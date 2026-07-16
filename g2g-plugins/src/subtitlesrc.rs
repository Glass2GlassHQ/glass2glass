//! Subtitle / text file source (M433): reads a `.srt` / `.vtt` / `.ssa` / `.ttml`
//! file and emits its contents as a [`Caps::Text`] stream, the streaming
//! counterpart of the out-of-band file `TextOverlay` loads with `location=`. It is
//! the head of a text pipeline: `subtitlesrc location=subs.srt ! subparse ! ...`
//! parses to timed `Text{Utf8}` cues that drive a [`TextOverlay`](crate::textoverlay)
//! or a caption encoder ([`CcInsert`](crate::ccinsert)), so a subtitle file can
//! author embedded closed captions or overlay text without hand-built Rust.
//!
//! The whole document is emitted as one frame (subtitle files are small, like the
//! batch demuxers), then `Eos`; `SubParse` downstream drains complete cues and
//! flushes the remainder at end of stream.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    TextFormat,
};

use crate::filesink::io_err;

/// A subtitle / text file source. Emits the file's bytes as a single
/// [`Caps::Text`] `format` frame, then `Eos`.
#[derive(Debug)]
pub struct SubtitleSrc {
    path: PathBuf,
    /// The text syntax the file carries (its output caps); sniffed from the path
    /// extension by [`from_location`](SubtitleSrc::from_location), or set explicitly.
    format: TextFormat,
    configured: bool,
}

impl SubtitleSrc {
    /// A source emitting the file at `path` as `Caps::Text { format }`.
    pub fn new(path: impl Into<PathBuf>, format: TextFormat) -> Self {
        Self { path: path.into(), format, configured: false }
    }

    /// A source whose text format is sniffed from the `path` extension
    /// (`.srt` -> SubRip, `.vtt` -> WebVTT, `.ssa` / `.ass` -> SSA, `.ttml` /
    /// `.xml` / `.dfxp` -> TTML; SubRip otherwise).
    pub fn from_location(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let format = format_from_path(&path);
        Self { path, format, configured: false }
    }

    fn output_caps(&self) -> Caps {
        Caps::Text { format: self.format }
    }
}

/// Sniff a subtitle [`TextFormat`] from a file extension (case-insensitive),
/// defaulting to SubRip.
fn format_from_path(path: &std::path::Path) -> TextFormat {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("vtt") => TextFormat::WebVtt,
        Some("ssa" | "ass") => TextFormat::Ssa,
        Some("ttml" | "xml" | "dfxp") => TextFormat::Ttml,
        // .srt and anything unrecognised parse as SubRip (SubParse's default).
        _ => TextFormat::Srt,
    }
}

/// Map a `format=` property string to a structured subtitle [`TextFormat`].
fn format_from_str(s: &str) -> Option<TextFormat> {
    Some(match s.to_ascii_lowercase().as_str() {
        "srt" | "subrip" => TextFormat::Srt,
        "vtt" | "webvtt" => TextFormat::WebVtt,
        "ssa" | "ass" => TextFormat::Ssa,
        "ttml" | "dfxp" => TextFormat::Ttml,
        _ => return None,
    })
}

/// The inverse of [`format_from_str`] for `get_property`.
fn format_to_str(f: TextFormat) -> &'static str {
    match f {
        TextFormat::WebVtt => "vtt",
        TextFormat::Ssa => "ssa",
        TextFormat::Ttml => "ttml",
        // Srt and the non-structured formats report as srt (only structured ones
        // are valid subtitle inputs).
        _ => "srt",
    }
}

impl SourceLoop for SubtitleSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.output_caps()))
    }

    /// Produces the file's text caps, so a downstream `decodebin` / `subparse`
    /// negotiates against the concrete format.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.output_caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        Some(self.output_caps())
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let bytes = std::fs::read(&self.path).map_err(io_err)?;
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SUBTITLESRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Subtitle file source",
            "Source/File/Subtitle",
            "Reads a SubRip / WebVTT / SSA / TTML file as a Text stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                let path = PathBuf::from(value.as_str().ok_or(PropError::Type)?);
                // A bare `location` re-sniffs the format from the new extension
                // (an explicit `format=` afterwards still overrides it).
                self.format = format_from_path(&path);
                self.path = path;
                Ok(())
            }
            "format" => {
                self.format = format_from_str(value.as_str().ok_or(PropError::Type)?)
                    .ok_or(PropError::Value)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.path.to_string_lossy().into_owned())),
            "format" => Some(PropValue::Str(format_to_str(self.format).into())),
            _ => None,
        }
    }
}

/// `SubtitleSrc`'s settable properties: the input file path, and an explicit text
/// format override (otherwise sniffed from the extension).
static SUBTITLESRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "input subtitle file path"),
    PropertySpec::new("format", PropKind::Str, "text format: srt | vtt | ssa | ttml (else sniffed from the extension)"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::vec::Vec;
    use core::future::Future;
    use core::pin::Pin;
    use g2g_core::PushOutcome;

    #[derive(Default)]
    struct Collect {
        packets: Vec<PipelinePacket>,
    }
    impl OutputSink for Collect {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                self.packets.push(packet);
                Ok(PushOutcome::Accepted)
            })
        }
    }

    #[test]
    fn sniffs_format_from_extension() {
        assert_eq!(SubtitleSrc::from_location("/x/subs.vtt").format, TextFormat::WebVtt);
        assert_eq!(SubtitleSrc::from_location("/x/subs.ass").format, TextFormat::Ssa);
        assert_eq!(SubtitleSrc::from_location("/x/subs.ttml").format, TextFormat::Ttml);
        assert_eq!(SubtitleSrc::from_location("/x/subs.srt").format, TextFormat::Srt);
        assert_eq!(SubtitleSrc::from_location("/x/unknown").format, TextFormat::Srt);
    }

    #[tokio::test]
    async fn emits_the_file_as_a_text_frame_then_eos() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("g2g_subsrc_{}.srt", std::process::id()));
        let doc = "1\n00:00:01,000 --> 00:00:03,000\nHELLO\n";
        std::fs::write(&path, doc).unwrap();

        let mut src = SubtitleSrc::from_location(&path);
        // Negotiation: the source advertises the sniffed text format.
        let caps = src.intercept_caps().await.unwrap();
        assert_eq!(caps, Caps::Text { format: TextFormat::Srt });
        src.configure_pipeline(&caps).unwrap();

        let mut sink = Collect::default();
        let pushed = src.run(&mut sink).await.unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(pushed, 1, "one document frame emitted");
        assert!(matches!(sink.packets.last(), Some(PipelinePacket::Eos)), "ends with Eos");
        match &sink.packets[0] {
            PipelinePacket::DataFrame(f) => {
                let MemoryDomain::System(s) = &f.domain else { panic!("system buffer") };
                assert_eq!(s.as_slice(), doc.as_bytes(), "the file bytes are emitted verbatim");
            }
            _ => panic!("first packet is the document frame"),
        }
    }

    #[tokio::test]
    async fn run_before_configure_is_an_error() {
        let mut src = SubtitleSrc::new("/nonexistent.srt", TextFormat::Srt);
        let mut sink = Collect::default();
        assert!(src.run(&mut sink).await.is_err());
    }
}
