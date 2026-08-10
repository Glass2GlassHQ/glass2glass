//! VobSub sidecar source (M926): reads a `.idx` / `.sub` pair from disk and
//! emits it as a [`Caps::SubPicture`] `VobSub` stream, so DVD subtitles that sit
//! next to a video as two files play like a muxed track:
//!
//! ```text
//! vobsubsrc location=movie.idx ! vobsubdec ! compositor.
//! ```
//!
//! The `.idx` text is both halves of the sidecar's index: the palette and
//! display geometry the cues are decoded with, and a `timestamp:` / `filepos:`
//! list saying when each cue shows and where its subpicture unit starts in the
//! `.sub`. So the stream opens on the `.idx` text itself, which is what
//! [`VobSubDec`](crate::vobsubdec::VobSubDec) expects in band ahead of the first
//! cue (a Matroska `S_VOBSUB` track's `CodecPrivate` is the same text), then one
//! frame per cue stamped with the `.idx` time and the unit's own duration.
//!
//! An `.idx` may index several languages; `language=` picks one by its `id:`
//! code, otherwise the stream the file's `langidx:` names.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};

use std::path::{Path, PathBuf};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
    SubPictureFormat,
};

use crate::filesink::io_err;
use crate::vobsub::{parse_idx_index, read_spu_packet, spu_timing, IdxStream};

/// A VobSub `.idx` / `.sub` sidecar source.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::vobsubsrc::VobSubSrc;
///
/// // vobsubsrc location=movie.idx ! vobsubdec ! compositor.
/// let src = VobSubSrc::new("movie.idx").with_language("en");
/// ```
#[derive(Debug)]
pub struct VobSubSrc {
    idx_path: PathBuf,
    /// An explicit `.sub` path; without one it is the `.idx` path with its
    /// extension swapped, which is how the pair is always named on disk.
    sub_path: Option<PathBuf>,
    language: Option<String>,
    configured: bool,
}

impl VobSubSrc {
    /// A source reading the `.idx` at `path` and the `.sub` beside it.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            idx_path: path.into(),
            sub_path: None,
            language: None,
            configured: false,
        }
    }

    /// Read the subpicture data from `path` rather than the `.idx`'s neighbour.
    pub fn with_sub_location(mut self, path: impl Into<PathBuf>) -> Self {
        self.sub_path = Some(path.into());
        self
    }

    /// Emit the cues of the `.idx` stream whose `id:` is this language code.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    fn sub_path(&self) -> PathBuf {
        match &self.sub_path {
            Some(path) => path.clone(),
            None => self.idx_path.with_extension("sub"),
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::SubPicture {
            format: SubPictureFormat::VobSub,
        }
    }
}

/// Pick the language stream to emit: the requested `id:` code, else the one the
/// file's `langidx:` names, else the first.
fn select<'a>(
    streams: &'a [IdxStream],
    language: Option<&str>,
    langidx: Option<u32>,
) -> &'a IdxStream {
    let by_language = language.and_then(|want| {
        streams
            .iter()
            .find(|s| s.language.eq_ignore_ascii_case(want))
    });
    let by_index = || langidx.and_then(|want| streams.iter().find(|s| s.index == want));
    by_language.or_else(by_index).unwrap_or_else(|| &streams[0])
}

fn frame(bytes: alloc::vec::Vec<u8>, timing: FrameTiming, seq: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing,
        seq,
    ))
}

impl SourceLoop for VobSubSrc {
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

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            self.output_caps(),
        ))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn configured_output_caps(&self) -> Option<Caps> {
        Some(self.output_caps())
    }

    /// Emits the `.idx` text, then one frame per indexed cue, then `Eos`.
    /// Returns the number of frames pushed. A cue whose offset or packet does
    /// not hold together is skipped, the way the decoder drops a malformed one,
    /// so one bad entry does not cost the rest of the subtitles.
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let idx = std::fs::read(&self.idx_path).map_err(io_err)?;
            let index = parse_idx_index(&idx).ok_or(G2gError::CapsMismatch)?;
            if index.streams.is_empty() {
                return Err(G2gError::CapsMismatch);
            }
            let stream = select(&index.streams, self.language.as_deref(), index.langidx);
            let entries = stream.entries.clone();
            let sub = std::fs::read(self.sub_path()).map_err(io_err)?;

            // The out-of-band config first: the decoder tells it from a cue by
            // parsing it as `.idx` text, and needs it before the first one.
            let mut pushed = 0u64;
            out.push(frame(idx, FrameTiming::default(), pushed)).await?;
            pushed += 1;

            for entry in entries {
                let Some(spu) = read_spu_packet(&sub, entry.filepos) else {
                    continue;
                };
                // The unit's own control sequence carries the hide time, so the
                // duration is exact rather than "until the next cue".
                let duration_ns = spu_timing(&spu)
                    .and_then(|(start, stop)| stop.map(|stop| stop.saturating_sub(start)))
                    .unwrap_or(0);
                let timing = FrameTiming {
                    pts_ns: entry.time_ns,
                    dts_ns: entry.time_ns,
                    duration_ns,
                    ..FrameTiming::default()
                };
                out.push(frame(spu, timing, pushed)).await?;
                pushed += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(pushed)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        VOBSUBSRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "VobSub sidecar source",
            "Source/File/Subtitle",
            "Reads a VobSub .idx / .sub pair as a DVD subpicture stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let text = value.as_str().ok_or(PropError::Type)?;
        match name {
            "location" => {
                self.idx_path = PathBuf::from(text);
                Ok(())
            }
            "sub-location" => {
                self.sub_path = (!text.is_empty()).then(|| PathBuf::from(text));
                Ok(())
            }
            "language" => {
                self.language = (!text.is_empty()).then(|| text.to_string());
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let path = |p: &Path| PropValue::Str(p.to_string_lossy().into_owned());
        Some(match name {
            "location" => path(&self.idx_path),
            "sub-location" => path(&self.sub_path()),
            "language" => PropValue::Str(self.language.clone().unwrap_or_default()),
            _ => return None,
        })
    }
}

/// `VobSubSrc`'s settable properties: the `.idx` path, an override for the
/// `.sub` beside it, and which indexed language to emit.
static VOBSUBSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "input VobSub .idx file path"),
    PropertySpec::new(
        "sub-location",
        PropKind::Str,
        "subpicture .sub file path (else the .idx path with a .sub extension)",
    ),
    PropertySpec::new(
        "language",
        PropKind::Str,
        "`id:` language code to emit (else the .idx `langidx:` stream)",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::vec::Vec;
    use g2g_core::PushOutcome;

    #[derive(Default)]
    struct Collect {
        packets: Vec<PipelinePacket>,
    }
    impl OutputSink for Collect {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                self.packets.push(packet);
                Ok(PushOutcome::Accepted)
            })
        }
    }

    #[test]
    fn derives_the_sub_path_from_the_idx_path() {
        let src = VobSubSrc::new("/x/movie.idx");
        assert_eq!(src.sub_path(), PathBuf::from("/x/movie.sub"));
        let src = src.with_sub_location("/y/other.sub");
        assert_eq!(src.sub_path(), PathBuf::from("/y/other.sub"));
    }

    #[test]
    fn selects_the_language_stream() {
        let streams = Vec::from([
            IdxStream {
                language: String::from("en"),
                index: 0,
                entries: Vec::new(),
            },
            IdxStream {
                language: String::from("fr"),
                index: 1,
                entries: Vec::new(),
            },
        ]);
        assert_eq!(select(&streams, Some("fr"), Some(0)).index, 1);
        assert_eq!(select(&streams, None, Some(1)).index, 1);
        // an unknown language or langidx falls back to the file's first stream
        assert_eq!(select(&streams, Some("de"), None).index, 0);
        assert_eq!(select(&streams, None, Some(7)).index, 0);
    }

    #[tokio::test]
    async fn run_before_configure_is_an_error() {
        let mut src = VobSubSrc::new("/nonexistent.idx");
        let mut sink = Collect::default();
        assert!(src.run(&mut sink).await.is_err());
    }

    #[tokio::test]
    async fn a_sidecar_without_an_index_is_an_error() {
        let path = std::env::temp_dir().join(format!("g2g_vobsubsrc_{}.idx", std::process::id()));
        std::fs::write(&path, b"not an idx file at all\n").unwrap();
        let mut src = VobSubSrc::new(&path);
        let caps = src.output_caps();
        src.configure_pipeline(&caps).unwrap();
        let mut sink = Collect::default();
        let err = src.run(&mut sink).await;
        std::fs::remove_file(&path).ok();
        assert!(matches!(err, Err(G2gError::CapsMismatch)));
    }
}
