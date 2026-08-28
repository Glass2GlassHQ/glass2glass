//! Split-file source (`splitfilesrc`). Reads every file matching a wildcard
//! pattern, in name order, as one continuous byte stream: the playback half of a
//! recording that was cut into parts (`multifilesink`, or `splitmuxsink`'s
//! segments read as plain bytes).
//!
//! The parts are joined without a boundary of any kind, so this is right only
//! when the parts are pieces of one file (the `multifilesink next-file=bytes`
//! case). Segments that each carry their own container header are separate
//! files, one demuxer run each.
//!
//! The media type comes from an explicit `bytestream-format` (the same names
//! `filesrc` takes), the first match's extension, or that part's own header: a
//! name like `clip.ts.part003` has no usable extension, so the header is what
//! types it. Nothing is guessed: a first part that sniffs as nothing fails to
//! configure rather than picking a container.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use g2g_core::frame::Frame;
use g2g_core::log::short_type_name;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::filesink::{io_err, path_io_err};
use crate::filesrc::{encoding_from_str, encoding_to_str};

/// Default read chunk size, matching `filesrc`: large enough to amortize
/// syscalls, small enough that a demuxer downstream sees steady progress.
const DEFAULT_BLOCKSIZE: usize = 64 * 1024;

/// The same value as declared text, for `gst-inspect`.
const DEFAULT_BLOCKSIZE_TEXT: &str = "65536";

static SPLITFILESRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "wildcard pattern matching the parts, e.g. recording.part*; matched on the file name and read in name order",
    ),
    PropertySpec::new("blocksize", PropKind::Uint, "bytes to read per buffer")
        .with_default(DEFAULT_BLOCKSIZE_TEXT),
    PropertySpec::new(
        "bytestream-format",
        PropKind::Str,
        "container of the joined stream: mpegts | matroska | ogg | flv | raw (default: from the first part's extension)",
    ),
];

/// # Example
///
/// ```no_run
/// use g2g_plugins::splitfilesrc::SplitFileSrc;
///
/// // gst-launch equivalent: splitfilesrc location="clip.ts.part*" ! tsdemux
/// let src = SplitFileSrc::new("clip.ts.part*");
/// ```
#[derive(Debug)]
pub struct SplitFileSrc {
    location: String,
    /// The media type of the joined stream: from `bytestream-format`, else the
    /// first match's extension at negotiation.
    caps: Option<Caps>,
    /// `true` once `bytestream-format` pinned the type, so the extension does
    /// not override it.
    format_explicit: bool,
    blocksize: usize,
    configured: bool,
}

impl SplitFileSrc {
    pub fn new(location: impl Into<String>) -> Self {
        Self {
            location: location.into(),
            caps: None,
            format_explicit: false,
            blocksize: DEFAULT_BLOCKSIZE,
            configured: false,
        }
    }

    /// Declare the joined stream's media type instead of deriving it.
    pub fn with_caps(mut self, caps: Caps) -> Self {
        self.caps = Some(caps);
        self.format_explicit = true;
        self
    }

    pub fn with_blocksize(mut self, bytes: usize) -> Self {
        if bytes > 0 {
            self.blocksize = bytes;
        }
        self
    }

    /// The parts the pattern matches, in name order. The pattern's directory
    /// part selects the directory and its file-name part is the wildcard, the
    /// way gst's `splitfilesrc` reads it.
    pub fn parts(&self) -> Result<Vec<PathBuf>, G2gError> {
        let pattern = Path::new(&self.location);
        let directory = match pattern.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
            _ => PathBuf::from("."),
        };
        let Some(name) = pattern.file_name().and_then(|name| name.to_str()) else {
            return Err(G2gError::CapsMismatch);
        };
        let entries = std::fs::read_dir(&directory)
            .map_err(|e| path_io_err(short_type_name::<Self>(), "read_dir", &directory, e))?;
        let mut matched = Vec::new();
        for entry in entries {
            let entry = entry.map_err(io_err)?;
            let file_name = entry.file_name();
            let Some(candidate) = file_name.to_str() else {
                continue;
            };
            if wildcard_matches(name, candidate) {
                matched.push(entry.path());
            }
        }
        matched.sort();
        Ok(matched)
    }

    /// The media type of the joined stream: the explicit one, else the first
    /// part's extension, else what its header sniffs as. A part name like
    /// `clip.ts.part003` has no usable extension, which is why the header is
    /// read as well.
    fn resolve_caps(&self) -> Result<Caps, G2gError> {
        if let Some(caps) = &self.caps {
            return Ok(caps.clone());
        }
        let parts = self.parts()?;
        let first = parts.first().ok_or(G2gError::CapsMismatch)?;
        if let Some(caps) = crate::filesrc::caps_from_extension(first) {
            return Ok(caps);
        }
        crate::filesrc::sniff_file_caps(first, short_type_name::<Self>())
    }
}

/// Whether `candidate` matches a `*` / `?` wildcard `pattern`, the shell
/// globbing gst's `location` takes. `*` spans any run of characters (including
/// none), `?` exactly one; every other character matches itself.
fn wildcard_matches(pattern: &str, candidate: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let candidate: Vec<char> = candidate.chars().collect();
    // Backtracking over the last `*`: on a mismatch, the star swallows one more
    // character and matching resumes from there.
    let (mut at_pattern, mut at_candidate) = (0usize, 0usize);
    let mut star: Option<(usize, usize)> = None;
    loop {
        match pattern.get(at_pattern) {
            Some('*') => {
                star = Some((at_pattern, at_candidate));
                at_pattern += 1;
            }
            Some('?') if at_candidate < candidate.len() => {
                at_pattern += 1;
                at_candidate += 1;
            }
            Some(expected) if candidate.get(at_candidate) == Some(expected) => {
                at_pattern += 1;
                at_candidate += 1;
            }
            None if at_candidate == candidate.len() => return true,
            _ => match star {
                Some((star_at, swallowed)) if swallowed < candidate.len() => {
                    at_pattern = star_at + 1;
                    at_candidate = swallowed + 1;
                    star = Some((star_at, at_candidate));
                }
                _ => return false,
            },
        }
    }
}

impl SourceLoop for SplitFileSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(self.resolve_caps())
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        let caps = self.resolve_caps();
        core::future::ready(caps.map(|caps| CapsConstraint::Produces(CapsSet::one(caps))))
    }

    /// The solved caps are a fixation of the derived type (a still sequence's
    /// placeholder geometry gets fixed downstream), so they are intersected
    /// rather than compared.
    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let resolved = self.resolve_caps()?;
        self.caps = Some(absolute_caps.intersect(&resolved)?);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Only an explicit `bytestream-format`, since deriving the type reads the
    /// directory and the first part's header.
    fn configured_output_caps(&self) -> Option<Caps> {
        self.caps.clone()
    }

    /// Parse-time caps for `decodebin`: read the first part now, so the demuxer
    /// is picked from what the recording actually is.
    fn probe_output_caps(&mut self) -> Option<Caps> {
        self.resolve_caps().ok()
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut sequence = 0u64;
            for path in self.parts()? {
                let mut file = File::open(&path)
                    .map_err(|e| path_io_err(short_type_name::<Self>(), "open", &path, e))?;
                loop {
                    let mut buf = alloc::vec![0u8; self.blocksize];
                    let read = file.read(&mut buf).map_err(io_err)?;
                    if read == 0 {
                        break;
                    }
                    buf.truncate(read);
                    let frame = Frame::new(
                        MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                        // A byte stream carries no timing; a demuxer downstream
                        // recovers it.
                        FrameTiming::default(),
                        sequence,
                    );
                    sequence += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SPLITFILESRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Split-file source",
            "Source/File",
            "Reads the files matching a pattern as one continuous byte stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.location = value.as_str().ok_or(PropError::Type)?.into();
                if !self.format_explicit {
                    // The type comes from the first match, which needs the
                    // directory read: leave it for negotiation.
                    self.caps = None;
                }
            }
            "blocksize" => {
                let bytes = value.as_uint().ok_or(PropError::Type)?;
                if bytes == 0 || bytes > u64::from(u32::MAX) {
                    return Err(PropError::Value);
                }
                self.blocksize = bytes as usize;
            }
            "bytestream-format" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                let encoding = encoding_from_str(text).ok_or(PropError::Value)?;
                self.caps = Some(Caps::ByteStream { encoding });
                self.format_explicit = true;
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.location.clone())),
            "blocksize" => Some(PropValue::Uint(self.blocksize as u64)),
            "bytestream-format" => match &self.caps {
                Some(Caps::ByteStream { encoding }) => {
                    Some(PropValue::Str(encoding_to_str(*encoding).into()))
                }
                // Not a container (a still-image sequence), or not resolved yet.
                _ => Some(PropValue::Str(String::new())),
            },
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::format;

    use g2g_core::ByteStreamEncoding;

    #[test]
    fn wildcards_match_the_way_a_shell_glob_does() {
        assert!(wildcard_matches("clip.ts.part*", "clip.ts.part003"));
        assert!(wildcard_matches("*", "anything"));
        assert!(wildcard_matches("a*c", "ac"));
        assert!(wildcard_matches("a*b*c", "azzbzzc"));
        assert!(wildcard_matches("part??.ts", "part07.ts"));
        assert!(!wildcard_matches("part??.ts", "part7.ts"));
        assert!(!wildcard_matches("clip.ts.part*", "clip.mp4.part1"));
        assert!(!wildcard_matches("a*c", "abcd"));
        assert!(!wildcard_matches("exact", "exact.ts"));
    }

    /// A directory of parts, named so the sort order is the read order.
    fn write_parts(tag: &str, parts: &[&[u8]]) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("g2g-splitfilesrc-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the part directory is created");
        for (index, bytes) in parts.iter().enumerate() {
            std::fs::write(dir.join(format!("clip.ts.part{index:03}")), bytes)
                .expect("a part is written");
        }
        // A file the pattern must not match.
        std::fs::write(dir.join("notes.txt"), b"skip me").expect("the decoy is written");
        dir
    }

    struct CollectSink {
        payload: Vec<u8>,
        buffers: usize,
        eos: bool,
    }

    impl OutputSink for CollectSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            match packet_slot.take().expect("poll_push without a packet") {
                PipelinePacket::DataFrame(f) => {
                    self.payload
                        .extend_from_slice(f.domain.as_system_slice().expect("system"));
                    self.buffers += 1;
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
        }
    }

    #[tokio::test]
    async fn joins_every_part_in_name_order() {
        let parts: [&[u8]; 3] = [b"first", b"second", b"third"];
        let dir = write_parts("join", &parts);
        let mut src = SplitFileSrc::new(dir.join("clip.ts.part*").to_string_lossy().into_owned());
        // Bytes that sniff as nothing: the format is stated instead. A real
        // container's parts type themselves, which `m1088_split_and_data.rs`
        // covers against a checked-in `.ts` fixture.
        src.set_property("bytestream-format", PropValue::Str("raw".into()))
            .expect("raw is a format value");
        assert_eq!(
            src.resolve_caps(),
            Ok(Caps::ByteStream {
                encoding: ByteStreamEncoding::Raw
            })
        );
        src.configure_pipeline(&src.resolve_caps().expect("typed"))
            .expect("the derived caps are accepted");
        let mut out = CollectSink {
            payload: Vec::new(),
            buffers: 0,
            eos: false,
        };
        src.run(&mut out).await.expect("the parts read");
        assert_eq!(out.payload, parts.concat(), "one continuous byte stream");
        assert!(out.eos);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn blocksize_bounds_each_buffer() {
        let part = alloc::vec![7u8; 10];
        let dir = write_parts("blocksize", &[&part]);
        let mut src = SplitFileSrc::new(dir.join("clip.ts.part*").to_string_lossy().into_owned())
            .with_blocksize(4)
            .with_caps(Caps::ByteStream {
                encoding: ByteStreamEncoding::Raw,
            });
        src.configure_pipeline(&src.resolve_caps().expect("typed"))
            .expect("the derived caps are accepted");
        let mut out = CollectSink {
            payload: Vec::new(),
            buffers: 0,
            eos: false,
        };
        src.run(&mut out).await.expect("the part reads");
        assert_eq!(out.buffers, 3, "4 + 4 + 2 bytes");
        assert_eq!(out.payload, part);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unmatched_pattern_and_an_unknown_extension_fail_loud() {
        let dir =
            std::env::temp_dir().join(format!("g2g-splitfilesrc-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("the directory is created");
        let src = SplitFileSrc::new(dir.join("nothing*").to_string_lossy().into_owned());
        assert_eq!(
            src.resolve_caps(),
            Err(G2gError::CapsMismatch),
            "no part, no media type"
        );
        std::fs::write(dir.join("nothing.unknown"), b"x").expect("a part is written");
        assert_eq!(
            src.resolve_caps(),
            Err(G2gError::CapsMismatch),
            "an unknown extension is not guessed"
        );
        // An explicit format types it anyway.
        let mut src = src;
        src.set_property("bytestream-format", PropValue::Str("raw".into()))
            .expect("raw is a format value");
        assert_eq!(
            src.resolve_caps(),
            Ok(Caps::ByteStream {
                encoding: ByteStreamEncoding::Raw
            })
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
