//! File source. Reads a file and emits its bytes as `DataFrame` chunks, the
//! playback half of M20 (`FileSink` records, `FileSrc` replays). Feed an
//! Annex-B `.h264` recording through `H264Parse` to recover access units for
//! a decoder.
//!
//! A raw byte stream carries no caps, so the caller declares them at
//! construction (`FileSrc::new(path, caps)`); the source produces exactly
//! that declaration to the solver. Chunks carry no timing (`pts_ns` 0):
//! timing for a compressed stream is recovered downstream (parser/decoder),
//! matching how a raw recording loses per-frame boundaries.
//!
//! For a text pipeline / registry build the caps come from the
//! `bytestream-format` property instead (M112): `mpegts` / `matroska` name the
//! container directly, and `auto` sniffs the file header at negotiation (the one
//! case `FileSrc` does I/O before `run`) so `filesrc location=x.webm
//! bytestream-format=auto ! matroskademux` works without naming the container.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use std::fs::File;
use std::io::{Read, Seek as _, SeekFrom};
use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{SeekController, SourceLoop};
use g2g_core::{
    ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::filesink::io_err;

/// Default read chunk size: large enough to amortize syscalls, small enough
/// that a parser downstream sees steady progress.
const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

/// Bytes read to sniff the container in `bytestream-format=auto` mode. Enough to
/// confirm an MPEG-TS sync byte across several 188-byte packets.
const SNIFF_LEN: usize = 4 * 188;

#[derive(Debug)]
pub struct FileSrc {
    path: PathBuf,
    caps: Caps,
    /// `bytestream-format=auto`: sniff the container from the file header at
    /// negotiation, replacing `caps` with the detected `ByteStream{..}`.
    auto_detect: bool,
    chunk_size: usize,
    configured: bool,
    /// Optional byte-offset seek channel (M361). A `FileSrc` is a byte source, so
    /// a [`Seek`](g2g_core::Seek) it observes is in **BYTES** format: `start` is a
    /// file byte offset, not a timestamp. A downstream demuxer drives this to
    /// reposition the read for a time seek it resolved to a byte offset.
    seek: Option<SeekController>,
}

impl FileSrc {
    /// `caps` is the stream's declared format (e.g.
    /// `Caps::CompressedVideo { codec: H264, .. }` for an Annex-B
    /// elementary-stream recording); the file is opened in `run`, so
    /// construction has no filesystem side effects.
    pub fn new(path: impl Into<PathBuf>, caps: Caps) -> Self {
        Self {
            path: path.into(),
            caps,
            auto_detect: false,
            chunk_size: DEFAULT_CHUNK_SIZE,
            configured: false,
            seek: None,
        }
    }

    /// Make the source byte-seekable (M361): `run` polls `controller` between
    /// chunks and, on a flushing seek, emits `Flush`, repositions the file read
    /// to `seek.start` (a **byte** offset, since `FileSrc` is a byte source), and
    /// resumes. A downstream demuxer that resolved a time seek to a byte offset
    /// holds a clone of this controller to drive the reposition.
    pub fn with_seek(mut self, controller: SeekController) -> Self {
        self.seek = Some(controller);
        self
    }

    /// Resolve `bytestream-format=auto`: read the file header once and sniff the
    /// container, replacing `caps`. A no-op unless auto mode is armed; idempotent
    /// (clears the flag), so calling it from both negotiation entry points reads
    /// the header at most once.
    fn resolve_auto_caps(&mut self) -> Result<(), G2gError> {
        if !self.auto_detect {
            return Ok(());
        }
        let mut file = File::open(&self.path).map_err(io_err)?;
        let mut header = alloc::vec![0u8; SNIFF_LEN];
        let mut filled = 0;
        while filled < header.len() {
            let n = file.read(&mut header[filled..]).map_err(io_err)?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        header.truncate(filled);
        let encoding = crate::typefind::sniff(&header).ok_or(G2gError::CapsMismatch)?;
        self.caps = Caps::ByteStream { encoding };
        self.auto_detect = false;
        Ok(())
    }

    /// Bytes per emitted `DataFrame`. Clamped to 1 so a misconfigured zero
    /// cannot spin without progress.
    pub fn with_chunk_size(mut self, bytes: usize) -> Self {
        self.chunk_size = bytes.max(1);
        self
    }
}

impl SourceLoop for FileSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        if let Err(e) = self.resolve_auto_caps() {
            return core::future::ready(Err(e));
        }
        core::future::ready(Ok(self.caps.clone()))
    }

    /// Produces the declared caps (or, in `auto` mode, the sniffed container).
    /// Synchronous override; auto mode reads the file header once here, otherwise
    /// the file is opened in `run`.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        let result = match self.resolve_auto_caps() {
            Ok(()) => Ok(CapsConstraint::Produces(CapsSet::one(self.caps.clone()))),
            Err(e) => Err(e),
        };
        core::future::ready(result)
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Expose the container caps set by `bytestream-format` (M195), so a
    /// downstream `decodebin` auto-plugs the right demuxer. `None` in
    /// `bytestream-format=auto` mode, where the container is only known after
    /// sniffing the file header at run time.
    fn configured_output_caps(&self) -> Option<Caps> {
        (!self.auto_detect).then(|| self.caps.clone())
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }

            let mut file = File::open(&self.path).map_err(io_err)?;
            let mut sequence = 0u64;
            loop {
                // A flushing byte-seek repositions the read before the next chunk
                // (GStreamer BYTES-format seek: `start` is a file offset). Emit
                // `Flush` so a downstream demuxer drops its parse buffer and
                // re-syncs from the new position.
                if let Some(seek) = self.seek.as_ref().and_then(|c| c.take_pending()) {
                    if seek.is_flush() {
                        out.push(PipelinePacket::Flush).await?;
                        file.seek(SeekFrom::Start(seek.start)).map_err(io_err)?;
                    }
                    continue; // re-evaluate from the repositioned offset
                }

                let mut buf = alloc::vec![0u8; self.chunk_size];
                let mut filled = 0usize;
                // A reader may return short reads; fill the chunk until EOF
                // so every frame but the last is exactly chunk_size.
                while filled < buf.len() {
                    let n = file.read(&mut buf[filled..]).map_err(io_err)?;
                    if n == 0 {
                        break;
                    }
                    filled += n;
                }
                if filled == 0 {
                    // EOF: honor a seek that arrived as the read drained (a
                    // reposition before ending), else end the stream.
                    if self.seek.as_ref().is_some_and(|c| c.has_pending()) {
                        continue;
                    }
                    break;
                }
                buf.truncate(filled);

                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                    timing: FrameTiming {
                        // Stamped so downstream sinks can record
                        // glass-to-glass latency; this module implies std.
                        arrival_ns: g2g_core::metrics::monotonic_ns(),
                        ..FrameTiming::default()
                    },
                    sequence,
                    meta: Default::default(),
                };
                sequence += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FILESRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "File source",
            "Source/File",
            "Reads a local file as a byte stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.path = PathBuf::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            "bytestream-format" => {
                match value.as_str().ok_or(PropError::Type)? {
                    "auto" => self.auto_detect = true,
                    s => {
                        let encoding = encoding_from_str(s).ok_or(PropError::Value)?;
                        self.caps = Caps::ByteStream { encoding };
                        self.auto_detect = false;
                    }
                }
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.path.to_string_lossy().into_owned())),
            "bytestream-format" => {
                if self.auto_detect {
                    Some(PropValue::Str("auto".into()))
                } else if let Caps::ByteStream { encoding } = &self.caps {
                    Some(PropValue::Str(encoding_to_str(*encoding).into()))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// `FileSrc`'s settable properties (M107, M112): the input file path, and the
/// container of a raw byte stream (so a text pipeline can feed a demuxer).
static FILESRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "input file path"),
    PropertySpec::new(
        "bytestream-format",
        PropKind::Str,
        "container of a raw byte stream: mpegts | matroska | ogg | flv | auto (sniff the header)",
    ),
];

/// Parse a `bytestream-format` value to an encoding (the `auto` value is handled
/// separately in `set_property`).
fn encoding_from_str(s: &str) -> Option<ByteStreamEncoding> {
    match s {
        "mpegts" | "ts" => Some(ByteStreamEncoding::MpegTs),
        "matroska" | "mkv" | "webm" => Some(ByteStreamEncoding::Matroska),
        "ogg" | "opus" => Some(ByteStreamEncoding::Ogg),
        "flv" => Some(ByteStreamEncoding::Flv),
        "mp4" | "isobmff" | "cmaf" | "fmp4" => Some(ByteStreamEncoding::IsoBmff),
        _ => None,
    }
}

/// The canonical `bytestream-format` string for an encoding.
fn encoding_to_str(encoding: ByteStreamEncoding) -> &'static str {
    match encoding {
        ByteStreamEncoding::MpegTs => "mpegts",
        ByteStreamEncoding::Matroska => "matroska",
        ByteStreamEncoding::Ogg => "ogg",
        ByteStreamEncoding::Flv => "flv",
        ByteStreamEncoding::IsoBmff => "mp4",
    }
}
