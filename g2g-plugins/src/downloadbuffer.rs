//! Spill-to-storage byte buffer (M861), the `downloadbuffer` analog: absorbs a
//! pushed, non-seekable byte stream (HTTP, a pipe) into a temp file and serves
//! downstream from that file, so the link below it is a *seekable* byte source.
//!
//! Two things follow from owning the whole stream on disk:
//!
//! - **Byte seeks.** A [`SeekController`] handed in with
//!   [`with_seek`](DownloadBuffer::with_seek) is polled between chunks. A
//!   flushing seek emits `Flush` and re-serves from the requested byte offset,
//!   the same BYTES-format contract [`FileSrc`](crate::filesrc) implements. A
//!   seek past the high-water mark (bytes received so far) is not an error: the
//!   read waits and resumes once the download reaches it.
//! - **Whole-file container typing.** `ByteStream{IsoBmff}` (the streaming,
//!   incremental form HLS / DASH / `httpsrc` produce) becomes `ByteStream{Mp4}`
//!   on the output, because a spilled stream is a whole file with random access,
//!   exactly the rewrite `FileSrc` applies to a sniffed ISO-BMFF header. That is
//!   what lets `httpsrc bytestream-format=mp4 ! downloadbuffer ! qtdemux` play a
//!   moov-at-end MP4 that `httpsrc ! qtdemux` cannot even negotiate. Every other
//!   encoding passes through unchanged.
//!
//! Temp file lifecycle: created at `configure_pipeline` from `temp-template`
//! (a trailing `XXXXXX` is replaced, `mkstemp`-style, and a bare filename is
//! resolved under the system temp dir), readable back as `temp-location`,
//! opened read+write, and removed on drop unless `temp-remove=false`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek as _, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SeekController;
use g2g_core::{
    BusHandle, BusMessage, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PadTemplate, PadTemplates, PropError, PropKind,
    PropValue, PropertySpec,
};

use crate::filesink::io_err;

/// Bytes per `DataFrame` served out of the spill file.
const DEFAULT_BLOCKSIZE: usize = 64 * 1024;

/// Spilled bytes at which `Buffering` reports 100, matching `downloadbuffer`'s
/// 2 MB `max-size-bytes` default.
const DEFAULT_MAX_SIZE_BYTES: u64 = 2 * 1024 * 1024;

/// `mkstemp` placeholder in `temp-template`.
const TEMPLATE_MARK: &str = "XXXXXX";

/// Distinguishes concurrent instances within one process. The pid separates
/// processes, and `create_new` catches any residual collision.
static INSTANCE: AtomicU64 = AtomicU64::new(0);

/// # Example
///
/// ```no_run
/// use g2g_plugins::downloadbuffer::DownloadBuffer;
///
/// let buffer = DownloadBuffer::new()
///     .with_temp_template("/var/tmp/g2g-downloadXXXXXX")
///     .with_max_size_bytes(4 * 1024 * 1024);
/// ```
#[derive(Debug)]
pub struct DownloadBuffer {
    temp_template: String,
    /// The spill file's path, once created.
    temp_location: Option<PathBuf>,
    temp_remove: bool,
    max_size_bytes: u64,
    blocksize: usize,
    file: Option<File>,
    /// High-water mark: bytes received from upstream and spilled.
    write_pos: u64,
    /// Next byte to serve downstream. Moved by a byte seek, and may sit past
    /// `write_pos`, in which case serving waits for the download to reach it.
    read_pos: u64,
    seek: Option<SeekController>,
    bus: Option<BusHandle>,
    last_bucket: Option<u8>,
    sequence: u64,
    configured: bool,
}

impl Default for DownloadBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl DownloadBuffer {
    pub fn new() -> Self {
        Self {
            temp_template: String::from("g2g-downloadbufferXXXXXX"),
            temp_location: None,
            temp_remove: true,
            max_size_bytes: DEFAULT_MAX_SIZE_BYTES,
            blocksize: DEFAULT_BLOCKSIZE,
            file: None,
            write_pos: 0,
            read_pos: 0,
            seek: None,
            bus: None,
            last_bucket: None,
            sequence: 0,
            configured: false,
        }
    }

    /// Where to spill. A trailing `XXXXXX` is replaced to make the name unique,
    /// and a template with no directory component lands in the system temp dir.
    pub fn with_temp_template(mut self, template: impl Into<String>) -> Self {
        self.temp_template = template.into();
        self
    }

    /// Keep the spill file after this element drops (`temp-remove=false`), so a
    /// download can be reused.
    pub fn with_temp_remove(mut self, remove: bool) -> Self {
        self.temp_remove = remove;
        self
    }

    /// Spilled bytes at which `Buffering` reports 100. `0` silences the reports.
    pub fn with_max_size_bytes(mut self, bytes: u64) -> Self {
        self.max_size_bytes = bytes;
        self
    }

    /// Bytes per served `DataFrame`. Clamped to 1 so a zero cannot spin.
    pub fn with_blocksize(mut self, bytes: usize) -> Self {
        self.blocksize = bytes.max(1);
        self
    }

    /// Make the served stream byte-seekable: `controller` carries a
    /// **BYTES**-format [`Seek`](g2g_core::Seek) (`start` is a byte offset into
    /// the stream), polled between chunks. A downstream demuxer that resolved a
    /// time seek to a byte offset holds a clone.
    pub fn with_seek(mut self, controller: SeekController) -> Self {
        self.seek = Some(controller);
        self
    }

    /// Attach the pipeline bus so the spill posts [`BusMessage::Buffering`].
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// The spill file in use, once `configure_pipeline` has created it.
    pub fn temp_location(&self) -> Option<&std::path::Path> {
        self.temp_location.as_deref()
    }

    /// Bytes received from upstream so far (the high-water mark a downstream
    /// read cannot pass).
    pub fn high_water(&self) -> u64 {
        self.write_pos
    }

    /// A byte stream spilled to disk is a whole file with random access, so the
    /// streaming ISO-BMFF form becomes the whole-file MP4 form. Everything else
    /// is carried unchanged.
    fn output_encoding(encoding: ByteStreamEncoding) -> ByteStreamEncoding {
        match encoding {
            ByteStreamEncoding::IsoBmff => ByteStreamEncoding::Mp4,
            other => other,
        }
    }

    /// The (input, output) byte-stream pairs the solver picks from.
    fn mapping() -> Vec<(CapsSet, CapsSet)> {
        [
            ByteStreamEncoding::MpegTs,
            ByteStreamEncoding::Matroska,
            ByteStreamEncoding::Ogg,
            ByteStreamEncoding::Flv,
            ByteStreamEncoding::IsoBmff,
            ByteStreamEncoding::Mp4,
            ByteStreamEncoding::Ivf,
        ]
        .into_iter()
        .map(|encoding| {
            (
                CapsSet::one(Caps::ByteStream { encoding }),
                CapsSet::one(Caps::ByteStream {
                    encoding: Self::output_encoding(encoding),
                }),
            )
        })
        .collect()
    }

    /// Resolve `temp-template` to a concrete path and create the file. Two
    /// instances never share one: the `XXXXXX` token carries the pid and an
    /// instance counter, and `create_new` retries rather than truncating a file
    /// that somehow exists already.
    fn open_spill(&mut self) -> Result<(), G2gError> {
        if self.file.is_some() {
            return Ok(());
        }
        let dir_given = std::path::Path::new(&self.temp_template)
            .parent()
            .is_some_and(|p| !p.as_os_str().is_empty());
        let base = if dir_given {
            PathBuf::from(&self.temp_template)
        } else {
            std::env::temp_dir().join(&self.temp_template)
        };
        let template = base.to_string_lossy().into_owned();

        if !template.ends_with(TEMPLATE_MARK) {
            // A literal path: the caller named the file, so use it as given.
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&template)
                .map_err(io_err)?;
            self.file = Some(file);
            self.temp_location = Some(PathBuf::from(template));
            return Ok(());
        }

        let stem = &template[..template.len() - TEMPLATE_MARK.len()];
        let pid = std::process::id() as u64;
        // Bounded: each attempt uses a fresh counter value, so the loop only
        // repeats against a genuinely colliding name.
        for _ in 0..64 {
            let token = pid
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add(INSTANCE.fetch_add(1, Ordering::Relaxed));
            let path = PathBuf::from(alloc::format!("{stem}{:06x}", token & 0xFF_FFFF));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    self.file = Some(file);
                    self.temp_location = Some(path);
                    return Ok(());
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io_err(e)),
            }
        }
        Err(G2gError::Hardware(g2g_core::HardwareError::Other))
    }

    /// Append one received chunk to the spill file and advance the high-water
    /// mark. Fails loud if the stream would push the offset past `u64`.
    fn spill(&mut self, bytes: &[u8]) -> Result<(), G2gError> {
        let end = self
            .write_pos
            .checked_add(bytes.len() as u64)
            .ok_or(G2gError::CapsMismatch)?;
        let file = self.file.as_mut().ok_or(G2gError::NotConfigured)?;
        file.seek(SeekFrom::Start(self.write_pos)).map_err(io_err)?;
        file.write_all(bytes).map_err(io_err)?;
        self.write_pos = end;
        Ok(())
    }

    /// Push everything between the read cursor and the high-water mark, honoring
    /// byte seeks as they arrive. Returns when the reader has caught up, or when
    /// a seek left it past the high-water mark (waiting for those bytes).
    async fn serve(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        loop {
            // A flushing byte-seek repositions the read before the next chunk.
            // `Flush` tells downstream to drop its parse buffer and re-sync.
            if let Some(seek) = self.seek.as_ref().and_then(|c| c.take_pending()) {
                if seek.is_flush() {
                    out.push(PipelinePacket::Flush).await?;
                    self.read_pos = seek.start;
                }
                continue;
            }
            // `None` means the read cursor sits past the high-water mark: the
            // requested bytes have not arrived yet, so stop and wait.
            let Some(available) = self.write_pos.checked_sub(self.read_pos) else {
                return Ok(());
            };
            if available == 0 {
                return Ok(());
            }
            let n = available.min(self.blocksize as u64) as usize;
            let mut buf = alloc::vec![0u8; n];
            let file = self.file.as_mut().ok_or(G2gError::NotConfigured)?;
            file.seek(SeekFrom::Start(self.read_pos)).map_err(io_err)?;
            file.read_exact(&mut buf).map_err(io_err)?;
            self.read_pos = self.read_pos.saturating_add(n as u64);

            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                timing: FrameTiming {
                    arrival_ns: g2g_core::metrics::monotonic_ns(),
                    ..FrameTiming::default()
                },
                sequence: self.sequence,
                meta: Default::default(),
            };
            self.sequence += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
    }

    /// Post the spill level on quartile-band transitions only, like `HttpSrc`.
    /// `element` is `None`: this is the element's own buffer, not a runner link.
    fn post_level(&mut self) {
        if self.max_size_bytes == 0 {
            return;
        }
        let pct = (self.write_pos.saturating_mul(100) / self.max_size_bytes).min(100) as u8;
        if let Some(b) = &self.bus {
            let bucket = (pct / 25).min(4);
            if self.last_bucket != Some(bucket) {
                self.last_bucket = Some(bucket);
                b.try_post(BusMessage::Buffering {
                    percent: pct,
                    element: None,
                });
            }
        }
    }
}

impl Drop for DownloadBuffer {
    fn drop(&mut self) {
        // Close before unlinking so Windows lets the removal through.
        self.file = None;
        if self.temp_remove {
            if let Some(path) = &self.temp_location {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

impl AsyncElement for DownloadBuffer {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::ByteStream { .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn is_format_boundary(&self) -> bool {
        true
    }

    fn propose_output_caps(&self, input: &Caps) -> Caps {
        match input {
            Caps::ByteStream { encoding } => Caps::ByteStream {
                encoding: Self::output_encoding(*encoding),
            },
            other => other.clone(),
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Mapping(Self::mapping())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !matches!(absolute_caps, Caps::ByteStream { .. }) {
            return Err(G2gError::CapsMismatch);
        }
        self.open_spill()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_output(&mut self, output_caps: &Caps) -> Result<(), G2gError> {
        match output_caps {
            Caps::ByteStream { .. } => Ok(()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Download buffer",
            "Generic",
            "Spills a pushed byte stream to a temp file and serves it as a seekable byte source",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DOWNLOADBUFFER_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "temp-template" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                if s.is_empty() {
                    return Err(PropError::Value);
                }
                self.temp_template = String::from(s);
                Ok(())
            }
            "temp-remove" => {
                self.temp_remove = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "max-size-bytes" => {
                self.max_size_bytes = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            "blocksize" => {
                let bytes = value.as_uint().ok_or(PropError::Type)?;
                if bytes == 0 {
                    return Err(PropError::Value);
                }
                self.blocksize = bytes.min(1 << 30) as usize;
                Ok(())
            }
            "temp-location" => Err(PropError::ReadOnly),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "temp-template" => Some(PropValue::Str(self.temp_template.clone())),
            "temp-location" => Some(PropValue::Str(
                self.temp_location
                    .as_ref()
                    .map(|p| p.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            )),
            "temp-remove" => Some(PropValue::Bool(self.temp_remove)),
            "max-size-bytes" => Some(PropValue::Uint(self.max_size_bytes)),
            "blocksize" => Some(PropValue::Uint(self.blocksize as u64)),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.spill(slice)?;
                    self.post_level();
                    self.serve(out).await?;
                }
                // Upstream ended: the spill is now the whole stream, so publish
                // its length (a seeking demuxer bounds its guesses with it), then
                // serve the tail (and any seek that lands as it drains). The
                // runner forwards the EOS.
                PipelinePacket::Eos => {
                    if let Some(ctl) = self.seek.as_ref() {
                        ctl.set_stream_len(self.write_pos);
                    }
                    self.serve(out).await?
                }
                // The runner already mapped this through our constraint.
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// `DownloadBuffer`'s settable properties, named after `downloadbuffer`'s.
static DOWNLOADBUFFER_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "temp-template",
        PropKind::Str,
        "spill file name template, a trailing XXXXXX is replaced, a bare name lands in the system temp dir",
    )
    .with_default("g2g-downloadbufferXXXXXX"),
    PropertySpec::new("temp-location", PropKind::Str, "the spill file in use").read_only(),
    PropertySpec::new(
        "temp-remove",
        PropKind::Bool,
        "delete the spill file when the element drops",
    )
    .with_default("true"),
    PropertySpec::new(
        "max-size-bytes",
        PropKind::Uint,
        "spilled bytes at which Buffering reports 100 (0 = post nothing)",
    )
    .with_default("2097152")
    .with_range("0", "1099511627776"),
    PropertySpec::new(
        "blocksize",
        PropKind::Uint,
        "bytes per DataFrame served from the spill file",
    )
    .with_default("65536")
    .with_range("1", "1073741824"),
];

impl PadTemplates for DownloadBuffer {
    fn pad_templates() -> Vec<PadTemplate> {
        let mapping = DownloadBuffer::mapping();
        let sink = CapsSet::from_alternatives(
            mapping
                .iter()
                .flat_map(|(i, _)| i.alternatives().to_vec())
                .collect(),
        );
        let mut outputs: Vec<Caps> = Vec::new();
        for (_, o) in &mapping {
            for caps in o.alternatives() {
                if !outputs.contains(caps) {
                    outputs.push(caps.clone());
                }
            }
        }
        Vec::from([
            PadTemplate::sink(sink),
            PadTemplate::source(CapsSet::from_alternatives(outputs)),
        ])
    }
}
