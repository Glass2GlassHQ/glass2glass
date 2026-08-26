//! Raw file-descriptor source and sink (`fdsrc` / `fdsink`): read from an
//! already-open descriptor, write to another, so a g2g pipeline sits in a shell
//! pipe or takes a descriptor from the process that spawned it.
//!
//! **Ownership:** the descriptor belongs to whoever opened it. These elements
//! borrow it for the run and never close it, so every `File` built over one is
//! wrapped in `ManuallyDrop`: a plain `File` would close a descriptor its
//! owner still holds. Unix only, since a `RawFd` is what the `fd` property
//! names.
//!
//! Reads and writes are the blocking `std::fs::File` ones inside the async
//! loop, as `filesrc` / `filesink` do.

use core::future::Future;
use core::mem::ManuallyDrop;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use std::fs::File;
use std::io::{Read, Write};
use std::os::fd::{FromRawFd, RawFd};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::filesink::io_err;

/// gst `fdsrc`'s default descriptor: standard input.
const STDIN_FD: RawFd = 0;

/// gst `fdsink`'s default descriptor: standard output.
const STDOUT_FD: RawFd = 1;

/// gst `basesrc`'s `blocksize` default.
const DEFAULT_BLOCKSIZE: usize = 4096;

/// Largest `blocksize` accepted, gst's own bound (`blocksize` is a 32-bit
/// unsigned int).
const MAX_BLOCKSIZE: u64 = u32::MAX as u64;

/// The descriptor as a `File` that will not close it. See the module's
/// ownership note.
fn borrowed_file(fd: RawFd) -> ManuallyDrop<File> {
    // SAFETY: `fd` is a descriptor the caller opened and keeps open across the
    // run. `ManuallyDrop` means the `File` is never dropped, so the ownership
    // `from_raw_fd` would otherwise take is given straight back: nothing here
    // closes the descriptor.
    ManuallyDrop::new(unsafe { File::from_raw_fd(fd) })
}

/// Read the `fd` property, rejecting a negative descriptor.
fn set_fd(target: &mut RawFd, value: &PropValue) -> Result<(), PropError> {
    let fd = value.as_int().ok_or(PropError::Type)?;
    if fd < 0 || fd > RawFd::MAX as i64 {
        return Err(PropError::Value);
    }
    *target = fd as RawFd;
    Ok(())
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::fd::FdSrc;
///
/// // gst-launch equivalent: fdsrc fd=3 blocksize=512
/// let source = FdSrc::new(3).with_blocksize(512);
/// ```
#[derive(Debug)]
pub struct FdSrc {
    fd: RawFd,
    blocksize: usize,
    /// Chunks to emit before EOS; `u64::MAX` is unlimited (the descriptor's own
    /// EOF ends the stream first in every finite case).
    target_chunks: u64,
    configured: bool,
}

impl Default for FdSrc {
    /// Standard input, gst `fdsrc`'s default descriptor.
    fn default() -> Self {
        Self::new(STDIN_FD)
    }
}

impl FdSrc {
    pub fn new(fd: RawFd) -> Self {
        Self {
            fd,
            blocksize: DEFAULT_BLOCKSIZE,
            target_chunks: u64::MAX,
            configured: false,
        }
    }

    /// Bytes per emitted `DataFrame`. Clamped to 1 so a misconfigured zero
    /// cannot spin without progress.
    pub fn with_blocksize(mut self, bytes: usize) -> Self {
        self.blocksize = bytes.max(1);
        self
    }

    /// The type an untyped `filesrc` gives a raw byte stream. A descriptor
    /// carries no container declaration, so put a `typefind` after it to type
    /// the content.
    fn caps() -> Caps {
        Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        }
    }
}

impl SourceLoop for FdSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(Self::caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(Self::caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut file = borrowed_file(self.fd);
            let mut sequence = 0u64;
            while sequence < self.target_chunks {
                let mut buf = vec![0u8; self.blocksize];
                let mut filled = 0usize;
                // A reader may return short reads; fill the chunk until EOF so
                // every frame but the last is exactly blocksize.
                while filled < buf.len() {
                    let n = file.read(&mut buf[filled..]).map_err(io_err)?;
                    if n == 0 {
                        break;
                    }
                    filled += n;
                }
                if filled == 0 {
                    break;
                }
                buf.truncate(filled);
                // Source-side wall-clock stamp so a sink can record
                // glass-to-glass latency, as filesrc does.
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                    FrameTiming {
                        arrival_ns,
                        ..FrameTiming::default()
                    },
                    sequence,
                );
                sequence += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(sequence)
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FDSRC_PROPS
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Filedescriptor source",
            "Source/File",
            "Reads from an open file descriptor as a byte stream",
            "g2g",
        )
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "fd" => set_fd(&mut self.fd, &value),
            "blocksize" => {
                let bytes = value.as_uint().ok_or(PropError::Type)?;
                if bytes == 0 || bytes > MAX_BLOCKSIZE {
                    return Err(PropError::Value);
                }
                self.blocksize = bytes as usize;
                Ok(())
            }
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.target_chunks, &value),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "fd" => Some(PropValue::Int(self.fd as i64)),
            "blocksize" => Some(PropValue::Uint(self.blocksize as u64)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.target_chunks)),
            _ => None,
        }
    }
}

/// `FdSrc`'s settable properties, named and defaulted as gst `fdsrc`.
static FDSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new("fd", PropKind::Int, "open file descriptor to read from")
        .with_range("0", "2147483647")
        .with_default("0"),
    PropertySpec::new(
        "blocksize",
        PropKind::Uint,
        "bytes per emitted DataFrame chunk",
    )
    .with_range("1", "4294967295")
    .with_default("4096"),
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "chunks to emit then EOS (-1 = until EOF)",
    )
    .with_range("-1", "9223372036854775807")
    .with_default("-1"),
];

/// # Example
///
/// ```no_run
/// use g2g_plugins::fd::FdSink;
///
/// // gst-launch equivalent: fdsink fd=1
/// let sink = FdSink::new(1);
/// ```
#[derive(Debug)]
pub struct FdSink {
    fd: RawFd,
    bytes_written: u64,
    configured: bool,
}

impl Default for FdSink {
    /// Standard output, gst `fdsink`'s default descriptor.
    fn default() -> Self {
        Self::new(STDOUT_FD)
    }
}

impl FdSink {
    pub fn new(fd: RawFd) -> Self {
        Self {
            fd,
            bytes_written: 0,
            configured: false,
        }
    }

    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

impl AsyncElement for FdSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// Wildcard sink: a descriptor takes whatever bytes arrive.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Filedescriptor sink",
            "Sink/File",
            "Writes incoming buffers to an open file descriptor",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut file = borrowed_file(self.fd);
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let bytes = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    file.write_all(bytes).map_err(io_err)?;
                    self.bytes_written += bytes.len() as u64;
                }
                PipelinePacket::Eos => {
                    file.flush().map_err(io_err)?;
                }
                // A raw descriptor has nothing to reset on a flush, carries no
                // caps, and takes no segment.
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FDSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "fd" => set_fd(&mut self.fd, &value),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "fd" => Some(PropValue::Int(self.fd as i64)),
            _ => None,
        }
    }
}

/// `FdSink`'s settable properties, named and defaulted as gst `fdsink`.
static FDSINK_PROPS: &[PropertySpec] =
    &[
        PropertySpec::new("fd", PropKind::Int, "open file descriptor to write to")
            .with_range("0", "2147483647")
            .with_default("1"),
    ];

impl PadTemplates for FdSink {
    /// Wildcard sink, matching the runtime `AcceptsAny` constraint.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}
