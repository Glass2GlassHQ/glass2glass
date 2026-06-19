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

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelinePacket,
};

use crate::filesink::io_err;

/// Default read chunk size: large enough to amortize syscalls, small enough
/// that a parser downstream sees steady progress.
const DEFAULT_CHUNK_SIZE: usize = 64 * 1024;

#[derive(Debug)]
pub struct FileSrc {
    path: PathBuf,
    caps: Caps,
    chunk_size: usize,
    configured: bool,
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
            chunk_size: DEFAULT_CHUNK_SIZE,
            configured: false,
        }
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
        core::future::ready(Ok(self.caps.clone()))
    }

    /// Produces exactly the caller-declared caps. Synchronous override (no
    /// I/O during negotiation; the file is opened in `run`).
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps.clone()))))
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

            let mut file = File::open(&self.path).map_err(io_err)?;
            let mut sequence = 0u64;
            loop {
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
}
