//! Synthetic video source. Emits a deterministic gradient pattern at a
//! fixed framerate in the system memory domain. CPU-only.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    BufferPool, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec, RawVideoFormat,
};

#[derive(Debug)]
pub struct VideoTestSrc {
    width: u32,
    height: u32,
    framerate_q16: u32,
    target_frames: u64,
    configured: bool,
    pool: Option<BufferPool<Box<[u8]>>>,
}

impl VideoTestSrc {
    /// `framerate` is in nominal fps; stored internally as Q16 fixed-point.
    pub fn new(width: u32, height: u32, framerate: u32, target_frames: u64) -> Self {
        Self {
            width,
            height,
            framerate_q16: framerate << 16,
            target_frames,
            configured: false,
            pool: None,
        }
    }

    /// Pool-backed variant: every emitted frame draws its `width * height * 4`
    /// bytes from the pool, and the buffer returns to the pool when the
    /// downstream `Frame` is dropped. The pool's buffer size MUST be at
    /// least `width * height * 4`; this is checked at run time.
    pub fn with_pool(
        width: u32,
        height: u32,
        framerate: u32,
        target_frames: u64,
        pool: BufferPool<Box<[u8]>>,
    ) -> Self {
        Self {
            width,
            height,
            framerate_q16: framerate << 16,
            target_frames,
            configured: false,
            pool: Some(pool),
        }
    }

    fn caps(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.framerate_q16),
        }
    }
}

impl SourceLoop for VideoTestSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    /// M16 step 5f: native `Produces` constraint. The chain is now
    /// fully-native when paired with `AcceptsAny`-migrated sinks
    /// (e.g. `FakeSink`, `syncsink`), exercising the all-native
    /// arc-consistency solver path instead of the mixed cascade.
    /// Synchronous override (no I/O), so we sidestep the default's
    /// `async move` indirection.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(
        &mut self,
        _absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(
        &'a mut self,
        out: &'a mut dyn OutputSink,
    ) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }

            let fps_q16 = u64::from(self.framerate_q16);
            let pts_step_ns: u64 = (1_000_000_000u64 << 16)
                .checked_div(fps_q16)
                .unwrap_or(0);

            let bytes_per_frame = (self.width as usize)
                .checked_mul(self.height as usize)
                .and_then(|n| n.checked_mul(4))
                .ok_or(G2gError::CapsMismatch)?;

            for seq in 0..self.target_frames {
                let domain = if let Some(pool) = &self.pool {
                    let mut buf = pool.acquire().await;
                    if buf.len() < bytes_per_frame {
                        return Err(G2gError::CapsMismatch);
                    }
                    let slice = buf.as_mut();
                    for (i, b) in slice.iter_mut().take(bytes_per_frame).enumerate() {
                        *b = ((i as u64).wrapping_add(seq) & 0xFF) as u8;
                    }
                    MemoryDomain::System(SystemSlice::from_pool(buf))
                } else {
                    let mut buf = vec![0u8; bytes_per_frame].into_boxed_slice();
                    for (i, b) in buf.iter_mut().enumerate() {
                        *b = ((i as u64).wrapping_add(seq) & 0xFF) as u8;
                    }
                    MemoryDomain::System(SystemSlice::from_boxed(buf))
                };

                let pts = seq * pts_step_ns;
                // Source-side wall-clock stamp so downstream sinks can
                // record glass-to-glass latency via
                // `monotonic_ns() - arrival_ns`. Matches the convention
                // used by RtspSrc for production sources. Std-gated
                // because `monotonic_ns` lives behind g2g-core's `std`
                // feature; in no_std builds `arrival_ns` stays zero
                // and downstream sinks silently skip latency recording.
                #[cfg(feature = "std")]
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                #[cfg(not(feature = "std"))]
                let arrival_ns: u64 = 0;
                let frame = Frame {
                    domain,
                    timing: FrameTiming {
                        pts_ns: pts,
                        dts_ns: pts,
                        duration_ns: pts_step_ns,
                        capture_ns: pts,
                        arrival_ns,
                    },
                    sequence: seq,
                };

                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(self.target_frames)
        })
    }
}

impl PadTemplates for VideoTestSrc {
    /// Static superset: the type always produces RGBA at any geometry /
    /// framerate. A constructed instance narrows to its configured dims via
    /// `SourceLoop::caps_constraint`.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))])
    }
}
