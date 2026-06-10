//! M18 α — element-local re-allocation on a mid-stream caps change.
//!
//! First observable M18 behavior change. Before α, M12 allocation ran
//! only at startup: in a `source → transform → sink` chain the sink's
//! `propose_allocation` answer feeds the *transform* (cross-element),
//! and the sink's own `configure_allocation` is never called. α makes
//! the runner re-derive an element's own pool from the new caps and
//! store it (`propose_allocation` then `configure_allocation` on the
//! same element) whenever a mid-stream `CapsChanged` is applied
//! (DESIGN-M16-workaround3-reconfigure.md §9.4 α). No cross-element
//! cascade, that is β.
//!
//! Because startup never touches the sink's `configure_allocation`, the
//! sink is a clean probe: any recorded call comes solely from α. So:
//!   1. A mid-stream geometry change records exactly one re-allocation,
//!      sized from the *new* caps.
//!   2. With no mid-stream change the sink is never re-allocated.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate,
    VideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::Video {
        format: VideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Scripted source: emits `initial` caps, then `target_frames` frames + EOS.
struct NvSource {
    initial: Caps,
    target_frames: u32,
}

impl SourceLoop for NvSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn intercept_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.initial.clone())
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..self.target_frames {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        vec![i as u8].into_boxed_slice(),
                    )),
                    caps: self.initial.clone(),
                    timing: FrameTiming::default(),
                    sequence: i as u64,
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.target_frames as u64)
        })
    }
}

/// Pass-through transform that optionally injects one `CapsChanged`
/// downstream after a given input-frame count.
struct Boundary {
    inject: Option<(u32, Caps)>,
    injected: bool,
}

impl AsyncElement for Boundary {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let inject = self.inject.clone();
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    let seq = f.sequence as u32;
                    out.push(PipelinePacket::DataFrame(f)).await?;
                    if let Some((after, caps)) = inject {
                        if !self.injected && seq + 1 >= after {
                            out.push(PipelinePacket::CapsChanged(caps)).await?;
                            self.injected = true;
                        }
                    }
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Sink whose own pool size is a function of caps geometry. It records
/// every `configure_allocation` size it is handed, so the test can see
/// exactly when (and at what geometry) α re-allocated it.
#[derive(Default)]
struct PoolLog {
    configured_sizes: Vec<usize>,
}

struct PoolSink {
    log: Arc<Mutex<PoolLog>>,
}

fn geometry_size(caps: &Caps) -> Option<usize> {
    match caps {
        Caps::Video {
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } => Some(*w as usize * *h as usize),
        _ => None,
    }
}

impl AsyncElement for PoolSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::Video {
            format: VideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        geometry_size(caps).map(|size| AllocationParams::system(size, 1))
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.log.lock().unwrap().configured_sizes.push(params.size_bytes);
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// A mid-stream geometry change re-allocates the sink's own pool exactly
/// once, sized from the new caps. Startup never configures the sink's
/// allocation, so the single recorded size isolates the α hook.
#[tokio::test]
async fn mid_stream_caps_change_reallocates_sink_locally() {
    let log = Arc::new(Mutex::new(PoolLog::default()));
    let mut src = NvSource {
        initial: nv12_caps(640, 480),
        target_frames: 4,
    };
    let mut tx = Boundary {
        inject: Some((2, nv12_caps(1920, 1080))),
        injected: false,
    };
    let mut snk = PoolSink {
        log: Arc::clone(&log),
    };
    let clock = ZeroClock;

    g2g_core::runtime::run_source_transform_sink(&mut src, &mut tx, &mut snk, &clock, 8)
        .await
        .expect("pipeline completes");

    let g = log.lock().unwrap();
    assert_eq!(
        g.configured_sizes,
        vec![1920 * 1080],
        "α re-allocates the sink once under the new caps; startup never \
         configures the sink, so 640x480 must not appear"
    );
}

/// No mid-stream caps change: the sink is never re-allocated. Confirms α
/// is the only path that configures the sink's allocation, and that it
/// fires solely on a mid-stream change.
#[tokio::test]
async fn no_caps_change_leaves_sink_allocation_untouched() {
    let log = Arc::new(Mutex::new(PoolLog::default()));
    let mut src = NvSource {
        initial: nv12_caps(1280, 720),
        target_frames: 5,
    };
    let mut tx = Boundary {
        inject: None,
        injected: false,
    };
    let mut snk = PoolSink {
        log: Arc::clone(&log),
    };
    let clock = ZeroClock;

    g2g_core::runtime::run_source_transform_sink(&mut src, &mut tx, &mut snk, &clock, 8)
        .await
        .expect("pipeline completes");

    let g = log.lock().unwrap();
    assert!(
        g.configured_sizes.is_empty(),
        "no mid-stream caps change means α never fires (got {:?})",
        g.configured_sizes
    );
}
