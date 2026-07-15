//! M18 β — cross-element allocation re-cascade (single hop).
//!
//! α (m18_alpha_realloc) re-derives an element's *own* pool on a
//! mid-stream caps change. β adds the cross-element step the startup
//! cascade does once at setup: the sink's re-derived `propose_allocation`
//! answer flows one hop upstream to the transform's `configure_allocation`
//! (DESIGN-M16-workaround3-reconfigure.md §9.4 β, §9.4.1). The runner now
//! routes this through the coordinator: the sink arm reports the applied
//! `CapsChanged` plus its proposal, and the coordinator forwards an
//! `ArmDirective::Recascade` to the transform arm, which selects on the
//! control channel alongside its data link.
//!
//! The transform here is a clean probe for the *combined* cascade: its
//! `configure_allocation` is called once at startup (the sink's initial
//! proposal) and once more per mid-stream caps change (β). So a single
//! geometry change records exactly `[startup_size, new_size]`, and no
//! change records only `[startup_size]`.

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
    RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Pool size as a pure function of caps geometry, so a geometry change is
/// observable as a different recorded size.
fn geometry_size(caps: &Caps) -> Option<usize> {
    match caps.dims()? {
        (Dim::Fixed(w), Dim::Fixed(h), _) => Some(*w as usize * *h as usize),
        _ => None,
    }
}

/// Scripted source: emits `initial` caps, then `target_frames` frames + EOS.
struct NvSource {
    initial: Caps,
    target_frames: u32,
}

impl SourceLoop for NvSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.initial.clone()))
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
                    timing: FrameTiming::default(),
                    sequence: i as u64,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.target_frames as u64)
        })
    }
}

/// Boundary transform that optionally injects one `CapsChanged` downstream
/// after a given input-frame count, and records every `configure_allocation`
/// size it receives. The recorded sizes are the cross-element cascade made
/// visible: startup (the sink's initial proposal) plus one per β hop.
struct Boundary {
    inject: Option<(u32, Caps)>,
    injected: bool,
    alloc_log: Arc<Mutex<Vec<usize>>>,
}

impl AsyncElement for Boundary {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.alloc_log.lock().unwrap().push(params.size_bytes);
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

/// NV12 sink whose pool size is a function of caps geometry.
struct PoolSink;

impl AsyncElement for PoolSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
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

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// A mid-stream geometry change re-cascades the sink's new proposal one hop
/// upstream to the transform. The transform records the startup proposal
/// (640x480) and then the β proposal (1920x1080), in order.
#[tokio::test]
async fn mid_stream_caps_change_recascades_to_transform() {
    let alloc_log = Arc::new(Mutex::new(Vec::new()));
    let mut src = NvSource {
        initial: nv12_caps(640, 480),
        target_frames: 4,
    };
    let mut tx = Boundary {
        inject: Some((2, nv12_caps(1920, 1080))),
        injected: false,
        alloc_log: Arc::clone(&alloc_log),
    };
    let mut snk = PoolSink;
    let clock = ZeroClock;

    g2g_core::runtime::run_source_transform_sink(&mut src, &mut tx, &mut snk, &clock, 8)
        .await
        .expect("pipeline completes");

    let log = alloc_log.lock().unwrap();
    assert_eq!(
        *log,
        vec![640 * 480, 1920 * 1080],
        "transform records the startup proposal then the β re-cascade \
         proposal from the sink's new geometry"
    );
}

/// No mid-stream change: the transform is configured once, at startup. β
/// adds nothing without a caps change, proving the second entry above is
/// the re-cascade and not a duplicate of startup.
#[tokio::test]
async fn no_caps_change_leaves_transform_at_startup_proposal() {
    let alloc_log = Arc::new(Mutex::new(Vec::new()));
    let mut src = NvSource {
        initial: nv12_caps(1280, 720),
        target_frames: 5,
    };
    let mut tx = Boundary {
        inject: None,
        injected: false,
        alloc_log: Arc::clone(&alloc_log),
    };
    let mut snk = PoolSink;
    let clock = ZeroClock;

    g2g_core::runtime::run_source_transform_sink(&mut src, &mut tx, &mut snk, &clock, 8)
        .await
        .expect("pipeline completes");

    let log = alloc_log.lock().unwrap();
    assert_eq!(
        *log,
        vec![1280 * 720],
        "no mid-stream change means only the startup cascade configures the \
         transform; β never fires"
    );
}
