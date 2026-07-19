//! M18 β N-hop — cross-element allocation re-cascade over a multi-element
//! chain (`run_linear_chain`).
//!
//! The single-hop β (m18_beta_recascade) re-cascades the sink's proposal one
//! hop to the lone transform. Over a chain `source -> t0 -> t1 -> sink`, a
//! mid-stream `CapsChanged` must re-cascade through *every* interior element:
//! the sink's proposal reaches `t1`, which re-derives and forwards its own to
//! `t0` (DESIGN-M16-caps-nego.md §13.4 item 4 follow-up). The coordinator
//! drives this reactively, one hop per arm reply.
//!
//! Each interior element records the `configure_allocation` sizes it receives.
//! On a received mid-stream `CapsChanged` an interior element is configured
//! three times: startup (the M12 fold hands it its *downstream* neighbour's
//! proposal), α (element-local, it re-derives its *own* pool when it forwards
//! the change), then β (the coordinator delivers its downstream neighbour's
//! re-derived proposal). To tell β apart from α, each element proposes a
//! distinct constant marker, so the recorded sequence pins exactly which
//! neighbour's proposal arrived when. The change is emitted early so the
//! reactive cascade completes well before EOS.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_linear_chain, SourceLoop};
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

fn nv12(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Source: emits NV12(640x480) caps, then after the first frame pushes a
/// `CapsChanged` to NV12(1920x1080), then the rest of the frames + EOS.
struct NvSource {
    total_frames: u32,
}

impl SourceLoop for NvSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(nv12(640, 480)))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..self.total_frames {
                out.push(PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                    timing: FrameTiming::default(),
                    sequence: i as u64,
                    meta: Default::default(),
                }))
                .await?;
                if i == 0 {
                    out.push(PipelinePacket::CapsChanged(nv12(1920, 1080)))
                        .await?;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.total_frames as u64)
        })
    }
}

/// Interior pass-through. Proposes a fixed per-element `marker` (so a recorded
/// size identifies *which* element's proposal it came from) and records every
/// `configure_allocation` size, making the cascade through it traceable.
struct RecordingTransform {
    marker: usize,
    log: Arc<Mutex<Vec<usize>>>,
}

impl AsyncElement for RecordingTransform {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        Some(AllocationParams::system(self.marker, 1))
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.log.lock().unwrap().push(params.size_bytes);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
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

    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        // Distinct marker (see `RecordingTransform`); the value the sink's
        // β proposal carries to its immediate upstream, `t1`.
        Some(AllocationParams::system(100, 1))
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// The mid-stream change re-cascades through both interior elements. Markers:
/// sink proposes 100, t1 proposes 200, t0 proposes 300. For each element the
/// recorded sequence is `[startup, α, β]`:
///   - startup: the M12 fold hands it its downstream neighbour's proposal.
///   - α: forwarding the `CapsChanged`, it re-derives its own (its marker).
///   - β: the coordinator delivers its downstream neighbour's proposal.
/// So `t1` ends on 100 (the sink's marker reached it) and `t0` ends on 200
/// (t1's marker reached it) — the cascade walked sink -> t1 -> t0.
#[tokio::test]
async fn mid_stream_change_recascades_through_every_interior_element() {
    let t0_log = Arc::new(Mutex::new(Vec::new()));
    let t1_log = Arc::new(Mutex::new(Vec::new()));
    // The change is emitted after frame 0; the remaining frames give the
    // reactive cascade ample room to walk sink -> t1 -> t0 before EOS.
    let mut src = NvSource { total_frames: 40 };
    let mut t0 = RecordingTransform {
        marker: 300,
        log: Arc::clone(&t0_log),
    };
    let mut t1 = RecordingTransform {
        marker: 200,
        log: Arc::clone(&t1_log),
    };
    let mut sink = PoolSink;
    let clock = ZeroClock;

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut t0, &mut t1];
    let stats = run_linear_chain(&mut src, transforms, &mut sink, &clock, 4)
        .await
        .expect("chain runs");

    assert_eq!(stats.frames_consumed, 40);
    assert_eq!(
        *t1_log.lock().unwrap(),
        vec![100, 200, 100],
        "t1: startup=sink(100), α=self(200), β=sink(100)"
    );
    assert_eq!(
        *t0_log.lock().unwrap(),
        vec![200, 300, 200],
        "t0: startup=t1(200), α=self(300), β=t1(200) — the cascade reached the furthest element"
    );
}
