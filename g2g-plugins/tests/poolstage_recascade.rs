//! The mid-stream β allocation re-cascade driving a *real* resizable pool.
//!
//! `m18_beta_recascade` proved the β control path with a probe transform that
//! only *recorded* the proposal size. This drives the same path through
//! `PoolStage`, a real element that rebuilds a live [`BufferPool`] on each
//! `configure_allocation`, and asserts the pool actually resized: a mid-stream
//! geometry change re-cascades the sink's new proposal to the stage, which
//! rebuilds its pool to the new size while frames keep flowing through it.
//!
//! Pipeline: scripted source (emits a `CapsChanged` mid-stream) -> `PoolStage`
//! -> geometry-sized sink.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate,
    RawVideoFormat,
};
use g2g_plugins::poolstage::PoolStage;

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

/// The sink's pool size is the frame's pixel count (geometry), so a geometry
/// change yields a distinct proposal the β re-cascade carries upstream.
fn geometry_size(caps: &Caps) -> Option<usize> {
    match caps.dims()? {
        (Dim::Fixed(w), Dim::Fixed(h), _) => Some(*w as usize * *h as usize),
        _ => None,
    }
}

/// Scripted source: `initial` caps, then `before` frames, a `CapsChanged` to
/// `switch_to`, then `after` frames, EOS. Frames are small fixed-size System
/// buffers (they fit either pool size, so they always stage).
struct ScriptedSource {
    initial: Caps,
    switch_to: Caps,
    before: u32,
    after: u32,
}

impl SourceLoop for ScriptedSource {
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
        let switch_to = self.switch_to.clone();
        Box::pin(async move {
            let total = self.before + self.after;
            for i in 0..total {
                if i == self.before {
                    out.push(PipelinePacket::CapsChanged(switch_to.clone())).await?;
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        vec![i as u8; 64].into_boxed_slice(),
                    )),
                    timing: FrameTiming::default(),
                    sequence: i as u64,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(total as u64)
        })
    }
}

/// NV12 sink whose allocation proposal is a function of the caps geometry.
struct GeometrySink;

impl AsyncElement for GeometrySink {
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

#[tokio::test]
async fn mid_stream_recascade_resizes_the_live_pool() {
    let (w0, h0) = (640u32, 480u32);
    let (w1, h1) = (1920u32, 1080u32);

    let mut src = ScriptedSource {
        initial: nv12_caps(w0, h0),
        switch_to: nv12_caps(w1, h1),
        before: 2,
        after: 2,
    };
    let mut stage = PoolStage::new();
    let mut snk = GeometrySink;
    let clock = ZeroClock;

    g2g_core::runtime::run_source_transform_sink(&mut src, &mut stage, &mut snk, &clock, 8)
        .await
        .expect("pipeline completes");

    // The pool was built at startup (640x480) and rebuilt once on the β
    // re-cascade from the mid-stream 1920x1080 change: two distinct shapes.
    assert_eq!(stage.reconfigures(), 2, "startup build + one mid-stream β rebuild");
    assert_eq!(
        stage.pool_shape(),
        Some((1, (w1 * h1) as usize)),
        "the live pool ended sized to the new geometry, not the startup one"
    );
    assert_eq!(stage.pool_capacity(), Some(1));
    // Every frame fit and was staged through the pool (real acquisition / reuse).
    assert_eq!(stage.staged(), 4, "all four frames flowed through the pool");
}
