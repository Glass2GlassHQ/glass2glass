//! M18 Phase C fan-out — per-branch downstream re-solve (FO-2) with the
//! FO-1 strict failure default, driven through `run_source_fanout`.
//!
//! A mid-stream `CapsChanged` is broadcast to every branch (`Router`).
//! Before FO-2 each branch applied it with the adjacent-only
//! `configure_pipeline` (Phase B's solver gate was not run in the fan-out
//! runner). FO-2 re-solves each branch's new caps against that branch's
//! declared `caps_constraint_as_sink()` first. Because each branch runs in
//! its own arm, the re-solves happen concurrently. FO-1 strict default: a
//! branch that rejects the new caps fails the whole fan-out loud (matches
//! GStreamer's `tee` with a rejecting downstream).

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_fanout, SourceLoop};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, Router, VideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::Video {
        format: VideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn nv12(w: u32, h: u32) -> Caps {
    Caps::Video {
        format: VideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn rgba_any() -> Caps {
    Caps::Video {
        format: VideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn frame(seq: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        timing: FrameTiming::default(),
        sequence: seq,
    }
}

/// Source: `before` frames under `initial`, a mid-stream
/// `CapsChanged(switch_to)`, `after` frames under `switch_to`, then EOS.
struct ReconfigSrc {
    initial: Caps,
    switch_to: Caps,
    before: u64,
    after: u64,
    configured: bool,
}

impl SourceLoop for ReconfigSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn intercept_caps(&self) -> Result<Caps, G2gError> {
        Ok(self.initial.clone())
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        assert!(self.configured, "runner must configure source before run");
        let switch_to = self.switch_to.clone();
        let before = self.before;
        let after = self.after;
        Box::pin(async move {
            for i in 0..before {
                out.push(PipelinePacket::DataFrame(frame(i))).await?;
            }
            out.push(PipelinePacket::CapsChanged(switch_to.clone())).await?;
            for j in 0..after {
                out.push(PipelinePacket::DataFrame(frame(before + j)))
                    .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(before + after)
        })
    }
}

#[derive(Default)]
struct BranchInner {
    caps_changes: Vec<Caps>,
    configured_sizes: Vec<usize>,
    eos: bool,
}

fn geometry_size(caps: &Caps) -> Option<usize> {
    match caps {
        Caps::Video { width: Dim::Fixed(w), height: Dim::Fixed(h), .. } => {
            Some(*w as usize * *h as usize)
        }
        _ => None,
    }
}

/// Branch sink that declares a constraint (`accepts`: `Some` => only that
/// shape; `None` => `AcceptsAny`) and records every `CapsChanged` that
/// actually reaches `process` after FO-2's gate.
struct RecordBranch {
    accepts: Option<Caps>,
    inner: Arc<Mutex<BranchInner>>,
}

impl AsyncElement for RecordBranch {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        match &self.accepts {
            Some(c) => CapsConstraint::Accepts(CapsSet::one(c.clone())),
            None => CapsConstraint::AcceptsAny,
        }
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        geometry_size(caps).map(|size| AllocationParams::system(size, 1))
    }

    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.inner.lock().unwrap().configured_sizes.push(params.size_bytes);
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let inner = Arc::clone(&self.inner);
        Box::pin(async move {
            let mut g = inner.lock().unwrap();
            match packet {
                PipelinePacket::CapsChanged(c) => g.caps_changes.push(c),
                PipelinePacket::Eos => g.eos = true,
                _ => {}
            }
            Ok(())
        })
    }
}

/// FO-2 accept: a mid-stream geometry change every branch declares it can
/// take is re-solved and applied per branch, reaching each branch's
/// `process`.
#[tokio::test]
async fn fo2_accepted_caps_change_propagates_to_every_branch() {
    let inner_a = Arc::new(Mutex::new(BranchInner::default()));
    let inner_b = Arc::new(Mutex::new(BranchInner::default()));

    let mut src = ReconfigSrc {
        initial: rgba(640, 480),
        switch_to: rgba(1920, 1080),
        before: 2,
        after: 2,
        configured: false,
    };
    let mut router = Router::new(2);
    let mut sink_a = RecordBranch { accepts: Some(rgba_any()), inner: Arc::clone(&inner_a) };
    let mut sink_b = RecordBranch { accepts: Some(rgba_any()), inner: Arc::clone(&inner_b) };
    let clock = ZeroClock;

    {
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut sink_a, &mut sink_b];
        run_source_fanout(&mut src, &mut router, sinks, &clock, 4)
            .await
            .expect("fan-out completes when every branch accepts the new caps");
    }

    for (name, inner) in [("A", &inner_a), ("B", &inner_b)] {
        let g = inner.lock().unwrap();
        assert!(g.eos, "branch {name} must see EOS");
        assert_eq!(
            g.caps_changes,
            vec![rgba(1920, 1080)],
            "branch {name} must receive the re-solved CapsChanged"
        );
        // α: each branch re-allocates its own pool under the new caps. The
        // fan-out runner never configures branch allocation at startup, so
        // the single recorded size isolates the per-branch α hook.
        assert_eq!(
            g.configured_sizes,
            vec![1920 * 1080],
            "branch {name} must re-allocate locally under the new caps"
        );
    }
}

/// FO-1 strict: if any branch's declared constraint rejects the mid-stream
/// caps, the whole fan-out fails loud rather than silently feeding that
/// branch a shape it advertised it cannot handle.
#[tokio::test]
async fn fo1_strict_branch_reject_fails_the_fanout() {
    let inner_a = Arc::new(Mutex::new(BranchInner::default()));
    let inner_b = Arc::new(Mutex::new(BranchInner::default()));

    // Source starts RGBA (both branches happy), then switches to NV12.
    let mut src = ReconfigSrc {
        initial: rgba(640, 480),
        switch_to: nv12(640, 480),
        before: 2,
        after: 2,
        configured: false,
    };
    let mut router = Router::new(2);
    // Branch A accepts anything; branch B only RGBA, so it rejects the NV12 switch.
    let mut sink_a = RecordBranch { accepts: None, inner: Arc::clone(&inner_a) };
    let mut sink_b = RecordBranch { accepts: Some(rgba_any()), inner: Arc::clone(&inner_b) };
    let clock = ZeroClock;

    let result = {
        let sinks: Vec<&mut dyn DynAsyncElement> = vec![&mut sink_a, &mut sink_b];
        run_source_fanout(&mut src, &mut router, sinks, &clock, 4).await
    };

    assert!(
        result.is_err(),
        "a branch rejecting the mid-stream caps must fail the fan-out (FO-1 strict)"
    );
    // The rejecting branch never got the bad caps through its process.
    let g = inner_b.lock().unwrap();
    assert!(
        g.caps_changes.iter().all(|c| c != &nv12(640, 480)),
        "rejected NV12 caps must not reach branch B's process (got {:?})",
        g.caps_changes
    );
}
