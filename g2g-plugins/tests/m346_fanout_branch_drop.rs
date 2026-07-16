//! M346: graceful per-branch drop on fan-out (`FanOutPolicy::AllowBranchDrop`).
//!
//! A tee built with `add_tee_with_policy(.., AllowBranchDrop)` lets a branch that
//! cannot follow a mid-stream `CapsChanged` fall away (its arm ends, the tee
//! stops broadcasting to it) while the siblings keep flowing. The default
//! (`add_tee` / `FailLoud`) fails the whole run, which `m70_dag_recascade`'s
//! `rejecting_branch_fails_loud` already covers; this is the opt-in success path.
//!
//! Pure-fake elements (no hardware).

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::frame::Frame;
use g2g_core::graph::FanOutPolicy;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    Graph, MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

struct NullClock;
impl PipelineClock for NullClock {
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

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn nv12_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn frame(seq: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![seq as u8].into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: seq,
        meta: Default::default(),
    })
}

/// Emits `before` frames under `initial`, a mid-stream `CapsChanged(switch)`,
/// then `after` frames, then EOS.
struct ScriptedSource {
    initial: Caps,
    switch: Caps,
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
        Box::pin(async move {
            let mut seq = 0u64;
            for _ in 0..self.before {
                out.push(frame(seq)).await?;
                seq += 1;
            }
            out.push(PipelinePacket::CapsChanged(self.switch.clone())).await?;
            for _ in 0..self.after {
                out.push(frame(seq)).await?;
                seq += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

/// Counts the `DataFrame`s it consumes. `accept = None` accepts any caps;
/// `Some(c)` accepts only `c` (and rejects a mid-stream switch away from it).
struct CountingSink {
    accept: Option<Caps>,
    count: Arc<Mutex<u64>>,
}

impl AsyncElement for CountingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        match &self.accept {
            None => CapsConstraint::AcceptsAny,
            Some(c) => CapsConstraint::Accepts(CapsSet::one(c.clone())),
        }
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = packet {
                *self.count.lock().unwrap() += 1;
            }
            Ok(())
        })
    }
}

/// A tee under `AllowBranchDrop`: the NV12-only branch drops out when the source
/// switches to RGBA mid-stream, while the accept-any branch keeps consuming. The
/// run succeeds (no loud failure), the surviving branch sees every frame, and the
/// dropped branch stops at the pre-switch ones.
#[tokio::test]
async fn allow_branch_drop_keeps_surviving_branch_flowing() {
    let any_count = Arc::new(Mutex::new(0u64));
    let nv12_count = Arc::new(Mutex::new(0u64));

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(ScriptedSource {
        initial: nv12(8, 8),
        switch: rgba(8, 8),
        before: 2,
        after: 3,
    }));
    let tee = g.add_tee_with_policy(2, FanOutPolicy::AllowBranchDrop);
    let any = g.add_sink(GraphNode::element(CountingSink {
        accept: None,
        count: Arc::clone(&any_count),
    }));
    let nv12_only = g.add_sink(GraphNode::element(CountingSink {
        accept: Some(nv12_any()),
        count: Arc::clone(&nv12_count),
    }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), any).unwrap();
    g.link(tee.out(1), nv12_only).unwrap();

    let stats = run_graph(g, &NullClock, 8).await.expect("branch drop does not fail the run");
    assert_eq!(
        *any_count.lock().unwrap(),
        5,
        "the accept-any branch consumed all 2 + 3 frames across the switch"
    );
    assert_eq!(
        *nv12_count.lock().unwrap(),
        2,
        "the NV12-only branch dropped at the RGBA switch, keeping only the 2 pre-switch frames"
    );
    assert_eq!(stats.frames_consumed, 7, "5 + 2 across both sinks");
}

/// The same graph under the default `FailLoud` policy fails the whole run loud
/// when the NV12-only branch rejects the switch, proving the opt-in is what
/// changes the behaviour.
#[tokio::test]
async fn fail_loud_is_the_default() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(ScriptedSource {
        initial: nv12(8, 8),
        switch: rgba(8, 8),
        before: 2,
        after: 3,
    }));
    let tee = g.add_tee(2);
    let any = g.add_sink(GraphNode::element(CountingSink {
        accept: None,
        count: Arc::new(Mutex::new(0)),
    }));
    let nv12_only = g.add_sink(GraphNode::element(CountingSink {
        accept: Some(nv12_any()),
        count: Arc::new(Mutex::new(0)),
    }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), any).unwrap();
    g.link(tee.out(1), nv12_only).unwrap();

    let result = run_graph(g, &NullClock, 8).await;
    assert_eq!(result.err(), Some(G2gError::CapsMismatch), "default tee fails loud");
}
