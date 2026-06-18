//! M78 - state machine + preroll over the DAG runner (`run_graph_stateful`).
//!
//! Rolls the M76/M77 flow gate into `run_graph`, so an arbitrary topology
//! honors `NULL → READY → PAUSED → PLAYING`. The load-bearing new behavior is
//! N-sink preroll aggregation: a tee fan-out with two sinks completes its async
//! `Paused` transition (one `AsyncDone`, `await_prerolled` resolves) only once
//! *both* sinks have taken their preroll buffer.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::element::{AsyncElement, BoxFuture, ConfigureOutcome, OutputSink};
use g2g_core::graph::Graph;
use g2g_core::runtime::{run_graph_stateful, GraphNode, StateController};
use g2g_core::{
    Bus, BusMessage, Caps, G2gError, PipelineClock, PipelinePacket, PipelineState,
    StateChangeReturn,
};

use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Sink that bumps a shared counter on every `DataFrame`, readable mid-run.
struct CountingSink {
    seen: Arc<AtomicU64>,
}

impl AsyncElement for CountingSink {
    type ProcessFuture<'a> = BoxFuture<'a, Result<(), G2gError>>;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = packet {
                self.seen.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })
    }
}

/// Non-live tee fan-out: each of the two sinks prerolls exactly one frame in
/// `Paused`; preroll aggregates to a single `AsyncDone`; `Playing` then drains
/// the whole stream to both sinks.
#[tokio::test]
async fn tee_fanout_prerolls_both_sinks_then_plays() {
    let target = 5u64;
    let s0 = Arc::new(AtomicU64::new(0));
    let s1 = Arc::new(AtomicU64::new(0));

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(16, 16, 30, target)));
    let tee = g.add_tee(2);
    let sink0 = g.add_sink(GraphNode::element(CountingSink { seen: s0.clone() }));
    let sink1 = g.add_sink(GraphNode::element(CountingSink { seen: s1.clone() }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), sink0).unwrap();
    g.link(tee.out(1), sink1).unwrap();

    let (bus, handle) = Bus::new(16);
    let ctrl = StateController::with_bus(PipelineState::Ready, handle); // non-live

    let pipeline = run_graph_stateful(g, &ZeroClock, 4, &ctrl);

    let s0d = s0.clone();
    let s1d = s1.clone();
    let ctrl_d = ctrl.clone();
    let driver = async move {
        assert_eq!(
            ctrl_d.set_state(PipelineState::Paused),
            StateChangeReturn::Async
        );
        // Completes only when *both* sinks preroll.
        ctrl_d.await_prerolled().await;
        assert!(ctrl_d.is_prerolled());
        // Each sink took exactly its one preroll frame.
        assert_eq!(s0d.load(Ordering::SeqCst), 1, "sink0 prerolled one frame");
        assert_eq!(s1d.load(Ordering::SeqCst), 1, "sink1 prerolled one frame");
        // Confirm the gate then holds.
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        assert_eq!(s0d.load(Ordering::SeqCst), 1, "sink0 holds after preroll");
        assert_eq!(s1d.load(Ordering::SeqCst), 1, "sink1 holds after preroll");
        ctrl_d.set_state(PipelineState::Playing);
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    res.expect("DAG runs to completion");

    assert_eq!(
        s0.load(Ordering::SeqCst),
        target,
        "sink0 drains all once Playing"
    );
    assert_eq!(
        s1.load(Ordering::SeqCst),
        target,
        "sink1 drains all once Playing"
    );

    // Exactly one AsyncDone for the whole pipeline, despite two sinks.
    let async_dones = std::iter::from_fn(|| bus.try_recv())
        .filter(|m| matches!(m, BusMessage::AsyncDone))
        .count();
    assert_eq!(async_dones, 1, "two sinks aggregate to a single AsyncDone");
}

/// A live DAG reports `NoPreroll` and admits no buffer at either sink until
/// `Playing` (full hold across the fan-out).
#[tokio::test]
async fn live_tee_fanout_full_holds_until_playing() {
    let target = 4u64;
    let s0 = Arc::new(AtomicU64::new(0));
    let s1 = Arc::new(AtomicU64::new(0));

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(16, 16, 30, target)));
    let tee = g.add_tee(2);
    let sink0 = g.add_sink(GraphNode::element(CountingSink { seen: s0.clone() }));
    let sink1 = g.add_sink(GraphNode::element(CountingSink { seen: s1.clone() }));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), sink0).unwrap();
    g.link(tee.out(1), sink1).unwrap();

    let ctrl = StateController::new(PipelineState::Ready);
    ctrl.set_live(true);

    let pipeline = run_graph_stateful(g, &ZeroClock, 4, &ctrl);

    let s0d = s0.clone();
    let s1d = s1.clone();
    let ctrl_d = ctrl.clone();
    let driver = async move {
        assert_eq!(
            ctrl_d.set_state(PipelineState::Paused),
            StateChangeReturn::NoPreroll
        );
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            s0d.load(Ordering::SeqCst),
            0,
            "live Paused admits no buffer"
        );
        assert_eq!(
            s1d.load(Ordering::SeqCst),
            0,
            "live Paused admits no buffer"
        );
        ctrl_d.set_state(PipelineState::Playing);
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    res.expect("DAG runs");
    assert_eq!(s0.load(Ordering::SeqCst), target);
    assert_eq!(s1.load(Ordering::SeqCst), target);
}
