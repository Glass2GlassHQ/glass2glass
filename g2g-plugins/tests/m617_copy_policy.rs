//! M617: the runner enforces a memory-domain copy budget as a graph-level contract.
//!
//! `run_graph_with_copy_policy` runs the copy plan after negotiation and refuses to
//! start a graph that violates the policy, turning "is this pipeline zero-copy?" into
//! a guarantee checked at construction. This test proves the gate is wired and does
//! not false-positive: an all-System pipeline (no memory-domain transfer) satisfies
//! the strictest `CopyPolicy::DenyAll` and runs to completion. The catching side (a
//! device<->host raw-frame transfer failing `DenyAll`) is unit-tested on the copy
//! plan itself in `g2g_core::copyplan` (`a_host_download_of_a_raw_frame_is_a_counted_copy`).
#![cfg(feature = "std")]

use g2g_core::copyplan::CopyPolicy;
use g2g_core::graph::Graph;
use g2g_core::runtime::{run_graph, run_graph_with_copy_policy, GraphNodeRef};
use g2g_core::{G2gError, PipelineClock};

use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn system_pipeline() -> Graph<GraphNodeRef<'static>> {
    // videotestsrc -> videoflip -> fakesink, all in System memory (no domain change,
    // so the copy plan finds zero frame copies).
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(VideoTestSrc::new(16, 16, 30, 2)));
    let flip = g.add_transform(GraphNodeRef::element(VideoFlip::new(FlipMethod::Rotate180)));
    let sink = g.add_sink(GraphNodeRef::element(FakeSink::new()));
    g.link(src, flip).unwrap();
    g.link(flip, sink).unwrap();
    g
}

#[tokio::test]
async fn system_pipeline_satisfies_deny_all_and_runs() {
    // The strict zero-copy contract passes (nothing crosses a memory domain), so the
    // graph negotiates, clears the gate, and runs.
    let stats = run_graph_with_copy_policy(system_pipeline(), &ZeroClock, 4, CopyPolicy::DenyAll)
        .await
        .expect("an all-System pipeline satisfies DenyAll and runs");
    assert!(
        stats.frames_emitted > 0 || stats.frames_consumed > 0,
        "the pipeline actually ran"
    );
}

#[tokio::test]
async fn at_most_and_allow_also_pass_a_zero_copy_pipeline() {
    // A budget of one copy, and the report-only policy, both admit a zero-copy graph.
    run_graph_with_copy_policy(system_pipeline(), &ZeroClock, 4, CopyPolicy::AtMost(1))
        .await
        .expect("AtMost(1) admits a zero-copy pipeline");
    run_graph_with_copy_policy(system_pipeline(), &ZeroClock, 4, CopyPolicy::Allow)
        .await
        .expect("Allow never rejects");
}

#[tokio::test]
async fn plain_run_graph_is_unaffected() {
    // The default entry point (no policy) runs the same graph, confirming the gate is
    // opt-in and does not change existing behavior.
    let r: Result<_, G2gError> = run_graph(system_pipeline(), &ZeroClock, 4).await;
    assert!(r.is_ok(), "run_graph without a policy is unchanged");
}
