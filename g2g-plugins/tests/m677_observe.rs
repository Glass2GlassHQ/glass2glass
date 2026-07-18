//! M677 live telemetry tap: `run_graph_observed` shares the running graph's
//! topology + per-element probes with an [`Observer`], so a dashboard / TUI can
//! read live `process()` latency and input-link fill while the pipeline runs.
//!
//! std-gated (uses `default_registry` + the graph runner): run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph_observed, NodeRole, Observer};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn observed_run_captures_topology_and_per_element() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=8 ! videoscale width=160 height=120 ! fakesink",
    )
    .expect("pipeline parses");

    let obs = Observer::new();
    let stats =
        run_graph_observed(graph, &ZeroClock, 4, &obs, None).await.expect("observed run");
    assert_eq!(stats.frames_consumed, 8, "all frames reached the sink");

    let snap = obs.snapshot();
    assert_eq!(snap.nodes.len(), 3, "src -> scale -> sink");
    assert_eq!(
        snap.edges.iter().map(|e| (e.from, e.to)).collect::<Vec<_>>(),
        vec![(0, 1), (1, 2)],
    );
    // Each edge carries its negotiated caps (raw video after the decode-free chain).
    assert!(snap.edges.iter().all(|e| e.caps.is_some()), "edges carry negotiated caps");
    assert!(snap.edges[0].caps.as_ref().unwrap().contains("video"));

    // Source has no `process()` probe; its cost surfaces as downstream fill.
    assert_eq!(snap.nodes[0].role, NodeRole::Source);
    assert!(snap.nodes[0].latency.is_none());

    // The interior transform was probed and timed real frames.
    assert_eq!(snap.nodes[1].role, NodeRole::Transform);
    let scale = snap.nodes[1].latency.as_ref().expect("transform probed");
    assert!(scale.proc.count > 0, "transform process() was timed, count={}", scale.proc.count);

    // The sink saw all 8 frames through its probe too.
    assert_eq!(snap.nodes[2].role, NodeRole::Sink);
    let sink = snap.nodes[2].latency.as_ref().expect("sink probed");
    assert_eq!(sink.proc.count, 8, "sink timed every frame");
}

#[tokio::test]
async fn observer_is_registered_before_frames_flow() {
    // Prove the topology is readable mid-run: `register` happens in the runner's
    // prepare phase, before any arm processes a frame. A poller joined with the
    // run must observe the 3 nodes while the pipeline is still executing.
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=400 ! videoscale width=64 height=64 ! fakesink",
    )
    .expect("pipeline parses");

    let obs = Observer::new();
    let poller = {
        let obs = obs.clone();
        async move {
            for _ in 0..1_000_000 {
                if obs.snapshot().nodes.len() == 3 {
                    return true;
                }
                tokio::task::yield_now().await;
            }
            false
        }
    };

    let (stats, saw_topology) =
        tokio::join!(run_graph_observed(graph, &ZeroClock, 2, &obs, None), poller);
    stats.expect("observed run");
    assert!(saw_topology, "topology became visible while the run was in flight");
}
