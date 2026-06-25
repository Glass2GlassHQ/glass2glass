//! M282: negotiated caps in the DOT dump. `negotiate_graph` runs the startup
//! caps solve (source probe + whole-graph CSP) without running the pipeline and
//! returns the per-edge fixated caps; `ValidatedGraph::to_dot` then renders them
//! on the edges. This is what `g2g-launch --dot` does, so the dump shows the
//! caps that *got chosen*, not just the topology.

#![cfg(feature = "std")]

use g2g_core::runtime::{negotiate_graph, parse_launch};
use g2g_core::DotAnnotations;
use g2g_plugins::registry::default_registry;

#[tokio::test]
async fn dot_dump_carries_negotiated_caps() {
    let reg = default_registry();
    let graph = parse_launch(&reg, "videotestsrc num-buffers=1 ! videoconvert ! fakesink")
        .expect("pipeline parses");

    let (vg, caps) = negotiate_graph(graph).await.expect("negotiation succeeds");
    // One fixated caps per edge (a 3-element chain has 2 links).
    assert_eq!(caps.len(), 2);

    let dot = vg.to_dot(
        "pipeline",
        |n| vg.element(n).map(|e| e.log_category().to_string()),
        &DotAnnotations { edge_caps: Some(&caps), edge_memory: None },
    );

    // Both edges carry the chosen caps as their label (videotestsrc defaults to
    // RGBA, which passes through videoconvert to the wildcard sink).
    assert_eq!(dot.matches("label=\"video/x-raw,format=RGBA").count(), 2, "{dot}");
    assert!(dot.contains("VideoTestSrc") && dot.contains("FakeSink"), "{dot}");
}

#[tokio::test]
async fn negotiation_failure_is_reported_not_panicked() {
    // videotestsrc (video) into a capsfilter pinned to audio: no overlap.
    let reg = default_registry();
    let graph = parse_launch(&reg, "videotestsrc ! capsfilter caps=audio/x-raw ! fakesink")
        .expect("pipeline parses");
    assert!(negotiate_graph(graph).await.is_err(), "video -> audio must fail to negotiate");
}
