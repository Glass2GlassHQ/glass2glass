//! DAG runner D3: `run_graph` over arbitrary source / transform / sink / tee
//! topologies, negotiated by `solve_graph` (D2) and driven by one arm per node.
//! Pure-fake elements (no hardware); the `rtspsrc -> tee -> {decode, mux}`
//! integration is owed a Linux run.

use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{Caps, Dim, G2gError, Graph, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::capsfilter::CapsFilter;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videocrop::VideoCrop;
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn linear_chain_flows_through_run_graph() {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let flip = g.add_transform(GraphNode::element(VideoFlip::new(FlipMethod::Rotate180)));
    let sink = g.add_sink(GraphNode::element(FakeSink::new()));
    g.link(src, flip).unwrap();
    g.link(flip, sink).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("linear DAG runs");
    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(stats.frames_consumed, 4);
}

#[tokio::test]
async fn tee_fans_out_to_two_sinks() {
    // src -> tee(2) -> {sink, sink}. The tee deep-copies each System frame to
    // both branches, so each sink consumes all 4 frames.
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let tee = g.add_tee(2);
    let s0 = g.add_sink(GraphNode::element(FakeSink::new()));
    let s1 = g.add_sink(GraphNode::element(FakeSink::new()));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), s0).unwrap();
    g.link(tee.out(1), s1).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("tee diamond runs");
    assert_eq!(stats.frames_emitted, 4, "source emitted 4 frames");
    assert_eq!(stats.frames_consumed, 8, "each of the 2 sinks consumed all 4");
}

#[tokio::test]
async fn tee_branches_run_independent_transforms() {
    // src -> tee(2) -> {flip -> sink, crop -> sink}. Each branch negotiates and
    // transforms independently off the shared source caps.
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let tee = g.add_tee(2);
    let flip = g.add_transform(GraphNode::element(VideoFlip::new(FlipMethod::Rotate90Cw)));
    let crop = g.add_transform(GraphNode::element(VideoCrop::new(2, 2, 2, 2)));
    let s0 = g.add_sink(GraphNode::element(FakeSink::new()));
    let s1 = g.add_sink(GraphNode::element(FakeSink::new()));
    g.link(src, tee.input()).unwrap();
    g.link(tee.out(0), flip).unwrap();
    g.link(tee.out(1), crop).unwrap();
    g.link(flip, s0).unwrap();
    g.link(crop, s1).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("tee + per-branch transforms run");
    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(stats.frames_consumed, 8, "both branches delivered all 4 frames");
}

#[tokio::test]
async fn incompatible_branch_fails_negotiation() {
    // An RGBA source into an NV12-pinned filter has no overlap, so the
    // whole-graph solve fails loud before any data flows.
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let nv12_only = CapsFilter::new(Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    });
    let filter = g.add_transform(GraphNode::element(nv12_only));
    let sink = g.add_sink(GraphNode::element(FakeSink::new()));
    g.link(src, filter).unwrap();
    g.link(filter, sink).unwrap();

    let result = run_graph(g, &NullClock, 4).await;
    assert_eq!(result.err(), Some(G2gError::CapsMismatch), "RGBA into NV12 filter must fail");
}
