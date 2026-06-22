//! M209 flattening bins: a `Bin` (a reusable subgraph with ghost pads) flattens
//! into a host `Graph` via `add_bin` and runs end to end through `run_graph`.
//! Because flattening is construction-time only (no new `NodeKind`), the runner
//! drives the bin's interior nodes as first-class host nodes, exactly as if they
//! had been added to the host directly. Pure-fake / software elements, no
//! hardware.

use core::future::Future;
use core::pin::Pin;

use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{
    AsyncElement, Bin, Caps, CapsConstraint, ConfigureOutcome, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket,
};
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::videocrop::VideoCrop;
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Accepts any packet without `FakeSink`'s monotonic-sequence assumption (a tee
/// branch can reorder), counted via `RunStats`.
#[derive(Default)]
struct AnySink;

impl AsyncElement for AnySink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
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
async fn flattened_bin_runs_source_to_sink() {
    // A bin wrapping flip -> crop, exposing flip's input and crop's output as
    // ghost pads, sits between a source and a sink: src -> [bin] -> sink.
    let mut bin: Bin<GraphNode> = Bin::new();
    let flip = bin.add_transform(GraphNode::element(VideoFlip::new(FlipMethod::Rotate180)));
    let crop = bin.add_transform(GraphNode::element(VideoCrop::new(0, 4, 0, 4)));
    bin.link(flip, crop).unwrap();
    bin.ghost_input(flip).unwrap();
    bin.ghost_output(crop).unwrap();

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 4)));
    let inst = g.add_bin(bin);
    let sink = g.add_sink(GraphNode::element(AnySink));
    g.link(src, inst.input(0)).unwrap();
    g.link(inst.output(0), sink).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("flattened bin runs");
    assert_eq!(stats.frames_emitted, 4, "source emitted 4 frames");
    assert_eq!(stats.frames_consumed, 4, "all 4 flowed through the bin's interior to the sink");
}

#[tokio::test]
async fn bin_with_ghosted_tee_fans_out_after_flattening() {
    // The bin exposes two ghost outputs off an interior tee: each source frame
    // reaches both sinks once flattened. Proves a multi-output ghost bin (and the
    // interior tee) runs through the runner unchanged. src -> [id -> tee(2)] -> {a, b}.
    let mut bin: Bin<GraphNode> = Bin::new();
    let id = bin.add_transform(GraphNode::element(IdentityTransform::new()));
    let tee = bin.add_tee(2);
    bin.link(id, tee.input()).unwrap();
    bin.ghost_input(id).unwrap();
    bin.ghost_output(tee.out(0)).unwrap();
    bin.ghost_output(tee.out(1)).unwrap();

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 3)));
    let inst = g.add_bin(bin);
    let a = g.add_sink(GraphNode::element(AnySink));
    let b = g.add_sink(GraphNode::element(AnySink));
    g.link(src, inst.input(0)).unwrap();
    g.link(inst.output(0), a).unwrap();
    g.link(inst.output(1), b).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("ghosted-tee bin runs");
    assert_eq!(stats.frames_emitted, 3, "source emitted 3 frames");
    assert_eq!(stats.frames_consumed, 6, "the ghosted tee broadcast each frame to both sinks");
}
