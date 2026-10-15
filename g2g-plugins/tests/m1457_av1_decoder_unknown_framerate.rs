#![cfg(all(feature = "std", feature = "rav1d"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::Mutex;

use g2g_core::runtime::{run_graph, GraphNode, LatencyProfile};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, Graph, OutputSink,
    PipelinePacket, Rate,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::mp4src::Mp4Src;
use g2g_plugins::rav1ddec::Rav1dDec;

// av1_321x181.obu stream-copied with a 24 fps then 12 fps sample spacing
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av1_321x181_vfr.mp4"
);

static LAST_CAPS_CHANGE: Mutex<Option<Caps>> = Mutex::new(None);

struct CapsProbe;

impl AsyncElement for CapsProbe {
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
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::CapsChanged(caps) = packet {
            *LAST_CAPS_CHANGE.lock().unwrap() = Some(caps);
        }
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn av1_decoder_reports_no_rate_its_input_never_declared() {
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(Mp4Src::new(FIXTURE)));
    let decoder = graph.add_transform(GraphNode::element(Rav1dDec::new()));
    let sink = graph.add_sink(GraphNode::element(CapsProbe));
    graph.link(source, decoder).unwrap();
    graph.link(decoder, sink).unwrap();
    let stats = run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
        .await
        .expect("mp4src ! rav1ddec runs");
    assert!(stats.frames_consumed > 0, "decoded frames reach the sink");
    let last_caps = LAST_CAPS_CHANGE
        .lock()
        .unwrap()
        .clone()
        .expect("the sink saw the decoded caps");
    let Caps::RawVideo { framerate, .. } = last_caps else {
        panic!("raw video at the sink, got {last_caps:?}");
    };
    assert_eq!(
        framerate,
        Rate::Any,
        "no frame rate invented for the stream"
    );
}
