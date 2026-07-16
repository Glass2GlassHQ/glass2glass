//! M87: `Buffering` bus messages from link occupancy. With a bus attached via
//! `run_graph_with_bus`, the sink arm samples its input link fill and posts a
//! `BusMessage::Buffering { percent }` on each quartile crossing. g2g has no
//! `queue` element, so this reports the bounded link channel's own occupancy.
//!
//! Exact percents are timing-dependent under the runtime, so we assert the
//! deterministic guarantees: at least one report is posted (the first sink
//! iteration samples the link before it is full), and every percent is 0..=100.
//! The quartile-band logic and `fill_percent` are unit-tested in g2g-core.

use g2g_core::runtime::{run_graph_with_bus, GraphNode};
use g2g_core::{Bus, BusMessage, Graph, PipelineClock};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn sink_posts_buffering_levels_to_the_bus() {
    let (bus, handle) = Bus::new(64);

    let stats = {
        let mut g: Graph<GraphNode> = Graph::new();
        let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, 8)));
        let sink = g.add_sink(GraphNode::element(FakeSink::new()));
        g.link(src, sink).unwrap();
        run_graph_with_bus(g, &NullClock, 4, &handle).await.expect("graph runs with bus")
    };
    assert_eq!(stats.frames_consumed, 8);

    let mut levels = Vec::new();
    while let Some(m) = bus.try_recv() {
        if let BusMessage::Buffering { percent } = m {
            assert!(percent <= 100, "fill percent in range");
            levels.push(percent);
        }
    }
    assert!(!levels.is_empty(), "the sink posts at least one buffering level");
}
