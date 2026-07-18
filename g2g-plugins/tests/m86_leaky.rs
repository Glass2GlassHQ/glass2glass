//! M86: leaky `LinkPolicy` end-to-end through the DAG runner. A `DropOldest`
//! link runs to completion without deadlock, and every emitted frame is
//! accounted for, either delivered to the sink or dropped (the conservation
//! invariant, which holds regardless of scheduling). The exact drop semantics
//! (which frame is evicted, control packets never dropped) are covered by the
//! channel unit tests; here we prove the policy is wired through `graph.link_with`
//! into `run_graph` and that `RunStats::frames_dropped` is surfaced.

use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{Graph, LinkPolicy, PipelineClock};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn run_with(policy: LinkPolicy, frames: u64, capacity: usize) -> g2g_core::runtime::RunStats {
    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(VideoTestSrc::new(8, 8, 30, frames)));
    let sink = g.add_sink(GraphNode::element(FakeSink::new()));
    g.link_with(src, sink, policy).unwrap();
    run_graph(g, &NullClock, capacity)
        .await
        .expect("leaky pipeline runs to completion")
}

#[tokio::test]
async fn drop_oldest_link_conserves_every_frame() {
    let stats = run_with(LinkPolicy::DropOldest, 16, 2).await;
    assert_eq!(stats.frames_emitted, 16);
    // Conservation: each emitted frame was either consumed or dropped. Holds
    // whatever the drop count turns out to be under the runtime's scheduling.
    assert_eq!(
        stats.frames_consumed + stats.frames_dropped,
        16,
        "every emitted frame is consumed or dropped"
    );
    assert!(stats.frames_consumed > 0, "the sink still receives frames");
}

#[tokio::test]
async fn drop_newest_link_conserves_every_frame() {
    let stats = run_with(LinkPolicy::DropNewest, 16, 2).await;
    assert_eq!(stats.frames_emitted, 16);
    assert_eq!(stats.frames_consumed + stats.frames_dropped, 16);
}

#[tokio::test]
async fn block_link_drops_nothing() {
    // The default lossless policy: backpressure paces the source, nothing drops.
    let stats = run_with(LinkPolicy::Block, 16, 2).await;
    assert_eq!(stats.frames_emitted, 16);
    assert_eq!(stats.frames_consumed, 16);
    assert_eq!(stats.frames_dropped, 0);
}
