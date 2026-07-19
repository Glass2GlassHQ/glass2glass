//! DAG runner: a borrowing `Graph<GraphNodeRef<'a>>` built from `&mut` element
//! references (the shape the convenience wrappers use to delegate to
//! `run_graph` without taking ownership). The caller keeps its elements and can
//! inspect them after the run.

use g2g_core::runtime::{run_graph, GraphNodeRef};
use g2g_core::{Graph, PipelineClock};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn borrowing_graph_runs_and_caller_keeps_elements() {
    let mut src = VideoTestSrc::new(8, 8, 30, 4);
    let mut flip = VideoFlip::new(FlipMethod::Rotate180);
    let mut sink = FakeSink::new();

    let stats = {
        let mut g: Graph<GraphNodeRef> = Graph::new();
        let s = g.add_source(GraphNodeRef::source_ref(&mut src));
        let f = g.add_transform(GraphNodeRef::element_ref(&mut flip));
        let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
        g.link(s, f).unwrap();
        g.link(f, k).unwrap();
        run_graph(g, &NullClock, 4)
            .await
            .expect("borrowing graph runs")
    };

    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(stats.frames_consumed, 4);
    // The borrows ended with the graph; the caller still owns the elements.
    assert_eq!(
        sink.received(),
        4,
        "caller reads the borrowed sink after the run"
    );
    assert!(sink.eos_seen());
}
