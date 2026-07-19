//! Phase 2 observability: QoS bus messages (M85). A `SyncSink` under a clock
//! that has already run past every frame's deadline drops the late frames
//! (instead of presenting them late) and posts a `BusMessage::Qos` per drop.
//! The control case (an on-time clock) presents every frame and posts nothing.

use core::future::{ready, Ready};

use g2g_core::runtime::{run_graph, GraphNodeRef};
use g2g_core::{AsyncClock, Bus, BusMessage, Graph, PipelineClock};
use g2g_plugins::syncsink::SyncSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

/// Pipeline clock for the runner's base-time election (unused by the sink,
/// which carries its own clock).
struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// A clock pinned to a fixed instant, with instant `sleep_until` (a test never
/// waits). Setting the instant ahead of the frame PTSs makes them all "late".
struct FixedClock(u64);
impl PipelineClock for FixedClock {
    fn now_ns(&self) -> u64 {
        self.0
    }
}
impl AsyncClock for FixedClock {
    type SleepFuture<'a>
        = Ready<()>
    where
        Self: 'a;
    fn sleep_until_ns(&self, _deadline_ns: u64) -> Self::SleepFuture<'_> {
        ready(())
    }
}

#[tokio::test]
async fn syncsink_drops_late_frames_and_posts_qos() {
    let (bus, handle) = Bus::new(16);
    let mut src = VideoTestSrc::new(8, 8, 30, 4);
    // now = 1 s; the 4 frames at 30 fps (PTS 0, 33, 66, 100 ms) are all past
    // their deadline, so with a 0 ns lateness bound every one is dropped.
    let mut sink = SyncSink::new(FixedClock(1_000_000_000))
        .with_max_lateness_ns(0)
        .with_bus(handle);

    let stats = {
        let mut g: Graph<GraphNodeRef> = Graph::new();
        let s = g.add_source(GraphNodeRef::source_ref(&mut src));
        let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
        g.link(s, k).unwrap();
        run_graph(g, &NullClock, 4).await.expect("qos graph runs")
    };

    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(sink.received(), 0, "every frame was late, none presented");
    assert_eq!(sink.dropped(), 4);

    // The bus carried one QoS report per drop, with positive jitter and a
    // monotonically rising dropped count.
    let mut qos = Vec::new();
    while let Some(m) = bus.try_recv() {
        if let BusMessage::Qos {
            jitter_ns,
            processed,
            dropped,
            ..
        } = m
        {
            assert!(jitter_ns > 0, "late frame has positive jitter");
            assert_eq!(processed, 0, "nothing was ever presented");
            qos.push(dropped);
        }
    }
    assert_eq!(
        qos,
        &[1, 2, 3, 4],
        "cumulative dropped count rises with each report"
    );
}

#[tokio::test]
async fn on_time_frames_present_with_no_qos() {
    let (bus, handle) = Bus::new(16);
    let mut src = VideoTestSrc::new(8, 8, 30, 4);
    // now = 0: no frame is past its deadline, so none is dropped even with the
    // tightest (0 ns) lateness bound.
    let mut sink = SyncSink::new(FixedClock(0))
        .with_max_lateness_ns(0)
        .with_bus(handle);

    {
        let mut g: Graph<GraphNodeRef> = Graph::new();
        let s = g.add_source(GraphNodeRef::source_ref(&mut src));
        let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
        g.link(s, k).unwrap();
        run_graph(g, &NullClock, 4)
            .await
            .expect("on-time graph runs");
    }

    assert_eq!(sink.received(), 4, "all frames presented");
    assert_eq!(sink.dropped(), 0);
    assert!(bus.try_recv().is_none(), "no QoS posted when on time");
}
