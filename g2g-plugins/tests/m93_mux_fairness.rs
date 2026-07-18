//! Fan-in fairness / backpressure regression (M93). Two sources feed a
//! compositor whose output is consumed by a deliberately slow (paced) sink, so
//! the muxer's input channels stay full and both per-input forwarders block on
//! send at the same time.
//!
//! The old fan-in runner merged every input into ONE shared bounded channel.
//! That channel's mpsc parks a single sender waker (last writer wins), so with
//! two forwarders blocked on a full channel one forwarder's wakeup was lost: it
//! never ran again, its source never reached EOS, and the all-inputs-EOS
//! aggregation hung forever. Live, this showed up as a frozen picture-in-picture
//! overlay and a pipeline that never terminated. The fix gives each input pad
//! its own channel (its own waker) drained round-robin. This test would hang
//! (then trip the timeout) under the old design and completes under the fix.

use core::future::Future;
use core::pin::Pin;
use std::time::Duration;

use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket,
};
use g2g_plugins::compositor::{Compositor, CompositorPad};
use g2g_plugins::videotestsrc::VideoTestSrc;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// A sink that sleeps briefly per frame, modelling a display-paced consumer.
/// The pacing keeps the muxer's input channels full so both forwarders contend
/// for send capacity at once, which is what surfaced the lost-wakeup deadlock.
#[derive(Default)]
struct PacedSink {
    seen: u32,
}

impl AsyncElement for PacedSink {
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
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = packet {
                self.seen += 1;
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn fan_in_does_not_deadlock_under_backpressure() {
    // Background (input 0, the cadence driver) and an overlay (input 1) both
    // free-run; the paced sink throttles the output so both input channels stay
    // full. With the old shared-FIFO runner this hangs; the timeout converts a
    // hang into a hard failure.
    const FRAMES: u64 = 40;
    let mut g: Graph<GraphNode> = Graph::new();
    let bg = g.add_source(GraphNode::source(VideoTestSrc::new(32, 32, 30, FRAMES)));
    let overlay = g.add_source(GraphNode::source(VideoTestSrc::new(16, 16, 30, FRAMES)));
    let comp = g.add_muxer(
        GraphNode::muxer(Compositor::new(
            32,
            32,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(8, 8).with_zorder(1),
            ]),
        )),
        2,
    );
    let sink = g.add_sink(GraphNode::element(PacedSink::default()));

    g.link(bg, comp.input(0)).unwrap();
    g.link(overlay, comp.input(1)).unwrap();
    g.link(comp.output(), sink).unwrap();

    let stats = tokio::time::timeout(Duration::from_secs(10), run_graph(g, &NullClock, 4))
        .await
        .expect("fan-in must terminate, not deadlock under backpressure")
        .expect("fan-in DAG runs");

    // Both sources ran to completion (so each contributed its EOS, the very
    // thing that hung before), and the compositor emitted one composited frame
    // per input-0 frame down to the sink.
    assert_eq!(
        stats.frames_emitted,
        2 * FRAMES,
        "both sources emitted every frame"
    );
    assert_eq!(
        stats.frames_consumed, FRAMES,
        "one composited frame per background frame"
    );
}
