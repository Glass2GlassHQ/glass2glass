//! Phase 2 breadth: `AudioResample` end-to-end through the DAG runner. A real
//! `AudioTestSrc` (44.1 kHz S16 stereo) feeds `AudioResample` retargeting to
//! 48 kHz into a `FakeSink`; the run proves the async `process` path flows
//! frames and emits the retargeted output caps. The resampling math itself is
//! covered by the element's unit tests.

use g2g_core::runtime::{run_graph, GraphNodeRef};
use g2g_core::{AudioFormat, Caps, Graph, PipelineClock};
use g2g_plugins::audioresample::AudioResample;
use g2g_plugins::audiotestsrc::AudioTestSrc;
use g2g_plugins::fakesink::FakeSink;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn audioresample_retargets_rate_through_run_graph() {
    let mut src = AudioTestSrc::new(44_100, 2, 440, 4);
    let mut resample = AudioResample::new(48_000);
    let mut sink = FakeSink::new();

    let stats = {
        let mut g: Graph<GraphNodeRef> = Graph::new();
        let s = g.add_source(GraphNodeRef::source_ref(&mut src));
        let r = g.add_transform(GraphNodeRef::element_ref(&mut resample));
        let k = g.add_sink(GraphNodeRef::element_ref(&mut sink));
        g.link(s, r).unwrap();
        g.link(r, k).unwrap();
        run_graph(g, &NullClock, 4).await.expect("audio resample graph runs")
    };

    // One output buffer per input buffer; all reach the sink.
    assert_eq!(stats.frames_emitted, 4);
    assert_eq!(stats.frames_consumed, 4);
    assert_eq!(sink.received(), 4);

    // The sink saw the retargeted caps: same format + channels, 48 kHz out.
    let retargeted = sink
        .caps_changes()
        .iter()
        .any(|c| matches!(c.caps, Caps::Audio { format: AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 }));
    assert!(retargeted, "sink observed 48 kHz S16 stereo caps: {:?}", sink.caps_changes());
}
