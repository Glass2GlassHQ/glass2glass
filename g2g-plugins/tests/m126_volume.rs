//! M126 `volume`: the audio-gain transform is registered in `default_registry`
//! and runs in a text pipeline with its `Double` / `Bool` properties parsed from
//! text. The per-sample gain math is unit-tested in the module itself.

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn volume_runs_in_a_text_pipeline() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "audiotestsrc num-buffers=3 wave=saw ! volume volume=0.5 ! fakesink",
    )
    .expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("volume pipeline runs");
    assert_eq!(stats.frames_consumed, 3, "all gain-adjusted buffers reached the sink");
}
