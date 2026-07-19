//! M128 `audiopanorama`: the stereo-pan transform is registered in
//! `default_registry` and runs in a text pipeline with its `Double` property
//! parsed from text. The per-sample balance math is unit-tested in the module.

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
async fn audiopanorama_runs_in_a_text_pipeline() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "audiotestsrc num-buffers=3 channels=2 wave=sine ! audiopanorama panorama=0.5 ! fakesink",
    )
    .expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("panorama pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "all panned buffers reached the sink"
    );
}
