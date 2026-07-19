//! M124 `videobalance`: the colour-balance transform is registered in
//! `default_registry` and runs in a text pipeline, with its `Double` properties
//! parsed from text. The per-pixel math is unit-tested in the module itself.

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
async fn videobalance_runs_in_a_text_pipeline() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=3 pattern=smpte ! videobalance saturation=0.0 contrast=1.2 ! fakesink",
    )
    .expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("balanced pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "all balanced frames reached the sink"
    );
}

#[tokio::test]
async fn videobalance_hue_is_settable_from_launch() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=3 pattern=smpte ! videobalance hue=0.5 ! fakesink",
    )
    .expect("hue property parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("hue-rotated pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "all hue-rotated frames reached the sink"
    );
}
