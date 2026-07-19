//! M127 `alpha`: the alpha / chroma-key transform is registered in
//! `default_registry` and runs in a text pipeline with its `Str` / `Double`
//! properties parsed from text. The per-pixel keying is unit-tested in the
//! module itself.

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
async fn alpha_runs_in_a_text_pipeline() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=3 pattern=smpte ! alpha method=green ! fakesink",
    )
    .expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("alpha pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "all keyed frames reached the sink"
    );
}
