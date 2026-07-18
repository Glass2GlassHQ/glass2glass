//! M129 `videobox`: the box element is registered in `default_registry` and
//! runs in a text pipeline (changing geometry), with its signed `Int` / `Str`
//! properties parsed from text. The canvas math is unit-tested in the module.

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
async fn videobox_runs_in_a_text_pipeline() {
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=3 pattern=smpte ! videobox left=-8 right=-8 fill=black ! fakesink",
    )
    .expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("videobox pipeline runs");
    assert_eq!(
        stats.frames_consumed, 3,
        "all bordered frames reached the sink"
    );
}
