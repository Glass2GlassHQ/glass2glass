//! M130 `audiomixer`: a summing fan-in muxer registered in `default_registry`,
//! reachable from the M122 text fan-in syntax. Two sources of unequal length mix
//! into one stream that runs end to end through `run_graph`. The summing math is
//! unit-tested in the module.

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
async fn audiomixer_sums_two_sources() {
    let reg = default_registry();
    // Feeding chains first (each ending `! m.`), the muxer chain last. The mixer
    // emits a single merged stream, so a plain fakesink (monotonic sequence) is
    // fine, unlike the interleaving `funnel`.
    let graph = parse_launch(
        &reg,
        "audiotestsrc num-buffers=4 channels=2 wave=sine ! m.   \
         audiotestsrc num-buffers=3 channels=2 wave=sine ! m.   \
         audiomixer name=m ! fakesink",
    )
    .expect("fan-in pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .expect("mixer pipeline runs");
    // Mixed buffers span the longer source: 3 paired + 1 from the longer alone.
    assert_eq!(
        stats.frames_consumed, 4,
        "mixed stream spans the longer input"
    );
}
