//! Regression: an `appsink` downstream of a caps-driven transform must still
//! deliver frames. A format/size transform (here `videoconvert`) makes the
//! runner cascade caps a second time, calling the sink's `configure_pipeline`
//! again; the claim of the registered delivery mode must be idempotent, or the
//! re-configure clobbers it and every frame is silently dropped.
//!
//! `default_registry` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn appsink_delivers_through_a_caps_driven_transform() {
    // Pre-fill the feed (one RGBA frame then EOS) so the graph drives on this
    // one thread without a concurrent producer.
    let feed = register_appsrc("recfg_in");
    assert!(feed.push(&[7u8; 16], 0));
    feed.end_of_stream();

    let pull = register_appsink_pull("recfg_out");

    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "appsrc channel=recfg_in caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1 \
         ! videoconvert \
         ! appsink channel=recfg_out",
    )
    .expect("parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("runs");
    assert_eq!(stats.frames_consumed, 1, "sink consumed the frame");

    // The frame must have reached the pull channel (the bug: consumed but never
    // delivered, because the second configure_pipeline dropped the sender).
    let mut delivered = 0;
    loop {
        match pull.try_pull() {
            Pull::Frame(_) => delivered += 1,
            Pull::Ended => break,
            Pull::Empty => break,
        }
    }
    assert_eq!(delivered, 1, "appsink delivered the frame past the re-configure");
}
