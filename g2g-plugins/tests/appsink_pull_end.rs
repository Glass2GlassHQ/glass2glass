//! An `appsink` pull handle ends once its graph is gone, whether the run
//! finished or failed before the sink configured.
//!
//! `default_registry` is `std`-gated, so this file is too.
#![cfg(feature = "std")]

use std::time::Duration;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::appsink::{register_appsink_pull, AppSink, AppSinkPull, Pull};
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::registry::default_registry;

const PULL_END_TIMEOUT: Duration = Duration::from_secs(5);
const VIDEO_CAPS: &str = "video/x-raw,format=RGBA,width=2,height=2,framerate=30/1";
const AUDIO_CAPS: &str = "audio/x-raw,format=S16LE,rate=48000,channels=2";
const FRAME_BYTES: [u8; 16] = [7; 16];

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn assert_pull_ended(pull: &AppSinkPull) {
    assert!(matches!(pull.try_pull(), Pull::Ended), "try_pull ended");
    let next = tokio::time::timeout(PULL_END_TIMEOUT, pull.pull())
        .await
        .expect("pull returns instead of hanging");
    assert!(next.is_none(), "pull ended");
}

async fn run_failing_before_appsink_configures(in_channel: &str, appsink_channel_prop: &str) {
    let _feed = register_appsrc(in_channel);
    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        &format!(
            "appsrc channel={in_channel} caps={VIDEO_CAPS} \
             ! appsink {appsink_channel_prop} caps={AUDIO_CAPS}"
        ),
    )
    .expect("parses");
    assert!(
        run_graph(graph, &ZeroClock, 4).await.is_err(),
        "video into an audio-only appsink fails negotiation"
    );
}

#[tokio::test]
async fn named_pull_ends_when_the_graph_fails_before_the_appsink_configures() {
    let pull = register_appsink_pull("pull_end_named_out");
    run_failing_before_appsink_configures("pull_end_named_in", "channel=pull_end_named_out").await;
    assert_pull_ended(&pull).await;
}

#[tokio::test]
async fn default_pull_ends_when_the_graph_fails_before_the_appsink_configures() {
    let pull = register_appsink_pull("default");
    drop(AppSink::new());
    assert!(
        matches!(pull.try_pull(), Pull::Empty),
        "an appsink no graph negotiated with leaves the registration"
    );
    run_failing_before_appsink_configures("pull_end_default_in", "").await;
    assert_pull_ended(&pull).await;
}

#[tokio::test]
async fn pull_ends_after_a_finished_run_is_drained() {
    let feed = register_appsrc("pull_end_drained_in");
    assert!(feed.push(&FRAME_BYTES, 0));
    feed.end_of_stream();
    let pull = register_appsink_pull("pull_end_drained_out");

    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        &format!(
            "appsrc channel=pull_end_drained_in caps={VIDEO_CAPS} \
             ! appsink channel=pull_end_drained_out"
        ),
    )
    .expect("parses");
    run_graph(graph, &ZeroClock, 4).await.expect("runs");

    let Pull::Frame(frame) = pull.try_pull() else {
        panic!("the pushed frame is queued");
    };
    assert_eq!(frame.domain.as_system_slice(), Some(&FRAME_BYTES[..]));
    assert!(
        matches!(pull.try_pull(), Pull::Ended),
        "end of stream marker"
    );
    assert_pull_ended(&pull).await;
}
