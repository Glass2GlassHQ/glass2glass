#![cfg(feature = "std")]

use std::time::Duration;

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{parse_launch, MutationError, RunBuilder};
use g2g_core::RawVideoFormat;
use g2g_plugins::appsink::register_appsink_pull;
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::{converters, default_registry};
use g2g_plugins::videoconvert::VideoConvert;

const CHANNEL: &str = "m1234-converters";
const LINK_CAPACITY: usize = 2;
const PULL_TIMEOUT: Duration = Duration::from_secs(5);

#[tokio::test]
async fn a_format_changing_splice_gets_a_converter_from_the_registry() {
    let pull = register_appsink_pull(CHANNEL);
    let line = format!(
        "videotestsrc name=src num-buffers=30 ! clocksync ! appsink channel={CHANNEL} caps=video/x-raw,format=RGBA"
    );
    let graph = parse_launch(&default_registry(), &line).expect("the line parses");
    let clock = WallClock::new();
    let (mutator, run) = RunBuilder::new(graph, &clock, LINK_CAPACITY).run_mutable();
    let to_i420 = || Box::new(VideoConvert::new(RawVideoFormat::I420)) as Box<dyn DynAsyncElement>;

    // dropping the pull handle at the end lets appsink discard the frames nobody reads
    let driver = async move {
        tokio::time::timeout(PULL_TIMEOUT, pull.pull())
            .await
            .expect("a frame before the splice")
            .expect("the stream is running");
        let refused = mutator
            .insert_after("src", to_i420())
            .await
            .expect_err("the sink takes no I420 and the plain handle has no converters");
        let names = mutator
            .clone()
            .with_converters(converters())
            .insert_chain_after("src", vec![to_i420()])
            .await
            .expect("a videoconvert takes I420 back to RGBA");
        let after = tokio::time::timeout(PULL_TIMEOUT, pull.pull())
            .await
            .expect("a frame after the splice");
        (refused, names, after.is_some())
    };
    let (stats, (refused, names, frame_after)) = tokio::join!(run, driver);

    stats.expect("the sink never saw a shape outside its caps");
    assert_eq!(refused, MutationError::DownstreamRefused);
    assert_eq!(
        names.len(),
        2,
        "the I420 convert and its converter: {names:?}"
    );
    assert!(frame_after, "frames still reach the sink after the splice");
}
