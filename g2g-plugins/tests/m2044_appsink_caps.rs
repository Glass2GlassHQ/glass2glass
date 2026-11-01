//! M2044: an appsink pull channel reports the caps of each frame it hands over,
//! including after a mid-stream resolution change.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{Caps, Dim, PipelineClock};
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::registry::default_registry;

const RECONFIG_CLIP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_reconfig_640x480_to_320x240.h264"
);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn geometry(caps: &Caps) -> (u32, u32) {
    match caps {
        Caps::CompressedVideo {
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } => (*width, *height),
        other => panic!("expected fixed compressed video caps, got {other:?}"),
    }
}

#[tokio::test]
async fn pulled_frames_carry_the_caps_they_crossed_under() {
    let pull = register_appsink_pull("m2044_caps");
    let graph = parse_launch(
        &default_registry(),
        &format!("filesrc location={RECONFIG_CLIP} ! h264parse ! appsink channel=m2044_caps"),
    )
    .expect("parses");
    assert_eq!(pull.caps(), None, "no caps before the first frame");

    let drain = async {
        let mut geometries = Vec::new();
        loop {
            match pull.try_pull() {
                Pull::Frame(_) => {
                    let caps = pull.caps().expect("caps arrive before the first frame");
                    geometries.push(geometry(&caps));
                }
                Pull::Empty => tokio::task::yield_now().await,
                Pull::Ended => return geometries,
            }
        }
    };
    let (run, geometries) = tokio::join!(run_graph(graph, &ZeroClock, 4), drain);
    run.expect("runs");

    let first = geometries.first().copied().expect("frames");
    let last = geometries.last().copied().expect("frames");
    assert_eq!(first, (640, 480));
    assert_eq!(last, (320, 240));
    let changes = geometries
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count();
    assert_eq!(changes, 1, "one switch, at the new SPS: {geometries:?}");
}
