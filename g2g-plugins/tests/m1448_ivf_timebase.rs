#![cfg(all(feature = "std", any(feature = "dav1d", feature = "rav1d")))]

use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// av1_321x181.obu stream-copied into IVF by ffmpeg without `-r`
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av1_321x181.ivf"
);
const IVF_RATE_OFFSET: usize = 16;
const IVF_FRAME_COUNT_OFFSET: usize = 24;
// the timebase ffmpeg writes for a stream with no timestamps of its own
const UNTIMED_STREAM_TIMEBASE_RATE: u32 = 1_200_000;

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32 field"))
}

#[tokio::test]
async fn an_ivf_timebase_past_any_frame_rate_still_plays() {
    let ivf = std::fs::read(FIXTURE).expect("fixture reads");
    assert_eq!(
        read_u32_le(&ivf, IVF_RATE_OFFSET),
        UNTIMED_STREAM_TIMEBASE_RATE
    );
    let frames = read_u32_le(&ivf, IVF_FRAME_COUNT_OFFSET);

    let line = format!("filesrc location={FIXTURE} ! decodebin ! fakesink");
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    assert_eq!(stats.frames_consumed, u64::from(frames));
}
