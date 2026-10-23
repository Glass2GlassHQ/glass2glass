#![cfg(all(
    feature = "std",
    feature = "rtsp",
    feature = "ffmpeg",
    target_os = "linux"
))]

use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
use ffmpeg_tool::have_ffmpeg;
#[path = "common/rtsp_h265_relay.rs"]
mod rtsp_h265_relay;

const DECODED_FRAMES: u64 = 10;

#[tokio::test]
async fn rtspsrc_decodes_an_h265_track_packetized_by_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the reference-peer RTSP H.265 decode");
        return;
    }
    let relay = rtsp_h265_relay::serve_h265_fixture(&[], None);

    let line = format!(
        "rtspsrc location={} num-buffers={DECODED_FRAMES} \
         ! h265parse ! avdec_h265 ! fakesink",
        relay.url
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    assert_eq!(
        stats.frames_emitted, DECODED_FRAMES,
        "rtspsrc delivers every frame"
    );
    assert!(
        stats.frames_consumed > 0,
        "decoded H.265 frames reach the sink"
    );
    assert_eq!(relay.sessions(), 1, "the run plays the negotiation session");
}
