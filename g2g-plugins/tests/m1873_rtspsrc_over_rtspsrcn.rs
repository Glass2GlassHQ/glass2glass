// `rtspsrc` is `rtspsrcn` with the video pad alone, so what one of them does at negotiation and
// across a reconnect the other does too.
#![cfg(all(
    feature = "std",
    feature = "rtsp",
    feature = "ffmpeg",
    target_os = "linux"
))]

use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile, SourceLoop};
use g2g_core::PipelinePacket;
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::rtspsrc::RtspSrc;

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
#[path = "common/rtsp_h265_relay.rs"]
mod rtsp_h265_relay;

use collect_sink::CollectSink;
use ffmpeg_tool::have_ffmpeg;

const DECODED_FRAMES: u64 = 10;
const RECONNECT_BACKOFF_MS: u64 = 10;

// a launch-built fan-out learns the codec at negotiation, then plays the session it set up
#[tokio::test]
async fn rtspsrcn_negotiates_an_h265_track_and_plays_the_session_it_set_up() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the RTSP H.265 fan-out negotiation");
        return;
    }
    let relay = rtsp_h265_relay::serve_h265_fixture(0);
    let line = format!(
        "rtspsrcn name=s location={} num-buffers={DECODED_FRAMES} \
         s. ! h265parse ! avdec_h265 ! fakesink",
        relay.url
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    assert!(
        stats.frames_consumed > 0,
        "decoded H.265 frames reach the sink"
    );
    assert_eq!(relay.sessions(), 1, "the run plays the negotiation session");
}

#[tokio::test]
async fn a_session_that_dies_before_any_frame_leaves_the_timeline_at_zero() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the RTSP reconnect timeline");
        return;
    }
    let relay = rtsp_h265_relay::serve_h265_fixture(1);
    let mut src = RtspSrc::new(relay.url.clone())
        .with_frame_limit(DECODED_FRAMES)
        .with_reconnect(1)
        .with_reconnect_backoff(RECONNECT_BACKOFF_MS, RECONNECT_BACKOFF_MS);
    src.intercept_caps().await.expect("negotiation connects");
    let mut sink = CollectSink::default();
    src.run(&mut sink).await.expect("the second session plays");

    let first_pts = sink.packets.iter().find_map(|packet| match packet {
        PipelinePacket::DataFrame(frame) => Some(frame.timing.pts_ns),
        _ => None,
    });
    assert_eq!(first_pts, Some(0));
    assert_eq!(relay.sessions(), 2, "one broken session, one that played");
}
