#![cfg(all(
    feature = "std",
    feature = "rtsp",
    feature = "ffmpeg",
    target_os = "linux"
))]

use g2g_core::runtime::SourceLoop;
use g2g_core::PipelinePacket;
use g2g_plugins::rtspsrc::RtspSrc;

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
#[path = "common/rtsp_h265_relay.rs"]
mod rtsp_h265_relay;

use collect_sink::CollectSink;
use ffmpeg_tool::have_ffmpeg;
use rtsp_h265_relay::EarlyEnd;

const DECODED_FRAMES: u64 = 10;
const RECONNECT_BACKOFF_MS: u64 = 10;
// fewer than the datagrams of the frames the source waits for
const DATAGRAMS_BEFORE_CLOSE: usize = 3;
const RECORDING_RANGE: &str = "npt=0-10";

async fn run_until_the_server_closes(range: Option<&str>) -> (CollectSink, usize) {
    let relay =
        rtsp_h265_relay::serve_h265_fixture(&[EarlyEnd::CloseAfter(DATAGRAMS_BEFORE_CLOSE)], range);
    let mut src = RtspSrc::new(relay.url.clone())
        .with_frame_limit(DECODED_FRAMES)
        .with_reconnect(1)
        .with_reconnect_backoff(RECONNECT_BACKOFF_MS, RECONNECT_BACKOFF_MS);
    src.intercept_caps().await.expect("negotiation connects");
    let mut sink = CollectSink::default();
    src.run(&mut sink).await.expect("the run ends cleanly");
    (sink, relay.sessions())
}

fn frames(sink: &CollectSink) -> usize {
    sink.packets
        .iter()
        .filter(|packet| matches!(packet, PipelinePacket::DataFrame(_)))
        .count()
}

#[tokio::test]
async fn a_live_server_that_closes_the_connection_is_reconnected() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the RTSP server close");
        return;
    }
    let (sink, sessions) = run_until_the_server_closes(None).await;
    assert_eq!(sessions, 2, "the closed session, then the one that played");
    assert_eq!(frames(&sink), DECODED_FRAMES as usize);
}

#[tokio::test]
async fn a_recording_that_closes_the_connection_ends_the_stream() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the RTSP server close");
        return;
    }
    let (sink, sessions) = run_until_the_server_closes(Some(RECORDING_RANGE)).await;
    assert_eq!(sessions, 1, "an ended recording is not reconnected");
    assert!(matches!(sink.packets.last(), Some(PipelinePacket::Eos)));
}
