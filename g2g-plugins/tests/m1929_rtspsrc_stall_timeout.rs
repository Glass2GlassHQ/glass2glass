#![cfg(all(
    feature = "std",
    feature = "rtsp",
    feature = "ffmpeg",
    target_os = "linux"
))]

use std::time::Duration;

use g2g_core::runtime::SourceLoop;
use g2g_core::{PipelinePacket, PropValue};
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
const DATAGRAMS_BEFORE_STALL: usize = 3;
const STALL_TIMEOUT_MS: u64 = 300;
// without the stall timeout the run waits on the silent connection forever
const RUN_DEADLINE: Duration = Duration::from_secs(20);

#[tokio::test]
async fn a_session_that_goes_silent_is_reconnected() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the RTSP stall");
        return;
    }
    let relay =
        rtsp_h265_relay::serve_h265_fixture(&[EarlyEnd::StallAfter(DATAGRAMS_BEFORE_STALL)], None);
    let mut src = RtspSrc::new(relay.url.clone())
        .with_frame_limit(DECODED_FRAMES)
        .with_reconnect(1)
        .with_reconnect_backoff(RECONNECT_BACKOFF_MS, RECONNECT_BACKOFF_MS);
    src.set_property("stall-timeout", PropValue::Uint(STALL_TIMEOUT_MS))
        .unwrap();
    src.intercept_caps().await.expect("negotiation connects");
    let mut sink = CollectSink::default();
    tokio::time::timeout(RUN_DEADLINE, src.run(&mut sink))
        .await
        .expect("the stalled session ends")
        .expect("the run ends cleanly");

    assert_eq!(
        relay.sessions(),
        2,
        "the stalled session, then the one that played"
    );
    let frames = sink
        .packets
        .iter()
        .filter(|packet| matches!(packet, PipelinePacket::DataFrame(_)))
        .count();
    assert_eq!(frames, DECODED_FRAMES as usize);
}
