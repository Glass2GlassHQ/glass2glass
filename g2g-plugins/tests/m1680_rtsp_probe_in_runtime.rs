// The parse-time DESCRIBE of an `rtsp://` URI runs from inside a tokio runtime too, so a line
// parsed by async code still fans out to the session's video and audio pads.
#![cfg(all(
    feature = "std",
    feature = "rtsp",
    feature = "ffmpeg",
    target_os = "linux"
))]

use g2g_core::graph::NodeKind;
use g2g_core::runtime::parse_launch;
use g2g_plugins::registry::default_registry;

#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
#[path = "common/rtsp_av_relay.rs"]
mod rtsp_av_relay;

const AV_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/av_h264_aac.mp4"
);
const STREAM_PADS: u8 = 2;

#[test]
fn a_line_parsed_inside_a_runtime_fans_out() {
    if !ffmpeg_tool::have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the RTSP probe inside a runtime");
        return;
    }
    let relay = rtsp_av_relay::serve_av_fixture(AV_FIXTURE);
    let line = format!(
        "uridecodebin uri={} name=d  d. ! fakesink  d. ! fakesink",
        relay.url
    );
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("build a runtime");
    let graph = runtime
        .block_on(async { parse_launch(&default_registry(), &line) })
        .unwrap_or_else(|e| panic!("parses `{line}`: {e}"))
        .finish()
        .expect("valid graph");
    let kinds: Vec<NodeKind> = graph.topo().iter().map(|&node| graph.kind(node)).collect();
    assert!(
        kinds.contains(&NodeKind::FanoutSrc(STREAM_PADS)),
        "{kinds:?}"
    );
}
