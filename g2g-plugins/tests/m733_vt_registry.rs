//! M733: the VideoToolbox elements in the launch registry. `vtdec` /
//! `vtdech265` decode inside a text pipeline (and `avdec_h264` resolves to
//! `vtdec` when ffmpeg is absent), and `vtenc_h264` encodes from a text
//! pipeline with its `bitrate` property parsed from the line. Runs on the
//! macOS CI runner like `m731_videotoolbox`.
#![cfg(all(target_os = "macos", feature = "vtdecode"))]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

const H264_CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const H265_CLIP: &[u8] = include_bytes!("fixtures/h265_640x480.h265");

/// Frames in each checked-in fixture (asserted exactly by `m731_videotoolbox`).
const FIXTURE_FRAMES: u64 = 10;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp(tag: &str, ext: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path =
        std::env::temp_dir().join(format!("g2g-m733-{}-{}.{}", std::process::id(), tag, ext));
    std::fs::write(&path, bytes).expect("write temp fixture");
    path
}

/// Run `line` and return the sink's consumed-frame count.
async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"))
        .frames_consumed
}

#[tokio::test(flavor = "current_thread")]
async fn vtdec_decodes_in_a_text_pipeline() {
    let path = temp("h264", "h264", H264_CLIP);
    let line = format!(
        "filesrc location={} ! h264parse ! vtdec ! fakesink",
        path.display()
    );
    let consumed = run_line(&line).await;
    std::fs::remove_file(&path).ok();
    assert_eq!(consumed, FIXTURE_FRAMES, "every fixture frame decoded");
}

#[tokio::test(flavor = "current_thread")]
async fn vtdech265_decodes_in_a_text_pipeline() {
    let path = temp("h265", "h265", H265_CLIP);
    let line = format!(
        "filesrc location={} ! h265parse ! vtdech265 ! fakesink",
        path.display()
    );
    let consumed = run_line(&line).await;
    std::fs::remove_file(&path).ok();
    assert_eq!(consumed, FIXTURE_FRAMES, "every fixture frame decoded");
}

/// The gst-canonical `avdec_h264` alias reaches a working H.264 decoder here:
/// `vtdec` when ffmpeg is off (the macOS CI build), `ffmpegdec` when it is on.
#[tokio::test(flavor = "current_thread")]
async fn avdec_h264_alias_resolves_and_decodes() {
    let path = temp("alias", "h264", H264_CLIP);
    let line = format!(
        "filesrc location={} ! h264parse ! avdec_h264 ! fakesink",
        path.display()
    );
    let consumed = run_line(&line).await;
    std::fs::remove_file(&path).ok();
    assert_eq!(consumed, FIXTURE_FRAMES, "every fixture frame decoded");
}

#[cfg(feature = "vtencode")]
#[tokio::test(flavor = "current_thread")]
async fn vtenc_h264_encodes_in_a_text_pipeline() {
    let consumed = run_line(
        "videotestsrc num-buffers=5 ! videoconvert ! vtenc_h264 bitrate=1000000 ! fakesink",
    )
    .await;
    assert_eq!(consumed, 5, "every test picture encoded to an access unit");
}

#[cfg(feature = "vtencode")]
#[tokio::test(flavor = "current_thread")]
async fn vtenc_h265_encodes_in_a_text_pipeline() {
    let consumed =
        run_line("videotestsrc num-buffers=5 ! videoconvert ! vtenc_h265 ! fakesink").await;
    assert_eq!(consumed, 5, "every test picture encoded to an access unit");
}
