#![cfg(feature = "ffmpeg")]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

// 30 frames at 25 fps with B-frames, its elst cut to 710 ms starting on the sixth frame
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_bframes_elst.mp4"
);
const WIDTH: usize = 64;
const HEIGHT: usize = 48;
const I420_FRAME_LEN: usize = WIDTH * HEIGHT * 3 / 2;
// what ffprobe -count_frames reads for the fixture
const FRAMES_IN_EDIT: usize = 18;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1548-{}-{name}", std::process::id()))
}

async fn decoded_frame_count(decode_chain: &str, name: &str) -> usize {
    let out = temp_path(name);
    let line = format!(
        "filesrc location={FIXTURE} ! {decode_chain} ! video/x-raw,format=I420 ! filesink location={}",
        out.display()
    );
    let graph = parse_launch(&default_registry(), &line).expect("launch line parses");
    run_graph(graph, &ZeroClock, 4).await.expect("decode runs");
    let raw = std::fs::read(&out).expect("decode output reads");
    let _ = std::fs::remove_file(&out);
    assert_eq!(raw.len() % I420_FRAME_LEN, 0, "whole I420 frames");
    raw.len() / I420_FRAME_LEN
}

#[tokio::test]
async fn decodebin_presents_only_the_edit() {
    assert_eq!(
        decoded_frame_count("decodebin", "decodebin.yuv").await,
        FRAMES_IN_EDIT
    );
}

#[tokio::test]
async fn multi_track_demux_presents_only_the_edit() {
    assert_eq!(
        decoded_frame_count("qtdemux name=d d.video_0 ! ffmpegdec", "qtdemux.yuv").await,
        FRAMES_IN_EDIT
    );
}
