#![cfg(any(feature = "ffmpeg", feature = "rav1d"))]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{PipelineClock, RawVideoFormat};
use g2g_plugins::registry::default_registry;

struct EditListFixture {
    path: &'static str,
    width: u32,
    height: u32,
    // what ffprobe -count_frames reads for the fixture
    frames_in_edit: usize,
}

// 30 frames at 25 fps with B-frames, its elst cut to 710 ms starting on the sixth frame
#[cfg(feature = "ffmpeg")]
const H264_BFRAMES: EditListFixture = EditListFixture {
    path: concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/h264_64x48_bframes_elst.mp4"
    ),
    width: 64,
    height: 48,
    frames_in_edit: 18,
};

// av1_321x181_24fps.mp4 with its elst cut from 417 ms to 250 ms
#[cfg(feature = "rav1d")]
const AV1_CUT_END: EditListFixture = EditListFixture {
    path: concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/av1_321x181_24fps_elst.mp4"
    ),
    width: 321,
    height: 181,
    frames_in_edit: 6,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m1548-{}-{name}", std::process::id()))
}

async fn decoded_frame_count(fixture: &EditListFixture, decode_chain: &str, name: &str) -> usize {
    let out = temp_path(name);
    let line = format!(
        "filesrc location={} ! {decode_chain} ! video/x-raw,format=I420 ! filesink location={}",
        fixture.path,
        out.display()
    );
    let graph = parse_launch(&default_registry(), &line).expect("launch line parses");
    run_graph(graph, &ZeroClock, 4).await.expect("decode runs");
    let raw = std::fs::read(&out).expect("decode output reads");
    let _ = std::fs::remove_file(&out);
    let frame_len = RawVideoFormat::I420
        .unpadded_frame_bytes(fixture.width, fixture.height)
        .expect("I420 frame size") as usize;
    assert_eq!(raw.len() % frame_len, 0, "whole I420 frames");
    raw.len() / frame_len
}

#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn decodebin_presents_only_the_edit() {
    assert_eq!(
        decoded_frame_count(&H264_BFRAMES, "decodebin", "decodebin.yuv").await,
        H264_BFRAMES.frames_in_edit
    );
}

#[cfg(feature = "ffmpeg")]
#[tokio::test]
async fn multi_track_demux_presents_only_the_edit() {
    assert_eq!(
        decoded_frame_count(
            &H264_BFRAMES,
            "qtdemux name=d d.video_0 ! ffmpegdec",
            "qtdemux.yuv"
        )
        .await,
        H264_BFRAMES.frames_in_edit
    );
}

#[cfg(feature = "rav1d")]
#[tokio::test]
async fn av1_decoder_drops_frames_past_the_edit() {
    assert_eq!(
        decoded_frame_count(
            &AV1_CUT_END,
            "qtdemux name=d d.video_0 ! rav1ddec",
            "rav1ddec.yuv"
        )
        .await,
        AV1_CUT_END.frames_in_edit
    );
}
