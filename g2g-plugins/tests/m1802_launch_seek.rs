#![cfg(feature = "ffmpeg")]

use std::process::{Command, Output};

use g2g_plugins::checksumsink::checksum_line;

// all three fixtures: ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=3 -c:v libx264 -g 10 -keyint_min 10 -sc_threshold 0 -bf 0
const MP4_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_25fps_gop10.mp4"
);
const MKV_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_25fps_gop10.mkv"
);
const TS_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_25fps_gop10.ts"
);
// ffprobe reports start_time=1.400000 for the TS fixture, the first PTS its muxer wrote
const TS_FIRST_PTS_NS: u64 = 1_400_000_000;
const FRAME_RATE: u64 = 25;
const FRAME_COUNT: u64 = 75;
const NANOS_PER_SECOND: u64 = 1_000_000_000;
const NANOS_PER_MILLISECOND: u64 = 1_000_000;
const FRAME_PERIOD_NS: u64 = NANOS_PER_SECOND / FRAME_RATE;
// between the keyframes at 0.8 s and 1.2 s, and between two frames
const SEEK_MILLISECONDS: u64 = 1_020;
// ffmpeg -ss 1.02 -copyts -i <mp4 or mkv fixture> -frames:v 1 -f framemd5 -, the frame at 1.04
// the TS fixture needs an output -ss 2.42 (first PTS plus 1.02), ffmpeg's input seek in it lands on the 1.2 s keyframe
const FRAME_AFTER_SEEK_MD5: &str = "732b16a0536b40e66a6005fa0cca8569";

fn seek_seconds(milliseconds: u64) -> String {
    format!("{}", milliseconds as f64 / 1_000.0)
}

fn launch_with_seek(seconds: &str, pipeline: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .args(["-q", "--seek", seconds])
        .args(pipeline.split_whitespace())
        .output()
        .expect("g2g-launch starts")
}

fn checksum_lines(output: &Output) -> Vec<String> {
    assert!(
        output.status.success(),
        "g2g-launch fails: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

fn assert_starts_at_the_frame_after_the_seek(lines: &[String], first_pts_ns: u64) {
    let first_frame = (SEEK_MILLISECONDS * NANOS_PER_MILLISECOND).div_ceil(FRAME_PERIOD_NS);
    assert_eq!(
        lines[0],
        checksum_line(
            first_pts_ns + first_frame * FRAME_PERIOD_NS,
            FRAME_AFTER_SEEK_MD5.to_string()
        ),
        "the first frame out is ffmpeg's first frame after the seek"
    );
    assert_eq!(
        lines.len() as u64,
        FRAME_COUNT - first_frame,
        "the run continues from the seek to the end and nothing before it"
    );
}

fn mp4_pipeline() -> String {
    format!("uridecodebin uri=file://{MP4_FIXTURE} ! checksumsink hash=md5")
}

fn mkv_pipeline() -> String {
    format!("filesrc location={MKV_FIXTURE} ! decodebin ! checksumsink hash=md5")
}

fn mp4_decodebin_pipeline() -> String {
    format!("filesrc location={MP4_FIXTURE} ! decodebin ! checksumsink hash=md5")
}

fn ts_pipeline() -> String {
    format!("filesrc location={TS_FIXTURE} ! decodebin ! checksumsink hash=md5")
}

#[test]
fn mp4_seek_decodes_from_the_first_frame_at_or_after_the_target() {
    for pipeline in [mp4_pipeline(), mp4_decodebin_pipeline()] {
        let output = launch_with_seek(&seek_seconds(SEEK_MILLISECONDS), &pipeline);
        assert_starts_at_the_frame_after_the_seek(&checksum_lines(&output), 0);
    }
}

#[test]
fn mkv_seek_decodes_from_the_first_frame_at_or_after_the_target() {
    let output = launch_with_seek(&seek_seconds(SEEK_MILLISECONDS), &mkv_pipeline());
    assert_starts_at_the_frame_after_the_seek(&checksum_lines(&output), 0);
}

#[test]
fn ts_seek_counts_the_target_from_the_first_pts() {
    let output = launch_with_seek(&seek_seconds(SEEK_MILLISECONDS), &ts_pipeline());
    assert_starts_at_the_frame_after_the_seek(&checksum_lines(&output), TS_FIRST_PTS_NS);
}

#[test]
fn seek_past_the_end_fails_the_run() {
    let past_the_end = seek_seconds((FRAME_COUNT / FRAME_RATE + 1) * 1_000);
    // the TS timestamps run on past this target, the stream does not
    for pipeline in [
        mp4_pipeline(),
        mp4_decodebin_pipeline(),
        mkv_pipeline(),
        ts_pipeline(),
    ] {
        let output = launch_with_seek(&past_the_end, &pipeline);
        assert!(!output.status.success(), "{pipeline} ran past the end");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("past the end"),
            "{pipeline}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stdout.is_empty(), "{pipeline} output a frame");
    }
}

#[test]
fn seek_without_a_seekable_element_fails() {
    let output = launch_with_seek(
        &seek_seconds(SEEK_MILLISECONDS),
        "videotestsrc num-buffers=3 ! fakesink",
    );
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("nothing in the pipeline can seek"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
