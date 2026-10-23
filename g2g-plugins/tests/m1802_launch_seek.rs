#![cfg(feature = "ffmpeg")]

use std::process::{Command, Output};

use g2g_core::runtime::{parse_launch, run_graph_with_progress, PipelineProgress};
use g2g_core::PipelineClock;
use g2g_plugins::checksumsink::checksum_line;
use g2g_plugins::registry::default_registry;

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
// g2g-launch's link capacity
const LINK_CAPACITY: usize = 4;
// ffmpeg -ss 1.02 -copyts -i <mp4 or mkv fixture> -frames:v 1 -f framemd5 -, the frame at 1.04
// the TS fixture needs an output -ss 2.42 (first PTS plus 1.02), ffmpeg's input seek in it lands on the 1.2 s keyframe
const FRAME_AFTER_SEEK_MD5: &str = "732b16a0536b40e66a6005fa0cca8569";

fn offset_fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

// The same encode with -bf 3 -output_ts_offset 0.4, the fragmented one also -movflags +frag_keyframe+empty_moov.
// ffprobe start_time is 0.4 for mkv, flv and mp4 (an edit list) and 0.08 for the fragmented mp4.
const H264_OFFSET_FIXTURES: [&str; 4] = [
    "h264_64x48_25fps_gop10_bframes_offset.mkv",
    "h264_64x48_25fps_gop10_bframes_offset.flv",
    "h264_64x48_25fps_gop10_bframes_offset.mp4",
    "h264_64x48_25fps_gop10_bframes_offset_frag.mp4",
];
// ffmpeg -ss 1.02 -copyts -i <any H264_OFFSET_FIXTURES> -frames:v 1 -f framemd5 -
const H264_OFFSET_FRAME_AFTER_SEEK_MD5: &str = "0f522e7ea384962044bb0cb76385ec24";
// ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=3 -c:v libvpx -g 10 -auto-alt-ref 0 -output_ts_offset 0.4, start_time 0.4
const VP8_OFFSET_FIXTURE: &str = "vp8_64x48_25fps_gop10_offset.ogg";
// ffmpeg -ss 1.02 -copyts -i <VP8_OFFSET_FIXTURE> -frames:v 1 -f framemd5 -
const VP8_OFFSET_FRAME_AFTER_SEEK_MD5: &str = "b125eb8b291a171ef049315747ed794a";
// ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=1.2 -f lavfi -i sine=frequency=440:sample_rate=48000:duration=1.2
//   -c:v libx264 -g 10 -keyint_min 10 -sc_threshold 0 -bf 0 -c:a mp2 -ac 1 -b:a 32k -output_ts_offset 10
// its audio starts 10 ms before its video, where ffprobe's start_time is
const AUDIO_FIRST_TS_FIXTURE: &str = "h264_mp2_audio_first.ts";
// lands one frame later when counted from the video's first pts
const AUDIO_FIRST_SEEK_MILLISECONDS: u64 = 410;
// ffmpeg -ss 0.41 -copyts -i <AUDIO_FIRST_TS_FIXTURE> -map 0:v -frames:v 1 -f framemd5 -
const AUDIO_FIRST_FRAME_AFTER_SEEK_MD5: &str = "f8be4f7e5b101f797f1fcecb3159cf25";

fn seek_seconds(milliseconds: u64) -> String {
    format!("{}", milliseconds as f64 / 1_000.0)
}

fn launch(options: &[&str], pipeline: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .arg("-q")
        .args(options)
        .args(pipeline.split_whitespace())
        .output()
        .expect("g2g-launch starts")
}

fn launch_with_seek(seconds: &str, pipeline: &str) -> Output {
    launch(&["--seek", seconds], pipeline)
}

fn first_frame_after_seek() -> u64 {
    (SEEK_MILLISECONDS * NANOS_PER_MILLISECOND).div_ceil(FRAME_PERIOD_NS)
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
    let first_frame = first_frame_after_seek();
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

// each pipeline with the md5 ffmpeg gives for the first frame after the seek
fn offset_pipelines() -> Vec<(String, &'static str)> {
    let mut pipelines: Vec<(String, &'static str)> = Vec::new();
    for name in H264_OFFSET_FIXTURES {
        let location = offset_fixture(name);
        pipelines.push((
            format!("filesrc location={location} ! decodebin ! checksumsink hash=md5"),
            H264_OFFSET_FRAME_AFTER_SEEK_MD5,
        ));
        if name.ends_with(".mp4") {
            pipelines.push((
                format!("uridecodebin uri=file://{location} ! checksumsink hash=md5"),
                H264_OFFSET_FRAME_AFTER_SEEK_MD5,
            ));
        }
    }
    pipelines.push((
        format!(
            "filesrc location={} ! decodebin ! checksumsink hash=md5",
            offset_fixture(VP8_OFFSET_FIXTURE)
        ),
        VP8_OFFSET_FRAME_AFTER_SEEK_MD5,
    ));
    pipelines
}

#[test]
fn seek_counts_the_target_from_the_start_time() {
    let first_frame = first_frame_after_seek() as usize;
    for (pipeline, ffmpeg_md5) in offset_pipelines() {
        let unseeked = checksum_lines(&launch(&[], &pipeline));
        let seeked = checksum_lines(&launch_with_seek(
            &seek_seconds(SEEK_MILLISECONDS),
            &pipeline,
        ));
        assert_eq!(
            seeked,
            unseeked[first_frame..],
            "{pipeline}: the seek lands on frame {first_frame} of the unseeked run"
        );
        assert!(
            seeked[0].ends_with(ffmpeg_md5),
            "{pipeline}: {} is not ffmpeg's frame after the seek",
            seeked[0]
        );
    }
}

#[test]
fn ts_seek_counts_the_target_from_the_earliest_stream() {
    let pipeline = format!(
        "filesrc location={} ! decodebin ! checksumsink hash=md5",
        offset_fixture(AUDIO_FIRST_TS_FIXTURE)
    );
    let seeked = checksum_lines(&launch_with_seek(
        &seek_seconds(AUDIO_FIRST_SEEK_MILLISECONDS),
        &pipeline,
    ));
    assert!(
        seeked[0].ends_with(AUDIO_FIRST_FRAME_AFTER_SEEK_MD5),
        "{} is not ffmpeg's frame after the seek",
        seeked[0]
    );
}

struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// the position g2g reports once the run's last frame has reached the sink
async fn final_position_ns(pipeline: &str) -> u64 {
    let graph = parse_launch(&default_registry(), pipeline)
        .unwrap_or_else(|err| panic!("parses `{pipeline}`: {err}"));
    let progress = PipelineProgress::new();
    run_graph_with_progress(graph, &ZeroClock, LINK_CAPACITY, &progress, None)
        .await
        .unwrap_or_else(|err| panic!("runs `{pipeline}`: {err:?}"));
    progress.position().expect("a frame reached the sink")
}

fn nanoseconds_as_seconds(nanoseconds: u64) -> String {
    format!(
        "{}.{:09}",
        nanoseconds / NANOS_PER_SECOND,
        nanoseconds % NANOS_PER_SECOND
    )
}

#[tokio::test]
async fn a_position_read_back_seeks_to_its_frame() {
    let mut pipelines: Vec<String> = offset_pipelines()
        .into_iter()
        .map(|(pipeline, _)| pipeline)
        .collect();
    pipelines.push(ts_pipeline());
    for pipeline in pipelines {
        let unseeked = checksum_lines(&launch(&[], &pipeline));
        let position_ns = final_position_ns(&pipeline).await;
        let seeked = checksum_lines(&launch_with_seek(
            &nanoseconds_as_seconds(position_ns),
            &pipeline,
        ));
        assert_eq!(
            seeked,
            unseeked[unseeked.len() - 1..],
            "{pipeline}: a seek to the last frame's position {position_ns} gives that frame"
        );
    }
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
