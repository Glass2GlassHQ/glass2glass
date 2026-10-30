#![cfg(feature = "ffmpeg")]

use std::process::{Command, Output};

use g2g_core::runtime::{parse_launch, run_graph_with_progress, PipelineProgress};
use g2g_core::PipelineClock;
use g2g_plugins::checksumsink::checksum_line;
use g2g_plugins::registry::default_registry;

#[cfg(feature = "hls")]
#[path = "common/http_routes.rs"]
mod http_routes;

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
// the same encode with -bf 2 -c:a aac into FLV: its AAC starts 21 ms before its video, after the video's first tag
const AUDIO_FIRST_FLV_FIXTURE: &str = "h264_aac_audio_first.flv";
// the keyframe at 10.4 s, one frame earlier than the target counted from the video's first pts
const AUDIO_FIRST_FLV_SEEK_MILLISECONDS: u64 = 421;
// ffmpeg -ss 0.421 -copyts -i <AUDIO_FIRST_FLV_FIXTURE> -map 0:v -frames:v 1 -f framemd5 -
const AUDIO_FIRST_FLV_FRAME_AFTER_SEEK_MD5: &str = "b7d8a515f21729cd2be1368bbe822d11";

// ffmpeg -i <TS_FIXTURE> -c copy -f hls -hls_time 1 -hls_playlist_type vod, its third segment starts at 2 s of playlist time
#[cfg(feature = "hls")]
const HLS_FIXTURE_DIRECTORY: &str = "hls_h264_64x48_25fps_gop10";
// in the third segment
#[cfg(feature = "hls")]
const HLS_SEEK_MILLISECONDS: u64 = 2_020;
// ffmpeg -copyts -i <TS_FIXTURE> -ss 3.42 -frames:v 1 -f framemd5 -, the frame at 2.04 s of playlist time
#[cfg(feature = "hls")]
const HLS_FRAME_AFTER_SEEK_MD5: &str = "fb27f06b296297026de6b636c57050df";

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

fn first_frame_at(milliseconds: u64) -> u64 {
    (milliseconds * NANOS_PER_MILLISECOND).div_ceil(FRAME_PERIOD_NS)
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
    let first_frame = first_frame_at(SEEK_MILLISECONDS);
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

// the frame where the seeked run leaves the unseeked one, checked to continue as the unseeked run from the target
fn seek_at_resume_frame(
    pipeline: &str,
    unseeked: &[String],
    position_milliseconds: u64,
    target_milliseconds: u64,
) -> usize {
    let seek_at = format!(
        "{}:{}",
        seek_seconds(position_milliseconds),
        seek_seconds(target_milliseconds)
    );
    let seeked = checksum_lines(&launch(&["--seek-at", &seek_at], pipeline));
    assert_ne!(
        seeked, unseeked,
        "{pipeline} --seek-at {seek_at} left the output unchanged"
    );
    let resumed = seeked
        .iter()
        .zip(unseeked)
        .position(|(seeked, unseeked)| seeked != unseeked)
        .unwrap_or(unseeked.len());
    assert!(
        resumed as u64 >= first_frame_at(position_milliseconds),
        "{pipeline} --seek-at {seek_at}: frame {resumed} came after the seek, before the sink reached the position"
    );
    assert_eq!(
        seeked[resumed..],
        unseeked[first_frame_at(target_milliseconds) as usize..],
        "{pipeline} --seek-at {seek_at}: the run resumes at the target and plays to the end"
    );
    resumed
}

#[test]
fn seek_during_playback_resumes_at_the_target() {
    let pipeline = mp4_pipeline();
    let unseeked = checksum_lines(&launch(&[], &pipeline));
    // forward past the frames in flight, and backward
    for (position_milliseconds, target_milliseconds) in [(200, 2_000), (2_000, 400)] {
        seek_at_resume_frame(
            &pipeline,
            &unseeked,
            position_milliseconds,
            target_milliseconds,
        );
    }
}

#[test]
fn seek_after_the_end_resumes_at_the_target() {
    let last_frame_milliseconds = (FRAME_COUNT - 1) * FRAME_PERIOD_NS / NANOS_PER_MILLISECOND;
    // decodebin picks mp4demux for a fragmented file, so fmp4demux is named
    let fmp4demux_pipeline = format!(
        "filesrc location={} ! fmp4demux ! decodebin ! checksumsink hash=md5",
        offset_fixture("h264_64x48_25fps_gop10_bframes_offset_frag.mp4")
    );
    let flv_pipeline = format!(
        "filesrc location={} ! decodebin ! checksumsink hash=md5",
        offset_fixture("h264_64x48_25fps_gop10_bframes_offset.flv")
    );
    for pipeline in [
        mp4_pipeline(),
        mkv_pipeline(),
        mp4_decodebin_pipeline(),
        fmp4demux_pipeline,
        flv_pipeline,
    ] {
        let unseeked = checksum_lines(&launch(&[], &pipeline));
        // the sink has every frame before the seek goes out
        let resumed = seek_at_resume_frame(&pipeline, &unseeked, last_frame_milliseconds, 400);
        assert_eq!(
            resumed,
            unseeked.len(),
            "{pipeline}: the whole unseeked run comes before the seek"
        );
    }
    // filesrc has read the whole mkv before the demuxer takes the seek, the mp4 source is near its end
    for (pipeline, position_milliseconds, target_milliseconds) in
        [(mkv_pipeline(), 400, 2_000), (mp4_pipeline(), 2_900, 400)]
    {
        let unseeked = checksum_lines(&launch(&[], &pipeline));
        seek_at_resume_frame(
            &pipeline,
            &unseeked,
            position_milliseconds,
            target_milliseconds,
        );
    }
}

#[test]
fn seek_after_the_source_ended_fails_in_a_muxer_pipeline() {
    let output_location = format!("{}/m1978_remuxed.ts", env!("CARGO_TARGET_TMPDIR"));
    // a muxer keeps the run ending at the end of the stream, filesrc has read the whole file by then
    let pipeline = format!(
        "filesrc location={TS_FIXTURE} ! tsdemux ! h264parse ! mpegtsmux ! filesink location={output_location}"
    );
    let output = launch(&["--seek-at", "2:0.4"], &pipeline);
    assert!(!output.status.success(), "the lost seek went unreported");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("the seek is not served"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
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

#[cfg(feature = "hls")]
fn hls_pipeline() -> String {
    let base = http_routes::serve(|path| {
        std::fs::read(offset_fixture(&format!("{HLS_FIXTURE_DIRECTORY}{path}"))).ok()
    });
    format!("hlssrc location={base}/media.m3u8 ! decodebin ! checksumsink hash=md5")
}

// the fixture's segments twice, the second time after an `#EXT-X-DISCONTINUITY` that sets its timestamps back to the first PTS
#[cfg(feature = "hls")]
fn hls_discontinuity_pipeline() -> String {
    const PLAYLIST_PATH: &str = "/twice.m3u8";
    let fixture = std::fs::read_to_string(offset_fixture(&format!(
        "{HLS_FIXTURE_DIRECTORY}/media.m3u8"
    )))
    .unwrap();
    let (head, segments) = fixture.split_at(fixture.find("#EXTINF").unwrap());
    let segments = segments.trim_end().strip_suffix("#EXT-X-ENDLIST").unwrap();
    let playlist = format!("{head}{segments}#EXT-X-DISCONTINUITY\n{segments}#EXT-X-ENDLIST\n");
    let base = http_routes::serve(move |path| match path {
        PLAYLIST_PATH => Some(playlist.clone().into_bytes()),
        segment => std::fs::read(offset_fixture(&format!("{HLS_FIXTURE_DIRECTORY}{segment}"))).ok(),
    });
    format!("hlssrc location={base}{PLAYLIST_PATH} ! decodebin ! checksumsink hash=md5")
}

#[cfg(feature = "hls")]
#[test]
fn hls_plays_on_across_a_discontinuity() {
    let once = checksum_lines(&launch(&[], &hls_pipeline()));
    let twice = checksum_lines(&launch(&[], &hls_discontinuity_pipeline()));
    assert_eq!(twice, [once.clone(), once].concat());
}

#[cfg(feature = "hls")]
#[test]
fn hls_seek_past_a_discontinuity_counts_from_the_playlist_start() {
    let once = checksum_lines(&launch(&[], &hls_pipeline()));
    let stream_seconds = FRAME_COUNT / FRAME_RATE;
    let target_milliseconds = stream_seconds * 1_000 + SEEK_MILLISECONDS;
    let seeked = checksum_lines(&launch_with_seek(
        &seek_seconds(target_milliseconds),
        &hls_discontinuity_pipeline(),
    ));
    assert_eq!(seeked, once[first_frame_at(SEEK_MILLISECONDS) as usize..]);
}

// the position after the discontinuity counts on in playlist time, so it seeks back to the last frame
#[cfg(feature = "hls")]
#[tokio::test]
async fn hls_position_counts_on_across_a_discontinuity() {
    let pipeline = hls_discontinuity_pipeline();
    let position_ns = final_position_ns(&pipeline).await;
    let last_frame_ns = (2 * FRAME_COUNT - 1) * FRAME_PERIOD_NS;
    assert_eq!(position_ns, last_frame_ns);
}

#[cfg(feature = "hls")]
#[test]
fn hls_seek_counts_the_target_from_the_playlist_start() {
    let output = launch_with_seek(&seek_seconds(HLS_SEEK_MILLISECONDS), &hls_pipeline());
    let lines = checksum_lines(&output);
    let first_frame = first_frame_at(HLS_SEEK_MILLISECONDS);
    assert_eq!(
        lines[0],
        checksum_line(
            TS_FIRST_PTS_NS + first_frame * FRAME_PERIOD_NS,
            HLS_FRAME_AFTER_SEEK_MD5.to_string()
        ),
        "the first frame out is ffmpeg's first frame after the seek"
    );
    assert_eq!(lines.len() as u64, FRAME_COUNT - first_frame);
}

#[cfg(feature = "hls")]
#[test]
fn hls_seek_after_the_end_resumes_at_the_target() {
    let last_frame_milliseconds = (FRAME_COUNT - 1) * FRAME_PERIOD_NS / NANOS_PER_MILLISECOND;
    let pipeline = hls_pipeline();
    let unseeked = checksum_lines(&launch(&[], &pipeline));
    let resumed = seek_at_resume_frame(&pipeline, &unseeked, last_frame_milliseconds, 400);
    assert_eq!(
        resumed,
        unseeked.len(),
        "the whole unseeked run comes before the seek"
    );
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
    let first_frame = first_frame_at(SEEK_MILLISECONDS) as usize;
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
fn seek_counts_the_target_from_the_earliest_stream() {
    for (fixture, seek_milliseconds, ffmpeg_md5) in [
        (
            AUDIO_FIRST_TS_FIXTURE,
            AUDIO_FIRST_SEEK_MILLISECONDS,
            AUDIO_FIRST_FRAME_AFTER_SEEK_MD5,
        ),
        (
            AUDIO_FIRST_FLV_FIXTURE,
            AUDIO_FIRST_FLV_SEEK_MILLISECONDS,
            AUDIO_FIRST_FLV_FRAME_AFTER_SEEK_MD5,
        ),
    ] {
        let pipeline = format!(
            "filesrc location={} ! decodebin ! checksumsink hash=md5",
            offset_fixture(fixture)
        );
        let seeked = checksum_lines(&launch_with_seek(
            &seek_seconds(seek_milliseconds),
            &pipeline,
        ));
        assert!(
            seeked[0].ends_with(ffmpeg_md5),
            "{fixture}: {} is not ffmpeg's frame after the seek",
            seeked[0]
        );
    }
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
