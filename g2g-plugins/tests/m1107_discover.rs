//! M1107 media discovery: `g2g-discover` reports a file's container, its
//! elementary streams and their caps, and its duration, from container headers
//! alone.
//!
//! ffmpeg writes each fixture from the parameters below, so every assertion is
//! against the geometry / rate / length the fixture was *asked* for, never a
//! transcribed literal. ffprobe reads the same file as the reference peer: a
//! loopback against g2g's own muxers would hide a misread of what another
//! writer produces, which is the case a discoverer exists for.
//!
//! Run with the feature explicitly (`--json` needs `tooling-json`) and check the
//! reported test count is nonzero:
//! `cargo test -p g2g-plugins --features tooling-json --test m1107_discover`.
#![cfg(feature = "tooling-json")]

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// The shape every fixture is written at. Assertions read these, so changing a
/// fixture changes what is expected.
const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FRAMERATE: u32 = 30;
const DURATION_SECONDS: u32 = 2;
const AUDIO_SAMPLE_RATE: u32 = 44_100;
const AUDIO_CHANNELS: u32 = 2;
/// How far a reported duration may sit from ffprobe's for the same file. A
/// container rounds its length to whatever granularity its timescale (or its
/// last audio frame) allows.
const DURATION_TOLERANCE_SECONDS: f64 = 0.15;

/// The g2g caps media type for a codec ffprobe names. This correspondence is
/// what the cross-check asserts, so it is spelled out rather than guessed from
/// substrings: ffprobe's `aac` and g2g's `audio/mpeg` share no text.
const CODEC_MEDIA_TYPES: &[(&str, &str)] = &[
    ("h264", "video/x-h264"),
    ("aac", "audio/mpeg"),
    ("pcm_s16le", "audio/x-raw"),
];

fn tool_present(tool: &str) -> bool {
    Command::new(tool)
        .arg("-version")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn fixture_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g2g_m1107_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    dir
}

/// The lavfi video source at the fixture geometry / rate / length.
fn video_input() -> [String; 2] {
    [
        "-i".to_string(),
        format!("testsrc=size={WIDTH}x{HEIGHT}:rate={FRAMERATE}:duration={DURATION_SECONDS}"),
    ]
}

/// The lavfi audio source at the fixture rate / length.
fn audio_input() -> [String; 2] {
    [
        "-i".to_string(),
        format!("sine=frequency=440:duration={DURATION_SECONDS}:sample_rate={AUDIO_SAMPLE_RATE}"),
    ]
}

fn run_ffmpeg(args: &[String], out: &Path) {
    let mut command = Command::new("ffmpeg");
    command.args(["-v", "error", "-y"]).args(args).arg(out);
    let result = command.output().expect("run ffmpeg");
    assert!(
        result.status.success(),
        "ffmpeg failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}

/// H.264 video in MP4, no audio track.
fn write_mp4(dir: &Path) -> PathBuf {
    let path = dir.join("clip.mp4");
    let mut args = vec!["-f".to_string(), "lavfi".to_string()];
    args.extend(video_input());
    args.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"].map(String::from));
    run_ffmpeg(&args, &path);
    path
}

/// H.264 video plus AAC audio in Matroska, the multi-stream case.
fn write_mkv(dir: &Path) -> PathBuf {
    let path = dir.join("clip.mkv");
    let mut args = vec!["-f".to_string(), "lavfi".to_string()];
    args.extend(video_input());
    args.extend(["-f".to_string(), "lavfi".to_string()]);
    args.extend(audio_input());
    args.extend(
        [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ac",
        ]
        .map(String::from),
    );
    args.push(AUDIO_CHANNELS.to_string());
    run_ffmpeg(&args, &path);
    path
}

/// Uncompressed PCM in WAV, the single-stream container with no track list.
fn write_wav(dir: &Path) -> PathBuf {
    let path = dir.join("clip.wav");
    let mut args = vec!["-f".to_string(), "lavfi".to_string()];
    args.extend(audio_input());
    args.extend(["-ac".to_string(), AUDIO_CHANNELS.to_string()]);
    args.extend(["-c:a".to_string(), "pcm_s16le".to_string()]);
    run_ffmpeg(&args, &path);
    path
}

/// `g2g-discover <path> --json`, parsed.
fn discover(path: &Path) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_g2g-discover"))
        .arg(path)
        .arg("--json")
        .output()
        .expect("run g2g-discover");
    assert!(
        out.status.success(),
        "g2g-discover {} failed: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("valid JSON")
}

/// The reference peer's view of the same file: `(format duration in seconds,
/// each stream's codec_name in file order)`.
fn ffprobe(path: &Path) -> (f64, Vec<String>) {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(out.status.success(), "ffprobe failed on {}", path.display());
    let v: Value = serde_json::from_slice(&out.stdout).expect("ffprobe JSON");
    let duration = v["format"]["duration"]
        .as_str()
        .expect("ffprobe format duration")
        .parse()
        .expect("duration is a number");
    let codecs = v["streams"]
        .as_array()
        .expect("ffprobe streams")
        .iter()
        .map(|s| s["codec_name"].as_str().expect("codec_name").to_string())
        .collect();
    (duration, codecs)
}

fn media_type_for(codec_name: &str) -> &'static str {
    CODEC_MEDIA_TYPES
        .iter()
        .find(|(name, _)| *name == codec_name)
        .map(|(_, media_type)| *media_type)
        .unwrap_or_else(|| panic!("no g2g media type known for ffprobe codec {codec_name}"))
}

fn duration_seconds(info: &Value) -> f64 {
    info["duration_ns"].as_f64().expect("duration reported") / 1e9
}

/// Every stream g2g reported carries the media type ffprobe named for the
/// stream at the same position, and the two agree on the file's length.
fn cross_check_against_ffprobe(info: &Value, path: &Path) {
    let (probe_duration, probe_codecs) = ffprobe(path);
    let streams = info["streams"].as_array().expect("streams array");
    assert_eq!(
        streams.len(),
        probe_codecs.len(),
        "g2g and ffprobe disagree on the stream count of {}",
        path.display()
    );
    for (stream, codec_name) in streams.iter().zip(&probe_codecs) {
        let caps = stream["caps"].as_str().expect("caps string");
        assert!(
            caps.starts_with(media_type_for(codec_name)),
            "{}: g2g reported {caps} where ffprobe found {codec_name}",
            path.display()
        );
    }
    let delta = (duration_seconds(info) - probe_duration).abs();
    assert!(
        delta <= DURATION_TOLERANCE_SECONDS,
        "{}: g2g says {}s, ffprobe says {probe_duration}s",
        path.display(),
        duration_seconds(info)
    );
}

/// The duration is the one the fixture was written at, within what a container
/// rounds to.
fn assert_fixture_duration(info: &Value) {
    let seconds = duration_seconds(info);
    let delta = (seconds - f64::from(DURATION_SECONDS)).abs();
    assert!(
        delta <= DURATION_TOLERANCE_SECONDS,
        "expected the {DURATION_SECONDS}s fixture length, got {seconds}s"
    );
}

fn video_stream(info: &Value) -> Value {
    info["streams"]
        .as_array()
        .expect("streams array")
        .iter()
        .find(|s| s["type"] == "video")
        .expect("a video stream")
        .clone()
}

fn audio_stream(info: &Value) -> Value {
    info["streams"]
        .as_array()
        .expect("streams array")
        .iter()
        .find(|s| s["type"] == "audio")
        .expect("an audio stream")
        .clone()
}

#[test]
fn mp4_reports_container_video_geometry_and_duration() {
    if !tool_present("ffmpeg") || !tool_present("ffprobe") {
        eprintln!("skipping: ffmpeg / ffprobe not on PATH");
        return;
    }
    let dir = fixture_dir();
    let path = write_mp4(&dir);
    let info = discover(&path);

    assert_eq!(info["container"], "video/quicktime");
    assert_fixture_duration(&info);
    let video = video_stream(&info);
    assert_eq!(video["width"], WIDTH, "the geometry ffmpeg encoded at");
    assert_eq!(video["height"], HEIGHT);
    assert!(video["caps"].as_str().unwrap().starts_with("video/x-h264"));
    cross_check_against_ffprobe(&info, &path);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn matroska_reports_both_streams_with_their_shapes() {
    if !tool_present("ffmpeg") || !tool_present("ffprobe") {
        eprintln!("skipping: ffmpeg / ffprobe not on PATH");
        return;
    }
    let dir = fixture_dir();
    let path = write_mkv(&dir);
    let info = discover(&path);

    assert_eq!(info["container"], "video/x-matroska");
    assert_fixture_duration(&info);
    assert_eq!(
        info["streams"].as_array().unwrap().len(),
        2,
        "the container's whole track list, not just the forwarded stream"
    );
    let video = video_stream(&info);
    assert_eq!(video["width"], WIDTH);
    assert_eq!(video["height"], HEIGHT);
    let audio = audio_stream(&info);
    assert_eq!(
        audio["sample_rate"], AUDIO_SAMPLE_RATE,
        "the rate ffmpeg encoded at"
    );
    cross_check_against_ffprobe(&info, &path);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn wav_reports_channels_rate_and_duration_without_a_track_list() {
    if !tool_present("ffmpeg") || !tool_present("ffprobe") {
        eprintln!("skipping: ffmpeg / ffprobe not on PATH");
        return;
    }
    let dir = fixture_dir();
    let path = write_wav(&dir);
    let info = discover(&path);

    assert_eq!(info["container"], "audio/x-wav");
    assert_fixture_duration(&info);
    let audio = audio_stream(&info);
    assert_eq!(audio["channels"], AUDIO_CHANNELS);
    assert_eq!(audio["sample_rate"], AUDIO_SAMPLE_RATE);
    cross_check_against_ffprobe(&info, &path);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_network_uri_is_refused_rather_than_fetched() {
    let out = Command::new(env!("CARGO_BIN_EXE_g2g-discover"))
        .arg("rtsp://camera.invalid/stream")
        .output()
        .expect("run g2g-discover");
    assert!(!out.status.success(), "an unsupported scheme must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("rtsp") && stderr.contains("local files"),
        "the error should name the scheme it cannot open, got: {stderr}"
    );
}
