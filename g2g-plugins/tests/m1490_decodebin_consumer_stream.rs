//! M1490: a bare `decodebin` on an A/V container decodes the stream the element
//! after it accepts, so `decodebin ! audioconvert` gets the audio track.

#![cfg(all(feature = "std", feature = "ffmpeg"))]

use std::path::PathBuf;

use g2g_core::runtime::parse_launch;
use g2g_plugins::registry::default_registry;

const AV_MP4: &[u8] = include_bytes!("fixtures/av_h264_aac.mp4");
const AV_TS: &[u8] = include_bytes!("fixtures/av_h264_aac44100.ts");

fn temp(tag: &str, ext: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("g2g-m1490-{tag}-{}.{ext}", std::process::id()));
    std::fs::write(&path, bytes).expect("write fixture");
    path
}

fn chain_names(bytes: &[u8], ext: &str, consumer: &str) -> Vec<String> {
    let path = temp(consumer, ext, bytes);
    let line = format!(
        "filesrc location={} ! decodebin ! {consumer} ! fakesink",
        path.display()
    );
    let reg = default_registry();
    let parsed = parse_launch(&reg, &line);
    std::fs::remove_file(&path).ok();
    let graph = parsed.unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    vg.topo()
        .iter()
        .filter_map(|&n| vg.element(n).map(|e| e.log_category().to_string()))
        .collect()
}

fn has(names: &[String], category: &str) -> bool {
    names.iter().any(|n| n == category)
}

#[test]
fn audio_consumer_gets_the_mp4_audio_track() {
    let names = chain_names(AV_MP4, "mp4", "audioconvert");
    assert!(has(&names, "FfmpegAudioDec"), "audio decoded: {names:?}");
    assert!(
        !has(&names, "FfmpegVideoDec"),
        "no video decoder: {names:?}"
    );
}

#[test]
fn video_consumer_gets_the_mp4_video_track() {
    let names = chain_names(AV_MP4, "mp4", "videoconvert");
    assert!(has(&names, "FfmpegVideoDec"), "video decoded: {names:?}");
    assert!(
        !has(&names, "FfmpegAudioDec"),
        "no audio decoder: {names:?}"
    );
}

#[test]
fn audio_consumer_gets_the_ts_audio_stream() {
    let names = chain_names(AV_TS, "ts", "audioconvert");
    assert!(has(&names, "FfmpegAudioDec"), "audio decoded: {names:?}");
    assert!(
        !has(&names, "FfmpegVideoDec"),
        "no video decoder: {names:?}"
    );
}

#[test]
fn any_consumer_keeps_the_video_stream() {
    let names = chain_names(AV_TS, "ts", "identity");
    assert!(has(&names, "FfmpegVideoDec"), "video decoded: {names:?}");
}
