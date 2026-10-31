#![cfg(feature = "ffmpeg")]

use std::process::Command;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn checksum_lines(pipeline: &str) -> Vec<String> {
    let output = Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .arg("-q")
        .args(pipeline.split_whitespace())
        .output()
        .expect("g2g-launch starts");
    assert!(
        output.status.success(),
        "{pipeline} fails: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}

fn assert_named_demux_decodes_like_decodebin(file: &str, demux: &str) {
    let location = fixture(file);
    let named = checksum_lines(&format!(
        "filesrc location={location} ! {demux} ! decodebin ! checksumsink hash=md5"
    ));
    let plugged = checksum_lines(&format!(
        "filesrc location={location} ! decodebin ! checksumsink hash=md5"
    ));
    assert!(!plugged.is_empty(), "{file} decodes through decodebin");
    assert_eq!(named, plugged, "{demux} ! decodebin decodes {file}");
}

#[test]
fn matroskademux_then_decodebin_decodes_an_h264_track() {
    assert_named_demux_decodes_like_decodebin(
        "h264_64x48_25fps_gop10_bframes_offset.mkv",
        "matroskademux",
    );
}

#[test]
fn fmp4demux_reads_a_fragmented_file() {
    assert_named_demux_decodes_like_decodebin(
        "h264_64x48_25fps_gop10_bframes_offset_frag.mp4",
        "fmp4demux",
    );
}

fn assert_demux_picks_stream(file: &str, demux: &str, consumer: &str, stream: &str) {
    let location = fixture(file);
    let picked = checksum_lines(&format!(
        "filesrc location={location} ! {demux} ! {consumer}checksumsink hash=md5"
    ));
    let named = checksum_lines(&format!(
        "filesrc location={location} ! {demux} stream={stream} ! {consumer}checksumsink hash=md5"
    ));
    assert!(!named.is_empty(), "{demux} stream={stream} reads {file}");
    assert_eq!(
        picked, named,
        "{demux} ! {consumer} takes the {stream} stream"
    );
}

#[test]
fn tsdemux_feeds_aacparse_without_a_stream_property() {
    assert_demux_picks_stream("av_h264_aac44100.ts", "tsdemux", "aacparse ! ", "aac");
}

#[test]
fn matroskademux_feeds_aacparse_without_a_stream_property() {
    assert_demux_picks_stream(
        "multitrack_h264_aac.mkv",
        "matroskademux",
        "aacparse ! ",
        "aac",
    );
}

#[test]
fn qtdemux_feeds_aacparse_without_a_stream_property() {
    assert_demux_picks_stream("av_h264_aac.mp4", "qtdemux", "aacparse ! ", "aac");
}

#[test]
fn flvdemux_feeds_aacparse_without_a_stream_property() {
    assert_demux_picks_stream("flv_h264_aac.flv", "flvdemux", "aacparse ! ", "aac");
}

#[test]
fn tsdemux_forwards_an_audio_only_file_to_any_consumer() {
    assert_demux_picks_stream("aac_44100.ts", "tsdemux", "", "aac");
}

#[test]
fn flvdemux_forwards_a_vp6_file_to_any_consumer() {
    assert_demux_picks_stream("vp6f_112x80.flv", "flvdemux", "", "vp6");
}

#[test]
fn oggdemux_forwards_a_video_only_file_to_any_consumer() {
    assert_demux_picks_stream("vp8_64x48_25fps_gop10_offset.ogg", "oggdemux", "", "vp8");
}
