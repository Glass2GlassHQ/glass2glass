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
