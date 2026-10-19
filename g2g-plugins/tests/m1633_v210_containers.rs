//! QuickTime, AVI and Matroska `v210` tracks demux as v210 raw video, checked
//! against ffmpeg's stream copy and decode of the fixtures
//! `tools/packed-10bit-fixtures.sh` writes.
#![cfg(feature = "std")]

#[path = "common/raw_convert.rs"]
mod raw_convert;
use raw_convert::{fixture, read_fixture, temp_path, ZeroClock};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::registry::default_registry;

// (two-frame file, the demuxer that reads it)
const CONTAINERS: [(&str, &str); 3] = [
    ("v210_52x4.mov", "qtdemux"),
    ("v210_52x4.avi", "avidemux"),
    ("v210_52x4.mkv", "matroskademux"),
];
const STREAM: &str = "raw-v210";
// beside each file, ffmpeg's stream copy of its frames and its decode of them
const STREAM_COPY_SUFFIX: &str = "v210";
const DECODE_SUFFIX: &str = "yuv422p10le";
const DECODE_CAPS: &str = "video/x-raw,format=I422_10LE";

// what `filesrc ! <demux> stream=raw-v210 <tail> ! filesink` writes for `file`
async fn demux(file: &str, demuxer: &str, tail: &str, tag: &str) -> Vec<u8> {
    let out = temp_path(&format!("{file}-{tag}"));
    let line = format!(
        "filesrc location={} ! {demuxer} stream={STREAM}{tail} ! filesink location={}",
        fixture(file).display(),
        out.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let bytes = std::fs::read(&out).expect("the sink wrote its file");
    std::fs::remove_file(&out).ok();
    bytes
}

#[tokio::test]
async fn a_v210_track_demuxes_to_the_frames_ffmpeg_copies() {
    for (file, demuxer) in CONTAINERS {
        let frames = demux(file, demuxer, "", "copy").await;
        assert!(
            frames == read_fixture(&format!("{file}.{STREAM_COPY_SUFFIX}")),
            "{demuxer}"
        );
    }
}

#[tokio::test]
async fn a_demuxed_v210_track_converts_to_ffmpegs_decode() {
    let tail = format!(" ! videoconvert ! {DECODE_CAPS}");
    for (file, demuxer) in CONTAINERS {
        let planar = demux(file, demuxer, &tail, "decode").await;
        assert!(
            planar == read_fixture(&format!("{file}.{DECODE_SUFFIX}")),
            "{demuxer}"
        );
    }
}
