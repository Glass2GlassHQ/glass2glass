#![cfg(feature = "std")]

#[path = "common/raw_convert.rs"]
mod raw_convert;
use raw_convert::{fixture, read_fixture, temp_path, ZeroClock};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::registry::default_registry;

// (two-frame file from `tools/raw-video-fixtures.sh`, the demuxer, the stream it selects, the reference beside the file)
const LAYOUTS: [(&str, &str, &str, &str); 11] = [
    ("avi_bgr24_7x5.avi", "avidemux", "raw-bgr", "bgr24"),
    ("avi_gst_bgr_17x16.avi", "avidemux", "raw-bgr", "bgr24"),
    ("avi_bgra_7x5.avi", "avidemux", "raw-bgrx", "bgra"),
    ("avi_gst_bgrx_17x16.avi", "avidemux", "raw-bgrx", "bgra"),
    ("avi_yv12_7x5.avi", "avidemux", "raw-i420", "yuv420p"),
    ("avi_y800_7x5.avi", "avidemux", "raw-gray8", "gray"),
    ("avi_grey_7x5.avi", "avidemux", "raw-gray8", "gray"),
    ("avi_gst_gray8_17x16.avi", "avidemux", "raw-gray8", "gray"),
    ("mov_rgb24_6x5.mov", "qtdemux", "raw-rgb", "rgb24"),
    ("mov_argb_7x5.mov", "qtdemux", "raw-argb", "argb"),
    // ffmpeg reads this file's two-byte row padding as pixels, so it is checked against the frames ffmpeg encoded
    ("mov_rgb24_7x5.mov", "qtdemux", "raw-rgb", "source.rgb24"),
];

// what `filesrc ! <demuxer> stream=<stream> ! filesink` writes for `file`
async fn demux(file: &str, demuxer: &str, stream: &str) -> Vec<u8> {
    let out = temp_path(&format!("m1729-{file}-{stream}"));
    let line = format!(
        "filesrc location={} ! {demuxer} stream={stream} ! filesink location={}",
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
async fn uncompressed_layouts_demux_to_ffmpegs_decode() {
    for (file, demuxer, stream, reference) in LAYOUTS {
        let frames = demux(file, demuxer, stream).await;
        assert!(
            frames == read_fixture(&format!("{file}.{reference}")),
            "{file} through {demuxer}"
        );
    }
}
