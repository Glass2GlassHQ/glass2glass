// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

// tools/uncv-fixtures.sh writes the GStreamer isomp4mux files
#![cfg(feature = "std")]

use std::path::PathBuf;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn scratch(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m2421-{}-{name}", std::process::id()))
}

async fn run(line: &str) {
    let graph =
        parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
}

// the bytes `chain` writes from `source`, the scratch file removed
async fn output_of(source: &str, chain: &str, name: &str) -> Vec<u8> {
    let output = scratch(name);
    run(&format!(
        "{source} ! {chain} ! filesink location={}",
        output.display()
    ))
    .await;
    let bytes = std::fs::read(&output).unwrap_or_else(|e| panic!("{name}: {e}"));
    std::fs::remove_file(&output).ok();
    bytes
}

#[tokio::test]
async fn gstreamer_uncv_files_demux_to_their_frames() {
    for (name, chain) in [
        ("uncv_rgb_18x4", "qtdemux stream=raw-rgb"),
        ("uncv_rgba_16x4", "qtdemux stream=raw-rgba"),
        ("uncv_i420_16x4", "qtdemux stream=raw-i420"),
        ("uncv_yuy2_16x4", "qtdemux stream=raw-yuy2"),
        ("uncv_grbg_8x4", "qtdemux stream=raw-grbg"),
        ("uncv_rggb16le_8x4", "qtdemux stream=raw-rggb16le"),
        (
            "uncv_nv12_16x4_zlib",
            "qtdemux stream=zlib-raw-nv12 ! zlibdecompress",
        ),
        ("uncv_nv12_16x4_zlib", "decodebin"),
    ] {
        let source = format!(
            "filesrc location={}",
            fixture(&format!("{name}.mp4")).display()
        );
        let frames = output_of(&source, chain, name).await;
        let expected = std::fs::read(fixture(&format!("{name}.raw"))).expect("fixture frames");
        assert!(frames == expected, "{name} through {chain}");
    }
}

#[tokio::test]
async fn raw_and_compressed_video_round_trip_through_mp4() {
    for (name, caps, wrap, unwrap) in [
        (
            "i420",
            "video/x-raw,format=I420,width=16,height=4",
            "identity",
            "qtdemux stream=raw-i420",
        ),
        (
            "rgb",
            "video/x-raw,format=RGB,width=18,height=4",
            "identity",
            "qtdemux stream=raw-rgb",
        ),
        (
            "bayer",
            "video/x-raw,format=ARGB,width=6,height=4 ! rgb2bayer ! video/x-bayer,format=gbrg",
            "identity",
            "qtdemux stream=raw-gbrg",
        ),
        (
            "nv12-brotli",
            "video/x-raw,format=NV12,width=16,height=4",
            "brotlicompress",
            "decodebin",
        ),
    ] {
        let source = format!("videotestsrc num-buffers=3 pattern=ball ! {caps}");
        let direct = output_of(&source, "identity", &format!("{name}.raw")).await;
        let file = scratch(&format!("{name}.mp4"));
        run(&format!(
            "{source} ! {wrap} ! mp4mux ! filesink location={}",
            file.display()
        ))
        .await;
        let read_back = output_of(
            &format!("filesrc location={}", file.display()),
            unwrap,
            &format!("{name}.read.raw"),
        )
        .await;
        std::fs::remove_file(&file).ok();
        assert!(!direct.is_empty(), "{name}");
        assert!(direct == read_back, "{name} changed through mp4");
    }
}
