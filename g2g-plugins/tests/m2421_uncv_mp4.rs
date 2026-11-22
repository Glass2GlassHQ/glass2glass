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
async fn libheif_planes_decode_to_the_graph_layout() {
    // libheif 1.23.7 writes uncC entries and planes in addition order and
    // bit-packs high depths by row. The raw references hold the original
    // frames as graph buffers, with LE words and padding for Bayer rows.
    for (name, stream) in [
        ("uncv_libheif_brg_6x4", "raw-rgbp"),
        ("uncv_libheif_vyu_6x4", "raw-i420"),
        ("uncv_libheif_packed_rggb12_5x3", "raw-rggb12le"),
        ("uncv_libheif_packed_vyu10_6x4", "raw-i420_10le"),
        ("uncv_libheif_packed_brg10_5x3", "raw-gbr_10le"),
    ] {
        let source = format!(
            "filesrc location={}",
            fixture(&format!("{name}.mp4")).display()
        );
        let output = output_of(&source, &format!("qtdemux stream={stream}"), name).await;
        let reference = std::fs::read(fixture(&format!("{name}.raw"))).expect("fixture frames");
        assert_eq!(output, reference, "{name}");
    }
}

#[tokio::test]
async fn subsampled_packed_rows_accept_both_chroma_alignments() {
    // Both clips use the libheif V,Y,U fixture's samples with row_align_size=16.
    // The ISO clip pads chroma rows to 8 bytes. The libheif clip uses 16 bytes.
    let reference = std::fs::read(fixture("uncv_libheif_packed_vyu10_6x4.raw")).unwrap();
    for layout in ["iso", "libheif"] {
        let name = format!("uncv_padded_vyu10_{layout}");
        let source = format!(
            "filesrc location={}",
            fixture(&format!("{name}.mp4")).display()
        );
        let output = output_of(&source, "qtdemux stream=raw-i420_10le", &name).await;
        assert_eq!(output, reference, "{layout}");
    }
}

#[tokio::test]
async fn row_interleaved_uncv_decodes_to_planar_frames() {
    // The libheif BRG samples are interleaved one component row at a time,
    // padded to eight bytes per row, and checked with libheif's row decoder.
    for (name, reference, stream) in [
        ("uncv_row_brg_6x4", "uncv_libheif_brg_6x4.raw", "raw-rgbp"),
        (
            "uncv_row_brg10_5x3",
            "uncv_libheif_packed_brg10_5x3.raw",
            "raw-gbr_10le",
        ),
    ] {
        let source = format!(
            "filesrc location={}",
            fixture(&format!("{name}.mp4")).display()
        );
        let output = output_of(&source, &format!("qtdemux stream={stream}"), name).await;
        assert_eq!(output, std::fs::read(fixture(reference)).unwrap(), "{name}");
    }
}

#[tokio::test]
async fn chroma_first_uncv_decodes_to_graph_formats() {
    // The fixtures put UV or VU first, with padded 8- and 10-bit rows and
    // unpadded 12-bit rows. Libheif's mixed decoder verifies the source samples.
    for (name, reference, stream) in [
        (
            "uncv_mixed_vuy420_6x4",
            "uncv_mixed_vuy420_6x4.raw",
            "raw-nv12",
        ),
        (
            "uncv_mixed_uvy422_6x4",
            "uncv_mixed_uvy422_6x4.raw",
            "raw-y42b",
        ),
        (
            "uncv_mixed_vuy420p10_6x4",
            "uncv_libheif_packed_vyu10_6x4.raw",
            "raw-i420_10le",
        ),
        (
            "uncv_mixed_uvy422p12_6x4",
            "uncv_mixed_uvy422p12_6x4.raw",
            "raw-i422_12le",
        ),
    ] {
        let source = format!(
            "filesrc location={}",
            fixture(&format!("{name}.mp4")).display()
        );
        let output = output_of(&source, &format!("qtdemux stream={stream}"), name).await;
        assert_eq!(output, std::fs::read(fixture(reference)).unwrap(), "{name}");
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
        assert!(!direct.is_empty(), "{name}");
        for fragmented in [true, false] {
            let file = scratch(&format!("{name}-{fragmented}.mp4"));
            run(&format!(
                "{source} ! {wrap} ! mp4mux fragmented={fragmented} ! filesink location={}",
                file.display()
            ))
            .await;
            let bytes = std::fs::read(&file).expect("MP4 output");
            assert_eq!(bytes.windows(4).any(|fourcc| fourcc == b"moof"), fragmented);
            let read_back = output_of(
                &format!("filesrc location={}", file.display()),
                unwrap,
                &format!("{name}-{fragmented}.read.raw"),
            )
            .await;
            std::fs::remove_file(&file).ok();
            assert!(
                direct == read_back,
                "{name} changed through mp4 (fragmented={fragmented})"
            );
        }
    }
}
