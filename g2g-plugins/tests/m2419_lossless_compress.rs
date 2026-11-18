// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

// tools/zlib-fixtures.sh writes the C zlib streams
#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, CapsSet, FrameTiming, LosslessCompression, PipelineClock, PipelinePacket,
    PropValue,
};
use g2g_plugins::losslesscompress::{LosslessCompress, LosslessDecompress};
use g2g_plugins::registry::default_registry;

const NV12_CAPS: &str = "video/x-raw,format=NV12,width=64,height=48";

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

fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixture(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

fn caps(text: &str) -> Caps {
    let set = CapsSet::from_gst_string(text).unwrap_or_else(|| panic!("caps {text}"));
    let [caps] = set.alternatives() else {
        panic!("{text} is not fixed");
    };
    caps.clone()
}

fn compressed(compression: LosslessCompression, original: &str) -> Caps {
    Caps::LosslesslyCompressed {
        compression,
        original: Some(Box::new(caps(original))),
    }
}

fn system(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into())),
        FrameTiming::default(),
        0,
    ))
}

fn frames(out: &CollectSink) -> Vec<Vec<u8>> {
    out.packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(
                frame
                    .domain
                    .as_system_slice()
                    .expect("system frame")
                    .to_vec(),
            ),
            _ => None,
        })
        .collect()
}

fn decompressor(compression: LosslessCompression) -> LosslessDecompress {
    let mut element = LosslessDecompress::new(compression);
    element
        .configure_pipeline(&compressed(compression, NV12_CAPS))
        .expect("compressed NV12 input");
    element
}

#[tokio::test]
async fn c_zlib_streams_decompress_to_the_original_frame() {
    let raw = read_fixture("nv12_64x48.raw");
    for (compression, stream) in [
        (LosslessCompression::Zlib, "nv12_64x48.zlib"),
        (LosslessCompression::Deflate, "nv12_64x48.deflate"),
    ] {
        let mut element = decompressor(compression);
        let mut out = CollectSink::default();
        element
            .process(system(&read_fixture(stream)), &mut out)
            .await
            .expect("decompresses");
        assert_eq!(frames(&out), vec![raw.clone()], "{stream}");
        assert!(
            out.packets.iter().any(
                |packet| matches!(packet, PipelinePacket::CapsChanged(c) if *c == caps(NV12_CAPS))
            ),
            "{stream} restores the original caps"
        );
    }
}

#[tokio::test]
async fn streams_split_across_buffers_and_joined_in_one_decompress_whole() {
    let raw = read_fixture("nv12_64x48.raw");
    let stream = read_fixture("nv12_64x48.zlib");
    let two_streams = [stream.clone(), stream].concat();
    let mut element = decompressor(LosslessCompression::Zlib);
    let mut out = CollectSink::default();
    for chunk in two_streams.chunks(97) {
        element
            .process(system(chunk), &mut out)
            .await
            .expect("decompresses");
    }
    assert_eq!(frames(&out), vec![raw.clone(), raw]);
}

#[tokio::test]
async fn a_stream_larger_than_the_original_frame_fails() {
    let raw = read_fixture("nv12_64x48.raw");
    let mut compress = LosslessCompress::new(LosslessCompression::Deflate);
    compress
        .configure_pipeline(&caps(NV12_CAPS))
        .expect("raw input");
    let mut out = CollectSink::default();
    compress
        .process(system(&[raw.clone(), raw].concat()), &mut out)
        .await
        .expect("compresses");
    let oversized = frames(&out).remove(0);
    let result = decompressor(LosslessCompression::Deflate)
        .process(system(&oversized), &mut CollectSink::default())
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn a_truncated_stream_fails_at_end_of_stream() {
    let stream = read_fixture("nv12_64x48.zlib");
    let mut element = decompressor(LosslessCompression::Zlib);
    let mut out = CollectSink::default();
    element
        .process(system(&stream[..stream.len() / 2]), &mut out)
        .await
        .expect("half a stream waits for more");
    assert!(frames(&out).is_empty());
    assert!(element
        .process(PipelinePacket::Eos, &mut out)
        .await
        .is_err());
}

#[tokio::test]
async fn the_level_property_takes_zero_to_nine() {
    let mut element = LosslessCompress::new(LosslessCompression::Zlib);
    assert_eq!(element.get_property("level"), Some(PropValue::Uint(6)));
    element
        .set_property("level", PropValue::Uint(9))
        .expect("level 9");
    assert_eq!(element.get_property("level"), Some(PropValue::Uint(9)));
    assert!(element.set_property("level", PropValue::Uint(10)).is_err());
}

#[tokio::test]
async fn video_survives_a_compress_and_decompress_round_trip() {
    let directory = std::env::temp_dir();
    for (compress, decompress) in [
        ("zlibcompress level=1", "zlibdecompress"),
        ("deflatecompress level=9", "deflatedecompress"),
    ] {
        let source =
            "videotestsrc num-buffers=5 pattern=ball ! video/x-raw,format=NV12,width=64,height=48";
        let direct = directory.join(format!("g2g-m2419-direct-{}.raw", std::process::id()));
        let round_trip = directory.join(format!("g2g-m2419-round-trip-{}.raw", std::process::id()));
        for (line, output) in [
            (
                format!("{source} ! filesink location={}", direct.display()),
                &direct,
            ),
            (
                format!(
                    "{source} ! {compress} ! {decompress} ! filesink location={}",
                    round_trip.display()
                ),
                &round_trip,
            ),
        ] {
            let graph = parse_launch(&default_registry(), &line)
                .unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
            run_graph(graph, &ZeroClock, 4)
                .await
                .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
            assert!(output.exists(), "{line}");
        }
        let (direct_bytes, round_trip_bytes) = (
            std::fs::read(&direct).expect("direct output"),
            std::fs::read(&round_trip).expect("round trip output"),
        );
        std::fs::remove_file(&direct).ok();
        std::fs::remove_file(&round_trip).ok();
        assert_eq!(direct_bytes.len(), 5 * 64 * 48 * 3 / 2, "{compress}");
        assert!(
            direct_bytes == round_trip_bytes,
            "{compress} changed the frames"
        );
    }
}

#[test]
fn decodebin_plugs_the_decompressor_and_then_demosaics_bayer() {
    let registry = default_registry();
    let decode = |input: &Caps| registry.autoplug_names(input, &g2g_core::runtime::is_raw_video, 4);
    assert_eq!(
        decode(&compressed(LosslessCompression::Zlib, NV12_CAPS)),
        Some(vec!["zlibdecompress"])
    );
    assert_eq!(
        decode(&compressed(
            LosslessCompression::Deflate,
            "video/x-bayer,format=rggb,width=12,height=8"
        )),
        Some(vec!["deflatedecompress", "bayer2rgb"])
    );
}

#[tokio::test]
async fn decodebin_demosaics_a_bayer_dump() {
    let output = std::env::temp_dir().join(format!("g2g-m2419-bayer-{}.bgra", std::process::id()));
    let line = format!(
        "filesrc location={} ! rawvideoparse format=rggb width=12 height=8 ! decodebin ! video/x-raw,format=BGRA ! filesink location={}",
        fixture("bayer_12x8_8bit.raw").display(),
        output.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    let written = std::fs::read(&output).expect("the sink wrote a file");
    std::fs::remove_file(&output).ok();
    assert!(written == read_fixture("bayer_12x8_rggb_to_bgra.raw"));
}
