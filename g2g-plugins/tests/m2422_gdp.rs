// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

// tools/gdp-fixtures.sh writes the GStreamer gdppay streams
#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use std::path::PathBuf;

use g2g_core::frame::Frame;
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, CapsSet, FrameTiming, PipelineClock, PipelinePacket, PropValue, Segment,
};
use g2g_plugins::gdp::{GdpDepay, GdpPay};
use g2g_plugins::registry::default_registry;

// the caps gst videotestsrc writes into the fixture
const NV12_CAPS: &str = "video/x-raw,format=NV12,width=64,height=48,framerate=30/1,pixel-aspect-ratio=1/1,interlace-mode=progressive";
const NV12_FRAME_BYTES: usize = 64 * 48 * 3 / 2;
const FRAME_PERIOD_NS: u64 = 1_000_000_000 / 30;

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

fn gdp(original: &str) -> Caps {
    Caps::Gdp {
        original: Some(Box::new(caps(original))),
    }
}

fn system(bytes: &[u8], timing: FrameTiming) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into())),
        timing,
        0,
    ))
}

fn frames(out: &CollectSink) -> Vec<(Vec<u8>, FrameTiming)> {
    out.packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some((
                frame
                    .domain
                    .as_system_slice()
                    .expect("system frame")
                    .to_vec(),
                frame.timing,
            )),
            _ => None,
        })
        .collect()
}

fn depayloader(original: &str) -> GdpDepay {
    let mut element = GdpDepay::new();
    element
        .configure_pipeline(&gdp(original))
        .expect("gdp input");
    element
}

async fn depayload(stream: &[u8], chunk_size: usize) -> CollectSink {
    let mut element = depayloader(NV12_CAPS);
    let mut out = CollectSink::default();
    for chunk in stream.chunks(chunk_size) {
        element
            .process(system(chunk, FrameTiming::default()), &mut out)
            .await
            .expect("depayloads");
    }
    element
        .process(PipelinePacket::Eos, &mut out)
        .await
        .expect("ends on a packet boundary");
    out
}

#[tokio::test]
async fn gst_streams_depayload_to_their_frames_caps_and_times() {
    let raw = read_fixture("gdp_nv12_64x48.raw");
    let stream = read_fixture("gdp_nv12_64x48.gdp");
    for chunk_size in [7, stream.len()] {
        let out = depayload(&stream, chunk_size).await;
        let frames = frames(&out);
        let bytes: Vec<Vec<u8>> = frames.iter().map(|(bytes, _)| bytes.clone()).collect();
        let expected: Vec<Vec<u8>> = raw.chunks(NV12_FRAME_BYTES).map(<[u8]>::to_vec).collect();
        assert_eq!(bytes, expected, "{chunk_size} byte chunks");
        let pts: Vec<u64> = frames.iter().map(|(_, timing)| timing.pts_ns).collect();
        assert_eq!(pts, [0, FRAME_PERIOD_NS]);
        assert!(frames.iter().all(|(_, timing)| timing.keyframe));
        assert!(out.packets.iter().any(
            |packet| matches!(packet, PipelinePacket::CapsChanged(c) if *c == caps(NV12_CAPS))
        ));
        assert!(out
            .packets
            .iter()
            .any(|packet| matches!(packet, PipelinePacket::Segment(s) if *s == Segment::new())));
    }
}

#[tokio::test]
async fn corrupt_headers_and_checksummed_payloads_fail() {
    let stream = read_fixture("gdp_nv12_64x48.gdp");
    // the first header's pts field, then the last byte of the last buffer's checksummed payload
    for corrupt_at in [12, stream.len() - 1] {
        let mut corrupt = stream.clone();
        corrupt[corrupt_at] ^= 0xFF;
        let result = depayloader(NV12_CAPS)
            .process(
                system(&corrupt, FrameTiming::default()),
                &mut CollectSink::default(),
            )
            .await;
        assert!(result.is_err(), "byte {corrupt_at}");
    }
}

#[tokio::test]
async fn a_truncated_stream_fails_at_end_of_stream() {
    let stream = read_fixture("gdp_nv12_64x48.gdp");
    let mut element = depayloader(NV12_CAPS);
    let mut out = CollectSink::default();
    element
        .process(
            system(&stream[..stream.len() - 1], FrameTiming::default()),
            &mut out,
        )
        .await
        .expect("waits for the rest of the last packet");
    assert_eq!(frames(&out).len(), 1);
    assert!(element
        .process(PipelinePacket::Eos, &mut out)
        .await
        .is_err());
}

#[tokio::test]
async fn payloaded_frames_depayload_with_their_times_and_segment() {
    let segment = Segment {
        rate: 2.0,
        start: 5 * FRAME_PERIOD_NS,
        stop: Some(9 * FRAME_PERIOD_NS),
        time: 5 * FRAME_PERIOD_NS as i64,
        ..Segment::new()
    };
    let raw = read_fixture("gdp_nv12_64x48.raw");
    let timings: Vec<FrameTiming> = (0..2u64)
        .map(|index| FrameTiming {
            pts_ns: (5 + index) * FRAME_PERIOD_NS,
            dts_ns: (4 + index) * FRAME_PERIOD_NS,
            duration_ns: FRAME_PERIOD_NS,
            keyframe: index == 0,
            ..FrameTiming::default()
        })
        .collect();
    let mut pay = GdpPay::new();
    pay.set_property("crc-payload", PropValue::Bool(true))
        .expect("crc-payload is a boolean");
    pay.configure_pipeline(&caps(NV12_CAPS)).expect("raw input");
    let mut payloaded = CollectSink::default();
    pay.process(PipelinePacket::Segment(segment), &mut payloaded)
        .await
        .expect("payloads the segment");
    for (frame, timing) in raw.chunks(NV12_FRAME_BYTES).zip(&timings) {
        pay.process(system(frame, *timing), &mut payloaded)
            .await
            .expect("payloads");
    }
    assert!(matches!(
        payloaded.packets.first(),
        Some(PipelinePacket::CapsChanged(c)) if *c == gdp(NV12_CAPS)
    ));
    let stream: Vec<u8> = frames(&payloaded)
        .into_iter()
        .flat_map(|(bytes, _)| bytes)
        .collect();

    let out = depayload(&stream, 13).await;
    let depayloaded = frames(&out);
    let expected: Vec<(Vec<u8>, FrameTiming)> = raw
        .chunks(NV12_FRAME_BYTES)
        .map(<[u8]>::to_vec)
        .zip(timings)
        .collect();
    assert_eq!(depayloaded, expected);
    let segments: Vec<&Segment> = out
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::Segment(segment) => Some(segment),
            _ => None,
        })
        .collect();
    assert_eq!(segments, [&segment]);
}

#[tokio::test]
async fn ts_offset_moves_timestamps_and_stops_at_zero() {
    let stream = read_fixture("gdp_nv12_64x48.gdp");
    let mut element = depayloader(NV12_CAPS);
    element
        .set_property("ts-offset", PropValue::Int(-(FRAME_PERIOD_NS as i64) - 1))
        .expect("ts-offset is an integer");
    let mut out = CollectSink::default();
    element
        .process(system(&stream, FrameTiming::default()), &mut out)
        .await
        .expect("depayloads");
    let pts: Vec<u64> = frames(&out)
        .iter()
        .map(|(_, timing)| timing.pts_ns)
        .collect();
    assert_eq!(pts, [0, 0]);
}

async fn run_launch(line: &str) {
    let graph =
        parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
}

#[tokio::test]
async fn filesrc_types_gst_files_and_g2g_streams_round_trip() {
    let directory = std::env::temp_dir();
    let output = |name: &str| directory.join(format!("g2g-m2422-{name}-{}", std::process::id()));
    for name in ["gdp_nv12_64x48", "gdp_s16_48k"] {
        let depayloaded = output(&format!("{name}.raw"));
        run_launch(&format!(
            "filesrc location={} ! gdpdepay ! filesink location={}",
            fixture(&format!("{name}.gdp")).display(),
            depayloaded.display()
        ))
        .await;
        let bytes = std::fs::read(&depayloaded).expect("depayloaded output");
        std::fs::remove_file(&depayloaded).ok();
        assert!(bytes == read_fixture(&format!("{name}.raw")), "{name}");
    }

    let source =
        "videotestsrc num-buffers=3 pattern=ball ! video/x-raw,format=NV12,width=64,height=48";
    let (direct, payloaded, round_trip) = (
        output("direct.raw"),
        output("payloaded.gdp"),
        output("round-trip.raw"),
    );
    run_launch(&format!(
        "{source} ! filesink location={}",
        direct.display()
    ))
    .await;
    run_launch(&format!(
        "{source} ! gdppay ! filesink location={}",
        payloaded.display()
    ))
    .await;
    run_launch(&format!(
        "filesrc location={} ! gdpdepay ! filesink location={}",
        payloaded.display(),
        round_trip.display()
    ))
    .await;
    let (direct_bytes, round_trip_bytes) = (
        std::fs::read(&direct).expect("direct output"),
        std::fs::read(&round_trip).expect("round trip output"),
    );
    for path in [&direct, &payloaded, &round_trip] {
        std::fs::remove_file(path).ok();
    }
    assert_eq!(direct_bytes.len(), 3 * NV12_FRAME_BYTES);
    assert!(direct_bytes == round_trip_bytes, "gdp changed the frames");
}
