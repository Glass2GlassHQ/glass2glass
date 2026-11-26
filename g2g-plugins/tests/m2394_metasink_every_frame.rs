// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "analytics-json")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::meta::{RelationKind, Tracking};
use g2g_core::{
    AnalyticsMeta, AnalyticsNode, BBox, Caps, Colorimetry, Dim, Interlace, ObjectDetection, Rate,
    RawVideoFormat,
};
use g2g_plugins::metasink::MetaSink;

const FRAME_SIZE: u32 = 100;
const TRACK_ID: u64 = 7;

fn frame_at(pts_ns: u64) -> Frame {
    Frame::from_system_bytes(
        vec![0u8; 4],
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        pts_ns,
    )
}

fn detection() -> ObjectDetection {
    ObjectDetection {
        bbox: BBox {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        },
        label: 0,
        confidence: 0.5,
    }
}

fn tracked_frame(pts_ns: u64) -> Frame {
    let mut frame = frame_at(pts_ns);
    let mut analytics = AnalyticsMeta::new();
    let tracked = analytics.add_detection(detection());
    let tracking = analytics.push(AnalyticsNode::Tracking(Tracking {
        object_id: TRACK_ID,
    }));
    analytics.relate(tracked, tracking, RelationKind::Tracks);
    analytics.add_detection(detection());
    frame.meta.attach(analytics);
    frame
}

#[tokio::test]
async fn metasink_writes_track_ids_and_a_line_for_every_frame() {
    let path = std::env::temp_dir().join("m2394_metasink.jsonl");
    let _ = std::fs::remove_file(&path);
    let mut sink = MetaSink::new().with_location(path.to_string_lossy().into_owned());
    sink.configure_pipeline(&Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(FRAME_SIZE),
        height: Dim::Fixed(FRAME_SIZE),
        framerate: Rate::Fixed(30 << 16),
        interlace: Interlace::Progressive,
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    })
    .expect("metasink accepts the caps");
    let mut out = CollectSink::default();
    for frame in [frame_at(0), tracked_frame(1_000), frame_at(2_000)] {
        sink.process(PipelinePacket::DataFrame(frame), &mut out)
            .await
            .expect("metasink writes the record");
    }
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("metasink flushes");

    let records: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .expect("the record file")
        .lines()
        .map(|line| serde_json::from_str(line).expect("a JSON line"))
        .collect();
    assert_eq!(records.len(), 3, "a line for each frame: {records:?}");
    assert_eq!(records[0]["detections"], serde_json::json!([]));
    assert_eq!(records[2]["detections"], serde_json::json!([]));
    let track_ids: Vec<&serde_json::Value> = records[1]["detections"]
        .as_array()
        .expect("the detections")
        .iter()
        .map(|detection| &detection["track_id"])
        .collect();
    assert_eq!(
        track_ids,
        [&serde_json::json!(TRACK_ID), &serde_json::Value::Null]
    );
    let _ = std::fs::remove_file(&path);
}
