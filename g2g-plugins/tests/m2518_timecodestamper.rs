// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "std", feature = "metadata"))]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, CapsSet, Frame, FrameTiming, MemoryDomain, PipelinePacket, PropValue,
    Segment, TimecodeMeta,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::timecodestamper::TimecodeStamper;

use collect_sink::CollectSink;

const SECOND: u64 = 1_000_000_000;

fn video_caps(fps: &str) -> Caps {
    CapsSet::from_gst_string(&format!(
        "video/x-raw,format=RGBA,width=2,height=2,framerate={fps}"
    ))
    .unwrap()
    .fixate()
    .unwrap()
}

fn frame(pts_ns: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([42; 16]))),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        0,
    )
}

fn set(element: &mut TimecodeStamper, name: &str, nick: &str) {
    element
        .set_property(name, PropValue::Str(nick.into()))
        .unwrap();
}

type Label = (u8, u8, u8, u16, bool);

fn labels(sink: &CollectSink) -> Vec<Option<Label>> {
    sink.packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(
                frame
                    .meta
                    .get::<TimecodeMeta>()
                    .map(|tc| (tc.hours, tc.minutes, tc.seconds, tc.frames, tc.drop_frame)),
            ),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn internal_count_skips_drop_frame_numbers_and_reuses_pixels() {
    for (fps, start, last, first) in [
        ("30000/1001", "00:00:59;29", (0, 0, 59, 29), (0, 1, 0, 2)),
        ("60000/1001", "00:00:59;59", (0, 0, 59, 59), (0, 1, 0, 4)),
        ("30000/1001", "00:09:59;29", (0, 9, 59, 29), (0, 10, 0, 0)),
    ] {
        let mut stamper = TimecodeStamper::new();
        set(&mut stamper, "set-internal-timecode", start);
        stamper.configure_pipeline(&video_caps(fps)).unwrap();
        let first_frame = frame(SECOND);
        let pixel_ptr = first_frame.domain.as_system_slice().unwrap().as_ptr();
        let mut sink = CollectSink::default();
        for frame in [first_frame, frame(5 * SECOND)] {
            stamper
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        assert_eq!(
            labels(&sink),
            [
                Some((last.0, last.1, last.2, last.3, true)),
                Some((first.0, first.1, first.2, first.3, true)),
            ],
            "{fps} from {start}"
        );
        assert!(matches!(&sink.packets[0], PipelinePacket::DataFrame(frame)
            if frame.domain.as_system_slice().unwrap().as_ptr() == pixel_ptr
                && frame.timing.pts_ns == SECOND));
    }
}

#[tokio::test]
async fn upstream_timecode_is_kept_then_followed_until_it_expires() {
    let mut stamper = TimecodeStamper::new();
    set(&mut stamper, "source", "last-known");
    stamper
        .set_property("timeout", PropValue::Uint(2 * SECOND))
        .unwrap();
    stamper
        .set_property("timecode-offset", PropValue::Int(1))
        .unwrap();
    stamper.configure_pipeline(&video_caps("25/1")).unwrap();
    let mut anchor = frame(0);
    anchor.meta.attach(TimecodeMeta {
        hours: 1,
        minutes: 2,
        seconds: 3,
        frames: 4,
        drop_frame: false,
        framerate_q16: Some(25 << 16),
    });
    let mut sink = CollectSink::default();
    for frame in [anchor, frame(SECOND), frame(2 * SECOND)] {
        stamper
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
    }
    assert_eq!(
        labels(&sink),
        [
            Some((1, 2, 3, 4, false)),
            Some((1, 2, 3, 6, false)),
            Some((0, 0, 0, 3, false)),
        ]
    );
}

#[tokio::test]
async fn running_time_tracks_segments_and_refined_caps() {
    let mut stamper = TimecodeStamper::new();
    set(&mut stamper, "source", "running-time");
    set(&mut stamper, "set", "always");
    stamper.configure_pipeline(&video_caps("25/1")).unwrap();
    let segment = Segment {
        start: 2 * SECOND,
        base: SECOND,
        rate: 2.0,
        ..Segment::new()
    };
    let refined_caps = video_caps("50/1");
    let mut upstream = frame(FrameTiming::PTS_NONE);
    upstream.meta.attach(TimecodeMeta {
        hours: 1,
        minutes: 0,
        seconds: 0,
        frames: 0,
        drop_frame: false,
        framerate_q16: None,
    });
    let mut sink = CollectSink::default();
    let pts = 2 * SECOND + SECOND / 2;
    for packet in [
        PipelinePacket::Segment(segment),
        PipelinePacket::DataFrame(frame(pts)),
        PipelinePacket::CapsChanged(refined_caps.clone()),
        PipelinePacket::DataFrame(frame(pts)),
        PipelinePacket::DataFrame(upstream),
        PipelinePacket::Flush,
        PipelinePacket::DataFrame(frame(0)),
    ] {
        stamper.process(packet, &mut sink).await.unwrap();
    }
    assert_eq!(
        labels(&sink),
        [
            Some((0, 0, 1, 6, false)),
            Some((0, 0, 1, 12, false)),
            None,
            Some((0, 0, 0, 0, false)),
        ]
    );
    assert!(matches!(&sink.packets[2], PipelinePacket::CapsChanged(c) if *c == refined_caps));
    assert!(matches!(sink.packets[5], PipelinePacket::Flush));
}

#[tokio::test]
async fn internal_count_resumes_at_the_seek_and_wraps_negative_offsets() {
    let mut stamper = TimecodeStamper::new();
    stamper
        .set_property("timecode-offset", PropValue::Int(-1))
        .unwrap();
    stamper.configure_pipeline(&video_caps("25/1")).unwrap();
    let seek = Segment {
        start: 2 * SECOND,
        position: 2 * SECOND,
        ..Segment::new()
    };
    let mut sink = CollectSink::default();
    for packet in [
        PipelinePacket::DataFrame(frame(0)),
        PipelinePacket::Flush,
        PipelinePacket::Segment(seek),
        PipelinePacket::DataFrame(frame(seek.position)),
    ] {
        stamper.process(packet, &mut sink).await.unwrap();
    }
    assert_eq!(
        labels(&sink),
        [Some((23, 59, 59, 24, false)), Some((0, 0, 1, 24, false))]
    );
}

#[tokio::test]
async fn launch_accepts_properties_and_runs_the_transform() {
    let graph = parse_launch(
        &default_registry(),
        "videotestsrc num-buffers=2 ! timecodestamper source=zero timecode-offset=-1 ! fakesink",
    )
    .unwrap();
    let stats = run_graph(graph, &g2g_core::clock::MonotonicClock, 4)
        .await
        .unwrap();
    assert_eq!(stats.frames_consumed, 2);
}
