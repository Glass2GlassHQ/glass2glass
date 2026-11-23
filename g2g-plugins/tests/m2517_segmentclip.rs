// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{
    AsyncElement, Caps, CapsSet, Frame, FrameTiming, MemoryDomain, PipelinePacket, Segment,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::segmentclip::{AudioSegmentClip, VideoSegmentClip};

use collect_sink::CollectSink;

const MS: u64 = 1_000_000;

fn frame(pts_ns: u64, duration_ns: u64, bytes: Vec<u8>) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            duration_ns,
            ..FrameTiming::default()
        },
        17,
    )
}

fn caps(text: &str) -> Caps {
    CapsSet::from_gst_string(text).unwrap().fixate().unwrap()
}

fn data_frames(sink: &CollectSink) -> impl Iterator<Item = &Frame> {
    sink.packets.iter().filter_map(|packet| match packet {
        PipelinePacket::DataFrame(frame) => Some(frame),
        _ => None,
    })
}

#[tokio::test]
async fn audio_trims_whole_channel_frames_and_uses_refined_caps() {
    let mut clip = AudioSegmentClip::new();
    clip.configure_pipeline(&caps("audio/x-raw,format=S16LE,rate=1000,channels=2"))
        .unwrap();
    let mut sink = CollectSink::default();
    let original: Vec<u8> = (0..32).collect();
    let initial = Segment {
        start: 12 * MS + MS / 2,
        stop: Some(16 * MS + MS / 2),
        ..Segment::new()
    };
    let changed = caps("audio/x-raw,format=S24LE,rate=2000,channels=1");
    let next = Segment {
        start: 31 * MS,
        stop: Some(35 * MS),
        ..Segment::new()
    };
    for packet in [
        PipelinePacket::Segment(initial),
        PipelinePacket::DataFrame(frame(0, 8 * MS, original.clone())),
        PipelinePacket::DataFrame(frame(10 * MS, 8 * MS, original.clone())),
        PipelinePacket::DataFrame(frame(19 * MS, 8 * MS, original.clone())),
        PipelinePacket::CapsChanged(changed.clone()),
        PipelinePacket::Segment(next),
        PipelinePacket::DataFrame(frame(30 * MS, 0, original[..24].to_vec())),
    ] {
        clip.process(packet, &mut sink).await.unwrap();
    }
    assert!(matches!(sink.packets.as_slice(), [
        PipelinePacket::Segment(a), PipelinePacket::DataFrame(_),
        PipelinePacket::CapsChanged(c), PipelinePacket::Segment(b), PipelinePacket::DataFrame(_),
    ] if *a == initial && *c == changed && *b == next));
    let frames: Vec<_> = data_frames(&sink).collect();
    assert_eq!(
        frames[0].domain.as_system_slice().unwrap(),
        &original[8..28]
    );
    assert_eq!(
        (frames[0].timing.pts_ns, frames[0].timing.duration_ns),
        (initial.start, 4 * MS)
    );
    assert_eq!(
        frames[1].domain.as_system_slice().unwrap(),
        &original[6..24]
    );
    assert_eq!(
        (frames[1].timing.pts_ns, frames[1].timing.duration_ns),
        (next.start, 3 * MS)
    );
}

#[tokio::test]
async fn video_clips_only_timing_and_flush_releases_the_old_segment() {
    let mut clip = VideoSegmentClip::new();
    clip.configure_pipeline(&caps(
        "video/x-raw,format=RGBA,width=2,height=2,framerate=25/1",
    ))
    .unwrap();
    let mut sink = CollectSink::default();
    let segment = Segment {
        start: 50 * MS,
        stop: Some(110 * MS),
        ..Segment::new()
    };
    let overlapping = frame(30 * MS, 0, vec![3; 16]);
    let pixel_ptr = overlapping.domain.as_system_slice().unwrap().as_ptr();
    for packet in [
        PipelinePacket::Segment(segment),
        PipelinePacket::DataFrame(frame(10 * MS, 0, vec![2; 16])),
        PipelinePacket::DataFrame(overlapping),
        PipelinePacket::DataFrame(frame(72 * MS, 40 * MS, vec![4; 16])),
        PipelinePacket::DataFrame(frame(110 * MS, 0, vec![5; 16])),
        PipelinePacket::DataFrame(frame(FrameTiming::PTS_NONE, 0, vec![6; 16])),
        PipelinePacket::Flush,
        PipelinePacket::DataFrame(frame(0, 40 * MS, vec![7; 16])),
    ] {
        clip.process(packet, &mut sink).await.unwrap();
    }
    let frames: Vec<_> = data_frames(&sink).collect();
    assert_eq!(frames.len(), 4);
    assert_eq!(
        frames[0].domain.as_system_slice().unwrap().as_ptr(),
        pixel_ptr
    );
    assert_eq!(
        frames
            .iter()
            .map(|f| (f.timing.pts_ns, f.timing.duration_ns))
            .collect::<Vec<_>>(),
        [
            (50 * MS, 20 * MS),
            (72 * MS, 38 * MS),
            (FrameTiming::PTS_NONE, 0),
            (0, 40 * MS),
        ]
    );
    assert!(matches!(sink.packets[4], PipelinePacket::Flush));
}

#[tokio::test]
async fn launch_runs_both_media_specific_clip_elements() {
    let registry = default_registry();
    for line in [
        "audiotestsrc num-buffers=2 ! audiosegmentclip ! fakesink",
        "videotestsrc num-buffers=2 ! videosegmentclip ! fakesink",
    ] {
        let graph = parse_launch(&registry, line).unwrap();
        let stats = run_graph(graph, &g2g_core::clock::MonotonicClock, 4)
            .await
            .unwrap();
        assert_eq!(stats.frames_consumed, 2, "{line}");
    }
    assert!(AudioSegmentClip::new()
        .intercept_caps(&caps(
            "video/x-raw,format=RGBA,width=2,height=2,framerate=25/1"
        ))
        .is_err());
    assert!(VideoSegmentClip::new()
        .intercept_caps(&caps("audio/x-raw,format=S16LE,rate=1000,channels=2"))
        .is_err());
}
