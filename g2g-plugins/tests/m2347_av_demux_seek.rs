// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use g2g_core::runtime::{attach_seek, parse_launch, run_graph_with_bus, GraphNode, SeekOutcome};
use g2g_core::{Bus, BusMessage, Graph, Seek};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// every fixture: ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=2
//   -f lavfi -i sine=frequency=440:sample_rate=48000:duration=2 -ac 1, video with -g 10 (-keyint_min 10
//   -sc_threshold 0 -bf 0 for libx264, -bf 0 for mpeg2video, -keyint_min 10 -auto-alt-ref 0 for libvpx)
const FIXTURES: [(&str, &str); 7] = [
    ("av_seek_h264_aac.ts", "tsdemux"),
    ("av_seek_h264_aac.mkv", "matroskademux"),
    ("av_seek_h264_aac.flv", "flvdemux"),
    ("av_seek_vp8_vorbis.ogg", "oggdemux"),
    ("av_seek_mpeg2_mp2.mpg", "mpegpsdemux"),
    ("av_seek_h264_mp3.avi", "avidemux"),
    ("av_seek_h264_aac.mp4", "qtdemux"),
];
const FRAMES_PER_KEYFRAME: usize = 10;
// between the keyframes at 0.8 s and 1.2 s of video time
const TARGET_NS: u64 = 1_020_000_000;
const LINK_CAPACITY: usize = 4;
const BUS_CAPACITY: usize = 64;

#[derive(Debug, Default)]
struct PortLines {
    video: Vec<(u64, String)>,
    audio: Vec<(u64, String)>,
}

fn fixture_graph(fixture: &str, demux: &str) -> Graph<GraphNode> {
    let line = format!(
        "filesrc location={}/tests/fixtures/{fixture} ! {demux} name=d \
         d.video_0 ! checksumsink name=video  d.audio_0 ! checksumsink name=audio",
        env!("CARGO_MANIFEST_DIR")
    );
    parse_launch(&default_registry(), &line)
        .unwrap_or_else(|error| panic!("{fixture}: the line does not build: {error}"))
}

async fn run_to_end(graph: Graph<GraphNode>) -> PortLines {
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let clock = WallClock::new();
    let run = run_graph_with_bus(graph, &clock, LINK_CAPACITY, &handle);
    tokio::pin!(run);
    let mut lines = PortLines::default();
    let mut collect = |message: BusMessage| {
        let BusMessage::Checksum {
            element,
            pts_ns,
            digest,
        } = message
        else {
            return;
        };
        match element.as_str() {
            "video" => lines.video.push((pts_ns, digest)),
            "audio" => lines.audio.push((pts_ns, digest)),
            _ => {}
        }
    };
    loop {
        tokio::select! {
            result = &mut run => {
                result.expect("the graph runs");
                while let Some(message) = bus.try_recv() {
                    collect(message);
                }
                return lines;
            }
            Some(message) = bus.recv() => collect(message),
        }
    }
}

// after a flush a port sends its decoder config again, as units of its own or in front of the first unit, which matches by pts alone
fn assert_suffix<'a>(
    fixture: &str,
    port: &str,
    seeked: &'a [(u64, String)],
    whole: &[(u64, String)],
) -> &'a [(u64, String)] {
    let config_units = seeked
        .iter()
        .zip(whole)
        .take_while(|(seeked, whole)| seeked == whole)
        .count();
    let seeked = &seeked[config_units..];
    let first_pts_matches = whole
        .len()
        .checked_sub(seeked.len())
        .is_some_and(|start| !seeked.is_empty() && whole[start].0 == seeked[0].0);
    assert!(
        first_pts_matches && whole.ends_with(&seeked[1..]),
        "{fixture} {port}: the seeked run is not the end of the whole run"
    );
    seeked
}

#[tokio::test]
async fn a_seek_restarts_every_port_at_the_target() {
    for (fixture, demux) in FIXTURES {
        let whole = run_to_end(fixture_graph(fixture, demux)).await;
        let stream_start_ns = whole.video[0].0.min(whole.audio[0].0);
        let target_pts_ns = stream_start_ns + TARGET_NS;

        let mut graph = fixture_graph(fixture, demux);
        let (_app, ticket) = attach_seek(&mut graph, Seek::flush_to(TARGET_NS))
            .unwrap_or_else(|error| panic!("{fixture}: {error}"));
        let seeked = run_to_end(graph).await;
        assert_eq!(ticket.outcome(), Some(SeekOutcome::Served), "{fixture}");

        let video = assert_suffix(fixture, "video", &seeked.video, &whole.video);
        let skipped_frames = whole.video.len() - video.len();
        assert!(
            skipped_frames > 0 && skipped_frames.is_multiple_of(FRAMES_PER_KEYFRAME),
            "{fixture}: video restarts at a keyframe after skipping {skipped_frames} frames"
        );
        assert!(
            video[0].0 <= target_pts_ns && video[FRAMES_PER_KEYFRAME].0 > target_pts_ns,
            "{fixture}: video restarts at the last keyframe at or before the target"
        );

        let audio = assert_suffix(fixture, "audio", &seeked.audio, &whole.audio);
        assert!(
            audio[0].0 <= target_pts_ns && audio[1].0 > target_pts_ns,
            "{fixture}: audio restarts at its last unit at or before the target"
        );
    }
}
