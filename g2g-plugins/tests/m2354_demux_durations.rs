// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::time::Duration;

use g2g_core::runtime::{
    attach_seek, attach_seek_controller, parse_launch, GraphNode, RunBuilder, SeekController,
};
use g2g_core::{Bus, BusMessage, Graph, Seek};
use g2g_plugins::clock::WallClock;
use g2g_plugins::discover::discover;
use g2g_plugins::registry::default_registry;

// the fixtures of m2347, 2 s of 25 fps video and 48 kHz audio, with the durations
// `ffprobe -show_entries format=duration` reports for them
const FIXTURES: [(&str, &str, u64); 7] = [
    ("av_seek_h264_aac.flv", "flvdemux", 2_021_000_000),
    ("av_seek_vp8_vorbis.ogg", "oggdemux", 2_000_000_000),
    ("av_seek_h264_aac.ts", "tsdemux", 2_021_333_000),
    ("av_seek_mpeg2_mp2.mpg", "mpegpsdemux", 1_730_022_000),
    ("av_seek_h264_mp3.avi", "avidemux", 2_040_000_000),
    ("av_seek_h264_aac.mkv", "matroskademux", 2_021_000_000),
    ("av_seek_h264_aac.mp4", "qtdemux", 2_000_000_000),
];
// ffprobe adds the duration of the last PES to its pts, which a scan of the PES headers in the tail of a TS or PS file cannot read
const FRAME_PERIOD_NS: u64 = 40_000_000;
const CHUNK_LEN: usize = 4096;
const LINK_CAPACITY: usize = 4;
const BUS_CAPACITY: usize = 4096;
const RUN_TIMEOUT: Duration = Duration::from_secs(10);

const AUDIO_AND_VIDEO_BRANCHES: &str = "name=d d.video_0 ! fakesink  d.audio_0 ! fakesink";

fn launch_graph(fixture: &str, demux: &str, branches: &str) -> Graph<GraphNode> {
    let line = format!(
        "filesrc location={}/tests/fixtures/{fixture} blocksize={CHUNK_LEN} ! {demux} {branches}",
        env!("CARGO_MANIFEST_DIR")
    );
    parse_launch(&default_registry(), &line)
        .unwrap_or_else(|error| panic!("{fixture}: the line does not build: {error}"))
}

async fn run_to_bus(
    graph: Graph<GraphNode>,
    seek_controller: Option<&SeekController>,
) -> Vec<BusMessage> {
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let clock = WallClock::new();
    let mut builder = RunBuilder::new(graph, &clock, LINK_CAPACITY).with_bus(&handle);
    if let Some(app) = seek_controller {
        builder = builder.with_seek_controller(app);
    }
    tokio::time::timeout(RUN_TIMEOUT, builder.run())
        .await
        .expect("the run ends")
        .expect("the pipeline runs");
    std::iter::from_fn(|| bus.try_recv()).collect()
}

fn durations(messages: &[BusMessage]) -> Vec<u64> {
    messages
        .iter()
        .filter_map(|message| match message {
            BusMessage::DurationChanged { duration_ns } => Some(*duration_ns),
            _ => None,
        })
        .collect()
}

async fn reported_duration_ns(fixture: &str, demux: &str, branches: &str) -> Option<u64> {
    let mut graph = launch_graph(fixture, demux, branches);
    let app = attach_seek_controller(&mut graph).expect("the demuxer takes seeks");
    durations(&run_to_bus(graph, Some(&app)).await)
        .last()
        .copied()
}

#[tokio::test]
async fn every_demuxer_reports_the_duration_ffprobe_reads() {
    for (fixture, demux, ffprobe_ns) in FIXTURES {
        let duration_ns = reported_duration_ns(fixture, demux, AUDIO_AND_VIDEO_BRANCHES)
            .await
            .unwrap_or_else(|| panic!("{fixture}: no duration"));
        assert!(
            duration_ns.abs_diff(ffprobe_ns) <= FRAME_PERIOD_NS,
            "{fixture}: {duration_ns} ns, ffprobe reads {ffprobe_ns} ns"
        );
    }
}

// ffprobe -show_entries format=start_time,duration: the first is 0.4 s and 3.4 s, the second 0.36 s and 0.64 s
const LATE_START_MKV_FIXTURES: [(&str, u64, u64); 2] = [
    (
        "h264_64x48_25fps_gop10_bframes_offset.mkv",
        400_000_000,
        3_400_000_000,
    ),
    (
        "h264_64x48_25fps_leading_bframes.mkv",
        360_000_000,
        640_000_000,
    ),
];

#[tokio::test]
async fn a_late_starting_matroska_file_reports_its_duration_from_the_first_frame() {
    for (fixture, start_ns, ffprobe_ns) in LATE_START_MKV_FIXTURES {
        let expected_ns = Some(ffprobe_ns - start_ns);
        assert_eq!(
            reported_duration_ns(fixture, "matroskademux", "name=d d.video_0 ! fakesink").await,
            expected_ns,
            "{fixture} through matroskademux name=d"
        );
        let uri = format!(
            "file://{}/tests/fixtures/{fixture}",
            env!("CARGO_MANIFEST_DIR")
        );
        let discovery = discover(&default_registry(), &uri)
            .await
            .unwrap_or_else(|error| panic!("{fixture}: {error:?}"));
        assert_eq!(
            discovery.duration_ns, expected_ns,
            "{fixture} through discover"
        );
    }
}

fn posts_streams_or_tags(messages: &[BusMessage]) -> bool {
    messages.iter().any(|message| {
        matches!(
            message,
            BusMessage::StreamCollection(_) | BusMessage::Tag { .. }
        )
    })
}

// the demuxers that post a stream list or tags for their FIXTURES file
const POSTING_DEMUXERS: [&str; 6] = [
    "flvdemux",
    "tsdemux",
    "mpegpsdemux",
    "avidemux",
    "matroskademux",
    "qtdemux",
];

#[tokio::test]
async fn a_single_stream_demuxer_in_a_launch_line_posts_to_the_bus() {
    for (fixture, demux, _) in FIXTURES {
        if POSTING_DEMUXERS.contains(&demux) {
            assert!(
                posts_streams_or_tags(
                    &run_to_bus(launch_graph(fixture, demux, "! fakesink"), None).await
                ),
                "{fixture}: {demux} ! fakesink posts nothing"
            );
        }
    }
}

// FlvDemuxN takes no bus
const UNPOSTING_NAMED_DEMUXERS: [&str; 1] = ["flvdemux"];

#[tokio::test]
async fn a_named_demuxer_posts_to_the_bus_and_each_duration_once() {
    for (fixture, demux, _) in FIXTURES {
        if !POSTING_DEMUXERS.contains(&demux) || UNPOSTING_NAMED_DEMUXERS.contains(&demux) {
            continue;
        }
        let messages =
            run_to_bus(launch_graph(fixture, demux, AUDIO_AND_VIDEO_BRANCHES), None).await;
        assert!(
            posts_streams_or_tags(&messages),
            "{fixture}: {demux} name=d posts nothing"
        );
        let durations = durations(&messages);
        assert!(
            durations.windows(2).all(|pair| pair[0] != pair[1]),
            "{fixture}: {demux} name=d posts a duration twice: {durations:?}"
        );
    }
}

// between the keyframes at 0.8 s and 1.2 s of video time
const TARGET_NS: u64 = 1_020_000_000;

async fn checksums(
    fixture: &str,
    demux: &str,
    reads_tail: bool,
    first_seek: Option<Seek>,
) -> Vec<(String, u64, String)> {
    let mut graph = launch_graph(
        fixture,
        demux,
        "name=d d.video_0 ! checksumsink name=video  d.audio_0 ! checksumsink name=audio",
    );
    let app = match first_seek {
        Some(seek) => Some(attach_seek(&mut graph, seek).expect("the demuxer seeks").0),
        None => reads_tail.then(|| attach_seek_controller(&mut graph).expect("the demuxer seeks")),
    };
    let tail_reader = app.as_ref().filter(|_| reads_tail);
    let mut lines: Vec<_> = run_to_bus(graph, tail_reader)
        .await
        .into_iter()
        .filter_map(|message| match message {
            BusMessage::Checksum {
                element,
                pts_ns,
                digest,
            } => Some((element, pts_ns, digest)),
            _ => None,
        })
        .collect();
    lines.sort();
    lines
}

#[tokio::test]
async fn reading_the_tail_first_leaves_the_frames_unchanged() {
    for (fixture, demux, _) in FIXTURES {
        let plain = checksums(fixture, demux, false, None).await;
        assert!(!plain.is_empty(), "{fixture}");
        assert_eq!(
            checksums(fixture, demux, true, None).await,
            plain,
            "{fixture}"
        );
    }
}

#[tokio::test]
async fn a_seek_queued_before_the_tail_read_lands_where_it_does_without_one() {
    for (fixture, demux, _) in FIXTURES {
        let seek = Some(Seek::flush_to(TARGET_NS));
        let plain = checksums(fixture, demux, false, seek).await;
        assert!(!plain.is_empty(), "{fixture}");
        assert_eq!(
            checksums(fixture, demux, true, seek).await,
            plain,
            "{fixture}"
        );
    }
}
