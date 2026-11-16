// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::time::Duration;

use g2g_core::runtime::{attach_seek, attach_seek_controller, parse_launch, RunBuilder};
use g2g_core::{Bus, BusMessage, Seek};
use g2g_plugins::clock::WallClock;
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

async fn reported_duration_ns(fixture: &str, demux: &str) -> Option<u64> {
    let line = format!(
        "filesrc location={}/tests/fixtures/{fixture} blocksize={CHUNK_LEN} ! {demux} name=d \
         d.video_0 ! fakesink  d.audio_0 ! fakesink",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut graph = parse_launch(&default_registry(), &line)
        .unwrap_or_else(|error| panic!("{fixture}: the line does not build: {error}"));
    let app = attach_seek_controller(&mut graph).expect("the demuxer takes seeks");
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let clock = WallClock::new();
    let run = RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .with_bus(&handle)
        .with_seek_controller(&app)
        .run();
    tokio::time::timeout(RUN_TIMEOUT, run)
        .await
        .expect("the run ends")
        .expect("the pipeline runs");
    let mut duration_ns = None;
    while let Some(message) = bus.try_recv() {
        if let BusMessage::DurationChanged { duration_ns: ns } = message {
            duration_ns = Some(ns);
        }
    }
    duration_ns
}

#[tokio::test]
async fn every_demuxer_reports_the_duration_ffprobe_reads() {
    for (fixture, demux, ffprobe_ns) in FIXTURES {
        let duration_ns = reported_duration_ns(fixture, demux)
            .await
            .unwrap_or_else(|| panic!("{fixture}: no duration"));
        assert!(
            duration_ns.abs_diff(ffprobe_ns) <= FRAME_PERIOD_NS,
            "{fixture}: {duration_ns} ns, ffprobe reads {ffprobe_ns} ns"
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
    let line = format!(
        "filesrc location={}/tests/fixtures/{fixture} blocksize={CHUNK_LEN} ! {demux} name=d \
         d.video_0 ! checksumsink name=video  d.audio_0 ! checksumsink name=audio",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut graph = parse_launch(&default_registry(), &line).expect("the line builds");
    let app = match first_seek {
        Some(seek) => Some(attach_seek(&mut graph, seek).expect("the demuxer seeks").0),
        None => reads_tail.then(|| attach_seek_controller(&mut graph).expect("the demuxer seeks")),
    };
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let clock = WallClock::new();
    let mut builder = RunBuilder::new(graph, &clock, LINK_CAPACITY).with_bus(&handle);
    if let Some(app) = app.as_ref().filter(|_| reads_tail) {
        builder = builder.with_seek_controller(app);
    }
    tokio::time::timeout(RUN_TIMEOUT, builder.run())
        .await
        .expect("the run ends")
        .expect("the pipeline runs");
    let mut lines = Vec::new();
    while let Some(message) = bus.try_recv() {
        if let BusMessage::Checksum {
            element,
            pts_ns,
            digest,
        } = message
        {
            lines.push((element, pts_ns, digest));
        }
    }
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
