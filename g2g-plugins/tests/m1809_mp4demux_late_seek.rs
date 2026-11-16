// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::time::Duration;

use g2g_core::runtime::{attach_seek_controller, parse_launch, RunBuilder, SeekController};
use g2g_core::{Bus, BusMessage, Seek, SeekFlags, SeekType};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// ffmpeg -f lavfi -i testsrc=size=64x48:rate=25:duration=3 -c:v libx264 -g 10 -keyint_min 10 -sc_threshold 0 -bf 0
const MP4_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_25fps_gop10.mp4"
);
const FRAME_COUNT: usize = 75;
const SEEK_NS: u64 = 1_000_000_000;
// -g 10 at 25 fps puts a keyframe every 400 ms, the one at or before the target is at 800 ms
const KEYFRAME_BEFORE_SEEK_NS: u64 = 800_000_000;
const FRAME_PERIOD_NS: u64 = 40_000_000;

const LINK_CAPACITY: usize = 4;
// every frame of the run and of the replay, so none is dropped before the test reads it
const BUS_CAPACITY: usize = 4 * FRAME_COUNT;
const SEGMENT_DONE_POLL: Duration = Duration::from_millis(5);
const RUN_TIMEOUT: Duration = Duration::from_secs(10);

struct LateSeek {
    app: SeekController,
    // the pts of every frame after the seek
    replayed_pts_ns: Vec<u64>,
}

// queues `seek` once the sink has every frame of the file, as filesrc parks at the end in a run that takes seeks there
async fn run_late_seek(seek: Seek) -> LateSeek {
    let line = format!("filesrc location={MP4_FIXTURE} ! qtdemux ! checksumsink");
    let mut graph = parse_launch(&default_registry(), &line).expect("the line negotiates");
    let app = attach_seek_controller(&mut graph).expect("qtdemux takes seeks");
    app.keep_run_open();
    let segment_seek = seek.flags.contains(SeekFlags::SEGMENT);
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let clock = WallClock::new();
    let run = RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .with_bus(&handle)
        .with_seek_controller(&app)
        .run();
    let mut pts_ns = Vec::new();
    let mut record = |message: BusMessage| {
        if let BusMessage::Checksum { pts_ns: pts, .. } = message {
            pts_ns.push(pts);
            if pts_ns.len() == FRAME_COUNT {
                app.seek(seek);
                // the run ends once the sink has the seek's frames, a segment seek waits for the next one instead
                app.let_run_end();
            }
        }
    };
    tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::pin!(run);
        loop {
            tokio::select! {
                result = &mut run => {
                    result.expect("the pipeline runs");
                    while let Some(message) = bus.try_recv() {
                        record(message);
                    }
                    return;
                }
                Some(message) = bus.recv() => record(message),
                _ = tokio::time::sleep(SEGMENT_DONE_POLL), if segment_seek => {
                    if app.segment_done_count() > 0 {
                        app.shutdown();
                    }
                }
            }
        }
    })
    .await
    .expect("the run ends");
    assert!(
        pts_ns.len() >= FRAME_COUNT,
        "every sample went out before the seek"
    );
    LateSeek {
        replayed_pts_ns: pts_ns.split_off(FRAME_COUNT),
        app,
    }
}

#[tokio::test]
async fn a_seek_after_the_drain_replays_from_the_keyframe_before_the_target() {
    let sink = run_late_seek(Seek::flush_to(SEEK_NS)).await;

    let replayed: Vec<u64> = (KEYFRAME_BEFORE_SEEK_NS..FRAME_COUNT as u64 * FRAME_PERIOD_NS)
        .step_by(FRAME_PERIOD_NS as usize)
        .collect();
    assert_eq!(
        sink.replayed_pts_ns, replayed,
        "after a Flush the samples play again from the keyframe at or before the target"
    );
}

#[tokio::test]
async fn a_seek_stop_ends_the_replay_at_the_last_frame_by_the_stop() {
    let sink = run_late_seek(Seek {
        stop_type: SeekType::Set,
        stop: STOP_NS,
        ..Seek::flush_to(SEEK_NS)
    })
    .await;
    assert_eq!(sink.replayed_pts_ns, replayed_to_the_stop());
}

const STOP_NS: u64 = 1_500_000_000;
const DURATION_NS: u64 = FRAME_COUNT as u64 * FRAME_PERIOD_NS;

fn replayed_to_the_stop() -> Vec<u64> {
    (KEYFRAME_BEFORE_SEEK_NS..=STOP_NS)
        .step_by(FRAME_PERIOD_NS as usize)
        .collect()
}

#[tokio::test]
async fn a_stop_counted_from_the_end_lands_on_the_duration() {
    let sink = run_late_seek(Seek {
        stop_type: SeekType::End,
        stop: DURATION_NS - STOP_NS,
        ..Seek::flush_to(SEEK_NS)
    })
    .await;
    assert_eq!(sink.replayed_pts_ns, replayed_to_the_stop());
}

#[tokio::test]
async fn a_segment_seek_posts_segment_done_at_its_stop() {
    let sink = run_late_seek(Seek {
        flags: SeekFlags::FLUSH | SeekFlags::SEGMENT,
        stop_type: SeekType::Set,
        stop: STOP_NS,
        ..Seek::flush_to(SEEK_NS)
    })
    .await;
    assert_eq!(sink.replayed_pts_ns, replayed_to_the_stop());
    assert_eq!(sink.app.take_segment_done(), Some(STOP_NS));
}
