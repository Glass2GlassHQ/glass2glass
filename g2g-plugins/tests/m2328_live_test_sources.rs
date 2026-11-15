// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::time::{Duration, Instant};

use g2g_core::runtime::{parse_launch, run_graph, RunStats};
use g2g_core::PipelineClock;

const LINK_CAPACITY: usize = 4;
const BUFFERS: u64 = 4;
const VIDEO_FPS: u64 = 30;
const VIDEO_PERIOD: Duration = Duration::from_nanos(1_000_000_000 / VIDEO_FPS);
// audiotestsrc emits 10 ms buffers
const AUDIO_PERIOD: Duration = Duration::from_millis(10);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn timed_run(line: &str) -> (RunStats, Duration) {
    let registry = g2g_plugins::registry::default_registry();
    let graph = parse_launch(&registry, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let started = Instant::now();
    let stats = run_graph(graph, &ZeroClock, LINK_CAPACITY)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    (stats, started.elapsed())
}

#[tokio::test]
async fn live_videotestsrc_produces_on_the_clock() {
    let line = format!(
        "videotestsrc is-live=true num-buffers={BUFFERS} framerate={VIDEO_FPS}/1 ! fakesink"
    );
    let (stats, elapsed) = timed_run(&line).await;
    assert!(stats.latency.live, "is-live makes the path live");
    assert_eq!(
        stats.latency.min_ns,
        VIDEO_PERIOD.as_nanos() as u64,
        "one frame of capture latency"
    );
    let last_buffer_due = VIDEO_PERIOD * (BUFFERS as u32 - 1);
    assert!(
        elapsed >= last_buffer_due,
        "{BUFFERS} frames into an unsynced sink took {elapsed:?}, under {last_buffer_due:?}"
    );
}

#[tokio::test]
async fn live_audiotestsrc_produces_on_the_clock() {
    let line = format!("audiotestsrc is-live=true num-buffers={BUFFERS} ! fakesink");
    let (stats, elapsed) = timed_run(&line).await;
    assert!(stats.latency.live, "is-live makes the path live");
    assert_eq!(
        stats.latency.min_ns,
        AUDIO_PERIOD.as_nanos() as u64,
        "one buffer of capture latency"
    );
    let last_buffer_due = AUDIO_PERIOD * (BUFFERS as u32 - 1);
    assert!(
        elapsed >= last_buffer_due,
        "{BUFFERS} buffers into an unsynced sink took {elapsed:?}, under {last_buffer_due:?}"
    );
}
