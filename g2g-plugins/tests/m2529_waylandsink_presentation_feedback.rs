// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2529: `WaylandSink` counts, times and scores each frame by the
//! compositor's presentation feedback.
//!
//! Ignored by default: needs a Wayland session (`WAYLAND_DISPLAY`).
//!
//! ```sh
//! cargo test -p g2g-plugins --features std,wayland-sink \
//!     --test m2529_waylandsink_presentation_feedback -- --ignored
//! ```

#![cfg(all(target_os = "linux", feature = "wayland-sink"))]

use g2g_core::metrics::monotonic_ns;
use g2g_core::{PipelineClock, RawVideoFormat};
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videotestsrc::VideoTestSrc;
use g2g_plugins::waylandsink::WaylandSink;

struct MonotonicClock;
impl PipelineClock for MonotonicClock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

#[tokio::test]
#[ignore = "needs a Wayland session (WAYLAND_DISPLAY)"]
async fn every_presented_frame_is_timed_at_its_presentation() {
    const FRAMES: u64 = 60;
    const FRAME_RATE: u32 = 30;
    let mut src = VideoTestSrc::new(640, 480, FRAME_RATE, FRAMES);
    let mut convert = VideoConvert::new(RawVideoFormat::Nv12);
    let mut sink = WaylandSink::new().with_title("g2g m2529 presentation feedback");

    tokio::time::timeout(
        std::time::Duration::from_secs(30),
        g2g_core::runtime::run_source_transform_sink(
            &mut src,
            &mut convert,
            &mut sink,
            &MonotonicClock,
            2,
        ),
    )
    .await
    .expect("presents within 30s")
    .expect("pipeline runs");

    let presented = sink.frames_presented();
    assert_eq!(
        presented + sink.late_dropped() + sink.frames_dropped(),
        FRAMES,
        "every committed frame gets presented or discarded feedback"
    );
    assert!(presented > 0, "the compositor presented no frame");
    assert_eq!(sink.latency_samples().len() as u64, presented);

    let errors = sink.deadline_error_samples();
    assert_eq!(errors.len() as u64, presented);
    // without timed commits the sink commits early by the median delay to the screen
    let frame_period_ns = 1_000_000_000 / i64::from(FRAME_RATE);
    assert!(
        errors
            .iter()
            .all(|&error| (-frame_period_ns..10 * frame_period_ns).contains(&error)),
        "presentation minus deadline, nanoseconds: {errors:?}"
    );
}
