// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2472: every frame `WaylandSink` commits gets the compositor's frame
//! callback before the next one, so none is counted dropped. Weston sends no
//! callback to a subsurface that no parent commit has placed on an output.
//!
//! Ignored by default: needs a Wayland session (`WAYLAND_DISPLAY`). Run it
//! under Weston, where the missing callbacks showed up:
//!
//! ```sh
//! cargo test -p g2g-plugins --features std,wayland-sink \
//!     --test m2472_waylandsink_frame_callbacks -- --ignored
//! ```

#![cfg(all(target_os = "linux", feature = "wayland-sink"))]

use g2g_core::{PipelineClock, RawVideoFormat};
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videotestsrc::VideoTestSrc;
use g2g_plugins::waylandsink::WaylandSink;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
#[ignore = "needs a Wayland session (WAYLAND_DISPLAY)"]
async fn every_committed_frame_gets_its_frame_callback() {
    const FRAMES: u64 = 60;
    let mut src = VideoTestSrc::new(640, 480, 30, FRAMES);
    let mut convert = VideoConvert::new(RawVideoFormat::Nv12);
    let mut sink = WaylandSink::new().with_title("g2g m2472 frame callbacks");

    tokio::time::timeout(
        std::time::Duration::from_secs(30),
        g2g_core::runtime::run_source_transform_sink(
            &mut src,
            &mut convert,
            &mut sink,
            &ZeroClock,
            2,
        ),
    )
    .await
    .expect("presents within 30s")
    .expect("pipeline runs");

    // slow hosts late-drop some frames, which is QoS and not what this checks
    assert_eq!(
        (
            sink.frames_presented() + sink.late_dropped(),
            sink.frames_dropped()
        ),
        (FRAMES, 0),
        "(presented or late-dropped, replaced before their frame callback)"
    );
}
