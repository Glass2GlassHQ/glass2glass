// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "windows", feature = "d3d11-sink"))]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{ClockPriority, PipelineClock};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

const FRAMES: u64 = 10;

#[tokio::test]
async fn autovideosink_presents_system_frames_through_d3d11() {
    let reg = default_registry();
    let line = format!(
        "videotestsrc num-buffers={FRAMES} ! video/x-raw,format=NV12,width=320,height=240 ! autovideosink"
    );
    let graph = parse_launch(&reg, &line).unwrap_or_else(|e| panic!("{line:?} parse: {e:?}"));
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line:?} runs: {e:?}"));
    assert_eq!(stats.frames_consumed, FRAMES, "{line}");
    // d3d11videosink provides the pipeline clock, fakevideosink provides none
    assert_eq!(stats.clock_priority, ClockPriority::Provider, "{line}");
    // a failing present still consumes the frame, so count what reached the swapchain
    let presented: u64 = stats
        .per_element
        .iter()
        .filter_map(|e| e.presentation)
        .map(|p| p.presented)
        .sum();
    assert_eq!(presented, FRAMES, "{line}");
}
