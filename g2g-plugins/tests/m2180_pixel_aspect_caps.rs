// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2180: video caps carry gst's `pixel-aspect-ratio`. `videotestsrc` draws
//! square pixels, so a launch caps filter asking for another ratio refuses to
//! negotiate, and one asking for 1/1 runs.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn negotiates(pixel_aspect_ratio: &str) -> bool {
    let line = format!(
        "videotestsrc num-buffers=1 ! video/x-raw,pixel-aspect-ratio={pixel_aspect_ratio} ! fakesink"
    );
    let graph = parse_launch(&default_registry(), &line).expect("the line parses");
    run_graph(graph, &ZeroClock, 2).await.is_ok()
}

#[tokio::test]
async fn a_square_source_meets_a_square_filter_and_refuses_another_ratio() {
    assert!(negotiates("1/1").await);
    assert!(!negotiates("10/11").await);
}
