//! M184: format-less / partial geometry caps. A `video/x-raw` capsfilter with no
//! `format` field (the gst-idiomatic geometry-only caps) parses and negotiates,
//! expanding to all raw formats at the pinned geometry and intersecting down to
//! whatever the upstream produces.
//!
//! `default_registry` is `std`-gated, so this file is too.
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

async fn run_line(line: &str) -> u64 {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("{line:?} should parse: {e:?}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line:?} should run: {e:?}"))
        .frames_consumed
}

#[tokio::test]
async fn format_less_caps_pins_geometry_only() {
    // No format on the capsfilter: it expands to every raw format at 320x240 and
    // the solver intersects with videotestsrc's RGBA to pin RGBA 320x240.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! video/x-raw,width=320,height=240 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn format_less_framerate_only_caps() {
    // Even just a framerate constraint (no format, no geometry) parses now.
    let line = "videotestsrc num-buffers=2 ! video/x-raw,framerate=30/1 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}
