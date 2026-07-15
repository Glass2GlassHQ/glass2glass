//! M185: caps-driven transforms. A `videoscale` with no `width`/`height`
//! properties takes its output geometry from a downstream capsfilter (the gst
//! idiom `videoscale ! video/x-raw,width=...,height=...`), via the new
//! `configure_output` hook that hands a transform its negotiated output caps.
//! The convenience properties still work and a bare videoscale is a passthrough.
//!
//! Why a green run proves the resize: if videoscale passed the source's 320x240
//! through, the downstream 160x120 capsfilter would reject it (the pre-M185
//! CapsMismatch). The pipeline only negotiates and flows because videoscale took
//! 160x120 from the solve and resized to it.
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
async fn capsfilter_drives_videoscale_geometry() {
    // No width/height on videoscale: the downstream capsfilter (160x120) pins
    // the output, and configure_output hands it to videoscale, which resizes.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoscale ! video/x-raw,format=RGBA,width=160,height=120 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn caps_driven_scale_to_format_less_geometry() {
    // Combine with M184: a format-less geometry-only capsfilter still drives the
    // scale (format stays the source's RGBA, geometry pinned to 160x120).
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoscale ! video/x-raw,width=160,height=120 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn bare_videoscale_is_passthrough() {
    // No properties, no downstream geometry constraint: videoscale defaults to
    // the input geometry (an identity scale), instead of failing to negotiate.
    assert_eq!(
        run_line("videotestsrc num-buffers=2 width=160 height=120 ! videoscale ! fakesink").await,
        2
    );
}

#[tokio::test]
async fn videoscale_properties_still_work() {
    // The convenience property route is unchanged.
    assert_eq!(
        run_line("videotestsrc num-buffers=2 ! videoscale width=160 height=120 ! fakesink").await,
        2
    );
}
