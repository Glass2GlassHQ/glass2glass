//! M186: caps-driven videoconvert. A `videoconvert` with no `format` property
//! takes its output format from a downstream capsfilter (gst's `videoconvert !
//! video/x-raw,format=NV12`), via the M185 `configure_output` hook. The property
//! still wins when set, and a bare videoconvert is a passthrough.
//!
//! Why a green run proves the conversion: the downstream capsfilter pins NV12,
//! so the only way the pipeline negotiates is videoconvert producing NV12 (an
//! RGBA source can't satisfy an NV12 filter directly).
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
async fn capsfilter_drives_videoconvert_format() {
    // No format on videoconvert: the downstream capsfilter (NV12) pins the
    // output, and configure_output hands it over, so videoconvert converts
    // RGBA -> NV12.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoconvert ! video/x-raw,format=NV12 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn caps_driven_convert_and_scale_each_pinned() {
    // Two caps-driven transforms, each pinned by its own immediately-downstream
    // capsfilter (the idiomatic gst form): convert to NV12, then scale to
    // 160x120. NOTE: stacking two auto transforms before a single far caps
    // (`videoconvert ! videoscale ! caps`) does NOT propagate the format back
    // through the passthrough-format scaler, that needs the forward-resolve walk
    // tracked in DESIGN_TODO. A capsfilter after each transform is the supported
    // (and conventional) form.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoconvert ! video/x-raw,format=NV12 \
                ! videoscale ! video/x-raw,width=160,height=120 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn bare_videoconvert_is_passthrough() {
    // No property, no downstream format constraint: passthrough (no conversion),
    // instead of forcing the old hardcoded RGBA default.
    assert_eq!(
        run_line("videotestsrc num-buffers=2 ! videoconvert ! fakesink").await,
        2
    );
}

#[tokio::test]
async fn videoconvert_format_property_still_works() {
    assert_eq!(
        run_line("videotestsrc num-buffers=2 ! videoconvert format=NV12 ! fakesink").await,
        2
    );
}
