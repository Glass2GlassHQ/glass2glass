//! M183: videocrop / videobox property-model alignment with GStreamer. Both now
//! use per-edge `top`/`bottom`/`left`/`right` properties. videocrop crops off
//! each edge; videobox is signed (>0 crops, <0 adds a border of `fill`). The old
//! g2g `x`/`y`/`width`/`height` (videocrop) and `border-*` (videobox) property
//! names are replaced, not aliased (pre-release, gst names are canonical).
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
async fn gst_videocrop_edge_insets_port() {
    // gst videocrop: crop pixels off each edge. 320x240 - (8+8) x (4+4).
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videocrop left=8 right=8 top=4 bottom=4 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn videocrop_default_is_passthrough() {
    // All-zero insets (gst default) is an identity crop.
    assert_eq!(
        run_line("videotestsrc num-buffers=2 ! videocrop ! fakesink").await,
        2
    );
}

#[tokio::test]
async fn gst_videobox_negative_edges_add_border() {
    // gst videobox: negative edge values add a border of `fill`.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videobox left=-8 right=-8 fill=black ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn gst_videobox_positive_edges_crop() {
    // gst videobox: positive edge values crop (the videocrop overlap).
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videobox top=10 bottom=10 fill=blue ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}
