//! M227: field-level bidirectional caps coupling. A transform declared
//! `DerivedCoupled` passes some caps fields through unchanged; the solver
//! couples those fields in both directions, so a downstream pin on a
//! passthrough field narrows the corresponding input field *within* an
//! alternative (`Range ∩ Fixed = Fixed`), not just drops whole alternatives.
//! This unblocks a geometry / rate pin sitting behind a passthrough transform,
//! which the M188 alternative-dropping walk could not express.
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

async fn run_line_err(line: &str) {
    let reg = default_registry();
    // Either the parse rejects the line or the solve fails; both are acceptable
    // "loud" outcomes (never a silent mis-fixate).
    if let Ok(graph) = parse_launch(&reg, line) {
        assert!(
            run_graph(graph, &ZeroClock, 4).await.is_err(),
            "{line:?} must fail loud"
        );
    }
}

#[tokio::test]
async fn geometry_pin_couples_through_three_transforms() {
    // The pin (160x120, NV12) sits behind TWO geometry-passthrough hops
    // (videoconvert, then a second videoconvert) plus the scaler. Coupling must
    // walk the geometry back through both converters to the scaler.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoscale ! videoconvert ! videoconvert \
                ! video/x-raw,format=NV12,width=160,height=120 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn geometry_only_pin_behind_convert_couples() {
    // A format-less geometry-only pin behind a format-retargeting converter:
    // the width/height couple back to the scaler, the format stays free.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoscale ! videoconvert ! video/x-raw,width=160,height=120 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn audio_rate_pin_couples_through_passthrough() {
    // audioresample retargets sample_rate but passes format+channels through;
    // a downstream rate pin behind it still resolves (no audio passthrough
    // transform sits between here, but this pins the audioresample directly
    // through a format-passthrough hop is not available, so this is the
    // single-hop coupling baseline staying correct).
    let line = "audiotestsrc num-buffers=3 freq=440 \
                ! audioresample ! audio/x-raw,format=S16LE,rate=16000 ! fakesink";
    assert_eq!(run_line(line).await, 3, "{line}");
}

#[tokio::test]
async fn odd_yuv420_geometry_pin_fails_loud() {
    // 4:2:0 (NV12) requires even dims; an odd target behind the converter must
    // fail negotiation, not fixate impossible caps.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoscale ! videoconvert ! video/x-raw,format=NV12,width=161,height=120 ! fakesink";
    run_line_err(line).await;
}

#[tokio::test]
async fn unproducible_format_pin_fails_loud() {
    // Yuyv is input-only for videoconvert (it unpacks, never produces it). A
    // Yuyv output pin behind the scaler must fail loud.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoscale ! videoconvert ! video/x-raw,format=YUY2,width=160,height=120 ! fakesink";
    run_line_err(line).await;
}

#[tokio::test]
async fn geometry_pin_without_scaler_fails_loud() {
    // No scaler upstream: videoconvert passes geometry through, so a 160x120 pin
    // can't be met by a fixed 320x240 source. Loud, never silent.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoconvert ! video/x-raw,format=NV12,width=160,height=120 ! fakesink";
    run_line_err(line).await;
}
