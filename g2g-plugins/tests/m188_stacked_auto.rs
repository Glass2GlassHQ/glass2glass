//! M188: forward-resolve walk. Two caps-driven (auto) transforms stacked before
//! a single far capsfilter now resolve: the downstream pin propagates *back*
//! through the transforms. The solver does this by filtering each auto
//! transform's ambiguous input to the alternatives whose forward image can still
//! satisfy the constrained output (DerivedOutput isn't invertible, but it is
//! evaluable per candidate). This is the case M186 explicitly couldn't do.
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
async fn convert_then_scale_before_one_caps() {
    // The M186 hard case: videoconvert(auto) ! videoscale(auto) ! caps. The NV12
    // + 160x120 pin two hops downstream back-propagates: videoscale's format must
    // be NV12 (it passes format through), so videoconvert must produce NV12.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoconvert ! videoscale ! video/x-raw,format=NV12,width=160,height=120 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn scale_then_convert_before_one_caps() {
    // M227 (field-level coupling): the other order, videoscale(auto) !
    // videoconvert(auto) ! caps, now resolves too. The geometry pin (160x120)
    // sits behind a geometry-passthrough transform (videoconvert); the
    // `DerivedCoupled` backward sweep intersects the pin *into* videoconvert's
    // input width/height fields (`Range ∩ Fixed = Fixed`) rather than only
    // dropping alternatives, so videoscale receives a pinned 160x120 output and
    // resizes. Was the documented M188 KNOWN-LIMIT; closed by M227.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoscale ! videoconvert ! video/x-raw,format=NV12,width=160,height=120 ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn stacked_auto_still_passthrough_without_a_pin() {
    // No downstream constraint: both auto transforms pass through (identity),
    // rather than failing to fixate.
    let line = "videotestsrc num-buffers=2 width=320 height=240 \
                ! videoconvert ! videoscale ! fakesink";
    assert_eq!(run_line(line).await, 2, "{line}");
}

#[tokio::test]
async fn incompatible_stacked_pin_fails_loud() {
    // A pin no producible format can reach still fails negotiation, not silently
    // mis-fixate. (videotestsrc only emits RGBA; an audio pin can't be satisfied.)
    let reg = default_registry();
    let parsed = parse_launch(
        &reg,
        "videotestsrc num-buffers=2 ! videoconvert ! videoscale ! audio/x-raw,rate=48000 ! fakesink",
    );
    // Either the parse rejects the video->audio capsfilter link, or the solve
    // fails; both are acceptable "loud" outcomes.
    if let Ok(graph) = parsed {
        assert!(
            run_graph(graph, &ZeroClock, 4).await.is_err(),
            "video->audio pin must fail"
        );
    }
}
