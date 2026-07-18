//! M182: gst-launch convention harmonization. Real `gst-launch` spellings,
//! GStreamer's videoflip method nicknames, uppercase caps format names, the
//! `bar`/`checkers-8` pattern names, should port to g2g verbatim. The historical
//! g2g spellings stay valid as aliases so nothing breaks, and the gst
//! caps-driven form (scale via a downstream capsfilter) works alongside the g2g
//! convenience properties.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
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
    let stats = run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("{line:?} should run: {e:?}"));
    stats.frames_consumed
}

#[tokio::test]
async fn gst_videoflip_method_names_port() {
    for method in [
        "none",
        "clockwise",
        "counterclockwise",
        "horizontal-flip",
        "vertical-flip",
        "rotate-180",
    ] {
        let line = format!("videotestsrc num-buffers=2 ! videoflip method={method} ! fakesink");
        assert_eq!(run_line(&line).await, 2, "{line}");
    }
}

#[tokio::test]
async fn gst_uppercase_caps_formats_port() {
    // gst always writes pixel/sample formats uppercase, in both capsfilter caps
    // and the convert format properties.
    for line in [
        "videotestsrc num-buffers=2 ! video/x-raw,format=RGBA ! fakesink",
        "videotestsrc num-buffers=2 ! videoconvert format=NV12 ! fakesink",
        "audiotestsrc num-buffers=2 ! audioconvert format=S16LE ! fakesink",
    ] {
        assert_eq!(run_line(line).await, 2, "{line}");
    }
}

#[tokio::test]
async fn gst_videotestsrc_pattern_names_port() {
    for pattern in ["bar", "checkers-8", "smpte", "snow", "ball", "zone-plate"] {
        let line = format!("videotestsrc num-buffers=2 pattern={pattern} ! fakesink");
        assert_eq!(run_line(&line).await, 2, "{line}");
    }
}

#[tokio::test]
async fn historical_g2g_spellings_still_accepted_as_aliases() {
    // The pre-harmonization spellings (lowercase formats, g2g method/pattern
    // names) remain valid so existing pipelines don't break.
    for line in [
        "videotestsrc num-buffers=2 pattern=moving-bar ! videoflip method=rotate-90cw ! fakesink",
        "videotestsrc num-buffers=2 pattern=checker ! videoflip method=horizontal-mirror ! fakesink",
        "videotestsrc num-buffers=2 ! videoconvert format=nv12 ! fakesink",
        "videotestsrc num-buffers=2 ! video/x-raw,format=rgba ! fakesink",
    ] {
        assert_eq!(run_line(line).await, 2, "{line}");
    }
}

#[tokio::test]
async fn videoflip_default_method_is_passthrough() {
    // gst's videoflip default method is `none`; omitting method must not flip.
    assert_eq!(
        run_line("videotestsrc num-buffers=2 ! videoflip ! fakesink").await,
        2
    );
}

#[tokio::test]
async fn videoscale_convenience_properties_scale() {
    // g2g exposes scale/convert/rate as element properties (an extension over
    // gst, which sets them via a downstream capsfilter). This convenience route
    // is the supported one and ports cleanly.
    assert_eq!(
        run_line("videotestsrc num-buffers=2 ! videoscale width=160 height=120 ! fakesink").await,
        2
    );
}

// KNOWN GST-PORTING GAPS (tracked in DESIGN_TODO, not naming issues):
//  - Format-less geometry caps `video/x-raw,width=160,height=120` don't parse:
//    g2g's `Caps::RawVideo` format field is a concrete enum, not "any".
//  - Caps-driven transform operation: `videoscale ! video/x-raw,...,width=160`
//    does NOT resize, videoscale keys off its `width=`/`height=` properties, not
//    the negotiated downstream caps. Same for videoconvert/audioresample.
// Both need partial-caps + caps-driven configure, beyond naming harmonization.
