//! M469 gst-launch equivalence corpus: a regression guard turning "g2g accepts
//! gst-launch syntax" from a claim into a checked guarantee. Two halves:
//!
//! 1. **Portable** lines (canonical GStreamer recipes using elements g2g has on
//!    the baseline `std` registry) must `parse_launch` into a runnable graph; a
//!    representative subset also `run_graph`s to prove end-to-end flow.
//! 2. **Needs-porting** lines (an element g2g lacks) must be flagged by the
//!    launch linter with guidance, so a porter gets a pointer instead of a
//!    silent failure.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::gst_compat::{gst_equivalent, lint_launch, GstEquivalent};
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// gst-launch lines that must parse into a runnable graph on the baseline
/// registry. Adapted from common GStreamer recipes; each exercises a distinct
/// piece of the DSL (inline caps, tee branches, quoted spaced properties,
/// caps-driven transforms, fan-out).
const PORTABLE: &[&str] = &[
    // The canonical smoke line.
    "videotestsrc ! videoconvert ! fakesink",
    // Inline caps shorthand becomes a capsfilter (a format conversion).
    "videotestsrc ! videoconvert ! video/x-raw,format=NV12 ! fakesink",
    // Caps-driven videoscale target via a trailing filter.
    "videotestsrc ! videoscale ! video/x-raw,width=640,height=480 ! videoconvert ! fakesink",
    // Caps-driven videorate.
    "videotestsrc ! videorate ! video/x-raw,framerate=15/1 ! fakesink",
    // Multiple transforms with numeric / enum properties.
    "videotestsrc ! videoflip method=horizontal-flip ! videobalance saturation=0.5 contrast=1.2 ! videoconvert ! fakesink",
    // A quoted property value with a space (the M467 tokenizer) in a real recipe.
    "filesrc location=\"/tmp/my video.ts\" ! fakesink",
    // tee fan-out with per-branch queues (queue -> LinkPolicy).
    "videotestsrc ! tee name=t ! queue ! fakesink t. ! queue ! videoconvert ! fakesink",
    // An audio chain with a property.
    "audiotestsrc ! volume volume=0.5 ! audioconvert ! audioresample ! fakesink",
    // A file source feeding a sink (parses without the file present).
    "filesrc location=/tmp/input.ts ! fakesink",
];

/// The subset of PORTABLE that also runs end to end on the baseline registry (a
/// test source with a bounded buffer count and transforms that negotiate on
/// videotestsrc's RGBA / audiotestsrc's PCM without extra features), with the
/// frame count expected at the sink(s). The tee line has two sinks, so the
/// runner consumes each of its 3 frames twice.
const RUNNABLE: &[(&str, u64)] = &[
    ("videotestsrc num-buffers=3 ! videoconvert ! fakesink", 3),
    ("videotestsrc num-buffers=3 ! videoflip method=horizontal-flip ! videobalance saturation=0.5 ! fakesink", 3),
    ("videotestsrc num-buffers=3 ! tee name=t ! queue ! fakesink t. ! queue ! fakesink", 6),
    ("audiotestsrc num-buffers=3 ! audioconvert ! audioresample ! fakesink", 3),
];

#[test]
fn portable_lines_parse() {
    let reg = default_registry();
    for line in PORTABLE {
        assert!(
            parse_launch(&reg, line).is_ok(),
            "portable gst-launch line should build a graph: {line}"
        );
    }
}

#[test]
fn portable_lines_lint_clean() {
    // The linter must not false-positive on a fully-supported line: no element
    // in these recipes is unportable, so it reports nothing.
    let reg = default_registry();
    for line in PORTABLE {
        let report = lint_launch(&reg, line);
        assert!(
            report.findings.is_empty(),
            "no porting findings expected for a portable line: {line}\n  got: {:?}",
            report.findings
        );
    }
}

#[tokio::test]
async fn runnable_lines_run_end_to_end() {
    for (line, expected) in RUNNABLE {
        let reg = default_registry();
        let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses: {line}: {e}"));
        let stats = run_graph(graph, &ZeroClock, 4)
            .await
            .unwrap_or_else(|e| panic!("runs: {line}: {e:?}"));
        assert_eq!(
            stats.frames_consumed, *expected,
            "expected frame count at sink(s): {line}"
        );
    }
}

#[test]
fn unportable_elements_are_flagged() {
    let reg = default_registry();
    // theoraenc / x265enc have no g2g equivalent under any feature, so the
    // linter reports them regardless of the compiled feature set.
    for (line, elem) in [
        (
            "videotestsrc ! theoraenc ! filesink location=out.ogv",
            "theoraenc",
        ),
        (
            "videotestsrc ! x265enc ! filesink location=out.h265",
            "x265enc",
        ),
    ] {
        let report = lint_launch(&reg, line);
        assert!(
            report.findings.iter().any(|f| f.contains(elem)),
            "linter should flag the unportable `{elem}` in: {line}\n  got: {:?}",
            report.findings
        );
    }
}

#[test]
fn unportable_elements_map_to_guidance() {
    // The porting table gives feature-stable advice for these (no g2g element
    // exists under any feature), so a porter gets a pointer, not a dead end.
    let reg = default_registry();
    assert!(matches!(
        gst_equivalent(&reg, "theoraenc"),
        GstEquivalent::Unsupported(_)
    ));
    assert!(matches!(
        gst_equivalent(&reg, "x265enc"),
        GstEquivalent::Unsupported(_)
    ));
    // A baseline element resolves as available.
    assert_eq!(
        gst_equivalent(&reg, "videoconvert"),
        GstEquivalent::Available
    );
}
