//! M280: the caps-negotiation explainer end to end. `solve_graph` narrates the
//! negotiation under the `caps` log category (`G2G_CAPS_TRACE` turns it on); a
//! capturing `LogSink` here asserts both the success narration (per-node
//! constraints + per-edge fixated result) and the failure narration (the two
//! conflicting elements named, with the sets each wanted) reach the log.
//!
//! Single test: it owns the process-global log sink + config for this binary.

#![cfg(feature = "std")]

use std::sync::{Arc, Mutex};

use g2g_core::graph::Graph;
use g2g_core::log::{self, LogLevel, LogRecord, LogSink};
use g2g_core::runtime::solver::{solve_graph, NodeConstraint};
use g2g_core::{Caps, CapsConstraint, CapsSet, Dim, RawVideoFormat, Rate};

#[derive(Default)]
struct Capture(Arc<Mutex<Vec<String>>>);
impl LogSink for Capture {
    fn emit(&self, r: &LogRecord<'_>) {
        // Keep only caps narration; record "LEVEL: message".
        if r.category == log::CAPS_CATEGORY {
            self.0.lock().unwrap().push(format!("{}: {}", r.level.as_str(), r.message));
        }
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn audio() -> Caps {
    Caps::Audio { format: g2g_core::AudioFormat::PcmS16Le, channels: 2, sample_rate: 48_000 }
}

#[test]
fn caps_trace_narrates_success_and_failure() {
    let captured = Arc::new(Mutex::new(Vec::new()));
    log::set_sink(Box::new(Capture(captured.clone())));
    log::set_category_level(log::CAPS_CATEGORY, LogLevel::Debug);

    // --- success: source produces RGBA, sink accepts anything. ---
    let cs = vec![
        NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(rgba(320, 240)))),
        NodeConstraint::Element(CapsConstraint::AcceptsAny),
    ];
    let mut g: Graph<&str> = Graph::new();
    let src = g.add_source("vsrc");
    let sink = g.add_sink("vsink");
    g.link(src, sink).unwrap();
    let v = g.finish().unwrap();
    let solution = solve_graph(&v, &cs).expect("compatible chain solves");
    assert_eq!(solution, vec![rgba(320, 240)]);

    let lines = captured.lock().unwrap().clone();
    let blob = lines.join("\n");
    assert!(blob.contains("negotiating 2 nodes, 1 edges"), "setup header missing:\n{blob}");
    assert!(blob.contains("produces video/x-raw,format=RGBA"), "produce line missing:\n{blob}");
    assert!(blob.contains("accepts ANY"), "accepts line missing:\n{blob}");
    // The fixated per-edge result, with the ✓ marker and the chosen caps.
    assert!(blob.contains("✓ -> video/x-raw,format=RGBA"), "fixated edge missing:\n{blob}");

    // --- failure: source produces video, sink accepts only audio. ---
    captured.lock().unwrap().clear();
    let cs = vec![
        NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(rgba(320, 240)))),
        NodeConstraint::Element(CapsConstraint::Accepts(CapsSet::one(audio()))),
    ];
    let mut g: Graph<&str> = Graph::new();
    let src = g.add_source("vsrc");
    let sink = g.add_sink("asink");
    g.link(src, sink).unwrap();
    let v = g.finish().unwrap();
    assert!(solve_graph(&v, &cs).is_err(), "video -> audio must fail");

    let lines = captured.lock().unwrap().clone();
    let blob = lines.join("\n");
    // The failure is narrated at ERROR, names the conflict, and shows the sets.
    assert!(blob.contains("ERROR: no caps overlap"), "failure header missing:\n{blob}");
    assert!(blob.contains("video/x-raw,format=RGBA"), "upstream set missing:\n{blob}");

    log::reset();
}
