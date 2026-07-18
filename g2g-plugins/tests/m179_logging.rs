//! M179: element-granular logging with instance names. Running a graph through
//! `run_graph` assigns each element a `<category>N` instance name (the GStreamer
//! `videotestsrc0` convention), logs each element's addition, and an element that
//! logs about itself (here `VideoFlip`) carries its instance name in its lines.
//! A capturing sink + a `*:trace` config lets the test assert all of this.

#![cfg(feature = "std")]

use std::sync::{Arc, Mutex};

use g2g_core::graph::Graph;
use g2g_core::log::{self, LogLevel, LogRecord, LogSink};
use g2g_core::runtime::{run_graph, GraphNodeRef};
use g2g_core::PipelineClock;

use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoflip::{FlipMethod, VideoFlip};
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// One captured record, flattened to owned data.
#[derive(Clone)]
struct Rec {
    level: LogLevel,
    category: String,
    instance: Option<String>,
    message: String,
}

/// A sink that appends every record to a shared vec, for assertions.
struct CaptureSink(Arc<Mutex<Vec<Rec>>>);
impl LogSink for CaptureSink {
    fn emit(&self, r: &LogRecord<'_>) {
        self.0.lock().unwrap().push(Rec {
            level: r.level,
            category: r.category.to_string(),
            instance: r.instance.map(|s| s.to_string()),
            message: format!("{}", r.message),
        });
    }
}

#[tokio::test]
async fn run_graph_names_instances_and_elements_self_log() {
    // Capture everything: install the sink and open all categories to TRACE.
    let captured = Arc::new(Mutex::new(Vec::new()));
    log::reset();
    log::set_sink(Box::new(CaptureSink(captured.clone())));
    log::configure("*:trace");

    // videotestsrc -> videoflip -> fakesink, two frames then EOS.
    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(VideoTestSrc::new(16, 16, 30, 2)));
    let flip = g.add_transform(GraphNodeRef::element(VideoFlip::new(FlipMethod::Rotate180)));
    let sink = g.add_sink(GraphNodeRef::element(FakeSink::new()));
    g.link(src, flip).unwrap();
    g.link(flip, sink).unwrap();

    run_graph(g, &ZeroClock, 4).await.expect("graph runs");

    let recs = captured.lock().unwrap().clone();
    log::reset();

    // Each element was named <category>N and logged on addition.
    let added: Vec<(&str, Option<&str>)> = recs
        .iter()
        .filter(|r| r.message == "added to pipeline")
        .map(|r| (r.category.as_str(), r.instance.as_deref()))
        .collect();
    assert!(
        added.contains(&("VideoTestSrc", Some("VideoTestSrc0"))),
        "source named + logged, got: {added:?}"
    );
    assert!(
        added.contains(&("VideoFlip", Some("VideoFlip0"))),
        "transform named, got: {added:?}"
    );
    assert!(
        added.contains(&("FakeSink", Some("FakeSink0"))),
        "sink named, got: {added:?}"
    );

    // VideoFlip logs about itself, carrying its assigned instance name and its
    // own category (same as the runner-derived one, so one filter covers both).
    let flip_configured = recs.iter().find(|r| {
        r.category == "VideoFlip"
            && r.instance.as_deref() == Some("VideoFlip0")
            && r.message.starts_with("configured")
    });
    assert!(
        flip_configured.is_some(),
        "videoflip self-logged its configure with its name"
    );
    assert_eq!(flip_configured.unwrap().level, LogLevel::Info);

    // Per-frame TRACE lines from the element, two frames in.
    let flip_frames = recs
        .iter()
        .filter(|r| r.level == LogLevel::Trace && r.instance.as_deref() == Some("VideoFlip0"))
        .count();
    assert_eq!(flip_frames, 2, "one per-frame trace per processed frame");

    // --- Phase 2: category filtering. Same process, so run sequentially (the
    // log config is process-global); only VideoFlip at debug, everything off. ---
    let captured = Arc::new(Mutex::new(Vec::new()));
    log::reset();
    log::set_sink(Box::new(CaptureSink(captured.clone())));
    log::configure("*:off,VideoFlip:debug");

    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(VideoTestSrc::new(16, 16, 30, 1)));
    let flip = g.add_transform(GraphNodeRef::element(VideoFlip::new(FlipMethod::Rotate180)));
    let sink = g.add_sink(GraphNodeRef::element(FakeSink::new()));
    g.link(src, flip).unwrap();
    g.link(flip, sink).unwrap();
    run_graph(g, &ZeroClock, 4).await.expect("graph runs");

    let recs = captured.lock().unwrap().clone();
    log::reset();

    assert!(
        recs.iter().all(|r| r.category == "VideoFlip"),
        "only VideoFlip passed the filter: {:?}",
        recs.iter().map(|r| &r.category).collect::<Vec<_>>()
    );
    assert!(
        recs.iter().any(|r| r.message == "added to pipeline"),
        "videoflip addition logged"
    );
    assert!(
        recs.iter().all(|r| r.level != LogLevel::Trace),
        "per-frame TRACE is above the DEBUG threshold, so suppressed"
    );
}
