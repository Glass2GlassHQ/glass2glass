#![cfg(feature = "analytics")]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::log::{LogLevel, LogRecord, LogSink};
use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{
    AnalyticsMeta, AnalyticsNode, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    Dim, G2gError, Graph, OutputSink, PipelineClock, PipelinePacket, PropValue, Rate,
    RawVideoFormat,
};
use g2g_plugins::batchdemux::BatchDemux;
use g2g_plugins::batchmux::BatchMux;
use g2g_plugins::videotestsrc::VideoTestSrc;
use g2g_python::PyTransform;
use pyo3::prelude::*;

const FIXTURE_MODULE: &str = "batch_element";
const FIXTURE_CLASS: &str = "BatchDetector";
const FRAMES_PER_SOURCE: u64 = 3;
const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const FIRST_FPS: u32 = 30;
const SECOND_FPS: u32 = 25;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixtures_on_path() {
    static SET: std::sync::Once = std::sync::Once::new();
    SET.call_once(|| {
        std::env::set_var(
            "PYTHONPATH",
            concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
        );
    });
}

fn rgba(width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

#[derive(Debug, PartialEq)]
struct Detection {
    class_name: String,
    x_pixels: f32,
}

#[derive(Debug, Default)]
struct Received {
    frames: Vec<(u64, Vec<Detection>)>,
    eos: bool,
}

struct RecordingSink(Arc<Mutex<Received>>);

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let mut received = self.0.lock().unwrap();
        match packet {
            PipelinePacket::DataFrame(frame) => {
                let detections = frame
                    .meta
                    .get::<AnalyticsMeta>()
                    .map(|analytics| {
                        analytics
                            .nodes
                            .iter()
                            .filter_map(|node| match node {
                                AnalyticsNode::Detection(detection) => Some(Detection {
                                    class_name: analytics
                                        .class_name(detection.label)
                                        .unwrap_or_default()
                                        .to_string(),
                                    x_pixels: detection.bbox.x * WIDTH as f32,
                                }),
                                _ => None,
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                received.frames.push((frame.timing.pts_ns, detections));
            }
            PipelinePacket::Eos => received.eos = true,
            _ => {}
        }
        Box::pin(async { Ok(()) })
    }
}

#[derive(Clone, Default)]
struct PyelementErrors(Arc<Mutex<Vec<String>>>);

impl LogSink for PyelementErrors {
    fn emit(&self, record: &LogRecord<'_>) {
        if record.level == LogLevel::Error && record.category == "pyelement" {
            self.0.lock().unwrap().push(record.message.to_string());
        }
    }
}

fn logged_error_containing(errors: &PyelementErrors, needle: &str) -> bool {
    errors
        .0
        .lock()
        .unwrap()
        .iter()
        .any(|message| message.contains(needle))
}

async fn run_through_pyelement(second_width: u32) -> (Result<(), G2gError>, Received, Received) {
    fixtures_on_path();
    let sinks = [
        Arc::new(Mutex::new(Received::default())),
        Arc::new(Mutex::new(Received::default())),
    ];
    let mut g: Graph<GraphNode> = Graph::new();
    let first = g.add_source(GraphNode::source(VideoTestSrc::new(
        WIDTH,
        HEIGHT,
        FIRST_FPS,
        FRAMES_PER_SOURCE,
    )));
    let second = g.add_source(GraphNode::source(VideoTestSrc::new(
        second_width,
        HEIGHT,
        SECOND_FPS,
        FRAMES_PER_SOURCE,
    )));
    let mux = g.add_muxer(GraphNode::muxer(BatchMux::new(2)), 2);
    let detector = g.add_transform(GraphNode::element(PyTransform::new(
        FIXTURE_MODULE,
        FIXTURE_CLASS,
    )));
    let demux = g.add_demux(GraphNode::demux(BatchDemux::new(2)), 2);
    g.link(first, mux.input(0)).unwrap();
    g.link(second, mux.input(1)).unwrap();
    g.link(mux.output(), detector).unwrap();
    g.link(detector, demux.input()).unwrap();
    for (port, received) in sinks.iter().enumerate() {
        let sink = g.add_sink(GraphNode::element(RecordingSink(received.clone())));
        g.link(demux.out(port as u8), sink).unwrap();
    }
    let result = run_graph(g, &ZeroClock, 4).await.map(|_| ());
    let [first, second] = sinks.map(|s| std::mem::take(&mut *s.lock().unwrap()));
    (result, first, second)
}

fn fixture_batch_sizes() -> Vec<usize> {
    Python::attach(|py| {
        PyModule::import(py, FIXTURE_MODULE)?
            .getattr("BATCH_SIZES")?
            .extract()
    })
    .expect("fixture records its batch sizes")
}

fn expected_pts(fps: u32) -> Vec<u64> {
    let step = 1_000_000_000 / fps as u64;
    (0..FRAMES_PER_SOURCE).map(|i| i * step).collect()
}

#[tokio::test]
async fn each_stream_gets_its_own_results_from_one_call_per_batch() {
    let (result, first, second) = run_through_pyelement(WIDTH).await;
    result.expect("batch graph runs to EOS");

    for (stream, received, fps) in [(0, &first, FIRST_FPS), (1, &second, SECOND_FPS)] {
        assert!(received.eos, "stream {stream} ended");
        let pts: Vec<u64> = received.frames.iter().map(|(pts, _)| *pts).collect();
        assert_eq!(
            pts,
            expected_pts(fps),
            "stream {stream} kept its own timing"
        );
        for (_, detections) in &received.frames {
            assert_eq!(
                detections,
                &vec![Detection {
                    class_name: String::from("thing"),
                    x_pixels: (stream + 1) as f32,
                }],
                "stream {stream} carries only its own object"
            );
        }
    }
    assert_eq!(
        fixture_batch_sizes(),
        vec![2; FRAMES_PER_SOURCE as usize],
        "every call saw both streams"
    );
}

#[tokio::test]
async fn streams_of_different_geometry_fail_naming_the_stream() {
    let errors = PyelementErrors::default();
    let sink_id = g2g_core::log::add_sink(Box::new(errors.clone()));
    let (result, first, second) = run_through_pyelement(2 * WIDTH).await;
    g2g_core::log::remove_sink(sink_id);

    assert_eq!(result, Err(G2gError::CapsMismatch));
    assert!(first.frames.is_empty() && second.frames.is_empty());
    assert!(
        logged_error_containing(&errors, "batch stream 1 is"),
        "{:?}",
        errors.0.lock().unwrap()
    );
}

#[test]
fn cuda_frames_refuse_a_batch_at_configure() {
    let errors = PyelementErrors::default();
    let sink_id = g2g_core::log::add_sink(Box::new(errors.clone()));
    let mut detector = PyTransform::new(FIXTURE_MODULE, FIXTURE_CLASS);
    detector
        .set_property("cuda-frames", PropValue::Bool(true))
        .unwrap();
    let batch = Caps::Batch {
        streams: vec![rgba(WIDTH, HEIGHT), rgba(WIDTH, HEIGHT)],
    };
    let result = detector.configure_pipeline(&batch);
    g2g_core::log::remove_sink(sink_id);

    assert_eq!(result.err(), Some(G2gError::UnsupportedDomain));
    assert!(
        logged_error_containing(&errors, "system memory only"),
        "{:?}",
        errors.0.lock().unwrap()
    );
}
