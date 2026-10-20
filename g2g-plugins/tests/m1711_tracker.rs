// M1711: `tracker ! analyticsoverlay show-track=true` through the graph runner.
// The source attaches detections the way `detectionpostprocess` does, two boxes
// that cross paths plus a frame with no analytics at all, and the sink reads the
// track id of each detection back off the frames the overlay forwards.

#![cfg(all(feature = "std", feature = "analytics"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::meta::{AnalyticsMeta, AnalyticsNode, BBox, ObjectDetection, RelationKind};
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::analyticsoverlay::AnalyticsOverlay;
use g2g_plugins::tracker::Tracker;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 64;
const DETECTED_FRAMES: usize = 20;
// The frame index, among all frames sent, that carries no analytics.
const BARE_FRAME: usize = 3;
const SIDE: f32 = 0.1;
const SPEED: f32 = 0.04;
const FROM_LEFT_LABEL: u32 = 1;
const FROM_RIGHT_LABEL: u32 = 2;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn detection(x: f32, label: u32) -> ObjectDetection {
    ObjectDetection {
        bbox: BBox {
            x,
            y: 0.4,
            w: SIDE,
            h: SIDE,
        },
        label,
        confidence: 0.9,
    }
}

// Two boxes moving towards each other, crossing half way through.
fn crossing(step: usize) -> AnalyticsMeta {
    let travelled = SPEED * step as f32;
    let mut meta = AnalyticsMeta::new();
    meta.add_detection(detection(0.1 + travelled, FROM_LEFT_LABEL));
    meta.add_detection(detection(0.8 - travelled, FROM_RIGHT_LABEL));
    meta
}

struct DetectionSrc;

impl SourceLoop for DetectionSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(rgba()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let mut step = 0;
            let total = DETECTED_FRAMES + 1;
            for index in 0..total {
                let mut frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(
                        vec![0u8; (WIDTH * HEIGHT * 4) as usize].into_boxed_slice(),
                    )),
                    FrameTiming::default(),
                    index as u64,
                );
                if index != BARE_FRAME {
                    frame.meta.attach(crossing(step));
                    step += 1;
                }
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(total as u64)
        })
    }
}

// The (label, track id) of every detection on a frame, `None` for a frame
// without analytics.
type SeenFrame = Option<Vec<(u32, Option<u64>)>>;

fn track_ids(meta: &AnalyticsMeta) -> Vec<(u32, Option<u64>)> {
    meta.nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| match node {
            AnalyticsNode::Detection(detection) => Some((index, detection.label)),
            _ => None,
        })
        .map(|(index, label)| {
            let id = meta
                .relations
                .iter()
                .find(|relation| relation.from == index && relation.kind == RelationKind::Tracks)
                .map(|relation| match meta.nodes[relation.to] {
                    AnalyticsNode::Tracking(tracking) => tracking.object_id,
                    _ => panic!("a Tracks relation points at a Tracking node"),
                });
            (label, id)
        })
        .collect()
}

struct RecordingSink {
    seen: Arc<Mutex<Vec<SeenFrame>>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = core::future::Ready<Result<(), G2gError>>
    where
        Self: 'a;
    fn intercept_caps(&self, caps: &Caps) -> Result<Caps, G2gError> {
        Ok(caps.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        if let PipelinePacket::DataFrame(frame) = packet {
            let ids = frame.meta.get::<AnalyticsMeta>().map(track_ids);
            self.seen.lock().unwrap().push(ids);
        }
        core::future::ready(Ok(()))
    }
}

#[tokio::test]
async fn crossing_boxes_reach_the_overlay_with_their_track_ids() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut graph: Graph<GraphNode> = Graph::new();
    let source = graph.add_source(GraphNode::source(DetectionSrc));
    let tracker = graph.add_transform(GraphNode::element(Tracker::new()));
    let overlay = graph.add_transform(GraphNode::element(AnalyticsOverlay::new().with_track(true)));
    let sink = graph.add_sink(GraphNode::element(RecordingSink { seen: seen.clone() }));
    graph.link(source, tracker).unwrap();
    graph.link(tracker, overlay).unwrap();
    graph.link(overlay, sink).unwrap();

    run_graph(graph, &NullClock, 4).await.expect("graph runs");

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), DETECTED_FRAMES + 1, "every frame arrives");
    for (index, frame) in seen.iter().enumerate() {
        if index == BARE_FRAME {
            assert_eq!(*frame, None, "a frame without analytics gains none");
            continue;
        }
        assert_eq!(
            frame.as_deref(),
            Some(&[(FROM_LEFT_LABEL, Some(1)), (FROM_RIGHT_LABEL, Some(2))][..]),
            "frame {index}"
        );
    }
}
