#![cfg(feature = "analytics")]

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AnalyticsMeta, AnalyticsNode, AsyncElement, BBox, Caps, Dim, Frame, FrameTiming, G2gError,
    MemoryDomain, ObjectDetection, OutputSink, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
    RelationKind,
};
use g2g_python::PyTransform;

const WIDTH: u32 = 4;
const HEIGHT: u32 = 2;
const BOX: BBox = BBox {
    x: 0.0,
    y: 0.0,
    w: 0.5,
    h: 0.5,
};

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame(analytics: Option<AnalyticsMeta>) -> Frame {
    let bytes = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    let mut frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    if let Some(analytics) = analytics {
        frame.meta.attach(analytics);
    }
    frame
}

fn run(class: &str, frame: Frame) -> AnalyticsMeta {
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );
    let mut element = PyTransform::new("m1904_element", class);
    element.configure_pipeline(&caps()).unwrap();
    let mut sink = CollectSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    runtime
        .block_on(element.process(PipelinePacket::DataFrame(frame), &mut sink))
        .unwrap();
    let PipelinePacket::DataFrame(frame) = sink.packets.remove(0) else {
        panic!("expected a DataFrame downstream");
    };
    frame
        .meta
        .get::<AnalyticsMeta>()
        .expect("the frame carries analytics")
        .clone()
}

fn tracked_object(analytics: &AnalyticsMeta, detection: usize) -> Option<u64> {
    analytics.relations.iter().find_map(|relation| {
        if relation.from != detection || relation.kind != RelationKind::Tracks {
            return None;
        }
        match analytics.nodes.get(relation.to) {
            Some(AnalyticsNode::Tracking(tracking)) => Some(tracking.object_id),
            _ => None,
        }
    })
}

#[test]
fn a_tracking_id_lands_on_the_upstream_detection_it_names() {
    let mut upstream = AnalyticsMeta::new();
    let detections: Vec<usize> = (0..2)
        .map(|label| {
            upstream.add_detection(ObjectDetection {
                bbox: BOX,
                label,
                confidence: 0.9,
            })
        })
        .collect();

    let analytics = run("UpstreamTagger", frame(Some(upstream)));

    let detection_count = analytics
        .nodes
        .iter()
        .filter(|node| matches!(node, AnalyticsNode::Detection(_)))
        .count();
    assert_eq!(detection_count, 2, "no detection is added");
    assert_eq!(tracked_object(&analytics, detections[0]), None);
    assert_eq!(tracked_object(&analytics, detections[1]), Some(7));
}

#[test]
fn the_stream_prefix_stays_off_the_class_names() {
    let analytics = run("StreamLabeller", frame(None));
    assert_eq!(analytics.class_name(0), Some("person"));
}
