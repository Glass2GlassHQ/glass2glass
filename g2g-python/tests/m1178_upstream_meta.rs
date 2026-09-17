//! M1178: the hosted element reads what upstream attached (`analytics` feature).
//!
//! The `meta` sink handed to `g2g_process` is filled from the incoming frame
//! before the call, so a hosted tracker or alert element sees a native
//! detector's results. The fixture echoes each reading back as a blob, and what
//! the element stages is appended to the upstream metadata rather than replacing
//! it. Needs libpython + the `metadata`-enabled core, so the whole file compiles
//! away without the feature.
#![cfg(feature = "analytics")]

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AnalyticsMeta, AnalyticsNode, AsyncElement, BBox, BlobMeta, Caps, Dim, Frame, FrameTiming,
    G2gError, MemoryDomain, ObjectDetection, OutputSink, PipelinePacket, PushOutcome, Rate,
    RawVideoFormat, RelationKind, Tracking,
};
use g2g_python::PyTransform;

/// Frame geometry every expected pixel value below is derived from.
const WIDTH: u32 = 4;
const HEIGHT: u32 = 2;
/// The one detection upstream attached, normalized as `BBox` always is.
const UPSTREAM_BOX: BBox = BBox {
    x: 0.25,
    y: 0.5,
    w: 0.5,
    h: 0.5,
};
const UPSTREAM_SCORE: f32 = 0.75;
/// Index into `CLASS_NAMES`, so the element reads back "car" rather than "1".
const UPSTREAM_LABEL: u32 = 1;
const CLASS_NAMES: [&str; 2] = ["person", "car"];
const UPSTREAM_OBJECT_ID: u64 = 42;
/// Spelled the way gst-python-ml puts it on the wire; the canonical header the
/// element reads back is `alert`.
const ALERT_HEADER_ON_THE_WIRE: &str = "GST-ALERT:";
const ALERT_HEADER: &str = "alert";
const ALERT_PAYLOAD: &[u8] = b"fire";

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

/// A frame carrying one tracked detection and one alert blob, as a native
/// detector plus tracker upstream would leave it.
fn frame_with_upstream_meta() -> Frame {
    let bytes = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
    let mut frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    };
    let mut analytics = AnalyticsMeta::new();
    analytics.set_class_names(CLASS_NAMES);
    let detection = analytics.add_detection(ObjectDetection {
        bbox: UPSTREAM_BOX,
        label: UPSTREAM_LABEL,
        confidence: UPSTREAM_SCORE,
    });
    let tracking = analytics.push(AnalyticsNode::Tracking(Tracking {
        object_id: UPSTREAM_OBJECT_ID,
    }));
    analytics.relate(detection, tracking, RelationKind::Tracks);
    frame.meta.attach(analytics);
    let mut blobs = BlobMeta::new();
    blobs.push(ALERT_HEADER_ON_THE_WIRE, ALERT_PAYLOAD.to_vec());
    frame.meta.attach(blobs);
    frame
}

/// Run one frame through the hosted `class` and return the frame that came out.
fn run(class: &str, frame: Frame) -> Frame {
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );
    let mut element = PyTransform::new("m1178_element", class);
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
}

fn blob<'a>(blobs: &'a BlobMeta, header: &str) -> &'a [u8] {
    &blobs
        .get(header)
        .unwrap_or_else(|| panic!("no blob tagged {header}"))
        .payload
}

fn text<'a>(blobs: &'a BlobMeta, header: &str) -> &'a str {
    core::str::from_utf8(blob(blobs, header)).expect("fixture blobs are utf-8")
}

#[test]
fn upstream_detections_and_blobs_reach_the_hosted_element() {
    let out = run("MetaReader", frame_with_upstream_meta());
    let blobs = out
        .meta
        .get::<BlobMeta>()
        .expect("the element's own blobs plus the upstream one");

    assert_eq!(blob(blobs, "seen-count"), [1], "one upstream detection");
    assert_eq!(
        text(blobs, "seen-label"),
        CLASS_NAMES[UPSTREAM_LABEL as usize],
        "the label resolves through the class table the producer published"
    );
    // The normalized box is reported in pixels of the frame being processed.
    let expected_box = format!(
        "{},{},{},{}",
        (UPSTREAM_BOX.x * WIDTH as f32) as i64,
        (UPSTREAM_BOX.y * HEIGHT as f32) as i64,
        (UPSTREAM_BOX.w * WIDTH as f32) as i64,
        (UPSTREAM_BOX.h * HEIGHT as f32) as i64,
    );
    assert_eq!(text(blobs, "seen-box"), expected_box);
    assert_eq!(text(blobs, "seen-score"), format!("{UPSTREAM_SCORE:.3}"));
    assert_eq!(text(blobs, "seen-names"), CLASS_NAMES.join(","));
    assert_eq!(
        text(blobs, "seen-headers"),
        ALERT_HEADER,
        "the wire spelling reads back canonical"
    );
    assert_eq!(blob(blobs, "seen-alert"), ALERT_PAYLOAD);
    assert_eq!(
        text(blobs, "seen-tracking"),
        UPSTREAM_OBJECT_ID.to_string(),
        "the tracking identity related to the detection"
    );
    // The blob upstream attached travels on next to the element's own.
    assert_eq!(blob(blobs, ALERT_HEADER), ALERT_PAYLOAD);
}

#[test]
fn staged_records_append_to_upstream_metadata() {
    // Mirrors the fixture's AppendingTransform.
    const STAGED_LABEL: u32 = 3;
    const STAGED_OBJECT_ID: u64 = 99;

    let out = run("AppendingTransform", frame_with_upstream_meta());

    let blobs = out.meta.get::<BlobMeta>().expect("blobs on the way out");
    assert_eq!(
        blob(blobs, ALERT_HEADER),
        ALERT_PAYLOAD,
        "the upstream blob survives the element's own"
    );
    assert_eq!(blob(blobs, "verdict"), b"ok");

    let analytics = out
        .meta
        .get::<AnalyticsMeta>()
        .expect("analytics on the way out");
    let detections: Vec<_> = analytics.detections().collect();
    assert_eq!(
        detections.len(),
        2,
        "upstream's detection plus the staged one"
    );
    assert_eq!(detections[0].label, UPSTREAM_LABEL);
    assert_eq!(detections[1].label, STAGED_LABEL);
    assert_eq!(
        analytics.class_name(UPSTREAM_LABEL),
        Some(CLASS_NAMES[UPSTREAM_LABEL as usize]),
        "the element published no table, so upstream's is kept"
    );

    // Both relations still point at their own tracking node after the append
    // shifted the staged indices.
    let tracked: Vec<u64> = analytics
        .relations
        .iter()
        .filter(|relation| relation.kind == RelationKind::Tracks)
        .filter_map(|relation| match analytics.nodes.get(relation.to) {
            Some(AnalyticsNode::Tracking(tracking)) => Some(tracking.object_id),
            _ => None,
        })
        .collect();
    assert_eq!(tracked, [UPSTREAM_OBJECT_ID, STAGED_OBJECT_ID]);

    // The staged box was given in pixels of the whole frame.
    assert_eq!(detections[1].bbox.w, 1.0);
    assert_eq!(detections[1].bbox.h, 1.0);
}
