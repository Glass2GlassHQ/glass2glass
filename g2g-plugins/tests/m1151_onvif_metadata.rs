//! M1151: the ONVIF analytics metadata path, from the RTSP track to the video
//! frames the analytics describe.
//!
//! The scene descriptions come from fixtures transcribed out of the ONVIF
//! Analytics Specification 21.12 and Streaming Specification 26.06 (each
//! fixture cites its section and page); the expected boxes, class names and
//! instants are the ones those specifications state, not a restatement of what
//! the parser computes. No ONVIF camera is involved: the RTSP half is checked
//! against an in-process server that speaks the real protocol over loopback.
//!
//! ```sh
//! cargo test -p g2g-plugins --features onvif,rtsp --test m1151_onvif_metadata
//! ```

#![cfg(all(feature = "onvif", feature = "rtsp"))]

use std::path::PathBuf;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::meta::{
    AnalyticsMeta, AnalyticsNode, BBox, ObjectDetection, RelationKind, WallClockMeta,
};
use g2g_core::{Caps, G2gError, MultiInputElement, OutputSink, PropValue, PushOutcome, VideoCodec};
use g2g_plugins::onvifmetadata::{
    parse_metadata_documents, OnvifMetadataCombiner, OnvifMetadataParse, MAX_ELEMENT_DEPTH,
    MAX_OBJECTS_PER_DOCUMENT, UNCLASSIFIED_LABEL,
};

/// Slack for the f32 coordinate arithmetic. Far tighter than one pixel of the
/// 640x480 picture the specification's examples describe (1/640 = 0.0016).
const TOLERANCE: f32 = 1e-5;

/// The instants the specification's example frames name, as nanoseconds since
/// the Unix epoch.
const FRAME_1_NANOS: i64 = 1_223_641_497_321_000_000; // 2008-10-10T12:24:57.321
const FRAME_2_NANOS: i64 = 1_223_641_497_421_000_000; // 2008-10-10T12:24:57.421
const FRAME_3_NANOS: i64 = 1_223_641_497_521_000_000; // 2008-10-10T12:24:57.521
const FRAME_4_NANOS: i64 = 1_223_641_497_621_000_000; // 2008-10-10T12:24:57.621
const FRAME_5_NANOS: i64 = 1_223_641_497_721_000_000; // 2008-10-10T12:24:57.721

/// The normalized box the Analytics Specification's `left=20 top=80 right=100
/// bottom=30` object occupies under its `Translate(-1,-1) Scale(0.003125,
/// 0.00416667)` frame transformation: the top row is 80 pixels above the bottom
/// of a 640x480 picture.
const SPEC_BOX: BBox = BBox {
    x: 0.031_25,
    y: 0.833_333,
    w: 0.125,
    h: 0.104_167,
};

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn assert_box(got: &BBox, want: &BBox) {
    for (label, got, want) in [
        ("x", got.x, want.x),
        ("y", got.y, want.y),
        ("w", got.w, want.w),
        ("h", got.h, want.h),
    ] {
        assert!(
            (got - want).abs() <= TOLERANCE,
            "{label}: got {got}, want {want} (box {got:?})",
        );
    }
}

/// The one detection a frame holds, panicking if it holds a different number.
fn only_detection(meta: &AnalyticsMeta) -> &ObjectDetection {
    let mut detections = meta.detections();
    let first = detections.next().expect("one detection");
    assert!(
        detections.next().is_none(),
        "expected exactly one detection"
    );
    first
}

/// The object ids the frame's `Tracking` nodes carry, in order.
fn tracked_ids(meta: &AnalyticsMeta) -> Vec<u64> {
    meta.nodes
        .iter()
        .filter_map(|n| match n {
            AnalyticsNode::Tracking(t) => Some(t.object_id),
            _ => None,
        })
        .collect()
}

// ---- document parsing ----

#[test]
fn the_spec_example_frames_become_normalized_boxes() {
    let frames = parse_metadata_documents(&fixture("onvif_transformed_frames.xml"));
    assert_eq!(frames.len(), 5, "one output per tt:Frame");
    assert_eq!(
        frames.iter().map(|f| f.unix_nanos).collect::<Vec<_>>(),
        [
            Some(FRAME_1_NANOS),
            Some(FRAME_2_NANOS),
            Some(FRAME_3_NANOS),
            Some(FRAME_4_NANOS),
            Some(FRAME_5_NANOS),
        ],
    );

    assert_box(&only_detection(&frames[0].analytics).bbox, &SPEC_BOX);
    // The Idle behaviour changes nothing about where the object is.
    assert_box(&only_detection(&frames[1].analytics).bbox, &SPEC_BOX);
    // The fourth frame's object has moved five pixels right, which is
    // 5 * 0.003125 / 2 of the normalized width.
    let moved = only_detection(&frames[3].analytics).bbox;
    assert_box(
        &moved,
        &BBox {
            x: SPEC_BOX.x + 5.0 * 0.003_125 / 2.0,
            ..SPEC_BOX
        },
    );
    // Object ids are carried through as tracking identities.
    assert_eq!(tracked_ids(&frames[0].analytics), [12]);
    assert_eq!(tracked_ids(&frames[4].analytics), [19]);
}

#[test]
fn an_empty_frame_still_yields_a_frame() {
    let frames = parse_metadata_documents(&fixture("onvif_transformed_frames.xml"));
    // The specification's third example frame carries a transformation and no
    // objects: the receiver has to see it, since it means the scene emptied.
    let empty = &frames[2];
    assert_eq!(empty.unix_nanos, Some(FRAME_3_NANOS));
    assert_eq!(empty.analytics.nodes.len(), 0);
    assert_eq!(empty.analytics.relations.len(), 0);
}

#[test]
fn a_detection_is_related_to_the_tracking_node_holding_its_object_id() {
    let frames = parse_metadata_documents(&fixture("onvif_transformed_frames.xml"));
    let meta = &frames[0].analytics;
    assert_eq!(meta.nodes.len(), 2, "one detection and one tracking node");
    assert_eq!(
        meta.relations,
        [g2g_core::meta::Relation {
            from: 0,
            to: 1,
            kind: RelationKind::Tracks,
        }],
    );
    assert!(matches!(meta.nodes[0], AnalyticsNode::Detection(_)));
    assert!(matches!(
        meta.nodes[1],
        AnalyticsNode::Tracking(g2g_core::meta::Tracking { object_id: 12 })
    ));
}

#[test]
fn the_likeliest_class_type_names_the_detection() {
    let frames = parse_metadata_documents(&fixture("onvif_multi_class.xml"));
    assert_eq!(frames.len(), 1);
    let meta = &frames[0].analytics;
    let detection = only_detection(meta);
    // Vehicle 0.8 outranks Car 0.75 and Truck 0.3.
    assert_eq!(meta.class_name(detection.label), Some("Vehicle"));
    assert!((detection.confidence - 0.8).abs() <= TOLERANCE);
    assert_box(&detection.bbox, &SPEC_BOX);
}

#[test]
fn the_legacy_class_candidate_form_reads_the_same_way() {
    let frames = parse_metadata_documents(&fixture("onvif_legacy_class_candidate.xml"));
    let meta = &frames[0].analytics;
    let detection = only_detection(meta);
    assert_eq!(meta.class_name(detection.label), Some("Vehicle"));
    assert!((detection.confidence - 0.91).abs() <= TOLERANCE);
}

#[test]
fn an_object_with_no_class_carries_no_name() {
    let frames = parse_metadata_documents(&fixture("onvif_transformed_frames.xml"));
    let detection = only_detection(&frames[0].analytics);
    assert_eq!(detection.label, UNCLASSIFIED_LABEL);
    assert_eq!(frames[0].analytics.class_name(detection.label), None);
}

#[test]
fn a_parent_in_the_same_frame_becomes_a_contains_relation() {
    let frames = parse_metadata_documents(&fixture("onvif_parent_same_frame.xml"));
    assert_eq!(frames.len(), 1);
    let meta = &frames[0].analytics;
    // Objects 14 (the plate, listed first and naming 12 as its Parent) then 12.
    assert_eq!(tracked_ids(meta), [14, 12]);
    let plate = 0;
    let vehicle = 2;
    assert!(
        meta.relations.contains(&g2g_core::meta::Relation {
            from: vehicle,
            to: plate,
            kind: RelationKind::Contains,
        }),
        "the vehicle contains the plate: {:?}",
        meta.relations,
    );
    assert_eq!(meta.class_name(0), Some("LicensePlate"));
    assert_eq!(meta.class_name(1), Some("Vehicle"));
}

#[test]
fn a_parent_in_another_frame_relates_nothing() {
    // The specification prints the vehicle and its plate in consecutive frames,
    // where nothing can be related: a relation is within one frame's graph.
    let frames = parse_metadata_documents(&fixture("onvif_parent_object.xml"));
    assert_eq!(frames.len(), 2);
    for frame in &frames {
        assert!(
            !frame
                .analytics
                .relations
                .iter()
                .any(|r| r.kind == RelationKind::Contains),
            "no Contains relation across frames",
        );
    }
    let vehicle = only_detection(&frames[0].analytics);
    assert_eq!(
        frames[0].analytics.class_name(vehicle.label),
        Some("Vehicle")
    );
    let plate = only_detection(&frames[1].analytics);
    assert_eq!(
        frames[1].analytics.class_name(plate.label),
        Some("LicensePlate")
    );
}

#[test]
fn an_untransformed_box_is_already_in_the_normalized_system() {
    let frames = parse_metadata_documents(&fixture("onvif_axis_untransformed.xml"));
    assert_eq!(frames.len(), 1);
    let detection = only_detection(&frames[0].analytics);
    // left=-0.6 top=0.6 right=-0.2 bottom=0.2, y up about the picture centre.
    assert_box(
        &detection.bbox,
        &BBox {
            x: 0.2,
            y: 0.2,
            w: 0.2,
            h: 0.2,
        },
    );
    // 2024-03-01T08:15:30.500+02:00 is 06:15:30.5 UTC.
    assert_eq!(frames[0].unix_nanos, Some(1_709_273_730_500_000_000));
}

#[test]
fn two_concatenated_roots_both_parse() {
    let frames = parse_metadata_documents(&fixture("onvif_two_roots.xml"));
    // Two frames in the first root; the second root carries only an event.
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].unix_nanos, Some(FRAME_1_NANOS));
    assert_eq!(frames[1].unix_nanos, Some(FRAME_4_NANOS));
    assert_eq!(tracked_ids(&frames[0].analytics), [3]);
}

#[test]
fn a_declaration_inside_cdata_is_content_not_a_boundary() {
    let plain = fixture("onvif_axis_untransformed.xml");
    let want = parse_metadata_documents(&plain).len();
    assert!(want > 0);
    let with_cdata = String::from_utf8(plain).unwrap().replace(
        "</tt:Frame>",
        "<tt:Extension><![CDATA[<?xml version=\"1.0\"?>]]></tt:Extension></tt:Frame>",
    );
    assert_eq!(parse_metadata_documents(with_cdata.as_bytes()).len(), want);
}

#[test]
fn an_object_naming_itself_as_parent_relates_nothing() {
    let document = String::from_utf8(fixture("onvif_parent_same_frame.xml"))
        .unwrap()
        .replace("Parent=\"12\"", "Parent=\"14\"")
        // Negative ids are legal xs:integer but name no trackable object.
        .replace("ObjectId=\"12\"", "ObjectId=\"-12\"");
    let frames = parse_metadata_documents(document.as_bytes());
    assert_eq!(frames.len(), 1);
    let meta = &frames[0].analytics;
    assert_eq!(tracked_ids(meta), [14]);
    assert!(
        !meta
            .relations
            .iter()
            .any(|r| r.kind == RelationKind::Contains),
        "no self-loop: {:?}",
        meta.relations,
    );
}

#[test]
fn a_document_nested_past_the_depth_bound_yields_nothing() {
    // Deep enough to overflow the XML parser's stack if it were handed over.
    const LEVELS: usize = 20_000;
    let mut document = String::from("<?xml version=\"1.0\"?>");
    document.push_str(&"<x>".repeat(LEVELS));
    document.push_str(&"</x>".repeat(LEVELS));
    assert!(parse_metadata_documents(document.as_bytes()).is_empty());

    // Siblings are not levels: a frame of self-closing and paired objects at
    // the bound's depth still parses.
    let plain = String::from_utf8(fixture("onvif_axis_untransformed.xml")).unwrap();
    let padding = "<tt:Extension/>".repeat(MAX_ELEMENT_DEPTH * 2);
    let padded = plain.replace("</tt:Frame>", &format!("{padding}</tt:Frame>"));
    assert_eq!(
        parse_metadata_documents(padded.as_bytes()).len(),
        parse_metadata_documents(plain.as_bytes()).len(),
    );
}

#[test]
fn a_stream_with_an_event_part_still_yields_its_frames() {
    let frames = parse_metadata_documents(&fixture("onvif_cell_motion_stream.xml"));
    // The cell-motion example's two frames describe no objects, and the
    // tt:Event part beside them contributes none.
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].unix_nanos, Some(1_287_577_497_321_000_000));
    assert_eq!(frames[1].unix_nanos, Some(1_287_577_497_621_000_000));
    assert!(frames.iter().all(|f| f.analytics.nodes.is_empty()));
}

#[test]
fn a_missing_bounding_box_attribute_skips_only_that_object() {
    let frames = parse_metadata_documents(&fixture("onvif_missing_bbox_attribute.xml"));
    assert_eq!(frames.len(), 1);
    // Object 12's box has no `bottom`; object 13's is whole.
    assert_eq!(tracked_ids(&frames[0].analytics), [13]);
    assert_box(&only_detection(&frames[0].analytics).bbox, &SPEC_BOX);
}

#[test]
fn a_truncated_or_non_xml_document_yields_nothing() {
    let whole = fixture("onvif_transformed_frames.xml");
    assert!(!parse_metadata_documents(&whole).is_empty());
    assert!(parse_metadata_documents(&whole[..whole.len() / 2]).is_empty());
    assert!(parse_metadata_documents(b"not xml at all").is_empty());
    assert!(parse_metadata_documents(&[]).is_empty());
    // Invalid UTF-8 where the document should be.
    assert!(parse_metadata_documents(&[0xff, 0xfe, 0x00]).is_empty());
}

#[test]
fn the_object_count_bound_holds() {
    let over = MAX_OBJECTS_PER_DOCUMENT + 10;
    let mut document = String::from(
        "<?xml version=\"1.0\"?>\
         <tt:MetadataStream xmlns:tt=\"http://www.onvif.org/ver10/schema\">\
         <tt:VideoAnalytics><tt:Frame UtcTime=\"2008-10-10T12:24:57.321\">",
    );
    for id in 0..over {
        document.push_str(&format!(
            "<tt:Object ObjectId=\"{id}\"><tt:Appearance><tt:Shape>\
             <tt:BoundingBox left=\"-0.5\" top=\"0.5\" right=\"0.0\" bottom=\"0.0\"/>\
             </tt:Shape></tt:Appearance></tt:Object>",
        ));
    }
    document.push_str("</tt:Frame></tt:VideoAnalytics></tt:MetadataStream>");

    let frames = parse_metadata_documents(document.as_bytes());
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].analytics.detections().count(),
        MAX_OBJECTS_PER_DOCUMENT,
        "the bound caps how many objects one document may describe",
    );
}

// ---- onvifmetadataparse ----

#[derive(Default)]
struct RecordingSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for RecordingSink {
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

impl RecordingSink {
    fn take_frames(&mut self) -> Vec<Frame> {
        core::mem::take(&mut self.packets)
            .into_iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }
}

fn metadata_packet(document: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(document.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            ..Default::default()
        },
        0,
    ))
}

async fn run_parse(document: Vec<u8>, pts_ns: u64) -> Vec<Frame> {
    let mut parse = OnvifMetadataParse::new();
    parse
        .configure_pipeline(&Caps::OnvifMetadata)
        .expect("takes ONVIF metadata caps");
    let mut sink = RecordingSink::default();
    parse
        .process(metadata_packet(document, pts_ns), &mut sink)
        .await
        .expect("a well-formed document parses");
    sink.take_frames()
}

#[tokio::test]
async fn the_parse_element_emits_one_frame_per_tt_frame() {
    const PTS_NS: u64 = 1_500_000_000;
    let document = fixture("onvif_transformed_frames.xml");
    let frames = run_parse(document.clone(), PTS_NS).await;
    assert_eq!(frames.len(), 5);
    for (index, frame) in frames.iter().enumerate() {
        // The payload is the document itself, shared rather than copied, so a
        // recorder downstream still sees the XML the camera sent.
        assert_eq!(
            frame.domain.as_system_slice().expect("system memory"),
            document.as_slice(),
        );
        assert_eq!(frame.timing.pts_ns, PTS_NS);
        assert_eq!(frame.sequence, index as u64);
        assert!(frame.meta.get::<AnalyticsMeta>().is_some());
    }
    assert_eq!(
        frames
            .iter()
            .map(|f| f.meta.get::<WallClockMeta>().map(|w| w.unix_nanos))
            .collect::<Vec<_>>(),
        [
            Some(FRAME_1_NANOS),
            Some(FRAME_2_NANOS),
            Some(FRAME_3_NANOS),
            Some(FRAME_4_NANOS),
            Some(FRAME_5_NANOS),
        ],
    );
}

#[tokio::test]
async fn the_parse_element_drops_a_document_it_cannot_read() {
    assert!(run_parse(b"not xml at all".to_vec(), 0).await.is_empty());
}

#[tokio::test]
async fn the_parse_element_refuses_caps_that_are_not_onvif_metadata() {
    let mut parse = OnvifMetadataParse::new();
    assert!(parse.intercept_caps(&Caps::OnvifMetadata).is_ok());
    assert!(parse.intercept_caps(&h264_caps()).is_err());
    assert_eq!(
        parse.configure_pipeline(&h264_caps()).err(),
        Some(G2gError::CapsMismatch)
    );
}

// ---- onvifmetadatacombiner ----

/// 30 fps, the cadence the combiner tests step the video at.
const FRAME_NS: u64 = 33_333_333;
/// A wall clock for the synthetic video: the instant the specification's first
/// example frame names.
const VIDEO_START_NANOS: i64 = FRAME_1_NANOS;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: g2g_core::Dim::Any,
        height: g2g_core::Dim::Any,
        framerate: g2g_core::Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn combiner() -> OnvifMetadataCombiner {
    let mut combiner = OnvifMetadataCombiner::new();
    combiner
        .configure_pipeline(OnvifMetadataCombiner::VIDEO, &h264_caps())
        .expect("takes the video pad's caps");
    combiner
        .configure_pipeline(OnvifMetadataCombiner::METADATA, &Caps::OnvifMetadata)
        .expect("takes the metadata pad's caps");
    combiner
}

/// A synthetic video frame: one byte of payload, a PTS, a duration, and
/// optionally the sender's wall clock an `RtspSrcN` would have attached.
fn video_frame(pts_ns: u64, wall_nanos: Option<i64>) -> PipelinePacket {
    let mut frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8]))),
        FrameTiming {
            pts_ns,
            duration_ns: FRAME_NS,
            ..Default::default()
        },
        0,
    );
    if let Some(unix_nanos) = wall_nanos {
        frame.meta.attach(WallClockMeta { unix_nanos });
    }
    PipelinePacket::DataFrame(frame)
}

/// A parsed metadata frame carrying one detection whose tracking id is
/// `object_id`, so the test can tell which node landed on which picture.
fn parsed_metadata(object_id: u64, pts_ns: u64, wall_nanos: Option<i64>) -> PipelinePacket {
    let mut analytics = AnalyticsMeta::new();
    let detection = analytics.add_detection(ObjectDetection {
        bbox: BBox {
            x: 0.1,
            y: 0.1,
            w: 0.2,
            h: 0.2,
        },
        label: UNCLASSIFIED_LABEL,
        confidence: 1.0,
    });
    let tracking = analytics.push(AnalyticsNode::Tracking(g2g_core::meta::Tracking {
        object_id,
    }));
    analytics.relate(detection, tracking, RelationKind::Tracks);

    let mut frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8]))),
        FrameTiming {
            pts_ns,
            ..Default::default()
        },
        0,
    );
    frame.meta.attach(analytics);
    if let Some(unix_nanos) = wall_nanos {
        frame.meta.attach(WallClockMeta { unix_nanos });
    }
    PipelinePacket::DataFrame(frame)
}

/// The object ids each emitted video frame ended up carrying.
fn ids_per_frame(frames: &[Frame]) -> Vec<Vec<u64>> {
    frames
        .iter()
        .map(|f| {
            f.meta
                .get::<AnalyticsMeta>()
                .map(tracked_ids)
                .unwrap_or_default()
        })
        .collect()
}

#[tokio::test]
async fn each_video_frame_gets_the_metadata_of_its_own_window() {
    let mut combiner = combiner();
    let mut sink = RecordingSink::default();
    // Four pictures 33 ms apart, each preceded by a metadata frame whose wall
    // clock falls inside it (10 ms in), the order a PTS-ordered merge delivers.
    for index in 0..4u64 {
        let pts = index * FRAME_NS;
        let wall = VIDEO_START_NANOS + (index * FRAME_NS) as i64;
        combiner
            .process(
                OnvifMetadataCombiner::METADATA,
                parsed_metadata(index, pts, Some(wall + 10_000_000)),
                &mut sink,
            )
            .await
            .expect("the metadata queues");
        combiner
            .process(
                OnvifMetadataCombiner::VIDEO,
                video_frame(pts, Some(wall)),
                &mut sink,
            )
            .await
            .expect("the video frame is held or released");
    }
    combiner
        .process(OnvifMetadataCombiner::VIDEO, PipelinePacket::Eos, &mut sink)
        .await
        .expect("EOS flushes what is held");

    let frames = sink.take_frames();
    assert_eq!(frames.len(), 4, "every picture goes out exactly once");
    assert_eq!(ids_per_frame(&frames), [[0], [1], [2], [3]]);
}

#[tokio::test]
async fn metadata_older_than_max_lateness_is_dropped() {
    let mut combiner = combiner();
    combiner
        .set_property("max-lateness", PropValue::Uint(FRAME_NS))
        .expect("max-lateness is a declared property");
    let mut sink = RecordingSink::default();

    // Ten frames of video first, so the stream clock has run well past the
    // start; then metadata for the very first picture, long gone.
    for index in 0..10u64 {
        let pts = index * FRAME_NS;
        combiner
            .process(
                OnvifMetadataCombiner::VIDEO,
                video_frame(pts, Some(VIDEO_START_NANOS + (index * FRAME_NS) as i64)),
                &mut sink,
            )
            .await
            .expect("video");
    }
    combiner
        .process(
            OnvifMetadataCombiner::METADATA,
            parsed_metadata(99, 0, Some(VIDEO_START_NANOS)),
            &mut sink,
        )
        .await
        .expect("the late metadata is dropped, not an error");
    combiner
        .process(OnvifMetadataCombiner::VIDEO, PipelinePacket::Eos, &mut sink)
        .await
        .expect("EOS flushes");

    let frames = sink.take_frames();
    assert_eq!(frames.len(), 10);
    assert!(
        ids_per_frame(&frames).iter().all(|ids| ids.is_empty()),
        "nothing carries the late object",
    );
}

#[tokio::test]
async fn pts_stands_in_when_a_side_carries_no_wall_clock() {
    let mut combiner = combiner();
    let mut sink = RecordingSink::default();
    // The video knows its wall clock, the metadata does not (a camera that
    // sends no sender report, or a graph built by hand), so both fall back to
    // the play timeline.
    for index in 0..3u64 {
        let pts = index * FRAME_NS;
        combiner
            .process(
                OnvifMetadataCombiner::METADATA,
                parsed_metadata(index, pts + 1_000_000, None),
                &mut sink,
            )
            .await
            .expect("metadata");
        combiner
            .process(
                OnvifMetadataCombiner::VIDEO,
                video_frame(pts, Some(VIDEO_START_NANOS + (index * FRAME_NS) as i64)),
                &mut sink,
            )
            .await
            .expect("video");
    }
    combiner
        .process(
            OnvifMetadataCombiner::METADATA,
            PipelinePacket::Eos,
            &mut sink,
        )
        .await
        .expect("EOS flushes");

    let frames = sink.take_frames();
    assert_eq!(frames.len(), 3);
    assert_eq!(ids_per_frame(&frames), [[0], [1], [2]]);
}

#[tokio::test]
async fn the_wall_clock_takes_over_once_the_video_gets_its_first_sender_report() {
    let mut combiner = combiner();
    let mut sink = RecordingSink::default();
    // The first video frame precedes the sender report, so it carries no wall
    // clock; every later one does. The metadata track's PTS runs five pictures
    // ahead of the video's, which is the skew the wall clock exists to ignore.
    const SKEW_FRAMES: u64 = 5;
    const PICTURES: u64 = 8;
    for index in 0..PICTURES {
        let pts = index * FRAME_NS;
        let wall = VIDEO_START_NANOS + (index * FRAME_NS) as i64;
        combiner
            .process(
                OnvifMetadataCombiner::METADATA,
                parsed_metadata(index, pts + SKEW_FRAMES * FRAME_NS, Some(wall + 10_000_000)),
                &mut sink,
            )
            .await
            .expect("metadata");
        let video_wall = (index > 0).then_some(wall);
        combiner
            .process(
                OnvifMetadataCombiner::VIDEO,
                video_frame(pts, video_wall),
                &mut sink,
            )
            .await
            .expect("video");
    }
    combiner
        .process(OnvifMetadataCombiner::VIDEO, PipelinePacket::Eos, &mut sink)
        .await
        .expect("EOS flushes");

    let frames = sink.take_frames();
    assert_eq!(frames.len(), PICTURES as usize);
    let want: Vec<Vec<u64>> = (0..PICTURES)
        .map(|index| if index == 0 { vec![] } else { vec![index] })
        .collect();
    assert_eq!(ids_per_frame(&frames), want);
}

#[tokio::test]
async fn a_video_frame_with_no_metadata_goes_out_within_latency() {
    let mut combiner = combiner();
    combiner
        .set_property("latency", PropValue::Uint(2 * FRAME_NS))
        .expect("latency is a declared property");
    let mut sink = RecordingSink::default();

    // The metadata pad never produces anything. Each picture waits two frame
    // periods and then leaves on its own, so the element never stalls.
    for index in 0..6u64 {
        combiner
            .process(
                OnvifMetadataCombiner::VIDEO,
                video_frame(
                    index * FRAME_NS,
                    Some(VIDEO_START_NANOS + (index * FRAME_NS) as i64),
                ),
                &mut sink,
            )
            .await
            .expect("video");
    }
    // The wait is two frame periods, so the six pictures leave all but the two
    // newest behind.
    assert_eq!(sink.take_frames().len(), 4);
}

#[tokio::test]
async fn an_existing_analytics_meta_keeps_its_nodes() {
    let mut combiner = combiner();
    let mut sink = RecordingSink::default();

    // A detector upstream already wrote a classified detection onto the frame.
    let mut existing = AnalyticsMeta::new();
    existing.add_detection(ObjectDetection {
        bbox: BBox {
            x: 0.5,
            y: 0.5,
            w: 0.1,
            h: 0.1,
        },
        label: 0,
        confidence: 0.75,
    });
    existing.set_class_names(["person"]);
    let PipelinePacket::DataFrame(mut frame) = video_frame(0, Some(VIDEO_START_NANOS)) else {
        unreachable!("video_frame builds a DataFrame")
    };
    frame.meta.attach(existing);

    combiner
        .process(
            OnvifMetadataCombiner::METADATA,
            parsed_metadata(42, 0, Some(VIDEO_START_NANOS)),
            &mut sink,
        )
        .await
        .expect("metadata");
    combiner
        .process(
            OnvifMetadataCombiner::VIDEO,
            PipelinePacket::DataFrame(frame),
            &mut sink,
        )
        .await
        .expect("video");
    combiner
        .process(OnvifMetadataCombiner::VIDEO, PipelinePacket::Eos, &mut sink)
        .await
        .expect("EOS flushes");

    let frames = sink.take_frames();
    assert_eq!(frames.len(), 1);
    let meta = frames[0].meta.get::<AnalyticsMeta>().expect("analytics");
    // The detector's detection, then the camera's detection and its tracking
    // node.
    assert_eq!(meta.nodes.len(), 3);
    assert_eq!(tracked_ids(meta), [42]);
    let detections: Vec<&ObjectDetection> = meta.detections().collect();
    assert_eq!(detections.len(), 2);
    assert_eq!(meta.class_name(detections[0].label), Some("person"));
    assert!((detections[0].confidence - 0.75).abs() <= TOLERANCE);
    // The camera named no class, so its detection still resolves to no name.
    assert_eq!(meta.class_name(detections[1].label), None);
    // The relation the camera's frame carried still points at its own nodes.
    assert!(meta.relations.contains(&g2g_core::meta::Relation {
        from: 1,
        to: 2,
        kind: RelationKind::Tracks,
    }));
}

// ---- launch ----

/// The metadata pad is the second linked one, so `onvif-metadata=true` has to
/// trade the audio pad for it rather than growing the element to three outputs.
#[test]
fn the_metadata_branch_graph_parses() {
    let line = "rtspsrcn location=rtsp://camera/stream onvif-metadata=true name=s \
                s. ! h264parse ! onvifmetadatacombiner name=c ! fakesink \
                s. ! onvifmetadataparse ! c.";
    parse_line(line);
}

/// The whole line the module documents, drawing the camera's objects over the
/// decoded picture. `analyticsoverlay` needs the `analytics` feature.
#[cfg(feature = "analytics")]
#[test]
fn the_analytics_overlay_graph_parses() {
    let line = "rtspsrcn location=rtsp://camera/stream onvif-metadata=true name=s \
                s. ! decodebin ! videoconvert ! onvifmetadatacombiner name=c \
                ! analyticsoverlay ! fakesink \
                s. ! onvifmetadataparse ! c.";
    parse_line(line);
}

fn parse_line(line: &str) {
    let registry = g2g_plugins::registry::default_registry();
    g2g_core::runtime::parse_launch(&registry, line).unwrap_or_else(|e| panic!("{line}\n{e:?}"));
}

// ---- rtspsrcn over a real RTSP session ----

/// The metadata track's dynamic payload type in the served SDP, and the RTP
/// clock the Streaming Specification recommends for it (section 5.1.2.1.1).
const METADATA_PAYLOAD_TYPE: u8 = 107;
/// The two SDP encoding names a served metadata track is offered under.
const ONVIF_METADATA_PLAIN: &str = "vnd.onvif.metadata";
const ONVIF_METADATA_GZIP: &str = "vnd.onvif.metadata.gzip";
const METADATA_CLOCK_HZ: u32 = 90_000;
/// The metadata stream's synchronization source, the same on its RTP packets
/// and its sender report so the client accepts both as one stream.
const METADATA_SSRC: u32 = 0x1122_3344;
/// The instant the served sender report pins the stream to.
const SERVER_REPORT_NANOS: i64 = FRAME_1_NANOS;
/// The RTP timestamp that report names, and the step between documents (one
/// second of the metadata clock).
const REPORT_RTP_TIMESTAMP: u32 = 900_000;

/// Serve one RTSP session on loopback offering an H.264 video track and an
/// ONVIF metadata track, stream `documents` on the metadata track over
/// TCP-interleaved RTP, then close. Returns the URL to play.
///
/// The first document is split across two RTP packets so the client has to
/// concatenate to the marker bit, and the sender report is sent after it, so
/// the frames before and after the report can be told apart.
fn serve_onvif_metadata(
    documents: Vec<Vec<u8>>,
    encoding_name: &'static str,
    report_ntp: u64,
) -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (stream, _) = listener.accept().expect("one client");
        serve_session(stream, documents, encoding_name, report_ntp);
    });
    format!("rtsp://127.0.0.1:{port}/stream")
}

fn serve_session(
    mut stream: std::net::TcpStream,
    documents: Vec<Vec<u8>>,
    encoding_name: &'static str,
    report_ntp: u64,
) {
    use std::io::{Read, Write};

    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    // The interleaved channels the client asked for, per SETUP in order: video
    // then metadata.
    let mut channels: Vec<(u8, u8)> = Vec::new();

    loop {
        let read = stream.read(&mut chunk).unwrap_or(0);
        if read == 0 {
            return;
        }
        buffer.extend_from_slice(&chunk[..read]);
        while let Some(end) = find_header_end(&buffer) {
            let request = String::from_utf8_lossy(&buffer[..end]).to_string();
            buffer.drain(..end + 4);
            let cseq = header_value(&request, "cseq").unwrap_or_default();
            let method = request.split_whitespace().next().unwrap_or("").to_string();
            let uri = request
                .split_whitespace()
                .nth(1)
                .unwrap_or("rtsp://127.0.0.1/stream")
                .to_string();
            let response = match method.as_str() {
                "OPTIONS" => rtsp_response(
                    &cseq,
                    &[("Public", "OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN")],
                    b"",
                ),
                "DESCRIBE" => {
                    let sdp = sdp(encoding_name);
                    rtsp_response(
                        &cseq,
                        &[
                            ("Content-Type", "application/sdp"),
                            ("Content-Base", &format!("{uri}/")),
                        ],
                        sdp.as_bytes(),
                    )
                }
                "SETUP" => {
                    let transport = header_value(&request, "transport").unwrap_or_default();
                    let pair = interleaved_channels(&transport)
                        .expect("the client asks for TCP-interleaved transport");
                    channels.push(pair);
                    let transport = format!(
                        "RTP/AVP/TCP;unicast;interleaved={}-{};ssrc={METADATA_SSRC:08X}",
                        pair.0, pair.1,
                    );
                    rtsp_response(
                        &cseq,
                        &[
                            ("Transport", &transport),
                            ("Session", "12345678;timeout=60"),
                        ],
                        b"",
                    )
                }
                "PLAY" => rtsp_response(
                    &cseq,
                    &[
                        ("Session", "12345678"),
                        (
                            "RTP-Info",
                            &format!(
                                "url={uri}/streamid=0;seq=0;rtptime=0,\
                                 url={uri}/streamid=1;seq=0;rtptime=0"
                            ),
                        ),
                    ],
                    b"",
                ),
                _ => rtsp_response(&cseq, &[], b""),
            };
            if stream.write_all(&response).is_err() {
                return;
            }
            if method == "PLAY" {
                let (rtp_channel, rtcp_channel) =
                    *channels.get(1).expect("the metadata track was set up");
                stream_documents(
                    &mut stream,
                    &documents,
                    rtp_channel,
                    rtcp_channel,
                    report_ntp,
                );
                let _ = stream.flush();
                let _ = stream.shutdown(std::net::Shutdown::Both);
                return;
            }
        }
    }
}

fn sdp(encoding_name: &str) -> String {
    format!(
        "v=0\r\n\
         o=- 0 0 IN IP4 127.0.0.1\r\n\
         s=g2g onvif metadata test\r\n\
         c=IN IP4 0.0.0.0\r\n\
         t=0 0\r\n\
         m=video 0 RTP/AVP 96\r\n\
         a=rtpmap:96 H264/90000\r\n\
         a=control:streamid=0\r\n\
         m=application 0 RTP/AVP {METADATA_PAYLOAD_TYPE}\r\n\
         a=rtpmap:{METADATA_PAYLOAD_TYPE} {encoding_name}/{METADATA_CLOCK_HZ}\r\n\
         a=control:streamid=1\r\n\
         a=recvonly\r\n",
    )
}

/// Send each document as RTP on `rtp_channel`, with an RTCP sender report on
/// `rtcp_channel` after the first.
fn stream_documents(
    stream: &mut std::net::TcpStream,
    documents: &[Vec<u8>],
    rtp_channel: u8,
    rtcp_channel: u8,
    report_ntp: u64,
) {
    use std::io::Write;
    let mut sequence: u16 = 0;
    for (index, document) in documents.iter().enumerate() {
        let timestamp = REPORT_RTP_TIMESTAMP + (index as u32) * METADATA_CLOCK_HZ;
        // The first document is split in two packets, so the client has to
        // concatenate up to the marker bit rather than taking one packet as one
        // document.
        let split = if index == 0 { document.len() / 2 } else { 0 };
        if split > 0 {
            let packet = rtp_packet(sequence, timestamp, false, &document[..split]);
            sequence = sequence.wrapping_add(1);
            let _ = stream.write_all(&interleaved(rtp_channel, &packet));
        }
        let packet = rtp_packet(sequence, timestamp, true, &document[split..]);
        sequence = sequence.wrapping_add(1);
        let _ = stream.write_all(&interleaved(rtp_channel, &packet));

        if index == 0 {
            let report = g2g_plugins::rtcp::build_sender_report(
                METADATA_SSRC,
                report_ntp,
                REPORT_RTP_TIMESTAMP,
                sequence as u32,
                document.len() as u32,
                &[],
            );
            let _ = stream.write_all(&interleaved(rtcp_channel, &report));
        }
    }
}

fn rtp_packet(sequence: u16, timestamp: u32, marker: bool, payload: &[u8]) -> Vec<u8> {
    const VERSION_2: u8 = 0x80;
    const MARKER_BIT: u8 = 0x80;
    let mut packet = Vec::with_capacity(12 + payload.len());
    packet.push(VERSION_2);
    packet.push(if marker {
        METADATA_PAYLOAD_TYPE | MARKER_BIT
    } else {
        METADATA_PAYLOAD_TYPE
    });
    packet.extend_from_slice(&sequence.to_be_bytes());
    packet.extend_from_slice(&timestamp.to_be_bytes());
    packet.extend_from_slice(&METADATA_SSRC.to_be_bytes());
    packet.extend_from_slice(payload);
    packet
}

/// RFC 2326 section 10.12 framing: `$`, the channel, a 16-bit length.
fn interleaved(channel: u8, packet: &[u8]) -> Vec<u8> {
    let mut framed = Vec::with_capacity(4 + packet.len());
    framed.push(b'$');
    framed.push(channel);
    framed.extend_from_slice(&(packet.len() as u16).to_be_bytes());
    framed.extend_from_slice(packet);
    framed
}

/// The inverse of the source's own NTP reader, so the served report names the
/// instant the test expects back.
fn unix_nanos_to_ntp(unix_nanos: i64) -> u64 {
    let secs = (unix_nanos / 1_000_000_000) as u64 + g2g_core::NTP_TO_UNIX_EPOCH_SECS;
    let frac = (((unix_nanos % 1_000_000_000) as u64) << 32) / 1_000_000_000;
    (secs << 32) | frac
}

fn rtsp_response(cseq: &str, headers: &[(&str, &str)], body: &[u8]) -> Vec<u8> {
    let mut text = format!("RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nServer: g2g-test\r\n");
    for (name, value) in headers {
        text.push_str(&format!("{name}: {value}\r\n"));
    }
    if !body.is_empty() {
        text.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    text.push_str("\r\n");
    let mut bytes = text.into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

fn find_header_end(buffer: &[u8]) -> Option<usize> {
    buffer.windows(4).position(|w| w == b"\r\n\r\n")
}

fn header_value(request: &str, name: &str) -> Option<String> {
    request.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().to_string())
    })
}

fn interleaved_channels(transport: &str) -> Option<(u8, u8)> {
    let range = transport.split("interleaved=").nth(1)?.split(';').next()?;
    let (rtp, rtcp) = range.trim().split_once('-')?;
    Some((rtp.parse().ok()?, rtcp.parse().ok()?))
}

/// Every packet each output pad received.
#[derive(Default)]
struct RecordingFanout {
    ports: Vec<Vec<PipelinePacket>>,
}

impl RecordingFanout {
    fn with_ports(ports: usize) -> Self {
        Self {
            ports: (0..ports).map(|_| Vec::new()).collect(),
        }
    }
}

impl g2g_core::MultiOutputSink for RecordingFanout {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        port: usize,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet.take().expect("poll_push_to without a packet");
        self.ports[port].push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        self.ports.len()
    }
}

/// Play a served metadata track and return what the metadata pad emitted.
async fn play_metadata_track(documents: Vec<Vec<u8>>, encoding_name: &'static str) -> Vec<Frame> {
    play_metadata_track_with_report(
        documents,
        encoding_name,
        unix_nanos_to_ntp(SERVER_REPORT_NANOS),
    )
    .await
}

async fn play_metadata_track_with_report(
    documents: Vec<Vec<u8>>,
    encoding_name: &'static str,
    report_ntp: u64,
) -> Vec<Frame> {
    use g2g_core::MultiOutputSource;
    use g2g_plugins::rtspsrcn::RtspSrcN;

    let url = serve_onvif_metadata(documents, encoding_name, report_ntp);
    let mut src = RtspSrcN::new(url).with_outputs(2);
    src.set_property("onvif-metadata", PropValue::Bool(true))
        .expect("onvif-metadata is a declared property");
    assert_eq!(src.output_count(), 2, "video then metadata");
    assert_eq!(src.output_caps(1), Ok(Caps::OnvifMetadata));

    let mut sink = RecordingFanout::with_ports(2);
    src.run(&mut sink)
        .await
        .expect("the session plays and ends");
    sink.ports[1]
        .drain(..)
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some(f),
            _ => None,
        })
        .collect()
}

/// The documents the loopback tests stream: three one-frame scene descriptions
/// built from the specification's example geometry, so each arrives as its own
/// `tt:Frame`.
fn streamed_documents() -> Vec<Vec<u8>> {
    (0..3u32)
        .map(|index| {
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
                 <tt:MetadataStream xmlns:tt=\"http://www.onvif.org/ver10/schema\">\
                 <tt:VideoAnalytics>\
                 <tt:Frame UtcTime=\"2008-10-10T12:24:57.321\">\
                 <tt:Transformation>\
                 <tt:Translate x=\"-1.0\" y=\"-1.0\"/>\
                 <tt:Scale x=\"0.003125\" y=\"0.00416667\"/>\
                 </tt:Transformation>\
                 <tt:Object ObjectId=\"{index}\"><tt:Appearance><tt:Shape>\
                 <tt:BoundingBox left=\"20.0\" top=\"80.0\" right=\"100.0\" bottom=\"30.0\"/>\
                 </tt:Shape></tt:Appearance></tt:Object>\
                 </tt:Frame></tt:VideoAnalytics></tt:MetadataStream>",
            )
            .into_bytes()
        })
        .collect()
}

#[tokio::test]
async fn an_uncompressed_metadata_track_arrives_whole() {
    let documents = streamed_documents();
    let frames = play_metadata_track(documents.clone(), ONVIF_METADATA_PLAIN).await;
    assert_eq!(frames.len(), documents.len());
    for (frame, document) in frames.iter().zip(&documents) {
        // The first document was split across two RTP packets, so this also
        // says the payloads were concatenated to the marker bit.
        assert_eq!(
            frame.domain.as_system_slice().expect("system memory"),
            document.as_slice(),
        );
    }
    // The documents are one second of the metadata clock apart, and the play
    // timeline starts at the first one.
    assert_eq!(
        frames.iter().map(|f| f.timing.pts_ns).collect::<Vec<_>>(),
        [0, 1_000_000_000, 2_000_000_000],
    );
    // The sender report is served after the first document, so only the ones
    // after it can name the sender's wall clock.
    assert_eq!(frames[0].meta.get::<WallClockMeta>(), None);
    assert_wall_clock(&frames[1], SERVER_REPORT_NANOS + 1_000_000_000);
    assert_wall_clock(&frames[2], SERVER_REPORT_NANOS + 2_000_000_000);
}

/// A sender report names its instant as a 32-bit binary fraction of a second,
/// which is about a quarter of a nanosecond, so the wall clock read back off a
/// frame is exact to one.
fn assert_wall_clock(frame: &Frame, want: i64) {
    let got = frame
        .meta
        .get::<WallClockMeta>()
        .map(|w| w.unix_nanos)
        .expect("the sender report reached this frame");
    assert!((got - want).abs() <= 1, "got {got}, want {want}");
}

/// RFC 3550 lets a sender with no wall clock report NTP zero, which names no
/// instant, so the frames after it stay unstamped rather than dated 1900.
#[tokio::test]
async fn a_sender_report_with_no_wall_clock_stamps_nothing() {
    let documents = streamed_documents();
    let frames = play_metadata_track_with_report(documents.clone(), ONVIF_METADATA_PLAIN, 0).await;
    assert_eq!(frames.len(), documents.len());
    for frame in &frames {
        assert_eq!(frame.meta.get::<WallClockMeta>(), None);
    }
}

/// The gzip half, against a member a different implementation produced: the
/// checked-in `.gz` fixture is python's `gzip.compress` over the plain fixture,
/// so this is not the inflater reading its own deflater's output.
#[tokio::test]
async fn a_gzip_metadata_track_is_inflated() {
    let plain = fixture("onvif_transformed_frames.xml");
    let compressed = fixture("onvif_transformed_frames.xml.gz");
    assert!(compressed.len() < plain.len(), "the fixture is compressed");

    let frames = play_metadata_track(vec![compressed], ONVIF_METADATA_GZIP).await;
    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].domain.as_system_slice().expect("system memory"),
        plain.as_slice(),
        "the pad emits the inflated document",
    );
    // And the inflated document is the one the parser reads.
    let parsed = parse_metadata_documents(frames[0].domain.as_system_slice().unwrap());
    assert_eq!(parsed.len(), 5);
    assert_box(&only_detection(&parsed[0].analytics).bbox, &SPEC_BOX);
}
