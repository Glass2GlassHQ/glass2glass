//! M1195: the `tt:Event` part of an ONVIF metadata document, parsed into
//! `OnvifEventMeta` and carried onto the video by the combiner.
//!
//! The fixtures are real event documents: a Hikvision-style motion alarm
//! captured from a TRENDnet camera, an Axis PIR Initialized / Changed pair from
//! the Kane610/axis library's captures, and the ONVIF Core Specification's
//! keyed property example. Each fixture's comment names its source. The
//! expected values are read out of the fixture XML by local name, not retyped.
//!
//! ```sh
//! cargo test -p g2g-plugins --features onvif --test m1195_onvif_events
//! ```

#![cfg(feature = "onvif")]

use std::path::PathBuf;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::meta::{AnalyticsMeta, WallClockMeta};
use g2g_core::{Caps, G2gError, MultiInputElement, OutputSink, PushOutcome, VideoCodec};
use g2g_plugins::onvifmetadata::{
    parse_metadata_documents, parse_utc_time, OnvifEventMessage, OnvifEventMeta,
    OnvifMetadataCombiner, OnvifMetadataParse, PropertyOperation, SimpleItem,
    MAX_EVENT_MESSAGES_PER_DOCUMENT, MAX_EVENT_TEXT_BYTES, MAX_ITEMS_PER_MESSAGE,
};

const HIKVISION_MOTION: &str = "onvif_event_hikvision_motion.xml";
const AXIS_PIR_INITIALIZED: &str = "onvif_event_axis_pir_initialized.xml";
const AXIS_PIR_CHANGED: &str = "onvif_event_axis_pir_changed.xml";
const FIELD_DETECTOR_KEY: &str = "onvif_event_field_detector_key.xml";
/// Analytics frames in one root and an event in the other (M1151's fixture).
const FRAMES_AND_EVENT: &str = "onvif_two_roots.xml";

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn fixture_text(name: &str) -> String {
    String::from_utf8(fixture(name)).expect("fixtures are UTF-8")
}

// ---- expectations read off the fixture ----

/// A notification message as the fixture spells it, read by local name alone
/// (no namespace or structure checks), so it does not share the parser's rules.
#[derive(Debug)]
struct FixtureMessage {
    topic: Option<String>,
    utc_time: String,
    property_operation: Option<String>,
    source: Vec<(String, String)>,
    key: Vec<(String, String)>,
    data: Vec<(String, String)>,
}

fn fixture_messages(name: &str) -> Vec<FixtureMessage> {
    let text = fixture_text(name);
    // A multi-root fixture keeps its events in the last root.
    let last_root = &text[text.rfind("<?xml").unwrap_or(0)..];
    let document = roxmltree::Document::parse(last_root).expect("the fixture is well-formed");
    let named = |node: &roxmltree::Node, local: &str| node.tag_name().name() == local;
    document
        .descendants()
        .filter(|n| named(n, "NotificationMessage"))
        .map(|notification| {
            let message = notification
                .descendants()
                .find(|n| named(n, "Message") && n.has_attribute("UtcTime"))
                .expect("every fixture notification has a timed message");
            let items = |group: &str| {
                message
                    .children()
                    .find(|n| named(n, group))
                    .map(|list| {
                        list.children()
                            .filter(|n| named(n, "SimpleItem"))
                            .map(|item| {
                                (
                                    item.attribute("Name").unwrap().to_string(),
                                    item.attribute("Value").unwrap().to_string(),
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };
            FixtureMessage {
                topic: notification
                    .descendants()
                    .find(|n| named(n, "Topic"))
                    .and_then(|n| n.text())
                    .map(|t| t.trim().to_string()),
                utc_time: message.attribute("UtcTime").unwrap().to_string(),
                property_operation: message.attribute("PropertyOperation").map(String::from),
                source: items("Source"),
                key: items("Key"),
                data: items("Data"),
            }
        })
        .collect()
}

fn pairs(items: &[SimpleItem]) -> Vec<(String, String)> {
    items
        .iter()
        .map(|item| (item.name.clone(), item.value.clone()))
        .collect()
}

fn assert_message(parsed: &OnvifEventMessage, expected: &FixtureMessage) {
    assert_eq!(parsed.topic, expected.topic, "topic");
    assert_eq!(
        Some(parsed.unix_nanos),
        parse_utc_time(&expected.utc_time),
        "UtcTime {}",
        expected.utc_time,
    );
    assert_eq!(
        parsed.property_operation.map(PropertyOperation::as_str),
        expected.property_operation.as_deref(),
        "PropertyOperation",
    );
    assert_eq!(pairs(&parsed.source), expected.source, "Source items");
    assert_eq!(pairs(&parsed.key), expected.key, "Key items");
    assert_eq!(pairs(&parsed.data), expected.data, "Data items");
}

fn assert_messages(parsed: &[OnvifEventMessage], fixture_name: &str) {
    let expected = fixture_messages(fixture_name);
    assert!(!expected.is_empty(), "{fixture_name} holds a message");
    assert_eq!(parsed.len(), expected.len(), "one message per notification");
    for (parsed, expected) in parsed.iter().zip(&expected) {
        assert_message(parsed, expected);
    }
}

// ---- document parsing ----

#[test]
fn a_hikvision_motion_alarm_parses() {
    let parsed = parse_metadata_documents(&fixture(HIKVISION_MOTION));
    assert!(parsed.frames.is_empty(), "an event-only document");
    assert_messages(&parsed.events, HIKVISION_MOTION);
    assert_eq!(
        parsed.events[0].property_operation,
        Some(PropertyOperation::Changed)
    );
}

#[test]
fn the_core_spec_property_lifecycle_keeps_its_key_items() {
    let parsed = parse_metadata_documents(&fixture(FIELD_DETECTOR_KEY));
    assert_messages(&parsed.events, FIELD_DETECTOR_KEY);
    assert_eq!(
        parsed
            .events
            .iter()
            .map(|m| m.property_operation)
            .collect::<Vec<_>>(),
        [
            Some(PropertyOperation::Initialized),
            Some(PropertyOperation::Changed),
            Some(PropertyOperation::Deleted),
        ],
    );
    // The key names the object, so every message about it carries the same one.
    assert!(!parsed.events[0].key.is_empty());
    assert!(parsed.events.iter().all(|m| m.key == parsed.events[0].key));
    // A deleted property has no state left to report.
    assert!(parsed.events[2].data.is_empty());
}

#[test]
fn a_message_breaking_the_schema_is_refused_alone() {
    let expected = fixture_messages(FIELD_DETECTOR_KEY);
    let text = fixture_text(FIELD_DETECTOR_KEY);
    // Break the first message's time and the second's operation, leave the third.
    let broken = text
        .replacen(
            &format!("UtcTime=\"{}\"", expected[0].utc_time),
            "UtcTime=\"yesterday\"",
            1,
        )
        .replacen(
            &format!(
                "PropertyOperation=\"{}\"",
                expected[1].property_operation.as_deref().unwrap()
            ),
            "PropertyOperation=\"Toggled\"",
            1,
        );
    let parsed = parse_metadata_documents(broken.as_bytes());
    assert_eq!(parsed.events.len(), 1);
    assert_message(&parsed.events[0], &expected[2]);
}

#[test]
fn a_truncated_event_document_fails_the_parse() {
    let whole = fixture(HIKVISION_MOTION);
    assert!(!parse_metadata_documents(&whole).events.is_empty());
    // Half way, and with every notification whole but the document unclosed.
    let event_close = String::from_utf8_lossy(&whole).find("</tt:Event>").unwrap();
    for cut in [whole.len() / 2, event_close] {
        let parsed = parse_metadata_documents(&whole[..cut]);
        assert!(
            parsed.events.is_empty() && parsed.frames.is_empty(),
            "cut at {cut}"
        );
    }
}

/// An event-only document holding `count` copies of the Hikvision alarm.
fn repeated_notifications(count: usize) -> String {
    let text = fixture_text(HIKVISION_MOTION);
    let start = text.find("<wsnt:NotificationMessage>").unwrap();
    let end_tag = "</wsnt:NotificationMessage>";
    let end = text.find(end_tag).unwrap() + end_tag.len();
    let notification = &text[start..end];
    format!(
        "{}{}{}",
        &text[..start],
        notification.repeat(count),
        &text[end..]
    )
}

#[test]
fn the_message_count_bound_holds() {
    let over = MAX_EVENT_MESSAGES_PER_DOCUMENT + 10;
    let parsed = parse_metadata_documents(repeated_notifications(over).as_bytes());
    assert_eq!(parsed.events.len(), MAX_EVENT_MESSAGES_PER_DOCUMENT);
}

#[test]
fn a_message_past_the_item_bound_is_refused() {
    let expected = fixture_messages(HIKVISION_MOTION);
    let items_in_fixture =
        expected[0].source.len() + expected[0].key.len() + expected[0].data.len();
    let text = fixture_text(HIKVISION_MOTION);
    let data_open = "<tt:Data>";
    let with_extra_items = |extra: usize| {
        let items: String = (0..extra)
            .map(|index| format!("<tt:SimpleItem Name=\"Extra{index}\" Value=\"0\"/>"))
            .collect();
        text.replacen(data_open, &format!("{data_open}{items}"), 1)
    };

    let at_bound = with_extra_items(MAX_ITEMS_PER_MESSAGE - items_in_fixture);
    let parsed = parse_metadata_documents(at_bound.as_bytes());
    assert_eq!(parsed.events.len(), 1);
    let message = &parsed.events[0];
    assert_eq!(
        message.source.len() + message.key.len() + message.data.len(),
        MAX_ITEMS_PER_MESSAGE
    );

    let past_bound = with_extra_items(MAX_ITEMS_PER_MESSAGE - items_in_fixture + 1);
    assert!(parse_metadata_documents(past_bound.as_bytes())
        .events
        .is_empty());
}

#[test]
fn a_message_with_an_overlong_text_is_refused() {
    let expected = fixture_messages(HIKVISION_MOTION);
    let (_, value) = &expected[0].data[0];
    let text = fixture_text(HIKVISION_MOTION);
    let with_value = |length: usize| {
        text.replacen(
            &format!("Value=\"{value}\""),
            &format!("Value=\"{}\"", "v".repeat(length)),
            1,
        )
    };
    let at_bound = parse_metadata_documents(with_value(MAX_EVENT_TEXT_BYTES).as_bytes());
    assert_eq!(at_bound.events[0].data[0].value.len(), MAX_EVENT_TEXT_BYTES);
    assert!(
        parse_metadata_documents(with_value(MAX_EVENT_TEXT_BYTES + 1).as_bytes())
            .events
            .is_empty()
    );

    let topic = expected[0].topic.as_deref().unwrap();
    let long_topic = text.replacen(topic, &"t".repeat(MAX_EVENT_TEXT_BYTES + 1), 1);
    assert!(parse_metadata_documents(long_topic.as_bytes())
        .events
        .is_empty());
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

/// A metadata document as `RtspSrcN` hands it over, with the sender's wall
/// clock when a sender report has arrived.
fn metadata_packet(document: Vec<u8>, pts_ns: u64, wall_nanos: Option<i64>) -> PipelinePacket {
    let mut frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(document.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            ..Default::default()
        },
        0,
    );
    if let Some(unix_nanos) = wall_nanos {
        frame.meta.attach(WallClockMeta { unix_nanos });
    }
    PipelinePacket::DataFrame(frame)
}

fn parse_element() -> OnvifMetadataParse {
    let mut parse = OnvifMetadataParse::new();
    parse
        .configure_pipeline(&Caps::OnvifMetadata)
        .expect("takes ONVIF metadata caps");
    parse
}

fn events_of(frame: &Frame) -> &[OnvifEventMessage] {
    &frame
        .meta
        .get::<OnvifEventMeta>()
        .expect("an event frame")
        .messages
}

#[tokio::test]
async fn the_parse_element_reads_an_axis_initialized_changed_pair() {
    const PTS_STEP_NS: u64 = 2_000_000_000;
    // How long after the event the document reached the sender's clock, so the
    // frame's wall clock and the message's UtcTime differ.
    const DELIVERY_DELAY_NS: i64 = 40_000_000;
    let mut parse = parse_element();
    let mut sink = RecordingSink::default();
    let documents = [
        (AXIS_PIR_INITIALIZED, PropertyOperation::Initialized),
        (AXIS_PIR_CHANGED, PropertyOperation::Changed),
    ];
    for (index, (name, operation)) in documents.into_iter().enumerate() {
        let expected = fixture_messages(name);
        let wall_nanos = parse_utc_time(&expected[0].utc_time).map(|t| t + DELIVERY_DELAY_NS);
        let pts_ns = PTS_STEP_NS * index as u64;
        let document = fixture(name);
        parse
            .process(
                metadata_packet(document.clone(), pts_ns, wall_nanos),
                &mut sink,
            )
            .await
            .expect("a well-formed document parses");
        let frames = sink.take_frames();
        assert_eq!(frames.len(), 1, "one event frame for {name}");
        let frame = &frames[0];
        assert_messages(events_of(frame), name);
        assert_eq!(events_of(frame)[0].property_operation, Some(operation));
        assert!(frame.meta.get::<AnalyticsMeta>().is_none());
        assert_eq!(frame.timing.pts_ns, pts_ns);
        assert_eq!(
            frame.meta.get::<WallClockMeta>().map(|w| w.unix_nanos),
            wall_nanos,
            "the event frame keeps the document's wall clock",
        );
        assert_eq!(
            frame.domain.as_system_slice().expect("system memory"),
            document.as_slice(),
        );
    }
}

#[tokio::test]
async fn frames_and_events_in_one_payload_both_come_out() {
    const PTS_NS: u64 = 500_000_000;
    let mut parse = parse_element();
    let mut sink = RecordingSink::default();
    parse
        .process(
            metadata_packet(fixture(FRAMES_AND_EVENT), PTS_NS, None),
            &mut sink,
        )
        .await
        .expect("a well-formed payload parses");
    let frames = sink.take_frames();
    let analytics_frames = parse_metadata_documents(&fixture(FRAMES_AND_EVENT))
        .frames
        .len();
    assert_eq!(frames.len(), analytics_frames + 1);
    assert!(frames[..analytics_frames].iter().all(
        |f| f.meta.get::<AnalyticsMeta>().is_some() && f.meta.get::<OnvifEventMeta>().is_none()
    ));
    let event_frame = &frames[analytics_frames];
    assert_messages(events_of(event_frame), FRAMES_AND_EVENT);
    assert_eq!(event_frame.timing.pts_ns, PTS_NS);
    assert!(event_frame.meta.get::<WallClockMeta>().is_none());
    assert_eq!(
        frames.iter().map(|f| f.sequence).collect::<Vec<_>>(),
        (0..frames.len() as u64).collect::<Vec<_>>(),
    );
}

#[tokio::test]
async fn the_parse_element_drops_a_truncated_event_document() {
    let whole = fixture(HIKVISION_MOTION);
    let mut parse = parse_element();
    let mut sink = RecordingSink::default();
    parse
        .process(
            metadata_packet(whole[..whole.len() / 2].to_vec(), 0, None),
            &mut sink,
        )
        .await
        .expect("a malformed document is dropped, not an error");
    assert!(sink.take_frames().is_empty());
}

// ---- onvifmetadatacombiner ----

/// 30 fps, the cadence the video steps at.
const FRAME_NS: u64 = 33_333_333;

fn combiner() -> OnvifMetadataCombiner {
    let video_caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: g2g_core::Dim::Any,
        height: g2g_core::Dim::Any,
        framerate: g2g_core::Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    };
    let mut combiner = OnvifMetadataCombiner::new();
    combiner
        .configure_pipeline(OnvifMetadataCombiner::VIDEO, &video_caps)
        .expect("takes the video pad's caps");
    combiner
        .configure_pipeline(OnvifMetadataCombiner::METADATA, &Caps::OnvifMetadata)
        .expect("takes the metadata pad's caps");
    combiner
}

fn video_frame(pts_ns: u64, wall_nanos: i64) -> PipelinePacket {
    let mut frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8]))),
        FrameTiming {
            pts_ns,
            duration_ns: FRAME_NS,
            ..Default::default()
        },
        0,
    );
    frame.meta.attach(WallClockMeta {
        unix_nanos: wall_nanos,
    });
    PipelinePacket::DataFrame(frame)
}

#[tokio::test]
async fn the_combiner_carries_events_onto_the_video_frame_of_their_instant() {
    const VIDEO_FRAMES: u64 = 4;
    const EVENT_FRAME_INDEX: u64 = 2;
    let expected = fixture_messages(HIKVISION_MOTION);
    let start_nanos = parse_utc_time(&expected[0].utc_time).unwrap();
    let wall_of = |index: u64| start_nanos + (FRAME_NS * index) as i64;

    let mut parse = parse_element();
    let mut parsed = RecordingSink::default();
    parse
        .process(
            metadata_packet(
                fixture(HIKVISION_MOTION),
                FRAME_NS * EVENT_FRAME_INDEX,
                Some(wall_of(EVENT_FRAME_INDEX)),
            ),
            &mut parsed,
        )
        .await
        .expect("the alarm parses");
    let event_frame = parsed.take_frames().pop().expect("one event frame");

    let mut combiner = combiner();
    let mut sink = RecordingSink::default();
    combiner
        .process(
            OnvifMetadataCombiner::METADATA,
            PipelinePacket::DataFrame(event_frame),
            &mut sink,
        )
        .await
        .unwrap();
    for index in 0..VIDEO_FRAMES {
        combiner
            .process(
                OnvifMetadataCombiner::VIDEO,
                video_frame(FRAME_NS * index, wall_of(index)),
                &mut sink,
            )
            .await
            .unwrap();
    }
    combiner
        .process(OnvifMetadataCombiner::VIDEO, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();

    let video = sink.take_frames();
    assert_eq!(video.len(), VIDEO_FRAMES as usize);
    for (index, frame) in video.iter().enumerate() {
        let events = frame.meta.get::<OnvifEventMeta>();
        if index as u64 == EVENT_FRAME_INDEX {
            assert_messages(
                &events.expect("the alarm lands here").messages,
                HIKVISION_MOTION,
            );
        } else {
            assert!(events.is_none(), "frame {index} carries no event");
        }
    }
}
