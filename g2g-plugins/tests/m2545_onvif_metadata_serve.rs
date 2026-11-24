// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "onvif", feature = "rtsp", feature = "rtsp-server"))]

#[path = "common/annexb.rs"]
mod annexb;
#[path = "common/collect_sink.rs"]
mod collect_sink;

use core::time::Duration;
use std::net::TcpListener as StdTcpListener;
use std::sync::atomic::{AtomicBool, Ordering};

use annexb::access_units;
use collect_sink::CollectSink;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::meta::{
    AnalyticsMeta, AnalyticsNode, BBox, ObjectDetection, RelationKind, Tracking, WallClockMeta,
};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, MultiInputElement, MultiOutputElement, MultiOutputSink,
    MultiOutputSource, PropValue, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::onvifmetadata::{
    parse_metadata_documents, write_metadata_document, OnvifMetadataCombiner,
    OnvifMetadataExtractor, OnvifMetadataParse, RelationMeta2OnvifMeta,
};
use g2g_plugins::rtspserversink::RtspServerSink;
use g2g_plugins::rtspsrcn::RtspSrcN;

const FIXTURE: &[u8] = include_bytes!("fixtures/h264_64x48_no_vui_timing.h264");
// 30 fps truncated to whole nanoseconds, which is not a whole number of 90 kHz ticks
const FRAME_PERIOD: Duration = Duration::from_nanos(1_000_000_000 / 30);
const FRAMES: u64 = 60;
const RUN_LIMIT: Duration = Duration::from_secs(30);
const SSRC: u32 = 0x2545_0000;
const PAYLOAD_TYPE: u8 = 96;
const VIDEO_TRACK: usize = 0;
const METADATA_TRACK: usize = 1;
// f32 coordinates through decimal text and back
const COORDINATE_SLACK: f32 = 1e-6;

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}

fn tracked_detection(analytics: &mut AnalyticsMeta, bbox: BBox, label: u32, object_id: u64) {
    let detection = analytics.add_detection(ObjectDetection {
        bbox,
        label,
        confidence: 0.75,
    });
    let tracking = analytics.push(AnalyticsNode::Tracking(Tracking { object_id }));
    analytics.relate(detection, tracking, RelationKind::Tracks);
}

fn tracked_ids(analytics: &AnalyticsMeta) -> Vec<u64> {
    analytics
        .nodes
        .iter()
        .filter_map(|node| match node {
            AnalyticsNode::Tracking(tracking) => Some(tracking.object_id),
            _ => None,
        })
        .collect()
}

#[test]
fn a_written_document_reads_back_as_the_same_detections() {
    const UNIX_NANOS: i64 = 1_700_000_000_123_456_789;
    const TRACKED_ID: u64 = 42;
    let top_left = BBox {
        x: 0.125,
        y: 0.0625,
        w: 0.25,
        h: 0.5,
    };
    let bottom_right = BBox {
        x: 0.5,
        y: 0.75,
        w: 0.375,
        h: 0.125,
    };
    let mut analytics = AnalyticsMeta::new();
    analytics.set_class_names(["deer", "fox & cub"]);
    tracked_detection(&mut analytics, top_left, 1, TRACKED_ID);
    let untracked = analytics.add_detection(ObjectDetection {
        bbox: bottom_right,
        label: 0,
        confidence: 0.5,
    });

    let document = write_metadata_document(&analytics, UNIX_NANOS);
    let frames = parse_metadata_documents(document.as_bytes()).frames;

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].unix_nanos, Some(UNIX_NANOS));
    let read = &frames[0].analytics;
    let detections: Vec<&ObjectDetection> = read.detections().collect();
    assert_eq!(detections.len(), 2);
    for (got, want) in detections.iter().zip([top_left, bottom_right]) {
        for (g, w) in [
            (got.bbox.x, want.x),
            (got.bbox.y, want.y),
            (got.bbox.w, want.w),
            (got.bbox.h, want.h),
        ] {
            assert!(
                (g - w).abs() < COORDINATE_SLACK,
                "got {:?}, want {want:?}",
                got.bbox
            );
        }
    }
    assert_eq!(read.class_name(detections[0].label), Some("fox & cub"));
    assert_eq!(read.class_name(detections[1].label), Some("deer"));
    assert_eq!(detections[0].confidence, 0.75);
    // an untracked detection is named by its node index in the frame
    assert_eq!(tracked_ids(read), [TRACKED_ID, untracked as u64]);
}

struct RecordingFanout {
    ports: Vec<Vec<PipelinePacket>>,
}

impl RecordingFanout {
    fn with_ports(ports: usize) -> Self {
        Self {
            ports: (0..ports).map(|_| Vec::new()).collect(),
        }
    }

    fn frames(&mut self, port: usize) -> Vec<Frame> {
        self.ports[port]
            .drain(..)
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some(frame),
                _ => None,
            })
            .collect()
    }
}

impl MultiOutputSink for RecordingFanout {
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

fn described_frame(index: u64, access_unit: &[u8]) -> Frame {
    let mut frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(
            access_unit.to_vec().into_boxed_slice(),
        )),
        FrameTiming {
            pts_ns: index * FRAME_PERIOD.as_nanos() as u64,
            ..FrameTiming::default()
        },
        index,
    );
    let mut analytics = AnalyticsMeta::new();
    let bbox = BBox {
        x: 0.25,
        y: 0.25,
        w: 0.5,
        h: 0.5,
    };
    tracked_detection(&mut analytics, bbox, 0, index);
    frame.meta.attach(analytics);
    frame
}

async fn serve(server: &mut RtspServerSink, client_done: &AtomicBool) -> Vec<i64> {
    let mut writer = RelationMeta2OnvifMeta::new();
    let mut extractor = OnvifMetadataExtractor::new(2);
    let mut written = CollectSink::default();
    let mut extracted = RecordingFanout::with_ports(2);
    let mut served = CollectSink::default();
    let mut written_times = Vec::new();
    for (index, access_unit) in access_units(FIXTURE).iter().cycle().enumerate() {
        if client_done.load(Ordering::Relaxed) {
            break;
        }
        let frame = described_frame(index as u64, access_unit);
        AsyncElement::process(&mut writer, PipelinePacket::DataFrame(frame), &mut written)
            .await
            .expect("the writer describes the frame");
        for frame in written.packets.drain(..) {
            MultiOutputElement::process(&mut extractor, frame, &mut extracted)
                .await
                .expect("the extractor splits the frame");
        }
        for (track, frame) in [VIDEO_TRACK, METADATA_TRACK]
            .into_iter()
            .flat_map(|track| extracted.frames(track).into_iter().map(move |f| (track, f)))
        {
            if track == METADATA_TRACK {
                let document = frame.domain.as_system_slice().expect("system memory");
                let parsed = parse_metadata_documents(document).frames;
                written_times.push(parsed[0].unix_nanos.expect("the writer sets UtcTime"));
            }
            MultiInputElement::process(
                server,
                track,
                PipelinePacket::DataFrame(frame),
                &mut served,
            )
            .await
            .expect("the server sends the frame");
        }
        tokio::time::sleep(FRAME_PERIOD).await;
    }
    written_times
}

async fn combine(video: Vec<Frame>, metadata: Vec<Frame>) -> Vec<Frame> {
    let mut combiner = OnvifMetadataCombiner::new();
    combiner
        .configure_pipeline(OnvifMetadataCombiner::VIDEO, &h264_caps())
        .expect("takes the video pad's caps");
    combiner
        .configure_pipeline(OnvifMetadataCombiner::METADATA, &Caps::OnvifMetadata)
        .expect("takes the metadata pad's caps");
    // each track's play timeline starts at its own first packet, so arrival order is wall-clock order
    let mut merged: Vec<(usize, Frame)> = metadata
        .into_iter()
        .map(|frame| (OnvifMetadataCombiner::METADATA, frame))
        .chain(
            video
                .into_iter()
                .map(|frame| (OnvifMetadataCombiner::VIDEO, frame)),
        )
        .collect();
    merged.sort_by_key(|(_, frame)| {
        frame
            .meta
            .get::<WallClockMeta>()
            .map_or(i64::MIN, |wall_clock| wall_clock.unix_nanos)
    });
    let mut combined = CollectSink::default();
    for (pad, frame) in merged {
        combiner
            .process(pad, PipelinePacket::DataFrame(frame), &mut combined)
            .await
            .expect("the combiner takes the frame");
    }
    combiner
        .process(
            OnvifMetadataCombiner::VIDEO,
            PipelinePacket::Eos,
            &mut combined,
        )
        .await
        .expect("EOS flushes what is held");
    combined
        .packets
        .into_iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect()
}

async fn parse(documents: Vec<Frame>) -> Vec<Frame> {
    let mut parse = OnvifMetadataParse::new();
    parse
        .configure_pipeline(&Caps::OnvifMetadata)
        .expect("takes ONVIF metadata caps");
    let mut parsed = CollectSink::default();
    for document in documents {
        parse
            .process(PipelinePacket::DataFrame(document), &mut parsed)
            .await
            .expect("a written document parses");
    }
    parsed
        .packets
        .into_iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn each_received_picture_carries_its_own_frames_detection() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp control");
    let port = listener.local_addr().unwrap().port();
    let mut server = RtspServerSink::from_listener(listener)
        .unwrap()
        .with_tracks(2)
        .with_rtp(PAYLOAD_TYPE, SSRC)
        .with_rtcp_sr_interval(Duration::from_millis(1));
    MultiInputElement::configure_pipeline(&mut server, VIDEO_TRACK, &h264_caps())
        .expect("the first track serves H.264");
    MultiInputElement::configure_pipeline(&mut server, METADATA_TRACK, &Caps::OnvifMetadata)
        .expect("the second track serves ONVIF metadata");

    let client_done = AtomicBool::new(false);
    let mut src = RtspSrcN::new(format!("rtsp://127.0.0.1:{port}/stream"))
        .with_outputs(2)
        .with_frame_limit(FRAMES);
    src.set_property("onvif-metadata", PropValue::Bool(true))
        .expect("onvif-metadata is a declared property");
    let mut received = RecordingFanout::with_ports(2);
    let play = async {
        let result = tokio::time::timeout(RUN_LIMIT, src.run(&mut received)).await;
        client_done.store(true, Ordering::Relaxed);
        result
    };
    let (played, written_times) = tokio::join!(play, serve(&mut server, &client_done));
    played
        .expect("the session ends within the run limit")
        .expect("the session plays");

    let video = received.frames(VIDEO_TRACK);
    let documents = received.frames(METADATA_TRACK);
    assert!(!documents.is_empty(), "the metadata track arrived");
    let parsed = parse(documents).await;
    let described: Vec<u64> = parsed
        .iter()
        .filter_map(|frame| frame.meta.get::<AnalyticsMeta>())
        .flat_map(tracked_ids)
        .collect();
    let pictures = combine(video, parsed).await;

    // the served frame whose document time is nearest the wall clock the player rebuilt
    let half_period_ns = FRAME_PERIOD.as_nanos() as i64 / 2;
    let served_index = |picture: &Frame| -> Option<u64> {
        let rebuilt = picture.meta.get::<WallClockMeta>()?.unix_nanos;
        let index = written_times
            .iter()
            .position(|written| (written - rebuilt).abs() < half_period_ns)?;
        Some(index as u64)
    };
    let mut checked = 0;
    for picture in &pictures {
        // the frame limit ends the two tracks at different pictures
        let Some(index) = served_index(picture).filter(|index| described.contains(index)) else {
            continue;
        };
        let ids = picture
            .meta
            .get::<AnalyticsMeta>()
            .map(tracked_ids)
            .unwrap_or_default();
        assert_eq!(ids, [index], "picture {index} carries its own detection");
        checked += 1;
    }
    assert!(
        checked >= FRAMES as usize / 2,
        "only {checked} of {} pictures had a sender wall clock and a document",
        pictures.len()
    );
}

#[tokio::test]
async fn the_served_graph_negotiates_and_runs() {
    let clip = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/h264_64x48_no_vui_timing.h264"
    );
    let line = format!(
        "filesrc location={clip} ! h264parse ! relationmeta2onvifmeta \
         ! onvifmetadataextractor name=e \
         e.src ! s.sink_0 \
         e.meta_src ! s.sink_1 \
         rtspserversink name=s address=127.0.0.1 port=0"
    );
    let registry = g2g_plugins::registry::default_registry();
    let graph = g2g_core::runtime::parse_launch(&registry, &line)
        .unwrap_or_else(|e| panic!("{line}\n{e:?}"));
    tokio::time::timeout(
        RUN_LIMIT,
        g2g_core::runtime::run_graph(graph, &g2g_plugins::clock::WallClock::new(), 2),
    )
    .await
    .expect("the clip ends the run")
    .expect("the graph negotiates and runs");
}
