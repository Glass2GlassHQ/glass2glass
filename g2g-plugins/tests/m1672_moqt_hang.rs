#![cfg(feature = "moqt")]

use serde::Deserialize;

use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, VideoCodec};

use g2g_plugins::moqt::coding::Params;
use g2g_plugins::moqt::coding::Reader;
use g2g_plugins::moqt::v18::coding::MessageParams;
use g2g_plugins::moqt::v18::message::{publish_done_code, ControlMessage};
use g2g_plugins::moqtsrc::MoqtSrc;
use g2g_plugins::textencoding::base64_decode;

mod moqt_common;
use moqt_common::peer18::{accept_request, accept_v18, exchange_setup, serve_group};
use moqt_common::{bind_server, TestCert};

#[path = "common/collect_sink.rs"]
mod collect_sink;
use collect_sink::CollectSink;

const FIXTURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/hang");
const NAMESPACE: &str = "demo/fixture.hang";
const CATALOG_TRACK: &str = "catalog.json";
const VIDEO_TRACK: &str = "0.avc3";
const CATALOG_ALIAS: u64 = 1;
const VIDEO_ALIAS: u64 = 2;
const NANOSECONDS_PER_MICROSECOND: u64 = 1_000;

// one line of `moq fetch --json`, which recorded the fixture off a moq-relay
#[derive(Deserialize)]
struct RecordedFrame {
    group: u64,
    payload: String,
}

fn recorded_groups() -> Vec<(u64, Vec<Vec<u8>>)> {
    let text = std::fs::read_to_string(format!("{FIXTURE_DIR}/{VIDEO_TRACK}.jsonl")).unwrap();
    let mut groups: Vec<(u64, Vec<Vec<u8>>)> = Vec::new();
    for line in text.lines() {
        let frame: RecordedFrame = serde_json::from_str(line).unwrap();
        let payload = base64_decode(&frame.payload).unwrap();
        match groups.last_mut() {
            Some((group, objects)) if *group == frame.group => objects.push(payload),
            _ => groups.push((frame.group, vec![payload])),
        }
    }
    groups
}

// what each recorded object holds after its microsecond timestamp
fn timestamp_and_codec_payload(object: &[u8]) -> (u64, &[u8]) {
    let mut reader = Reader::new(object);
    let timestamp_us = reader.varint().unwrap();
    (timestamp_us, reader.rest())
}

// the scripted relay: answers the catalog SUBSCRIBE, then the video one, and
// serves the recorded objects on draft-18 subgroup streams
async fn serve_recording(mut server: web_transport_quinn::Server) -> impl Sized {
    let catalog = std::fs::read(format!("{FIXTURE_DIR}/{CATALOG_TRACK}")).unwrap();
    let groups = recorded_groups();
    let session = accept_v18(&mut server).await;
    let control = exchange_setup(&session).await;
    let mut kept = Vec::new();
    for (track, alias) in [(CATALOG_TRACK, CATALOG_ALIAS), (VIDEO_TRACK, VIDEO_ALIAS)] {
        let (mut request, first) = accept_request(&session).await;
        match first {
            ControlMessage::Subscribe { track_name, .. } => {
                assert_eq!(track_name.as_str_lossy(), track)
            }
            other => panic!("expected SUBSCRIBE, got {}", other.name()),
        }
        request
            .send(ControlMessage::SubscribeOk {
                track_alias: alias,
                params: MessageParams::new(),
                properties: Params::new(),
            })
            .await;
        if track == CATALOG_TRACK {
            serve_group(&session, alias, 0, std::slice::from_ref(&catalog)).await;
        } else {
            for (group, objects) in &groups {
                serve_group(&session, alias, *group, objects).await;
            }
            request
                .send(ControlMessage::PublishDone {
                    status_code: publish_done_code::TRACK_ENDED,
                    stream_count: groups.len() as u64,
                    reason: String::from("end of recording"),
                })
                .await;
        }
        kept.push(request);
    }
    (session, control, kept)
}

#[tokio::test]
async fn a_hang_broadcast_plays_as_annexb_access_units_with_their_timestamps() {
    let tls = TestCert::generate();
    let (server, port) = bind_server(&tls);
    let mut src = MoqtSrc::new(format!("https://127.0.0.1:{port}/"), NAMESPACE)
        .with_server_certificate_hashes(&tls.hash_hex);

    let play = async {
        let caps = src
            .intercept_caps()
            .await
            .expect("the catalog decides the caps");
        assert!(matches!(
            caps,
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        ));
        src.configure_pipeline(&caps.fixate().unwrap()).unwrap();
        let mut sink = CollectSink::default();
        src.run(&mut sink).await.expect("the run completes");
        sink.packets
    };
    let (packets, _kept) = tokio::join!(play, serve_recording(server));

    let played: Vec<(u64, Vec<u8>, bool)> = packets
        .iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => Some((
                f.timing.pts_ns,
                f.domain.as_system_slice().unwrap().to_vec(),
                f.timing.keyframe,
            )),
            _ => None,
        })
        .collect();
    let mut expected = Vec::new();
    for (_, objects) in recorded_groups() {
        for (index, object) in objects.iter().enumerate() {
            let (timestamp_us, codec_payload) = timestamp_and_codec_payload(object);
            // an empty payload marks the previous frame's end and is not media
            if codec_payload.is_empty() {
                continue;
            }
            expected.push((
                timestamp_us * NANOSECONDS_PER_MICROSECOND,
                codec_payload.to_vec(),
                index == 0,
            ));
        }
    }
    assert_eq!(played, expected);
    assert_eq!(src.selected_track(), VIDEO_TRACK);
}
