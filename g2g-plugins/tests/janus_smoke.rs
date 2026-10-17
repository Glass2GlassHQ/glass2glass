//! `JanusVrSink` against a real Janus. Ignored by default: it needs a running
//! Janus with the videoroom plugin and the WebSocket + HTTP transports, so it
//! is a user-run harness, not a CI gate.
//!
//! ```sh
//! docker run -d --network host --name g2g-janus canyan/janus-gateway:latest
//! cargo test -p g2g-plugins --features webrtc-janus --test janus_smoke -- --ignored --nocapture
//! ```
//!
//! `G2G_JANUS_URL` (default `ws://127.0.0.1:8188`) and `G2G_JANUS_HTTP`
//! (default `http://127.0.0.1:8088`) point elsewhere; `G2G_JANUS_ROOM` picks a
//! room (default the stock demo room 1234, VP8 only). Green means the join + publish
//! handshake completed, media flowed, and Janus listed the publisher in the
//! room while it was live and not after the EOS `leave`.

#![cfg(all(target_os = "linux", feature = "webrtc-janus"))]

mod webrtc_streams_common;

use core::task::{Context, Poll};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
};
use g2g_plugins::janusvrsink::JanusVrSink;
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use webrtc_streams_common::{fixed_caps, synthetic_stream, FPS};

const DEFAULT_WS: &str = "ws://127.0.0.1:8188";
const DEFAULT_HTTP: &str = "http://127.0.0.1:8088";
const DEFAULT_ROOM: &str = "1234";
const DISPLAY_NAME: &str = "g2g-smoke";
// the stock demo room 1234 allows vp8 only
const ROOM_CODEC: WebRtcVideoCodec = WebRtcVideoCodec::Vp8;
const PUBLISH_SECONDS: u32 = 6;
const VIDEOROOM_PLUGIN: &str = "janus.plugin.videoroom";
const VERIFY_DEADLINE: Duration = Duration::from_secs(9);
const VERIFY_POLL: Duration = Duration::from_millis(500);
const DRAIN_BEFORE_EOS: Duration = Duration::from_millis(500);

struct Discard;

impl OutputSink for Discard {
    fn poll_push(
        &mut self,
        _cx: &mut Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        packet.take();
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

/// The videoroom participants of `room`, through the Janus HTTP API.
async fn list_participants(http: &str, room: u64) -> Result<Vec<Value>, String> {
    let client = reqwest::Client::new();
    let post = |url: String, body: Value| {
        let client = client.clone();
        async move {
            let text = client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(body.to_string())
                .send()
                .await
                .map_err(|e| format!("POST {url}: {e}"))?
                .text()
                .await
                .map_err(|e| format!("POST {url}: {e}"))?;
            serde_json::from_str::<Value>(&text).map_err(|e| format!("POST {url}: {e}: {text}"))
        }
    };
    let created = post(
        format!("{http}/janus"),
        json!({ "janus": "create", "transaction": "t1" }),
    )
    .await?;
    let session = created["data"]["id"].as_u64().ok_or(format!("{created}"))?;
    let attached = post(
        format!("{http}/janus/{session}"),
        json!({ "janus": "attach", "plugin": VIDEOROOM_PLUGIN, "transaction": "t2" }),
    )
    .await?;
    let handle = attached["data"]["id"]
        .as_u64()
        .ok_or(format!("{attached}"))?;
    let listed = post(
        format!("{http}/janus/{session}/{handle}"),
        json!({ "janus": "message", "transaction": "t3",
                "body": { "request": "listparticipants", "room": room } }),
    )
    .await?;
    let _ = post(
        format!("{http}/janus/{session}"),
        json!({ "janus": "destroy", "transaction": "t4" }),
    )
    .await;
    listed["plugindata"]["data"]["participants"]
        .as_array()
        .cloned()
        .ok_or(format!("{listed}"))
}

fn is_our_publisher(participant: &Value) -> bool {
    participant["display"] == DISPLAY_NAME && participant["publisher"] == true
}

#[tokio::test]
#[ignore = "needs a running Janus (see the module docs)"]
async fn janus_lists_the_publisher_while_it_publishes() {
    let ws = env_or("G2G_JANUS_URL", DEFAULT_WS);
    let http = env_or("G2G_JANUS_HTTP", DEFAULT_HTTP);
    let room_id = env_or("G2G_JANUS_ROOM", DEFAULT_ROOM);
    let room: u64 = room_id.parse().expect("numeric room id");
    let codec = ROOM_CODEC;
    eprintln!("publishing synthetic {codec:?} -> {ws} room {room} as {DISPLAY_NAME}");

    let interval = Duration::from_secs(1) / FPS;
    let units = synthetic_stream(codec, (PUBLISH_SECONDS * FPS) as usize);
    let mut sink = JanusVrSink::new(ws)
        .with_room(room_id)
        .with_display_name(DISPLAY_NAME);
    sink.configure_pipeline(&fixed_caps(codec)).unwrap();

    let publisher = async {
        for (sequence, unit) in units.iter().enumerate() {
            let timing = FrameTiming {
                pts_ns: sequence as u64 * interval.as_nanos() as u64,
                ..FrameTiming::default()
            };
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(unit.clone().into_boxed_slice())),
                timing,
                sequence as u64,
            );
            sink.process(PipelinePacket::DataFrame(frame), &mut Discard)
                .await
                .expect("publish frame");
            tokio::time::sleep(interval).await;
        }
        tokio::time::sleep(DRAIN_BEFORE_EOS).await;
        let stats = sink.stats_handle().snapshot();
        sink.process(PipelinePacket::Eos, &mut Discard)
            .await
            .expect("eos");
        stats
    };
    let verifier = async {
        let deadline = Instant::now() + VERIFY_DEADLINE;
        let mut last = String::new();
        while Instant::now() < deadline {
            tokio::time::sleep(VERIFY_POLL).await;
            match list_participants(&http, room).await {
                Ok(participants) if participants.iter().any(is_our_publisher) => {
                    return Ok(participants);
                }
                Ok(participants) => last = format!("{participants:?}"),
                Err(e) => last = e,
            }
        }
        Err(last)
    };
    let (stats, seen) = tokio::join!(publisher, verifier);
    let seen = seen.expect("janus listed our publisher");
    eprintln!("participants while live: {seen:?}");
    eprintln!("sink stats at eos: {stats:?}");
    assert!(stats.packets_sent() > 0, "{stats:?}");
    assert!(
        stats.peer.as_ref().is_some_and(|p| p.bytes_tx > 0),
        "{stats:?}"
    );

    // the EOS `leave` removed us from the room
    tokio::time::sleep(VERIFY_POLL).await;
    let after = list_participants(&http, room)
        .await
        .expect("list after leave");
    assert!(!after.iter().any(is_our_publisher), "{after:?}");
}

#[tokio::test]
#[ignore = "needs a running Janus (see the module docs)"]
async fn a_codec_the_room_refuses_fails_the_first_frame() {
    let ws = env_or("G2G_JANUS_URL", DEFAULT_WS);
    let room_id = env_or("G2G_JANUS_ROOM", DEFAULT_ROOM);
    // the demo room takes vp8 only, so h264 comes back disabled
    let codec = WebRtcVideoCodec::H264;
    let units = synthetic_stream(codec, 1);
    let mut sink = JanusVrSink::new(ws)
        .with_room(room_id)
        .with_display_name(DISPLAY_NAME);
    sink.configure_pipeline(&fixed_caps(codec)).unwrap();
    let frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(units[0].clone().into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    let result = sink
        .process(PipelinePacket::DataFrame(frame), &mut Discard)
        .await;
    assert!(matches!(result, Err(G2gError::Hardware(_))), "{result:?}");
    assert_eq!(sink.frames_sent(), 0);
}
