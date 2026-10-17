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
//! room while it was live and not after the EOS `leave`. The subscriber test
//! publishes the same way and has `JanusVrSrc` pick that publisher from the
//! room, receive its video, and end its unbounded run on the all-inactive
//! `updated` Janus sends once the publisher's EOS unpublishes.

#![cfg(all(target_os = "linux", feature = "webrtc-janus"))]

mod numbuffers_common;
mod webrtc_streams_common;

use core::task::{Context, Poll};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use g2g_core::frame::Frame;
use g2g_core::log::{LogLevel, LogRecord, LogSink};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, FrameTiming, G2gError, MemoryDomain, MultiOutputSource, OutputSink,
    PipelinePacket, PushOutcome, VideoCodec,
};
use g2g_plugins::janusvrsink::JanusVrSink;
use g2g_plugins::janusvrsrc::JanusVrSrc;
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use g2g_plugins::webrtc_stats::WebRtcStats;
use numbuffers_common::CollectPorts;
use webrtc_streams_common::{data_frames, fixed_caps, frame_bytes, synthetic_stream, FPS};

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
// janus sends the all-inactive update within a second of the unpublish
const LEAVE_NOTICE_MARGIN: Duration = Duration::from_secs(5);
const SUBSCRIBE_DEADLINE: Duration =
    Duration::from_secs(PUBLISH_SECONDS as u64).saturating_add(LEAVE_NOTICE_MARGIN);
const ELEMENT_LOG_SPEC: &str = "JanusVr*:debug";
const SOURCE_LOG_CATEGORY: &str = "JanusVrSrc";
// the line janusvrsrc logs when an update leaves the feed with no active stream
const FEED_STOPPED_LOG: &str = "the feed stopped publishing";

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

    let units = synthetic_stream(codec, (PUBLISH_SECONDS * FPS) as usize);
    let mut sink = JanusVrSink::new(ws)
        .with_room(room_id)
        .with_display_name(DISPLAY_NAME);
    sink.configure_pipeline(&fixed_caps(codec)).unwrap();

    let (stats, seen) = tokio::join!(publish(&mut sink, &units), wait_until_listed(&http, room));
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

// the sink's stats just before its eos
async fn publish(sink: &mut JanusVrSink, units: &[Vec<u8>]) -> WebRtcStats {
    let interval = Duration::from_secs(1) / FPS;
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
}

async fn wait_until_listed(http: &str, room: u64) -> Result<Vec<Value>, String> {
    let deadline = Instant::now() + VERIFY_DEADLINE;
    let mut last = String::new();
    while Instant::now() < deadline {
        tokio::time::sleep(VERIFY_POLL).await;
        match list_participants(http, room).await {
            Ok(participants) if participants.iter().any(is_our_publisher) => {
                return Ok(participants);
            }
            Ok(participants) => last = format!("{participants:?}"),
            Err(e) => last = e,
        }
    }
    Err(last)
}

type CapturedLog = Arc<Mutex<Vec<(LogLevel, String, String)>>>;

struct CaptureLog(CapturedLog);

impl LogSink for CaptureLog {
    fn emit(&self, record: &LogRecord<'_>) {
        let line = (
            record.level,
            record.category.to_string(),
            record.message.to_string(),
        );
        eprintln!("{:?} {} {}", line.0, line.1, line.2);
        self.0.lock().unwrap().push(line);
    }
}

#[tokio::test]
#[ignore = "needs a running Janus (see the module docs)"]
async fn janusvrsrc_ends_when_the_publisher_leaves() {
    let captured = CapturedLog::default();
    g2g_core::log::reset();
    g2g_core::log::set_sink(Box::new(CaptureLog(captured.clone())));
    g2g_core::log::configure(ELEMENT_LOG_SPEC);
    let ws = env_or("G2G_JANUS_URL", DEFAULT_WS);
    let http = env_or("G2G_JANUS_HTTP", DEFAULT_HTTP);
    let room_id = env_or("G2G_JANUS_ROOM", DEFAULT_ROOM);
    let room: u64 = room_id.parse().expect("numeric room id");
    let codec = ROOM_CODEC;
    let units = synthetic_stream(codec, (PUBLISH_SECONDS * FPS) as usize);
    let mut sink = JanusVrSink::new(ws.clone())
        .with_room(room_id.clone())
        .with_display_name(DISPLAY_NAME);
    sink.configure_pipeline(&fixed_caps(codec)).unwrap();
    let mut src = JanusVrSrc::new(ws)
        .with_room(room_id)
        .with_video_codec(codec);
    let mut out = CollectPorts::new(src.output_count());

    let subscriber = async {
        let seen = wait_until_listed(&http, room)
            .await
            .expect("janus listed our publisher");
        eprintln!("subscribing, participants: {seen:?}");
        tokio::time::timeout(SUBSCRIBE_DEADLINE, src.run(&mut out)).await
    };
    let (_, emitted) = tokio::join!(publish(&mut sink, &units), subscriber);
    let emitted = emitted
        .expect("the run ends by itself once the publisher leaves")
        .expect("run succeeds");
    g2g_core::log::reset();

    let [video, audio] = out.packets.as_slice() else {
        panic!("two output ports")
    };
    let video_frames = data_frames(video);
    let video_bytes: usize = video_frames.iter().map(|f| frame_bytes(f).len()).sum();
    eprintln!(
        "subscriber emitted {emitted} buffers: {} video frames ({video_bytes} bytes, first is keyframe {:?}), {} audio frames, eos on video {} audio {}",
        video_frames.len(),
        video_frames.first().map(|f| f.timing.keyframe),
        data_frames(audio).len(),
        matches!(video.last(), Some(PipelinePacket::Eos)),
        matches!(audio.last(), Some(PipelinePacket::Eos)),
    );
    assert!(
        matches!(
            video.first(),
            Some(PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: VideoCodec::Vp8,
                ..
            }))
        ),
        "{:?}",
        video.first()
    );
    let source_log: Vec<_> = captured
        .lock()
        .unwrap()
        .iter()
        .filter(|(_, category, _)| category == SOURCE_LOG_CATEGORY)
        .cloned()
        .collect();
    assert!(
        source_log
            .iter()
            .any(|(level, _, message)| *level == LogLevel::Info
                && message.contains(FEED_STOPPED_LOG)),
        "the run did not end on janus's all-inactive update: {source_log:?}"
    );
    assert!(
        !source_log
            .iter()
            .any(|(level, _, _)| *level <= LogLevel::Warn),
        "{source_log:?}"
    );
    assert!(!video_frames.is_empty(), "no video reached the subscriber");
    assert_eq!(emitted, video_frames.len() as u64, "the feed is video only");
    assert!(
        video_frames[0].timing.keyframe,
        "video starts on a keyframe"
    );
    assert!(matches!(video.last(), Some(PipelinePacket::Eos)));
    assert!(matches!(audio.last(), Some(PipelinePacket::Eos)));
}
