//! `JanusVrSink` and `JanusVrSrc` against a mock Janus VideoRoom on localhost:
//! a WebSocket server speaking the publisher flow (create / attach / join /
//! publish / unpublish / leave), whose answer comes from a real str0m peer that receives
//! the media, and the subscriber flow (listparticipants / join / start / leave),
//! whose offer comes from a real str0m peer that sends media. No Janus install,
//! so it runs by default; a real Janus is the user-side check.

#![cfg(all(target_os = "linux", feature = "webrtc-janus"))]

mod numbuffers_common;
mod webrtc_streams_common;

use core::task::{Context, Poll};
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::handshake::server::{
    Callback, ErrorResponse, Request, Response,
};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

use str0m::change::{SdpAnswer, SdpOffer, SdpPendingOffer};
use str0m::crypto::from_feature_flags;
use str0m::format::Codec;
use str0m::media::{Direction, Frequency, MediaKind, MediaTime, Mid};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, Input, Output, Rtc, RtcConfig};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, MultiOutputSource,
    OutputSink, PipelinePacket, PushOutcome, VideoCodec,
};
use g2g_plugins::janusvrsink::JanusVrSink;
use g2g_plugins::janusvrsrc::JanusVrSrc;
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use numbuffers_common::CollectPorts;
use webrtc_streams_common::{
    data_frames, fixed_caps, frame_bytes, payload, synthetic_stream, FPS, GOP,
};

const SESSION_ID: u64 = 4_815_162_342;
const HANDLE_ID: u64 = 1_234_567;
const FEED_ID: u64 = 77;
const VIEWER_ID: u64 = 5;
const ROOM_ID: u64 = 1234;
const API_SECRET: &str = "s3";
const DISPLAY_NAME: &str = "g2g-test";
const VIDEOROOM_PLUGIN: &str = "janus.plugin.videoroom";
const JANUS_SUBPROTOCOL: &str = "janus-protocol";
const NO_SUCH_ROOM: u64 = 426;
const VIDEO_UNITS: usize = 60;
const MIN_DELIVERED_PERCENT: usize = 90;
const UNIT_TAIL_BYTES: usize = 64;
// let the sender's pacer flush before EOS leaves the room
const DRAIN_BEFORE_EOS: Duration = Duration::from_millis(500);
const SERVER_EXIT_TIMEOUT: Duration = Duration::from_secs(5);
const UDP_RECV_BYTES: usize = 2000;
const SUBSCRIBED_VIDEO_CODEC: WebRtcVideoCodec = WebRtcVideoCodec::H264;
const OPUS_CODEC_NAME: &str = "opus";
const OPUS_PACKET_BYTES: usize = 120;
const OPUS_RATE: u64 = 48_000;
const VIDEO_RATE: u64 = 90_000;
// lets the subscriber finish its dtls handshake before the first keyframe
const SETTLE_BEFORE_MEDIA: Duration = Duration::from_millis(200);
const SUBSCRIBE_TIMEOUT: Duration = Duration::from_secs(15);

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

/// What the mock saw, for the assertions.
#[derive(Default)]
struct MockState {
    subprotocol: Option<String>,
    /// `janus` kinds in order, with the plugin request appended for messages.
    requests: Vec<String>,
    secrets: Vec<Value>,
    join: Value,
    publish: Value,
    start: Value,
    reject_join: bool,
}

type Received = Arc<Mutex<Vec<Vec<u8>>>>;

/// Records the offered subprotocol and echoes it, as a real server must.
struct EchoSubprotocol(Arc<Mutex<MockState>>);

impl Callback for EchoSubprotocol {
    fn on_request(
        self,
        request: &Request,
        mut response: Response,
    ) -> Result<Response, ErrorResponse> {
        let offered = request.headers().get("Sec-WebSocket-Protocol").cloned();
        self.0.lock().unwrap().subprotocol = offered
            .as_ref()
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        if let Some(offered) = offered {
            response
                .headers_mut()
                .insert("Sec-WebSocket-Protocol", offered);
        }
        Ok(response)
    }
}

fn event(transaction: &Value, data: Value, jsep: Option<Value>) -> Value {
    let mut message = json!({
        "janus": "event",
        "session_id": SESSION_ID,
        "sender": HANDLE_ID,
        "transaction": transaction,
        "plugindata": { "plugin": VIDEOROOM_PLUGIN, "data": data },
    });
    if let Some(jsep) = jsep {
        message["jsep"] = jsep;
    }
    message
}

async fn send(ws: &mut WebSocketStream<TcpStream>, message: Value) {
    ws.send(Message::Text(message.to_string())).await.unwrap();
}

/// One Janus connection: the core and videoroom replies both flows need.
async fn mock_janus(listener: TcpListener, state: Arc<Mutex<MockState>>, received: Received) {
    let (tcp, _) = listener.accept().await.unwrap();
    let mut ws = tokio_tungstenite::accept_hdr_async(tcp, EchoSubprotocol(state.clone()))
        .await
        .unwrap();
    let mut offerer: Option<PublisherPeer> = None;
    let (publisher_gone_tx, mut publisher_gone) = mpsc::channel(1);

    loop {
        let frame = tokio::select! {
            frame = ws.next() => frame,
            Some(streams) = publisher_gone.recv() => {
                let data = json!({ "videoroom": "updated", "room": ROOM_ID, "streams": streams });
                send(&mut ws, event(&Value::Null, data, None)).await;
                continue;
            }
        };
        let Some(Ok(frame)) = frame else { return };
        let Message::Text(text) = frame else { continue };
        let message: Value = serde_json::from_str(&text).unwrap();
        let transaction = message["transaction"].clone();
        let kind = message["janus"].as_str().unwrap_or_default().to_string();
        let request = message["body"]["request"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        {
            let mut state = state.lock().unwrap();
            state.requests.push(if request.is_empty() {
                kind.clone()
            } else {
                request.clone()
            });
            state.secrets.push(message["apisecret"].clone());
        }
        match kind.as_str() {
            "create" => {
                send(&mut ws, json!({ "janus": "success", "transaction": transaction, "data": { "id": SESSION_ID } })).await;
            }
            "attach" => {
                assert_eq!(message["plugin"], VIDEOROOM_PLUGIN);
                assert_eq!(message["session_id"], SESSION_ID);
                send(&mut ws, json!({ "janus": "success", "session_id": SESSION_ID, "transaction": transaction, "data": { "id": HANDLE_ID } })).await;
            }
            "keepalive" | "trickle" => {
                send(
                    &mut ws,
                    json!({ "janus": "ack", "session_id": SESSION_ID, "transaction": transaction }),
                )
                .await;
            }
            "message" if request == "listparticipants" => {
                assert_eq!(message["body"]["room"], ROOM_ID);
                let participants = json!([
                    { "id": VIEWER_ID, "publisher": false },
                    { "id": FEED_ID, "display": DISPLAY_NAME, "publisher": true },
                ]);
                let data = json!({ "videoroom": "participants", "room": ROOM_ID, "participants": participants });
                send(&mut ws, json!({ "janus": "success", "session_id": SESSION_ID, "sender": HANDLE_ID, "transaction": transaction,
                    "plugindata": { "plugin": VIDEOROOM_PLUGIN, "data": data } })).await;
            }
            "message" => {
                assert_eq!(message["session_id"], SESSION_ID);
                assert_eq!(message["handle_id"], HANDLE_ID);
                send(
                    &mut ws,
                    json!({ "janus": "ack", "session_id": SESSION_ID, "transaction": transaction }),
                )
                .await;
                let subscriber = message["body"]["ptype"] == "subscriber"
                    || state.lock().unwrap().join["ptype"] == "subscriber";
                match request.as_str() {
                    "join" if subscriber => {
                        state.lock().unwrap().join = message["body"].clone();
                        let (built, offer_sdp) = subscriber_offer();
                        let data = json!({ "videoroom": "attached", "room": ROOM_ID,
                            "streams": built.streams(true) });
                        let jsep = json!({ "type": "offer", "sdp": offer_sdp });
                        offerer = Some(built);
                        send(&mut ws, event(&transaction, data, Some(jsep))).await;
                    }
                    "join" => {
                        let reject = {
                            let mut state = state.lock().unwrap();
                            state.join = message["body"].clone();
                            state.reject_join
                        };
                        let data = if reject {
                            json!({ "videoroom": "event", "error_code": NO_SUCH_ROOM, "error": "No such room" })
                        } else {
                            json!({ "videoroom": "joined", "room": ROOM_ID, "id": FEED_ID, "private_id": 1, "publishers": [] })
                        };
                        send(&mut ws, event(&transaction, data, None)).await;
                    }
                    "start" => {
                        state.lock().unwrap().start = message.clone();
                        let mut offerer = offerer.take().expect("start before join");
                        let answer_sdp = message["jsep"]["sdp"].as_str().unwrap();
                        let socket = UdpSocket::bind((candidate_ip(answer_sdp), 0))
                            .await
                            .unwrap();
                        let local = socket.local_addr().unwrap();
                        let candidate = Candidate::host(local, "udp").unwrap();
                        offerer.rtc.add_local_candidate(candidate.clone());
                        offerer.accept_answer(answer_sdp);
                        let data =
                            json!({ "videoroom": "event", "room": ROOM_ID, "started": "ok" });
                        send(&mut ws, event(&transaction, data, None)).await;
                        send(&mut ws, json!({ "janus": "trickle", "session_id": SESSION_ID, "sender": HANDLE_ID,
                            "candidate": { "candidate": candidate.to_sdp_string(), "sdpMid": offerer.video_mid.to_string(), "sdpMLineIndex": 0 } })).await;
                        tokio::spawn(send_media_then_unpublish(
                            offerer,
                            socket,
                            local,
                            publisher_gone_tx.clone(),
                        ));
                    }
                    "publish" => {
                        state.lock().unwrap().publish = message.clone();
                        let answer_sdp = answer_offer(
                            message["jsep"]["sdp"].as_str().unwrap(),
                            received.clone(),
                        )
                        .await;
                        let data = json!({ "videoroom": "event", "configured": "ok" });
                        let jsep = json!({ "type": "answer", "sdp": answer_sdp });
                        send(&mut ws, event(&transaction, data, Some(jsep))).await;
                        send(&mut ws, json!({ "janus": "webrtcup", "session_id": SESSION_ID, "sender": HANDLE_ID })).await;
                    }
                    "unpublish" => {
                        let data =
                            json!({ "videoroom": "event", "room": ROOM_ID, "unpublished": "ok" });
                        send(&mut ws, event(&transaction, data, None)).await;
                    }
                    "leave" => {
                        let acknowledgement = if subscriber { "left" } else { "leaving" };
                        let mut data = json!({ "videoroom": "event" });
                        data[acknowledgement] = json!("ok");
                        send(&mut ws, event(&transaction, data, None)).await;
                        return;
                    }
                    other => panic!("unexpected videoroom request {other}"),
                }
            }
            other => panic!("unexpected janus message {other}"),
        }
    }
}

struct PublisherPeer {
    rtc: Rtc,
    pending: Option<SdpPendingOffer>,
    video_mid: Mid,
    audio_mid: Mid,
}

impl PublisherPeer {
    // the streams summary janus puts in attached and updated events
    fn streams(&self, active: bool) -> Value {
        json!([
            { "type": "video", "active": active, "mindex": 0, "mid": self.video_mid.to_string(),
              "codec": SUBSCRIBED_VIDEO_CODEC.name(), "feed_id": FEED_ID },
            { "type": "audio", "active": active, "mindex": 1, "mid": self.audio_mid.to_string(),
              "codec": OPUS_CODEC_NAME, "feed_id": FEED_ID },
        ])
    }

    fn accept_answer(&mut self, answer_sdp: &str) {
        let answer = SdpAnswer::from_sdp_string(answer_sdp).unwrap();
        let pending = self.pending.take().expect("one answer per offer");
        self.rtc.sdp_api().accept_answer(pending, answer).unwrap();
    }

    fn write(&mut self, mid: Mid, codec: Codec, rtp_time: MediaTime, data: &[u8]) {
        let writer = self.rtc.writer(mid).expect("negotiated mid");
        let pt = writer
            .payload_params()
            .find(|p| p.spec().codec == codec)
            .expect("negotiated codec")
            .pt();
        writer
            .write(pt, Instant::now(), rtp_time, data.to_vec())
            .unwrap();
    }
}

// no candidates: the mock binds next to the answer's address and trickles one
fn subscriber_offer() -> (PublisherPeer, String) {
    let mut rtc = RtcConfig::new()
        .set_crypto_provider(Arc::new(from_feature_flags()))
        .clear_codecs()
        .enable_h264(true)
        .enable_opus(true)
        .build(Instant::now());
    let mut api = rtc.sdp_api();
    let video_mid = api.add_media(MediaKind::Video, Direction::SendOnly, None, None, None);
    let audio_mid = api.add_media(MediaKind::Audio, Direction::SendOnly, None, None, None);
    let (offer, pending) = api.apply().unwrap();
    let offer_sdp = offer.to_sdp_string();
    let offerer = PublisherPeer {
        rtc,
        pending: Some(pending),
        video_mid,
        audio_mid,
    };
    (offerer, offer_sdp)
}

fn video_units() -> Vec<Vec<u8>> {
    synthetic_stream(SUBSCRIBED_VIDEO_CODEC, VIDEO_UNITS)
}

fn opus_packets() -> Vec<Vec<u8>> {
    (0..VIDEO_UNITS)
        .map(|i| payload(OPUS_PACKET_BYTES, i as u8))
        .collect()
}

async fn send_media_then_unpublish(
    mut offerer: PublisherPeer,
    socket: UdpSocket,
    local: SocketAddr,
    publisher_gone: mpsc::Sender<Value>,
) {
    let units = video_units();
    let packets = opus_packets();
    let interval = Duration::from_secs(1) / FPS;
    let mut media_start: Option<Instant> = None;
    let mut written = 0usize;
    let mut gone_reported = false;
    let mut buf = vec![0u8; UDP_RECV_BYTES];
    loop {
        let deadline = loop {
            match offerer.rtc.poll_output() {
                Ok(Output::Timeout(t)) => break t,
                Ok(Output::Transmit(t)) => {
                    let _ = socket.send_to(&t.contents, t.destination).await;
                }
                Ok(Output::Event(Event::Connected)) => {
                    media_start = Some(Instant::now() + SETTLE_BEFORE_MEDIA)
                }
                Ok(Output::Event(_)) => {}
                Err(_) => return,
            }
        };
        let mut wake = deadline;
        if let Some(start) = media_start {
            let due = start + interval * written as u32;
            if written < units.len() && Instant::now() >= due {
                let video_time = MediaTime::new(
                    written as u64 * VIDEO_RATE / FPS as u64,
                    Frequency::NINETY_KHZ,
                );
                let audio_time = MediaTime::new(
                    written as u64 * OPUS_RATE / FPS as u64,
                    Frequency::FORTY_EIGHT_KHZ,
                );
                let (video_mid, audio_mid) = (offerer.video_mid, offerer.audio_mid);
                offerer.write(video_mid, Codec::H264, video_time, &units[written]);
                offerer.write(audio_mid, Codec::Opus, audio_time, &packets[written]);
                written += 1;
                continue;
            }
            if written < units.len() {
                wake = wake.min(due);
            } else if !gone_reported {
                let gone_at = due + DRAIN_BEFORE_EOS;
                if Instant::now() >= gone_at {
                    gone_reported = true;
                    let _ = publisher_gone.send(offerer.streams(false)).await;
                } else {
                    wake = wake.min(gone_at);
                }
            }
        }
        let timeout = wake.saturating_duration_since(Instant::now());
        tokio::select! {
            r = socket.recv_from(&mut buf) => {
                let Ok((n, source)) = r else { return };
                if let Ok(contents) = buf[..n].try_into() {
                    let input = Input::Receive(Instant::now(), Receive { proto: Protocol::Udp, source, destination: local, contents });
                    if offerer.rtc.handle_input(input).is_err() {
                        return;
                    }
                }
            }
            _ = tokio::time::sleep(timeout) => {
                if offerer.rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                    return;
                }
            }
        }
    }
}

/// The IP the publisher put in its host candidate, so the answerer binds the
/// same interface.
fn candidate_ip(offer_sdp: &str) -> IpAddr {
    let line = offer_sdp
        .lines()
        .find(|l| l.starts_with("a=candidate:"))
        .expect("offer carries a host candidate");
    line.split_whitespace().nth(4).unwrap().parse().unwrap()
}

/// Answer `offer_sdp` from a receiving str0m peer and keep it running.
async fn answer_offer(offer_sdp: &str, received: Received) -> String {
    let mut rtc = RtcConfig::new()
        .set_crypto_provider(Arc::new(from_feature_flags()))
        .build(Instant::now());
    let socket = UdpSocket::bind((candidate_ip(offer_sdp), 0)).await.unwrap();
    let local = socket.local_addr().unwrap();
    rtc.add_local_candidate(Candidate::host(local, "udp").unwrap());
    let answer = rtc
        .sdp_api()
        .accept_offer(SdpOffer::from_sdp_string(offer_sdp).unwrap())
        .unwrap();
    tokio::spawn(receive_media(rtc, socket, local, received));
    answer.to_sdp_string()
}

async fn receive_media(mut rtc: Rtc, socket: UdpSocket, local: SocketAddr, received: Received) {
    let mut buf = vec![0u8; UDP_RECV_BYTES];
    loop {
        let deadline = loop {
            match rtc.poll_output() {
                Ok(Output::Timeout(t)) => break t,
                Ok(Output::Transmit(t)) => {
                    let _ = socket.send_to(&t.contents, t.destination).await;
                }
                Ok(Output::Event(Event::MediaData(d))) => {
                    received.lock().unwrap().push(d.data.to_vec())
                }
                Ok(Output::Event(_)) => {}
                Err(_) => return,
            }
        };
        let timeout = deadline.saturating_duration_since(Instant::now());
        tokio::select! {
            r = socket.recv_from(&mut buf) => {
                let Ok((n, source)) = r else { return };
                if let Ok(contents) = buf[..n].try_into() {
                    let input = Input::Receive(Instant::now(), Receive { proto: Protocol::Udp, source, destination: local, contents });
                    if rtc.handle_input(input).is_err() {
                        return;
                    }
                }
            }
            _ = tokio::time::sleep(timeout) => {
                if rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                    return;
                }
            }
        }
    }
}

fn frame(sequence: usize, unit: &[u8], interval: Duration) -> PipelinePacket {
    let timing = FrameTiming {
        pts_ns: sequence as u64 * interval.as_nanos() as u64,
        ..FrameTiming::default()
    };
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(unit.to_vec().into_boxed_slice())),
        timing,
        sequence as u64,
    ))
}

async fn start_mock(
    reject_join: bool,
) -> (
    u16,
    Arc<Mutex<MockState>>,
    Received,
    tokio::task::JoinHandle<()>,
) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let state = Arc::new(Mutex::new(MockState {
        reject_join,
        ..MockState::default()
    }));
    let received: Received = Arc::default();
    let server = tokio::spawn(mock_janus(listener, state.clone(), received.clone()));
    (port, state, received, server)
}

fn sink(port: u16) -> JanusVrSink {
    JanusVrSink::new(format!("ws://127.0.0.1:{port}"))
        .with_room(ROOM_ID.to_string())
        .with_display_name(DISPLAY_NAME)
        .with_api_secret(API_SECRET)
}

#[tokio::test]
async fn publishes_into_a_videoroom() {
    let (port, state, received, server) = start_mock(false).await;
    let codec = WebRtcVideoCodec::H264;
    let units = synthetic_stream(codec, VIDEO_UNITS);
    let interval = Duration::from_secs(1) / FPS;
    let mut sink = sink(port);
    sink.configure_pipeline(&fixed_caps(codec)).unwrap();
    for (sequence, unit) in units.iter().enumerate() {
        sink.process(frame(sequence, unit, interval), &mut Discard)
            .await
            .unwrap();
        tokio::time::sleep(interval).await;
    }
    tokio::time::sleep(DRAIN_BEFORE_EOS).await;
    sink.process(PipelinePacket::Eos, &mut Discard)
        .await
        .unwrap();
    assert_eq!(sink.frames_sent(), units.len() as u64);
    tokio::time::timeout(SERVER_EXIT_TIMEOUT, server)
        .await
        .expect("mock janus never saw the leave")
        .unwrap();

    let state = state.lock().unwrap();
    assert_eq!(state.subprotocol.as_deref(), Some(JANUS_SUBPROTOCOL));
    assert_eq!(
        state.requests,
        ["create", "attach", "join", "publish", "unpublish", "leave"]
    );
    assert!(
        state.secrets.iter().all(|s| *s == API_SECRET),
        "{:?}",
        state.secrets
    );
    assert_eq!(state.join["ptype"], "publisher");
    assert_eq!(state.join["room"], ROOM_ID);
    assert_eq!(state.join["display"], DISPLAY_NAME);
    assert_eq!(state.publish["jsep"]["type"], "offer");
    assert_eq!(state.publish["jsep"]["trickle"], false);
    assert!(state.publish["jsep"]["sdp"]
        .as_str()
        .unwrap()
        .contains("a=candidate:"));

    let received = received.lock().unwrap();
    assert!(
        received.len() >= units.len() * MIN_DELIVERED_PERCENT / 100,
        "{} of {} units arrived",
        received.len(),
        units.len()
    );
    let tail = &units[0][units[0].len() - UNIT_TAIL_BYTES..];
    assert!(
        received[0].ends_with(tail),
        "first unit does not end with the sent bytes"
    );
    let stats = sink.stats_handle().snapshot();
    assert!(stats.packets_sent() >= received.len() as u64, "{stats:?}");
    assert!(
        stats.peer.as_ref().is_some_and(|p| p.bytes_tx > 0),
        "{stats:?}"
    );
}

#[tokio::test]
async fn a_room_error_fails_the_first_frame() {
    let (port, state, _received, server) = start_mock(true).await;
    let codec = WebRtcVideoCodec::H264;
    let units = synthetic_stream(codec, 1);
    let mut sink = sink(port);
    sink.configure_pipeline(&fixed_caps(codec)).unwrap();
    let result = sink
        .process(
            frame(0, &units[0], Duration::from_secs(1) / FPS),
            &mut Discard,
        )
        .await;
    assert!(matches!(result, Err(G2gError::Hardware(_))), "{result:?}");
    assert_eq!(sink.frames_sent(), 0);
    server.abort();
    assert_eq!(state.lock().unwrap().requests, ["create", "attach", "join"]);
}

#[tokio::test]
async fn subscribes_to_the_first_publisher_until_it_leaves() {
    let (port, state, _received, server) = start_mock(false).await;
    let mut src = JanusVrSrc::new(format!("ws://127.0.0.1:{port}"))
        .with_room(ROOM_ID.to_string())
        .with_api_secret(API_SECRET)
        .with_video_codec(SUBSCRIBED_VIDEO_CODEC);
    let mut out = CollectPorts::new(src.output_count());
    let emitted = tokio::time::timeout(SUBSCRIBE_TIMEOUT, src.run(&mut out))
        .await
        .expect("the run ends when the publisher leaves")
        .expect("run succeeds");
    tokio::time::timeout(SERVER_EXIT_TIMEOUT, server)
        .await
        .expect("mock janus never saw the leave")
        .unwrap();

    let state = state.lock().unwrap();
    assert_eq!(state.subprotocol.as_deref(), Some(JANUS_SUBPROTOCOL));
    assert_eq!(
        state.requests,
        [
            "create",
            "attach",
            "listparticipants",
            "join",
            "start",
            "leave"
        ]
    );
    assert!(
        state.secrets.iter().all(|s| *s == API_SECRET),
        "{:?}",
        state.secrets
    );
    assert_eq!(state.join["ptype"], "subscriber");
    assert_eq!(state.join["room"], ROOM_ID);
    assert_eq!(state.join["streams"], json!([{ "feed": FEED_ID }]));
    assert_eq!(state.start["jsep"]["type"], "answer");
    assert!(state.start["jsep"]["sdp"]
        .as_str()
        .unwrap()
        .contains("a=candidate:"));

    let [video, audio] = out.packets.as_slice() else {
        panic!("two output ports")
    };
    assert!(
        matches!(
            video.first(),
            Some(PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }))
        ),
        "{:?}",
        video.first()
    );
    assert!(
        matches!(
            audio.first(),
            Some(PipelinePacket::CapsChanged(Caps::Audio {
                format: AudioFormat::Opus,
                ..
            }))
        ),
        "{:?}",
        audio.first()
    );
    assert!(matches!(video.last(), Some(PipelinePacket::Eos)));
    assert!(matches!(audio.last(), Some(PipelinePacket::Eos)));

    let video_frames = data_frames(video);
    let audio_frames = data_frames(audio);
    assert_eq!(emitted, (video_frames.len() + audio_frames.len()) as u64);
    let units = video_units();
    let first_video = video_frames.first().expect("video frames arrived");
    assert!(first_video.timing.keyframe, "video starts on a keyframe");
    let keyframe_tails: Vec<&[u8]> = units
        .iter()
        .step_by(GOP)
        .map(|u| &u[u.len() - UNIT_TAIL_BYTES..])
        .collect();
    assert!(
        keyframe_tails
            .iter()
            .any(|tail| frame_bytes(first_video).ends_with(tail)),
        "first video frame is not a sent keyframe"
    );
    let packets = opus_packets();
    assert!(
        audio_frames.len() >= packets.len() * MIN_DELIVERED_PERCENT / 100,
        "{} of {} opus packets arrived",
        audio_frames.len(),
        packets.len()
    );
    assert!(
        packets
            .iter()
            .any(|p| p.as_slice() == frame_bytes(audio_frames[0])),
        "first audio frame is not a sent opus packet"
    );
}
