#![cfg(all(target_os = "linux", feature = "webrtc-kvs"))]

mod numbuffers_common;
mod webrtc_streams_common;

use core::task::{Context, Poll};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::handshake::server::{
    Callback, ErrorResponse, Request, Response,
};
use tokio_tungstenite::tungstenite::Message;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, AudioFormat, Caps, FrameTiming, G2gError, MemoryDomain, MultiOutputSource,
    OutputSink, PipelinePacket, PushOutcome, VideoCodec,
};
use g2g_plugins::kvswebrtcsink::KvsWebRtcSink;
use g2g_plugins::kvswebrtcsrc::KvsWebRtcSrc;
use g2g_plugins::webrtc_codec::WebRtcVideoCodec;
use numbuffers_common::CollectPorts;
use webrtc_streams_common::{data_frames, fixed_caps, frame_bytes, synthetic_stream, FPS};

const CHANNEL_ARN: &str =
    "arn:aws:kinesisvideo:us-west-2:123456789012:channel/g2g-loopback/1234567890123";
const REGION: &str = "us-west-2";
const ACCESS_KEY: &str = "AKIDEXAMPLE";
const SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
const SESSION_TOKEN: &str = "session-token-example";
const VIEWER_CLIENT_ID: &str = "g2g-viewer-1";
const VIDEO_CODEC: WebRtcVideoCodec = WebRtcVideoCodec::H264;
const VIDEO_UNITS: usize = 60;
const MIN_DELIVERED_PERCENT: usize = 90;
const UNIT_TAIL_BYTES: usize = 64;
// let the sender's pacer flush before EOS
const DRAIN_BEFORE_EOS: Duration = Duration::from_millis(500);
const RUN_TIMEOUT: Duration = Duration::from_secs(30);
const ICE_CONFIG_PATH: &str = "/v1/get-ice-server-config";
const SIGNATURE_HEX_DIGITS: usize = 64;
const PRESIGNED_EXPIRES_SECONDS: &str = "299";
const HEADER_END: &[u8] = b"\r\n\r\n";
const HTTP_READ_BYTES: usize = 4096;

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

#[derive(Debug, Clone)]
struct Relayed {
    from_viewer: bool,
    action: String,
    recipient: Option<String>,
}

#[derive(Debug, Clone)]
struct IceConfigRequest {
    path: String,
    headers: HashMap<String, String>,
    body: Value,
}

#[derive(Default)]
struct MockState {
    connections: Vec<HashMap<String, String>>,
    relayed: Vec<Relayed>,
    ice_config_requests: Vec<IceConfigRequest>,
}

#[derive(Default)]
struct Router {
    master: Option<mpsc::UnboundedSender<String>>,
    viewers: HashMap<String, mpsc::UnboundedSender<String>>,
    // kvs holds messages for an absent recipient until its ttl expires
    held_for_master: Vec<String>,
    held_for_viewers: HashMap<String, Vec<String>>,
}

impl Router {
    fn deliver(&mut self, viewer: Option<&str>, message: String) {
        match viewer {
            None => match &self.master {
                Some(master) => {
                    let _ = master.send(message);
                }
                None => self.held_for_master.push(message),
            },
            Some(id) => match self.viewers.get(id) {
                Some(viewer) => {
                    let _ = viewer.send(message);
                }
                None => self
                    .held_for_viewers
                    .entry(id.to_string())
                    .or_default()
                    .push(message),
            },
        }
    }

    fn register(&mut self, viewer: Option<&str>, sender: mpsc::UnboundedSender<String>) {
        let held = match viewer {
            None => core::mem::take(&mut self.held_for_master),
            Some(id) => self.held_for_viewers.remove(id).unwrap_or_default(),
        };
        for message in held {
            let _ = sender.send(message);
        }
        match viewer {
            None => self.master = Some(sender),
            Some(id) => {
                self.viewers.insert(id.to_string(), sender);
            }
        }
    }
}

struct RecordUri(Arc<Mutex<String>>);

impl Callback for RecordUri {
    fn on_request(self, request: &Request, response: Response) -> Result<Response, ErrorResponse> {
        *self.0.lock().unwrap() = request.uri().to_string();
        Ok(response)
    }
}

fn query_parameters(uri: &str) -> HashMap<String, String> {
    reqwest::Url::parse(&format!("ws://mock{uri}"))
        .unwrap()
        .query_pairs()
        .map(|(name, value)| (name.into_owned(), value.into_owned()))
        .collect()
}

async fn serve_signalling(
    tcp: TcpStream,
    state: Arc<Mutex<MockState>>,
    router: Arc<Mutex<Router>>,
) {
    let uri = Arc::new(Mutex::new(String::new()));
    let ws = tokio_tungstenite::accept_hdr_async(tcp, RecordUri(uri.clone()))
        .await
        .unwrap();
    let parameters = query_parameters(&uri.lock().unwrap());
    let viewer = parameters.get("X-Amz-ClientId").cloned();
    state.lock().unwrap().connections.push(parameters);
    let (mut write, mut read) = ws.split();
    let (tx, mut rx) = mpsc::unbounded_channel();
    router.lock().unwrap().register(viewer.as_deref(), tx);
    loop {
        tokio::select! {
            Some(outgoing) = rx.recv() => {
                if write.send(Message::Text(outgoing)).await.is_err() {
                    return;
                }
            }
            frame = read.next() => {
                let text = match frame {
                    Some(Ok(Message::Text(text))) => text,
                    Some(Ok(_)) => continue,
                    _ => return,
                };
                let message: Value = serde_json::from_str(&text).unwrap();
                let action = message["action"].as_str().unwrap().to_string();
                let recipient = message["recipientClientId"].as_str().map(str::to_string);
                state.lock().unwrap().relayed.push(Relayed {
                    from_viewer: viewer.is_some(),
                    action: action.clone(),
                    recipient: recipient.clone(),
                });
                let mut delivered = json!({ "messageType": action, "messagePayload": message["messagePayload"] });
                let target = match &viewer {
                    Some(id) => {
                        delivered["senderClientId"] = json!(id);
                        None
                    }
                    None => Some(recipient.expect("master messages name a viewer")),
                };
                router.lock().unwrap().deliver(target.as_deref(), delivered.to_string());
            }
        }
    }
}

async fn mock_signalling(listener: TcpListener, state: Arc<Mutex<MockState>>) {
    let router = Arc::new(Mutex::new(Router::default()));
    loop {
        let (tcp, _) = listener.accept().await.unwrap();
        tokio::spawn(serve_signalling(tcp, state.clone(), router.clone()));
    }
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

async fn serve_ice_config(mut tcp: TcpStream, state: Arc<Mutex<MockState>>) {
    let mut data = Vec::new();
    let mut buf = [0u8; HTTP_READ_BYTES];
    let (head, body) = loop {
        let n = tcp.read(&mut buf).await.unwrap();
        assert!(n > 0, "request ended early");
        data.extend_from_slice(&buf[..n]);
        let Some(end) = find(&data, HEADER_END) else {
            continue;
        };
        let head = String::from_utf8(data[..end].to_vec()).unwrap();
        let length: usize = head
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
            .map(|(_, value)| value.trim().parse().unwrap())
            .unwrap_or(0);
        let body_start = end + HEADER_END.len();
        if data.len() >= body_start + length {
            break (head, data[body_start..body_start + length].to_vec());
        }
    };
    let mut lines = head.lines();
    let path = lines
        .next()
        .and_then(|request_line| request_line.split_whitespace().nth(1))
        .unwrap()
        .to_string();
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.to_ascii_lowercase(), value.trim().to_string()))
        .collect();
    state
        .lock()
        .unwrap()
        .ice_config_requests
        .push(IceConfigRequest {
            path,
            headers,
            body: serde_json::from_slice(&body).unwrap(),
        });
    let reply = json!({ "IceServerList": [] }).to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
        reply.len()
    );
    tcp.write_all(response.as_bytes()).await.unwrap();
}

async fn mock_ice_config(listener: TcpListener, state: Arc<Mutex<MockState>>) {
    loop {
        let (tcp, _) = listener.accept().await.unwrap();
        tokio::spawn(serve_ice_config(tcp, state.clone()));
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

async fn publish(sink: &mut KvsWebRtcSink, units: &[Vec<u8>]) {
    let interval = Duration::from_secs(1) / FPS;
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
}

fn assert_presigned(parameters: &HashMap<String, String>) {
    assert_eq!(parameters["X-Amz-ChannelARN"], CHANNEL_ARN);
    assert_eq!(parameters["X-Amz-Algorithm"], "AWS4-HMAC-SHA256");
    assert!(
        parameters["X-Amz-Credential"].starts_with(&format!("{ACCESS_KEY}/")),
        "{parameters:?}"
    );
    assert!(
        parameters["X-Amz-Credential"].ends_with(&format!("/{REGION}/kinesisvideo/aws4_request"))
    );
    assert_eq!(parameters["X-Amz-SignedHeaders"], "host");
    assert_eq!(parameters["X-Amz-Expires"], PRESIGNED_EXPIRES_SECONDS);
    assert_eq!(parameters["X-Amz-Security-Token"], SESSION_TOKEN);
    let signature = &parameters["X-Amz-Signature"];
    assert_eq!(signature.len(), SIGNATURE_HEX_DIGITS);
    assert!(signature.bytes().all(|b| b.is_ascii_hexdigit()));
}

#[tokio::test]
async fn master_streams_to_a_viewer_through_the_signalling_relay() {
    g2g_core::log::init_from_env();
    let state = Arc::new(Mutex::new(MockState::default()));
    let signalling = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let ice_config = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let websocket = format!("ws://{}", signalling.local_addr().unwrap());
    let https = format!("http://{}", ice_config.local_addr().unwrap());
    let signalling_server = tokio::spawn(mock_signalling(signalling, state.clone()));
    let ice_config_server = tokio::spawn(mock_ice_config(ice_config, state.clone()));

    let mut sink = KvsWebRtcSink::new()
        .with_channel_arn(CHANNEL_ARN)
        .with_region(REGION)
        .with_credentials(ACCESS_KEY, SECRET_KEY, Some(SESSION_TOKEN.to_string()))
        .with_endpoints(websocket.clone(), https.clone());
    sink.configure_pipeline(&fixed_caps(VIDEO_CODEC)).unwrap();
    let mut src = KvsWebRtcSrc::new()
        .with_channel_arn(CHANNEL_ARN)
        .with_region(REGION)
        .with_credentials(ACCESS_KEY, SECRET_KEY, Some(SESSION_TOKEN.to_string()))
        .with_endpoints(websocket, https)
        .with_client_id(VIEWER_CLIENT_ID)
        .with_video_codec(VIDEO_CODEC)
        .with_frame_limit(VIDEO_UNITS as u64);
    let units = synthetic_stream(VIDEO_CODEC, VIDEO_UNITS);
    let mut out = CollectPorts::new(src.output_count());

    let (_, emitted) = tokio::time::timeout(RUN_TIMEOUT, async {
        tokio::join!(publish(&mut sink, &units), src.run(&mut out))
    })
    .await
    .expect("the viewer run ends");
    let emitted = emitted.expect("run succeeds");
    signalling_server.abort();
    ice_config_server.abort();
    assert_eq!(sink.frames_sent(), units.len() as u64);

    let state = state.lock().unwrap();
    let [master, viewer] = state.connections.as_slice() else {
        panic!("{:?}", state.connections)
    };
    let (master, viewer) = if master.contains_key("X-Amz-ClientId") {
        (viewer, master)
    } else {
        (master, viewer)
    };
    assert_presigned(master);
    assert_presigned(viewer);
    assert_eq!(viewer["X-Amz-ClientId"], VIEWER_CLIENT_ID);

    let viewer_actions: Vec<&str> = state
        .relayed
        .iter()
        .filter(|r| r.from_viewer)
        .map(|r| r.action.as_str())
        .collect();
    assert_eq!(
        viewer_actions.first(),
        Some(&"SDP_OFFER"),
        "{viewer_actions:?}"
    );
    assert!(
        viewer_actions.contains(&"ICE_CANDIDATE"),
        "{viewer_actions:?}"
    );
    assert!(state
        .relayed
        .iter()
        .filter(|r| r.from_viewer)
        .all(|r| r.recipient.is_none()));
    let master_messages: Vec<&Relayed> = state.relayed.iter().filter(|r| !r.from_viewer).collect();
    assert_eq!(
        master_messages.first().map(|r| r.action.as_str()),
        Some("SDP_ANSWER"),
        "{master_messages:?}"
    );
    assert!(master_messages.iter().any(|r| r.action == "ICE_CANDIDATE"));
    assert!(master_messages
        .iter()
        .all(|r| r.recipient.as_deref() == Some(VIEWER_CLIENT_ID)));

    assert_eq!(state.ice_config_requests.len(), 2);
    for request in &state.ice_config_requests {
        assert_eq!(request.path, ICE_CONFIG_PATH);
        assert_eq!(request.body["ChannelARN"], CHANNEL_ARN);
        assert!(
            request.headers["authorization"]
                .starts_with(&format!("AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/")),
            "{request:?}"
        );
        assert!(request.headers["authorization"]
            .contains("SignedHeaders=host;x-amz-date;x-amz-security-token"));
        assert_eq!(request.headers["x-amz-security-token"], SESSION_TOKEN);
    }

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
    assert_eq!(
        emitted,
        (video_frames.len() + data_frames(audio).len()) as u64
    );
    assert!(
        video_frames.len() >= units.len() * MIN_DELIVERED_PERCENT / 100,
        "{} of {} units arrived",
        video_frames.len(),
        units.len()
    );
    let first_video = video_frames[0];
    assert!(first_video.timing.keyframe, "video starts on a keyframe");
    let first_unit = &units[0];
    assert!(
        frame_bytes(first_video).ends_with(&first_unit[first_unit.len() - UNIT_TAIL_BYTES..]),
        "first video frame is not the first sent keyframe"
    );
}
