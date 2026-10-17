//! Janus VideoRoom publisher: `janusvrsink` joins a room over the Janus
//! WebSocket API (JSON on the `janus-protocol` subprotocol) and publishes one
//! track, any carried video codec or Opus, through a str0m PeerConnection.
//!
//! Signalling follows the Janus core + videoroom plugin API: `create` a
//! session, `attach` the plugin, `join` as a publisher, `publish` with the SDP
//! offer, apply the answer from the `configured` event, then `keepalive` every
//! ten seconds while the media loop runs. Our candidates ride in the offer
//! (`trickle: false`), Janus's in its answer plus any `trickle` events after
//! it. EOS sends `leave` and closes the socket.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::{TcpStream, UdpSocket};
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use str0m::bwe::{Bitrate, BweKind};
use str0m::change::SdpAnswer;
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, Mid, Pt};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_debug, g2g_info, g2g_warn, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, HardwareError, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Reconfigure,
};

use crate::turn::TurnSet;
use crate::webrtc_codec::{sink_caps_alternatives, Track, WebRtcVideoCodec};
use crate::webrtc_stats::{WebRtcStatsHandle, STATS_INTERVAL};
use crate::webrtc_util::{
    add_ice_candidates, drain_pacer, feed_datagram, random_hex_id, select_host_ip, send_transmit,
};
use crate::webrtcsink::{MediaUnit, DEFAULT_QUEUE_DEPTH, INITIAL_BITRATE_BPS};

pub const DEFAULT_ENDPOINT: &str = "ws://127.0.0.1:8188";
const JANUS_SUBPROTOCOL: &str = "janus-protocol";
const VIDEOROOM_PLUGIN: &str = "janus.plugin.videoroom";
// janus drops a session after 60 s without traffic
const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(10);
const SIGNAL_TIMEOUT: Duration = Duration::from_secs(20);
const LEAVE_TIMEOUT: Duration = Duration::from_secs(2);
const LEAVE_ACK_TIMEOUT: Duration = Duration::from_millis(500);
const UDP_RECV_BYTES: usize = 2000;

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;
type WsWrite = SplitSink<Ws, Message>;
type WsRead = SplitStream<Ws>;

/// Janus VideoRoom publisher. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::janusvrsink::JanusVrSink;
///
/// let sink = JanusVrSink::new("ws://janus.example:8188")
///     .with_room("1234")
///     .with_display_name("camera-1");
/// ```
pub struct JanusVrSink {
    endpoint: String,
    room_id: String,
    feed_id: Option<String>,
    display_name: Option<String>,
    api_secret: Option<String>,
    string_ids: bool,
    stun_server: Option<String>,
    track: Track,
    configured: bool,
    tx: Option<mpsc::Sender<MediaUnit>>,
    left: Option<oneshot::Receiver<()>>,
    keyframe_requested: Arc<AtomicBool>,
    bitrate_estimate: Arc<AtomicU64>,
    last_bitrate_sent: u64,
    frames_sent: u64,
    stats: WebRtcStatsHandle,
    log_name: LogName,
}

impl core::fmt::Debug for JanusVrSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("JanusVrSink")
            .field("endpoint", &self.endpoint)
            .field("room_id", &self.room_id)
            .field("frames_sent", &self.frames_sent)
            .finish()
    }
}

impl JanusVrSink {
    /// Publish through the Janus WebSocket API at `endpoint` (`ws://` or `wss://`).
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            room_id: String::new(),
            feed_id: None,
            display_name: None,
            api_secret: None,
            string_ids: false,
            stun_server: None,
            track: Track::Video(WebRtcVideoCodec::default()),
            configured: false,
            tx: None,
            left: None,
            keyframe_requested: Arc::new(AtomicBool::new(false)),
            bitrate_estimate: Arc::new(AtomicU64::new(0)),
            last_bitrate_sent: 0,
            frames_sent: 0,
            stats: WebRtcStatsHandle::default(),
            log_name: LogName::new(),
        }
    }

    /// The VideoRoom to publish into (a number unless `with_string_ids`).
    pub fn with_room(mut self, room_id: impl Into<String>) -> Self {
        self.room_id = room_id.into();
        self
    }

    /// Ask Janus for this publisher id instead of an assigned one.
    pub fn with_feed_id(mut self, feed_id: impl Into<String>) -> Self {
        self.feed_id = Some(feed_id.into());
        self
    }

    /// The display name the room sees.
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// The `apisecret` a server configured with one requires on every request.
    pub fn with_api_secret(mut self, secret: impl Into<String>) -> Self {
        self.api_secret = Some(secret.into());
        self
    }

    /// Send room and feed ids as strings (a Janus built with `string_ids`).
    pub fn with_string_ids(mut self, string_ids: bool) -> Self {
        self.string_ids = string_ids;
        self
    }

    /// Add a server-reflexive candidate from this STUN server (`host:port`).
    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.stun_server = Some(server.into());
        self
    }

    /// Access units handed to the session so far.
    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// Live counters of the publisher's connection, readable while it runs.
    pub fn stats_handle(&self) -> WebRtcStatsHandle {
        self.stats.clone()
    }

    async fn start_session(&mut self) -> Result<(), G2gError> {
        match self.signal_and_publish().await {
            Ok(()) => Ok(()),
            Err(reason) => {
                g2g_warn!(self, "{reason}");
                Err(G2gError::Hardware(HardwareError::Other))
            }
        }
    }

    async fn signal_and_publish(&mut self) -> Result<(), String> {
        let room = id_value(&self.room_id, self.string_ids).ok_or_else(|| {
            format!(
                "room-id {:?} is not a number (set use-string-ids for a string room)",
                self.room_id
            )
        })?;
        let feed = match &self.feed_id {
            Some(feed_id) => Some(
                id_value(feed_id, self.string_ids)
                    .ok_or_else(|| format!("feed-id {feed_id:?} is not a number"))?,
            ),
            None => None,
        };
        let secret = self.api_secret.as_deref();

        let mut request = self
            .endpoint
            .as_str()
            .into_client_request()
            .map_err(|e| format!("janus-endpoint {}: {e}", self.endpoint))?;
        request.headers_mut().insert(
            "Sec-WebSocket-Protocol",
            JANUS_SUBPROTOCOL
                .parse()
                .map_err(|_| "subprotocol header".to_string())?,
        );
        let (mut ws, _) = tokio::time::timeout(SIGNAL_TIMEOUT, connect_async(request))
            .await
            .map_err(|_| format!("connect to {} timed out", self.endpoint))?
            .map_err(|e| format!("connect to {} failed: {e}", self.endpoint))?;

        send_json(&mut ws, envelope("create", None, None, secret)).await?;
        let session_id = wait_for(&mut ws, success_id).await?;
        let mut attach = envelope("attach", Some(session_id), None, secret);
        attach["plugin"] = json!(VIDEOROOM_PLUGIN);
        send_json(&mut ws, attach).await?;
        let handle_id = wait_for(&mut ws, success_id).await?;
        let signaller = Signaller {
            session_id,
            handle_id,
            api_secret: self.api_secret.clone(),
        };

        let mut join = json!({ "request": "join", "ptype": "publisher", "room": room });
        if let Some(feed) = feed {
            join["id"] = feed;
        }
        if let Some(name) = &self.display_name {
            join["display"] = json!(name);
        }
        send_json(&mut ws, signaller.plugin_message(join, None)).await?;
        let joined = wait_for(&mut ws, |m| {
            (plugin_data(m)["videoroom"] == "joined").then(|| plugin_data(m)["id"].clone())
        })
        .await?;
        g2g_info!(self, "joined room {} as publisher {joined}", self.room_id);

        let socket = UdpSocket::bind((select_host_ip(), 0))
            .await
            .map_err(|e| format!("udp bind: {e}"))?;
        let local = socket.local_addr().map_err(|e| format!("udp bind: {e}"))?;
        let config = RtcConfig::new()
            .set_crypto_provider(Arc::new(from_feature_flags()))
            .enable_bwe(Some(Bitrate::bps(INITIAL_BITRATE_BPS)))
            .set_stats_interval(Some(STATS_INTERVAL))
            .clear_codecs();
        let mut rtc = self.track.enable(config).build(Instant::now());
        // every candidate rides in the offer, so gather before it
        add_ice_candidates(&mut rtc, &socket, self.stun_server.as_deref())
            .await
            .map_err(|_| "ice candidate gathering failed".to_string())?;
        let (offer_sdp, pending, mid) = {
            let mut api = rtc.sdp_api();
            let mid = api.add_media(
                self.track.media_kind(),
                Direction::SendOnly,
                None,
                None,
                None,
            );
            let (offer, pending) = api.apply().ok_or("offer creation failed")?;
            (offer.to_sdp_string(), pending, mid)
        };
        let jsep = json!({ "type": "offer", "sdp": offer_sdp, "trickle": false });
        send_json(
            &mut ws,
            signaller.plugin_message(json!({ "request": "publish" }), Some(jsep)),
        )
        .await?;
        let answer_sdp = wait_for(&mut ws, |m| {
            (plugin_data(m)["configured"] == "ok" && m["jsep"]["type"] == "answer")
                .then(|| m["jsep"]["sdp"].as_str().map(str::to_string))
                .flatten()
        })
        .await?;
        let answer = SdpAnswer::from_sdp_string(&answer_sdp)
            .map_err(|e| format!("answer sdp: {e:?}\n{answer_sdp}"))?;
        rtc.sdp_api()
            .accept_answer(pending, answer)
            .map_err(|e| format!("answer rejected: {e:?}\n{answer_sdp}"))?;
        // a room whose codec list excludes ours answers the m-line disabled
        let direction = rtc.media(mid).map(|m| m.direction());
        if direction != Some(Direction::SendOnly) {
            return Err(format!(
                "janus answered our {:?} m-line as {direction:?}: the room does not take this codec",
                self.track.codec()
            ));
        }
        g2g_debug!(self, "published, rtc socket {local}");

        let (ws_write, ws_read) = ws.split();
        let (tx, rx) = mpsc::channel(DEFAULT_QUEUE_DEPTH);
        let (left_tx, left_rx) = oneshot::channel();
        tokio::spawn(run_session(SessionArgs {
            rtc,
            socket,
            local,
            mid,
            track: self.track,
            signaller,
            ws_write,
            ws_read,
            keyframe_requested: Arc::clone(&self.keyframe_requested),
            bitrate_estimate: Arc::clone(&self.bitrate_estimate),
            stats: self.stats.clone(),
            rx,
            left: Some(left_tx),
        }));
        self.tx = Some(tx);
        self.left = Some(left_rx);
        Ok(())
    }
}

/// A room or feed id in the JSON shape the server expects.
fn id_value(id: &str, string_ids: bool) -> Option<Value> {
    if string_ids {
        return Some(Value::String(id.to_string()));
    }
    id.parse::<u64>().ok().map(Value::from)
}

/// The core request envelope: kind, a fresh transaction, and the ids a
/// request past `create` / `attach` carries.
fn envelope(
    kind: &str,
    session_id: Option<u64>,
    handle_id: Option<u64>,
    secret: Option<&str>,
) -> Value {
    let mut message = json!({ "janus": kind, "transaction": random_hex_id() });
    if let Some(session_id) = session_id {
        message["session_id"] = json!(session_id);
    }
    if let Some(handle_id) = handle_id {
        message["handle_id"] = json!(handle_id);
    }
    if let Some(secret) = secret {
        message["apisecret"] = json!(secret);
    }
    message
}

/// The attached session + plugin handle every later request names.
#[derive(Debug, Clone)]
struct Signaller {
    session_id: u64,
    handle_id: u64,
    api_secret: Option<String>,
}

impl Signaller {
    fn envelope(&self, kind: &str) -> Value {
        envelope(
            kind,
            Some(self.session_id),
            Some(self.handle_id),
            self.api_secret.as_deref(),
        )
    }

    /// A videoroom plugin request (`body`), with a JSEP when it carries SDP.
    fn plugin_message(&self, body: Value, jsep: Option<Value>) -> Value {
        let mut message = self.envelope("message");
        message["body"] = body;
        if let Some(jsep) = jsep {
            message["jsep"] = jsep;
        }
        message
    }

    fn keepalive(&self) -> Value {
        envelope(
            "keepalive",
            Some(self.session_id),
            None,
            self.api_secret.as_deref(),
        )
    }
}

fn plugin_data(message: &Value) -> &Value {
    &message["plugindata"]["data"]
}

fn success_id(message: &Value) -> Option<u64> {
    (message["janus"] == "success")
        .then(|| message["data"]["id"].as_u64())
        .flatten()
}

/// Why the server ended things, when `message` says so.
fn failure(message: &Value) -> Option<String> {
    match message["janus"].as_str() {
        Some("error") => Some(format!(
            "janus error {}: {}",
            message["error"]["code"], message["error"]["reason"]
        )),
        Some("hangup") => Some(format!("janus hangup: {}", message["reason"])),
        _ => {
            let data = plugin_data(message);
            data.get("error_code")
                .map(|code| format!("videoroom error {code}: {}", data["error"]))
        }
    }
}

async fn send_json(ws: &mut Ws, message: Value) -> Result<(), String> {
    ws.send(Message::Text(message.to_string()))
        .await
        .map_err(|e| format!("websocket send: {e}"))
}

async fn recv_json(ws: &mut Ws) -> Result<Value, String> {
    loop {
        let frame = ws
            .next()
            .await
            .ok_or("websocket closed")?
            .map_err(|e| format!("websocket: {e}"))?;
        if let Some(message) = parse_frame(&frame)? {
            return Ok(message);
        }
    }
}

/// The JSON of a text or binary frame; control frames give `None`.
fn parse_frame(frame: &Message) -> Result<Option<Value>, String> {
    let bytes: &[u8] = match frame {
        Message::Text(text) => text.as_bytes(),
        Message::Binary(bytes) => bytes,
        Message::Close(_) => return Err("websocket closed".to_string()),
        _ => return Ok(None),
    };
    serde_json::from_slice(bytes)
        .map(Some)
        .map_err(|e| format!("janus sent invalid json: {e}"))
}

/// Read until `accept` yields, failing on a server error or the timeout.
async fn wait_for<T>(
    ws: &mut Ws,
    mut accept: impl FnMut(&Value) -> Option<T>,
) -> Result<T, String> {
    let deadline = tokio::time::Instant::now() + SIGNAL_TIMEOUT;
    loop {
        let message = tokio::time::timeout_at(deadline, recv_json(ws))
            .await
            .map_err(|_| "janus did not answer in time".to_string())??;
        if let Some(reason) = failure(&message) {
            return Err(reason);
        }
        if let Some(value) = accept(&message) {
            return Ok(value);
        }
    }
}

/// A `trickle` event's candidate line, `None` for anything else (including the
/// end-of-candidates marker).
fn trickle_candidate(message: &Value) -> Option<Candidate> {
    if message["janus"] != "trickle" {
        return None;
    }
    let line = message["candidate"]["candidate"].as_str()?;
    Candidate::from_sdp_string(line).ok()
}

struct SessionArgs {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    mid: Mid,
    track: Track,
    signaller: Signaller,
    ws_write: WsWrite,
    ws_read: WsRead,
    keyframe_requested: Arc<AtomicBool>,
    bitrate_estimate: Arc<AtomicU64>,
    stats: WebRtcStatsHandle,
    rx: mpsc::Receiver<MediaUnit>,
    left: Option<oneshot::Sender<()>>,
}

/// The sans-IO loop: owns the `Rtc`, the UDP socket and the WebSocket. Drains
/// `poll_output`, writes each `MediaUnit`, applies server trickle, keeps the
/// session alive, and leaves the room when the unit channel closes.
async fn run_session(mut a: SessionArgs) {
    let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
    let mut turn = TurnSet::empty();
    let mut pt: Option<Pt> = None;
    let mut keepalive_at = Instant::now() + KEEPALIVE_INTERVAL;
    loop {
        let deadline = loop {
            match a.rtc.poll_output() {
                Ok(Output::Timeout(t)) => break t,
                Ok(Output::Transmit(t)) => send_transmit(&a.socket, &mut turn, &t).await,
                Ok(Output::Event(Event::KeyframeRequest(_))) => {
                    a.keyframe_requested.store(true, Ordering::Relaxed);
                }
                Ok(Output::Event(Event::EgressBitrateEstimate(kind))) => {
                    let bps = match kind {
                        BweKind::Twcc(b) | BweKind::Remb(_, b) => Some(b.as_u64()),
                        // `BweKind` is non-exhaustive; ignore unknown future kinds.
                        _ => None,
                    };
                    if let Some(bps) = bps {
                        a.bitrate_estimate.store(bps, Ordering::Relaxed);
                    }
                }
                Ok(Output::Event(Event::IceConnectionStateChange(
                    IceConnectionState::Disconnected,
                ))) => return leave(&mut a).await,
                Ok(Output::Event(event)) => a.stats.record(&event),
                Err(_) => return leave(&mut a).await,
            }
        };
        let timeout = deadline.saturating_duration_since(Instant::now());
        tokio::select! {
            r = a.socket.recv_from(&mut buf) => {
                let Ok((n, source)) = r else { return leave(&mut a).await };
                if !feed_datagram(&mut a.rtc, &mut turn, a.local, &buf[..n], source) {
                    return leave(&mut a).await;
                }
            }
            frame = a.ws_read.next() => {
                let Some(Ok(frame)) = frame else { return finish(&mut a) };
                let Ok(Some(message)) = parse_frame(&frame) else { return finish(&mut a) };
                if failure(&message).is_some() {
                    return finish(&mut a);
                }
                if let Some(candidate) = trickle_candidate(&message) {
                    a.rtc.add_remote_candidate(candidate);
                }
            }
            unit = a.rx.recv() => {
                let Some(unit) = unit else {
                    drain_pacer(&mut a.rtc, &a.socket, &mut turn).await;
                    return leave(&mut a).await;
                };
                if pt.is_none() {
                    if let Some(writer) = a.rtc.writer(a.mid) {
                        pt = writer
                            .payload_params()
                            .find(|p| p.spec().codec == a.track.codec())
                            .map(|p| p.pt());
                    }
                }
                if let Some(p) = pt {
                    let rtp_time = a.track.media_time(unit.pts_ns);
                    if let Some(writer) = a.rtc.writer(a.mid) {
                        let _ = writer.write(p, Instant::now(), rtp_time, unit.data);
                    }
                }
            }
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(keepalive_at)) => {
                let keepalive = a.signaller.keepalive().to_string();
                if a.ws_write.send(Message::Text(keepalive)).await.is_err() {
                    return finish(&mut a);
                }
                keepalive_at = Instant::now() + KEEPALIVE_INTERVAL;
            }
            _ = tokio::time::sleep(timeout) => {
                if a.rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                    return leave(&mut a).await;
                }
            }
        }
    }
}

/// Tell the room we are gone, wait briefly for its `leaving` reply, close the
/// socket, and release the element's EOS.
async fn leave(a: &mut SessionArgs) {
    let leave = a
        .signaller
        .plugin_message(json!({ "request": "leave" }), None)
        .to_string();
    if a.ws_write.send(Message::Text(leave)).await.is_ok() {
        let deadline = tokio::time::Instant::now() + LEAVE_ACK_TIMEOUT;
        while let Ok(Some(Ok(frame))) = tokio::time::timeout_at(deadline, a.ws_read.next()).await {
            let left =
                matches!(parse_frame(&frame), Ok(Some(m)) if plugin_data(&m)["leaving"] == "ok");
            if left {
                break;
            }
        }
    }
    let _ = a.ws_write.close().await;
    finish(a);
}

fn finish(a: &mut SessionArgs) {
    if let Some(left) = a.left.take() {
        let _ = left.send(());
    }
}

static JANUSVRSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "janus-endpoint",
        PropKind::Str,
        "Janus WebSocket endpoint (ws:// or wss://)",
    )
    .with_default(DEFAULT_ENDPOINT),
    PropertySpec::new(
        "room-id",
        PropKind::Str,
        "VideoRoom to publish into (a number unless use-string-ids)",
    ),
    PropertySpec::new(
        "feed-id",
        PropKind::Str,
        "publisher id to request (empty = Janus assigns one)",
    ),
    PropertySpec::new("display-name", PropKind::Str, "display name the room sees"),
    PropertySpec::new(
        "secret-key",
        PropKind::Str,
        "Janus API secret (apisecret) when the server requires one",
    ),
    PropertySpec::new(
        "use-string-ids",
        PropKind::Bool,
        "send room-id and feed-id as strings (Janus string_ids)",
    )
    .with_default("false"),
    PropertySpec::new(
        "stun-server",
        PropKind::Str,
        "STUN server host:port for a server-reflexive candidate (empty = host only)",
    ),
];

impl AsyncElement for JanusVrSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Track::from_caps(upstream_caps)
            .map(|_| upstream_caps.clone())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(sink_caps_alternatives()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.track = Track::from_caps(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Janus VideoRoom sink",
            "Sink/Network/WebRTC",
            "Publishes video (H.264/VP8/VP9/H.265/AV1) or Opus into a Janus VideoRoom over WebRTC (str0m: ICE/DTLS/SRTP)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        JANUSVRSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name == "use-string-ids" {
            self.string_ids = value.as_bool().ok_or(PropError::Type)?;
            return Ok(());
        }
        let v = value.as_str().ok_or(PropError::Type)?;
        let optional = || (!v.is_empty()).then(|| v.into());
        match name {
            "janus-endpoint" => self.endpoint = v.into(),
            "room-id" => self.room_id = v.into(),
            "feed-id" => self.feed_id = optional(),
            "display-name" => self.display_name = optional(),
            "secret-key" => self.api_secret = optional(),
            "stun-server" => self.stun_server = optional(),
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let text = |value: &Option<String>| PropValue::Str(value.clone().unwrap_or_default());
        match name {
            "janus-endpoint" => Some(PropValue::Str(self.endpoint.clone())),
            "room-id" => Some(PropValue::Str(self.room_id.clone())),
            "feed-id" => Some(text(&self.feed_id)),
            "display-name" => Some(text(&self.display_name)),
            "secret-key" => Some(text(&self.api_secret)),
            "use-string-ids" => Some(PropValue::Bool(self.string_ids)),
            "stun-server" => Some(text(&self.stun_server)),
            _ => None,
        }
    }

    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        self.keyframe_requested
            .swap(false, Ordering::Relaxed)
            .then_some(Reconfigure::ForceKeyframe)
    }

    fn take_bitrate(&mut self) -> Option<u32> {
        let current = self.bitrate_estimate.load(Ordering::Relaxed);
        if current == 0 || current == self.last_bitrate_sent {
            return None;
        }
        self.last_bitrate_sent = current;
        Some(current.min(u32::MAX as u64) as u32)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(short_type_name::<Self>())?;
                    let unit = MediaUnit {
                        pts_ns: frame.timing.pts_ns,
                        data: slice.to_vec(),
                    };
                    if self.tx.is_none() {
                        self.start_session().await?;
                    }
                    if let Some(tx) = &self.tx {
                        // a closed channel means the session ended
                        tx.send(unit).await.map_err(|_| G2gError::Shutdown)?;
                    }
                    self.frames_sent += 1;
                }
                // closing the unit channel makes the task flush, leave and signal
                PipelinePacket::Eos => {
                    self.tx = None;
                    if let Some(left) = self.left.take() {
                        let _ = tokio::time::timeout(LEAVE_TIMEOUT, left).await;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }
}

impl LogSource for JanusVrSink {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_follow_the_string_ids_switch() {
        assert_eq!(id_value("1234", false), Some(json!(1234)));
        assert_eq!(id_value("1234", true), Some(json!("1234")));
        assert_eq!(id_value("lobby", false), None);
        assert_eq!(id_value("lobby", true), Some(json!("lobby")));
    }

    #[test]
    fn requests_carry_the_session_handle_and_secret() {
        let signaller = Signaller {
            session_id: 7,
            handle_id: 9,
            api_secret: Some("s3".into()),
        };
        let publish = signaller.plugin_message(
            json!({ "request": "publish" }),
            Some(json!({ "type": "offer", "sdp": "v=0", "trickle": false })),
        );
        assert_eq!(publish["janus"], "message");
        assert_eq!(publish["session_id"], 7);
        assert_eq!(publish["handle_id"], 9);
        assert_eq!(publish["apisecret"], "s3");
        assert_eq!(publish["body"]["request"], "publish");
        assert_eq!(publish["jsep"]["type"], "offer");
        assert_eq!(publish["transaction"].as_str().map(str::len), Some(16));
        let keepalive = signaller.keepalive();
        assert_eq!(keepalive["janus"], "keepalive");
        assert_eq!(keepalive["session_id"], 7);
        assert!(keepalive.get("handle_id").is_none());
        let create = envelope("create", None, None, None);
        assert!(create.get("session_id").is_none());
        assert!(create.get("apisecret").is_none());
    }

    #[test]
    fn server_failures_are_named() {
        assert_eq!(
            failure(
                &json!({ "janus": "error", "error": { "code": 403, "reason": "Unauthorized request" } })
            ),
            Some("janus error 403: \"Unauthorized request\"".to_string())
        );
        assert_eq!(
            failure(&json!({ "janus": "hangup", "reason": "DTLS alert" })),
            Some("janus hangup: \"DTLS alert\"".to_string())
        );
        let room_error = json!({ "janus": "event", "plugindata": { "plugin": VIDEOROOM_PLUGIN,
            "data": { "videoroom": "event", "error_code": 426, "error": "No such room" } } });
        assert_eq!(
            failure(&room_error),
            Some("videoroom error 426: \"No such room\"".to_string())
        );
        assert_eq!(failure(&json!({ "janus": "ack" })), None);
        assert_eq!(failure(&json!({ "janus": "webrtcup", "sender": 9 })), None);
    }

    #[test]
    fn success_and_trickle_messages_parse() {
        assert_eq!(
            success_id(&json!({ "janus": "success", "data": { "id": 42 } })),
            Some(42)
        );
        assert_eq!(success_id(&json!({ "janus": "ack" })), None);
        let trickle = json!({ "janus": "trickle", "sender": 9, "candidate": {
            "candidate": "candidate:1 1 udp 2130706431 192.0.2.1 5000 typ host", "sdpMid": "0", "sdpMLineIndex": 0 } });
        assert!(trickle_candidate(&trickle).is_some());
        let completed = json!({ "janus": "trickle", "candidate": { "completed": true } });
        assert!(trickle_candidate(&completed).is_none());
        assert!(trickle_candidate(&json!({ "janus": "ack" })).is_none());
    }

    #[test]
    fn properties_round_trip() {
        let mut sink = JanusVrSink::new(DEFAULT_ENDPOINT);
        for (name, value) in [
            ("janus-endpoint", "wss://janus.example/ws"),
            ("room-id", "1234"),
            ("feed-id", "77"),
            ("display-name", "camera-1"),
            ("secret-key", "s3"),
            ("stun-server", "stun.example.com:3478"),
        ] {
            sink.set_property(name, PropValue::Str(value.into()))
                .unwrap();
            assert_eq!(sink.get_property(name), Some(PropValue::Str(value.into())));
        }
        sink.set_property("use-string-ids", PropValue::Bool(true))
            .unwrap();
        assert_eq!(
            sink.get_property("use-string-ids"),
            Some(PropValue::Bool(true))
        );
        assert_eq!(
            sink.set_property("room-id", PropValue::Bool(true)),
            Err(PropError::Type)
        );
        for spec in sink.properties() {
            assert!(sink.get_property(spec.name).is_some(), "{}", spec.name);
        }
    }

    #[test]
    fn launch_line_sets_the_properties() {
        use g2g_core::runtime::parse_launch;

        let registry = crate::registry::default_registry();
        parse_launch(
            &registry,
            "videotestsrc num-buffers=1 ! janusvrsink janus-endpoint=ws://127.0.0.1:8188 room-id=1234 display-name=g2g use-string-ids=false",
        )
        .expect("launch line parses")
        .finish()
        .expect("valid graph");
    }

    #[test]
    fn accepts_every_carried_codec_and_opus() {
        let sink = JanusVrSink::new(DEFAULT_ENDPOINT);
        for caps in sink_caps_alternatives() {
            assert!(sink.intercept_caps(&caps).is_ok(), "{caps:?}");
        }
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: g2g_core::Dim::Fixed(640),
            height: g2g_core::Dim::Fixed(480),
            framerate: g2g_core::Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(sink.intercept_caps(&raw).is_err());
    }
}
