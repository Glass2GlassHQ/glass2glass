//! Janus VideoRoom publisher: `janusvrsink` joins a room over the Janus
//! WebSocket API (JSON on the `janus-protocol` subprotocol) and publishes one
//! track, any carried video codec or Opus, through a str0m PeerConnection.
//!
//! Signalling follows the Janus core + videoroom plugin API: `create` a
//! session, `attach` the plugin, `join` as a publisher, `publish` with the SDP
//! offer, apply the answer from the `configured` event, then `keepalive` every
//! ten seconds while the media loop runs. Our candidates ride in the offer
//! (`trickle: false`), Janus's in its answer plus any `trickle` events after
//! it. EOS sends `unpublish` then `leave` and closes the socket.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde_json::json;
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, oneshot};

use str0m::bwe::{Bitrate, BweKind};
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, Mid, Pt};
use str0m::{Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_debug, g2g_info, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Reconfigure,
};

pub use crate::janus_signal::DEFAULT_ENDPOINT;
use crate::janus_signal::{
    failure, id_value, open_session, parse_frame, plugin_data, send_json, trickle_candidate,
    wait_for, Signaller, WsRead, WsWrite, KEEPALIVE_INTERVAL,
};
use crate::turn::TurnSet;
use crate::webrtc_codec::{
    enable_tracks, frame_capture_time, sink_caps_alternatives, stamp_capture_time, Track,
    WebRtcVideoCodec,
};
use crate::webrtc_receive::UDP_RECV_BYTES;
use crate::webrtc_stats::{WebRtcStatsHandle, STATS_INTERVAL};
use crate::webrtc_util::{
    accept_answer_sdp, add_ice_candidates, connect_failed, drain_pacer, feed_datagram,
    select_host_ip, send_transmit, STUN_SERVER_PROP,
};
use crate::webrtcsink::{MediaUnit, DEFAULT_QUEUE_DEPTH, INITIAL_BITRATE_BPS};

const LEAVE_TIMEOUT: Duration = Duration::from_secs(2);

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
            Err(reason) => Err(connect_failed(self, &reason)),
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
        let (mut ws, signaller) = open_session(&self.endpoint, self.api_secret.as_deref()).await?;

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
        let mut rtc = enable_tracks(config, [self.track]).build(Instant::now());
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
        accept_answer_sdp(&mut rtc, pending, &answer_sdp)?;
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
                        let _ = stamp_capture_time(writer, unit.capture_time)
                            .write(p, Instant::now(), rtp_time, unit.data);
                    }
                }
            }
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(keepalive_at)) => {
                if send_json(&mut a.ws_write, a.signaller.keepalive()).await.is_err() {
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

async fn leave(a: &mut SessionArgs) {
    // janus tells subscribers the feed ended on unpublish, never on a bare leave
    let unpublish = a
        .signaller
        .plugin_message(json!({ "request": "unpublish" }), None);
    let _ = send_json(&mut a.ws_write, unpublish).await;
    crate::janus_signal::leave(&mut a.ws_write, &mut a.ws_read, &a.signaller).await;
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
    STUN_SERVER_PROP,
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
                        capture_time: frame_capture_time(&frame),
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
