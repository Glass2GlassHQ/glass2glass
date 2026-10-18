use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use tokio::net::UdpSocket;
use tokio::sync::{mpsc, oneshot};

use str0m::bwe::{Bitrate, BweKind};
use str0m::change::SdpOffer;
use str0m::crypto::from_feature_flags;
use str0m::media::{MediaKind, Mid, Pt};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_debug, g2g_info, g2g_warn, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, G2gError, HardwareError, OutputSink, PipelinePacket, PropError, PropValue,
    PropertySpec, Reconfigure,
};

use crate::kvs_signal::{
    candidate_payloads, connect, connect_url, ice_servers, receive, sdp_payload, send_message,
    Action, Credentials, KvsSettings, Received, Role, SignalMessage, SignalingChannel,
    SignalingEndpoints, WsRead, WsWrite, ACCESS_KEY_PROP, ANSWER_TYPE, CHANNEL_ARN_PROP,
    CHANNEL_NAME_PROP, KVS_STUN_SERVER_PROP, REGION_PROP, SECRET_KEY_PROP, SESSION_TOKEN_PROP,
};
use crate::turn::{TurnSet, REFRESH_INTERVAL};
use crate::webrtc_codec::{enable_tracks, sink_caps_alternatives, Track, WebRtcVideoCodec};
use crate::webrtc_stats::{WebRtcStatsHandle, STATS_INTERVAL};
use crate::webrtc_util::{
    drain_pacer, feed_datagram, select_host_ip, send_transmit, IceTransportPolicy,
    ICE_TRANSPORT_POLICY_PROP,
};
use crate::webrtcsink::{MediaUnit, DEFAULT_QUEUE_DEPTH, INITIAL_BITRATE_BPS};

const LEAVE_TIMEOUT: Duration = Duration::from_secs(2);
// the service closes an idle socket after 10 minutes and any socket after an hour
const RECONNECT_DELAY: Duration = Duration::from_secs(2);
const UDP_RECV_BYTES: usize = 2000;
const REJECTED_PORT: &str = "0";
const SENDING_DIRECTIONS: [&str; 2] = ["a=sendonly", "a=sendrecv"];
const MID_LINE_PREFIX: &str = "a=mid:";

pub struct KvsWebRtcSink {
    settings: KvsSettings,
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

impl core::fmt::Debug for KvsWebRtcSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KvsWebRtcSink")
            .field("settings", &self.settings)
            .field("frames_sent", &self.frames_sent)
            .finish()
    }
}

impl Default for KvsWebRtcSink {
    fn default() -> Self {
        Self::new()
    }
}

impl KvsWebRtcSink {
    pub fn new() -> Self {
        Self {
            settings: KvsSettings::default(),
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

    pub fn with_channel_name(mut self, name: impl Into<String>) -> Self {
        self.settings.set_channel_name(name);
        self
    }

    pub fn with_channel_arn(mut self, arn: impl Into<String>) -> Self {
        self.settings.set_channel_arn(arn);
        self
    }

    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.settings.set_region(region);
        self
    }

    pub fn with_credentials(
        mut self,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
        session_token: Option<String>,
    ) -> Self {
        self.settings
            .set_credentials(access_key, secret_key, session_token);
        self
    }

    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.settings.stun_server = Some(server.into());
        self
    }

    pub fn with_ice_transport_policy(mut self, policy: IceTransportPolicy) -> Self {
        self.settings.transport_policy = policy;
        self
    }

    // skips the control-plane lookups
    pub fn with_endpoints(
        mut self,
        websocket: impl Into<String>,
        https: impl Into<String>,
    ) -> Self {
        self.settings.set_endpoints(SignalingEndpoints {
            websocket: websocket.into(),
            https: https.into(),
        });
        self
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    pub fn stats_handle(&self) -> WebRtcStatsHandle {
        self.stats.clone()
    }

    async fn start_session(&mut self) -> Result<(), G2gError> {
        match self.connect_as_master().await {
            Ok(()) => Ok(()),
            Err(reason) => {
                g2g_warn!(self, "{reason}");
                Err(G2gError::Hardware(HardwareError::Other))
            }
        }
    }

    async fn connect_as_master(&mut self) -> Result<(), String> {
        let credentials = self.settings.credentials()?;
        let region = self.settings.region()?;
        let channel = self
            .settings
            .resolve(Role::Master, &credentials, &region)
            .await?;
        let url = connect_url(&credentials, &region, &channel, None)?;
        let ws = connect(&url).await?;
        g2g_info!(
            self,
            "connected to {} as master, waiting for a viewer",
            channel.arn
        );
        let (tx, rx) = mpsc::channel(DEFAULT_QUEUE_DEPTH);
        let (left_tx, left_rx) = oneshot::channel();
        tokio::spawn(run_master(Master {
            settings: self.settings.clone(),
            credentials,
            region,
            channel,
            track: self.track,
            signalling: Some(ws.split()),
            reconnect_at: None,
            rx,
            keyframe_requested: Arc::clone(&self.keyframe_requested),
            bitrate_estimate: Arc::clone(&self.bitrate_estimate),
            stats: self.stats.clone(),
            left: Some(left_tx),
            log: MasterLog(self.log_name.clone()),
        }));
        self.tx = Some(tx);
        self.left = Some(left_rx);
        Ok(())
    }
}

struct MasterLog(LogName);

impl LogSource for MasterLog {
    fn log_category(&self) -> &'static str {
        short_type_name::<KvsWebRtcSink>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.0.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.0.category()
    }
}

type Signalling = (WsWrite, WsRead);

struct Master {
    settings: KvsSettings,
    credentials: Credentials,
    region: String,
    channel: SignalingChannel,
    track: Track,
    signalling: Option<Signalling>,
    reconnect_at: Option<Instant>,
    rx: mpsc::Receiver<MediaUnit>,
    keyframe_requested: Arc<AtomicBool>,
    bitrate_estimate: Arc<AtomicU64>,
    stats: WebRtcStatsHandle,
    left: Option<oneshot::Sender<()>>,
    log: MasterLog,
}

struct Session {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    turn: TurnSet,
    mid: Mid,
    pt: Pt,
    viewer: String,
}

enum Waited {
    Offer {
        viewer: String,
        sdp: String,
        candidates: Vec<Candidate>,
    },
    InputEnded,
}

#[derive(PartialEq, Eq)]
enum SessionEnd {
    ViewerLeft,
    InputEnded,
}

// MediaAdded comes too late for the first frame, the answer names the mid up front
fn sending_mid(answer_sdp: &str, kind: MediaKind) -> Option<Mid> {
    let kind_token = match kind {
        MediaKind::Video => "video",
        MediaKind::Audio => "audio",
    };
    answer_sdp.split("\nm=").skip(1).find_map(|section| {
        let mut lines = section.lines().map(str::trim_end);
        let mut media_line = lines.next()?.split_whitespace();
        let wanted =
            media_line.next() == Some(kind_token) && media_line.next() != Some(REJECTED_PORT);
        let attributes: Vec<&str> = lines.collect();
        let sends = attributes
            .iter()
            .any(|line| SENDING_DIRECTIONS.contains(line));
        let mid = attributes
            .iter()
            .find_map(|line| line.strip_prefix(MID_LINE_PREFIX))?;
        (wanted && sends).then(|| Mid::from(mid))
    })
}

async fn run_master(mut m: Master) {
    loop {
        let Waited::Offer {
            viewer,
            sdp,
            candidates,
        } = wait_for_offer(&mut m).await
        else {
            return leave(&mut m).await;
        };
        g2g_info!(&m.log, "offer from viewer {viewer}");
        let session = match answer(&mut m, &viewer, &sdp, candidates).await {
            Ok(session) => session,
            Err(reason) => {
                g2g_warn!(&m.log, "viewer {viewer}: {reason}");
                continue;
            }
        };
        // a viewer joining mid-stream waits for the next keyframe
        m.keyframe_requested.store(true, Ordering::Relaxed);
        if run_session(&mut m, session).await == SessionEnd::InputEnded {
            return leave(&mut m).await;
        }
        g2g_info!(&m.log, "waiting for the next viewer");
    }
}

// pending while the socket is down, so a select branch can sit on it
async fn next_signal(signalling: &mut Option<Signalling>) -> Received {
    match signalling {
        Some((_, read)) => receive(read.next().await),
        None => core::future::pending().await,
    }
}

fn wake(at: Option<Instant>) -> tokio::time::Sleep {
    tokio::time::sleep_until(tokio::time::Instant::from_std(
        at.unwrap_or_else(Instant::now),
    ))
}

fn drop_signalling(m: &mut Master, reason: &str) {
    g2g_info!(&m.log, "{reason}, reconnecting in {RECONNECT_DELAY:?}");
    m.signalling = None;
    m.reconnect_at = Some(Instant::now() + RECONNECT_DELAY);
}

async fn reconnect(m: &mut Master) {
    m.reconnect_at = None;
    let connected = match connect_url(&m.credentials, &m.region, &m.channel, None) {
        Ok(url) => connect(&url).await,
        Err(reason) => Err(reason),
    };
    match connected {
        Ok(ws) => {
            g2g_info!(&m.log, "reconnected to {} as master", m.channel.arn);
            m.signalling = Some(ws.split());
        }
        Err(reason) => {
            g2g_warn!(&m.log, "{reason}, retrying in {RECONNECT_DELAY:?}");
            m.reconnect_at = Some(Instant::now() + RECONNECT_DELAY);
        }
    }
}

// handles what both loops treat alike, hands back the offer and candidate messages
fn on_signal(m: &mut Master, received: Received) -> Option<SignalMessage> {
    match received {
        Received::Signal(SignalMessage::Status(status)) => {
            g2g_warn!(&m.log, "signalling status: {status}");
        }
        Received::Signal(SignalMessage::GoAway) => {
            drop_signalling(m, "signalling service sent GO_AWAY");
        }
        Received::Signal(SignalMessage::ReconnectIceServer) => {
            g2g_debug!(
                &m.log,
                "relay servers are going away, the next viewer gets a fresh ice config"
            );
        }
        Received::Signal(other) => return Some(other),
        Received::Skipped(reason) => g2g_debug!(&m.log, "skipped a message: {reason}"),
        Received::Control => {}
        Received::Closed(reason) => drop_signalling(m, &reason),
    }
    None
}

// media units arriving while no viewer is connected are dropped
async fn wait_for_offer(m: &mut Master) -> Waited {
    let mut early_candidates: Vec<(Option<String>, Candidate)> = Vec::new();
    loop {
        tokio::select! {
            received = next_signal(&mut m.signalling) => match on_signal(m, received) {
                Some(SignalMessage::Offer { sender, sdp }) => {
                    let candidates = early_candidates
                        .into_iter()
                        .filter(|(from, _)| from.as_deref() == Some(sender.as_str()))
                        .map(|(_, candidate)| candidate)
                        .collect();
                    return Waited::Offer { viewer: sender, sdp, candidates };
                }
                Some(SignalMessage::Candidate { sender, candidate }) => {
                    early_candidates.push((sender, candidate));
                }
                _ => {}
            },
            unit = m.rx.recv() => {
                if unit.is_none() {
                    return Waited::InputEnded;
                }
            }
            _ = wake(m.reconnect_at), if m.reconnect_at.is_some() => reconnect(m).await,
        }
    }
}

async fn answer(
    m: &mut Master,
    viewer: &str,
    offer_sdp: &str,
    candidates: Vec<Candidate>,
) -> Result<Session, String> {
    // fetched per viewer, the relay credentials carry a ttl
    let servers = ice_servers(&m.credentials, &m.region, &m.channel)
        .await
        .unwrap_or_else(|reason| {
            g2g_warn!(&m.log, "no turn servers, continuing without: {reason}");
            Vec::new()
        });
    let socket = UdpSocket::bind((select_host_ip(), 0))
        .await
        .map_err(|e| format!("udp bind: {e}"))?;
    let local = socket.local_addr().map_err(|e| format!("udp bind: {e}"))?;
    let config = RtcConfig::new()
        .set_crypto_provider(Arc::new(from_feature_flags()))
        .enable_bwe(Some(Bitrate::bps(INITIAL_BITRATE_BPS)))
        .set_stats_interval(Some(STATS_INTERVAL))
        .clear_codecs();
    // str0m answers an m-line it has no codec for with an empty payload list
    let every_track = WebRtcVideoCodec::ALL
        .map(Track::Video)
        .into_iter()
        .chain([Track::Audio]);
    let mut rtc = enable_tracks(config, every_track).build(Instant::now());
    let turn = m
        .settings
        .gather(&mut rtc, &socket, &m.region, &servers)
        .await?;
    g2g_debug!(
        &m.log,
        "ice policy {}, {} turn allocation(s) at {:?}",
        m.settings.transport_policy.name(),
        turn.relay_addrs().len(),
        turn.relay_addrs()
    );
    let offer = SdpOffer::from_sdp_string(offer_sdp)
        .map_err(|e| format!("offer sdp: {e:?}\n{offer_sdp}"))?;
    let answer_sdp = rtc
        .sdp_api()
        .accept_offer(offer)
        .map_err(|e| format!("offer rejected: {e:?}\n{offer_sdp}"))?
        .to_sdp_string();
    let mid = sending_mid(&answer_sdp, m.track.media_kind()).ok_or_else(|| {
        format!(
            "offered no {:?} m-line we can send on",
            m.track.media_kind()
        )
    })?;
    let pt = rtc
        .writer(mid)
        .and_then(|writer| {
            writer
                .payload_params()
                .find(|p| p.spec().codec == m.track.codec())
                .map(|p| p.pt())
        })
        .ok_or_else(|| format!("does not take {:?}", m.track.codec()))?;
    let Some((ws_write, _)) = m.signalling.as_mut() else {
        return Err("signalling closed before the answer went out".to_string());
    };
    send_message(
        ws_write,
        Action::SdpAnswer,
        Some(viewer),
        &sdp_payload(ANSWER_TYPE, &answer_sdp),
    )
    .await?;
    for payload in candidate_payloads(&answer_sdp) {
        send_message(ws_write, Action::IceCandidate, Some(viewer), &payload).await?;
    }
    for candidate in candidates {
        rtc.add_remote_candidate(candidate);
    }
    g2g_debug!(&m.log, "answered viewer {viewer}, rtc socket {local}");
    Ok(Session {
        rtc,
        socket,
        local,
        turn,
        mid,
        pt,
        viewer: viewer.to_string(),
    })
}

async fn run_session(m: &mut Master, mut s: Session) -> SessionEnd {
    let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
    let mut refresh_at = Instant::now() + REFRESH_INTERVAL;
    loop {
        let deadline = loop {
            match s.rtc.poll_output() {
                Ok(Output::Timeout(t)) => break t,
                Ok(Output::Transmit(t)) => send_transmit(&s.socket, &mut s.turn, &t).await,
                Ok(Output::Event(Event::KeyframeRequest(_))) => {
                    m.keyframe_requested.store(true, Ordering::Relaxed);
                }
                Ok(Output::Event(Event::EgressBitrateEstimate(kind))) => {
                    let bps = match kind {
                        BweKind::Twcc(b) | BweKind::Remb(_, b) => Some(b.as_u64()),
                        // `BweKind` is non-exhaustive
                        _ => None,
                    };
                    if let Some(bps) = bps {
                        m.bitrate_estimate.store(bps, Ordering::Relaxed);
                    }
                }
                Ok(Output::Event(Event::Closed)) => {
                    g2g_info!(&m.log, "viewer {} closed the session", s.viewer);
                    return SessionEnd::ViewerLeft;
                }
                Ok(Output::Event(Event::IceConnectionStateChange(
                    IceConnectionState::Disconnected,
                ))) => {
                    g2g_info!(&m.log, "viewer {} disconnected", s.viewer);
                    return SessionEnd::ViewerLeft;
                }
                Ok(Output::Event(Event::IceConnectionStateChange(state))) => {
                    g2g_debug!(&m.log, "viewer {} ice {state:?}", s.viewer);
                }
                Ok(Output::Event(event)) => m.stats.record(&event),
                Err(e) => {
                    g2g_warn!(&m.log, "rtc failed: {e:?}");
                    return SessionEnd::ViewerLeft;
                }
            }
        };
        let timeout = deadline.saturating_duration_since(Instant::now());
        tokio::select! {
            r = s.socket.recv_from(&mut buf) => {
                let Ok((n, source)) = r else { return SessionEnd::ViewerLeft };
                if !feed_datagram(&mut s.rtc, &mut s.turn, s.local, &buf[..n], source) {
                    return SessionEnd::ViewerLeft;
                }
            }
            received = next_signal(&mut m.signalling) => match on_signal(m, received) {
                Some(SignalMessage::Candidate { sender, candidate })
                    if sender.as_deref() == Some(s.viewer.as_str()) =>
                {
                    s.rtc.add_remote_candidate(candidate);
                }
                Some(SignalMessage::Offer { sender, .. }) => {
                    g2g_info!(&m.log, "ignoring the offer from viewer {sender}: viewer {} is live", s.viewer);
                }
                _ => {}
            },
            unit = m.rx.recv() => {
                let Some(unit) = unit else {
                    // the last frames, then close_notify so the viewer ends now
                    drain_pacer(&mut s.rtc, &s.socket, &mut s.turn).await;
                    let _ = s.rtc.close();
                    drain_pacer(&mut s.rtc, &s.socket, &mut s.turn).await;
                    return SessionEnd::InputEnded;
                };
                let rtp_time = m.track.media_time(unit.pts_ns);
                if let Some(writer) = s.rtc.writer(s.mid) {
                    let _ = writer.write(s.pt, Instant::now(), rtp_time, unit.data);
                }
            }
            _ = wake(Some(refresh_at)), if !s.turn.is_empty() => {
                s.turn.refresh_all(&s.socket).await;
                refresh_at = Instant::now() + REFRESH_INTERVAL;
            }
            _ = wake(m.reconnect_at), if m.reconnect_at.is_some() => reconnect(m).await,
            _ = tokio::time::sleep(timeout) => {
                if s.rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                    return SessionEnd::ViewerLeft;
                }
            }
        }
    }
}

async fn leave(m: &mut Master) {
    if let Some((mut ws_write, _)) = m.signalling.take() {
        let _ = ws_write.close().await;
    }
    if let Some(left) = m.left.take() {
        let _ = left.send(());
    }
}

static KVSWEBRTCSINK_PROPS: &[PropertySpec] = &[
    CHANNEL_NAME_PROP,
    CHANNEL_ARN_PROP,
    REGION_PROP,
    ACCESS_KEY_PROP,
    SECRET_KEY_PROP,
    SESSION_TOKEN_PROP,
    KVS_STUN_SERVER_PROP,
    ICE_TRANSPORT_POLICY_PROP,
];

impl AsyncElement for KvsWebRtcSink {
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
            "Kinesis Video Streams WebRTC sink",
            "Sink/Network/WebRTC",
            "Answers a Kinesis Video Streams signalling channel viewer as master and sends it video (H.264/VP8/VP9/H.265/AV1) or Opus over WebRTC (str0m: ICE/DTLS/SRTP)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        KVSWEBRTCSINK_PROPS
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.settings
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.settings.get_property(name)
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
                        // a closed channel means the master task died
                        tx.send(unit).await.map_err(|_| G2gError::Shutdown)?;
                    }
                    self.frames_sent += 1;
                }
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

impl LogSource for KvsWebRtcSink {
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
        let mut sink = KvsWebRtcSink::new();
        for (name, value) in [
            ("channel-name", "g2g-test"),
            (
                "channel-arn",
                "arn:aws:kinesisvideo:us-west-2:123456789012:channel/g2g-test/1",
            ),
            ("region", "us-west-2"),
            ("access-key", "AKIDEXAMPLE"),
            ("secret-key", "secret"),
            ("session-token", "token"),
            ("stun-server", "stun.example.com:3478"),
            ("ice-transport-policy", "relay"),
        ] {
            sink.set_property(name, PropValue::Str(value.into()))
                .unwrap();
            assert_eq!(sink.get_property(name), Some(PropValue::Str(value.into())));
        }
        assert_eq!(
            sink.set_property("region", PropValue::Bool(true)),
            Err(PropError::Type)
        );
        assert_eq!(
            sink.set_property("room-id", PropValue::Str("1".into())),
            Err(PropError::Unknown)
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
            "videotestsrc num-buffers=1 ! kvswebrtcsink channel-name=g2g-test region=us-west-2 access-key=AKIDEXAMPLE secret-key=secret stun-server=stun.example.com:3478",
        )
        .expect("launch line parses")
        .finish()
        .expect("valid graph");
    }

    #[test]
    fn the_answer_names_the_mid_to_send_on() {
        let answer = "v=0\r\n\
            m=video 0 UDP/TLS/RTP/SAVPF 96\r\na=mid:v0\r\na=sendonly\r\n\
            m=video 9 UDP/TLS/RTP/SAVPF 96\r\na=mid:v1\r\na=recvonly\r\n\
            m=video 9 UDP/TLS/RTP/SAVPF 96\r\na=mid:v2\r\na=sendrecv\r\n\
            m=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=sendonly\r\na=mid:a0\r\n";
        assert_eq!(sending_mid(answer, MediaKind::Video), Some(Mid::from("v2")));
        assert_eq!(sending_mid(answer, MediaKind::Audio), Some(Mid::from("a0")));
        assert_eq!(sending_mid("v=0\r\n", MediaKind::Video), None);
    }

    #[test]
    fn accepts_every_carried_codec_and_opus() {
        let sink = KvsWebRtcSink::new();
        for caps in sink_caps_alternatives() {
            assert!(sink.intercept_caps(&caps).is_ok(), "{caps:?}");
        }
    }
}
