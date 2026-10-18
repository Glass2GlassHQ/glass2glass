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
    candidate_payloads, connect, connect_url, gather, ice_servers, receive, sdp_payload,
    send_message, Action, KvsSettings, Received, Role, SignalMessage, SignalingEndpoints, WsRead,
    WsWrite, ACCESS_KEY_PROP, ANSWER_TYPE, CHANNEL_ARN_PROP, CHANNEL_NAME_PROP, REGION_PROP,
    SECRET_KEY_PROP, SESSION_TOKEN_PROP, STUN_SERVER_PROP,
};
use crate::turn::{TurnSet, REFRESH_INTERVAL};
use crate::webrtc_codec::{enable_tracks, sink_caps_alternatives, Track, WebRtcVideoCodec};
use crate::webrtc_stats::{WebRtcStatsHandle, STATS_INTERVAL};
use crate::webrtc_util::{drain_pacer, feed_datagram, select_host_ip, send_transmit};
use crate::webrtcsink::{MediaUnit, DEFAULT_QUEUE_DEPTH, INITIAL_BITRATE_BPS};

const LEAVE_TIMEOUT: Duration = Duration::from_secs(2);
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
        match self.answer_a_viewer().await {
            Ok(()) => Ok(()),
            Err(reason) => {
                g2g_warn!(self, "{reason}");
                Err(G2gError::Hardware(HardwareError::Other))
            }
        }
    }

    async fn answer_a_viewer(&mut self) -> Result<(), String> {
        let credentials = self.settings.credentials()?;
        let region = self.settings.region()?;
        let channel = self
            .settings
            .resolve(Role::Master, &credentials, &region)
            .await?;
        let servers = ice_servers(&credentials, &region, &channel)
            .await
            .unwrap_or_else(|reason| {
                g2g_warn!(self, "no turn servers, continuing without: {reason}");
                Vec::new()
            });
        let url = connect_url(&credentials, &region, &channel, None)?;
        let ws = connect(&url).await?;
        g2g_info!(
            self,
            "connected to {} as master, waiting for a viewer",
            channel.arn
        );
        let (mut ws_write, mut ws_read) = ws.split();

        let mut early_candidates: Vec<(Option<String>, Candidate)> = Vec::new();
        let (viewer, offer_sdp) = loop {
            match receive(ws_read.next().await) {
                Received::Signal(SignalMessage::Offer { sender, sdp }) => break (sender, sdp),
                Received::Signal(SignalMessage::Candidate { sender, candidate }) => {
                    early_candidates.push((sender, candidate));
                }
                Received::Signal(SignalMessage::GoAway) => {
                    return Err("signalling service sent GO_AWAY before any offer".to_string())
                }
                Received::Signal(SignalMessage::Status(status)) => {
                    g2g_warn!(self, "signalling status: {status}");
                }
                Received::Signal(other) => g2g_debug!(self, "ignoring {other:?} before an offer"),
                Received::Skipped(reason) => g2g_debug!(self, "skipped a message: {reason}"),
                Received::Control => {}
                Received::Closed(reason) => return Err(reason),
            }
        };
        g2g_info!(self, "offer from viewer {viewer}");

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
        let turn = gather(
            &mut rtc,
            &socket,
            self.settings.stun_server.as_deref(),
            &servers,
        )
        .await?;
        g2g_debug!(
            self,
            "{} turn allocation(s) at {:?}",
            turn.relay_addrs().len(),
            turn.relay_addrs()
        );
        let offer = SdpOffer::from_sdp_string(&offer_sdp)
            .map_err(|e| format!("offer sdp: {e:?}\n{offer_sdp}"))?;
        let answer_sdp = rtc
            .sdp_api()
            .accept_offer(offer)
            .map_err(|e| format!("offer rejected: {e:?}\n{offer_sdp}"))?
            .to_sdp_string();
        let mid = sending_mid(&answer_sdp, self.track.media_kind()).ok_or_else(|| {
            format!(
                "viewer {viewer} offered no {:?} m-line we can send on",
                self.track.media_kind()
            )
        })?;
        let pt = rtc
            .writer(mid)
            .and_then(|writer| {
                writer
                    .payload_params()
                    .find(|p| p.spec().codec == self.track.codec())
                    .map(|p| p.pt())
            })
            .ok_or_else(|| format!("viewer {viewer} does not take {:?}", self.track.codec()))?;
        send_message(
            &mut ws_write,
            Action::SdpAnswer,
            Some(&viewer),
            &sdp_payload(ANSWER_TYPE, &answer_sdp),
        )
        .await?;
        for payload in candidate_payloads(&answer_sdp) {
            send_message(&mut ws_write, Action::IceCandidate, Some(&viewer), &payload).await?;
        }
        for (sender, candidate) in early_candidates {
            if sender.as_deref() == Some(viewer.as_str()) {
                rtc.add_remote_candidate(candidate);
            }
        }
        g2g_debug!(self, "answered viewer {viewer}, rtc socket {local}");

        let (tx, rx) = mpsc::channel(DEFAULT_QUEUE_DEPTH);
        let (left_tx, left_rx) = oneshot::channel();
        tokio::spawn(run_session(SessionArgs {
            rtc,
            socket,
            local,
            turn,
            track: self.track,
            mid,
            pt,
            viewer,
            ws_write,
            ws_read,
            keyframe_requested: Arc::clone(&self.keyframe_requested),
            bitrate_estimate: Arc::clone(&self.bitrate_estimate),
            stats: self.stats.clone(),
            rx,
            left: Some(left_tx),
            log: SessionLog(self.log_name.clone()),
        }));
        self.tx = Some(tx);
        self.left = Some(left_rx);
        Ok(())
    }
}

struct SessionLog(LogName);

impl LogSource for SessionLog {
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

struct SessionArgs {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    turn: TurnSet,
    track: Track,
    mid: Mid,
    pt: Pt,
    viewer: String,
    ws_write: WsWrite,
    ws_read: WsRead,
    keyframe_requested: Arc<AtomicBool>,
    bitrate_estimate: Arc<AtomicU64>,
    stats: WebRtcStatsHandle,
    rx: mpsc::Receiver<MediaUnit>,
    left: Option<oneshot::Sender<()>>,
    log: SessionLog,
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

async fn run_session(mut a: SessionArgs) {
    let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
    let mut signalling_open = true;
    let mut refresh_at = Instant::now() + REFRESH_INTERVAL;
    loop {
        let deadline = loop {
            match a.rtc.poll_output() {
                Ok(Output::Timeout(t)) => break t,
                Ok(Output::Transmit(t)) => send_transmit(&a.socket, &mut a.turn, &t).await,
                Ok(Output::Event(Event::KeyframeRequest(_))) => {
                    a.keyframe_requested.store(true, Ordering::Relaxed);
                }
                Ok(Output::Event(Event::EgressBitrateEstimate(kind))) => {
                    let bps = match kind {
                        BweKind::Twcc(b) | BweKind::Remb(_, b) => Some(b.as_u64()),
                        // `BweKind` is non-exhaustive
                        _ => None,
                    };
                    if let Some(bps) = bps {
                        a.bitrate_estimate.store(bps, Ordering::Relaxed);
                    }
                }
                Ok(Output::Event(Event::IceConnectionStateChange(
                    IceConnectionState::Disconnected,
                ))) => {
                    g2g_info!(&a.log, "viewer {} disconnected", a.viewer);
                    return leave(&mut a).await;
                }
                Ok(Output::Event(event)) => a.stats.record(&event),
                Err(e) => {
                    g2g_warn!(&a.log, "rtc failed: {e:?}");
                    return leave(&mut a).await;
                }
            }
        };
        let timeout = deadline.saturating_duration_since(Instant::now());
        tokio::select! {
            r = a.socket.recv_from(&mut buf) => {
                let Ok((n, source)) = r else { return leave(&mut a).await };
                if !feed_datagram(&mut a.rtc, &mut a.turn, a.local, &buf[..n], source) {
                    return leave(&mut a).await;
                }
            }
            frame = a.ws_read.next(), if signalling_open => match receive(frame) {
                Received::Signal(SignalMessage::Candidate { sender, candidate })
                    if sender.as_deref() == Some(a.viewer.as_str()) =>
                {
                    a.rtc.add_remote_candidate(candidate);
                }
                Received::Signal(SignalMessage::Offer { sender, .. }) => {
                    g2g_info!(&a.log, "ignoring the offer from viewer {sender}: viewer {} is live", a.viewer);
                }
                Received::Signal(SignalMessage::Status(status)) => {
                    g2g_warn!(&a.log, "signalling status: {status}");
                }
                Received::Signal(SignalMessage::GoAway) => {
                    g2g_info!(&a.log, "signalling service sent GO_AWAY, media continues");
                    signalling_open = false;
                }
                Received::Signal(other) => g2g_debug!(&a.log, "ignoring {other:?}"),
                Received::Skipped(reason) => g2g_debug!(&a.log, "skipped a message: {reason}"),
                Received::Control => {}
                Received::Closed(reason) => {
                    g2g_info!(&a.log, "{reason}, media continues without signalling");
                    signalling_open = false;
                }
            },
            unit = a.rx.recv() => {
                let Some(unit) = unit else {
                    drain_pacer(&mut a.rtc, &a.socket, &mut a.turn).await;
                    return leave(&mut a).await;
                };
                let rtp_time = a.track.media_time(unit.pts_ns);
                if let Some(writer) = a.rtc.writer(a.mid) {
                    let _ = writer.write(a.pt, Instant::now(), rtp_time, unit.data);
                }
            }
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(refresh_at)), if !a.turn.is_empty() => {
                a.turn.refresh_all(&a.socket).await;
                refresh_at = Instant::now() + REFRESH_INTERVAL;
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
    let _ = a.ws_write.close().await;
    if let Some(left) = a.left.take() {
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
    STUN_SERVER_PROP,
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
                        // a closed channel means the viewer left
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
