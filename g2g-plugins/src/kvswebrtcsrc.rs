use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use tokio::net::UdpSocket;

use str0m::change::SdpAnswer;
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, KeyframeRequestKind, MediaKind, Mid};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_debug, g2g_info, g2g_warn, Caps, FrameTiming, G2gError, HardwareError, MemoryDomain,
    MultiOutputSink, MultiOutputSource, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec,
};

use crate::kvs_signal::{
    candidate_payloads, client_id_is_valid, connect, connect_url, ice_servers, receive,
    sdp_payload, send_message, Action, KvsSettings, Received, Role, SignalMessage,
    SignalingEndpoints, WsRead, WsWrite, ACCESS_KEY_PROP, CHANNEL_ARN_PROP, CHANNEL_NAME_PROP,
    KVS_STUN_SERVER_PROP, OFFER_TYPE, REGION_PROP, SECRET_KEY_PROP, SESSION_TOKEN_PROP,
};
use crate::turn::{TurnSet, REFRESH_INTERVAL};
use crate::webrtc_codec::{
    enable_tracks, get_video_codec_property, opus_caps, set_video_codec_property, Track,
    WebRtcVideoCodec, VIDEO_CODEC_PROP,
};
use crate::webrtc_util::{
    drain_pacer, feed_datagram, media_time_ns, random_hex_id, select_host_ip, send_transmit,
    IceTransportPolicy, ICE_TRANSPORT_POLICY_PROP,
};

const VIDEO_PORT: usize = 0;
const AUDIO_PORT: usize = 1;
const OUTPUT_COUNT: usize = 2;
const PLI_INTERVAL: Duration = Duration::from_secs(1);
const UDP_RECV_BYTES: usize = 2000;
const STEREO: u8 = 2;
// kvs holds an offer for an absent master until its message ttl runs out
const ANSWER_TIMEOUT: Duration = Duration::from_secs(60);
const CLIENT_ID_PREFIX: &str = "g2g-";

pub struct KvsWebRtcSrc {
    settings: KvsSettings,
    client_id: String,
    video_codec: WebRtcVideoCodec,
    frame_limit: u64,
    log_name: LogName,
}

impl core::fmt::Debug for KvsWebRtcSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KvsWebRtcSrc")
            .field("settings", &self.settings)
            .field("client_id", &self.client_id)
            .field("video_codec", &self.video_codec)
            .finish()
    }
}

impl Default for KvsWebRtcSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl KvsWebRtcSrc {
    pub fn new() -> Self {
        Self {
            settings: KvsSettings::default(),
            client_id: format!("{CLIENT_ID_PREFIX}{}", random_hex_id()),
            video_codec: WebRtcVideoCodec::default(),
            frame_limit: u64::MAX,
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

    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = client_id.into();
        self
    }

    pub fn with_video_codec(mut self, codec: WebRtcVideoCodec) -> Self {
        self.video_codec = codec;
        self
    }

    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    async fn offer_to_the_master(&self) -> Result<Session, String> {
        if !client_id_is_valid(&self.client_id) {
            return Err(format!(
                "client-id {:?} must be 1 to 256 of [a-zA-Z0-9_.-] and not start with AWS_",
                self.client_id
            ));
        }
        let credentials = self.settings.credentials()?;
        let region = self.settings.region()?;
        let channel = self
            .settings
            .resolve(Role::Viewer, &credentials, &region)
            .await?;
        let servers = ice_servers(&credentials, &region, &channel)
            .await
            .unwrap_or_else(|reason| {
                g2g_warn!(self, "no turn servers, continuing without: {reason}");
                Vec::new()
            });
        let url = connect_url(&credentials, &region, &channel, Some(&self.client_id))?;
        let ws = connect(&url).await?;
        g2g_info!(
            self,
            "connected to {} as viewer {}",
            channel.arn,
            self.client_id
        );
        let (mut ws_write, mut ws_read) = ws.split();

        let socket = UdpSocket::bind((select_host_ip(), 0))
            .await
            .map_err(|e| format!("udp bind: {e}"))?;
        let local = socket.local_addr().map_err(|e| format!("udp bind: {e}"))?;
        let config = RtcConfig::new()
            .set_crypto_provider(Arc::new(from_feature_flags()))
            .clear_codecs();
        let mut rtc = enable_tracks(config, [Track::Video(self.video_codec), Track::Audio])
            .build(Instant::now());
        let turn = self
            .settings
            .gather(&mut rtc, &socket, &region, &servers)
            .await?;
        g2g_debug!(
            self,
            "ice policy {}, {} turn allocation(s) at {:?}",
            self.settings.transport_policy.name(),
            turn.relay_addrs().len(),
            turn.relay_addrs()
        );
        let (offer_sdp, pending, video_mid, audio_mid) = {
            let mut api = rtc.sdp_api();
            let video_mid = api.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None);
            let audio_mid = api.add_media(MediaKind::Audio, Direction::RecvOnly, None, None, None);
            let (offer, pending) = api.apply().ok_or("offer creation failed")?;
            (offer.to_sdp_string(), pending, video_mid, audio_mid)
        };
        send_message(
            &mut ws_write,
            Action::SdpOffer,
            None,
            &sdp_payload(OFFER_TYPE, &offer_sdp),
        )
        .await?;
        for payload in candidate_payloads(&offer_sdp) {
            send_message(&mut ws_write, Action::IceCandidate, None, &payload).await?;
        }

        let mut early_candidates: Vec<Candidate> = Vec::new();
        let deadline = tokio::time::Instant::now() + ANSWER_TIMEOUT;
        let answer_sdp = loop {
            let frame = tokio::time::timeout_at(deadline, ws_read.next())
                .await
                .map_err(|_| "the master did not answer in time".to_string())?;
            match receive(frame) {
                Received::Signal(SignalMessage::Answer { sdp }) => break sdp,
                Received::Signal(SignalMessage::Candidate { candidate, .. }) => {
                    early_candidates.push(candidate);
                }
                Received::Signal(SignalMessage::GoAway) => {
                    return Err("signalling service sent GO_AWAY before the answer".to_string())
                }
                Received::Signal(SignalMessage::Status(status)) => {
                    g2g_warn!(self, "signalling status: {status}");
                }
                Received::Signal(other) => g2g_debug!(self, "ignoring {other:?} before the answer"),
                Received::Skipped(reason) => g2g_debug!(self, "skipped a message: {reason}"),
                Received::Control => {}
                Received::Closed(reason) => return Err(reason),
            }
        };
        let answer = SdpAnswer::from_sdp_string(&answer_sdp)
            .map_err(|e| format!("answer sdp: {e:?}\n{answer_sdp}"))?;
        rtc.sdp_api()
            .accept_answer(pending, answer)
            .map_err(|e| format!("answer rejected: {e:?}\n{answer_sdp}"))?;
        for candidate in early_candidates {
            rtc.add_remote_candidate(candidate);
        }
        g2g_debug!(self, "master answered, rtc socket {local}");
        Ok(Session {
            rtc,
            socket,
            local,
            turn,
            video_mid,
            audio_mid,
            ws_write,
            ws_read,
        })
    }
}

struct Session {
    rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    turn: TurnSet,
    video_mid: Mid,
    audio_mid: Mid,
    ws_write: WsWrite,
    ws_read: WsRead,
}

static KVSWEBRTCSRC_PROPS: &[PropertySpec] = &[
    CHANNEL_NAME_PROP,
    CHANNEL_ARN_PROP,
    REGION_PROP,
    ACCESS_KEY_PROP,
    SECRET_KEY_PROP,
    SESSION_TOKEN_PROP,
    KVS_STUN_SERVER_PROP,
    ICE_TRANSPORT_POLICY_PROP,
    PropertySpec::new(
        "client-id",
        PropKind::Str,
        "viewer client id, unique on the channel (default g2g- plus a random id)",
    ),
    VIDEO_CODEC_PROP,
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "access units across both tracks then EOS (-1 = until the master goes away)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
];

impl MultiOutputSource for KvsWebRtcSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }

    fn output_count(&self) -> usize {
        OUTPUT_COUNT
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        match output {
            VIDEO_PORT => Ok(self.video_codec.caps_placeholder()),
            AUDIO_PORT => Ok(opus_caps(STEREO)),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        KVSWEBRTCSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(result) = self.settings.set_property(name, &value) {
            return result;
        }
        match name {
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value),
            "video-codec" => set_video_codec_property(&mut self.video_codec, &value),
            "client-id" => {
                self.client_id = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.settings.get_property(name) {
            return Some(value);
        }
        match name {
            "client-id" => Some(PropValue::Str(self.client_id.clone())),
            "video-codec" => Some(get_video_codec_property(self.video_codec)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if crate::numbuffers::finished_at_zero_limit_multi(self.frame_limit, OUTPUT_COUNT, out)
                .await?
            {
                return Ok(0);
            }
            let mut s = match self.offer_to_the_master().await {
                Ok(session) => session,
                Err(reason) => {
                    g2g_warn!(self, "{reason}");
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
            };

            let video_codec = self.video_codec;
            out.push_to(
                VIDEO_PORT,
                PipelinePacket::CapsChanged(video_codec.caps_placeholder()),
            )
            .await?;
            out.push_to(AUDIO_PORT, PipelinePacket::CapsChanged(opus_caps(STEREO)))
                .await?;

            let mut video_keyframed = false;
            let mut last_pli = Instant::now();
            let mut signalling_open = true;
            let mut refresh_at = Instant::now() + REFRESH_INTERVAL;
            let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
            let mut seq = 0u64;
            macro_rules! finish {
                () => {{
                    // close_notify tells the master now, not at the ice timeout
                    let _ = s.rtc.close();
                    drain_pacer(&mut s.rtc, &s.socket, &mut s.turn).await;
                    let _ = s.ws_write.close().await;
                    out.push_to(VIDEO_PORT, PipelinePacket::Eos).await?;
                    out.push_to(AUDIO_PORT, PipelinePacket::Eos).await?;
                    return Ok(seq);
                }};
            }

            loop {
                let mut frames: Vec<(usize, u64, Vec<u8>)> = Vec::new();
                let deadline = loop {
                    match s.rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => send_transmit(&s.socket, &mut s.turn, &t).await,
                        Ok(Output::Event(Event::MediaData(d))) => {
                            let pts_ns = media_time_ns(d.time);
                            if d.mid == s.video_mid {
                                if !video_keyframed {
                                    if !video_codec.is_keyframe(&d.data) {
                                        continue;
                                    }
                                    g2g_debug!(self, "first video keyframe, forwarding");
                                    video_keyframed = true;
                                }
                                frames.push((
                                    VIDEO_PORT,
                                    pts_ns,
                                    video_codec.pipeline_unit(&d.data),
                                ));
                            } else if d.mid == s.audio_mid {
                                frames.push((AUDIO_PORT, pts_ns, d.data.to_vec()));
                            }
                        }
                        Ok(Output::Event(Event::Closed)) => {
                            g2g_info!(self, "the master closed the session, ending run");
                            finish!()
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => {
                            g2g_info!(self, "ice disconnected, ending run");
                            finish!()
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(state))) => {
                            g2g_debug!(self, "ice {state:?}");
                        }
                        Ok(Output::Event(_)) => {}
                        Err(e) => {
                            g2g_warn!(self, "rtc failed: {e:?}, ending run");
                            finish!()
                        }
                    }
                };

                if !video_keyframed && last_pli.elapsed() >= PLI_INTERVAL {
                    last_pli = Instant::now();
                    if let Some(rx) = s.rtc.direct_api().stream_rx_by_mid(s.video_mid, None) {
                        g2g_debug!(self, "no keyframe yet, sending pli");
                        rx.request_keyframe(KeyframeRequestKind::Pli);
                    }
                }

                for (port, pts_ns, data) in frames {
                    let keyframe = port == VIDEO_PORT && video_codec.is_keyframe(&data);
                    let frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(
                            data.into_boxed_slice(),
                        )),
                        timing: FrameTiming {
                            pts_ns,
                            dts_ns: pts_ns,
                            duration_ns: 0,
                            capture_ns: pts_ns,
                            arrival_ns: g2g_core::metrics::monotonic_ns(),
                            keyframe,
                        },
                        sequence: seq,
                        meta: Default::default(),
                    };
                    out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
                    seq += 1;
                    if seq >= self.frame_limit {
                        finish!();
                    }
                }

                let timeout = deadline.saturating_duration_since(Instant::now());
                tokio::select! {
                    r = s.socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else {
                            g2g_warn!(self, "rtc socket recv failed, ending run");
                            finish!()
                        };
                        let _ = feed_datagram(&mut s.rtc, &mut s.turn, s.local, &buf[..n], source);
                    }
                    frame = s.ws_read.next(), if signalling_open => match receive(frame) {
                        Received::Signal(SignalMessage::Candidate { candidate, .. }) => {
                            s.rtc.add_remote_candidate(candidate);
                        }
                        Received::Signal(SignalMessage::Status(status)) => {
                            g2g_warn!(self, "signalling status: {status}");
                        }
                        Received::Signal(SignalMessage::GoAway) => {
                            g2g_info!(self, "signalling service sent GO_AWAY, media continues");
                            signalling_open = false;
                        }
                        Received::Signal(SignalMessage::ReconnectIceServer) => {
                            g2g_info!(self, "relay servers are going away, media continues until ice fails");
                        }
                        Received::Signal(other) => g2g_debug!(self, "ignoring {other:?}"),
                        Received::Skipped(reason) => g2g_debug!(self, "skipped a message: {reason}"),
                        Received::Control => {}
                        Received::Closed(reason) => {
                            g2g_info!(self, "{reason}, media continues without signalling");
                            signalling_open = false;
                        }
                    },
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(refresh_at)), if !s.turn.is_empty() => {
                        s.turn.refresh_all(&s.socket).await;
                        refresh_at = Instant::now() + REFRESH_INTERVAL;
                    }
                    _ = tokio::time::sleep(timeout) => {
                        if s.rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                            g2g_warn!(self, "rtc timeout handling failed, ending run");
                            finish!();
                        }
                    }
                }
            }
        })
    }
}

impl LogSource for KvsWebRtcSrc {
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
    use g2g_core::{AudioFormat, VideoCodec};

    #[test]
    fn outputs_follow_the_video_codec() {
        let src = KvsWebRtcSrc::new().with_video_codec(WebRtcVideoCodec::Vp8);
        assert!(matches!(
            src.output_caps(VIDEO_PORT),
            Ok(Caps::CompressedVideo {
                codec: VideoCodec::Vp8,
                ..
            })
        ));
        assert!(matches!(
            src.output_caps(AUDIO_PORT),
            Ok(Caps::Audio {
                format: AudioFormat::Opus,
                ..
            })
        ));
        assert!(src.output_caps(OUTPUT_COUNT).is_err());
    }

    #[test]
    fn the_default_client_id_is_random_and_valid() {
        let first = KvsWebRtcSrc::new();
        assert!(client_id_is_valid(&first.client_id), "{}", first.client_id);
        assert!(first.client_id.starts_with(CLIENT_ID_PREFIX));
        assert_ne!(first.client_id, KvsWebRtcSrc::new().client_id);
    }

    #[test]
    fn properties_round_trip() {
        let mut src = KvsWebRtcSrc::new();
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
            ("client-id", "viewer-1"),
            ("video-codec", "vp8"),
        ] {
            src.set_property(name, PropValue::Str(value.into()))
                .unwrap();
            assert_eq!(src.get_property(name), Some(PropValue::Str(value.into())));
        }
        src.set_property("num-buffers", PropValue::Int(12)).unwrap();
        assert_eq!(src.get_property("num-buffers"), Some(PropValue::Int(12)));
        assert_eq!(
            src.set_property("client-id", PropValue::Bool(true)),
            Err(PropError::Type)
        );
        assert_eq!(
            src.set_property("video-codec", PropValue::Str("theora".into())),
            Err(PropError::Value)
        );
        assert_eq!(
            src.set_property("room-id", PropValue::Str("1".into())),
            Err(PropError::Unknown)
        );
        for spec in src.properties() {
            assert!(src.get_property(spec.name).is_some(), "{}", spec.name);
        }
    }

    #[test]
    fn launch_line_sets_the_properties() {
        use g2g_core::graph::NodeKind;
        use g2g_core::runtime::parse_launch;

        let registry = crate::registry::default_registry();
        let graph = parse_launch(
            &registry,
            "kvswebrtcsrc name=s channel-name=g2g-test region=us-west-2 client-id=viewer-1 video-codec=vp8 num-buffers=10  s. ! fakesink  s. ! fakesink",
        )
        .expect("launch line parses")
        .finish()
        .expect("valid graph");
        let fanouts: Vec<NodeKind> = graph
            .topo()
            .iter()
            .map(|&n| graph.kind(n))
            .filter(|k| matches!(k, NodeKind::FanoutSrc(_)))
            .collect();
        assert_eq!(fanouts, [NodeKind::FanoutSrc(OUTPUT_COUNT as u8)]);
    }
}
