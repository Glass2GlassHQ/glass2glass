//! Bidirectional LiveKit participant (`LiveKitDuplex`): publishes local
//! video + Opus audio AND subscribes to the room's remote tracks over one
//! signalling WebSocket, the real-SFU signaller for the duplex shape (M728).
//!
//! LiveKit does not use sendrecv m-lines: a participant owns TWO
//! PeerConnections, a publisher PC (client offers, as in
//! [`crate::livekitsink::LiveKitSink`]) and a subscriber PC (the SERVER
//! offers, as in [`crate::livekitsrc::LiveKitSrc`]). This element runs both
//! `Rtc` instances in one loop over one WebSocket, routing trickle by
//! `SignalTarget` and answering every server re-offer, and exposes the
//! [`MultiDuplexSession`] shape (send inputs 0 = video, 1 = audio; recv
//! outputs 0 = video, 1 = audio) so it drops onto
//! [`run_duplex_session`](g2g_core::runtime::run_duplex_session).
//!
//! Send side is a single video stream + audio (simulcast on the duplex is a
//! follow-up); the recv side takes the first remote video / audio m-line
//! offered, gates video until the first keyframe, and repeats a PLI until it
//! arrives (the `LiveKitSrc` behavior).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use tokio::net::UdpSocket;
use tokio_tungstenite::tungstenite::Message;

use str0m::change::SdpPendingOffer;
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, MediaKind, Mid, Pt};
use str0m::{Event, IceConnectionState, Input, Output, RtcConfig};

use g2g_core::fanout::{DuplexInbound, MultiDuplexSession};
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, HardwareError, MultiOutputSink,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, ReverseChannel,
};

use crate::filesink::io_err;
use crate::livekit_signal::{
    candidate_from_init_json, ping_timestamp, room_property_table, send_signal, AddTrackRequest,
    IceSettings, IceTransportPolicy, LiveKitSignalLog, RoomSettings, SessionDescription,
    SignalRequest, SignalResponse, SignalTarget, TrackSource, TrackType, VideoGrant,
    ICE_TRANSPORT_POLICY_PROP, LIVEKIT_TURN_SERVER_PROP, TURN_PASS_PROP, TURN_USER_PROP,
};
use crate::livekitsrc::answer_offer;
use crate::webrtc_codec::{
    enable_tracks, frame_capture_time, get_video_codec_property, opus_caps,
    set_video_codec_property, sink_caps_alternatives, stamp_capture_time, Track, WebRtcVideoCodec,
    VIDEO_CODEC_PROP,
};
use crate::webrtc_receive::{MediaReceiver, ReceivedUnit, UDP_RECV_BYTES};
use crate::webrtc_util::{
    accept_answer_sdp, connect_failed, feed_datagram, get_linger_ms_property, send_transmit,
    set_linger_ms_property, DEFAULT_LINGER_MS, LINGER_MS_PROP,
};

/// Send input / recv output pad layout: 0 = video, 1 = audio.
const VIDEO: usize = 0;
const AUDIO: usize = 1;
const TOKEN_TTL_SECS: u64 = 3600;
const UNLIMITED_BUFFERS_TEXT: &str = "-1";
const MAX_BUFFERS_TEXT: &str = "9223372036854775807";

static LIVEKITDUPLEX_PROPS: &[PropertySpec] = room_property_table![
    VIDEO_CODEC_PROP,
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "access units to receive across both tracks then EOS (-1 = until the room ends)",
    )
    .with_default(UNLIMITED_BUFFERS_TEXT)
    .with_range(UNLIMITED_BUFFERS_TEXT, MAX_BUFFERS_TEXT),
    LINGER_MS_PROP,
    LIVEKIT_TURN_SERVER_PROP,
    TURN_USER_PROP,
    TURN_PASS_PROP,
    ICE_TRANSPORT_POLICY_PROP,
];

/// Bidirectional LiveKit participant. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::livekitduplex::LiveKitDuplex;
///
/// let duplex = LiveKitDuplex::new("ws://127.0.0.1:7880", "demo-room", "g2g")
///     .with_api_key("devkey", "devsecret")
///     .with_frame_limit(300);
/// ```
pub struct LiveKitDuplex {
    room_settings: RoomSettings,
    /// Track kind per send input pad, set in `configure_input`.
    inputs: Vec<Option<Track>>,
    /// The video codec answered on the subscriber PeerConnection and emitted
    /// on output 0.
    video_codec: WebRtcVideoCodec,
    /// Per send-input reverse channel (remote PLI / BWE back to the source).
    reverse: Vec<ReverseChannel>,
    /// Stop after this many RECEIVED access units and emit EOS (`u64::MAX` =
    /// unbounded).
    frame_limit: u64,
    /// How long to keep draining the peer after the local send side ends.
    linger: Duration,
    ice: IceSettings,
    log_name: LogName,
}

impl LogSource for LiveKitDuplex {
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

impl core::fmt::Debug for LiveKitDuplex {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LiveKitDuplex")
            .field("url", &self.room_settings.url)
            .field("room", &self.room_settings.room)
            .field("identity", &self.room_settings.identity)
            .finish()
    }
}

impl LiveKitDuplex {
    /// Join `room` at the LiveKit `url` as `identity`, publishing the two send
    /// pads (video, audio) and emitting the first remote video / audio tracks
    /// on the two output pads.
    pub fn new(
        url: impl Into<String>,
        room: impl Into<String>,
        identity: impl Into<String>,
    ) -> Self {
        Self {
            room_settings: RoomSettings::new(url, room, identity),
            inputs: alloc::vec![None; 2],
            video_codec: WebRtcVideoCodec::default(),
            reverse: (0..2).map(|_| ReverseChannel::new()).collect(),
            frame_limit: u64::MAX,
            linger: Duration::from_millis(DEFAULT_LINGER_MS),
            ice: IceSettings::default(),
            log_name: LogName::new(),
        }
    }

    /// Subscribe to remote video in `codec` instead of H.264. The published
    /// video keeps the codec of its own pad's caps, the two PeerConnections
    /// being independent.
    pub fn with_video_codec(mut self, codec: WebRtcVideoCodec) -> Self {
        self.video_codec = codec;
        self
    }

    /// Mint the access token locally from an API key + secret.
    pub fn with_api_key(mut self, key: impl Into<String>, secret: impl Into<String>) -> Self {
        self.room_settings.api_key = key.into();
        self.room_settings.api_secret = secret.into();
        self
    }

    /// Explicit TURN servers (comma-separated, see `turn-server`) with long-term
    /// credentials, overriding the JoinResponse's relays.
    pub fn with_turn_server(
        mut self,
        server: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.ice.turn_server = Some(server.into());
        self.ice.turn_user = username.into();
        self.ice.turn_pass = password.into();
        self
    }

    /// Offer every candidate kind, or relays only (`ice-transport-policy`).
    pub fn with_ice_transport_policy(mut self, policy: IceTransportPolicy) -> Self {
        self.ice.transport_policy = policy;
        self
    }

    /// Use a pre-minted access token.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.room_settings.token = Some(token.into());
        self
    }

    /// Stop after `n` received access units (then EOS on both outputs). 0 emits
    /// EOS on both without joining the room, so nothing is published either.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    fn grant(&self) -> VideoGrant {
        VideoGrant {
            room_join: true,
            room: self.room_settings.room.clone(),
            can_publish: true,
            can_subscribe: true,
            can_publish_data: false,
            room_admin: false,
        }
    }
}

impl MultiDuplexSession for LiveKitDuplex {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        2
    }

    fn output_count(&self) -> usize {
        2
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match Track::from_caps(upstream_caps) {
            Some(_) => Ok(upstream_caps.clone()),
            None => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(sink_caps_alternatives()))
    }

    fn configure_input(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let track = Track::from_caps(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        *self.inputs.get_mut(input).ok_or(G2gError::CapsMismatch)? = Some(track);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        match output {
            VIDEO => Ok(self.video_codec.caps_placeholder()),
            AUDIO => Ok(opus_caps(2)),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn reverse_channel(&self, input: usize) -> Option<ReverseChannel> {
        self.reverse.get(input).cloned()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LIVEKITDUPLEX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "num-buffers" => {
                return crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value)
            }
            "video-codec" => return set_video_codec_property(&mut self.video_codec, &value),
            "linger-ms" => return set_linger_ms_property(&mut self.linger, &value),
            _ => {}
        }
        if let Some(result) = self.ice.set_property(name, &value) {
            return result;
        }
        self.room_settings
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(value) = self.room_settings.get_property(name) {
            return Some(value);
        }
        match name {
            "video-codec" => Some(get_video_codec_property(self.video_codec)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            "linger-ms" => Some(get_linger_ms_property(self.linger)),
            _ => self.ice.get_property(name),
        }
    }

    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if crate::numbuffers::finished_at_zero_limit_multi(
                self.frame_limit,
                self.output_count(),
                out,
            )
            .await?
            {
                return Ok(0);
            }
            let (mut ws, join) = self
                .room_settings
                .join(&self.grant(), TOKEN_TTL_SECS, true, &*self)
                .await?;
            let ping_interval = join.ping_period();

            let host_ip = crate::webrtc_util::select_host_ip();

            // ---- Publisher PC (client offers, mirrors LiveKitSink). ----
            let pub_socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
            // The publisher offers whatever the video send pad carries (H.264
            // until a pad says otherwise) plus Opus.
            let send_video = self
                .inputs
                .iter()
                .flatten()
                .copied()
                .find(|t| t.is_video())
                .unwrap_or(Track::Video(WebRtcVideoCodec::default()));
            let recv_video = self.video_codec;
            let pub_config = RtcConfig::new()
                .set_crypto_provider(Arc::new(from_feature_flags()))
                .clear_codecs();
            let mut pub_rtc =
                enable_tracks(pub_config, [send_video, Track::Audio]).build(Instant::now());
            let mut pub_turn = self
                .ice
                .gather(&mut pub_rtc, &pub_socket, &join, &*self)
                .await?;
            let pub_local = pub_socket.local_addr().map_err(io_err)?;
            let stream_id = format!("g2g-{}", self.room_settings.identity);
            let video_cid = format!("{stream_id}-video");
            let audio_cid = format!("{stream_id}-audio");
            let (offer_sdp, pending, send_video_mid, send_audio_mid): (
                String,
                SdpPendingOffer,
                Mid,
                Mid,
            ) = {
                let mut api = pub_rtc.sdp_api();
                let v = api.add_media(
                    MediaKind::Video,
                    Direction::SendOnly,
                    Some(stream_id.clone()),
                    Some(video_cid.clone()),
                    None,
                );
                let a = api.add_media(
                    MediaKind::Audio,
                    Direction::SendOnly,
                    Some(stream_id.clone()),
                    Some(audio_cid.clone()),
                    None,
                );
                let (offer, pending) = api
                    .apply()
                    .ok_or(G2gError::Hardware(HardwareError::Other))?;
                (offer.to_sdp_string(), pending, v, a)
            };
            for (cid, ty, source, name) in [
                (&video_cid, TrackType::Video, TrackSource::Camera, "video"),
                (
                    &audio_cid,
                    TrackType::Audio,
                    TrackSource::Microphone,
                    "audio",
                ),
            ] {
                send_signal(
                    &mut ws,
                    &SignalRequest::AddTrack(AddTrackRequest {
                        cid: cid.clone(),
                        name: name.into(),
                        track_type: ty,
                        width: 0,
                        height: 0,
                        source,
                        layers: Vec::new(),
                    }),
                )
                .await?;
            }

            // ---- Subscriber PC (server offers, mirrors LiveKitSrc). ----
            let sub_socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
            let sub_local = sub_socket.local_addr().map_err(io_err)?;
            let sub_config = RtcConfig::new()
                .set_crypto_provider(Arc::new(from_feature_flags()))
                .clear_codecs();
            let mut sub_rtc = enable_tracks(sub_config, [Track::Video(recv_video), Track::Audio])
                .build(Instant::now());
            let mut sub_turn = self
                .ice
                .gather(&mut sub_rtc, &sub_socket, &join, &*self)
                .await?;
            let mut refresh_at = Instant::now() + crate::turn::REFRESH_INTERVAL;

            // Publisher handshake completion state (the answer + TrackPublished
            // arrive interleaved with subscriber offers on the same socket, so
            // everything is handled in the main loop).
            let mut pub_pending = Some(pending);
            let mut published = 0usize;

            // Send the publisher offer now that both tracks are announced.
            send_signal(
                &mut ws,
                &SignalRequest::Offer(SessionDescription {
                    sdp_type: "offer".into(),
                    sdp: offer_sdp,
                }),
            )
            .await?;

            out.push_to(
                VIDEO,
                PipelinePacket::CapsChanged(recv_video.caps_placeholder()),
            )
            .await?;
            out.push_to(AUDIO, PipelinePacket::CapsChanged(opus_caps(2)))
                .await?;

            let mut media_receiver = MediaReceiver::new(recv_video);
            let mut video_pt: Option<Pt> = None;
            let mut audio_pt: Option<Pt> = None;
            let mut next_ping = Instant::now() + ping_interval;
            let mut send_done = false;
            let mut drain_deadline: Option<Instant> = None;

            let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
            let mut sub_buf = alloc::vec![0u8; UDP_RECV_BYTES];
            let mut received = 0u64;
            macro_rules! finish {
                () => {{
                    let _ = send_signal(&mut ws, &SignalRequest::Leave).await;
                    out.push_to(VIDEO, PipelinePacket::Eos).await?;
                    out.push_to(AUDIO, PipelinePacket::Eos).await?;
                    return Ok(received);
                }};
            }

            loop {
                // Drain both PCs' outputs; the loop deadline is the earlier one.
                let mut frames: Vec<(usize, ReceivedUnit)> = Vec::new();
                let pub_deadline = loop {
                    match pub_rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => {
                            send_transmit(&pub_socket, &mut pub_turn, &t).await;
                        }
                        Ok(Output::Event(Event::KeyframeRequest(req))) => {
                            if req.mid == send_video_mid {
                                self.reverse[VIDEO].request_keyframe();
                            }
                        }
                        Ok(Output::Event(Event::EgressBitrateEstimate(kind))) => {
                            use str0m::bwe::BweKind;
                            let bps = match kind {
                                BweKind::Twcc(b) | BweKind::Remb(_, b) => Some(b.as_u64()),
                                _ => None,
                            };
                            if let Some(bps) = bps {
                                self.reverse[VIDEO].set_bitrate(bps.min(u32::MAX as u64) as u32);
                            }
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => finish!(),
                        Ok(Output::Event(_)) => {}
                        Err(_) => finish!(),
                    }
                };
                let sub_deadline = loop {
                    match sub_rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => {
                            send_transmit(&sub_socket, &mut sub_turn, &t).await;
                        }
                        Ok(Output::Event(Event::MediaAdded(m))) => media_receiver.media_added(&m),
                        Ok(Output::Event(Event::MediaData(d))) => {
                            let Some(unit) = media_receiver.media_data(&d) else {
                                continue;
                            };
                            let port = match unit.kind {
                                MediaKind::Video => VIDEO,
                                MediaKind::Audio => AUDIO,
                            };
                            frames.push((port, unit));
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => finish!(),
                        Ok(Output::Event(_)) => {}
                        Err(_) => finish!(),
                    }
                };
                let deadline = pub_deadline.min(sub_deadline);

                media_receiver.request_keyframe_if_due(&mut sub_rtc);

                for (port, unit) in frames {
                    let frame = unit.into_frame(received);
                    out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
                    received += 1;
                    if received >= self.frame_limit {
                        finish!();
                    }
                }

                if let Some(dl) = drain_deadline {
                    if Instant::now() >= dl {
                        finish!();
                    }
                }

                let timeout = deadline.saturating_duration_since(Instant::now());
                tokio::select! {
                    msg = ws.next() => {
                        match msg {
                            Some(Ok(Message::Binary(bytes))) => match SignalResponse::decode(&bytes) {
                                // Publisher answer: completes our offer.
                                Some(SignalResponse::Answer(sd)) => {
                                    if let Some(pending) = pub_pending.take() {
                                        accept_answer_sdp(&mut pub_rtc, pending, &sd.sdp)
                                            .map_err(|reason| {
                                                connect_failed(&LiveKitSignalLog, &format!("publisher {reason}"))
                                            })?;
                                    }
                                }
                                // Subscriber (re-)offer: answer on the sub PC.
                                Some(SignalResponse::Offer(sd)) => {
                                    answer_offer(&mut sub_rtc, &mut ws, &sd).await?;
                                }
                                Some(SignalResponse::Trickle(t)) => {
                                    if let Some(c) = candidate_from_init_json(&t.candidate_init) {
                                        if let Ok(c) = str0m::Candidate::from_sdp_string(&c) {
                                            match t.target {
                                                SignalTarget::Subscriber => {
                                                    sub_rtc.add_remote_candidate(c)
                                                }
                                                SignalTarget::Publisher => {
                                                    pub_rtc.add_remote_candidate(c)
                                                }
                                            };
                                        }
                                    }
                                }
                                Some(SignalResponse::TrackPublished(_)) => published += 1,
                                Some(SignalResponse::Leave) => finish!(),
                                _ => {}
                            },
                            Some(Ok(Message::Close(_))) | None => finish!(),
                            Some(Ok(_)) => {}
                            Some(Err(_)) => finish!(),
                        }
                        let _ = published;
                    }
                    inb = inbound.recv(), if !send_done => {
                        match inb {
                            None => {
                                send_done = true;
                                drain_deadline = Some(Instant::now() + self.linger);
                            }
                            Some((idx, PipelinePacket::DataFrame(frame))) => {
                                let Some(track) = self.inputs.get(idx).copied().flatten() else {
                                    continue;
                                };
                                let Some(slice) = frame.domain.as_system_slice() else {
                                    continue;
                                };
                                let (mid, pt_slot) = match track {
                                    Track::Video(_) => (send_video_mid, &mut video_pt),
                                    Track::Audio => (send_audio_mid, &mut audio_pt),
                                };
                                if pt_slot.is_none() {
                                    if let Some(writer) = pub_rtc.writer(mid) {
                                        let mode1 = writer.payload_params().find(|p| {
                                            p.spec().codec == track.codec()
                                                && p.spec().format.packetization_mode == Some(1)
                                        });
                                        let any = writer
                                            .payload_params()
                                            .find(|p| p.spec().codec == track.codec());
                                        *pt_slot = mode1.or(any).map(|p| p.pt());
                                    }
                                }
                                if let Some(p) = *pt_slot {
                                    let rtp_time = track.media_time(frame.timing.pts_ns);
                                    if let Some(writer) = pub_rtc.writer(mid) {
                                        let capture_time = frame_capture_time(&frame);
                                        let _ = stamp_capture_time(writer, capture_time).write(
                                            p,
                                            Instant::now(),
                                            rtp_time,
                                            slice.to_vec(),
                                        );
                                    }
                                }
                            }
                            Some(_) => {}
                        }
                    }
                    r = pub_socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else { finish!() };
                        let _ = feed_datagram(&mut pub_rtc, &mut pub_turn, pub_local, &buf[..n], source);
                    }
                    r = sub_socket.recv_from(&mut sub_buf) => {
                        let Ok((n, source)) = r else { finish!() };
                        let _ = feed_datagram(&mut sub_rtc, &mut sub_turn, sub_local, &sub_buf[..n], source);
                    }
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(refresh_at)), if !pub_turn.is_empty() || !sub_turn.is_empty() => {
                        pub_turn.refresh_all(&pub_socket).await;
                        sub_turn.refresh_all(&sub_socket).await;
                        refresh_at = Instant::now() + crate::turn::REFRESH_INTERVAL;
                    }
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(next_ping)) => {
                        let _ = send_signal(&mut ws, &SignalRequest::Ping(ping_timestamp())).await;
                        next_ping = Instant::now() + ping_interval;
                    }
                    _ = tokio::time::sleep(timeout) => {
                        let now = Instant::now();
                        let _ = pub_rtc.handle_input(Input::Timeout(now));
                        let _ = sub_rtc.handle_input(Input::Timeout(now));
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplex_shape_is_two_by_two() {
        let d = LiveKitDuplex::new("ws://h:7880", "room", "id");
        assert_eq!(MultiDuplexSession::input_count(&d), 2);
        assert_eq!(MultiDuplexSession::output_count(&d), 2);
        assert!(MultiDuplexSession::output_caps(&d, 0).is_ok());
        assert!(MultiDuplexSession::output_caps(&d, 2).is_err());
        assert!(MultiDuplexSession::reverse_channel(&d, 0).is_some());
    }

    #[test]
    fn every_property_round_trips() {
        let mut d = LiveKitDuplex::new("ws://h:7880", "room", "id");
        assert_eq!(
            MultiDuplexSession::get_property(&d, LINGER_MS_PROP.name),
            Some(
                LINGER_MS_PROP
                    .parse_value(LINGER_MS_PROP.default.unwrap())
                    .unwrap()
            )
        );
        for spec in MultiDuplexSession::properties(&d) {
            let value = crate::webrtc_util::sample_property_value(spec);
            MultiDuplexSession::set_property(&mut d, spec.name, value.clone()).unwrap();
            assert_eq!(
                MultiDuplexSession::get_property(&d, spec.name),
                Some(value),
                "{}",
                spec.name
            );
        }
        MultiDuplexSession::set_property(&mut d, "num-buffers", PropValue::Int(-1)).unwrap();
        assert_eq!(d.frame_limit, u64::MAX);
    }

    #[test]
    fn unknown_and_mistyped_properties_are_distinct_errors() {
        let mut d = LiveKitDuplex::new("ws://h:7880", "room", "id");
        assert_eq!(
            MultiDuplexSession::set_property(&mut d, "no-such-property", PropValue::Uint(0)),
            Err(PropError::Unknown)
        );
        assert_eq!(
            MultiDuplexSession::get_property(&d, "no-such-property"),
            None
        );
        for name in ["url", "linger-ms", "num-buffers", "turn-server"] {
            assert_eq!(
                MultiDuplexSession::set_property(&mut d, name, PropValue::Double(0.0)),
                Err(PropError::Type),
                "{name}"
            );
        }
        assert_eq!(
            MultiDuplexSession::set_property(
                &mut d,
                "ice-transport-policy",
                PropValue::Str("nearby".into())
            ),
            Err(PropError::Value)
        );
    }

    #[test]
    fn video_codec_property_retags_the_video_output() {
        let mut d = LiveKitDuplex::new("ws://h:7880", "room", "id");
        MultiDuplexSession::set_property(&mut d, "video-codec", PropValue::Str("vp9".into()))
            .unwrap();
        assert_eq!(
            MultiDuplexSession::output_caps(&d, VIDEO),
            Ok(WebRtcVideoCodec::Vp9.caps_placeholder())
        );
        assert_eq!(
            MultiDuplexSession::set_property(
                &mut d,
                "video-codec",
                PropValue::Str("theora".into())
            ),
            Err(PropError::Value)
        );
    }

    #[test]
    fn configure_reads_track_kind_from_caps() {
        let mut d = LiveKitDuplex::new("ws://h:7880", "room", "id");
        let h264 = WebRtcVideoCodec::H264.caps_placeholder();
        MultiDuplexSession::configure_input(&mut d, 1, &h264).unwrap();
        MultiDuplexSession::configure_input(&mut d, 0, &opus_caps(2)).unwrap();
        assert_eq!(d.inputs[1], Some(Track::Video(WebRtcVideoCodec::H264)));
        assert_eq!(d.inputs[0], Some(Track::Audio));
    }
}
