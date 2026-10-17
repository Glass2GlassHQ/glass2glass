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
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;

use str0m::change::{SdpAnswer, SdpPendingOffer};
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, KeyframeRequestKind, MediaKind, Mid, Pt};
use str0m::{Event, IceConnectionState, Input, Output, RtcConfig};

use g2g_core::fanout::{DuplexInbound, MultiDuplexSession};
use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::memory::SystemSlice;
use g2g_core::{
    g2g_warn, Caps, CapsConstraint, CapsSet, ConfigureOutcome, FrameTiming, G2gError,
    HardwareError, MemoryDomain, MultiOutputSink, PipelinePacket, ReverseChannel,
};

use crate::filesink::io_err;
use crate::livekit_signal::{
    access_token, candidate_from_init_json, signal_ws_url, AddTrackRequest, IceSettings,
    SessionDescription, SignalRequest, SignalResponse, SignalTarget, TrackSource, TrackType,
    VideoGrant,
};
use crate::livekitsrc::{answer_offer, recv_signal, send_signal};
use crate::webrtc_codec::{
    enable_tracks, opus_caps, sink_caps_alternatives, Track, WebRtcVideoCodec,
};
use crate::webrtc_util::{feed_datagram, media_time_ns, send_transmit, IceTransportPolicy};

/// Send input / recv output pad layout: 0 = video, 1 = audio.
const VIDEO: usize = 0;
const AUDIO: usize = 1;
const PLI_INTERVAL: Duration = Duration::from_secs(1);
const TOKEN_TTL_SECS: u64 = 3600;

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
    url: String,
    room: String,
    identity: String,
    api_key: String,
    api_secret: String,
    token: Option<String>,
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
            .field("url", &self.url)
            .field("room", &self.room)
            .field("identity", &self.identity)
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
            url: url.into(),
            room: room.into(),
            identity: identity.into(),
            api_key: String::new(),
            api_secret: String::new(),
            token: None,
            inputs: alloc::vec![None; 2],
            video_codec: WebRtcVideoCodec::default(),
            reverse: (0..2).map(|_| ReverseChannel::new()).collect(),
            frame_limit: u64::MAX,
            linger: Duration::from_millis(1500),
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
        self.api_key = key.into();
        self.api_secret = secret.into();
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
        self.token = Some(token.into());
        self
    }

    /// Stop after `n` received access units (then EOS on both outputs). 0 emits
    /// EOS on both without joining the room, so nothing is published either.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    fn access_token(&self) -> Result<String, G2gError> {
        access_token(
            self.token.as_deref(),
            &self.api_key,
            &self.api_secret,
            &self.identity,
            &VideoGrant {
                room_join: true,
                room: self.room.clone(),
                can_publish: true,
                can_subscribe: true,
                can_publish_data: false,
                room_admin: false,
            },
            TOKEN_TTL_SECS,
        )
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
            let hw = || G2gError::Hardware(HardwareError::Other);
            let token = self.access_token()?;
            let ws_url = signal_ws_url(&self.url, &token, true);
            let (mut ws, _resp) = connect_async(&ws_url).await.map_err(|e| {
                g2g_warn!(self, "WebSocket connect to {} failed: {e}", self.url);
                hw()
            })?;
            let join = loop {
                match recv_signal(&mut ws).await? {
                    Some(SignalResponse::Join(j)) => break j,
                    Some(SignalResponse::Leave) | None => return Err(hw()),
                    Some(_) => {}
                }
            };
            let ping_interval = if join.ping_interval > 0 {
                Duration::from_secs(join.ping_interval as u64)
            } else {
                Duration::from_secs(15)
            };

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
            let stream_id = format!("g2g-{}", self.identity);
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
                let (offer, pending) = api.apply().ok_or_else(hw)?;
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

            let mut recv_video_mid: Option<Mid> = None;
            let mut recv_audio_mid: Option<Mid> = None;
            let mut video_keyframed = false;
            let mut last_pli = Instant::now();
            let mut video_pt: Option<Pt> = None;
            let mut audio_pt: Option<Pt> = None;
            let mut next_ping = Instant::now() + ping_interval;
            let mut send_done = false;
            let mut drain_deadline: Option<Instant> = None;

            let mut buf = alloc::vec![0u8; 2000];
            let mut sub_buf = alloc::vec![0u8; 2000];
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
                let mut frames: Vec<(usize, u64, Vec<u8>)> = Vec::new();
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
                        Ok(Output::Event(Event::MediaAdded(m))) => match m.kind {
                            MediaKind::Video if recv_video_mid.is_none() => {
                                recv_video_mid = Some(m.mid)
                            }
                            MediaKind::Audio if recv_audio_mid.is_none() => {
                                recv_audio_mid = Some(m.mid)
                            }
                            _ => {}
                        },
                        Ok(Output::Event(Event::MediaData(d))) => {
                            let pts_ns = media_time_ns(d.time);
                            if Some(d.mid) == recv_video_mid {
                                if !video_keyframed {
                                    if !recv_video.is_keyframe(&d.data) {
                                        continue;
                                    }
                                    video_keyframed = true;
                                }
                                frames.push((VIDEO, pts_ns, recv_video.pipeline_unit(&d.data)));
                            } else if Some(d.mid) == recv_audio_mid {
                                frames.push((AUDIO, pts_ns, d.data.to_vec()));
                            }
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => finish!(),
                        Ok(Output::Event(_)) => {}
                        Err(_) => finish!(),
                    }
                };
                let deadline = pub_deadline.min(sub_deadline);

                if let Some(mid) = recv_video_mid {
                    if !video_keyframed && last_pli.elapsed() >= PLI_INTERVAL {
                        last_pli = Instant::now();
                        if let Some(rx) = sub_rtc.direct_api().stream_rx_by_mid(mid, None) {
                            rx.request_keyframe(KeyframeRequestKind::Pli);
                        }
                    }
                }

                for (port, pts_ns, data) in frames {
                    let keyframe = port == VIDEO && recv_video.is_keyframe(&data);
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
                        sequence: received,
                        meta: Default::default(),
                    };
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
                                        let answer = SdpAnswer::from_sdp_string(&sd.sdp)
                                            .map_err(|_| hw())?;
                                        pub_rtc
                                            .sdp_api()
                                            .accept_answer(pending, answer)
                                            .map_err(|_| hw())?;
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
                                        let _ = writer.write(
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
                        let now_ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as i64;
                        let _ = send_signal(&mut ws, &SignalRequest::Ping(now_ms)).await;
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
    fn configure_reads_track_kind_from_caps() {
        let mut d = LiveKitDuplex::new("ws://h:7880", "room", "id");
        let h264 = WebRtcVideoCodec::H264.caps_placeholder();
        MultiDuplexSession::configure_input(&mut d, 1, &h264).unwrap();
        MultiDuplexSession::configure_input(&mut d, 0, &opus_caps(2)).unwrap();
        assert_eq!(d.inputs[1], Some(Track::Video(WebRtcVideoCodec::H264)));
        assert_eq!(d.inputs[0], Some(Track::Audio));
    }
}
