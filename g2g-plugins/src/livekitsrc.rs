//! Native LiveKit subscriber source (`LiveKitSrc`): joins a room over the
//! LiveKit WebSocket signalling protocol and emits the subscribed video (`video-codec`, H.264 by default)
//! and Opus audio on two output pads, from one PeerConnection. The ingest
//! mirror of [`crate::livekitsink::LiveKitSink`], reusing the same
//! [`crate::livekit_signal`] seam (JWT + hand-rolled protobuf codec).
//!
//! Protocol shape: unlike WHIP/WHEP (client offers), the LiveKit SFU OFFERS the
//! subscriber PeerConnection: the client joins with `auto_subscribe`, the server
//! sends an SDP offer over the signalling socket (re-offering whenever the
//! room's track set changes), and this element answers each one
//! (`SdpApi::accept_offer`). Tracks are discovered from `Event::MediaAdded`
//! (the answerer learns its mids from the offer, see the duplex session): the
//! first video m-line feeds output 0 and the first audio m-line output 1;
//! additional tracks in the room are ignored (one-subscription element).
//!
//! Mid-GOP subscribe: video is gated until the first keyframe (a decoder can't
//! start on a P slice), and a PLI is repeated once a second until it arrives,
//! like [`crate::webrtcwhepsrc::WebRtcWhepSrc`].
//!
//! Shape: a [`MultiOutputSource`] (output 0 = video, output 1 = Opus
//! audio) driven by the terminal fan-out runner
//! [`run_fanout_session`](g2g_core::runtime::run_fanout_session), like
//! [`crate::webrtcwhepsession::WebRtcWhepSessionSrc`]. Host-candidate ICE (the
//! local dev server); STUN/TURN toward LiveKit Cloud is a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::time::Instant;

use futures_util::StreamExt;
use tokio::net::UdpSocket;
use tokio_tungstenite::tungstenite::Message;

use str0m::crypto::from_feature_flags;
use str0m::media::MediaKind;
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_warn, Caps, G2gError, MultiOutputSink, MultiOutputSource, NetworkFailure, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec,
};

use crate::filesink::io_err;
use crate::livekit_signal::{
    candidate_from_init_json, ping_timestamp, room_property_table, send_signal, IceSettings,
    IceTransportPolicy, LiveKitSignalLog, RoomSettings, SessionDescription, SignalRequest,
    SignalResponse, SignalSocket, SignalTarget, VideoGrant, ICE_TRANSPORT_POLICY_PROP,
    LIVEKIT_TURN_SERVER_PROP, TURN_PASS_PROP, TURN_USER_PROP,
};
use crate::webrtc_codec::{
    enable_tracks, get_video_codec_property, opus_caps, set_video_codec_property, Track,
    WebRtcVideoCodec, VIDEO_CODEC_PROP,
};
use crate::webrtc_receive::{MediaReceiver, ReceivedUnit, UDP_RECV_BYTES};
use crate::webrtc_util::{accept_offer_sdp, send_transmit};

/// Output port for the video track.
const VIDEO_PORT: usize = 0;
/// Output port for the Opus audio track.
const AUDIO_PORT: usize = 1;
/// Repeat the PLI at this interval until the first keyframe arrives.
/// Access-token lifetime, matching the sink.
const TOKEN_TTL_SECS: u64 = 3600;

/// Native LiveKit room subscriber. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::livekitsrc::LiveKitSrc;
///
/// let src = LiveKitSrc::new("ws://127.0.0.1:7880", "room", "g2g-subscriber")
///     .with_api_key("devkey", "secret")
///     .with_frame_limit(300);
/// ```
pub struct LiveKitSrc {
    room_settings: RoomSettings,
    /// The video codec answered to the server's offer and emitted on output 0.
    video_codec: WebRtcVideoCodec,
    /// Stop after this many access units across both tracks and emit EOS
    /// (`u64::MAX` = unbounded). For tests / bounded runs.
    frame_limit: u64,
    ice: IceSettings,
    log_name: LogName,
}

impl LogSource for LiveKitSrc {
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

impl core::fmt::Debug for LiveKitSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LiveKitSrc")
            .field("url", &self.room_settings.url)
            .field("room", &self.room_settings.room)
            .field("identity", &self.room_settings.identity)
            .finish()
    }
}

impl LiveKitSrc {
    /// Subscribe to `room` on the LiveKit server at `url` (`ws://` / `wss://`)
    /// as participant `identity`, emitting video on output 0 and audio on
    /// output 1. Set credentials with [`Self::with_api_key`] (+ secret) or a
    /// pre-minted [`Self::with_token`].
    pub fn new(
        url: impl Into<String>,
        room: impl Into<String>,
        identity: impl Into<String>,
    ) -> Self {
        Self {
            room_settings: RoomSettings::new(url, room, identity),
            video_codec: WebRtcVideoCodec::default(),
            frame_limit: u64::MAX,
            ice: IceSettings::default(),
            log_name: LogName::new(),
        }
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

    /// Subscribe to video in `codec` instead of H.264 (the `video-codec`
    /// property). A remote track in another codec is left unanswered.
    pub fn with_video_codec(mut self, codec: WebRtcVideoCodec) -> Self {
        self.video_codec = codec;
        self
    }

    /// Use a pre-minted access token (LiveKit Cloud, external token service).
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.room_settings.token = Some(token.into());
        self
    }

    /// Stop after `n` access units across both tracks (then EOS on both). 0 emits
    /// EOS on both without joining the room.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    fn grant(&self) -> VideoGrant {
        VideoGrant::subscriber(self.room_settings.room.clone())
    }
}

/// Accept one server offer on the subscriber PC and send the answer back.
pub(crate) async fn answer_offer(
    rtc: &mut Rtc,
    ws: &mut SignalSocket,
    sd: &SessionDescription,
) -> Result<(), G2gError> {
    let answer = accept_offer_sdp(rtc, &sd.sdp).map_err(|reason| {
        g2g_warn!(&LiveKitSignalLog, "subscriber {reason}");
        G2gError::Network(NetworkFailure::Stream)
    })?;
    send_signal(
        ws,
        &SignalRequest::Answer(SessionDescription {
            sdp_type: "answer".into(),
            sdp: answer.to_sdp_string(),
        }),
    )
    .await
}

/// Settable properties, so a `gst-launch` line can subscribe without the
/// builder (`livekitsrc url=... room=... api-key=... api-secret=...`).
static LIVEKITSRC_PROPS: &[PropertySpec] = room_property_table![
    VIDEO_CODEC_PROP,
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "access units to emit across both tracks then EOS (-1 = until the room ends)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
    LIVEKIT_TURN_SERVER_PROP,
    TURN_USER_PROP,
    TURN_PASS_PROP,
    ICE_TRANSPORT_POLICY_PROP,
];

impl MultiOutputSource for LiveKitSrc {
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
        2
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        match output {
            VIDEO_PORT => Ok(self.video_codec.caps_placeholder()),
            AUDIO_PORT => Ok(opus_caps(2)),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LIVEKITSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if name == "num-buffers" {
            return crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value);
        }
        if name == "video-codec" {
            return set_video_codec_property(&mut self.video_codec, &value);
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
            _ => self.ice.get_property(name),
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
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

            let socket = UdpSocket::bind((crate::webrtc_util::select_host_ip(), 0))
                .await
                .map_err(io_err)?;
            let local = socket.local_addr().map_err(io_err)?;
            let video_codec = self.video_codec;
            let config = RtcConfig::new()
                .set_crypto_provider(Arc::new(from_feature_flags()))
                .clear_codecs();
            let mut rtc = enable_tracks(config, [Track::Video(video_codec), Track::Audio])
                .build(Instant::now());
            let mut turn = self.ice.gather(&mut rtc, &socket, &join, &*self).await?;
            let mut refresh_at = Instant::now() + crate::turn::REFRESH_INTERVAL;

            out.push_to(
                VIDEO_PORT,
                PipelinePacket::CapsChanged(video_codec.caps_placeholder()),
            )
            .await?;
            out.push_to(AUDIO_PORT, PipelinePacket::CapsChanged(opus_caps(2)))
                .await?;

            // Track routing state: the first video / audio m-line the server
            // offers claims its port; video is gated until the first keyframe.
            let mut media_receiver = MediaReceiver::new(video_codec);
            let mut next_ping = Instant::now() + ping_interval;

            let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
            let mut seq = 0u64;
            macro_rules! finish {
                () => {{
                    let _ = send_signal(&mut ws, &SignalRequest::Leave).await;
                    out.push_to(VIDEO_PORT, PipelinePacket::Eos).await?;
                    out.push_to(AUDIO_PORT, PipelinePacket::Eos).await?;
                    return Ok(seq);
                }};
            }

            loop {
                let mut frames: Vec<(usize, ReceivedUnit)> = Vec::new();
                let deadline = loop {
                    match rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => {
                            send_transmit(&socket, &mut turn, &t).await;
                        }
                        Ok(Output::Event(Event::MediaAdded(m))) => media_receiver.media_added(&m),
                        Ok(Output::Event(Event::MediaData(d))) => {
                            let Some(unit) = media_receiver.media_data(&d) else {
                                continue;
                            };
                            let port = match unit.kind {
                                MediaKind::Video => VIDEO_PORT,
                                MediaKind::Audio => AUDIO_PORT,
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

                // Re-PLI until the video decodes from a keyframe.
                media_receiver.request_keyframe_if_due(&mut rtc);

                for (port, unit) in frames {
                    let frame = unit.into_frame(seq);
                    out.push_to(port, PipelinePacket::DataFrame(frame)).await?;
                    seq += 1;
                    if seq >= self.frame_limit {
                        finish!();
                    }
                }

                let timeout = deadline.saturating_duration_since(Instant::now());
                tokio::select! {
                    msg = ws.next() => {
                        match msg {
                            Some(Ok(Message::Binary(bytes))) => {
                                match SignalResponse::decode(&bytes) {
                                    // The SFU (re-)offers the subscriber PC when
                                    // the room's track set changes; answer each.
                                    Some(SignalResponse::Offer(sd)) => {
                                        answer_offer(&mut rtc, &mut ws, &sd).await?;
                                    }
                                    Some(SignalResponse::Trickle(t))
                                        if t.target == SignalTarget::Subscriber =>
                                    {
                                        if let Some(c) = candidate_from_init_json(&t.candidate_init) {
                                            if let Ok(c) = Candidate::from_sdp_string(&c) {
                                                rtc.add_remote_candidate(c);
                                            }
                                        }
                                    }
                                    Some(SignalResponse::Leave) => finish!(),
                                    _ => {}
                                }
                            }
                            Some(Ok(Message::Close(_))) | None => finish!(),
                            Some(Ok(_)) => {}
                            Some(Err(e)) => {
                                g2g_warn!(self, "signal recv failed: {e}");
                                finish!();
                            }
                        }
                    }
                    r = socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else { finish!() };
                        let _ = crate::webrtc_util::feed_datagram(
                            &mut rtc, &mut turn, local, &buf[..n], source,
                        );
                    }
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(refresh_at)), if !turn.is_empty() => {
                        turn.refresh_all(&socket).await;
                        refresh_at = Instant::now() + crate::turn::REFRESH_INTERVAL;
                    }
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(next_ping)) => {
                        let _ = send_signal(&mut ws, &SignalRequest::Ping(ping_timestamp())).await;
                        next_ping = Instant::now() + ping_interval;
                    }
                    _ = tokio::time::sleep(timeout) => {
                        let _ = rtc.handle_input(Input::Timeout(Instant::now()));
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{AudioFormat, VideoCodec};

    #[test]
    fn two_outputs_video_then_audio() {
        let src = LiveKitSrc::new("ws://h:7880", "room", "id");
        assert_eq!(src.output_count(), 2);
        assert!(matches!(
            src.output_caps(VIDEO_PORT),
            Ok(Caps::CompressedVideo {
                codec: VideoCodec::H264,
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
        assert!(src.output_caps(2).is_err());
    }

    #[test]
    fn unknown_property_is_unknown_whatever_the_value_type() {
        let mut src = LiveKitSrc::new("ws://h:7880", "room", "id");
        assert_eq!(
            src.set_property("nope", PropValue::Uint(1)),
            Err(PropError::Unknown)
        );
        assert_eq!(
            src.set_property("url", PropValue::Uint(1)),
            Err(PropError::Type)
        );
    }

    #[test]
    fn token_requires_credentials() {
        let src = LiveKitSrc::new("ws://h:7880", "room", "id");
        let grant = src.grant();
        assert!(src
            .room_settings
            .access_token(&grant, TOKEN_TTL_SECS)
            .is_err());
        let src = src.with_token("tok");
        assert_eq!(
            src.room_settings
                .access_token(&grant, TOKEN_TTL_SECS)
                .unwrap(),
            "tok"
        );
    }
}
