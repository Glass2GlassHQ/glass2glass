//! Native WebRTC ingest source (`WebRtcWhepSrc`): subscribes to a WHEP endpoint
//! and emits the received video or Opus as `DataFrame`s. The receive-side
//! inverse of [`crate::webrtcsink::WebRtcSink`], on the same sans-IO `str0m`
//! stack (ICE / DTLS / SRTP), and distinct from the wasm-only data-channel
//! [`crate::webrtcsrc::WebRtcSrc`].
//!
//! WHEP is client-offers-recvonly: the source builds a str0m `Rtc` with a single
//! recv-only m-line for the chosen track, POSTs the SDP offer to the WHEP
//! endpoint, applies the answer, then drives str0m's `poll_output` /
//! `handle_input` loop on a tokio `UdpSocket`. Each `Event::MediaData` is a
//! depacketized access unit, forwarded downstream with its RTP-clock PTS: for
//! video the depayloader emits the pipeline framing (Annex-B for H.264 and H.265,
//! a raw frame for VP8 and VP9, a temporal unit for AV1); Opus packets as-is.
//!
//! The track is video (H.264 unless `video-codec` says otherwise) by default, or Opus audio via
//! [`WebRtcWhepSrc::audio`] / `media=audio`; one m-line per source (a
//! `SourceLoop` has a single output pad). NAT traversal: a STUN server-reflexive
//! candidate via `stun-server`, and a TURN relay via
//! [`WebRtcWhepSrc::with_turn_server`] for the cases STUN cannot punch through.
//!
//! Status: compile-validated against str0m 0.20. The live subscribe path
//! (ICE/DTLS/SRTP handshake against a real WHEP server + real media, and the
//! TURN relay) is owed an on-network validation, like `WebRtcSink`; the sandbox
//! blocks the ports.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::time::Instant;

use tokio::net::UdpSocket;

use str0m::change::SdpAnswer;
use str0m::crypto::from_feature_flags;
use str0m::media::Direction;
use str0m::{Event, IceConnectionState, Input, Output, RtcConfig};

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, ElementMetadata, G2gError, HardwareError,
    LatencyReport, OutputSink, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};

use crate::filesink::io_err;
use crate::turn::{self, TurnSet};
use crate::webrtc_codec::{
    get_video_codec_property, opus_caps, set_video_codec_property, Track, WebRtcVideoCodec,
    VIDEO_CODEC_PROP,
};
use crate::webrtc_receive::{MediaReceiver, ReceivedUnit, UDP_RECV_BYTES};
use crate::webrtc_stats::{WebRtcStatsHandle, STATS_INTERVAL};
use crate::webrtc_util::{
    add_host_candidate, delete_resource, feed_datagram, ice_restart, post_sdp, select_host_ip,
    send_transmit, trickle_candidates, TricklePatch, TurnConfig, ICE_RESTART_TIMEOUT,
    STUN_SERVER_PROP, TURN_PASS_PROP, TURN_SERVER_PROP, TURN_USER_PROP,
};

/// Which media this source subscribes to. WHEP offers one recv-only m-line, and
/// a `SourceLoop` has one output pad, so the source carries a single track; the
/// kind is chosen up front (video by default, audio via [`WebRtcWhepSrc::audio`]
/// or `media=audio`), the video codec by `video-codec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Media {
    /// Video in the configured codec.
    Video,
    /// Opus audio.
    Audio,
}

/// WHEP-subscribing WebRTC ingest source. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::webrtcwhepsrc::WebRtcWhepSrc;
///
/// let source = WebRtcWhepSrc::new("http://localhost:8889/mystream/whep")
///     .with_stun_server("stun.l.google.com:19302")
///     .with_frame_limit(300);
/// ```
pub struct WebRtcWhepSrc {
    whep_url: String,
    bearer: Option<String>,
    /// STUN server (`host:port`) for ICE NAT traversal toward a cloud SFU.
    /// `None` = host candidate only (LAN / self-hosted same network).
    stun_server: Option<String>,
    /// TURN relay (`host:port`) + long-term credentials for the NAT cases a
    /// server-reflexive candidate cannot punch through. `None` = no relay.
    turn_server: Option<String>,
    turn_user: String,
    turn_pass: String,
    /// Which track to subscribe to (video by default, or Opus audio).
    media: Media,
    /// The video codec offered and emitted when `media` is video.
    video_codec: WebRtcVideoCodec,
    /// Stop after this many access units and emit EOS (`u64::MAX` = unbounded).
    /// The bounded path is for tests / smoke runs.
    frame_limit: u64,
    configured: bool,
    stats: WebRtcStatsHandle,
}

impl core::fmt::Debug for WebRtcWhepSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebRtcWhepSrc")
            .field("whep_url", &self.whep_url)
            .field("frame_limit", &self.frame_limit)
            .finish()
    }
}

impl WebRtcWhepSrc {
    /// Subscribe to the given WHEP endpoint URL (e.g.
    /// `http://localhost:8889/mystream/whep` on a mediamtx server).
    pub fn new(whep_url: impl Into<String>) -> Self {
        Self {
            whep_url: whep_url.into(),
            bearer: None,
            stun_server: None,
            turn_server: None,
            turn_user: String::new(),
            turn_pass: String::new(),
            media: Media::Video,
            video_codec: WebRtcVideoCodec::default(),
            frame_limit: u64::MAX,
            configured: false,
            stats: WebRtcStatsHandle::default(),
        }
    }

    /// Live counters of the WHEP session, readable during the run.
    pub fn stats_handle(&self) -> WebRtcStatsHandle {
        self.stats.clone()
    }

    /// Subscribe to video in `codec` instead of H.264 (the `video-codec`
    /// property).
    pub fn with_video_codec(mut self, codec: WebRtcVideoCodec) -> Self {
        self.video_codec = codec;
        self
    }

    /// Attach a bearer token, sent as `Authorization: Bearer <token>` on the
    /// WHEP POST.
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Set a STUN server (`host:port`, e.g. `stun.l.google.com:19302`) for ICE
    /// NAT traversal. Required to reach a cloud SFU from behind NAT; unset means
    /// host candidate only (works on a LAN).
    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.stun_server = Some(server.into());
        self
    }

    /// Set a TURN relay (`host:port`) with long-term credentials, the fallback
    /// for NAT/firewall situations a STUN server-reflexive candidate cannot
    /// traverse. Composes with [`Self::with_stun_server`]; ICE prefers the
    /// direct paths and falls back to the relay only if they fail.
    pub fn with_turn_server(
        mut self,
        server: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        self.turn_server = Some(server.into());
        self.turn_user = username.into();
        self.turn_pass = password.into();
        self
    }

    /// Subscribe to the Opus audio track instead of H.264 video. The source
    /// then produces `Caps::Audio { Opus }` and forwards each Opus packet.
    pub fn audio(mut self) -> Self {
        self.media = Media::Audio;
        self
    }

    /// Stop after `n` access units (then EOS). 0 emits EOS without POSTing to
    /// the server. For tests / bounded runs.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// The one track this source subscribes to.
    fn track(&self) -> Track {
        match self.media {
            Media::Video => Track::Video(self.video_codec),
            Media::Audio => Track::Audio,
        }
    }

    /// The produced caps for the configured track: a placeholder geometry for
    /// video (see `WebRtcVideoCodec::caps_placeholder`), stereo Opus for audio.
    fn caps(&self) -> Caps {
        match self.track() {
            Track::Video(codec) => codec.caps_placeholder(),
            Track::Audio => opus_caps(2),
        }
    }
}

impl SourceLoop for WebRtcWhepSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // The WHEP handshake is async + needs the socket, so it runs at the
        // top of `run`; configure only marks readiness (the m-line is fixed).
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "WebRTC source",
            "Source/Network/WebRTC",
            "Subscribes to a WHEP server over WebRTC and emits video (H.264/VP8/VP9/H.265/AV1) or Opus (str0m: ICE/DTLS/SRTP)",
            "g2g",
        )
    }

    /// Live source: one frame period of latency so the sink keeps a frame in
    /// hand. Unknown framerate at startup, so report the network-driven default.
    fn latency(&self) -> LatencyReport {
        LatencyReport::live(0, None)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WEBRTCSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" | "whep-url" => {
                self.whep_url = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "auth-token" => {
                let token = value.as_str().ok_or(PropError::Type)?;
                self.bearer = if token.is_empty() {
                    None
                } else {
                    Some(token.into())
                };
                Ok(())
            }
            "stun-server" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.stun_server = if s.is_empty() { None } else { Some(s.into()) };
                Ok(())
            }
            "turn-server" => {
                let s = value.as_str().ok_or(PropError::Type)?;
                self.turn_server = if s.is_empty() { None } else { Some(s.into()) };
                Ok(())
            }
            "turn-user" => {
                self.turn_user = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "turn-pass" => {
                self.turn_pass = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            // `video` / `audio` pick the subscribed track; default is video.
            "media" => {
                self.media = match value.as_str().ok_or(PropError::Type)? {
                    "audio" => Media::Audio,
                    "video" => Media::Video,
                    _ => return Err(PropError::Value),
                };
                Ok(())
            }
            "video-codec" => set_video_codec_property(&mut self.video_codec, &value),
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" | "whep-url" => Some(PropValue::Str(self.whep_url.clone())),
            "auth-token" => Some(PropValue::Str(self.bearer.clone().unwrap_or_default())),
            "stun-server" => Some(PropValue::Str(self.stun_server.clone().unwrap_or_default())),
            "turn-server" => Some(PropValue::Str(self.turn_server.clone().unwrap_or_default())),
            "turn-user" => Some(PropValue::Str(self.turn_user.clone())),
            "turn-pass" => Some(PropValue::Str(self.turn_pass.clone())),
            "media" => Some(PropValue::Str(
                match self.media {
                    Media::Video => "video",
                    Media::Audio => "audio",
                }
                .into(),
            )),
            "video-codec" => Some(get_video_codec_property(self.video_codec)),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if crate::numbuffers::finished_at_zero_limit(self.frame_limit, out).await? {
                return Ok(0);
            }
            let hw = || G2gError::Hardware(HardwareError::Other);

            // ICE host candidate + UDP socket (see WebRtcSink for the rationale).
            let host_ip = select_host_ip();
            let socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
            let local = socket.local_addr().map_err(io_err)?;

            let track = self.track();
            let config = RtcConfig::new()
                .set_crypto_provider(alloc::sync::Arc::new(from_feature_flags()))
                .set_stats_interval(Some(STATS_INTERVAL))
                .clear_codecs();
            let mut rtc = track.enable(config).build(Instant::now());
            // Trickle ICE: the offer carries the host candidate only; reflexive /
            // relay candidates are gathered after the POST and trickled by PATCH.
            add_host_candidate(&mut rtc, &socket)?;

            // WHEP: offer a single recv-only m-line for the chosen track, POST
            // it, apply the answer.
            let (offer_sdp, pending, mid) = {
                let mut api = rtc.sdp_api();
                let mid = api.add_media(track.media_kind(), Direction::RecvOnly, None, None, None);
                let (offer, pending) = api.apply().ok_or_else(hw)?;
                (offer.to_sdp_string(), pending, mid)
            };
            let session =
                post_sdp(&self.whep_url, self.bearer.as_deref(), offer_sdp.clone()).await?;
            let answer = SdpAnswer::from_sdp_string(&session.answer).map_err(|_| hw())?;
            rtc.sdp_api()
                .accept_answer(pending, answer)
                .map_err(|_| hw())?;

            // Gather reflexive / relay candidates and trickle them to the
            // resource; allocation failure degrades to host/srflx.
            let mut turn: TurnSet = trickle_candidates(
                &mut rtc,
                &socket,
                &offer_sdp,
                &session,
                self.bearer.as_deref(),
                self.stun_server.as_deref(),
                TurnConfig {
                    server: self.turn_server.as_deref(),
                    user: &self.turn_user,
                    pass: &self.turn_pass,
                },
            )
            .await;
            let resource = session.resource;
            let etag = session.etag;
            let bearer = self.bearer.clone();
            let mut disconnected_since: Option<Instant> = None;
            let mut refresh_at = Instant::now() + turn::REFRESH_INTERVAL;

            // Announce the produced caps before the first frame.
            out.push(PipelinePacket::CapsChanged(self.caps())).await?;

            let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
            let mut seq = 0u64;
            // Mid-GOP join: video is dropped until the first keyframe, with a
            // PLI asked on a coarse interval so playback starts without waiting
            // for the next natural IDR. See [[rtsp_first_keyframe]].
            let mut media_receiver = match track {
                Track::Video(codec) => MediaReceiver::with_mids(codec, Some(mid), None),
                Track::Audio => MediaReceiver::with_mids(self.video_codec, None, Some(mid)),
            };
            // Emit EOS and DELETE the WHEP resource (RFC 9725 teardown) on a clean
            // end.
            macro_rules! finish {
                () => {{
                    if let Some(res) = resource.as_deref() {
                        delete_resource(res, bearer.as_deref()).await;
                    }
                    out.push(PipelinePacket::Eos).await?;
                    return Ok(seq);
                }};
            }
            loop {
                // Drain str0m's output to a deadline, collecting decoded access
                // units to push after (poll_output is sync; pushes are async).
                let mut frames: Vec<ReceivedUnit> = Vec::new();
                let deadline = loop {
                    match rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => send_transmit(&socket, &mut turn, &t).await,
                        Ok(Output::Event(Event::MediaAdded(m))) => media_receiver.media_added(&m),
                        Ok(Output::Event(Event::MediaData(d))) => {
                            if let Some(unit) = media_receiver.media_data(&d) {
                                frames.push(unit);
                            }
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(state))) => match state {
                            IceConnectionState::Disconnected => {
                                disconnected_since.get_or_insert_with(Instant::now);
                            }
                            IceConnectionState::Connected | IceConnectionState::Completed => {
                                disconnected_since = None;
                            }
                            _ => {}
                        },
                        Ok(Output::Event(event)) => self.stats.record(&event),
                        Err(_) => finish!(),
                    }
                };

                // Sustained ICE disconnect: attempt an ICE restart against the
                // resource. A server that rejects it ends the source (fresh
                // re-POST reconnect is owned by the element restarting).
                if disconnected_since.is_some_and(|t| t.elapsed() >= ICE_RESTART_TIMEOUT) {
                    disconnected_since = None;
                    match resource.as_deref() {
                        Some(res) => {
                            if !matches!(
                                ice_restart(&mut rtc, res, bearer.as_deref(), etag.as_deref())
                                    .await,
                                TricklePatch::Accepted
                            ) {
                                finish!();
                            }
                        }
                        None => finish!(),
                    }
                }

                media_receiver.request_keyframe_if_due(&mut rtc);

                for unit in frames {
                    let frame = unit.into_frame(seq);
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                    seq += 1;
                    if seq >= self.frame_limit {
                        finish!();
                    }
                }

                let timeout = deadline.saturating_duration_since(Instant::now());
                tokio::select! {
                    r = socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else { finish!() };
                        let _ = feed_datagram(&mut rtc, &mut turn, local, &buf[..n], source);
                    }
                    // Keep the TURN allocation + permissions alive.
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(refresh_at)), if !turn.is_empty() => {
                        turn.refresh_all(&socket).await;
                        refresh_at = Instant::now() + turn::REFRESH_INTERVAL;
                    }
                    _ = tokio::time::sleep(timeout) => {
                        let _ = rtc.handle_input(Input::Timeout(Instant::now()));
                    }
                }
            }
        })
    }
}

/// `WebRtcWhepSrc`'s settable properties: the WHEP endpoint URL + an optional
/// bearer token, so a `gst-launch` line can target a server without the builder.
static WEBRTCSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "WHEP endpoint URL to subscribe to",
    ),
    PropertySpec::new(
        "auth-token",
        PropKind::Str,
        "optional Authorization: Bearer token for the WHEP POST",
    ),
    STUN_SERVER_PROP,
    TURN_SERVER_PROP,
    TURN_USER_PROP,
    TURN_PASS_PROP,
    PropertySpec::new(
        "media",
        PropKind::Str,
        "track to subscribe to: video or audio (Opus)",
    ),
    VIDEO_CODEC_PROP,
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "access units to emit then EOS (-1 = until the session ends)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
];

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{AudioFormat, VideoCodec};

    #[test]
    fn produces_h264_caps() {
        let src = WebRtcWhepSrc::new("http://localhost:8889/s/whep");
        assert!(matches!(
            src.caps(),
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            }
        ));
    }

    #[test]
    fn video_codec_selects_the_offered_codec_and_caps() {
        let src = WebRtcWhepSrc::new("http://h/whep").with_video_codec(WebRtcVideoCodec::Vp9);
        assert_eq!(src.track(), Track::Video(WebRtcVideoCodec::Vp9));
        assert!(matches!(
            src.caps(),
            Caps::CompressedVideo {
                codec: VideoCodec::Vp9,
                ..
            }
        ));
        let mut src = WebRtcWhepSrc::new("http://h/whep");
        src.set_property("video-codec", PropValue::Str("h265".into()))
            .unwrap();
        assert_eq!(
            src.get_property("video-codec"),
            Some(PropValue::Str("h265".into()))
        );
        assert_eq!(
            src.set_property("video-codec", PropValue::Str("mjpeg".into())),
            Err(PropError::Value)
        );
        // Audio ignores the video codec.
        assert_eq!(src.audio().track(), Track::Audio);
    }

    #[test]
    fn location_and_bearer_properties_round_trip() {
        let mut src = WebRtcWhepSrc::new("http://h/whep").with_frame_limit(10);
        assert_eq!(src.frame_limit, 10);
        src.set_property("location", PropValue::Str("http://srv/whep".into()))
            .unwrap();
        assert_eq!(src.whep_url, "http://srv/whep");
        assert_eq!(
            src.get_property("location"),
            Some(PropValue::Str("http://srv/whep".into()))
        );
        src.set_property("auth-token", PropValue::Str("tok".into()))
            .unwrap();
        assert_eq!(src.bearer.as_deref(), Some("tok"));
        src.set_property(
            "stun-server",
            PropValue::Str("stun.l.google.com:19302".into()),
        )
        .unwrap();
        assert_eq!(src.stun_server.as_deref(), Some("stun.l.google.com:19302"));
        assert_eq!(
            src.set_property("nope", PropValue::Str("x".into())),
            Err(PropError::Unknown)
        );
        assert_eq!(
            src.set_property("location", PropValue::Int(1)),
            Err(PropError::Type)
        );
    }

    #[test]
    fn turn_builder_and_properties() {
        let src = WebRtcWhepSrc::new("http://h/whep").with_turn_server("turn:3478", "u", "p");
        assert_eq!(src.turn_server.as_deref(), Some("turn:3478"));
        assert_eq!(src.turn_user, "u");
        assert_eq!(src.turn_pass, "p");

        let mut src = WebRtcWhepSrc::new("http://h/whep");
        src.set_property("turn-server", PropValue::Str("relay:3478".into()))
            .unwrap();
        src.set_property("turn-user", PropValue::Str("user".into()))
            .unwrap();
        src.set_property("turn-pass", PropValue::Str("secret".into()))
            .unwrap();
        assert_eq!(src.turn_server.as_deref(), Some("relay:3478"));
        assert_eq!(
            src.get_property("turn-user"),
            Some(PropValue::Str("user".into()))
        );
        // Empty turn-server clears the relay (host/srflx only).
        src.set_property("turn-server", PropValue::Str(String::new()))
            .unwrap();
        assert_eq!(src.turn_server, None);
    }

    #[test]
    fn audio_selects_opus_track_and_caps() {
        let src = WebRtcWhepSrc::new("http://h/whep").audio();
        assert_eq!(src.media, Media::Audio);
        assert!(matches!(
            src.caps(),
            Caps::Audio {
                format: AudioFormat::Opus,
                ..
            }
        ));

        // Default is video; the `media` property flips it and rejects garbage.
        let mut src = WebRtcWhepSrc::new("http://h/whep");
        assert_eq!(src.media, Media::Video);
        src.set_property("media", PropValue::Str("audio".into()))
            .unwrap();
        assert_eq!(src.media, Media::Audio);
        assert_eq!(
            src.get_property("media"),
            Some(PropValue::Str("audio".into()))
        );
        src.set_property("media", PropValue::Str("video".into()))
            .unwrap();
        assert_eq!(src.media, Media::Video);
        assert_eq!(
            src.set_property("media", PropValue::Str("subtitle".into())),
            Err(PropError::Value)
        );
    }

    #[test]
    fn run_before_configure_is_not_configured() {
        // configure_pipeline gates run; without it run returns NotConfigured.
        let src = WebRtcWhepSrc::new("http://h/whep");
        assert!(!src.configured);
    }
}
