//! Native WebRTC ingest source (`WebRtcWhepSrc`): subscribes to a WHEP endpoint
//! and emits the received H.264 as Annex-B `DataFrame`s. The receive-side
//! inverse of [`crate::webrtcsink::WebRtcSink`], on the same sans-IO `str0m`
//! stack (ICE / DTLS / SRTP), and distinct from the wasm-only data-channel
//! [`crate::webrtcsrc::WebRtcSrc`].
//!
//! WHEP is client-offers-recvonly: the source builds a str0m `Rtc` with a single
//! recv-only m-line for the chosen track, POSTs the SDP offer to the WHEP
//! endpoint, applies the answer, then drives str0m's `poll_output` /
//! `handle_input` loop on a tokio `UdpSocket`. Each `Event::MediaData` is a
//! depacketized access unit, forwarded downstream with its RTP-clock PTS: for
//! video the H.264 depayloader emits Annex-B start-code framing (exactly g2g's
//! convention); for audio each Opus packet is forwarded as-is.
//!
//! The track is video (H.264) by default, or Opus audio via
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

use core::time::Duration;
use std::time::Instant;

use tokio::net::UdpSocket;

use str0m::change::SdpAnswer;
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, KeyframeRequestKind, MediaKind, Mid};
use str0m::net::{Protocol, Receive};
use str0m::{Event, IceConnectionState, Input, Output, RtcConfig};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming,
    G2gError, HardwareError, LatencyReport, MemoryDomain, OutputSink, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::turn::{self, TurnClient};
use crate::webrtc_util::{add_ice_candidates, post_sdp, select_host_ip};

/// Minimum gap between keyframe (PLI) requests while waiting for the first one,
/// so a slow producer is not spammed every frame period.
const PLI_INTERVAL: Duration = Duration::from_secs(1);

/// Which media this source subscribes to. WHEP offers one recv-only m-line, and
/// a `SourceLoop` has one output pad, so the source carries a single track; the
/// kind is chosen up front (video by default, audio via [`WebRtcWhepSrc::audio`]
/// or `media=audio`). This is the receive-side mirror of the sink's `Track`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Media {
    /// H.264 video.
    Video,
    /// Opus audio.
    Audio,
}

impl Media {
    fn media_kind(self) -> MediaKind {
        match self {
            Media::Video => MediaKind::Video,
            Media::Audio => MediaKind::Audio,
        }
    }

    /// The caps this source produces for the chosen media. Video geometry is
    /// unknown until the in-band SPS, so it is advertised as a `Range`
    /// placeholder (a downstream H.264 parser recovers the real dimensions and
    /// re-announces them via `CapsChanged`): negotiation fixates before any data
    /// flows and `fixate()` rejects `Dim::Any`, so a wildcard would fail startup.
    /// Opus is the WebRTC default stereo 48 kHz.
    fn caps(self) -> Caps {
        match self {
            Media::Video => Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Range { min: 2, max: 8192 },
                height: Dim::Range { min: 2, max: 8192 },
                framerate: Rate::Range { min_q16: 1 << 16, max_q16: 240 << 16 },
            },
            Media::Audio => {
                Caps::Audio { format: AudioFormat::Opus, channels: 2, sample_rate: 48_000 }
            }
        }
    }
}

/// WHEP-subscribing WebRTC ingest source. See the module docs.
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
    /// Which track to subscribe to (H.264 video by default, or Opus audio).
    media: Media,
    /// Stop after this many access units and emit EOS (0 = unbounded). The
    /// bounded path is for tests / smoke runs.
    frame_limit: u64,
    configured: bool,
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
            frame_limit: 0,
            configured: false,
        }
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

    /// Stop after `n` access units (then EOS). For tests / bounded runs.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// The produced caps for the configured track (see [`Media::caps`]).
    fn caps(&self) -> Caps {
        self.media.caps()
    }
}

impl SourceLoop for WebRtcWhepSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
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
            "Subscribes to a WHEP server over WebRTC and emits H.264 or Opus (str0m: ICE/DTLS/SRTP)",
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
            "bearer" => {
                let token = value.as_str().ok_or(PropError::Type)?;
                self.bearer = if token.is_empty() { None } else { Some(token.into()) };
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
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" | "whep-url" => Some(PropValue::Str(self.whep_url.clone())),
            "bearer" => Some(PropValue::Str(self.bearer.clone().unwrap_or_default())),
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
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let hw = || G2gError::Hardware(HardwareError::Other);

            // ICE host candidate + UDP socket (see WebRtcSink for the rationale).
            let host_ip = select_host_ip();
            let socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
            let local = socket.local_addr().map_err(io_err)?;

            let media = self.media;
            let config = RtcConfig::new()
                .set_crypto_provider(alloc::sync::Arc::new(from_feature_flags()))
                .clear_codecs();
            let config = match media {
                Media::Video => config.enable_h264(true),
                Media::Audio => config.enable_opus(true),
            };
            let mut rtc = config.build(Instant::now());
            // Host candidate, plus a STUN server-reflexive candidate when set
            // (needed to reach a cloud SFU across NAT).
            add_ice_candidates(&mut rtc, &socket, self.stun_server.as_deref()).await?;

            // TURN relay candidate when configured (fallback for NATs STUN
            // cannot traverse); allocation failure degrades to host/srflx.
            let mut turn: Option<TurnClient> = match &self.turn_server {
                Some(server) => {
                    turn::setup(&mut rtc, &socket, server, &self.turn_user, &self.turn_pass).await
                }
                None => None,
            };
            let mut refresh_at = Instant::now() + turn::REFRESH_INTERVAL;

            // WHEP: offer a single recv-only m-line for the chosen track, POST
            // it, apply the answer.
            let (offer_sdp, pending) = {
                let mut api = rtc.sdp_api();
                api.add_media(media.media_kind(), Direction::RecvOnly, None, None, None);
                let (offer, pending) = api.apply().ok_or_else(hw)?;
                (offer.to_sdp_string(), pending)
            };
            let answer_sdp = post_sdp(&self.whep_url, self.bearer.as_deref(), offer_sdp).await?;
            let answer = SdpAnswer::from_sdp_string(&answer_sdp).map_err(|_| hw())?;
            rtc.sdp_api().accept_answer(pending, answer).map_err(|_| hw())?;

            // Announce the produced caps before the first frame.
            out.push(PipelinePacket::CapsChanged(self.caps())).await?;

            let mut buf = alloc::vec![0u8; 2000];
            let mut seq = 0u64;
            // Mid-GOP join recovery (video): until the first keyframe arrives, ask
            // the remote for one (PLI) on a coarse interval so playback starts
            // without waiting for the next natural IDR. See [[rtsp_first_keyframe]].
            let mut seen_keyframe = false;
            let mut last_pli: Option<Instant> = None;
            let mut media_mid: Option<Mid> = None;
            loop {
                // Drain str0m's output to a deadline, collecting decoded access
                // units to push after (poll_output is sync; pushes are async).
                let mut frames: Vec<(u64, Vec<u8>)> = Vec::new();
                let deadline = loop {
                    match rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => {
                            // Relay-sourced datagrams go through TURN; direct ones
                            // (host / srflx) go straight out.
                            match turn.as_mut() {
                                Some(tc) if t.source == tc.relay_addr() => {
                                    let _ = tc.ensure_permission(&socket, t.destination).await;
                                    let wrapped = tc.wrap_send(t.destination, &t.contents);
                                    let _ = socket.send_to(&wrapped, tc.server_addr()).await;
                                }
                                _ => {
                                    let _ = socket.send_to(&t.contents, t.destination).await;
                                }
                            }
                        }
                        Ok(Output::Event(Event::MediaData(d))) => {
                            // d.time is the RTP MediaTime (90 kHz for H.264);
                            // map its rational value to nanoseconds.
                            let denom = d.time.denom().max(1) as u128;
                            let pts_ns = (d.time.numer() as u128 * 1_000_000_000 / denom) as u64;
                            media_mid = Some(d.mid);
                            frames.push((pts_ns, d.data.to_vec()));
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => {
                            out.push(PipelinePacket::Eos).await?;
                            return Ok(seq);
                        }
                        Ok(Output::Event(_)) => {}
                        Err(_) => {
                            out.push(PipelinePacket::Eos).await?;
                            return Ok(seq);
                        }
                    }
                };

                for (pts_ns, data) in frames {
                    // Video keyframe = IDR (the sink/parser cares); every Opus
                    // packet is independently decodable, so the flag is moot.
                    let keyframe = match media {
                        Media::Video => crate::h264util::h264_au_is_keyframe(&data),
                        Media::Audio => false,
                    };
                    seen_keyframe |= keyframe;
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
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                    seq += 1;
                    if self.frame_limit != 0 && seq >= self.frame_limit {
                        out.push(PipelinePacket::Eos).await?;
                        return Ok(seq);
                    }
                }

                // Ask the remote for a keyframe while we are still waiting for the
                // first one (video only; Opus has no keyframes). PLI, rate-limited.
                if media == Media::Video && !seen_keyframe {
                    if let Some(mid) = media_mid {
                        let now = Instant::now();
                        let due = last_pli.map_or(true, |t| now.duration_since(t) >= PLI_INTERVAL);
                        if due {
                            if let Some(rx) = rtc.direct_api().stream_rx_by_mid(mid, None) {
                                rx.request_keyframe(KeyframeRequestKind::Pli);
                                last_pli = Some(now);
                            }
                        }
                    }
                }

                let timeout = deadline.saturating_duration_since(Instant::now());
                tokio::select! {
                    r = socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else {
                            out.push(PipelinePacket::Eos).await?;
                            return Ok(seq);
                        };
                        let from_turn = turn.as_ref().is_some_and(|tc| tc.is_server(source));
                        if from_turn {
                            // Unwrap a relayed Data indication and feed str0m as
                            // if it arrived on the relay candidate; control
                            // responses parse to None and are discarded.
                            if let Some(tc) = turn.as_mut() {
                                if let Some((peer, payload)) = tc.parse_data(&buf[..n]) {
                                    let relay = tc.relay_addr();
                                    if let Ok(contents) = payload.as_slice().try_into() {
                                        let input = Input::Receive(
                                            Instant::now(),
                                            Receive { proto: Protocol::Udp, source: peer, destination: relay, contents },
                                        );
                                        let _ = rtc.handle_input(input);
                                    }
                                }
                            }
                        } else if let Ok(contents) = (&buf[..n]).try_into() {
                            let input = Input::Receive(
                                Instant::now(),
                                Receive { proto: Protocol::Udp, source, destination: local, contents },
                            );
                            let _ = rtc.handle_input(input);
                        }
                    }
                    // Keep the TURN allocation + permissions alive.
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(refresh_at)), if turn.is_some() => {
                        if let Some(tc) = turn.as_mut() {
                            let _ = tc.refresh(&socket).await;
                        }
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
    PropertySpec::new("location", PropKind::Str, "WHEP endpoint URL to subscribe to"),
    PropertySpec::new("bearer", PropKind::Str, "optional Authorization: Bearer token for the WHEP POST"),
    PropertySpec::new(
        "stun-server",
        PropKind::Str,
        "STUN server host:port for ICE NAT traversal to a cloud SFU (empty = host-only)",
    ),
    PropertySpec::new(
        "turn-server",
        PropKind::Str,
        "TURN relay host:port for the NAT cases STUN cannot traverse (empty = no relay)",
    ),
    PropertySpec::new("turn-user", PropKind::Str, "TURN long-term credential username"),
    PropertySpec::new("turn-pass", PropKind::Str, "TURN long-term credential password"),
    PropertySpec::new("media", PropKind::Str, "track to subscribe to: video (H.264) or audio (Opus)"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn produces_h264_caps() {
        let src = WebRtcWhepSrc::new("http://localhost:8889/s/whep");
        assert!(matches!(
            src.caps(),
            Caps::CompressedVideo { codec: VideoCodec::H264, .. }
        ));
    }

    #[test]
    fn location_and_bearer_properties_round_trip() {
        let mut src = WebRtcWhepSrc::new("http://h/whep").with_frame_limit(10);
        assert_eq!(src.frame_limit, 10);
        src.set_property("location", PropValue::Str("http://srv/whep".into())).unwrap();
        assert_eq!(src.whep_url, "http://srv/whep");
        assert_eq!(src.get_property("location"), Some(PropValue::Str("http://srv/whep".into())));
        src.set_property("bearer", PropValue::Str("tok".into())).unwrap();
        assert_eq!(src.bearer.as_deref(), Some("tok"));
        src.set_property("stun-server", PropValue::Str("stun.l.google.com:19302".into())).unwrap();
        assert_eq!(src.stun_server.as_deref(), Some("stun.l.google.com:19302"));
        assert_eq!(src.set_property("nope", PropValue::Str("x".into())), Err(PropError::Unknown));
        assert_eq!(src.set_property("location", PropValue::Int(1)), Err(PropError::Type));
    }

    #[test]
    fn turn_builder_and_properties() {
        let src = WebRtcWhepSrc::new("http://h/whep").with_turn_server("turn:3478", "u", "p");
        assert_eq!(src.turn_server.as_deref(), Some("turn:3478"));
        assert_eq!(src.turn_user, "u");
        assert_eq!(src.turn_pass, "p");

        let mut src = WebRtcWhepSrc::new("http://h/whep");
        src.set_property("turn-server", PropValue::Str("relay:3478".into())).unwrap();
        src.set_property("turn-user", PropValue::Str("user".into())).unwrap();
        src.set_property("turn-pass", PropValue::Str("secret".into())).unwrap();
        assert_eq!(src.turn_server.as_deref(), Some("relay:3478"));
        assert_eq!(src.get_property("turn-user"), Some(PropValue::Str("user".into())));
        // Empty turn-server clears the relay (host/srflx only).
        src.set_property("turn-server", PropValue::Str(String::new())).unwrap();
        assert_eq!(src.turn_server, None);
    }

    #[test]
    fn audio_selects_opus_track_and_caps() {
        let src = WebRtcWhepSrc::new("http://h/whep").audio();
        assert_eq!(src.media, Media::Audio);
        assert!(matches!(src.caps(), Caps::Audio { format: AudioFormat::Opus, .. }));

        // Default is video; the `media` property flips it and rejects garbage.
        let mut src = WebRtcWhepSrc::new("http://h/whep");
        assert_eq!(src.media, Media::Video);
        src.set_property("media", PropValue::Str("audio".into())).unwrap();
        assert_eq!(src.media, Media::Audio);
        assert_eq!(src.get_property("media"), Some(PropValue::Str("audio".into())));
        src.set_property("media", PropValue::Str("video".into())).unwrap();
        assert_eq!(src.media, Media::Video);
        assert_eq!(src.set_property("media", PropValue::Str("subtitle".into())), Err(PropError::Value));
    }

    #[test]
    fn run_before_configure_is_not_configured() {
        // configure_pipeline gates run; without it run returns NotConfigured.
        let src = WebRtcWhepSrc::new("http://h/whep");
        assert!(!src.configured);
    }
}
