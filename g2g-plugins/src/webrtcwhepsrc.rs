//! Native WebRTC ingest source (`WebRtcWhepSrc`): subscribes to a WHEP endpoint
//! and emits the received H.264 as Annex-B `DataFrame`s. The receive-side
//! inverse of [`crate::webrtcsink::WebRtcSink`], on the same sans-IO `str0m`
//! stack (ICE / DTLS / SRTP), and distinct from the wasm-only data-channel
//! [`crate::webrtcsrc::WebRtcSrc`].
//!
//! WHEP is client-offers-recvonly: the source builds a str0m `Rtc` with a single
//! recv-only H.264 m-line, POSTs the SDP offer to the WHEP endpoint, applies the
//! answer, then drives str0m's `poll_output` / `handle_input` loop on a tokio
//! `UdpSocket`. Each `Event::MediaData` is a depacketized Annex-B access unit
//! (str0m's H.264 depayloader emits start-code framing, which is exactly g2g's
//! convention), forwarded downstream with its RTP-clock PTS.
//!
//! Status: compile-validated against str0m 0.20. The live subscribe path
//! (ICE/DTLS/SRTP handshake against a real WHEP server + real media) is owed an
//! on-network validation, like `WebRtcSink`; the sandbox blocks the ports.
//! v1 is video-only (H.264); an Opus audio track is a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use std::time::Instant;

use tokio::net::UdpSocket;

use str0m::change::SdpAnswer;
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, MediaKind};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, RtcConfig};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, FrameTiming, G2gError,
    HardwareError, LatencyReport, MemoryDomain, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::webrtc_util::{post_sdp, select_host_ip};

/// WHEP-subscribing WebRTC ingest source. See the module docs.
pub struct WebRtcWhepSrc {
    whep_url: String,
    bearer: Option<String>,
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
        Self { whep_url: whep_url.into(), bearer: None, frame_limit: 0, configured: false }
    }

    /// Attach a bearer token, sent as `Authorization: Bearer <token>` on the
    /// WHEP POST.
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Stop after `n` access units (then EOS). For tests / bounded runs.
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// The produced caps: H.264 with geometry unknown until the in-band SPS, so
    /// a downstream parser / decoder recovers the real dimensions.
    fn caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
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
            "Subscribes to a WHEP server over WebRTC and emits H.264 (str0m: ICE/DTLS/SRTP)",
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
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" | "whep-url" => Some(PropValue::Str(self.whep_url.clone())),
            "bearer" => Some(PropValue::Str(self.bearer.clone().unwrap_or_default())),
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

            let mut rtc = RtcConfig::new()
                .set_crypto_provider(alloc::sync::Arc::new(from_feature_flags()))
                .clear_codecs()
                .enable_h264(true)
                .build(Instant::now());
            rtc.add_local_candidate(Candidate::host(local, "udp").map_err(|_| hw())?);

            // WHEP: offer a single recv-only H.264 m-line, POST it, apply answer.
            let (offer_sdp, pending) = {
                let mut api = rtc.sdp_api();
                api.add_media(MediaKind::Video, Direction::RecvOnly, None, None, None);
                let (offer, pending) = api.apply().ok_or_else(hw)?;
                (offer.to_sdp_string(), pending)
            };
            let answer_sdp = post_sdp(&self.whep_url, self.bearer.as_deref(), offer_sdp).await?;
            let answer = SdpAnswer::from_sdp_string(&answer_sdp).map_err(|_| hw())?;
            rtc.sdp_api().accept_answer(pending, answer).map_err(|_| hw())?;

            // Announce the H.264 caps before the first frame.
            out.push(PipelinePacket::CapsChanged(self.caps())).await?;

            let mut buf = alloc::vec![0u8; 2000];
            let mut seq = 0u64;
            loop {
                // Drain str0m's output to a deadline, collecting decoded access
                // units to push after (poll_output is sync; pushes are async).
                let mut frames: Vec<(u64, Vec<u8>)> = Vec::new();
                let deadline = loop {
                    match rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => {
                            let _ = socket.send_to(&t.contents, t.destination).await;
                        }
                        Ok(Output::Event(Event::MediaData(d))) => {
                            // d.time is the RTP MediaTime (90 kHz for H.264);
                            // map its rational value to nanoseconds.
                            let denom = d.time.denom().max(1) as u128;
                            let pts_ns = (d.time.numer() as u128 * 1_000_000_000 / denom) as u64;
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
                    let keyframe = crate::h264util::h264_au_is_keyframe(&data);
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

                let timeout = deadline.saturating_duration_since(Instant::now());
                tokio::select! {
                    r = socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else {
                            out.push(PipelinePacket::Eos).await?;
                            return Ok(seq);
                        };
                        if let Ok(contents) = (&buf[..n]).try_into() {
                            let input = Input::Receive(
                                Instant::now(),
                                Receive { proto: Protocol::Udp, source, destination: local, contents },
                            );
                            let _ = rtc.handle_input(input);
                        }
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
        assert_eq!(src.set_property("nope", PropValue::Str("x".into())), Err(PropError::Unknown));
        assert_eq!(src.set_property("location", PropValue::Int(1)), Err(PropError::Type));
    }

    #[test]
    fn run_before_configure_is_not_configured() {
        // configure_pipeline gates run; without it run returns NotConfigured.
        let src = WebRtcWhepSrc::new("http://h/whep");
        assert!(!src.configured);
    }
}
