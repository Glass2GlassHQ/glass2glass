//! Native WebRTC egress sink (`WebRtcSink`): publishes a pipeline's encoded
//! H.264 to a WHIP server over a WebRTC PeerConnection, built on the sans-IO
//! [`str0m`] stack (ICE / DTLS / SRTP / RTP). This is the native counterpart of
//! the browser-only data-channel [`crate::webrtcsrc::WebRtcSrc`], and the
//! WebRTC analog of `RtmpSink` (publish encoded media to a server endpoint).
//!
//! Shape, mirroring the other tokio network sinks (`UdpSink`, `SrtSink`):
//! - `configure_pipeline` accepts `Caps::CompressedVideo { codec: H264 }`.
//! - On the first `DataFrame`, the sink performs the WHIP handshake: it builds a
//!   str0m `Rtc` offering a single send-only H.264 video m-line, POSTs the SDP
//!   offer to the WHIP endpoint (reqwest, `application/sdp`), applies the
//!   answer, then spawns a background task that owns the `Rtc` + a tokio
//!   `UdpSocket` and runs str0m's `poll_output` / `handle_input` loop.
//! - Each subsequent access unit is handed to that task over a bounded channel;
//!   the task feeds it to str0m's H.264 writer, which packetizes + SRTP-encrypts
//!   and emits the UDP datagrams. The element itself never touches the `Rtc`
//!   (it lives on the task), so `WebRtcSink` is naturally `Send`.
//!
//! str0m is sans-IO, so g2g owns the socket and the timer, exactly the contract
//! the `srt` / `rtspserver` modules already follow. The pure-Rust `rust-crypto`
//! backend is selected (no OpenSSL system dep). Behind the `webrtc` feature.
//!
//! Status: compile-validated against str0m 0.20. The live WHIP publish path
//! (real ICE host-candidate selection, the DTLS/SRTP handshake against a real
//! server, and browser playback) is owed an on-network validation, like the
//! other egress elements (`rtmpsink`); the sandbox blocks the required ports.
//!
//! H.264 framing: the sink forwards each access unit as g2g delivers it
//! (Annex-B, the pipeline convention); str0m's H.264 packetizer splits NAL
//! units for RTP. Confirming the exact framing str0m expects is part of the
//! owed runtime validation.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::SocketAddr;
use std::time::Instant;

use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use str0m::bwe::{Bitrate, BweKind};
use str0m::change::{SdpAnswer, SdpPendingOffer};
use str0m::crypto::from_feature_flags;
use str0m::format::Codec;
use str0m::media::{Direction, Frequency, MediaKind, MediaTime, Mid, Pt};
use str0m::{Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::{
    AsyncElement, AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, G2gError, HardwareError, MemoryDomain, OutputSink, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, Reconfigure, VideoCodec,
};

use crate::filesink::io_err;
use crate::turn::{self, TurnClient};
use crate::webrtc_util::{add_ice_candidates, feed_datagram, post_sdp, select_host_ip, send_transmit};

/// Default bounded depth of the element->session media channel. Backpressures
/// the pipeline if the session task falls behind the encoder.
const DEFAULT_QUEUE_DEPTH: usize = 256;

/// Initial BWE estimate seeded into str0m before any feedback arrives. The
/// congestion controller adapts from here as TWCC/REMB reports come in.
const INITIAL_BITRATE_BPS: u64 = 2_000_000;

/// One encoded access unit handed from the element to the session task.
#[derive(Debug)]
struct MediaUnit {
    pts_ns: u64,
    data: Vec<u8>,
}

/// Which media this sink carries, chosen from the configured caps. WebRTC needs
/// the codec, the m-line `MediaKind`, and the RTP clock to agree; one input pad
/// means one track per sink (simultaneous A/V over one PeerConnection is a
/// `MultiInputElement` follow-up).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Track {
    /// H.264 video on a 90 kHz clock.
    Video,
    /// Opus audio on a 48 kHz clock.
    Audio,
}

impl Track {
    pub(crate) fn media_kind(self) -> MediaKind {
        match self {
            Track::Video => MediaKind::Video,
            Track::Audio => MediaKind::Audio,
        }
    }

    pub(crate) fn codec(self) -> Codec {
        match self {
            Track::Video => Codec::H264,
            Track::Audio => Codec::Opus,
        }
    }

    pub(crate) fn frequency(self) -> Frequency {
        match self {
            Track::Video => Frequency::NINETY_KHZ,
            Track::Audio => Frequency::FORTY_EIGHT_KHZ,
        }
    }

    fn rate_hz(self) -> u64 {
        match self {
            Track::Video => 90_000,
            Track::Audio => 48_000,
        }
    }

    /// Map a nanosecond PTS to this track's RTP `MediaTime`. `u128` intermediate
    /// avoids overflow on large timestamps.
    pub(crate) fn media_time(self, pts_ns: u64) -> MediaTime {
        let ticks = (pts_ns as u128 * self.rate_hz() as u128 / 1_000_000_000) as u64;
        MediaTime::new(ticks, self.frequency())
    }
}

/// WHIP-publishing WebRTC egress sink. See the module docs.
pub struct WebRtcSink {
    whip_url: String,
    bearer: Option<String>,
    /// STUN server (`host:port`) for ICE NAT traversal toward a cloud SFU.
    /// `None` = host candidate only (LAN / self-hosted same network).
    stun_server: Option<String>,
    /// TURN relay (`host:port`) + long-term credentials for the NAT cases a
    /// server-reflexive candidate cannot punch through. `None` = no relay.
    turn_server: Option<String>,
    turn_user: String,
    turn_pass: String,
    queue_depth: usize,
    configured: bool,
    /// The media kind, decided from the configured caps (H.264 video or Opus
    /// audio). Defaults to video until `configure_pipeline` runs.
    track: Track,
    /// Set on the first frame, after the WHIP handshake spawns the session task.
    tx: Option<mpsc::Sender<MediaUnit>>,
    /// Set by the session task when the remote sends a PLI (keyframe request);
    /// read + cleared by `take_reconfigure`, which forwards a
    /// `Reconfigure::ForceKeyframe` up the reverse channel to the encoder. Shared
    /// because the str0m loop lives on a separate task.
    keyframe_requested: Arc<AtomicBool>,
    /// Latest congestion-control / BWE estimate (bits/second, 0 = none yet),
    /// written by the session task from `Event::EgressBitrateEstimate`. Shared
    /// with the task; `take_bitrate` forwards changes up the reverse channel.
    bitrate_estimate: Arc<AtomicU64>,
    /// Last estimate `take_bitrate` forwarded, so it only reports changes (it is
    /// polled every frame; the encoder need not see the same value repeatedly).
    last_bitrate_sent: u64,
    frames_sent: u64,
}

impl core::fmt::Debug for WebRtcSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebRtcSink")
            .field("whip_url", &self.whip_url)
            .field("configured", &self.configured)
            .field("frames_sent", &self.frames_sent)
            .finish()
    }
}

impl WebRtcSink {
    /// Publish to the given WHIP endpoint URL (e.g.
    /// `http://localhost:8889/mystream/whip` on a mediamtx server).
    pub fn new(whip_url: impl Into<String>) -> Self {
        Self {
            whip_url: whip_url.into(),
            bearer: None,
            stun_server: None,
            turn_server: None,
            turn_user: String::new(),
            turn_pass: String::new(),
            queue_depth: DEFAULT_QUEUE_DEPTH,
            configured: false,
            track: Track::Video,
            tx: None,
            keyframe_requested: Arc::new(AtomicBool::new(false)),
            bitrate_estimate: Arc::new(AtomicU64::new(0)),
            last_bitrate_sent: 0,
            frames_sent: 0,
        }
    }

    /// Attach a bearer token, sent as `Authorization: Bearer <token>` on the
    /// WHIP POST (some servers require it).
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Override the bounded element->session media-channel depth.
    pub fn with_queue_depth(mut self, depth: usize) -> Self {
        self.queue_depth = depth.max(1);
        self
    }

    /// Set a STUN server (`host:port`, e.g. `stun.l.google.com:19302`) for ICE
    /// NAT traversal. Required to reach a cloud SFU (LiveKit Cloud, etc.) from
    /// behind NAT; unset means host candidate only (works on a LAN).
    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.stun_server = Some(server.into());
        self
    }

    /// Set a TURN relay (`host:port`) with long-term credentials. The relay is
    /// the fallback for NAT/firewall situations a STUN server-reflexive
    /// candidate cannot traverse (symmetric NAT, restrictive networks); LiveKit
    /// Cloud and other SFUs sometimes require it. Compose with
    /// [`Self::with_stun_server`]: ICE picks the relay only if the direct paths
    /// fail.
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

    /// Number of access units handed to the WebRTC session so far.
    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// Build the `Rtc`, do the WHIP offer/answer exchange, and spawn the session
    /// task. Runs on the first frame because it is async (the runner drives
    /// `process` inside a tokio runtime, as for the other network sinks).
    async fn start_session(&mut self) -> Result<(), G2gError> {
        // ICE needs a routable host candidate; pick the route-local IP via the
        // UDP connect trick (no packet is sent, the OS just resolves the source
        // address for the route). Falls back to loopback for offline use.
        let host_ip = select_host_ip();
        let socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
        let local = socket.local_addr().map_err(io_err)?;

        // rust-crypto backend (pure Rust, no OpenSSL); offer only this track's
        // codec so the answer's payload type is unambiguous.
        let config = RtcConfig::new()
            .set_crypto_provider(Arc::new(from_feature_flags()))
            // Congestion control: str0m runs the TWCC/REMB estimator and emits
            // `EgressBitrateEstimate`; the sink relays it to the encoder (BWE).
            .enable_bwe(Some(Bitrate::bps(INITIAL_BITRATE_BPS)))
            .clear_codecs();
        let config = match self.track {
            Track::Video => config.enable_h264(true),
            Track::Audio => config.enable_opus(true),
        };
        let mut rtc = config.build(Instant::now());

        // Host candidate, plus a STUN-discovered server-reflexive candidate when
        // a STUN server is set (needed to reach a cloud SFU across NAT).
        add_ice_candidates(&mut rtc, &socket, self.stun_server.as_deref()).await?;

        // TURN relay candidate when configured: the fallback path for NATs a
        // server-reflexive candidate cannot traverse. Allocation failure degrades
        // gracefully to the host/srflx candidates (the publish still attempts).
        let turn = match &self.turn_server {
            Some(server) => {
                turn::setup(&mut rtc, &socket, server, &self.turn_user, &self.turn_pass).await
            }
            None => None,
        };

        // Offer a single send-only m-line for the configured track.
        let (offer_sdp, pending, mid): (String, SdpPendingOffer, Mid) = {
            let mut api = rtc.sdp_api();
            let mid = api.add_media(self.track.media_kind(), Direction::SendOnly, None, None, None);
            let (offer, pending) = api.apply().ok_or(G2gError::Hardware(HardwareError::Other))?;
            (offer.to_sdp_string(), pending, mid)
        };

        // WHIP: POST the offer, receive the answer SDP, apply it.
        let answer_sdp = post_sdp(&self.whip_url, self.bearer.as_deref(), offer_sdp).await?;
        let answer = SdpAnswer::from_sdp_string(&answer_sdp).map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        rtc.sdp_api().accept_answer(pending, answer).map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        let (tx, rx) = mpsc::channel::<MediaUnit>(self.queue_depth);
        let keyframe_requested = Arc::clone(&self.keyframe_requested);
        let bitrate_estimate = Arc::clone(&self.bitrate_estimate);
        tokio::spawn(run_session(
            rtc,
            socket,
            local,
            mid,
            self.track,
            turn,
            keyframe_requested,
            bitrate_estimate,
            rx,
        ));
        self.tx = Some(tx);
        Ok(())
    }
}

/// `WebRtcSink`'s settable properties: the WHIP endpoint URL and an optional
/// bearer token, so a `gst-launch` line can target a server without the builder.
static WEBRTCSINK_PROPS: &[PropertySpec] = &[
    PropertySpec::new("location", PropKind::Str, "WHIP endpoint URL to publish to"),
    PropertySpec::new("bearer", PropKind::Str, "optional Authorization: Bearer token for the WHIP POST"),
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
];

/// The H.264 sink caps this element accepts (any geometry / framerate).
fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// The Opus audio caps this element accepts. Stereo 48 kHz is the WebRTC /
/// `opusenc` default; other channel counts / rates are a follow-up (the audio
/// `Caps` has no wildcard fields, so the declared sink caps must be concrete).
fn opus_stereo() -> Caps {
    Caps::Audio { format: AudioFormat::Opus, channels: 2, sample_rate: 48_000 }
}

impl AsyncElement for WebRtcSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Accept H.264 video or Opus audio; one track per sink instance.
        match upstream_caps {
            Caps::CompressedVideo { codec: VideoCodec::H264, .. }
            | Caps::Audio { format: AudioFormat::Opus, .. } => Ok(upstream_caps.clone()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(Vec::from([h264_any(), opus_stereo()])))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.track = match absolute_caps {
            Caps::CompressedVideo { codec: VideoCodec::H264, .. } => Track::Video,
            Caps::Audio { format: AudioFormat::Opus, .. } => Track::Audio,
            _ => return Err(G2gError::CapsMismatch),
        };
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "WebRTC sink",
            "Sink/Network/WebRTC",
            "Publishes H.264 to a WHIP server over WebRTC (str0m: ICE/DTLS/SRTP)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WEBRTCSINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            // `location` is the gst-canonical name; `whip-url` is accepted too.
            "location" | "whip-url" => {
                self.whip_url = value.as_str().ok_or(PropError::Type)?.into();
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
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" | "whip-url" => Some(PropValue::Str(self.whip_url.clone())),
            "bearer" => Some(PropValue::Str(self.bearer.clone().unwrap_or_default())),
            "stun-server" => Some(PropValue::Str(self.stun_server.clone().unwrap_or_default())),
            "turn-server" => Some(PropValue::Str(self.turn_server.clone().unwrap_or_default())),
            "turn-user" => Some(PropValue::Str(self.turn_user.clone())),
            "turn-pass" => Some(PropValue::Str(self.turn_pass.clone())),
            _ => None,
        }
    }

    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        // The session task set this on a remote PLI; clear and forward it as a
        // keyframe request to the upstream encoder.
        if self.keyframe_requested.swap(false, Ordering::Relaxed) {
            Some(Reconfigure::ForceKeyframe)
        } else {
            None
        }
    }

    fn take_bitrate(&mut self) -> Option<u32> {
        // Forward the session task's latest BWE estimate, but only when it
        // changed (this is polled every frame; the encoder need not re-see it).
        let cur = self.bitrate_estimate.load(Ordering::Relaxed);
        if cur != 0 && cur != self.last_bitrate_sent {
            self.last_bitrate_sent = cur;
            Some(cur.min(u32::MAX as u64) as u32)
        } else {
            None
        }
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let unit =
                        MediaUnit { pts_ns: frame.timing.pts_ns, data: slice.as_slice().to_vec() };
                    if self.tx.is_none() {
                        self.start_session().await?;
                    }
                    if let Some(tx) = &self.tx {
                        // Bounded send: backpressures the pipeline if the session
                        // task is behind. A closed channel means the session ended
                        // (ICE disconnect); surface it as shutdown.
                        tx.send(unit).await.map_err(|_| G2gError::Shutdown)?;
                    }
                    self.frames_sent += 1;
                }
                // The session keeps running on the spawned task; dropping the
                // sender on element drop closes the channel and the task exits
                // once the peer disconnects. (Graceful WHIP DELETE on EOS is a
                // follow-up.)
                PipelinePacket::Eos => {}
                // Geometry refinement lives in the in-band SPS, not the m-line.
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Flush => {}
                PipelinePacket::Segment(_) => {}
            }
            Ok(())
        })
    }
}

/// The sans-IO driving loop: owns the `Rtc` and the UDP socket, drains
/// `poll_output` to a deadline, then waits on incoming UDP, an outgoing access
/// unit, or the str0m timeout, whichever comes first.
#[allow(clippy::too_many_arguments)]
async fn run_session(
    mut rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    mid: Mid,
    track: Track,
    mut turn: Option<TurnClient>,
    keyframe_requested: Arc<AtomicBool>,
    bitrate_estimate: Arc<AtomicU64>,
    mut rx: mpsc::Receiver<MediaUnit>,
) {
    let mut buf = alloc::vec![0u8; 2000];
    // The negotiated payload type for this track's codec, discovered once the
    // writer exists.
    let mut pt: Option<Pt> = None;
    // Keep the TURN allocation + permissions alive while the session runs.
    let mut refresh_at = Instant::now() + turn::REFRESH_INTERVAL;

    loop {
        // Drain pending output until str0m asks us to wait for a deadline.
        let deadline = loop {
            match rtc.poll_output() {
                Ok(Output::Timeout(t)) => break t,
                Ok(Output::Transmit(t)) => send_transmit(&socket, &mut turn, &t).await,
                Ok(Output::Event(Event::IceConnectionStateChange(
                    IceConnectionState::Disconnected,
                ))) => return,
                // Remote PLI: the peer lost data and needs a fresh keyframe.
                // Flag it; the element's `take_reconfigure` forwards a
                // `ForceKeyframe` up the reverse channel to the encoder.
                Ok(Output::Event(Event::KeyframeRequest(_))) => {
                    keyframe_requested.store(true, Ordering::Relaxed);
                }
                // Congestion-control estimate: stash the latest target bitrate
                // (TWCC or REMB carry a `Bitrate`); `take_bitrate` relays changes.
                Ok(Output::Event(Event::EgressBitrateEstimate(kind))) => {
                    let bps = match kind {
                        BweKind::Twcc(b) | BweKind::Remb(_, b) => Some(b.as_u64()),
                        // `BweKind` is non-exhaustive; ignore unknown future kinds.
                        _ => None,
                    };
                    if let Some(bps) = bps {
                        bitrate_estimate.store(bps, Ordering::Relaxed);
                    }
                }
                Ok(Output::Event(_)) => {}
                Err(_) => return,
            }
        };

        let timeout = deadline.saturating_duration_since(Instant::now());

        tokio::select! {
            // Incoming UDP (STUN / DTLS / RTCP from the peer, or TURN-framed
            // traffic from the relay server).
            r = socket.recv_from(&mut buf) => {
                let Ok((n, source)) = r else { return };
                if !feed_datagram(&mut rtc, &mut turn, local, &buf[..n], source) {
                    return;
                }
            }
            // Keep the TURN allocation + permissions alive.
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(refresh_at)), if turn.is_some() => {
                if let Some(tc) = turn.as_mut() {
                    let _ = tc.refresh(&socket).await;
                }
                refresh_at = Instant::now() + turn::REFRESH_INTERVAL;
            }
            // An encoded access unit to publish.
            unit = rx.recv() => {
                let Some(unit) = unit else { return };
                if pt.is_none() {
                    if let Some(writer) = rtc.writer(mid) {
                        pt = writer
                            .payload_params()
                            .find(|p| p.spec().codec == track.codec())
                            .map(|p| p.pt());
                    }
                }
                if let Some(p) = pt {
                    let rtp_time = track.media_time(unit.pts_ns);
                    if let Some(writer) = rtc.writer(mid) {
                        let _ = writer.write(p, Instant::now(), rtp_time, unit.data);
                    }
                }
            }
            // str0m's timer fired.
            _ = tokio::time::sleep(timeout) => {
                if rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_h264_rejects_raw() {
        let sink = WebRtcSink::new("http://localhost:8889/s/whip");
        assert!(sink.intercept_caps(&h264_any()).is_ok());
        let rgba = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert_eq!(sink.intercept_caps(&rgba), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn configure_requires_h264() {
        let mut sink = WebRtcSink::new("http://localhost:8889/s/whip");
        assert!(sink.configure_pipeline(&h264_any()).is_ok());
        let mut sink2 = WebRtcSink::new("http://localhost:8889/s/whip");
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::I420,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert!(matches!(sink2.configure_pipeline(&raw), Err(G2gError::CapsMismatch)));
    }

    #[test]
    fn builders_set_fields() {
        let sink = WebRtcSink::new("http://h/whip")
            .with_bearer("tok")
            .with_queue_depth(8)
            .with_turn_server("turn:3478", "u", "p");
        assert_eq!(sink.bearer.as_deref(), Some("tok"));
        assert_eq!(sink.queue_depth, 8);
        assert_eq!(sink.turn_server.as_deref(), Some("turn:3478"));
        assert_eq!(sink.turn_user, "u");
        assert_eq!(sink.turn_pass, "p");
        assert_eq!(sink.frames_sent(), 0);
    }

    #[test]
    fn location_and_bearer_properties_round_trip() {
        let mut sink = WebRtcSink::new("http://h/whip");
        sink.set_property("location", PropValue::Str("http://srv:8889/s/whip".into())).unwrap();
        assert_eq!(sink.whip_url, "http://srv:8889/s/whip");
        assert_eq!(sink.get_property("location"), Some(PropValue::Str("http://srv:8889/s/whip".into())));
        // `whip-url` is an accepted alias for the same field.
        sink.set_property("whip-url", PropValue::Str("http://x/whip".into())).unwrap();
        assert_eq!(sink.whip_url, "http://x/whip");
        sink.set_property("bearer", PropValue::Str("secret".into())).unwrap();
        assert_eq!(sink.bearer.as_deref(), Some("secret"));
        sink.set_property("stun-server", PropValue::Str("stun.l.google.com:19302".into())).unwrap();
        assert_eq!(sink.stun_server.as_deref(), Some("stun.l.google.com:19302"));
        sink.set_property("turn-server", PropValue::Str("relay:3478".into())).unwrap();
        sink.set_property("turn-user", PropValue::Str("u".into())).unwrap();
        sink.set_property("turn-pass", PropValue::Str("p".into())).unwrap();
        assert_eq!(sink.turn_server.as_deref(), Some("relay:3478"));
        assert_eq!(sink.get_property("turn-user"), Some(PropValue::Str("u".into())));
        // Empty turn-server clears the relay.
        sink.set_property("turn-server", PropValue::Str(String::new())).unwrap();
        assert_eq!(sink.turn_server, None);
        // Empty bearer clears it; unknown name and wrong type are distinct errors.
        sink.set_property("bearer", PropValue::Str(String::new())).unwrap();
        assert_eq!(sink.bearer, None);
        assert_eq!(sink.set_property("nope", PropValue::Str("x".into())), Err(PropError::Unknown));
        assert_eq!(sink.set_property("location", PropValue::Int(1)), Err(PropError::Type));
    }

    #[test]
    fn media_time_uses_the_track_clock() {
        // Video: 90 kHz. 1 s -> 90000 ticks; 1/30 s -> ~3000.
        assert_eq!(Track::Video.media_time(1_000_000_000).numer(), 90_000);
        assert_eq!(Track::Video.media_time(33_333_333).numer(), 2_999);
        assert_eq!(Track::Video.frequency(), Frequency::NINETY_KHZ);
        // Audio: 48 kHz. 1 s -> 48000 ticks; 20 ms Opus frame -> 960 samples.
        assert_eq!(Track::Audio.media_time(1_000_000_000).numer(), 48_000);
        assert_eq!(Track::Audio.media_time(20_000_000).numer(), 960);
        assert_eq!(Track::Audio.frequency(), Frequency::FORTY_EIGHT_KHZ);
    }

    #[test]
    fn keyframe_request_flag_surfaces_as_force_keyframe() {
        let sink = WebRtcSink::new("http://h/whip");
        // No PLI yet: nothing to forward.
        let mut s = sink;
        assert_eq!(s.take_reconfigure(), None);
        // Session task observed a remote PLI -> sets the shared flag.
        s.keyframe_requested.store(true, Ordering::Relaxed);
        assert_eq!(s.take_reconfigure(), Some(Reconfigure::ForceKeyframe));
        // Consumed: a second poll yields nothing until the next PLI.
        assert_eq!(s.take_reconfigure(), None);
    }

    #[test]
    fn bitrate_estimate_surfaces_as_take_bitrate_on_change() {
        let mut s = WebRtcSink::new("http://h/whip");
        // No estimate yet.
        assert_eq!(s.take_bitrate(), None);
        // Session task stored a BWE estimate.
        s.bitrate_estimate.store(1_500_000, Ordering::Relaxed);
        assert_eq!(s.take_bitrate(), Some(1_500_000));
        // Same value again: not re-reported (the encoder need not re-see it).
        assert_eq!(s.take_bitrate(), None);
        // A new estimate is reported.
        s.bitrate_estimate.store(900_000, Ordering::Relaxed);
        assert_eq!(s.take_bitrate(), Some(900_000));
    }

    #[test]
    fn accepts_opus_audio() {
        let sink = WebRtcSink::new("http://h/whip");
        assert!(sink.intercept_caps(&opus_stereo()).is_ok());
        let mut s = WebRtcSink::new("http://h/whip");
        assert!(s.configure_pipeline(&opus_stereo()).is_ok());
        assert_eq!(s.track, Track::Audio);
    }
}
