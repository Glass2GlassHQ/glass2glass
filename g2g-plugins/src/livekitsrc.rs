//! Native LiveKit subscriber source (`LiveKitSrc`): joins a room over the
//! LiveKit WebSocket signalling protocol and emits the subscribed H.264 video
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
//! Shape: a [`MultiOutputSource`] (output 0 = H.264 video, output 1 = Opus
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

use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use tokio::net::{TcpStream, UdpSocket};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use str0m::change::SdpOffer;
use str0m::crypto::from_feature_flags;
use str0m::media::{KeyframeRequestKind, MediaKind, Mid};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, Caps, Dim, FrameTiming, G2gError, HardwareError, MemoryDomain, MultiOutputSink,
    MultiOutputSource, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    VideoCodec,
};

use crate::filesink::io_err;
use crate::livekit_signal::{
    candidate_from_init_json, mint_token, signal_ws_url, SessionDescription, SignalRequest,
    SignalResponse, SignalTarget, VideoGrant,
};
use crate::webrtc_util::add_ice_candidates;

/// Output port for the H.264 video track.
const VIDEO_PORT: usize = 0;
/// Output port for the Opus audio track.
const AUDIO_PORT: usize = 1;
/// Repeat the PLI at this interval until the first keyframe arrives.
const PLI_INTERVAL: Duration = Duration::from_secs(1);
/// Access-token lifetime, matching the sink.
const TOKEN_TTL_SECS: u64 = 3600;

/// Native LiveKit room subscriber. See the module docs.
pub struct LiveKitSrc {
    url: String,
    room: String,
    identity: String,
    api_key: String,
    api_secret: String,
    token: Option<String>,
    /// Stop after this many access units across both tracks and emit EOS
    /// (0 = unbounded). For tests / bounded runs.
    frame_limit: u64,
}

impl core::fmt::Debug for LiveKitSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("LiveKitSrc")
            .field("url", &self.url)
            .field("room", &self.room)
            .field("identity", &self.identity)
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
            url: url.into(),
            room: room.into(),
            identity: identity.into(),
            api_key: String::new(),
            api_secret: String::new(),
            token: None,
            frame_limit: 0,
        }
    }

    /// Mint the access token locally from an API key + secret.
    pub fn with_api_key(mut self, key: impl Into<String>, secret: impl Into<String>) -> Self {
        self.api_key = key.into();
        self.api_secret = secret.into();
        self
    }

    /// Use a pre-minted access token (LiveKit Cloud, external token service).
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Stop after `n` access units across both tracks (then EOS on both).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    fn access_token(&self) -> Result<String, G2gError> {
        if let Some(t) = &self.token {
            return Ok(t.clone());
        }
        if self.api_key.is_empty() || self.api_secret.is_empty() {
            std::eprintln!("livekit: no token and no api key/secret configured");
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let grant = VideoGrant::subscriber(self.room.clone());
        Ok(mint_token(
            &self.api_key,
            &self.api_secret,
            &self.identity,
            &grant,
            now,
            TOKEN_TTL_SECS,
        ))
    }
}

pub(crate) fn video_caps() -> Caps {
    // Geometry is unknown until the in-band SPS, so advertise a `Range`
    // placeholder (a downstream parser recovers the real dimensions).
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Range { min: 2, max: 8192 },
        height: Dim::Range { min: 2, max: 8192 },
        framerate: Rate::Range {
            min_q16: 1 << 16,
            max_q16: 240 << 16,
        },
    }
}

pub(crate) fn audio_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
    }
}

/// Read the next server `SignalResponse`, skipping non-binary frames. `Ok(None)`
/// on a clean close. (The sink has an identical helper on its own socket half.)
pub(crate) async fn recv_signal(
    ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
) -> Result<Option<SignalResponse>, G2gError> {
    loop {
        match ws.next().await {
            Some(Ok(Message::Binary(bytes))) => return Ok(SignalResponse::decode(&bytes)),
            Some(Ok(Message::Close(_))) | None => return Ok(None),
            Some(Ok(_)) => continue,
            Some(Err(e)) => {
                std::eprintln!("livekit: signal recv failed: {e}");
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }
    }
}

pub(crate) async fn send_signal(
    ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
    req: &SignalRequest,
) -> Result<(), G2gError> {
    ws.send(Message::Binary(req.encode())).await.map_err(|e| {
        std::eprintln!("livekit: signal send failed: {e}");
        G2gError::Hardware(HardwareError::Other)
    })
}

/// Accept one server offer on the subscriber PC and send the answer back.
pub(crate) async fn answer_offer(
    rtc: &mut Rtc,
    ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
    sd: &SessionDescription,
) -> Result<(), G2gError> {
    let hw = || G2gError::Hardware(HardwareError::Other);
    let offer = SdpOffer::from_sdp_string(&sd.sdp).map_err(|e| {
        std::eprintln!("livekit: subscriber offer parse failed: {e:?}");
        hw()
    })?;
    let answer = rtc.sdp_api().accept_offer(offer).map_err(|e| {
        std::eprintln!("livekit: subscriber offer rejected: {e:?}");
        hw()
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
static LIVEKITSRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "url",
        PropKind::Str,
        "LiveKit signalling URL (ws:// or wss://)",
    ),
    PropertySpec::new("room", PropKind::Str, "room name to join"),
    PropertySpec::new("identity", PropKind::Str, "participant identity"),
    PropertySpec::new("api-key", PropKind::Str, "LiveKit API key (mints a token)"),
    PropertySpec::new(
        "api-secret",
        PropKind::Str,
        "LiveKit API secret (mints a token)",
    ),
    PropertySpec::new(
        "token",
        PropKind::Str,
        "pre-minted access token (overrides api-key/secret)",
    ),
];

impl MultiOutputSource for LiveKitSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn output_count(&self) -> usize {
        2
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        match output {
            VIDEO_PORT => Ok(video_caps()),
            AUDIO_PORT => Ok(audio_caps()),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LIVEKITSRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let v = value.as_str().ok_or(PropError::Type)?;
        match name {
            "url" => self.url = v.into(),
            "room" => self.room = v.into(),
            "identity" => self.identity = v.into(),
            "api-key" => self.api_key = v.into(),
            "api-secret" => self.api_secret = v.into(),
            "token" => self.token = (!v.is_empty()).then(|| v.into()),
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "url" => Some(PropValue::Str(self.url.clone())),
            "room" => Some(PropValue::Str(self.room.clone())),
            "identity" => Some(PropValue::Str(self.identity.clone())),
            "api-key" => Some(PropValue::Str(self.api_key.clone())),
            "api-secret" => Some(PropValue::Str(self.api_secret.clone())),
            "token" => Some(PropValue::Str(self.token.clone().unwrap_or_default())),
            _ => None,
        }
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let hw = || G2gError::Hardware(HardwareError::Other);
            let token = self.access_token()?;
            let ws_url = signal_ws_url(&self.url, &token, true);
            let (mut ws, _resp) = connect_async(&ws_url).await.map_err(|e| {
                std::eprintln!("livekit: WebSocket connect to {} failed: {e}", self.url);
                hw()
            })?;

            // The server's first envelope is the JoinResponse.
            let join = loop {
                match recv_signal(&mut ws).await? {
                    Some(SignalResponse::Join(j)) => break j,
                    Some(SignalResponse::Leave) | None => {
                        std::eprintln!("livekit: closed before JoinResponse");
                        return Err(hw());
                    }
                    Some(_) => {}
                }
            };
            let ping_interval = if join.ping_interval > 0 {
                Duration::from_secs(join.ping_interval as u64)
            } else {
                Duration::from_secs(15)
            };

            let socket = UdpSocket::bind((crate::webrtc_util::select_host_ip(), 0))
                .await
                .map_err(io_err)?;
            let local = socket.local_addr().map_err(io_err)?;
            let mut rtc = RtcConfig::new()
                .set_crypto_provider(Arc::new(from_feature_flags()))
                .clear_codecs()
                .enable_h264(true)
                .enable_opus(true)
                .build(Instant::now());
            // Host candidate only (rides inline in each answer SDP); the server
            // trickles its own candidates after the offer.
            add_ice_candidates(&mut rtc, &socket, None).await?;

            out.push_to(VIDEO_PORT, PipelinePacket::CapsChanged(video_caps()))
                .await?;
            out.push_to(AUDIO_PORT, PipelinePacket::CapsChanged(audio_caps()))
                .await?;

            // Track routing state: the first video / audio m-line the server
            // offers claims its port; video is gated until the first keyframe.
            let mut video_mid: Option<Mid> = None;
            let mut audio_mid: Option<Mid> = None;
            let mut video_keyframed = false;
            let mut last_pli = Instant::now();
            let mut next_ping = Instant::now() + ping_interval;

            let mut buf = alloc::vec![0u8; 2000];
            // No TURN on the LiveKit path yet; the empty set feeds direct.
            let mut turn = crate::turn::TurnSet::empty();
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
                let mut frames: Vec<(usize, u64, Vec<u8>)> = Vec::new();
                let deadline = loop {
                    match rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => {
                            let _ = socket.send_to(&t.contents, t.destination).await;
                        }
                        Ok(Output::Event(Event::MediaAdded(m))) => match m.kind {
                            MediaKind::Video if video_mid.is_none() => video_mid = Some(m.mid),
                            MediaKind::Audio if audio_mid.is_none() => audio_mid = Some(m.mid),
                            _ => {}
                        },
                        Ok(Output::Event(Event::MediaData(d))) => {
                            let denom = d.time.denom().max(1) as u128;
                            let pts_ns = (d.time.numer() as u128 * 1_000_000_000 / denom) as u64;
                            if Some(d.mid) == video_mid {
                                // Mid-GOP subscribe: drop until the first IDR.
                                if !video_keyframed {
                                    if !crate::h264util::h264_au_is_keyframe(&d.data) {
                                        continue;
                                    }
                                    video_keyframed = true;
                                }
                                frames.push((VIDEO_PORT, pts_ns, d.data.to_vec()));
                            } else if Some(d.mid) == audio_mid {
                                frames.push((AUDIO_PORT, pts_ns, d.data.to_vec()));
                            }
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => finish!(),
                        Ok(Output::Event(_)) => {}
                        Err(_) => finish!(),
                    }
                };

                // Re-PLI until the video decodes from a keyframe.
                if let Some(mid) = video_mid {
                    if !video_keyframed && last_pli.elapsed() >= PLI_INTERVAL {
                        last_pli = Instant::now();
                        if let Some(rx) = rtc.direct_api().stream_rx_by_mid(mid, None) {
                            rx.request_keyframe(KeyframeRequestKind::Pli);
                        }
                    }
                }

                for (port, pts_ns, data) in frames {
                    let keyframe =
                        port == VIDEO_PORT && crate::h264util::h264_au_is_keyframe(&data);
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
                    if self.frame_limit != 0 && seq >= self.frame_limit {
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
                                std::eprintln!("livekit: signal recv failed: {e}");
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
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(next_ping)) => {
                        let now_ms = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as i64;
                        let _ = send_signal(&mut ws, &SignalRequest::Ping(now_ms)).await;
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
    fn token_requires_credentials() {
        let src = LiveKitSrc::new("ws://h:7880", "room", "id");
        assert!(src.access_token().is_err());
        let src = src.with_token("tok");
        assert_eq!(src.access_token().unwrap(), "tok");
    }
}
