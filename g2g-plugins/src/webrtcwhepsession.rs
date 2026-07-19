//! Multi-track WebRTC ingest session (`WebRtcWhepSessionSrc`): subscribes to a
//! WHEP endpoint and emits the received H.264 video **and** Opus audio on two
//! output pads, from one PeerConnection. The receive-side mirror of
//! [`crate::webrtcsession::WebRtcSessionSink`], and the multi-track counterpart
//! of the one-track [`crate::webrtcwhepsrc::WebRtcWhepSrc`].
//!
//! Shape: a [`MultiOutputSource`] (output 0 = H.264 video, output 1 = Opus
//! audio) driven by the terminal fan-out runner
//! [`run_fanout_session`](g2g_core::runtime::run_fanout_session) (no upstream,
//! the session generates both streams from the network). It offers two recv-only
//! m-lines in the WHEP handshake, drives str0m's loop on a tokio `UdpSocket`, and
//! routes each `Event::MediaData` to the matching output by its `Mid` (video AUs
//! are Annex-B from str0m's depayloader; audio is forwarded as Opus packets).
//! STUN / TURN NAT traversal mirror [`crate::webrtcwhepsrc::WebRtcWhepSrc`].
//!
//! Status: on-network validated (M248) against a local mediamtx, behind the
//! `webrtc` feature: subscribes to a WHEP session publishing both tracks
//! (`webrtc_av_session_loopback`) and emits video + audio on its two outputs,
//! mediamtx logging the read as `2 tracks (H264, Opus)`.

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

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, Caps, Dim, FrameTiming, G2gError, HardwareError, MemoryDomain, MultiOutputSink,
    MultiOutputSource, PipelinePacket, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::turn::{self, TurnClient};
use crate::webrtc_util::{
    add_ice_candidates, feed_datagram, post_sdp, select_host_ip, send_transmit,
};

/// Output port for the H.264 video track.
const VIDEO_PORT: usize = 0;
/// Output port for the Opus audio track.
const AUDIO_PORT: usize = 1;

/// Multi-track WHEP-subscribing WebRTC ingest session. See the module docs.
pub struct WebRtcWhepSessionSrc {
    whep_url: String,
    bearer: Option<String>,
    stun_server: Option<String>,
    turn_server: Option<String>,
    turn_user: String,
    turn_pass: String,
    /// Stop after this many access units across both tracks and emit EOS
    /// (0 = unbounded). For tests / bounded runs.
    frame_limit: u64,
}

impl core::fmt::Debug for WebRtcWhepSessionSrc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebRtcWhepSessionSrc")
            .field("whep_url", &self.whep_url)
            .finish()
    }
}

impl WebRtcWhepSessionSrc {
    /// Subscribe to the given WHEP endpoint, emitting video on output 0 and audio
    /// on output 1.
    pub fn new(whep_url: impl Into<String>) -> Self {
        Self {
            whep_url: whep_url.into(),
            bearer: None,
            stun_server: None,
            turn_server: None,
            turn_user: String::new(),
            turn_pass: String::new(),
            frame_limit: 0,
        }
    }

    /// Attach a bearer token for the WHEP POST.
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Set a STUN server (`host:port`) for ICE NAT traversal.
    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.stun_server = Some(server.into());
        self
    }

    /// Set a TURN relay (`host:port`) + long-term credentials.
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

    /// Stop after `n` access units across both tracks (then EOS on both).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }
}

fn video_caps() -> Caps {
    // Geometry is unknown until the in-band SPS, so advertise a `Range`
    // placeholder (a downstream parser recovers the real dimensions): negotiation
    // fixates before data flows and `fixate()` rejects `Dim::Any`.
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

fn audio_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
    }
}

impl MultiOutputSource for WebRtcWhepSessionSrc {
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

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let hw = || G2gError::Hardware(HardwareError::Other);
            let host_ip = select_host_ip();
            let socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
            let local = socket.local_addr().map_err(io_err)?;

            let mut rtc = RtcConfig::new()
                .set_crypto_provider(alloc::sync::Arc::new(from_feature_flags()))
                .clear_codecs()
                .enable_h264(true)
                .enable_opus(true)
                .build(Instant::now());
            add_ice_candidates(&mut rtc, &socket, self.stun_server.as_deref()).await?;
            let mut turn: Option<TurnClient> = match &self.turn_server {
                Some(server) => {
                    turn::setup(&mut rtc, &socket, server, &self.turn_user, &self.turn_pass).await
                }
                None => None,
            };
            let mut refresh_at = Instant::now() + turn::REFRESH_INTERVAL;

            // WHEP: offer recv-only video + audio m-lines, POST, apply answer.
            let (offer_sdp, pending, video_mid, audio_mid) = {
                let mut api = rtc.sdp_api();
                let video_mid = api.add_media(
                    str0m::media::MediaKind::Video,
                    Direction::RecvOnly,
                    None,
                    None,
                    None,
                );
                let audio_mid = api.add_media(
                    str0m::media::MediaKind::Audio,
                    Direction::RecvOnly,
                    None,
                    None,
                    None,
                );
                let (offer, pending) = api.apply().ok_or_else(hw)?;
                (offer.to_sdp_string(), pending, video_mid, audio_mid)
            };
            let answer_sdp = post_sdp(&self.whep_url, self.bearer.as_deref(), offer_sdp).await?;
            let answer = SdpAnswer::from_sdp_string(&answer_sdp).map_err(|_| hw())?;
            rtc.sdp_api()
                .accept_answer(pending, answer)
                .map_err(|_| hw())?;

            // Announce each output's caps before its first frame.
            out.push_to(VIDEO_PORT, PipelinePacket::CapsChanged(video_caps()))
                .await?;
            out.push_to(AUDIO_PORT, PipelinePacket::CapsChanged(audio_caps()))
                .await?;

            let mut buf = alloc::vec![0u8; 2000];
            let mut seq = 0u64;
            // Emit EOS to both outputs and finish.
            macro_rules! finish {
                () => {{
                    out.push_to(VIDEO_PORT, PipelinePacket::Eos).await?;
                    out.push_to(AUDIO_PORT, PipelinePacket::Eos).await?;
                    return Ok(seq);
                }};
            }

            loop {
                // (port, pts_ns, data) collected while draining poll_output.
                let mut frames: Vec<(usize, u64, Vec<u8>)> = Vec::new();
                let deadline = loop {
                    match rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => send_transmit(&socket, &mut turn, &t).await,
                        Ok(Output::Event(Event::MediaData(d))) => {
                            let denom = d.time.denom().max(1) as u128;
                            let pts_ns = (d.time.numer() as u128 * 1_000_000_000 / denom) as u64;
                            let port = if d.mid == video_mid {
                                VIDEO_PORT
                            } else if d.mid == audio_mid {
                                AUDIO_PORT
                            } else {
                                continue;
                            };
                            frames.push((port, pts_ns, d.data.to_vec()));
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => finish!(),
                        Ok(Output::Event(_)) => {}
                        Err(_) => finish!(),
                    }
                };

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
                    r = socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else { finish!() };
                        let _ = feed_datagram(&mut rtc, &mut turn, local, &buf[..n], source);
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_outputs_video_then_audio() {
        let src = WebRtcWhepSessionSrc::new("http://h/whep");
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
    fn builders_set_fields() {
        let src = WebRtcWhepSessionSrc::new("http://h/whep")
            .with_bearer("tok")
            .with_turn_server("relay:3478", "u", "p")
            .with_frame_limit(20);
        assert_eq!(src.bearer.as_deref(), Some("tok"));
        assert_eq!(src.turn_server.as_deref(), Some("relay:3478"));
        assert_eq!(src.frame_limit, 20);
    }
}
