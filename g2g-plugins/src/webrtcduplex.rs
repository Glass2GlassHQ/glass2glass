//! Bidirectional (sendrecv) WebRTC session (`WebRtcDuplexSession`): one
//! PeerConnection that **both** publishes local tracks and receives the peer's
//! tracks, the `webrtcbin` sendrecv shape. Where [`crate::webrtcsession`] is
//! send-only (WHIP, N inputs) and [`crate::webrtcwhepsession`] is receive-only
//! (WHEP, N outputs), this is the union: N send inputs **and** N recv outputs on
//! one [`MultiDuplexSession`], driven by the terminal duplex runner
//! [`run_duplex_session`](g2g_core::runtime::run_duplex_session).
//!
//! WHIP / WHEP are unidirectional by spec, so sendrecv cannot use them. Instead
//! each m-line is offered `Direction::SendRecv` and the two peers exchange SDP
//! directly (no media server): one is the [`SignalRole::Offerer`], the other the
//! [`SignalRole::Answerer`], swapping offer/answer over an [`SdpChannel`]
//! (in-process for a P2P loopback; a real signaller, e.g. LiveKit, plugs in the
//! same place). ICE host candidates ride in the SDP, so two peers on one host
//! connect over localhost UDP with no STUN. The track kind per pad is read from
//! the negotiated `Event::MediaAdded` (so offerer and answerer discover the same
//! `Mid`s the same way), and each m-line carries one send direction (written from
//! the matching input pad) and one recv direction (emitted on the matching output
//! pad).
//!
//! Unlike the send-only session (which spawns a detached task to own the `Rtc`
//! and dodge `process` / run-loop aliasing), the duplex runner gives this element
//! a single `run` that owns the connection outright: it selects over the inbound
//! send packets and the network, so the send and recv halves share state with no
//! task hop. Status: on-network validated (M249) by in-process P2P loopbacks
//! (`webrtc_duplex_p2p_loopback` video, `webrtc_duplex_p2p_av_loopback` A/V),
//! behind the `webrtc` feature. STUN / TURN NAT traversal and a pluggable
//! real-SFU signaller are follow-ups.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::time::Instant;

use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use str0m::bwe::{Bitrate, BweKind};
use str0m::change::{SdpAnswer, SdpOffer};
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, MediaKind, Mid, Pt};
use str0m::net::{Protocol, Receive};
use str0m::{Event, IceConnectionState, Input, Output, RtcConfig};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, ConfigureOutcome, Dim, DuplexInbound, G2gError,
    HardwareError, MemoryDomain, MultiDuplexSession, MultiOutputSink, PipelinePacket, Rate,
    ReverseChannel, VideoCodec,
};

use crate::filesink::io_err;
use crate::h264util::h264_au_is_keyframe;
use crate::webrtc_util::{add_ice_candidates, select_host_ip};
use crate::webrtcsink::Track;

/// The two tracks a duplex session can carry, in pad order: video on pad 0, audio
/// on pad 1. `track_count` selects how many are active (1 = video only).
const KINDS: [Track; 2] = [Track::Video, Track::Audio];

/// Which peer originates the SDP offer in the sendrecv handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalRole {
    /// Generates the SDP offer, sends it, awaits the answer.
    Offerer,
    /// Awaits the SDP offer, accepts it, sends back the answer.
    Answerer,
}

/// In-process SDP signaling transport for a P2P sendrecv handshake. The offerer
/// sends its offer on `tx` and reads the answer from `rx`; the answerer reads the
/// offer from `rx` and sends the answer on `tx`. [`SdpChannel::pair`] wires two
/// of these crossed so the two sessions exchange SDP with no media server.
#[derive(Debug)]
pub struct SdpChannel {
    tx: mpsc::Sender<String>,
    rx: mpsc::Receiver<String>,
}

impl SdpChannel {
    /// Build a crossed pair: the offerer's channel and the answerer's channel,
    /// such that each one's `tx` feeds the other's `rx`.
    pub fn pair() -> (SdpChannel, SdpChannel) {
        let (a_tx, a_rx) = mpsc::channel(4);
        let (b_tx, b_rx) = mpsc::channel(4);
        // Offerer sends on a_tx (-> b_rx), reads on b_rx... wire crossed:
        // offerer.tx -> answerer.rx, answerer.tx -> offerer.rx.
        (
            SdpChannel { tx: a_tx, rx: b_rx },
            SdpChannel { tx: b_tx, rx: a_rx },
        )
    }
}

/// Bidirectional sendrecv WebRTC session. See the module docs.
pub struct WebRtcDuplexSession {
    role: SignalRole,
    sig: Option<SdpChannel>,
    /// Number of sendrecv m-lines: 1 (video) or 2 (video + audio). Equal to both
    /// `input_count` and `output_count` (input i and output i share m-line i).
    track_count: usize,
    stun_server: Option<String>,
    /// How long to keep draining the peer after the local send side ends (its
    /// sources reached EOS), so in-flight received frames are not cut off.
    linger: Duration,
    /// Track kind per send input pad, set in `configure_input`.
    inputs: Vec<Option<Track>>,
    /// Per send-input reverse channel: a remote PLI / BWE that names a track's
    /// m-line is routed back to the source feeding that pad. Shared (Arc-backed)
    /// with the runner, which polls each after every push from its source.
    reverse: Vec<ReverseChannel>,
}

impl core::fmt::Debug for WebRtcDuplexSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebRtcDuplexSession")
            .field("role", &self.role)
            .field("track_count", &self.track_count)
            .field("inputs", &self.inputs)
            .finish()
    }
}

impl WebRtcDuplexSession {
    /// A sendrecv session carrying `track_count` tracks (1 = video; 2 = video +
    /// audio), with the given `role` and SDP signaling channel.
    pub fn new(role: SignalRole, sig: SdpChannel, track_count: usize) -> Self {
        assert!(
            track_count >= 1 && track_count <= KINDS.len(),
            "track_count must be 1 or 2"
        );
        Self {
            role,
            sig: Some(sig),
            track_count,
            stun_server: None,
            linger: Duration::from_millis(1500),
            inputs: alloc::vec![None; track_count],
            reverse: (0..track_count).map(|_| ReverseChannel::new()).collect(),
        }
    }

    /// Set a STUN server (`host:port`) for ICE NAT traversal (host-only by
    /// default, which is all a same-host P2P loopback needs).
    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.stun_server = Some(server.into());
        self
    }

    /// Override the post-send linger window (default 1.5 s).
    pub fn with_linger(mut self, linger: Duration) -> Self {
        self.linger = linger;
        self
    }
}

fn video_caps() -> Caps {
    // Geometry unknown until the in-band SPS, so a `Range` placeholder that
    // fixates (a downstream parser recovers the real dimensions), as in the WHEP
    // session source.
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

/// The output caps for a given track kind.
fn caps_for(kind: Track) -> Caps {
    match kind {
        Track::Video => video_caps(),
        Track::Audio => audio_caps(),
    }
}

/// The track kind an input's caps select (H.264 video or Opus audio).
fn track_of(caps: &Caps) -> Option<Track> {
    match caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            ..
        } => Some(Track::Video),
        Caps::Audio {
            format: AudioFormat::Opus,
            ..
        } => Some(Track::Audio),
        _ => None,
    }
}

impl MultiDuplexSession for WebRtcDuplexSession {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.track_count
    }

    fn output_count(&self) -> usize {
        self.track_count
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match track_of(upstream_caps) {
            Some(_) => Ok(upstream_caps.clone()),
            None => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_input(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let track = track_of(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        *self.inputs.get_mut(input).ok_or(G2gError::CapsMismatch)? = Some(track);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        let kind = KINDS.get(output).copied().ok_or(G2gError::CapsMismatch)?;
        Ok(caps_for(kind))
    }

    fn reverse_channel(&self, input: usize) -> Option<ReverseChannel> {
        self.reverse.get(input).cloned()
    }

    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a> {
        let role = self.role;
        let track_count = self.track_count;
        let inputs = self.inputs.clone();
        let stun = self.stun_server.clone();
        let linger = self.linger;
        let sig = self.sig.take();
        // The reverse channel of the input pad carrying each track kind, so a
        // remote PLI / BWE naming that track's m-line routes back to the source
        // feeding it (the send pads may be wired in either order).
        let reverse_for = |kind: Track| {
            self.inputs
                .iter()
                .position(|t| *t == Some(kind))
                .and_then(|i| self.reverse.get(i).cloned())
        };
        let video_reverse = reverse_for(Track::Video);
        let audio_reverse = reverse_for(Track::Audio);
        Box::pin(async move {
            let hw = || G2gError::Hardware(HardwareError::Other);
            let mut sig = sig.ok_or_else(hw)?;

            let host_ip = select_host_ip();
            let socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
            let local = socket.local_addr().map_err(io_err)?;

            let mut rtc = RtcConfig::new()
                .set_crypto_provider(Arc::new(from_feature_flags()))
                .clear_codecs()
                .enable_h264(true)
                .enable_opus(true)
                // Congestion control so the peer's BWE estimate arrives as
                // `Event::EgressBitrateEstimate`, routed to the video track below.
                .enable_bwe(Some(Bitrate::bps(2_000_000)))
                .build(Instant::now());
            // Host (and optional reflexive) candidates ride in the SDP, so they
            // must be added before the offer/answer is generated below.
            add_ice_candidates(&mut rtc, &socket, stun.as_deref()).await?;

            // Per-kind `Mid` and negotiated payload type, for both directions.
            // The offerer learns its `Mid`s from `add_media` (str0m does not emit
            // `MediaAdded` for media the local side added); the answerer learns
            // them from `MediaAdded` when it accepts the offer.
            let mut video_mid: Option<Mid> = None;
            let mut audio_mid: Option<Mid> = None;
            let mut video_pt: Option<Pt> = None;
            let mut audio_pt: Option<Pt> = None;

            // SDP handshake: each m-line is sendrecv. The offerer adds the media
            // and creates the offer; the answerer accepts the offer (whose m-lines
            // it inherits).
            match role {
                SignalRole::Offerer => {
                    let (offer_sdp, pending) = {
                        let mut api = rtc.sdp_api();
                        for kind in KINDS.iter().take(track_count) {
                            let mid = api.add_media(
                                kind.media_kind(),
                                Direction::SendRecv,
                                None,
                                None,
                                None,
                            );
                            match kind {
                                Track::Video => video_mid = Some(mid),
                                Track::Audio => audio_mid = Some(mid),
                            }
                        }
                        let (offer, pending) = api.apply().ok_or_else(hw)?;
                        (offer.to_sdp_string(), pending)
                    };
                    sig.tx.send(offer_sdp).await.map_err(|_| hw())?;
                    let answer_sdp = sig.rx.recv().await.ok_or_else(hw)?;
                    let answer = SdpAnswer::from_sdp_string(&answer_sdp).map_err(|_| hw())?;
                    rtc.sdp_api()
                        .accept_answer(pending, answer)
                        .map_err(|_| hw())?;
                }
                SignalRole::Answerer => {
                    let offer_sdp = sig.rx.recv().await.ok_or_else(hw)?;
                    let offer = SdpOffer::from_sdp_string(&offer_sdp).map_err(|_| hw())?;
                    let answer = rtc.sdp_api().accept_offer(offer).map_err(|_| hw())?;
                    sig.tx
                        .send(answer.to_sdp_string())
                        .await
                        .map_err(|_| hw())?;
                }
            }

            // Announce each output's caps before its first frame.
            for (o, kind) in KINDS.iter().enumerate().take(track_count) {
                out.push_to(o, PipelinePacket::CapsChanged(caps_for(*kind)))
                    .await?;
            }

            let mut buf = alloc::vec![0u8; 2000];
            let mut seq = 0u64;
            let mut received = 0u64;
            let mut send_done = false;
            // Set when the local send side ends; the loop finishes after it.
            let mut drain_deadline: Option<Instant> = None;

            macro_rules! finish {
                () => {{
                    for o in 0..track_count {
                        out.push_to(o, PipelinePacket::Eos).await?;
                    }
                    return Ok(received);
                }};
            }

            loop {
                // (output port, pts_ns, data) collected while draining poll_output.
                let mut frames: Vec<(usize, u64, Vec<u8>)> = Vec::new();
                let deadline = loop {
                    match rtc.poll_output() {
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => {
                            let _ = socket.send_to(&t.contents, t.destination).await;
                        }
                        // The answerer learns its m-line `Mid`s here (the offerer
                        // captured them from `add_media`); harmless to set again.
                        Ok(Output::Event(Event::MediaAdded(m))) => match m.kind {
                            MediaKind::Video => video_mid = Some(m.mid),
                            MediaKind::Audio => audio_mid = Some(m.mid),
                        },
                        Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => finish!(),
                        // Remote PLI: route the keyframe request to the send source
                        // feeding the track whose m-line it names (by mid), so only
                        // that encoder emits an IDR.
                        Ok(Output::Event(Event::KeyframeRequest(req))) => {
                            let rc = if Some(req.mid) == video_mid {
                                video_reverse.as_ref()
                            } else if Some(req.mid) == audio_mid {
                                audio_reverse.as_ref()
                            } else {
                                None
                            };
                            if let Some(rc) = rc {
                                rc.request_keyframe();
                            }
                        }
                        // Congestion-control estimate (whole-connection): relay it to
                        // the video track, the bitrate-adaptive one (Opus bitrate
                        // adaptation is a separate follow-up), as the fan-in session does.
                        Ok(Output::Event(Event::EgressBitrateEstimate(kind))) => {
                            let bps = match kind {
                                BweKind::Twcc(b) | BweKind::Remb(_, b) => Some(b.as_u64()),
                                _ => None,
                            };
                            if let (Some(bps), Some(rc)) = (bps, video_reverse.as_ref()) {
                                rc.set_bitrate(bps.min(u32::MAX as u64) as u32);
                            }
                        }
                        Ok(Output::Event(Event::MediaData(d))) => {
                            let denom = d.time.denom().max(1) as u128;
                            let pts_ns = (d.time.numer() as u128 * 1_000_000_000 / denom) as u64;
                            let port = if Some(d.mid) == video_mid {
                                0
                            } else if Some(d.mid) == audio_mid {
                                1
                            } else {
                                continue;
                            };
                            if port < track_count {
                                frames.push((port, pts_ns, d.data.to_vec()));
                            }
                        }
                        Ok(Output::Event(_)) => {}
                        Err(_) => finish!(),
                    }
                };

                for (port, pts_ns, data) in frames {
                    let keyframe = port == 0 && h264_au_is_keyframe(&data);
                    let frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(
                            data.into_boxed_slice(),
                        )),
                        timing: g2g_core::FrameTiming {
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
                    received += 1;
                }

                if let Some(dl) = drain_deadline {
                    if Instant::now() >= dl {
                        finish!();
                    }
                }

                let timeout = deadline.saturating_duration_since(Instant::now());
                tokio::select! {
                    r = socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else { finish!() };
                        if let Ok(contents) = (&buf[..n]).try_into() {
                            let input = Input::Receive(Instant::now(),
                                Receive { proto: Protocol::Udp, source, destination: local, contents });
                            if rtc.handle_input(input).is_err() { finish!(); }
                        }
                    }
                    inb = inbound.recv(), if !send_done => {
                        match inb {
                            None => {
                                // All send sources ended: drain the peer for `linger`,
                                // then finish (flushes both directions).
                                send_done = true;
                                drain_deadline = Some(Instant::now() + linger);
                            }
                            Some((idx, PipelinePacket::DataFrame(frame))) => {
                                // Route by the track configured for this send pad,
                                // not the fixed KINDS position (a pipeline may wire
                                // audio to pad 0 and video to pad 1).
                                let kind = inputs
                                    .get(idx)
                                    .copied()
                                    .flatten()
                                    .unwrap_or(KINDS[idx.min(track_count - 1)]);
                                let (mid, pt_slot) = match kind {
                                    Track::Video => (video_mid, &mut video_pt),
                                    Track::Audio => (audio_mid, &mut audio_pt),
                                };
                                // Drop send frames until the m-line is negotiated
                                // (its `Mid` arrives via `MediaAdded`).
                                if let (Some(mid), MemoryDomain::System(slice)) =
                                    (mid, &frame.domain)
                                {
                                    if pt_slot.is_none() {
                                        if let Some(w) = rtc.writer(mid) {
                                            *pt_slot = w
                                                .payload_params()
                                                .find(|p| p.spec().codec == kind.codec())
                                                .map(|p| p.pt());
                                        }
                                    }
                                    if let Some(p) = *pt_slot {
                                        let rtp_time = kind.media_time(frame.timing.pts_ns);
                                        if let Some(w) = rtc.writer(mid) {
                                            let _ = w.write(p, Instant::now(), rtp_time,
                                                slice.as_slice().to_vec());
                                        }
                                    }
                                }
                            }
                            // Per-input EOS / control: drained, not forwarded (the
                            // session owns its own per-output EOS).
                            Some(_) => {}
                        }
                    }
                    _ = tokio::time::sleep(timeout) => {
                        if rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                            finish!();
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    #[test]
    fn pad_counts_match_track_count() {
        let (a, _b) = SdpChannel::pair();
        let s = WebRtcDuplexSession::new(SignalRole::Offerer, a, 2);
        assert_eq!(s.input_count(), 2);
        assert_eq!(s.output_count(), 2);
        assert!(matches!(
            s.output_caps(0),
            Ok(Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            })
        ));
        assert!(matches!(
            s.output_caps(1),
            Ok(Caps::Audio {
                format: AudioFormat::Opus,
                ..
            })
        ));
        assert!(s.output_caps(2).is_err());
    }

    #[test]
    fn configure_input_reads_track_kind_from_caps() {
        let (a, _b) = SdpChannel::pair();
        let mut s = WebRtcDuplexSession::new(SignalRole::Answerer, a, 2);
        assert!(s.configure_input(0, &h264_caps()).is_ok());
        assert!(s.configure_input(1, &audio_caps()).is_ok());
        assert_eq!(
            s.inputs,
            alloc::vec![Some(Track::Video), Some(Track::Audio)]
        );
        // Non-A/V caps rejected.
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::I420,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert_eq!(s.intercept_caps(0, &raw), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn sdp_channel_pair_is_crossed() {
        // Offerer's tx must reach the answerer's rx and vice versa.
        let (mut off, mut ans) = SdpChannel::pair();
        off.tx.try_send("offer".into()).unwrap();
        ans.tx.try_send("answer".into()).unwrap();
        assert_eq!(ans.rx.try_recv().unwrap(), "offer");
        assert_eq!(off.rx.try_recv().unwrap(), "answer");
    }
}
