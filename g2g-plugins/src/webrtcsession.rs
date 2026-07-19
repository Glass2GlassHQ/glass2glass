//! Multi-track WebRTC egress session (`WebRtcSessionSink`): publishes a video
//! **and** an audio track over a single WHIP PeerConnection, the `webrtcbin`
//! analog for A/V-on-one-connection. Where [`crate::webrtcsink::WebRtcSink`] is
//! one track per element (one `Rtc` per element), this owns one `Rtc` carrying
//! both an H.264 video m-line and an Opus audio m-line, so the two share one
//! ICE/DTLS/SRTP session, one WHIP handshake, and one bundle.
//!
//! Shape: a [`MultiInputElement`] (input 0 + input 1) driven by the terminal
//! fan-in runner [`run_fanin_session`](g2g_core::runtime::run_fanin_session) (no
//! trailing sink, the session is the destination). Each input is negotiated
//! independently; the session reads the track kind (H.264 video vs Opus audio)
//! from each input's caps, so the pad order does not matter. On the first frame
//! it performs the WHIP handshake offering both send-only m-lines, then spawns a
//! task that owns the `Rtc` + `UdpSocket` and routes each access unit to the
//! matching track writer. STUN / TURN NAT traversal mirror `WebRtcSink`.
//!
//! Status: on-network validated (M248) against a local mediamtx, behind the
//! `webrtc` feature: A/V published over one PeerConnection and read back via
//! `WebRtcWhepSessionSrc` (`webrtc_av_session_loopback`), mediamtx logging the
//! path as `2 tracks (Opus, H264)`. Reverse signals (keyframe-request / BWE) are
//! not yet routed per-input through the multi-track runner, and bidirectional
//! sendrecv on one connection is a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::net::SocketAddr;
use std::time::Instant;

use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use str0m::change::{SdpAnswer, SdpPendingOffer};
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, Mid, Pt};
use str0m::{Event, IceConnectionState, Input, Output, Rtc, RtcConfig};

use str0m::bwe::BweKind;

use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    MemoryDomain, MultiInputElement, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, Rate, ReverseChannel, VideoCodec,
};

use crate::filesink::io_err;
use crate::turn::{self, TurnSet};
use crate::webrtc_util::{
    add_host_candidate, delete_resource, feed_datagram, ice_restart, post_sdp, select_host_ip,
    send_transmit, trickle_candidates, TricklePatch, TurnConfig, ICE_RESTART_TIMEOUT,
};
use crate::webrtcsink::Track;

/// Default bounded depth of the element->session media channel (per direction).
const DEFAULT_QUEUE_DEPTH: usize = 256;

/// Number of tracks this session carries (one video + one audio).
const TRACK_COUNT: usize = 2;

/// One encoded access unit handed to the session task, tagged with its track so
/// the task picks the matching m-line writer.
#[derive(Debug)]
struct MediaUnit {
    track: Track,
    pts_ns: u64,
    data: Vec<u8>,
}

/// Multi-track WHIP-publishing WebRTC egress session. See the module docs.
pub struct WebRtcSessionSink {
    whip_url: String,
    bearer: Option<String>,
    stun_server: Option<String>,
    turn_server: Option<String>,
    turn_user: String,
    turn_pass: String,
    queue_depth: usize,
    /// Track kind per input pad, set in `configure_pipeline(input, caps)`.
    tracks: [Option<Track>; TRACK_COUNT],
    /// Per-input reverse-signal channel: the session task posts a remote PLI /
    /// BWE for a track here, and the fan-in runner surfaces it to that input's
    /// upstream source (so a video PLI forces only the video encoder's IDR).
    reverse: [ReverseChannel; TRACK_COUNT],
    /// Set on the first frame, after the WHIP handshake spawns the session task.
    tx: Option<mpsc::Sender<MediaUnit>>,
    /// WHIP resource URL, so the element DELETEs it synchronously on EOS (RFC
    /// 9725 teardown); the detached task cannot reliably finish a DELETE on clean
    /// end. See `WebRtcSink`.
    resource: Option<String>,
    frames_sent: u64,
}

impl core::fmt::Debug for WebRtcSessionSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebRtcSessionSink")
            .field("whip_url", &self.whip_url)
            .field("tracks", &self.tracks)
            .field("frames_sent", &self.frames_sent)
            .finish()
    }
}

impl WebRtcSessionSink {
    /// Publish A/V to the given WHIP endpoint over one PeerConnection. Two input
    /// pads: connect the H.264 video stream to one and the Opus audio stream to
    /// the other (either order, the track kind is read from the caps).
    pub fn new(whip_url: impl Into<String>) -> Self {
        Self {
            whip_url: whip_url.into(),
            bearer: None,
            stun_server: None,
            turn_server: None,
            turn_user: String::new(),
            turn_pass: String::new(),
            queue_depth: DEFAULT_QUEUE_DEPTH,
            tracks: [None; TRACK_COUNT],
            reverse: Default::default(),
            tx: None,
            resource: None,
            frames_sent: 0,
        }
    }

    /// Attach a bearer token for the WHIP POST.
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.bearer = Some(token.into());
        self
    }

    /// Set a STUN server (`host:port`) for ICE NAT traversal (see `WebRtcSink`).
    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.stun_server = Some(server.into());
        self
    }

    /// Set a TURN relay (`host:port`) + long-term credentials (see `WebRtcSink`).
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

    /// Access units handed to the session so far.
    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    /// True once every input pad has been configured (so both track kinds are
    /// known and the offer can carry both m-lines).
    fn all_configured(&self) -> bool {
        self.tracks.iter().all(|t| t.is_some())
    }

    /// Build the `Rtc` with one m-line per configured track, do the WHIP
    /// offer/answer, and spawn the session task. Runs on the first frame.
    async fn start_session(&mut self) -> Result<(), G2gError> {
        let hw = || G2gError::Hardware(HardwareError::Other);
        let host_ip = select_host_ip();
        let socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
        let local = socket.local_addr().map_err(io_err)?;

        // Enable both codecs so the single Rtc can carry video + audio.
        let mut rtc = RtcConfig::new()
            .set_crypto_provider(Arc::new(from_feature_flags()))
            .clear_codecs()
            .enable_h264(true)
            .enable_opus(true)
            .build(Instant::now());

        // Trickle ICE: the offer carries the host candidate only; reflexive /
        // relay candidates are gathered after the POST and trickled by PATCH.
        add_host_candidate(&mut rtc, &socket)?;

        // One send-only m-line per distinct track kind present on the inputs.
        let has_video = self.tracks.contains(&Some(Track::Video));
        let has_audio = self.tracks.contains(&Some(Track::Audio));
        let (offer_sdp, pending, video_mid, audio_mid): (
            String,
            SdpPendingOffer,
            Option<Mid>,
            Option<Mid>,
        ) = {
            let mut api = rtc.sdp_api();
            let video_mid = has_video.then(|| {
                api.add_media(
                    Track::Video.media_kind(),
                    Direction::SendOnly,
                    None,
                    None,
                    None,
                )
            });
            let audio_mid = has_audio.then(|| {
                api.add_media(
                    Track::Audio.media_kind(),
                    Direction::SendOnly,
                    None,
                    None,
                    None,
                )
            });
            let (offer, pending) = api.apply().ok_or_else(hw)?;
            (offer.to_sdp_string(), pending, video_mid, audio_mid)
        };

        let session = post_sdp(&self.whip_url, self.bearer.as_deref(), offer_sdp.clone()).await?;
        let answer = SdpAnswer::from_sdp_string(&session.answer).map_err(|_| hw())?;
        rtc.sdp_api()
            .accept_answer(pending, answer)
            .map_err(|_| hw())?;

        // Gather reflexive / relay candidates and trickle them to the resource.
        let turn: TurnSet = trickle_candidates(
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

        // The reverse channel of the input pad carrying each track, so the
        // session task can route a per-mid PLI / BWE back to the right source.
        let reverse_for = |kind: Track| {
            self.tracks
                .iter()
                .position(|t| *t == Some(kind))
                .map(|i| self.reverse[i].clone())
        };
        let video_reverse = reverse_for(Track::Video);
        let audio_reverse = reverse_for(Track::Audio);

        self.resource = session.resource.clone();
        let (tx, rx) = mpsc::channel::<MediaUnit>(self.queue_depth);
        tokio::spawn(run_session(
            rtc,
            socket,
            local,
            video_mid,
            audio_mid,
            video_reverse,
            audio_reverse,
            turn,
            session.resource,
            session.etag,
            self.bearer.clone(),
            rx,
        ));
        self.tx = Some(tx);
        Ok(())
    }
}

/// Settable properties, mirroring `WebRtcSink` (so a `gst-launch` line can target
/// a server without the builder).
static WEBRTCSESSION_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "location",
        PropKind::Str,
        "WHIP endpoint URL to publish A/V to",
    ),
    PropertySpec::new(
        "bearer",
        PropKind::Str,
        "optional Authorization: Bearer token",
    ),
    PropertySpec::new(
        "stun-server",
        PropKind::Str,
        "STUN server host:port (empty = host-only)",
    ),
    PropertySpec::new(
        "turn-server",
        PropKind::Str,
        "TURN relay host:port (empty = no relay)",
    ),
    PropertySpec::new(
        "turn-user",
        PropKind::Str,
        "TURN long-term credential username",
    ),
    PropertySpec::new(
        "turn-pass",
        PropKind::Str,
        "TURN long-term credential password",
    ),
];

fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn opus_stereo() -> Caps {
    Caps::Audio {
        format: AudioFormat::Opus,
        channels: 2,
        sample_rate: 48_000,
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

impl MultiInputElement for WebRtcSessionSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_count(&self) -> usize {
        TRACK_COUNT
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match track_of(upstream_caps) {
            Some(_) => Ok(upstream_caps.clone()),
            None => Err(G2gError::CapsMismatch),
        }
    }

    fn caps_constraint_as_input(&self, _input: usize) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::from_alternatives(Vec::from([
            h264_any(),
            opus_stereo(),
        ])))
    }

    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        let track = track_of(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        let slot = self.tracks.get_mut(input).ok_or(G2gError::CapsMismatch)?;
        *slot = Some(track);
        Ok(ConfigureOutcome::Accepted)
    }

    /// Terminal session: there is no merged output (the network is the
    /// destination), so this is never consulted by `run_fanin_session`. A
    /// placeholder keeps the trait total for any muxer-style wiring.
    fn output_caps(&self) -> Result<Caps, G2gError> {
        Ok(h264_any())
    }

    fn reverse_channel(&self, input: usize) -> Option<ReverseChannel> {
        self.reverse.get(input).cloned()
    }

    fn is_terminal(&self) -> bool {
        true
    }

    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let track = self
                        .tracks
                        .get(input)
                        .copied()
                        .flatten()
                        .ok_or(G2gError::NotConfigured)?;
                    let unit = MediaUnit {
                        track,
                        pts_ns: frame.timing.pts_ns,
                        data: slice.as_slice().to_vec(),
                    };
                    // Start the session once every track is known (so the offer
                    // carries both m-lines), on the first frame from any input.
                    if self.tx.is_none() {
                        if !self.all_configured() {
                            return Err(G2gError::NotConfigured);
                        }
                        self.start_session().await?;
                    }
                    if let Some(tx) = &self.tx {
                        tx.send(unit).await.map_err(|_| G2gError::Shutdown)?;
                    }
                    self.frames_sent += 1;
                }
                // Clean end: DELETE the WHIP resource (RFC 9725 teardown) here in
                // the element so it completes before the runtime tears the session
                // task down (as for `WebRtcSink`).
                PipelinePacket::Eos => {
                    if let Some(res) = self.resource.take() {
                        delete_resource(&res, self.bearer.as_deref()).await;
                    }
                }
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Flush => {}
                PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WEBRTCSESSION_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" | "whip-url" => {
                self.whip_url = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            "bearer" => {
                let t = value.as_str().ok_or(PropError::Type)?;
                self.bearer = if t.is_empty() { None } else { Some(t.into()) };
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
}

/// The sans-IO driving loop for the multi-track session: owns the `Rtc` + socket,
/// drains `poll_output` (routing relay datagrams through TURN), and routes each
/// incoming `MediaUnit` to its track's writer. Mirrors `WebRtcSink::run_session`
/// generalised to two writers.
#[allow(clippy::too_many_arguments)]
async fn run_session(
    mut rtc: Rtc,
    socket: UdpSocket,
    local: SocketAddr,
    video_mid: Option<Mid>,
    audio_mid: Option<Mid>,
    video_reverse: Option<ReverseChannel>,
    audio_reverse: Option<ReverseChannel>,
    mut turn: TurnSet,
    resource: Option<String>,
    etag: Option<String>,
    bearer: Option<String>,
    mut rx: mpsc::Receiver<MediaUnit>,
) {
    let mut buf = alloc::vec![0u8; 2000];
    // Negotiated payload type per track, discovered once each writer exists.
    let mut video_pt: Option<Pt> = None;
    let mut audio_pt: Option<Pt> = None;
    let mut refresh_at = Instant::now() + turn::REFRESH_INTERVAL;
    let mut disconnected_since: Option<Instant> = None;

    loop {
        let deadline = loop {
            match rtc.poll_output() {
                Ok(Output::Timeout(t)) => break t,
                Ok(Output::Transmit(t)) => send_transmit(&socket, &mut turn, &t).await,
                Ok(Output::Event(Event::IceConnectionStateChange(state))) => match state {
                    IceConnectionState::Disconnected => {
                        disconnected_since.get_or_insert_with(Instant::now);
                    }
                    IceConnectionState::Connected | IceConnectionState::Completed => {
                        disconnected_since = None;
                    }
                    _ => {}
                },
                // Remote PLI: route the keyframe request to the source feeding
                // the track whose m-line it names (by mid), so only that encoder
                // emits an IDR.
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
                // Congestion-control estimate (whole-connection): relay it to the
                // video track, the bitrate-adaptive one (Opus bitrate adaptation
                // is a separate follow-up).
                Ok(Output::Event(Event::EgressBitrateEstimate(kind))) => {
                    let bps = match kind {
                        BweKind::Twcc(b) | BweKind::Remb(_, b) => Some(b.as_u64()),
                        _ => None,
                    };
                    if let (Some(bps), Some(rc)) = (bps, video_reverse.as_ref()) {
                        rc.set_bitrate(bps.min(u32::MAX as u64) as u32);
                    }
                }
                Ok(Output::Event(_)) => {}
                Err(_) => {
                    teardown(resource.as_deref(), bearer.as_deref()).await;
                    return;
                }
            }
        };

        // Sustained ICE disconnect: attempt an ICE restart against the resource.
        if disconnected_since.is_some_and(|t| t.elapsed() >= ICE_RESTART_TIMEOUT) {
            disconnected_since = None;
            match resource.as_deref() {
                Some(res) => {
                    if !matches!(
                        ice_restart(&mut rtc, res, bearer.as_deref(), etag.as_deref()).await,
                        TricklePatch::Accepted
                    ) {
                        teardown(resource.as_deref(), bearer.as_deref()).await;
                        return;
                    }
                }
                None => return,
            }
        }

        let timeout = deadline.saturating_duration_since(Instant::now());
        tokio::select! {
            r = socket.recv_from(&mut buf) => {
                let Ok((n, source)) = r else {
                    teardown(resource.as_deref(), bearer.as_deref()).await;
                    return;
                };
                if !feed_datagram(&mut rtc, &mut turn, local, &buf[..n], source) {
                    teardown(resource.as_deref(), bearer.as_deref()).await;
                    return;
                }
            }
            // A closed channel = element drop. Clean-EOS teardown is done by the
            // element (`process`); just exit here.
            unit = rx.recv() => {
                let Some(unit) = unit else { return };
                // Pick this unit's track writer (by mid), discovering the codec's
                // negotiated payload type on first use.
                let (mid, pt_slot) = match unit.track {
                    Track::Video => (video_mid, &mut video_pt),
                    Track::Audio => (audio_mid, &mut audio_pt),
                };
                let Some(mid) = mid else { continue };
                if pt_slot.is_none() {
                    if let Some(writer) = rtc.writer(mid) {
                        *pt_slot = writer
                            .payload_params()
                            .find(|p| p.spec().codec == unit.track.codec())
                            .map(|p| p.pt());
                    }
                }
                if let Some(p) = *pt_slot {
                    let rtp_time = unit.track.media_time(unit.pts_ns);
                    if let Some(writer) = rtc.writer(mid) {
                        let _ = writer.write(p, Instant::now(), rtp_time, unit.data);
                    }
                }
            }
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(refresh_at)), if !turn.is_empty() => {
                turn.refresh_all(&socket).await;
                refresh_at = Instant::now() + turn::REFRESH_INTERVAL;
            }
            _ = tokio::time::sleep(timeout) => {
                if rtc.handle_input(Input::Timeout(Instant::now())).is_err() {
                    teardown(resource.as_deref(), bearer.as_deref()).await;
                    return;
                }
            }
        }
    }
}

/// Best-effort WHIP resource teardown (RFC 9725 `DELETE`) before the session
/// task exits.
async fn teardown(resource: Option<&str>, bearer: Option<&str>) {
    if let Some(res) = resource {
        delete_resource(res, bearer).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_inputs_take_video_and_audio_by_caps() {
        let mut s = WebRtcSessionSink::new("http://h/whip");
        assert_eq!(s.input_count(), 2);
        // Either pad order works; the track is read from the caps.
        assert!(s.configure_pipeline(0, &h264_any()).is_ok());
        assert!(s.configure_pipeline(1, &opus_stereo()).is_ok());
        assert_eq!(s.tracks, [Some(Track::Video), Some(Track::Audio)]);
        assert!(s.all_configured());
    }

    #[test]
    fn per_input_reverse_channel_is_shared_with_the_runner() {
        // The fan-in runner obtains each input's reverse channel via
        // reverse_channel(i); a signal the session posts on its own handle must
        // be visible on that same (Arc-shared) channel, and each input's channel
        // is independent so a PLI for one track never fires the other's.
        let s = WebRtcSessionSink::new("http://h/whip");
        let rc0 = s.reverse_channel(0).expect("input 0 has a reverse channel");
        let rc1 = s.reverse_channel(1).expect("input 1 has a reverse channel");
        assert!(
            s.reverse_channel(2).is_none(),
            "no channel past the track count"
        );

        // The session posts a keyframe request for input 0; the runner-side
        // handle for input 0 sees it (once), input 1's does not.
        s.reverse[0].request_keyframe();
        assert!(
            matches!(rc0.take(), Some(g2g_core::PushOutcome::Reconfigure(_))),
            "input 0's channel surfaces the keyframe request"
        );
        assert!(rc0.take().is_none(), "the request is consumed once");
        assert!(rc1.take().is_none(), "input 1's channel is untouched");

        // A BWE estimate posted for input 0 surfaces as a bitrate outcome.
        s.reverse[0].set_bitrate(1_200_000);
        assert!(matches!(
            rc0.take(),
            Some(g2g_core::PushOutcome::Bitrate(1_200_000))
        ));
    }

    #[test]
    fn rejects_non_av_caps() {
        let s = WebRtcSessionSink::new("http://h/whip");
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::I420,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
        };
        assert_eq!(s.intercept_caps(0, &raw), Err(G2gError::CapsMismatch));
        assert!(s.intercept_caps(0, &h264_any()).is_ok());
        assert!(s.intercept_caps(1, &opus_stereo()).is_ok());
    }

    #[test]
    fn properties_round_trip() {
        let mut s = WebRtcSessionSink::new("http://h/whip")
            .with_bearer("tok")
            .with_turn_server("relay:3478", "u", "p");
        assert_eq!(s.bearer.as_deref(), Some("tok"));
        assert_eq!(s.turn_server.as_deref(), Some("relay:3478"));
        s.set_property("location", PropValue::Str("http://srv/whip".into()))
            .unwrap();
        assert_eq!(
            s.get_property("location"),
            Some(PropValue::Str("http://srv/whip".into()))
        );
        s.set_property("stun-server", PropValue::Str("stun:3478".into()))
            .unwrap();
        assert_eq!(s.stun_server.as_deref(), Some("stun:3478"));
        assert_eq!(
            s.set_property("nope", PropValue::Str("x".into())),
            Err(PropError::Unknown)
        );
    }
}
