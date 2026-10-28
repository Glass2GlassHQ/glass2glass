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
//! task hop. Status: on-network validated by in-process P2P loopbacks
//! (`webrtc_duplex_p2p_loopback` video, `webrtc_duplex_p2p_av_loopback` A/V),
//! behind the `webrtc` feature. Mid-session tracks ride on spare pads
//! declared up front by [`WebRtcDuplexSession::with_spare_tracks`]: a spare
//! carries no m-line at the handshake and binds later, when its send pad gets a
//! frame (which re-offers a new m-line) or when the peer adds one.
//! [`DuplexControl::remove_track`] is the inverse: it stops the m-line,
//! freeing both of its pads on both peers. A freed pad is claimable again by the
//! same two paths, and its next track always negotiates a NEW m-line, since a
//! stopped one cannot be reactivated. Under the dynamic runner
//! (`run_duplex_session_dynamic`) no reserve is needed at all: a send
//! track added through the runner's handle announces itself on a fresh input
//! index and the session grows to take it, and a peer track with no free pad
//! grows a recv port through `MultiOutputSink::add_port` instead of being
//! skipped. STUN / TURN NAT traversal and a pluggable real-SFU signaller are
//! follow-ups.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use core::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use std::time::Instant;

use tokio::net::UdpSocket;
use tokio::sync::mpsc;

use str0m::bwe::{Bitrate, BweKind};
use str0m::change::SdpAnswer;
use str0m::crypto::from_feature_flags;
use str0m::media::{Direction, MediaKind, Mid, Pt};
use str0m::{Event, IceConnectionState, Input, Output, RtcConfig};

use g2g_core::frame::Frame;
use g2g_core::g2g_warn;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    Caps, CapsConstraint, ConfigureOutcome, DuplexInbound, G2gError, HardwareError, MemoryDomain,
    MultiDuplexSession, MultiOutputSink, NetworkFailure, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, ReverseChannel,
};

use crate::filesink::io_err;
use crate::turn::TurnSet;
use crate::webrtc_codec::{
    enable_tracks, get_video_codec_property, opus_caps, set_video_codec_property, Track,
    WebRtcVideoCodec, VIDEO_CODEC_PROP,
};
use crate::webrtc_receive::UDP_RECV_BYTES;
use crate::webrtc_util::{
    accept_answer_sdp, accept_offer_sdp, add_ice_candidates, connect_failed, feed_datagram,
    get_linger_ms_property, media_time_ns, non_empty, select_host_ip, send_transmit,
    set_linger_ms_property, DEFAULT_LINGER_MS, LINGER_MS_PROP, STUN_SERVER_PROP, TURN_PASS_PROP,
    TURN_SERVER_PROP, TURN_USER_PROP,
};

/// The tracks a duplex session offers at the handshake, in pad order: video on
/// pad 0, audio on pad 1. `track_count` selects how many are active (1 = video
/// only); [`WebRtcDuplexSession::with_spare_tracks`] appends spare pads after
/// them.
fn active_kinds(video_codec: WebRtcVideoCodec, track_count: usize) -> Vec<Track> {
    [Track::Video(video_codec), Track::Audio][..track_count].to_vec()
}

/// The most tracks a session offers at the handshake (video + audio).
const MAX_ACTIVE_TRACKS: usize = 2;

const DEFAULT_SPARE_TRACKS: usize = 0;
const DEFAULT_SPARE_TRACKS_TEXT: &str = "0";

static WEBRTCDUPLEX_PROPS: &[PropertySpec] = &[
    STUN_SERVER_PROP,
    TURN_SERVER_PROP,
    TURN_USER_PROP,
    TURN_PASS_PROP,
    VIDEO_CODEC_PROP,
    LINGER_MS_PROP,
    PropertySpec::new(
        "spare-video-tracks",
        PropKind::Uint,
        "video pads reserved beyond the active tracks, bound when a video track is added mid-session",
    )
    .with_default(DEFAULT_SPARE_TRACKS_TEXT),
    PropertySpec::new(
        "spare-audio-tracks",
        PropKind::Uint,
        "audio pads reserved beyond the active tracks, bound when an audio track is added mid-session",
    )
    .with_default(DEFAULT_SPARE_TRACKS_TEXT),
];

fn spare_track_count(value: &PropValue) -> Result<usize, PropError> {
    let count = value.as_uint().ok_or(PropError::Type)?;
    usize::try_from(count).map_err(|_| PropError::Value)
}

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

    /// Build a channel from raw halves, so a signaller (or a test relay) can
    /// splice itself between two sessions, e.g. to rewrite ICE candidates through
    /// a proxy socket. `pair` is the direct-connect shortcut over this.
    pub fn from_halves(tx: mpsc::Sender<String>, rx: mpsc::Receiver<String>) -> Self {
        SdpChannel { tx, rx }
    }

    /// Send one SDP blob (offer or answer) to the peer. `false` if the peer
    /// dropped its receiver.
    pub async fn send_sdp(&self, sdp: String) -> bool {
        self.tx.send(sdp).await.is_ok()
    }

    /// Await the next SDP blob from the peer, `None` once the peer closed.
    pub async fn recv_sdp(&mut self) -> Option<String> {
        self.rx.recv().await
    }
}

/// Bidirectional sendrecv WebRTC session. See the module docs.
/// Cloneable mid-session control for a [`WebRtcDuplexSession`]: toggle
/// a track on/off, which renegotiates that m-line's direction with the peer
/// (SendRecv <-> Inactive) over the session's `SdpChannel`, or remove it
/// outright. The handle can be used from any task; the session applies
/// pending toggles and removes in its loop.
#[derive(Debug, Clone, Default)]
pub struct DuplexControl {
    toggles: Arc<std::sync::Mutex<Vec<(usize, bool)>>>,
    removes: Arc<std::sync::Mutex<Vec<usize>>>,
}

impl DuplexControl {
    /// Enable / disable send+receive on track `input` (0 = video, 1 = audio).
    pub fn set_track_enabled(&self, input: usize, enabled: bool) {
        self.toggles.lock().unwrap().push((input, enabled));
    }

    /// Remove track `input`: its m-line is stopped (port 0, out of the
    /// BUNDLE group) in the next re-offer, which frees both of its pads for a
    /// later track. Unlike a disable this cannot be undone, since a stopped
    /// m-line never reactivates: reusing the pad negotiates a NEW m-line. The
    /// freed output pad gets no `Eos` (it may be recycled; the session's own
    /// end-of-run EOS covers every pad).
    pub fn remove_track(&self, input: usize) {
        self.removes.lock().unwrap().push(input);
    }

    fn drain(&self) -> Vec<(usize, bool)> {
        core::mem::take(&mut *self.toggles.lock().unwrap())
    }

    fn drain_removes(&self) -> Vec<usize> {
        core::mem::take(&mut *self.removes.lock().unwrap())
    }
}

/// Message prefixes for MID-SESSION renegotiation over the `SdpChannel` (the
/// initial offer/answer stays a bare SDP string, so existing relays keep
/// working): the receiver needs to know whether an SDP is a peer re-offer to
/// answer or the answer to its own pending re-offer.
const RENEGO_OFFER: &str = "offer\n";
const RENEGO_ANSWER: &str = "answer\n";

/// # Example
///
/// ```no_run
/// use g2g_plugins::webrtcduplex::{SdpChannel, SignalRole, WebRtcDuplexSession};
///
/// let (offerer_channel, _answerer_channel) = SdpChannel::pair();
/// let session = WebRtcDuplexSession::new(SignalRole::Offerer, offerer_channel, 2)
///     .with_stun_server("stun.example.com:3478");
/// ```
pub struct WebRtcDuplexSession {
    role: SignalRole,
    sig: Option<SdpChannel>,
    /// Number of sendrecv m-lines offered at the handshake: 1 (video) or 2
    /// (video + audio).
    track_count: usize,
    /// Track kind per pad: the `track_count` active pads first, then any spares
    /// reserved by [`Self::with_spare_tracks`]. Its length is both `input_count`
    /// and `output_count` (input i and output i share m-line i).
    pad_kinds: Vec<Track>,
    spare_video_tracks: usize,
    spare_audio_tracks: usize,
    /// The one video codec on every video m-line: a sendrecv m-line carries
    /// one codec both ways, so the send pads must match it and the recv pads
    /// emit it.
    video_codec: WebRtcVideoCodec,
    stun_server: Option<String>,
    turn_server: Option<String>,
    turn_user: String,
    turn_pass: String,
    /// How long to keep draining the peer after the local send side ends (its
    /// sources reached EOS), so in-flight received frames are not cut off.
    linger: Duration,
    /// Track kind per send input pad, set in `configure_input`.
    inputs: Vec<Option<Track>>,
    /// Per send-input reverse channel: a remote PLI / BWE that names a track's
    /// m-line is routed back to the source feeding that pad. Shared (Arc-backed)
    /// with the runner, which polls each after every push from its source.
    reverse: Vec<ReverseChannel>,
    /// Peak cumulative NACK count observed across str0m's ingress / egress stats,
    /// so a caller can confirm loss-recovery feedback actually flowed. Shared with
    /// the run loop.
    nacks_seen: Arc<AtomicU64>,
    /// Mid-session renegotiation control (track enable/disable), shared with
    /// [`Self::control`] handles.
    control: DuplexControl,
}

impl core::fmt::Debug for WebRtcDuplexSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebRtcDuplexSession")
            .field("role", &self.role)
            .field("track_count", &self.track_count)
            .field("pad_kinds", &self.pad_kinds)
            .field("inputs", &self.inputs)
            .finish()
    }
}

impl WebRtcDuplexSession {
    /// A sendrecv session carrying `track_count` tracks (1 = video; 2 = video +
    /// audio), with the given `role` and SDP signaling channel.
    pub fn new(role: SignalRole, sig: SdpChannel, track_count: usize) -> Self {
        assert!(
            (1..=MAX_ACTIVE_TRACKS).contains(&track_count),
            "track_count must be 1 or 2"
        );
        let video_codec = WebRtcVideoCodec::default();
        Self {
            role,
            sig: Some(sig),
            track_count,
            pad_kinds: active_kinds(video_codec, track_count),
            spare_video_tracks: DEFAULT_SPARE_TRACKS,
            spare_audio_tracks: DEFAULT_SPARE_TRACKS,
            video_codec,
            stun_server: None,
            turn_server: None,
            turn_user: String::new(),
            turn_pass: String::new(),
            linger: Duration::from_millis(DEFAULT_LINGER_MS),
            inputs: alloc::vec![None; track_count],
            reverse: (0..track_count).map(|_| ReverseChannel::new()).collect(),
            nacks_seen: Arc::new(AtomicU64::new(0)),
            control: DuplexControl::default(),
        }
    }

    /// Reserve `video` + `audio` extra pads beyond the active tracks.
    /// A spare pad carries no m-line at the handshake: it binds mid-session,
    /// either when its send pad gets its first frame (the session offers the
    /// peer a new m-line) or when the peer adds an m-line of that kind. Under
    /// the fixed-arity runner, tracks beyond the reserve are rejected; the
    /// dynamic runner (`run_duplex_session_dynamic`) grows the pad count
    /// instead, so it needs no reserve.
    pub fn with_spare_tracks(mut self, video: usize, audio: usize) -> Self {
        self.set_spare_tracks(video, audio);
        self
    }

    fn set_spare_tracks(&mut self, video: usize, audio: usize) {
        self.spare_video_tracks = video;
        self.spare_audio_tracks = audio;
        self.rebuild_pad_kinds();
    }

    /// Carry `codec` on every video m-line instead of H.264. A video send pad
    /// configured with another codec is refused, since the m-line it would
    /// share with the matching recv pad negotiates one codec for both ways.
    pub fn with_video_codec(mut self, codec: WebRtcVideoCodec) -> Self {
        self.set_video_codec(codec);
        self
    }

    fn set_video_codec(&mut self, codec: WebRtcVideoCodec) {
        self.video_codec = codec;
        self.rebuild_pad_kinds();
    }

    fn rebuild_pad_kinds(&mut self) {
        let mut pad_kinds = active_kinds(self.video_codec, self.track_count);
        pad_kinds.extend(core::iter::repeat_n(
            Track::Video(self.video_codec),
            self.spare_video_tracks,
        ));
        pad_kinds.extend(core::iter::repeat_n(Track::Audio, self.spare_audio_tracks));
        let pads = pad_kinds.len();
        self.pad_kinds = pad_kinds;
        self.inputs.resize(pads, None);
        self.reverse.resize_with(pads, ReverseChannel::new);
    }

    /// A cloneable mid-session control handle: toggling a track
    /// renegotiates its m-line direction with the peer.
    pub fn control(&self) -> DuplexControl {
        self.control.clone()
    }

    /// Peak cumulative NACK count str0m has reported for this session (the max of
    /// nacks sent as a receiver and nacks received as a sender). Non-zero proves
    /// loss-recovery feedback flowed, e.g. under a lossy link with RTX active.
    pub fn nacks_seen(&self) -> u64 {
        self.nacks_seen.load(Ordering::Relaxed)
    }

    /// Set a STUN server (`host:port`) for ICE NAT traversal (host-only by
    /// default, which is all a same-host P2P loopback needs).
    pub fn with_stun_server(mut self, server: impl Into<String>) -> Self {
        self.stun_server = Some(server.into());
        self
    }

    /// Set a TURN relay (a `host:port` / `turn:` / `turns:` server, or a
    /// comma-separated list) + long-term credentials, as on the WHIP/WHEP
    /// elements. The relayed candidates ride in the offer/answer SDP (the
    /// duplex signal channel has no trickle), so allocation happens before the
    /// exchange.
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

    /// Override the post-send linger window (default 1.5 s).
    pub fn with_linger(mut self, linger: Duration) -> Self {
        self.linger = linger;
        self
    }
}

/// The output caps for a given track kind.
fn caps_for(kind: Track) -> Caps {
    match kind {
        Track::Video(codec) => codec.caps_placeholder(),
        Track::Audio => opus_caps(2),
    }
}

/// The log category this session reports its pad decisions on.
const DUPLEX_CATEGORY: &str = "webrtcduplex";

/// Ask the runner for one more recv pad of `kind` when no declared pad is free,
/// and take it into the pad tables so the new port announces its caps
/// before its first frame like any other. `None` when the runner cannot grow (the
/// fixed-arity duplex runner), which leaves the caller with today's behavior.
fn grow_out_pad(
    out: &mut dyn MultiOutputSink,
    pad_kinds: &mut Vec<Track>,
    announced: &mut Vec<bool>,
    kind: Track,
) -> Option<usize> {
    let port = out.add_port(&caps_for(kind))?;
    if port >= pad_kinds.len() {
        pad_kinds.resize(port + 1, kind);
        announced.resize(port + 1, false);
    }
    Some(port)
}

/// One negotiated m-line and the pads it serves: `out_pad` emits the peer's
/// media, `in_pad` feeds the send direction (either may be missing when no pad
/// of that kind is free). A spare pad has no binding until it activates.
#[derive(Debug)]
struct Binding {
    mid: Mid,
    kind: Track,
    /// Negotiated payload type, discovered from the writer on the first frame.
    pt: Option<Pt>,
    in_pad: Option<usize>,
    out_pad: Option<usize>,
}

fn binding_of_mid(bindings: &[Binding], mid: Mid) -> Option<usize> {
    bindings.iter().position(|b| b.mid == mid)
}

/// The lowest output pad of `kind` no binding has claimed (a spare once the
/// active pads are taken).
fn free_out_pad(bindings: &[Binding], pad_kinds: &[Track], kind: Track) -> Option<usize> {
    (0..pad_kinds.len())
        .find(|o| pad_kinds[*o] == kind && !bindings.iter().any(|b| b.out_pad == Some(*o)))
}

/// The same on the send side: the lowest input pad configured for `kind` that no
/// binding has claimed (the send pads may be wired in either order).
fn free_in_pad(bindings: &[Binding], inputs: &[Option<Track>], kind: Track) -> Option<usize> {
    (0..inputs.len())
        .find(|i| inputs[*i] == Some(kind) && !bindings.iter().any(|b| b.in_pad == Some(*i)))
}

/// The track a send pad's caps select: Opus audio, or video in the session's
/// one codec. Video in another codec is `None`, since its m-line would have to
/// carry that codec back as well.
fn send_track_for(video_codec: WebRtcVideoCodec, caps: &Caps) -> Option<Track> {
    match Track::from_caps(caps)? {
        Track::Video(codec) if codec != video_codec => None,
        track => Some(track),
    }
}

impl WebRtcDuplexSession {
    fn send_track(&self, caps: &Caps) -> Option<Track> {
        send_track_for(self.video_codec, caps)
    }
}

impl MultiDuplexSession for WebRtcDuplexSession {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;

    fn input_count(&self) -> usize {
        self.pad_kinds.len()
    }

    fn output_count(&self) -> usize {
        self.pad_kinds.len()
    }

    fn intercept_caps(&self, _input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match self.send_track(upstream_caps) {
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
        let track = self
            .send_track(absolute_caps)
            .ok_or(G2gError::CapsMismatch)?;
        *self.inputs.get_mut(input).ok_or(G2gError::CapsMismatch)? = Some(track);
        Ok(ConfigureOutcome::Accepted)
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        let kind = self
            .pad_kinds
            .get(output)
            .copied()
            .ok_or(G2gError::CapsMismatch)?;
        Ok(caps_for(kind))
    }

    fn reverse_channel(&self, input: usize) -> Option<ReverseChannel> {
        self.reverse.get(input).cloned()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        WEBRTCDUPLEX_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "stun-server" => self.stun_server = non_empty(value.as_str().ok_or(PropError::Type)?),
            "turn-server" => self.turn_server = non_empty(value.as_str().ok_or(PropError::Type)?),
            "turn-user" => self.turn_user = value.as_str().ok_or(PropError::Type)?.into(),
            "turn-pass" => self.turn_pass = value.as_str().ok_or(PropError::Type)?.into(),
            "video-codec" => {
                let mut codec = self.video_codec;
                set_video_codec_property(&mut codec, &value)?;
                self.set_video_codec(codec);
            }
            "linger-ms" => set_linger_ms_property(&mut self.linger, &value)?,
            "spare-video-tracks" => {
                self.set_spare_tracks(spare_track_count(&value)?, self.spare_audio_tracks)
            }
            "spare-audio-tracks" => {
                self.set_spare_tracks(self.spare_video_tracks, spare_track_count(&value)?)
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let value = match name {
            "stun-server" => PropValue::Str(self.stun_server.clone().unwrap_or_default()),
            "turn-server" => PropValue::Str(self.turn_server.clone().unwrap_or_default()),
            "turn-user" => PropValue::Str(self.turn_user.clone()),
            "turn-pass" => PropValue::Str(self.turn_pass.clone()),
            "video-codec" => get_video_codec_property(self.video_codec),
            "linger-ms" => get_linger_ms_property(self.linger),
            "spare-video-tracks" => PropValue::Uint(self.spare_video_tracks as u64),
            "spare-audio-tracks" => PropValue::Uint(self.spare_audio_tracks as u64),
            _ => return None,
        };
        Some(value)
    }

    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a> {
        let role = self.role;
        let track_count = self.track_count;
        // Pad tables grow with the session (M1014): a send pad appears when a
        // track the runner attached mid-run announces its caps, a recv pad when
        // the runner hands out a port for a track no declared pad can carry.
        let mut pad_kinds = self.pad_kinds.clone();
        let mut inputs = self.inputs.clone();
        let stun = self.stun_server.clone();
        let turn_server = self.turn_server.clone();
        let turn_user = self.turn_user.clone();
        let turn_pass = self.turn_pass.clone();
        let linger = self.linger;
        let sig = self.sig.take();
        let control_handle = self.control.clone();
        // Per-input reverse channels, so a remote PLI / BWE naming a track's
        // m-line routes back to the source feeding that m-line's send pad.
        let mut reverse = self.reverse.clone();
        let nacks_seen = self.nacks_seen.clone();
        Box::pin(async move {
            let hw = || G2gError::Hardware(HardwareError::Other);
            let log = g2g_core::log::Target::category(DUPLEX_CATEGORY);
            let mut sig = sig.ok_or_else(hw)?;

            let host_ip = select_host_ip();
            let socket = UdpSocket::bind((host_ip, 0)).await.map_err(io_err)?;
            let local = socket.local_addr().map_err(io_err)?;

            let config = RtcConfig::new()
                .set_crypto_provider(Arc::new(from_feature_flags()))
                .clear_codecs()
                // Congestion control so the peer's BWE estimate arrives as
                // `Event::EgressBitrateEstimate`, routed to the video track below.
                .enable_bwe(Some(Bitrate::bps(2_000_000)))
                // Periodic stats surface NACK counts (loss-recovery feedback), read
                // back via `nacks_seen`.
                .set_stats_interval(Some(Duration::from_millis(500)));
            let video_codec = self.video_codec;
            let mut rtc = enable_tracks(config, active_kinds(video_codec, MAX_ACTIVE_TRACKS))
                .build(Instant::now());
            // Host (and optional reflexive / relayed) candidates ride in the
            // SDP, so they must be added before the offer/answer is generated
            // below (M719: the duplex path gained TURN).
            add_ice_candidates(&mut rtc, &socket, stun.as_deref()).await?;
            let mut turn = match turn_server.as_deref() {
                Some(servers) => {
                    TurnSet::setup(&mut rtc, &socket, servers, &turn_user, &turn_pass).await
                }
                None => TurnSet::empty(),
            };
            let mut refresh_at = Instant::now() + crate::turn::REFRESH_INTERVAL;

            // One binding per negotiated m-line, holding the pads it serves.
            // The offerer learns its `Mid`s from `add_media` (str0m does not emit
            // `MediaAdded` for media the local side added); the answerer learns
            // them from `MediaAdded` when it accepts the offer.
            let mut bindings: Vec<Binding> = Vec::new();

            // SDP handshake: each ACTIVE pad gets one sendrecv m-line (spares get
            // none). The offerer adds the media and creates the offer; the
            // answerer accepts the offer (whose m-lines it inherits).
            match role {
                SignalRole::Offerer => {
                    let (offer_sdp, pending) = {
                        let mut api = rtc.sdp_api();
                        for (o, kind) in pad_kinds.iter().enumerate().take(track_count) {
                            let mid = api.add_media(
                                kind.media_kind(),
                                Direction::SendRecv,
                                None,
                                None,
                                None,
                            );
                            let in_pad = free_in_pad(&bindings, &inputs, *kind);
                            bindings.push(Binding {
                                mid,
                                kind: *kind,
                                pt: None,
                                in_pad,
                                out_pad: Some(o),
                            });
                        }
                        let (offer, pending) = api.apply().ok_or_else(hw)?;
                        (offer.to_sdp_string(), pending)
                    };
                    sig.tx
                        .send(offer_sdp)
                        .await
                        .map_err(|_| G2gError::Network(NetworkFailure::Connect))?;
                    let answer_sdp = sig
                        .rx
                        .recv()
                        .await
                        .ok_or(G2gError::Network(NetworkFailure::Connect))?;
                    accept_answer_sdp(&mut rtc, pending, &answer_sdp)
                        .map_err(|reason| connect_failed(&log, &reason))?;
                }
                SignalRole::Answerer => {
                    let offer_sdp = sig
                        .rx
                        .recv()
                        .await
                        .ok_or(G2gError::Network(NetworkFailure::Connect))?;
                    let answer = accept_offer_sdp(&mut rtc, &offer_sdp)
                        .map_err(|reason| connect_failed(&log, &reason))?;
                    sig.tx
                        .send(answer.to_sdp_string())
                        .await
                        .map_err(|_| G2gError::Network(NetworkFailure::Connect))?;
                }
            }

            // Announce each ACTIVE output's caps before its first frame; a spare
            // pad is announced when it binds mid-session.
            let pad_count = pad_kinds.len();
            let mut announced = alloc::vec![false; pad_count];
            for (o, kind) in pad_kinds.iter().enumerate().take(track_count) {
                out.push_to(o, PipelinePacket::CapsChanged(caps_for(*kind)))
                    .await?;
                announced[o] = true;
            }

            let mut buf = alloc::vec![0u8; UDP_RECV_BYTES];
            let mut seq = 0u64;
            let mut received = 0u64;
            let mut send_done = false;
            // Set when the local send side ends; the loop finishes after it.
            let mut drain_deadline: Option<Instant> = None;
            // Mid-session renegotiation: pending toggles and removes come in via
            // the control handle (M729 / M785) and spare pads add m-lines (M784);
            // one exchange in flight at a time (its answer clears
            // `renego_pending`). On glare (a peer re-offer arriving while ours is
            // pending) the ANSWERER role yields: it drops its pending exchange and
            // answers the peer's offer.
            let control = control_handle;
            let mut renego_pending: Option<str0m::change::SdpPendingOffer> = None;
            // Send pads added mid-run, waiting for their reverse channel. The
            // lookup cannot happen where the pad is learned: that is inside the
            // select below, which holds `inbound` borrowed for its `recv`.
            let mut pending_reverse: Vec<usize> = Vec::new();

            // Run after every SDP application: drop the bindings whose m-line
            // str0m no longer has (a retracted ADD, e.g. one this side yielded on
            // glare) or which either peer stopped (M785). Their pads go back to
            // spare, free for a later track, and the output pad re-announces its
            // caps when one claims it. No `Eos` on a freed pad: it may be
            // recycled, and the end of the run EOSes every pad anyway.
            macro_rules! prune_bindings {
                () => {
                    bindings.retain(|b| {
                        let live = rtc.media(b.mid).is_some_and(|m| !m.stopped());
                        if !live {
                            if let Some(o) = b.out_pad {
                                announced[o] = false;
                            }
                        }
                        live
                    });
                };
            }

            macro_rules! finish {
                () => {{
                    for o in 0..pad_kinds.len() {
                        out.push_to(o, PipelinePacket::Eos).await?;
                    }
                    return Ok(received);
                }};
            }

            loop {
                for idx in core::mem::take(&mut pending_reverse) {
                    if let Some(rc) = inbound.reverse_channel(idx) {
                        reverse[idx] = rc;
                    }
                }

                // (output port, pts_ns, data) collected while draining poll_output.
                let mut frames: Vec<(usize, u64, Vec<u8>)> = Vec::new();
                let deadline = loop {
                    match rtc.poll_output() {
                        // after a DTLS close_notify str0m returns a timeout that never fires
                        Ok(Output::Timeout(_)) if !rtc.is_alive() => finish!(),
                        Ok(Output::Timeout(t)) => break t,
                        Ok(Output::Transmit(t)) => send_transmit(&socket, &mut turn, &t).await,
                        // A remote-created m-line: the answerer learns its initial
                        // `Mid`s here (the offerer captured them from `add_media`),
                        // and either side learns a mid-session ADD by the peer
                        // (M784). Bind it to the first free pad of its kind, or to
                        // one grown for it (M1014); only a runner that cannot grow
                        // leaves it unbound, its media skipped by the unknown-mid
                        // path below.
                        Ok(Output::Event(Event::MediaAdded(m))) => {
                            if binding_of_mid(&bindings, m.mid).is_none() {
                                let kind = match m.kind {
                                    MediaKind::Video => Track::Video(video_codec),
                                    MediaKind::Audio => Track::Audio,
                                };
                                let pad = free_out_pad(&bindings, &pad_kinds, kind).or_else(|| {
                                    grow_out_pad(out, &mut pad_kinds, &mut announced, kind)
                                });
                                if let Some(out_pad) = pad {
                                    let in_pad = free_in_pad(&bindings, &inputs, kind);
                                    bindings.push(Binding {
                                        mid: m.mid,
                                        kind,
                                        pt: None,
                                        in_pad,
                                        out_pad: Some(out_pad),
                                    });
                                }
                            }
                        }
                        Ok(Output::Event(Event::IceConnectionStateChange(
                            IceConnectionState::Disconnected,
                        ))) => finish!(),
                        // Remote PLI: route the keyframe request to the send source
                        // feeding the track whose m-line it names (by mid), so only
                        // that encoder emits an IDR.
                        Ok(Output::Event(Event::KeyframeRequest(req))) => {
                            if let Some(rc) = binding_of_mid(&bindings, req.mid)
                                .and_then(|b| bindings[b].in_pad)
                                .and_then(|i| reverse.get(i))
                            {
                                rc.request_keyframe();
                            }
                        }
                        // Congestion-control estimate (whole-connection): relay it to
                        // the first bound video track, the bitrate-adaptive one
                        // (Opus bitrate adaptation is a separate follow-up), as the
                        // fan-in session does.
                        Ok(Output::Event(Event::EgressBitrateEstimate(kind))) => {
                            let bps = match kind {
                                BweKind::Twcc(b) | BweKind::Remb(_, b) => Some(b.as_u64()),
                                _ => None,
                            };
                            let rc = bindings
                                .iter()
                                .find(|b| b.kind.is_video())
                                .and_then(|b| b.in_pad)
                                .and_then(|i| reverse.get(i));
                            if let (Some(bps), Some(rc)) = (bps, rc) {
                                rc.set_bitrate(bps.min(u32::MAX as u64) as u32);
                            }
                        }
                        Ok(Output::Event(Event::MediaData(d))) => {
                            let pts_ns = media_time_ns(d.time);
                            // Unknown mid, or an m-line with no output pad: skip.
                            let Some(port) =
                                binding_of_mid(&bindings, d.mid).and_then(|b| bindings[b].out_pad)
                            else {
                                continue;
                            };
                            frames.push((port, pts_ns, pad_kinds[port].pipeline_unit(&d.data)));
                        }
                        // Loss-recovery feedback counters (cumulative): nacks sent
                        // as a receiver (ingress) and nacks received as a sender
                        // (egress). Keep the peak so a caller can confirm RTX was
                        // exercised under loss.
                        Ok(Output::Event(Event::MediaIngressStats(s))) => {
                            nacks_seen.fetch_max(s.nacks, Ordering::Relaxed);
                        }
                        Ok(Output::Event(Event::MediaEgressStats(s))) => {
                            nacks_seen.fetch_max(s.nacks, Ordering::Relaxed);
                        }
                        Ok(Output::Event(_)) => {}
                        Err(_) => finish!(),
                    }
                };

                for (port, pts_ns, data) in frames {
                    // A pad bound mid-session announces its caps before its first
                    // frame (the active pads were announced at session start).
                    if !announced[port] {
                        out.push_to(port, PipelinePacket::CapsChanged(caps_for(pad_kinds[port])))
                            .await?;
                        announced[port] = true;
                    }
                    let keyframe = pad_kinds[port].is_keyframe(&data);
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

                // Apply pending track toggles and removes as ONE renegotiation
                // exchange (batched into a single re-offer).
                if renego_pending.is_none() {
                    let toggles = control.drain();
                    let removes = control.drain_removes();
                    if !toggles.is_empty() || !removes.is_empty() {
                        let mut api = rtc.sdp_api();
                        let mut changed = false;
                        for (idx, enabled) in toggles {
                            let mid = bindings
                                .iter()
                                .find(|b| b.in_pad == Some(idx))
                                .map(|b| b.mid);
                            if let Some(mid) = mid {
                                let dir = if enabled {
                                    Direction::SendRecv
                                } else {
                                    Direction::Inactive
                                };
                                api.set_direction(mid, dir);
                                changed = true;
                            }
                        }
                        // A removed track stops its m-line (M785): str0m marks it
                        // stopped straight away, so the prune below frees its pads
                        // whether or not the exchange reaches the peer.
                        for idx in removes {
                            let mid = bindings
                                .iter()
                                .find(|b| b.in_pad == Some(idx))
                                .map(|b| b.mid);
                            if let Some(mid) = mid {
                                api.stop_media(mid);
                                changed = true;
                            }
                        }
                        if changed {
                            if let Some((offer, pending)) = api.apply() {
                                let msg = alloc::format!("{RENEGO_OFFER}{}", offer.to_sdp_string());
                                if sig.tx.send(msg).await.is_ok() {
                                    renego_pending = Some(pending);
                                }
                            }
                            prune_bindings!();
                        }
                    }
                }

                let timeout = deadline.saturating_duration_since(Instant::now());
                tokio::select! {
                    msg = sig.rx.recv() => {
                        match msg {
                            // Peer signalling closed: media keeps flowing (the
                            // channel is only needed for renegotiation).
                            None => {}
                            Some(m) => {
                                if let Some(sdp) = m.strip_prefix(RENEGO_ANSWER) {
                                    if let (Some(pending), Ok(answer)) = (
                                        renego_pending.take(),
                                        SdpAnswer::from_sdp_string(sdp),
                                    ) {
                                        let _ = rtc.sdp_api().accept_answer(pending, answer);
                                        prune_bindings!();
                                    }
                                } else if let Some(sdp) = m.strip_prefix(RENEGO_OFFER) {
                                    // Glare rule: the answerer role yields its
                                    // own pending exchange to the peer's offer.
                                    // Its unanswered ADD is retracted by the
                                    // prune (str0m never created that media), so
                                    // a later frame on the pad re-offers it.
                                    if renego_pending.is_some()
                                        && matches!(role, SignalRole::Answerer)
                                    {
                                        renego_pending = None;
                                        prune_bindings!();
                                    }
                                    if renego_pending.is_none() {
                                        if let Ok(answer) = accept_offer_sdp(&mut rtc, sdp) {
                                            let msg = alloc::format!(
                                                "{RENEGO_ANSWER}{}",
                                                answer.to_sdp_string()
                                            );
                                            let _ = sig.tx.send(msg).await;
                                            // The peer may have stopped an
                                            // m-line: free its pads (M785).
                                            prune_bindings!();
                                        }
                                    }
                                }
                            }
                        }
                    }
                    r = socket.recv_from(&mut buf) => {
                        let Ok((n, source)) = r else { finish!() };
                        if !feed_datagram(&mut rtc, &mut turn, local, &buf[..n], source) {
                            finish!();
                        }
                    }
                    _ = tokio::time::sleep_until(tokio::time::Instant::from_std(refresh_at)), if !turn.is_empty() => {
                        turn.refresh_all(&socket).await;
                        refresh_at = Instant::now() + crate::turn::REFRESH_INTERVAL;
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
                                // Route by the m-line this send pad feeds, not the
                                // fixed KINDS position (a pipeline may wire audio to
                                // pad 0 and video to pad 1). Falls back to a kind
                                // match for an initial track whose binding claimed
                                // no input pad.
                                let kind = inputs.get(idx).copied().flatten();
                                let bound = bindings
                                    .iter()
                                    .position(|b| b.in_pad == Some(idx))
                                    .or_else(|| {
                                        kind.and_then(|k| {
                                            bindings.iter().position(|b| {
                                                b.kind == k && b.in_pad.is_none()
                                            })
                                        })
                                    });
                                match bound {
                                    Some(b) => {
                                        let b = &mut bindings[b];
                                        let kind = b.kind;
                                        // Drop send frames until the m-line is
                                        // negotiated (no writer before that).
                                        if let MemoryDomain::System(slice) = &frame.domain {
                                            if b.pt.is_none() {
                                                if let Some(w) = rtc.writer(b.mid) {
                                                    b.pt = w
                                                        .payload_params()
                                                        .find(|p| p.spec().codec == kind.codec())
                                                        .map(|p| p.pt());
                                                }
                                            }
                                            if let Some(p) = b.pt {
                                                let rtp_time =
                                                    kind.media_time(frame.timing.pts_ns);
                                                if let Some(w) = rtc.writer(b.mid) {
                                                    let _ = w.write(p, Instant::now(), rtp_time,
                                                        slice.as_slice().to_vec());
                                                }
                                            }
                                        }
                                    }
                                    // A send pad with no m-line yet, either a spare
                                    // (M784) or one added mid-run (M1014): offer the
                                    // peer a new sendrecv m-line and claim an output
                                    // pad of the same kind, growing the recv side if
                                    // none is free. This frame is dropped (nothing
                                    // can be written before the answer lands); a
                                    // later one starts the stream. With another
                                    // exchange in flight, retry later.
                                    None => {
                                        let add = kind.filter(|_| {
                                            idx >= track_count && renego_pending.is_none()
                                        });
                                        if let Some(kind) = add {
                                            let out_pad =
                                                free_out_pad(&bindings, &pad_kinds, kind).or_else(
                                                    || {
                                                        grow_out_pad(
                                                            out,
                                                            &mut pad_kinds,
                                                            &mut announced,
                                                            kind,
                                                        )
                                                    },
                                                );
                                            if let Some(out_pad) = out_pad {
                                                let mut api = rtc.sdp_api();
                                                let mid = api.add_media(
                                                    kind.media_kind(),
                                                    Direction::SendRecv,
                                                    None,
                                                    None,
                                                    None,
                                                );
                                                if let Some((offer, pending)) = api.apply() {
                                                    let msg = alloc::format!(
                                                        "{RENEGO_OFFER}{}",
                                                        offer.to_sdp_string()
                                                    );
                                                    if sig.tx.send(msg).await.is_ok() {
                                                        renego_pending = Some(pending);
                                                        bindings.push(Binding {
                                                            mid,
                                                            kind,
                                                            pt: None,
                                                            in_pad: Some(idx),
                                                            out_pad: Some(out_pad),
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // A send pad the runner attached mid-run (M1014)
                            // introduces itself with its caps on an index past the
                            // ones configured at startup. Its kind comes from those
                            // caps; the local-ADD path above then offers the peer an
                            // m-line for it on its first frame.
                            Some((idx, PipelinePacket::CapsChanged(caps))) => {
                                if idx >= inputs.len() {
                                    match send_track_for(video_codec, &caps) {
                                        Some(kind) => {
                                            inputs.resize(idx + 1, None);
                                            inputs[idx] = Some(kind);
                                            reverse.resize_with(idx + 1, ReverseChannel::new);
                                            pending_reverse.push(idx);
                                        }
                                        None => g2g_warn!(
                                            g2g_core::log::Target::category(DUPLEX_CATEGORY),
                                            "send pad {idx} announced caps this session cannot \
                                             carry ({caps:?}): dropping its frames"
                                        ),
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
    use g2g_core::{AudioFormat, Dim, Rate, VideoCodec};

    fn h264_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
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
    fn spare_tracks_add_pads_beyond_the_active_ones() {
        let (a, _b) = SdpChannel::pair();
        let s = WebRtcDuplexSession::new(SignalRole::Offerer, a, 2).with_spare_tracks(1, 1);
        assert_eq!(s.input_count(), 4);
        assert_eq!(s.output_count(), 4);
        // Spares follow the active pads: video on 2, audio on 3.
        assert!(matches!(
            s.output_caps(2),
            Ok(Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            })
        ));
        assert!(matches!(
            s.output_caps(3),
            Ok(Caps::Audio {
                format: AudioFormat::Opus,
                ..
            })
        ));
        assert!(s.output_caps(4).is_err());
        assert_eq!(s.reverse.len(), 4);
    }

    #[test]
    fn configure_input_reads_track_kind_from_caps() {
        let (a, _b) = SdpChannel::pair();
        let mut s = WebRtcDuplexSession::new(SignalRole::Answerer, a, 2);
        assert!(s.configure_input(0, &h264_caps()).is_ok());
        assert!(s.configure_input(1, &opus_caps(2)).is_ok());
        assert_eq!(
            s.inputs,
            alloc::vec![
                Some(Track::Video(WebRtcVideoCodec::H264)),
                Some(Track::Audio)
            ]
        );
        // Video in another codec cannot share the session's video m-line.
        let vp8 = WebRtcVideoCodec::Vp8.caps_any();
        assert_eq!(s.intercept_caps(0, &vp8), Err(G2gError::CapsMismatch));
        assert!(s.configure_input(0, &vp8).is_err());
        // A session built for VP8 takes VP8 pads and announces VP8 outputs.
        let (c, _d) = SdpChannel::pair();
        let mut vp8_session = WebRtcDuplexSession::new(SignalRole::Answerer, c, 2)
            .with_video_codec(WebRtcVideoCodec::Vp8)
            .with_spare_tracks(1, 0);
        assert!(vp8_session.configure_input(0, &vp8).is_ok());
        assert!(matches!(
            vp8_session.output_caps(2),
            Ok(Caps::CompressedVideo {
                codec: VideoCodec::Vp8,
                ..
            })
        ));
        // Non-A/V caps rejected.
        let raw = Caps::RawVideo {
            format: g2g_core::RawVideoFormat::I420,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(s.intercept_caps(0, &raw), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn every_property_round_trips() {
        let (a, _b) = SdpChannel::pair();
        let mut s = WebRtcDuplexSession::new(SignalRole::Offerer, a, 2);
        assert_eq!(
            s.get_property(LINGER_MS_PROP.name),
            Some(
                LINGER_MS_PROP
                    .parse_value(LINGER_MS_PROP.default.unwrap())
                    .unwrap()
            )
        );
        for spec in s.properties() {
            let value = crate::webrtc_util::sample_property_value(spec);
            s.set_property(spec.name, value.clone()).unwrap();
            assert_eq!(s.get_property(spec.name), Some(value), "{}", spec.name);
        }
        s.set_property("stun-server", PropValue::Str(String::new()))
            .unwrap();
        assert_eq!(s.stun_server, None);
    }

    #[test]
    fn unknown_and_mistyped_properties_are_distinct_errors() {
        let (a, _b) = SdpChannel::pair();
        let mut s = WebRtcDuplexSession::new(SignalRole::Offerer, a, 2);
        assert_eq!(
            s.set_property("no-such-property", PropValue::Uint(0)),
            Err(PropError::Unknown)
        );
        assert_eq!(s.get_property("no-such-property"), None);
        assert_eq!(
            s.set_property("linger-ms", PropValue::Str("fast".into())),
            Err(PropError::Type)
        );
        assert_eq!(
            s.set_property("stun-server", PropValue::Uint(0)),
            Err(PropError::Type)
        );
    }

    #[test]
    fn video_codec_property_retags_every_video_pad() {
        let (a, _b) = SdpChannel::pair();
        let mut s = WebRtcDuplexSession::new(SignalRole::Offerer, a, 2).with_spare_tracks(1, 0);
        s.set_property("video-codec", PropValue::Str("vp8".into()))
            .unwrap();
        for pad in [0, 2] {
            assert!(matches!(
                s.output_caps(pad),
                Ok(Caps::CompressedVideo {
                    codec: VideoCodec::Vp8,
                    ..
                })
            ));
        }
        assert!(s.intercept_caps(0, &h264_caps()).is_err());
        assert!(s
            .intercept_caps(0, &WebRtcVideoCodec::Vp8.caps_any())
            .is_ok());
    }

    #[test]
    fn spare_track_properties_set_absolute_counts() {
        let (a, _b) = SdpChannel::pair();
        let mut s = WebRtcDuplexSession::new(SignalRole::Offerer, a, 2);
        s.set_property("spare-video-tracks", PropValue::Uint(1))
            .unwrap();
        s.set_property("spare-video-tracks", PropValue::Uint(1))
            .unwrap();
        assert_eq!(s.input_count(), 3);
        s.set_property("spare-audio-tracks", PropValue::Uint(2))
            .unwrap();
        assert_eq!(s.input_count(), 5);
        assert_eq!(s.output_count(), 5);
        assert_eq!(s.reverse.len(), 5);
        assert!(matches!(
            s.output_caps(4),
            Ok(Caps::Audio {
                format: AudioFormat::Opus,
                ..
            })
        ));
        s.set_property("spare-video-tracks", PropValue::Uint(0))
            .unwrap();
        assert_eq!(s.input_count(), 4);
        assert!(matches!(
            s.output_caps(2),
            Ok(Caps::Audio {
                format: AudioFormat::Opus,
                ..
            })
        ));
    }

    #[test]
    fn control_queues_toggles_and_removes_separately() {
        let control = DuplexControl::default();
        control.set_track_enabled(1, false);
        control.remove_track(0);
        control.remove_track(2);
        assert_eq!(control.drain(), alloc::vec![(1, false)]);
        assert_eq!(control.drain_removes(), alloc::vec![0, 2]);
        // Draining takes the queue.
        assert!(control.drain().is_empty());
        assert!(control.drain_removes().is_empty());
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
