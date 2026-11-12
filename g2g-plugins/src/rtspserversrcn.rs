// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Concurrent multi-publisher RTSP ingest (`RtspServerSrcN`, `rtsp-server`
//! feature): one RTSP endpoint, N output pads, one recording publisher per pad.
//! The fan-out sibling of [`RtspServerSrc`](crate::rtspserversrc), which serves
//! publishers *sequentially* on its single pad. Both drive the same per-session
//! machinery (`rtspingest`): ANNOUNCE / SETUP / RECORD over the TCP
//! control channel, then RTP over unicast UDP or TCP-interleaved, depayloaded to
//! Annex-B H.264 access units.
//!
//! Shape: a [`MultiOutputSource`] driven by
//! [`run_fanout_session`](g2g_core::runtime::run_fanout_session), so a
//! contribution server can ingest several encoders at once
//! (`rtspserversrcn name=s port=8554  s. ! ...  s. ! ...`).
//!
//! With `audio=true` each publisher records onto two pads, its video and then
//! its first AAC track as ADTS, so publisher k owns pads 2k and 2k+1.
//!
//! Pad lifecycle: the pad count is fixed at construction (`max-sessions`, the
//! linked pad count on a launch line). One acceptor binds each new publisher to
//! the first free pad and hands it to that pad's session; the pads then run
//! concurrently, so a slow handshake or a departing publisher on one never stalls
//! another. A publisher arriving with every pad busy is refused with `503 Service
//! Unavailable` rather than queued in the accept backlog. When a publisher
//! leaves, its pad is freed for the next one **without** an `Eos`: the pad's
//! downstream branch stays live and PTS continues forward across the handover, as
//! on the single-pad element. `Eos` is per pad and final: on `num-buffers` for
//! that pad, and on every remaining pad when the element stops.
//!
//! The pad sessions are polled concurrently in one task and push through a
//! channel the `run` loop drains into the outputs, since only one holder of the
//! [`MultiOutputSink`] can exist.

use core::cell::Cell;
use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use tokio::sync::mpsc;

use g2g_core::element::BoxFuture;
use g2g_core::log::Target;
use g2g_core::runtime::join_all;
use g2g_core::{
    g2g_error, Caps, G2gError, MultiOutputSink, MultiOutputSource, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec,
};

use crate::filesink::io_err;
use crate::rtpdepay::{RtpEncoding, ADTS_AAC_CAPS};
use crate::rtpjitter::JitterConfig;
use crate::rtprecv::RtpRecvConfig;
use crate::rtspingest::{
    end_session, forward_pads, handshake, inactive_for, ingest_codec, receive_session, refuse_one,
    IngestFormat, IngestPad, PadSink, SessionEnd, SessionTap, PAD_QUEUE,
};
use crate::rtspserver::{sdp_h264, RtspResponder, DEFAULT_SESSION_TIMEOUT_SECS};

/// Default dynamic RTP payload type for H.264.
const DEFAULT_PAYLOAD_TYPE: u8 = 96;
/// Declared geometry hint (SPS is authoritative; a downstream decoder corrects).
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const DEFAULT_FPS: u32 = 30;

/// # Example
///
/// ```no_run
/// use g2g_plugins::rtspserversrcn::RtspServerSrcN;
///
/// let src = RtspServerSrcN::new("0.0.0.0:8554".parse().unwrap(), 4)
///     .with_video_size(1920, 1080)
///     .with_framerate(30);
/// ```
#[derive(Debug)]
pub struct RtspServerSrcN {
    rtsp_addr: SocketAddr,
    payload_type: u8,
    ssrc: u32,
    width: u32,
    height: u32,
    fps: u32,
    /// `u64::MAX` means run until downstream shuts down; otherwise each pad
    /// stops after this many access units and emits EOS (the test / bounded
    /// path), and the element stops once every pad has.
    frame_limit: u64,
    /// Receive-path tuning (jitter reorder + optional RTCP/NACK), shared with
    /// [`UdpSrc`](crate::udpsrc) via [`crate::rtprecv`], applied to every pad.
    recv: RtpRecvConfig,
    // the output pad count, the linked pad count on a launch line
    outputs: usize,
    // set by `max-sessions`, which then fixes the pad count whatever `audio` is set to
    max_sessions: Option<usize>,
    // each publisher's AAC track rides the pad after its video pad
    audio: bool,
    /// Per-session inactivity budget: a publisher that sends neither media nor a
    /// control request within it is torn down and its pad freed. `u64::MAX`
    /// disables reaping.
    session_timeout_ns: u64,
    listener: Option<StdTcpListener>,
    /// Publishers that reached RECORD, and connections refused (503) because
    /// every pad was busy. Readable once `run` returns.
    sessions_served: u64,
    sessions_refused: u64,
}

impl RtspServerSrcN {
    /// Listen for publishers on `rtsp_addr` (e.g. `0.0.0.0:8554`), serving up to
    /// `pads` of them at once, one per output pad.
    pub fn new(rtsp_addr: SocketAddr, pads: usize) -> Self {
        Self {
            rtsp_addr,
            payload_type: DEFAULT_PAYLOAD_TYPE,
            ssrc: 0,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            fps: DEFAULT_FPS,
            frame_limit: u64::MAX,
            recv: RtpRecvConfig {
                jitter: JitterConfig::default(),
                // Off until a separate RTCP port / rtcp-mux is negotiated.
                rtcp_rr_interval_ms: 0,
                nack_enabled: false,
                rtx: None,
                fec_pt: None,
                flexfec_pt: None,
                declared_caps: None,
                capture_time_ids: Default::default(),
                encoding: RtpEncoding::H264,
                parameter_sets: Vec::new(),
                ..RtpRecvConfig::default()
            },
            outputs: pads.max(1),
            max_sessions: None,
            audio: false,
            session_timeout_ns: DEFAULT_SESSION_TIMEOUT_SECS as u64 * 1_000_000_000,
            listener: None,
            sessions_served: 0,
            sessions_refused: 0,
        }
    }

    /// Use an already-bound listener (so a test can pick an ephemeral port).
    pub fn from_listener(listener: StdTcpListener, pads: usize) -> Result<Self, G2gError> {
        let addr = listener.local_addr().map_err(io_err)?;
        Ok(Self {
            listener: Some(listener),
            ..Self::new(addr, pads)
        })
    }

    /// Set the RTP payload type and the base SSRC negotiated in SETUP. Each pad
    /// answers with its own SSRC derived from this one.
    pub fn with_rtp(mut self, payload_type: u8, ssrc: u32) -> Self {
        self.payload_type = payload_type & 0x7F;
        self.ssrc = ssrc;
        self
    }

    /// Declared output geometry (a negotiation hint; SPS is authoritative).
    pub fn with_video_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Declared output frame rate (a negotiation hint).
    pub fn with_framerate(mut self, fps: u32) -> Self {
        self.fps = fps;
        self
    }

    /// Stop each pad after `n` access units and emit EOS on it; the element ends
    /// once every pad has (0 emits EOS on every pad without receiving). Without
    /// this the source runs until downstream shuts down (RTP has no in-band end
    /// marker).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Tune the receive-side jitter buffer for every pad: hold a gap up to
    /// `max_hold_ms` before declaring it lost, buffering at most `max_depth`
    /// packets. A `max_depth` of 0 disables reordering (in-order passthrough).
    pub fn with_jitter(mut self, max_hold_ms: u64, max_depth: usize) -> Self {
        self.recv.jitter = JitterConfig::new(max_hold_ms, max_depth);
        self
    }

    /// Enable RTCP receiver reports (every `rr_interval_ms`, 0 disables) and
    /// Generic NACK (when `nack`) on each session's RTP socket. Off by default:
    /// it is only useful once the publisher muxes RTCP onto the RTP port
    /// (RFC 5761), which a classic RTSP publisher does not do.
    pub fn with_rtcp(mut self, rr_interval_ms: u64, nack: bool) -> Self {
        self.recv.rtcp_rr_interval_ms = rr_interval_ms;
        self.recv.nack_enabled = nack;
        self
    }

    /// Serve `n` publishers concurrently, each on its own output pads. Takes
    /// effect on the next run; on a launch line the pad count is the number of
    /// linked pads.
    pub fn with_max_sessions(mut self, n: usize) -> Self {
        self.max_sessions = Some(n.max(1));
        self
    }

    /// Record each publisher's first AAC track, as ADTS, on the pad after its
    /// video pad.
    pub fn with_audio(mut self, audio: bool) -> Self {
        self.audio = audio;
        self
    }

    fn pads_per_publisher(&self) -> usize {
        1 + usize::from(self.audio)
    }

    fn publisher_formats(&self) -> Vec<IngestFormat> {
        let video = IngestFormat::Video(ingest_codec(&self.recv));
        if self.audio {
            alloc::vec![video, IngestFormat::Aac]
        } else {
            alloc::vec![video]
        }
    }

    /// Per-session inactivity timeout: a publisher that sends neither media nor a
    /// control request within it is torn down and its pad freed, so a peer that
    /// vanishes without closing its TCP connection cannot pin a pad. A zero
    /// duration disables reaping.
    pub fn with_session_timeout(mut self, timeout: Duration) -> Self {
        self.session_timeout_ns = match timeout.as_nanos() as u64 {
            0 => u64::MAX,
            ns => ns,
        };
        self
    }

    /// The TCP control port actually bound, once a listener exists (ephemeral
    /// lookup for tests).
    pub fn local_port(&self) -> Option<u16> {
        self.listener
            .as_ref()
            .and_then(|l| l.local_addr().ok())
            .map(|a| a.port())
    }

    /// Publishers that reached RECORD during the last `run`.
    pub fn sessions_served(&self) -> u64 {
        self.sessions_served
    }

    /// Publishers refused with 503 because every pad was busy.
    pub fn sessions_refused(&self) -> u64 {
        self.sessions_refused
    }

    /// The timeout to advertise in `Session: id;timeout=N`, in whole seconds.
    fn session_timeout_secs(&self) -> u32 {
        if self.session_timeout_ns == u64::MAX {
            return DEFAULT_SESSION_TIMEOUT_SECS;
        }
        self.session_timeout_ns.div_ceil(1_000_000_000).max(1) as u32
    }

    fn caps(&self) -> Caps {
        crate::rtspingest::ingest_caps(&self.recv, self.width, self.height, self.fps)
    }

    fn pad_config(&self) -> PadConfig {
        PadConfig {
            payload_type: self.payload_type,
            ssrc: self.ssrc,
            recv: self.recv.clone(),
            frame_limit: self.frame_limit,
            timeout_ns: self.session_timeout_ns,
            frame_period_ns: match self.fps {
                0 => 0,
                fps => 1_000_000_000 / fps as u64,
            },
            session_timeout_secs: self.session_timeout_secs(),
        }
    }
}

/// What a pad session needs from the element, copied once so the pad tasks do
/// not borrow the element itself.
#[derive(Debug)]
struct PadConfig {
    payload_type: u8,
    ssrc: u32,
    recv: RtpRecvConfig,
    frame_limit: u64,
    timeout_ns: u64,
    frame_period_ns: u64,
    session_timeout_secs: u32,
}

impl PadConfig {
    /// A fresh responder for one session on `port`. Each pad answers with its own
    /// SSRC, and so its own `Session` id, so concurrent publishers are never
    /// handed the same session identity.
    fn responder(&self, port: usize) -> RtspResponder {
        RtspResponder::new(
            // Ingest: the publisher describes the stream, not this end.
            sdp_h264(self.payload_type, None, None),
            // the handshake binds and advertises a port pair per announced track
            0,
            self.ssrc.wrapping_add(port as u32),
        )
        .accepting_codec(crate::rtspingest::ingest_codec(&self.recv))
        .with_session_timeout_secs(self.session_timeout_secs)
    }
}

// per pad, the access units it emitted and whether it pushed its own `Eos`
type PadOutcomes = Vec<(u64, bool)>;

/// Serve one publisher's pads, starting at `first_port`: take each publisher the
/// acceptor binds to them, drive the handshake, receive until that publisher
/// leaves, then free the pads for the next one. Returns, per pad, the access
/// units it emitted and whether it already pushed its `Eos` (the `num-buffers`
/// path does).
async fn publisher_worker(
    first_port: usize,
    formats: &[IngestFormat],
    mut inbox: mpsc::Receiver<tokio::net::TcpStream>,
    busy: &Cell<bool>,
    served: &Cell<u64>,
    cfg: &PadConfig,
    tx: g2g_core::runtime::Sender<(usize, PipelinePacket)>,
) -> Result<PadOutcomes, G2gError> {
    let mut sinks: Vec<PadSink> = (first_port..first_port + formats.len())
        .map(|port| PadSink::new(port, tx.clone()))
        .collect();
    drop(tx);
    if cfg.frame_limit == 0 {
        for sink in &mut sinks {
            crate::numbuffers::finished_at_zero_limit(cfg.frame_limit, sink).await?;
        }
        return Ok(alloc::vec![(0, true); formats.len()]);
    }
    let activity = Cell::new(g2g_core::metrics::monotonic_ns());
    // One tap per pad for its whole life: a publisher taking the pads over
    // continues the sequence numbering and the timeline of the previous one.
    let mut pads: Vec<IngestPad<'_>> = sinks
        .iter_mut()
        .zip(formats)
        .map(|(sink, &format)| IngestPad {
            format,
            tap: SessionTap::new(sink, &activity, cfg.frame_period_ns),
        })
        .collect();
    let outcomes = |pads: &[IngestPad<'_>]| {
        pads.iter()
            .map(|pad| (pad.tap.frames(), pad.finished(cfg.frame_limit)))
            .collect::<Vec<_>>()
    };

    loop {
        let Some(control) = inbox.recv().await else {
            return Ok(outcomes(&pads)); // no further publisher can arrive
        };
        activity.set(g2g_core::metrics::monotonic_ns());
        let session = tokio::select! {
            r = handshake(control, cfg.responder(first_port), &activity) => r?,
            _ = inactive_for(&activity, cfg.timeout_ns) => None,
        };
        let Some(session) = session else {
            busy.set(false); // gone before RECORD: the pad is free again
            continue;
        };
        served.set(served.get().saturating_add(1));

        let end = tokio::select! {
            r = receive_session(session, &mut pads, &cfg.recv, cfg.frame_limit, &activity) => r?,
            _ = inactive_for(&activity, cfg.timeout_ns) => SessionEnd::PeerGone,
        };
        match end {
            // `AccessUnitEmitter::push` already emitted the EOS of each pad at `num-buffers`.
            SessionEnd::Limit => return Ok(outcomes(&pads)),
            SessionEnd::PeerGone => {
                end_session(&mut pads);
                busy.set(false);
            }
        }
    }
}

/// Accept publishers and bind each to the first free pad, handing the control
/// connection to that pad's session (the handshake runs there, so a slow
/// publisher never holds up the next accept). With every pad busy there is
/// nowhere to put a new publisher, so it is answered with `503 Service
/// Unavailable` (RFC 2326 §7.1.1) instead of sitting in the backlog. Loops
/// forever; the caller runs it alongside the pads and drops it when they finish.
async fn accept_loop(
    listener: &tokio::net::TcpListener,
    dispatch: &[mpsc::Sender<tokio::net::TcpStream>],
    busy: &[Cell<bool>],
    refused: &Cell<u64>,
) {
    loop {
        let Ok((sock, _peer)) = listener.accept().await else {
            // A listener that cannot accept right now must not spin the task.
            tokio::time::sleep(Duration::from_millis(10)).await;
            continue;
        };
        // Refusals answer on a detached task: one must not delay the next accept,
        // and a publisher that connected as a pad freed up still gets an answer
        // rather than a silently closed connection.
        match busy.iter().position(|b| !b.get()) {
            Some(port) => {
                busy[port].set(true);
                if let Err(gone) = dispatch[port].send(sock).await {
                    // That pad has finished for good (num-buffers), so it stays
                    // marked busy and is never picked again.
                    refused.set(refused.get().saturating_add(1));
                    tokio::spawn(refuse_one(gone.0));
                }
            }
            None => {
                refused.set(refused.get().saturating_add(1));
                tokio::spawn(refuse_one(sock));
            }
        }
    }
}

static SRCN_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "address",
        PropKind::Str,
        "IP to listen for RTSP publishers on",
    )
    .with_default("0.0.0.0"),
    PropertySpec::new("port", PropKind::Uint, "RTSP TCP port to listen on")
        .with_default("8554")
        .with_range("0", "65535"),
    PropertySpec::new(
        "payload-type",
        PropKind::Uint,
        "RTP payload type advertised in the SDP (96..=127)",
    )
    .with_default("96")
    .with_range("0", "127"),
    PropertySpec::new("ssrc", PropKind::Uint, "base RTP synchronization source id"),
    PropertySpec::new(
        "width",
        PropKind::Uint,
        "declared frame width hint (the in-band SPS corrects it)",
    )
    .with_default("1280"),
    PropertySpec::new(
        "height",
        PropKind::Uint,
        "declared frame height hint (the in-band SPS corrects it)",
    )
    .with_default("720"),
    PropertySpec::new("framerate", PropKind::Uint, "declared frame rate hint, fps")
        .with_default("30"),
    crate::rtspingest::ENCODING_NAME_PROPERTY,
    PropertySpec::new(
        "num-buffers",
        PropKind::Int,
        "access units to emit per pad then EOS (-1 = until downstream stops)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
    PropertySpec::new(
        "max-sessions",
        PropKind::Uint,
        "publishers served at once, each on its own output pads (the linked pad count)",
    )
    .with_default("2")
    .with_range("1", "65535"),
    PropertySpec::new(
        "audio",
        PropKind::Bool,
        "record each publisher's first AAC track, as ADTS, on the pad after its video pad",
    )
    .with_default("false"),
    PropertySpec::new(
        "timeout",
        PropKind::Uint,
        "session timeout in seconds (a silent publisher is torn down; 0 = never)",
    )
    .with_default("60")
    .with_range("0", "604800"),
    PropertySpec::new(
        "jitter-latency",
        PropKind::Uint,
        "max time to hold a sequence gap before declaring it lost, milliseconds",
    )
    .with_default("50"),
    PropertySpec::new(
        "jitter-depth",
        PropKind::Uint,
        "max packets buffered for reorder (0 = in-order passthrough)",
    )
    .with_default("64"),
    PropertySpec::new(
        "rtcp-rr-interval",
        PropKind::Uint,
        "RTCP receiver-report interval in milliseconds (0 = off; needs rtcp-mux)",
    )
    .with_default("0"),
    PropertySpec::new(
        "nack",
        PropKind::Bool,
        "request retransmission of detected gaps via RTPFB Generic NACK",
    )
    .with_default("false"),
    PropertySpec::new(
        "rtx-payload-type",
        PropKind::Uint,
        "RFC 4588 RTX stream payload type (0 = off; set rtx-apt too)",
    )
    .with_default("0")
    .with_range("0", "127"),
    PropertySpec::new(
        "rtx-apt",
        PropKind::Uint,
        "original (associated) payload type RTX packets rebuild to",
    )
    .with_default("0")
    .with_range("0", "127"),
    PropertySpec::new(
        "fec-payload-type",
        PropKind::Uint,
        "RFC 5109 ULPFEC repair-stream payload type (0 = off)",
    )
    .with_default("0")
    .with_range("0", "127"),
    PropertySpec::new(
        "flexfec-payload-type",
        PropKind::Uint,
        "RFC 8627 FlexFEC repair-stream payload type (0 = off)",
    )
    .with_default("0")
    .with_range("0", "127"),
    crate::rtprecv::WAIT_FOR_KEYFRAME_PROPERTY,
    crate::rtprecv::REQUEST_KEYFRAME_PROPERTY,
];

impl MultiOutputSource for RtspServerSrcN {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn output_count(&self) -> usize {
        match self.max_sessions {
            Some(publishers) => publishers * self.pads_per_publisher(),
            None => self.outputs,
        }
    }

    /// Every video pad declares the same `encoding-name` hint caps; a downstream
    /// decoder corrects the geometry from the in-band SPS.
    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        if output >= self.output_count() {
            return Err(G2gError::CapsMismatch);
        }
        match self.publisher_formats()[output % self.pads_per_publisher()] {
            IngestFormat::Video(_) => Ok(self.caps()),
            IngestFormat::Aac => Ok(ADTS_AAC_CAPS),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SRCN_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Some(r) = crate::netprop::set_addr_prop(&mut self.rtsp_addr, "address", name, &value)
        {
            return r;
        }
        if let Some(r) = crate::rtprecv::set_recv_prop(&mut self.recv, name, &value) {
            return r;
        }
        match name {
            "payload-type" => {
                let pt = value.as_uint().ok_or(PropError::Type)?;
                if pt > 127 {
                    return Err(PropError::Value);
                }
                self.payload_type = pt as u8;
                Ok(())
            }
            "ssrc" => {
                self.ssrc = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "width" => {
                self.width = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "height" => {
                self.height = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "framerate" => {
                self.fps = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "encoding-name" => crate::rtspingest::set_encoding_name(&mut self.recv, &value),
            "num-buffers" => crate::numbuffers::set_num_buffers(&mut self.frame_limit, &value),
            "max-sessions" => {
                let n = value.as_uint().ok_or(PropError::Type)?;
                if n == 0 || n > u16::MAX as u64 {
                    return Err(PropError::Value);
                }
                self.max_sessions = Some(n as usize);
                Ok(())
            }
            "audio" => {
                self.audio = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "timeout" => {
                let secs = value.as_uint().ok_or(PropError::Type)?;
                if secs > 604_800 {
                    return Err(PropError::Value);
                }
                self.session_timeout_ns = match secs {
                    0 => u64::MAX, // never reap
                    s => s * 1_000_000_000,
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Some(v) = crate::netprop::get_addr_prop(&self.rtsp_addr, "address", name) {
            return Some(v);
        }
        if let Some(v) = crate::rtprecv::get_recv_prop(&self.recv, name) {
            return Some(v);
        }
        match name {
            "payload-type" => Some(PropValue::Uint(self.payload_type as u64)),
            "ssrc" => Some(PropValue::Uint(self.ssrc as u64)),
            "width" => Some(PropValue::Uint(self.width as u64)),
            "height" => Some(PropValue::Uint(self.height as u64)),
            "framerate" => Some(PropValue::Uint(self.fps as u64)),
            "encoding-name" => Some(PropValue::Str(self.recv.encoding.name().into())),
            "num-buffers" => Some(crate::numbuffers::get_num_buffers(self.frame_limit)),
            "max-sessions" => Some(PropValue::Uint(
                (self.output_count() / self.pads_per_publisher()) as u64,
            )),
            "audio" => Some(PropValue::Bool(self.audio)),
            "timeout" => Some(PropValue::Uint(match self.session_timeout_ns {
                u64::MAX => 0,
                _ => self.session_timeout_secs() as u64,
            })),
            _ => None,
        }
    }

    /// Bind the listener, then run one accept loop and N pad sessions
    /// concurrently, draining what the pads receive into their outputs. Ends when
    /// every pad has hit `num-buffers` (or downstream stops), emitting the `Eos`
    /// each pad has not already sent.
    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let std_listener = match self.listener.take() {
                Some(l) => l,
                None => StdTcpListener::bind(self.rtsp_addr).map_err(io_err)?,
            };
            std_listener.set_nonblocking(true).map_err(io_err)?;
            let listener = tokio::net::TcpListener::from_std(std_listener).map_err(io_err)?;

            let outputs = self.output_count();
            let pads_per_publisher = self.pads_per_publisher();
            if !outputs.is_multiple_of(pads_per_publisher) {
                g2g_error!(
                    Target::category("rtspserversrcn"),
                    "audio=true takes two pads per publisher, but {outputs} pads are linked"
                );
                return Err(G2gError::NotConfigured);
            }
            let publishers = outputs / pads_per_publisher;
            let formats = self.publisher_formats();
            let cfg = self.pad_config();
            let busy: Vec<Cell<bool>> = (0..publishers).map(|_| Cell::new(false)).collect();
            let served = Cell::new(0u64);
            let refused = Cell::new(0u64);

            // Per-pad handover of an accepted control connection, and the shared
            // channel every pad pushes its packets onto.
            let mut dispatch = Vec::with_capacity(publishers);
            let mut inboxes = Vec::with_capacity(publishers);
            for _ in 0..publishers {
                let (tx, rx) = mpsc::channel::<tokio::net::TcpStream>(1);
                dispatch.push(tx);
                inboxes.push(rx);
            }
            let (tx, rx) =
                g2g_core::runtime::bounded::<(usize, PipelinePacket)>(PAD_QUEUE * outputs);
            let workers: Vec<BoxFuture<'_, Result<PadOutcomes, G2gError>>> = inboxes
                .into_iter()
                .enumerate()
                .map(|(publisher, inbox)| {
                    Box::pin(publisher_worker(
                        publisher * pads_per_publisher,
                        &formats,
                        inbox,
                        &busy[publisher],
                        &served,
                        &cfg,
                        tx.clone(),
                    )) as BoxFuture<'_, _>
                })
                .collect();
            // Only the pads hold senders now, so the drain ends when they do.
            drop(tx);

            // The pads are the producers; the acceptor only feeds them, so it is
            // dropped once every pad has finished.
            let producers = async {
                tokio::select! {
                    outcomes = join_all(workers) => outcomes,
                    _ = accept_loop(&listener, &dispatch, &busy, &refused) => {
                        unreachable!("the acceptor never ends")
                    }
                }
            };

            let result: Result<u64, G2gError> = async {
                let outcomes = forward_pads(producers, &rx, out).await?;
                let mut pad_outcomes = Vec::with_capacity(outputs);
                for outcome in outcomes {
                    pad_outcomes.extend(outcome?);
                }
                let mut frames = 0u64;
                for (port, (pad_frames, eos_sent)) in pad_outcomes.into_iter().enumerate() {
                    frames = frames.saturating_add(pad_frames);
                    if !eos_sent {
                        out.push_to(port, PipelinePacket::Eos).await?;
                    }
                }
                Ok(frames)
            }
            .await;

            self.sessions_served = served.get();
            self.sessions_refused = refused.get();
            result
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::VideoCodec;

    fn src(pads: usize) -> RtspServerSrcN {
        RtspServerSrcN::new("0.0.0.0:8554".parse().unwrap(), pads)
    }

    #[test]
    fn pads_are_output_ports() {
        let s = src(3);
        assert_eq!(s.output_count(), 3);
        assert!(matches!(
            s.output_caps(2),
            Ok(Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            })
        ));
        assert!(s.output_caps(3).is_err());
        // A pad count of zero would leave the element with no output at all.
        assert_eq!(src(0).output_count(), 1);
    }

    #[test]
    fn audio_gives_each_publisher_a_video_pad_then_an_aac_pad() {
        let s = src(4).with_audio(true);
        let caps: Vec<Caps> = (0..s.output_count())
            .map(|port| s.output_caps(port).unwrap())
            .collect();
        assert_eq!(caps, [s.caps(), ADTS_AAC_CAPS, s.caps(), ADTS_AAC_CAPS]);
        assert_eq!(s.get_property("max-sessions"), Some(PropValue::Uint(2)));
    }

    #[test]
    fn max_sessions_counts_publishers_whichever_property_comes_first() {
        const PUBLISHERS: u64 = 3;
        let mut audio_last = src(1);
        audio_last
            .set_property("max-sessions", PropValue::Uint(PUBLISHERS))
            .unwrap();
        audio_last
            .set_property("audio", PropValue::Bool(true))
            .unwrap();
        let mut audio_first = src(1);
        audio_first
            .set_property("audio", PropValue::Bool(true))
            .unwrap();
        audio_first
            .set_property("max-sessions", PropValue::Uint(PUBLISHERS))
            .unwrap();
        assert_eq!(audio_last.output_count(), 2 * PUBLISHERS as usize);
        assert_eq!(audio_first.output_count(), 2 * PUBLISHERS as usize);
    }

    #[test]
    fn each_pad_gets_its_own_session_identity() {
        let cfg = src(2).with_rtp(96, 0x1234_5678).pad_config();
        assert_ne!(cfg.responder(0).session_id(), cfg.responder(1).session_id());
    }
}
