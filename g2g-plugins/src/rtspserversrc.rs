// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! RTSP server source (RtspServerSrc, `rtsp-server` feature): hosts an RTSP
//! endpoint that a *publisher* pushes media into (the ANNOUNCE / SETUP / RECORD
//! direction, the inverse of [`RtspServerSink`](crate::rtspserversink) which
//! serves players). The sans-IO [`rtspserver::RtspResponder`](crate::rtspserver)
//! already speaks ANNOUNCE/RECORD; this element wraps it in the tokio TCP
//! control channel and the UDP RTP receive transport, reusing the depayloader
//! ([`RtpH264Depayloader`](crate::rtpdepay)) the UDP source uses.
//!
//! Shape: a contribution endpoint (e.g. an encoder/camera that publishes to an
//! RTSP server with `ffmpeg -f rtsp -rtsp_transport udp ...`). The TCP listener
//! is bound in `configure_pipeline`; `run` accepts a publisher, drives the
//! handshake to RECORD, then depayloads the RTP it receives into H.264 / H.265
//! access units emitted downstream.
//!
//! Audio: with an audio pad (`with_audio_pad`, or a launch line linking a second
//! pad) the publisher's first AAC track is recorded onto pad 1 as ADTS. Video and
//! audio PTS share one timeline, placed from the publisher's RTCP sender reports
//! (a track with no report before its first frame is placed by arrival time).
//! Announced tracks no pad records are SETUP and their packets dropped.
//!
//! Transport: both unicast UDP (`RTP/AVP;client_port=`, the jitter/RTCP/FEC
//! receive path shared with `UdpSrc`) and TCP-interleaved (`RTP/AVP/TCP;
//! interleaved=`, RFC 2326 §10.12: RTP rides the control connection as `$`-framed
//! binary, in order, so no jitter buffer is needed), chosen by what the publisher
//! negotiates in SETUP. What `ffmpeg -rtsp_transport tcp` uses.
//!
//! Multi-client: the element has one output pad, so one publisher records
//! at a time and sessions are *sequential*. When a publisher disconnects, tears
//! down, or falls silent past `timeout`, its session state (control connection,
//! RTP port, interleaved channel, depayloader) is dropped and the element goes
//! back to listening, so a reconnecting encoder resumes on the same pad without
//! restarting the graph; downstream PTS continues forward across the handover
//! rather than jumping back to zero. A publisher that connects while another is
//! recording is refused with `503 Service Unavailable` instead of being queued in
//! the accept backlog. `max-sessions` bounds how many publishers are served
//! before EOS (the default, 0, keeps serving). To ingest several publishers at
//! the same time, one per pad, use
//! [`RtspServerSrcN`](crate::rtspserversrcn::RtspServerSrcN); the per-session
//! machinery both drive lives in `rtspingest`.

use core::cell::Cell;
use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::vec::Vec;

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, LatencyReport,
    MultiOutputSink, MultiOutputSource, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::oneport::OnePortSink;
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
const VIDEO_PORT: usize = 0;
const AUDIO_PORT: usize = 1;

/// # Example
///
/// ```no_run
/// use g2g_plugins::rtspserversrc::RtspServerSrc;
///
/// let src = RtspServerSrc::new("0.0.0.0:8554".parse().unwrap())
///     .with_video_size(1920, 1080)
///     .with_framerate(30);
/// ```
#[derive(Debug)]
pub struct RtspServerSrc {
    rtsp_addr: SocketAddr,
    payload_type: u8,
    ssrc: u32,
    width: u32,
    height: u32,
    fps: u32,
    /// `u64::MAX` means run until the connection drops / downstream shuts down;
    /// otherwise stop after this many access units and emit EOS (the test /
    /// bounded path).
    frame_limit: u64,
    /// Receive-path tuning (jitter reorder + optional RTCP/NACK), shared with
    /// [`UdpSrc`](crate::udpsrc) via [`crate::rtprecv`]. RTCP defaults off: a
    /// classic RTSP publisher puts RTCP on a separate port (not muxed onto the
    /// RTP socket), so receiver-report / NACK feedback needs `with_rtcp` plus a
    /// negotiated `rtcp-mux` (a follow-up) to actually reach the sender.
    recv: RtpRecvConfig,
    /// Publishers to serve (sequentially) before emitting EOS and stopping.
    /// 0 keeps listening for the next publisher forever.
    max_sessions: u64,
    /// Per-session inactivity budget: a publisher that sends neither media nor a
    /// control request within it is torn down and the element goes back to
    /// listening. `u64::MAX` disables reaping.
    session_timeout_ns: u64,
    listener: Option<StdTcpListener>,
    configured: bool,
    audio_pad: bool,
    /// Publishers that reached RECORD, and connections refused (503) because one
    /// was already recording. Readable once `run` returns.
    sessions_served: u64,
    sessions_refused: u64,
}

impl RtspServerSrc {
    /// Listen for an RTSP publisher on `rtsp_addr` (e.g. `0.0.0.0:8554`).
    pub fn new(rtsp_addr: SocketAddr) -> Self {
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
            },
            max_sessions: 0,
            session_timeout_ns: DEFAULT_SESSION_TIMEOUT_SECS as u64 * 1_000_000_000,
            listener: None,
            configured: false,
            audio_pad: false,
            sessions_served: 0,
            sessions_refused: 0,
        }
    }

    /// Use an already-bound listener (so a test can pick an ephemeral port).
    pub fn from_listener(listener: StdTcpListener) -> Result<Self, G2gError> {
        let addr = listener.local_addr().map_err(io_err)?;
        Ok(Self {
            listener: Some(listener),
            configured: true,
            ..Self::new(addr)
        })
    }

    /// Set the RTP payload type and SSRC negotiated in SETUP.
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

    /// Stop after `n` access units and emit EOS (0 emits EOS without
    /// listening). Without this the source runs until the publisher disconnects
    /// (RTP has no in-band end marker).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Tune the receive-side jitter buffer: hold a gap up to `max_hold_ms`
    /// before declaring it lost, buffering at most `max_depth` packets. A
    /// `max_depth` of 0 disables reordering (in-order passthrough). Default is
    /// [`JitterConfig::default`] (50 ms / 64 packets), so a lossy / reordering
    /// link is tolerated even without RTCP retransmission.
    pub fn with_jitter(mut self, max_hold_ms: u64, max_depth: usize) -> Self {
        self.recv.jitter = JitterConfig::new(max_hold_ms, max_depth);
        self
    }

    /// Enable RTCP receiver reports (every `rr_interval_ms`, 0 disables) and
    /// Generic NACK (when `nack`) on the RTP socket. Off by default: it is only
    /// useful once the publisher muxes RTCP onto the RTP port (RFC 5761), which
    /// a classic RTSP publisher does not do without a negotiated `rtcp-mux`.
    pub fn with_rtcp(mut self, rr_interval_ms: u64, nack: bool) -> Self {
        self.recv.rtcp_rr_interval_ms = rr_interval_ms;
        self.recv.nack_enabled = nack;
        self
    }

    /// Serve `n` publishers (one at a time, in sequence) then emit EOS and stop.
    /// 0 (the default) keeps listening for the next publisher forever, the live
    /// contribution-endpoint shape; 1 is the one-shot "record until the publisher
    /// leaves" behaviour a file-writing graph wants.
    pub fn with_max_sessions(mut self, n: u64) -> Self {
        self.max_sessions = n;
        self
    }

    /// Record the publisher's first AAC track onto a second output pad, as ADTS.
    /// Only the multi-output form ([`MultiOutputSource`]) has that pad.
    pub fn with_audio_pad(mut self, on: bool) -> Self {
        self.audio_pad = on;
        self
    }

    /// Per-session inactivity timeout: a publisher that sends neither media nor a
    /// control request within it is torn down and the element returns to
    /// listening, so a peer that vanishes without closing its TCP connection
    /// cannot stall the graph. A zero duration disables reaping.
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

    /// Publishers refused with 503 because another was already recording.
    pub fn sessions_refused(&self) -> u64 {
        self.sessions_refused
    }

    /// The timeout to advertise in `Session: id;timeout=N`, in whole seconds.
    /// With reaping disabled the RFC default is advertised, so a publisher keeps
    /// a standard keepalive cadence.
    fn session_timeout_secs(&self) -> u32 {
        if self.session_timeout_ns == u64::MAX {
            return DEFAULT_SESSION_TIMEOUT_SECS;
        }
        self.session_timeout_ns.div_ceil(1_000_000_000).max(1) as u32
    }

    fn caps(&self) -> Caps {
        crate::rtspingest::ingest_caps(&self.recv, self.width, self.height, self.fps)
    }

    fn responder(&self) -> RtspResponder {
        // An ingest server: the publisher describes the stream, not this end, so
        // there are no parameter sets of its own to offer.
        RtspResponder::new(
            sdp_h264(self.payload_type, None, None),
            // the handshake binds and advertises a port pair per announced track
            0,
            self.ssrc,
        )
        .accepting_codec(ingest_codec(&self.recv))
        .with_session_timeout_secs(self.session_timeout_secs())
    }

    /// Serve publishers one at a time onto the ports of `out` (video, then AAC)
    /// until `max-sessions` of them have recorded (0 = forever) or `num-buffers`
    /// access units have been emitted on every recorded track. Each session owns
    /// its sockets and responder, dropped when it ends, and anyone who connects
    /// mid-session is refused with a 503.
    async fn serve(&mut self, out: &mut dyn MultiOutputSink) -> Result<u64, G2gError> {
        let ports = out.port_count();
        if crate::numbuffers::finished_at_zero_limit_multi(self.frame_limit, ports, out).await? {
            return Ok(0);
        }
        let std_listener = match self.listener.take() {
            Some(listener) => listener,
            None => StdTcpListener::bind(self.rtsp_addr).map_err(io_err)?,
        };
        std_listener.set_nonblocking(true).map_err(io_err)?;
        let listener = tokio::net::TcpListener::from_std(std_listener).map_err(io_err)?;

        let this = &*self;
        let frame_limit = this.frame_limit;
        let timeout_ns = this.session_timeout_ns;
        let frame_period_ns = match this.fps {
            0 => 0,
            fps => 1_000_000_000 / fps as u64,
        };
        let formats = [
            IngestFormat::Video(ingest_codec(&this.recv)),
            IngestFormat::Aac,
        ];

        let activity = Cell::new(g2g_core::metrics::monotonic_ns());
        let refused = Cell::new(0u64);
        let served = Cell::new(0u64);
        let (tx, rx) = g2g_core::runtime::bounded::<(usize, PipelinePacket)>(PAD_QUEUE * ports);
        let sinks: Vec<PadSink> = (0..ports)
            .map(|port| PadSink::new(port, tx.clone()))
            .collect();
        drop(tx);

        let sessions = async {
            // Moved in, so the channel closes once the sessions end.
            let mut sinks = sinks;
            // One tap per pad for the whole run: a re-publish continues the
            // sequence numbering and the timeline of the previous publisher.
            let mut pads: Vec<IngestPad<'_>> = sinks
                .iter_mut()
                .zip(formats)
                .map(|(sink, format)| IngestPad {
                    format,
                    tap: SessionTap::new(sink, &activity, frame_period_ns),
                })
                .collect();
            loop {
                let (control, _peer) = listener.accept().await.map_err(io_err)?;
                activity.set(g2g_core::metrics::monotonic_ns());
                // Anyone connecting during the handshake is refused too, so a
                // second publisher never waits in the backlog.
                let session = tokio::select! {
                    r = handshake(control, this.responder(), &activity) => r?,
                    _ = inactive_for(&activity, timeout_ns) => None,
                    _ = refuse_extras(&listener, &refused) => unreachable!("refusal never ends"),
                };
                let Some(session) = session else {
                    continue; // gone before RECORD: listen for the next one
                };
                let end = tokio::select! {
                    r = receive_session(session, &mut pads, &this.recv, frame_limit, &activity) => r?,
                    _ = inactive_for(&activity, timeout_ns) => SessionEnd::PeerGone,
                    _ = refuse_extras(&listener, &refused) => unreachable!("refusal never ends"),
                };
                served.set(served.get().saturating_add(1));
                match end {
                    SessionEnd::Limit => break,
                    SessionEnd::PeerGone => {
                        end_session(&mut pads);
                        if this.max_sessions != 0 && served.get() >= this.max_sessions {
                            break;
                        }
                    }
                }
            }
            Ok::<_, G2gError>(
                pads.iter()
                    .map(|pad| (pad.tap.frames(), pad.finished(frame_limit)))
                    .collect::<Vec<_>>(),
            )
        };

        let result: Result<u64, G2gError> = async {
            let pads = forward_pads(sessions, &rx, out).await??;
            let mut frames = 0u64;
            for (port, (pad_frames, eos_sent)) in pads.into_iter().enumerate() {
                frames = frames.saturating_add(pad_frames);
                // `AccessUnitEmitter::push` already emitted the EOS of a pad at `num-buffers`.
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
    }
}

/// Refuse every publisher that connects while another is recording: answer its
/// first request with `503 Service Unavailable` (RFC 2326 §7.1.1) and close, so a
/// second ffmpeg fails fast instead of sitting in the accept backlog. Loops
/// forever; the caller runs it as a `select!` arm alongside the live session.
async fn refuse_extras(listener: &tokio::net::TcpListener, refused: &Cell<u64>) {
    loop {
        let Ok((sock, _peer)) = listener.accept().await else {
            // A listener that cannot accept right now must not spin the task.
            tokio::time::sleep(Duration::from_millis(10)).await;
            continue;
        };
        refused.set(refused.get().saturating_add(1));
        // Answer on a detached task: this loop is cancelled the instant the live
        // session ends, and a publisher that connected in that same instant must
        // still get its 503 instead of a silently closed connection.
        tokio::spawn(refuse_one(sock));
    }
}

static SRC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "address",
        PropKind::Str,
        "IP to listen for an RTSP publisher on",
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
    PropertySpec::new("ssrc", PropKind::Uint, "RTP synchronization source id"),
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
        "access units to emit then EOS (-1 = until the publisher disconnects)",
    )
    .with_default("-1")
    .with_range("-1", "9223372036854775807"),
    PropertySpec::new(
        "max-sessions",
        PropKind::Uint,
        "publishers to serve in sequence then EOS (0 = keep listening)",
    )
    .with_default("0"),
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
];

impl SourceLoop for RtspServerSrc {
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

    /// Produces the declared H.264 hint caps (no I/O at negotiation; the TCP
    /// listener binds in `configure_pipeline`). A downstream decoder corrects the
    /// real geometry from the in-band SPS via a mid-stream `CapsChanged`.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if self.listener.is_none() {
            self.listener = Some(StdTcpListener::bind(self.rtsp_addr).map_err(io_err)?);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "RTSP server source",
            "Source/Network",
            "Hosts an RTSP endpoint a publisher pushes H.264 / H.265 and AAC into (ANNOUNCE/RECORD)",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SRC_PROPS
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
                self.max_sessions = value.as_uint().ok_or(PropError::Type)?;
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
            "max-sessions" => Some(PropValue::Uint(self.max_sessions)),
            "timeout" => Some(PropValue::Uint(match self.session_timeout_ns {
                u64::MAX => 0,
                _ => self.session_timeout_secs() as u64,
            })),
            _ => None,
        }
    }

    /// Live source: contributes one frame period so the sink keeps a frame in
    /// hand and never runs dry waiting on the network.
    fn latency(&self) -> LatencyReport {
        let period_ns = if self.fps > 0 {
            1_000_000_000 / self.fps as u64
        } else {
            0
        };
        LatencyReport::live(period_ns, None)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            self.serve(&mut OnePortSink(out)).await
        })
    }
}

impl MultiOutputSource for RtspServerSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    fn output_count(&self) -> usize {
        1 + usize::from(self.audio_pad)
    }

    fn output_caps(&self, output: usize) -> Result<Caps, G2gError> {
        match output {
            VIDEO_PORT => Ok(self.caps()),
            AUDIO_PORT if self.audio_pad => Ok(ADTS_AAC_CAPS),
            _ => Err(G2gError::CapsMismatch),
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        SRC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        SourceLoop::set_property(self, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        SourceLoop::get_property(self, name)
    }

    fn latency(&self) -> LatencyReport {
        SourceLoop::latency(self)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
        Box::pin(self.serve(out))
    }
}

impl PadTemplates for RtspServerSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        let ingested = [VideoCodec::H264, VideoCodec::H265].map(|codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        });
        Vec::from([
            PadTemplate::source(CapsSet::from_alternatives(Vec::from(ingested))),
            PadTemplate::source(CapsSet::one(ADTS_AAC_CAPS)),
        ])
    }
}
