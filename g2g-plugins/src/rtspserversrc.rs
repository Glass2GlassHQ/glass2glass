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
//! handshake to RECORD, then depayloads the RTP it receives into H.264 access
//! units emitted downstream.
//!
//! Transport: both unicast UDP (`RTP/AVP;client_port=`, the jitter/RTCP/FEC
//! receive path shared with `UdpSrc`) and TCP-interleaved (`RTP/AVP/TCP;
//! interleaved=`, RFC 2326 §10.12: RTP rides the control connection as `$`-framed
//! binary, in order, so no jitter buffer is needed), chosen by what the publisher
//! negotiates in SETUP. What `ffmpeg -rtsp_transport tcp` uses.
//!
//! Multi-client (M834): the element has one output pad, so one publisher records
//! at a time and sessions are *sequential*. When a publisher disconnects, tears
//! down, or falls silent past `timeout`, its session state (control connection,
//! RTP port, interleaved channel, depayloader) is dropped and the element goes
//! back to listening, so a reconnecting encoder resumes on the same pad without
//! restarting the graph; downstream PTS continues forward across the handover
//! rather than jumping back to zero. A publisher that connects while another is
//! recording is refused with `503 Service Unavailable` instead of being queued in
//! the accept backlog. `max-sessions` bounds how many publishers are served
//! before EOS (the default, 0, keeps serving).

use core::cell::Cell;
use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use std::net::{SocketAddr, TcpListener as StdTcpListener};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError, LatencyReport,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, PushOutcome, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::rtpdepay::RtpH264Depayloader;
use crate::rtpjitter::JitterConfig;
use crate::rtprecv::{push_access_unit, RtpRecvConfig};
use crate::rtspserver::{
    sdp_h264, RtspEvent, RtspRequest, RtspResponder, DEFAULT_SESSION_TIMEOUT_SECS,
};

/// Default dynamic RTP payload type for H.264.
const DEFAULT_PAYLOAD_TYPE: u8 = 96;
/// Declared geometry hint (SPS is authoritative; a downstream decoder corrects).
const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const DEFAULT_FPS: u32 = 30;
/// TCP read buffer for RTSP control requests.
const CTRL_BUF: usize = 8192;
/// Cap on buffered-but-unparsed control bytes, as in the serving sink: a client
/// dripping a never-terminating request (or an oversized Content-Length) is
/// dropped instead of growing the buffer without bound.
const MAX_PENDING: usize = 64 * 1024;
/// How long a refused publisher is given to send the request its 503 answers.
const REFUSE_READ_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug)]
pub struct RtspServerSrc {
    rtsp_addr: SocketAddr,
    payload_type: u8,
    ssrc: u32,
    width: u32,
    height: u32,
    fps: u32,
    /// 0 means run until the connection drops / downstream shuts down; otherwise
    /// stop after this many access units and emit EOS (the test / bounded path).
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
            frame_limit: 0,
            recv: RtpRecvConfig {
                jitter: JitterConfig::default(),
                // Off until a separate RTCP port / rtcp-mux is negotiated.
                rtcp_rr_interval_ms: 0,
                nack_enabled: false,
                rtx: None,
                fec_pt: None,
                flexfec_pt: None,
                declared_caps: None,
            },
            max_sessions: 0,
            session_timeout_ns: DEFAULT_SESSION_TIMEOUT_SECS as u64 * 1_000_000_000,
            listener: None,
            configured: false,
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

    /// Stop after `n` access units and emit EOS. Without this the source runs
    /// until the publisher disconnects (RTP has no in-band end marker).
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
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.fps << 16),
        }
    }

    fn responder(&self, server_rtp_port: u16) -> RtspResponder {
        RtspResponder::new(sdp_h264(self.payload_type), server_rtp_port, self.ssrc)
            .with_session_timeout_secs(self.session_timeout_secs())
    }
}

/// Drive one accepted publisher's RTSP handshake to RECORD: bind this session's
/// RTP UDP receive socket (its port is advertised in a UDP SETUP), run
/// OPTIONS/ANNOUNCE/SETUP/RECORD over the TCP control channel, and return the
/// negotiated [`RecordTransport`] once the publisher has issued RECORD. Either
/// way the control stream is kept (never dropped): a real RTSP publisher (ffmpeg)
/// holds the control connection open while it records and treats the server
/// closing it as a fatal "broken pipe", so dropping it would abort the publish.
/// For TCP-interleaved, that same stream *is* the RTP transport, and any bytes
/// already buffered past RECORD are handed on as `leftover` so a pipelined first
/// frame is not lost.
///
/// `Ok(None)` means this publisher went away before RECORD (closed, tore down, or
/// overflowed the control buffer): the caller drops it and listens for the next
/// one rather than failing the graph.
async fn handshake(
    mut control: tokio::net::TcpStream,
    mut responder: RtspResponder,
    rtp_socket: tokio::net::UdpSocket,
    activity: &Cell<u64>,
) -> Result<Option<RecordTransport>, G2gError> {
    // Set when SETUP negotiated TCP-interleaved: the RTP channel to demux the
    // `$`-framed control-connection binary on.
    let mut interleaved_rtp_channel: Option<u8> = None;
    let mut pending: Vec<u8> = Vec::new();
    let mut buf = [0u8; CTRL_BUF];

    loop {
        let Ok(n) = control.read(&mut buf).await else {
            return Ok(None);
        };
        if n == 0 {
            return Ok(None); // publisher closed before RECORD
        }
        activity.set(g2g_core::metrics::monotonic_ns());
        pending.extend_from_slice(&buf[..n]);

        while let Some((req, consumed)) = RtspRequest::parse(&pending) {
            pending.drain(..consumed);
            let (response, event) = responder.handle_request(&req);
            if control.write_all(&response).await.is_err() {
                return Ok(None);
            }
            match event {
                RtspEvent::SetupInterleaved { rtp_channel, .. } => {
                    interleaved_rtp_channel = Some(rtp_channel);
                }
                RtspEvent::Record => {
                    // Media now flows; hand the control stream back so the caller
                    // keeps it open for the session (interleaved also receives its
                    // RTP on it). `pending` holds any bytes read past RECORD (a
                    // pipelined first interleaved frame).
                    return Ok(Some(match interleaved_rtp_channel {
                        Some(rtp_channel) => RecordTransport::Interleaved {
                            control,
                            responder,
                            rtp_channel,
                            leftover: pending,
                        },
                        None => RecordTransport::Udp {
                            rtp_socket,
                            control,
                            responder,
                        },
                    }));
                }
                RtspEvent::Teardown => return Ok(None),
                _ => {}
            }
        }
        if pending.len() > MAX_PENDING {
            return Ok(None);
        }
    }
}

/// The transport a publisher negotiated by RECORD time. Both keep the control
/// `TcpStream` (a UDP publisher needs it open; an interleaved one receives its RTP
/// on it) and the session's responder, so keepalives and a mid-RECORD TEARDOWN
/// are answered from the same protocol state.
#[derive(Debug)]
enum RecordTransport {
    /// Unicast UDP: RTP arrives on `rtp_socket`; `control` is held open.
    Udp {
        rtp_socket: tokio::net::UdpSocket,
        control: tokio::net::TcpStream,
        responder: RtspResponder,
    },
    /// TCP-interleaved (RFC 2326 §10.12): RTP arrives on `control` as `$`-framed
    /// binary on `rtp_channel`; `leftover` is any binary already buffered.
    Interleaved {
        control: tokio::net::TcpStream,
        responder: RtspResponder,
        rtp_channel: u8,
        leftover: Vec<u8>,
    },
}

/// Why a publisher's session stopped.
#[derive(Debug, PartialEq, Eq)]
enum SessionEnd {
    /// `num-buffers` access units were emitted (EOS is already pushed).
    Limit,
    /// The publisher disconnected, tore down, or fell silent past the timeout.
    PeerGone,
}

/// Downstream wrapper for the whole run, spanning every publisher session: it
/// counts the access units emitted so far (so the next session continues the
/// sequence numbering), stamps liveness for the inactivity watchdog, and shifts
/// each session's timestamps past the previous one, so a re-publish continues
/// forward instead of restarting downstream time at zero.
struct SessionTap<'a> {
    out: &'a mut dyn OutputSink,
    activity: &'a Cell<u64>,
    frames: u64,
    pts_offset_ns: u64,
    last_pts_ns: u64,
    frame_period_ns: u64,
}

impl core::fmt::Debug for SessionTap<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SessionTap")
            .field("frames", &self.frames)
            .field("pts_offset_ns", &self.pts_offset_ns)
            .finish()
    }
}

impl<'a> SessionTap<'a> {
    fn new(out: &'a mut dyn OutputSink, activity: &'a Cell<u64>, frame_period_ns: u64) -> Self {
        Self {
            out,
            activity,
            frames: 0,
            pts_offset_ns: 0,
            last_pts_ns: 0,
            frame_period_ns,
        }
    }

    /// Access units pushed downstream so far, across every session.
    fn frames(&self) -> u64 {
        self.frames
    }

    /// Note liveness on a session that is receiving bytes but has not completed
    /// an access unit yet, so the inactivity watchdog does not reap it.
    fn mark_activity(&self) {
        self.activity.set(g2g_core::metrics::monotonic_ns());
    }

    /// Close out a session: the next publisher's timestamps start one frame past
    /// the last one emitted.
    fn end_session(&mut self) {
        self.pts_offset_ns = self.last_pts_ns.saturating_add(self.frame_period_ns);
    }
}

impl OutputSink for SessionTap<'_> {
    fn push<'b>(
        &'b mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'b>> {
        let mut packet = packet;
        if let PipelinePacket::DataFrame(frame) = &mut packet {
            let t = &mut frame.timing;
            t.pts_ns = t.pts_ns.saturating_add(self.pts_offset_ns);
            t.dts_ns = t.dts_ns.saturating_add(self.pts_offset_ns);
            t.capture_ns = t.capture_ns.saturating_add(self.pts_offset_ns);
            self.last_pts_ns = t.pts_ns;
            self.frames += 1;
            self.activity.set(g2g_core::metrics::monotonic_ns());
        }
        self.out.push(packet)
    }
}

/// Wait until the session has been silent for `timeout_ns`. Never completes when
/// reaping is disabled (`u64::MAX`), so it can sit in a `select!` unconditionally.
async fn inactive_for(activity: &Cell<u64>, timeout_ns: u64) {
    if timeout_ns == u64::MAX {
        core::future::pending::<()>().await;
    }
    loop {
        let idle = g2g_core::metrics::monotonic_ns().saturating_sub(activity.get());
        let Some(remaining) = timeout_ns.checked_sub(idle) else {
            return;
        };
        tokio::time::sleep(Duration::from_nanos(remaining.max(1))).await;
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

/// Answer one refused publisher: echo the `CSeq` it asked with in a 503, then
/// close.
async fn refuse_one(mut sock: tokio::net::TcpStream) {
    let cseq = tokio::time::timeout(REFUSE_READ_TIMEOUT, first_request_cseq(&mut sock))
        .await
        .ok()
        .flatten()
        .unwrap_or(0);
    let busy = format!("RTSP/1.0 503 Service Unavailable\r\nCSeq: {cseq}\r\nServer: g2g\r\n\r\n");
    let _ = sock.write_all(busy.as_bytes()).await;
}

/// Read until one complete RTSP request has arrived and report its `CSeq`, so a
/// refusal echoes the sequence number the client expects.
async fn first_request_cseq(sock: &mut tokio::net::TcpStream) -> Option<u32> {
    let mut pending: Vec<u8> = Vec::new();
    let mut buf = [0u8; CTRL_BUF];
    loop {
        let n = sock.read(&mut buf).await.ok()?;
        if n == 0 {
            return None;
        }
        pending.extend_from_slice(&buf[..n]);
        if let Some((req, _)) = RtspRequest::parse(&pending) {
            return Some(req.cseq);
        }
        if pending.len() > MAX_PENDING {
            return None;
        }
    }
}

/// Watch a UDP publisher's control channel for the length of its session:
/// answer keepalives (OPTIONS / GET_PARAMETER) and return once the publisher
/// tears down or closes the connection. Without this the UDP receive loop, which
/// only ever sees datagrams, would keep waiting for RTP from a publisher that has
/// already left.
async fn watch_control(
    control: &mut tokio::net::TcpStream,
    responder: &mut RtspResponder,
    activity: &Cell<u64>,
) {
    let mut pending: Vec<u8> = Vec::new();
    let mut buf = [0u8; CTRL_BUF];
    loop {
        let Ok(n) = control.read(&mut buf).await else {
            return;
        };
        if n == 0 {
            return;
        }
        activity.set(g2g_core::metrics::monotonic_ns());
        pending.extend_from_slice(&buf[..n]);
        while let Some((req, consumed)) = RtspRequest::parse(&pending) {
            pending.drain(..consumed);
            let (response, event) = responder.handle_request(&req);
            if control.write_all(&response).await.is_err() || event == RtspEvent::Teardown {
                return;
            }
        }
        if pending.len() > MAX_PENDING {
            return;
        }
    }
}

/// One parsed item from an interleaved control stream (RFC 2326 §10.12).
#[derive(Debug, PartialEq, Eq)]
enum Interleaved {
    /// A `$`-framed binary packet on `channel`; its payload is `buf[start..end]`,
    /// and `consumed` bytes (header + payload) form the whole frame.
    Binary {
        channel: u8,
        start: usize,
        end: usize,
        consumed: usize,
    },
    /// An embedded RTSP request occupying `consumed` bytes, interleaved between
    /// binary frames.
    Rtsp { consumed: usize },
    /// Not enough bytes buffered yet for a complete item.
    NeedMore,
}

/// Parse the next interleaved item at the front of `buf`. A `$` (0x24) begins a
/// 4-byte binary header (`$`, channel, 2-byte big-endian length) then that many
/// payload bytes; anything else is an interleaved RTSP request.
fn next_interleaved(buf: &[u8]) -> Interleaved {
    match buf.first() {
        None => Interleaved::NeedMore,
        Some(&0x24) => {
            if buf.len() < 4 {
                return Interleaved::NeedMore;
            }
            let channel = buf[1];
            let len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
            let consumed = 4 + len;
            if buf.len() < consumed {
                return Interleaved::NeedMore;
            }
            Interleaved::Binary {
                channel,
                start: 4,
                end: consumed,
                consumed,
            }
        }
        // Not a binary frame: an interleaved RTSP request (e.g. TEARDOWN).
        Some(_) => match RtspRequest::parse(buf) {
            Some((_req, consumed)) => Interleaved::Rtsp { consumed },
            None => Interleaved::NeedMore,
        },
    }
}

/// Receive H.264 over a TCP-interleaved control stream (RFC 2326 §10.12): demux
/// `$`-framed RTP on `rtp_channel` and depayload it into access units, answering
/// any RTSP request interleaved between the binary frames. TCP is ordered and
/// lossless, so no jitter buffer / RTCP / FEC is needed (unlike the UDP path);
/// packets depayload straight through. Ends on the publisher closing the
/// connection, a TEARDOWN, or `frame_limit` access units. `leftover` is any
/// binary already buffered past RECORD.
async fn receive_interleaved(
    control: &mut tokio::net::TcpStream,
    responder: &mut RtspResponder,
    rtp_channel: u8,
    mut pending: Vec<u8>,
    frame_limit: u64,
    seq_base: u64,
    out: &mut SessionTap<'_>,
) -> Result<SessionEnd, G2gError> {
    let mut depay = RtpH264Depayloader::new();
    let mut seq = seq_base;
    let mut ts_base: Option<u32> = None;
    let mut buf = [0u8; CTRL_BUF];
    loop {
        // Drain every complete interleaved item currently buffered.
        loop {
            match next_interleaved(&pending) {
                Interleaved::Binary {
                    channel,
                    start,
                    end,
                    consumed,
                } => {
                    // Depayload only the RTP channel (skip RTCP / other channels).
                    if channel == rtp_channel {
                        if let Some(au) = depay.depacketize(&pending[start..end]) {
                            if push_access_unit(
                                au,
                                &mut ts_base,
                                &mut seq,
                                seq_base,
                                frame_limit,
                                out,
                            )
                            .await?
                            {
                                return Ok(SessionEnd::Limit);
                            }
                        }
                    }
                    pending.drain(..consumed);
                }
                Interleaved::Rtsp { consumed } => {
                    let teardown = match RtspRequest::parse(&pending) {
                        Some((req, _)) => {
                            let (response, event) = responder.handle_request(&req);
                            control.write_all(&response).await.is_err()
                                || event == RtspEvent::Teardown
                        }
                        None => false,
                    };
                    pending.drain(..consumed);
                    if teardown {
                        return Ok(SessionEnd::PeerGone);
                    }
                }
                Interleaved::NeedMore => break,
            }
        }
        let Ok(n) = control.read(&mut buf).await else {
            return Ok(SessionEnd::PeerGone);
        };
        if n == 0 {
            // Publisher closed the connection: RTP has no in-band end marker, so
            // the close is what ends the session.
            return Ok(SessionEnd::PeerGone);
        }
        out.mark_activity();
        pending.extend_from_slice(&buf[..n]);
    }
}

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
            "Hosts an RTSP endpoint a publisher pushes H.264 into (ANNOUNCE/RECORD)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
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
        PROPS
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
            "num-buffers" => crate::netprop::set_frame_limit(&mut self.frame_limit, &value),
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
            "num-buffers" => Some(crate::netprop::get_frame_limit(self.frame_limit)),
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

    /// Serve publishers one at a time until `max-sessions` of them have recorded
    /// (0 = forever) or `num-buffers` access units have been emitted. Each session
    /// owns its own RTP socket / interleaved channel and responder, dropped when
    /// it ends, and anyone who connects mid-session is refused with a 503.
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let std_listener = self.listener.take().ok_or(G2gError::NotConfigured)?;
            std_listener.set_nonblocking(true).map_err(io_err)?;
            let listener = tokio::net::TcpListener::from_std(std_listener).map_err(io_err)?;

            let recv = self.recv.clone();
            let frame_limit = self.frame_limit;
            let max_sessions = self.max_sessions;
            let timeout_ns = self.session_timeout_ns;
            let frame_period_ns = match self.fps {
                0 => 0,
                fps => 1_000_000_000 / fps as u64,
            };

            let activity = Cell::new(g2g_core::metrics::monotonic_ns());
            let refused = Cell::new(0u64);
            let served = Cell::new(0u64);
            let mut tap = SessionTap::new(out, &activity, frame_period_ns);

            let result: Result<u64, G2gError> = async {
                loop {
                    let (control, _peer) = listener.accept().await.map_err(io_err)?;
                    activity.set(g2g_core::metrics::monotonic_ns());
                    // This session's RTP socket: its port is advertised in a UDP
                    // SETUP, and it is released with the session (bound even for an
                    // interleaved publisher, which then simply drops it).
                    let rtp_socket = tokio::net::UdpSocket::bind(("0.0.0.0", 0))
                        .await
                        .map_err(io_err)?;
                    let server_rtp_port = rtp_socket.local_addr().map_err(io_err)?.port();
                    let responder = self.responder(server_rtp_port);

                    // Anyone connecting during the handshake is refused too, so a
                    // second publisher never waits in the backlog.
                    let session = tokio::select! {
                        r = handshake(control, responder, rtp_socket, &activity) => r?,
                        _ = inactive_for(&activity, timeout_ns) => None,
                        _ = refuse_extras(&listener, &refused) => unreachable!("refusal never ends"),
                    };
                    let Some(session) = session else {
                        continue; // gone before RECORD: listen for the next one
                    };

                    // Remaining budget, so `num-buffers` counts across sessions and
                    // the sequence numbering continues over a re-publish.
                    let seq_base = tap.frames();
                    let remaining = match frame_limit {
                        0 => 0,
                        limit => limit.saturating_sub(seq_base),
                    };
                    let end = match session {
                        // UDP: the jitter + (optional) RTCP + depayload path shared
                        // with UdpSrc, while the control channel is watched for the
                        // publisher leaving (datagrams alone never reveal that).
                        RecordTransport::Udp {
                            rtp_socket,
                            mut control,
                            mut responder,
                        } => tokio::select! {
                            r = crate::rtprecv::receive_rtp_h264(&rtp_socket, &recv, remaining, seq_base, &mut tap) => {
                                r?;
                                SessionEnd::Limit
                            }
                            _ = watch_control(&mut control, &mut responder, &activity) => SessionEnd::PeerGone,
                            _ = inactive_for(&activity, timeout_ns) => SessionEnd::PeerGone,
                            _ = refuse_extras(&listener, &refused) => unreachable!("refusal never ends"),
                        },
                        // TCP-interleaved: the control stream itself carries the
                        // `$`-framed RTP, so its close ends the session.
                        RecordTransport::Interleaved {
                            mut control,
                            mut responder,
                            rtp_channel,
                            leftover,
                        } => tokio::select! {
                            r = receive_interleaved(&mut control, &mut responder, rtp_channel, leftover, remaining, seq_base, &mut tap) => r?,
                            _ = inactive_for(&activity, timeout_ns) => SessionEnd::PeerGone,
                            _ = refuse_extras(&listener, &refused) => unreachable!("refusal never ends"),
                        },
                    };
                    served.set(served.get().saturating_add(1));
                    match end {
                        // `push_access_unit` already emitted the EOS.
                        SessionEnd::Limit => return Ok(tap.frames()),
                        SessionEnd::PeerGone => {
                            tap.end_session();
                            if max_sessions != 0 && served.get() >= max_sessions {
                                tap.push(PipelinePacket::Eos).await?;
                                return Ok(tap.frames());
                            }
                        }
                    }
                }
            }
            .await;

            self.sessions_served = served.get();
            self.sessions_refused = refused.get();
            result
        })
    }
}

impl PadTemplates for RtspServerSrc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))])
    }
}
