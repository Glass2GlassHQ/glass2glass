//! Shared RTSP ingest session machinery: the per-publisher pieces both ingest
//! elements ride, from the accepted TCP control connection to depayloaded access
//! units. [`RtspServerSrc`](crate::rtspserversrc) drives one session at a time on
//! its single pad; [`RtspServerSrcN`](crate::rtspserversrcn) drives N of them
//! concurrently, one per output pad. Each session owns its transport
//! ([`RecordTransport`]), its inactivity watchdog ([`inactive_for`]) and its
//! downstream tap ([`SessionTap`]), so nothing carries over to the next
//! publisher.

use core::cell::Cell;

use core::time::Duration;

use alloc::format;
use alloc::vec::Vec;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::{G2gError, OutputSink, PipelinePacket, PushOutcome};

use crate::rtpdepay::RtpH264Depayloader;
use crate::rtprecv::push_access_unit;
use crate::rtspserver::{RtspEvent, RtspRequest, RtspResponder};

/// TCP read buffer for RTSP control requests.
const CTRL_BUF: usize = 8192;
/// Cap on buffered-but-unparsed control bytes, as in the serving sink: a client
/// dripping a never-terminating request (or an oversized Content-Length) is
/// dropped instead of growing the buffer without bound.
const MAX_PENDING: usize = 64 * 1024;
/// How long a refused publisher is given to send the request its 503 answers.
const REFUSE_READ_TIMEOUT: Duration = Duration::from_secs(5);

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
pub(crate) async fn handshake(
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
pub(crate) enum RecordTransport {
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
pub(crate) enum SessionEnd {
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
pub(crate) struct SessionTap<'a> {
    out: &'a mut dyn OutputSink,
    activity: &'a Cell<u64>,
    frames: u64,
    pts_offset_ns: u64,
    last_pts_ns: u64,
    frame_period_ns: u64,
    /// Whether the packet in the caller's slot already got the PTS offset, so
    /// a re-poll under downstream backpressure never shifts twice.
    shifted: bool,
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
    pub(crate) fn new(
        out: &'a mut dyn OutputSink,
        activity: &'a Cell<u64>,
        frame_period_ns: u64,
    ) -> Self {
        Self {
            out,
            activity,
            frames: 0,
            pts_offset_ns: 0,
            last_pts_ns: 0,
            frame_period_ns,
            shifted: false,
        }
    }

    /// Access units pushed downstream so far, across every session.
    pub(crate) fn frames(&self) -> u64 {
        self.frames
    }

    /// Note liveness on a session that is receiving bytes but has not completed
    /// an access unit yet, so the inactivity watchdog does not reap it.
    pub(crate) fn mark_activity(&self) {
        self.activity.set(g2g_core::metrics::monotonic_ns());
    }

    /// Close out a session: the next publisher's timestamps start one frame past
    /// the last one emitted.
    pub(crate) fn end_session(&mut self) {
        self.pts_offset_ns = self.last_pts_ns.saturating_add(self.frame_period_ns);
    }
}

impl OutputSink for SessionTap<'_> {
    fn begin_push(&mut self) {
        self.shifted = false;
        self.out.begin_push();
    }

    fn poll_push(
        &mut self,
        cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        // Shift exactly once per packet: a re-poll under downstream
        // backpressure must not re-apply the offset.
        if !self.shifted {
            if let Some(PipelinePacket::DataFrame(frame)) = packet.as_mut() {
                let t = &mut frame.timing;
                t.pts_ns = t.pts_ns.saturating_add(self.pts_offset_ns);
                t.dts_ns = t.dts_ns.saturating_add(self.pts_offset_ns);
                t.capture_ns = t.capture_ns.saturating_add(self.pts_offset_ns);
                self.last_pts_ns = t.pts_ns;
                self.frames += 1;
                self.activity.set(g2g_core::metrics::monotonic_ns());
            }
            self.shifted = true;
        }
        self.out.poll_push(cx, packet)
    }
}

/// Wait until the session has been silent for `timeout_ns`. Never completes when
/// reaping is disabled (`u64::MAX`), so it can sit in a `select!` unconditionally.
pub(crate) async fn inactive_for(activity: &Cell<u64>, timeout_ns: u64) {
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

/// Answer one refused publisher: echo the `CSeq` it asked with in a 503, then
/// close.
pub(crate) async fn refuse_one(mut sock: tokio::net::TcpStream) {
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
pub(crate) async fn watch_control(
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
pub(crate) async fn receive_interleaved(
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
