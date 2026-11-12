// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Shared RTSP ingest session machinery: the per-publisher pieces both ingest
//! elements ride, from the accepted TCP control connection to depayloaded access
//! units. [`RtspServerSrc`](crate::rtspserversrc) drives one session at a time,
//! recording its video track and, with an audio pad, its AAC track;
//! [`RtspServerSrcN`](crate::rtspserversrcn) drives N of them concurrently, one
//! per output pad. Each session owns its sockets ([`RecordSession`]) and its
//! inactivity watchdog ([`inactive_for`]); the downstream taps ([`SessionTap`])
//! span every session on a pad.

use core::cell::Cell;
use core::future::Future;
use core::time::Duration;

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::element::BoxFuture;
use g2g_core::rtp::CaptureTimeExtensionIds;
use g2g_core::runtime::{join_all, Receiver, Sender};
use g2g_core::{
    Caps, Dim, G2gError, MultiOutputSink, OutputSink, PipelinePacket, PropError, PropKind,
    PropValue, PropertySpec, PushOutcome, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::rtcp::SenderClock;
use crate::rtpdepay::{RtpDepayloader, RtpEncoding};
use crate::rtprecv::{
    AccessUnitEmitter, CaptureTimeLatch, RtpRecvConfig, SessionTimeline, TrackTimeline,
};
use crate::rtspserver::{
    interleaved_frame, RtspEvent, RtspRequest, RtspResponder, TrackTransport,
    INTERLEAVED_HEADER_LEN, INTERLEAVED_MARKER,
};
use crate::rtspserversink::bind_rtp_pair;
use crate::sdp::SdpMedia;
use crate::udprecv::{UdpReceiveConfig, UdpReceiver};

/// TCP read buffer for RTSP control requests.
const CTRL_BUF: usize = 8192;
/// Cap on buffered-but-unparsed control bytes, as in the serving sink: a client
/// dripping a never-terminating request (or an oversized Content-Length) is
/// dropped instead of growing the buffer without bound.
const MAX_PENDING: usize = 64 * 1024;
/// How long a refused publisher is given to send the request its 503 answers.
const REFUSE_READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Drive one accepted publisher's RTSP handshake to RECORD: run
/// OPTIONS/ANNOUNCE/SETUP/RECORD over the TCP control channel, binding an
/// RTP/RTCP socket pair per announced track (its ports are advertised in a UDP
/// SETUP of that track), and return the [`RecordSession`] once the publisher has
/// issued RECORD. The control stream is kept (never dropped): a real RTSP
/// publisher (ffmpeg) holds the control connection open while it records and
/// treats the server closing it as a fatal "broken pipe", so dropping it would
/// abort the publish. For TCP-interleaved, that same stream *is* the RTP
/// transport, and any bytes already buffered past RECORD are handed on so a
/// pipelined first frame is not lost.
///
/// `Ok(None)` means this publisher went away before RECORD (closed, tore down, or
/// overflowed the control buffer): the caller drops it and listens for the next
/// one rather than failing the graph.
pub(crate) async fn handshake(
    mut control: tokio::net::TcpStream,
    mut responder: RtspResponder,
    activity: &Cell<u64>,
) -> Result<Option<RecordSession>, G2gError> {
    let mut track_sockets = Vec::new();
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
                RtspEvent::Announce { tracks } => {
                    track_sockets = Vec::with_capacity(tracks);
                    for _ in 0..tracks {
                        track_sockets.push(bind_rtp_pair().await?);
                    }
                    let ports = track_sockets
                        .iter()
                        .map(|(rtp, _)| rtp.local_addr().map(|addr| addr.port()))
                        .collect::<Result<Vec<u16>, _>>()
                        .map_err(io_err)?;
                    responder.set_track_server_ports(ports);
                }
                RtspEvent::Record => {
                    return Ok(Some(RecordSession {
                        control,
                        responder,
                        track_sockets,
                        leftover: pending,
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

/// A publisher that reached RECORD. The control `TcpStream` stays open for the
/// session (a UDP publisher needs it open; an interleaved one sends its RTP on
/// it), and the session's responder answers keepalives and a mid-RECORD
/// TEARDOWN from the same protocol state.
#[derive(Debug)]
pub(crate) struct RecordSession {
    control: tokio::net::TcpStream,
    responder: RtspResponder,
    // the RTP and RTCP sockets of each announced track, in announce order
    track_sockets: Vec<(tokio::net::UdpSocket, tokio::net::UdpSocket)>,
    // bytes read past RECORD, such as a pipelined first interleaved frame
    leftover: Vec<u8>,
}

pub(crate) const ENCODING_NAME_PROPERTY: PropertySpec = PropertySpec::new(
    "encoding-name",
    PropKind::Str,
    "video codec the publishers send, as in gst application/x-rtp caps",
)
.with_default("H264")
.with_enum_values("H264 | H265");

pub(crate) fn set_encoding_name(
    recv: &mut RtpRecvConfig,
    value: &PropValue,
) -> Result<(), PropError> {
    let name = value.as_str().ok_or(PropError::Type)?;
    let encoding = RtpEncoding::from_name(name)
        .filter(|encoding| encoding.video_codec().is_some())
        .ok_or(PropError::Value)?;
    recv.encoding = encoding;
    Ok(())
}

// the caps an ingest pad declares before the publisher's SPS refines them
pub(crate) fn ingest_caps(recv: &RtpRecvConfig, width: u32, height: u32, fps: u32) -> Caps {
    Caps::CompressedVideo {
        codec: ingest_codec(recv),
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(fps << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}

pub(crate) fn ingest_codec(recv: &RtpRecvConfig) -> VideoCodec {
    recv.encoding
        .video_codec()
        .expect("set_encoding_name admits video encodings only")
}

/// What an ingest pad records: the first announced track of this format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IngestFormat {
    Video(VideoCodec),
    Aac,
}

impl IngestFormat {
    // the receive path for `media`, `None` when this format does not record it
    fn recv_config(self, recv: &RtpRecvConfig, media: &SdpMedia) -> Option<RtpRecvConfig> {
        let encoding = match self {
            IngestFormat::Video(codec) => {
                let Caps::CompressedVideo {
                    codec: announced, ..
                } = media.caps
                else {
                    return None;
                };
                (announced == codec).then(|| recv.encoding.clone())?
            }
            IngestFormat::Aac => media.aac_encoding()?,
        };
        Some(RtpRecvConfig {
            encoding,
            capture_time_ids: media.capture_time_ids(),
            parameter_sets: media.parameter_sets.clone(),
            ..recv.clone()
        })
    }
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

/// One output pad of an ingest element, across every publisher session on it.
#[derive(Debug)]
pub(crate) struct IngestPad<'a> {
    pub(crate) format: IngestFormat,
    pub(crate) tap: SessionTap<'a>,
}

impl IngestPad<'_> {
    // true once `frame_limit` access units are out, which also pushed the pad's `Eos`
    pub(crate) fn finished(&self, frame_limit: u64) -> bool {
        self.tap.frames() >= frame_limit
    }
}

/// Close out a session on every pad: the next publisher's timestamps start one
/// frame past the last one any pad emitted, so its tracks stay in step.
pub(crate) fn end_session(pads: &mut [IngestPad<'_>]) {
    let next = pads
        .iter()
        .map(|pad| pad.tap.last_pts_ns.saturating_add(pad.tap.frame_period_ns))
        .max()
        .unwrap_or(0);
    for pad in pads {
        pad.tap.pts_offset_ns = next;
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
        Some(&INTERLEAVED_MARKER) => match interleaved_frame(buf) {
            Some(frame) => Interleaved::Binary {
                channel: frame.channel,
                start: INTERLEAVED_HEADER_LEN,
                end: frame.consumed,
                consumed: frame.consumed,
            },
            None => Interleaved::NeedMore,
        },
        // Not a binary frame: an interleaved RTSP request (e.g. TEARDOWN).
        Some(_) => match RtspRequest::parse(buf) {
            Some((_req, consumed)) => Interleaved::Rtsp { consumed },
            None => Interleaved::NeedMore,
        },
    }
}

// the largest interleaved frame, a 16-bit length of RTP, still waiting for one more read
const MAX_INTERLEAVED_PENDING: usize = INTERLEAVED_HEADER_LEN + u16::MAX as usize + CTRL_BUF;

// one recorded track of an interleaved session, demuxed off the control stream
struct InterleavedTrack<'t, 'o> {
    rtp_channel: u8,
    rtcp_channel: u8,
    depay: RtpDepayloader,
    capture_time_ids: CaptureTimeExtensionIds,
    capture_time: CaptureTimeLatch,
    emitter: AccessUnitEmitter<'t>,
    timeline: TrackTimeline<'t>,
    out: &'o mut dyn OutputSink,
    finished: bool,
}

/// Receive the recorded tracks of a TCP-interleaved session (RFC 2326 §10.12):
/// demux `$`-framed RTP by channel and depayload it into access units, read the
/// sender reports on the RTCP channels, and answer any RTSP request interleaved
/// between the binary frames. TCP is ordered and lossless, so no jitter buffer /
/// FEC is needed (unlike the UDP path); packets depayload straight through. Ends
/// on the publisher closing the connection or a TEARDOWN, or once every track
/// has emitted its `frame_limit`. `pending` is any binary already buffered past
/// RECORD.
async fn receive_interleaved(
    control: &mut tokio::net::TcpStream,
    responder: &mut RtspResponder,
    mut pending: Vec<u8>,
    tracks: &mut [InterleavedTrack<'_, '_>],
    activity: &Cell<u64>,
) -> Result<SessionEnd, G2gError> {
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
                    let packet = &pending[start..end];
                    if let Some(track) = tracks
                        .iter_mut()
                        .find(|track| track.rtp_channel == channel && !track.finished)
                    {
                        track.capture_time.observe(&track.capture_time_ids, packet);
                        for au in track.depay.depacketize(packet) {
                            let capture_unix_ns = track.capture_time.take(au.rtp_timestamp);
                            if track
                                .emitter
                                .push(au, capture_unix_ns, &mut *track.out)
                                .await?
                            {
                                track.finished = true;
                                break;
                            }
                        }
                    } else if let Some(track) =
                        tracks.iter().find(|track| track.rtcp_channel == channel)
                    {
                        track.timeline.note_sender_reports(
                            packet,
                            track.emitter.encoding().clock_rate_hz(),
                            g2g_core::metrics::monotonic_ns(),
                        );
                    }
                    pending.drain(..consumed);
                    if !tracks.is_empty() && tracks.iter().all(|track| track.finished) {
                        return Ok(SessionEnd::Limit);
                    }
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
        activity.set(g2g_core::metrics::monotonic_ns());
        pending.extend_from_slice(&buf[..n]);
        if pending.len() > MAX_INTERLEAVED_PENDING {
            return Ok(SessionEnd::PeerGone);
        }
    }
}

/// Receive one UDP track through the jitter + (optional) RTCP + depayload path
/// shared with UdpSrc until it has emitted `frame_limit` access units.
async fn receive_udp_track(
    (rtp_socket, rtcp_socket): (tokio::net::UdpSocket, tokio::net::UdpSocket),
    cfg: &RtpRecvConfig,
    frame_limit: u64,
    seq_base: u64,
    timeline: TrackTimeline<'_>,
    out: &mut dyn OutputSink,
) -> Result<(), G2gError> {
    let mut receiver = UdpReceiver::new(rtp_socket, UdpReceiveConfig::default()).map_err(io_err)?;
    crate::rtprecv::receive_rtp(
        &mut receiver,
        Some(&rtcp_socket),
        cfg,
        frame_limit,
        seq_base,
        Some(timeline),
        out,
    )
    .await
    .map(drop)
}

/// Record one publisher session onto `pads`: each pad that has not reached
/// `frame_limit` takes the first SETUP track of its format, over UDP or
/// interleaved, and every recorded track is placed on one timeline from the
/// publisher's sender reports. Tracks no pad records are SETUP but dropped.
/// Ends `Limit` once every recorded track has emitted `frame_limit` access
/// units, and `PeerGone` when the publisher tears down or closes.
pub(crate) async fn receive_session(
    session: RecordSession,
    pads: &mut [IngestPad<'_>],
    recv: &RtpRecvConfig,
    frame_limit: u64,
    activity: &Cell<u64>,
) -> Result<SessionEnd, G2gError> {
    let RecordSession {
        mut control,
        mut responder,
        track_sockets,
        leftover,
    } = session;
    let transports = responder.track_transports().to_vec();
    let mut track_sockets: Vec<_> = track_sockets.into_iter().map(Some).collect();
    let mut claimed = alloc::vec![false; transports.len()];
    // (pad, track, receive path) for every pad this session records
    let mut recorded = Vec::new();
    for (pad_index, pad) in pads.iter().enumerate() {
        if pad.finished(frame_limit) {
            continue;
        }
        let track = responder
            .announced_tracks()
            .iter()
            .enumerate()
            .find_map(|(index, track)| {
                let media = track.media.as_ref()?;
                let setup = transports[index].is_some() && !claimed[index];
                setup
                    .then(|| pad.format.recv_config(recv, media))?
                    .map(|cfg| (index, cfg))
            });
        if let Some((index, cfg)) = track {
            claimed[index] = true;
            recorded.push((pad_index, index, cfg));
        }
    }
    let interleaved_setup = transports
        .iter()
        .any(|transport| matches!(transport, Some(TrackTransport::Interleaved { .. })));

    let timeline = SessionTimeline::default();
    let sender_clocks: Vec<Cell<Option<SenderClock>>> =
        recorded.iter().map(|_| Cell::new(None)).collect();
    let mut udp_tracks: Vec<BoxFuture<'_, Result<(), G2gError>>> = Vec::new();
    let mut interleaved_tracks = Vec::new();
    for (pad_index, pad) in pads.iter_mut().enumerate() {
        let Some(slot) = recorded
            .iter()
            .position(|(recorded_pad, _, _)| *recorded_pad == pad_index)
        else {
            continue;
        };
        let (_, track, cfg) = &recorded[slot];
        let track_timeline = TrackTimeline {
            session: &timeline,
            sender_clock: &sender_clocks[slot],
        };
        let seq_base = pad.tap.frames();
        let remaining = frame_limit.saturating_sub(seq_base);
        match transports[*track].expect("recorded tracks were SETUP") {
            TrackTransport::Udp { .. } => {
                let sockets = track_sockets[*track]
                    .take()
                    .expect("every announced track has its sockets bound");
                udp_tracks.push(Box::pin(receive_udp_track(
                    sockets,
                    cfg,
                    remaining,
                    seq_base,
                    track_timeline,
                    &mut pad.tap,
                )));
            }
            TrackTransport::Interleaved {
                rtp_channel,
                rtcp_channel,
            } => interleaved_tracks.push(InterleavedTrack {
                rtp_channel,
                rtcp_channel,
                depay: RtpDepayloader::new(&cfg.encoding),
                capture_time_ids: cfg.capture_time_ids,
                capture_time: CaptureTimeLatch::default(),
                emitter: AccessUnitEmitter::new(cfg, seq_base, remaining, Some(track_timeline)),
                timeline: track_timeline,
                out: &mut pad.tap,
                finished: false,
            }),
        }
    }
    // Sockets of tracks no pad records stay open, so the publisher's datagrams to
    // them are dropped here rather than refused.
    let _unrecorded = track_sockets;

    let records_interleaved = !interleaved_tracks.is_empty();
    let records_udp = !udp_tracks.is_empty();
    let mut udp = core::pin::pin!(join_all(udp_tracks));
    let mut control_end = core::pin::pin!(async {
        if interleaved_setup {
            receive_interleaved(
                &mut control,
                &mut responder,
                leftover,
                &mut interleaved_tracks,
                activity,
            )
            .await
        } else {
            watch_control(&mut control, &mut responder, activity).await;
            Ok(SessionEnd::PeerGone)
        }
    });
    tokio::select! {
        end = &mut control_end => match end? {
            SessionEnd::PeerGone => Ok(SessionEnd::PeerGone),
            SessionEnd::Limit => {
                for result in udp.await {
                    result?;
                }
                Ok(SessionEnd::Limit)
            }
        },
        results = &mut udp, if records_udp => {
            for result in results {
                result?;
            }
            if records_interleaved {
                control_end.await
            } else {
                Ok(SessionEnd::Limit)
            }
        }
    }
}

/// Access units a pad may run ahead of the drain loop before its session blocks,
/// so a stalled branch backs its own publisher up rather than the others.
pub(crate) const PAD_QUEUE: usize = 4;

/// One pad's downstream end. The sessions are polled concurrently in a single
/// task, so they cannot each hold the [`MultiOutputSink`]; each pushes onto the
/// shared channel tagged with its output port, and [`forward_pads`], which owns
/// the sink, forwards it.
#[derive(Debug)]
pub(crate) struct PadSink {
    port: usize,
    tx: Sender<(usize, PipelinePacket)>,
    /// The in-flight tagged packet of a blocked push (the drain channel wants
    /// `(port, packet)`, so the caller's slot cannot hold it directly).
    staged: Option<(usize, PipelinePacket)>,
}

impl PadSink {
    pub(crate) fn new(port: usize, tx: Sender<(usize, PipelinePacket)>) -> Self {
        Self {
            port,
            tx,
            staged: None,
        }
    }
}

impl OutputSink for PadSink {
    fn begin_push(&mut self) {
        self.staged = None;
    }

    fn poll_push(
        &mut self,
        cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        use core::task::Poll;
        if self.staged.is_none() {
            let taken = packet.take().expect("poll_push without a packet");
            self.staged = Some((self.port, taken));
        }
        // The drain loop is gone only once the graph is shutting down.
        match self.tx.poll_send(cx, &mut self.staged) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(())) => Poll::Ready(Ok(PushOutcome::Accepted)),
            Poll::Ready(Err(_)) => {
                self.staged = None;
                Poll::Ready(Err(G2gError::Shutdown))
            }
        }
    }
}

/// Run `producers`, which push through [`PadSink`]s, while forwarding what they
/// queue on `rx` to `out`, then flush the queue once they and their senders are
/// gone.
pub(crate) async fn forward_pads<T>(
    producers: impl Future<Output = T>,
    rx: &Receiver<(usize, PipelinePacket)>,
    out: &mut dyn MultiOutputSink,
) -> Result<T, G2gError> {
    let mut producers = core::pin::pin!(producers);
    let finished = loop {
        tokio::select! {
            biased;
            outcome = &mut producers => break outcome,
            item = rx.recv() => {
                if let Some((port, packet)) = item {
                    out.push_to(port, packet).await?;
                }
            }
        }
    };
    while let Some((port, packet)) = rx.recv().await {
        out.push_to(port, packet).await?;
    }
    Ok(finished)
}
