// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Shared RTP receive loop: the jitter-buffer + RTCP RR/NACK +
//! FEC/RTX + depayload-to-access-unit path used by both [`UdpSrc`](crate::udpsrc)
//! (raw RTP) and [`RtspServerSrc`](crate::rtspserversrc) (RTSP RECORD ingest).
//!
//! Given a bound UDP socket, read in batches through a
//! [`UdpReceiver`](crate::udprecv::UdpReceiver), it reorders (jitter buffer),
//! recovers (ULPFEC / FlexFEC / RTX), and depayloads RTP into H.264 Annex-B
//! access units or Opus packets ([`RtpRecvConfig::encoding`]), pushing each as
//! one frame downstream, while driving RTCP receiver reports and Generic NACK
//! retransmission requests back to the sender.
//!
//! Both ingest sources differ only in how they obtain the socket (a raw bind vs
//! the RTP port negotiated over an RTSP control channel); the receive path is
//! identical, so it lives here once.

use core::cell::Cell;

use std::net::SocketAddr;
use std::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::rtp::{CaptureTimeExtensionIds, RtpHeader};
use g2g_core::{
    Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket, PropError,
    PropValue, Rate,
};

use crate::filesink::io_err;
use crate::flexfec::FlexFecDecoder;
use crate::nalparse::NalCodec;
use crate::rist::NackScheduler;
use crate::rtcp::{self, ReceptionStats, RtcpPacket, SenderClock};
use crate::rtpdepay::{AccessUnit, RtpDepayloader, RtpEncoding};
use crate::rtpjitter::{JitterConfig, RtpJitterBuffer, RtpPtsRebase, SequenceContinuity};
use crate::rtx;
use crate::udprecv::UdpReceiver;
use crate::ulpfec::FecDecoder;

/// Default receiver-report cadence.
pub const DEFAULT_RR_INTERVAL_MS: u64 = 1000;
// GStreamer's rtpjitterbuffer, before it has measured jitter and RTT, first requests a loss after 20 ms and retries every 40 ms
const FIRST_NACK_DELAY_NS: u64 = 20_000_000;
pub const NACK_RETRY_INTERVAL_NS: u64 = 40_000_000;

/// Receive-path policy shared by the RTP ingest sources. Mirrors the tuning
/// each source exposes through its `with_jitter` / `with_rtcp` / `with_rtx` /
/// `with_fec` / `with_flexfec` builders.
#[derive(Debug, Clone)]
pub struct RtpRecvConfig {
    /// Reorder/loss-resilience policy for the receive path.
    pub jitter: JitterConfig,
    /// RTCP receiver-report cadence in ms (0 disables RTCP feedback entirely).
    pub rtcp_rr_interval_ms: u64,
    /// Emit RTPFB Generic NACK for detected gaps (requests retransmission).
    pub nack_enabled: bool,
    /// RFC 4588 RTX `(rtx payload type, apt)`: packets on `rtx payload type` are
    /// reconstructed to the original stream (restamped with `apt`) before reorder.
    pub rtx: Option<(u8, u8)>,
    /// RFC 5109 ULPFEC repair-packet payload type, if enabled.
    pub fec_pt: Option<u8>,
    /// RFC 8627 FlexFEC repair-packet payload type, if enabled.
    pub flexfec_pt: Option<u8>,
    /// The caps the source negotiated. When set, each depayloaded access unit's
    /// SPS refines them and a `CapsChanged` precedes the frame that carries the
    /// new geometry ([`SpsCapsRefiner`]). `None` leaves the stream unrefined.
    pub declared_caps: Option<Caps>,
    pub capture_time_ids: CaptureTimeExtensionIds,
    pub encoding: RtpEncoding,
    // ffmpeg's RTSP publisher sends the parameter sets only in the SDP sprop attributes
    pub parameter_sets: Vec<u8>,
}

impl Default for RtpRecvConfig {
    fn default() -> Self {
        Self {
            jitter: JitterConfig::default(),
            rtcp_rr_interval_ms: DEFAULT_RR_INTERVAL_MS,
            nack_enabled: true,
            rtx: None,
            fec_pt: None,
            flexfec_pt: None,
            declared_caps: None,
            capture_time_ids: CaptureTimeExtensionIds::default(),
            encoding: RtpEncoding::default(),
            parameter_sets: Vec::new(),
        }
    }
}

/// Runtime caps refinement from the bitstream: an RTP receiver has no in-band
/// stream description, so the geometry it negotiated with is whatever the SDP
/// declared or the element was told. The H.264 SPS in the stream is
/// authoritative, so each access unit is checked against the caps in force and a
/// mismatch (the first SPS, or a mid-stream resolution change) yields the
/// corrected caps for the caller to emit as `CapsChanged`.
#[derive(Debug)]
pub struct SpsCapsRefiner {
    current: Caps,
}

impl SpsCapsRefiner {
    pub fn new(declared: Caps) -> Self {
        Self { current: declared }
    }

    /// The caps in force right now (the declared ones until an SPS refines them).
    pub fn caps(&self) -> &Caps {
        &self.current
    }

    /// Refine from the SPS in one Annex-B access unit. Returns the new caps the
    /// first time they differ from the ones in force (and on any later change),
    /// `None` when the unit carries no SPS or nothing changed.
    pub fn refine(&mut self, au: &[u8]) -> Option<Caps> {
        let Caps::CompressedVideo {
            codec, framerate, ..
        } = &self.current
        else {
            return None;
        };
        let (codec, framerate) = (*codec, framerate.clone());
        let sps = match codec {
            g2g_core::VideoCodec::H264 => crate::h264parse::H264Codec::extract_sps_info(au)?,
            g2g_core::VideoCodec::H265 => crate::h265parse::H265Codec::extract_sps_info(au)?,
            _ => return None,
        };
        let refined = Caps::CompressedVideo {
            codec,
            width: Dim::Fixed(sps.width),
            height: Dim::Fixed(sps.height),
            // Keep the declared rate when the SPS has no VUI timing: a live
            // source must never hand downstream a wildcard rate.
            framerate: sps.framerate.map_or(framerate, Rate::Fixed),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        if refined == self.current {
            return None;
        }
        self.current = refined.clone();
        Some(refined)
    }
}

/// Handle the receive-tuning half of an RTP ingest source's `set_property`
/// (`jitter-latency`, `jitter-depth`, `rtcp-rr-interval`, `nack`,
/// `rtx-payload-type`, `rtx-apt`, `fec-payload-type`, `flexfec-payload-type`),
/// the [`crate::netprop`] pattern applied to [`RtpRecvConfig`]. Returns
/// `Some(result)` when `name` is one of these, `None` otherwise. A payload type
/// of 0 disables the corresponding recovery stream (dynamic PTs are 96..=127,
/// so 0 is never a real repair stream).
pub(crate) fn set_recv_prop(
    cfg: &mut RtpRecvConfig,
    name: &str,
    value: &PropValue,
) -> Option<Result<(), PropError>> {
    fn uint(value: &PropValue) -> Result<u64, PropError> {
        value.as_uint().ok_or(PropError::Type)
    }
    fn pt(value: &PropValue) -> Result<u8, PropError> {
        let v = uint(value)?;
        if v > 127 {
            return Err(PropError::Value);
        }
        Ok(v as u8)
    }
    Some(match name {
        "jitter-latency" => uint(value).map(|ms| {
            cfg.jitter.max_hold_ns = ms.saturating_mul(1_000_000);
        }),
        "jitter-depth" => uint(value).map(|d| {
            cfg.jitter.max_depth = d.min(usize::MAX as u64) as usize;
        }),
        "rtcp-rr-interval" => uint(value).map(|ms| {
            cfg.rtcp_rr_interval_ms = ms;
        }),
        "nack" => value
            .as_bool()
            .ok_or(PropError::Type)
            .map(|b| cfg.nack_enabled = b),
        // rtx needs both legs; either may be set first (the other leg keeps its
        // current value), and rtx-payload-type=0 disables RTX.
        "rtx-payload-type" => pt(value).map(|v| {
            cfg.rtx = match v {
                0 => None,
                _ => Some((v, cfg.rtx.map(|(_, apt)| apt).unwrap_or(0))),
            };
        }),
        "rtx-apt" => pt(value).map(|apt| {
            let rtx_pt = cfg.rtx.map(|(p, _)| p).unwrap_or(0);
            cfg.rtx = Some((rtx_pt, apt));
        }),
        "fec-payload-type" => pt(value).map(|v| cfg.fec_pt = (v != 0).then_some(v)),
        "flexfec-payload-type" => pt(value).map(|v| cfg.flexfec_pt = (v != 0).then_some(v)),
        _ => return None,
    })
}

/// The read half of [`set_recv_prop`]. Disabled optional streams read as 0.
pub(crate) fn get_recv_prop(cfg: &RtpRecvConfig, name: &str) -> Option<PropValue> {
    Some(match name {
        "jitter-latency" => PropValue::Uint(cfg.jitter.max_hold_ns / 1_000_000),
        "jitter-depth" => PropValue::Uint(cfg.jitter.max_depth as u64),
        "rtcp-rr-interval" => PropValue::Uint(cfg.rtcp_rr_interval_ms),
        "nack" => PropValue::Bool(cfg.nack_enabled),
        "rtx-payload-type" => PropValue::Uint(cfg.rtx.map(|(p, _)| p).unwrap_or(0) as u64),
        "rtx-apt" => PropValue::Uint(cfg.rtx.map(|(_, apt)| apt).unwrap_or(0) as u64),
        "fec-payload-type" => PropValue::Uint(cfg.fec_pt.unwrap_or(0) as u64),
        "flexfec-payload-type" => PropValue::Uint(cfg.flexfec_pt.unwrap_or(0) as u64),
        _ => return None,
    })
}

#[derive(Debug, Default)]
pub(crate) struct CaptureTimeLatch {
    latest: Option<(u32, i64)>,
}

impl CaptureTimeLatch {
    pub(crate) fn observe(&mut self, ids: &CaptureTimeExtensionIds, packet: &[u8]) {
        let Some(ntp) = ids.read(packet) else { return };
        let Some(parsed) = RtpHeader::parse_header(packet) else {
            return;
        };
        self.latest =
            g2g_core::unix_nanos_from_ntp(ntp).map(|unix_ns| (parsed.header.timestamp, unix_ns));
    }

    pub(crate) fn take(&mut self, rtp_timestamp: u32) -> Option<i64> {
        let (stamped, unix_ns) = self.latest?;
        if stamped != rtp_timestamp {
            return None;
        }
        self.latest = None;
        Some(unix_ns)
    }
}

// the instants one sender's RTP streams share, so their PTS line up on one timeline
#[derive(Debug, Default)]
pub(crate) struct SessionTimeline {
    // the local monotonic instant PTS zero stands for, set by the session's first frame
    origin_ns: Cell<Option<i64>>,
    // the sender's wall clock minus local monotonic time, from the session's first sender report
    sender_minus_local_ns: Cell<Option<i64>>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TrackTimeline<'a> {
    pub(crate) session: &'a SessionTimeline,
    pub(crate) sender_clock: &'a Cell<Option<SenderClock>>,
}

impl TrackTimeline<'_> {
    #[cfg(feature = "rtsp-server")]
    pub(crate) fn note_sender_report(&self, clock: SenderClock, arrival_ns: u64) {
        self.sender_clock.set(Some(clock));
        let offsets = &self.session.sender_minus_local_ns;
        if offsets.get().is_none() {
            offsets.set(Some(clock.unix_nanos - arrival_ns as i64));
        }
    }

    // where on the session timeline the frame stamped `rtp_timestamp` falls, from the
    // sender report when one came first and from the frame's arrival otherwise
    fn session_offset_ns(&self, rtp_timestamp: u32, arrival_ns: u64) -> i64 {
        let local_ns = match (
            self.sender_clock.get(),
            self.session.sender_minus_local_ns.get(),
        ) {
            (Some(clock), Some(offset)) => clock.wall_clock_ns(rtp_timestamp) - offset,
            _ => arrival_ns as i64,
        };
        let origin = self.session.origin_ns.get().unwrap_or(local_ns);
        self.session.origin_ns.set(Some(origin));
        local_ns - origin
    }
}

// turns depayloaded access units into frames with PTS rebased to start at zero,
// shared by the UDP (`receive_rtp`) and TCP-interleaved (`rtspingest`) ingest paths
#[derive(Debug)]
pub(crate) struct AccessUnitEmitter<'a> {
    // taken by the first access unit, which places this stream on the session timeline
    timeline: Option<TrackTimeline<'a>>,
    session_offset_ns: i64,
    encoding: RtpEncoding,
    // emptied at the first keyframe
    parameter_sets: Vec<u8>,
    pts_rebase: RtpPtsRebase,
    // a replacing sender's timeline continues past the last frame instead of restarting at zero
    pts_offset_ns: u64,
    last_pts_ns: u64,
    seq: u64,
    seq_base: u64,
    // u64::MAX never reaches the limit, which is how "run forever" is spelled
    frame_limit: u64,
    // decoders reject the pictures before the first keyframe and after a loss
    awaiting_keyframe: bool,
}

impl<'a> AccessUnitEmitter<'a> {
    // `seq_base` is the sequence number the first emitted frame carries
    pub(crate) fn new(
        cfg: &RtpRecvConfig,
        seq_base: u64,
        frame_limit: u64,
        timeline: Option<TrackTimeline<'a>>,
    ) -> Self {
        Self {
            timeline,
            session_offset_ns: 0,
            encoding: cfg.encoding.clone(),
            parameter_sets: cfg.parameter_sets.clone(),
            pts_rebase: RtpPtsRebase::new(),
            pts_offset_ns: 0,
            last_pts_ns: 0,
            seq: seq_base,
            seq_base,
            frame_limit,
            awaiting_keyframe: true,
        }
    }

    fn emitted(&self) -> u64 {
        self.seq - self.seq_base
    }

    fn restart_timeline_after(&mut self, silence_ns: u64) {
        self.pts_offset_ns = self.last_pts_ns.saturating_add(silence_ns);
        self.pts_rebase = RtpPtsRebase::new();
    }

    #[cfg(feature = "rtsp-server")]
    pub(crate) fn encoding(&self) -> &RtpEncoding {
        &self.encoding
    }

    // true once `frame_limit` access units are out and `Eos` is pushed
    pub(crate) async fn push(
        &mut self,
        au: AccessUnit,
        capture_unix_ns: Option<i64>,
        out: &mut dyn OutputSink,
    ) -> Result<bool, G2gError> {
        // false (the safe default) for H.264 without an IDR NAL.
        let keyframe = self.encoding.is_keyframe(&au.data);
        if self.awaiting_keyframe && !keyframe {
            return Ok(false);
        }
        self.awaiting_keyframe = false;
        let arrival_ns = g2g_core::metrics::monotonic_ns();
        if let Some(timeline) = self.timeline.take() {
            self.session_offset_ns = timeline.session_offset_ns(au.rtp_timestamp, arrival_ns);
        }
        let stream_pts = self.pts_offset_ns
            + self
                .pts_rebase
                .pts_ns(au.rtp_timestamp, self.encoding.clock_rate_hz());
        // a stream that started before the session's first frame drops what precedes it
        let Ok(pts) = u64::try_from(stream_pts as i64 + self.session_offset_ns) else {
            return Ok(false);
        };
        self.last_pts_ns = pts;
        let announced = if keyframe {
            core::mem::take(&mut self.parameter_sets)
        } else {
            Vec::new()
        };
        let in_band = self
            .encoding
            .video_codec()
            .is_some_and(|codec| crate::annexb::starts_with_param_set(&au.data, codec));
        let data = if announced.is_empty() || in_band {
            au.data
        } else {
            [announced, au.data].concat()
        };
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns: pts,
                dts_ns: pts,
                duration_ns: 0,
                capture_ns: pts,
                arrival_ns,
                keyframe,
                ..FrameTiming::default()
            },
            sequence: self.seq,
            meta: Default::default(),
        };
        #[cfg(feature = "metadata")]
        let frame = {
            let mut frame = frame;
            if let Some(unix_nanos) = capture_unix_ns {
                frame.meta.attach(g2g_core::WallClockMeta { unix_nanos });
            }
            frame
        };
        #[cfg(not(feature = "metadata"))]
        let _ = capture_unix_ns;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        self.seq += 1;
        if self.emitted() >= self.frame_limit {
            out.push(PipelinePacket::Eos).await?;
            return Ok(true);
        }
        Ok(false)
    }
}

// the retries spread over the jitter hold, after which the buffer gives up on the loss
fn nack_scheduler(jitter: &JitterConfig) -> NackScheduler {
    let first_request_ns = FIRST_NACK_DELAY_NS.min(jitter.max_hold_ns);
    let retry_window_ns = jitter.max_hold_ns - first_request_ns;
    let max_requests = (retry_window_ns / NACK_RETRY_INTERVAL_NS).max(1);
    NackScheduler::new(
        first_request_ns,
        jitter.max_hold_ns,
        u32::try_from(max_requests).unwrap_or(u32::MAX),
    )
}

// the receive state that belongs to one sender, rebuilt when another sender replaces it
#[derive(Debug)]
struct SenderReceiveState {
    jitter: RtpJitterBuffer,
    depay: RtpDepayloader,
    stats: ReceptionStats,
    // each consulted only when its payload type is set
    fec_decoder: FecDecoder,
    flexfec_decoder: FlexFecDecoder,
    nacks: NackScheduler,
    capture_time: CaptureTimeLatch,
}

impl SenderReceiveState {
    fn new(cfg: &RtpRecvConfig) -> Self {
        Self {
            jitter: RtpJitterBuffer::new(cfg.jitter),
            depay: RtpDepayloader::new(&cfg.encoding),
            stats: ReceptionStats::new(0, cfg.encoding.clock_rate_hz()),
            fec_decoder: FecDecoder::default(),
            flexfec_decoder: FlexFecDecoder::default(),
            nacks: nack_scheduler(&cfg.jitter),
            capture_time: CaptureTimeLatch::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum SsrcAdmission {
    Locked,
    Replaced { silence_ns: u64 },
    Rejected,
}

#[derive(Debug, Default)]
struct SsrcLock {
    locked: Option<u32>,
    last_arrival_ns: u64,
}

impl SsrcLock {
    fn admit(&mut self, ssrc: u32, arrival_ns: u64, silence_limit_ns: u64) -> SsrcAdmission {
        let silence_ns = arrival_ns.saturating_sub(self.last_arrival_ns);
        let admission = match self.locked {
            None => SsrcAdmission::Locked,
            Some(locked) if locked == ssrc => SsrcAdmission::Locked,
            Some(_) if silence_ns >= silence_limit_ns => SsrcAdmission::Replaced { silence_ns },
            Some(_) => return SsrcAdmission::Rejected,
        };
        self.locked = Some(ssrc);
        self.last_arrival_ns = arrival_ns;
        admission
    }
}

const KEYFRAME_REQUEST_INTERVAL_NS: u64 = 1_000_000_000;

/// Run the RTP receive loop on `socket` until `frame_limit` access units
/// have been emitted (`u64::MAX` = run until a socket error or downstream
/// shutdown, 0 = emit nothing), pushing each completed access unit downstream
/// and emitting `Eos` on the bounded path. `seq_base` is the sequence number the
/// first emitted frame carries. Returns the number of access units pushed.
pub(crate) async fn receive_rtp(
    receiver: &mut UdpReceiver,
    cfg: &RtpRecvConfig,
    frame_limit: u64,
    seq_base: u64,
    timeline: Option<TrackTimeline<'_>>,
    out: &mut dyn OutputSink,
) -> Result<u64, G2gError> {
    if crate::numbuffers::finished_at_zero_limit(frame_limit, out).await? {
        return Ok(0);
    }
    let mut sender = SenderReceiveState::new(cfg);
    let mut emitter = AccessUnitEmitter::new(cfg, seq_base, frame_limit, timeline);
    // RFC 3550 8.1 has every participant pick a random SSRC
    let local_ssrc = crate::rist::random_u32();
    let mut ssrc_lock = SsrcLock::default();
    // The original media stream's SSRC, learned from the first non-RTX packet;
    // an RFC 4588 RTX resend is restamped back onto it.
    let mut media_ssrc_seen: Option<u32> = None;
    // In-band caps discovery: correct the declared geometry from the stream's SPS.
    let mut refiner = cfg.declared_caps.clone().map(SpsCapsRefiner::new);

    // RTCP feedback state (RTP/RTCP-muxed on this socket): the peer to report to
    // (learned from the first datagram) and report timers.
    let rtcp_on = cfg.rtcp_rr_interval_ms > 0;
    let rr_interval_ns = cfg.rtcp_rr_interval_ms * 1_000_000;
    let mut peer: Option<SocketAddr> = None;
    let mut last_rr_ns = g2g_core::metrics::monotonic_ns();

    let mut last_keyframe_request_ns: Option<u64> = None;

    loop {
        // Drain every packet the jitter buffer will release now, turning each
        // completed access unit into a downstream frame.
        let mut lost = sender.jitter.stats().lost;
        while let Some(packet) = sender.jitter.pop(g2g_core::metrics::monotonic_ns()) {
            // the pictures after a loss reference the lost one
            if sender.jitter.stats().lost > lost {
                lost = sender.jitter.stats().lost;
                emitter.awaiting_keyframe = true;
            }
            sender.capture_time.observe(&cfg.capture_time_ids, &packet);
            for au in sender.depay.depacketize(&packet) {
                if let Some(caps) = refiner.as_mut().and_then(|r| r.refine(&au.data)) {
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                let capture_unix_ns = sender.capture_time.take(au.rtp_timestamp);
                if emitter.push(au, capture_unix_ns, out).await? {
                    return Ok(emitter.emitted());
                }
            }
        }

        let now = g2g_core::metrics::monotonic_ns();
        let keyframe_request_due = last_keyframe_request_ns
            .is_none_or(|sent| now.saturating_sub(sent) >= KEYFRAME_REQUEST_INTERVAL_NS);
        let request_keyframe = rtcp_on
            && cfg.encoding.video_codec().is_some()
            && emitter.awaiting_keyframe
            && keyframe_request_due;
        if let (true, Some(dest), Some(media_ssrc)) = (request_keyframe, peer, media_ssrc_seen) {
            let pli = rtcp::build_picture_loss_indication(local_ssrc, media_ssrc);
            let _ = receiver.socket().send_to(&pli, dest).await;
            last_keyframe_request_ns = Some(now);
        }

        // Periodic receiver report back to the peer (best-effort).
        if rtcp_on {
            if let Some(dest) = peer {
                if now.saturating_sub(last_rr_ns) >= rr_interval_ns {
                    let rr =
                        rtcp::build_receiver_report(local_ssrc, &[sender.stats.report_block(now)]);
                    let _ = receiver.socket().send_to(&rr, dest).await;
                    last_rr_ns = now;
                }
            }
        }

        // Wait for the next datagram, but no longer than the soonest of the
        // jitter hold deadline (flush a gap) and the next report.
        let now = g2g_core::metrics::monotonic_ns();
        let jitter_deadline = sender.jitter.next_deadline_ns(now);
        let rr_deadline = if rtcp_on && peer.is_some() {
            Some(rr_interval_ns.saturating_sub(now.saturating_sub(last_rr_ns)))
        } else {
            None
        };
        let timeout = match (jitter_deadline, rr_deadline) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        let received = match timeout {
            Some(delay) if delay > 0 => {
                // Err (deadline elapsed, no packet) => None: loop to flush / report.
                tokio::time::timeout(core::time::Duration::from_nanos(delay), receiver.recv())
                    .await
                    .ok()
            }
            // Deadline already elapsed (delay == 0): don't block on recv; loop
            // back so the jitter flush / receiver report at the top of the loop
            // fires now instead of waiting for a packet.
            Some(_) => None,
            None => Some(receiver.recv().await),
        };

        let Some(r) = received else { continue };
        let datagram = r.map_err(io_err)?;
        let (payload, from, now) = (datagram.payload, datagram.from, datagram.arrival_ns);

        // RTCP (muxed) feedback from the sender: a sender report fills the
        // LSR/DLSR fields for round-trip estimation.
        if rtcp::is_rtcp(payload) {
            if rtcp_on {
                for p in rtcp::parse_compound(payload) {
                    if let RtcpPacket::SenderReport { ntp, .. } = p {
                        sender.stats.on_sender_report(ntp, now);
                    }
                }
            }
            continue;
        }

        // RTP media: account it for reception stats, buffer it for reordering,
        // and remember the peer for feedback.
        if payload.len() >= 12 {
            let pt = payload[1] & 0x7F;
            // ULPFEC repair packet: feed the decoder and inject any recovered
            // media into the jitter buffer (no NAK round trip).
            if cfg.fec_pt == Some(pt) {
                sender.fec_decoder.push_fec(payload);
                for rec in sender.fec_decoder.take_recovered() {
                    sender.jitter.push(&rec, now);
                }
                continue;
            }
            // FlexFEC repair packet (RFC 8627): same, via its own decoder.
            if cfg.flexfec_pt == Some(pt) {
                sender.flexfec_decoder.push_fec(payload);
                for rec in sender.flexfec_decoder.take_recovered() {
                    sender.jitter.push(&rec, now);
                }
                continue;
            }
            // RFC 4588: a packet on the RTX payload type carries an original
            // packet (sequence prepended); rebuild it onto the media stream's
            // SSRC before reordering, so the resend simply fills its gap. Drop it
            // if no original SSRC is known yet.
            // PT 0 means "not configured" (the property half-set state, see
            // `set_recv_prop`), never a real RTX stream.
            let is_rtx = cfg
                .rtx
                .is_some_and(|(rtx_pt, _)| rtx_pt != 0 && pt == rtx_pt);
            let reconstructed = match (is_rtx, cfg.rtx, media_ssrc_seen) {
                (true, Some((_, apt)), Some(ssrc)) => rtx::parse_rtx_packet(payload, apt, ssrc),
                _ => None,
            };
            if is_rtx && reconstructed.is_none() {
                continue; // unusable RTX packet (no media SSRC yet / malformed)
            }
            let pkt: &[u8] = reconstructed.as_deref().unwrap_or(payload);

            let media_ssrc = u32::from_be_bytes([pkt[8], pkt[9], pkt[10], pkt[11]]);
            let pkt_seq = u16::from_be_bytes([pkt[2], pkt[3]]);
            let rtp_ts = u32::from_be_bytes([pkt[4], pkt[5], pkt[6], pkt[7]]);
            // RFC 4588 counts a resend in the RTX stream's reception stats, not the original's
            if !is_rtx {
                match ssrc_lock.admit(media_ssrc, now, cfg.jitter.max_hold_ns) {
                    SsrcAdmission::Rejected => continue,
                    SsrcAdmission::Replaced { silence_ns } => {
                        sender = SenderReceiveState::new(cfg);
                        emitter.restart_timeline_after(silence_ns);
                    }
                    SsrcAdmission::Locked => {}
                }
                media_ssrc_seen = Some(media_ssrc);
                sender.stats.on_rtp(media_ssrc, pkt_seq, rtp_ts, now);
            }
            peer = Some(from);
            if let SequenceContinuity::Restarted { silence_ns } = sender.jitter.push(pkt, now) {
                let jitter =
                    core::mem::replace(&mut sender.jitter, RtpJitterBuffer::new(cfg.jitter));
                sender = SenderReceiveState {
                    jitter,
                    ..SenderReceiveState::new(cfg)
                };
                emitter.restart_timeline_after(silence_ns);
            }
            // Record the packet for FEC and inject anything its arrival now lets
            // a buffered repair packet recover.
            if cfg.fec_pt.is_some() {
                sender.fec_decoder.push_media(pkt_seq, pkt);
                for rec in sender.fec_decoder.take_recovered() {
                    sender.jitter.push(&rec, now);
                }
            }
            if cfg.flexfec_pt.is_some() {
                sender.flexfec_decoder.push_media(pkt_seq, pkt);
                for rec in sender.flexfec_decoder.take_recovered() {
                    sender.jitter.push(&rec, now);
                }
            }

            if cfg.nack_enabled {
                let requests = sender
                    .nacks
                    .due(&sender.jitter.missing_seqs(), now, usize::MAX);
                if !requests.is_empty() {
                    let nack = rtcp::build_nack(local_ssrc, media_ssrc, &requests);
                    let _ = receiver.socket().send_to(&nack, from).await;
                }
            }
        }
    }
}

#[cfg(all(test, feature = "rtsp-server"))]
mod tests {
    use super::*;
    use crate::testutil::CollectSink;
    use g2g_core::runtime::block_on;

    const VIDEO_CLOCK_HZ: u32 = 90_000;
    const AUDIO_CLOCK_HZ: u32 = 48_000;
    const NANOS_PER_MS: u64 = 1_000_000;
    // 2026-01-01T00:00:00Z, the instant both sender reports name
    const REPORT_UNIX_NS: i64 = 1_767_225_600_000_000_000;
    const VIDEO_REPORT_RTP: u32 = 1_000;
    const AUDIO_REPORT_RTP: u32 = 5_000;
    // local monotonic time the first sender report arrived at
    const REPORT_ARRIVAL_NS: u64 = 1_000_000_000;

    fn report(rtp_timestamp: u32, clock_rate_hz: u32) -> SenderClock {
        SenderClock {
            unix_nanos: REPORT_UNIX_NS,
            rtp_timestamp,
            clock_rate_hz,
        }
    }

    fn ticks_after(base: u32, clock_rate_hz: u32, ms: u64) -> u32 {
        base + (u64::from(clock_rate_hz) * ms / 1_000) as u32
    }

    fn emitter<'a>(encoding: RtpEncoding, timeline: TrackTimeline<'a>) -> AccessUnitEmitter<'a> {
        let cfg = RtpRecvConfig {
            encoding,
            ..RtpRecvConfig::default()
        };
        AccessUnitEmitter::new(&cfg, 0, u64::MAX, Some(timeline))
    }

    const H264_IDR: [u8; 5] = [0, 0, 0, 1, 0x65];
    const H264_NON_IDR_SLICE: [u8; 5] = [0, 0, 0, 1, 0x41];

    fn access_unit(rtp_timestamp: u32) -> AccessUnit {
        AccessUnit {
            data: Vec::from(H264_IDR),
            rtp_timestamp,
        }
    }

    #[test]
    fn video_waits_for_a_keyframe_at_the_start_and_after_a_loss() {
        let cfg = RtpRecvConfig {
            encoding: RtpEncoding::H264,
            ..RtpRecvConfig::default()
        };
        let mut emitter = AccessUnitEmitter::new(&cfg, 0, u64::MAX, None);
        let mut out = CollectSink::default();
        let units = [
            H264_NON_IDR_SLICE,
            H264_IDR,
            H264_NON_IDR_SLICE,
            H264_NON_IDR_SLICE,
            H264_IDR,
        ];
        block_on(async {
            for (index, data) in units.into_iter().enumerate() {
                // the fourth unit follows a loss
                if index == 3 {
                    emitter.awaiting_keyframe = true;
                }
                let unit = AccessUnit {
                    data: Vec::from(data),
                    rtp_timestamp: index as u32,
                };
                emitter.push(unit, None, &mut out).await.unwrap();
            }
        });
        let emitted: Vec<bool> = out
            .packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some(frame.timing.keyframe),
                _ => None,
            })
            .collect();
        assert_eq!(emitted, [true, false, true]);
    }

    fn pts_ms(sink: &CollectSink) -> Vec<u64> {
        sink.packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some(frame.timing.pts_ns / NANOS_PER_MS),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn sender_reports_put_both_tracks_on_one_timeline() {
        let session = SessionTimeline::default();
        let (video_clock, audio_clock) = (Cell::new(None), Cell::new(None));
        let video = TrackTimeline {
            session: &session,
            sender_clock: &video_clock,
        };
        let audio = TrackTimeline {
            session: &session,
            sender_clock: &audio_clock,
        };
        video.note_sender_report(report(VIDEO_REPORT_RTP, VIDEO_CLOCK_HZ), REPORT_ARRIVAL_NS);
        // A later arrival of the second report moves nothing: both name one instant.
        audio.note_sender_report(
            report(AUDIO_REPORT_RTP, AUDIO_CLOCK_HZ),
            REPORT_ARRIVAL_NS + 3 * NANOS_PER_MS,
        );
        let mut video_emitter = emitter(RtpEncoding::H264, video);
        let mut audio_emitter = emitter(RtpEncoding::Opus, audio);
        let (mut video_out, mut audio_out) = (CollectSink::default(), CollectSink::default());

        block_on(async {
            // Video's first frame, 100 ms after the reports, is the session origin.
            for ms in [100, 200] {
                let rtp = ticks_after(VIDEO_REPORT_RTP, VIDEO_CLOCK_HZ, ms);
                video_emitter
                    .push(access_unit(rtp), None, &mut video_out)
                    .await
                    .unwrap();
            }
            // Audio's frame at 40 ms precedes the origin and is dropped.
            for ms in [40, 140] {
                let rtp = ticks_after(AUDIO_REPORT_RTP, AUDIO_CLOCK_HZ, ms);
                audio_emitter
                    .push(access_unit(rtp), None, &mut audio_out)
                    .await
                    .unwrap();
            }
        });

        assert_eq!(pts_ms(&video_out), [0, 100]);
        assert_eq!(pts_ms(&audio_out), [40]);
    }

    #[test]
    fn a_track_without_a_sender_report_is_placed_by_arrival() {
        let session = SessionTimeline::default();
        let (video_clock, audio_clock) = (Cell::new(None), Cell::new(None));
        let video = TrackTimeline {
            session: &session,
            sender_clock: &video_clock,
        };
        let audio = TrackTimeline {
            session: &session,
            sender_clock: &audio_clock,
        };
        video.note_sender_report(report(VIDEO_REPORT_RTP, VIDEO_CLOCK_HZ), REPORT_ARRIVAL_NS);
        // The frame the report names maps to the report's own arrival.
        assert_eq!(
            video.session_offset_ns(VIDEO_REPORT_RTP, REPORT_ARRIVAL_NS + 7 * NANOS_PER_MS),
            0
        );
        let audio_arrival_ns = REPORT_ARRIVAL_NS + 250 * NANOS_PER_MS;
        assert_eq!(
            audio.session_offset_ns(AUDIO_REPORT_RTP, audio_arrival_ns),
            (250 * NANOS_PER_MS) as i64
        );
    }
}
