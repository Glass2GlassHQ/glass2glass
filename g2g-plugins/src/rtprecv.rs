//! Shared RTP H.264 receive loop (M520): the jitter-buffer + RTCP RR/NACK +
//! FEC/RTX + depayload-to-access-unit path used by both [`UdpSrc`](crate::udpsrc)
//! (raw RTP) and [`RtspServerSrc`](crate::rtspserversrc) (RTSP RECORD ingest).
//!
//! Given a bound UDP socket it reorders (jitter buffer), recovers (ULPFEC /
//! FlexFEC / RTX), and depayloads RTP into Annex-B access units, pushing each as
//! a `CompressedVideo` H.264 frame downstream, while driving RTCP receiver
//! reports and Generic NACK retransmission requests back to the sender.
//!
//! Both ingest sources differ only in how they obtain the socket (a raw bind vs
//! the RTP port negotiated over an RTSP control channel); the receive path is
//! identical, so it lives here once.

use alloc::vec;

use std::net::SocketAddr;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket};

use crate::filesink::io_err;
use crate::flexfec::FlexFecDecoder;
use crate::rtcp::{self, ReceptionStats, RtcpPacket};
use crate::rtpdepay::{AccessUnit, RtpH264Depayloader};
use crate::rtpjitter::{JitterConfig, RtpJitterBuffer};
use crate::rtx;
use crate::ulpfec::FecDecoder;

/// H.264 RTP media clock (RFC 6184): timestamps tick at 90 kHz.
const RTP_CLOCK_HZ: u64 = 90_000;
/// Receive buffer: a UDP datagram tops out at 65507 payload bytes.
const RECV_BUF: usize = 65_536;
/// Default receiver-report cadence.
pub const DEFAULT_RR_INTERVAL_MS: u64 = 1000;
/// Minimum spacing between NACK feedback packets, so a persistent gap is not
/// re-requested every datagram (give a retransmit time to arrive).
const NACK_MIN_INTERVAL_NS: u64 = 20_000_000;
/// Our own SSRC as the RTCP reporter (`g2g` + 1).
const LOCAL_SSRC: u32 = 0x6732_6701;

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
        }
    }
}

/// Turn one depayloaded access unit into a downstream H.264 frame and push it,
/// tracking the RTP-timestamp rebase (`ts_base`, so downstream PTS starts near
/// zero) and the emitted sequence counter. Returns `Ok(true)` once `frame_limit`
/// access units have been emitted (after pushing `Eos`), so the caller stops.
/// Shared by the UDP ([`receive_rtp_h264`]) and TCP-interleaved
/// ([`crate::rtspserversrc`]) ingest paths, which differ only in how they obtain
/// each RTP packet.
pub(crate) async fn push_access_unit(
    au: AccessUnit,
    ts_base: &mut Option<u32>,
    seq: &mut u64,
    seq_base: u64,
    frame_limit: u64,
    out: &mut dyn OutputSink,
) -> Result<bool, G2gError> {
    let base = *ts_base.get_or_insert(au.rtp_timestamp);
    let rel = au.rtp_timestamp.wrapping_sub(base) as u64;
    let pts = rel * 1_000_000_000 / RTP_CLOCK_HZ;
    let arrival_ns = g2g_core::metrics::monotonic_ns();
    // IDR NAL => keyframe; false (the safe default) otherwise.
    let keyframe = crate::h264util::h264_au_is_keyframe(&au.data);
    let frame = Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(au.data.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: pts,
            dts_ns: pts,
            duration_ns: 0,
            capture_ns: pts,
            arrival_ns,
            keyframe,
        },
        sequence: *seq,
        meta: Default::default(),
    };
    out.push(PipelinePacket::DataFrame(frame)).await?;
    *seq += 1;
    if frame_limit != 0 && seq.saturating_sub(seq_base) >= frame_limit {
        out.push(PipelinePacket::Eos).await?;
        return Ok(true);
    }
    Ok(false)
}

/// Run the RTP H.264 receive loop on `socket` until `frame_limit` access units
/// have been emitted (0 = run until a socket error or downstream shutdown),
/// pushing each completed access unit downstream and emitting `Eos` on the
/// bounded path. `seq_base` is the sequence number the first emitted frame
/// carries. Returns the number of access units pushed.
pub async fn receive_rtp_h264(
    socket: &tokio::net::UdpSocket,
    cfg: &RtpRecvConfig,
    frame_limit: u64,
    seq_base: u64,
    out: &mut dyn OutputSink,
) -> Result<u64, G2gError> {
    let mut depay = RtpH264Depayloader::new();
    let mut jitter = RtpJitterBuffer::new(cfg.jitter);
    let mut stats = ReceptionStats::new(0, RTP_CLOCK_HZ as u32);
    let mut buf = vec![0u8; RECV_BUF];
    let mut seq = seq_base;
    // RTP timestamps start at a random offset; rebase so downstream sees PTS
    // near zero.
    let mut ts_base: Option<u32> = None;
    // The original media stream's SSRC, learned from the first non-RTX packet;
    // an RFC 4588 RTX resend is restamped back onto it.
    let mut media_ssrc_seen: Option<u32> = None;
    // FEC decoders; each consulted only when its payload type is set.
    let mut fec_decoder = FecDecoder::default();
    let mut flexfec_decoder = FlexFecDecoder::default();

    // RTCP feedback state (RTP/RTCP-muxed on this socket): the peer to report to
    // (learned from the first datagram) and report timers.
    let rtcp_on = cfg.rtcp_rr_interval_ms > 0;
    let rr_interval_ns = cfg.rtcp_rr_interval_ms * 1_000_000;
    let mut peer: Option<SocketAddr> = None;
    let mut last_rr_ns = g2g_core::metrics::monotonic_ns();
    let mut last_nack_ns = 0u64;

    loop {
        // Drain every packet the jitter buffer will release now, turning each
        // completed access unit into a downstream frame.
        while let Some(packet) = jitter.pop(g2g_core::metrics::monotonic_ns()) {
            let Some(au) = depay.depacketize(&packet) else {
                continue;
            };
            if push_access_unit(au, &mut ts_base, &mut seq, seq_base, frame_limit, out).await? {
                return Ok(seq - seq_base);
            }
        }

        // Periodic receiver report back to the peer (best-effort).
        let now = g2g_core::metrics::monotonic_ns();
        if rtcp_on {
            if let Some(dest) = peer {
                if now.saturating_sub(last_rr_ns) >= rr_interval_ns {
                    let rr = rtcp::build_receiver_report(LOCAL_SSRC, &[stats.report_block(now)]);
                    let _ = socket.send_to(&rr, dest).await;
                    last_rr_ns = now;
                }
            }
        }

        // Wait for the next datagram, but no longer than the soonest of the
        // jitter hold deadline (flush a gap) and the next report.
        let now = g2g_core::metrics::monotonic_ns();
        let jitter_deadline = jitter.next_deadline_ns(now);
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
                tokio::time::timeout(
                    core::time::Duration::from_nanos(delay),
                    socket.recv_from(&mut buf),
                )
                .await
                .ok()
            }
            // Deadline already elapsed (delay == 0): don't block on recv; loop
            // back so the jitter flush / receiver report at the top of the loop
            // fires now instead of waiting for a packet.
            Some(_) => None,
            None => Some(socket.recv_from(&mut buf).await),
        };

        let Some(r) = received else { continue };
        let (n, from) = r.map_err(io_err)?;
        let now = g2g_core::metrics::monotonic_ns();

        // RTCP (muxed) feedback from the sender: a sender report fills the
        // LSR/DLSR fields for round-trip estimation.
        if rtcp_on && rtcp::is_rtcp(&buf[..n]) {
            for p in rtcp::parse_compound(&buf[..n]) {
                if let RtcpPacket::SenderReport { ntp, .. } = p {
                    stats.on_sender_report(ntp, now);
                }
            }
            continue;
        }

        // RTP media: account it for reception stats, buffer it for reordering,
        // and remember the peer for feedback.
        if n >= 12 {
            let pt = buf[1] & 0x7F;
            // ULPFEC repair packet: feed the decoder and inject any recovered
            // media into the jitter buffer (no NAK round trip).
            if cfg.fec_pt == Some(pt) {
                fec_decoder.push_fec(&buf[..n]);
                for rec in fec_decoder.take_recovered() {
                    jitter.push(&rec, now);
                }
                continue;
            }
            // FlexFEC repair packet (RFC 8627): same, via its own decoder.
            if cfg.flexfec_pt == Some(pt) {
                flexfec_decoder.push_fec(&buf[..n]);
                for rec in flexfec_decoder.take_recovered() {
                    jitter.push(&rec, now);
                }
                continue;
            }
            // RFC 4588: a packet on the RTX payload type carries an original
            // packet (sequence prepended); rebuild it onto the media stream's
            // SSRC before reordering, so the resend simply fills its gap. Drop it
            // if no original SSRC is known yet.
            let is_rtx = cfg.rtx.is_some_and(|(rtx_pt, _)| pt == rtx_pt);
            let reconstructed = match (is_rtx, cfg.rtx, media_ssrc_seen) {
                (true, Some((_, apt)), Some(ssrc)) => rtx::parse_rtx_packet(&buf[..n], apt, ssrc),
                _ => None,
            };
            if is_rtx && reconstructed.is_none() {
                continue; // unusable RTX packet (no media SSRC yet / malformed)
            }
            let pkt: &[u8] = reconstructed.as_deref().unwrap_or(&buf[..n]);

            let media_ssrc = u32::from_be_bytes([pkt[8], pkt[9], pkt[10], pkt[11]]);
            let pkt_seq = u16::from_be_bytes([pkt[2], pkt[3]]);
            let rtp_ts = u32::from_be_bytes([pkt[4], pkt[5], pkt[6], pkt[7]]);
            if !is_rtx {
                media_ssrc_seen = Some(media_ssrc);
            }
            stats.on_rtp(media_ssrc, pkt_seq, rtp_ts, now);
            peer = Some(from);
            jitter.push(pkt, now);
            // Record the packet for FEC and inject anything its arrival now lets
            // a buffered repair packet recover.
            if cfg.fec_pt.is_some() {
                fec_decoder.push_media(pkt_seq, pkt);
                for rec in fec_decoder.take_recovered() {
                    jitter.push(&rec, now);
                }
            }
            if cfg.flexfec_pt.is_some() {
                flexfec_decoder.push_media(pkt_seq, pkt);
                for rec in flexfec_decoder.take_recovered() {
                    jitter.push(&rec, now);
                }
            }

            // Request retransmission of any open gaps, rate-limited.
            if cfg.nack_enabled && now.saturating_sub(last_nack_ns) >= NACK_MIN_INTERVAL_NS {
                let missing = jitter.missing_seqs();
                if !missing.is_empty() {
                    let nack = rtcp::build_nack(LOCAL_SSRC, media_ssrc, &missing);
                    let _ = socket.send_to(&nack, from).await;
                    last_nack_ns = now;
                }
            }
        }
    }
}
