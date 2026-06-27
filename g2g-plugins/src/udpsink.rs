//! UDP egress sink for H.264 over RTP (M47). The I/O half of the live-egress
//! path whose Sans-IO half is `rtppay::RtpH264Packetizer` (M46): this sink
//! drives the packetizer over each Annex-B access unit and sends the resulting
//! RTP packets to a destination on a UDP socket, the send-side inverse of
//! `RtspSrc`'s receive path.
//!
//! The RTP timestamp is derived from `FrameTiming::pts_ns` at the 90 kHz H.264
//! media clock; sequence numbers and the per-AU marker bit come from the
//! packetizer. Opt-in RFC 3550 sender reports (`with_rtcp_sender_reports`) let a
//! receiver sync the RTP clock to wall time. The RTSP `ANNOUNCE`/`RECORD`
//! handshake for Wowza-style ingest is the remaining live-egress follow-up (it
//! needs the network port the sandbox blocks).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use std::net::{SocketAddr, UdpSocket as StdUdpSocket};

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, VideoCodec,
};

use crate::filesink::io_err;
use crate::rtcp::{self, RtcpPacket};
use crate::rtppay::RtpH264Packetizer;
use crate::rtx;
use crate::flexfec::FlexFecEncoder;
use crate::ulpfec::{FecEncoder, InterleavedFecEncoder};

/// FEC mode for the sink: off, single-level ULPFEC (one repair per contiguous
/// group, recovers one loss per group), interleaved ULPFEC (column repairs
/// recovering a burst), or FlexFEC (RFC 8627, a wide-mask repair on a dedicated
/// FEC SSRC protecting more than 16 packets).
#[derive(Debug)]
enum FecMode {
    None,
    Single(FecEncoder),
    Interleaved(InterleavedFecEncoder),
    FlexFec(FlexFecEncoder),
}

/// H.264 RTP media clock (RFC 6184): timestamps tick at 90 kHz.
const RTP_CLOCK_HZ: u64 = 90_000;
/// Default dynamic RTP payload type for H.264.
const DEFAULT_PAYLOAD_TYPE: u8 = 96;
/// Default max RTP payload bytes, leaving headroom under a 1500-byte MTU.
const DEFAULT_MAX_PAYLOAD: usize = 1400;
/// Default depth of the retransmission history (recently sent packets kept for
/// NACK-triggered resend).
const DEFAULT_RETX_CAPACITY: usize = 1024;

/// The H.264-at-any-geometry caps the sink accepts. Geometry rides in-band in
/// the SPS, so the sink imposes no concrete dimensions.
fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

#[derive(Debug)]
pub struct UdpSink {
    dest: SocketAddr,
    payload_type: u8,
    ssrc: u32,
    max_payload: usize,
    packetizer: Option<RtpH264Packetizer>,
    // Bound synchronously in `configure_pipeline` (no runtime needed); wrapped
    // into the tokio socket lazily on first `process`, where a runtime context
    // is guaranteed (`UdpSocket::from_std` requires one).
    std_socket: Option<StdUdpSocket>,
    socket: Option<tokio::net::UdpSocket>,
    /// Honor RTPFB NACK by resending from the retransmission history.
    retransmit: bool,
    /// RFC 4588 RTX: when set, NACK resends are wrapped in this `(payload type,
    /// SSRC)` with the original sequence prepended, instead of a plain resend.
    rtx: Option<(u8, u32)>,
    /// Sequence counter for the RTX stream (its own numbering space).
    rtx_seq: u16,
    /// RFC 5109 ULPFEC: emits repair packets for latency-free loss recovery
    /// (independent of NACK/RTX). Single-level recovers one loss per group;
    /// interleaved recovers a burst of consecutive losses.
    fec: FecMode,
    /// Recently sent packets, `(sequence, bytes)`, oldest first, capped at
    /// [`retx_cap`](Self::retx_cap). Resent on a matching NACK.
    retx_buf: VecDeque<(u16, Vec<u8>)>,
    retx_cap: usize,
    /// RFC 3550 sender reports: emit an RTCP SR at most this often (ns) on the
    /// RTP/RTCP-muxed socket, so a receiver can sync this SSRC's RTP clock to
    /// wall time. `None` disables them (the default; SR is a session concern a
    /// caller opts into via [`with_rtcp_sender_reports`](Self::with_rtcp_sender_reports)).
    rtcp_sr_interval_ns: Option<u64>,
    last_sr_ns: u64,
    /// Media packets / payload octets sent on `ssrc` (the SR sender counters;
    /// FEC and RTX ride other SSRCs and do not count here). Wrap as RFC fields.
    rtp_packets: u32,
    rtp_octets: u32,
    /// The most recent media RTP timestamp, reported in the SR alongside NTP now.
    last_rtp_ts: u32,
    packets_sent: u64,
    bytes_sent: u64,
    frames_sent: u64,
    retransmits_sent: u64,
    sender_reports_sent: u64,
    eos_seen: bool,
}

impl UdpSink {
    pub fn new(dest: SocketAddr) -> Self {
        Self {
            dest,
            payload_type: DEFAULT_PAYLOAD_TYPE,
            ssrc: 0,
            max_payload: DEFAULT_MAX_PAYLOAD,
            packetizer: None,
            std_socket: None,
            socket: None,
            retransmit: true,
            rtx: None,
            rtx_seq: 0,
            fec: FecMode::None,
            retx_buf: VecDeque::new(),
            retx_cap: DEFAULT_RETX_CAPACITY,
            rtcp_sr_interval_ns: None,
            last_sr_ns: 0,
            rtp_packets: 0,
            rtp_octets: 0,
            last_rtp_ts: 0,
            packets_sent: 0,
            bytes_sent: 0,
            frames_sent: 0,
            retransmits_sent: 0,
            sender_reports_sent: 0,
            eos_seen: false,
        }
    }

    /// Set the RTP payload type (commonly 96..=127) and the synchronization
    /// source identifier carried in every packet.
    pub fn with_rtp(mut self, payload_type: u8, ssrc: u32) -> Self {
        self.payload_type = payload_type;
        self.ssrc = ssrc;
        self
    }

    /// Max RTP payload bytes per packet; larger NALs fragment into FU-A.
    pub fn with_max_payload(mut self, bytes: usize) -> Self {
        self.max_payload = bytes;
        self
    }

    /// Enable/disable NACK-triggered retransmission and size the history of
    /// recently sent packets kept for resend. Default: on, 1024 packets.
    pub fn with_retransmit(mut self, enabled: bool, capacity: usize) -> Self {
        self.retransmit = enabled;
        self.retx_cap = capacity.max(1);
        self
    }

    /// Send NACK resends as RFC 4588 RTX packets on `rtx_payload_type` /
    /// `rtx_ssrc` (the original sequence is prepended) instead of a plain
    /// same-stream resend. The receiver must be told the same `apt` mapping.
    pub fn with_rtx(mut self, rtx_payload_type: u8, rtx_ssrc: u32) -> Self {
        self.rtx = Some((rtx_payload_type & 0x7F, rtx_ssrc));
        self
    }

    /// Emit an RFC 5109 ULPFEC repair packet every `group` media packets on
    /// `fec_payload_type` / `fec_ssrc`, recovering a single per-group loss at the
    /// receiver with no round trip. Complements (does not replace) NACK/RTX.
    pub fn with_fec(mut self, group: usize, fec_payload_type: u8, fec_ssrc: u32) -> Self {
        self.fec = FecMode::Single(FecEncoder::new(group, fec_payload_type, fec_ssrc));
        self
    }

    /// Emit interleaved (column) ULPFEC over a `rows * stride` block on
    /// `fec_payload_type` / `fec_ssrc`: `stride` repair packets per block,
    /// recovering a burst of up to `stride` consecutive losses (`rows * stride`
    /// is capped at 16). The burst-loss counterpart of [`with_fec`](Self::with_fec).
    pub fn with_interleaved_fec(
        mut self,
        rows: usize,
        stride: usize,
        fec_payload_type: u8,
        fec_ssrc: u32,
    ) -> Self {
        self.fec = FecMode::Interleaved(InterleavedFecEncoder::new(
            rows,
            stride,
            fec_payload_type,
            fec_ssrc,
        ));
        self
    }

    /// Emit an RFC 8627 FlexFEC repair every `group` media packets on
    /// `fec_payload_type` / `fec_ssrc` (a dedicated FEC stream). One repair
    /// protects up to 109 packets via the variable-length mask, beyond ULPFEC's
    /// 16; recovers a single loss per group at the receiver with no round trip.
    pub fn with_flexfec(mut self, group: usize, fec_payload_type: u8, fec_ssrc: u32) -> Self {
        self.fec = FecMode::FlexFec(FlexFecEncoder::new(group, fec_payload_type, fec_ssrc));
        self
    }

    /// Emit an RFC 3550 RTCP sender report every `interval_ms` on the RTP socket
    /// (RTP/RTCP-muxed): it carries the NTP wall time, the matching RTP timestamp,
    /// and this SSRC's packet / octet counts, so a receiver can map the RTP clock
    /// to wall time (the basis of inter-stream, e.g. A/V, synchronization). Off by
    /// default. A receiver disambiguates SR from media by RTCP payload type.
    pub fn with_rtcp_sender_reports(mut self, interval_ms: u64) -> Self {
        self.rtcp_sr_interval_ns = Some(interval_ms.saturating_mul(1_000_000));
        self
    }

    /// RTCP sender reports emitted so far.
    pub fn sender_reports_sent(&self) -> u64 {
        self.sender_reports_sent
    }

    /// Retransmitted packets sent in response to NACK feedback.
    pub fn retransmits_sent(&self) -> u64 {
        self.retransmits_sent
    }

    pub fn packets_sent(&self) -> u64 {
        self.packets_sent
    }

    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent
    }

    pub fn frames_sent(&self) -> u64 {
        self.frames_sent
    }

    pub fn eos_seen(&self) -> bool {
        self.eos_seen
    }

    /// 90 kHz RTP timestamp for a presentation time. Wraps the u32 RTP field
    /// as the protocol expects.
    fn rtp_timestamp(pts_ns: u64) -> u32 {
        ((pts_ns as u128 * RTP_CLOCK_HZ as u128) / 1_000_000_000) as u32
    }

    /// The current time as a 64-bit NTP timestamp (seconds since 1900 in the high
    /// 32 bits, fraction in the low 32), for the RTCP sender report.
    fn ntp_now() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        // NTP epoch (1900) precedes the Unix epoch (1970) by this many seconds.
        const NTP_UNIX_OFFSET: u64 = 2_208_988_800;
        let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
        let secs = d.as_secs().wrapping_add(NTP_UNIX_OFFSET);
        let frac = ((d.subsec_nanos() as u64) << 32) / 1_000_000_000;
        (secs << 32) | frac
    }

    fn ensure_socket(&mut self) -> Result<(), G2gError> {
        if self.socket.is_none() {
            let std = self.std_socket.take().ok_or(G2gError::NotConfigured)?;
            self.socket = Some(tokio::net::UdpSocket::from_std(std).map_err(io_err)?);
        }
        Ok(())
    }

    /// Drain any pending RTCP (RTP/RTCP-muxed on the send socket) without
    /// blocking, and retransmit every requested-and-still-buffered packet.
    /// Returns the number of packets resent.
    async fn service_nacks(&mut self) -> Result<u64, G2gError> {
        let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
        // Collect requested packets first so the resend loop does not borrow the
        // history while sending.
        let mut to_resend: Vec<Vec<u8>> = Vec::new();
        let mut rb = [0u8; 1500];
        loop {
            match socket.try_recv(&mut rb) {
                Ok(0) => break,
                Ok(n) => {
                    for p in rtcp::parse_compound(&rb[..n]) {
                        if let RtcpPacket::Nack { missing, .. } = p {
                            for seq in missing {
                                if let Some((_, pkt)) = self.retx_buf.iter().find(|(s, _)| *s == seq)
                                {
                                    to_resend.push(pkt.clone());
                                }
                            }
                        }
                    }
                }
                // WouldBlock (no more datagrams) or a transient error: stop.
                Err(_) => break,
            }
        }
        for pkt in &to_resend {
            // RFC 4588: wrap the resend in an RTX packet (own sequence space),
            // else resend the original packet verbatim.
            if let Some((rtx_pt, rtx_ssrc)) = self.rtx {
                if let Some(wrapped) = rtx::build_rtx_packet(pkt, rtx_pt, rtx_ssrc, self.rtx_seq) {
                    self.rtx_seq = self.rtx_seq.wrapping_add(1);
                    socket.send(&wrapped).await.map_err(io_err)?;
                    continue;
                }
            }
            socket.send(pkt).await.map_err(io_err)?;
        }
        Ok(to_resend.len() as u64)
    }
}

impl AsyncElement for UdpSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&h264_any())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(h264_any()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => {}
            _ => return Err(G2gError::CapsMismatch),
        }
        self.packetizer =
            Some(RtpH264Packetizer::new(self.payload_type, self.ssrc).with_max_payload(self.max_payload));
        let socket = StdUdpSocket::bind(("0.0.0.0", 0)).map_err(io_err)?;
        socket.set_nonblocking(true).map_err(io_err)?;
        socket.connect(self.dest).map_err(io_err)?;
        self.std_socket = Some(socket);
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "UDP RTP sink",
            "Sink/Network",
            "Sends RTP H.264 over UDP, honoring NACK retransmit",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let timestamp = Self::rtp_timestamp(frame.timing.pts_ns);
                    self.last_rtp_ts = timestamp;
                    let packets = {
                        let packetizer = self.packetizer.as_mut().ok_or(G2gError::NotConfigured)?;
                        packetizer.packetize(slice.as_slice(), timestamp)
                    };
                    self.ensure_socket()?;
                    // Generate any ULPFEC repair packets for the just-built media
                    // (before the borrow of `self.socket` below).
                    let mut fec_packets = Vec::new();
                    match &mut self.fec {
                        FecMode::None => {}
                        FecMode::Single(enc) => {
                            for pkt in &packets {
                                if let Some(repair) = enc.push(pkt) {
                                    fec_packets.push(repair);
                                }
                            }
                        }
                        // Interleaved emits a batch of column repairs per block.
                        FecMode::Interleaved(enc) => {
                            for pkt in &packets {
                                fec_packets.extend(enc.push(pkt));
                            }
                        }
                        FecMode::FlexFec(enc) => {
                            for pkt in &packets {
                                if let Some(repair) = enc.push(pkt) {
                                    fec_packets.push(repair);
                                }
                            }
                        }
                    }
                    let mut sent = 0u64;
                    let mut bytes = 0u64;
                    {
                        let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                        for pkt in &packets {
                            socket.send(pkt).await.map_err(io_err)?;
                            sent += 1;
                            bytes += pkt.len() as u64;
                            // SR sender counters: this SSRC's media packets and
                            // their RTP payload octets (past the 12-byte header).
                            self.rtp_packets = self.rtp_packets.wrapping_add(1);
                            self.rtp_octets =
                                self.rtp_octets.wrapping_add(pkt.len().saturating_sub(12) as u32);
                        }
                        // Repair packets follow the media they protect.
                        for repair in &fec_packets {
                            socket.send(repair).await.map_err(io_err)?;
                            sent += 1;
                            bytes += repair.len() as u64;
                        }
                    }
                    // Keep each sent packet in the bounded retransmission history,
                    // keyed by its RTP sequence, for NACK-triggered resend.
                    if self.retransmit {
                        for pkt in packets {
                            let s = u16::from_be_bytes([pkt[2], pkt[3]]);
                            if self.retx_buf.len() >= self.retx_cap {
                                self.retx_buf.pop_front();
                            }
                            self.retx_buf.push_back((s, pkt));
                        }
                    }
                    self.packets_sent += sent;
                    self.bytes_sent += bytes;
                    self.frames_sent += 1;

                    // Service any NACK feedback that arrived (non-blocking),
                    // resending the requested sequences from the history.
                    if self.retransmit {
                        let resent = self.service_nacks().await?;
                        self.retransmits_sent += resent;
                    }

                    // Emit an RTCP sender report when due (rate-limited): NTP now
                    // paired with the last RTP timestamp lets a receiver map this
                    // SSRC's clock to wall time.
                    if let Some(interval) = self.rtcp_sr_interval_ns {
                        let now = g2g_core::metrics::monotonic_ns();
                        if now.saturating_sub(self.last_sr_ns) >= interval {
                            let sr = rtcp::build_sender_report(
                                self.ssrc,
                                Self::ntp_now(),
                                self.last_rtp_ts,
                                self.rtp_packets,
                                self.rtp_octets,
                                &[],
                            );
                            let socket = self.socket.as_ref().ok_or(G2gError::NotConfigured)?;
                            socket.send(&sr).await.map_err(io_err)?;
                            self.last_sr_ns = now;
                            self.sender_reports_sent += 1;
                        }
                    }
                }
                // RTP carries no in-band end marker; an RTCP BYE is the M47
                // follow-up. Sequence numbers persist so a receiver sees clean
                // termination of the flow.
                PipelinePacket::Eos => self.eos_seen = true,
                // A seek does not reset the RTP sequence: a receiver tracks loss
                // by gaps, so the numbering continues across the flush.
                PipelinePacket::Flush => {}
                // Geometry refinement lives in the in-band SPS, not in RTP.
                PipelinePacket::CapsChanged(_) => {}
                // Segment is control: ignored at sink.
                PipelinePacket::Segment(_) => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for UdpSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(h264_any()))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h264(w: u32, h: u32) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    fn rgba(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    fn sink() -> UdpSink {
        UdpSink::new("127.0.0.1:5004".parse().unwrap())
    }

    #[test]
    fn intercept_narrows_h264_and_rejects_raw() {
        let s = sink();
        assert!(s.intercept_caps(&h264(640, 480)).is_ok());
        assert_eq!(
            s.intercept_caps(&rgba(640, 480)),
            Err(G2gError::CapsMismatch),
            "an RTP H.264 packetizer cannot take raw video"
        );
    }

    #[test]
    fn configure_rejects_non_h264_before_binding() {
        let mut s = sink();
        let err = s
            .configure_pipeline(&rgba(640, 480))
            .expect_err("non-h264 caps must be rejected");
        assert_eq!(err, G2gError::CapsMismatch);
        assert!(s.std_socket.is_none(), "no socket bound on a rejected caps");
    }

    #[test]
    fn rtp_timestamp_is_90khz_of_pts() {
        assert_eq!(UdpSink::rtp_timestamp(0), 0);
        // 1 second of pts -> 90000 ticks.
        assert_eq!(UdpSink::rtp_timestamp(1_000_000_000), 90_000);
        // 1/30 s -> 3000 ticks.
        assert_eq!(UdpSink::rtp_timestamp(33_333_333), 2999);
    }
}
