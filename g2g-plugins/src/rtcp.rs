//! Sans-IO RTCP (RFC 3550) plus the RTPFB Generic NACK feedback message
//! (RFC 4585), the control-protocol half of the RTP receive-side stack. It
//! builds and parses the compound packets that flow alongside the media:
//! receiver reports (loss / jitter feedback), sender reports (for round-trip
//! timing), BYE (clean termination), and Generic NACK (request retransmission
//! of lost sequence numbers).
//!
//! [`ReceptionStats`] tracks the RFC 3550 reception statistics for one source
//! (extended highest sequence, cumulative + interval loss, interarrival jitter)
//! and emits a [`ReportBlock`] on demand. No I/O and no clock: the caller feeds
//! per-packet arrival times and pumps the built bytes onto a socket. `UdpSrc`
//! (receiver) and `UdpSink` (sender) wire it to a socket; the unit tests drive
//! it directly.

use alloc::vec::Vec;

/// RTCP packet types (the `PT` byte).
pub const PT_SR: u8 = 200;
pub const PT_RR: u8 = 201;
pub const PT_SDES: u8 = 202;
pub const PT_BYE: u8 = 203;
/// Transport-layer feedback (RFC 4585); Generic NACK is `FMT == 1`.
pub const PT_RTPFB: u8 = 205;
pub const FMT_GENERIC_NACK: u8 = 1;

/// True if a datagram on an RTP/RTCP-muxed socket (RFC 5761) is RTCP: the
/// packet-type byte lands in the 200..=207 range that RTP payload types avoid.
pub fn is_rtcp(buf: &[u8]) -> bool {
    buf.len() >= 2 && (200..=207).contains(&buf[1])
}

/// One RFC 3550 report block: per-source reception quality, carried in SR/RR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReportBlock {
    /// SSRC the statistics describe.
    pub ssrc: u32,
    /// Loss fraction since the previous report, as an 8-bit fixed-point
    /// numerator over 256.
    pub fraction_lost: u8,
    /// Cumulative packets lost (24-bit signed in the wire format; widened here).
    pub cumulative_lost: u32,
    /// Extended highest sequence number received.
    pub highest_seq: u32,
    /// Interarrival jitter, in media-clock units.
    pub jitter: u32,
    /// Middle 32 bits of the NTP timestamp of the last SR from this source.
    pub last_sr: u32,
    /// Delay since the last SR, in units of 1/65536 s.
    pub delay_since_last_sr: u32,
}

/// A parsed RTCP packet (only the fields this stack consumes are surfaced).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtcpPacket {
    SenderReport { ssrc: u32, ntp: u64, rtp_ts: u32, reports: Vec<ReportBlock> },
    ReceiverReport { ssrc: u32, reports: Vec<ReportBlock> },
    /// RTPFB Generic NACK: the media sender SSRC plus the lost sequence numbers.
    Nack { sender_ssrc: u32, media_ssrc: u32, missing: Vec<u16> },
    Bye { ssrc: Vec<u32> },
    /// A type we don't decode (e.g. SDES); kept so compound parsing can skip it.
    Other { pt: u8 },
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

/// Write the 4-byte RTCP header. `length` is filled by the caller afterward;
/// here we reserve it and patch once the body length is known.
fn header(out: &mut Vec<u8>, count: u8, pt: u8) {
    out.push(0x80 | (count & 0x1F)); // V=2, P=0, RC/FMT=count
    out.push(pt);
    out.push(0); // length hi (patched)
    out.push(0); // length lo (patched)
}

/// Patch the length field of the RTCP packet that starts at `start`: length is
/// in 32-bit words minus one (RFC 3550).
fn patch_length(out: &mut [u8], start: usize) {
    let words = ((out.len() - start) / 4 - 1) as u16;
    out[start + 2..start + 4].copy_from_slice(&words.to_be_bytes());
}

fn write_report_block(out: &mut Vec<u8>, b: &ReportBlock) {
    push_u32(out, b.ssrc);
    push_u32(out, (b.fraction_lost as u32) << 24 | (b.cumulative_lost & 0x00FF_FFFF));
    push_u32(out, b.highest_seq);
    push_u32(out, b.jitter);
    push_u32(out, b.last_sr);
    push_u32(out, b.delay_since_last_sr);
}

/// Build a Receiver Report (PT 201) from `reporter_ssrc` carrying `blocks`.
pub fn build_receiver_report(reporter_ssrc: u32, blocks: &[ReportBlock]) -> Vec<u8> {
    let mut out = Vec::new();
    header(&mut out, blocks.len() as u8, PT_RR);
    push_u32(&mut out, reporter_ssrc);
    for b in blocks {
        write_report_block(&mut out, b);
    }
    patch_length(&mut out, 0);
    out
}

/// Build a Sender Report (PT 200): NTP + RTP timestamp and sender counters,
/// optionally with reception blocks.
pub fn build_sender_report(
    ssrc: u32,
    ntp: u64,
    rtp_ts: u32,
    packet_count: u32,
    octet_count: u32,
    blocks: &[ReportBlock],
) -> Vec<u8> {
    let mut out = Vec::new();
    header(&mut out, blocks.len() as u8, PT_SR);
    push_u32(&mut out, ssrc);
    out.extend_from_slice(&ntp.to_be_bytes());
    push_u32(&mut out, rtp_ts);
    push_u32(&mut out, packet_count);
    push_u32(&mut out, octet_count);
    for b in blocks {
        write_report_block(&mut out, b);
    }
    patch_length(&mut out, 0);
    out
}

/// Build a BYE (PT 203) for one source.
pub fn build_bye(ssrc: u32) -> Vec<u8> {
    let mut out = Vec::new();
    header(&mut out, 1, PT_BYE);
    push_u32(&mut out, ssrc);
    patch_length(&mut out, 0);
    out
}

/// Build an RTPFB Generic NACK (PT 205, FMT 1) requesting `missing` sequence
/// numbers from `media_ssrc`. Consecutive losses pack into one FCI word via the
/// 16-bit bitmask (BLP); each word covers a PID plus the 16 sequences after it.
pub fn build_nack(sender_ssrc: u32, media_ssrc: u32, missing: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    header(&mut out, FMT_GENERIC_NACK, PT_RTPFB);
    push_u32(&mut out, sender_ssrc);
    push_u32(&mut out, media_ssrc);

    let mut sorted = missing.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut i = 0;
    while i < sorted.len() {
        let pid = sorted[i];
        let mut blp: u16 = 0;
        let mut j = i + 1;
        // Fold any of the next 16 sequences into this word's bitmask.
        while j < sorted.len() {
            let delta = sorted[j].wrapping_sub(pid);
            if delta == 0 || delta > 16 {
                break;
            }
            blp |= 1 << (delta - 1);
            j += 1;
        }
        push_u32(&mut out, (pid as u32) << 16 | blp as u32);
        i = j;
    }
    patch_length(&mut out, 0);
    out
}

/// Parse a (possibly compound) RTCP datagram into its constituent packets.
/// Unknown packet types are surfaced as [`RtcpPacket::Other`] so the walk can
/// skip them by their length field.
pub fn parse_compound(buf: &[u8]) -> Vec<RtcpPacket> {
    let mut out = Vec::new();
    let mut off = 0;
    while off + 4 <= buf.len() {
        let count = (buf[off] & 0x1F) as usize;
        let pt = buf[off + 1];
        let words = u16::from_be_bytes([buf[off + 2], buf[off + 3]]) as usize;
        let total = (words + 1) * 4;
        let end = off + total;
        if total < 4 || end > buf.len() {
            break;
        }
        let body = &buf[off + 4..end];
        match pt {
            PT_RR => {
                if body.len() >= 4 {
                    let ssrc = be32(body, 0);
                    let reports = parse_report_blocks(&body[4..], count);
                    out.push(RtcpPacket::ReceiverReport { ssrc, reports });
                }
            }
            PT_SR => {
                if body.len() >= 24 {
                    let ssrc = be32(body, 0);
                    let ntp = u64::from_be_bytes(body[4..12].try_into().unwrap());
                    let rtp_ts = be32(body, 12);
                    let reports = parse_report_blocks(&body[24..], count);
                    out.push(RtcpPacket::SenderReport { ssrc, ntp, rtp_ts, reports });
                }
            }
            PT_RTPFB if count as u8 == FMT_GENERIC_NACK => {
                if body.len() >= 8 {
                    let sender_ssrc = be32(body, 0);
                    let media_ssrc = be32(body, 4);
                    let mut missing = Vec::new();
                    let mut p = 8;
                    while p + 4 <= body.len() {
                        let pid = u16::from_be_bytes([body[p], body[p + 1]]);
                        let blp = u16::from_be_bytes([body[p + 2], body[p + 3]]);
                        missing.push(pid);
                        for bit in 0..16u16 {
                            if blp & (1 << bit) != 0 {
                                missing.push(pid.wrapping_add(bit + 1));
                            }
                        }
                        p += 4;
                    }
                    out.push(RtcpPacket::Nack { sender_ssrc, media_ssrc, missing });
                }
            }
            PT_BYE => {
                let mut ssrc = Vec::new();
                for k in 0..count {
                    if (k + 1) * 4 <= body.len() {
                        ssrc.push(be32(body, k * 4));
                    }
                }
                out.push(RtcpPacket::Bye { ssrc });
            }
            _ => out.push(RtcpPacket::Other { pt }),
        }
        off = end;
    }
    out
}

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn parse_report_blocks(mut b: &[u8], count: usize) -> Vec<ReportBlock> {
    let mut out = Vec::new();
    for _ in 0..count {
        if b.len() < 24 {
            break;
        }
        let fl_cum = be32(b, 4);
        out.push(ReportBlock {
            ssrc: be32(b, 0),
            fraction_lost: (fl_cum >> 24) as u8,
            cumulative_lost: fl_cum & 0x00FF_FFFF,
            highest_seq: be32(b, 8),
            jitter: be32(b, 12),
            last_sr: be32(b, 16),
            delay_since_last_sr: be32(b, 20),
        });
        b = &b[24..];
    }
    out
}

/// RFC 3550 reception statistics for one source: extended sequence tracking,
/// cumulative + per-interval loss, and the interarrival jitter estimate. Feeds
/// a [`ReportBlock`] for receiver reports.
#[derive(Debug)]
pub struct ReceptionStats {
    source_ssrc: u32,
    clock_hz: u32,
    inited: bool,
    base_seq: u32,
    cycles: u32,
    max_seq16: u16,
    received: u32,
    expected_prior: u32,
    received_prior: u32,
    /// Arrival of the first packet, ns; subsequent arrivals rebase off it so the
    /// media-unit conversion stays in range.
    arrival0_ns: u64,
    have_transit: bool,
    transit: i64,
    /// Interarrival jitter scaled by 16 (fixed-point), per the RFC update rule.
    jitter_x16: u32,
    last_sr_mid: u32,
    last_sr_arrival_ns: u64,
}

impl ReceptionStats {
    pub fn new(source_ssrc: u32, clock_hz: u32) -> Self {
        Self {
            source_ssrc,
            clock_hz,
            inited: false,
            base_seq: 0,
            cycles: 0,
            max_seq16: 0,
            received: 0,
            expected_prior: 0,
            received_prior: 0,
            arrival0_ns: 0,
            have_transit: false,
            transit: 0,
            jitter_x16: 0,
            last_sr_mid: 0,
            last_sr_arrival_ns: 0,
        }
    }

    /// The SSRC these statistics describe (0 until the first packet is seen).
    pub fn source_ssrc(&self) -> u32 {
        self.source_ssrc
    }

    /// Account one received RTP packet: sequence (for loss), RTP timestamp +
    /// arrival (for jitter). `ssrc` adopts the stream's source on the first
    /// packet.
    pub fn on_rtp(&mut self, ssrc: u32, seq: u16, rtp_ts: u32, arrival_ns: u64) {
        if !self.inited {
            self.inited = true;
            self.source_ssrc = ssrc;
            self.base_seq = seq as u32;
            self.max_seq16 = seq;
            self.arrival0_ns = arrival_ns;
            self.received = 1;
        } else {
            self.received = self.received.wrapping_add(1);
            let delta = seq.wrapping_sub(self.max_seq16) as i16;
            if delta > 0 {
                // Forward step; a numeric drop means the 16-bit field wrapped.
                if seq < self.max_seq16 {
                    self.cycles = self.cycles.wrapping_add(1);
                }
                self.max_seq16 = seq;
            }
        }

        // Interarrival jitter (RFC 3550 6.4.1): D is the change in transit time
        // between consecutive packets; J += (|D| - J)/16. Arrival is rebased and
        // converted to media-clock units so the subtraction stays small.
        let rel_ns = arrival_ns.saturating_sub(self.arrival0_ns);
        let arrival_units = ((rel_ns as u128 * self.clock_hz as u128) / 1_000_000_000) as i64;
        let transit = arrival_units - rtp_ts as i64;
        if self.have_transit {
            let d = (transit - self.transit).unsigned_abs() as u32;
            // jitter_x16 += d - jitter_x16/16, i.e. a 1/16-gain low-pass.
            self.jitter_x16 = self.jitter_x16 + d - ((self.jitter_x16 + 8) >> 4);
        }
        self.transit = transit;
        self.have_transit = true;
    }

    /// Record the arrival of a sender report (for the LSR / DLSR fields): store
    /// the middle 32 bits of its NTP timestamp and when it arrived.
    pub fn on_sender_report(&mut self, ntp: u64, arrival_ns: u64) {
        self.last_sr_mid = (ntp >> 16) as u32;
        self.last_sr_arrival_ns = arrival_ns;
    }

    /// Extended highest sequence number received.
    pub fn extended_highest(&self) -> u32 {
        (self.cycles << 16) | self.max_seq16 as u32
    }

    /// Total packets expected so far (highest - base + 1).
    fn expected(&self) -> u32 {
        self.extended_highest().wrapping_sub(self.base_seq).wrapping_add(1)
    }

    /// Cumulative packets lost (expected - received), clamped at 0.
    pub fn cumulative_lost(&self) -> u32 {
        self.expected().saturating_sub(self.received)
    }

    /// Build the report block for this source as of `now_ns`, advancing the
    /// per-interval loss baseline.
    pub fn report_block(&mut self, now_ns: u64) -> ReportBlock {
        let expected = self.expected();
        let expected_interval = expected.wrapping_sub(self.expected_prior);
        let received_interval = self.received.wrapping_sub(self.received_prior);
        self.expected_prior = expected;
        self.received_prior = self.received;
        let lost_interval = expected_interval.saturating_sub(received_interval);
        let fraction_lost = if expected_interval == 0 || lost_interval == 0 {
            0
        } else {
            ((lost_interval << 8) / expected_interval) as u8
        };

        // DLSR in 1/65536 s units, 0 if no SR has been received.
        let dlsr = if self.last_sr_arrival_ns == 0 {
            0
        } else {
            let elapsed_ns = now_ns.saturating_sub(self.last_sr_arrival_ns);
            ((elapsed_ns as u128 * 65536) / 1_000_000_000) as u32
        };

        ReportBlock {
            ssrc: self.source_ssrc,
            fraction_lost,
            cumulative_lost: self.cumulative_lost() & 0x00FF_FFFF,
            highest_seq: self.extended_highest(),
            jitter: self.jitter_x16 >> 4,
            last_sr: self.last_sr_mid,
            delay_since_last_sr: dlsr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn receiver_report_round_trips() {
        let block = ReportBlock {
            ssrc: 0x1234_5678,
            fraction_lost: 12,
            cumulative_lost: 345,
            highest_seq: 0x0001_0009,
            jitter: 77,
            last_sr: 0xABCD_0000,
            delay_since_last_sr: 65536,
        };
        let bytes = build_receiver_report(0xDEAD_BEEF, &[block]);
        assert!(is_rtcp(&bytes));
        let parsed = parse_compound(&bytes);
        assert_eq!(
            parsed,
            alloc::vec![RtcpPacket::ReceiverReport {
                ssrc: 0xDEAD_BEEF,
                reports: alloc::vec![block],
            }]
        );
    }

    #[test]
    fn sender_report_round_trips() {
        let bytes = build_sender_report(0xAAAA, 0x1122_3344_5566_7788, 9000, 100, 20000, &[]);
        let parsed = parse_compound(&bytes);
        assert_eq!(
            parsed,
            alloc::vec![RtcpPacket::SenderReport {
                ssrc: 0xAAAA,
                ntp: 0x1122_3344_5566_7788,
                rtp_ts: 9000,
                reports: Vec::new(),
            }]
        );
    }

    #[test]
    fn nack_packs_consecutive_losses_into_one_word() {
        // 100 and 102 differ by 2: one PID=100 word with bit 1 set in the BLP.
        let bytes = build_nack(1, 2, &[100, 102]);
        let parsed = parse_compound(&bytes);
        let RtcpPacket::Nack { sender_ssrc, media_ssrc, missing } = &parsed[0] else {
            panic!("expected NACK, got {parsed:?}");
        };
        assert_eq!((*sender_ssrc, *media_ssrc), (1, 2));
        assert_eq!(missing, &alloc::vec![100, 102], "PID + bitmask recovers both");
    }

    #[test]
    fn nack_spans_multiple_words_when_losses_are_far_apart() {
        // 10 and 200 are >16 apart: two FCI words, both recovered.
        let bytes = build_nack(1, 2, &[10, 200]);
        let RtcpPacket::Nack { missing, .. } = &parse_compound(&bytes)[0] else {
            panic!("expected NACK");
        };
        assert_eq!(missing, &alloc::vec![10, 200]);
    }

    #[test]
    fn bye_round_trips() {
        let parsed = parse_compound(&build_bye(0x42));
        assert_eq!(parsed, alloc::vec![RtcpPacket::Bye { ssrc: alloc::vec![0x42] }]);
    }

    #[test]
    fn reception_stats_counts_loss_and_fraction() {
        let mut rs = ReceptionStats::new(0, 90_000);
        // Receive seq 0,1,2,4,5 (3 missing) across one interval.
        for (i, seq) in [0u16, 1, 2, 4, 5].into_iter().enumerate() {
            rs.on_rtp(0x99, seq, (i as u32) * 3000, (i as u64) * 33_000_000);
        }
        assert_eq!(rs.extended_highest(), 5);
        assert_eq!(rs.cumulative_lost(), 1, "expected 6 (0..=5), received 5");
        let b = rs.report_block(0);
        assert_eq!(b.ssrc, 0x99, "adopts the stream SSRC");
        // 1 lost of 6 expected -> 256/6 = 42.
        assert_eq!(b.fraction_lost, (256u32 / 6) as u8);
        assert_eq!(b.cumulative_lost, 1);
        assert_eq!(b.highest_seq, 5);
    }

    #[test]
    fn reception_stats_jitter_zero_for_perfectly_paced_stream() {
        // Arrival cadence exactly matches the RTP timestamp cadence -> D is
        // always 0 -> jitter stays 0.
        let mut rs = ReceptionStats::new(0, 90_000);
        for i in 0..10u32 {
            // 3000 ticks = 1/30 s = 33_333_333 ns per step.
            rs.on_rtp(1, i as u16, i * 3000, i as u64 * 33_333_333);
        }
        let b = rs.report_block(0);
        assert!(b.jitter <= 1, "paced stream has ~zero jitter, got {}", b.jitter);
    }

    #[test]
    fn reception_stats_jitter_rises_with_arrival_variance() {
        let mut rs = ReceptionStats::new(0, 90_000);
        // Same RTP cadence but wildly irregular arrivals -> nonzero jitter.
        let arrivals = [0u64, 5_000_000, 90_000_000, 95_000_000, 200_000_000];
        for (i, &a) in arrivals.iter().enumerate() {
            rs.on_rtp(1, i as u16, i as u32 * 3000, a);
        }
        assert!(rs.report_block(0).jitter > 0, "irregular arrivals raise jitter");
    }
}
