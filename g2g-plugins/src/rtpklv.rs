//! Sans-IO KLV RTP payloader / depayloader (RFC 3550 header + RFC 6597 SMPTE ST
//! 336 payload format), the low-latency alternative to carrying metadata in a
//! transport stream. Same shape as the H.264 pair
//! ([`rtppay`](crate::rtppay) / [`rtpdepay`](crate::rtpdepay)): no I/O, a UDP
//! sink or source wraps these and moves the datagrams.
//!
//! RFC 6597 is a thin format: the RTP payload is the KLVunit bytes verbatim,
//! with no payload header of its own. A KLVunit (one or more ST 336 packets
//! sampled at one instant) that exceeds the MTU is split across consecutive
//! packets sharing one timestamp, and the marker bit is set on the last packet
//! of the unit. The clock rate is 90 kHz, so a KLV stream and the video it
//! describes carry comparable timestamps.
//!
//! Because there is no per-fragment header, a receiver cannot tell a middle
//! fragment from the start of a unit; RFC 6597 therefore requires discarding a
//! whole KLVunit when any of its fragments is missing. The depayloader detects
//! that from the RTP sequence number, then discards up to and including the next
//! marker (the end of the damaged unit) so a loss never yields half a unit.

use alloc::vec::Vec;

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};

/// RFC 6597 media clock: KLV RTP timestamps tick at 90 kHz.
pub const RTP_CLOCK_HZ: u64 = 90_000;

/// Cap the in-flight reassembly buffer. A fragmented KLVunit whose marker never
/// arrives would otherwise grow without bound on untrusted RTP; past this the
/// in-flight bytes are dropped and the next marker resyncs. 16 MiB matches the
/// PES bound in `mpegts`.
const MAX_UNIT_BYTES: usize = 16 * 1024 * 1024;

/// A 90 kHz RTP timestamp for a frame presented at `pts_ns`, truncated into the
/// 32-bit RTP field (the wrap is what the wire carries).
pub fn rtp_timestamp_from_pts(pts_ns: u64) -> u32 {
    (pts_ns.wrapping_mul(RTP_CLOCK_HZ) / 1_000_000_000) as u32
}

/// The nanosecond presentation time a 90 kHz RTP timestamp names, relative to
/// whatever epoch the sender's timestamps started from. Rounding is truncating,
/// so a round trip through [`rtp_timestamp_from_pts`] lands within one 90 kHz
/// tick (11.1 us).
pub fn pts_ns_from_rtp(rtp_timestamp: u32) -> u64 {
    rtp_timestamp as u64 * 1_000_000_000 / RTP_CLOCK_HZ
}

#[derive(Debug, Clone)]
pub struct RtpKlvPacketizer {
    payload_type: u8,
    ssrc: u32,
    sequence: u16,
    /// Max RTP payload bytes per packet (the bytes after the 12-byte header).
    max_payload: usize,
}

impl RtpKlvPacketizer {
    /// `payload_type` is the dynamic RTP PT (KLV has no static assignment, so a
    /// signalled 96..=127 as with H.264).
    pub fn new(payload_type: u8, ssrc: u32) -> Self {
        Self {
            payload_type: payload_type & 0x7F,
            ssrc,
            sequence: 0,
            max_payload: 1400,
        }
    }

    /// Max RTP payload bytes per packet. Floored at 1: RFC 6597 adds no payload
    /// header, so every byte of a packet is KLVunit data.
    pub fn with_max_payload(mut self, bytes: usize) -> Self {
        self.max_payload = bytes.max(1);
        self
    }

    /// The sequence number the next packet will carry. Useful in tests.
    pub fn next_sequence(&self) -> u16 {
        self.sequence
    }

    /// Packetize one KLVunit at `rtp_timestamp` into complete RTP packets.
    /// Sequence numbers increment across packets and calls; every packet of the
    /// unit carries the same timestamp and only the last one is marked.
    pub fn packetize(&mut self, klv_unit: &[u8], rtp_timestamp: u32) -> Vec<Vec<u8>> {
        if klv_unit.is_empty() {
            return Vec::new();
        }
        let count = klv_unit.len().div_ceil(self.max_payload);
        let mut packets = Vec::with_capacity(count);
        for (i, part) in klv_unit.chunks(self.max_payload).enumerate() {
            let header = RtpHeader {
                payload_type: self.payload_type,
                marker: i + 1 == count,
                sequence: self.sequence,
                timestamp: rtp_timestamp,
                ssrc: self.ssrc,
            };
            let mut packet = Vec::with_capacity(RTP_HEADER_LEN + part.len());
            packet.extend_from_slice(&header.to_bytes());
            packet.extend_from_slice(part);
            packets.push(packet);
            self.sequence = self.sequence.wrapping_add(1);
        }
        packets
    }
}

/// One depayloaded KLVunit: the ST 336 bytes plus the 90 kHz RTP timestamp its
/// packets shared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KlvUnit {
    pub data: Vec<u8>,
    pub rtp_timestamp: u32,
}

impl KlvUnit {
    /// Presentation time of this unit in nanoseconds, from its RTP timestamp.
    pub fn pts_ns(&self) -> u64 {
        pts_ns_from_rtp(self.rtp_timestamp)
    }
}

#[derive(Debug, Default)]
pub struct RtpKlvDepayloader {
    /// Bytes accumulated for the KLVunit currently being reassembled.
    unit: Vec<u8>,
    /// Set after a loss: the in-flight unit is unrecoverable and incoming
    /// payloads are discarded through the marker that ends it.
    dropping: bool,
    /// Last RTP sequence number seen, for gap detection.
    last_seq: Option<u16>,
}

impl RtpKlvDepayloader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one RTP packet. Returns `Some(unit)` when the packet's marker bit
    /// closes a KLVunit, otherwise accumulates and returns `None`. A malformed
    /// datagram, or a unit with a missing fragment (RFC 6597: discard it whole),
    /// yields nothing.
    pub fn depacketize(&mut self, packet: &[u8]) -> Option<KlvUnit> {
        let parsed = RtpHeader::parse(packet)?;
        let seq = parsed.header.sequence;
        let marker = parsed.header.marker;

        if let Some(prev) = self.last_seq {
            if seq != prev.wrapping_add(1) {
                self.unit.clear();
                self.dropping = true;
            }
        }
        self.last_seq = Some(seq);

        let payload =
            packet.get(parsed.payload_offset..parsed.payload_offset + parsed.payload_len)?;
        if payload.is_empty() {
            return None;
        }

        if self.dropping {
            // Nothing in a damaged unit is usable and a fragment carries no
            // header saying which unit it belongs to, so discard through the
            // marker that ends it and start clean on the next packet.
            self.dropping = !marker;
            return None;
        }

        if self.unit.len().saturating_add(payload.len()) > MAX_UNIT_BYTES {
            // A unit whose marker never arrives; drop it rather than buffer
            // without bound, and resync at the next marker.
            self.unit.clear();
            self.dropping = true;
            return None;
        }
        self.unit.extend_from_slice(payload);

        marker.then(|| KlvUnit {
            data: core::mem::take(&mut self.unit),
            rtp_timestamp: parsed.header.timestamp,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_bytes(len: usize) -> Vec<u8> {
        (0..len as u32).map(|i| i as u8).collect()
    }

    fn marker(packet: &[u8]) -> bool {
        packet[1] & 0x80 != 0
    }

    #[test]
    fn small_unit_is_one_marked_packet() {
        let mut pay = RtpKlvPacketizer::new(98, 0x1234_5678);
        let unit = unit_bytes(20);
        let packets = pay.packetize(&unit, 9000);
        assert_eq!(packets.len(), 1);
        let pkt = &packets[0];
        assert_eq!(pkt[0], 0x80, "V=2, no padding/extension/CSRC");
        assert_eq!(pkt[1] & 0x7F, 98, "payload type");
        assert!(marker(pkt), "marker closes the KLVunit");
        assert_eq!(
            &pkt[RTP_HEADER_LEN..],
            &unit[..],
            "KLVunit carried verbatim, no payload header"
        );
    }

    #[test]
    fn oversized_unit_fragments_with_one_timestamp() {
        let mut pay = RtpKlvPacketizer::new(98, 1).with_max_payload(16);
        let unit = unit_bytes(100);
        let packets = pay.packetize(&unit, 4500);
        assert!(packets.len() > 1, "unit fragments across packets");
        for (i, pkt) in packets.iter().enumerate() {
            assert_eq!(
                u32::from_be_bytes([pkt[4], pkt[5], pkt[6], pkt[7]]),
                4500,
                "one timestamp for the whole unit"
            );
            assert_eq!(
                u16::from_be_bytes([pkt[2], pkt[3]]),
                i as u16,
                "sequence increments per fragment"
            );
            assert_eq!(
                marker(pkt),
                i + 1 == packets.len(),
                "marker on the last only"
            );
        }
        let mut depay = RtpKlvDepayloader::new();
        let out: Vec<_> = packets
            .iter()
            .filter_map(|p| depay.depacketize(p))
            .collect();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].data, unit, "fragments reassemble the unit");
    }

    #[test]
    fn unterminated_unit_is_bounded_and_resyncs() {
        let mut depay = RtpKlvDepayloader::new();
        let mut pay = RtpKlvPacketizer::new(98, 1).with_max_payload(1024 * 1024);
        // 20 MiB with no marker in sight: the buffer must cap instead of growing.
        let huge = unit_bytes(20 * 1024 * 1024);
        let mut packets = pay.packetize(&huge, 1);
        packets.pop(); // drop the marked tail so the unit never closes
        for p in &packets {
            assert!(depay.depacketize(p).is_none());
        }
        // The next marker ends the discarded unit; the one after it decodes.
        let unit = unit_bytes(8);
        assert!(pay
            .packetize(&unit, 2)
            .iter()
            .all(|p| depay.depacketize(p).is_none()));
        let next = pay.packetize(&unit, 3);
        let out: Vec<_> = next.iter().filter_map(|p| depay.depacketize(p)).collect();
        assert_eq!(out.len(), 1, "resyncs once the damaged unit is behind us");
        assert_eq!(out[0].data, unit);
        assert_eq!(out[0].rtp_timestamp, 3);
    }

    #[test]
    fn pts_round_trips_through_the_90khz_clock() {
        for pts in [0u64, 33_366_667, 1_000_000_000, 42_123_456_789] {
            let ts = rtp_timestamp_from_pts(pts);
            let back = pts_ns_from_rtp(ts);
            assert!(
                pts.abs_diff(back) < 1_000_000_000 / RTP_CLOCK_HZ + 1,
                "pts {pts} -> {ts} -> {back} beyond one 90 kHz tick"
            );
        }
    }
}
