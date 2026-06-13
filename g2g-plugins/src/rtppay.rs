//! Sans-IO H.264 RTP packetizer (RFC 3550 RTP header + RFC 6184 payload), the
//! live-egress counterpart of `RtspSrc`'s receive path. Given an Annex-B access
//! unit and its RTP timestamp, it produces complete RTP packets: a single-NAL
//! packet when the NAL fits the MTU, else FU-A fragments. No I/O: a UDP sink
//! wraps this and sends the packets (the I/O follow-up).

use alloc::vec::Vec;

use crate::annexb::nal_units;

/// RTP header size with no CSRC list and no extension.
const RTP_HEADER_LEN: usize = 12;
/// RFC 6184 FU-A NAL type.
const FU_A_TYPE: u8 = 28;

#[derive(Debug, Clone)]
pub struct RtpH264Packetizer {
    payload_type: u8,
    ssrc: u32,
    sequence: u16,
    /// Max RTP payload bytes per packet (the bytes after the 12-byte header).
    max_payload: usize,
}

impl RtpH264Packetizer {
    /// `payload_type` is the dynamic RTP PT (commonly 96..=127 for H.264).
    pub fn new(payload_type: u8, ssrc: u32) -> Self {
        Self {
            payload_type: payload_type & 0x7F,
            ssrc,
            sequence: 0,
            max_payload: 1400,
        }
    }

    /// Max RTP payload bytes per packet. Floored at 3 so an FU-A packet always
    /// carries at least one body byte past its 2-byte header.
    pub fn with_max_payload(mut self, bytes: usize) -> Self {
        self.max_payload = bytes.max(3);
        self
    }

    /// The sequence number the next packet will carry. Useful in tests.
    pub fn next_sequence(&self) -> u16 {
        self.sequence
    }

    /// Packetize one Annex-B access unit at `rtp_timestamp` into complete RTP
    /// packets. Sequence numbers increment across packets and calls; the marker
    /// bit is set on the last packet of the access unit (RFC 6184 frame end).
    pub fn packetize(&mut self, access_unit: &[u8], rtp_timestamp: u32) -> Vec<Vec<u8>> {
        let payloads = self.payloads(access_unit);
        let count = payloads.len();
        let mut packets = Vec::with_capacity(count);
        for (i, payload) in payloads.into_iter().enumerate() {
            let marker = i + 1 == count;
            let mut packet = Vec::with_capacity(RTP_HEADER_LEN + payload.len());
            self.write_header(&mut packet, marker, rtp_timestamp);
            packet.extend_from_slice(&payload);
            packets.push(packet);
            self.sequence = self.sequence.wrapping_add(1);
        }
        packets
    }

    /// One RTP payload per output packet: a whole NAL when it fits, else FU-A
    /// fragments of the oversized NAL.
    fn payloads(&self, access_unit: &[u8]) -> Vec<Vec<u8>> {
        let mut payloads = Vec::new();
        for nal in nal_units(access_unit).filter(|n| !n.is_empty()) {
            if nal.len() <= self.max_payload {
                payloads.push(nal.to_vec());
                continue;
            }
            // FU-A: the original NAL header byte splits into the FU indicator
            // (F | NRI | type=28) and the FU header (S | E | original type).
            let header = nal[0];
            let fu_indicator = (header & 0xE0) | FU_A_TYPE;
            let nal_type = header & 0x1F;
            let body = &nal[1..];
            let chunk = self.max_payload - 2;
            let chunks = body.len().div_ceil(chunk);
            for (i, part) in body.chunks(chunk).enumerate() {
                let start = u8::from(i == 0);
                let end = u8::from(i + 1 == chunks);
                let fu_header = (start << 7) | (end << 6) | nal_type;
                let mut p = Vec::with_capacity(2 + part.len());
                p.push(fu_indicator);
                p.push(fu_header);
                p.extend_from_slice(part);
                payloads.push(p);
            }
        }
        payloads
    }

    fn write_header(&self, out: &mut Vec<u8>, marker: bool, timestamp: u32) {
        out.push(0x80); // V=2, P=0, X=0, CC=0
        out.push((u8::from(marker) << 7) | self.payload_type);
        out.extend_from_slice(&self.sequence.to_be_bytes());
        out.extend_from_slice(&timestamp.to_be_bytes());
        out.extend_from_slice(&self.ssrc.to_be_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wrap(nals: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in nals {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
        }
        out
    }

    fn seq(packet: &[u8]) -> u16 {
        u16::from_be_bytes([packet[2], packet[3]])
    }
    fn timestamp(packet: &[u8]) -> u32 {
        u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]])
    }
    fn marker(packet: &[u8]) -> bool {
        packet[1] & 0x80 != 0
    }
    fn payload(packet: &[u8]) -> &[u8] {
        &packet[RTP_HEADER_LEN..]
    }

    #[test]
    fn single_small_nal_is_one_packet() {
        let mut p = RtpH264Packetizer::new(96, 0xDEAD_BEEF);
        let au = wrap(&[&[0x65, 1, 2, 3]]);
        let packets = p.packetize(&au, 9000);
        assert_eq!(packets.len(), 1);
        let pkt = &packets[0];
        assert_eq!(pkt[0], 0x80, "V=2, no padding/extension/CSRC");
        assert_eq!(pkt[1] & 0x7F, 96, "payload type");
        assert!(marker(pkt), "marker on the last (only) packet");
        assert_eq!(seq(pkt), 0);
        assert_eq!(timestamp(pkt), 9000);
        assert_eq!(
            u32::from_be_bytes([pkt[8], pkt[9], pkt[10], pkt[11]]),
            0xDEAD_BEEF,
            "ssrc"
        );
        assert_eq!(payload(pkt), &[0x65, 1, 2, 3], "single NAL carried verbatim");
    }

    #[test]
    fn two_nals_increment_seq_and_mark_only_last() {
        let mut p = RtpH264Packetizer::new(96, 1);
        let au = wrap(&[&[0x67, 0x42], &[0x65, 9]]);
        let packets = p.packetize(&au, 100);
        assert_eq!(packets.len(), 2);
        assert_eq!(seq(&packets[0]), 0);
        assert_eq!(seq(&packets[1]), 1);
        assert!(!marker(&packets[0]));
        assert!(marker(&packets[1]), "only the AU's last packet is marked");
        assert_eq!(timestamp(&packets[0]), 100);
        assert_eq!(timestamp(&packets[1]), 100, "one timestamp per access unit");
    }

    #[test]
    fn oversized_nal_fragments_into_fu_a_and_reassembles() {
        let mut p = RtpH264Packetizer::new(96, 1).with_max_payload(5);
        // NAL header 0x65: F=0, NRI=3 (0x60), type=5 (IDR); 10-byte body.
        let body: Vec<u8> = (0..10).collect();
        let mut nal = alloc::vec![0x65u8];
        nal.extend_from_slice(&body);
        let packets = p.packetize(&wrap(&[&nal]), 7);
        assert!(packets.len() > 1, "oversized NAL fragments");

        let mut reassembled = Vec::new();
        for (i, pkt) in packets.iter().enumerate() {
            let pl = payload(pkt);
            assert_eq!(pl[0] & 0x1F, FU_A_TYPE, "FU-A indicator type");
            assert_eq!(pl[0] & 0xE0, 0x60, "F|NRI preserved from 0x65");
            assert_eq!(pl[1] & 0x1F, 5, "original NAL type in FU header");
            assert_eq!(pl[1] & 0x80 != 0, i == 0, "start bit on first only");
            assert_eq!(pl[1] & 0x40 != 0, i + 1 == packets.len(), "end bit on last only");
            reassembled.extend_from_slice(&pl[2..]);
        }
        assert_eq!(reassembled, body, "fragments reassemble the NAL body");
        assert!(marker(packets.last().unwrap()), "marker on the last fragment");
        assert!(!marker(&packets[0]));
    }

    #[test]
    fn sequence_persists_across_access_units() {
        let mut p = RtpH264Packetizer::new(96, 1);
        let au = wrap(&[&[0x65, 1]]);
        let _ = p.packetize(&au, 0);
        let next = p.packetize(&au, 3000);
        assert_eq!(seq(&next[0]), 1, "sequence continues across access units");
        assert_eq!(p.next_sequence(), 2);
    }
}
