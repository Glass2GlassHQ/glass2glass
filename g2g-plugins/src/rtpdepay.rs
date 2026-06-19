//! Sans-IO H.264 RTP depayloader (RFC 3550 header + RFC 6184 payload), the
//! receive-side inverse of [`rtppay::RtpH264Packetizer`](crate::rtppay). It
//! turns a stream of RTP packets back into Annex-B access units: single-NAL
//! and STAP-A payloads pass through; FU-A fragments reassemble; the RTP marker
//! bit closes an access unit. No I/O: `UdpSrc` wraps this and feeds it the
//! datagrams a socket receives.
//!
//! This is the basic in-order depayloader. A jitter buffer (packet reorder,
//! loss concealment, RTCP) is the larger receive-side follow-up (DESIGN_TODO);
//! out-of-order or lost packets are detected via the sequence number and reset
//! the in-flight reassembly so a gap never welds two access units together.

use alloc::vec::Vec;

/// Minimum RTP header: V/P/X/CC, M/PT, sequence, timestamp, ssrc.
const RTP_HEADER_LEN: usize = 12;
/// RFC 6184 aggregation / fragmentation NAL types.
const STAP_A_TYPE: u8 = 24;
const FU_A_TYPE: u8 = 28;

/// One depayloaded access unit: Annex-B bytes plus the RTP timestamp shared by
/// its packets (90 kHz for H.264).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessUnit {
    pub data: Vec<u8>,
    pub rtp_timestamp: u32,
}

#[derive(Debug, Default)]
pub struct RtpH264Depayloader {
    /// Annex-B NALs accumulated for the access unit currently being assembled.
    au: Vec<u8>,
    /// FU-A reassembly buffer (one NAL spanning multiple packets).
    fu: Vec<u8>,
    fu_active: bool,
    /// Last RTP sequence number seen, for gap detection.
    last_seq: Option<u16>,
    /// RTP timestamp of the in-flight access unit.
    timestamp: u32,
}

impl RtpH264Depayloader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one RTP packet. Returns `Some(access_unit)` when the packet's
    /// marker bit closes an access unit; otherwise accumulates and returns
    /// `None`. Packets that are too short or use unsupported aggregation modes
    /// (STAP-B / MTAP / FU-B) are skipped.
    pub fn depacketize(&mut self, packet: &[u8]) -> Option<AccessUnit> {
        if packet.len() < RTP_HEADER_LEN {
            return None;
        }
        let cc = (packet[0] & 0x0F) as usize;
        let has_ext = packet[0] & 0x10 != 0;
        let marker = packet[1] & 0x80 != 0;
        let seq = u16::from_be_bytes([packet[2], packet[3]]);
        let timestamp = u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]]);

        // A sequence gap (loss / reorder) means the in-flight FU-A and access
        // unit can no longer be trusted; drop them so we never splice across
        // the discontinuity. The next start-of-AU rebuilds cleanly.
        if let Some(prev) = self.last_seq {
            if seq != prev.wrapping_add(1) {
                self.fu.clear();
                self.fu_active = false;
                self.au.clear();
            }
        }
        self.last_seq = Some(seq);
        self.timestamp = timestamp;

        // Skip the CSRC list and a one-word-or-more extension header to find
        // the payload. Most senders emit neither (CC=0, X=0).
        let mut offset = RTP_HEADER_LEN + 4 * cc;
        if has_ext {
            if packet.len() < offset + 4 {
                return None;
            }
            let ext_words = u16::from_be_bytes([packet[offset + 2], packet[offset + 3]]) as usize;
            offset += 4 + 4 * ext_words;
        }
        let payload = packet.get(offset..)?;
        if payload.is_empty() {
            return None;
        }

        match payload[0] & 0x1F {
            // Single NAL unit (types 1..=23): the payload is the NAL verbatim.
            1..=23 => self.push_nal(payload),
            // STAP-A: [type 24][ (16-bit size)(NAL) ]* aggregated NALs, used
            // mostly to carry SPS+PPS in one packet.
            t if t == STAP_A_TYPE => {
                let mut i = 1;
                while i + 2 <= payload.len() {
                    let size = u16::from_be_bytes([payload[i], payload[i + 1]]) as usize;
                    i += 2;
                    let Some(nal) = payload.get(i..i + size) else { break };
                    self.push_nal(nal);
                    i += size;
                }
            }
            // FU-A: a single NAL fragmented across packets. The original NAL
            // header byte is rebuilt from the FU indicator's F|NRI and the FU
            // header's type; Start/End bits bound the fragment run.
            t if t == FU_A_TYPE => {
                if payload.len() < 2 {
                    return None;
                }
                let fu_header = payload[1];
                let start = fu_header & 0x80 != 0;
                let end = fu_header & 0x40 != 0;
                if start {
                    self.fu.clear();
                    self.fu.push((payload[0] & 0xE0) | (fu_header & 0x1F));
                    self.fu_active = true;
                }
                if self.fu_active {
                    self.fu.extend_from_slice(&payload[2..]);
                    if end {
                        let nal = core::mem::take(&mut self.fu);
                        self.push_nal(&nal);
                        self.fu_active = false;
                    }
                }
            }
            // STAP-B (25), MTAP (26/27), FU-B (29): unsupported, skip.
            _ => {}
        }

        if marker && !self.au.is_empty() {
            Some(AccessUnit {
                data: core::mem::take(&mut self.au),
                rtp_timestamp: self.timestamp,
            })
        } else {
            None
        }
    }

    /// Append one NAL to the in-flight access unit as a 4-byte-start-code
    /// Annex-B unit.
    fn push_nal(&mut self, nal: &[u8]) {
        if nal.is_empty() {
            return;
        }
        self.au.extend_from_slice(&[0, 0, 0, 1]);
        self.au.extend_from_slice(nal);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtppay::RtpH264Packetizer;

    fn wrap(nals: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for nal in nals {
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
        }
        out
    }

    #[test]
    fn round_trips_a_small_single_nal_access_unit() {
        let au = wrap(&[&[0x67, 0x42, 0x00], &[0x65, 1, 2, 3, 4]]);
        let mut pkt = RtpH264Packetizer::new(96, 7);
        let mut depay = RtpH264Depayloader::new();
        let packets = pkt.packetize(&au, 9000);

        let mut out = None;
        for p in &packets {
            if let Some(unit) = depay.depacketize(p) {
                out = Some(unit);
            }
        }
        let unit = out.expect("marker closes the access unit");
        assert_eq!(unit.data, au, "depayloaded AU matches the original Annex-B");
        assert_eq!(unit.rtp_timestamp, 9000);
    }

    #[test]
    fn reassembles_fu_a_fragments() {
        // One oversized NAL packetized into FU-A fragments must reassemble to
        // the exact original NAL.
        let mut nal = alloc::vec![0x65u8];
        nal.extend_from_slice(&(0..40u8).collect::<Vec<_>>());
        let au = wrap(&[&nal]);
        let mut pkt = RtpH264Packetizer::new(96, 1).with_max_payload(8);
        let packets = pkt.packetize(&au, 3000);
        assert!(packets.len() > 1, "precondition: NAL actually fragmented");

        let mut depay = RtpH264Depayloader::new();
        let mut out = None;
        for p in &packets {
            if let Some(unit) = depay.depacketize(p) {
                out = Some(unit);
            }
        }
        assert_eq!(out.expect("reassembled AU").data, au);
    }

    #[test]
    fn parses_stap_a_aggregated_nals() {
        // Hand-build a STAP-A packet carrying SPS + PPS, marker set.
        let sps: &[u8] = &[0x67, 0x42, 0x00, 0x1f];
        let pps: &[u8] = &[0x68, 0xce, 0x38, 0x80];
        let mut payload = alloc::vec![STAP_A_TYPE | 0x60];
        for nal in [sps, pps] {
            payload.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            payload.extend_from_slice(nal);
        }
        let mut packet = alloc::vec![0x80u8, 0x80 | 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        packet.extend_from_slice(&payload);

        let mut depay = RtpH264Depayloader::new();
        let unit = depay.depacketize(&packet).expect("marker closes AU");
        assert_eq!(unit.data, wrap(&[sps, pps]), "both NALs in start-code form");
    }

    #[test]
    fn sequence_gap_drops_partial_access_unit() {
        // First an unmarked single-NAL packet (AU still open), then a packet
        // with a non-consecutive sequence: the partial AU is discarded so the
        // gap never welds the two together.
        let mut depay = RtpH264Depayloader::new();
        // seq 0, no marker: starts an AU.
        let p0 = alloc::vec![0x80u8, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x61, 0xAA];
        assert!(depay.depacketize(&p0).is_none());
        // seq 5 (gap), marker set, a fresh single NAL.
        let p1 = alloc::vec![0x80u8, 0x80 | 96, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0x65, 0xBB];
        let unit = depay.depacketize(&p1).expect("marker closes AU");
        assert_eq!(
            unit.data,
            wrap(&[&[0x65, 0xBB]]),
            "only the post-gap NAL survives; the dropped one is gone"
        );
    }
}
