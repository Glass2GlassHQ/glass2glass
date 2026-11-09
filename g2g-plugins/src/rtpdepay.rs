//! Sans-IO H.264 and H.265 RTP depayloaders (RFC 3550 header + RFC 6184 /
//! RFC 7798 payload), the receive-side inverses of the
//! [`rtppay`](crate::rtppay) packetizers. They turn a stream of RTP packets
//! back into Annex-B access units: single-NAL and aggregation payloads pass
//! through; fragmentation units reassemble; the RTP marker bit closes an
//! access unit. No I/O: `UdpSrc` wraps this and feeds it the
//! datagrams a socket receives. Opus (RFC 7587) needs no reassembly: each RTP
//! payload is one Opus packet.
//!
//! This is the basic in-order depayloader. A jitter buffer (packet reorder,
//! loss concealment, RTCP) is the larger receive-side follow-up (design/TODO.md);
//! out-of-order or lost packets are detected via the sequence number and reset
//! the in-flight reassembly so a gap never welds two access units together.

use alloc::vec::Vec;

use g2g_core::mediaclock::MediaClock;
use g2g_core::rtp::RtpHeader;
use g2g_core::VideoCodec;

use crate::rtppay::{FU_A_TYPE, FU_END_BIT, FU_START_BIT, H265_FU_TYPE, H265_NAL_HEADER_LEN};
use crate::sdp::OPUS_CLOCK_HZ;

/// RFC 6184 aggregation packet NAL type.
const STAP_A_TYPE: u8 = 24;
const ANNEX_B_START_CODE: [u8; 4] = [0, 0, 0, 1];

/// One depayloaded access unit: Annex-B bytes for H.264, one Opus packet for
/// Opus, plus the RTP timestamp shared by its packets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessUnit {
    pub data: Vec<u8>,
    pub rtp_timestamp: u32,
}

/// Cap the in-flight FU-A and access-unit reassembly buffers. A never-ending
/// fragment run (start without end) or an access unit whose marker never arrives
/// would otherwise grow without bound on untrusted RTP; past this we drop the
/// in-flight data and resync at the next start-of-AU. 8 MiB covers a large
/// keyframe.
const MAX_ASSEMBLY_BYTES: usize = 8 * 1024 * 1024;

// the payload formats the plain RTP receive path depayloads, named as in gst `encoding-name`
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RtpEncoding {
    #[default]
    H264,
    Opus,
    H265,
}

impl RtpEncoding {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "H264" => Some(RtpEncoding::H264),
            "OPUS" => Some(RtpEncoding::Opus),
            "H265" => Some(RtpEncoding::H265),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            RtpEncoding::H264 => "H264",
            RtpEncoding::Opus => "OPUS",
            RtpEncoding::H265 => "H265",
        }
    }

    pub fn clock_rate_hz(self) -> u32 {
        match self {
            RtpEncoding::H264 | RtpEncoding::H265 => MediaClock::VIDEO_RATE_HZ,
            RtpEncoding::Opus => OPUS_CLOCK_HZ,
        }
    }

    // every Opus packet decodes without the ones before it
    pub fn is_keyframe(self, unit: &[u8]) -> bool {
        match self {
            RtpEncoding::H264 => crate::annexb::au_is_keyframe(VideoCodec::H264, unit),
            RtpEncoding::H265 => crate::annexb::au_is_keyframe(VideoCodec::H265, unit),
            RtpEncoding::Opus => true,
        }
    }
}

pub type AccessUnits =
    core::iter::Chain<core::option::IntoIter<AccessUnit>, core::option::IntoIter<AccessUnit>>;

#[derive(Debug)]
pub enum RtpDepayloader {
    H264(RtpH264Depayloader),
    Opus,
    H265(RtpH265Depayloader),
}

impl RtpDepayloader {
    pub fn new(encoding: RtpEncoding) -> Self {
        match encoding {
            RtpEncoding::H264 => RtpDepayloader::H264(RtpH264Depayloader::new()),
            RtpEncoding::Opus => RtpDepayloader::Opus,
            RtpEncoding::H265 => RtpDepayloader::H265(RtpH265Depayloader::new()),
        }
    }

    pub fn depacketize(&mut self, packet: &[u8]) -> AccessUnits {
        match self {
            RtpDepayloader::H264(depayloader) => depayloader.depacketize(packet),
            RtpDepayloader::H265(depayloader) => depayloader.depacketize(packet),
            RtpDepayloader::Opus => depacketize_opus(packet).into_iter().chain(None),
        }
    }
}

// RFC 7587 never fragments, so each payload is one whole Opus packet
fn depacketize_opus(packet: &[u8]) -> Option<AccessUnit> {
    let parsed = RtpHeader::parse(packet)?;
    let end = parsed.payload_offset.checked_add(parsed.payload_len)?;
    let payload = packet.get(parsed.payload_offset..end)?;
    if payload.is_empty() {
        return None;
    }
    Some(AccessUnit {
        data: payload.to_vec(),
        rtp_timestamp: parsed.header.timestamp,
    })
}

#[derive(Debug, Default)]
pub struct RtpH264Depayloader {
    assembler: NalAssembler,
}

impl RtpH264Depayloader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one RTP packet. Yields the access unit a new RTP timestamp closed,
    /// then the one this packet's marker bit closed. Packets that are too short
    /// or use unsupported aggregation modes (STAP-B / MTAP / FU-B) are skipped.
    pub fn depacketize(&mut self, packet: &[u8]) -> AccessUnits {
        self.assembler.depacketize(packet, add_h264_nals)
    }
}

fn add_h264_nals(assembler: &mut NalAssembler, payload: &[u8]) {
    match payload[0] & 0x1F {
        // Single NAL unit (types 1..=23): the payload is the NAL verbatim.
        1..=23 => {
            assembler.abandon_fragment();
            assembler.push_nal(payload);
        }
        // STAP-A: [type 24][ (16-bit size)(NAL) ]* aggregated NALs, used
        // mostly to carry SPS+PPS in one packet.
        t if t == STAP_A_TYPE => {
            assembler.abandon_fragment();
            assembler.push_aggregated_nals(&payload[1..]);
        }
        // FU-A: a single NAL fragmented across packets. The original NAL
        // header byte is rebuilt from the FU indicator's F|NRI and the FU
        // header's type; Start/End bits bound the fragment run.
        t if t == FU_A_TYPE => {
            let Some(&fu_header) = payload.get(1) else {
                return;
            };
            if fu_header & FU_START_BIT != 0 {
                assembler.start_fragment(&[(payload[0] & 0xE0) | (fu_header & 0x1F)]);
            }
            assembler.extend_fragment(&payload[2..], fu_header & FU_END_BIT != 0);
        }
        // STAP-B (25), MTAP (26/27), FU-B (29): unsupported, skip.
        _ => {}
    }
}

/// Sans-IO H.265 RTP depayloader (RFC 7798), for senders that keep
/// `sprop-max-don-diff` at 0 and so carry no DONL fields.
#[derive(Debug, Default)]
pub struct RtpH265Depayloader {
    assembler: NalAssembler,
}

impl RtpH265Depayloader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one RTP packet, with the same output as [`RtpH264Depayloader::depacketize`].
    /// PACI packets are skipped.
    pub fn depacketize(&mut self, packet: &[u8]) -> AccessUnits {
        self.assembler.depacketize(packet, add_h265_nals)
    }
}

/// RFC 7798 aggregation packet NAL type.
const H265_AP_TYPE: u8 = 48;

fn add_h265_nals(assembler: &mut NalAssembler, payload: &[u8]) {
    if payload.len() < H265_NAL_HEADER_LEN {
        return;
    }
    match (payload[0] >> 1) & 0x3F {
        0..=47 => {
            assembler.abandon_fragment();
            assembler.push_nal(payload);
        }
        H265_AP_TYPE => {
            assembler.abandon_fragment();
            assembler.push_aggregated_nals(&payload[H265_NAL_HEADER_LEN..]);
        }
        // the FU header after the payload header carries S, E and the fragmented NAL's type
        H265_FU_TYPE => {
            let Some(&fu_header) = payload.get(H265_NAL_HEADER_LEN) else {
                return;
            };
            if fu_header & FU_START_BIT != 0 {
                let nal_type = fu_header & 0x3F;
                assembler.start_fragment(&[(payload[0] & 0x81) | (nal_type << 1), payload[1]]);
            }
            assembler.extend_fragment(
                &payload[H265_NAL_HEADER_LEN + 1..],
                fu_header & FU_END_BIT != 0,
            );
        }
        _ => {}
    }
}

// rebuilds Annex-B access units from the NALs of consecutive RTP packets
#[derive(Debug, Default)]
struct NalAssembler {
    /// Annex-B NALs accumulated for the access unit currently being assembled.
    au: Vec<u8>,
    /// Where the fragmented NAL being reassembled starts in `au`, while one is.
    fragment_start: Option<usize>,
    /// Last RTP sequence number seen, for gap detection.
    last_seq: Option<u16>,
    /// RTP timestamp of the in-flight access unit.
    timestamp: u32,
    /// Part of the current access unit was lost or dropped, so its remaining
    /// packets are skipped until the next access-unit boundary.
    discarding: bool,
}

impl NalAssembler {
    fn depacketize(
        &mut self,
        packet: &[u8],
        add_nals: fn(&mut NalAssembler, &[u8]),
    ) -> AccessUnits {
        // One RTP header parser for the whole workspace (the CSRC / extension /
        // padding offset math lives in `g2g_core::rtp`, shared with the MCU RTP
        // source); a malformed or non-v2 datagram is rejected.
        let Some(parsed) = RtpHeader::parse(packet) else {
            return None.into_iter().chain(None);
        };
        let seq = parsed.header.sequence;

        // A sequence gap (loss / reorder) means the in-flight FU-A and access
        // unit can no longer be trusted; drop them so we never splice across
        // the discontinuity. The next start-of-AU rebuilds cleanly.
        if let Some(prev) = self.last_seq {
            if seq != prev.wrapping_add(1) {
                self.drop_access_unit();
            }
        }
        self.last_seq = Some(seq);

        // senders that omit the marker bit end an access unit only by starting
        // the next one with a new RTP timestamp
        let mut closed_by_timestamp = None;
        if parsed.header.timestamp != self.timestamp {
            self.abandon_fragment();
            if !self.discarding && !self.au.is_empty() {
                closed_by_timestamp = Some(self.take_access_unit());
            }
            self.au.clear();
            self.discarding = false;
            self.timestamp = parsed.header.timestamp;
        }

        let payload = packet
            .get(parsed.payload_offset..parsed.payload_offset + parsed.payload_len)
            .unwrap_or_default();
        if !self.discarding && !payload.is_empty() {
            add_nals(self, payload);
        }
        let closed_by_marker = self.close_on_marker(parsed.header.marker);
        closed_by_timestamp.into_iter().chain(closed_by_marker)
    }

    fn close_on_marker(&mut self, marker: bool) -> Option<AccessUnit> {
        if self.discarding {
            self.discarding = !marker;
            return None;
        }
        if !marker {
            return None;
        }
        self.abandon_fragment();
        if self.au.is_empty() {
            return None;
        }
        Some(self.take_access_unit())
    }

    // `[ (16-bit size)(NAL) ]*`, the body of a STAP-A or an H.265 aggregation packet
    fn push_aggregated_nals(&mut self, mut body: &[u8]) {
        while let [high, low, rest @ ..] = body {
            let size = usize::from(u16::from_be_bytes([*high, *low]));
            let Some(nal) = rest.get(..size) else {
                break;
            };
            self.push_nal(nal);
            body = &rest[size..];
        }
    }

    fn start_fragment(&mut self, nal_header: &[u8]) {
        self.abandon_fragment();
        self.fragment_start = Some(self.au.len());
        self.au.extend_from_slice(&ANNEX_B_START_CODE);
        self.au.extend_from_slice(nal_header);
    }

    fn extend_fragment(&mut self, data: &[u8], end: bool) {
        if self.fragment_start.is_none() {
            return;
        }
        self.au.extend_from_slice(data);
        if self.au.len() > MAX_ASSEMBLY_BYTES {
            // A fragment run that never ends; drop the in-flight data
            // and resync at the next start-of-AU.
            self.drop_access_unit();
            return;
        }
        if end {
            self.fragment_start = None;
        }
    }

    fn take_access_unit(&mut self) -> AccessUnit {
        // the next access unit is likely about as large as this one
        let next_capacity = self.au.len();
        let data = core::mem::replace(&mut self.au, Vec::with_capacity(next_capacity));
        AccessUnit {
            data,
            rtp_timestamp: self.timestamp,
        }
    }

    fn drop_access_unit(&mut self) {
        self.fragment_start = None;
        self.au.clear();
        self.discarding = true;
    }

    // a fragment run cut short by another NAL or the marker is incomplete, drop it
    fn abandon_fragment(&mut self) {
        if let Some(start) = self.fragment_start.take() {
            self.au.truncate(start);
        }
    }

    /// Append one NAL to the in-flight access unit as a 4-byte-start-code
    /// Annex-B unit.
    fn push_nal(&mut self, nal: &[u8]) {
        if nal.is_empty() {
            return;
        }
        self.au.extend_from_slice(&ANNEX_B_START_CODE);
        self.au.extend_from_slice(nal);
        if self.au.len() > MAX_ASSEMBLY_BYTES {
            // An access unit whose marker never arrives; drop it rather than
            // buffer without bound.
            self.drop_access_unit();
        }
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
            for unit in depay.depacketize(p) {
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
            for unit in depay.depacketize(p) {
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
        let unit = depay.depacketize(&packet).next().expect("marker closes AU");
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
        assert!(depay.depacketize(&p0).next().is_none());
        // seq 5 (gap), marker set, a fresh single NAL of the next access unit.
        let p1 = alloc::vec![
            0x80u8,
            0x80 | 96,
            0,
            5,
            0,
            0,
            0x0B,
            0xB8,
            0,
            0,
            0,
            0,
            0x65,
            0xBB
        ];
        let unit = depay.depacketize(&p1).next().expect("marker closes AU");
        assert_eq!(
            unit.data,
            wrap(&[&[0x65, 0xBB]]),
            "only the post-gap NAL survives; the dropped one is gone"
        );
    }

    fn single_nal_packet(seq: u16, ts: u32, marker: bool, nal: &[u8]) -> Vec<u8> {
        let mut p = alloc::vec![0x80u8, if marker { 0x80 | 96 } else { 96 }];
        p.extend_from_slice(&seq.to_be_bytes());
        p.extend_from_slice(&ts.to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]);
        p.extend_from_slice(nal);
        p
    }

    #[test]
    fn gap_inside_an_access_unit_drops_its_remaining_packets() {
        let mut depay = RtpH264Depayloader::new();
        assert!(depay
            .depacketize(&single_nal_packet(0, 9000, false, &[0x65, 1]))
            .next()
            .is_none());
        assert!(
            depay
                .depacketize(&single_nal_packet(2, 9000, true, &[0x65, 3]))
                .next()
                .is_none(),
            "the tail of an access unit that lost a packet is not emitted"
        );
        let unit = depay
            .depacketize(&single_nal_packet(3, 12000, true, &[0x41, 4]))
            .next()
            .expect("the next access unit is complete");
        assert_eq!(unit.data, wrap(&[&[0x41, 4]]));
        assert_eq!(unit.rtp_timestamp, 12000);
    }

    #[test]
    fn timestamp_change_closes_an_unmarked_access_unit() {
        let mut depay = RtpH264Depayloader::new();
        assert!(depay
            .depacketize(&single_nal_packet(0, 9000, false, &[0x65, 1]))
            .next()
            .is_none());
        let unit = depay
            .depacketize(&single_nal_packet(1, 12000, false, &[0x41, 2]))
            .next()
            .expect("the new timestamp closes the previous access unit");
        assert_eq!(unit.data, wrap(&[&[0x65, 1]]));
        assert_eq!(unit.rtp_timestamp, 9000);
        let units: Vec<_> = depay
            .depacketize(&single_nal_packet(2, 15000, true, &[0x41, 3]))
            .map(|unit| (unit.data, unit.rtp_timestamp))
            .collect();
        assert_eq!(
            units,
            [(wrap(&[&[0x41, 2]]), 12000), (wrap(&[&[0x41, 3]]), 15000)],
            "the timestamp closes the unmarked unit and the marker closes this one"
        );
    }

    fn fu_a_packet(
        seq: u16,
        ts: u32,
        marker: bool,
        start: bool,
        end: bool,
        frag: &[u8],
    ) -> Vec<u8> {
        let mut p = alloc::vec![0x80u8, if marker { 0x80 | 96 } else { 96 }];
        p.extend_from_slice(&seq.to_be_bytes());
        p.extend_from_slice(&ts.to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0]); // ssrc
        p.push(0x60 | FU_A_TYPE); // FU indicator (F=0, NRI=3, type 28)
        let mut fu_header = 0x05u8; // type 5 (IDR slice)
        if start {
            fu_header |= 0x80;
        }
        if end {
            fu_header |= 0x40;
        }
        p.push(fu_header);
        p.extend_from_slice(frag);
        p
    }

    #[test]
    fn unterminated_fu_a_run_is_bounded_and_resyncs() {
        let mut depay = RtpH264Depayloader::new();
        let frag = alloc::vec![0xABu8; 1024 * 1024]; // 1 MiB per packet
                                                     // Start a fragment run, then never send the end bit: the buffer must cap
                                                     // and emit nothing instead of growing without bound.
        assert!(depay
            .depacketize(&fu_a_packet(0, 1, false, true, false, &frag))
            .next()
            .is_none());
        for seq in 1..16u16 {
            assert!(depay
                .depacketize(&fu_a_packet(seq, 1, false, false, false, &frag))
                .next()
                .is_none());
        }
        // A fresh complete FU-A NAL still reassembles after the run is dropped.
        assert!(depay
            .depacketize(&fu_a_packet(100, 2, false, true, false, &[1, 2, 3]))
            .next()
            .is_none());
        let unit = depay
            .depacketize(&fu_a_packet(101, 2, true, false, true, &[4, 5, 6]))
            .next()
            .expect("resyncs and completes a new AU");
        assert_eq!(unit.data, wrap(&[&[0x65, 1, 2, 3, 4, 5, 6]]));
    }

    #[test]
    fn h265_access_unit_round_trips_through_fragmentation() {
        let mut idr = alloc::vec![0x26u8, 0x01];
        idr.extend((0..200u8).collect::<Vec<_>>());
        let au = wrap(&[
            &[0x40, 0x01, 0x0C],
            &[0x42, 0x01, 0x01],
            &[0x44, 0x01, 0xC1],
            &idr,
        ]);
        let mut pay = crate::rtppay::RtpH265Packetizer::new(96, 5).with_max_payload(64);
        let packets = pay.packetize(&au, 9000);
        assert!(packets.len() > 4, "precondition: the IDR is fragmented");
        let mut depay = RtpH265Depayloader::new();
        let units: Vec<_> = packets.iter().flat_map(|p| depay.depacketize(p)).collect();
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].data, au);
        assert!(RtpEncoding::H265.is_keyframe(&units[0].data));
    }

    #[test]
    fn h265_aggregation_packet_splits_into_its_nals() {
        let vps: &[u8] = &[0x40, 0x01, 0x0C];
        let sps: &[u8] = &[0x42, 0x01, 0x01, 0x60];
        let mut payload = alloc::vec![H265_AP_TYPE << 1, 0x01];
        for nal in [vps, sps] {
            payload.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            payload.extend_from_slice(nal);
        }
        let mut packet = alloc::vec![0x80u8, 0x80 | 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        packet.extend_from_slice(&payload);
        let mut depay = RtpH265Depayloader::new();
        let unit = depay
            .depacketize(&packet)
            .next()
            .expect("marker closes the unit");
        assert_eq!(unit.data, wrap(&[vps, sps]));
    }

    #[test]
    fn opus_payload_round_trips_and_short_packets_drop() {
        let opus = [0xFCu8, 1, 2, 3];
        let packets = crate::rtppay::RtpOpusPacketizer::new(111, 9).packetize(&opus, 960);
        let mut depay = RtpDepayloader::new(RtpEncoding::Opus);
        let unit = depay
            .depacketize(&packets[0])
            .next()
            .expect("one RTP packet carries one Opus packet");
        assert_eq!(unit.data, opus);
        assert_eq!(unit.rtp_timestamp, 960);

        let header_only = &packets[0][..g2g_core::rtp::RTP_HEADER_LEN];
        assert!(
            depay.depacketize(header_only).next().is_none(),
            "empty payload"
        );
        for len in 0..g2g_core::rtp::RTP_HEADER_LEN {
            assert!(
                depay.depacketize(&packets[0][..len]).next().is_none(),
                "{len} bytes"
            );
        }
    }

    #[test]
    fn fu_a_run_cut_short_by_another_nal_is_dropped() {
        let mut d = RtpH264Depayloader::new();
        assert!(d
            .depacketize(&fu_a_packet(0, 9000, false, true, false, &[1, 2]))
            .next()
            .is_none());
        let mut single = alloc::vec![0x80u8, 0x80 | 96];
        single.extend_from_slice(&1u16.to_be_bytes());
        single.extend_from_slice(&9000u32.to_be_bytes());
        single.extend_from_slice(&[0, 0, 0, 0]);
        single.extend_from_slice(&[0x41, 7]);
        let au = d
            .depacketize(&single)
            .next()
            .expect("the marker closes the access unit");
        assert_eq!(au.data, wrap(&[&[0x41, 7]]), "only the complete NAL");
    }
}
