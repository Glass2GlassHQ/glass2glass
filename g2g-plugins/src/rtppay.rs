//! Sans-IO RTP packetizers (RFC 3550 RTP header), the live-egress counterpart
//! of `RtspSrc`'s receive path. Given one access unit and its RTP timestamp,
//! each produces complete RTP packets. H.264 (RFC 6184) and H.265 (RFC 7798)
//! send a single-NAL packet when the NAL fits the MTU, else FU fragments. AAC
//! (RFC 3640 AAC-hbr) sends one access unit per packet, fragmented past the
//! MTU, and Opus (RFC 7587) one frame per packet. No I/O: the UDP sink, the
//! RTSP server sink and `rtspclientsink` wrap these and send the packets.

use alloc::vec::Vec;

use g2g_core::rtp::{CaptureTimeExtensionIds, RtpHeader, RtpHeaderExtension, RTP_HEADER_LEN};

use crate::annexb::nal_units;
/// RFC 6184 FU-A NAL type.
const FU_A_TYPE: u8 = 28;
const FU_A_HEADER_LEN: usize = 2;
const H265_FU_TYPE: u8 = 49;
// the two-byte payload header then the one-byte FU header
const H265_FU_HEADER_LEN: usize = 3;
const H265_NAL_HEADER_LEN: usize = 2;
pub(crate) const AAC_AU_SIZE_BITS: u32 = 13;
pub(crate) const AAC_AU_INDEX_BITS: u32 = 3;
// the 16-bit AU-headers-length field, then one AU header of size and index
const AAC_AU_HEADER_SECTION_LEN: usize = 4;
const AAC_AU_HEADERS_LENGTH_BITS: u16 = (AAC_AU_SIZE_BITS + AAC_AU_INDEX_BITS) as u16;
const MAX_PREFIX_LEN: usize = 4;
const DEFAULT_MAX_PAYLOAD: usize = 1400;
const FU_START_BIT: u8 = 0x80;
const FU_END_BIT: u8 = 0x40;

// borrows the access unit so its bytes are copied once, into the packet
struct Payload<'a> {
    prefix: [u8; MAX_PREFIX_LEN],
    prefix_len: usize,
    body: &'a [u8],
}

impl<'a> Payload<'a> {
    fn whole(body: &'a [u8]) -> Self {
        Payload {
            prefix: [0; MAX_PREFIX_LEN],
            prefix_len: 0,
            body,
        }
    }

    fn len(&self) -> usize {
        self.prefix_len + self.body.len()
    }
}

#[derive(Debug, Clone)]
struct RtpPacketWriter {
    payload_type: u8,
    ssrc: u32,
    sequence: u16,
}

impl RtpPacketWriter {
    fn new(payload_type: u8, ssrc: u32) -> Self {
        Self {
            payload_type: payload_type & 0x7F,
            ssrc,
            sequence: 0,
        }
    }

    // the marker bit goes on the last packet, the end of the access unit
    fn write(
        &mut self,
        payloads: &[Payload<'_>],
        timestamp: u32,
        extensions: &[RtpHeaderExtension<'_>],
    ) -> Vec<Vec<u8>> {
        let count = payloads.len();
        let mut packets = Vec::with_capacity(count);
        for (i, payload) in payloads.iter().enumerate() {
            let marker = i + 1 == count;
            // only the first packet, as GStreamer's ntp-64 writer does by default
            let packet_extensions = if i == 0 { extensions } else { &[] };
            let mut packet = Vec::with_capacity(RTP_HEADER_LEN + payload.len());
            self.write_header(&mut packet, marker, timestamp, packet_extensions);
            packet.extend_from_slice(&payload.prefix[..payload.prefix_len]);
            packet.extend_from_slice(payload.body);
            packets.push(packet);
            self.sequence = self.sequence.wrapping_add(1);
        }
        packets
    }

    fn write_header(
        &self,
        out: &mut Vec<u8>,
        marker: bool,
        timestamp: u32,
        extensions: &[RtpHeaderExtension<'_>],
    ) {
        let header = RtpHeader {
            payload_type: self.payload_type,
            marker,
            sequence: self.sequence,
            timestamp,
            ssrc: self.ssrc,
        };
        let start = out.len();
        let len = RtpHeader::extended_header_len(extensions)
            .expect("a capture-time element fits the two-byte form");
        out.resize(start + len, 0);
        header.write_with_extensions(extensions, &mut out[start..]);
    }
}

// how one codec splits a NAL too big for a packet into fragmentation units
struct NalFragmentation {
    nal_header_len: usize,
    fu_prefix_len: usize,
    // the FU prefix for a fragment of the NAL with this header, given its start and end bits
    fu_prefix: fn(&[u8], u8) -> [u8; MAX_PREFIX_LEN],
}

// FU-A: the original NAL header byte splits into the FU indicator
// (F | NRI | type=28) and the FU header (S | E | original type).
const H264_FRAGMENTATION: NalFragmentation = NalFragmentation {
    nal_header_len: 1,
    fu_prefix_len: FU_A_HEADER_LEN,
    fu_prefix: |header, start_end| {
        [
            (header[0] & 0xE0) | FU_A_TYPE,
            start_end | (header[0] & 0x1F),
            0,
            0,
        ]
    },
};

// the payload header keeps F, LayerId and TID with type 49, the FU header carries the original type
const H265_FRAGMENTATION: NalFragmentation = NalFragmentation {
    nal_header_len: H265_NAL_HEADER_LEN,
    fu_prefix_len: H265_FU_HEADER_LEN,
    fu_prefix: |header, start_end| {
        [
            (header[0] & 0x81) | (H265_FU_TYPE << 1),
            header[1],
            start_end | ((header[0] >> 1) & 0x3F),
            0,
        ]
    },
};

// one payload per output packet: a whole NAL when it fits, else fragments of the oversized NAL
fn nal_payloads<'a>(
    access_unit: &'a [u8],
    max_payload: usize,
    fragmentation: &NalFragmentation,
) -> Vec<Payload<'a>> {
    let mut payloads = Vec::new();
    for nal in nal_units(access_unit).filter(|n| !n.is_empty()) {
        if nal.len() <= max_payload {
            payloads.push(Payload::whole(nal));
            continue;
        }
        let (header, body) = nal.split_at(fragmentation.nal_header_len);
        let chunk = max_payload - fragmentation.fu_prefix_len;
        let chunks = body.len().div_ceil(chunk);
        for (i, part) in body.chunks(chunk).enumerate() {
            let start = if i == 0 { FU_START_BIT } else { 0 };
            let end = if i + 1 == chunks { FU_END_BIT } else { 0 };
            payloads.push(Payload {
                prefix: (fragmentation.fu_prefix)(header, start | end),
                prefix_len: fragmentation.fu_prefix_len,
                body: part,
            });
        }
    }
    payloads
}

#[derive(Debug, Clone)]
pub struct RtpH264Packetizer {
    writer: RtpPacketWriter,
    /// Max RTP payload bytes per packet (the bytes after the 12-byte header).
    max_payload: usize,
}

impl RtpH264Packetizer {
    /// `payload_type` is the dynamic RTP PT (commonly 96..=127 for H.264).
    pub fn new(payload_type: u8, ssrc: u32) -> Self {
        Self {
            writer: RtpPacketWriter::new(payload_type, ssrc),
            max_payload: DEFAULT_MAX_PAYLOAD,
        }
    }

    /// Max RTP payload bytes per packet. Floored at 3 so an FU-A packet always
    /// carries at least one body byte past its 2-byte header.
    pub fn with_max_payload(mut self, bytes: usize) -> Self {
        self.max_payload = bytes.max(FU_A_HEADER_LEN + 1);
        self
    }

    /// The sequence number the next packet will carry. Useful in tests.
    pub fn next_sequence(&self) -> u16 {
        self.writer.sequence
    }

    /// Packetize one Annex-B access unit at `rtp_timestamp` into complete RTP
    /// packets. Sequence numbers increment across packets and calls; the marker
    /// bit is set on the last packet of the access unit (RFC 6184 frame end).
    pub fn packetize(&mut self, access_unit: &[u8], rtp_timestamp: u32) -> Vec<Vec<u8>> {
        self.packetize_with_capture_time(
            access_unit,
            rtp_timestamp,
            CaptureTimeExtensionIds::default(),
            0,
        )
    }

    pub fn packetize_with_capture_time(
        &mut self,
        access_unit: &[u8],
        rtp_timestamp: u32,
        capture_time_ids: CaptureTimeExtensionIds,
        capture_ntp: u64,
    ) -> Vec<Vec<u8>> {
        let capture_ntp_bytes = capture_ntp.to_be_bytes();
        let extensions: Vec<RtpHeaderExtension<'_>> =
            capture_time_ids.elements(&capture_ntp_bytes).collect();
        let payloads = nal_payloads(access_unit, self.max_payload, &H264_FRAGMENTATION);
        self.writer.write(&payloads, rtp_timestamp, &extensions)
    }
}

#[derive(Debug, Clone)]
pub struct RtpH265Packetizer {
    writer: RtpPacketWriter,
    max_payload: usize,
}

impl RtpH265Packetizer {
    pub fn new(payload_type: u8, ssrc: u32) -> Self {
        Self {
            writer: RtpPacketWriter::new(payload_type, ssrc),
            max_payload: DEFAULT_MAX_PAYLOAD,
        }
    }

    // floored so a fragment always carries one body byte past its header
    pub fn with_max_payload(mut self, bytes: usize) -> Self {
        self.max_payload = bytes.max(H265_FU_HEADER_LEN + 1);
        self
    }

    pub fn packetize(&mut self, access_unit: &[u8], rtp_timestamp: u32) -> Vec<Vec<u8>> {
        let payloads = nal_payloads(access_unit, self.max_payload, &H265_FRAGMENTATION);
        self.writer.write(&payloads, rtp_timestamp, &[])
    }
}

#[derive(Debug, Clone)]
pub struct RtpAacPacketizer {
    writer: RtpPacketWriter,
    max_payload: usize,
}

impl RtpAacPacketizer {
    pub fn new(payload_type: u8, ssrc: u32) -> Self {
        Self {
            writer: RtpPacketWriter::new(payload_type, ssrc),
            max_payload: DEFAULT_MAX_PAYLOAD,
        }
    }

    // floored so a fragment always carries one body byte past its AU header section
    pub fn with_max_payload(mut self, bytes: usize) -> Self {
        self.max_payload = bytes.max(AAC_AU_HEADER_SECTION_LEN + 1);
        self
    }

    // `access_unit` is a raw AAC frame without ADTS, None when it is too big for the 13-bit AU size
    pub fn packetize(&mut self, access_unit: &[u8], rtp_timestamp: u32) -> Option<Vec<Vec<u8>>> {
        let size = u16::try_from(access_unit.len())
            .ok()
            .filter(|size| *size < 1 << AAC_AU_SIZE_BITS)?;
        let au_header = (size << AAC_AU_INDEX_BITS).to_be_bytes();
        let headers_length = AAC_AU_HEADERS_LENGTH_BITS.to_be_bytes();
        let prefix = [
            headers_length[0],
            headers_length[1],
            au_header[0],
            au_header[1],
        ];
        // every fragment repeats the AU header with the whole AU size (RFC 3640 §3.2.3)
        let payloads: Vec<Payload<'_>> = access_unit
            .chunks(self.max_payload - AAC_AU_HEADER_SECTION_LEN)
            .map(|part| Payload {
                prefix,
                prefix_len: AAC_AU_HEADER_SECTION_LEN,
                body: part,
            })
            .collect();
        Some(self.writer.write(&payloads, rtp_timestamp, &[]))
    }
}

#[derive(Debug, Clone)]
pub struct RtpOpusPacketizer {
    writer: RtpPacketWriter,
}

impl RtpOpusPacketizer {
    pub fn new(payload_type: u8, ssrc: u32) -> Self {
        Self {
            writer: RtpPacketWriter::new(payload_type, ssrc),
        }
    }

    // RFC 7587 has no fragmentation, a packet carries one whole Opus packet
    pub fn packetize(&mut self, opus_packet: &[u8], rtp_timestamp: u32) -> Vec<Vec<u8>> {
        self.writer
            .write(&[Payload::whole(opus_packet)], rtp_timestamp, &[])
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
        assert_eq!(
            payload(pkt),
            &[0x65, 1, 2, 3],
            "single NAL carried verbatim"
        );
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
            assert_eq!(
                pl[1] & 0x40 != 0,
                i + 1 == packets.len(),
                "end bit on last only"
            );
            reassembled.extend_from_slice(&pl[2..]);
        }
        assert_eq!(reassembled, body, "fragments reassemble the NAL body");
        assert!(
            marker(packets.last().unwrap()),
            "marker on the last fragment"
        );
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

    #[test]
    fn capture_time_rides_the_first_packet_and_the_frame_still_reassembles() {
        const CAPTURE_NTP: u64 = 0xE8F0_0000_4000_0000;
        let ids = CaptureTimeExtensionIds {
            abs_capture_time: core::num::NonZeroU8::new(2),
            ntp_64: core::num::NonZeroU8::new(3),
        };
        let body: Vec<u8> = (0..10).collect();
        let mut nal = alloc::vec![0x65u8];
        nal.extend_from_slice(&body);
        let mut p = RtpH264Packetizer::new(96, 1).with_max_payload(5);
        let packets = p.packetize_with_capture_time(&wrap(&[&nal]), 7, ids, CAPTURE_NTP);
        assert!(packets.len() > 1);
        assert_eq!(ids.read(&packets[0]), Some(CAPTURE_NTP));
        assert!(
            packets[1..].iter().all(|packet| packet[0] == 0x80),
            "no X bit after the first"
        );

        let mut depayloader = crate::rtpdepay::RtpH264Depayloader::new();
        let access_unit = packets
            .iter()
            .find_map(|packet| depayloader.depacketize(packet))
            .expect("the marker closes the access unit");
        assert_eq!(access_unit.data, wrap(&[&nal]));
    }

    // IDR_W_RADL (type 19) on layer 0 with temporal id 1
    const H265_IDR_HEADER: [u8; 2] = [19 << 1, 1];

    #[test]
    fn oversized_h265_nal_fragments_into_rfc7798_units() {
        let body: Vec<u8> = (0..10).collect();
        let mut nal = H265_IDR_HEADER.to_vec();
        nal.extend_from_slice(&body);
        let mut p = RtpH265Packetizer::new(96, 1).with_max_payload(6);
        let packets = p.packetize(&wrap(&[&nal]), 7);
        assert!(packets.len() > 1, "oversized NAL fragments");

        let mut reassembled = Vec::new();
        for (i, pkt) in packets.iter().enumerate() {
            let pl = payload(pkt);
            assert_eq!((pl[0] >> 1) & 0x3F, H265_FU_TYPE, "FU payload header type");
            assert_eq!(pl[1], H265_IDR_HEADER[1], "LayerId and TID kept");
            assert_eq!(pl[2] & 0x3F, 19, "original NAL type in the FU header");
            assert_eq!(pl[2] & FU_START_BIT != 0, i == 0, "start bit on first only");
            assert_eq!(
                pl[2] & FU_END_BIT != 0,
                i + 1 == packets.len(),
                "end bit on last only"
            );
            assert_eq!(marker(pkt), i + 1 == packets.len());
            reassembled.extend_from_slice(&pl[H265_FU_HEADER_LEN..]);
        }
        assert_eq!(reassembled, body, "fragments reassemble to the NAL body");
    }

    #[test]
    fn aac_access_unit_rides_behind_one_au_header() {
        let au: Vec<u8> = (0..10).collect();
        let mut p = RtpAacPacketizer::new(97, 1);
        let packets = p.packetize(&au, 1024).expect("fits the 13-bit size");
        assert_eq!(packets.len(), 1);
        let pl = payload(&packets[0]);
        let headers_length = u16::from_be_bytes([pl[0], pl[1]]);
        assert_eq!(
            u32::from(headers_length),
            AAC_AU_SIZE_BITS + AAC_AU_INDEX_BITS
        );
        let au_header = u16::from_be_bytes([pl[2], pl[3]]);
        assert_eq!(
            usize::from(au_header >> AAC_AU_INDEX_BITS),
            au.len(),
            "AU size"
        );
        assert_eq!(au_header & ((1 << AAC_AU_INDEX_BITS) - 1), 0, "AU index");
        assert_eq!(&pl[AAC_AU_HEADER_SECTION_LEN..], &au[..]);
        assert!(marker(&packets[0]));
    }

    #[test]
    fn oversized_aac_access_unit_fragments_with_the_whole_size() {
        let au: Vec<u8> = (0..10).collect();
        let mut p = RtpAacPacketizer::new(97, 1).with_max_payload(AAC_AU_HEADER_SECTION_LEN + 4);
        let packets = p.packetize(&au, 1024).expect("fits the 13-bit size");
        assert_eq!(packets.len(), au.len().div_ceil(4));
        let mut reassembled = Vec::new();
        for (i, pkt) in packets.iter().enumerate() {
            let pl = payload(pkt);
            let au_header = u16::from_be_bytes([pl[2], pl[3]]);
            assert_eq!(usize::from(au_header >> AAC_AU_INDEX_BITS), au.len());
            assert_eq!(marker(pkt), i + 1 == packets.len(), "marker ends the AU");
            reassembled.extend_from_slice(&pl[AAC_AU_HEADER_SECTION_LEN..]);
        }
        assert_eq!(reassembled, au);
    }

    #[test]
    fn aac_access_unit_past_the_size_field_is_refused() {
        let au = alloc::vec![0u8; 1 << AAC_AU_SIZE_BITS];
        assert!(RtpAacPacketizer::new(97, 1).packetize(&au, 0).is_none());
    }

    #[test]
    fn opus_packet_is_one_marked_rtp_packet() {
        let opus = [0xFCu8, 1, 2, 3];
        let mut p = RtpOpusPacketizer::new(111, 1);
        let packets = p.packetize(&opus, 960);
        assert_eq!(packets.len(), 1);
        assert_eq!(payload(&packets[0]), &opus);
        assert_eq!(timestamp(&packets[0]), 960);
        assert!(marker(&packets[0]));
    }
}
