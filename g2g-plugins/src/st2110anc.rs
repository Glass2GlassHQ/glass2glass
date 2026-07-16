//! ST 2110-40 ancillary data over RTP (M596): SMPTE ST 291 ANC packets (closed
//! captions, timecode, AFD, ...) carried per RFC 8331, timestamped off the PTP
//! media clock.
//!
//! This is where the existing caption stack meets ST 2110: a CEA-608/708 ANC
//! packet (DID 0x61) rides -40 with an RTP timestamp from the 90 kHz video
//! [`MediaClock`], so a receiver on the same grandmaster aligns the captions with
//! the video frame. Sans-IO like `st2110audio` / `rtppay` (pure `no_std` + alloc,
//! CI round-trip testable); an element wrapper sits on top later.
//!
//! The wire format is exact per RFC 8331: an 8-byte payload header (extended
//! sequence number, length, ANC_Count, field) then one or more ANC data packets.
//! Each ANC packet has a 32-bit header (C / Line_Number / Horizontal_Offset /
//! StreamNum) followed by 10-bit words (DID, SDID, Data_Count, the user data
//! words, and a Checksum_Word) bit-packed MSB-first and zero-padded to a 32-bit
//! boundary. Each 10-bit word carries SMPTE 291 parity (b8 = even parity of
//! b0..b7, b9 = NOT b8); the checksum is the low 9 bits of the sum of the low 9
//! bits of DID..last-UDW. Parsing is fully bounds-checked and rejects a message
//! whose parity or checksum does not verify (never trust the stream, AGENTS.md).

use alloc::vec::Vec;

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::MediaClock;

/// RFC 8331 payload header length (ext-seq + length + ANC_Count + F/reserved).
const ANC_HEADER_LEN: usize = 8;

/// The field a frame's ancillary data belongs to (RFC 8331 `F`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AncField {
    /// Progressive or unspecified (0b00).
    Progressive,
    /// Interlaced field 1 (0b10).
    Field1,
    /// Interlaced field 2 (0b11).
    Field2,
}

impl AncField {
    fn to_bits(self) -> u8 {
        match self {
            Self::Progressive => 0b00,
            Self::Field1 => 0b10,
            Self::Field2 => 0b11,
        }
    }
    fn from_bits(b: u8) -> Self {
        match b & 0b11 {
            0b10 => Self::Field1,
            0b11 => Self::Field2,
            _ => Self::Progressive,
        }
    }
}

/// One SMPTE ST 291 ancillary data packet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AncPacket {
    /// Channel: `true` = colour-difference (Cb/Cr), `false` = luma / no channel.
    pub c: bool,
    /// SDI raster line (11 bits); `0x7FF` = generic / unspecified.
    pub line_number: u16,
    /// Distance from SAV in words (12 bits); `0xFFF` = unspecified.
    pub horizontal_offset: u16,
    /// Data stream number (7 bits) when the S flag is set, else `None`.
    pub stream_num: Option<u8>,
    /// Data Identification word (e.g. 0x61 for CEA captions).
    pub did: u8,
    /// Secondary DID (e.g. 0x01 CEA-708, 0x02 CEA-608).
    pub sdid: u8,
    /// The user data words (at most 255; Data_Count is 8 bits).
    pub user_data: Vec<u8>,
}

impl AncPacket {
    /// A generic (line/offset unspecified) ANC packet, the common caption shape.
    pub fn generic(did: u8, sdid: u8, user_data: Vec<u8>) -> Self {
        Self {
            c: false,
            line_number: 0x7FF,
            horizontal_offset: 0xFFF,
            stream_num: None,
            did,
            sdid,
            user_data,
        }
    }

    fn encode(&self, w: &mut BitWriter) {
        w.write(u32::from(self.c), 1);
        w.write(u32::from(self.line_number), 11);
        w.write(u32::from(self.horizontal_offset), 12);
        match self.stream_num {
            Some(n) => {
                w.write(1, 1);
                w.write(u32::from(n), 7);
            }
            None => {
                w.write(0, 1);
                w.write(0, 7);
            }
        }
        // Data_Count is 8 bits; a longer payload cannot be represented, so cap it.
        let dc = self.user_data.len().min(0xFF) as u8;
        let mut sum = low9(self.did) + low9(self.sdid) + low9(dc);
        w.write(encode_word(self.did), 10);
        w.write(encode_word(self.sdid), 10);
        w.write(encode_word(dc), 10);
        for &u in self.user_data.iter().take(0xFF) {
            w.write(encode_word(u), 10);
            sum += low9(u);
        }
        let sum9 = sum & 0x1FF;
        // Checksum_Word: b0..b8 = sum9, b9 = NOT b8.
        w.write(sum9 | (((sum9 >> 8) & 1) ^ 1) << 9, 10);
        w.align32();
    }

    fn decode(r: &mut BitReader) -> Option<Self> {
        let c = r.read(1)? != 0;
        let line_number = r.read(11)? as u16;
        let horizontal_offset = r.read(12)? as u16;
        let s = r.read(1)?;
        let stream = r.read(7)? as u8;
        let did = decode_word(r)?;
        let sdid = decode_word(r)?;
        let dc = decode_word(r)?;
        let mut sum = low9(did) + low9(sdid) + low9(dc);
        let mut user_data = Vec::with_capacity(dc as usize);
        for _ in 0..dc {
            let u = decode_word(r)?;
            user_data.push(u);
            sum += low9(u);
        }
        // Checksum_Word: verify inverse-parity bit and the 9-bit sum.
        let cs = r.read(10)?;
        if (cs >> 9) & 1 != ((cs >> 8) & 1) ^ 1 || (cs & 0x1FF) != (sum & 0x1FF) {
            return None;
        }
        r.align32();
        Some(Self {
            c,
            line_number,
            horizontal_offset,
            stream_num: (s != 0).then_some(stream),
            did,
            sdid,
            user_data,
        })
    }
}

/// Even parity (b8) of an 8-bit value: the XOR of its bits.
fn parity(v: u8) -> u32 {
    v.count_ones() & 1
}

/// The low 9 bits (value + parity b8) of a data word, as the checksum sums them.
fn low9(v: u8) -> u32 {
    u32::from(v) | (parity(v) << 8)
}

/// Encode an 8-bit value as a 10-bit SMPTE 291 word: value, b8 = even parity,
/// b9 = NOT b8.
fn encode_word(v: u8) -> u32 {
    let p = parity(v);
    u32::from(v) | (p << 8) | ((p ^ 1) << 9)
}

/// Read one 10-bit data word, verifying its parity bits; `None` on a parity error.
fn decode_word(r: &mut BitReader) -> Option<u8> {
    let w = r.read(10)?;
    let value = (w & 0xFF) as u8;
    let b8 = (w >> 8) & 1;
    let b9 = (w >> 9) & 1;
    if b8 != parity(value) || b9 != (b8 ^ 1) {
        return None;
    }
    Some(value)
}

/// Packetizes ANC packets into an RFC 8331 RTP payload (one RTP packet per call,
/// marker set; fragmentation across packets is a later refinement).
#[derive(Debug)]
pub struct St2110AncPacketizer {
    payload_type: u8,
    ssrc: u32,
    sequence: u32,
    clock: MediaClock,
}

impl St2110AncPacketizer {
    /// A packetizer with the given dynamic RTP payload type and SSRC. The RTP
    /// timestamp uses the 90 kHz video media clock (ANC is aligned to video).
    pub fn new(payload_type: u8, ssrc: u32) -> Self {
        Self { payload_type: payload_type & 0x7F, ssrc, sequence: 0, clock: MediaClock::video() }
    }

    /// The media clock, for recovering a packet's PTP time on the receive side.
    pub fn media_clock(&self) -> MediaClock {
        self.clock
    }

    /// Build one RTP packet carrying `packets` (up to 255), timestamped at the
    /// video frame's PTP/TAI time. The RTP marker is set (all this frame's ANC is
    /// in this packet).
    pub fn packetize(&mut self, packets: &[AncPacket], tai_ns: u64, field: AncField) -> Vec<u8> {
        let count = packets.len().min(0xFF);
        // Bit-pack the ANC data section first so its length is known for the header.
        let mut bw = BitWriter::new();
        for p in packets.iter().take(count) {
            p.encode(&mut bw);
        }
        let anc = bw.finish();

        let mut out = Vec::with_capacity(RTP_HEADER_LEN + ANC_HEADER_LEN + anc.len());
        // RTP header, marker set (the low 16 sequence bits; the high 16 ride
        // the RFC 8331 extended sequence field below).
        let header = RtpHeader {
            payload_type: self.payload_type,
            marker: true,
            sequence: self.sequence as u16,
            timestamp: self.clock.rtp_timestamp(g2g_core::TaiNs(tai_ns)).get(),
            ssrc: self.ssrc,
        };
        out.extend_from_slice(&header.to_bytes());
        // RFC 8331 payload header.
        out.extend_from_slice(&((self.sequence >> 16) as u16).to_be_bytes()); // ext seq (high)
        out.extend_from_slice(&(anc.len() as u16).to_be_bytes()); // Length: the ANC data section
        out.push(count as u8); // ANC_Count
        out.push(field.to_bits() << 6); // F in the top 2 bits
        out.extend_from_slice(&[0, 0]); // reserved (22 bits total with the F byte's low 6)
        out.extend_from_slice(&anc);

        self.sequence = self.sequence.wrapping_add(1);
        out
    }
}

/// A depacketized ST 2110-40 frame: the RTP fields and the ANC packets it carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct St2110AncFrame {
    /// The full 32-bit extended sequence number (payload high 16 | RTP low 16).
    pub sequence: u32,
    pub rtp_timestamp: u32,
    pub field: AncField,
    pub packets: Vec<AncPacket>,
}

/// Depacketizes RFC 8331 RTP packets back into ANC packets.
#[derive(Debug, Default)]
pub struct St2110AncDepacketizer;

impl St2110AncDepacketizer {
    pub fn new() -> Self {
        Self
    }

    /// Parse one RTP packet into its ANC frame, or `None` if it is too short, the
    /// declared Length overruns the buffer, or any ANC packet fails parity /
    /// checksum validation.
    pub fn depacketize(&self, packet: &[u8]) -> Option<St2110AncFrame> {
        if packet.len() < RTP_HEADER_LEN + ANC_HEADER_LEN || packet[0] & 0xC0 != 0x80 {
            return None;
        }
        let seq_lo = u16::from_be_bytes([packet[2], packet[3]]);
        let rtp_timestamp = u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]]);

        let hdr = &packet[RTP_HEADER_LEN..];
        let seq_hi = u16::from_be_bytes([hdr[0], hdr[1]]);
        let length = usize::from(u16::from_be_bytes([hdr[2], hdr[3]]));
        let anc_count = hdr[4];
        let field = AncField::from_bits(hdr[5] >> 6);

        // Length must not overrun the datagram (attacker-controlled).
        let anc_data = hdr.get(ANC_HEADER_LEN..ANC_HEADER_LEN + length)?;
        let sequence = (u32::from(seq_hi) << 16) | u32::from(seq_lo);

        let mut r = BitReader::new(anc_data);
        let mut packets = Vec::with_capacity(usize::from(anc_count));
        for _ in 0..anc_count {
            packets.push(AncPacket::decode(&mut r)?);
        }
        Some(St2110AncFrame { sequence, rtp_timestamp, field, packets })
    }
}

// ================================================================
// MSB-first bit cursors for the 10-bit-word packing.
// ================================================================

/// Appends bits MSB-first into a byte buffer.
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        Self { out: Vec::new(), acc: 0, nbits: 0 }
    }

    /// Append the low `bits` (<= 24) of `value`, most-significant bit first.
    fn write(&mut self, value: u32, bits: u32) {
        let mask = if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 };
        self.acc = (self.acc << bits) | (value & mask);
        self.nbits += bits;
        while self.nbits >= 8 {
            self.nbits -= 8;
            self.out.push((self.acc >> self.nbits) as u8);
        }
    }

    fn bit_len(&self) -> usize {
        self.out.len() * 8 + self.nbits as usize
    }

    /// Zero-pad to the next 32-bit boundary (RFC 8331 word_align).
    fn align32(&mut self) {
        while self.bit_len() % 32 != 0 {
            self.write(0, 1);
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.write(0, pad);
        }
        self.out
    }
}

/// Reads bits MSB-first from a byte slice.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Read `bits` (<= 24) MSB-first, or `None` past the end.
    fn read(&mut self, bits: usize) -> Option<u32> {
        if self.pos + bits > self.data.len() * 8 {
            return None;
        }
        let mut v = 0u32;
        for _ in 0..bits {
            let byte = self.data[self.pos / 8];
            let bit = (byte >> (7 - self.pos % 8)) & 1;
            v = (v << 1) | u32::from(bit);
            self.pos += 1;
        }
        Some(v)
    }

    /// Skip padding to the next 32-bit boundary (word_align).
    fn align32(&mut self) {
        while self.pos % 32 != 0 && self.pos < self.data.len() * 8 {
            self.pos += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn word_parity_is_even_with_inverse_b9() {
        // 0x61 = 0110_0001, three 1s -> odd -> parity bit 1; b9 = 0.
        let w = encode_word(0x61);
        assert_eq!(w & 0xFF, 0x61);
        assert_eq!((w >> 8) & 1, 1, "even-parity bit");
        assert_eq!((w >> 9) & 1, 0, "b9 = NOT b8");
        // 0x03 = two 1s -> even -> parity 0; b9 = 1.
        assert_eq!((encode_word(0x03) >> 8) & 1, 0);
        assert_eq!((encode_word(0x03) >> 9) & 1, 1);
    }

    #[test]
    fn anc_packet_round_trips() {
        // A CEA-708 caption ANC packet (DID 0x61 / SDID 0x01) with a small payload.
        let pkt = AncPacket::generic(0x61, 0x01, vec![0x96, 0x69, 0x52, 0x00, 0xFF]);
        let mut w = BitWriter::new();
        pkt.encode(&mut w);
        let bytes = w.finish();
        assert_eq!(bytes.len() % 4, 0, "padded to a 32-bit boundary");

        let mut r = BitReader::new(&bytes);
        assert_eq!(AncPacket::decode(&mut r), Some(pkt));
    }

    #[test]
    fn rtp_round_trips_multiple_anc_packets() {
        let tai = 1_700_000_000_000_000_000u64;
        let mut tx = St2110AncPacketizer::new(100, 0xABCD);
        let rx = St2110AncDepacketizer::new();
        let packets = vec![
            AncPacket::generic(0x61, 0x01, vec![0x01, 0x02, 0x03]), // CEA-708
            AncPacket::generic(0x61, 0x02, vec![0xAA, 0xBB]),       // CEA-608
        ];

        let rtp = tx.packetize(&packets, tai, AncField::Progressive);
        // Marker set on an ANC frame.
        assert_eq!(rtp[1] & 0x80, 0x80, "RTP marker set");

        let frame = rx.depacketize(&rtp).expect("valid ANC frame");
        assert_eq!(frame.packets, packets);
        assert_eq!(frame.field, AncField::Progressive);
        assert_eq!(frame.sequence, 0);
        assert_eq!(frame.rtp_timestamp, MediaClock::video().rtp_timestamp(g2g_core::TaiNs(tai)).get());

        // Extended sequence spans the 16-bit RTP field: after 0x1_0000 packets the
        // high half is 1. Cheaply check the split by driving the counter.
        tx.sequence = 0x0001_2345;
        let rtp2 = tx.packetize(&packets, tai, AncField::Field1);
        let f2 = rx.depacketize(&rtp2).unwrap();
        assert_eq!(f2.sequence, 0x0001_2345);
        assert_eq!(f2.field, AncField::Field1);
    }

    #[test]
    fn rejects_corrupted_parity_and_checksum() {
        let mut tx = St2110AncPacketizer::new(100, 1);
        let rx = St2110AncDepacketizer::new();
        let good = tx.packetize(&[AncPacket::generic(0x61, 0x01, vec![0x11, 0x22])], 0, AncField::Progressive);

        // Flip a bit inside the 10-bit-word region (past the RTP header, the
        // 8-byte ANC header, and the 4-byte ANC-packet header): that breaks a
        // data word's parity or the checksum, so it must be rejected. (A flip in
        // the trailing zero word_align padding would be legitimately ignored.)
        let mut bad = good.clone();
        let word_region = RTP_HEADER_LEN + ANC_HEADER_LEN + 4;
        bad[word_region] ^= 0x80;
        assert!(rx.depacketize(&bad).is_none(), "corrupted ANC data is rejected");

        // A Length that overruns the datagram is rejected, not read out of bounds.
        let mut overrun = good.clone();
        overrun[RTP_HEADER_LEN + 2] = 0xFF; // Length high byte huge
        overrun[RTP_HEADER_LEN + 3] = 0xFF;
        assert!(rx.depacketize(&overrun).is_none(), "over-long Length rejected");
    }

    #[test]
    fn rejects_short_packets() {
        let rx = St2110AncDepacketizer::new();
        assert!(rx.depacketize(&[0u8; 12]).is_none(), "no room for the ANC header");
        assert!(rx.depacketize(&[]).is_none());
    }
}
