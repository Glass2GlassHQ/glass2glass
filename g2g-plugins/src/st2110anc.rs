//! ST 2110-40 ancillary data over RTP: SMPTE ST 291 ANC packets (closed
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
//! boundary. The words are the ST 291 words of `st291`. Parsing is fully
//! bounds-checked and rejects a message whose words fail `st291`'s parity or
//! checksum checks (never trust the stream, AGENTS.md).

use alloc::vec::Vec;

use g2g_core::rtp::{RtpHeader, RTP_HEADER_LEN};
use g2g_core::MediaClock;

use crate::annexb::{BitReader, BitWriter};
use crate::st291::{AncFieldOverflow, AncPacket};

/// RFC 8331 payload header length (ext-seq + length + ANC_Count + F/reserved).
const ANC_HEADER_LEN: usize = 8;
const WORD_ALIGN_BITS: usize = 32;
// RFC 8331 S flag and 7-bit StreamNum, written as zero and skipped on receive
const STREAM_FLAG_AND_NUMBER_BITS: u32 = 8;

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
        Self {
            payload_type: payload_type & 0x7F,
            ssrc,
            sequence: 0,
            clock: MediaClock::video(),
        }
    }

    /// The media clock, for recovering a packet's PTP time on the receive side.
    pub fn media_clock(&self) -> MediaClock {
        self.clock
    }

    /// Build one RTP packet carrying `packets` (up to 255), timestamped at the
    /// video frame's PTP/TAI time. The RTP marker is set (all this frame's ANC is
    /// in this packet).
    pub fn packetize(
        &mut self,
        packets: &[AncPacket],
        tai_ns: u64,
        field: AncField,
    ) -> Result<Vec<u8>, AncFieldOverflow> {
        let count = packets.len().min(0xFF);
        // Bit-pack the ANC data section first so its length is known for the header.
        let mut bw = BitWriter::default();
        for p in packets.iter().take(count) {
            write_rfc8331_packet(p, &mut bw)?;
        }
        let anc = bw.into_bytes();

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
        Ok(out)
    }
}

fn write_rfc8331_packet(packet: &AncPacket, w: &mut BitWriter) -> Result<(), AncFieldOverflow> {
    packet.write(w, STREAM_FLAG_AND_NUMBER_BITS)?;
    while !w.bit_pos().is_multiple_of(WORD_ALIGN_BITS) {
        w.write_bit(0);
    }
    Ok(())
}

fn read_rfc8331_packet(r: &mut BitReader) -> Option<AncPacket> {
    let packet = AncPacket::read(r, STREAM_FLAG_AND_NUMBER_BITS)?;
    skip_word_align(r);
    Some(packet)
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
        let parsed = RtpHeader::parse(packet)?;
        let seq_lo = parsed.header.sequence;
        let rtp_timestamp = parsed.header.timestamp;

        let hdr = &packet[parsed.payload_offset..][..parsed.payload_len];
        if hdr.len() < ANC_HEADER_LEN {
            return None;
        }
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
            packets.push(read_rfc8331_packet(&mut r)?);
        }
        Some(St2110AncFrame {
            sequence,
            rtp_timestamp,
            field,
            packets,
        })
    }
}

// word_align, a last packet with its padding cut short still decodes
fn skip_word_align(r: &mut BitReader) {
    let misalignment = r.bit_pos() % WORD_ALIGN_BITS;
    if misalignment != 0 {
        let _ = r.skip_bits(WORD_ALIGN_BITS - misalignment);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn anc_packet_round_trips() {
        // A CEA-708 caption ANC packet (DID 0x61 / SDID 0x01) with a small payload.
        let pkt = AncPacket::from_bytes(0x61, 0x01, &[0x96, 0x69, 0x52, 0x00, 0xFF]);
        let mut w = BitWriter::default();
        write_rfc8331_packet(&pkt, &mut w).unwrap();
        let bytes = w.into_bytes();
        assert_eq!(bytes.len() % 4, 0, "padded to a 32-bit boundary");

        let mut r = BitReader::new(&bytes);
        assert_eq!(read_rfc8331_packet(&mut r), Some(pkt));
    }

    #[test]
    fn depacketizes_past_csrcs_extension_and_padding() {
        let mut tx = St2110AncPacketizer::new(100, 0xABCD);
        let packets = vec![AncPacket::from_bytes(0x61, 0x01, &[0x01, 0x02, 0x03])];
        let rtp = tx.packetize(&packets, 0, AncField::Progressive).unwrap();
        let shaped = crate::st2110video::with_csrc_extension_and_padding(&rtp);
        let frame = St2110AncDepacketizer::new().depacketize(&shaped);
        assert_eq!(frame.expect("valid ANC frame").packets, packets);
    }

    #[test]
    fn rtp_round_trips_multiple_anc_packets() {
        let tai = 1_700_000_000_000_000_000u64;
        let mut tx = St2110AncPacketizer::new(100, 0xABCD);
        let rx = St2110AncDepacketizer::new();
        let packets = vec![
            AncPacket::from_bytes(0x61, 0x01, &[0x01, 0x02, 0x03]), // CEA-708
            AncPacket::from_bytes(0x61, 0x02, &[0xAA, 0xBB]),       // CEA-608
        ];

        let rtp = tx.packetize(&packets, tai, AncField::Progressive).unwrap();
        // Marker set on an ANC frame.
        assert_eq!(rtp[1] & 0x80, 0x80, "RTP marker set");

        let frame = rx.depacketize(&rtp).expect("valid ANC frame");
        assert_eq!(frame.packets, packets);
        assert_eq!(frame.field, AncField::Progressive);
        assert_eq!(frame.sequence, 0);
        assert_eq!(
            frame.rtp_timestamp,
            MediaClock::video()
                .rtp_timestamp(g2g_core::TaiNs(tai))
                .get()
        );

        // Extended sequence spans the 16-bit RTP field: after 0x1_0000 packets the
        // high half is 1. Cheaply check the split by driving the counter.
        tx.sequence = 0x0001_2345;
        let rtp2 = tx.packetize(&packets, tai, AncField::Field1).unwrap();
        let f2 = rx.depacketize(&rtp2).unwrap();
        assert_eq!(f2.sequence, 0x0001_2345);
        assert_eq!(f2.field, AncField::Field1);
    }

    #[test]
    fn rejects_corrupted_parity_and_checksum() {
        let mut tx = St2110AncPacketizer::new(100, 1);
        let rx = St2110AncDepacketizer::new();
        let good = tx
            .packetize(
                &[AncPacket::from_bytes(0x61, 0x01, &[0x11, 0x22])],
                0,
                AncField::Progressive,
            )
            .unwrap();

        // Flip a bit inside the 10-bit-word region (past the RTP header, the
        // 8-byte ANC header, and the 4-byte ANC-packet header): that breaks a
        // data word's parity or the checksum, so it must be rejected. (A flip in
        // the trailing zero word_align padding would be legitimately ignored.)
        let mut bad = good.clone();
        let word_region = RTP_HEADER_LEN + ANC_HEADER_LEN + 4;
        bad[word_region] ^= 0x80;
        assert!(
            rx.depacketize(&bad).is_none(),
            "corrupted ANC data is rejected"
        );

        // A Length that overruns the datagram is rejected, not read out of bounds.
        let mut overrun = good.clone();
        overrun[RTP_HEADER_LEN + 2] = 0xFF; // Length high byte huge
        overrun[RTP_HEADER_LEN + 3] = 0xFF;
        assert!(
            rx.depacketize(&overrun).is_none(),
            "over-long Length rejected"
        );
    }

    #[test]
    fn rejects_short_packets() {
        let rx = St2110AncDepacketizer::new();
        assert!(
            rx.depacketize(&[0u8; 12]).is_none(),
            "no room for the ANC header"
        );
        assert!(rx.depacketize(&[]).is_none());
    }
}
