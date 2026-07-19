//! RFC 3550 RTP fixed header (M643): the 12-byte wire header every RTP
//! packetizer in the workspace emits, defined once. Pure `const fn`
//! byte-building, part of the heap-free subset, so the MCU packet sink and
//! the std packetizers in `g2g-plugins` (H.264, the ST 2110 essences) share
//! one implementation instead of five hand-rolled copies.
//!
//! Only the fixed header lives here: V=2, no padding, no extension, no CSRC
//! list, which is the shape every g2g payload format uses. Payload-format
//! headers (FU-A, RFC 4175 SRDs, RFC 8331 ANC) stay with their packetizers.

/// RTP fixed header length: V=2 with no CSRC list and no extension.
pub const RTP_HEADER_LEN: usize = 12;

/// The RFC 3550 fixed header fields a packetizer chooses per packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtpHeader {
    /// 7-bit payload type (a static PT like 0 = PCMU, or a dynamic 96..=127).
    pub payload_type: u8,
    /// Frame/talkspurt boundary marker; payload-format-defined semantics.
    pub marker: bool,
    /// Per-packet sequence number (the packetizer increments, wrapping).
    pub sequence: u16,
    /// Media-clock timestamp of the payload's sampling instant.
    pub timestamp: u32,
    /// Synchronization source identifier.
    pub ssrc: u32,
}

/// A parsed RTP packet: the fixed-header fields plus where the payload sits in
/// the datagram, after any CSRC list / extension header and before any padding.
/// [`RtpHeader::parse`] returns this; a depacketizer reads
/// `buf[payload_offset..payload_offset + payload_len]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtpParsed {
    /// The fixed-header fields (`RtpHeader::to_bytes` round-trips these).
    pub header: RtpHeader,
    /// Byte offset of the payload within the parsed datagram.
    pub payload_offset: usize,
    /// Payload length in bytes (padding already stripped).
    pub payload_len: usize,
}

impl RtpHeader {
    /// Parse an RTP datagram received from an arbitrary peer, returning the
    /// fixed-header fields and the payload's byte range. The inverse of
    /// [`to_bytes`](Self::to_bytes), but tolerant of the header variations a
    /// real sender may set that `to_bytes` never emits: a CSRC list (`CC`), a
    /// profile extension header (`X`), and payload padding (`P`).
    ///
    /// Parser discipline (never trust the wire, per the demuxer rule): every
    /// offset and length is folded with checked arithmetic and bounds-checked
    /// against `buf`, so a malformed or truncated datagram returns `None`
    /// rather than panicking or reading out of bounds. A non-RTP-v2 packet is
    /// rejected. Heap-free (part of the no-alloc subset), so an MCU RTP source
    /// uses it directly.
    pub fn parse(buf: &[u8]) -> Option<RtpParsed> {
        let b0 = *buf.first()?;
        if b0 >> 6 != 2 {
            return None; // only RTP version 2
        }
        let has_padding = b0 & 0x20 != 0;
        let has_extension = b0 & 0x10 != 0;
        let csrc_count = (b0 & 0x0F) as usize;
        let b1 = *buf.get(1)?;
        let marker = b1 & 0x80 != 0;
        let payload_type = b1 & 0x7F;
        let sequence = u16::from_be_bytes([*buf.get(2)?, *buf.get(3)?]);
        let timestamp =
            u32::from_be_bytes([*buf.get(4)?, *buf.get(5)?, *buf.get(6)?, *buf.get(7)?]);
        let ssrc = u32::from_be_bytes([*buf.get(8)?, *buf.get(9)?, *buf.get(10)?, *buf.get(11)?]);

        // Fixed header + the CSRC list (CC 32-bit identifiers).
        let mut offset = RTP_HEADER_LEN.checked_add(csrc_count.checked_mul(4)?)?;
        if has_extension {
            // The extension header is a 2-byte profile id + a 2-byte length in
            // 32-bit words, then that many words of extension data.
            let len_hi = *buf.get(offset.checked_add(2)?)?;
            let len_lo = *buf.get(offset.checked_add(3)?)?;
            let ext_words = u16::from_be_bytes([len_hi, len_lo]) as usize;
            offset = offset
                .checked_add(4)?
                .checked_add(ext_words.checked_mul(4)?)?;
        }

        // Padding, if present, is counted by the datagram's last byte.
        let mut end = buf.len();
        if has_padding {
            let pad = *buf.get(end.checked_sub(1)?)? as usize;
            end = end.checked_sub(pad)?;
        }
        if offset > end {
            return None; // header (and padding) overrun the datagram
        }
        Some(RtpParsed {
            header: RtpHeader {
                payload_type,
                marker,
                sequence,
                timestamp,
                ssrc,
            },
            payload_offset: offset,
            payload_len: end - offset,
        })
    }

    /// The header as it goes on the wire. Pure and heap-free; a packetizer
    /// prepends this to its payload.
    pub const fn to_bytes(self) -> [u8; RTP_HEADER_LEN] {
        let seq = self.sequence.to_be_bytes();
        let ts = self.timestamp.to_be_bytes();
        let ssrc = self.ssrc.to_be_bytes();
        [
            0x80, // V=2, P=0, X=0, CC=0
            (if self.marker { 0x80 } else { 0 }) | (self.payload_type & 0x7F),
            seq[0],
            seq[1],
            ts[0],
            ts[1],
            ts[2],
            ts[3],
            ssrc[0],
            ssrc[1],
            ssrc[2],
            ssrc[3],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_bytes_match_the_wire_layout() {
        let h = RtpHeader {
            payload_type: 96,
            marker: true,
            sequence: 0x0102,
            timestamp: 0x0304_0506,
            ssrc: 0x0708_090A,
        };
        assert_eq!(
            h.to_bytes(),
            [0x80, 0x80 | 96, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x0A],
            "V=2 | M+PT | seq | timestamp | ssrc, all big-endian"
        );
        let unmarked = RtpHeader { marker: false, ..h };
        assert_eq!(
            unmarked.to_bytes()[1],
            96,
            "marker bit clear leaves the bare PT"
        );
        // PT is 7 bits: bit 7 of an oversized PT must not leak into marker.
        let overwide = RtpHeader {
            payload_type: 0xFF,
            marker: false,
            ..h
        };
        assert_eq!(
            overwide.to_bytes()[1],
            0x7F,
            "payload type masked to 7 bits"
        );
    }

    #[test]
    fn parse_round_trips_to_bytes_and_finds_the_payload() {
        let h = RtpHeader {
            payload_type: 0, // PCMU
            marker: true,
            sequence: 0x1234,
            timestamp: 0xAABB_CCDD,
            ssrc: 0x0011_2233,
        };
        let mut dgram = [0u8; RTP_HEADER_LEN + 4];
        dgram[..RTP_HEADER_LEN].copy_from_slice(&h.to_bytes());
        dgram[RTP_HEADER_LEN..].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let p = RtpHeader::parse(&dgram).expect("valid RTP");
        assert_eq!(
            p.header, h,
            "fixed fields round-trip through to_bytes/parse"
        );
        assert_eq!(p.payload_offset, RTP_HEADER_LEN);
        assert_eq!(p.payload_len, 4);
        assert_eq!(
            &dgram[p.payload_offset..p.payload_offset + p.payload_len],
            &[0xDE, 0xAD, 0xBE, 0xEF]
        );
    }

    #[test]
    fn parse_handles_csrc_extension_and_padding() {
        // V=2, P=1, X=1, CC=1; PT=96, no marker; then 1 CSRC word, a 1-word
        // extension, 2 payload bytes, and 2 padding bytes (last byte = 2).
        let d: [u8; 26] = [
            0b1011_0001,
            96, // V/P/X/CC, M/PT
            0x00,
            0x01, // sequence
            0,
            0,
            0,
            0, // timestamp
            0,
            0,
            0,
            0, // ssrc
            9,
            9,
            9,
            9, // 1 CSRC identifier
            0xBE,
            0xDE,
            0x00,
            0x01, // ext profile + len = 1 word
            1,
            2,
            3,
            4, // 1 word of extension data
            0x55,
            0x66, // payload (padding count byte follows in the next 2)
        ];
        // Append the 2 padding bytes (0x00, count=2) to a 28-byte datagram.
        let mut dg = [0u8; 28];
        dg[..26].copy_from_slice(&d);
        dg[26] = 0;
        dg[27] = 2;
        let p = RtpHeader::parse(&dg).expect("valid RTP with CC/X/P");
        assert_eq!(p.header.payload_type, 96);
        assert_eq!(p.payload_len, 2, "CSRC, extension and padding all excluded");
        assert_eq!(
            &dg[p.payload_offset..p.payload_offset + p.payload_len],
            &[0x55, 0x66]
        );
    }

    #[test]
    fn parse_rejects_malformed_input_without_panicking() {
        assert!(RtpHeader::parse(&[]).is_none(), "empty");
        assert!(
            RtpHeader::parse(&[0x80, 0]).is_none(),
            "truncated fixed header"
        );
        assert!(RtpHeader::parse(&[0x00; 12]).is_none(), "version != 2");
        // CC=15 claims 60 CSRC bytes a 12-byte datagram does not have.
        let mut d = [0u8; RTP_HEADER_LEN];
        d[0] = 0x8F; // V=2, CC=15
        assert!(
            RtpHeader::parse(&d).is_none(),
            "CSRC list overruns the datagram"
        );
        // Padding count larger than the datagram.
        let mut pad = [0u8; RTP_HEADER_LEN + 1];
        pad[0] = 0xA0; // V=2, P=1
        pad[RTP_HEADER_LEN] = 0xFF; // pad count 255 > available
        assert!(RtpHeader::parse(&pad).is_none(), "padding underflows");
    }
}
