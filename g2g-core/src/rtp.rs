//! RFC 3550 RTP fixed header (M643): the 12-byte wire header every RTP
//! packetizer in the workspace emits, defined once. Pure `const fn`
//! byte-building, part of the heap-free subset, so the MCU packet sink and
//! the std packetizers in `g2g-plugins` (H.264, the ST 2110 essences) share
//! one implementation instead of five hand-rolled copies.
//!
//! The fixed header lives here: V=2, no padding, no CSRC list, which is the
//! shape every g2g payload format uses, plus the RFC 8285 header extension
//! block and the two capture-time extensions carried in it. Payload-format
//! headers (FU-A, RFC 4175 SRDs, RFC 8331 ANC) stay with their packetizers.

use core::num::NonZeroU8;

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

/// The fields and byte length of an RTP header. Unlike [`RtpParsed`], this
/// does not inspect the payload or its padding count, so SRTP can parse the
/// authenticated header before decrypting the payload and padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtpParsedHeader {
    /// The fixed-header fields (`RtpHeader::to_bytes` round-trips these).
    pub header: RtpHeader,
    /// Byte offset immediately after the CSRC list and extension header.
    pub payload_offset: usize,
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
        let parsed_header = Self::parse_header(buf)?;
        let b0 = *buf.first()?;
        let has_padding = b0 & 0x20 != 0;

        // Padding, if present, is counted by the datagram's last byte.
        let mut end = buf.len();
        if has_padding {
            let pad = *buf.get(end.checked_sub(1)?)? as usize;
            end = end.checked_sub(pad)?;
        }
        if parsed_header.payload_offset > end {
            return None; // header (and padding) overrun the datagram
        }
        Some(RtpParsed {
            header: parsed_header.header,
            payload_offset: parsed_header.payload_offset,
            payload_len: end - parsed_header.payload_offset,
        })
    }

    /// Parse the authenticated RTP header without reading the payload. This
    /// accepts CSRC identifiers and an RTP extension and rejects any header
    /// whose declared fields exceed `buf`.
    pub fn parse_header(buf: &[u8]) -> Option<RtpParsedHeader> {
        let b0 = *buf.first()?;
        if b0 >> 6 != 2 {
            return None;
        }
        let has_extension = b0 & 0x10 != 0;
        let csrc_count = (b0 & 0x0F) as usize;
        let b1 = *buf.get(1)?;
        let marker = b1 & 0x80 != 0;
        let payload_type = b1 & 0x7F;
        let sequence = u16::from_be_bytes([*buf.get(2)?, *buf.get(3)?]);
        let timestamp =
            u32::from_be_bytes([*buf.get(4)?, *buf.get(5)?, *buf.get(6)?, *buf.get(7)?]);
        let ssrc = u32::from_be_bytes([*buf.get(8)?, *buf.get(9)?, *buf.get(10)?, *buf.get(11)?]);

        let mut payload_offset = RTP_HEADER_LEN.checked_add(csrc_count.checked_mul(4)?)?;
        if has_extension {
            let extension_length_high = *buf.get(payload_offset.checked_add(2)?)?;
            let extension_length_low = *buf.get(payload_offset.checked_add(3)?)?;
            let extension_words =
                u16::from_be_bytes([extension_length_high, extension_length_low]) as usize;
            payload_offset = payload_offset
                .checked_add(4)?
                .checked_add(extension_words.checked_mul(4)?)?;
        }
        if payload_offset > buf.len() {
            return None;
        }

        Some(RtpParsedHeader {
            header: RtpHeader {
                payload_type,
                marker,
                sequence,
                timestamp,
                ssrc,
            },
            payload_offset,
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

const RTP_VERSION: u8 = 2;
const RTP_VERSION_SHIFT: u8 = 6;
const EXTENSION_BIT: u8 = 0x10;
const CSRC_COUNT_MASK: u8 = 0x0F;
const CSRC_LEN: usize = 4;
const EXTENSION_HEADER_LEN: usize = 4;
const EXTENSION_WORD_LEN: usize = 4;

pub const ONE_BYTE_EXTENSION_PROFILE: u16 = 0xBEDE;
pub const TWO_BYTE_EXTENSION_PROFILE: u16 = 0x1000;
const TWO_BYTE_PROFILE_MASK: u16 = 0xFFF0;
const PADDING_BYTE: u8 = 0;
const ONE_BYTE_ID_SHIFT: u8 = 4;
const ONE_BYTE_LENGTH_MASK: u8 = 0x0F;
const ONE_BYTE_PADDING_ID: u8 = 0;
const ONE_BYTE_MAX_ID: u8 = 14;
const ONE_BYTE_RESERVED_ID: u8 = 15;
const ONE_BYTE_MAX_DATA_LEN: usize = 16;
const ONE_BYTE_ELEMENT_HEADER_LEN: usize = 1;
const TWO_BYTE_MAX_DATA_LEN: usize = u8::MAX as usize;
const TWO_BYTE_ELEMENT_HEADER_LEN: usize = 2;

pub const ABS_CAPTURE_TIME_URI: &str =
    "http://www.webrtc.org/experiments/rtp-hdrext/abs-capture-time";
pub const NTP_64_URI: &str = "urn:ietf:params:rtp-hdrext:ntp-64";
pub const NTP_TIMESTAMP_LEN: usize = 8;
// abs-capture-time may append a signed capture clock offset
const ABS_CAPTURE_TIME_WITH_OFFSET_LEN: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RtpHeaderExtension<'a> {
    pub id: u8,
    pub data: &'a [u8],
}

fn fits_one_byte_form(extension: &RtpHeaderExtension<'_>) -> bool {
    (1..=ONE_BYTE_MAX_ID).contains(&extension.id)
        && (1..=ONE_BYTE_MAX_DATA_LEN).contains(&extension.data.len())
}

fn fits_two_byte_form(extension: &RtpHeaderExtension<'_>) -> bool {
    extension.id != PADDING_BYTE && extension.data.len() <= TWO_BYTE_MAX_DATA_LEN
}

fn extension_elements_len(extensions: &[RtpHeaderExtension<'_>]) -> Option<(usize, bool)> {
    let one_byte = extensions.iter().all(fits_one_byte_form);
    if !one_byte && !extensions.iter().all(fits_two_byte_form) {
        return None;
    }
    let element_header_len = if one_byte {
        ONE_BYTE_ELEMENT_HEADER_LEN
    } else {
        TWO_BYTE_ELEMENT_HEADER_LEN
    };
    let len = extensions.iter().try_fold(0usize, |total, extension| {
        total
            .checked_add(element_header_len)?
            .checked_add(extension.data.len())
    })?;
    Some((len, one_byte))
}

fn extension_block(packet: &[u8]) -> Option<(u16, &[u8])> {
    let first = *packet.first()?;
    if first >> RTP_VERSION_SHIFT != RTP_VERSION || first & EXTENSION_BIT == 0 {
        return None;
    }
    let csrc_count = (first & CSRC_COUNT_MASK) as usize;
    let start = RTP_HEADER_LEN.checked_add(csrc_count.checked_mul(CSRC_LEN)?)?;
    let header = packet.get(start..start.checked_add(EXTENSION_HEADER_LEN)?)?;
    let profile = u16::from_be_bytes([header[0], header[1]]);
    let words = u16::from_be_bytes([header[2], header[3]]) as usize;
    let data_start = start + EXTENSION_HEADER_LEN;
    let data_end = data_start.checked_add(words.checked_mul(EXTENSION_WORD_LEN)?)?;
    Some((profile, packet.get(data_start..data_end)?))
}

impl RtpHeader {
    pub fn extended_header_len(extensions: &[RtpHeaderExtension<'_>]) -> Option<usize> {
        if extensions.is_empty() {
            return Some(RTP_HEADER_LEN);
        }
        let (elements_len, _) = extension_elements_len(extensions)?;
        let padded = elements_len.checked_next_multiple_of(EXTENSION_WORD_LEN)?;
        u16::try_from(padded / EXTENSION_WORD_LEN).ok()?;
        RTP_HEADER_LEN
            .checked_add(EXTENSION_HEADER_LEN)?
            .checked_add(padded)
    }

    pub fn write_with_extensions(
        self,
        extensions: &[RtpHeaderExtension<'_>],
        out: &mut [u8],
    ) -> Option<usize> {
        let total = Self::extended_header_len(extensions)?;
        let out = out.get_mut(..total)?;
        out[..RTP_HEADER_LEN].copy_from_slice(&self.to_bytes());
        if extensions.is_empty() {
            return Some(total);
        }
        let (_, one_byte) = extension_elements_len(extensions)?;
        out[0] |= EXTENSION_BIT;
        let profile = if one_byte {
            ONE_BYTE_EXTENSION_PROFILE
        } else {
            TWO_BYTE_EXTENSION_PROFILE
        };
        let words = ((total - RTP_HEADER_LEN - EXTENSION_HEADER_LEN) / EXTENSION_WORD_LEN) as u16;
        let mut at = RTP_HEADER_LEN;
        out[at..at + 2].copy_from_slice(&profile.to_be_bytes());
        out[at + 2..at + EXTENSION_HEADER_LEN].copy_from_slice(&words.to_be_bytes());
        at += EXTENSION_HEADER_LEN;
        for extension in extensions {
            let len = extension.data.len();
            if one_byte {
                out[at] = (extension.id << ONE_BYTE_ID_SHIFT) | (len - 1) as u8;
                at += ONE_BYTE_ELEMENT_HEADER_LEN;
            } else {
                out[at] = extension.id;
                out[at + 1] = len as u8;
                at += TWO_BYTE_ELEMENT_HEADER_LEN;
            }
            out[at..at + len].copy_from_slice(extension.data);
            at += len;
        }
        out[at..].fill(PADDING_BYTE);
        Some(total)
    }

    pub fn find_extension(packet: &[u8], id: u8) -> Option<&[u8]> {
        let (profile, mut data) = extension_block(packet)?;
        let one_byte = profile == ONE_BYTE_EXTENSION_PROFILE;
        if !one_byte && profile & TWO_BYTE_PROFILE_MASK != TWO_BYTE_EXTENSION_PROFILE {
            return None;
        }
        let mut found = None;
        while let Some((&first, rest)) = data.split_first() {
            if first == PADDING_BYTE {
                data = rest;
                continue;
            }
            let (element_id, len, body) = if one_byte {
                let element_id = first >> ONE_BYTE_ID_SHIFT;
                // RFC 8285 stops parsing at these, keeping the elements before them
                if element_id == ONE_BYTE_RESERVED_ID || element_id == ONE_BYTE_PADDING_ID {
                    break;
                }
                (
                    element_id,
                    (first & ONE_BYTE_LENGTH_MASK) as usize + 1,
                    rest,
                )
            } else {
                let (&len, body) = rest.split_first()?;
                (first, len as usize, body)
            };
            let element = body.get(..len)?;
            if element_id == id && found.is_none() {
                found = Some(element);
            }
            data = &body[len..];
        }
        found
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaptureTimeExtensionIds {
    pub abs_capture_time: Option<NonZeroU8>,
    pub ntp_64: Option<NonZeroU8>,
}

impl CaptureTimeExtensionIds {
    pub fn is_empty(&self) -> bool {
        self.abs_capture_time.is_none() && self.ntp_64.is_none()
    }

    pub fn assign_extmap(&mut self, id: NonZeroU8, uri: &str) -> bool {
        match uri {
            ABS_CAPTURE_TIME_URI => self.abs_capture_time = Some(id),
            NTP_64_URI => self.ntp_64 = Some(id),
            _ => return false,
        }
        true
    }

    pub fn elements<'a>(
        &self,
        ntp_bytes: &'a [u8; NTP_TIMESTAMP_LEN],
    ) -> impl Iterator<Item = RtpHeaderExtension<'a>> {
        [self.abs_capture_time, self.ntp_64]
            .into_iter()
            .flatten()
            .map(move |id| RtpHeaderExtension {
                id: id.get(),
                data: ntp_bytes,
            })
    }

    pub fn read(&self, packet: &[u8]) -> Option<u64> {
        let abs_capture_time = self
            .abs_capture_time
            .and_then(|id| RtpHeader::find_extension(packet, id.get()))
            .filter(|data| {
                data.len() == NTP_TIMESTAMP_LEN || data.len() == ABS_CAPTURE_TIME_WITH_OFFSET_LEN
            });
        let ntp_64 = self
            .ntp_64
            .and_then(|id| RtpHeader::find_extension(packet, id.get()))
            .filter(|data| data.len() == NTP_TIMESTAMP_LEN);
        [abs_capture_time, ntp_64]
            .into_iter()
            .flatten()
            .filter_map(|data| data.get(..NTP_TIMESTAMP_LEN)?.try_into().ok())
            .map(u64::from_be_bytes)
            // GStreamer sends zero when it has no time to stamp
            .find(|&ntp| ntp != 0)
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

        dg[27] = 0xFF;
        let header = RtpHeader::parse_header(&dg).expect("payload is opaque to header parsing");
        assert_eq!(header.payload_offset, 24);
        assert!(
            RtpHeader::parse(&dg).is_none(),
            "full parsing checks padding"
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

    const CAPTURE_NTP: u64 = 0xE8F0_1234_8000_0001;
    const PAYLOAD: [u8; 3] = [0x65, 0xAA, 0xBB];

    fn header() -> RtpHeader {
        RtpHeader {
            payload_type: 96,
            marker: true,
            sequence: 7,
            timestamp: 9000,
            ssrc: 0x1234_5678,
        }
    }

    fn stamped_packet(ids: CaptureTimeExtensionIds) -> ([u8; 64], usize) {
        let ntp_bytes = CAPTURE_NTP.to_be_bytes();
        let mut elements = [RtpHeaderExtension { id: 0, data: &[] }; 2];
        let mut count = 0;
        for element in ids.elements(&ntp_bytes) {
            elements[count] = element;
            count += 1;
        }
        let mut packet = [0u8; 64];
        let header_len = header()
            .write_with_extensions(&elements[..count], &mut packet)
            .expect("fits");
        packet[header_len..header_len + PAYLOAD.len()].copy_from_slice(&PAYLOAD);
        (packet, header_len + PAYLOAD.len())
    }

    #[test]
    fn one_byte_capture_time_extensions_round_trip_in_the_spec_layout() {
        let ids = CaptureTimeExtensionIds {
            abs_capture_time: NonZeroU8::new(3),
            ntp_64: NonZeroU8::new(5),
        };
        let (packet, len) = stamped_packet(ids);
        let packet = &packet[..len];
        assert_eq!(packet[0] & EXTENSION_BIT, EXTENSION_BIT, "X bit set");
        assert_eq!(&packet[12..14], &ONE_BYTE_EXTENSION_PROFILE.to_be_bytes());
        // Two 9-byte elements pad to 20 bytes, five words.
        assert_eq!(&packet[14..16], &[0, 5]);
        assert_eq!(packet[16], 0x37, "id 3, length field 7 (8 bytes)");
        assert_eq!(&packet[17..25], &CAPTURE_NTP.to_be_bytes());
        assert_eq!(packet[25], 0x57, "id 5, length field 7");
        assert_eq!(&packet[34..36], &[0, 0], "zero padding to the word");

        let parsed = RtpHeader::parse(packet).expect("valid RTP");
        assert_eq!(parsed.header, header());
        assert_eq!(
            &packet[parsed.payload_offset..parsed.payload_offset + parsed.payload_len],
            &PAYLOAD
        );
        assert_eq!(ids.read(packet), Some(CAPTURE_NTP));
        let ntp_only = CaptureTimeExtensionIds {
            abs_capture_time: None,
            ntp_64: NonZeroU8::new(5),
        };
        assert_eq!(ntp_only.read(packet), Some(CAPTURE_NTP));
        let unknown = CaptureTimeExtensionIds {
            abs_capture_time: NonZeroU8::new(4),
            ntp_64: None,
        };
        assert_eq!(unknown.read(packet), None, "id 4 is not in the packet");
    }

    #[test]
    fn ids_past_fourteen_take_the_two_byte_form() {
        let ids = CaptureTimeExtensionIds {
            abs_capture_time: NonZeroU8::new(20),
            ntp_64: NonZeroU8::new(200),
        };
        let (packet, len) = stamped_packet(ids);
        let packet = &packet[..len];
        assert_eq!(&packet[12..14], &TWO_BYTE_EXTENSION_PROFILE.to_be_bytes());
        assert_eq!(&packet[16..18], &[20, 8], "id byte, length byte");
        assert_eq!(&packet[26..28], &[200, 8]);
        assert_eq!(
            RtpHeader::find_extension(packet, 200),
            Some(&packet[28..36])
        );
        assert_eq!(ids.read(packet), Some(CAPTURE_NTP));
    }

    #[test]
    fn abs_capture_time_with_a_clock_offset_is_read() {
        let mut data = [0u8; ABS_CAPTURE_TIME_WITH_OFFSET_LEN];
        data[..8].copy_from_slice(&CAPTURE_NTP.to_be_bytes());
        data[8..].copy_from_slice(&(-1i64).to_be_bytes());
        let mut packet = [0u8; 40];
        header()
            .write_with_extensions(&[RtpHeaderExtension { id: 9, data: &data }], &mut packet)
            .unwrap();
        let ids = CaptureTimeExtensionIds {
            abs_capture_time: NonZeroU8::new(9),
            ntp_64: None,
        };
        assert_eq!(packet[16], 0x9F, "id 9, length field 15 (16 bytes)");
        assert_eq!(ids.read(&packet), Some(CAPTURE_NTP));
    }

    #[test]
    fn malformed_extensions_fail_the_lookup_without_panicking() {
        let ids = CaptureTimeExtensionIds {
            abs_capture_time: NonZeroU8::new(3),
            ntp_64: NonZeroU8::new(5),
        };
        let (good, len) = stamped_packet(ids);

        // The second element claims 16 bytes where the block holds 8 more.
        let mut overrun = good;
        overrun[25] = 0x5F;
        assert_eq!(RtpHeader::find_extension(&overrun[..len], 5), None);
        assert_eq!(
            RtpHeader::find_extension(&overrun[..len], 3),
            None,
            "one bad element distrusts the whole block"
        );
        assert_eq!(ids.read(&overrun[..len]), None);

        // The block length runs past the datagram.
        let mut long_block = good;
        long_block[15] = 0xFF;
        assert_eq!(ids.read(&long_block[..len]), None);
        assert!(RtpHeader::parse(&long_block[..len]).is_none());

        // Truncated inside the extension header and inside an element.
        for cut in [13, 15, 20] {
            assert_eq!(ids.read(&good[..cut]), None, "cut at {cut}");
        }

        // An unknown profile is not RFC 8285.
        let mut profile = good;
        profile[12] = 0xAB;
        assert_eq!(ids.read(&profile[..len]), None);

        // Id 15 stops parsing, so the elements after it are not read.
        let mut reserved = good;
        reserved[25] = 0xF7;
        assert_eq!(
            RtpHeader::find_extension(&reserved[..len], 3).map(<[u8]>::len),
            Some(8)
        );
        assert_eq!(RtpHeader::find_extension(&reserved[..len], 5), None);

        // A zero NTP time is a sender without a clock reading.
        let mut zero = good;
        zero[17..25].fill(0);
        zero[26..34].fill(0);
        assert_eq!(ids.read(&zero[..len]), None);

        // An ntp-64 element of the wrong length is ignored.
        let mut short = [0u8; 40];
        let written = header()
            .write_with_extensions(
                &[RtpHeaderExtension {
                    id: 5,
                    data: &[1; 4],
                }],
                &mut short,
            )
            .unwrap();
        assert_eq!(ids.read(&short[..written]), None);
    }

    #[test]
    fn writing_rejects_elements_no_form_can_carry() {
        let mut out = [0u8; 600];
        let reserved_padding_id = RtpHeaderExtension { id: 0, data: &[1] };
        assert_eq!(
            header().write_with_extensions(&[reserved_padding_id], &mut out),
            None
        );
        let oversized = [0u8; 256];
        let too_long = RtpHeaderExtension {
            id: 1,
            data: &oversized,
        };
        assert_eq!(header().write_with_extensions(&[too_long], &mut out), None);
        let element = RtpHeaderExtension { id: 1, data: &[1] };
        assert_eq!(
            header().write_with_extensions(&[element], &mut out[..12]),
            None,
            "output too short"
        );
        assert_eq!(
            header().write_with_extensions(&[], &mut out),
            Some(RTP_HEADER_LEN),
            "no elements is the plain header"
        );
        assert_eq!(out[0], 0x80, "and leaves the X bit clear");
    }
}
