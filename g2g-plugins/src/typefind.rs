//! Container content sniffing (M112): guess a [`ByteStreamEncoding`] from the
//! first bytes of a stream, the `typefind` analog.
//!
//! The typed `Caps` model has no "untyped bytes" variant, so a byte source must
//! declare which container it carries before a demuxer can negotiate. This lets
//! `FileSrc` (and a future HTTP source) pick that automatically instead of the
//! caller naming it: read a header, match a magic signature, emit the matching
//! `Caps::ByteStream{encoding}`. Pure `no_std`, no allocation.

use g2g_core::ByteStreamEncoding;

/// MPEG-TS packet stride; the sync byte recurs at this interval.
const TS_PACKET_LEN: usize = 188;
const TS_SYNC: u8 = 0x47;
/// EBML magic (Matroska / WebM): the leading bytes of the EBML header element.
const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];
/// Ogg page capture pattern.
const OGG_MAGIC: [u8; 4] = *b"OggS";
/// FLV signature: the first three bytes of an FLV header.
const FLV_MAGIC: [u8; 3] = *b"FLV";

/// Guess the container encoding from a stream's leading bytes, or `None` if no
/// signature matches. Pass at least a few hundred bytes so MPEG-TS can be
/// confirmed across packet boundaries (a lone `0x47` is too weak to trust).
pub fn sniff(header: &[u8]) -> Option<ByteStreamEncoding> {
    if header.starts_with(&EBML_MAGIC) {
        return Some(ByteStreamEncoding::Matroska);
    }
    if header.starts_with(&OGG_MAGIC) {
        return Some(ByteStreamEncoding::Ogg);
    }
    if header.starts_with(&FLV_MAGIC) {
        return Some(ByteStreamEncoding::Flv);
    }
    if looks_like_mpegts(header) {
        return Some(ByteStreamEncoding::MpegTs);
    }
    None
}

/// True when the sync byte recurs at the 188-byte packet stride. Requires at
/// least one recurrence so a stray leading `0x47` is not a false positive,
/// unless fewer than two packets are present (then the lead byte is all we have).
fn looks_like_mpegts(header: &[u8]) -> bool {
    if header.first() != Some(&TS_SYNC) {
        return false;
    }
    let mut confirmed = 0;
    let mut off = TS_PACKET_LEN;
    while off < header.len() {
        if header[off] != TS_SYNC {
            return false;
        }
        confirmed += 1;
        off += TS_PACKET_LEN;
    }
    confirmed >= 1 || header.len() <= TS_PACKET_LEN
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn detects_matroska_by_ebml_magic() {
        let mut data = vec![0x1A, 0x45, 0xDF, 0xA3];
        data.extend_from_slice(&[0x01, 0x02, 0x03]);
        assert_eq!(sniff(&data), Some(ByteStreamEncoding::Matroska));
    }

    #[test]
    fn detects_ogg_by_capture_pattern() {
        assert_eq!(sniff(b"OggS\0\x02\0\0"), Some(ByteStreamEncoding::Ogg));
    }

    #[test]
    fn detects_flv_by_signature() {
        assert_eq!(sniff(b"FLV\x01\x05\0\0\0\x09"), Some(ByteStreamEncoding::Flv));
    }

    #[test]
    fn detects_mpegts_by_sync_stride() {
        // Two 188-byte packets: sync byte at 0 and 188.
        let mut data = vec![0u8; TS_PACKET_LEN * 2 + 1];
        data[0] = 0x47;
        data[TS_PACKET_LEN] = 0x47;
        data[TS_PACKET_LEN * 2] = 0x47;
        assert_eq!(sniff(&data), Some(ByteStreamEncoding::MpegTs));
    }

    #[test]
    fn rejects_stray_sync_byte() {
        // 0x47 at offset 0 but not at the packet stride: not TS.
        let mut data = vec![0u8; TS_PACKET_LEN * 2];
        data[0] = 0x47;
        // offset 188 is 0x00, so the stride check fails.
        assert_eq!(sniff(&data), None);
    }

    #[test]
    fn returns_none_for_unknown() {
        assert_eq!(sniff(&[0xDE, 0xAD, 0xBE, 0xEF]), None);
        assert_eq!(sniff(&[]), None);
        // RIFF/AVI, not a container we sniff.
        assert_eq!(sniff(b"RIFF\0\0\0\0AVI "), None);
    }

    #[test]
    fn ebml_takes_precedence_over_a_leading_0x47() {
        // EBML magic never starts with 0x47, so no ambiguity; sanity check that
        // a Matroska stream is not misread as TS.
        let data: Vec<u8> = EBML_MAGIC.iter().chain([0x47u8; 200].iter()).copied().collect();
        assert_eq!(sniff(&data), Some(ByteStreamEncoding::Matroska));
    }
}
