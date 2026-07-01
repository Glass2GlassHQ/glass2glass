//! Content sniffing (M112, text + MP4 M478): guess a media type from the first
//! bytes of a stream, the `typefind` analog.
//!
//! The typed `Caps` model has no "untyped bytes" variant, so a byte source must
//! declare what it carries before a demuxer / parser can negotiate. This lets
//! `FileSrc` (and a future HTTP source) pick that automatically instead of the
//! caller naming it: read a header, match a magic signature, and emit the
//! matching `Caps` ([`sniff_caps`]). Container magic yields a
//! `Caps::ByteStream{encoding}`; a subtitle document yields a `Caps::Text{format}`
//! (so `filesrc ! subparse` types without an explicit source). Pure `no_std`, no
//! allocation.

use g2g_core::{ByteStreamEncoding, Caps, TextFormat};

/// MPEG-TS packet stride; the sync byte recurs at this interval.
const TS_PACKET_LEN: usize = 188;
const TS_SYNC: u8 = 0x47;
/// EBML magic (Matroska / WebM): the leading bytes of the EBML header element.
const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];
/// Ogg page capture pattern.
const OGG_MAGIC: [u8; 4] = *b"OggS";
/// FLV signature: the first three bytes of an FLV header.
const FLV_MAGIC: [u8; 3] = *b"FLV";

/// Guess a media type from a stream's leading bytes, or `None` if nothing matches
/// (a `typefind` failure). Tries container magic first (binary signatures), then a
/// subtitle-document text sniff. Pass at least a few hundred bytes so MPEG-TS can
/// be confirmed across packet boundaries (a lone `0x47` is too weak to trust).
pub fn sniff_caps(header: &[u8]) -> Option<Caps> {
    if let Some(encoding) = sniff(header) {
        return Some(Caps::ByteStream { encoding });
    }
    sniff_text(header).map(|format| Caps::Text { format })
}

/// Guess the container encoding from a stream's leading bytes, or `None` if no
/// container signature matches. Pass at least a few hundred bytes so MPEG-TS can
/// be confirmed across packet boundaries (a lone `0x47` is too weak to trust).
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
    // ISO-BMFF (MP4 / QuickTime): both progressive (`moov`-based) and fragmented
    // (CMAF) map to the one `IsoBmff` encoding; the demuxer handles either.
    if looks_like_iso_bmff(header) {
        return Some(ByteStreamEncoding::IsoBmff);
    }
    if looks_like_mpegts(header) {
        return Some(ByteStreamEncoding::MpegTs);
    }
    None
}

/// True when the header is the start of an ISO Base Media File (MP4 / QuickTime):
/// a leading box whose 4-byte type at offset 4 is a known top-level box. `ftyp` is
/// the near-universal first box; `moov` / `mdat` / `styp` / `free` / `skip` / `wide`
/// cover header-less QuickTime and fragment / mdat-first layouts.
fn looks_like_iso_bmff(header: &[u8]) -> bool {
    if header.len() < 8 {
        return false;
    }
    matches!(
        &header[4..8],
        b"ftyp" | b"styp" | b"moov" | b"moof" | b"mdat" | b"free" | b"skip" | b"wide"
    )
}

/// Sniff a subtitle document from its text header, or `None` if it is not one we
/// parse. Content-based (not extension), the `subparse` typefind analog: WebVTT by
/// its mandatory `WEBVTT` signature, SSA/ASS by its `[Script Info]` / `[V4...]`
/// section, TTML by its `<tt>` root, and SubRip by a `-->` cue arrow with a comma
/// decimal (WebVTT uses a dot, and is already caught by its signature).
fn sniff_text(header: &[u8]) -> Option<TextFormat> {
    // A subtitle document is UTF-8 text; a lossy view is enough to match signatures
    // and never allocates beyond the borrowed slice for valid input.
    let text = core::str::from_utf8(header).ok()?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let head = text.trim_start();
    if head.starts_with("WEBVTT") {
        return Some(TextFormat::WebVtt);
    }
    if head.starts_with("[Script Info]") || head.starts_with("[V4+ Styles]") || head.starts_with("[V4 Styles]") {
        return Some(TextFormat::Ssa);
    }
    // TTML: an XML doc whose root (possibly namespaced) is `<tt`.
    if head.starts_with("<tt") || (head.starts_with("<?xml") && text.contains("<tt")) {
        return Some(TextFormat::Ttml);
    }
    // SubRip: a `-->` cue arrow whose preceding timestamp uses SRT's comma-
    // millisecond decimal (`00:00:20,000 --> ...`). WebVTT's arrow uses a dot
    // decimal and is caught by its signature above, so a `:` + `,` in the short
    // window before the arrow disambiguates SRT without matching prose commas.
    if let Some(pos) = text.find("-->") {
        // Walk back to a char boundary so a multibyte char before the arrow can't
        // panic the slice (timestamps are ASCII, so this is the identity in practice).
        let start = (pos.saturating_sub(14)..=pos).find(|&i| text.is_char_boundary(i)).unwrap_or(pos);
        let window = &text[start..pos];
        if window.contains(':') && window.contains(',') {
            return Some(TextFormat::Srt);
        }
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
    fn detects_iso_bmff_by_ftyp_box() {
        // `[size=0x18]ftypisom...` : the near-universal MP4 first box.
        let data = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2";
        assert_eq!(sniff(data), Some(ByteStreamEncoding::IsoBmff));
    }

    #[test]
    fn detects_iso_bmff_mdat_first_and_moov() {
        // Progressive `mdat`-first (moov at end) and moov-first both sniff as MP4.
        assert_eq!(sniff(b"\x00\x00\x00\x10mdat\0\0\0\0\0\0\0\0"), Some(ByteStreamEncoding::IsoBmff));
        assert_eq!(sniff(b"\x00\x00\x01\x00moov\0\0\0\0"), Some(ByteStreamEncoding::IsoBmff));
    }

    #[test]
    fn sniff_caps_maps_container_and_text() {
        assert_eq!(
            sniff_caps(b"\x00\x00\x00\x18ftypmp42"),
            Some(Caps::ByteStream { encoding: ByteStreamEncoding::IsoBmff })
        );
        assert_eq!(
            sniff_caps(b"WEBVTT\n\n00:00.000 --> 00:02.000\nhi"),
            Some(Caps::Text { format: TextFormat::WebVtt })
        );
    }

    #[test]
    fn detects_subtitle_documents_by_content() {
        assert_eq!(sniff_text(b"WEBVTT\n"), Some(TextFormat::WebVtt));
        assert_eq!(sniff_text(b"\xEF\xBB\xBFWEBVTT FILE\n"), Some(TextFormat::WebVtt));
        assert_eq!(sniff_text(b"1\n00:00:20,000 --> 00:00:24,400\nHello\n"), Some(TextFormat::Srt));
        assert_eq!(sniff_text(b"[Script Info]\nTitle: x\n"), Some(TextFormat::Ssa));
        assert_eq!(sniff_text(b"<?xml version=\"1.0\"?>\n<tt xmlns=\"...\">"), Some(TextFormat::Ttml));
        // Prose with a comma but no timestamp, and a dot-decimal (WebVTT-style)
        // arrow, must not be misread as SubRip.
        assert_eq!(sniff_text(b"Hello, world. No cues here."), None);
        assert_eq!(sniff_text(b"foo\n00:00.000 --> 00:02.000\nbar"), None);
    }

    #[test]
    fn ebml_takes_precedence_over_a_leading_0x47() {
        // EBML magic never starts with 0x47, so no ambiguity; sanity check that
        // a Matroska stream is not misread as TS.
        let data: Vec<u8> = EBML_MAGIC.iter().chain([0x47u8; 200].iter()).copied().collect();
        assert_eq!(sniff(&data), Some(ByteStreamEncoding::Matroska));
    }
}
