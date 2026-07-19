//! Content sniffing (M112, text + MP4 M478): guess a media type from the first
//! bytes of a stream, the `typefind` analog.
//!
//! The typed `Caps` model has no "untyped bytes" variant, so a byte source must
//! declare what it carries before a demuxer / parser can negotiate. This lets
//! `FileSrc` (and a future HTTP source) pick that automatically instead of the
//! caller naming it: read a header, match a magic signature, and emit the
//! matching `Caps` ([`sniff_caps`]). Container magic yields a
//! `Caps::ByteStream{encoding}`; a raw Annex-B H.264/H.265 elementary stream
//! yields a `Caps::CompressedVideo{..}` (so `filesrc ! decodebin` types a bare
//! `.264` / `.jsv` recording by content); a subtitle document yields a
//! `Caps::Text{format}` (so `filesrc ! subparse` types without an explicit
//! source). Pure `no_std`, no allocation.

use g2g_core::{ByteStreamEncoding, Caps, Dim, Rate, TextFormat, VideoCodec};

/// MPEG-TS packet stride; the sync byte recurs at this interval.
const TS_PACKET_LEN: usize = 188;
const TS_SYNC: u8 = 0x47;
/// EBML magic (Matroska / WebM): the leading bytes of the EBML header element.
const EBML_MAGIC: [u8; 4] = [0x1A, 0x45, 0xDF, 0xA3];
/// Ogg page capture pattern.
const OGG_MAGIC: [u8; 4] = *b"OggS";
/// FLV signature: the first three bytes of an FLV header.
const FLV_MAGIC: [u8; 3] = *b"FLV";
/// IVF file signature (the first 4 bytes of the 32-byte `DKIF` header).
const IVF_MAGIC: [u8; 4] = *b"DKIF";

/// Guess a media type from a stream's leading bytes, or `None` if nothing matches
/// (a `typefind` failure). Tries container magic first (binary signatures), then a
/// raw Annex-B video elementary stream, then a subtitle-document text sniff. Pass
/// at least a few hundred bytes so MPEG-TS can be confirmed across packet
/// boundaries (a lone `0x47` is too weak to trust).
pub fn sniff_caps(header: &[u8]) -> Option<Caps> {
    if let Some(encoding) = sniff(header) {
        return Some(Caps::ByteStream { encoding });
    }
    if let Some(codec) = sniff_annexb_video(header) {
        return Some(elementary_video_caps(codec));
    }
    sniff_text(header).map(|format| Caps::Text { format })
}

/// Caps for a raw Annex-B video elementary stream at a fixable `Range` placeholder
/// geometry: never `Dim::Any` (which cannot fixate), the parser refines it from
/// the SPS (M676). Shared by content sniffing and `FileSrc`'s extension typing so
/// the two never drift.
pub fn elementary_video_caps(codec: VideoCodec) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Range {
            min: 16,
            max: 65535,
        },
        height: Dim::Range {
            min: 16,
            max: 65535,
        },
        framerate: Rate::Range {
            min_q16: 1 << 16,
            max_q16: 240 << 16,
        },
    }
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
    if header.starts_with(&IVF_MAGIC) {
        return Some(ByteStreamEncoding::Ivf);
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
    if head.starts_with("[Script Info]")
        || head.starts_with("[V4+ Styles]")
        || head.starts_with("[V4 Styles]")
    {
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
        let start = (pos.saturating_sub(14)..=pos)
            .find(|&i| text.is_char_boundary(i))
            .unwrap_or(pos);
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

/// Guess H.264 vs H.265 from a raw Annex-B elementary stream, or `None` if the
/// header is not one. Scans for start-code-prefixed NAL units and keys on a
/// parameter-set NAL, whose type is decisive between the two codecs: HEVC VPS/
/// SPS/PPS (32/33/34) read as H.264 types 0/1/2 (never parameter sets), and H.264
/// SPS/PPS (7/8) read as HEVC types 51/4. Returns on the first parameter set,
/// which every real elementary stream carries before its slices. A malformed
/// stream that leads with a bare slice is undecodeable anyway and stays `None`.
fn sniff_annexb_video(header: &[u8]) -> Option<VideoCodec> {
    const MAX_NALS: usize = 8;
    let mut pos = 0;
    for _ in 0..MAX_NALS {
        let nal = find_start_code(header, pos)?;
        pos = nal + 1;
        let b0 = *header.get(nal)?;
        // forbidden_zero_bit must be 0 in both codecs; anything else is not a
        // NAL header (likely a start-code-like byte run in unrelated data).
        if b0 & 0x80 != 0 {
            continue;
        }
        // HEVC: 2-byte header, nal_unit_type = bits 1..6, temporal_id_plus1 (b1
        // low 3 bits) is mandatory and nonzero.
        let hevc_type = (b0 >> 1) & 0x3f;
        if matches!(hevc_type, 32..=34) {
            if let Some(&b1) = header.get(nal + 1) {
                if b1 & 0x07 != 0 {
                    return Some(VideoCodec::H265);
                }
            }
        }
        // H.264: 1-byte header, nal_unit_type = low 5 bits; a parameter set has a
        // nonzero nal_ref_idc (bits 5..6).
        let h264_type = b0 & 0x1f;
        if matches!(h264_type, 7 | 8) && b0 & 0x60 != 0 {
            return Some(VideoCodec::H264);
        }
    }
    None
}

/// Offset of the NAL header byte after the next Annex-B start code (`00 00 01`,
/// which also matches the 4-byte `00 00 00 01` form) at or after `from`, or `None`.
fn find_start_code(data: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            return Some(i + 3);
        }
        i += 1;
    }
    None
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
        assert_eq!(
            sniff(b"FLV\x01\x05\0\0\0\x09"),
            Some(ByteStreamEncoding::Flv)
        );
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
        assert_eq!(
            sniff(b"\x00\x00\x00\x10mdat\0\0\0\0\0\0\0\0"),
            Some(ByteStreamEncoding::IsoBmff)
        );
        assert_eq!(
            sniff(b"\x00\x00\x01\x00moov\0\0\0\0"),
            Some(ByteStreamEncoding::IsoBmff)
        );
    }

    #[test]
    fn sniff_caps_maps_container_and_text() {
        assert_eq!(
            sniff_caps(b"\x00\x00\x00\x18ftypmp42"),
            Some(Caps::ByteStream {
                encoding: ByteStreamEncoding::IsoBmff
            })
        );
        assert_eq!(
            sniff_caps(b"WEBVTT\n\n00:00.000 --> 00:02.000\nhi"),
            Some(Caps::Text {
                format: TextFormat::WebVtt
            })
        );
    }

    #[test]
    fn detects_subtitle_documents_by_content() {
        assert_eq!(sniff_text(b"WEBVTT\n"), Some(TextFormat::WebVtt));
        assert_eq!(
            sniff_text(b"\xEF\xBB\xBFWEBVTT FILE\n"),
            Some(TextFormat::WebVtt)
        );
        assert_eq!(
            sniff_text(b"1\n00:00:20,000 --> 00:00:24,400\nHello\n"),
            Some(TextFormat::Srt)
        );
        assert_eq!(
            sniff_text(b"[Script Info]\nTitle: x\n"),
            Some(TextFormat::Ssa)
        );
        assert_eq!(
            sniff_text(b"<?xml version=\"1.0\"?>\n<tt xmlns=\"...\">"),
            Some(TextFormat::Ttml)
        );
        // Prose with a comma but no timestamp, and a dot-decimal (WebVTT-style)
        // arrow, must not be misread as SubRip.
        assert_eq!(sniff_text(b"Hello, world. No cues here."), None);
        assert_eq!(sniff_text(b"foo\n00:00.000 --> 00:02.000\nbar"), None);
    }

    #[test]
    fn detects_h264_annexb_by_sps_nal() {
        // 4-byte start code, then an SPS NAL (0x67: nal_ref_idc=3, type=7).
        let data = [0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e];
        assert_eq!(sniff_annexb_video(&data), Some(VideoCodec::H264));
        assert!(matches!(
            sniff_caps(&data),
            Some(Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            })
        ));
    }

    #[test]
    fn detects_h264_annexb_after_aud_and_3byte_start_code() {
        // AUD (0x09) then SPS (0x68 -> nal_ref_idc=3, type=8 PPS), 3-byte codes.
        let data = [0x00, 0x00, 0x01, 0x09, 0x10, 0x00, 0x00, 0x01, 0x67, 0x42];
        assert_eq!(sniff_annexb_video(&data), Some(VideoCodec::H264));
    }

    #[test]
    fn detects_h265_annexb_by_vps_nal() {
        // 4-byte start code, then a VPS NAL (0x40 0x01: type=32, temporal_id_plus1=1).
        let data = [0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0c, 0x01];
        assert_eq!(sniff_annexb_video(&data), Some(VideoCodec::H265));
        assert!(matches!(
            sniff_caps(&data),
            Some(Caps::CompressedVideo {
                codec: VideoCodec::H265,
                ..
            })
        ));
    }

    #[test]
    fn annexb_sniff_rejects_non_video() {
        // No start code at all.
        assert_eq!(sniff_annexb_video(b"just some plain text bytes here"), None);
        // A start code but the NAL header has the forbidden bit set and no param set.
        assert_eq!(sniff_annexb_video(&[0x00, 0x00, 0x01, 0x80, 0x00]), None);
        // HEVC VPS type but temporal_id_plus1 = 0 (invalid): not accepted.
        assert_eq!(sniff_annexb_video(&[0x00, 0x00, 0x01, 0x40, 0x00]), None);
    }

    #[test]
    fn ebml_takes_precedence_over_a_leading_0x47() {
        // EBML magic never starts with 0x47, so no ambiguity; sanity check that
        // a Matroska stream is not misread as TS.
        let data: Vec<u8> = EBML_MAGIC
            .iter()
            .chain([0x47u8; 200].iter())
            .copied()
            .collect();
        assert_eq!(sniff(&data), Some(ByteStreamEncoding::Matroska));
    }
}
