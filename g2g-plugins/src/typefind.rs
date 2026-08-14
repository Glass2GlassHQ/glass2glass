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
//! `.264` / `.jsv` recording by content), as does a still image (PNG / WebP, one
//! frame per file); a subtitle document yields a
//! `Caps::Text{format}` (so `filesrc ! subparse` types without an explicit
//! source). The sniff functions themselves are pure `no_std`, no allocation.
//!
//! [`TypeFind`] is the same sniff as a mid-graph element, for a byte stream that
//! did not come from a file: it holds back the leading frames, sniffs them, and
//! re-declares its output caps with a `CapsChanged` before letting the data
//! through, so a source that could only guess its type is corrected downstream.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::log::{short_type_name, LogName, LogSource};
use g2g_core::{
    g2g_error, g2g_info, AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, ConfigureOutcome,
    Dim, ElementMetadata, G2gError, OutputSink, PipelinePacket, Rate, TextFormat, VideoCodec,
};

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
/// MPEG program stream: a pack header start code. Every PS (MPEG-1 `.mpg`,
/// MPEG-2 `.vob`) opens on one, and packs recur throughout.
const PS_PACK_MAGIC: [u8; 4] = [0x00, 0x00, 0x01, 0xBA];
/// RIFF/WAVE: the `RIFF` container magic, with `WAVE` after the 4-byte size.
pub(crate) const RIFF_MAGIC: [u8; 4] = *b"RIFF";
const WAVE_MAGIC: [u8; 4] = *b"WAVE";
/// WebP rides the same RIFF header as WAVE, tagged `WEBP` after the size.
pub(crate) const WEBP_MAGIC: [u8; 4] = *b"WEBP";
/// PNG signature (ISO/IEC 15948 5.2): the 8 bytes every PNG opens with.
const PNG_MAGIC: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
/// Every RIFF tag is a four-character code, and the 4-byte size sits between the
/// magic and the form type, so a complete RIFF header is three of them.
const FOURCC_LEN: usize = 4;
pub(crate) const RIFF_FORM_OFFSET: usize = FOURCC_LEN * 2;
pub(crate) const RIFF_HEADER_LEN: usize = RIFF_FORM_OFFSET + FOURCC_LEN;

/// Header bytes [`sniff_caps`] needs to decide: enough to confirm an MPEG-TS
/// sync byte across several packets, the longest signature here.
pub const SNIFF_LEN: usize = 4 * TS_PACKET_LEN;

/// Guess a media type from a stream's leading bytes, or `None` if nothing matches
/// (a `typefind` failure). Tries container magic first (binary signatures), then a
/// raw Annex-B video elementary stream, then a subtitle-document text sniff. Pass
/// at least a few hundred bytes so MPEG-TS can be confirmed across packet
/// boundaries (a lone `0x47` is too weak to trust).
pub fn sniff_caps(header: &[u8]) -> Option<Caps> {
    if let Some(encoding) = sniff(header) {
        return Some(Caps::ByteStream { encoding });
    }
    // Native FLAC: the `fLaC` stream marker (M774); `flacparse` frames it.
    if header.starts_with(b"fLaC") {
        return Some(elementary_flac_caps());
    }
    if let Some(codec) = sniff_still_image(header) {
        return Some(still_image_caps(codec));
    }
    if let Some(codec) = sniff_annexb_video(header) {
        return Some(elementary_video_caps(codec));
    }
    sniff_text(header).map(|format| Caps::Text { format })
}

/// The form type of a RIFF file (`WAVE`, `WEBP`, ...), or `None` when the header
/// is not RIFF or is too short to carry one.
pub(crate) fn riff_form(header: &[u8]) -> Option<[u8; 4]> {
    if !header.starts_with(&RIFF_MAGIC) {
        return None;
    }
    header
        .get(RIFF_FORM_OFFSET..RIFF_HEADER_LEN)?
        .try_into()
        .ok()
}

/// Guess a still-image codec from a file's magic bytes, or `None` if it is not
/// one we decode. A still image is a one-frame `CompressedVideo` stream here, so
/// `filesrc location=x.png ! decodebin` plugs the matching image decoder.
///
/// JPEG is deliberately absent: `mjpegdec` takes one whole access unit per
/// buffer, and nothing here reassembles a JPEG that a byte source split across
/// reads, so typing a `.jpg` by content would plug a decoder that fails on any
/// file past the source's chunk size. It needs a `jpegparse` first.
fn sniff_still_image(header: &[u8]) -> Option<VideoCodec> {
    if header.starts_with(&PNG_MAGIC) {
        return Some(VideoCodec::Png);
    }
    if riff_form(header) == Some(WEBP_MAGIC) {
        return Some(VideoCodec::WebP);
    }
    None
}

/// Caps for a native FLAC byte stream at the channels/rate placeholders
/// (`flacparse` refines them from STREAMINFO). Shared by content sniffing and
/// `FileSrc`'s extension typing so the two never drift.
pub fn elementary_flac_caps() -> Caps {
    Caps::Audio {
        format: g2g_core::AudioFormat::Flac,
        channels: 0,
        sample_rate: 0,
    }
}

/// Caps for a raw Annex-B video elementary stream at a fixable `Range` placeholder
/// geometry: never `Dim::Any` (which cannot fixate), the parser refines it from
/// the SPS (M676). Shared by content sniffing and `FileSrc`'s extension typing so
/// the two never drift.
pub fn elementary_video_caps(codec: VideoCodec) -> Caps {
    /// A coded video stream is at least one macroblock.
    const MIN_DIM: u32 = 16;
    placeholder_video_caps(codec, MIN_DIM)
}

/// Caps for a still image (PNG / WebP) at a fixable `Range` placeholder geometry,
/// the same shape as [`elementary_video_caps`] but down to a single pixel.
pub fn still_image_caps(codec: VideoCodec) -> Caps {
    /// An icon is a legitimate still, where a coded video stream has a floor.
    const MIN_DIM: u32 = 1;
    placeholder_video_caps(codec, MIN_DIM)
}

fn placeholder_video_caps(codec: VideoCodec, min_dim: u32) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Range {
            min: min_dim,
            max: 65535,
        },
        height: Dim::Range {
            min: min_dim,
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
    if header.starts_with(&PS_PACK_MAGIC) {
        return Some(ByteStreamEncoding::MpegPs);
    }
    if riff_form(header) == Some(WAVE_MAGIC) {
        return Some(ByteStreamEncoding::Wav);
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

/// Leading bytes held for the sniff before it is declared a failure. Every
/// signature [`sniff_caps`] matches decides well inside this, and the stream is
/// attacker-controlled, so the buffer is capped rather than growing with it.
const HEADER_BUDGET: usize = 8192;

/// Frames held back while sniffing. Every CPU-readable byte counts against the
/// budget above, so this only has to bound a stream of frames that contribute no
/// bytes at all (empty, or a device-domain handle nothing can sniff).
const MAX_HELD_FRAMES: usize = 1024;

/// Mid-graph content sniffing (`typefind`): re-declares the caps of a byte stream
/// from the bytes themselves.
///
/// A byte source that cannot know what it carries (a socket, an application push,
/// a mis-named file) still has to declare something to negotiate. This element
/// holds back the leading frames, runs [`sniff_caps`] over them, and emits a
/// `CapsChanged` with the sniffed type before releasing the held data, so the
/// downstream demuxer / parser sees the real type. The frames themselves are
/// forwarded unchanged. Caps equal to what is already declared are not re-emitted.
///
/// A stream that has not sniffed by the end of the header budget, or that ends
/// before it does, fails with [`G2gError::CapsMismatch`]: untyped bytes are never
/// forwarded as if they had been typed.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::typefind::TypeFind;
///
/// let element = TypeFind::new();
/// assert_eq!(element.sniffed_caps(), None);
/// ```
#[derive(Debug, Default)]
pub struct TypeFind {
    header: Vec<u8>,
    held: Vec<Frame>,
    /// Caps currently declared on the output link: the negotiated ones until a
    /// sniff replaces them. Emission is suppressed while the sniff agrees.
    declared: Option<Caps>,
    sniffed: Option<Caps>,
    configured: bool,
    log_name: LogName,
}

impl TypeFind {
    pub fn new() -> Self {
        Self::default()
    }

    /// The caps the content sniffed to, or `None` before enough bytes flowed.
    pub fn sniffed_caps(&self) -> Option<&Caps> {
        self.sniffed.as_ref()
    }

    /// Append a frame's system-memory bytes to the sniff buffer, up to the budget.
    /// A device-domain frame contributes nothing (its bytes are not CPU-readable),
    /// which the held-frame bound then turns into a sniff failure.
    fn accumulate(&mut self, frame: &Frame) {
        let Some(bytes) = frame.domain.as_system_slice() else {
            return;
        };
        let room = HEADER_BUDGET.saturating_sub(self.header.len());
        self.header
            .extend_from_slice(&bytes[..bytes.len().min(room)]);
    }

    fn out_of_budget(&self) -> bool {
        self.header.len() >= HEADER_BUDGET || self.held.len() >= MAX_HELD_FRAMES
    }
}

impl AsyncElement for TypeFind {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "TypeFind",
            "Generic",
            "Sniffs the media type of a byte stream and re-declares its caps",
            "g2g",
        )
    }

    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Pass-through at negotiation: the type is only known once bytes flow, so the
    /// element starts on whatever the source declared and refines it at runtime.
    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.declared = Some(absolute_caps.clone());
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) if self.sniffed.is_some() => {
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::DataFrame(frame) => {
                    self.accumulate(&frame);
                    self.held.push(frame);
                    let Some(caps) = sniff_caps(&self.header) else {
                        if self.out_of_budget() {
                            g2g_error!(
                                self,
                                "no media type in the first {} bytes ({} frames): the stream is not one we sniff",
                                self.header.len(),
                                self.held.len()
                            );
                            return Err(G2gError::CapsMismatch);
                        }
                        return Ok(());
                    };
                    g2g_info!(self, "sniffed {:?} from {} bytes", caps, self.header.len());
                    self.sniffed = Some(caps.clone());
                    self.header = Vec::new();
                    if self.declared.as_ref() != Some(&caps) {
                        out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                        self.declared = Some(caps);
                    }
                    for held in core::mem::take(&mut self.held) {
                        out.push(PipelinePacket::DataFrame(held)).await?;
                    }
                }
                PipelinePacket::CapsChanged(caps) => {
                    self.declared = Some(caps.clone());
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                // A flushing seek restarts the byte stream, so the bytes held for
                // an unfinished sniff are stale; the type already found still holds.
                PipelinePacket::Flush => {
                    self.header = Vec::new();
                    self.held = Vec::new();
                    out.push(PipelinePacket::Flush).await?;
                }
                // The runner forwards the EOS sentinel itself; a stream that ended
                // mid-sniff never got a type, and its held frames must not vanish.
                PipelinePacket::Eos => {
                    if !self.held.is_empty() {
                        g2g_error!(
                            self,
                            "stream ended after {} bytes with no media type sniffed",
                            self.header.len()
                        );
                        return Err(G2gError::CapsMismatch);
                    }
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn set_instance_name(&mut self, name: String) {
        self.log_name.set_instance(name);
    }

    fn set_log_category(&mut self, category: String) {
        self.log_name.set_category(category);
    }
}

impl LogSource for TypeFind {
    fn log_category(&self) -> &'static str {
        short_type_name::<Self>()
    }
    fn log_instance(&self) -> Option<&str> {
        self.log_name.instance()
    }
    fn log_category_override(&self) -> Option<&str> {
        self.log_name.category()
    }
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
    fn detects_still_images_by_magic() {
        assert_eq!(
            sniff_still_image(&PNG_MAGIC),
            Some(VideoCodec::Png),
            "the 8-byte PNG signature types as PNG"
        );
        assert_eq!(
            sniff_still_image(b"RIFF\x24\0\0\0WEBPVP8L"),
            Some(VideoCodec::WebP)
        );
        // JPEG is not typed by content: see the note on `sniff_still_image`.
        assert_eq!(sniff_still_image(b"\xff\xd8\xff\xe0\0\x10JFIF"), None);
        // A RIFF file that is not WebP, a truncated RIFF header, and a PNG
        // signature with one byte wrong are all misses.
        assert_eq!(sniff_still_image(b"RIFF\0\0\0\0AVI "), None);
        assert_eq!(sniff_still_image(b"RIFF\0\0\0"), None);
        assert_eq!(
            sniff_still_image(&[0x89, 0x50, 0x4E, 0x46, 0, 0, 0, 0]),
            None
        );
        assert_eq!(sniff_still_image(&[]), None);
    }

    #[test]
    fn sniff_caps_types_a_still_image_as_one_frame_video() {
        // A PNG must reach the caps layer as CompressedVideo{Png} at a fixable
        // placeholder geometry, or `filesrc ! decodebin` cannot plug pngdec.
        let caps = sniff_caps(&PNG_MAGIC).expect("PNG types");
        assert_eq!(caps, still_image_caps(VideoCodec::Png));
        assert!(matches!(
            caps,
            Caps::CompressedVideo {
                codec: VideoCodec::Png,
                width: Dim::Range { min: 1, .. },
                ..
            }
        ));
        assert_eq!(
            sniff_caps(b"RIFF\x24\0\0\0WEBPVP8 "),
            Some(still_image_caps(VideoCodec::WebP))
        );
        // RIFF/WAVE still wins the shared RIFF header.
        assert_eq!(
            sniff_caps(b"RIFF\x24\0\0\0WAVEfmt "),
            Some(Caps::ByteStream {
                encoding: ByteStreamEncoding::Wav
            })
        );
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
