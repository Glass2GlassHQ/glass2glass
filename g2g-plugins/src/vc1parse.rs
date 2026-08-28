//! VC-1 (SMPTE 421M) advanced-profile elementary-stream parser.
//!
//! Frames the stream into one access unit per frame (`00 00 01 0D`) together
//! with the sequence header (`00 00 01 0F`) and entry-point header
//! (`00 00 01 0E`) that lead it, and refines caps from the sequence header,
//! which carries `MAX_CODED_WIDTH` / `MAX_CODED_HEIGHT` and, in its display
//! extension, the frame rate and the sample aspect ratio.
//!
//! Advanced profile only. Simple and main profile carry no start codes at all:
//! their sequence layer is a fixed codec-configuration block the container
//! supplies out of band, so there is nothing in the byte stream for a parser to
//! frame or to read geometry from. Feeding this element a simple / main profile
//! stream leaves the caps unrefined.
//!
//! Advanced profile escapes its byte stream (SMPTE 421M Annex E): a `0x03`
//! sits after any `00 00` that would otherwise emulate a start code, so the
//! sequence header is de-escaped before its bits are read.
//!
//! An access unit counts as a keyframe when it carries a sequence header or an
//! entry-point header. Advanced profile places those at random access points,
//! ahead of an I frame, and nowhere else. Reading `PTYPE` out of the frame
//! header instead would need the sequence header's `INTERLACE` flag, which is
//! not in the access unit being classified.

use g2g_core::{PropertySpec, VideoCodec};

use crate::annexb::{strip_emulation_prevention, BitReader};
use crate::startcodeparse::{
    reduce_ratio, sample_aspect, start_code_units, StartCodeCodec, StartCodeParse, StartCodeRole,
    VideoGeometry, PIXEL_ASPECT_PROPERTY,
};

/// Sequence header BDU type.
const SEQUENCE_HEADER_CODE: u8 = 0x0F;
/// Entry-point header BDU type.
const ENTRY_POINT_CODE: u8 = 0x0E;
/// Frame BDU type: the coded picture.
const FRAME_CODE: u8 = 0x0D;

/// `PROFILE` value for advanced profile, the only one that uses start codes.
const PROFILE_ADVANCED: u32 = 3;
/// `MAX_CODED_WIDTH` and `MAX_CODED_HEIGHT` are coded in units of two samples,
/// minus one.
const CODED_SIZE_UNIT: u32 = 2;
/// `ASPECT_RATIO` value meaning the pair is coded explicitly.
const ASPECT_RATIO_CUSTOM: u32 = 15;
/// Highest `ASPECT_RATIO` code SMPTE 421M Table 43 defines.
const HIGHEST_DEFINED_ASPECT_CODE: u32 = 13;

/// `FRAMERATENR` (SMPTE 421M Table 45) in frames per second. Index 0 is
/// forbidden and 8..=255 reserved, both left at 0.
const FRAME_RATE_NUMERATOR_BY_CODE: [u32; 8] = [0, 24, 25, 30, 50, 60, 48, 72];
/// `FRAMERATEDR` (SMPTE 421M Table 46), the divisor the numerator above is
/// scaled by after multiplying it by 1000. Index 0 and 3..=15 are reserved.
const FRAME_RATE_DENOMINATOR_BY_CODE: [u32; 16] =
    [0, 1000, 1001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
/// The scale `FRAME_RATE_DENOMINATOR_BY_CODE` is expressed against.
const FRAME_RATE_SCALE: u32 = 1000;
/// `FRAMERATEEXP` codes the rate as `(FRAMERATEEXP + 1) / 32` frames per second.
const FRAME_RATE_EXPLICIT_DIVISOR: u32 = 32;

/// VC-1 advanced-profile parser.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::vc1parse::Vc1Parse;
///
/// let parse = Vc1Parse::new();
/// ```
pub type Vc1Parse = StartCodeParse<Vc1Codec>;

/// VC-1 hooks for [`StartCodeParse`].
#[derive(Debug)]
pub struct Vc1Codec;

impl StartCodeCodec for Vc1Codec {
    const CODEC: VideoCodec = VideoCodec::Vc1;
    const NAME: &'static str = "VC-1 parser";
    const DESCRIPTION: &'static str =
        "Frames a VC-1 advanced-profile stream into access units and reads its sequence header";
    const PROPERTIES: &'static [PropertySpec] = &[PIXEL_ASPECT_PROPERTY];

    fn start_code_role(code: u8) -> StartCodeRole {
        match code {
            FRAME_CODE => StartCodeRole::Picture,
            SEQUENCE_HEADER_CODE | ENTRY_POINT_CODE => StartCodeRole::Leads,
            // Fields, slices and the user-data BDUs belong to the frame in
            // progress.
            _ => StartCodeRole::Continues,
        }
    }

    fn geometry(au: &[u8]) -> Option<VideoGeometry> {
        start_code_units(au)
            .filter(|(code, _)| *code == SEQUENCE_HEADER_CODE)
            .find_map(|(_, payload)| parse_sequence_header(&strip_emulation_prevention(payload)))
    }

    fn au_is_keyframe(au: &[u8]) -> bool {
        start_code_units(au)
            .any(|(code, _)| code == SEQUENCE_HEADER_CODE || code == ENTRY_POINT_CODE)
    }
}

/// Parse a de-escaped advanced-profile sequence header for the coded geometry.
/// `None` for another profile, a truncated header, or a zero dimension.
fn parse_sequence_header(header: &[u8]) -> Option<VideoGeometry> {
    let mut bits = BitReader::new(header);
    if bits.read_bits(2)? != PROFILE_ADVANCED {
        return None;
    }
    bits.skip_bits(3)?; // LEVEL
    bits.skip_bits(2)?; // COLORDIFF_FORMAT
    bits.skip_bits(3)?; // FRMRTQ_POSTPROC
    bits.skip_bits(5)?; // BITRTQ_POSTPROC
    bits.skip_bits(1)?; // POSTPROCFLAG
    let max_coded_width = bits.read_bits(12)?;
    let max_coded_height = bits.read_bits(12)?;
    bits.skip_bits(1)?; // PULLDOWN
    bits.skip_bits(1)?; // INTERLACE
    bits.skip_bits(1)?; // TFCNTRFLAG
    bits.skip_bits(1)?; // FINTERPFLAG
    bits.skip_bits(1)?; // RESERVED
    bits.skip_bits(1)?; // PSF

    let width = max_coded_width
        .saturating_add(1)
        .saturating_mul(CODED_SIZE_UNIT);
    let height = max_coded_height
        .saturating_add(1)
        .saturating_mul(CODED_SIZE_UNIT);
    if width == 0 || height == 0 {
        return None;
    }

    let mut framerate = None;
    let mut pixel_aspect = None;
    // DISPLAY_EXT. Read without `?` so a header truncated here still yields the
    // coded size.
    if bits.read_bit() == Some(1) {
        let mut display_extension = || -> Option<()> {
            bits.skip_bits(14)?; // DISP_HORIZ_SIZE
            bits.skip_bits(14)?; // DISP_VERT_SIZE
            if bits.read_bit()? == 1 {
                // ASPECT_RATIO_FLAG
                let aspect_ratio = bits.read_bits(4)?;
                pixel_aspect = if aspect_ratio == ASPECT_RATIO_CUSTOM {
                    let horizontal = bits.read_bits(8)?;
                    let vertical = bits.read_bits(8)?;
                    reduce_ratio(horizontal, vertical)
                } else {
                    sample_aspect(aspect_ratio, HIGHEST_DEFINED_ASPECT_CODE)
                };
            }
            if bits.read_bit()? == 1 {
                // FRAMERATE_FLAG
                framerate = if bits.read_bit()? == 0 {
                    // FRAMERATEIND 0: the table pair.
                    let numerator_code = bits.read_bits(8)?;
                    let denominator_code = bits.read_bits(4)?;
                    table_frame_rate_q16(numerator_code, denominator_code)
                } else {
                    explicit_frame_rate_q16(bits.read_bits(16)?)
                };
            }
            Some(())
        };
        let _ = display_extension();
    }

    Some(VideoGeometry {
        width,
        height,
        framerate,
        pixel_aspect,
    })
}

/// Q16 frames per second for a `(FRAMERATENR, FRAMERATEDR)` code pair.
fn table_frame_rate_q16(numerator_code: u32, denominator_code: u32) -> Option<u32> {
    let numerator = *FRAME_RATE_NUMERATOR_BY_CODE.get(numerator_code as usize)?;
    let denominator = *FRAME_RATE_DENOMINATOR_BY_CODE.get(denominator_code as usize)?;
    if numerator == 0 || denominator == 0 {
        return None;
    }
    let q16 = (((numerator as u64) * (FRAME_RATE_SCALE as u64)) << 16) / denominator as u64;
    u32::try_from(q16).ok()
}

/// Q16 frames per second for an explicit `FRAMERATEEXP`.
fn explicit_frame_rate_q16(framerate_exponent: u32) -> Option<u32> {
    let q16 = ((framerate_exponent as u64 + 1) << 16) / FRAME_RATE_EXPLICIT_DIVISOR as u64;
    (q16 != 0).then_some(q16 as u32)
}

/// Fuzzing entry: parse a VC-1 advanced-profile sequence header out of an
/// access unit. Exposed only under `--cfg fuzzing` (cargo-fuzz).
#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = Vc1Codec::geometry(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{
        AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
        PropValue, PushOutcome, Rate,
    };

    use crate::annexb::{add_emulation_prevention, BitWriter};

    /// `FRAMERATENR` 3: 30 fps before the divisor.
    const FRAME_RATE_CODE_30: u32 = 3;
    /// `FRAMERATEDR` 2: the 1001 divisor that makes 30 into 29.97.
    const FRAME_RATE_DIVISOR_1001: u32 = 2;
    /// `FRAMERATEDR` 1: the 1000 divisor that leaves the rate whole.
    const FRAME_RATE_DIVISOR_1000: u32 = 1;
    /// `ASPECT_RATIO` 1: square samples.
    const ASPECT_CODE_SQUARE: u32 = 1;
    /// `ASPECT_RATIO` 3: 10:11, the 525-line 4:3 sample aspect.
    const ASPECT_CODE_10_11: u32 = 3;

    /// What a fixture's display extension carries.
    #[derive(Default)]
    struct DisplayExtension {
        aspect_ratio: Option<u32>,
        frame_rate: Option<(u32, u32)>,
        frame_rate_exponent: Option<u32>,
    }

    /// An advanced-profile sequence header unit for `width` x `height`, start
    /// code included and Annex-E escaped as a real stream would be.
    fn sequence_header(width: u32, height: u32, display: Option<DisplayExtension>) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(PROFILE_ADVANCED, 2); // PROFILE
        w.write_bits(4, 3); // LEVEL
        w.write_bits(1, 2); // COLORDIFF_FORMAT (4:2:0)
        w.write_bits(0, 3); // FRMRTQ_POSTPROC
        w.write_bits(0, 5); // BITRTQ_POSTPROC
        w.write_bit(0); // POSTPROCFLAG
        w.write_bits(width / CODED_SIZE_UNIT - 1, 12); // MAX_CODED_WIDTH
        w.write_bits(height / CODED_SIZE_UNIT - 1, 12); // MAX_CODED_HEIGHT
        w.write_bit(0); // PULLDOWN
        w.write_bit(0); // INTERLACE
        w.write_bit(0); // TFCNTRFLAG
        w.write_bit(1); // FINTERPFLAG
        w.write_bit(0); // RESERVED
        w.write_bit(0); // PSF
        match display {
            None => w.write_bit(0), // DISPLAY_EXT
            Some(display) => {
                w.write_bit(1); // DISPLAY_EXT
                w.write_bits(width, 14); // DISP_HORIZ_SIZE
                w.write_bits(height, 14); // DISP_VERT_SIZE
                match display.aspect_ratio {
                    None => w.write_bit(0), // ASPECT_RATIO_FLAG
                    Some(code) => {
                        w.write_bit(1);
                        w.write_bits(code, 4);
                    }
                }
                match (display.frame_rate, display.frame_rate_exponent) {
                    (Some((numerator, denominator)), _) => {
                        w.write_bit(1); // FRAMERATE_FLAG
                        w.write_bit(0); // FRAMERATEIND
                        w.write_bits(numerator, 8);
                        w.write_bits(denominator, 4);
                    }
                    (None, Some(exponent)) => {
                        w.write_bit(1); // FRAMERATE_FLAG
                        w.write_bit(1); // FRAMERATEIND
                        w.write_bits(exponent, 16);
                    }
                    (None, None) => w.write_bit(0),
                }
                w.write_bit(0); // COLOR_FORMAT_FLAG
            }
        }
        w.write_bit(0); // HRD_PARAM_FLAG
        w.align_to_byte();
        let mut unit = vec![0u8, 0, 1, SEQUENCE_HEADER_CODE];
        unit.extend_from_slice(&add_emulation_prevention(&w.into_bytes()));
        unit
    }

    /// An entry-point header unit, start code included.
    fn entry_point() -> Vec<u8> {
        vec![0, 0, 1, ENTRY_POINT_CODE, 0x88, 0x40]
    }

    /// A frame unit, start code included.
    fn frame(tag: u8) -> Vec<u8> {
        vec![0, 0, 1, FRAME_CODE, 0x9C, tag, 0x11]
    }

    /// A slice unit, start code included.
    fn slice(tag: u8) -> Vec<u8> {
        vec![0, 0, 1, 0x0B, 0x22, tag]
    }

    #[test]
    fn reads_the_coded_size_without_a_display_extension() {
        let au = sequence_header(1920, 1080, None);
        let info = Vc1Codec::geometry(&au).expect("sequence header must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.framerate, None);
        assert_eq!(info.pixel_aspect, None);
    }

    #[test]
    fn reads_the_frame_rate_and_sample_aspect_from_the_display_extension() {
        let au = sequence_header(
            720,
            480,
            Some(DisplayExtension {
                aspect_ratio: Some(ASPECT_CODE_10_11),
                frame_rate: Some((FRAME_RATE_CODE_30, FRAME_RATE_DIVISOR_1001)),
                frame_rate_exponent: None,
            }),
        );
        let info = Vc1Codec::geometry(&au).expect("sequence header must parse");
        assert_eq!((info.width, info.height), (720, 480));
        let expected = u32::try_from(((30u64 * 1000) << 16) / 1001).unwrap();
        assert_eq!(info.framerate, Some(expected));
        assert_eq!(info.framerate.unwrap() >> 16, 29, "~29.97 fps");
        assert_eq!(info.pixel_aspect, Some((10, 11)));
    }

    #[test]
    fn reads_a_whole_frame_rate_and_square_samples() {
        let au = sequence_header(
            1280,
            720,
            Some(DisplayExtension {
                aspect_ratio: Some(ASPECT_CODE_SQUARE),
                frame_rate: Some((FRAME_RATE_CODE_30, FRAME_RATE_DIVISOR_1000)),
                frame_rate_exponent: None,
            }),
        );
        let info = Vc1Codec::geometry(&au).expect("sequence header must parse");
        assert_eq!(info.framerate, Some(30 << 16));
        assert_eq!(info.pixel_aspect, Some((1, 1)));
    }

    #[test]
    fn reads_an_explicit_frame_rate_exponent() {
        // FRAMERATEEXP 799 is (799 + 1) / 32 = 25 fps.
        const EXPONENT_FOR_25_FPS: u32 = 799;
        let au = sequence_header(
            720,
            576,
            Some(DisplayExtension {
                aspect_ratio: None,
                frame_rate: None,
                frame_rate_exponent: Some(EXPONENT_FOR_25_FPS),
            }),
        );
        let info = Vc1Codec::geometry(&au).expect("sequence header must parse");
        assert_eq!(info.framerate, Some(25 << 16));
    }

    #[test]
    fn rejects_a_non_advanced_profile_and_a_truncated_header() {
        // PROFILE 0 is simple profile, which never carries a start code.
        let simple = vec![0u8, 0, 1, SEQUENCE_HEADER_CODE, 0x00, 0x00, 0x00, 0x00];
        assert!(Vc1Codec::geometry(&simple).is_none());

        // Advanced profile cut off inside MAX_CODED_WIDTH.
        let truncated = vec![0u8, 0, 1, SEQUENCE_HEADER_CODE, 0xC8, 0x40];
        assert!(Vc1Codec::geometry(&truncated).is_none());
        assert!(Vc1Codec::geometry(&[]).is_none());
    }

    #[test]
    fn a_reserved_frame_rate_code_leaves_the_rate_unknown() {
        const RESERVED_FRAME_RATE_CODE: u32 = 200;
        let au = sequence_header(
            720,
            576,
            Some(DisplayExtension {
                aspect_ratio: None,
                frame_rate: Some((RESERVED_FRAME_RATE_CODE, FRAME_RATE_DIVISOR_1000)),
                frame_rate_exponent: None,
            }),
        );
        let info = Vc1Codec::geometry(&au).expect("the coded size still parses");
        assert_eq!(info.framerate, None);
    }

    #[test]
    fn de_escapes_the_sequence_header_before_reading_it() {
        // MAX_CODED_WIDTH 0 leaves two zero bytes ahead of MAX_CODED_HEIGHT, so
        // the encoder escapes the header with a 0x03. Reading those bytes raw
        // shifts every field past the escape.
        const ESCAPING_WIDTH: u32 = 2;
        let au = sequence_header(ESCAPING_WIDTH, 1080, None);
        let payload = &au[4..];
        assert!(
            payload.windows(3).any(|w| w == [0x00, 0x00, 0x03]),
            "the fixture must actually carry an escape byte"
        );
        assert_ne!(
            parse_sequence_header(payload).map(|g| g.height),
            Some(1080),
            "the escaped bytes read raw mis-place MAX_CODED_HEIGHT"
        );
        let info = Vc1Codec::geometry(&au).expect("sequence header must parse");
        assert_eq!((info.width, info.height), (ESCAPING_WIDTH, 1080));
    }

    #[test]
    fn access_units_split_at_each_frame() {
        let mut stream = sequence_header(1920, 1080, None);
        stream.extend_from_slice(&entry_point());
        stream.extend_from_slice(&frame(0xAA));
        stream.extend_from_slice(&slice(0xCC));
        let second = stream.len();
        stream.extend_from_slice(&frame(0xBB));
        let starts = crate::startcodeparse::au_starts_by(&stream, Vc1Codec::start_code_role);
        assert_eq!(starts, vec![0, second]);
    }

    #[test]
    fn keyframe_flag_follows_the_random_access_headers() {
        let mut at_entry = entry_point();
        at_entry.extend_from_slice(&frame(0xAA));
        assert!(Vc1Codec::au_is_keyframe(&at_entry));
        assert!(!Vc1Codec::au_is_keyframe(&frame(0xBB)));
    }

    // -- Element-level tests (drive Vc1Parse::process directly) --------------

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            self.packets.push(packet);
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn frame_with_bytes(seq: u64, bytes: Vec<u8>) -> Frame {
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: seq,
            meta: Default::default(),
        }
    }

    fn vc1_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Vc1,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    fn data_payloads(sink: &RecordingSink) -> Vec<Vec<u8>> {
        sink.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn emits_caps_changed_then_one_frame_per_picture() {
        let mut parse = Vc1Parse::new();
        parse.configure_pipeline(&vc1_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let mut first = sequence_header(
            1280,
            720,
            Some(DisplayExtension {
                aspect_ratio: Some(ASPECT_CODE_SQUARE),
                frame_rate: Some((FRAME_RATE_CODE_30, FRAME_RATE_DIVISOR_1000)),
                frame_rate_exponent: None,
            }),
        );
        first.extend_from_slice(&entry_point());
        first.extend_from_slice(&frame(0xAA));
        let second = frame(0xBB);
        let mut buffer = first.clone();
        buffer.extend_from_slice(&second);

        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, buffer)),
                &mut sink,
            )
            .await
            .unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                width,
                height,
                framerate,
                ..
            }) => {
                assert_eq!(*width, Dim::Fixed(1280));
                assert_eq!(*height, Dim::Fixed(720));
                assert_eq!(*framerate, Rate::Fixed(30 << 16));
            }
            other => panic!("expected CapsChanged first, got {other:?}"),
        }
        assert_eq!(data_payloads(&sink), vec![first, second]);
        assert_eq!(parse.caps_changes_emitted(), 1);
        assert_eq!(
            parse.get_property("pixel-aspect-ratio"),
            Some(PropValue::Fraction(1, 1))
        );
    }

    #[tokio::test]
    async fn rejects_non_vc1_caps_in_intercept() {
        let parse = Vc1Parse::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(parse.intercept_caps(&h264), Err(G2gError::CapsMismatch));
    }
}
