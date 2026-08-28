//! MPEG-4 Part 2 (Visual, ISO/IEC 14496-2) elementary-stream parser.
//!
//! Frames the stream into one access unit per VOP (`00 00 01 B6`) together with
//! the Visual Object Sequence, Visual Object, Video Object Layer and Group of
//! VOP headers that lead it, and refines caps from the VOL header (14496-2
//! 6.2.3), which carries the coded width and height, the VOP time increment
//! resolution the framerate derives from, and the sample aspect ratio.
//!
//! `config-interval` re-inserts the cached configuration headers (everything
//! from the start of an access unit up to its first VOP, when that prefix
//! carries a VOL) before a later keyframe that lacks them, so a decoder joining
//! mid-stream can configure itself. Only rectangular VOLs
//! (`video_object_layer_shape == 0`) code a width and height; an arbitrary-shape
//! VOL leaves the caps geometry unrefined.

use alloc::vec::Vec;

use g2g_core::{PropertySpec, VideoCodec};

use crate::annexb::BitReader;
use crate::startcodeparse::{
    first_start_code_offset, sample_aspect, start_code_units, StartCodeCodec, StartCodeParse,
    StartCodeRole, VideoGeometry, CONFIG_INTERVAL_PROPERTY, PIXEL_ASPECT_PROPERTY,
};

/// `visual_object_sequence_start_code`.
const VISUAL_OBJECT_SEQUENCE_START_CODE: u8 = 0xB0;
/// `group_of_vop_start_code`.
const GROUP_OF_VOP_START_CODE: u8 = 0xB3;
/// `visual_object_start_code`.
const VISUAL_OBJECT_START_CODE: u8 = 0xB5;
/// `vop_start_code`: the coded picture.
const VOP_START_CODE: u8 = 0xB6;
/// `video_object_start_code` range.
const VIDEO_OBJECT_START_CODES: core::ops::RangeInclusive<u8> = 0x00..=0x1F;
/// `video_object_layer_start_code` range: the header carrying the geometry.
const VIDEO_OBJECT_LAYER_START_CODES: core::ops::RangeInclusive<u8> = 0x20..=0x2F;

/// `video_object_layer_shape` value for a plain rectangular layer, the only one
/// that codes a width and height in the VOL header.
const SHAPE_RECTANGULAR: u32 = 0;
/// `video_object_layer_shape` value for grayscale, whose shape extension field
/// only exists from version 2 of the layer syntax.
const SHAPE_GRAYSCALE: u32 = 3;
/// `video_object_layer_verid` when `is_object_layer_identifier` is absent.
const DEFAULT_VERID: u32 = 1;
/// `aspect_ratio_info` value meaning the pair is coded explicitly.
const ASPECT_RATIO_EXTENDED: u32 = 15;
/// Highest `aspect_ratio_info` code ISO/IEC 14496-2 Table 6-12 defines.
const HIGHEST_DEFINED_ASPECT_CODE: u32 = 5;
/// Width of the `vbv_parameters` block, in bits, when it is present.
const VBV_PARAMETERS_BITS: usize = 79;

/// MPEG-4 Part 2 video parser.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::mpeg4videoparse::Mpeg4VideoParse;
///
/// let parse = Mpeg4VideoParse::new().with_config_interval(-1);
/// ```
pub type Mpeg4VideoParse = StartCodeParse<Mpeg4VideoCodec>;

/// MPEG-4 Part 2 hooks for [`StartCodeParse`].
#[derive(Debug)]
pub struct Mpeg4VideoCodec;

impl StartCodeCodec for Mpeg4VideoCodec {
    const CODEC: VideoCodec = VideoCodec::Mpeg4Part2;
    const NAME: &'static str = "MPEG-4 Part 2 video parser";
    const DESCRIPTION: &'static str =
        "Frames an MPEG-4 Part 2 stream into access units and reads its VOL header";
    const PROPERTIES: &'static [PropertySpec] = &[CONFIG_INTERVAL_PROPERTY, PIXEL_ASPECT_PROPERTY];

    fn start_code_role(code: u8) -> StartCodeRole {
        match code {
            VOP_START_CODE => StartCodeRole::Picture,
            VISUAL_OBJECT_SEQUENCE_START_CODE
            | GROUP_OF_VOP_START_CODE
            | VISUAL_OBJECT_START_CODE => StartCodeRole::Leads,
            c if VIDEO_OBJECT_START_CODES.contains(&c) => StartCodeRole::Leads,
            c if VIDEO_OBJECT_LAYER_START_CODES.contains(&c) => StartCodeRole::Leads,
            _ => StartCodeRole::Continues,
        }
    }

    fn geometry(au: &[u8]) -> Option<VideoGeometry> {
        start_code_units(au)
            .filter(|(code, _)| VIDEO_OBJECT_LAYER_START_CODES.contains(code))
            .find_map(|(_, payload)| parse_video_object_layer(payload))
    }

    fn au_is_keyframe(au: &[u8]) -> bool {
        crate::annexb::au_is_keyframe(VideoCodec::Mpeg4Part2, au)
    }

    /// Everything from the start of `au` up to its first VOP, when that prefix
    /// carries a VOL header. A decoder needs the VOL to configure itself, so a
    /// prefix without one is no configuration.
    fn config_headers(au: &[u8]) -> Option<Vec<u8>> {
        let vop = first_start_code_offset(au, |code| code == VOP_START_CODE)?;
        if vop == 0 {
            return None;
        }
        let prefix = &au[..vop];
        let carries_layer = start_code_units(prefix)
            .any(|(code, _)| VIDEO_OBJECT_LAYER_START_CODES.contains(&code));
        carries_layer.then(|| prefix.to_vec())
    }
}

/// Parse a `video_object_layer` header payload (the bytes after its start code)
/// for the coded geometry. `None` when the layer is not rectangular, the header
/// is truncated, or it codes a zero dimension.
fn parse_video_object_layer(payload: &[u8]) -> Option<VideoGeometry> {
    let mut bits = BitReader::new(payload);
    bits.skip_bits(1)?; // random_accessible_vol
    bits.skip_bits(8)?; // video_object_type_indication

    let verid = if bits.read_bit()? == 1 {
        let verid = bits.read_bits(4)?;
        bits.skip_bits(3)?; // video_object_layer_priority
        verid
    } else {
        DEFAULT_VERID
    };

    let aspect_ratio_info = bits.read_bits(4)?;
    let pixel_aspect = if aspect_ratio_info == ASPECT_RATIO_EXTENDED {
        let par_width = bits.read_bits(8)?;
        let par_height = bits.read_bits(8)?;
        crate::startcodeparse::reduce_ratio(par_width, par_height)
    } else {
        sample_aspect(aspect_ratio_info, HIGHEST_DEFINED_ASPECT_CODE)
    };

    if bits.read_bit()? == 1 {
        // vol_control_parameters
        bits.skip_bits(2)?; // chroma_format
        bits.skip_bits(1)?; // low_delay
        if bits.read_bit()? == 1 {
            bits.skip_bits(VBV_PARAMETERS_BITS)?;
        }
    }

    let shape = bits.read_bits(2)?;
    if shape == SHAPE_GRAYSCALE && verid != DEFAULT_VERID {
        bits.skip_bits(4)?; // video_object_layer_shape_extension
    }
    bits.skip_bits(1)?; // marker_bit
    let vop_time_increment_resolution = bits.read_bits(16)?;
    bits.skip_bits(1)?; // marker_bit

    let framerate = if bits.read_bit()? == 1 {
        // fixed_vop_rate: the increment is coded in as many bits as the
        // resolution needs, so a zero resolution makes the header unreadable.
        if vop_time_increment_resolution == 0 {
            return None;
        }
        let increment = bits.read_bits(time_increment_bits(vop_time_increment_resolution))?;
        fixed_vop_rate_q16(vop_time_increment_resolution, increment)
    } else {
        None
    };

    if shape != SHAPE_RECTANGULAR {
        return None;
    }
    bits.skip_bits(1)?; // marker_bit
    let width = bits.read_bits(13)?;
    bits.skip_bits(1)?; // marker_bit
    let height = bits.read_bits(13)?;
    if width == 0 || height == 0 {
        return None;
    }

    Some(VideoGeometry {
        width,
        height,
        framerate,
        pixel_aspect,
    })
}

/// Width of `fixed_vop_time_increment` in bits: `ceil(log2(resolution))`, at
/// least one (14496-2 6.3.3).
fn time_increment_bits(resolution: u32) -> u32 {
    match resolution {
        0 | 1 => 1,
        r => 32 - (r - 1).leading_zeros(),
    }
}

/// Q16 frames per second from `vop_time_increment_resolution` ticks per second
/// and the `fixed_vop_time_increment` ticks each VOP lasts.
fn fixed_vop_rate_q16(resolution: u32, increment: u32) -> Option<u32> {
    if increment == 0 {
        return None;
    }
    u32::try_from(((resolution as u64) << 16) / increment as u64).ok()
}

/// Fuzzing entry: parse an MPEG-4 Part 2 VOL header out of an access unit.
/// Exposed only under `--cfg fuzzing` (cargo-fuzz).
#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = Mpeg4VideoCodec::geometry(data);
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

    use crate::annexb::BitWriter;

    /// The `video_object_layer_start_code` the fixtures use (layer 0).
    const VOL_START_CODE: u8 = 0x20;
    /// `aspect_ratio_info` 2: 12:11, the 625-line 4:3 sample aspect.
    const ASPECT_CODE_12_11: u32 = 2;
    /// `aspect_ratio_info` 1: square samples.
    const ASPECT_CODE_SQUARE: u32 = 1;
    /// A `vop_time_increment_resolution` of 30 with an increment of 1 is 30 fps.
    const TIME_INCREMENT_RESOLUTION_30: u32 = 30;

    /// A rectangular VOL header unit for `width` x `height`, start code
    /// included. `fixed_increment` of `None` leaves `fixed_vop_rate` clear.
    fn video_object_layer(
        width: u32,
        height: u32,
        aspect_ratio_info: u32,
        resolution: u32,
        fixed_increment: Option<u32>,
    ) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bit(1); // random_accessible_vol
        w.write_bits(1, 8); // video_object_type_indication (simple)
        w.write_bit(0); // is_object_layer_identifier
        w.write_bits(aspect_ratio_info, 4);
        if aspect_ratio_info == ASPECT_RATIO_EXTENDED {
            w.write_bits(1, 8); // par_width
            w.write_bits(1, 8); // par_height
        }
        w.write_bit(0); // vol_control_parameters
        w.write_bits(SHAPE_RECTANGULAR, 2); // video_object_layer_shape
        w.write_bit(1); // marker_bit
        w.write_bits(resolution, 16); // vop_time_increment_resolution
        w.write_bit(1); // marker_bit
        match fixed_increment {
            Some(increment) => {
                w.write_bit(1); // fixed_vop_rate
                w.write_bits(increment, time_increment_bits(resolution));
            }
            None => w.write_bit(0),
        }
        w.write_bit(1); // marker_bit
        w.write_bits(width, 13); // video_object_layer_width
        w.write_bit(1); // marker_bit
        w.write_bits(height, 13); // video_object_layer_height
        w.write_bit(1); // marker_bit
        w.write_bit(0); // interlaced
        w.align_to_byte();
        let mut unit = vec![0u8, 0, 1, VOL_START_CODE];
        unit.extend_from_slice(&w.into_bytes());
        unit
    }

    /// The configuration prefix a real stream leads with: VOS, VO, VOL.
    fn configuration(width: u32, height: u32) -> Vec<u8> {
        let mut out = vec![0u8, 0, 1, VISUAL_OBJECT_SEQUENCE_START_CODE, 0xF5];
        out.extend_from_slice(&[0, 0, 1, VISUAL_OBJECT_START_CODE, 0x09]);
        out.extend_from_slice(&video_object_layer(
            width,
            height,
            ASPECT_CODE_12_11,
            TIME_INCREMENT_RESOLUTION_30,
            Some(1),
        ));
        out
    }

    /// A VOP unit whose `vop_coding_type` is `coding_type` (0 = I, 1 = P),
    /// start code included.
    fn vop(coding_type: u8, tag: u8) -> Vec<u8> {
        vec![
            0,
            0,
            1,
            VOP_START_CODE,
            (coding_type << 6) | 0x2A,
            tag,
            0x11,
        ]
    }

    #[test]
    fn reads_geometry_framerate_and_sample_aspect_from_the_vol() {
        let au = video_object_layer(
            720,
            576,
            ASPECT_CODE_12_11,
            TIME_INCREMENT_RESOLUTION_30,
            Some(1),
        );
        let info = Mpeg4VideoCodec::geometry(&au).expect("VOL must parse");
        assert_eq!((info.width, info.height), (720, 576));
        assert_eq!(info.framerate, Some(30 << 16));
        assert_eq!(info.pixel_aspect, Some((12, 11)));
    }

    #[test]
    fn derives_a_fractional_framerate_from_the_time_increment() {
        // 30000 ticks per second, 1001 per VOP: 29.97 fps.
        let au = video_object_layer(640, 480, ASPECT_CODE_SQUARE, 30_000, Some(1001));
        let info = Mpeg4VideoCodec::geometry(&au).expect("VOL must parse");
        let expected = u32::try_from((30_000u64 << 16) / 1001).unwrap();
        assert_eq!(info.framerate, Some(expected));
        assert_eq!(info.framerate.unwrap() >> 16, 29, "~29.97 fps");
        assert_eq!(info.pixel_aspect, Some((1, 1)));
    }

    #[test]
    fn a_variable_vop_rate_leaves_the_framerate_unknown() {
        let au = video_object_layer(
            352,
            288,
            ASPECT_CODE_SQUARE,
            TIME_INCREMENT_RESOLUTION_30,
            None,
        );
        let info = Mpeg4VideoCodec::geometry(&au).expect("VOL must parse");
        assert_eq!((info.width, info.height), (352, 288));
        assert_eq!(info.framerate, None);
    }

    #[test]
    fn time_increment_bits_is_ceil_log2() {
        assert_eq!(time_increment_bits(1), 1);
        assert_eq!(time_increment_bits(2), 1);
        assert_eq!(time_increment_bits(30), 5);
        assert_eq!(time_increment_bits(32), 5);
        assert_eq!(time_increment_bits(33), 6);
    }

    #[test]
    fn rejects_a_zero_dimension_a_zero_resolution_and_a_truncated_vol() {
        let zero_width = video_object_layer(
            0,
            576,
            ASPECT_CODE_SQUARE,
            TIME_INCREMENT_RESOLUTION_30,
            Some(1),
        );
        assert!(
            Mpeg4VideoCodec::geometry(&zero_width).is_none(),
            "a zero coded width is not a geometry"
        );

        // A zero vop_time_increment_resolution would size the increment field
        // at ceil(log2(0)); the parse stops rather than guessing a width.
        let zero_resolution = video_object_layer(720, 576, ASPECT_CODE_SQUARE, 0, Some(1));
        assert!(Mpeg4VideoCodec::geometry(&zero_resolution).is_none());

        // Cut the header off inside the fields the geometry needs.
        let truncated = vec![0u8, 0, 1, VOL_START_CODE, 0x80, 0x40];
        assert!(Mpeg4VideoCodec::geometry(&truncated).is_none());
        assert!(Mpeg4VideoCodec::geometry(&[]).is_none());
    }

    #[test]
    fn access_units_split_at_each_vop() {
        let mut stream = configuration(720, 576);
        stream.extend_from_slice(&vop(0, 0xAA));
        let second = stream.len();
        stream.extend_from_slice(&[0, 0, 1, GROUP_OF_VOP_START_CODE, 0x00, 0x00, 0x0C]);
        stream.extend_from_slice(&vop(1, 0xBB));
        let starts = crate::startcodeparse::au_starts_by(&stream, Mpeg4VideoCodec::start_code_role);
        assert_eq!(starts, vec![0, second]);
    }

    #[test]
    fn keyframe_flag_follows_the_vop_coding_type() {
        assert!(Mpeg4VideoCodec::au_is_keyframe(&vop(0, 0xAA)), "an I-VOP");
        assert!(!Mpeg4VideoCodec::au_is_keyframe(&vop(1, 0xBB)), "a P-VOP");
    }

    #[test]
    fn config_headers_are_the_prefix_up_to_the_first_vop() {
        let config = configuration(720, 576);
        let mut au = config.clone();
        au.extend_from_slice(&vop(0, 0xAA));
        assert_eq!(Mpeg4VideoCodec::config_headers(&au), Some(config));
        // A bare VOP carries no configuration to cache.
        assert_eq!(Mpeg4VideoCodec::config_headers(&vop(0, 0xAA)), None);
    }

    // -- Element-level tests (drive Mpeg4VideoParse::process directly) -------

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

    fn mpeg4_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Mpeg4Part2,
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
    async fn emits_caps_changed_then_one_frame_per_vop() {
        let mut parse = Mpeg4VideoParse::new();
        parse.configure_pipeline(&mpeg4_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let mut first = configuration(720, 576);
        first.extend_from_slice(&vop(0, 0xAA));
        let second = vop(1, 0xBB);
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
                assert_eq!(*width, Dim::Fixed(720));
                assert_eq!(*height, Dim::Fixed(576));
                assert_eq!(*framerate, Rate::Fixed(30 << 16));
            }
            other => panic!("expected CapsChanged first, got {other:?}"),
        }
        assert_eq!(data_payloads(&sink), vec![first, second]);
        assert_eq!(
            parse.get_property("pixel-aspect-ratio"),
            Some(PropValue::Fraction(12, 11))
        );
    }

    #[test]
    fn config_interval_reinserts_the_headers_on_a_later_keyframe() {
        let mut parse = Mpeg4VideoParse::new().with_config_interval(-1);
        let config = configuration(720, 576);
        let mut configured = config.clone();
        configured.extend_from_slice(&vop(0, 0xAA));
        let out = parse.apply_config_interval(configured.clone(), 0, true);
        assert_eq!(out, configured, "an access unit with a VOL is untouched");

        let bare = vop(0, 0xBB);
        let out = parse.apply_config_interval(bare.clone(), 90_000, true);
        assert!(out.starts_with(&config), "the cached headers are prepended");
        assert!(out.ends_with(&bare), "the access unit is preserved");
    }

    #[test]
    fn config_interval_zero_never_reinserts() {
        let mut parse = Mpeg4VideoParse::new();
        let mut configured = configuration(720, 576);
        configured.extend_from_slice(&vop(0, 0xAA));
        let _ = parse.apply_config_interval(configured, 0, true);
        let bare = vop(0, 0xBB);
        assert_eq!(
            parse.apply_config_interval(bare.clone(), 90_000, true),
            bare
        );
    }

    #[test]
    fn config_interval_round_trips_as_a_property() {
        let mut parse = Mpeg4VideoParse::new();
        assert_eq!(
            parse.get_property("config-interval"),
            Some(PropValue::Int(0))
        );
        parse
            .set_property("config-interval", PropValue::Int(-1))
            .unwrap();
        assert_eq!(
            parse.get_property("config-interval"),
            Some(PropValue::Int(-1))
        );
        assert!(parse
            .set_property("config-interval", PropValue::Int(-2))
            .is_err());
    }

    #[tokio::test]
    async fn rejects_non_mpeg4_caps_in_intercept() {
        let parse = Mpeg4VideoParse::new();
        let mpeg2 = Caps::CompressedVideo {
            codec: VideoCodec::Mpeg2,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(parse.intercept_caps(&mpeg2), Err(G2gError::CapsMismatch));
    }
}
