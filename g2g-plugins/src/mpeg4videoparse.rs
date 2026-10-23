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

use g2g_core::caps::framerate_q16_from_fraction;
use g2g_core::time::ticks_to_ns;
use g2g_core::{PropertySpec, VideoCodec};

use crate::access_unit::StartCodeRole;
use crate::annexb::BitReader;
use crate::startcodeparse::{
    first_start_code_offset, sample_aspect, start_code_units, StartCodeCodec, StartCodeParse,
    VideoGeometry, CONFIG_INTERVAL_PROPERTY, PIXEL_ASPECT_PROPERTY,
};

/// `visual_object_sequence_start_code`.
pub(crate) const VISUAL_OBJECT_SEQUENCE_START_CODE: u8 = 0xB0;
/// `group_of_vop_start_code`.
const GROUP_OF_VOP_START_CODE: u8 = 0xB3;
/// `visual_object_start_code`.
pub(crate) const VISUAL_OBJECT_START_CODE: u8 = 0xB5;
/// `vop_start_code`: the coded picture.
const VOP_START_CODE: u8 = 0xB6;
/// `video_object_start_code` range.
pub(crate) const VIDEO_OBJECT_START_CODES: core::ops::RangeInclusive<u8> = 0x00..=0x1F;
/// `video_object_layer_start_code` range: the header carrying the geometry.
pub(crate) const VIDEO_OBJECT_LAYER_START_CODES: core::ops::RangeInclusive<u8> = 0x20..=0x2F;

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

// the `video_object_layer` fields ahead of `fixed_vop_rate`
struct LayerPrefix {
    pixel_aspect: Option<(u32, u32)>,
    shape: u32,
    vop_time_increment_resolution: u32,
}

fn parse_layer_prefix(bits: &mut BitReader) -> Option<LayerPrefix> {
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
    Some(LayerPrefix {
        pixel_aspect,
        shape,
        vop_time_increment_resolution,
    })
}

/// Parse a `video_object_layer` header payload (the bytes after its start code)
/// for the coded geometry. `None` when the layer is not rectangular, the header
/// is truncated, or it codes a zero dimension.
fn parse_video_object_layer(payload: &[u8]) -> Option<VideoGeometry> {
    let mut bits = BitReader::new(payload);
    let LayerPrefix {
        pixel_aspect,
        shape,
        vop_time_increment_resolution,
    } = parse_layer_prefix(&mut bits)?;

    let framerate = if bits.read_bit()? == 1 {
        // fixed_vop_rate: the increment is coded in as many bits as the
        // resolution needs, so a zero resolution makes the header unreadable.
        if vop_time_increment_resolution == 0 {
            return None;
        }
        let increment = bits.read_bits(time_increment_bits(vop_time_increment_resolution))?;
        framerate_q16_from_fraction(vop_time_increment_resolution, increment)
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

const VOP_CODING_TYPE_B: u32 = 2;

// seconds counts the `modulo_time_base` bits
struct VopTime {
    coding_type: u32,
    seconds: u64,
    increment: u32,
}

fn parse_vop_time(payload: &[u8], increment_bits: u32) -> Option<VopTime> {
    let mut bits = BitReader::new(payload);
    let coding_type = bits.read_bits(2)?;
    let mut seconds = 0u64;
    while bits.read_bit()? == 1 {
        seconds += 1;
    }
    bits.skip_bits(1)?; // marker_bit
    let increment = bits.read_bits(increment_bits)?;
    Some(VopTime {
        coding_type,
        seconds,
        increment,
    })
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct PackedInputSplit {
    pub(crate) b_frame_offset: usize,
    pub(crate) reference_pts_ns: u64,
}

// DivX packs a reference VOP and the B-VOP shown before it into one AVI chunk, whose stamp names the B-VOP
#[derive(Debug, Default)]
pub(crate) struct PackedVops {
    vop_time_increment_resolution: Option<u32>,
}

impl PackedVops {
    pub(crate) fn observe_layer_header(&mut self, input: &[u8]) {
        if let Some(resolution) = start_code_units(input)
            .filter(|(code, _)| VIDEO_OBJECT_LAYER_START_CODES.contains(code))
            .find_map(|(_, payload)| parse_layer_prefix(&mut BitReader::new(payload)))
            .map(|layer| layer.vop_time_increment_resolution)
            .filter(|resolution| *resolution > 0)
        {
            self.vop_time_increment_resolution = Some(resolution);
        }
    }

    pub(crate) fn split(&self, input: &[u8], pts_ns: u64) -> Option<PackedInputSplit> {
        let resolution = self.vop_time_increment_resolution?;
        let starts = crate::access_unit::access_unit_starts(
            input,
            false,
            crate::startcodeparse::unit_role::<Mpeg4VideoCodec>,
        );
        let [0, b_frame_offset] = starts[..] else {
            return None;
        };
        let increment_bits = time_increment_bits(resolution);
        let vop_time = |unit: &[u8]| {
            start_code_units(unit)
                .find(|(code, _)| *code == VOP_START_CODE)
                .and_then(|(_, payload)| parse_vop_time(payload, increment_bits))
        };
        let reference = vop_time(&input[..b_frame_offset])?;
        let b_frame = vop_time(&input[b_frame_offset..])?;
        if reference.coding_type == VOP_CODING_TYPE_B || b_frame.coding_type != VOP_CODING_TYPE_B {
            return None;
        }
        // ffmpeg counts a B-VOP's seconds from the same base as the reference coded before it
        let ticks_ahead = (i128::from(reference.seconds) - i128::from(b_frame.seconds))
            * i128::from(resolution)
            + i128::from(reference.increment)
            - i128::from(b_frame.increment);
        let ticks_ahead = u64::try_from(ticks_ahead).ok().filter(|ticks| *ticks > 0)?;
        Some(PackedInputSplit {
            b_frame_offset,
            reference_pts_ns: pts_ns.checked_add(ticks_to_ns(ticks_ahead, resolution))?,
        })
    }
}

/// Width of `fixed_vop_time_increment` in bits: `ceil(log2(resolution))`, at
/// least one (14496-2 6.3.3).
fn time_increment_bits(resolution: u32) -> u32 {
    match resolution {
        0 | 1 => 1,
        r => 32 - (r - 1).leading_zeros(),
    }
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
    use crate::testutil::{frame_with_bytes, CollectSink};
    use alloc::vec;
    use alloc::vec::Vec;

    use g2g_core::{AsyncElement, Caps, Dim, G2gError, PipelinePacket, PropValue, Rate};

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

    // a VOP shown `ticks` VOP time ticks into the stream, at TIME_INCREMENT_RESOLUTION_30
    fn timed_vop(coding_type: u32, ticks: u32) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(coding_type, 2);
        for _ in 0..ticks / TIME_INCREMENT_RESOLUTION_30 {
            w.write_bit(1); // modulo_time_base
        }
        w.write_bit(0);
        w.write_bit(1); // marker_bit
        w.write_bits(
            ticks % TIME_INCREMENT_RESOLUTION_30,
            time_increment_bits(TIME_INCREMENT_RESOLUTION_30),
        );
        w.write_bit(1); // marker_bit
        w.write_bit(1); // vop_coded
        w.write_bits(0x2A, 8);
        w.align_to_byte();
        let mut unit = vec![0u8, 0, 1, VOP_START_CODE];
        unit.extend_from_slice(&w.into_bytes());
        unit
    }

    const CODING_TYPE_I: u32 = 0;
    const CODING_TYPE_P: u32 = 1;

    fn pts_of_tick(ticks: u32) -> u64 {
        ticks_to_ns(u64::from(ticks), TIME_INCREMENT_RESOLUTION_30)
    }

    #[tokio::test]
    async fn a_packed_reference_vop_takes_its_own_display_time() {
        let mut parse = Mpeg4VideoParse::new();
        parse.configure_pipeline(&mpeg4_caps()).unwrap();
        let mut sink = CollectSink::default();
        // one stamp per AVI chunk: I, then P packed with the B shown first, then the second B
        let chunks = [
            (
                0,
                [configuration(720, 576), timed_vop(CODING_TYPE_I, 0)].concat(),
            ),
            (
                1,
                [timed_vop(CODING_TYPE_P, 3), timed_vop(VOP_CODING_TYPE_B, 1)].concat(),
            ),
            (2, timed_vop(VOP_CODING_TYPE_B, 2)),
        ];
        for (sequence, (tick, bytes)) in (0..).zip(chunks) {
            let mut frame = frame_with_bytes(sequence, bytes);
            frame.timing.pts_ns = pts_of_tick(tick);
            parse
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let stamps: Vec<u64> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f.timing.pts_ns),
                _ => None,
            })
            .collect();
        let packed_chunk_pts = pts_of_tick(1);
        let expected = vec![
            0,
            packed_chunk_pts + pts_of_tick(2),
            packed_chunk_pts,
            pts_of_tick(2),
        ];
        assert_eq!(stamps, expected);
    }

    #[test]
    fn only_a_reference_then_a_b_vop_is_split() {
        let mut packed = PackedVops::default();
        let two_references = [timed_vop(CODING_TYPE_P, 3), timed_vop(CODING_TYPE_P, 4)].concat();
        let reference_then_b =
            [timed_vop(CODING_TYPE_P, 3), timed_vop(VOP_CODING_TYPE_B, 1)].concat();
        let pts_ns = pts_of_tick(1);
        assert_eq!(
            packed.split(&reference_then_b, pts_ns),
            None,
            "no layer header has given the time resolution"
        );
        packed.observe_layer_header(&configuration(720, 576));
        assert_eq!(packed.split(&two_references, pts_ns), None);
        assert_eq!(
            packed.split(&reference_then_b, pts_ns),
            Some(PackedInputSplit {
                b_frame_offset: timed_vop(CODING_TYPE_P, 3).len(),
                reference_pts_ns: pts_ns + pts_of_tick(2),
            })
        );
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
        let starts = crate::access_unit::access_unit_starts(
            &stream,
            false,
            crate::startcodeparse::unit_role::<Mpeg4VideoCodec>,
        );
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

    fn mpeg4_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::Mpeg4Part2,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        }
    }

    fn data_payloads(sink: &CollectSink) -> Vec<Vec<u8>> {
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
        let mut sink = CollectSink::default();

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

    #[tokio::test]
    async fn a_variable_vop_rate_keeps_the_upstream_framerate() {
        const UPSTREAM_FRAMERATE_Q16: u32 = 25 << 16;
        let mut parse = Mpeg4VideoParse::new();
        parse.configure_pipeline(&mpeg4_caps()).unwrap();
        let mut sink = CollectSink::default();
        let upstream = Caps::CompressedVideo {
            codec: VideoCodec::Mpeg4Part2,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Fixed(UPSTREAM_FRAMERATE_Q16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        parse
            .process(PipelinePacket::CapsChanged(upstream), &mut sink)
            .await
            .unwrap();
        let mut buffer = video_object_layer(
            352,
            288,
            ASPECT_CODE_SQUARE,
            TIME_INCREMENT_RESOLUTION_30,
            None,
        );
        buffer.extend_from_slice(&vop(0, 0xAA));
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, buffer)),
                &mut sink,
            )
            .await
            .unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let refined = sink.packets.iter().find_map(|p| match p {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                width: Dim::Fixed(_),
                framerate,
                ..
            }) => Some(framerate.clone()),
            _ => None,
        });
        assert_eq!(refined, Some(Rate::Fixed(UPSTREAM_FRAMERATE_Q16)));
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(parse.intercept_caps(&mpeg2), Err(G2gError::CapsMismatch));
    }
}
