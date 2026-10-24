//! H.264 access-unit parser that refines source-side `Caps`.
//!
//! M6: scans each `DataFrame`'s bitstream for an SPS NAL unit, parses
//! width/height, and emits a `CapsChanged` packet with `Dim::Fixed` values
//! before forwarding the frame. This is the first element that refines
//! caps mid-stream — `RtspSrc` advertises `Dim::Any` at negotiation time
//! because the SPS only lands once bytes flow.
//!
//! Bitstream framing: both Annex-B (00 00 01 / 00 00 00 01 start codes) and
//! AVCC (4-byte length-prefixed NALs, what retina emits by default) are
//! accepted, detected per access unit via `annexb::nal_units_any`. The element
//! refines caps only; it does not rewrite the bitstream between framings.
//!
//! Framerate is recovered from the SPS VUI `timing_info` when present
//! (`time_scale / (2 * num_units_in_tick)`, emitted as a Q16 `Rate::Fixed`);
//! when the VUI omits it, caps carry the rate an upstream `CapsChanged` declared,
//! else `Rate::Any`.

use g2g_core::{Colorimetry, PropKind, PropertySpec, VideoCodec};

use crate::access_unit::StartCodeRole;
use crate::annexb::{h264_nal_type, strip_emulation_prevention, BitReader};
use crate::nalparse::{NalCodec, NalParse, SpsGeometry};
use crate::vui::{read_h264_vui_tail, read_vui_signal_type, H264VuiTail, VideoSignalType};

/// H.264 access-unit parser: `CompressedVideo{H264}` in and out, refining caps
/// from the SPS and (in re-framing mode) re-chunking to one access unit per
/// `DataFrame` with SPS/PPS re-insertion. The shared parser machinery lives in
/// [`NalParse`]; this file supplies only the H.264-specific hooks.
pub type H264Parse = NalParse<H264Codec>;

/// H.264 codec hooks for [`NalParse`].
#[derive(Debug)]
pub struct H264Codec;

impl NalCodec for H264Codec {
    const CODEC: VideoCodec = VideoCodec::H264;
    const NAME: &'static str = "H.264 parser";
    const DESCRIPTION: &'static str =
        "Parses an H.264 Annex-B stream and refines caps from SPS/PPS";
    const PROPERTIES: &'static [PropertySpec] = &[PropertySpec::new(
        "config-interval",
        PropKind::Int,
        "SPS/PPS re-insertion interval in seconds (0 = off, -1 = every IDR, N = every N s)",
    )
    .with_range("-1", "3600")
    .with_default("0")];
    // SPS (7) then PPS (8), the H.264 prepend order.
    const PARAM_SET_TYPES: &'static [u8] = &[7, 8];
    const SPS_TYPE: u8 = 7;

    fn nal_type(nal: &[u8]) -> Option<u8> {
        h264_nal_type(nal)
    }

    /// A new picture's first slice has `first_mb_in_slice == 0`, coded as a
    /// lone leading 1 bit of the slice RBSP. Slices 2..N of one picture carry a
    /// larger value and stay in its access unit. Ahead of a picture, an
    /// access-unit delimiter (9) and the parameter sets / SEI / prefix NALs that
    /// lead a picture (6, 7, 8, 14, 15, 18) open the next access unit
    /// (ISO/IEC 14496-10 7.4.1.2.3, the framing GStreamer's `h264parse` applies).
    fn unit_role(unit: &[u8]) -> Option<StartCodeRole> {
        let nal_type = h264_nal_type(unit)?;
        if (1..=5).contains(&nal_type) {
            let first_slice_of_picture = unit.get(1)? & 0x80 != 0;
            return Some(if first_slice_of_picture {
                StartCodeRole::Picture
            } else {
                StartCodeRole::PictureContinuation
            });
        }
        Some(if matches!(nal_type, 6 | 7 | 8 | 9 | 14 | 15 | 18) {
            StartCodeRole::Leads
        } else {
            StartCodeRole::Continues
        })
    }

    fn sps_info(nals: &[&[u8]]) -> Option<SpsGeometry> {
        sps_info(nals.iter().copied())
    }
}

/// The info from the first SPS NAL (nal_unit_type == 7) among `nals` we can
/// fully parse.
fn sps_info<'a>(nals: impl IntoIterator<Item = &'a [u8]>) -> Option<SpsGeometry> {
    for nal in nals {
        if nal.is_empty() {
            continue;
        }
        let nal_unit_type = nal[0] & 0x1F;
        if nal_unit_type != 7 {
            continue;
        }
        let rbsp = strip_emulation_prevention(&nal[1..]);
        if let Some(info) = parse_sps(&rbsp) {
            return Some(info);
        }
    }
    None
}

/// Profiles whose SPS carries the chroma format / bit depth / scaling block.
fn is_high_profile(profile_idc: u8) -> bool {
    matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    )
}

/// The H.264 SPS fields up to `vui_parameters_present_flag` (7.3.2.1.1), the
/// part every SPS consumer reads. Dimensions and crop offsets are the raw coded
/// values, unchecked: each consumer applies its own arithmetic policy.
// only the Vulkan decoder reads every field
#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct H264SpsHead {
    pub profile_idc: u8,
    pub level_idc: u8,
    pub seq_parameter_set_id: u32,
    pub chroma_format_idc: u32,
    pub bit_depth_luma_minus8: u32,
    pub bit_depth_chroma_minus8: u32,
    pub seq_scaling_matrix_present: bool,
    /// The picture-order-count fields, `frame_mbs_only_flag` and
    /// `separate_colour_plane_flag` included.
    pub poc: crate::poc::SpsPocParams,
    pub max_num_ref_frames: u32,
    pub gaps_in_frame_num_value_allowed_flag: bool,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub mb_adaptive_frame_field_flag: bool,
    pub direct_8x8_inference_flag: bool,
    pub frame_cropping_flag: bool,
    /// `frame_crop_{left,right,top,bottom}_offset`, zero without cropping.
    pub frame_crop_offsets: [u32; 4],
}

/// Parse an SPS RBSP (the bytes after the NAL header, emulation prevention
/// stripped) up to `vui_parameters_present_flag`, returning the reader
/// positioned at that flag. `None` on truncation or an out-of-range field.
pub(crate) fn parse_h264_sps_head(rbsp: &[u8]) -> Option<(H264SpsHead, BitReader<'_>)> {
    let [profile_idc, _constraint_flags, level_idc, ..] = *rbsp else {
        return None;
    };
    let mut br = BitReader::new(&rbsp[3..]);
    let seq_parameter_set_id = br.read_ue()?;

    // 4:2:0 8-bit unless a high profile signals otherwise.
    let mut chroma_format_idc = 1u32;
    let mut separate_colour_plane_flag = false;
    let mut bit_depth_luma_minus8 = 0;
    let mut bit_depth_chroma_minus8 = 0;
    let mut seq_scaling_matrix_present = false;
    if is_high_profile(profile_idc) {
        chroma_format_idc = br.read_ue()?;
        if chroma_format_idc == 3 {
            separate_colour_plane_flag = br.read_bit()? == 1;
        }
        bit_depth_luma_minus8 = br.read_ue()?;
        bit_depth_chroma_minus8 = br.read_ue()?;
        br.read_bit()?; // qpprime_y_zero_transform_bypass_flag
        seq_scaling_matrix_present = br.read_bit()? == 1;
        if seq_scaling_matrix_present {
            // variable length, so the walk has to cross them to reach the fields after
            let lists = if chroma_format_idc == 3 { 12 } else { 8 };
            crate::poc::skip_h264_scaling_lists(&mut br, lists)?;
        }
    }

    let mut poc = crate::poc::SpsPocParams {
        separate_colour_plane_flag,
        ..crate::poc::SpsPocParams::default()
    };
    let log2_max_frame_num_minus4 = br.read_ue()?;
    if log2_max_frame_num_minus4 > crate::poc::MAX_LOG2_MINUS4 {
        return None;
    }
    poc.log2_max_frame_num = log2_max_frame_num_minus4 + 4;
    crate::poc::read_h264_poc_block(&mut br, &mut poc)?;
    let max_num_ref_frames = br.read_ue()?;
    let gaps_in_frame_num_value_allowed_flag = br.read_bit()? == 1;

    let pic_width_in_mbs_minus1 = br.read_ue()?;
    let pic_height_in_map_units_minus1 = br.read_ue()?;
    poc.frame_mbs_only_flag = br.read_bit()? == 1;
    let mb_adaptive_frame_field_flag = !poc.frame_mbs_only_flag && br.read_bit()? == 1;
    let direct_8x8_inference_flag = br.read_bit()? == 1;

    let frame_cropping_flag = br.read_bit()? == 1;
    let frame_crop_offsets = if frame_cropping_flag {
        [br.read_ue()?, br.read_ue()?, br.read_ue()?, br.read_ue()?]
    } else {
        [0; 4]
    };

    let head = H264SpsHead {
        profile_idc,
        level_idc,
        seq_parameter_set_id,
        chroma_format_idc,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        seq_scaling_matrix_present,
        poc,
        max_num_ref_frames,
        gaps_in_frame_num_value_allowed_flag,
        pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1,
        mb_adaptive_frame_field_flag,
        direct_8x8_inference_flag,
        frame_cropping_flag,
        frame_crop_offsets,
    };
    Some((head, br))
}

/// Parse the SPS RBSP (post NAL-header byte) for the coded picture dimensions
/// and, when the VUI carries `timing_info`, the framerate. Returns `None` on a
/// parse failure up to the dimensions; a failure past them leaves only the
/// framerate unknown.
fn parse_sps(rbsp: &[u8]) -> Option<SpsGeometry> {
    let (head, mut br) = parse_h264_sps_head(rbsp)?;
    let frame_mbs_only_flag = u32::from(head.poc.frame_mbs_only_flag);
    let [crop_left, crop_right, crop_top, crop_bottom] = head.frame_crop_offsets;

    // Crop units in luma samples (H.264 7.4.2.1.1). ChromaArrayType 0
    // (monochrome, or 4:4:4 with separate colour planes) crops 1 x (2-fmof);
    // otherwise SubWidthC x SubHeightC*(2-fmof).
    let chroma_array_type = if head.poc.separate_colour_plane_flag {
        0
    } else {
        head.chroma_format_idc
    };
    let (sub_width_c, sub_height_c) = match chroma_array_type {
        1 => (2u32, 2u32), // 4:2:0
        2 => (2, 1),       // 4:2:2
        _ => (1, 1),       // 4:4:4 / monochrome
    };
    // Crop and dimension fields come from untrusted exp-Golomb, so fold with
    // saturating arithmetic (the additions and the *16 would otherwise overflow,
    // panicking in debug and wrapping to bogus caps in release).
    let crop_x = crop_left
        .saturating_add(crop_right)
        .saturating_mul(sub_width_c);
    let crop_y = crop_top
        .saturating_add(crop_bottom)
        .saturating_mul(sub_height_c.saturating_mul(2u32.saturating_sub(frame_mbs_only_flag)));

    let width = head
        .pic_width_in_mbs_minus1
        .saturating_add(1)
        .saturating_mul(16)
        .saturating_sub(crop_x);
    let height = (2 - frame_mbs_only_flag)
        .saturating_mul(head.pic_height_in_map_units_minus1.saturating_add(1))
        .saturating_mul(16)
        .saturating_sub(crop_y);

    // vui_parameters_present_flag follows frame cropping. Read it without `?`
    // so a stream truncated right here still yields the dimensions.
    let (colorimetry, vui) = match br.read_bit() {
        Some(1) => parse_vui(&mut br),
        _ => (Colorimetry::UNKNOWN, H264VuiTail::default()),
    };

    Some(SpsGeometry {
        width,
        height,
        framerate: vui.framerate,
        colorimetry,
        pic_timing: vui.pic_timing,
        // POC type 2 derives the picture order count from frame_num alone
        // (H.264 8.2.1.3), so coded order is display order by construction. The
        // VUI's max_num_reorder_frames is deliberately not read here: streams
        // declare 0 and reorder anyway (see `sps_reorder_frames`), and this flag
        // has to be a proof, not a hint.
        presents_in_decode_order: head.poc.pic_order_cnt_type == 2,
        poc: Some(head.poc),
    })
}

/// Walk the VUI parameters for the colour description (`UNKNOWN` when the
/// `video_signal_type` block is absent or truncated, never a guess) and the
/// rest of [`H264VuiTail`]. A VUI that ends early yields whatever was read
/// before the truncation.
fn parse_vui(br: &mut BitReader) -> (Colorimetry, H264VuiTail) {
    match read_vui_signal_type(br) {
        Some(signal) => (
            signal.map_or(Colorimetry::UNKNOWN, VideoSignalType::colorimetry),
            read_h264_vui_tail(br),
        ),
        None => (Colorimetry::UNKNOWN, H264VuiTail::default()),
    }
}

/// The output-reorder depth (`libavcodec` `has_b_frames`) a decoder must buffer
/// before emitting its first picture, read from the first SPS in `au`. Without
/// it, a decoder fed raw Annex-B access units emits the opening IDR early and
/// drops that GOP's leading pictures (lower POC, displayed before the IDR).
///
/// The value is the level-derived DPB bound for the frame size (0 for a baseline
/// / constrained profile that cannot reorder), matching how ffmpeg sizes the
/// buffer. The VUI `max_num_reorder_frames` is deliberately not trusted: a
/// stream may declare 0 yet still reorder (JVT conformance vectors do), and a
/// too-low value is exactly what drops the leading pictures. POC type 2 is the
/// one exception, returning 0: it pins output order to decode order by spec, a
/// proof rather than a declaration. `None` if there is no parseable SPS.
///
/// Consumed by the libavcodec decoder (`ffmpegdec`) and `vulkanvideodec`, so
/// gated like them.
#[cfg(any(
    all(target_os = "linux", feature = "ffmpeg"),
    all(
        any(target_os = "linux", target_os = "windows"),
        feature = "vulkan-video"
    )
))]
pub(crate) fn sps_reorder_frames(au: &[u8]) -> Option<u8> {
    for nal in crate::annexb::nal_units_any(au) {
        if nal.first().map(|b| b & 0x1F) != Some(7) {
            continue;
        }
        let rbsp = strip_emulation_prevention(&nal[1..]);
        if let Some(n) = parse_sps_reorder(&rbsp) {
            return Some(n);
        }
    }
    None
}

/// Parse an SPS RBSP (post NAL-header byte) for its level-derived reorder depth.
#[cfg(any(
    all(target_os = "linux", feature = "ffmpeg"),
    all(
        any(target_os = "linux", target_os = "windows"),
        feature = "vulkan-video"
    )
))]
fn parse_sps_reorder(rbsp: &[u8]) -> Option<u8> {
    // Baseline (profile 66) is not special-cased to zero: some JVT conformance
    // vectors declare Baseline yet still carry pictures libavcodec reorders (it
    // grows its reorder buffer at runtime and drops the GOP's leading pictures if
    // we seed too low). Baseline takes the same level-derived DPB bound below.
    let (head, _) = parse_h264_sps_head(rbsp)?;
    // POC type 2 pins output order to decode order (H.264 8.2.1.3), a proof the
    // stream cannot reorder, so the level DPB seed below would only buy latency
    // (5 frames at 720p). Live encoders with B-frames off emit this type.
    if head.poc.pic_order_cnt_type == 2 {
        return Some(0);
    }
    let pic_width_in_mbs = head.pic_width_in_mbs_minus1.saturating_add(1);
    let pic_height_in_map_units = head.pic_height_in_map_units_minus1.saturating_add(1);
    let frame_height_in_mbs = (2u32.saturating_sub(u32::from(head.poc.frame_mbs_only_flag)))
        .saturating_mul(pic_height_in_map_units);
    let pic_size_in_mbs = pic_width_in_mbs.saturating_mul(frame_height_in_mbs).max(1);
    let max_dpb_frames = (max_dpb_mbs(head.level_idc) / pic_size_in_mbs).min(16);
    Some(max_dpb_frames as u8)
}

/// `MaxDpbMbs` for an H.264 level (Table A-1), in macroblocks. The reorder /
/// DPB frame bound is this divided by the frame size in macroblocks.
#[cfg(any(
    all(target_os = "linux", feature = "ffmpeg"),
    all(
        any(target_os = "linux", target_os = "windows"),
        feature = "vulkan-video"
    )
))]
fn max_dpb_mbs(level_idc: u8) -> u32 {
    match level_idc {
        0..=10 => 396,
        11 => 900,
        12 | 13 | 20 => 2376,
        21 => 4752,
        22 | 30 => 8100,
        31 => 18000,
        32 => 20480,
        40 | 41 => 32768,
        42 => 34816,
        50 => 110400,
        51 | 52 => 184320,
        _ => 696320,
    }
}

/// Fuzzing entry: walk an H.264 access unit for the SPS and parse its geometry
/// (NAL scan, emulation-prevention strip, Exp-Golomb bit reader). Exposed only
/// under `--cfg fuzzing` (cargo-fuzz) so the normal public API is unchanged.
#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = H264Codec::extract_sps_info(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_unit::access_unit_starts;
    use crate::annexb::{nal_units, BitWriter};
    use crate::testutil::{frame_with_bytes, CollectSink};
    use alloc::vec;
    use alloc::vec::Vec;

    use g2g_core::{AsyncElement, Caps, CapsConstraint, Dim, G2gError, PipelinePacket, Rate};

    /// Build a minimal baseline-profile SPS RBSP for `width` x `height`
    /// (both multiples of 16), then frame it in Annex-B. Returns the
    /// full byte stream including a 4-byte start code and NAL header.
    fn build_test_annexb_sps(width: u32, height: u32) -> Vec<u8> {
        assert!(width.is_multiple_of(16) && height.is_multiple_of(16));
        let mut w = BitWriter::default();
        // Post NAL-header SPS fields:
        // seq_parameter_set_id = 0
        w.write_ue(0);
        // log2_max_frame_num_minus4 = 0
        w.write_ue(0);
        // pic_order_cnt_type = 0
        w.write_ue(0);
        // log2_max_pic_order_cnt_lsb_minus4 = 0
        w.write_ue(0);
        // max_num_ref_frames = 1
        w.write_ue(1);
        // gaps_in_frame_num_value_allowed_flag = 0
        w.write_bit(0);
        // pic_width_in_mbs_minus1
        w.write_ue(width / 16 - 1);
        // pic_height_in_map_units_minus1
        w.write_ue(height / 16 - 1);
        // frame_mbs_only_flag = 1
        w.write_bit(1);
        // direct_8x8_inference_flag = 0
        w.write_bit(0);
        // frame_cropping_flag = 0
        w.write_bit(0);
        // vui_parameters_present_flag = 0
        w.write_bit(0);
        // rbsp_trailing_bits: 1 then zero-pad
        w.write_bit(1);
        w.align_to_byte();

        let rbsp = w.into_bytes();

        // Annex-B framing: 00 00 00 01 | nal_header | profile/level bytes | rbsp
        // nal_header for SPS: forbidden_zero_bit=0, nal_ref_idc=3, nal_unit_type=7
        // → (3 << 5) | 7 = 0x67
        // Then the SPS's byte-aligned prefix: profile_idc=66 (baseline),
        // constraint flags + reserved = 0, level_idc=30 — chosen so the
        // parser takes the simple (non-high-profile) branch.
        let mut out = vec![0u8, 0, 0, 1, 0x67, 66, 0, 30];
        out.extend_from_slice(&rbsp);
        out
    }

    /// Frame an SPS RBSP in Annex-B with an explicit profile / level (unlike
    /// [`build_test_annexb_sps`], which hardcodes baseline). `rbsp` is the
    /// post-NAL-header bytes.
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    fn annexb_sps(profile_idc: u8, level_idc: u8, rbsp: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8, 0, 0, 1, 0x67, profile_idc, 0, level_idc];
        out.extend_from_slice(rbsp);
        out
    }

    /// Minimal SPS body (non-high-profile branch) at `width` x `height`.
    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    fn sps_body(width: u32, height: u32) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_ue(0); // seq_parameter_set_id
        w.write_ue(0); // log2_max_frame_num_minus4
        w.write_ue(0); // pic_order_cnt_type
        w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.write_ue(1); // max_num_ref_frames
        w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(width / 16 - 1);
        w.write_ue(height / 16 - 1);
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(0); // direct_8x8_inference_flag
        w.write_bit(0); // frame_cropping_flag
        w.write_bit(0); // vui_parameters_present_flag
        w.write_bit(1); // rbsp_trailing_bits
        w.align_to_byte();
        w.into_bytes()
    }

    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    #[test]
    fn sps_reorder_frames_uses_level_dpb_for_reordering_profile() {
        // Main profile, level 4.0, 1920x1088 (120x68 = 8160 MBs).
        // MaxDpbMbs(40) = 32768; 32768 / 8160 = 4 (capped at 16).
        let au = annexb_sps(77, 40, &sps_body(1920, 1088));
        assert_eq!(sps_reorder_frames(&au), Some(4));
    }

    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    #[test]
    fn sps_reorder_frames_uses_level_dpb_for_baseline_too() {
        // Baseline (66) is not special-cased to zero: some JVT conformance vectors
        // declare Baseline yet still reorder, so it takes the same level-derived
        // DPB bound as a reordering profile (32768 / 8160 = 4 here).
        let au = annexb_sps(66, 40, &sps_body(1920, 1088));
        assert_eq!(sps_reorder_frames(&au), Some(4));
    }

    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    #[test]
    fn sps_reorder_frames_zero_for_poc_type_2() {
        // POC type 2 pins output order to decode order, so the level DPB bound
        // does not apply: no reorder seed, whatever the profile / level.
        let mut w = BitWriter::default();
        w.write_ue(0); // seq_parameter_set_id
        w.write_ue(0); // log2_max_frame_num_minus4
        w.write_ue(2); // pic_order_cnt_type
        w.write_ue(1); // max_num_ref_frames
        w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(1920 / 16 - 1); // pic_width_in_mbs_minus1
        w.write_ue(1088 / 16 - 1); // pic_height_in_map_units_minus1
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(1); // direct_8x8_inference_flag
        w.write_bit(0); // frame_cropping_flag
        w.write_bit(0); // vui_parameters_present_flag
        w.write_bit(1); // rbsp_trailing_bits
        w.align_to_byte();
        let au = annexb_sps(77, 40, &w.into_bytes());
        assert_eq!(sps_reorder_frames(&au), Some(0));
    }

    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    #[test]
    fn sps_reorder_frames_crosses_scaling_lists() {
        // High profile with a custom scaling matrix: the lists sit before the
        // dimensions, so the walk has to skip them to reach the DPB bound.
        let mut w = BitWriter::default();
        w.write_ue(0); // seq_parameter_set_id
        w.write_ue(1); // chroma_format_idc (4:2:0)
        w.write_ue(0); // bit_depth_luma_minus8
        w.write_ue(0); // bit_depth_chroma_minus8
        w.write_bit(0); // qpprime_y_zero_transform_bypass_flag
        w.write_bit(1); // seq_scaling_matrix_present_flag
        w.write_bit(1); // seq_scaling_list_present_flag[0]
        w.write_ue(7); // delta_scale = se(+4)
        for _ in 1..16 {
            w.write_ue(0); // delta_scale = se(0)
        }
        for _ in 1..8 {
            w.write_bit(0); // seq_scaling_list_present_flag[1..8]
        }
        w.write_ue(0); // log2_max_frame_num_minus4
        w.write_ue(0); // pic_order_cnt_type
        w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.write_ue(1); // max_num_ref_frames
        w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(1920 / 16 - 1); // pic_width_in_mbs_minus1
        w.write_ue(1088 / 16 - 1); // pic_height_in_map_units_minus1
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(1); // rbsp_trailing_bits
        w.align_to_byte();
        let au = annexb_sps(100, 40, &w.into_bytes());
        // Same 1080p level-4.0 bound as the no-matrix case: 32768 / 8160 = 4.
        assert_eq!(sps_reorder_frames(&au), Some(4));
    }

    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    #[test]
    fn sps_reorder_frames_follows_each_level_dpb_size() {
        // MaxDpbMbs (Table A-1) over the frame's macroblocks, each under the 16 frame cap
        let cases: [(u8, (u32, u32), u32); 10] = [
            (10, (176, 144), 396 / 99),
            (11, (352, 288), 900 / 396),
            (12, (352, 288), 2376 / 396),
            (21, (352, 576), 4752 / 792),
            (30, (720, 576), 8100 / 1620),
            (31, (1280, 720), 18000 / 3600),
            (32, (1280, 1024), 20480 / 5120),
            (42, (1920, 1088), 34816 / 8160),
            (50, (1920, 1088), 110400 / 8160),
            (51, (4096, 2304), 184320 / 36864),
        ];
        for (level_idc, (width, height), expected_frames) in cases {
            let au = annexb_sps(77, level_idc, &sps_body(width, height));
            assert_eq!(
                sps_reorder_frames(&au).map(u32::from),
                Some(expected_frames),
                "level {level_idc} at {width}x{height}"
            );
        }
    }

    #[cfg(all(target_os = "linux", feature = "ffmpeg"))]
    #[test]
    fn sps_reorder_frames_none_without_sps() {
        // A stream carrying only a slice NAL (type 5 = IDR) has no SPS.
        assert_eq!(sps_reorder_frames(&[0, 0, 0, 1, 0x65, 0x88]), None);
    }

    #[test]
    fn parse_sps_saturates_adversarial_dimensions() {
        // Untrusted exp-Golomb dimension/crop fields must saturate, not
        // overflow-panic (debug) or wrap to bogus caps (release).
        let mut w = BitWriter::default();
        w.write_ue(0); // seq_parameter_set_id
        w.write_ue(0); // log2_max_frame_num_minus4
        w.write_ue(0); // pic_order_cnt_type
        w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.write_ue(1); // max_num_ref_frames
        w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(300_000_000); // pic_width_in_mbs_minus1 (overflows *16)
        w.write_ue(300_000_000); // pic_height_in_map_units_minus1
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(0); // direct_8x8_inference_flag
        w.write_bit(1); // frame_cropping_flag
        w.write_ue(3_000_000_000); // crop_left (sum overflows u32)
        w.write_ue(3_000_000_000); // crop_right
        w.write_ue(3_000_000_000); // crop_top
        w.write_ue(3_000_000_000); // crop_bottom
        w.write_bit(0); // vui_parameters_present_flag
        w.write_bit(1); // rbsp_trailing_bits
        w.align_to_byte();
        let mut au = vec![0u8, 0, 0, 1, 0x67, 66, 0, 30];
        au.extend_from_slice(&w.into_bytes());
        // Must not panic; the dimension *16 saturates to u32::MAX and the even
        // larger crop then saturating-subtracts it back to 0 (vs an overflow
        // panic in debug or a wrapped bogus value in release).
        let info = H264Codec::extract_sps_info(&au).expect("parses without overflow");
        assert_eq!((info.width, info.height), (0, 0));
    }

    /// Build an Annex-B SPS for `width` x `height` carrying a VUI `timing_info`
    /// block. Emulation-prevention bytes are inserted (as a real encoder would
    /// for the 32-bit fields' zero runs) so the parser's de-emulation
    /// round-trips them exactly.
    fn build_annexb_sps_with_timing(
        width: u32,
        height: u32,
        num_units_in_tick: u32,
        time_scale: u32,
    ) -> Vec<u8> {
        assert!(width.is_multiple_of(16) && height.is_multiple_of(16));
        let mut w = BitWriter::default();
        w.write_ue(0); // seq_parameter_set_id
        w.write_ue(0); // log2_max_frame_num_minus4
        w.write_ue(0); // pic_order_cnt_type
        w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.write_ue(1); // max_num_ref_frames
        w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(width / 16 - 1); // pic_width_in_mbs_minus1
        w.write_ue(height / 16 - 1); // pic_height_in_map_units_minus1
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(0); // direct_8x8_inference_flag
        w.write_bit(0); // frame_cropping_flag
        w.write_bit(1); // vui_parameters_present_flag
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(1); // timing_info_present_flag
        w.write_bits(num_units_in_tick, 32);
        w.write_bits(time_scale, 32);
        w.write_bit(0); // fixed_frame_rate_flag
        w.write_bit(1); // rbsp_trailing_bits
        w.align_to_byte();
        let rbsp = w.into_bytes();

        // NAL payload after the 0x67 header: profile/constraint/level + RBSP,
        // emulation-prevented like an encoder's output.
        let mut payload = vec![66u8, 0, 30];
        payload.extend_from_slice(&rbsp);
        let payload = crate::annexb::add_emulation_prevention(&payload);

        let mut out = vec![0u8, 0, 0, 1, 0x67];
        out.extend_from_slice(&payload);
        out
    }

    /// Build an Annex-B SPS whose VUI carries a `video_signal_type` block with
    /// the given CICP colour description (and no timing info).
    fn build_annexb_sps_with_color(
        primaries: u8,
        transfer: u8,
        matrix: u8,
        full_range: bool,
    ) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_ue(0); // seq_parameter_set_id
        w.write_ue(0); // log2_max_frame_num_minus4
        w.write_ue(0); // pic_order_cnt_type
        w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.write_ue(1); // max_num_ref_frames
        w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
        w.write_ue(1920 / 16 - 1); // pic_width_in_mbs_minus1
        w.write_ue(1088 / 16 - 1); // pic_height_in_map_units_minus1
        w.write_bit(1); // frame_mbs_only_flag
        w.write_bit(0); // direct_8x8_inference_flag
        w.write_bit(0); // frame_cropping_flag
        w.write_bit(1); // vui_parameters_present_flag
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(1); // video_signal_type_present_flag
        w.write_bits(5, 3); // video_format (unspecified)
        w.write_bit(u32::from(full_range)); // video_full_range_flag
        w.write_bit(1); // colour_description_present_flag
        w.write_bits(u32::from(primaries), 8);
        w.write_bits(u32::from(transfer), 8);
        w.write_bits(u32::from(matrix), 8);
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(0); // timing_info_present_flag
        w.write_bit(1); // rbsp_trailing_bits
        w.align_to_byte();
        let mut payload = vec![66u8, 0, 40];
        payload.extend_from_slice(&w.into_bytes());
        let payload = crate::annexb::add_emulation_prevention(&payload);
        let mut out = vec![0u8, 0, 0, 1, 0x67];
        out.extend_from_slice(&payload);
        out
    }

    #[test]
    fn recovers_bt709_colorimetry_from_vui() {
        // CICP 1/1/1 + limited range is exactly the bt709 preset.
        let stream = build_annexb_sps_with_color(1, 1, 1, false);
        let info = H264Codec::extract_sps_info(&stream).expect("SPS with colour must parse");
        assert_eq!(info.colorimetry, Colorimetry::BT709);
    }

    #[test]
    fn absent_vui_colorimetry_stays_unknown() {
        let stream = build_test_annexb_sps(1280, 720);
        let info = H264Codec::extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!(info.colorimetry, Colorimetry::UNKNOWN);
    }

    #[test]
    fn garbage_colour_codepoints_map_unknown_without_panic() {
        // Codepoints outside the modeled CICP set must not panic or leak a
        // guess into caps: each field falls back to Unknown. The coded
        // full-range flag is real, so the range stays concrete.
        let stream = build_annexb_sps_with_color(200, 250, 77, true);
        let info = H264Codec::extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!(
            info.colorimetry,
            Colorimetry {
                range: g2g_core::ColorRange::Full,
                matrix: g2g_core::MatrixCoefficients::Unknown,
                transfer: g2g_core::TransferCharacteristics::Unknown,
                primaries: g2g_core::ColorPrimaries::Unknown,
            }
        );
    }

    #[tokio::test]
    async fn emits_colorimetry_in_refined_caps() {
        // The 1080p BT.709 case this track exists for: the parser's CapsChanged
        // must carry the VUI colour description, not discard it.
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();
        let stream = build_annexb_sps_with_color(1, 1, 1, false);
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, stream)),
                &mut sink,
            )
            .await
            .unwrap();
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo { colorimetry, .. }) => {
                assert_eq!(*colorimetry, Colorimetry::BT709);
            }
            other => panic!("expected CapsChanged first, got {other:?}"),
        }
    }

    #[test]
    fn round_trips_a_1280x720_sps() {
        let stream = build_test_annexb_sps(1280, 720);
        let info = H264Codec::extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(
            info.framerate, None,
            "no VUI timing in the baseline fixture"
        );
    }

    #[test]
    fn round_trips_a_1920x1080_sps() {
        let stream = build_test_annexb_sps(1920, 1088);
        // height 1088 because 1080 is not a multiple of 16; the test
        // builder asserts on alignment. Real 1080p streams use cropping.
        let info = H264Codec::extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (1920, 1088));
    }

    #[test]
    fn parses_an_avcc_framed_sps() {
        // Re-frame the same SPS NAL as AVCC (4-byte length prefix instead of
        // the Annex-B start code) and confirm the dimensions still resolve.
        let annexb = build_test_annexb_sps(1280, 720);
        let nal = &annexb[4..]; // drop the 00 00 00 01 start code
        let mut avcc = (nal.len() as u32).to_be_bytes().to_vec();
        avcc.extend_from_slice(nal);
        let info = H264Codec::extract_sps_info(&avcc).expect("AVCC SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
    }

    #[test]
    fn recovers_framerate_from_vui_timing() {
        // num_units_in_tick = 1, time_scale = 60 -> 30 fps.
        let stream = build_annexb_sps_with_timing(1280, 720, 1, 60);
        let info = H264Codec::extract_sps_info(&stream).expect("SPS with VUI must parse");
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.framerate, Some(30 << 16), "30 fps in Q16");
    }

    #[test]
    fn ignores_non_sps_nals() {
        // A stream containing only a slice NAL (type 5 = IDR) returns None.
        let stream = [0u8, 0, 0, 1, 0x65, 0xAA, 0xBB, 0xCC];
        assert!(H264Codec::extract_sps_info(&stream).is_none());
    }

    #[test]
    fn returns_none_on_empty_input() {
        assert!(H264Codec::extract_sps_info(&[]).is_none());
    }

    #[test]
    fn strips_emulation_prevention_bytes() {
        // Input "00 00 03 01" should decode to "00 00 01".
        let ebsp = [0u8, 0, 3, 1, 2, 0, 0, 3, 0xFF];
        let rbsp = strip_emulation_prevention(&ebsp);
        assert_eq!(rbsp, [0u8, 0, 1, 2, 0, 0, 0xFF]);
    }

    #[test]
    // the binary grouping aligns to the Exp-Golomb code words, not byte nibbles.
    #[allow(clippy::unusual_byte_groupings)]
    fn bit_reader_decodes_known_ue_codes() {
        // Bits: 1 010 011 00100 → ue values 0, 1, 2, 3
        let bytes = [0b1_010_011_0, 0b0100_0000];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_ue(), Some(0));
        assert_eq!(r.read_ue(), Some(1));
        assert_eq!(r.read_ue(), Some(2));
        assert_eq!(r.read_ue(), Some(3));
    }

    #[test]
    fn parse_vui_reads_timing_info() {
        // num_units_in_tick = 1001, time_scale = 60000 -> 29.97 fps, exercising
        // the Q16 conversion on a non-integer rate.
        let mut w = BitWriter::default();
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(1); // timing_info_present_flag
        w.write_bits(1001, 32);
        w.write_bits(60000, 32);
        w.write_bit(0); // fixed_frame_rate_flag
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        let fps = parse_vui(&mut br).1.framerate.expect("timing info present");
        let expected = ((60000u64 << 16) / (2 * 1001)) as u32;
        assert_eq!(fps, expected);
        assert_eq!(fps >> 16, 29, "~29.97 fps");
    }

    #[test]
    fn parse_vui_absent_timing_is_none() {
        let mut w = BitWriter::default();
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(0); // timing_info_present_flag = 0
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(parse_vui(&mut br).1.framerate, None);
    }

    // -- Element-level tests (drive H264Parse::process directly) -----------

    fn h264_parse_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();

        let stream = build_test_annexb_sps(1280, 720);
        let frame = frame_with_bytes(0, stream);
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo { width, height, .. }) => {
                assert_eq!(*width, Dim::Fixed(1280));
                assert_eq!(*height, Dim::Fixed(720));
            }
            other => panic!("expected CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    fn last_caps_rate(sink: &CollectSink) -> Option<Rate> {
        sink.packets.iter().rev().find_map(|packet| match packet {
            PipelinePacket::CapsChanged(Caps::CompressedVideo { framerate, .. }) => {
                Some(framerate.clone())
            }
            _ => None,
        })
    }

    #[tokio::test]
    async fn keeps_an_upstream_rate_the_sps_does_not_code() {
        const CONTAINER_FPS: u32 = 25;
        // what configure sees: the negotiation placeholder fixated to its minimum
        const FIXATED_PLACEHOLDER_Q16: u32 = 1 << 16;
        let declared = |framerate| Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        let sps_without_timing = || frame_with_bytes(0, build_test_annexb_sps(1280, 720));

        let mut parse = H264Parse::new();
        parse
            .configure_pipeline(&declared(Rate::Fixed(FIXATED_PLACEHOLDER_Q16)))
            .unwrap();
        let mut sink = CollectSink::default();
        parse
            .process(PipelinePacket::DataFrame(sps_without_timing()), &mut sink)
            .await
            .unwrap();
        assert_eq!(
            last_caps_rate(&sink),
            Some(Rate::Any),
            "a negotiated placeholder is not a rate"
        );

        let mut parse = H264Parse::new();
        parse
            .configure_pipeline(&declared(Rate::Fixed(FIXATED_PLACEHOLDER_Q16)))
            .unwrap();
        let mut sink = CollectSink::default();
        let container_rate = Rate::Fixed(CONTAINER_FPS << 16);
        parse
            .process(
                PipelinePacket::CapsChanged(declared(container_rate.clone())),
                &mut sink,
            )
            .await
            .unwrap();
        parse
            .process(PipelinePacket::DataFrame(sps_without_timing()), &mut sink)
            .await
            .unwrap();
        assert_eq!(last_caps_rate(&sink), Some(container_rate));
    }

    #[tokio::test]
    async fn emits_caps_from_an_avcc_access_unit() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();

        // Re-frame the SPS NAL as AVCC (4-byte length prefix).
        let annexb = build_test_annexb_sps(1280, 720);
        let nal = &annexb[4..];
        let mut avcc = (nal.len() as u32).to_be_bytes().to_vec();
        avcc.extend_from_slice(nal);

        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, avcc)),
                &mut sink,
            )
            .await
            .unwrap();

        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo { width, height, .. }) => {
                assert_eq!(*width, Dim::Fixed(1280));
                assert_eq!(*height, Dim::Fixed(720));
            }
            other => panic!("expected CapsChanged from the AVCC AU, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn marks_avcc_keyframes() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();
        let idr_slice: &[u8] = &[0x65, 0x88];
        let non_idr_slice: &[u8] = &[0x41, 0x9A];
        for (seq, nal) in [idr_slice, non_idr_slice].into_iter().enumerate() {
            let mut avcc = (nal.len() as u32).to_be_bytes().to_vec();
            avcc.extend_from_slice(nal);
            parse
                .process(
                    PipelinePacket::DataFrame(frame_with_bytes(seq as u64, avcc)),
                    &mut sink,
                )
                .await
                .unwrap();
        }
        let keyframes: Vec<bool> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f.timing.keyframe),
                _ => None,
            })
            .collect();
        assert_eq!(keyframes, vec![true, false]);
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();

        for seq in 0..3 {
            let stream = build_test_annexb_sps(1280, 720);
            let frame = frame_with_bytes(seq, stream);
            parse
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }

        let caps_count = sink
            .packets
            .iter()
            .filter(|p| matches!(p, PipelinePacket::CapsChanged(_)))
            .count();
        assert_eq!(
            caps_count, 1,
            "CapsChanged must fire once for identical SPS"
        );
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn re_emits_caps_on_resolution_change() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();

        let frame_720 = frame_with_bytes(0, build_test_annexb_sps(1280, 720));
        parse
            .process(PipelinePacket::DataFrame(frame_720), &mut sink)
            .await
            .unwrap();

        let frame_1080 = frame_with_bytes(1, build_test_annexb_sps(1920, 1088));
        parse
            .process(PipelinePacket::DataFrame(frame_1080), &mut sink)
            .await
            .unwrap();

        let widths: Vec<Dim> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::CapsChanged(Caps::CompressedVideo { width, .. }) => {
                    Some(width.clone())
                }
                _ => None,
            })
            .collect();
        assert_eq!(widths, vec![Dim::Fixed(1280), Dim::Fixed(1920)]);
        assert_eq!(parse.caps_changes_emitted(), 2);
    }

    #[tokio::test]
    async fn rejects_non_h264_caps_in_intercept() {
        let parse = H264Parse::new();
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(parse.intercept_caps(&vp9), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_h264_any() {
        // M16 step 5g: native shape is `Identity` over H.264 with any
        // geometry. With a fully-native chain the solver enforces the
        // input/output coupling and the format requirement during
        // arc-consistency, not via the dynamic `intercept_caps`.
        let parse = H264Parse::new();
        let c = parse.caps_constraint_as_transform();
        match c {
            CapsConstraint::Identity(set) => {
                assert_eq!(
                    set.alternatives(),
                    &[Caps::CompressedVideo {
                        codec: VideoCodec::H264,
                        width: Dim::Any,
                        height: Dim::Any,
                        framerate: Rate::Any,
                        colorimetry: g2g_core::Colorimetry::UNKNOWN,
                        alignment: g2g_core::BitstreamAlignment::Unspecified,
                    }]
                );
            }
            _ => panic!("expected Identity"),
        }
    }

    // ---- M421 access-unit re-framing ----

    /// An Annex-B VCL slice NAL (type 1). `first` picks `first_mb_in_slice == 0`
    /// (a new picture's first slice: leading RBSP bit 1) vs a continuation slice
    /// (leading bit 0). `tag` makes the payload distinguishable; the payload bytes
    /// avoid `00 00` so they never look like a start code.
    fn annexb_vcl(first: bool, tag: u8) -> Vec<u8> {
        let mut v = vec![0, 0, 0, 1, 0x41]; // start code + NAL header (type 1, ref)
        v.push(if first { 0x80 } else { 0x40 }); // first RBSP byte: MSB = first_mb==0
        v.extend_from_slice(&[0xAA, 0xBB, tag, 0x11]);
        v
    }

    /// Pull the `DataFrame` byte payloads out of a sink, in order.
    fn data_payloads(sink: &CollectSink) -> Vec<Vec<u8>> {
        sink.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn au_starts_groups_slices_into_one_picture() {
        // Picture A: two slices (first_mb==0 then a continuation). Picture B: one
        // slice. Two access units, not three.
        let mut stream = annexb_vcl(true, 1);
        stream.extend_from_slice(&annexb_vcl(false, 2)); // same picture (continuation)
        let b_off = stream.len();
        stream.extend_from_slice(&annexb_vcl(true, 3)); // new picture
        let starts = access_unit_starts(&stream, true, H264Codec::unit_role);
        assert_eq!(
            starts,
            vec![0, b_off],
            "two access units: A(2 slices) then B"
        );
    }

    #[tokio::test]
    async fn reframing_splits_two_access_units_in_one_buffer() {
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();

        // One input buffer carrying two complete pictures back to back.
        let au0 = annexb_vcl(true, 1);
        let au1 = annexb_vcl(true, 2);
        let mut buf = au0.clone();
        buf.extend_from_slice(&au1);
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, buf)),
                &mut sink,
            )
            .await
            .unwrap();
        // The first AU is emitted once the second AU's start is seen; the second
        // stays buffered until end of stream.
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let payloads = data_payloads(&sink);
        assert_eq!(
            payloads.len(),
            2,
            "two pictures -> two access-unit DataFrames"
        );
        assert_eq!(payloads[0], au0, "first AU emitted whole");
        assert_eq!(payloads[1], au1, "second AU emitted whole on EOS");
    }

    #[tokio::test]
    async fn an_access_unit_aligned_buffer_stays_whole() {
        let mut parse = H264Parse::reframing();
        let sample_caps = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::AccessUnit,
        };
        parse.configure_pipeline(&sample_caps).unwrap();
        let mut sink = CollectSink::default();
        // a container sample holding two first slices, as fate-suite's two_first_slice.mp4 does
        let sample = [annexb_vcl(true, 1), annexb_vcl(true, 2)].concat();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, sample.clone())),
                &mut sink,
            )
            .await
            .unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        assert_eq!(data_payloads(&sink), vec![sample]);
    }

    #[tokio::test]
    async fn a_trailing_delimiter_leaves_each_sample_stamp_on_its_own_picture() {
        // fate-suite's mixed-nal-coding.mp4 ends every sample with the delimiter of the next access unit
        const ACCESS_UNIT_DELIMITER: [u8; 6] = [0, 0, 0, 1, 0x09, 0xF0];
        const SAMPLE_PERIOD_NS: u64 = 33_366_667;
        const CODED_SLICE_NAL_TYPE: u8 = 1;
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();
        let stamps: Vec<u64> = (1..=3).map(|n| n * SAMPLE_PERIOD_NS).collect();
        for (tag, &pts_ns) in (1u8..).zip(&stamps) {
            let mut frame = frame_with_bytes(
                u64::from(tag),
                [annexb_vcl(true, tag), ACCESS_UNIT_DELIMITER.to_vec()].concat(),
            );
            frame.timing.pts_ns = pts_ns;
            parse
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let picture_stamps: Vec<u64> = sink
            .packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some(frame),
                _ => None,
            })
            .filter(|frame| {
                frame.domain.as_system_slice().is_some_and(|au| {
                    nal_units(au).any(|nal| nal[0] & 0x1F == CODED_SLICE_NAL_TYPE)
                })
            })
            .map(|frame| frame.timing.pts_ns)
            .collect();
        assert_eq!(picture_stamps, stamps);
    }

    #[tokio::test]
    async fn reframing_reads_a_byte_stream_that_opens_with_junk_as_annex_b() {
        // fate-suite's crew_cif.nal opens with these two bytes before its first
        // start code, which made the parser take the file for length-prefixed NALs.
        const LEADING_JUNK: [u8; 2] = [0x00, 0x1F];
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();

        let au0 = annexb_vcl(true, 1);
        let au1 = annexb_vcl(true, 2);
        let mut buf = LEADING_JUNK.to_vec();
        buf.extend_from_slice(&au0);
        buf.extend_from_slice(&au1);
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, buf)),
                &mut sink,
            )
            .await
            .expect("the stream frames as Annex-B");
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let payloads = data_payloads(&sink);
        assert_eq!(payloads.len(), 2, "both pictures come out: {payloads:?}");
        assert_eq!(payloads[0], au0);
        assert_eq!(payloads[1], au1);
    }

    #[tokio::test]
    async fn reframing_reads_avcc_whose_length_prefix_looks_like_a_start_code() {
        use crate::annexb::{ANNEX_B_START_CODE, AVCC_LENGTH_SIZE, START_CODE_PREFIX};
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();

        // the shortest NAL whose 4-byte length prefix begins `00 00 01`
        let mut length_prefix = [0u8; AVCC_LENGTH_SIZE];
        length_prefix[..START_CODE_PREFIX.len()].copy_from_slice(&START_CODE_PREFIX);
        let nal_len = u32::from_be_bytes(length_prefix) as usize;
        let mut expected = Vec::new();
        for tag in 0..2u8 {
            // a first_mb == 0 slice, so each buffer is a picture of its own
            let mut slice = vec![0x41u8, 0x80, tag];
            slice.resize(nal_len, 0xAA);
            let mut avcc = length_prefix.to_vec();
            avcc.extend_from_slice(&slice);
            parse
                .process(
                    PipelinePacket::DataFrame(frame_with_bytes(tag.into(), avcc)),
                    &mut sink,
                )
                .await
                .expect("the stream frames as length-prefixed");
            expected.push([&ANNEX_B_START_CODE[..], &slice].concat());
        }
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert_eq!(data_payloads(&sink), expected);
    }

    /// An upstream that hands over one access unit per buffer (RtspSrc, the
    /// depayloader, the demuxers) must not have its trailing picture held back for
    /// the next buffer's start code, which cost a frame period of latency on every
    /// live stream.
    #[tokio::test]
    async fn reframing_stops_holding_once_the_input_proves_au_aligned() {
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();

        let aus: Vec<Vec<u8>> = (0..4).map(|tag| annexb_vcl(true, tag as u8)).collect();
        for (seq, au) in aus.iter().enumerate() {
            parse
                .process(
                    PipelinePacket::DataFrame(frame_with_bytes(seq as u64, au.clone())),
                    &mut sink,
                )
                .await
                .unwrap();
        }

        // No Eos: everything fed must already be downstream.
        let payloads = data_payloads(&sink);
        assert_eq!(
            payloads.len(),
            aus.len(),
            "every access unit is out without waiting for a successor"
        );
        assert_eq!(payloads, aus, "and each one whole, in order");
    }

    /// The other half: a chunked byte stream (filesrc) still waits, since
    /// a buffer boundary there says nothing about where an access unit ends.
    #[tokio::test]
    async fn reframing_still_holds_the_tail_of_a_chunked_byte_stream() {
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();

        let aus: Vec<Vec<u8>> = (0..4).map(|tag| annexb_vcl(true, tag as u8)).collect();
        let stream: Vec<u8> = aus.concat();
        // A chunk size coprime with the access-unit size, so no chunk boundary ever
        // lands on an access-unit start.
        let chunk = aus[0].len() + 3;
        for (seq, part) in stream.chunks(chunk).enumerate() {
            parse
                .process(
                    PipelinePacket::DataFrame(frame_with_bytes(seq as u64, part.to_vec())),
                    &mut sink,
                )
                .await
                .unwrap();
        }

        assert_eq!(
            data_payloads(&sink).len(),
            aus.len() - 1,
            "the last access unit is still open until its successor or Eos"
        );
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        assert_eq!(data_payloads(&sink), aus, "Eos flushes it, bytes intact");
    }

    #[tokio::test]
    async fn reframing_reassembles_an_au_split_across_buffers() {
        // The regression that made HLS video garbage: one Annex-B access unit
        // delivered as two buffers (one MPEG-TS PES carrying the tail), the second
        // with no leading start code. A per-buffer framing guess would misread the
        // tail as AVCC and corrupt it; latching the framing keeps it Annex-B.
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = CollectSink::default();

        let au = annexb_vcl(true, 7);
        let split = 6; // mid-NAL: after the start code + header + first RBSP byte
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, au[..split].to_vec())),
                &mut sink,
            )
            .await
            .unwrap();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(1, au[split..].to_vec())),
                &mut sink,
            )
            .await
            .unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let payloads = data_payloads(&sink);
        assert_eq!(
            payloads.len(),
            1,
            "the split access unit reassembles into one"
        );
        assert_eq!(
            payloads[0], au,
            "reassembled bytes are bit-for-bit the original AU"
        );
    }

    #[test]
    fn config_interval_reinserts_sps_on_idr_without_params() {
        let mut p = H264Parse::reframing().with_config_interval(-1);
        // An IDR AU carrying SPS + PPS: caches them and is returned unchanged.
        let mut au1 = build_test_annexb_sps(320, 240);
        au1.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE]); // PPS (type 8)
        au1.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88]); // IDR slice (type 5)
        let out1 = p.apply_config_interval(au1.clone(), 0, true);
        assert_eq!(out1, au1, "an IDR that already carries SPS is untouched");
        // A later IDR with no parameter sets gets the cached SPS/PPS prepended.
        let au2 = vec![0, 0, 0, 1, 0x65, 0x88];
        let out2 = p.apply_config_interval(au2.clone(), 90_000, true);
        assert!(
            nal_units(&out2).any(|n| h264_nal_type(n) == Some(7)),
            "result carries an SPS"
        );
        assert!(
            nal_units(&out2).any(|n| h264_nal_type(n) == Some(8)),
            "result carries a PPS"
        );
        assert!(
            out2.ends_with(&au2),
            "the original AU is preserved at the tail"
        );
    }

    #[test]
    fn config_interval_zero_leaves_idr_untouched() {
        let mut p = H264Parse::reframing(); // default config-interval = 0
        let mut au1 = build_test_annexb_sps(320, 240);
        au1.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88]);
        let _ = p.apply_config_interval(au1, 0, true); // caches the SPS
        let au2 = vec![0, 0, 0, 1, 0x65, 0x88];
        let out2 = p.apply_config_interval(au2.clone(), 90_000, true);
        assert_eq!(out2, au2, "interval 0 never re-inserts parameter sets");
    }

    #[test]
    fn config_interval_seconds_paces_reinsertion() {
        let mut p = H264Parse::reframing().with_config_interval(2); // every 2 s
        let mut au1 = build_test_annexb_sps(320, 240);
        au1.extend_from_slice(&[0, 0, 0, 1, 0x65, 0x88]);
        let _ = p.apply_config_interval(au1, 0, true); // last insert at pts 0
                                                       // 1 s later (pts is nanoseconds): under the 2 s interval, not re-inserted.
        let au = vec![0, 0, 0, 1, 0x65, 0x88];
        let early = p.apply_config_interval(au.clone(), 1_000_000_000, true);
        assert_eq!(early, au, "before the interval elapses, no re-insertion");
        // 2 s later: due.
        let late = p.apply_config_interval(au.clone(), 2_000_000_000, true);
        assert!(
            nal_units(&late).any(|n| h264_nal_type(n) == Some(7)),
            "re-inserted after 2 s"
        );
    }

    // a present scaling list codes delta_scale se(-8), the switch to the default matrix
    fn annexb_sps_from_head(head: &H264SpsHead, scaling_lists_present: &[bool]) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bits(u32::from(head.profile_idc), 8);
        w.write_bits(0, 8); // constraint_set flags + reserved_zero_2bits
        w.write_bits(u32::from(head.level_idc), 8);
        w.write_ue(head.seq_parameter_set_id);
        if is_high_profile(head.profile_idc) {
            w.write_ue(head.chroma_format_idc);
            if head.chroma_format_idc == 3 {
                w.write_bit(u32::from(head.poc.separate_colour_plane_flag));
            }
            w.write_ue(head.bit_depth_luma_minus8);
            w.write_ue(head.bit_depth_chroma_minus8);
            w.write_bit(0); // qpprime_y_zero_transform_bypass_flag
            w.write_bit(u32::from(head.seq_scaling_matrix_present));
            for &present in scaling_lists_present {
                w.write_bit(u32::from(present));
                if present {
                    w.write_ue(16); // delta_scale = se(-8)
                }
            }
        }
        w.write_ue(head.poc.log2_max_frame_num - 4);
        w.write_ue(u32::from(head.poc.pic_order_cnt_type));
        assert!(
            head.poc.pic_order_cnt_type != 1,
            "POC type 1 is not written here"
        );
        if head.poc.pic_order_cnt_type == 0 {
            w.write_ue(head.poc.log2_max_pic_order_cnt_lsb - 4);
        }
        w.write_ue(head.max_num_ref_frames);
        w.write_bit(u32::from(head.gaps_in_frame_num_value_allowed_flag));
        w.write_ue(head.pic_width_in_mbs_minus1);
        w.write_ue(head.pic_height_in_map_units_minus1);
        w.write_bit(u32::from(head.poc.frame_mbs_only_flag));
        if !head.poc.frame_mbs_only_flag {
            w.write_bit(u32::from(head.mb_adaptive_frame_field_flag));
        }
        w.write_bit(u32::from(head.direct_8x8_inference_flag));
        w.write_bit(u32::from(head.frame_cropping_flag));
        if head.frame_cropping_flag {
            for offset in head.frame_crop_offsets {
                w.write_ue(offset);
            }
        }
        w.write_bit(0); // vui_parameters_present_flag
        w.write_bit(1); // rbsp_stop_one_bit
        w.align_to_byte();
        let mut out = vec![0u8, 0, 0, 1, 0x67];
        out.extend_from_slice(&crate::annexb::add_emulation_prevention(&w.into_bytes()));
        out
    }

    const CODED_1080I: (u32, u32) = (1936, 1088);

    // 1080i High 4:2:2 10-bit, cropped from 1936x1088 to 1920x1080
    fn interlaced_422_head() -> H264SpsHead {
        H264SpsHead {
            profile_idc: 122,
            level_idc: 41,
            seq_parameter_set_id: 3,
            chroma_format_idc: 2,
            bit_depth_luma_minus8: 2,
            bit_depth_chroma_minus8: 2,
            seq_scaling_matrix_present: false,
            poc: crate::poc::SpsPocParams {
                pic_order_cnt_type: 0,
                log2_max_pic_order_cnt_lsb: 6,
                log2_max_frame_num: 5,
                frame_mbs_only_flag: false,
                ..crate::poc::SpsPocParams::default()
            },
            max_num_ref_frames: 4,
            gaps_in_frame_num_value_allowed_flag: true,
            pic_width_in_mbs_minus1: CODED_1080I.0 / 16 - 1,
            pic_height_in_map_units_minus1: CODED_1080I.1 / 32 - 1,
            mb_adaptive_frame_field_flag: true,
            direct_8x8_inference_flag: true,
            frame_cropping_flag: true,
            frame_crop_offsets: [0, 8, 0, 4],
        }
    }

    fn sps_rbsp(annexb_sps: &[u8]) -> Vec<u8> {
        strip_emulation_prevention(&annexb_sps[ANNEX_B_SPS_HEADER_BYTES..])
    }

    // start code and NAL header byte
    const ANNEX_B_SPS_HEADER_BYTES: usize = 5;

    #[test]
    fn sps_head_reads_every_field_as_coded() {
        let coded = interlaced_422_head();
        let rbsp = sps_rbsp(&annexb_sps_from_head(&coded, &[]));
        let (head, _) = parse_h264_sps_head(&rbsp).expect("SPS head parses");
        assert_eq!(head.profile_idc, coded.profile_idc);
        assert_eq!(head.level_idc, coded.level_idc);
        assert_eq!(head.seq_parameter_set_id, coded.seq_parameter_set_id);
        assert_eq!(head.chroma_format_idc, coded.chroma_format_idc);
        assert_eq!(head.bit_depth_luma_minus8, coded.bit_depth_luma_minus8);
        assert_eq!(head.bit_depth_chroma_minus8, coded.bit_depth_chroma_minus8);
        assert_eq!(head.poc, coded.poc);
        assert_eq!(head.max_num_ref_frames, coded.max_num_ref_frames);
        assert!(head.gaps_in_frame_num_value_allowed_flag);
        assert_eq!(head.pic_width_in_mbs_minus1, coded.pic_width_in_mbs_minus1);
        assert_eq!(
            head.pic_height_in_map_units_minus1,
            coded.pic_height_in_map_units_minus1
        );
        assert!(head.mb_adaptive_frame_field_flag);
        assert!(head.direct_8x8_inference_flag);
        assert_eq!(head.frame_crop_offsets, coded.frame_crop_offsets);
    }

    #[test]
    fn interlaced_422_sps_crops_in_chroma_and_field_units() {
        let info = H264Codec::extract_sps_info(&annexb_sps_from_head(&interlaced_422_head(), &[]))
            .expect("SPS parses");
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.poc, Some(interlaced_422_head().poc));
    }

    #[test]
    fn separate_colour_planes_reach_the_poc_fields() {
        // the slice header codes colour_plane_id only under this flag
        let mut head = interlaced_422_head();
        head.profile_idc = 244;
        head.chroma_format_idc = 3;
        head.poc.separate_colour_plane_flag = true;
        let info =
            H264Codec::extract_sps_info(&annexb_sps_from_head(&head, &[])).expect("SPS parses");
        assert_eq!(info.poc, Some(head.poc));
    }

    #[test]
    fn four_four_four_scaling_matrix_crosses_twelve_lists() {
        // 4:4:4 adds four 8x8 chroma lists after the eight every high profile codes
        let mut head = interlaced_422_head();
        head.profile_idc = 244;
        head.chroma_format_idc = 3;
        head.seq_scaling_matrix_present = true;
        head.frame_crop_offsets = [0; 4];
        head.frame_cropping_flag = false;
        let lists = [true; 12];
        let info =
            H264Codec::extract_sps_info(&annexb_sps_from_head(&head, &lists)).expect("SPS parses");
        assert_eq!((info.width, info.height), CODED_1080I);
    }

    #[test]
    fn log2_max_frame_num_is_bounded() {
        let mut head = interlaced_422_head();
        head.poc.log2_max_frame_num = crate::poc::MAX_LOG2_MINUS4 + 4;
        let info = H264Codec::extract_sps_info(&annexb_sps_from_head(&head, &[]))
            .expect("the largest frame_num width parses");
        assert_eq!(info.poc, Some(head.poc.clone()));
        head.poc.log2_max_frame_num += 1;
        assert!(H264Codec::extract_sps_info(&annexb_sps_from_head(&head, &[])).is_none());
    }

    #[test]
    fn only_poc_type_2_presents_in_decode_order() {
        let mut head = interlaced_422_head();
        let info =
            H264Codec::extract_sps_info(&annexb_sps_from_head(&head, &[])).expect("SPS parses");
        assert!(!info.presents_in_decode_order, "POC type 0 may reorder");
        head.poc.pic_order_cnt_type = 2;
        let info =
            H264Codec::extract_sps_info(&annexb_sps_from_head(&head, &[])).expect("SPS parses");
        assert!(info.presents_in_decode_order);
    }
}
