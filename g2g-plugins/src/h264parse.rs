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
//! caps carry `Rate::Any` when the VUI omits it.

use alloc::vec::Vec;

use g2g_core::{PropKind, PropertySpec, VideoCodec};

use crate::annexb::{h264_nal_type, next_start_code, strip_emulation_prevention, BitReader};
use crate::nalparse::{NalCodec, NalParse, SpsGeometry};

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

    fn au_starts(data: &[u8]) -> Vec<usize> {
        h264_au_starts(data)
    }

    fn au_is_keyframe(au: &[u8]) -> bool {
        crate::h264util::h264_au_is_keyframe(au)
    }

    fn extract_sps_info(au: &[u8]) -> Option<SpsGeometry> {
        extract_sps_info(au)
    }
}

/// Start-code offsets in an Annex-B buffer at which a new H.264 access unit
/// begins, per the ISO/IEC 14496-10 access-unit boundary rules. The first NAL
/// always opens the first AU. Once a VCL NAL (types 1..=5) has been seen in the
/// current AU, the next AU begins at: an access-unit delimiter (9), the parameter
/// sets / SEI / prefix NALs that lead a picture (6, 7, 8, 14, 15, 18), or the
/// first VCL NAL of a new coded picture (`first_mb_in_slice == 0`, i.e. the first
/// RBSP bit is 1). Slices 2..N of one picture carry `first_mb_in_slice > 0` and
/// stay in the same AU. This is the framing GStreamer's `h264parse` applies; a
/// decoder needs it because it parses one packet as one coded picture.
fn h264_au_starts(data: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut seen_vcl = false;
    let mut i = 0;
    while let Some((sc, begin)) = next_start_code(data, i) {
        let nal_type = data.get(begin).map(|b| b & 0x1F).unwrap_or(0);
        let is_vcl = (1..=5).contains(&nal_type);
        let starts_au = if !seen_vcl {
            // Leading NALs of the first AU: only the very first opens an AU; the
            // rest (more parameter sets, the first VCL) join it.
            starts.is_empty()
        } else if is_vcl {
            // A new picture's first slice has first_mb_in_slice == 0, encoded as a
            // leading 1 bit in the slice RBSP (ue(v) of 0).
            data.get(begin + 1).map(|b| b & 0x80 != 0).unwrap_or(false)
        } else {
            // A non-VCL that can only lead the next access unit.
            matches!(nal_type, 6 | 7 | 8 | 9 | 14 | 15 | 18)
        };
        if starts_au {
            starts.push(sc);
            seen_vcl = false;
        }
        if is_vcl {
            seen_vcl = true;
        }
        i = begin;
    }
    starts
}

/// Walk the NALs of `au` (Annex-B or AVCC, auto-detected), returning the info
/// from the first SPS NAL (nal_unit_type == 7) we can fully parse.
fn extract_sps_info(au: &[u8]) -> Option<SpsGeometry> {
    for nal in crate::annexb::nal_units_any(au) {
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

/// Parse the SPS RBSP (post NAL-header byte) for the coded picture dimensions
/// and, when the VUI carries `timing_info`, the framerate. Returns `None` on a
/// parse failure up to the dimensions; a failure past them leaves only the
/// framerate unknown.
fn parse_sps(rbsp: &[u8]) -> Option<SpsGeometry> {
    if rbsp.len() < 3 {
        return None;
    }
    let profile_idc = rbsp[0];
    // rbsp[1] = constraint_set flags + reserved zero bits (skipped)
    // rbsp[2] = level_idc (skipped)
    let mut br = BitReader::new(&rbsp[3..]);

    let _sps_id = br.read_ue()?;

    // 4:2:0 unless a high profile signals a different chroma format.
    let mut chroma_format_idc = 1u32;
    let mut separate_colour_plane_flag = 0u32;
    // Profiles that include chroma/scaling/etc. extra header fields.
    if matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        chroma_format_idc = br.read_ue()?;
        if chroma_format_idc == 3 {
            separate_colour_plane_flag = br.read_bit()?;
        }
        let _bit_depth_luma_minus8 = br.read_ue()?;
        let _bit_depth_chroma_minus8 = br.read_ue()?;
        let _qpprime_y_zero_transform_bypass_flag = br.read_bit()?;
        let seq_scaling_matrix_present_flag = br.read_bit()?;
        if seq_scaling_matrix_present_flag == 1 {
            // We don't decode the (optional) scaling lists. M6 test fixtures
            // never set this flag; live streams that do will leave dims
            // unknown until M7 lands a full SPS parser.
            return None;
        }
    }

    let _log2_max_frame_num_minus4 = br.read_ue()?;
    let pic_order_cnt_type = br.read_ue()?;
    if pic_order_cnt_type == 0 {
        let _log2_max_pic_order_cnt_lsb_minus4 = br.read_ue()?;
    } else if pic_order_cnt_type == 1 {
        let _delta_pic_order_always_zero_flag = br.read_bit()?;
        let _offset_for_non_ref_pic = br.read_se()?;
        let _offset_for_top_to_bottom_field = br.read_se()?;
        let num_ref_frames_in_pic_order_cnt_cycle = br.read_ue()?;
        for _ in 0..num_ref_frames_in_pic_order_cnt_cycle {
            let _offset = br.read_se()?;
        }
    }
    let _max_num_ref_frames = br.read_ue()?;
    let _gaps_in_frame_num_value_allowed_flag = br.read_bit()?;

    let pic_width_in_mbs_minus1 = br.read_ue()?;
    let pic_height_in_map_units_minus1 = br.read_ue()?;
    let frame_mbs_only_flag = br.read_bit()?;
    if frame_mbs_only_flag == 0 {
        let _mb_adaptive_frame_field_flag = br.read_bit()?;
    }
    let _direct_8x8_inference_flag = br.read_bit()?;

    let frame_cropping_flag = br.read_bit()?;
    let (crop_left, crop_right, crop_top, crop_bottom) = if frame_cropping_flag == 1 {
        let l = br.read_ue()?;
        let r = br.read_ue()?;
        let t = br.read_ue()?;
        let b = br.read_ue()?;
        (l, r, t, b)
    } else {
        (0, 0, 0, 0)
    };

    // Crop units in luma samples (H.264 7.4.2.1.1). ChromaArrayType 0
    // (monochrome, or 4:4:4 with separate colour planes) crops 1 x (2-fmof);
    // otherwise SubWidthC x SubHeightC*(2-fmof).
    let chroma_array_type = if separate_colour_plane_flag == 1 {
        0
    } else {
        chroma_format_idc
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

    let width = pic_width_in_mbs_minus1
        .saturating_add(1)
        .saturating_mul(16)
        .saturating_sub(crop_x);
    let height = (2 - frame_mbs_only_flag)
        .saturating_mul(pic_height_in_map_units_minus1.saturating_add(1))
        .saturating_mul(16)
        .saturating_sub(crop_y);

    // vui_parameters_present_flag follows frame cropping. Read it without `?`
    // so a stream truncated right here still yields the dimensions.
    let framerate = match br.read_bit() {
        Some(1) => parse_vui_framerate(&mut br),
        _ => None,
    };

    Some(SpsGeometry {
        width,
        height,
        framerate,
    })
}

/// Walk the VUI parameters up to `timing_info`, returning the framerate as Q16
/// fixed-point (`time_scale / (2 * num_units_in_tick)`). `None` if the VUI
/// omits timing or ends early; the caller already has the dimensions.
fn parse_vui_framerate(br: &mut BitReader) -> Option<u32> {
    // aspect_ratio_info_present_flag
    if br.read_bit()? == 1 {
        let aspect_ratio_idc = br.read_bits(8)?;
        // 255 = Extended_SAR: explicit sar_width / sar_height follow.
        if aspect_ratio_idc == 255 {
            br.read_bits(16)?; // sar_width
            br.read_bits(16)?; // sar_height
        }
    }
    // overscan_info_present_flag
    if br.read_bit()? == 1 {
        br.read_bit()?; // overscan_appropriate_flag
    }
    // video_signal_type_present_flag
    if br.read_bit()? == 1 {
        br.read_bits(3)?; // video_format
        br.read_bit()?; // video_full_range_flag
        if br.read_bit()? == 1 {
            // colour_description_present_flag
            br.read_bits(8)?; // colour_primaries
            br.read_bits(8)?; // transfer_characteristics
            br.read_bits(8)?; // matrix_coefficients
        }
    }
    // chroma_loc_info_present_flag
    if br.read_bit()? == 1 {
        br.read_ue()?; // chroma_sample_loc_type_top_field
        br.read_ue()?; // chroma_sample_loc_type_bottom_field
    }
    // timing_info_present_flag
    if br.read_bit()? == 1 {
        let num_units_in_tick = br.read_bits(32)?;
        let time_scale = br.read_bits(32)?;
        let _fixed_frame_rate_flag = br.read_bit()?;
        if num_units_in_tick == 0 {
            return None;
        }
        // fps = time_scale / (2 * num_units_in_tick); carry to Q16 in u64.
        let q16 = ((time_scale as u64) << 16) / (2 * num_units_in_tick as u64);
        return u32::try_from(q16).ok();
    }
    None
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
/// too-low value is exactly what drops the leading pictures. `None` if there is
/// no parseable SPS.
///
/// Consumed by the libavcodec decoder (`ffmpegdec`), so gated on that feature.
#[cfg(feature = "ffmpeg")]
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
/// Mirrors the field walk of [`parse_sps`] up to the frame dimensions.
#[cfg(feature = "ffmpeg")]
fn parse_sps_reorder(rbsp: &[u8]) -> Option<u8> {
    if rbsp.len() < 3 {
        return None;
    }
    let profile_idc = rbsp[0];
    let level_idc = rbsp[2];
    // The Baseline profile has no B-frames and no picture reordering, so it never
    // needs output buffering. (constraint_set1 means "also Main-conformant", not
    // "baseline", so it is not a no-reorder signal.)
    if profile_idc == 66 {
        return Some(0);
    }
    let mut br = BitReader::new(&rbsp[3..]);

    let _sps_id = br.read_ue()?;
    if matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    ) {
        let chroma_format_idc = br.read_ue()?;
        if chroma_format_idc == 3 {
            br.read_bit()?; // separate_colour_plane_flag
        }
        br.read_ue()?; // bit_depth_luma_minus8
        br.read_ue()?; // bit_depth_chroma_minus8
        br.read_bit()?; // qpprime_y_zero_transform_bypass_flag
        if br.read_bit()? == 1 {
            return None; // scaling lists not walked; leave reorder unknown
        }
    }

    br.read_ue()?; // log2_max_frame_num_minus4
    let pic_order_cnt_type = br.read_ue()?;
    if pic_order_cnt_type == 0 {
        br.read_ue()?; // log2_max_pic_order_cnt_lsb_minus4
    } else if pic_order_cnt_type == 1 {
        br.read_bit()?; // delta_pic_order_always_zero_flag
        br.read_se()?; // offset_for_non_ref_pic
        br.read_se()?; // offset_for_top_to_bottom_field
        let cycle = br.read_ue()?;
        for _ in 0..cycle {
            br.read_se()?;
        }
    }
    br.read_ue()?; // max_num_ref_frames
    br.read_bit()?; // gaps_in_frame_num_value_allowed_flag

    let pic_width_in_mbs = br.read_ue()?.saturating_add(1);
    let pic_height_in_map_units = br.read_ue()?.saturating_add(1);
    let frame_mbs_only_flag = br.read_bit()?;

    let frame_height_in_mbs =
        (2u32.saturating_sub(frame_mbs_only_flag)).saturating_mul(pic_height_in_map_units);
    let pic_size_in_mbs = pic_width_in_mbs.saturating_mul(frame_height_in_mbs).max(1);
    let max_dpb_frames = (max_dpb_mbs(level_idc) / pic_size_in_mbs).min(16);
    Some(max_dpb_frames as u8)
}

/// `MaxDpbMbs` for an H.264 level (Table A-1), in macroblocks. The reorder /
/// DPB frame bound is this divided by the frame size in macroblocks.
#[cfg(feature = "ffmpeg")]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annexb::{nal_units, BitWriter};
    use alloc::boxed::Box;
    use alloc::vec;
    use core::future::Future;
    use g2g_core::{
        AsyncElement, Caps, CapsConstraint, Dim, G2gError, OutputSink, PipelinePacket, Rate,
    };

    /// Build a minimal baseline-profile SPS RBSP for `width` x `height`
    /// (both multiples of 16), then frame it in Annex-B. Returns the
    /// full byte stream including a 4-byte start code and NAL header.
    fn build_test_annexb_sps(width: u32, height: u32) -> Vec<u8> {
        assert!(width % 16 == 0 && height % 16 == 0);
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
    #[cfg(feature = "ffmpeg")]
    fn annexb_sps(profile_idc: u8, level_idc: u8, rbsp: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8, 0, 0, 1, 0x67, profile_idc, 0, level_idc];
        out.extend_from_slice(rbsp);
        out
    }

    /// Minimal SPS body (non-high-profile branch) at `width` x `height`.
    #[cfg(feature = "ffmpeg")]
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

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn sps_reorder_frames_uses_level_dpb_for_reordering_profile() {
        // Main profile, level 4.0, 1920x1088 (120x68 = 8160 MBs).
        // MaxDpbMbs(40) = 32768; 32768 / 8160 = 4 (capped at 16).
        let au = annexb_sps(77, 40, &sps_body(1920, 1088));
        assert_eq!(sps_reorder_frames(&au), Some(4));
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn sps_reorder_frames_zero_for_baseline() {
        // Baseline (66) has no B-frames / reordering regardless of level or size.
        let au = annexb_sps(66, 40, &sps_body(1920, 1088));
        assert_eq!(sps_reorder_frames(&au), Some(0));
    }

    #[cfg(feature = "ffmpeg")]
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
        let info = extract_sps_info(&au).expect("parses without overflow");
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
        assert!(width % 16 == 0 && height % 16 == 0);
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

    #[test]
    fn round_trips_a_1280x720_sps() {
        let stream = build_test_annexb_sps(1280, 720);
        let info = extract_sps_info(&stream).expect("SPS must parse");
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
        let info = extract_sps_info(&stream).expect("SPS must parse");
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
        let info = extract_sps_info(&avcc).expect("AVCC SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
    }

    #[test]
    fn recovers_framerate_from_vui_timing() {
        // num_units_in_tick = 1, time_scale = 60 -> 30 fps.
        let stream = build_annexb_sps_with_timing(1280, 720, 1, 60);
        let info = extract_sps_info(&stream).expect("SPS with VUI must parse");
        assert_eq!((info.width, info.height), (1280, 720));
        assert_eq!(info.framerate, Some(30 << 16), "30 fps in Q16");
    }

    #[test]
    fn ignores_non_sps_nals() {
        // A stream containing only a slice NAL (type 5 = IDR) returns None.
        let stream = [0u8, 0, 0, 1, 0x65, 0xAA, 0xBB, 0xCC];
        assert!(extract_sps_info(&stream).is_none());
    }

    #[test]
    fn returns_none_on_empty_input() {
        assert!(extract_sps_info(&[]).is_none());
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
    fn parse_vui_framerate_reads_timing_info() {
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
        let fps = parse_vui_framerate(&mut br).expect("timing info present");
        let expected = ((60000u64 << 16) / (2 * 1001)) as u32;
        assert_eq!(fps, expected);
        assert_eq!(fps >> 16, 29, "~29.97 fps");
    }

    #[test]
    fn parse_vui_framerate_absent_timing_is_none() {
        let mut w = BitWriter::default();
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(0); // timing_info_present_flag = 0
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        assert_eq!(parse_vui_framerate(&mut br), None);
    }

    // -- Element-level tests (drive H264Parse::process directly) -----------

    use core::pin::Pin;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, MemoryDomain, PushOutcome};

    #[derive(Default)]
    struct RecordingSink {
        packets: Vec<PipelinePacket>,
    }

    impl OutputSink for RecordingSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                self.packets.push(packet);
                Ok(PushOutcome::Accepted)
            })
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

    fn h264_parse_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

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

    #[tokio::test]
    async fn emits_caps_from_an_avcc_access_unit() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

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
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = H264Parse::new();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

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
        let mut sink = RecordingSink::default();

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
    fn data_payloads(sink: &RecordingSink) -> Vec<Vec<u8>> {
        sink.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => match &f.domain {
                    MemoryDomain::System(s) => Some(s.as_slice().to_vec()),
                    _ => None,
                },
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
        let starts = h264_au_starts(&stream);
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
        let mut sink = RecordingSink::default();

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
    async fn reframing_reassembles_an_au_split_across_buffers() {
        // The regression that made HLS video garbage: one Annex-B access unit
        // delivered as two buffers (one MPEG-TS PES carrying the tail), the second
        // with no leading start code. A per-buffer framing guess would misread the
        // tail as AVCC and corrupt it; latching the framing keeps it Annex-B.
        let mut parse = H264Parse::reframing();
        parse.configure_pipeline(&h264_parse_caps()).unwrap();
        let mut sink = RecordingSink::default();

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
}
