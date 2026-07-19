//! H.265 (HEVC) access-unit parser that refines source-side `Caps`.
//!
//! The H.265 sibling of `h264parse`: it scans each `DataFrame` for an SPS NAL
//! (`nal_unit_type == 33`), recovers the coded picture dimensions, and emits a
//! `CapsChanged` with `Dim::Fixed` values before forwarding the frame. This
//! lets a raw H.265 elementary stream (which advertises `Dim::Any` at
//! negotiation, since the SPS only lands once bytes flow) be restreamed or
//! recorded with concrete geometry.
//!
//! H.265's NAL header is two bytes (type is bits `[1..7]` of the first), and
//! the SPS carries a variable-size `profile_tier_level` before the dimensions;
//! for a single-layer stream (`sps_max_sub_layers_minus1 == 0`) that block is a
//! fixed 96 bits. The Annex-B / AVCC framing, the RBSP de-emulation, and the
//! exp-Golomb bit reader are shared with `h264parse` via the `annexb` module.
//!
//! Framerate is recovered from the VUI `timing_info` (M663): the parse
//! continues past the scaling-list, PCM, short-term-ref-pic-set, and
//! long-term-ref blocks (the RPS parse is the driver-validated one shared with
//! the Vulkan Video decoder) to `vui_time_scale / vui_num_units_in_tick`,
//! best-effort: a stream whose tail cannot be crossed still refines geometry
//! and leaves the rate unrefined.

use alloc::vec::Vec;

use g2g_core::{PropKind, PropertySpec, VideoCodec};

use crate::annexb::{h265_nal_type, next_start_code, strip_emulation_prevention, BitReader};
use crate::nalparse::{NalCodec, NalParse, SpsGeometry};

/// H.265 (HEVC) access-unit parser: `CompressedVideo{H265}` in and out, refining
/// caps from the SPS and (in re-framing mode) re-chunking to one access unit per
/// `DataFrame` with VPS/SPS/PPS re-insertion. The shared parser machinery lives in
/// [`NalParse`]; this file supplies only the H.265-specific hooks.
pub type H265Parse = NalParse<H265Codec>;

/// H.265 codec hooks for [`NalParse`].
#[derive(Debug)]
pub struct H265Codec;

impl NalCodec for H265Codec {
    const CODEC: VideoCodec = VideoCodec::H265;
    const NAME: &'static str = "H.265 parser";
    const DESCRIPTION: &'static str =
        "Parses an H.265 Annex-B stream and refines caps from VPS/SPS/PPS";
    const PROPERTIES: &'static [PropertySpec] = &[PropertySpec::new(
        "config-interval",
        PropKind::Int,
        "VPS/SPS/PPS re-insertion interval in seconds (0 = off, -1 = every IRAP, N = every N s)",
    )
    .with_range("-1", "3600")
    .with_default("0")];
    // VPS (32) then SPS (33) then PPS (34), the H.265 prepend order.
    const PARAM_SET_TYPES: &'static [u8] = &[VPS_NUT, SPS_NUT, PPS_NUT];
    const SPS_TYPE: u8 = SPS_NUT;

    fn nal_type(nal: &[u8]) -> Option<u8> {
        h265_nal_type(nal)
    }

    fn au_starts(data: &[u8]) -> Vec<usize> {
        h265_au_starts(data)
    }

    fn au_is_keyframe(au: &[u8]) -> bool {
        h265_au_is_keyframe(au)
    }

    fn extract_sps_info(au: &[u8]) -> Option<SpsGeometry> {
        extract_sps_info(au)
    }
}

/// H.265 NAL unit type for a sequence parameter set (SPS_NUT).
const SPS_NUT: u8 = 33;
/// H.265 NAL unit types for the video and picture parameter sets.
const VPS_NUT: u8 = 32;
const PPS_NUT: u8 = 34;

/// Walk the NALs of `au` (Annex-B or AVCC, auto-detected), returning the info
/// from the first SPS NAL we can parse. H.265 NAL type is bits `[1..7]` of the
/// first header byte.
fn extract_sps_info(au: &[u8]) -> Option<SpsGeometry> {
    for nal in crate::annexb::nal_units_any(au) {
        if nal.len() < 2 {
            continue;
        }
        let nal_unit_type = (nal[0] >> 1) & 0x3F;
        if nal_unit_type != SPS_NUT {
            continue;
        }
        // Strip the 2-byte NAL header, then de-emulate the RBSP.
        let rbsp = strip_emulation_prevention(&nal[2..]);
        if let Some(info) = parse_sps(&rbsp) {
            return Some(info);
        }
    }
    None
}

/// Start-code offsets in an Annex-B buffer at which a new H.265 access unit
/// begins, per the ISO/IEC 23008-2 access-unit boundary rules. The first NAL
/// opens the first AU. Once a VCL NAL (`nal_unit_type` 0..=31) has been seen in
/// the current AU, the next AU begins at: a VCL NAL whose
/// `first_slice_segment_in_pic_flag` is 1 (the first coded picture slice, the MSB
/// of the slice RBSP after the 2-byte NAL header), an access-unit delimiter (35),
/// or the parameter-set / prefix-SEI NALs that lead a picture (VPS 32, SPS 33,
/// PPS 34, prefix SEI 39). Slices 2..N of a picture carry the flag as 0 and stay
/// in the same AU. The HEVC sibling of `h264parse`'s `h264_au_starts`.
fn h265_au_starts(data: &[u8]) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut seen_vcl = false;
    let mut i = 0;
    while let Some((sc, begin)) = next_start_code(data, i) {
        let nal_type = data.get(begin).map(|b| (b >> 1) & 0x3F).unwrap_or(0);
        let is_vcl = nal_type <= 31;
        let starts_au = if !seen_vcl {
            // Leading NALs of the first AU: only the very first opens it.
            starts.is_empty()
        } else if is_vcl {
            // A new picture's first slice has first_slice_segment_in_pic_flag == 1,
            // the MSB of the slice RBSP (the byte after the 2-byte NAL header).
            data.get(begin + 2).map(|b| b & 0x80 != 0).unwrap_or(false)
        } else {
            // A non-VCL that can only lead the next access unit.
            matches!(nal_type, 32 | 33 | 34 | 35 | 39)
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

/// True if the access unit contains an IRAP (random-access) picture: a keyframe.
/// HEVC IRAP `nal_unit_type`s are 16..=23 (BLA / IDR / CRA / reserved IRAP).
fn h265_au_is_keyframe(au: &[u8]) -> bool {
    crate::annexb::nal_units_any(au).any(|nal| {
        nal.first()
            .map(|b| (16..=23).contains(&((b >> 1) & 0x3F)))
            .unwrap_or(false)
    })
}

/// Parse the SPS RBSP (H.265 7.3.2.2): the cropped picture dimensions from the
/// head, then (best-effort) the framerate from the VUI `timing_info` (M663).
/// `None` on a parse failure before the dimensions resolve.
fn parse_sps(rbsp: &[u8]) -> Option<SpsGeometry> {
    let mut br = BitReader::new(rbsp);
    let _sps_video_parameter_set_id = br.read_bits(4)?;
    let sps_max_sub_layers_minus1 = br.read_bits(3)?;
    let _sps_temporal_id_nesting_flag = br.read_bit()?;
    skip_profile_tier_level(&mut br, sps_max_sub_layers_minus1)?;

    let _sps_seq_parameter_set_id = br.read_ue()?;
    let chroma_format_idc = br.read_ue()?;
    let separate_colour_plane_flag = if chroma_format_idc == 3 {
        br.read_bit()?
    } else {
        0
    };
    let pic_width = br.read_ue()?;
    let pic_height = br.read_ue()?;

    let conformance_window_flag = br.read_bit()?;
    let (left, right, top, bottom) = if conformance_window_flag == 1 {
        (br.read_ue()?, br.read_ue()?, br.read_ue()?, br.read_ue()?)
    } else {
        (0, 0, 0, 0)
    };

    // Crop offsets are in chroma sample units, scaled to luma by SubWidthC /
    // SubHeightC (H.265 7.4.3.2.1). ChromaArrayType 0 (monochrome or 4:4:4 with
    // separate colour planes) and 4:4:4 use 1x1.
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
    // Conformance offsets come from untrusted exp-Golomb; saturate the sums so
    // adversarial values cannot overflow before the subtract.
    let width = pic_width.saturating_sub(left.saturating_add(right).saturating_mul(sub_width_c));
    let height = pic_height.saturating_sub(top.saturating_add(bottom).saturating_mul(sub_height_c));
    // The framerate is best-effort: the VUI sits past the scaling-list / RPS /
    // long-term-ref blocks, and a stream this walk cannot cross still has valid
    // geometry, so a failure downgrades to `None` instead of failing the parse.
    let framerate = parse_vui_framerate(&mut br, sps_max_sub_layers_minus1);
    Some(SpsGeometry {
        width,
        height,
        framerate,
    })
}

/// Continue the SPS walk past the conformance window down to the VUI
/// `timing_info` (M663), returning the framerate as Q16 fixed-point fps
/// (`vui_time_scale / vui_num_units_in_tick`; H.265 ticks are per picture, so
/// there is no H.264-style field factor of 2). `None` when the VUI omits
/// timing or a block on the way ends early / is out of range.
fn parse_vui_framerate(br: &mut BitReader, sps_max_sub_layers_minus1: u32) -> Option<u32> {
    br.read_ue()?; // bit_depth_luma_minus8
    br.read_ue()?; // bit_depth_chroma_minus8
    let log2_max_pic_order_cnt_lsb_minus4 = br.read_ue()?;
    if log2_max_pic_order_cnt_lsb_minus4 > 12 {
        return None;
    }
    // sps_sub_layer_ordering_info_present_flag selects one triple or one per
    // sub-layer.
    let start = if br.read_bit()? == 1 {
        0
    } else {
        sps_max_sub_layers_minus1
    };
    for _ in start..=sps_max_sub_layers_minus1 {
        br.read_ue()?; // sps_max_dec_pic_buffering_minus1
        br.read_ue()?; // sps_max_num_reorder_pics
        br.read_ue()?; // sps_max_latency_increase_plus1
    }
    br.read_ue()?; // log2_min_luma_coding_block_size_minus3
    br.read_ue()?; // log2_diff_max_min_luma_coding_block_size
    br.read_ue()?; // log2_min_luma_transform_block_size_minus2
    br.read_ue()?; // log2_diff_max_min_luma_transform_block_size
    br.read_ue()?; // max_transform_hierarchy_depth_inter
    br.read_ue()?; // max_transform_hierarchy_depth_intra
    if br.read_bit()? == 1 {
        // scaling_list_enabled_flag -> sps_scaling_list_data_present_flag
        if br.read_bit()? == 1 {
            skip_scaling_list_data(br)?;
        }
    }
    br.read_bit()?; // amp_enabled_flag
    br.read_bit()?; // sample_adaptive_offset_enabled_flag
    if br.read_bit()? == 1 {
        // pcm_enabled_flag
        br.read_bits(4)?; // pcm_sample_bit_depth_luma_minus1
        br.read_bits(4)?; // pcm_sample_bit_depth_chroma_minus1
        br.read_ue()?; // log2_min_pcm_luma_coding_block_size_minus3
        br.read_ue()?; // log2_diff_max_min_pcm_luma_coding_block_size
        br.read_bit()?; // pcm_loop_filter_disabled_flag
    }
    let num_short_term_ref_pic_sets = br.read_ue()?;
    if num_short_term_ref_pic_sets > 64 {
        return None;
    }
    let mut sets = Vec::with_capacity(num_short_term_ref_pic_sets as usize);
    for idx in 0..num_short_term_ref_pic_sets as usize {
        sets.push(parse_h265_short_term_rps(br, idx, &sets)?);
    }
    if br.read_bit()? == 1 {
        // long_term_ref_pics_present_flag
        let num_long_term_ref_pics_sps = br.read_ue()?;
        if num_long_term_ref_pics_sps > 32 {
            return None;
        }
        for _ in 0..num_long_term_ref_pics_sps {
            br.skip_bits(log2_max_pic_order_cnt_lsb_minus4 as usize + 4)?; // lt_ref_pic_poc_lsb_sps
            br.read_bit()?; // used_by_curr_pic_lt_sps_flag
        }
    }
    br.read_bit()?; // sps_temporal_mvp_enabled_flag
    br.read_bit()?; // strong_intra_smoothing_enabled_flag
    if br.read_bit()? != 1 {
        return None; // vui_parameters_present_flag
    }

    // vui_parameters() (E.2.1) down to timing_info. The head mirrors H.264's,
    // then HEVC inserts four flags and the default display window.
    if br.read_bit()? == 1 {
        // aspect_ratio_info_present_flag; 255 = Extended_SAR.
        if br.read_bits(8)? == 255 {
            br.read_bits(16)?; // sar_width
            br.read_bits(16)?; // sar_height
        }
    }
    if br.read_bit()? == 1 {
        br.read_bit()?; // overscan_appropriate_flag
    }
    if br.read_bit()? == 1 {
        // video_signal_type_present_flag
        br.read_bits(3)?; // video_format
        br.read_bit()?; // video_full_range_flag
        if br.read_bit()? == 1 {
            br.read_bits(24)?; // colour primaries / transfer / matrix
        }
    }
    if br.read_bit()? == 1 {
        // chroma_loc_info_present_flag
        br.read_ue()?;
        br.read_ue()?;
    }
    br.read_bit()?; // neutral_chroma_indication_flag
    br.read_bit()?; // field_seq_flag
    br.read_bit()?; // frame_field_info_present_flag
    if br.read_bit()? == 1 {
        // default_display_window_flag
        br.read_ue()?;
        br.read_ue()?;
        br.read_ue()?;
        br.read_ue()?;
    }
    if br.read_bit()? != 1 {
        return None; // vui_timing_info_present_flag
    }
    let num_units_in_tick = br.read_bits(32)?;
    let time_scale = br.read_bits(32)?;
    if num_units_in_tick == 0 {
        return None;
    }
    let q16 = ((time_scale as u64) << 16) / (num_units_in_tick as u64);
    u32::try_from(q16).ok()
}

/// Skip `scaling_list_data()` (H.265 7.3.4): 4 size classes x 6 matrices (2 for
/// 32x32), each either predicted (one ue) or explicit (an optional DC delta and
/// `coefNum` signed deltas).
fn skip_scaling_list_data(br: &mut BitReader) -> Option<()> {
    for size_id in 0..4u32 {
        let matrices = if size_id == 3 { 2 } else { 6 };
        for _ in 0..matrices {
            if br.read_bit()? == 0 {
                br.read_ue()?; // scaling_list_pred_matrix_id_delta
            } else {
                let coef_num = 64u32.min(1 << (4 + (size_id << 1)));
                if size_id > 1 {
                    br.read_se()?; // scaling_list_dc_coef_minus8
                }
                for _ in 0..coef_num {
                    br.read_se()?; // scaling_list_delta_coef
                }
            }
        }
    }
    Some(())
}

/// Skip `profile_tier_level(1, max_sub_layers_minus1)` (H.265 7.3.3). The
/// general block is a fixed 96 bits (88-bit profile/tier/constraints + 8-bit
/// level); per-sub-layer blocks follow only when `max_sub_layers_minus1 > 0`.
fn skip_profile_tier_level(br: &mut BitReader, max_sub_layers_minus1: u32) -> Option<()> {
    br.skip_bits(88)?; // general profile/tier + constraint/reserved/inbld
    br.skip_bits(8)?; // general_level_idc

    let mut sub_profile_present = [false; 8];
    let mut sub_level_present = [false; 8];
    for i in 0..max_sub_layers_minus1 as usize {
        sub_profile_present[i] = br.read_bit()? == 1;
        sub_level_present[i] = br.read_bit()? == 1;
    }
    if max_sub_layers_minus1 > 0 {
        for _ in max_sub_layers_minus1..8 {
            br.read_bits(2)?; // reserved_zero_2bits
        }
    }
    for i in 0..max_sub_layers_minus1 as usize {
        if sub_profile_present[i] {
            br.skip_bits(88)?; // sub_layer profile/tier block
        }
        if sub_level_present[i] {
            br.skip_bits(8)?; // sub_layer_level_idc
        }
    }
    Some(())
}

/// A short-term reference-picture set in canonical (explicit) form: the derived
/// `DeltaPocS0/S1` deltas and used-by-current flags, whether the stream coded
/// the set explicitly or by inter-RPS prediction. Storing the derived form lets
/// the Vulkan Video `Std*` mapping always emit the explicit encoding (with
/// `inter_ref_pic_set_prediction_flag == 0`), sidestepping the driver-facing
/// ambiguity of the predicted form; the parser here only needs the pic counts
/// to cross the RPS list on the way to the VUI (M663).
#[derive(Debug, Clone, Default)]
pub struct H265ShortTermRps {
    pub num_negative_pics: u8,
    pub num_positive_pics: u8,
    /// Increasingly negative POC deltas (`DeltaPocS0`), one per negative pic.
    pub delta_poc_s0: [i32; 16],
    /// Increasingly positive POC deltas (`DeltaPocS1`), one per positive pic.
    pub delta_poc_s1: [i32; 16],
    pub used_s0: [bool; 16],
    pub used_s1: [bool; 16],
}

/// Parse the `st_ref_pic_set(stRpsIdx)` at index `idx` (H.265 7.3.7), deriving
/// the canonical explicit form. When the set is coded by inter-RPS prediction it
/// is derived from `prev[idx - 1]` per H.265 7.4.8; the bit reader is always
/// advanced by `NumDeltaPocs[RefRpsIdx] + 1` used/use-delta flags in that branch.
/// Within an SPS `stRpsIdx` is never `num_short_term_ref_pic_sets`, so
/// `delta_idx_minus1` is not present. `None` on truncation or an out-of-range
/// count.
pub(crate) fn parse_h265_short_term_rps(
    br: &mut BitReader,
    idx: usize,
    prev: &[H265ShortTermRps],
) -> Option<H265ShortTermRps> {
    let mut rps = H265ShortTermRps::default();
    let inter_pred = if idx != 0 { br.read_bit()? == 1 } else { false };

    if inter_pred {
        // RefRpsIdx = stRpsIdx - (delta_idx_minus1 + 1); delta_idx_minus1 is not
        // coded in the SPS, so it is 0 and RefRpsIdx = idx - 1.
        let reference = prev.get(idx.checked_sub(1)?)?;
        let delta_rps_sign = br.read_bit()? as i32;
        let abs_delta_rps_minus1 = br.read_ue()? as i32;
        let delta_rps =
            (1 - 2 * delta_rps_sign).checked_mul(abs_delta_rps_minus1.checked_add(1)?)?;
        let num_neg = reference.num_negative_pics as usize;
        let num_pos = reference.num_positive_pics as usize;
        let num_delta_pocs = num_neg + num_pos;

        // used_by_curr_pic_flag[j] / use_delta_flag[j] for j in 0..=NumDeltaPocs.
        let mut used_by_curr = [false; 33];
        let mut use_delta = [true; 33];
        for j in 0..=num_delta_pocs {
            let u = br.read_bit()? == 1;
            used_by_curr[j] = u;
            use_delta[j] = if u { true } else { br.read_bit()? == 1 };
        }

        // Derive the negative pics (S0) of the current set (H.265 7.4.8).
        let mut i = 0usize;
        for j in (0..num_pos).rev() {
            let d_poc = reference.delta_poc_s1[j].checked_add(delta_rps)?;
            if d_poc < 0 && use_delta[num_neg + j] && i < 16 {
                rps.delta_poc_s0[i] = d_poc;
                rps.used_s0[i] = used_by_curr[num_neg + j];
                i += 1;
            }
        }
        if delta_rps < 0 && use_delta[num_delta_pocs] && i < 16 {
            rps.delta_poc_s0[i] = delta_rps;
            rps.used_s0[i] = used_by_curr[num_delta_pocs];
            i += 1;
        }
        for j in 0..num_neg {
            let d_poc = reference.delta_poc_s0[j].checked_add(delta_rps)?;
            if d_poc < 0 && use_delta[j] && i < 16 {
                rps.delta_poc_s0[i] = d_poc;
                rps.used_s0[i] = used_by_curr[j];
                i += 1;
            }
        }
        rps.num_negative_pics = i as u8;

        // Derive the positive pics (S1).
        let mut i = 0usize;
        for j in (0..num_neg).rev() {
            let d_poc = reference.delta_poc_s0[j].checked_add(delta_rps)?;
            if d_poc > 0 && use_delta[j] && i < 16 {
                rps.delta_poc_s1[i] = d_poc;
                rps.used_s1[i] = used_by_curr[j];
                i += 1;
            }
        }
        if delta_rps > 0 && use_delta[num_delta_pocs] && i < 16 {
            rps.delta_poc_s1[i] = delta_rps;
            rps.used_s1[i] = used_by_curr[num_delta_pocs];
            i += 1;
        }
        for j in 0..num_pos {
            let d_poc = reference.delta_poc_s1[j].checked_add(delta_rps)?;
            if d_poc > 0 && use_delta[num_neg + j] && i < 16 {
                rps.delta_poc_s1[i] = d_poc;
                rps.used_s1[i] = used_by_curr[num_neg + j];
                i += 1;
            }
        }
        rps.num_positive_pics = i as u8;
    } else {
        let num_negative_pics = br.read_ue()?;
        let num_positive_pics = br.read_ue()?;
        if num_negative_pics > 16 || num_positive_pics > 16 {
            return None;
        }
        let mut prev_poc = 0i32;
        for k in 0..num_negative_pics as usize {
            let delta_poc_s0_minus1 = br.read_ue()? as i32;
            rps.used_s0[k] = br.read_bit()? == 1;
            let dp = prev_poc.checked_sub(delta_poc_s0_minus1.checked_add(1)?)?;
            rps.delta_poc_s0[k] = dp;
            prev_poc = dp;
        }
        let mut prev_poc = 0i32;
        for k in 0..num_positive_pics as usize {
            let delta_poc_s1_minus1 = br.read_ue()? as i32;
            rps.used_s1[k] = br.read_bit()? == 1;
            let dp = prev_poc.checked_add(delta_poc_s1_minus1.checked_add(1)?)?;
            rps.delta_poc_s1[k] = dp;
            prev_poc = dp;
        }
        rps.num_negative_pics = num_negative_pics as u8;
        rps.num_positive_pics = num_positive_pics as u8;
    }
    Some(rps)
}

/// Fuzzing entry: walk an H.265 access unit for the SPS and parse its geometry
/// (NAL scan, emulation-prevention strip, Exp-Golomb bit reader). Exposed only
/// under `--cfg fuzzing` (cargo-fuzz) so the normal public API is unchanged.
#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = extract_sps_info(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annexb::{nal_units, BitWriter};

    // A malformed HEVC SPS whose short-term RPS carries huge Exp-Golomb POC
    // deltas: the running POC accumulation overflowed i32 (found by cargo-fuzz).
    // It must parse to a rejection, not panic / wrap.
    #[test]
    fn malformed_short_term_rps_does_not_overflow() {
        let sps = [
            0x00, 0x00, 0x01, 0x42, 0x3b, 0x00, 0x01, 0xff, 0xff, 0xff, 0xff, 0xff, 0x20, 0xff,
            0xff, 0xfb, 0xff, 0xe4, 0xff, 0xfb, 0xff, 0x00, 0x8b, 0xf9, 0x0b, 0x00, 0x00, 0x00,
            0xff, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0x07, 0x03, 0x07, 0x07,
            0x07, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfb, 0x00, 0x07, 0x07,
        ];
        let _ = extract_sps_info(&sps);
    }
    use alloc::boxed::Box;
    use alloc::vec;
    use core::future::Future;
    use core::pin::Pin;
    use g2g_core::{
        AsyncElement, Caps, CapsConstraint, Dim, G2gError, OutputSink, PipelinePacket, Rate,
    };

    /// Build an Annex-B H.265 SPS for `pic_w` x `pic_h` luma samples at
    /// `chroma_format_idc`, optionally with a conformance window
    /// `(left, right, top, bottom)`. The profile_tier_level is written as 96
    /// zero bits (its content is skipped); the RBSP is emulation-prevented like
    /// a real encoder's output so the parser's de-emulation round-trips it.
    fn build_annexb_sps(
        pic_w: u32,
        pic_h: u32,
        chroma_format_idc: u32,
        conf: Option<(u32, u32, u32, u32)>,
    ) -> Vec<u8> {
        finish_sps(sps_head(pic_w, pic_h, chroma_format_idc, conf))
    }

    /// The SPS fields up to the conformance window, as a bit writer the caller
    /// can extend (the M663 VUI variant) before wrapping into a NAL.
    fn sps_head(
        pic_w: u32,
        pic_h: u32,
        chroma_format_idc: u32,
        conf: Option<(u32, u32, u32, u32)>,
    ) -> BitWriter {
        let mut w = BitWriter::default();
        w.write_bits(0, 4); // sps_video_parameter_set_id
        w.write_bits(0, 3); // sps_max_sub_layers_minus1
        w.write_bit(1); // sps_temporal_id_nesting_flag
        for _ in 0..96 {
            w.write_bit(0); // profile_tier_level (single layer = 96 bits)
        }
        w.write_ue(0); // sps_seq_parameter_set_id
        w.write_ue(chroma_format_idc);
        if chroma_format_idc == 3 {
            w.write_bit(0); // separate_colour_plane_flag
        }
        w.write_ue(pic_w);
        w.write_ue(pic_h);
        match conf {
            Some((l, r, t, b)) => {
                w.write_bit(1); // conformance_window_flag
                w.write_ue(l);
                w.write_ue(r);
                w.write_ue(t);
                w.write_ue(b);
            }
            None => w.write_bit(0),
        }
        w
    }

    /// Close the RBSP (stop bit + alignment), emulation-prevent it, and wrap it
    /// in a start code + SPS NAL header.
    fn finish_sps(mut w: BitWriter) -> Vec<u8> {
        w.write_bit(1); // rbsp_stop_one_bit
        w.align_to_byte();
        let rbsp = w.into_bytes();
        let ebsp = add_emulation_prevention(&rbsp);

        // 00 00 00 01 | NAL header (type 33, layer 0, tid+1 = 1) | EBSP
        let mut out = vec![0u8, 0, 0, 1, 0x42, 0x01];
        out.extend_from_slice(&ebsp);
        out
    }

    /// [`build_annexb_sps`] continued through the whole SPS tail to a VUI with
    /// `timing_info` (M663): zero/absent optional blocks, then
    /// `vui_num_units_in_tick` / `vui_time_scale`.
    fn build_annexb_sps_with_vui(
        pic_w: u32,
        pic_h: u32,
        num_units_in_tick: u32,
        time_scale: u32,
    ) -> Vec<u8> {
        let mut w = sps_head(pic_w, pic_h, 1, None);
        w.write_ue(0); // bit_depth_luma_minus8
        w.write_ue(0); // bit_depth_chroma_minus8
        w.write_ue(0); // log2_max_pic_order_cnt_lsb_minus4
        w.write_bit(1); // sps_sub_layer_ordering_info_present_flag
        w.write_ue(0); // sps_max_dec_pic_buffering_minus1
        w.write_ue(0); // sps_max_num_reorder_pics
        w.write_ue(0); // sps_max_latency_increase_plus1
        w.write_ue(0); // log2_min_luma_coding_block_size_minus3
        w.write_ue(0); // log2_diff_max_min_luma_coding_block_size
        w.write_ue(0); // log2_min_luma_transform_block_size_minus2
        w.write_ue(0); // log2_diff_max_min_luma_transform_block_size
        w.write_ue(0); // max_transform_hierarchy_depth_inter
        w.write_ue(0); // max_transform_hierarchy_depth_intra
        w.write_bit(0); // scaling_list_enabled_flag
        w.write_bit(0); // amp_enabled_flag
        w.write_bit(0); // sample_adaptive_offset_enabled_flag
        w.write_bit(0); // pcm_enabled_flag
        w.write_ue(0); // num_short_term_ref_pic_sets
        w.write_bit(0); // long_term_ref_pics_present_flag
        w.write_bit(0); // sps_temporal_mvp_enabled_flag
        w.write_bit(0); // strong_intra_smoothing_enabled_flag
        w.write_bit(1); // vui_parameters_present_flag
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        w.write_bit(0); // video_signal_type_present_flag
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(0); // neutral_chroma_indication_flag
        w.write_bit(0); // field_seq_flag
        w.write_bit(0); // frame_field_info_present_flag
        w.write_bit(0); // default_display_window_flag
        w.write_bit(1); // vui_timing_info_present_flag
        w.write_bits(num_units_in_tick, 32);
        w.write_bits(time_scale, 32);
        finish_sps(w)
    }

    /// Inverse of `annexb::strip_emulation_prevention`: insert `0x03` after each
    /// `00 00` run preceding a byte <= 0x03.
    fn add_emulation_prevention(rbsp: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(rbsp.len());
        let mut zeros = 0usize;
        for &b in rbsp {
            if zeros >= 2 && b <= 0x03 {
                out.push(0x03);
                zeros = 0;
            }
            out.push(b);
            zeros = if b == 0 { zeros + 1 } else { 0 };
        }
        out
    }

    #[test]
    fn recovers_dimensions_from_sps() {
        let stream = build_annexb_sps(1920, 1080, 1, None);
        let info = extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
        // The RBSP ends at the conformance window, so the VUI walk truncates:
        // geometry still resolves, the rate stays unrefined (best-effort M663).
        assert_eq!(info.framerate, None);
    }

    #[test]
    fn recovers_framerate_from_vui_timing_info() {
        // 25 fps: one tick per picture at a 25 Hz time scale (no H.264-style
        // field doubling in HEVC).
        let stream = build_annexb_sps_with_vui(1920, 1080, 1, 25);
        let info = extract_sps_info(&stream).expect("SPS with VUI must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.framerate, Some(25 << 16), "Q16 fps from timing_info");

        // 29.97 fps (30000/1001): the Q16 value is the truncated division.
        let ntsc = build_annexb_sps_with_vui(720, 480, 1001, 30_000);
        let info = extract_sps_info(&ntsc).expect("NTSC SPS must parse");
        assert_eq!(info.framerate, Some(((30_000u64 << 16) / 1001) as u32));
    }

    #[test]
    fn applies_conformance_window_cropping() {
        // 1920x1088 coded, 4:2:0 (SubHeightC = 2), crop 4 chroma rows off the
        // bottom -> 1088 - 2*4 = 1080.
        let stream = build_annexb_sps(1920, 1088, 1, Some((0, 0, 0, 4)));
        let info = extract_sps_info(&stream).expect("SPS with conf window must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
    }

    #[test]
    fn saturates_adversarial_conformance_offsets() {
        // Huge conformance offsets must saturate, not overflow-panic on the sum.
        let huge = 3_000_000_000u32;
        let stream = build_annexb_sps(1920, 1080, 1, Some((huge, huge, huge, huge)));
        let info = extract_sps_info(&stream).expect("parses without overflow");
        assert_eq!(
            (info.width, info.height),
            (0, 0),
            "offsets clamp dims to zero"
        );
    }

    #[test]
    fn parses_an_avcc_framed_sps() {
        // Re-frame the SPS NAL as length-prefixed (HVCC-style) and confirm the
        // dimensions still resolve.
        let annexb = build_annexb_sps(1280, 720, 1, None);
        let nal = &annexb[4..]; // drop the 00 00 00 01 start code
        let mut hvcc = (nal.len() as u32).to_be_bytes().to_vec();
        hvcc.extend_from_slice(nal);
        let info = extract_sps_info(&hvcc).expect("length-prefixed SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
    }

    #[test]
    fn ignores_non_sps_nals() {
        // A TRAIL_R slice NAL (type 1 -> first byte 0x02) carries no SPS.
        let stream = [0u8, 0, 0, 1, 0x02, 0x01, 0xAA, 0xBB];
        assert!(extract_sps_info(&stream).is_none());
    }

    #[test]
    fn returns_none_on_empty_input() {
        assert!(extract_sps_info(&[]).is_none());
    }

    #[test]
    fn skip_profile_tier_level_advances_96_bits_for_single_layer() {
        // 12 bytes of PTL then a ue(0) = single '1' bit. After the skip the
        // reader must sit exactly on that bit.
        let mut w = BitWriter::default();
        for _ in 0..96 {
            w.write_bit(0);
        }
        w.write_bit(1); // a marker the reader should land on (ue value 0)
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        skip_profile_tier_level(&mut br, 0).expect("skips the fixed 96-bit block");
        assert_eq!(br.read_ue(), Some(0), "reader landed just past the PTL");
    }

    // -- Element-level tests (drive H265Parse::process directly) -----------

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

    fn h265_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = H265Parse::new();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let frame = frame_with_bytes(0, build_annexb_sps(1920, 1080, 1, None));
        parse
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        assert_eq!(sink.packets.len(), 2, "expected CapsChanged then DataFrame");
        match &sink.packets[0] {
            PipelinePacket::CapsChanged(Caps::CompressedVideo {
                codec: VideoCodec::H265,
                width,
                height,
                ..
            }) => {
                assert_eq!(*width, Dim::Fixed(1920));
                assert_eq!(*height, Dim::Fixed(1080));
            }
            other => panic!("expected H.265 CapsChanged first, got {other:?}"),
        }
        assert!(matches!(sink.packets[1], PipelinePacket::DataFrame(_)));
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn does_not_re_emit_caps_when_unchanged() {
        let mut parse = H265Parse::new();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        for seq in 0..3 {
            let frame = frame_with_bytes(seq, build_annexb_sps(1280, 720, 1, None));
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
        assert_eq!(caps_count, 1, "CapsChanged fires once for identical SPS");
        assert_eq!(parse.caps_changes_emitted(), 1);
    }

    #[tokio::test]
    async fn re_emits_caps_on_resolution_change() {
        let mut parse = H265Parse::new();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(
                    0,
                    build_annexb_sps(1280, 720, 1, None),
                )),
                &mut sink,
            )
            .await
            .unwrap();
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(
                    1,
                    build_annexb_sps(1920, 1080, 1, None),
                )),
                &mut sink,
            )
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
    async fn rejects_non_h265_caps_in_intercept() {
        let parse = H265Parse::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(parse.intercept_caps(&h264), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn caps_constraint_is_identity_h265_any() {
        let parse = H265Parse::new();
        let c = parse.caps_constraint_as_transform();
        match c {
            CapsConstraint::Identity(set) => {
                assert_eq!(
                    set.alternatives(),
                    &[Caps::CompressedVideo {
                        codec: VideoCodec::H265,
                        width: Dim::Any,
                        height: Dim::Any,
                        framerate: Rate::Any,
                    }]
                );
            }
            _ => panic!("expected Identity"),
        }
    }

    // -- Re-framing (M425) ------------------------------------------------

    /// One Annex-B H.265 VCL NAL (TRAIL_R, type 1). `first` sets
    /// `first_slice_segment_in_pic_flag` (the MSB of the slice RBSP, the byte after
    /// the 2-byte NAL header), so `first == true` opens a new coded picture.
    fn annexb_vcl_h265(first: bool, tag: u8) -> Vec<u8> {
        // start code + NAL header: type 1 (TRAIL_R) = (1 << 1) = 0x02, layer 0;
        // second header byte 0x01 = temporal_id_plus1.
        let mut v = vec![0u8, 0, 0, 1, 0x02, 0x01];
        v.push(if first { 0x80 } else { 0x40 }); // slice RBSP: MSB = first_slice flag
        v.extend_from_slice(&[0xAA, 0xBB, tag, 0x11]);
        v
    }

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
        // Picture A: two slices (first then a continuation). Picture B: one slice.
        // Two access units, not three.
        let mut stream = annexb_vcl_h265(true, 1);
        stream.extend_from_slice(&annexb_vcl_h265(false, 2)); // same picture
        let b_off = stream.len();
        stream.extend_from_slice(&annexb_vcl_h265(true, 3)); // new picture
        let starts = h265_au_starts(&stream);
        assert_eq!(
            starts,
            vec![0, b_off],
            "two access units: A(2 slices) then B"
        );
    }

    #[tokio::test]
    async fn reframing_splits_two_access_units_in_one_buffer() {
        let mut parse = H265Parse::reframing();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let au0 = annexb_vcl_h265(true, 1);
        let au1 = annexb_vcl_h265(true, 2);
        let mut buf = au0.clone();
        buf.extend_from_slice(&au1);
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, buf)),
                &mut sink,
            )
            .await
            .unwrap();
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
        // One Annex-B access unit delivered as two buffers (e.g. one MPEG-TS PES
        // carrying the tail), the second with no leading start code. Latching the
        // framing keeps it Annex-B instead of misreading the tail as length-prefixed.
        let mut parse = H265Parse::reframing();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        let au = annexb_vcl_h265(true, 7);
        let split = 7; // mid-NAL: past start code + 2-byte header + first RBSP byte
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

    #[tokio::test]
    async fn reframing_stamps_keyframe_on_irap() {
        // An IDR_W_RADL (type 19) access unit is a keyframe; a TRAIL_R (type 1) is not.
        let mut parse = H265Parse::reframing();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = RecordingSink::default();

        // IDR: NAL header first byte = (19 << 1) = 0x26, first-slice flag set.
        let mut idr = vec![0u8, 0, 0, 1, 0x26, 0x01, 0x80, 0xAA];
        let trail = annexb_vcl_h265(true, 2);
        idr.extend_from_slice(&trail);
        parse
            .process(
                PipelinePacket::DataFrame(frame_with_bytes(0, idr)),
                &mut sink,
            )
            .await
            .unwrap();
        parse.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        let kf: Vec<bool> = sink
            .packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f.timing.keyframe),
                _ => None,
            })
            .collect();
        assert_eq!(
            kf,
            vec![true, false],
            "IDR AU is a keyframe, TRAIL_R is not"
        );
    }

    #[test]
    fn config_interval_reinserts_vps_sps_pps_on_irap() {
        let mut p = H265Parse::reframing().with_config_interval(-1);
        // An IRAP AU with VPS (32) + SPS (33) + PPS (34): cached, returned as-is.
        // H.265 NAL header type is bits 1..=6 of the first byte: 32<<1=0x40 etc.
        let mut au1 = vec![0, 0, 0, 1, 0x40, 0x01]; // VPS
        au1.extend_from_slice(&[0, 0, 0, 1, 0x42, 0x01]); // SPS
        au1.extend_from_slice(&[0, 0, 0, 1, 0x44, 0x01]); // PPS
        au1.extend_from_slice(&[0, 0, 0, 1, 0x26, 0x01]); // IDR_W_RADL (19) slice
        let out1 = p.apply_config_interval(au1.clone(), 0, true);
        assert_eq!(
            out1, au1,
            "an IRAP that already carries parameter sets is untouched"
        );
        // A later IRAP with no parameter sets gets VPS/SPS/PPS prepended.
        let au2 = vec![0, 0, 0, 1, 0x26, 0x01];
        let out2 = p.apply_config_interval(au2.clone(), 90_000, true);
        assert!(
            nal_units(&out2).any(|n| h265_nal_type(n) == Some(VPS_NUT)),
            "result carries a VPS"
        );
        assert!(
            nal_units(&out2).any(|n| h265_nal_type(n) == Some(SPS_NUT)),
            "result carries an SPS"
        );
        assert!(
            nal_units(&out2).any(|n| h265_nal_type(n) == Some(PPS_NUT)),
            "result carries a PPS"
        );
        assert!(
            out2.ends_with(&au2),
            "the original AU is preserved at the tail"
        );
    }
}
