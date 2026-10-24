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
//! Framerate is recovered from the VUI `timing_info`: the parse
//! continues past the scaling-list, PCM, short-term-ref-pic-set, and
//! long-term-ref blocks (the RPS parse is the driver-validated one shared with
//! the Vulkan Video decoder) to `vui_time_scale / vui_num_units_in_tick`,
//! best-effort: a stream whose tail cannot be crossed still refines geometry
//! and leaves the rate unrefined.

use alloc::vec::Vec;

// Only `h265_codec_string` (WebCodecs / wasm) needs an owned String.
#[cfg(any(test, all(target_arch = "wasm32", feature = "web-codecs")))]
use alloc::string::String;

use g2g_core::caps::framerate_q16_from_fraction;
use g2g_core::{Colorimetry, PropKind, PropertySpec, VideoCodec};

use crate::access_unit::StartCodeRole;
use crate::annexb::{h265_nal_type, strip_emulation_prevention, BitReader};
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

    /// A new picture's first slice has `first_slice_segment_in_pic_flag == 1`,
    /// the MSB of the slice RBSP after the 2-byte NAL header. Slices 2..N carry
    /// it as 0 and stay in the picture's access unit. Ahead of a picture, an
    /// access-unit delimiter (35) and the parameter-set / prefix-SEI NALs that
    /// lead a picture (VPS 32, SPS 33, PPS 34, prefix SEI 39) open the next
    /// access unit (ISO/IEC 23008-2 7.4.2.4.4).
    fn unit_role(unit: &[u8]) -> Option<StartCodeRole> {
        let nal_type = h265_nal_type(unit)?;
        if nal_type <= 31 {
            let first_slice_of_picture = unit.get(2)? & 0x80 != 0;
            return Some(if first_slice_of_picture {
                StartCodeRole::Picture
            } else {
                StartCodeRole::PictureContinuation
            });
        }
        Some(if matches!(nal_type, 32 | 33 | 34 | 35 | 39) {
            StartCodeRole::Leads
        } else {
            StartCodeRole::Continues
        })
    }

    fn sps_info(nals: &[&[u8]]) -> Option<SpsGeometry> {
        with_sps_rbsp(nals.iter().copied(), parse_sps)
    }
}

/// H.265 NAL unit type for a sequence parameter set (SPS_NUT).
const SPS_NUT: u8 = 33;
/// H.265 NAL unit types for the video and picture parameter sets.
const VPS_NUT: u8 = 32;
pub(crate) const PPS_NUT: u8 = 34;
const RASL_N_NUT: u8 = 8;
const RASL_R_NUT: u8 = 9;
const CRA_NUT: u8 = 21;
const END_OF_SEQUENCE_NUT: u8 = 36;
const END_OF_BITSTREAM_NUT: u8 = 37;

// RASL pictures reference pictures from before their CRA, which a decoder that
// starts at that CRA never receives (NoRaslOutputFlag, H.265 8.1.3).
#[derive(Debug)]
pub(crate) struct LeadingRaslFilter {
    next_irap_starts_sequence: bool,
    rasl_undecodable: bool,
}

impl Default for LeadingRaslFilter {
    fn default() -> Self {
        Self {
            next_irap_starts_sequence: true,
            rasl_undecodable: true,
        }
    }
}

impl LeadingRaslFilter {
    pub(crate) fn restart(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn keeps_access_unit(&mut self, nals: &[&[u8]]) -> bool {
        let nal_types = || nals.iter().filter_map(|nal| h265_nal_type(nal));
        let keep = match nal_types().find(|&nal_type| nal_type <= 31) {
            Some(nal_type @ 16..=23) => {
                self.rasl_undecodable = self.next_irap_starts_sequence || nal_type != CRA_NUT;
                self.next_irap_starts_sequence = false;
                true
            }
            Some(RASL_N_NUT | RASL_R_NUT) => !self.rasl_undecodable,
            _ => true,
        };
        if nal_types()
            .any(|nal_type| matches!(nal_type, END_OF_SEQUENCE_NUT | END_OF_BITSTREAM_NUT))
        {
            self.next_irap_starts_sequence = true;
        }
        keep
    }
}

/// Walk the NALs of `au` (Annex-B or AVCC, auto-detected), running `parse` over
/// the de-emulated RBSP of each SPS NAL and returning the first result it
/// yields. H.265 NAL type is bits `[1..7]` of the first header byte.
fn with_sps_rbsp<'a, T>(
    nals: impl IntoIterator<Item = &'a [u8]>,
    parse: impl Fn(&[u8]) -> Option<T>,
) -> Option<T> {
    for nal in nals {
        if nal.len() < 2 {
            continue;
        }
        let nal_unit_type = (nal[0] >> 1) & 0x3F;
        if nal_unit_type != SPS_NUT {
            continue;
        }
        // Strip the 2-byte NAL header, then de-emulate the RBSP.
        let rbsp = strip_emulation_prevention(&nal[2..]);
        if let Some(value) = parse(&rbsp) {
            return Some(value);
        }
    }
    None
}

/// Build the WebCodecs `codec` string for an H.265 stream from its SPS
/// (ISO/IEC 14496-15 Annex E.3, registered for WebCodecs by the W3C HEVC
/// registration): `"hev1."`, the profile space letter and profile_idc, the
/// compatibility flags in reverse bit order as hex, the tier letter and
/// level_idc, then the constraint bytes. Main profile, progressive, main tier,
/// level 3.1 gives `"hev1.1.6.L93.B0"`. `None` if the access unit carries no
/// parsable SPS.
// Only the WebCodecs decoder (wasm) needs this; the parser itself uses the
// geometry. Gated so the always-compiled module stays warning-clean elsewhere.
#[cfg(any(test, all(target_arch = "wasm32", feature = "web-codecs")))]
pub(crate) fn h265_codec_string(au: &[u8]) -> Option<String> {
    let ptl = with_sps_rbsp(crate::annexb::nal_units_any(au), |rbsp| {
        parse_h265_sps_head(rbsp).map(|(head, _)| head.ptl)
    })?;
    let space = match ptl.profile_space {
        0 => "",
        1 => "A",
        2 => "B",
        3 => "C",
        _ => return None,
    };
    let tier = if ptl.tier_flag == 0 { "L" } else { "H" };
    let compatibility = ptl.compatibility_flags.reverse_bits();
    let profile_idc = ptl.profile_idc;
    let level_idc = ptl.level_idc;
    let mut codec = alloc::format!("hev1.{space}{profile_idc}.{compatibility:X}.{tier}{level_idc}");
    // trailing zero constraint bytes are omitted, so an all-zero set ends the
    // string at the level
    if let Some(last) = ptl.constraint_bytes.iter().rposition(|byte| *byte != 0) {
        for byte in &ptl.constraint_bytes[..=last] {
            codec.push_str(&alloc::format!(".{byte:02X}"));
        }
    }
    Some(codec)
}

/// The H.265 SPS fields up to the conformance window (7.3.2.2), the part every
/// SPS consumer reads. Values are as coded, unchecked.
// only the Vulkan decoder reads every field
#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct H265SpsHead {
    pub sps_video_parameter_set_id: u8,
    pub sps_max_sub_layers_minus1: u8,
    pub sps_temporal_id_nesting_flag: bool,
    pub ptl: GeneralProfileTierLevel,
    pub sps_seq_parameter_set_id: u32,
    pub chroma_format_idc: u32,
    pub separate_colour_plane_flag: bool,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub conformance_window_flag: bool,
    /// `conf_win_{left,right,top,bottom}_offset`, zero without a window.
    pub conformance_window: [u32; 4],
}

/// Parse an SPS RBSP (the bytes after the 2-byte NAL header, emulation
/// prevention stripped) through the conformance window, returning the reader
/// positioned at `bit_depth_luma_minus8`. `None` on truncation.
pub(crate) fn parse_h265_sps_head(rbsp: &[u8]) -> Option<(H265SpsHead, BitReader<'_>)> {
    let mut br = BitReader::new(rbsp);
    let sps_video_parameter_set_id = br.read_bits(4)? as u8;
    let sps_max_sub_layers_minus1 = br.read_bits(3)? as u8;
    let sps_temporal_id_nesting_flag = br.read_bit()? == 1;
    let ptl = read_profile_tier_level(&mut br, u32::from(sps_max_sub_layers_minus1))?;
    let sps_seq_parameter_set_id = br.read_ue()?;
    let chroma_format_idc = br.read_ue()?;
    let separate_colour_plane_flag = chroma_format_idc == 3 && br.read_bit()? == 1;
    let pic_width_in_luma_samples = br.read_ue()?;
    let pic_height_in_luma_samples = br.read_ue()?;
    let conformance_window_flag = br.read_bit()? == 1;
    let conformance_window = if conformance_window_flag {
        [br.read_ue()?, br.read_ue()?, br.read_ue()?, br.read_ue()?]
    } else {
        [0; 4]
    };
    let head = H265SpsHead {
        sps_video_parameter_set_id,
        sps_max_sub_layers_minus1,
        sps_temporal_id_nesting_flag,
        ptl,
        sps_seq_parameter_set_id,
        chroma_format_idc,
        separate_colour_plane_flag,
        pic_width_in_luma_samples,
        pic_height_in_luma_samples,
        conformance_window_flag,
        conformance_window,
    };
    Some((head, br))
}

// the H.265 PPS fields up to entropy_coding_sync_enabled_flag (7.3.2.3.1), as coded
#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct H265PpsHead {
    pub poc: crate::poc::H265PpsPoc,
    pub pps_seq_parameter_set_id: u32,
    pub dependent_slice_segments_enabled_flag: u8,
    pub sign_data_hiding_enabled_flag: u8,
    pub cabac_init_present_flag: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub init_qp_minus26: i8,
    pub constrained_intra_pred_flag: u8,
    pub transform_skip_enabled_flag: u8,
    pub cu_qp_delta_enabled_flag: u8,
    pub diff_cu_qp_delta_depth: u8,
    pub pps_cb_qp_offset: i8,
    pub pps_cr_qp_offset: i8,
    pub pps_slice_chroma_qp_offsets_present_flag: u8,
    pub weighted_pred_flag: u8,
    pub weighted_bipred_flag: u8,
    pub transquant_bypass_enabled_flag: u8,
    pub tiles_enabled_flag: u8,
    pub entropy_coding_sync_enabled_flag: u8,
}

// takes the RBSP after the 2-byte NAL header and returns the reader at the tile layout
#[cfg_attr(
    not(any(feature = "vulkan-video", all(target_os = "linux", feature = "nvdec"))),
    allow(dead_code)
)]
pub(crate) fn parse_h265_pps_head(rbsp: &[u8]) -> Option<(H265PpsHead, BitReader<'_>)> {
    let mut br = BitReader::new(rbsp);
    let (poc, pps_seq_parameter_set_id, dependent_slice_segments_enabled_flag) =
        crate::poc::parse_h265_pps_poc_bits(&mut br)?;
    let sign_data_hiding_enabled_flag = br.read_bit()? as u8;
    let cabac_init_present_flag = br.read_bit()? as u8;
    let num_ref_idx_l0_default_active_minus1 = br.read_ue()? as u8;
    let num_ref_idx_l1_default_active_minus1 = br.read_ue()? as u8;
    let init_qp_minus26 = br.read_se()? as i8;
    let constrained_intra_pred_flag = br.read_bit()? as u8;
    let transform_skip_enabled_flag = br.read_bit()? as u8;
    let cu_qp_delta_enabled_flag = br.read_bit()? as u8;
    let diff_cu_qp_delta_depth = if cu_qp_delta_enabled_flag == 1 {
        br.read_ue()? as u8
    } else {
        0
    };
    let pps_cb_qp_offset = br.read_se()? as i8;
    let pps_cr_qp_offset = br.read_se()? as i8;
    let pps_slice_chroma_qp_offsets_present_flag = br.read_bit()? as u8;
    let weighted_pred_flag = br.read_bit()? as u8;
    let weighted_bipred_flag = br.read_bit()? as u8;
    let transquant_bypass_enabled_flag = br.read_bit()? as u8;
    let tiles_enabled_flag = br.read_bit()? as u8;
    let entropy_coding_sync_enabled_flag = br.read_bit()? as u8;
    let head = H265PpsHead {
        poc,
        pps_seq_parameter_set_id,
        dependent_slice_segments_enabled_flag: dependent_slice_segments_enabled_flag as u8,
        sign_data_hiding_enabled_flag,
        cabac_init_present_flag,
        num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1,
        init_qp_minus26,
        constrained_intra_pred_flag,
        transform_skip_enabled_flag,
        cu_qp_delta_enabled_flag,
        diff_cu_qp_delta_depth,
        pps_cb_qp_offset,
        pps_cr_qp_offset,
        pps_slice_chroma_qp_offsets_present_flag,
        weighted_pred_flag,
        weighted_bipred_flag,
        transquant_bypass_enabled_flag,
        tiles_enabled_flag,
        entropy_coding_sync_enabled_flag,
    };
    Some((head, br))
}

/// Parse the SPS RBSP (H.265 7.3.2.2): the cropped picture dimensions from the
/// head, then (best-effort) the framerate from the VUI `timing_info`.
/// `None` on a parse failure before the dimensions resolve.
fn parse_sps(rbsp: &[u8]) -> Option<SpsGeometry> {
    let (head, mut br) = parse_h265_sps_head(rbsp)?;
    let sps_max_sub_layers_minus1 = u32::from(head.sps_max_sub_layers_minus1);
    let [left, right, top, bottom] = head.conformance_window;
    let (pic_width, pic_height) = (
        head.pic_width_in_luma_samples,
        head.pic_height_in_luma_samples,
    );

    // Crop offsets are in chroma sample units, scaled to luma by SubWidthC /
    // SubHeightC (H.265 7.4.3.2.1). ChromaArrayType 0 (monochrome or 4:4:4 with
    // separate colour planes) and 4:4:4 use 1x1.
    let chroma_array_type = if head.separate_colour_plane_flag {
        0
    } else {
        head.chroma_format_idc
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
    // The rest of the walk is best-effort: it sits past the scaling-list / RPS /
    // long-term-ref blocks, and a stream this walk cannot cross still has valid
    // geometry, so a failure downgrades its fields instead of failing the parse.
    let mut tail = SpsTail::default();
    let framerate = parse_vui_framerate(&mut br, sps_max_sub_layers_minus1, &mut tail);
    let max_num_reorder_pics = tail.max_num_reorder_pics;
    let poc = tail
        .log2_max_pic_order_cnt_lsb
        .map(|log2_max_pic_order_cnt_lsb| crate::poc::SpsPocParams {
            log2_max_pic_order_cnt_lsb,
            separate_colour_plane_flag: head.separate_colour_plane_flag,
            ..crate::poc::SpsPocParams::default()
        });
    Some(SpsGeometry {
        width,
        height,
        framerate,
        colorimetry: tail.colorimetry,
        // H.265 codes the `time_code` SEI self-contained, so nothing from the
        // SPS is needed to read it.
        pic_timing: None,
        // sps_max_num_reorder_pics bounds how many pictures may precede a
        // picture in decode order and follow it in output order, so 0 forbids
        // reordering outright. Unlike H.264's optional VUI restriction this is a
        // normative SPS field the output process is specified against (C.5.2.2).
        presents_in_decode_order: max_num_reorder_pics == Some(0),
        poc,
    })
}

/// What the best-effort walk past the conformance window recovers on its way to
/// the VUI: fields that sit far ahead of it, so a stream the walk cannot cross
/// all the way still yields them.
#[derive(Debug, Default)]
struct SpsTail {
    max_num_reorder_pics: Option<u32>,
    /// Colour description from the VUI `video_signal_type` block,
    /// [`Colorimetry::UNKNOWN`] when absent or the walk ends before it.
    colorimetry: Colorimetry,
    /// `log2_max_pic_order_cnt_lsb_minus4 + 4`, the width of the slice header's
    /// order-count lsb.
    log2_max_pic_order_cnt_lsb: Option<u32>,
}

/// Continue the SPS walk past the conformance window down to the VUI
/// `timing_info`, returning the framerate as Q16 fixed-point fps
/// (`vui_time_scale / vui_num_units_in_tick`; H.265 ticks are per picture, so
/// there is no H.264-style field factor of 2). `None` when the VUI omits timing
/// or a block on the way ends early / is out of range. The [`SpsTail`] fields are
/// written on the way past the blocks that carry them, which sit far ahead of the
/// VUI, so a stream this walk cannot cross all the way still yields them.
fn parse_vui_framerate(
    br: &mut BitReader,
    sps_max_sub_layers_minus1: u32,
    tail: &mut SpsTail,
) -> Option<u32> {
    let mut middle = H265SpsMiddle::default();
    let walked = read_h265_sps_middle(br, sps_max_sub_layers_minus1, &mut middle);
    tail.log2_max_pic_order_cnt_lsb = middle
        .log2_max_pic_order_cnt_lsb_minus4
        .map(|log2_minus4| log2_minus4 + 4);
    // the highest sub-layer's bound is the one the output process uses
    tail.max_num_reorder_pics = middle
        .sub_layer_ordering
        .map(|ordering| ordering.max_num_reorder_pics[sps_max_sub_layers_minus1 as usize]);
    walked?;
    if br.read_bit()? != 1 {
        return None; // vui_parameters_present_flag
    }

    // vui_parameters() (E.2.1) down to timing_info. The head mirrors H.264's,
    // then HEVC inserts four flags and the default display window.
    if let Some(signal) = crate::vui::read_vui_signal_type(br)? {
        tail.colorimetry = signal.colorimetry();
    }
    skip_vui_display_fields(br)?;
    if br.read_bit()? != 1 {
        return None; // vui_timing_info_present_flag
    }
    let num_units_in_tick = br.read_bits(32)?;
    let time_scale = br.read_bits(32)?;
    framerate_q16_from_fraction(time_scale, num_units_in_tick)
}

// the VUI fields between `video_signal_type` and `vui_timing_info_present_flag`
fn skip_vui_display_fields(br: &mut BitReader) -> Option<()> {
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
    Some(())
}

// the rest of `vui_parameters()` (E.2.1) after `video_signal_type`
#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
pub(crate) fn skip_h265_vui_tail(br: &mut BitReader, sps_max_sub_layers_minus1: u32) -> Option<()> {
    skip_vui_display_fields(br)?;
    // vui_timing_info_present_flag
    if br.read_bit()? == 1 {
        br.read_bits(32)?; // vui_num_units_in_tick
        br.read_bits(32)?; // vui_time_scale
        if br.read_bit()? == 1 {
            br.read_ue()?; // vui_num_ticks_poc_diff_one_minus1
        }
        if br.read_bit()? == 1 {
            skip_h265_hrd_parameters(br, sps_max_sub_layers_minus1)?;
        }
    }
    // bitstream_restriction_flag
    if br.read_bit()? == 1 {
        br.read_bit()?; // tiles_fixed_structure_flag
        br.read_bit()?; // motion_vectors_over_pic_boundaries_flag
        br.read_bit()?; // restricted_ref_pic_lists_flag
        for _ in 0..5 {
            // min_spatial_segmentation_idc, max_bytes_per_pic_denom,
            // max_bits_per_min_cu_denom, log2_max_mv_length_horizontal / vertical
            br.read_ue()?;
        }
    }
    Some(())
}

// a CPB count past 32 means a desynced parse (E.3.2)
const H265_MAX_CPB_COUNT: u32 = 32;

// `hrd_parameters(1, maxNumSubLayersMinus1)` (E.2.2)
#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
fn skip_h265_hrd_parameters(br: &mut BitReader, max_sub_layers_minus1: u32) -> Option<()> {
    let nal_hrd_parameters_present_flag = br.read_bit()? == 1;
    let vcl_hrd_parameters_present_flag = br.read_bit()? == 1;
    let mut sub_pic_hrd_params_present_flag = false;
    if nal_hrd_parameters_present_flag || vcl_hrd_parameters_present_flag {
        sub_pic_hrd_params_present_flag = br.read_bit()? == 1;
        if sub_pic_hrd_params_present_flag {
            br.read_bits(8)?; // tick_divisor_minus2
            br.read_bits(5)?; // du_cpb_removal_delay_increment_length_minus1
            br.read_bit()?; // sub_pic_cpb_params_in_pic_timing_sei_flag
            br.read_bits(5)?; // dpb_output_delay_du_length_minus1
        }
        br.read_bits(4)?; // bit_rate_scale
        br.read_bits(4)?; // cpb_size_scale
        if sub_pic_hrd_params_present_flag {
            br.read_bits(4)?; // cpb_size_du_scale
        }
        br.read_bits(5)?; // initial_cpb_removal_delay_length_minus1
        br.read_bits(5)?; // au_cpb_removal_delay_length_minus1
        br.read_bits(5)?; // dpb_output_delay_length_minus1
    }
    for _ in 0..=max_sub_layers_minus1 {
        let fixed_pic_rate_general_flag = br.read_bit()? == 1;
        let fixed_pic_rate_within_cvs_flag = fixed_pic_rate_general_flag || br.read_bit()? == 1;
        let mut low_delay_hrd_flag = false;
        if fixed_pic_rate_within_cvs_flag {
            br.read_ue()?; // elemental_duration_in_tc_minus1
        } else {
            low_delay_hrd_flag = br.read_bit()? == 1;
        }
        let cpb_cnt_minus1 = if low_delay_hrd_flag { 0 } else { br.read_ue()? };
        if cpb_cnt_minus1 >= H265_MAX_CPB_COUNT {
            return None;
        }
        let present = [
            nal_hrd_parameters_present_flag,
            vcl_hrd_parameters_present_flag,
        ];
        for _ in present.into_iter().filter(|&present| present) {
            // sub_layer_hrd_parameters()
            for _ in 0..=cpb_cnt_minus1 {
                br.read_ue()?; // bit_rate_value_minus1
                br.read_ue()?; // cpb_size_value_minus1
                if sub_pic_hrd_params_present_flag {
                    br.read_ue()?; // cpb_size_du_value_minus1
                    br.read_ue()?; // bit_rate_du_value_minus1
                }
                br.read_bit()?; // cbr_flag
            }
        }
    }
    Some(())
}

// whether the SPS extensions after the VUI (7.3.2.2.1) turn on range extension or SCC tools
#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct H265SpsExtensions {
    pub range_extension_tools: bool,
    pub screen_content_coding: bool,
}

#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
pub(crate) fn read_h265_sps_extensions(br: &mut BitReader) -> Option<H265SpsExtensions> {
    // sps_extension_present_flag
    if br.read_bit()? == 0 {
        return Some(H265SpsExtensions::default());
    }
    let sps_range_extension_flag = br.read_bit()? == 1;
    br.read_bit()?; // sps_multilayer_extension_flag
    br.read_bit()?; // sps_3d_extension_flag
    let sps_scc_extension_flag = br.read_bit()? == 1;
    br.read_bits(4)?; // sps_extension_4bits
    let mut range_extension_tools = false;
    if sps_range_extension_flag {
        // transform_skip_rotation_enabled_flag through cabac_bypass_alignment_enabled_flag
        for _ in 0..9 {
            range_extension_tools |= br.read_bit()? == 1;
        }
    }
    Some(H265SpsExtensions {
        range_extension_tools,
        screen_content_coding: sps_scc_extension_flag,
    })
}

// sps_max_sub_layers_minus1 is at most 6
pub(crate) const H265_MAX_SUB_LAYERS: usize = 7;
const H265_MAX_SHORT_TERM_REF_PIC_SETS: u32 = 64;
pub(crate) const H265_MAX_LONG_TERM_REF_PICS_SPS: usize = 32;

/// `sps_max_dec_pic_buffering_minus1`, `sps_max_num_reorder_pics` and
/// `sps_max_latency_increase_plus1` per sub-layer. Without
/// `sps_sub_layer_ordering_info_present_flag` only the highest is coded.
#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct H265SubLayerOrdering {
    pub info_present_flag: bool,
    pub max_dec_pic_buffering_minus1: [u32; H265_MAX_SUB_LAYERS],
    pub max_num_reorder_pics: [u32; H265_MAX_SUB_LAYERS],
    pub max_latency_increase_plus1: [u32; H265_MAX_SUB_LAYERS],
}

/// The H.265 SPS fields between the conformance window and the VUI (7.3.2.2).
/// The `Option` fields stay `None` until the walk gets past them, so a caller
/// keeps what was read before a truncated or out-of-range field.
// only the Vulkan decoder reads every field
#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
#[derive(Debug, Default)]
pub(crate) struct H265SpsMiddle {
    pub bit_depth_luma_minus8: u32,
    pub bit_depth_chroma_minus8: u32,
    pub log2_max_pic_order_cnt_lsb_minus4: Option<u32>,
    pub sub_layer_ordering: Option<H265SubLayerOrdering>,
    pub log2_min_luma_coding_block_size_minus3: u32,
    pub log2_diff_max_min_luma_coding_block_size: u32,
    pub log2_min_luma_transform_block_size_minus2: u32,
    pub log2_diff_max_min_luma_transform_block_size: u32,
    pub max_transform_hierarchy_depth_inter: u32,
    pub max_transform_hierarchy_depth_intra: u32,
    pub scaling_list_enabled_flag: bool,
    pub sps_scaling_list_data_present_flag: bool,
    // the default lists when the SPS enables scaling lists without coding them
    pub scaling_lists: Option<H265ScalingLists>,
    pub amp_enabled_flag: bool,
    pub sample_adaptive_offset_enabled_flag: bool,
    pub pcm_enabled_flag: bool,
    pub pcm_sample_bit_depth_luma_minus1: u32,
    pub pcm_sample_bit_depth_chroma_minus1: u32,
    pub log2_min_pcm_luma_coding_block_size_minus3: u32,
    pub log2_diff_max_min_pcm_luma_coding_block_size: u32,
    pub pcm_loop_filter_disabled_flag: bool,
    pub short_term_rps: Vec<H265ShortTermRps>,
    pub long_term_ref_pics_present_flag: bool,
    pub num_long_term_ref_pics_sps: u32,
    pub lt_ref_pic_poc_lsb_sps: [u32; H265_MAX_LONG_TERM_REF_PICS_SPS],
    pub used_by_curr_pic_lt_sps_flag: [bool; H265_MAX_LONG_TERM_REF_PICS_SPS],
    pub sps_temporal_mvp_enabled_flag: bool,
    pub strong_intra_smoothing_enabled_flag: bool,
}

/// Walk the SPS from `bit_depth_luma_minus8` through
/// `strong_intra_smoothing_enabled_flag`, filling `middle` as it goes. `None`
/// on truncation or an out-of-range count, with `middle` holding what was read.
pub(crate) fn read_h265_sps_middle(
    br: &mut BitReader,
    sps_max_sub_layers_minus1: u32,
    middle: &mut H265SpsMiddle,
) -> Option<()> {
    middle.bit_depth_luma_minus8 = br.read_ue()?;
    middle.bit_depth_chroma_minus8 = br.read_ue()?;
    let log2_max_pic_order_cnt_lsb_minus4 = br.read_ue()?;
    if log2_max_pic_order_cnt_lsb_minus4 > crate::poc::MAX_LOG2_MINUS4 {
        return None;
    }
    middle.log2_max_pic_order_cnt_lsb_minus4 = Some(log2_max_pic_order_cnt_lsb_minus4);

    let highest_sub_layer = sps_max_sub_layers_minus1 as usize;
    if highest_sub_layer >= H265_MAX_SUB_LAYERS {
        return None;
    }
    let mut ordering = H265SubLayerOrdering {
        info_present_flag: br.read_bit()? == 1,
        ..H265SubLayerOrdering::default()
    };
    let first_coded = if ordering.info_present_flag {
        0
    } else {
        highest_sub_layer
    };
    for sub_layer in first_coded..=highest_sub_layer {
        ordering.max_dec_pic_buffering_minus1[sub_layer] = br.read_ue()?;
        ordering.max_num_reorder_pics[sub_layer] = br.read_ue()?;
        ordering.max_latency_increase_plus1[sub_layer] = br.read_ue()?;
    }
    middle.sub_layer_ordering = Some(ordering);

    middle.log2_min_luma_coding_block_size_minus3 = br.read_ue()?;
    middle.log2_diff_max_min_luma_coding_block_size = br.read_ue()?;
    middle.log2_min_luma_transform_block_size_minus2 = br.read_ue()?;
    middle.log2_diff_max_min_luma_transform_block_size = br.read_ue()?;
    middle.max_transform_hierarchy_depth_inter = br.read_ue()?;
    middle.max_transform_hierarchy_depth_intra = br.read_ue()?;
    middle.scaling_list_enabled_flag = br.read_bit()? == 1;
    if middle.scaling_list_enabled_flag {
        middle.sps_scaling_list_data_present_flag = br.read_bit()? == 1;
        middle.scaling_lists = Some(if middle.sps_scaling_list_data_present_flag {
            read_scaling_list_data(br)?
        } else {
            H265ScalingLists::default()
        });
    }
    middle.amp_enabled_flag = br.read_bit()? == 1;
    middle.sample_adaptive_offset_enabled_flag = br.read_bit()? == 1;
    middle.pcm_enabled_flag = br.read_bit()? == 1;
    if middle.pcm_enabled_flag {
        middle.pcm_sample_bit_depth_luma_minus1 = br.read_bits(4)?;
        middle.pcm_sample_bit_depth_chroma_minus1 = br.read_bits(4)?;
        middle.log2_min_pcm_luma_coding_block_size_minus3 = br.read_ue()?;
        middle.log2_diff_max_min_pcm_luma_coding_block_size = br.read_ue()?;
        middle.pcm_loop_filter_disabled_flag = br.read_bit()? == 1;
    }

    let num_short_term_ref_pic_sets = br.read_ue()?;
    if num_short_term_ref_pic_sets > H265_MAX_SHORT_TERM_REF_PIC_SETS {
        return None;
    }
    let num_sets = num_short_term_ref_pic_sets as usize;
    middle.short_term_rps = Vec::with_capacity(num_sets);
    for idx in 0..num_sets {
        let (set, _) = parse_h265_short_term_rps(br, idx, num_sets, &middle.short_term_rps)?;
        middle.short_term_rps.push(set);
    }

    middle.long_term_ref_pics_present_flag = br.read_bit()? == 1;
    if middle.long_term_ref_pics_present_flag {
        let num_long_term_ref_pics_sps = br.read_ue()?;
        if num_long_term_ref_pics_sps as usize > H265_MAX_LONG_TERM_REF_PICS_SPS {
            return None;
        }
        middle.num_long_term_ref_pics_sps = num_long_term_ref_pics_sps;
        for lt in 0..num_long_term_ref_pics_sps as usize {
            middle.lt_ref_pic_poc_lsb_sps[lt] =
                br.read_bits(log2_max_pic_order_cnt_lsb_minus4 + 4)?;
            middle.used_by_curr_pic_lt_sps_flag[lt] = br.read_bit()? == 1;
        }
    }
    middle.sps_temporal_mvp_enabled_flag = br.read_bit()? == 1;
    middle.strong_intra_smoothing_enabled_flag = br.read_bit()? == 1;
    Some(())
}

// Table 7-6, in coded (up-right diagonal) order
const DEFAULT_INTRA_SCALING_LIST: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 16, 17, 16, 17, 18, 17, 18, 18, 17, 18, 21, 19, 20,
    21, 20, 19, 21, 24, 22, 22, 24, 24, 22, 22, 24, 25, 25, 27, 30, 27, 25, 25, 29, 31, 35, 35, 31,
    29, 36, 41, 44, 41, 36, 47, 54, 54, 47, 65, 70, 65, 88, 88, 115,
];
const DEFAULT_INTER_SCALING_LIST: [u8; 64] = [
    16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 20, 20, 20,
    20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 28, 28, 28, 28, 28,
    28, 33, 33, 33, 33, 33, 41, 41, 41, 41, 54, 54, 54, 71, 71, 91,
];
const FLAT_SCALING_FACTOR: u8 = 16;
// 4x4, 8x8, 16x16 and 32x32
const SCALING_LIST_SIZES: usize = 4;
// six matrices per size, two for 32x32 outside 4:4:4
const SCALING_LIST_MATRICES: usize = 6;
const SCALING_LIST_32X32_MATRICES: usize = 2;
const SCALING_LIST_32X32_SIZE_ID: usize = 3;
// the first intra matrix id is 0, the first inter one 3
const SCALING_LIST_FIRST_INTER_MATRIX: usize = 3;

// `ScalingList[sizeId][matrixId][i]` of H.265 7.4.5, `i` in coded (up-right
// diagonal) order, and a 4x4 list uses the first 16 entries
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H265ScalingLists {
    pub coefficients: [[[u8; 64]; SCALING_LIST_MATRICES]; SCALING_LIST_SIZES],
    // `scaling_list_dc_coef_minus8 + 8` of the 16x16 and 32x32 lists
    pub dc_coefficients: [[u8; SCALING_LIST_MATRICES]; 2],
}

impl Default for H265ScalingLists {
    fn default() -> Self {
        let mut coefficients =
            [[[FLAT_SCALING_FACTOR; 64]; SCALING_LIST_MATRICES]; SCALING_LIST_SIZES];
        for size in coefficients.iter_mut().skip(1) {
            for (matrix_id, list) in size.iter_mut().enumerate() {
                *list = if matrix_id < SCALING_LIST_FIRST_INTER_MATRIX {
                    DEFAULT_INTRA_SCALING_LIST
                } else {
                    DEFAULT_INTER_SCALING_LIST
                };
            }
        }
        // the two coded 32x32 lists are one intra, one inter
        coefficients[SCALING_LIST_32X32_SIZE_ID][1] = DEFAULT_INTER_SCALING_LIST;
        Self {
            coefficients,
            dc_coefficients: [[FLAT_SCALING_FACTOR; SCALING_LIST_MATRICES]; 2],
        }
    }
}

// H.265 7.3.4
pub(crate) fn read_scaling_list_data(br: &mut BitReader) -> Option<H265ScalingLists> {
    let mut lists = H265ScalingLists::default();
    for size_id in 0..SCALING_LIST_SIZES {
        let matrices = if size_id == SCALING_LIST_32X32_SIZE_ID {
            SCALING_LIST_32X32_MATRICES
        } else {
            SCALING_LIST_MATRICES
        };
        let dc_index = size_id.checked_sub(2);
        for matrix_id in 0..matrices {
            let scaling_list_pred_mode_flag = br.read_bit()? == 1;
            if !scaling_list_pred_mode_flag {
                let scaling_list_pred_matrix_id_delta = br.read_ue()? as usize;
                // a zero delta keeps the default list this matrix starts with
                if scaling_list_pred_matrix_id_delta == 0 {
                    continue;
                }
                let reference = matrix_id.checked_sub(scaling_list_pred_matrix_id_delta)?;
                lists.coefficients[size_id][matrix_id] = lists.coefficients[size_id][reference];
                if let Some(dc) = dc_index {
                    lists.dc_coefficients[dc][matrix_id] = lists.dc_coefficients[dc][reference];
                }
                continue;
            }
            let coef_num = 64usize.min(1 << (4 + (size_id << 1)));
            let mut next_coef = 8i32;
            if let Some(dc) = dc_index {
                let scaling_list_dc_coef_minus8 = br.read_se()?;
                if !(-7..=247).contains(&scaling_list_dc_coef_minus8) {
                    return None;
                }
                next_coef = scaling_list_dc_coef_minus8 + 8;
                lists.dc_coefficients[dc][matrix_id] = next_coef as u8;
            }
            for coefficient in &mut lists.coefficients[size_id][matrix_id][..coef_num] {
                let scaling_list_delta_coef = br.read_se()?;
                if !(-128..=127).contains(&scaling_list_delta_coef) {
                    return None;
                }
                next_coef = (next_coef + scaling_list_delta_coef + 256) % 256;
                *coefficient = next_coef as u8;
            }
        }
    }
    Some(lists)
}

/// Number of constraint / reserved flag bytes in a general
/// `profile_tier_level`, between the compatibility flags and the level.
const GENERAL_CONSTRAINT_BYTES: usize = 6;

/// The general `profile_tier_level` fields (H.265 7.3.3), which the WebCodecs
/// codec string and the Vulkan `Std*` profile-tier-level are built from.
// the WebCodecs codec string and the Vulkan decoder read these
#[cfg_attr(
    not(any(
        test,
        feature = "vulkan-video",
        all(target_arch = "wasm32", feature = "web-codecs")
    )),
    allow(dead_code)
)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct GeneralProfileTierLevel {
    // only the WebCodecs codec string reads it
    #[cfg_attr(
        not(any(test, all(target_arch = "wasm32", feature = "web-codecs"))),
        allow(dead_code)
    )]
    pub profile_space: u8,
    pub tier_flag: u8,
    pub profile_idc: u8,
    // only the WebCodecs codec string reads it
    #[cfg_attr(
        not(any(test, all(target_arch = "wasm32", feature = "web-codecs"))),
        allow(dead_code)
    )]
    pub compatibility_flags: u32,
    /// Opens with the progressive, interlaced, non-packed and frame-only
    /// source flags, most significant bit first.
    pub constraint_bytes: [u8; GENERAL_CONSTRAINT_BYTES],
    pub level_idc: u8,
}

#[cfg_attr(not(feature = "vulkan-video"), allow(dead_code))]
impl GeneralProfileTierLevel {
    /// The `index`th of the four source flags that open the constraint bits.
    fn source_flag(&self, index: u32) -> u8 {
        const MOST_SIGNIFICANT_BIT: u32 = 7;
        (self.constraint_bytes[0] >> (MOST_SIGNIFICANT_BIT - index)) & 1
    }

    pub(crate) fn progressive_source_flag(&self) -> u8 {
        self.source_flag(0)
    }

    pub(crate) fn interlaced_source_flag(&self) -> u8 {
        self.source_flag(1)
    }

    pub(crate) fn non_packed_constraint_flag(&self) -> u8 {
        self.source_flag(2)
    }

    pub(crate) fn frame_only_constraint_flag(&self) -> u8 {
        self.source_flag(3)
    }
}

/// Read the general block of `profile_tier_level`: a fixed 96 bits (2 + 1 + 5
/// profile space/tier/idc, 32 compatibility flags, 48 constraint/reserved bits,
/// 8-bit level).
fn read_general_profile_tier_level(br: &mut BitReader) -> Option<GeneralProfileTierLevel> {
    let profile_space = br.read_bits(2)? as u8;
    let tier_flag = br.read_bit()? as u8;
    let profile_idc = br.read_bits(5)? as u8;
    let compatibility_flags = br.read_bits(32)?;
    let mut constraint_bytes = [0u8; GENERAL_CONSTRAINT_BYTES];
    for byte in &mut constraint_bytes {
        *byte = br.read_bits(8)? as u8;
    }
    let level_idc = br.read_bits(8)? as u8;
    Some(GeneralProfileTierLevel {
        profile_space,
        tier_flag,
        profile_idc,
        compatibility_flags,
        constraint_bytes,
        level_idc,
    })
}

/// Read `profile_tier_level(1, max_sub_layers_minus1)` (H.265 7.3.3). The
/// general block is a fixed 96 bits (88-bit profile/tier/constraints + 8-bit
/// level); per-sub-layer blocks follow only when `max_sub_layers_minus1 > 0`,
/// and are advanced past without being carried. `None` on truncation.
pub(crate) fn read_profile_tier_level(
    br: &mut BitReader,
    max_sub_layers_minus1: u32,
) -> Option<GeneralProfileTierLevel> {
    let general = read_general_profile_tier_level(br)?;

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
    Some(general)
}

/// A short-term reference-picture set in canonical (explicit) form: the derived
/// `DeltaPocS0/S1` deltas and used-by-current flags, whether the stream coded
/// the set explicitly or by inter-RPS prediction. Storing the derived form lets
/// the Vulkan Video `Std*` mapping always emit the explicit encoding (with
/// `inter_ref_pic_set_prediction_flag == 0`), sidestepping the driver-facing
/// ambiguity of the predicted form; the parser here only needs the pic counts
/// to cross the RPS list on the way to the VUI.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
/// is derived from `prev[RefRpsIdx]` per H.265 7.4.8; the bit reader is always
/// advanced by `NumDeltaPocs[RefRpsIdx] + 1` used/use-delta flags in that branch.
/// `num_sets` is the SPS's `num_short_term_ref_pic_sets`: only a slice-coded set
/// (`idx == num_sets`) carries `delta_idx_minus1`; within the SPS it is inferred
/// 0 (RefRpsIdx = idx - 1). Also returns `NumDeltaPocs[RefRpsIdx]` (0 for an
/// explicitly-coded set): a Vulkan driver re-parsing an inter-predicted slice
/// RPS needs it (`NumDeltaPocsOfRefRpsIdx`). `None` on truncation or an
/// out-of-range count.
pub(crate) fn parse_h265_short_term_rps(
    br: &mut BitReader,
    idx: usize,
    num_sets: usize,
    prev: &[H265ShortTermRps],
) -> Option<(H265ShortTermRps, u8)> {
    let mut rps = H265ShortTermRps::default();
    let mut ref_num_delta_pocs = 0u8;
    let inter_pred = if idx != 0 { br.read_bit()? == 1 } else { false };

    if inter_pred {
        // RefRpsIdx = stRpsIdx - (delta_idx_minus1 + 1); delta_idx_minus1 is
        // coded only for a slice-header set (stRpsIdx == num_sets).
        let delta_idx = if idx == num_sets {
            br.read_ue()? as usize + 1
        } else {
            1
        };
        let reference = prev.get(idx.checked_sub(delta_idx)?)?;
        let delta_rps_sign = br.read_bit()? as i32;
        let abs_delta_rps_minus1 = br.read_ue()? as i32;
        let delta_rps =
            (1 - 2 * delta_rps_sign).checked_mul(abs_delta_rps_minus1.checked_add(1)?)?;
        let num_neg = reference.num_negative_pics as usize;
        let num_pos = reference.num_positive_pics as usize;
        let num_delta_pocs = num_neg + num_pos;
        ref_num_delta_pocs = num_delta_pocs as u8;

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
    Some((rps, ref_num_delta_pocs))
}

/// Fuzzing entry: walk an H.265 access unit for the SPS and parse its geometry
/// (NAL scan, emulation-prevention strip, Exp-Golomb bit reader). Exposed only
/// under `--cfg fuzzing` (cargo-fuzz) so the normal public API is unchanged.
#[cfg(fuzzing)]
pub fn fuzz_parse(data: &[u8]) {
    let _ = H265Codec::extract_sps_info(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_unit::access_unit_starts;
    use crate::annexb::{nal_units, BitWriter};
    use crate::testutil::{frame_with_bytes, CollectSink};

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
        let _ = H265Codec::extract_sps_info(&sps);
    }
    use alloc::vec;

    use g2g_core::{AsyncElement, Caps, CapsConstraint, Dim, G2gError, PipelinePacket, Rate};

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
        finish_sps(sps_head(pic_w, pic_h, chroma_format_idc, conf, &ZERO_PTL))
    }

    /// A `profile_tier_level` of all-zero fields: what the geometry tests wrote
    /// before the codec-string parse read the block's contents.
    const ZERO_PTL: GeneralProfileTierLevel = GeneralProfileTierLevel {
        profile_space: 0,
        tier_flag: 0,
        profile_idc: 0,
        compatibility_flags: 0,
        constraint_bytes: [0; GENERAL_CONSTRAINT_BYTES],
        level_idc: 0,
    };

    /// The SPS fields up to the conformance window, as a bit writer the caller
    /// can extend (the VUI variant) before wrapping into a NAL.
    fn sps_head(
        pic_w: u32,
        pic_h: u32,
        chroma_format_idc: u32,
        conf: Option<(u32, u32, u32, u32)>,
        ptl: &GeneralProfileTierLevel,
    ) -> BitWriter {
        let (conformance_window_flag, conformance_window) = match conf {
            Some((left, right, top, bottom)) => (true, [left, right, top, bottom]),
            None => (false, [0; 4]),
        };
        write_sps_head(&H265SpsHead {
            sps_video_parameter_set_id: 0,
            sps_max_sub_layers_minus1: 0,
            sps_temporal_id_nesting_flag: true,
            ptl: *ptl,
            sps_seq_parameter_set_id: 0,
            chroma_format_idc,
            separate_colour_plane_flag: false,
            pic_width_in_luma_samples: pic_w,
            pic_height_in_luma_samples: pic_h,
            conformance_window_flag,
            conformance_window,
        })
    }

    // codes no sub-layer profile or level
    fn write_sps_head(head: &H265SpsHead) -> BitWriter {
        let mut w = BitWriter::default();
        w.write_bits(u32::from(head.sps_video_parameter_set_id), 4);
        w.write_bits(u32::from(head.sps_max_sub_layers_minus1), 3);
        w.write_bit(u32::from(head.sps_temporal_id_nesting_flag));
        write_profile_tier_level(&mut w, &head.ptl);
        for _ in 0..head.sps_max_sub_layers_minus1 {
            w.write_bit(0); // sub_layer_profile_present_flag
            w.write_bit(0); // sub_layer_level_present_flag
        }
        if head.sps_max_sub_layers_minus1 > 0 {
            for _ in head.sps_max_sub_layers_minus1..8 {
                w.write_bits(0, 2); // reserved_zero_2bits
            }
        }
        w.write_ue(head.sps_seq_parameter_set_id);
        w.write_ue(head.chroma_format_idc);
        if head.chroma_format_idc == 3 {
            w.write_bit(u32::from(head.separate_colour_plane_flag));
        }
        w.write_ue(head.pic_width_in_luma_samples);
        w.write_ue(head.pic_height_in_luma_samples);
        w.write_bit(u32::from(head.conformance_window_flag));
        if head.conformance_window_flag {
            for offset in head.conformance_window {
                w.write_ue(offset);
            }
        }
        w
    }

    /// The inverse of [`read_general_profile_tier_level`]: 96 bits of general
    /// profile / tier / compatibility / constraint / level fields.
    fn write_profile_tier_level(w: &mut BitWriter, ptl: &GeneralProfileTierLevel) {
        w.write_bits(u32::from(ptl.profile_space), 2);
        w.write_bit(u32::from(ptl.tier_flag));
        w.write_bits(u32::from(ptl.profile_idc), 5);
        w.write_bits(ptl.compatibility_flags, 32);
        for byte in &ptl.constraint_bytes {
            w.write_bits(u32::from(*byte), 8);
        }
        w.write_bits(u32::from(ptl.level_idc), 8);
    }

    /// A 640x480 Annex-B SPS carrying `ptl` in its general
    /// `profile_tier_level`, for the codec-string tests.
    fn build_annexb_sps_with_ptl(ptl: &GeneralProfileTierLevel) -> Vec<u8> {
        finish_sps(sps_head(640, 480, 1, None, ptl))
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
    /// `timing_info`: zero/absent optional blocks, then
    /// `vui_num_units_in_tick` / `vui_time_scale`. `color` optionally writes a
    /// `video_signal_type` block with a CICP colour description
    /// `(primaries, transfer, matrix, full_range)` ahead of the timing.
    fn build_annexb_sps_with_vui(
        pic_w: u32,
        pic_h: u32,
        num_units_in_tick: u32,
        time_scale: u32,
        color: Option<(u8, u8, u8, bool)>,
    ) -> Vec<u8> {
        let mut w = sps_head(pic_w, pic_h, 1, None, &ZERO_PTL);
        write_sps_middle(&mut w, &plain_middle(), 0);
        write_vui(&mut w, num_units_in_tick, time_scale, color);
        finish_sps(w)
    }

    // codes only the optional video_signal_type block and timing_info
    fn write_vui(
        w: &mut BitWriter,
        num_units_in_tick: u32,
        time_scale: u32,
        color: Option<(u8, u8, u8, bool)>,
    ) {
        w.write_bit(1); // vui_parameters_present_flag
        w.write_bit(0); // aspect_ratio_info_present_flag
        w.write_bit(0); // overscan_info_present_flag
        match color {
            Some((primaries, transfer, matrix, full_range)) => {
                w.write_bit(1); // video_signal_type_present_flag
                w.write_bits(5, 3); // video_format (unspecified)
                w.write_bit(u32::from(full_range));
                w.write_bit(1); // colour_description_present_flag
                w.write_bits(u32::from(primaries), 8);
                w.write_bits(u32::from(transfer), 8);
                w.write_bits(u32::from(matrix), 8);
            }
            None => w.write_bit(0), // video_signal_type_present_flag
        }
        w.write_bit(0); // chroma_loc_info_present_flag
        w.write_bit(0); // neutral_chroma_indication_flag
        w.write_bit(0); // field_seq_flag
        w.write_bit(0); // frame_field_info_present_flag
        w.write_bit(0); // default_display_window_flag
        w.write_bit(1); // vui_timing_info_present_flag
        w.write_bits(num_units_in_tick, 32);
        w.write_bits(time_scale, 32);
    }

    // every optional block absent, one sub-layer with zero ordering limits
    fn plain_middle() -> H265SpsMiddle {
        H265SpsMiddle {
            log2_max_pic_order_cnt_lsb_minus4: Some(0),
            sub_layer_ordering: Some(H265SubLayerOrdering {
                info_present_flag: true,
                ..H265SubLayerOrdering::default()
            }),
            ..H265SpsMiddle::default()
        }
    }

    fn write_sps_middle(w: &mut BitWriter, middle: &H265SpsMiddle, highest_sub_layer: usize) {
        write_sps_middle_through_short_term_refs(w, middle, highest_sub_layer);
        write_sps_middle_from_long_term_refs(w, middle);
    }

    fn log2_max_pic_order_cnt_lsb_minus4(middle: &H265SpsMiddle) -> u32 {
        middle
            .log2_max_pic_order_cnt_lsb_minus4
            .expect("the writer needs the order-count lsb width")
    }

    fn write_sps_middle_through_short_term_refs(
        w: &mut BitWriter,
        middle: &H265SpsMiddle,
        highest_sub_layer: usize,
    ) {
        w.write_ue(middle.bit_depth_luma_minus8);
        w.write_ue(middle.bit_depth_chroma_minus8);
        w.write_ue(log2_max_pic_order_cnt_lsb_minus4(middle));
        let ordering = middle
            .sub_layer_ordering
            .expect("the writer needs the sub-layer ordering");
        w.write_bit(u32::from(ordering.info_present_flag));
        let first_coded = if ordering.info_present_flag {
            0
        } else {
            highest_sub_layer
        };
        for sub_layer in first_coded..=highest_sub_layer {
            w.write_ue(ordering.max_dec_pic_buffering_minus1[sub_layer]);
            w.write_ue(ordering.max_num_reorder_pics[sub_layer]);
            w.write_ue(ordering.max_latency_increase_plus1[sub_layer]);
        }
        w.write_ue(middle.log2_min_luma_coding_block_size_minus3);
        w.write_ue(middle.log2_diff_max_min_luma_coding_block_size);
        w.write_ue(middle.log2_min_luma_transform_block_size_minus2);
        w.write_ue(middle.log2_diff_max_min_luma_transform_block_size);
        w.write_ue(middle.max_transform_hierarchy_depth_inter);
        w.write_ue(middle.max_transform_hierarchy_depth_intra);
        w.write_bit(u32::from(middle.scaling_list_enabled_flag));
        if middle.scaling_list_enabled_flag {
            w.write_bit(u32::from(middle.sps_scaling_list_data_present_flag));
            if middle.sps_scaling_list_data_present_flag {
                write_scaling_list_data(w);
            }
        }
        w.write_bit(u32::from(middle.amp_enabled_flag));
        w.write_bit(u32::from(middle.sample_adaptive_offset_enabled_flag));
        w.write_bit(u32::from(middle.pcm_enabled_flag));
        if middle.pcm_enabled_flag {
            w.write_bits(middle.pcm_sample_bit_depth_luma_minus1, 4);
            w.write_bits(middle.pcm_sample_bit_depth_chroma_minus1, 4);
            w.write_ue(middle.log2_min_pcm_luma_coding_block_size_minus3);
            w.write_ue(middle.log2_diff_max_min_pcm_luma_coding_block_size);
            w.write_bit(u32::from(middle.pcm_loop_filter_disabled_flag));
        }
        w.write_ue(middle.short_term_rps.len() as u32);
        for (idx, set) in middle.short_term_rps.iter().enumerate() {
            write_explicit_short_term_rps(w, idx, set);
        }
    }

    fn write_sps_middle_from_long_term_refs(w: &mut BitWriter, middle: &H265SpsMiddle) {
        w.write_bit(u32::from(middle.long_term_ref_pics_present_flag));
        if middle.long_term_ref_pics_present_flag {
            w.write_ue(middle.num_long_term_ref_pics_sps);
            for lt in 0..middle.num_long_term_ref_pics_sps as usize {
                w.write_bits(
                    middle.lt_ref_pic_poc_lsb_sps[lt],
                    log2_max_pic_order_cnt_lsb_minus4(middle) + 4,
                );
                w.write_bit(u32::from(middle.used_by_curr_pic_lt_sps_flag[lt]));
            }
        }
        w.write_bit(u32::from(middle.sps_temporal_mvp_enabled_flag));
        w.write_bit(u32::from(middle.strong_intra_smoothing_enabled_flag));
    }

    fn write_explicit_short_term_rps(w: &mut BitWriter, idx: usize, set: &H265ShortTermRps) {
        if idx != 0 {
            w.write_bit(0); // inter_ref_pic_set_prediction_flag
        }
        w.write_ue(u32::from(set.num_negative_pics));
        w.write_ue(u32::from(set.num_positive_pics));
        let mut previous = 0i32;
        for k in 0..usize::from(set.num_negative_pics) {
            w.write_ue((previous - set.delta_poc_s0[k] - 1) as u32); // delta_poc_s0_minus1
            w.write_bit(u32::from(set.used_s0[k]));
            previous = set.delta_poc_s0[k];
        }
        let mut previous = 0i32;
        for k in 0..usize::from(set.num_positive_pics) {
            w.write_ue((set.delta_poc_s1[k] - previous - 1) as u32); // delta_poc_s1_minus1
            w.write_bit(u32::from(set.used_s1[k]));
            previous = set.delta_poc_s1[k];
        }
    }

    // H.265 7.3.4: a 4x4 list codes 16 coefficients, the larger ones an 8x8 subsample
    const SCALING_LIST_COEFFICIENTS: [usize; 4] = [16, 64, 64, 64];
    const SCALING_LIST_MATRICES: [usize; 4] = [6, 6, 6, 2];

    // even matrices are explicit with every delta +1, odd ones copy the previous matrix
    fn write_scaling_list_data(w: &mut BitWriter) {
        for size_id in 0..SCALING_LIST_COEFFICIENTS.len() {
            for matrix_id in 0..SCALING_LIST_MATRICES[size_id] {
                let explicit = matrix_id % 2 == 0;
                w.write_bit(u32::from(explicit)); // scaling_list_pred_mode_flag
                if !explicit {
                    w.write_ue(1); // scaling_list_pred_matrix_id_delta
                    continue;
                }
                if size_id > 1 {
                    w.write_ue(1); // scaling_list_dc_coef_minus8 = se(+1)
                }
                for _ in 0..SCALING_LIST_COEFFICIENTS[size_id] {
                    w.write_ue(1); // scaling_list_delta_coef = se(+1)
                }
            }
        }
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

    /// Main profile, main tier, level 3.1, progressive + non-packed +
    /// frame-only: the everyday HEVC stream, whose codec string the WebCodecs
    /// HEVC registration (ISO/IEC 14496-15 Annex E.3) spells `hev1.1.6.L93.B0`.
    #[test]
    fn codec_string_for_main_profile_level_3_1() {
        let ptl = GeneralProfileTierLevel {
            profile_idc: 1,
            // Compatible with profiles 1 and 2; reverse bit order makes it 0x6.
            compatibility_flags: 0x6000_0000,
            constraint_bytes: [0xB0, 0, 0, 0, 0, 0],
            level_idc: 93,
            ..ZERO_PTL
        };
        let stream = build_annexb_sps_with_ptl(&ptl);
        assert_eq!(
            h265_codec_string(&stream).as_deref(),
            Some("hev1.1.6.L93.B0")
        );
        // The geometry parse still crosses the block it now reads.
        let info = H265Codec::extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (640, 480));
    }

    /// Profile space 1 ("A"), high tier ("H"), and a second nonzero constraint
    /// byte, which must be emitted after the first.
    #[test]
    fn codec_string_keeps_constraint_bytes_up_to_the_last_nonzero() {
        let ptl = GeneralProfileTierLevel {
            profile_space: 1,
            tier_flag: 1,
            profile_idc: 4,
            compatibility_flags: 0x0800_0000,
            constraint_bytes: [0x90, 0x08, 0, 0, 0, 0],
            level_idc: 120,
        };
        let stream = build_annexb_sps_with_ptl(&ptl);
        assert_eq!(
            h265_codec_string(&stream).as_deref(),
            Some("hev1.A4.10.H120.90.08")
        );
    }

    /// All six constraint bytes zero: every one is a trailing zero, so the
    /// string ends at the level.
    #[test]
    fn codec_string_omits_all_zero_constraint_bytes() {
        let ptl = GeneralProfileTierLevel {
            profile_idc: 2,
            compatibility_flags: 0x4000_0000,
            level_idc: 63,
            ..ZERO_PTL
        };
        let stream = build_annexb_sps_with_ptl(&ptl);
        assert_eq!(h265_codec_string(&stream).as_deref(), Some("hev1.2.2.L63"));
    }

    #[test]
    fn codec_string_none_without_sps() {
        // A lone IDR_W_RADL slice NAL (type 19), no SPS.
        let au = [0u8, 0, 0, 1, 0x26, 0x01, 0xAF];
        assert_eq!(h265_codec_string(&au), None);
    }

    #[test]
    fn recovers_dimensions_from_sps() {
        let stream = build_annexb_sps(1920, 1080, 1, None);
        let info = H265Codec::extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
        // The RBSP ends at the conformance window, so the VUI walk truncates:
        // geometry still resolves, the rate stays unrefined (best-effort M663).
        assert_eq!(info.framerate, None);
    }

    #[test]
    fn recovers_bt709_colorimetry_from_vui() {
        // CICP 1/1/1 + limited range is exactly the bt709 preset.
        let stream = build_annexb_sps_with_vui(1920, 1080, 1, 25, Some((1, 1, 1, false)));
        let info = H265Codec::extract_sps_info(&stream).expect("SPS with colour must parse");
        assert_eq!(info.colorimetry, Colorimetry::BT709);
    }

    #[test]
    fn absent_signal_type_colorimetry_stays_unknown() {
        let stream = build_annexb_sps_with_vui(1920, 1080, 1, 25, None);
        let info = H265Codec::extract_sps_info(&stream).expect("SPS must parse");
        assert_eq!(info.colorimetry, Colorimetry::UNKNOWN);
    }

    #[test]
    fn garbage_colour_codepoints_map_unknown_without_panic() {
        // Unmodeled CICP codepoints fall back to Unknown per field; only the
        // coded full-range flag stays concrete.
        let stream = build_annexb_sps_with_vui(1920, 1080, 1, 25, Some((222, 199, 33, true)));
        let info = H265Codec::extract_sps_info(&stream).expect("SPS must parse");
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

    #[test]
    fn recovers_framerate_from_vui_timing_info() {
        // 25 fps: one tick per picture at a 25 Hz time scale (no H.264-style
        // field doubling in HEVC).
        let stream = build_annexb_sps_with_vui(1920, 1080, 1, 25, None);
        let info = H265Codec::extract_sps_info(&stream).expect("SPS with VUI must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.framerate, Some(25 << 16), "Q16 fps from timing_info");

        // 29.97 fps (30000/1001): the Q16 value is the truncated division.
        let ntsc = build_annexb_sps_with_vui(720, 480, 1001, 30_000, None);
        let info = H265Codec::extract_sps_info(&ntsc).expect("NTSC SPS must parse");
        assert_eq!(info.framerate, Some(((30_000u64 << 16) / 1001) as u32));
    }

    #[test]
    fn applies_conformance_window_cropping() {
        // 1920x1088 coded, 4:2:0 (SubHeightC = 2), crop 4 chroma rows off the
        // bottom -> 1088 - 2*4 = 1080.
        let stream = build_annexb_sps(1920, 1088, 1, Some((0, 0, 0, 4)));
        let info = H265Codec::extract_sps_info(&stream).expect("SPS with conf window must parse");
        assert_eq!((info.width, info.height), (1920, 1080));
    }

    #[test]
    fn saturates_adversarial_conformance_offsets() {
        // Huge conformance offsets must saturate, not overflow-panic on the sum.
        let huge = 3_000_000_000u32;
        let stream = build_annexb_sps(1920, 1080, 1, Some((huge, huge, huge, huge)));
        let info = H265Codec::extract_sps_info(&stream).expect("parses without overflow");
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
        let info = H265Codec::extract_sps_info(&hvcc).expect("length-prefixed SPS must parse");
        assert_eq!((info.width, info.height), (1280, 720));
    }

    #[test]
    fn ignores_non_sps_nals() {
        // A TRAIL_R slice NAL (type 1 -> first byte 0x02) carries no SPS.
        let stream = [0u8, 0, 0, 1, 0x02, 0x01, 0xAA, 0xBB];
        assert!(H265Codec::extract_sps_info(&stream).is_none());
    }

    #[test]
    fn returns_none_on_empty_input() {
        assert!(H265Codec::extract_sps_info(&[]).is_none());
    }

    #[test]
    fn profile_tier_level_advances_96_bits_for_single_layer() {
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
        read_profile_tier_level(&mut br, 0).expect("reads the fixed 96-bit block");
        assert_eq!(br.read_ue(), Some(0), "reader landed just past the PTL");
    }

    // -- Element-level tests (drive H265Parse::process directly) -----------

    fn h265_caps() -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        }
    }

    #[tokio::test]
    async fn emits_caps_changed_before_first_data_frame() {
        let mut parse = H265Parse::new();
        parse.configure_pipeline(&h265_caps()).unwrap();
        let mut sink = CollectSink::default();

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
        let mut sink = CollectSink::default();

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
        let mut sink = CollectSink::default();

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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
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
                        colorimetry: g2g_core::Colorimetry::UNKNOWN,
                        alignment: g2g_core::BitstreamAlignment::Unspecified,
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
        // Picture A: two slices (first then a continuation). Picture B: one slice.
        // Two access units, not three.
        let mut stream = annexb_vcl_h265(true, 1);
        stream.extend_from_slice(&annexb_vcl_h265(false, 2)); // same picture
        let b_off = stream.len();
        stream.extend_from_slice(&annexb_vcl_h265(true, 3)); // new picture
        let starts = access_unit_starts(&stream, true, H265Codec::unit_role);
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
        let mut sink = CollectSink::default();

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
        let mut sink = CollectSink::default();

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
        let mut sink = CollectSink::default();

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

    #[test]
    fn codec_string_spells_profile_spaces_b_and_c() {
        let ptl = GeneralProfileTierLevel {
            profile_space: 2,
            profile_idc: 1,
            compatibility_flags: 0x6000_0000,
            level_idc: 93,
            ..ZERO_PTL
        };
        let stream = build_annexb_sps_with_ptl(&ptl);
        assert_eq!(h265_codec_string(&stream).as_deref(), Some("hev1.B1.6.L93"));
        let ptl = GeneralProfileTierLevel {
            profile_space: 3,
            ..ptl
        };
        let stream = build_annexb_sps_with_ptl(&ptl);
        assert_eq!(h265_codec_string(&stream).as_deref(), Some("hev1.C1.6.L93"));
    }

    #[test]
    fn source_flags_read_the_first_constraint_byte_most_significant_bit_first() {
        // progressive, interlaced, non-packed, frame-only
        let flags_by_byte = [
            (0b1000_0000, [1, 0, 0, 0]),
            (0b0100_0000, [0, 1, 0, 0]),
            (0b0010_0000, [0, 0, 1, 0]),
            (0b0001_0000, [0, 0, 0, 1]),
            (0b1011_1111, [1, 0, 1, 1]),
        ];
        for (first_byte, expected) in flags_by_byte {
            let ptl = GeneralProfileTierLevel {
                constraint_bytes: [first_byte, 0, 0, 0, 0, 0],
                ..ZERO_PTL
            };
            let flags = [
                ptl.progressive_source_flag(),
                ptl.interlaced_source_flag(),
                ptl.non_packed_constraint_flag(),
                ptl.frame_only_constraint_flag(),
            ];
            assert_eq!(flags, expected, "constraint byte {first_byte:#010b}");
        }
    }

    #[test]
    fn profile_tier_level_crosses_coded_sub_layer_blocks() {
        let mut w = BitWriter::default();
        write_profile_tier_level(&mut w, &ZERO_PTL);
        let max_sub_layers_minus1 = 2;
        // sub-layer 0 codes profile and level, sub-layer 1 only a level
        w.write_bit(1);
        w.write_bit(1);
        w.write_bit(0);
        w.write_bit(1);
        for _ in max_sub_layers_minus1..8 {
            w.write_bits(0, 2); // reserved_zero_2bits
        }
        // sub_layer profile/tier block of 88 bits, then the 8-bit level
        w.write_bits(0, 32);
        w.write_bits(0, 32);
        w.write_bits(0, 24);
        w.write_bits(0, 8); // sub_layer_level_idc[0]
        w.write_bits(0, 8); // sub_layer_level_idc[1]
        w.write_bit(1); // a marker the reader should land on (ue value 0)
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        read_profile_tier_level(&mut br, max_sub_layers_minus1).expect("reads every block");
        assert_eq!(
            br.read_ue(),
            Some(0),
            "reader landed just past the sub-layer blocks"
        );
        assert_eq!(br.read_bit(), Some(0), "only alignment padding is left");
    }

    fn head_with(chroma_format_idc: u32, separate_colour_plane_flag: bool) -> H265SpsHead {
        H265SpsHead {
            sps_video_parameter_set_id: 5,
            sps_max_sub_layers_minus1: 0,
            sps_temporal_id_nesting_flag: true,
            ptl: ZERO_PTL,
            sps_seq_parameter_set_id: 7,
            chroma_format_idc,
            separate_colour_plane_flag,
            pic_width_in_luma_samples: 1920,
            pic_height_in_luma_samples: 1088,
            conformance_window_flag: true,
            conformance_window: [0, 0, 0, 4],
        }
    }

    fn annexb_sps(head: &H265SpsHead, middle: &H265SpsMiddle, frame_rate: (u32, u32)) -> Vec<u8> {
        let (num_units_in_tick, time_scale) = frame_rate;
        let mut w = write_sps_head(head);
        write_sps_middle(&mut w, middle, usize::from(head.sps_max_sub_layers_minus1));
        write_vui(&mut w, num_units_in_tick, time_scale, None);
        finish_sps(w)
    }

    const FRAME_RATE_25: (u32, u32) = (1, 25);
    const FRAME_RATE_25_Q16: u32 = 25 << 16;

    #[test]
    fn sps_head_reads_every_field_as_coded() {
        let coded = head_with(3, true);
        let stream = annexb_sps(&coded, &plain_middle(), FRAME_RATE_25);
        let rbsp = strip_emulation_prevention(&stream[ANNEX_B_SPS_HEADER_BYTES..]);
        let (head, _) = parse_h265_sps_head(&rbsp).expect("SPS head parses");
        assert_eq!(
            head.sps_video_parameter_set_id,
            coded.sps_video_parameter_set_id
        );
        assert_eq!(
            head.sps_max_sub_layers_minus1,
            coded.sps_max_sub_layers_minus1
        );
        assert!(head.sps_temporal_id_nesting_flag);
        assert_eq!(
            head.sps_seq_parameter_set_id,
            coded.sps_seq_parameter_set_id
        );
        assert_eq!(head.chroma_format_idc, coded.chroma_format_idc);
        assert!(head.separate_colour_plane_flag);
        assert_eq!(
            head.pic_width_in_luma_samples,
            coded.pic_width_in_luma_samples
        );
        assert_eq!(
            head.pic_height_in_luma_samples,
            coded.pic_height_in_luma_samples
        );
        assert!(head.conformance_window_flag);
        assert_eq!(head.conformance_window, coded.conformance_window);
    }

    // start code and the two NAL header bytes
    const ANNEX_B_SPS_HEADER_BYTES: usize = 6;

    #[test]
    fn separate_colour_planes_and_poc_lsb_width_reach_the_poc_fields() {
        // the slice header codes colour_plane_id only under the flag
        let mut middle = plain_middle();
        middle.log2_max_pic_order_cnt_lsb_minus4 = Some(2);
        let stream = annexb_sps(&head_with(3, true), &middle, FRAME_RATE_25);
        let info = H265Codec::extract_sps_info(&stream).expect("SPS parses");
        let expected = crate::poc::SpsPocParams {
            log2_max_pic_order_cnt_lsb: 2 + 4,
            separate_colour_plane_flag: true,
            ..crate::poc::SpsPocParams::default()
        };
        assert_eq!(info.poc, Some(expected));
    }

    #[test]
    fn four_two_two_conformance_window_crops_in_chroma_units() {
        // SubWidthC 2, SubHeightC 1: 4 units right are 8 columns, 4 units bottom 4 rows
        let stream = build_annexb_sps(1928, 1084, 2, Some((0, 4, 0, 4)));
        let info = H265Codec::extract_sps_info(&stream).expect("SPS parses");
        assert_eq!((info.width, info.height), (1920, 1080));
    }

    #[test]
    fn only_zero_reorder_pictures_present_in_decode_order() {
        let stream = annexb_sps(&head_with(1, false), &plain_middle(), FRAME_RATE_25);
        let info = H265Codec::extract_sps_info(&stream).expect("SPS parses");
        assert!(info.presents_in_decode_order);
        let mut middle = plain_middle();
        let mut ordering = middle
            .sub_layer_ordering
            .expect("plain middle has ordering");
        ordering.max_num_reorder_pics[0] = 2;
        middle.sub_layer_ordering = Some(ordering);
        let stream = annexb_sps(&head_with(1, false), &middle, FRAME_RATE_25);
        let info = H265Codec::extract_sps_info(&stream).expect("SPS parses");
        assert!(!info.presents_in_decode_order);
    }

    #[test]
    fn poc_lsb_width_is_bounded() {
        let mut middle = plain_middle();
        middle.log2_max_pic_order_cnt_lsb_minus4 = Some(crate::poc::MAX_LOG2_MINUS4);
        let stream = annexb_sps(&head_with(1, false), &middle, FRAME_RATE_25);
        let info = H265Codec::extract_sps_info(&stream).expect("SPS parses");
        let lsb_width = info.poc.map(|poc| poc.log2_max_pic_order_cnt_lsb);
        assert_eq!(lsb_width, Some(crate::poc::MAX_LOG2_MINUS4 + 4));
        assert_eq!(info.framerate, Some(FRAME_RATE_25_Q16));

        middle.log2_max_pic_order_cnt_lsb_minus4 = Some(crate::poc::MAX_LOG2_MINUS4 + 1);
        let stream = annexb_sps(&head_with(1, false), &middle, FRAME_RATE_25);
        let info = H265Codec::extract_sps_info(&stream).expect("geometry still parses");
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.poc, None);
        assert_eq!(info.framerate, None);
    }

    #[test]
    fn sub_layer_ordering_reads_every_coded_sub_layer() {
        let mut head = head_with(1, false);
        head.sps_max_sub_layers_minus1 = 2;
        for info_present_flag in [true, false] {
            let mut middle = plain_middle();
            middle.sub_layer_ordering = Some(H265SubLayerOrdering {
                info_present_flag,
                max_dec_pic_buffering_minus1: [4, 4, 4, 0, 0, 0, 0],
                // only the highest sub-layer forbids reordering
                max_num_reorder_pics: [2, 1, 0, 0, 0, 0, 0],
                max_latency_increase_plus1: [0; H265_MAX_SUB_LAYERS],
            });
            let stream = annexb_sps(&head, &middle, FRAME_RATE_25);
            let info = H265Codec::extract_sps_info(&stream).expect("SPS parses");
            assert!(
                info.presents_in_decode_order,
                "info present {info_present_flag}"
            );
            assert_eq!(info.framerate, Some(FRAME_RATE_25_Q16));
        }
    }

    #[test]
    fn scaling_list_data_is_crossed_to_the_vui() {
        for data_present in [true, false] {
            let mut middle = plain_middle();
            middle.scaling_list_enabled_flag = true;
            middle.sps_scaling_list_data_present_flag = data_present;
            let stream = annexb_sps(&head_with(1, false), &middle, FRAME_RATE_25);
            let info = H265Codec::extract_sps_info(&stream).expect("SPS parses");
            assert_eq!(
                info.framerate,
                Some(FRAME_RATE_25_Q16),
                "scaling list data present {data_present}"
            );
        }
    }

    #[test]
    fn sps_middle_reads_every_field_as_coded() {
        let mut coded = plain_middle();
        coded.bit_depth_luma_minus8 = 2;
        coded.bit_depth_chroma_minus8 = 2;
        coded.amp_enabled_flag = true;
        coded.sample_adaptive_offset_enabled_flag = true;
        coded.pcm_enabled_flag = true;
        coded.pcm_sample_bit_depth_luma_minus1 = 7;
        coded.pcm_sample_bit_depth_chroma_minus1 = 6;
        coded.log2_min_pcm_luma_coding_block_size_minus3 = 1;
        coded.log2_diff_max_min_pcm_luma_coding_block_size = 2;
        coded.pcm_loop_filter_disabled_flag = true;
        coded.long_term_ref_pics_present_flag = true;
        coded.num_long_term_ref_pics_sps = 2;
        coded.lt_ref_pic_poc_lsb_sps[0] = 9;
        coded.lt_ref_pic_poc_lsb_sps[1] = 14;
        coded.used_by_curr_pic_lt_sps_flag[1] = true;
        coded.sps_temporal_mvp_enabled_flag = true;
        coded.strong_intra_smoothing_enabled_flag = true;
        let mut w = BitWriter::default();
        write_sps_middle(&mut w, &coded, 0);
        w.write_bit(1); // a marker the reader should land on
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        let mut middle = H265SpsMiddle::default();
        read_h265_sps_middle(&mut br, 0, &mut middle).expect("middle parses");
        assert_eq!(middle.bit_depth_luma_minus8, coded.bit_depth_luma_minus8);
        assert_eq!(
            middle.bit_depth_chroma_minus8,
            coded.bit_depth_chroma_minus8
        );
        assert!(middle.amp_enabled_flag);
        assert!(middle.sample_adaptive_offset_enabled_flag);
        assert!(middle.pcm_enabled_flag);
        assert_eq!(
            middle.pcm_sample_bit_depth_luma_minus1,
            coded.pcm_sample_bit_depth_luma_minus1
        );
        assert_eq!(
            middle.pcm_sample_bit_depth_chroma_minus1,
            coded.pcm_sample_bit_depth_chroma_minus1
        );
        assert_eq!(
            middle.log2_min_pcm_luma_coding_block_size_minus3,
            coded.log2_min_pcm_luma_coding_block_size_minus3
        );
        assert_eq!(
            middle.log2_diff_max_min_pcm_luma_coding_block_size,
            coded.log2_diff_max_min_pcm_luma_coding_block_size
        );
        assert!(middle.pcm_loop_filter_disabled_flag);
        assert!(middle.long_term_ref_pics_present_flag);
        assert_eq!(
            middle.num_long_term_ref_pics_sps,
            coded.num_long_term_ref_pics_sps
        );
        assert_eq!(middle.lt_ref_pic_poc_lsb_sps, coded.lt_ref_pic_poc_lsb_sps);
        assert_eq!(
            middle.used_by_curr_pic_lt_sps_flag,
            coded.used_by_curr_pic_lt_sps_flag
        );
        assert!(middle.sps_temporal_mvp_enabled_flag);
        assert!(middle.strong_intra_smoothing_enabled_flag);
        assert_eq!(br.read_bit(), Some(1), "reader landed on the marker");
    }

    #[test]
    fn short_term_ref_pic_set_count_is_bounded() {
        let one_back = H265ShortTermRps {
            num_negative_pics: 1,
            delta_poc_s0: [-1; 16],
            used_s0: [true; 16],
            ..H265ShortTermRps::default()
        };
        let mut middle = plain_middle();
        middle.short_term_rps = vec![one_back; H265_MAX_SHORT_TERM_REF_PIC_SETS as usize];
        let stream = annexb_sps(&head_with(1, false), &middle, FRAME_RATE_25);
        let info = H265Codec::extract_sps_info(&stream).expect("SPS parses");
        assert_eq!(info.framerate, Some(FRAME_RATE_25_Q16));

        middle.short_term_rps.push(middle.short_term_rps[0].clone());
        let stream = annexb_sps(&head_with(1, false), &middle, FRAME_RATE_25);
        let info = H265Codec::extract_sps_info(&stream).expect("geometry still parses");
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.framerate, None);
    }

    #[test]
    fn long_term_ref_pic_count_is_bounded() {
        let mut middle = plain_middle();
        middle.long_term_ref_pics_present_flag = true;
        middle.num_long_term_ref_pics_sps = H265_MAX_LONG_TERM_REF_PICS_SPS as u32;
        let stream = annexb_sps(&head_with(1, false), &middle, FRAME_RATE_25);
        let info = H265Codec::extract_sps_info(&stream).expect("SPS parses");
        assert_eq!(info.framerate, Some(FRAME_RATE_25_Q16));

        // past the array bound, so coded by hand from the long-term block on
        let middle = plain_middle();
        let mut w = write_sps_head(&head_with(1, false));
        write_sps_middle_through_short_term_refs(&mut w, &middle, 0);
        w.write_bit(1); // long_term_ref_pics_present_flag
        let too_many = H265_MAX_LONG_TERM_REF_PICS_SPS as u32 + 1;
        w.write_ue(too_many);
        for _ in 0..too_many {
            w.write_bits(0, log2_max_pic_order_cnt_lsb_minus4(&middle) + 4); // lt_ref_pic_poc_lsb_sps
            w.write_bit(0); // used_by_curr_pic_lt_sps_flag
        }
        w.write_bit(0); // sps_temporal_mvp_enabled_flag
        w.write_bit(0); // strong_intra_smoothing_enabled_flag
        write_vui(&mut w, FRAME_RATE_25.0, FRAME_RATE_25.1, None);
        let stream = finish_sps(w);
        let info = H265Codec::extract_sps_info(&stream).expect("geometry still parses");
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.framerate, None);
    }

    #[test]
    fn a_one_byte_nal_is_skipped_without_reading_past_it() {
        let mut stream = vec![0u8, 0, 0, 1, 0x42];
        stream.extend_from_slice(&build_annexb_sps(1280, 720, 1, None));
        let info = H265Codec::extract_sps_info(&stream).expect("the later SPS parses");
        assert_eq!((info.width, info.height), (1280, 720));
    }

    // (POC delta, used by the current picture) per picture, negative then positive
    type RpsPictures = (Vec<(i32, bool)>, Vec<(i32, bool)>);

    fn short_term_set(negative: &[(i32, bool)], positive: &[(i32, bool)]) -> H265ShortTermRps {
        let mut set = H265ShortTermRps {
            num_negative_pics: negative.len() as u8,
            num_positive_pics: positive.len() as u8,
            ..H265ShortTermRps::default()
        };
        for (k, &(delta, used)) in negative.iter().enumerate() {
            set.delta_poc_s0[k] = delta;
            set.used_s0[k] = used;
        }
        for (k, &(delta, used)) in positive.iter().enumerate() {
            set.delta_poc_s1[k] = delta;
            set.used_s1[k] = used;
        }
        set
    }

    fn pictures(set: &H265ShortTermRps) -> RpsPictures {
        let negative = usize::from(set.num_negative_pics);
        let positive = usize::from(set.num_positive_pics);
        (
            set.delta_poc_s0[..negative]
                .iter()
                .copied()
                .zip(set.used_s0[..negative].iter().copied())
                .collect(),
            set.delta_poc_s1[..positive]
                .iter()
                .copied()
                .zip(set.used_s1[..positive].iter().copied())
                .collect(),
        )
    }

    // flags holds (used_by_curr_pic_flag, use_delta_flag) per reference picture, then deltaRps
    fn inter_predicted_rps(
        delta_idx_minus1: Option<u32>,
        delta_rps: i32,
        flags: &[(bool, bool)],
    ) -> Vec<u8> {
        let mut w = BitWriter::default();
        w.write_bit(1); // inter_ref_pic_set_prediction_flag
        if let Some(delta_idx_minus1) = delta_idx_minus1 {
            w.write_ue(delta_idx_minus1);
        }
        w.write_bit(u32::from(delta_rps < 0)); // delta_rps_sign
        w.write_ue(delta_rps.unsigned_abs() - 1); // abs_delta_rps_minus1
        for &(used_by_curr_pic, use_delta) in flags {
            w.write_bit(u32::from(used_by_curr_pic));
            if !used_by_curr_pic {
                w.write_bit(u32::from(use_delta));
            }
        }
        w.write_bit(1); // a marker the reader should land on
        w.align_to_byte();
        w.into_bytes()
    }

    // two pictures before and three after, so S0 and S1 indices differ
    fn reference_set() -> H265ShortTermRps {
        short_term_set(
            &[(-1, true), (-3, true)],
            &[(2, true), (4, true), (5, true)],
        )
    }

    const REFERENCE_SET_PICTURES: u8 = 5;

    fn parse_rps(
        bytes: &[u8],
        idx: usize,
        num_sets: usize,
        prev: &[H265ShortTermRps],
    ) -> (RpsPictures, u8) {
        let mut br = BitReader::new(bytes);
        let (set, reference_pictures) =
            parse_h265_short_term_rps(&mut br, idx, num_sets, prev).expect("set parses");
        assert_eq!(br.read_bit(), Some(1), "reader landed on the marker");
        (pictures(&set), reference_pictures)
    }

    #[test]
    fn explicit_short_term_set_reads_deltas_and_flags() {
        let coded = short_term_set(&[(-1, true), (-4, false)], &[(2, false), (3, true)]);
        let mut w = BitWriter::default();
        write_explicit_short_term_rps(&mut w, 1, &coded);
        w.write_bit(1);
        w.align_to_byte();
        let (parsed, reference_pictures) = parse_rps(&w.into_bytes(), 1, 2, &[reference_set()]);
        assert_eq!(parsed, pictures(&coded));
        assert_eq!(
            reference_pictures, 0,
            "an explicit set references no other set"
        );
    }

    #[test]
    fn explicit_short_term_set_rejects_more_than_sixteen_pictures_a_side() {
        let too_many = 17;
        for (negative, positive) in [(too_many, 0), (0, too_many)] {
            let mut w = BitWriter::default();
            w.write_ue(negative); // num_negative_pics
            w.write_ue(positive); // num_positive_pics
            for _ in 0..negative + positive {
                w.write_ue(0); // delta_poc_s*_minus1
                w.write_bit(1); // used_by_curr_pic_s*_flag
            }
            w.align_to_byte();
            let bytes = w.into_bytes();
            let mut br = BitReader::new(&bytes);
            assert!(parse_h265_short_term_rps(&mut br, 0, 1, &[]).is_none());
        }
    }

    // H.265 7.4.8 for deltaRps -1 over S0 {-1, -3} and S1 {2, 4, 5}: S0 shifts to
    // {-2, -4}, S1 to {1, 3, 4}, and deltaRps itself is a picture at -1.
    const BACKWARD_FLAGS: [(bool, bool); 6] = [
        (true, true),
        (false, true),
        // dropped: S1[0] moved to 1
        (false, false),
        (true, true),
        (false, true),
        (true, true),
    ];

    fn backward_prediction() -> RpsPictures {
        (
            vec![(-1, true), (-2, true), (-4, false)],
            vec![(3, true), (4, false)],
        )
    }

    #[test]
    fn inter_predicted_set_shifts_the_reference_back() {
        let bytes = inter_predicted_rps(None, -1, &BACKWARD_FLAGS);
        let (parsed, reference_pictures) = parse_rps(&bytes, 1, 2, &[reference_set()]);
        assert_eq!(parsed, backward_prediction());
        assert_eq!(reference_pictures, REFERENCE_SET_PICTURES);
    }

    #[test]
    fn inter_predicted_set_shifts_the_reference_forward() {
        // deltaRps +2: S0 moves to {1, -1}, S1 to {4, 6, 7}, and +2 itself
        let mut flags = [(true, true); 6];
        flags[0] = (false, false);
        let bytes = inter_predicted_rps(None, 2, &flags);
        let (parsed, _) = parse_rps(&bytes, 1, 2, &[reference_set()]);
        assert_eq!(
            parsed,
            (
                vec![(-1, true)],
                vec![(2, true), (4, true), (6, true), (7, true)]
            )
        );
    }

    #[test]
    fn inter_predicted_set_drops_a_picture_moved_onto_the_current_one() {
        // deltaRps -4 moves S1[1] from 4 to 0, deltaRps +3 moves S0[1] from -3 to 0
        let flags = [(true, true); 6];
        let bytes = inter_predicted_rps(None, -4, &flags);
        let (parsed, _) = parse_rps(&bytes, 1, 2, &[reference_set()]);
        assert_eq!(
            parsed,
            (
                vec![(-2, true), (-4, true), (-5, true), (-7, true)],
                vec![(1, true)]
            )
        );
        let bytes = inter_predicted_rps(None, 3, &flags);
        let (parsed, _) = parse_rps(&bytes, 1, 2, &[reference_set()]);
        assert_eq!(
            parsed,
            (
                vec![],
                vec![(2, true), (3, true), (5, true), (7, true), (8, true)]
            )
        );
    }

    #[test]
    fn inter_predicted_set_moves_later_pictures_back_with_their_own_flags() {
        // deltaRps -6 moves all of S1 {2, 4, 5} before the current picture
        let flags = [
            (true, true),
            (false, false),
            (true, true),
            (false, true),
            (false, true),
            (true, true),
        ];
        let bytes = inter_predicted_rps(None, -6, &flags);
        let (parsed, _) = parse_rps(&bytes, 1, 2, &[reference_set()]);
        assert_eq!(
            parsed,
            (
                vec![(-1, false), (-2, false), (-4, true), (-6, true), (-7, true)],
                vec![]
            )
        );
    }

    #[test]
    fn slice_header_set_predicts_from_the_coded_reference_index() {
        // the slice's set sits at num_short_term_ref_pic_sets and names its reference
        let sets = [reference_set(), short_term_set(&[], &[])];
        let delta_idx_minus1 = 1;
        let bytes = inter_predicted_rps(Some(delta_idx_minus1), -1, &BACKWARD_FLAGS);
        let (parsed, reference_pictures) = parse_rps(&bytes, sets.len(), sets.len(), &sets);
        assert_eq!(parsed, backward_prediction());
        assert_eq!(reference_pictures, REFERENCE_SET_PICTURES);
    }

    #[test]
    fn inter_predicted_set_keeps_at_most_sixteen_pictures_a_side() {
        let side = 16;
        let negative: Vec<(i32, bool)> = (1..=side).map(|k| (-k, true)).collect();
        let positive: Vec<(i32, bool)> = (1..=side).map(|k| (k, true)).collect();
        let reference = short_term_set(&negative, &positive);
        let flags = vec![(true, true); 2 * side as usize + 1];
        // shifting past the far side moves all 33 candidates to one side
        let shift = side + 1;

        let bytes = inter_predicted_rps(None, -shift, &flags);
        let (parsed, _) = parse_rps(&bytes, 1, 2, core::slice::from_ref(&reference));
        assert_eq!(parsed, (negative.clone(), vec![]));

        let bytes = inter_predicted_rps(None, shift, &flags);
        let (parsed, _) = parse_rps(&bytes, 1, 2, core::slice::from_ref(&reference));
        assert_eq!(parsed, (vec![], positive));
    }
}
