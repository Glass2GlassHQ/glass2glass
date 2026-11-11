// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use g2g_core::caps::framerate_q16_from_fraction;
use g2g_core::{AspectRatio, Colorimetry, PixelAspect};

use crate::annexb::BitReader;
use crate::sei::PicTimingContext;
use crate::startcodeparse::sample_aspect;

/// CICP codepoint 2, "unspecified", for a colour description the VUI omits.
const UNSPECIFIED_CICP: u8 = 2;

/// `aspect_ratio_idc` that codes `sar_width` / `sar_height` explicitly.
const EXTENDED_SAR: u32 = 255;

/// The last `aspect_ratio_idc` H.264 Table E-1 and H.265 Table E.1 define.
const HIGHEST_DEFINED_ASPECT_RATIO_IDC: u32 = 16;

/// `cpb_cnt_minus1` is 0..=31 by spec, a larger value means a desynced parse.
const MAX_CPB_COUNT_MINUS1: u32 = 31;

/// A reorder bound past the 16-frame DPB ceiling means a desynced parse.
const MAX_REORDER_FRAMES: u32 = 16;

/// The `video_signal_type` colour description, as CICP codepoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VideoSignalType {
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub video_full_range_flag: bool,
}

impl VideoSignalType {
    pub(crate) const UNSPECIFIED: Self = Self {
        color_primaries: UNSPECIFIED_CICP,
        transfer_characteristics: UNSPECIFIED_CICP,
        matrix_coefficients: UNSPECIFIED_CICP,
        video_full_range_flag: false,
    };

    pub(crate) fn colorimetry(self) -> Colorimetry {
        Colorimetry::from_cicp(
            self.color_primaries,
            self.transfer_characteristics,
            self.matrix_coefficients,
            self.video_full_range_flag,
        )
    }
}

/// The head of an H.264 / H.265 VUI, laid out the same in both codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VuiHead {
    /// `Any` when the VUI leaves the sample aspect unspecified.
    pub pixel_aspect: PixelAspect,
    /// `None` when the VUI carries no `video_signal_type`.
    pub signal_type: Option<VideoSignalType>,
}

impl VuiHead {
    pub(crate) const ABSENT: Self = Self {
        pixel_aspect: PixelAspect::Any,
        signal_type: None,
    };

    pub(crate) fn colorimetry(self) -> Colorimetry {
        self.signal_type
            .map_or(Colorimetry::UNKNOWN, VideoSignalType::colorimetry)
    }
}

/// Read the head of an H.264 / H.265 VUI: aspect ratio, overscan, then
/// `video_signal_type`. Reads from just after `vui_parameters_present_flag`.
/// `None` on truncation.
pub(crate) fn read_vui_head(br: &mut BitReader) -> Option<VuiHead> {
    // aspect_ratio_info_present_flag
    let pixel_aspect = if br.read_bit()? == 1 {
        read_sample_aspect(br)?
    } else {
        PixelAspect::Any
    };
    // overscan_info_present_flag
    if br.read_bit()? == 1 {
        br.read_bit()?; // overscan_appropriate_flag
    }
    // video_signal_type_present_flag
    if br.read_bit()? == 0 {
        return Some(VuiHead {
            pixel_aspect,
            signal_type: None,
        });
    }
    br.read_bits(3)?; // video_format
    let mut signal = VideoSignalType {
        video_full_range_flag: br.read_bit()? == 1,
        ..VideoSignalType::UNSPECIFIED
    };
    // colour_description_present_flag
    if br.read_bit()? == 1 {
        signal.color_primaries = br.read_bits(8)? as u8;
        signal.transfer_characteristics = br.read_bits(8)? as u8;
        signal.matrix_coefficients = br.read_bits(8)? as u8;
    }
    Some(VuiHead {
        pixel_aspect,
        signal_type: Some(signal),
    })
}

fn read_sample_aspect(br: &mut BitReader) -> Option<PixelAspect> {
    let aspect_ratio_idc = br.read_bits(8)?;
    let ratio = if aspect_ratio_idc == EXTENDED_SAR {
        let sar_width = br.read_bits(16)?;
        let sar_height = br.read_bits(16)?;
        AspectRatio::new(sar_width, sar_height)
    } else {
        sample_aspect(aspect_ratio_idc, HIGHEST_DEFINED_ASPECT_RATIO_IDC)
    };
    Some(ratio.map_or(PixelAspect::Any, PixelAspect::Exact))
}

/// Walk an H.264 `hrd_parameters()` (E.1.2), returning the three delay bit
/// widths a `pic_timing` SEI is sized by: `(cpb_removal_delay_length,
/// dpb_output_delay_length, time_offset_length)`.
pub(crate) fn read_h264_hrd(br: &mut BitReader) -> Option<(u8, u8, u8)> {
    let cpb_cnt_minus1 = br.read_ue()?;
    if cpb_cnt_minus1 > MAX_CPB_COUNT_MINUS1 {
        return None;
    }
    br.read_bits(4)?; // bit_rate_scale
    br.read_bits(4)?; // cpb_size_scale
    for _ in 0..=cpb_cnt_minus1 {
        br.read_ue()?; // bit_rate_value_minus1
        br.read_ue()?; // cpb_size_value_minus1
        br.read_bit()?; // cbr_flag
    }
    br.read_bits(5)?; // initial_cpb_removal_delay_length_minus1
    let cpb_removal = br.read_bits(5)? as u8 + 1;
    let dpb_output = br.read_bits(5)? as u8 + 1;
    let time_offset = br.read_bits(5)? as u8;
    Some((cpb_removal, dpb_output, time_offset))
}

/// What the rest of an H.264 VUI yields after [`read_vui_head`]. A VUI
/// that ends early keeps whatever was read before the truncation.
#[derive(Debug, Default)]
pub(crate) struct H264VuiTail {
    /// `time_scale / (2 * num_units_in_tick)` as Q16 fixed-point fps.
    pub framerate: Option<u32>,
    pub pic_timing: Option<PicTimingContext>,
    /// `bitstream_restriction` `max_num_reorder_frames`.
    pub max_num_reorder_frames: Option<u8>,
    pub max_dec_frame_buffering: Option<u8>,
}

/// Walk an H.264 VUI from where [`read_vui_head`] stopped.
pub(crate) fn read_h264_vui_tail(br: &mut BitReader) -> H264VuiTail {
    let mut tail = H264VuiTail::default();
    let _ = walk_h264_vui_tail(br, &mut tail);
    tail
}

fn walk_h264_vui_tail(br: &mut BitReader, tail: &mut H264VuiTail) -> Option<()> {
    // chroma_loc_info_present_flag
    if br.read_bit()? == 1 {
        br.read_ue()?; // chroma_sample_loc_type_top_field
        br.read_ue()?; // chroma_sample_loc_type_bottom_field
    }
    // timing_info_present_flag
    if br.read_bit()? == 1 {
        let num_units_in_tick = br.read_bits(32)?;
        let time_scale = br.read_bits(32)?;
        br.read_bit()?; // fixed_frame_rate_flag
                        // H.264 counts field ticks, two per frame
        tail.framerate = num_units_in_tick
            .checked_mul(2)
            .and_then(|frame_ticks| framerate_q16_from_fraction(time_scale, frame_ticks))
            .filter(|&q16| q16 > 0);
    }
    // Either HRD block sizes the pic_timing delays and sets CpbDpbDelaysPresentFlag.
    let mut pic_timing = PicTimingContext::default();
    let nal_hrd = br.read_bit()? == 1;
    if nal_hrd {
        let (cpb, dpb, offset) = read_h264_hrd(br)?;
        pic_timing.cpb_removal_delay_length = cpb;
        pic_timing.dpb_output_delay_length = dpb;
        pic_timing.time_offset_length = offset;
    }
    let vcl_hrd = br.read_bit()? == 1;
    if vcl_hrd {
        let (cpb, dpb, offset) = read_h264_hrd(br)?;
        pic_timing.cpb_removal_delay_length = cpb;
        pic_timing.dpb_output_delay_length = dpb;
        pic_timing.time_offset_length = offset;
    }
    pic_timing.cpb_dpb_delays_present = nal_hrd || vcl_hrd;
    if pic_timing.cpb_dpb_delays_present {
        br.read_bit()?; // low_delay_hrd_flag
    }
    pic_timing.pic_struct_present = br.read_bit()? == 1;
    tail.pic_timing = Some(pic_timing);
    // bitstream_restriction_flag
    if br.read_bit()? == 0 {
        return None;
    }
    br.read_bit()?; // motion_vectors_over_pic_boundaries_flag
    br.read_ue()?; // max_bytes_per_pic_denom
    br.read_ue()?; // max_bits_per_mb_denom
    br.read_ue()?; // log2_max_mv_length_horizontal
    br.read_ue()?; // log2_max_mv_length_vertical
    let reorder = br.read_ue()?;
    if reorder <= MAX_REORDER_FRAMES {
        tail.max_num_reorder_frames = Some(reorder as u8);
    }
    let dec_frame_buffering = br.read_ue()?;
    if dec_frame_buffering <= MAX_REORDER_FRAMES {
        tail.max_dec_frame_buffering = Some(dec_frame_buffering as u8);
    }
    Some(())
}
