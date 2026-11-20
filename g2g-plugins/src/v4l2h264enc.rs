// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;
use core::mem::{offset_of, size_of};

use g2g_core::memory::MemoryDomain;
use g2g_core::{
    BitstreamAlignment, Colorimetry, ElementMetadata, G2gError, PropError, PropValue, PropertySpec,
    VideoCodec,
};

use crate::annexb::{add_emulation_prevention, BitWriter, ANNEX_B_START_CODE};
use crate::h264dpb::{H264_NAL_IDR_SLICE, H264_NAL_SLICE};
use crate::h264parse::{
    h264_level_for, H264_MACROBLOCK_SIZE, H264_MAX_QP, H264_NAL_PPS, H264_NAL_SPS,
};
use crate::v4l2stateless::{control, timestamp_ns, V4L2_PIX_FMT_H264_SLICE};
use crate::v4l2statelessenc::{
    v4l2_encoder_properties, EncodeSession, Picture, QuantizerScale, RateControl, RateSettings,
    SettingTakesEffect, V4l2StatelessEnc, V4l2StatelessEncoder, DEFAULT_KEYFRAME_INTERVAL,
};
use crate::vui::{write_vui_head, VideoSignalType, VuiHead};

// linux/v4l2-controls.h of the STMicroelectronics v6.6 kernel, mainline Linux has no H.264 encode controls
const V4L2_CID_STATELESS_H264_ENCODE_PARAMS: u32 = 0x00a4_0908;
const V4L2_CID_STATELESS_H264_ENCODE_RC: u32 = 0x00a4_0909;
const V4L2_H264_SLICE_TYPE_P: u8 = 0;
const V4L2_H264_SLICE_TYPE_I: u8 = 2;
const V4L2_H264_ENCODE_FLAG_ENTROPY_CABAC: u32 = 0x01;
const V4L2_H264_ENCODE_FLAG_TRANSFORM_8X8_MODE: u32 = 0x02;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct V4l2CtrlH264EncodeParams {
    slice_type: u8,
    pic_parameter_set_id: u8,
    frame_num: u16,
    idr_pic_id: u16,
    cabac_init_idc: u8,
    disable_deblocking_filter_idc: u8,
    slice_alpha_c0_offset_div2: i8,
    slice_beta_offset_div2: i8,
    slice_size_mb_rows: i32,
    pic_init_qp_minus26: i8,
    chroma_qp_index_offset: i8,
    flags: u32,
    nal_reference_idc: u8,
    nalu_type: u8,
    reference_ts: u64,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
struct V4l2CtrlH264EncodeRc {
    qp: u32,
    qp_min: u32,
    qp_max: u32,
    mad_qp_delta: i32,
    mad_threshold: u32,
    cp_distance_mbs: u32,
    cp_target: [u32; 10],
    cp_target_error: [i32; 6],
    cp_qp_delta: [i32; 7],
}

// sizes and offsets printed by a C program built against the STMicroelectronics v6.6 header
const _: () = assert!(size_of::<V4l2CtrlH264EncodeParams>() == 40);
const _: () = assert!(offset_of!(V4l2CtrlH264EncodeParams, idr_pic_id) == 4);
const _: () = assert!(offset_of!(V4l2CtrlH264EncodeParams, slice_size_mb_rows) == 12);
const _: () = assert!(offset_of!(V4l2CtrlH264EncodeParams, pic_init_qp_minus26) == 16);
const _: () = assert!(offset_of!(V4l2CtrlH264EncodeParams, flags) == 20);
const _: () = assert!(offset_of!(V4l2CtrlH264EncodeParams, nalu_type) == 25);
const _: () = assert!(offset_of!(V4l2CtrlH264EncodeParams, reference_ts) == 32);
const _: () = assert!(size_of::<V4l2CtrlH264EncodeRc>() == 116);
const _: () = assert!(offset_of!(V4l2CtrlH264EncodeRc, cp_target) == 24);
const _: () = assert!(offset_of!(V4l2CtrlH264EncodeRc, cp_qp_delta) == 88);

// the Hantro H1 writes frame_num in 16 bits and orders pictures by frame_num alone
const LOG2_MAX_FRAME_NUM_MINUS4: u32 = 12;
const PIC_ORDER_CNT_TYPE_FROM_FRAME_NUM: u32 = 2;
// gstreamer's v4l2slh264enc offsets chroma QP by this much
const CHROMA_QP_INDEX_OFFSET: i32 = 4;
const PIC_INIT_QP_BASE: i32 = 26;
const NAL_REF_IDC_HIGHEST: u8 = 3;
// H.264 E.2.1 defaults, written out because the reorder bound after them must be present
const MAX_BYTES_PER_PIC_DENOM: u32 = 2;
const MAX_BITS_PER_MB_DENOM: u32 = 1;
const LOG2_MAX_MV_LENGTH: u32 = 16;

const MAX_CABAC_INIT_IDC: u8 = 2;

// H.264 doubles the quantizer step size every six QP
const QP_PER_DOUBLING: f64 = 6.0;

fn h264_quantizer_step(qp: u8) -> f64 {
    (f64::from(qp) / QP_PER_DOUBLING).exp2()
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Settings {
    cabac: bool,
    cabac_init_idc: u8,
    dct8x8: bool,
}

struct Profile {
    profile_idc: u8,
    constraint_flags: u8,
}

// constrained baseline sets constraint_set0_flag and constraint_set1_flag
const CONSTRAINED_BASELINE: Profile = Profile {
    profile_idc: 66,
    constraint_flags: 0xc0,
};
const MAIN: Profile = Profile {
    profile_idc: 77,
    constraint_flags: 0,
};
const HIGH: Profile = Profile {
    profile_idc: 100,
    constraint_flags: 0,
};

impl Settings {
    fn profile(&self) -> &'static Profile {
        if self.dct8x8 {
            &HIGH
        } else if self.cabac {
            &MAIN
        } else {
            &CONSTRAINED_BASELINE
        }
    }

    fn encode_flags(&self) -> u32 {
        let cabac = if self.cabac {
            V4L2_H264_ENCODE_FLAG_ENTROPY_CABAC
        } else {
            0
        };
        let transform_8x8 = if self.dct8x8 {
            V4L2_H264_ENCODE_FLAG_TRANSFORM_8X8_MODE
        } else {
            0
        };
        cabac | transform_8x8
    }
}

fn nal_unit(nal_type: u8, mut rbsp: BitWriter) -> Vec<u8> {
    // rbsp_trailing_bits
    rbsp.write_bit(1);
    rbsp.align_to_byte();
    let mut nal = Vec::from(ANNEX_B_START_CODE);
    nal.push(NAL_REF_IDC_HIGHEST << 5 | nal_type);
    nal.extend(add_emulation_prevention(&rbsp.into_bytes()));
    nal
}

fn sequence_parameter_set(picture: &Picture, settings: &Settings) -> Option<Vec<u8>> {
    let (width_mbs, height_mbs) = (picture.width_mbs(), picture.height_mbs());
    let frame_mbs = u64::from(width_mbs) * u64::from(height_mbs);
    // without a frame rate only the frame size bounds the level
    let macroblocks_per_second =
        picture
            .frame_rate_fraction()
            .map_or(0, |(numerator, denominator)| {
                (frame_mbs * u64::from(numerator)).div_ceil(u64::from(denominator))
            });
    let level_idc = h264_level_for(width_mbs, height_mbs, macroblocks_per_second)?;
    let profile = settings.profile();
    let mut w = BitWriter::default();
    w.write_bits(u32::from(profile.profile_idc), 8);
    w.write_bits(u32::from(profile.constraint_flags), 8);
    w.write_bits(u32::from(level_idc), 8);
    w.write_ue(0); // seq_parameter_set_id
    if profile.profile_idc == HIGH.profile_idc {
        w.write_ue(1); // chroma_format_idc, 4:2:0
        w.write_ue(0); // bit_depth_luma_minus8
        w.write_ue(0); // bit_depth_chroma_minus8
        w.write_bit(0); // qpprime_y_zero_transform_bypass_flag
        w.write_bit(0); // seq_scaling_matrix_present_flag
    }
    w.write_ue(LOG2_MAX_FRAME_NUM_MINUS4);
    w.write_ue(PIC_ORDER_CNT_TYPE_FROM_FRAME_NUM);
    w.write_ue(1); // max_num_ref_frames
    w.write_bit(0); // gaps_in_frame_num_value_allowed_flag
    w.write_ue(width_mbs - 1);
    w.write_ue(height_mbs - 1);
    w.write_bit(1); // frame_mbs_only_flag
    w.write_bit(1); // direct_8x8_inference_flag
                    // 4:2:0 crops in units of two samples
    let crop_right = (width_mbs * H264_MACROBLOCK_SIZE - picture.width) / 2;
    let crop_bottom = (height_mbs * H264_MACROBLOCK_SIZE - picture.height) / 2;
    if crop_right == 0 && crop_bottom == 0 {
        w.write_bit(0); // frame_cropping_flag
    } else {
        w.write_bit(1);
        w.write_ue(0);
        w.write_ue(crop_right);
        w.write_ue(0);
        w.write_ue(crop_bottom);
    }
    w.write_bit(1); // vui_parameters_present_flag
    let signal_type = (picture.colorimetry != Colorimetry::UNKNOWN).then(|| VideoSignalType {
        color_primaries: picture.colorimetry.primaries.to_cicp(),
        transfer_characteristics: picture.colorimetry.transfer.to_cicp(),
        matrix_coefficients: picture.colorimetry.matrix.to_cicp(),
        video_full_range_flag: picture.colorimetry.range == g2g_core::ColorRange::Full,
    });
    let head = VuiHead {
        pixel_aspect: picture.pixel_aspect,
        signal_type,
    };
    write_vui_head(&mut w, &head)?;
    w.write_bit(0); // chroma_loc_info_present_flag
    match picture.frame_rate_fraction() {
        Some((numerator, denominator)) => {
            w.write_bit(1); // timing_info_present_flag
            w.write_bits(denominator, 32); // num_units_in_tick
                                           // H.264 counts field ticks, two per frame
            w.write_bits(numerator.checked_mul(2)?, 32);
            w.write_bit(1); // fixed_frame_rate_flag
        }
        None => w.write_bit(0),
    }
    w.write_bit(0); // nal_hrd_parameters_present_flag
    w.write_bit(0); // vcl_hrd_parameters_present_flag
    w.write_bit(0); // pic_struct_present_flag
    w.write_bit(1); // bitstream_restriction_flag
    w.write_bit(1); // motion_vectors_over_pic_boundaries_flag
    w.write_ue(MAX_BYTES_PER_PIC_DENOM);
    w.write_ue(MAX_BITS_PER_MB_DENOM);
    w.write_ue(LOG2_MAX_MV_LENGTH);
    w.write_ue(LOG2_MAX_MV_LENGTH);
    w.write_ue(0); // max_num_reorder_frames
    w.write_ue(1); // max_dec_frame_buffering
    Some(nal_unit(H264_NAL_SPS, w))
}

fn picture_parameter_set(settings: &Settings, pic_init_qp: u8) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.write_ue(0); // pic_parameter_set_id
    w.write_ue(0); // seq_parameter_set_id
    w.write_bit(u32::from(settings.cabac)); // entropy_coding_mode_flag
    w.write_bit(0); // bottom_field_pic_order_in_frame_present_flag
    w.write_ue(0); // num_slice_groups_minus1
    w.write_ue(0); // num_ref_idx_l0_default_active_minus1
    w.write_ue(0); // num_ref_idx_l1_default_active_minus1
    w.write_bit(0); // weighted_pred_flag
    w.write_bits(0, 2); // weighted_bipred_idc
    w.write_se(i32::from(pic_init_qp) - PIC_INIT_QP_BASE);
    w.write_se(0); // pic_init_qs_minus26
    w.write_se(CHROMA_QP_INDEX_OFFSET);
    w.write_bit(1); // deblocking_filter_control_present_flag
    w.write_bit(0); // constrained_intra_pred_flag
    w.write_bit(0); // redundant_pic_cnt_present_flag
    if settings.dct8x8 {
        w.write_bit(1); // transform_8x8_mode_flag
        w.write_bit(0); // pic_scaling_matrix_present_flag
        w.write_se(CHROMA_QP_INDEX_OFFSET); // second_chroma_qp_index_offset
    }
    nal_unit(H264_NAL_PPS, w)
}

#[derive(Debug, Default)]
pub struct V4l2H264Encoder {
    settings: Settings,
    frame_num: u16,
    idr_pic_id: u16,
    // the pic_init_qp of the PPS the stream carries, which the slice QP is coded against
    pic_init_qp: u8,
}

impl V4l2StatelessEncoder for V4l2H264Encoder {
    const LOG_CATEGORY: &'static str = "V4l2SlH264Enc";
    const CODEC: VideoCodec = VideoCodec::H264;
    const ALIGNMENT: BitstreamAlignment = BitstreamAlignment::AccessUnit;
    const CODED_PIXEL_FORMAT: u32 = V4L2_PIX_FMT_H264_SLICE;
    const QUANTIZER: QuantizerScale = QuantizerScale {
        max: H264_MAX_QP,
        step: h264_quantizer_step,
    };
    // gstreamer's v4l2slh264enc defaults, but for the quantizer
    const RATE_DEFAULTS: RateSettings = RateSettings {
        keyframe_interval: DEFAULT_KEYFRAME_INTERVAL,
        quantizer: 18,
        rate_control: RateControl::ConstantQp,
        bitrate: 0,
        qp_min: 10,
        qp_max: H264_MAX_QP,
        qp_step: 4,
    };

    fn metadata() -> ElementMetadata {
        ElementMetadata::new(
            "V4L2 stateless H.264 encoder",
            "Codec/Encoder/Video/Hardware",
            "Encodes H.264 on a V4L2 stateless encoder through the media request API",
            crate::AARON_BOXER,
        )
    }

    fn properties() -> &'static [PropertySpec] {
        v4l2_encoder_properties!(
            quantizer: "18",
            quantizer_range: "0..51",
            qp_min: "10",
            qp_max: "51",
            qp_step: "4",
            codec: [
                PropertySpec::new(
                    "cabac",
                    g2g_core::PropKind::Bool,
                    "CABAC entropy coding, Main profile",
                )
                .with_default("false"),
                PropertySpec::new("cabac-init-idc", g2g_core::PropKind::Uint, "CABAC init table, 0..2")
                    .with_default("0"),
                PropertySpec::new("dct8x8", g2g_core::PropKind::Bool, "8x8 transform, High profile")
                    .with_default("false"),
            ]
        )
    }

    // settings in the SPS or PPS take effect at the next IDR, which starts at once
    fn set_codec_property(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Result<SettingTakesEffect, PropError> {
        let previous = self.settings;
        let flag = || value.as_bool().ok_or(PropError::Type);
        match name {
            "cabac" => self.settings.cabac = flag()?,
            "cabac-init-idc" => {
                self.settings.cabac_init_idc = value
                    .as_uint()
                    .ok_or(PropError::Type)
                    .and_then(|number| u8::try_from(number).map_err(|_| PropError::Value))
                    .and_then(|number| {
                        (number <= MAX_CABAC_INIT_IDC)
                            .then_some(number)
                            .ok_or(PropError::Value)
                    })?;
            }
            "dct8x8" => self.settings.dct8x8 = flag()?,
            _ => return Err(PropError::Unknown),
        }
        Ok(if self.settings == previous {
            SettingTakesEffect::Now
        } else {
            SettingTakesEffect::AtNextKeyframe
        })
    }

    fn get_codec_property(&self, name: &str) -> Option<PropValue> {
        let settings = &self.settings;
        Some(match name {
            "cabac" => PropValue::Bool(settings.cabac),
            "cabac-init-idc" => PropValue::Uint(u64::from(settings.cabac_init_idc)),
            "dct8x8" => PropValue::Bool(settings.dct8x8),
            _ => return None,
        })
    }

    fn encode(
        &mut self,
        session: &mut EncodeSession,
        picture: &Picture,
        keyframe: bool,
        qp: u8,
        input: &MemoryDomain,
    ) -> Result<Vec<u8>, G2gError> {
        let settings = &self.settings;
        let mut params = V4l2CtrlH264EncodeParams {
            cabac_init_idc: settings.cabac_init_idc,
            chroma_qp_index_offset: CHROMA_QP_INDEX_OFFSET as i8,
            flags: settings.encode_flags(),
            nal_reference_idc: NAL_REF_IDC_HIGHEST,
            ..Default::default()
        };
        let mut parameter_sets = Vec::new();
        if keyframe {
            self.pic_init_qp = qp;
            parameter_sets = sequence_parameter_set(picture, settings).ok_or_else(|| {
                g2g_core::g2g_error!(
                    g2g_core::log::Target::category(Self::LOG_CATEGORY),
                    "cannot encode: no H.264 level or VUI fits this picture"
                );
                G2gError::CapsMismatch
            })?;
            parameter_sets.extend(picture_parameter_set(settings, self.pic_init_qp));
            self.idr_pic_id = self.idr_pic_id.wrapping_add(1);
            self.frame_num = 0;
            params.slice_type = V4L2_H264_SLICE_TYPE_I;
            params.nalu_type = H264_NAL_IDR_SLICE;
        } else {
            self.frame_num = self.frame_num.wrapping_add(1);
            params.slice_type = V4L2_H264_SLICE_TYPE_P;
            params.nalu_type = H264_NAL_SLICE;
            params.reference_ts = timestamp_ns(session.encoded - 1);
        }
        params.frame_num = self.frame_num;
        params.idr_pic_id = self.idr_pic_id;
        params.pic_init_qp_minus26 = (i32::from(self.pic_init_qp) - PIC_INIT_QP_BASE) as i8;
        let mut rate_control = V4l2CtrlH264EncodeRc {
            qp: u32::from(qp),
            ..Default::default()
        };
        let coded = session.run(
            picture,
            &mut [
                control(V4L2_CID_STATELESS_H264_ENCODE_PARAMS, &mut params),
                control(V4L2_CID_STATELESS_H264_ENCODE_RC, &mut rate_control),
            ],
            input,
        )?;
        let mut access_unit = parameter_sets;
        access_unit.extend_from_slice(coded);
        Ok(access_unit)
    }
}

pub type V4l2SlH264Enc = V4l2StatelessEnc<V4l2H264Encoder>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annexb::strip_emulation_prevention;
    use crate::h264parse::{parse_h264_pps, parse_h264_sps};
    use g2g_core::{AsyncElement, PixelAspect, Rate, RawVideoFormat};

    const FRAMES_PER_SECOND: u32 = 30;

    fn picture_1080p() -> Picture {
        Picture {
            format: RawVideoFormat::Nv12,
            width: 1920,
            height: 1080,
            framerate: Rate::Fixed(
                g2g_core::caps::framerate_q16_from_fraction(FRAMES_PER_SECOND, 1).unwrap(),
            ),
            colorimetry: Colorimetry::BT709,
            pixel_aspect: PixelAspect::SQUARE,
        }
    }

    // the RBSP after the start code and NAL header
    fn rbsp(nal: &[u8]) -> Vec<u8> {
        strip_emulation_prevention(&nal[ANNEX_B_START_CODE.len() + 1..])
    }

    #[test]
    fn the_sps_reads_back_as_the_hardware_codes_slices() {
        let picture = picture_1080p();
        let sps = sequence_parameter_set(&picture, &Settings::default()).unwrap();
        let parsed = parse_h264_sps(&rbsp(&sps)).unwrap();
        assert_eq!(parsed.profile_idc, CONSTRAINED_BASELINE.profile_idc);
        assert_eq!(parsed.level_idc, 40);
        assert_eq!(parsed.coded_extent(), (1920, 1088));
        // 1088 coded rows crop to 1080, in units of two rows
        assert_eq!(parsed.frame_crop_bottom_offset, 4);
        assert_eq!(
            u32::from(parsed.log2_max_frame_num_minus4),
            LOG2_MAX_FRAME_NUM_MINUS4
        );
        assert_eq!(
            u32::from(parsed.pic_order_cnt_type),
            PIC_ORDER_CNT_TYPE_FROM_FRAME_NUM
        );
        assert_eq!(parsed.max_num_ref_frames, 1);
        assert_eq!(parsed.max_num_reorder_frames, Some(0));
        assert_eq!(
            parsed.matrix_coefficients,
            Colorimetry::BT709.matrix.to_cicp()
        );
    }

    #[test]
    fn the_8x8_transform_writes_a_high_profile_sps_and_pps() {
        let settings = Settings {
            cabac: true,
            dct8x8: true,
            ..Settings::default()
        };
        let sps = sequence_parameter_set(&picture_1080p(), &settings).unwrap();
        assert_eq!(
            parse_h264_sps(&rbsp(&sps)).unwrap().profile_idc,
            HIGH.profile_idc
        );
        let pps = parse_h264_pps(&rbsp(&picture_parameter_set(&settings, 30))).unwrap();
        assert_eq!(pps.entropy_coding_mode_flag, 1);
        assert_eq!(pps.transform_8x8_mode_flag, 1);
        assert_eq!(pps.pic_init_qp_minus26, 4);
        assert_eq!(
            i32::from(pps.chroma_qp_index_offset),
            CHROMA_QP_INDEX_OFFSET
        );
        assert_eq!(
            i32::from(pps.second_chroma_qp_index_offset),
            CHROMA_QP_INDEX_OFFSET
        );
    }

    #[test]
    fn a_setting_the_pps_carries_starts_an_idr() {
        let mut encoder = V4l2SlH264Enc::new();
        let mid_gop = 5;
        assert!(!encoder.keyframe_due(mid_gop));
        encoder
            .set_property("cabac", PropValue::Bool(true))
            .unwrap();
        assert!(encoder.keyframe_due(mid_gop));
    }
}
