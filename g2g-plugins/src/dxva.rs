// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::mem::{offset_of, size_of};

use crate::h264dpb::{H264PictureOrder, RefPic};
use crate::h264parse::{H264Pps, H264SliceHeader, H264Sps};

// dxva.h packs these structs to one byte (`#pragma pack(push, 1)`)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct DxvaPictureParametersH264 {
    pub(crate) frame_width_in_mbs_minus1: u16,
    pub(crate) frame_height_in_mbs_minus1: u16,
    pub(crate) current_picture: u8,
    pub(crate) num_ref_frames: u8,
    pub(crate) picture_flags: u16,
    pub(crate) bit_depth_luma_minus8: u8,
    pub(crate) bit_depth_chroma_minus8: u8,
    pub(crate) reserved_16_bits: u16,
    pub(crate) status_report_feedback_number: u32,
    pub(crate) reference_frame_list: [u8; DXVA_H264_REFERENCE_FRAMES],
    pub(crate) current_field_order_count: [i32; 2],
    pub(crate) field_order_count_list: [[i32; 2]; DXVA_H264_REFERENCE_FRAMES],
    pub(crate) pic_init_qs_minus26: i8,
    pub(crate) chroma_qp_index_offset: i8,
    pub(crate) second_chroma_qp_index_offset: i8,
    pub(crate) continuation_flag: u8,
    pub(crate) pic_init_qp_minus26: i8,
    pub(crate) num_ref_idx_l0_active_minus1: u8,
    pub(crate) num_ref_idx_l1_active_minus1: u8,
    pub(crate) reserved_8_bits_a: u8,
    pub(crate) frame_num_list: [u16; DXVA_H264_REFERENCE_FRAMES],
    pub(crate) used_for_reference_flags: u32,
    pub(crate) non_existing_frame_flags: u16,
    pub(crate) frame_num: u16,
    pub(crate) log2_max_frame_num_minus4: u8,
    pub(crate) pic_order_cnt_type: u8,
    pub(crate) log2_max_pic_order_cnt_lsb_minus4: u8,
    pub(crate) delta_pic_order_always_zero_flag: u8,
    pub(crate) direct_8x8_inference_flag: u8,
    pub(crate) entropy_coding_mode_flag: u8,
    pub(crate) pic_order_present_flag: u8,
    pub(crate) num_slice_groups_minus1: u8,
    pub(crate) slice_group_map_type: u8,
    pub(crate) deblocking_filter_control_present_flag: u8,
    pub(crate) redundant_pic_cnt_present_flag: u8,
    pub(crate) reserved_8_bits_b: u8,
    pub(crate) slice_group_change_rate_minus1: u16,
    pub(crate) slice_group_map: [u8; 810],
}

// sizes and offsets printed by a C program built against dxva.h from Windows SDK 10.0.26100.0
const _: () = assert!(size_of::<DxvaPictureParametersH264>() == 1040);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, current_picture) == 4);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, picture_flags) == 6);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, status_report_feedback_number) == 12);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, reference_frame_list) == 16);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, current_field_order_count) == 32);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, field_order_count_list) == 40);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, pic_init_qs_minus26) == 168);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, pic_init_qp_minus26) == 172);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, frame_num_list) == 176);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, used_for_reference_flags) == 208);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, non_existing_frame_flags) == 212);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, frame_num) == 214);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, log2_max_frame_num_minus4) == 216);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, direct_8x8_inference_flag) == 220);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, slice_group_map_type) == 224);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, slice_group_change_rate_minus1) == 228);
const _: () = assert!(offset_of!(DxvaPictureParametersH264, slice_group_map) == 230);

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct DxvaScalingMatricesH264 {
    pub(crate) scaling_lists_4x4: [[u8; 16]; 6],
    pub(crate) scaling_lists_8x8: [[u8; 64]; 2],
}

const _: () = assert!(size_of::<DxvaScalingMatricesH264>() == 224);
const _: () = assert!(offset_of!(DxvaScalingMatricesH264, scaling_lists_8x8) == 96);

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct DxvaSliceH264Short {
    pub(crate) bitstream_data_location: u32,
    pub(crate) slice_bytes_in_buffer: u32,
    pub(crate) bad_slice_chopping: u16,
}

const _: () = assert!(size_of::<DxvaSliceH264Short>() == 10);
const _: () = assert!(offset_of!(DxvaSliceH264Short, slice_bytes_in_buffer) == 4);
const _: () = assert!(offset_of!(DxvaSliceH264Short, bad_slice_chopping) == 8);

pub(crate) const DXVA_H264_REFERENCE_FRAMES: usize = 16;

// dxva.h `DXVA_PicEntry_H264`: `Index7Bits` in the low seven bits, `AssociatedFlag` in the top bit
const DXVA_PICTURE_ENTRY_ASSOCIATED_FLAG: u8 = 0x80;
const DXVA_PICTURE_ENTRY_UNUSED: u8 = 0xff;

// MSVC allocates the `wBitFields` members of dxva.h from the least significant bit up
const PICTURE_FLAG_FIELD_PIC: u16 = 1 << 0;
const PICTURE_FLAG_MBAFF_FRAME: u16 = 1 << 1;
const PICTURE_FLAG_CHROMA_FORMAT_IDC_SHIFT: u16 = 4;
const PICTURE_FLAG_REFERENCE_PICTURE: u16 = 1 << 6;
const PICTURE_FLAG_CONSTRAINED_INTRA_PRED: u16 = 1 << 7;
const PICTURE_FLAG_WEIGHTED_PRED: u16 = 1 << 8;
const PICTURE_FLAG_WEIGHTED_BIPRED_IDC_SHIFT: u16 = 9;
const PICTURE_FLAG_MBS_CONSECUTIVE: u16 = 1 << 11;
const PICTURE_FLAG_FRAME_MBS_ONLY: u16 = 1 << 12;
const PICTURE_FLAG_TRANSFORM_8X8_MODE: u16 = 1 << 13;
const PICTURE_FLAG_MIN_LUMA_BIPRED_SIZE_8X8: u16 = 1 << 14;
const PICTURE_FLAG_INTRA_PICTURE: u16 = 1 << 15;

// H.264 table A-1: from level 3.1 up, bi-predicted luma blocks are at least 8x8
const MIN_LUMA_BIPRED_SIZE_8X8_LEVEL_IDC: u8 = 31;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DxvaH264Picture<'a> {
    pub(crate) sps: &'a H264Sps,
    pub(crate) pps: &'a H264Pps,
    pub(crate) header: &'a H264SliceHeader,
    pub(crate) order: &'a H264PictureOrder,
    pub(crate) target_slot: usize,
    pub(crate) references: &'a [(usize, RefPic)],
}

fn picture_entry(slot: usize, associated_flag: bool) -> u8 {
    let index = u8::try_from(slot).expect("DPB slots fit the seven-bit DXVA picture index");
    assert!(
        index < DXVA_PICTURE_ENTRY_ASSOCIATED_FLAG,
        "DPB slots fit the seven-bit DXVA picture index"
    );
    if associated_flag {
        index | DXVA_PICTURE_ENTRY_ASSOCIATED_FLAG
    } else {
        index
    }
}

fn picture_flags(picture: &DxvaH264Picture<'_>) -> u16 {
    let sps = picture.sps;
    let pps = picture.pps;
    let header = picture.header;
    let field_pic = header.field_pic_flag;
    let flags = [
        (field_pic, PICTURE_FLAG_FIELD_PIC),
        (
            sps.mb_adaptive_frame_field_flag == 1 && !field_pic,
            PICTURE_FLAG_MBAFF_FRAME,
        ),
        (header.nal_ref_idc != 0, PICTURE_FLAG_REFERENCE_PICTURE),
        (
            pps.constrained_intra_pred_flag == 1,
            PICTURE_FLAG_CONSTRAINED_INTRA_PRED,
        ),
        (pps.weighted_pred_flag == 1, PICTURE_FLAG_WEIGHTED_PRED),
        // the PPS parse refuses slice groups, so macroblocks always run in raster order
        (true, PICTURE_FLAG_MBS_CONSECUTIVE),
        (sps.frame_mbs_only_flag == 1, PICTURE_FLAG_FRAME_MBS_ONLY),
        (
            pps.transform_8x8_mode_flag == 1,
            PICTURE_FLAG_TRANSFORM_8X8_MODE,
        ),
        (
            sps.level_idc >= MIN_LUMA_BIPRED_SIZE_8X8_LEVEL_IDC,
            PICTURE_FLAG_MIN_LUMA_BIPRED_SIZE_8X8,
        ),
        (
            header.is_idr || header.is_intra_slice(),
            PICTURE_FLAG_INTRA_PICTURE,
        ),
    ];
    let set = flags
        .iter()
        .filter(|(on, _)| *on)
        .fold(0, |bits, (_, flag)| bits | flag);
    set | u16::from(sps.chroma_format_idc) << PICTURE_FLAG_CHROMA_FORMAT_IDC_SHIFT
        | u16::from(pps.weighted_bipred_idc) << PICTURE_FLAG_WEIGHTED_BIPRED_IDC_SHIFT
}

pub(crate) fn h264_picture_parameters(picture: &DxvaH264Picture<'_>) -> DxvaPictureParametersH264 {
    let sps = picture.sps;
    let pps = picture.pps;
    let header = picture.header;
    assert!(
        picture.references.len() <= DXVA_H264_REFERENCE_FRAMES,
        "an H.264 DPB holds at most 16 reference frames"
    );
    let mut reference_frame_list = [DXVA_PICTURE_ENTRY_UNUSED; DXVA_H264_REFERENCE_FRAMES];
    let mut field_order_count_list = [[0; 2]; DXVA_H264_REFERENCE_FRAMES];
    let mut frame_num_list = [0; DXVA_H264_REFERENCE_FRAMES];
    let mut used_for_reference_flags = 0u32;
    let mut non_existing_frame_flags = 0u16;
    for (entry, (slot, reference)) in picture.references.iter().enumerate() {
        let long_term = reference.long_term_frame_idx.is_some();
        reference_frame_list[entry] = picture_entry(*slot, long_term);
        field_order_count_list[entry] = reference.field_order_counts;
        frame_num_list[entry] = reference.long_term_frame_idx.unwrap_or(reference.frame_num) as u16;
        // dxva.h: bit 2i marks the top field of entry i used for reference, bit 2i+1 the bottom
        used_for_reference_flags |= 0b11 << (2 * entry);
        if reference.non_existing {
            non_existing_frame_flags |= 1 << entry;
        }
    }
    let frame_height_in_mbs =
        (sps.pic_height_in_map_units_minus1 + 1) * (2 - u32::from(sps.frame_mbs_only_flag));
    DxvaPictureParametersH264 {
        frame_width_in_mbs_minus1: sps.pic_width_in_mbs_minus1 as u16,
        frame_height_in_mbs_minus1: (frame_height_in_mbs - 1) as u16,
        current_picture: picture_entry(picture.target_slot, header.bottom_field_flag),
        num_ref_frames: sps.max_num_ref_frames,
        picture_flags: picture_flags(picture),
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        // gstreamer's and ffmpeg's dxva h264 decoders send 3 here and 1 as the feedback number
        reserved_16_bits: 3,
        status_report_feedback_number: 1,
        reference_frame_list,
        current_field_order_count: picture.order.field_order_counts,
        field_order_count_list,
        pic_init_qs_minus26: pps.pic_init_qs_minus26,
        chroma_qp_index_offset: pps.chroma_qp_index_offset,
        second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset,
        // dxva.h: one means the fields after this one, for the accelerator's own slice parse, are filled
        continuation_flag: 1,
        pic_init_qp_minus26: pps.pic_init_qp_minus26,
        num_ref_idx_l0_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        reserved_8_bits_a: 0,
        frame_num_list,
        used_for_reference_flags,
        non_existing_frame_flags,
        frame_num: header.frame_num as u16,
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
        pic_order_cnt_type: sps.pic_order_cnt_type,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        delta_pic_order_always_zero_flag: sps.delta_pic_order_always_zero_flag,
        direct_8x8_inference_flag: sps.direct_8x8_inference_flag,
        entropy_coding_mode_flag: pps.entropy_coding_mode_flag,
        pic_order_present_flag: pps.bottom_field_pic_order_in_frame_present_flag,
        num_slice_groups_minus1: 0,
        slice_group_map_type: 0,
        deblocking_filter_control_present_flag: pps.deblocking_filter_control_present_flag,
        redundant_pic_cnt_present_flag: pps.redundant_pic_cnt_present_flag,
        reserved_8_bits_b: 0,
        slice_group_change_rate_minus1: 0,
        slice_group_map: [0; 810],
    }
}

// H.264 8.5.6: without a coded scaling matrix every weight is the flat sixteen
const FLAT_SCALING_WEIGHT: u8 = 16;

pub(crate) fn h264_flat_scaling_matrices() -> DxvaScalingMatricesH264 {
    DxvaScalingMatricesH264 {
        scaling_lists_4x4: [[FLAT_SCALING_WEIGHT; 16]; 6],
        scaling_lists_8x8: [[FLAT_SCALING_WEIGHT; 64]; 2],
    }
}
