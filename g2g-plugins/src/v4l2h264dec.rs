// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::mem::{offset_of, size_of};
use core::task::{Context, Poll};
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::PathBuf;

use g2g_core::g2g_error;
use g2g_core::log::Target;
use g2g_core::memory::{DmaBufReleaseWaker, DomainSet, MemoryDomainKind, OwnedDmaBuf, SystemSlice};
use g2g_core::meta::{CropRectangle, Plane, PlaneLayout};
use g2g_core::{BufferPool, ElementMetadata, G2gError, MemoryDomain};
use v4l::v4l2::vidioc;
use v4l::v4l_sys::{v4l2_format, v4l2_memory_V4L2_MEMORY_MMAP, v4l2_plane};

use crate::annexb::START_CODE_PREFIX;
use crate::displayorder::PictureMeta;
use crate::h264dpb::{
    chroma_rectangle, h264_dpb_slots, h264_frame_num_wrap, h264_max_frame_num,
    h264_visible_picture, H264DecodeSession, H264Picture, H264PictureSlots,
};
use crate::h264hardwaredec::{H264HardwareDec, H264HardwareDecoder};
use crate::h264parse::{H264Pps, H264Sps, H264StreamRefused, MALFORMED_SLICE_HEADER};
use crate::v4l2ioctl::{export_buffer, v4l2_ioctl, zeroed_arg};
use crate::v4l2stateless::{
    buffer_arg, buffer_timestamp, control, find_m2m_nodes, menu_control, open_node,
    request_buffers, set_controls, set_stream, timestamp_ns, Mapping, MediaRequest, RequestError,
    BITSTREAM_BYTES_PER_PIXEL, CAPTURE_TYPE, OUTPUT_TYPE, V4L2_BUF_FLAG_ERROR,
    V4L2_BUF_FLAG_REQUEST_FD, V4L2_CTRL_WHICH_CUR_VAL, V4L2_CTRL_WHICH_REQUEST_VAL,
    V4L2_PIX_FMT_H264_SLICE,
};

// linux/v4l2-controls.h from kernel 5.11 on. These are declared here because the
// v4l2-sys-mit bindings come from the build host's headers, and older ones lack
// the stateless controls.
const V4L2_CID_STATELESS_H264_DECODE_MODE: u32 = 0x00a4_0900;
const V4L2_CID_STATELESS_H264_START_CODE: u32 = 0x00a4_0901;
const V4L2_CID_STATELESS_H264_SPS: u32 = 0x00a4_0902;
const V4L2_CID_STATELESS_H264_PPS: u32 = 0x00a4_0903;
const V4L2_CID_STATELESS_H264_DECODE_PARAMS: u32 = 0x00a4_0907;
const V4L2_STATELESS_H264_DECODE_MODE_FRAME_BASED: i32 = 1;
const V4L2_STATELESS_H264_START_CODE_ANNEX_B: i32 = 1;
const V4L2_PIX_FMT_NV12: u32 = u32::from_le_bytes(*b"NV12");

const V4L2_H264_SPS_FLAG_DELTA_PIC_ORDER_ALWAYS_ZERO: u32 = 0x04;
const V4L2_H264_SPS_FLAG_GAPS_IN_FRAME_NUM_VALUE_ALLOWED: u32 = 0x08;
const V4L2_H264_SPS_FLAG_FRAME_MBS_ONLY: u32 = 0x10;
const V4L2_H264_SPS_FLAG_MB_ADAPTIVE_FRAME_FIELD: u32 = 0x20;
const V4L2_H264_SPS_FLAG_DIRECT_8X8_INFERENCE: u32 = 0x40;

const V4L2_H264_PPS_FLAG_ENTROPY_CODING_MODE: u16 = 0x01;
const V4L2_H264_PPS_FLAG_BOTTOM_FIELD_PIC_ORDER_IN_FRAME_PRESENT: u16 = 0x02;
const V4L2_H264_PPS_FLAG_WEIGHTED_PRED: u16 = 0x04;
const V4L2_H264_PPS_FLAG_DEBLOCKING_FILTER_CONTROL_PRESENT: u16 = 0x08;
const V4L2_H264_PPS_FLAG_CONSTRAINED_INTRA_PRED: u16 = 0x10;
const V4L2_H264_PPS_FLAG_REDUNDANT_PIC_CNT_PRESENT: u16 = 0x20;
const V4L2_H264_PPS_FLAG_TRANSFORM_8X8_MODE: u16 = 0x40;

const V4L2_H264_FRAME_REF: u8 = 3;
const V4L2_H264_NUM_DPB_ENTRIES: usize = 16;
const V4L2_H264_DPB_ENTRY_FLAG_VALID: u32 = 0x01;
const V4L2_H264_DPB_ENTRY_FLAG_ACTIVE: u32 = 0x02;
const V4L2_H264_DPB_ENTRY_FLAG_LONG_TERM: u32 = 0x04;

const V4L2_H264_DECODE_PARAM_FLAG_IDR_PIC: u32 = 0x01;
const V4L2_H264_DECODE_PARAM_FLAG_PFRAME: u32 = 0x08;
const V4L2_H264_DECODE_PARAM_FLAG_BFRAME: u32 = 0x10;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct V4l2CtrlH264Sps {
    profile_idc: u8,
    constraint_set_flags: u8,
    level_idc: u8,
    seq_parameter_set_id: u8,
    chroma_format_idc: u8,
    bit_depth_luma_minus8: u8,
    bit_depth_chroma_minus8: u8,
    log2_max_frame_num_minus4: u8,
    pic_order_cnt_type: u8,
    log2_max_pic_order_cnt_lsb_minus4: u8,
    max_num_ref_frames: u8,
    num_ref_frames_in_pic_order_cnt_cycle: u8,
    offset_for_ref_frame: [i32; 255],
    offset_for_non_ref_pic: i32,
    offset_for_top_to_bottom_field: i32,
    pic_width_in_mbs_minus1: u16,
    pic_height_in_map_units_minus1: u16,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct V4l2CtrlH264Pps {
    pic_parameter_set_id: u8,
    seq_parameter_set_id: u8,
    num_slice_groups_minus1: u8,
    num_ref_idx_l0_default_active_minus1: u8,
    num_ref_idx_l1_default_active_minus1: u8,
    weighted_bipred_idc: u8,
    pic_init_qp_minus26: i8,
    pic_init_qs_minus26: i8,
    chroma_qp_index_offset: i8,
    second_chroma_qp_index_offset: i8,
    flags: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct V4l2H264DpbEntry {
    reference_ts: u64,
    pic_num: u32,
    frame_num: u16,
    fields: u8,
    reserved: [u8; 5],
    top_field_order_cnt: i32,
    bottom_field_order_cnt: i32,
    flags: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct V4l2CtrlH264DecodeParams {
    dpb: [V4l2H264DpbEntry; V4L2_H264_NUM_DPB_ENTRIES],
    nal_ref_idc: u16,
    frame_num: u16,
    top_field_order_cnt: i32,
    bottom_field_order_cnt: i32,
    idr_pic_id: u16,
    pic_order_cnt_lsb: u16,
    delta_pic_order_cnt_bottom: i32,
    delta_pic_order_cnt0: i32,
    delta_pic_order_cnt1: i32,
    dec_ref_pic_marking_bit_size: u32,
    pic_order_cnt_bit_size: u32,
    slice_group_change_cycle: u32,
    reserved: u32,
    flags: u32,
}

// sizes and offsets printed by a C program built against the kernel 7.3 uAPI headers
const _: () = assert!(size_of::<V4l2CtrlH264Sps>() == 1048);
const _: () = assert!(offset_of!(V4l2CtrlH264Sps, offset_for_ref_frame) == 12);
const _: () = assert!(offset_of!(V4l2CtrlH264Sps, offset_for_non_ref_pic) == 1032);
const _: () = assert!(offset_of!(V4l2CtrlH264Sps, pic_width_in_mbs_minus1) == 1040);
const _: () = assert!(offset_of!(V4l2CtrlH264Sps, flags) == 1044);
const _: () = assert!(size_of::<V4l2CtrlH264Pps>() == 12);
const _: () = assert!(offset_of!(V4l2CtrlH264Pps, pic_init_qp_minus26) == 6);
const _: () = assert!(offset_of!(V4l2CtrlH264Pps, flags) == 10);
const _: () = assert!(size_of::<V4l2H264DpbEntry>() == 32);
const _: () = assert!(offset_of!(V4l2H264DpbEntry, pic_num) == 8);
const _: () = assert!(offset_of!(V4l2H264DpbEntry, frame_num) == 12);
const _: () = assert!(offset_of!(V4l2H264DpbEntry, fields) == 14);
const _: () = assert!(offset_of!(V4l2H264DpbEntry, top_field_order_cnt) == 20);
const _: () = assert!(offset_of!(V4l2H264DpbEntry, flags) == 28);
const _: () = assert!(size_of::<V4l2CtrlH264DecodeParams>() == 560);
const _: () = assert!(offset_of!(V4l2CtrlH264DecodeParams, nal_ref_idc) == 512);
const _: () = assert!(offset_of!(V4l2CtrlH264DecodeParams, top_field_order_cnt) == 516);
const _: () = assert!(offset_of!(V4l2CtrlH264DecodeParams, idr_pic_id) == 524);
const _: () = assert!(offset_of!(V4l2CtrlH264DecodeParams, delta_pic_order_cnt_bottom) == 528);
const _: () = assert!(offset_of!(V4l2CtrlH264DecodeParams, dec_ref_pic_marking_bit_size) == 540);
const _: () = assert!(offset_of!(V4l2CtrlH264DecodeParams, slice_group_change_cycle) == 548);
const _: () = assert!(offset_of!(V4l2CtrlH264DecodeParams, flags) == 556);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum V4l2DecodeError {
    NoStatelessH264Decoder,
    StreamRefused(&'static str),
    Device(G2gError),
    TimedOut,
    CorruptPicture,
    NoFreePictureBuffer,
}

impl From<H264StreamRefused> for V4l2DecodeError {
    fn from(refused: H264StreamRefused) -> Self {
        Self::StreamRefused(refused.0)
    }
}

impl From<G2gError> for V4l2DecodeError {
    fn from(error: G2gError) -> Self {
        Self::Device(error)
    }
}

impl From<RequestError> for V4l2DecodeError {
    fn from(error: RequestError) -> Self {
        match error {
            RequestError::TimedOut => Self::TimedOut,
            RequestError::Device(error) => Self::Device(error),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct V4l2DecoderNodes {
    pub video: PathBuf,
    pub media: PathBuf,
}

pub fn find_v4l2_h264_decoder() -> Result<V4l2DecoderNodes, V4l2DecodeError> {
    let nodes = find_m2m_nodes(OUTPUT_TYPE, V4L2_PIX_FMT_H264_SLICE)?
        .ok_or(V4l2DecodeError::NoStatelessH264Decoder)?;
    Ok(V4l2DecoderNodes {
        video: nodes.video,
        media: nodes.media,
    })
}

const PICTURE_TOO_LARGE: &str = "H.264 picture is wider or taller than V4L2's SPS control holds";

fn sps_control(sps: &H264Sps) -> Result<V4l2CtrlH264Sps, V4l2DecodeError> {
    let mut control: V4l2CtrlH264Sps = zeroed_arg();
    control.profile_idc = sps.profile_idc;
    control.level_idc = sps.level_idc;
    control.seq_parameter_set_id = sps.seq_parameter_set_id;
    control.chroma_format_idc = sps.chroma_format_idc;
    control.bit_depth_luma_minus8 = sps.bit_depth_luma_minus8;
    control.bit_depth_chroma_minus8 = sps.bit_depth_chroma_minus8;
    control.log2_max_frame_num_minus4 = sps.log2_max_frame_num_minus4;
    control.pic_order_cnt_type = sps.pic_order_cnt_type;
    control.log2_max_pic_order_cnt_lsb_minus4 = sps.log2_max_pic_order_cnt_lsb_minus4;
    control.max_num_ref_frames = sps.max_num_ref_frames;
    // parse_h264_sps refuses a cycle longer than a byte holds
    control.num_ref_frames_in_pic_order_cnt_cycle = sps.offsets_for_ref_frame.len() as u8;
    control.offset_for_ref_frame[..sps.offsets_for_ref_frame.len()]
        .copy_from_slice(&sps.offsets_for_ref_frame);
    control.offset_for_non_ref_pic = sps.offset_for_non_ref_pic;
    control.offset_for_top_to_bottom_field = sps.offset_for_top_to_bottom_field;
    control.pic_width_in_mbs_minus1 = u16::try_from(sps.pic_width_in_mbs_minus1)
        .map_err(|_| V4l2DecodeError::StreamRefused(PICTURE_TOO_LARGE))?;
    control.pic_height_in_map_units_minus1 = u16::try_from(sps.pic_height_in_map_units_minus1)
        .map_err(|_| V4l2DecodeError::StreamRefused(PICTURE_TOO_LARGE))?;
    control.flags = [
        (
            sps.delta_pic_order_always_zero_flag,
            V4L2_H264_SPS_FLAG_DELTA_PIC_ORDER_ALWAYS_ZERO,
        ),
        (
            sps.gaps_in_frame_num_value_allowed_flag,
            V4L2_H264_SPS_FLAG_GAPS_IN_FRAME_NUM_VALUE_ALLOWED,
        ),
        (sps.frame_mbs_only_flag, V4L2_H264_SPS_FLAG_FRAME_MBS_ONLY),
        (
            sps.mb_adaptive_frame_field_flag,
            V4L2_H264_SPS_FLAG_MB_ADAPTIVE_FRAME_FIELD,
        ),
        (
            sps.direct_8x8_inference_flag,
            V4L2_H264_SPS_FLAG_DIRECT_8X8_INFERENCE,
        ),
    ]
    .iter()
    .filter(|(set, _)| *set == 1)
    .fold(0, |flags, (_, flag)| flags | flag);
    Ok(control)
}

fn pps_control(pps: &H264Pps) -> V4l2CtrlH264Pps {
    V4l2CtrlH264Pps {
        pic_parameter_set_id: pps.pic_parameter_set_id,
        seq_parameter_set_id: pps.seq_parameter_set_id,
        num_slice_groups_minus1: 0,
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        weighted_bipred_idc: pps.weighted_bipred_idc,
        pic_init_qp_minus26: pps.pic_init_qp_minus26,
        pic_init_qs_minus26: pps.pic_init_qs_minus26,
        chroma_qp_index_offset: pps.chroma_qp_index_offset,
        second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset,
        flags: [
            (
                pps.entropy_coding_mode_flag,
                V4L2_H264_PPS_FLAG_ENTROPY_CODING_MODE,
            ),
            (
                pps.bottom_field_pic_order_in_frame_present_flag,
                V4L2_H264_PPS_FLAG_BOTTOM_FIELD_PIC_ORDER_IN_FRAME_PRESENT,
            ),
            (pps.weighted_pred_flag, V4L2_H264_PPS_FLAG_WEIGHTED_PRED),
            (
                pps.deblocking_filter_control_present_flag,
                V4L2_H264_PPS_FLAG_DEBLOCKING_FILTER_CONTROL_PRESENT,
            ),
            (
                pps.constrained_intra_pred_flag,
                V4L2_H264_PPS_FLAG_CONSTRAINED_INTRA_PRED,
            ),
            (
                pps.redundant_pic_cnt_present_flag,
                V4L2_H264_PPS_FLAG_REDUNDANT_PIC_CNT_PRESENT,
            ),
            (
                pps.transform_8x8_mode_flag,
                V4L2_H264_PPS_FLAG_TRANSFORM_8X8_MODE,
            ),
        ]
        .iter()
        .filter(|(set, _)| *set == 1)
        .fold(0, |flags, (_, flag)| flags | flag),
    }
}

// V4L2 matches a reference to the capture buffer whose timestamp is `reference_ts`
fn decode_params_control(
    picture: &H264Picture<'_>,
    slots: &H264PictureSlots,
    slot_pictures: &[u64],
    picture_number: u64,
) -> Result<V4l2CtrlH264DecodeParams, V4l2DecodeError> {
    if slots.references.len() > V4L2_H264_NUM_DPB_ENTRIES {
        return Err(V4l2DecodeError::StreamRefused(
            "H.264 stream holds more references than V4L2's 16 DPB entries",
        ));
    }
    let header = &picture.header;
    let max_frame_num = h264_max_frame_num(&picture.sps);
    let mut control: V4l2CtrlH264DecodeParams = zeroed_arg();
    for (entry, (slot, reference)) in control.dpb.iter_mut().zip(&slots.references) {
        // frames a frame_num gap inferred have no picture, so they point at the one being decoded
        let holder = if reference.non_existing {
            picture_number
        } else {
            slot_pictures[*slot]
        };
        let (pic_num, long_term) = match reference.long_term_frame_idx {
            Some(index) => (index, V4L2_H264_DPB_ENTRY_FLAG_LONG_TERM),
            None => (
                h264_frame_num_wrap(reference.frame_num, header.frame_num, max_frame_num) as u32,
                0,
            ),
        };
        *entry = V4l2H264DpbEntry {
            reference_ts: timestamp_ns(holder),
            pic_num,
            frame_num: reference.frame_num as u16,
            fields: V4L2_H264_FRAME_REF,
            reserved: [0; 5],
            top_field_order_cnt: reference.field_order_counts[0],
            bottom_field_order_cnt: reference.field_order_counts[1],
            flags: V4L2_H264_DPB_ENTRY_FLAG_VALID | V4L2_H264_DPB_ENTRY_FLAG_ACTIVE | long_term,
        };
    }
    control.nal_ref_idc = u16::from(header.nal_ref_idc);
    control.frame_num = header.frame_num as u16;
    control.top_field_order_cnt = slots.order.field_order_counts[0];
    control.bottom_field_order_cnt = slots.order.field_order_counts[1];
    control.idr_pic_id = u16::try_from(header.idr_pic_id).map_err(|_| MALFORMED_SLICE_HEADER)?;
    control.pic_order_cnt_lsb = header.pic_order_cnt_lsb as u16;
    control.delta_pic_order_cnt_bottom = header.delta_pic_order_cnt_bottom;
    control.delta_pic_order_cnt0 = header.delta_pic_order_cnt[0];
    control.delta_pic_order_cnt1 = header.delta_pic_order_cnt[1];
    control.dec_ref_pic_marking_bit_size = header.dec_ref_pic_marking_bit_size;
    control.pic_order_cnt_bit_size = header.pic_order_cnt_bit_size;
    let slice_flag = match header.slice_type % 5 {
        0 => V4L2_H264_DECODE_PARAM_FLAG_PFRAME,
        1 => V4L2_H264_DECODE_PARAM_FLAG_BFRAME,
        _ => 0,
    };
    let idr_flag = if header.is_idr {
        V4L2_H264_DECODE_PARAM_FLAG_IDR_PIC
    } else {
        0
    };
    control.flags = slice_flag | idr_flag;
    Ok(control)
}

const DECODE_TIMEOUT_MS: i32 = 1000;

#[derive(Debug)]
struct CaptureLayout {
    bytes_per_line: usize,
    luma_rows: usize,
}

#[derive(Debug)]
pub struct CapturePicture {
    buffer: usize,
    dmabuf: OwnedDmaBuf,
}

#[derive(Debug)]
struct V4l2Session {
    h264: H264DecodeSession,
    request: MediaRequest,
    bitstream: Mapping,
    pictures: Vec<Mapping>,
    exports: Vec<OwnedDmaBuf>,
    release_waker: Arc<DmaBufReleaseWaker>,
    slot_buffers: Vec<Option<usize>>,
    // created on the first picture a system-memory consumer takes
    copies: Option<BufferPool<Box<[u8]>>>,
    reorder_depth: usize,
    layout: CaptureLayout,
    visible: CropRectangle,
    slot_pictures: Vec<u64>,
    decoded: u64,
}

// g2g-launch queues four frames per link, and Weston holds two buffers, the one on screen and the next
const DOWNSTREAM_PICTURES: usize = 6;

fn nv12_plane_layout(layout: &CaptureLayout, visible: CropRectangle) -> Option<PlaneLayout> {
    let stride = layout.bytes_per_line;
    let chroma = chroma_rectangle(visible);
    let luma_offset = stride
        .checked_mul(visible.top as usize)?
        .checked_add(visible.left as usize)?;
    let chroma_offset = stride
        .checked_mul(layout.luma_rows.checked_add(chroma.top as usize)?)?
        .checked_add(2 * chroma.left as usize)?;
    PlaneLayout::new(&[
        Plane {
            offset: luma_offset,
            stride,
        },
        Plane {
            offset: chroma_offset,
            stride,
        },
    ])
}

fn free_capture_buffer(exports: &[OwnedDmaBuf], slot_buffers: &[Option<usize>]) -> Option<usize> {
    // the DPB references a buffer through its slot, and a picture still downstream holds a share of its export
    (0..exports.len())
        .find(|&buffer| !slot_buffers.contains(&Some(buffer)) && exports[buffer].share_count() == 1)
}

impl V4l2Session {
    fn start(
        video: &OwnedFd,
        media: &OwnedFd,
        sps: &H264Sps,
        reorder_depth: usize,
    ) -> Result<Self, V4l2DecodeError> {
        // Hantro G1 and the other V4L2 H.264 decoders in mainline Linux output 8-bit 4:2:0 only
        if sps.chroma_format_idc != 1 || sps.bit_depth_luma_minus8 != 0 {
            return Err(V4l2DecodeError::StreamRefused(
                "H.264 other than 8-bit 4:2:0",
            ));
        }
        let visible = h264_visible_picture(sps)?;
        let (width, height) = sps.coded_extent();

        let mut output: v4l2_format = zeroed_arg();
        output.type_ = OUTPUT_TYPE;
        // SAFETY: `pix_mp` is the union member multi-plane buffer types use.
        let pix = unsafe { &mut output.fmt.pix_mp };
        pix.width = width;
        pix.height = height;
        pix.pixelformat = V4L2_PIX_FMT_H264_SLICE;
        pix.num_planes = 1;
        pix.plane_fmt[0].sizeimage = width * height * BITSTREAM_BYTES_PER_PIXEL;
        // SAFETY: `v4l2_format` is VIDIOC_S_FMT's argument type.
        unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_S_FMT, &mut output) }?;

        let mut sps_payload = sps_control(sps)?;
        set_controls(
            video,
            V4L2_CTRL_WHICH_CUR_VAL,
            0,
            &mut [
                menu_control(
                    V4L2_CID_STATELESS_H264_DECODE_MODE,
                    V4L2_STATELESS_H264_DECODE_MODE_FRAME_BASED,
                ),
                menu_control(
                    V4L2_CID_STATELESS_H264_START_CODE,
                    V4L2_STATELESS_H264_START_CODE_ANNEX_B,
                ),
                control(V4L2_CID_STATELESS_H264_SPS, &mut sps_payload),
            ],
        )?;

        let mut capture: v4l2_format = zeroed_arg();
        capture.type_ = CAPTURE_TYPE;
        // SAFETY: `v4l2_format` is VIDIOC_G_FMT's argument type.
        unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_G_FMT, &mut capture) }?;
        // SAFETY: the driver filled `pix_mp` for the multi-plane capture queue.
        let pix = unsafe { capture.fmt.pix_mp };
        if pix.pixelformat != V4L2_PIX_FMT_NV12 || pix.num_planes != 1 {
            return Err(V4l2DecodeError::StreamRefused(
                "the V4L2 decoder outputs no single-plane NV12 for this stream",
            ));
        }
        let layout = CaptureLayout {
            bytes_per_line: pix.plane_fmt[0].bytesperline as usize,
            luma_rows: pix.height as usize,
        };

        let slots = h264_dpb_slots(sps);
        request_buffers(video, OUTPUT_TYPE, v4l2_memory_V4L2_MEMORY_MMAP, 1)?;
        let bitstream = Mapping::of_buffer(video, OUTPUT_TYPE, 0)?;
        let wanted = slots + reorder_depth + DOWNSTREAM_PICTURES;
        let granted = request_buffers(
            video,
            CAPTURE_TYPE,
            v4l2_memory_V4L2_MEMORY_MMAP,
            wanted as u32,
        )?;
        // fewer leaves the decoder waiting forever on a picture the display holds until the next one
        if (granted as usize) < slots + reorder_depth + 1 {
            return Err(V4l2DecodeError::StreamRefused(
                "the V4L2 decoder grants fewer picture buffers than the stream's DPB and reordering need",
            ));
        }
        let pictures = (0..granted)
            .map(|index| Mapping::of_buffer(video, CAPTURE_TYPE, index))
            .collect::<Result<Vec<_>, _>>()?;
        let chroma_end = layout.bytes_per_line * (layout.luma_rows + layout.luma_rows.div_ceil(2));
        if pictures
            .iter()
            .any(|picture| picture.bytes().len() < chroma_end)
        {
            return Err(V4l2DecodeError::StreamRefused(
                "the V4L2 decoder's picture buffers are smaller than an NV12 picture",
            ));
        }
        let plane_layout = nv12_plane_layout(&layout, visible).ok_or(
            V4l2DecodeError::StreamRefused("the visible picture lies outside the decoded picture"),
        )?;
        let release_waker = Arc::new(DmaBufReleaseWaker::default());
        let exports = (0..granted)
            .map(|index| {
                export_buffer(
                    video.as_raw_fd(),
                    CAPTURE_TYPE,
                    index,
                    layout.bytes_per_line as u32,
                )
                .map(|dmabuf| {
                    dmabuf
                        .with_plane_layout(plane_layout)
                        .with_release_waker(Arc::clone(&release_waker))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        set_stream(video, OUTPUT_TYPE, true)?;
        set_stream(video, CAPTURE_TYPE, true)?;

        let request = MediaRequest::alloc(media)?;

        Ok(Self {
            h264: H264DecodeSession::new(sps),
            request,
            bitstream,
            pictures,
            exports,
            release_waker,
            slot_buffers: alloc::vec![None; slots],
            copies: None,
            reorder_depth,
            layout,
            visible,
            slot_pictures: alloc::vec![0; slots],
            decoded: 0,
        })
    }

    fn write_bitstream(&mut self, slices: &[&[u8]]) -> Result<u32, V4l2DecodeError> {
        let length: usize = slices
            .iter()
            .map(|slice| START_CODE_PREFIX.len() + slice.len())
            .sum();
        let bytes = self.bitstream.bytes_mut();
        if length > bytes.len() {
            return Err(V4l2DecodeError::StreamRefused(
                "H.264 picture is larger than the V4L2 decoder's bitstream buffer",
            ));
        }
        let mut written = 0;
        for slice in slices {
            for part in [&START_CODE_PREFIX[..], slice] {
                bytes[written..written + part.len()].copy_from_slice(part);
                written += part.len();
            }
        }
        Ok(written as u32)
    }

    fn run_request(&self, video: &OwnedFd) -> Result<(), V4l2DecodeError> {
        self.request.queue_and_wait(DECODE_TIMEOUT_MS)?;
        let mut plane: v4l2_plane = zeroed_arg();
        let mut output = buffer_arg(OUTPUT_TYPE, 0, &mut plane);
        // SAFETY: `V4l2Buffer` matches the kernel's `v4l2_buffer`, which VIDIOC_DQBUF takes.
        unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_DQBUF, &mut output) }?;
        Ok(())
    }

    fn release_unreferenced_buffers(&mut self) {
        let held = self.h264.held_slots();
        for (slot, buffer) in self.slot_buffers.iter_mut().enumerate() {
            if !held.contains(&slot) {
                *buffer = None;
            }
        }
    }

    fn decode_picture(
        &mut self,
        video: &OwnedFd,
        picture: &H264Picture<'_>,
    ) -> Result<(PictureMeta, CapturePicture), V4l2DecodeError> {
        let slots = self.h264.begin_picture(picture)?;
        self.release_unreferenced_buffers();
        let buffer = free_capture_buffer(&self.exports, &self.slot_buffers)
            .ok_or(V4l2DecodeError::NoFreePictureBuffer)?;
        self.slot_buffers[slots.target] = Some(buffer);
        self.decoded += 1;
        let picture_number = self.decoded;
        self.slot_pictures[slots.target] = picture_number;
        let mut decode_params =
            decode_params_control(picture, &slots, &self.slot_pictures, picture_number)?;
        let mut sps = sps_control(&picture.sps)?;
        let mut pps = pps_control(&picture.pps);
        let bytesused = self.write_bitstream(&picture.slices)?;

        self.request.reinit()?;
        set_controls(
            video,
            V4L2_CTRL_WHICH_REQUEST_VAL,
            self.request.fd(),
            &mut [
                control(V4L2_CID_STATELESS_H264_SPS, &mut sps),
                control(V4L2_CID_STATELESS_H264_PPS, &mut pps),
                control(V4L2_CID_STATELESS_H264_DECODE_PARAMS, &mut decode_params),
            ],
        )?;

        let target = buffer as u32;
        let mut capture_plane: v4l2_plane = zeroed_arg();
        let mut capture = buffer_arg(CAPTURE_TYPE, target, &mut capture_plane);
        // SAFETY: `V4l2Buffer` matches the kernel's `v4l2_buffer`, which VIDIOC_QBUF takes.
        unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_QBUF, &mut capture) }?;

        let mut output_plane: v4l2_plane = zeroed_arg();
        output_plane.bytesused = bytesused;
        let mut output = buffer_arg(OUTPUT_TYPE, 0, &mut output_plane);
        output.flags = V4L2_BUF_FLAG_REQUEST_FD;
        output.request_fd = self.request.fd();
        output.timestamp = buffer_timestamp(picture_number);
        // SAFETY: as for the capture buffer above.
        unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_QBUF, &mut output) }?;

        self.run_request(video)?;
        let mut done_plane: v4l2_plane = zeroed_arg();
        let mut done = buffer_arg(CAPTURE_TYPE, 0, &mut done_plane);
        // SAFETY: as for the capture buffer above.
        unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_DQBUF, &mut done) }?;
        if done.index != target {
            return Err(V4l2DecodeError::Device(G2gError::Hardware(
                g2g_core::HardwareError::V4l2(-1),
            )));
        }
        if done.flags & V4L2_BUF_FLAG_ERROR != 0 {
            return Err(V4l2DecodeError::CorruptPicture);
        }
        let meta = self.h264.end_picture(picture, &slots)?;
        self.release_unreferenced_buffers();
        Ok((
            meta,
            CapturePicture {
                buffer,
                dmabuf: self.exports[buffer].clone(),
            },
        ))
    }

    fn copy_picture(&mut self, buffer: usize) -> SystemSlice {
        let chroma = chroma_rectangle(self.visible);
        let length =
            (self.visible.width * self.visible.height + 2 * chroma.width * chroma.height) as usize;
        // draining the reorder buffer at end of stream copies every picture it holds at once
        let count = self.reorder_depth + DOWNSTREAM_PICTURES + 1;
        let pooled = self
            .copies
            .get_or_insert_with(|| BufferPool::new_byte_pool(count, length))
            .try_acquire();
        match pooled {
            Some(mut copy) => {
                self.write_visible_picture(buffer, &mut copy.as_mut()[..length]);
                SystemSlice::from_pool(copy, length)
            }
            // a consumer that keeps frames, such as a test collecting every one, empties the pool
            None => {
                let mut copy = alloc::vec![0; length].into_boxed_slice();
                self.write_visible_picture(buffer, &mut copy);
                SystemSlice::from_boxed(copy)
            }
        }
    }

    // tight NV12 of the visible picture, luma rows then interleaved chroma rows
    fn write_visible_picture(&self, buffer: usize, target: &mut [u8]) {
        let bytes = self.pictures[buffer].bytes();
        let stride = self.layout.bytes_per_line;
        let luma = self.visible;
        let chroma = chroma_rectangle(luma);
        let chroma_offset = stride * self.layout.luma_rows;
        let luma_rows = (luma.top..luma.top + luma.height).map(|row| {
            (
                row as usize * stride + luma.left as usize,
                luma.width as usize,
            )
        });
        let chroma_rows = (chroma.top..chroma.top + chroma.height).map(|row| {
            (
                chroma_offset + row as usize * stride + 2 * chroma.left as usize,
                2 * chroma.width as usize,
            )
        });
        let mut written = 0;
        for (start, width) in luma_rows.chain(chroma_rows) {
            target[written..written + width].copy_from_slice(&bytes[start..start + width]);
            written += width;
        }
    }
}

#[derive(Debug)]
pub struct V4l2H264Decoder {
    video: OwnedFd,
    media: OwnedFd,
    session: Option<V4l2Session>,
}

impl V4l2H264Decoder {
    pub fn open_nodes(nodes: &V4l2DecoderNodes) -> Result<Self, V4l2DecodeError> {
        Ok(Self {
            video: open_node(&nodes.video)?,
            media: open_node(&nodes.media)?,
            session: None,
        })
    }

    // stopping both queues hands every buffer back, so a failed decode leaves none queued
    fn restart_queues(&self) -> Result<(), V4l2DecodeError> {
        for buffer_type in [OUTPUT_TYPE, CAPTURE_TYPE] {
            set_stream(&self.video, buffer_type, false)?;
            set_stream(&self.video, buffer_type, true)?;
        }
        Ok(())
    }
}

fn decode_refused(error: V4l2DecodeError) -> G2gError {
    let target = Target::category(V4l2H264Decoder::LOG_CATEGORY);
    match error {
        V4l2DecodeError::StreamRefused(reason) => {
            g2g_error!(target, "cannot decode this stream: {reason}")
        }
        V4l2DecodeError::NoStatelessH264Decoder => g2g_error!(
            target,
            "no V4L2 stateless H.264 decoder with a media controller was found"
        ),
        V4l2DecodeError::TimedOut => g2g_error!(
            target,
            "the V4L2 decoder did not finish a picture within {DECODE_TIMEOUT_MS} ms"
        ),
        V4l2DecodeError::CorruptPicture => {
            g2g_error!(target, "the V4L2 decoder reported a corrupt picture")
        }
        V4l2DecodeError::NoFreePictureBuffer => g2g_error!(
            target,
            "every V4L2 picture buffer is referenced or still downstream"
        ),
        V4l2DecodeError::Device(error) => {
            g2g_error!(target, "the V4L2 decoder refused a request: {error:?}");
            return error;
        }
    }
    G2gError::CapsMismatch
}

impl H264HardwareDecoder for V4l2H264Decoder {
    type Picture = CapturePicture;
    type Error = V4l2DecodeError;
    const LOG_CATEGORY: &'static str = "V4l2SlH264Dec";
    const OUTPUT_DOMAINS: DomainSet =
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf);

    fn metadata() -> ElementMetadata {
        ElementMetadata::new(
            "V4L2 stateless H.264 decoder",
            "Codec/Decoder/Video/Hardware",
            "Decodes H.264 on a V4L2 stateless decoder through the media request API",
            crate::AARON_BOXER,
        )
    }

    fn decodes_h264() -> bool {
        static DECODES_H264: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *DECODES_H264.get_or_init(|| find_v4l2_h264_decoder().is_ok())
    }

    fn open() -> Result<Self, V4l2DecodeError> {
        Self::open_nodes(&find_v4l2_h264_decoder()?)
    }

    fn start_session(
        &mut self,
        sps: &H264Sps,
        reorder_depth: usize,
    ) -> Result<(), V4l2DecodeError> {
        if self.session.is_some() {
            return Ok(());
        }
        self.session = Some(V4l2Session::start(
            &self.video,
            &self.media,
            sps,
            reorder_depth,
        )?);
        Ok(())
    }

    fn has_session(&self) -> bool {
        self.session.is_some()
    }

    fn poll_free_picture(&self, cx: &mut Context<'_>) -> Poll<()> {
        let Some(session) = self.session.as_ref() else {
            return Poll::Ready(());
        };
        session.release_waker.register(cx.waker());
        if free_capture_buffer(&session.exports, &session.slot_buffers).is_some() {
            return Poll::Ready(());
        }
        Poll::Pending
    }

    fn split_pictures<'s>(
        &mut self,
        access_unit: &'s [u8],
    ) -> Result<Vec<H264Picture<'s>>, V4l2DecodeError> {
        let session = self
            .session
            .as_mut()
            .expect("H264HardwareDec decodes only once a session has started");
        Ok(session.h264.split_pictures(access_unit)?)
    }

    fn decode_picture(
        &mut self,
        picture: &H264Picture<'_>,
    ) -> Result<(PictureMeta, CapturePicture), V4l2DecodeError> {
        let video = &self.video;
        let session = self
            .session
            .as_mut()
            .expect("H264HardwareDec decodes only once a session has started");
        let decoded = session.decode_picture(video, picture);
        if decoded.is_err() {
            self.restart_queues()?;
        }
        decoded
    }

    fn reset(&mut self) {
        if let Some(session) = self.session.as_mut() {
            session.h264.reset();
            session.slot_buffers.fill(None);
        }
    }

    fn picture_extent(&self) -> (u32, u32) {
        let visible = self
            .session
            .as_ref()
            .expect("H264HardwareDec emits pictures only once a session has started")
            .visible;
        (visible.width, visible.height)
    }

    fn output(
        &mut self,
        picture: CapturePicture,
        domain: MemoryDomainKind,
    ) -> Result<MemoryDomain, V4l2DecodeError> {
        if domain == MemoryDomainKind::DmaBuf {
            return Ok(MemoryDomain::DmaBuf(picture.dmabuf));
        }
        let session = self
            .session
            .as_mut()
            .expect("H264HardwareDec emits pictures only once a session has started");
        Ok(MemoryDomain::System(session.copy_picture(picture.buffer)))
    }

    fn stream_refusal(error: &V4l2DecodeError) -> Option<&'static str> {
        match error {
            V4l2DecodeError::StreamRefused(reason) => Some(reason),
            _ => None,
        }
    }

    fn refused(error: V4l2DecodeError) -> G2gError {
        decode_refused(error)
    }
}

pub type V4l2SlH264Dec = H264HardwareDec<V4l2H264Decoder>;

#[cfg(test)]
mod tests {
    use super::*;

    fn exports(count: usize) -> Vec<OwnedDmaBuf> {
        (0..count)
            .map(|_| {
                // SAFETY: a fresh memfd with no other owner stands in for an exported buffer.
                let fd = unsafe { libc::memfd_create(c"capture".as_ptr(), 0) };
                assert!(fd >= 0, "memfd_create failed");
                // SAFETY: the fd was just created and nothing else owns it.
                unsafe { OwnedDmaBuf::from_raw(fd, 0, 0) }
            })
            .collect()
    }

    #[test]
    fn a_buffer_referenced_by_the_dpb_or_held_downstream_is_not_reused() {
        let exports = exports(3);
        let slot_buffers = [Some(0), None];
        let downstream = exports[1].clone();
        assert_eq!(free_capture_buffer(&exports, &slot_buffers), Some(2));
        let downstream_too = exports[2].clone();
        assert_eq!(free_capture_buffer(&exports, &slot_buffers), None);
        drop(downstream);
        assert_eq!(free_capture_buffer(&exports, &slot_buffers), Some(1));
        drop(downstream_too);
    }

    #[test]
    fn the_chroma_plane_starts_after_the_decoded_rows_not_the_visible_ones() {
        let layout = CaptureLayout {
            bytes_per_line: 1920,
            luma_rows: 1088,
        };
        let visible = CropRectangle {
            left: 0,
            top: 0,
            width: 1920,
            height: 1080,
        };
        let planes = nv12_plane_layout(&layout, visible).expect("the picture fits");
        assert_eq!(
            planes.plane(0),
            Some(Plane {
                offset: 0,
                stride: 1920
            })
        );
        assert_eq!(
            planes.plane(1),
            Some(Plane {
                offset: 1920 * 1088,
                stride: 1920
            })
        );
    }

    #[test]
    fn a_cropped_picture_starts_at_its_top_left_sample_in_both_planes() {
        let layout = CaptureLayout {
            bytes_per_line: 320,
            luma_rows: 192,
        };
        let visible = CropRectangle {
            left: 8,
            top: 4,
            width: 304,
            height: 180,
        };
        let planes = nv12_plane_layout(&layout, visible).expect("the picture fits");
        assert_eq!(planes.plane(0).map(|plane| plane.offset), Some(4 * 320 + 8));
        assert_eq!(
            planes.plane(1).map(|plane| plane.offset),
            Some((192 + 2) * 320 + 8)
        );
    }
}
