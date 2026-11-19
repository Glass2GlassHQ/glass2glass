// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::future::Future;
use core::mem::{offset_of, size_of};
use core::pin::Pin;
use std::os::fd::{AsRawFd, OwnedFd};

use g2g_core::caps::framerate_q16_to_fraction;
use g2g_core::g2g_error;
use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, OwnedDmaBuf};
use g2g_core::{
    AsyncElement, BitstreamAlignment, Caps, CapsConstraint, CapsSet, Colorimetry, ConfigureOutcome,
    Dim, ElementMetadata, G2gError, HardwareError, Interlace, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, PixelAspect, PropError, PropKind, PropValue, PropertySpec, Rate,
    RawVideoFormat, VideoCodec,
};
use v4l::v4l2::vidioc;
use v4l::v4l_sys::{
    v4l2_format, v4l2_memory_V4L2_MEMORY_DMABUF, v4l2_memory_V4L2_MEMORY_MMAP, v4l2_plane,
};

use crate::annexb::{add_emulation_prevention, BitWriter, ANNEX_B_START_CODE};
use crate::dmabufmap::dmabuf_len;
use crate::h264dpb::{H264_NAL_IDR_SLICE, H264_NAL_SLICE};
use crate::h264parse::{
    h264_level_for, H264_MACROBLOCK_SIZE, H264_MAX_QP, H264_NAL_PPS, H264_NAL_SPS,
};
use crate::paddedrows::dmabuf_plane_layout;
use crate::v4l2ioctl::{v4l2_ioctl, zeroed_arg};
use crate::v4l2stateless::{
    buffer_arg, buffer_timestamp, control, find_m2m_nodes, open_node, planes_buffer_arg,
    request_buffers, set_controls, set_stream, timestamp_ns, Mapping, MediaRequest, RequestError,
    BITSTREAM_BYTES_PER_PIXEL, CAPTURE_TYPE, OUTPUT_TYPE, V4L2_BUF_FLAG_ERROR,
    V4L2_BUF_FLAG_REQUEST_FD, V4L2_CTRL_WHICH_REQUEST_VAL, V4L2_PIX_FMT_H264_SLICE,
};
use crate::vui::{write_vui_head, VideoSignalType, VuiHead};

// linux/v4l2-controls.h of the STMicroelectronics v6.6 kernel, mainline Linux has no H.264 encode controls
const V4L2_CID_STATELESS_H264_ENCODE_PARAMS: u32 = 0x00a4_0908;
const V4L2_CID_STATELESS_H264_ENCODE_RC: u32 = 0x00a4_0909;
const V4L2_H264_SLICE_TYPE_P: u8 = 0;
const V4L2_H264_SLICE_TYPE_I: u8 = 2;
const V4L2_H264_ENCODE_FLAG_ENTROPY_CABAC: u32 = 0x01;
const V4L2_H264_ENCODE_FLAG_TRANSFORM_8X8_MODE: u32 = 0x02;
const V4L2_PIX_FMT_NV12M: u32 = u32::from_le_bytes(*b"NM12");
const V4L2_PIX_FMT_YUYV: u32 = u32::from_le_bytes(*b"YUYV");

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

const ENCODE_TIMEOUT_MS: i32 = 1000;
// the Hantro H1 writes frame_num in 16 bits and orders pictures by frame_num alone
const LOG2_MAX_FRAME_NUM_MINUS4: u32 = 12;
const PIC_ORDER_CNT_TYPE_FROM_FRAME_NUM: u32 = 2;
// gstreamer's v4l2slh264enc offsets chroma QP by this much
const CHROMA_QP_INDEX_OFFSET: i32 = 4;
const PIC_INIT_QP_BASE: i32 = 26;
const NAL_REF_IDC_HIGHEST: u8 = 3;
// one buffer slot per dma-buf the source cycles through, past that a slot is remapped
const IMPORT_SLOTS: usize = 8;
const MAX_INPUT_PLANES: usize = 2;
// H.264 E.2.1 defaults, written out because the reorder bound after them must be present
const MAX_BYTES_PER_PIC_DENOM: u32 = 2;
const MAX_BITS_PER_MB_DENOM: u32 = 1;
const LOG2_MAX_MV_LENGTH: u32 = 16;

const DEFAULT_KEYFRAME_INTERVAL: u32 = 30;
const DEFAULT_QUANTIZER: u8 = 18;
const MAX_CABAC_INIT_IDC: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Settings {
    keyframe_interval: u32,
    quantizer: u8,
    cabac: bool,
    cabac_init_idc: u8,
    dct8x8: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            keyframe_interval: DEFAULT_KEYFRAME_INTERVAL,
            quantizer: DEFAULT_QUANTIZER,
            cabac: false,
            cabac_init_idc: 0,
            dct8x8: false,
        }
    }
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

#[derive(Debug, Clone, PartialEq)]
struct Picture {
    format: RawVideoFormat,
    width: u32,
    height: u32,
    framerate: Rate,
    colorimetry: Colorimetry,
    pixel_aspect: PixelAspect,
}

impl Picture {
    fn frame_rate_fraction(&self) -> Option<(u32, u32)> {
        match self.framerate {
            Rate::Fixed(q16) if q16 > 0 => Some(framerate_q16_to_fraction(q16)),
            _ => None,
        }
    }

    fn width_mbs(&self) -> u32 {
        self.width.div_ceil(H264_MACROBLOCK_SIZE)
    }

    fn height_mbs(&self) -> u32 {
        self.height.div_ceil(H264_MACROBLOCK_SIZE)
    }

    fn v4l2_format(&self) -> u32 {
        match self.format {
            RawVideoFormat::Yuyv => V4L2_PIX_FMT_YUYV,
            _ => V4L2_PIX_FMT_NV12M,
        }
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

#[derive(Debug)]
enum RawQueue {
    // system frames are copied into the driver's own buffer, one mapping per plane
    Copied(Vec<Mapping>),
    // the dma-buf fd queued in each buffer slot
    Imported(Vec<Option<i32>>),
}

#[derive(Debug)]
struct EncodeSession {
    video: OwnedFd,
    request: MediaRequest,
    bitstream: Mapping,
    raw: RawQueue,
    bytes_per_line: Vec<usize>,
    // the bytes the H1 reads from each plane, macroblock rows included
    plane_sizes: Vec<usize>,
    encoded: u64,
    frame_num: u16,
    idr_pic_id: u16,
    // the pic_init_qp of the PPS the stream carries, which the slice QP is coded against
    pic_init_qp: u8,
}

fn encoder_refused(reason: &str) -> G2gError {
    g2g_error!(
        Target::category(short_type_name::<V4l2SlH264Enc>()),
        "cannot encode: {reason}"
    );
    G2gError::CapsMismatch
}

impl EncodeSession {
    fn open(picture: &Picture, memory: u32) -> Result<Self, G2gError> {
        let nodes = find_m2m_nodes(CAPTURE_TYPE, V4L2_PIX_FMT_H264_SLICE)?.ok_or_else(|| {
            encoder_refused("no V4L2 stateless H.264 encoder with a media controller was found")
        })?;
        let video = open_node(&nodes.video)?;
        let media = open_node(&nodes.media)?;

        // the coded format goes first, setting it resets the raw format
        let mut coded: v4l2_format = zeroed_arg();
        coded.type_ = CAPTURE_TYPE;
        // SAFETY: `pix_mp` is the union member multi-plane buffer types use.
        let pix = unsafe { &mut coded.fmt.pix_mp };
        pix.width = picture.width;
        pix.height = picture.height;
        pix.pixelformat = V4L2_PIX_FMT_H264_SLICE;
        pix.num_planes = 1;
        pix.plane_fmt[0].sizeimage = picture.width * picture.height * BITSTREAM_BYTES_PER_PIXEL;
        // SAFETY: `v4l2_format` is VIDIOC_S_FMT's argument type.
        unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_S_FMT, &mut coded) }?;

        let plane_count = picture.format.plane_count();
        let mut raw: v4l2_format = zeroed_arg();
        raw.type_ = OUTPUT_TYPE;
        // SAFETY: as for the coded format.
        let pix = unsafe { &mut raw.fmt.pix_mp };
        pix.width = picture.width;
        pix.height = picture.height;
        pix.pixelformat = picture.v4l2_format();
        pix.num_planes = plane_count as u8;
        // SAFETY: as for the coded format.
        unsafe { v4l2_ioctl(video.as_raw_fd(), vidioc::VIDIOC_S_FMT, &mut raw) }?;
        // SAFETY: the driver filled `pix_mp` for the multi-plane output queue.
        let pix = unsafe { raw.fmt.pix_mp };
        // the H1 reads whole macroblocks, so the driver rounds the raw size up to them
        let coded_extent = (
            picture.width_mbs() * H264_MACROBLOCK_SIZE,
            picture.height_mbs() * H264_MACROBLOCK_SIZE,
        );
        if pix.pixelformat != picture.v4l2_format()
            || (pix.width, pix.height) != coded_extent
            || usize::from(pix.num_planes) != plane_count
        {
            return Err(encoder_refused(
                "the V4L2 encoder does not take this format and size as it is",
            ));
        }
        let plane_formats = &pix.plane_fmt[..plane_count];
        let bytes_per_line = plane_formats
            .iter()
            .map(|plane| plane.bytesperline as usize)
            .collect();
        let plane_sizes = plane_formats
            .iter()
            .map(|plane| plane.sizeimage as usize)
            .collect();

        request_buffers(&video, CAPTURE_TYPE, v4l2_memory_V4L2_MEMORY_MMAP, 1)?;
        let bitstream = Mapping::of_buffer(&video, CAPTURE_TYPE, 0)?;
        let raw = if memory == v4l2_memory_V4L2_MEMORY_DMABUF {
            let slots = request_buffers(&video, OUTPUT_TYPE, memory, IMPORT_SLOTS as u32)?;
            RawQueue::Imported(alloc::vec![None; slots as usize])
        } else {
            request_buffers(&video, OUTPUT_TYPE, memory, 1)?;
            RawQueue::Copied(Mapping::of_planes(&video, OUTPUT_TYPE, 0, plane_count)?)
        };
        set_stream(&video, OUTPUT_TYPE, true)?;
        set_stream(&video, CAPTURE_TYPE, true)?;
        let request = MediaRequest::alloc(&media)?;
        Ok(Self {
            video,
            request,
            bitstream,
            raw,
            bytes_per_line,
            plane_sizes,
            encoded: 0,
            frame_num: 0,
            idr_pic_id: 0,
            pic_init_qp: 0,
        })
    }

    // copy a tight system frame into the driver's planes, each `bytes_per_line` apart
    fn copy_frame(&mut self, picture: &Picture, pixels: &[u8]) -> Result<(), G2gError> {
        let RawQueue::Copied(planes) = &mut self.raw else {
            return Err(G2gError::UnsupportedDomain);
        };
        let format = picture.format;
        for (index, plane) in planes.iter_mut().enumerate() {
            let offset = format
                .plane_offset(index, picture.width, picture.height)
                .and_then(|offset| usize::try_from(offset).ok());
            let row_bytes = format.plane_stride(index, picture.width);
            let rows = format.plane_rows(index, picture.height);
            let (Some(offset), Some(row_bytes), Some(rows)) = (offset, row_bytes, rows) else {
                return Err(G2gError::CapsMismatch);
            };
            let (row_bytes, stride) = (row_bytes as usize, self.bytes_per_line[index]);
            let target = plane.bytes_mut();
            for row in 0..rows as usize {
                let source = offset + row * row_bytes;
                let source = pixels
                    .get(source..source + row_bytes)
                    .ok_or(G2gError::CapsMismatch)?;
                target
                    .get_mut(row * stride..row * stride + row_bytes)
                    .ok_or(G2gError::CapsMismatch)?
                    .copy_from_slice(source);
            }
        }
        Ok(())
    }

    // the buffer slot that last carried this dma-buf, else the next one round-robin
    fn import_slot(&mut self, dmabuf: &OwnedDmaBuf) -> Result<u32, G2gError> {
        let RawQueue::Imported(slots) = &mut self.raw else {
            return Err(G2gError::UnsupportedDomain);
        };
        let fd = dmabuf.as_raw();
        let slot = slots
            .iter()
            .position(|queued| *queued == Some(fd))
            .unwrap_or(self.encoded as usize % slots.len());
        slots[slot] = Some(fd);
        Ok(slot as u32)
    }

    fn encode(
        &mut self,
        picture: &Picture,
        settings: &Settings,
        keyframe: bool,
        input: &MemoryDomain,
    ) -> Result<Vec<u8>, G2gError> {
        let picture_number = self.encoded;
        let mut params = V4l2CtrlH264EncodeParams {
            cabac_init_idc: settings.cabac_init_idc,
            chroma_qp_index_offset: CHROMA_QP_INDEX_OFFSET as i8,
            flags: settings.encode_flags(),
            nal_reference_idc: NAL_REF_IDC_HIGHEST,
            ..Default::default()
        };
        let mut parameter_sets = Vec::new();
        if keyframe {
            self.pic_init_qp = settings.quantizer;
            parameter_sets = sequence_parameter_set(picture, settings)
                .ok_or_else(|| encoder_refused("no H.264 level or VUI fits this picture"))?;
            parameter_sets.extend(picture_parameter_set(settings, self.pic_init_qp));
            self.idr_pic_id = self.idr_pic_id.wrapping_add(1);
            self.frame_num = 0;
            params.slice_type = V4L2_H264_SLICE_TYPE_I;
            params.nalu_type = H264_NAL_IDR_SLICE;
        } else {
            self.frame_num = self.frame_num.wrapping_add(1);
            params.slice_type = V4L2_H264_SLICE_TYPE_P;
            params.nalu_type = H264_NAL_SLICE;
            params.reference_ts = timestamp_ns(picture_number - 1);
        }
        params.frame_num = self.frame_num;
        params.idr_pic_id = self.idr_pic_id;
        params.pic_init_qp_minus26 = (i32::from(self.pic_init_qp) - PIC_INIT_QP_BASE) as i8;
        let mut rate_control = V4l2CtrlH264EncodeRc {
            qp: u32::from(settings.quantizer),
            ..Default::default()
        };

        let mut planes: [v4l2_plane; MAX_INPUT_PLANES] = [zeroed_arg(), zeroed_arg()];
        let plane_count = self.bytes_per_line.len();
        let (memory, index) = match input {
            MemoryDomain::System(pixels) => {
                self.copy_frame(picture, pixels.as_slice())?;
                for (plane, size) in planes.iter_mut().zip(&self.plane_sizes) {
                    plane.bytesused = *size as u32;
                }
                (v4l2_memory_V4L2_MEMORY_MMAP, 0)
            }
            MemoryDomain::DmaBuf(dmabuf) => {
                let (layout, _) =
                    dmabuf_plane_layout(dmabuf, picture.format, picture.width, picture.height)
                        .ok_or(G2gError::CapsMismatch)?;
                let buffer_length = dmabuf_len(dmabuf)?;
                for (index, plane) in planes[..plane_count].iter_mut().enumerate() {
                    let source = layout.plane(index).ok_or(G2gError::CapsMismatch)?;
                    if source.stride != self.bytes_per_line[index] {
                        return Err(encoder_refused(
                            "the dma-buf rows are not as far apart as the V4L2 encoder reads them",
                        ));
                    }
                    let end = source.offset + self.plane_sizes[index];
                    if end > buffer_length {
                        return Err(encoder_refused(
                            "the dma-buf ends before the macroblock rows the V4L2 encoder reads",
                        ));
                    }
                    plane.m.fd = dmabuf.as_raw();
                    // vb2 refuses a dma-buf plane whose length is short of its bytesused
                    plane.length = buffer_length as u32;
                    plane.data_offset = source.offset as u32;
                    plane.bytesused = end as u32;
                }
                (v4l2_memory_V4L2_MEMORY_DMABUF, self.import_slot(dmabuf)?)
            }
            _ => return Err(G2gError::UnsupportedDomain),
        };

        self.request.reinit()?;
        set_controls(
            &self.video,
            V4L2_CTRL_WHICH_REQUEST_VAL,
            self.request.fd(),
            &mut [
                control(V4L2_CID_STATELESS_H264_ENCODE_PARAMS, &mut params),
                control(V4L2_CID_STATELESS_H264_ENCODE_RC, &mut rate_control),
            ],
        )?;
        let mut coded_plane: v4l2_plane = zeroed_arg();
        let mut coded = buffer_arg(CAPTURE_TYPE, 0, &mut coded_plane);
        // SAFETY: `V4l2Buffer` matches the kernel's `v4l2_buffer`, which VIDIOC_QBUF takes.
        unsafe { v4l2_ioctl(self.video.as_raw_fd(), vidioc::VIDIOC_QBUF, &mut coded) }?;
        let mut raw = planes_buffer_arg(OUTPUT_TYPE, index, memory, &mut planes[..plane_count]);
        raw.flags = V4L2_BUF_FLAG_REQUEST_FD;
        raw.request_fd = self.request.fd();
        raw.timestamp = buffer_timestamp(picture_number);
        // SAFETY: as for the coded buffer, and the dma-buf stays open until the dequeue below.
        unsafe { v4l2_ioctl(self.video.as_raw_fd(), vidioc::VIDIOC_QBUF, &mut raw) }?;

        match self.request.queue_and_wait(ENCODE_TIMEOUT_MS) {
            Ok(()) => {}
            Err(RequestError::TimedOut) => {
                return Err(encoder_refused(
                    "the V4L2 encoder did not finish a frame within the timeout",
                ))
            }
            Err(RequestError::Device(error)) => return Err(error),
        }
        let mut done_planes: [v4l2_plane; MAX_INPUT_PLANES] = [zeroed_arg(), zeroed_arg()];
        let mut done_raw =
            planes_buffer_arg(OUTPUT_TYPE, 0, memory, &mut done_planes[..plane_count]);
        // SAFETY: as for the coded buffer.
        unsafe { v4l2_ioctl(self.video.as_raw_fd(), vidioc::VIDIOC_DQBUF, &mut done_raw) }?;
        let mut done_plane: v4l2_plane = zeroed_arg();
        let mut done = buffer_arg(CAPTURE_TYPE, 0, &mut done_plane);
        // SAFETY: as for the coded buffer.
        unsafe { v4l2_ioctl(self.video.as_raw_fd(), vidioc::VIDIOC_DQBUF, &mut done) }?;
        if done.flags & V4L2_BUF_FLAG_ERROR != 0 {
            return Err(encoder_refused(
                "the V4L2 encoder flagged the frame as failed",
            ));
        }
        self.encoded += 1;
        let coded = self
            .bitstream
            .bytes()
            .get(..done_plane.bytesused as usize)
            .ok_or(G2gError::Hardware(HardwareError::V4l2(-1)))?;
        let mut access_unit = parameter_sets;
        access_unit.extend_from_slice(coded);
        Ok(access_unit)
    }
}

#[derive(Debug, Default)]
pub struct V4l2SlH264Enc {
    settings: Settings,
    picture: Option<Picture>,
    session: Option<EncodeSession>,
    force_keyframe: bool,
    emitted: u64,
    caps_sent: bool,
}

impl V4l2SlH264Enc {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_available() -> bool {
        find_m2m_nodes(CAPTURE_TYPE, V4L2_PIX_FMT_H264_SLICE).is_ok_and(|nodes| nodes.is_some())
    }

    fn input_caps_set() -> CapsSet {
        CapsSet::from_alternatives(
            [RawVideoFormat::Nv12, RawVideoFormat::Yuyv]
                .into_iter()
                .map(|format| Caps::RawVideo {
                    format,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                    interlace: Interlace::Any,
                    colorimetry: Colorimetry::UNKNOWN,
                    pixel_aspect: PixelAspect::Any,
                })
                .collect(),
        )
    }

    fn output_caps(
        width: Dim,
        height: Dim,
        framerate: Rate,
        colorimetry: Colorimetry,
        pixel_aspect: PixelAspect,
    ) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width,
            height,
            framerate,
            colorimetry,
            pixel_aspect,
            alignment: BitstreamAlignment::AccessUnit,
        }
    }

    // settings in the SPS or PPS take effect at the next IDR, which starts at once
    fn set_stream_setting(&mut self, change: impl FnOnce(&mut Settings)) {
        let previous = self.settings;
        change(&mut self.settings);
        self.force_keyframe |= self.settings != previous;
    }

    fn keyframe_due(&self, encoded: u64) -> bool {
        let interval = u64::from(self.settings.keyframe_interval);
        // zero is a multiple of every interval, and the only multiple of a zero interval
        self.force_keyframe || encoded.is_multiple_of(interval)
    }

    async fn encode_frame(
        &mut self,
        domain: MemoryDomain,
        pts_ns: u64,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let picture = self.picture.clone().ok_or(G2gError::NotConfigured)?;
        if self.session.is_none() {
            let memory = match domain {
                MemoryDomain::DmaBuf(_) => v4l2_memory_V4L2_MEMORY_DMABUF,
                _ => v4l2_memory_V4L2_MEMORY_MMAP,
            };
            self.session = Some(EncodeSession::open(&picture, memory)?);
        }
        let encoded = self.session.as_ref().map_or(0, |session| session.encoded);
        let keyframe = self.keyframe_due(encoded);
        let session = self.session.as_mut().ok_or(G2gError::NotConfigured)?;
        let access_unit = session.encode(&picture, &self.settings, keyframe, &domain)?;
        if keyframe {
            self.force_keyframe = false;
        }
        let caps = Self::output_caps(
            Dim::Fixed(picture.width),
            Dim::Fixed(picture.height),
            picture.framerate.clone(),
            picture.colorimetry,
            picture.pixel_aspect,
        );
        let feedback = crate::encoder_base::emit_packets(
            &mut self.caps_sent,
            &mut self.emitted,
            Vec::from([(access_unit, pts_ns)]),
            &caps,
            out,
        )
        .await?;
        self.force_keyframe |= feedback.force_keyframe;
        Ok(())
    }
}

impl AsyncElement for V4l2SlH264Enc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        Some(g2g_core::meta::Transform::Encode)
    }

    fn handles_keyframe_requests(&self) -> bool {
        true
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::DmaBuf)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Self::input_caps_set()
            .alternatives()
            .iter()
            .find_map(|alternative| upstream_caps.intersect(alternative).ok())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12 | RawVideoFormat::Yuyv,
                width,
                height,
                framerate,
                colorimetry,
                pixel_aspect,
                ..
            } => CapsSet::one(Self::output_caps(
                width.clone(),
                height.clone(),
                framerate.clone(),
                *colorimetry,
                *pixel_aspect,
            )),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format: format @ (RawVideoFormat::Nv12 | RawVideoFormat::Yuyv),
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate,
            colorimetry,
            pixel_aspect,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        // 4:2:0 and 4:2:2 crop and subsample in pairs of samples
        if width % 2 != 0 || height % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }
        let picture = Picture {
            format: *format,
            width: *width,
            height: *height,
            framerate: framerate.clone(),
            colorimetry: *colorimetry,
            pixel_aspect: *pixel_aspect,
        };
        if self.picture.as_ref() != Some(&picture) {
            self.picture = Some(picture);
            self.session = None;
            self.caps_sent = false;
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "V4L2 stateless H.264 encoder",
            "Codec/Encoder/Video/Hardware",
            "Encodes H.264 on a V4L2 stateless encoder through the media request API",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "keyframe-interval",
                PropKind::Uint,
                "frames from one IDR to the next, 0 = only the first frame and requested ones",
            )
            .with_default("30"),
            PropertySpec::new("quantizer", PropKind::Uint, "QP of every frame, 0..51")
                .with_default("18"),
            PropertySpec::new(
                "cabac",
                PropKind::Bool,
                "CABAC entropy coding, Main profile",
            )
            .with_default("false"),
            PropertySpec::new("cabac-init-idc", PropKind::Uint, "CABAC init table, 0..2")
                .with_default("0"),
            PropertySpec::new("dct8x8", PropKind::Bool, "8x8 transform, High profile")
                .with_default("false"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let uint = |value: &PropValue| value.as_uint().ok_or(PropError::Type);
        let flag = |value: &PropValue| value.as_bool().ok_or(PropError::Type);
        let bounded = |value: &PropValue, max: u8| {
            u8::try_from(uint(value)?)
                .ok()
                .filter(|&number| number <= max)
                .ok_or(PropError::Value)
        };
        match name {
            "keyframe-interval" => {
                self.settings.keyframe_interval =
                    u32::try_from(uint(&value)?).map_err(|_| PropError::Value)?;
            }
            "quantizer" => self.settings.quantizer = bounded(&value, H264_MAX_QP)?,
            "cabac" => {
                let cabac = flag(&value)?;
                self.set_stream_setting(|settings| settings.cabac = cabac);
            }
            "cabac-init-idc" => {
                let cabac_init_idc = bounded(&value, MAX_CABAC_INIT_IDC)?;
                self.set_stream_setting(|settings| settings.cabac_init_idc = cabac_init_idc);
            }
            "dct8x8" => {
                let dct8x8 = flag(&value)?;
                self.set_stream_setting(|settings| settings.dct8x8 = dct8x8);
            }
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let settings = &self.settings;
        Some(match name {
            "keyframe-interval" => PropValue::Uint(u64::from(settings.keyframe_interval)),
            "quantizer" => PropValue::Uint(u64::from(settings.quantizer)),
            "cabac" => PropValue::Bool(settings.cabac),
            "cabac-init-idc" => PropValue::Uint(u64::from(settings.cabac_init_idc)),
            "dct8x8" => PropValue::Bool(settings.dct8x8),
            _ => return None,
        })
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let pts_ns = frame.timing.pts_ns;
                    self.encode_frame(frame.domain, pts_ns, out).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for V4l2SlH264Enc {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(Self::input_caps_set()),
            PadTemplate::source(CapsSet::one(Self::output_caps(
                Dim::Any,
                Dim::Any,
                Rate::Any,
                Colorimetry::UNKNOWN,
                PixelAspect::Any,
            ))),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annexb::strip_emulation_prevention;
    use crate::h264parse::{parse_h264_pps, parse_h264_sps};

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

    #[test]
    fn keyframes_follow_the_interval() {
        let mut encoder = V4l2SlH264Enc::new();
        encoder
            .set_property("keyframe-interval", PropValue::Uint(10))
            .unwrap();
        let keyframes: Vec<u64> = (0..25).filter(|&n| encoder.keyframe_due(n)).collect();
        assert_eq!(keyframes, [0, 10, 20]);
        encoder
            .set_property("keyframe-interval", PropValue::Uint(0))
            .unwrap();
        let keyframes: Vec<u64> = (0..25).filter(|&n| encoder.keyframe_due(n)).collect();
        assert_eq!(keyframes, [0]);
    }
}
