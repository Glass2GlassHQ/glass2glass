// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use std::os::fd::{AsRawFd, OwnedFd};

use g2g_core::caps::framerate_q16_to_fraction;
use g2g_core::log::Target;
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, OwnedDmaBuf};
use g2g_core::{
    g2g_error, g2g_log, g2g_warn, AsyncElement, BitstreamAlignment, Caps, CapsConstraint, CapsSet,
    Colorimetry, ConfigureOutcome, Dim, ElementMetadata, G2gError, HardwareError, Interlace,
    OutputSink, PadTemplate, PadTemplates, PipelinePacket, PixelAspect, PropError, PropValue,
    PropertySpec, Rate, RawVideoFormat, VideoCodec,
};
use v4l::v4l2::vidioc;
use v4l::v4l_sys::{
    v4l2_format, v4l2_memory_V4L2_MEMORY_DMABUF, v4l2_memory_V4L2_MEMORY_MMAP, v4l2_plane,
};

use crate::dmabufmap::dmabuf_len;
use crate::h264parse::H264_MACROBLOCK_SIZE;
use crate::paddedrows::dmabuf_plane_layout;
use crate::v4l2ioctl::{v4l2_ioctl, zeroed_arg};
use crate::v4l2stateless::{
    buffer_arg, buffer_timestamp, find_m2m_nodes, open_node, planes_buffer_arg, request_buffers,
    set_controls, set_stream, Mapping, MediaRequest, RequestError, V4l2ExtControl,
    BITSTREAM_BYTES_PER_PIXEL, CAPTURE_TYPE, OUTPUT_TYPE, V4L2_BUF_FLAG_ERROR,
    V4L2_BUF_FLAG_REQUEST_FD, V4L2_CTRL_WHICH_REQUEST_VAL,
};

const V4L2_PIX_FMT_NV12M: u32 = u32::from_le_bytes(*b"NM12");
const V4L2_PIX_FMT_YUYV: u32 = u32::from_le_bytes(*b"YUYV");

const ENCODE_TIMEOUT_MS: i32 = 1000;
// one buffer slot per dma-buf the source cycles through, past that a slot is remapped
const IMPORT_SLOTS: usize = 8;
const MAX_INPUT_PLANES: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RateControl {
    ConstantQp,
    ConstantBitrate,
}

const RATE_CONTROL_NAMES: &[(&str, RateControl)] = &[
    ("cqp", RateControl::ConstantQp),
    ("cbr", RateControl::ConstantBitrate),
];

pub(crate) const DEFAULT_KEYFRAME_INTERVAL: u32 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateSettings {
    pub(crate) keyframe_interval: u32,
    pub(crate) quantizer: u8,
    pub(crate) rate_control: RateControl,
    pub(crate) bitrate: u32,
    pub(crate) qp_min: u8,
    pub(crate) qp_max: u8,
    pub(crate) qp_step: u8,
}

// the properties every V4L2 stateless encoder takes, then the codec's own
macro_rules! v4l2_encoder_properties {
    (
        quantizer: $quantizer:literal,
        quantizer_range: $range:literal,
        qp_min: $qp_min:literal,
        qp_max: $qp_max:literal,
        qp_step: $qp_step:literal,
        codec: [$($codec:expr),* $(,)?]
    ) => {{
        const PROPERTIES: &[::g2g_core::PropertySpec] = &[
            ::g2g_core::PropertySpec::new(
                "keyframe-interval",
                ::g2g_core::PropKind::Uint,
                "frames from one keyframe to the next, 0 = only the first frame and requested ones",
            )
            .with_default("30"),
            ::g2g_core::PropertySpec::new(
                "quantizer",
                ::g2g_core::PropKind::Uint,
                concat!("quantizer of every frame under cqp, of the first frame under cbr, ", $range),
            )
            .with_default($quantizer),
            ::g2g_core::PropertySpec::new(
                "rate-control",
                ::g2g_core::PropKind::Str,
                "rate control: cqp | cbr",
            )
            .with_default("cqp"),
            ::g2g_core::PropertySpec::new(
                "bitrate",
                ::g2g_core::PropKind::Uint,
                "target bitrate under cbr, bits/second",
            )
            .with_default("0"),
            ::g2g_core::PropertySpec::new(
                "qp-min",
                ::g2g_core::PropKind::Uint,
                concat!("lowest quantizer cbr picks, ", $range),
            )
            .with_default($qp_min),
            ::g2g_core::PropertySpec::new(
                "qp-max",
                ::g2g_core::PropKind::Uint,
                concat!("highest quantizer cbr picks, ", $range),
            )
            .with_default($qp_max),
            ::g2g_core::PropertySpec::new(
                "qp-step",
                ::g2g_core::PropKind::Uint,
                "most cbr changes the quantizer from one frame to the next",
            )
            .with_default($qp_step),
            $($codec),*
        ];
        PROPERTIES
    }};
}
pub(crate) use v4l2_encoder_properties;

#[derive(Debug, Clone, Copy)]
pub struct QuantizerScale {
    pub(crate) max: u8,
    // the quantizer step size relative to the other quantizers, coded bits fall in proportion to it
    pub(crate) step: fn(u8) -> f64,
}

impl QuantizerScale {
    // the quantizer whose step is nearest `step` in ratio
    fn nearest(&self, step: f64) -> u8 {
        let distance = |quantizer: u8| ((self.step)(quantizer) / step).ln().abs();
        (1..=self.max).fold(0, |nearest, quantizer| {
            if distance(quantizer) < distance(nearest) {
                quantizer
            } else {
                nearest
            }
        })
    }
}

// the smallest share of its budget cbr gives a frame while it pays back an overspend
const MIN_FRAME_SHARE: f64 = 0.125;
// IDRs on the 30 fps 1080p clip code 0.8 to 3.4 times the bits of the P frame after them at QP 51, more at lower QP
const FIRST_KEYFRAME_BUDGET_FRAMES: f64 = 4.0;

#[derive(Debug)]
pub(crate) struct BitrateControl {
    scale: QuantizerScale,
    pub(crate) bits_per_frame: f64,
    pub(crate) frames_per_second: f64,
    // the bits coded past the bitrate so far, down to one second of unspent bits
    excess_bits: f64,
    // the bits the last inter frame and the last keyframe took, times their quantizer step
    inter_frame_complexity: Option<f64>,
    keyframe_complexity: Option<f64>,
    qp: u8,
    warned_at_qp_max: bool,
}

impl BitrateControl {
    pub(crate) fn new(
        settings: &RateSettings,
        frames_per_second: f64,
        scale: QuantizerScale,
    ) -> Self {
        Self {
            scale,
            bits_per_frame: f64::from(settings.bitrate) / frames_per_second,
            frames_per_second,
            excess_bits: 0.0,
            inter_frame_complexity: None,
            keyframe_complexity: None,
            qp: settings.quantizer,
            warned_at_qp_max: false,
        }
    }

    fn complexity(&self, qp: u8, coded_bits: f64) -> f64 {
        coded_bits * (self.scale.step)(qp)
    }

    pub(crate) fn next_qp(&self, settings: &RateSettings) -> u8 {
        let (inter_frame, keyframe) = match (self.inter_frame_complexity, self.keyframe_complexity)
        {
            (None, None) => return self.qp,
            (Some(inter_frame), Some(keyframe)) => (inter_frame, keyframe),
            (Some(measured), None) | (None, Some(measured)) => (measured, measured),
        };
        let interval = f64::from(settings.keyframe_interval);
        let mean_complexity = if settings.keyframe_interval == 0 {
            inter_frame
        } else {
            (inter_frame * (interval - 1.0) + keyframe) / interval
        };
        let wanted_bits = (self.bits_per_frame - self.excess_bits / self.frames_per_second)
            .max(self.bits_per_frame * MIN_FRAME_SHARE);
        let ideal = self.scale.nearest(mean_complexity / wanted_bits);
        ideal
            .clamp(
                self.qp.saturating_sub(settings.qp_step),
                self.qp.saturating_add(settings.qp_step),
            )
            .clamp(settings.qp_min, settings.qp_max)
    }

    // the quantizer that fits the first keyframe into its budget, from a trial encode of it at `trial_qp`
    pub(crate) fn first_keyframe_qp(
        &self,
        settings: &RateSettings,
        trial_qp: u8,
        trial_bits: f64,
    ) -> Option<u8> {
        let measured = self.inter_frame_complexity.is_some() || self.keyframe_complexity.is_some();
        if measured {
            return None;
        }
        let budget = self.bits_per_frame * FIRST_KEYFRAME_BUDGET_FRAMES;
        let qp = self
            .scale
            .nearest(self.complexity(trial_qp, trial_bits) / budget)
            .clamp(settings.qp_min, settings.qp_max);
        (qp != trial_qp).then_some(qp)
    }

    pub(crate) fn record(&mut self, qp: u8, coded_bits: f64, keyframe: bool) {
        let complexity = Some(self.complexity(qp, coded_bits));
        if keyframe {
            self.keyframe_complexity = complexity;
        } else {
            self.inter_frame_complexity = complexity;
        }
        let credit_limit = -self.bits_per_frame * self.frames_per_second;
        self.excess_bits = (self.excess_bits + coded_bits - self.bits_per_frame).max(credit_limit);
        self.qp = qp;
    }

    // at qp-max and a second of bits over the bitrate, for the first time
    fn first_stuck_at_qp_max(&mut self, settings: &RateSettings) -> bool {
        let stuck = self.qp == settings.qp_max
            && self.excess_bits > self.bits_per_frame * self.frames_per_second;
        let first = stuck && !self.warned_at_qp_max;
        self.warned_at_qp_max |= stuck;
        first
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Picture {
    pub(crate) format: RawVideoFormat,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) framerate: Rate,
    pub(crate) colorimetry: Colorimetry,
    pub(crate) pixel_aspect: PixelAspect,
}

impl Picture {
    pub(crate) fn frame_rate_fraction(&self) -> Option<(u32, u32)> {
        match self.framerate {
            Rate::Fixed(q16) if q16 > 0 => Some(framerate_q16_to_fraction(q16)),
            _ => None,
        }
    }

    // VP8 macroblocks are 16 samples square as well
    pub(crate) fn width_mbs(&self) -> u32 {
        self.width.div_ceil(H264_MACROBLOCK_SIZE)
    }

    pub(crate) fn height_mbs(&self) -> u32 {
        self.height.div_ceil(H264_MACROBLOCK_SIZE)
    }

    fn v4l2_format(&self) -> u32 {
        match self.format {
            RawVideoFormat::Yuyv => V4L2_PIX_FMT_YUYV,
            _ => V4L2_PIX_FMT_NV12M,
        }
    }
}

fn encoder_refused(log_category: &'static str, reason: &str) -> G2gError {
    g2g_error!(Target::category(log_category), "cannot encode: {reason}");
    G2gError::CapsMismatch
}

#[derive(Debug)]
enum RawQueue {
    // system frames are copied into the driver's own buffer, one mapping per plane
    Copied(Vec<Mapping>),
    // the dma-buf fd queued in each buffer slot
    Imported(Vec<Option<i32>>),
}

#[derive(Debug)]
pub struct EncodeSession {
    log_category: &'static str,
    video: OwnedFd,
    request: MediaRequest,
    bitstream: Mapping,
    raw: RawQueue,
    bytes_per_line: Vec<usize>,
    // the bytes the H1 reads from each plane, macroblock rows included
    plane_sizes: Vec<usize>,
    pub(crate) encoded: u64,
}

impl EncodeSession {
    fn open(
        picture: &Picture,
        coded_format: u32,
        memory: u32,
        log_category: &'static str,
    ) -> Result<Self, G2gError> {
        let nodes = find_m2m_nodes(CAPTURE_TYPE, coded_format)?.ok_or_else(|| {
            encoder_refused(
                log_category,
                "no V4L2 stateless encoder for this codec with a media controller was found",
            )
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
        pix.pixelformat = coded_format;
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
                log_category,
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
            log_category,
            video,
            request,
            bitstream,
            raw,
            bytes_per_line,
            plane_sizes,
            encoded: 0,
        })
    }

    fn refused(&self, reason: &str) -> G2gError {
        encoder_refused(self.log_category, reason)
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

    // encodes `input` under the codec's `controls` and returns the bytes the driver coded
    pub(crate) fn run(
        &mut self,
        picture: &Picture,
        controls: &mut [V4l2ExtControl],
        input: &MemoryDomain,
    ) -> Result<&[u8], G2gError> {
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
                        return Err(self.refused(
                            "the dma-buf rows are not as far apart as the V4L2 encoder reads them",
                        ));
                    }
                    let end = source.offset + self.plane_sizes[index];
                    if end > buffer_length {
                        return Err(self.refused(
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
            controls,
        )?;
        let mut coded_plane: v4l2_plane = zeroed_arg();
        let mut coded = buffer_arg(CAPTURE_TYPE, 0, &mut coded_plane);
        // SAFETY: `V4l2Buffer` matches the kernel's `v4l2_buffer`, which VIDIOC_QBUF takes.
        unsafe { v4l2_ioctl(self.video.as_raw_fd(), vidioc::VIDIOC_QBUF, &mut coded) }?;
        let mut raw = planes_buffer_arg(OUTPUT_TYPE, index, memory, &mut planes[..plane_count]);
        raw.flags = V4L2_BUF_FLAG_REQUEST_FD;
        raw.request_fd = self.request.fd();
        raw.timestamp = buffer_timestamp(self.encoded);
        // SAFETY: as for the coded buffer, and the dma-buf stays open until the dequeue below.
        unsafe { v4l2_ioctl(self.video.as_raw_fd(), vidioc::VIDIOC_QBUF, &mut raw) }?;

        match self.request.queue_and_wait(ENCODE_TIMEOUT_MS) {
            Ok(()) => {}
            Err(RequestError::TimedOut) => {
                return Err(
                    self.refused("the V4L2 encoder did not finish a frame within the timeout")
                )
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
            return Err(self.refused("the V4L2 encoder flagged the frame as failed"));
        }
        self.bitstream
            .bytes()
            .get(..done_plane.bytesused as usize)
            .ok_or(G2gError::Hardware(HardwareError::V4l2(-1)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettingTakesEffect {
    Now,
    AtNextKeyframe,
}

pub trait V4l2StatelessEncoder: Default + Send + core::fmt::Debug {
    const LOG_CATEGORY: &'static str;
    const CODEC: VideoCodec;
    const ALIGNMENT: BitstreamAlignment;
    const CODED_PIXEL_FORMAT: u32;
    const QUANTIZER: QuantizerScale;
    const RATE_DEFAULTS: RateSettings;

    fn metadata() -> ElementMetadata;
    fn properties() -> &'static [PropertySpec];
    fn set_codec_property(
        &mut self,
        _name: &str,
        _value: &PropValue,
    ) -> Result<SettingTakesEffect, PropError> {
        Err(PropError::Unknown)
    }
    fn get_codec_property(&self, _name: &str) -> Option<PropValue> {
        None
    }
    // the coded frame, from the controls for `qp` and `keyframe` and the bytes `session` codes under them
    fn encode(
        &mut self,
        session: &mut EncodeSession,
        picture: &Picture,
        keyframe: bool,
        qp: u8,
        input: &MemoryDomain,
    ) -> Result<Vec<u8>, G2gError>;
}

#[derive(Debug)]
pub struct V4l2StatelessEnc<C: V4l2StatelessEncoder> {
    rate: RateSettings,
    codec: C,
    picture: Option<Picture>,
    session: Option<EncodeSession>,
    bitrate_control: Option<BitrateControl>,
    force_keyframe: bool,
    emitted: u64,
    caps_sent: bool,
}

impl<C: V4l2StatelessEncoder> Default for V4l2StatelessEnc<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: V4l2StatelessEncoder> V4l2StatelessEnc<C> {
    pub fn new() -> Self {
        Self {
            rate: C::RATE_DEFAULTS,
            codec: C::default(),
            picture: None,
            session: None,
            bitrate_control: None,
            force_keyframe: false,
            emitted: 0,
            caps_sent: false,
        }
    }

    pub fn is_available() -> bool {
        find_m2m_nodes(CAPTURE_TYPE, C::CODED_PIXEL_FORMAT).is_ok_and(|nodes| nodes.is_some())
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
            codec: C::CODEC,
            width,
            height,
            framerate,
            colorimetry,
            pixel_aspect,
            alignment: C::ALIGNMENT,
        }
    }

    pub(crate) fn keyframe_due(&self, encoded: u64) -> bool {
        let interval = u64::from(self.rate.keyframe_interval);
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
            self.session = Some(EncodeSession::open(
                &picture,
                C::CODED_PIXEL_FORMAT,
                memory,
                C::LOG_CATEGORY,
            )?);
        }
        if self.rate.rate_control == RateControl::ConstantBitrate && self.bitrate_control.is_none()
        {
            let (numerator, denominator) = picture.frame_rate_fraction().ok_or_else(|| {
                encoder_refused(C::LOG_CATEGORY, "rate-control=cbr needs a fixed frame rate")
            })?;
            if self.rate.bitrate == 0 {
                return Err(encoder_refused(
                    C::LOG_CATEGORY,
                    "rate-control=cbr needs a bitrate",
                ));
            }
            let frames_per_second = f64::from(numerator) / f64::from(denominator);
            self.bitrate_control = Some(BitrateControl::new(
                &self.rate,
                frames_per_second,
                C::QUANTIZER,
            ));
        }
        let mut qp = self
            .bitrate_control
            .as_ref()
            .map_or(self.rate.quantizer, |control| control.next_qp(&self.rate));
        let encoded = self.session.as_ref().map_or(0, |session| session.encoded);
        let keyframe = self.keyframe_due(encoded);
        let session = self.session.as_mut().ok_or(G2gError::NotConfigured)?;
        let mut coded = self
            .codec
            .encode(session, &picture, keyframe, qp, &domain)?;
        // hantro h1 takes the next inter frame's reference from the latest encode
        let first_keyframe_qp =
            self.bitrate_control
                .as_ref()
                .filter(|_| keyframe)
                .and_then(|control| {
                    control.first_keyframe_qp(&self.rate, qp, (coded.len() * 8) as f64)
                });
        if let Some(fitted) = first_keyframe_qp {
            qp = fitted;
            coded = self
                .codec
                .encode(session, &picture, keyframe, qp, &domain)?;
        }
        session.encoded += 1;
        if let Some(control) = self.bitrate_control.as_mut() {
            control.record(qp, (coded.len() * 8) as f64, keyframe);
            g2g_log!(
                Target::category(C::LOG_CATEGORY),
                "cbr coded {} bits at qp {qp}, {} bits over the bitrate",
                coded.len() * 8,
                control.excess_bits as i64
            );
            if control.first_stuck_at_qp_max(&self.rate) {
                g2g_warn!(
                    Target::category(C::LOG_CATEGORY),
                    "cbr cannot reach {} bits/second: the quantizer is at qp-max {} and the stream is {} bits over",
                    self.rate.bitrate,
                    self.rate.qp_max,
                    control.excess_bits as i64
                );
            }
        }
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
            Vec::from([(coded, pts_ns)]),
            &caps,
            out,
        )
        .await?;
        self.force_keyframe |= feedback.force_keyframe;
        Ok(())
    }
}

impl<C: V4l2StatelessEncoder> AsyncElement for V4l2StatelessEnc<C> {
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
            self.bitrate_control = None;
            self.caps_sent = false;
        }
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        C::metadata()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        C::properties()
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        let uint = |value: &PropValue| value.as_uint().ok_or(PropError::Type);
        let bounded = |value: &PropValue, max: u8| {
            u8::try_from(uint(value)?)
                .ok()
                .filter(|&number| number <= max)
                .ok_or(PropError::Value)
        };
        let quantizer_max = C::QUANTIZER.max;
        match name {
            "keyframe-interval" => {
                self.rate.keyframe_interval =
                    u32::try_from(uint(&value)?).map_err(|_| PropError::Value)?;
            }
            "quantizer" => self.rate.quantizer = bounded(&value, quantizer_max)?,
            "rate-control" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                let (_, mode) = RATE_CONTROL_NAMES
                    .iter()
                    .find(|(known, _)| *known == name)
                    .ok_or(PropError::Value)?;
                if *mode != self.rate.rate_control {
                    self.rate.rate_control = *mode;
                    self.bitrate_control = None;
                }
            }
            "bitrate" => {
                let bitrate = u32::try_from(uint(&value)?).map_err(|_| PropError::Value)?;
                if let Some(control) = self.bitrate_control.as_mut() {
                    if bitrate == 0 {
                        return Err(PropError::Value);
                    }
                    control.bits_per_frame = f64::from(bitrate) / control.frames_per_second;
                }
                self.rate.bitrate = bitrate;
            }
            "qp-min" => self.rate.qp_min = bounded(&value, self.rate.qp_max)?,
            "qp-max" => {
                let qp_max = bounded(&value, quantizer_max)?;
                if qp_max < self.rate.qp_min {
                    return Err(PropError::Value);
                }
                self.rate.qp_max = qp_max;
            }
            "qp-step" => self.rate.qp_step = bounded(&value, quantizer_max)?,
            _ => {
                let takes_effect = self.codec.set_codec_property(name, &value)?;
                self.force_keyframe |= takes_effect == SettingTakesEffect::AtNextKeyframe;
            }
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let rate = &self.rate;
        Some(match name {
            "keyframe-interval" => PropValue::Uint(u64::from(rate.keyframe_interval)),
            "quantizer" => PropValue::Uint(u64::from(rate.quantizer)),
            "rate-control" => {
                let (name, _) = RATE_CONTROL_NAMES
                    .iter()
                    .find(|(_, mode)| *mode == rate.rate_control)?;
                PropValue::Str((*name).into())
            }
            "bitrate" => PropValue::Uint(u64::from(rate.bitrate)),
            "qp-min" => PropValue::Uint(u64::from(rate.qp_min)),
            "qp-max" => PropValue::Uint(u64::from(rate.qp_max)),
            "qp-step" => PropValue::Uint(u64::from(rate.qp_step)),
            _ => return self.codec.get_codec_property(name),
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

impl<C: V4l2StatelessEncoder> PadTemplates for V4l2StatelessEnc<C> {
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
    use crate::v4l2h264enc::V4l2H264Encoder;
    use crate::v4l2vp8enc::V4l2Vp8Encoder;

    const FRAMES_PER_SECOND: u32 = 30;
    // a stand-in for the H1: a keyframe codes this many times the bits of an inter frame of the same content
    const KEYFRAME_TO_INTER_BITS: f64 = 6.0;
    const TARGET_BITRATE: u32 = 4_000_000;
    const SECONDS: u64 = 10;
    const SCENE_CHANGE_SECOND: u64 = 5;
    // inter frame bits at a quantizer step of 1, which settle near H.264 QP 25 and QP 37 at the target bitrate
    const EASY_SCENE_BITS: f64 = 2.0e6;
    const HARD_SCENE_BITS: f64 = 8.0e6;

    fn cbr_settings<C: V4l2StatelessEncoder>() -> RateSettings {
        RateSettings {
            rate_control: RateControl::ConstantBitrate,
            bitrate: TARGET_BITRATE,
            ..C::RATE_DEFAULTS
        }
    }

    fn modelled_bits(scale: &QuantizerScale, qp: u8, scene_bits: f64, keyframe: bool) -> f64 {
        let frame_bits = if keyframe {
            scene_bits * KEYFRAME_TO_INTER_BITS
        } else {
            scene_bits
        };
        frame_bits / (scale.step)(qp)
    }

    // the bits each second codes, and the quantizer of each frame
    fn run_cbr<C: V4l2StatelessEncoder>(scene_bits_at: impl Fn(u64) -> f64) -> (Vec<f64>, Vec<u8>) {
        let settings = cbr_settings::<C>();
        let scale = C::QUANTIZER;
        let frames_per_second = u64::from(FRAMES_PER_SECOND);
        let mut control = BitrateControl::new(&settings, f64::from(FRAMES_PER_SECOND), scale);
        let mut seconds = alloc::vec![0.0; SECONDS as usize];
        let mut quantizers = Vec::new();
        for frame in 0..SECONDS * frames_per_second {
            let second = frame / frames_per_second;
            let scene_bits = scene_bits_at(second);
            let keyframe = frame.is_multiple_of(u64::from(settings.keyframe_interval));
            let mut qp = control.next_qp(&settings);
            let mut bits = modelled_bits(&scale, qp, scene_bits, keyframe);
            if let Some(fitted) = control
                .first_keyframe_qp(&settings, qp, bits)
                .filter(|_| keyframe)
            {
                qp = fitted;
                bits = modelled_bits(&scale, qp, scene_bits, keyframe);
            }
            control.record(qp, bits, keyframe);
            seconds[second as usize] += bits;
            quantizers.push(qp);
        }
        (seconds, quantizers)
    }

    fn scene_change(second: u64) -> f64 {
        if second < SCENE_CHANGE_SECOND {
            EASY_SCENE_BITS
        } else {
            HARD_SCENE_BITS
        }
    }

    // the H.264 model codes 0.95 of the bitrate in the first second, then settles within 0.5% from the second second
    const FIRST_SECOND_TOLERANCE: f64 = 0.1;
    const START_SETTLE_SECONDS: u64 = 1;
    // and settles within 0.5% four seconds after the scene change
    const SCENE_CHANGE_SETTLE_SECONDS: u64 = 4;
    const SETTLED_TOLERANCE: f64 = 0.01;
    // the H.264 model's ten seconds measure 0.995 of the bitrate, the rest of the scene change second's overshoot unpaid
    const WHOLE_RUN_TOLERANCE: f64 = 0.01;

    fn assert_cbr_holds_each_second_near_the_bitrate<C: V4l2StatelessEncoder>() {
        let target = f64::from(TARGET_BITRATE);
        let (seconds, _) = run_cbr::<C>(scene_change);
        let first_second = seconds[0] / target;
        assert!(
            (first_second - 1.0).abs() < FIRST_SECOND_TOLERANCE,
            "the first second coded {first_second} of the bitrate"
        );
        let settled = (START_SETTLE_SECONDS..SCENE_CHANGE_SECOND)
            .chain(SCENE_CHANGE_SECOND + SCENE_CHANGE_SETTLE_SECONDS..SECONDS);
        for second in settled {
            let ratio = seconds[second as usize] / target;
            assert!(
                (ratio - 1.0).abs() < SETTLED_TOLERANCE,
                "second {second} coded {ratio} of the bitrate"
            );
        }
        let whole_run = seconds.iter().sum::<f64>() / (target * SECONDS as f64);
        assert!(
            (whole_run - 1.0).abs() < WHOLE_RUN_TOLERANCE,
            "the run coded {whole_run} of the bitrate"
        );
    }

    #[test]
    fn h264_cbr_holds_each_second_near_the_bitrate_across_a_scene_change() {
        assert_cbr_holds_each_second_near_the_bitrate::<V4l2H264Encoder>();
    }

    #[test]
    fn vp8_cbr_holds_each_second_near_the_bitrate_across_a_scene_change() {
        assert_cbr_holds_each_second_near_the_bitrate::<V4l2Vp8Encoder>();
    }

    #[test]
    fn cbr_warns_once_when_qp_max_cannot_reach_the_bitrate() {
        let settings = cbr_settings::<V4l2H264Encoder>();
        let scale = V4l2H264Encoder::QUANTIZER;
        let mut control = BitrateControl::new(&settings, f64::from(FRAMES_PER_SECOND), scale);
        // twice the bitrate at qp-max
        let overspend_bits = 2.0 * control.bits_per_frame;
        let mut warnings = 0;
        for _ in 0..SECONDS * u64::from(FRAMES_PER_SECOND) {
            control.record(settings.qp_max, overspend_bits, false);
            warnings += u32::from(control.first_stuck_at_qp_max(&settings));
        }
        assert_eq!(warnings, 1);
    }

    #[test]
    fn cbr_does_not_warn_at_qp_max_on_budget_or_over_budget_below_it() {
        let settings = cbr_settings::<V4l2H264Encoder>();
        let scale = V4l2H264Encoder::QUANTIZER;
        let frames = SECONDS * u64::from(FRAMES_PER_SECOND);
        let mut on_budget = BitrateControl::new(&settings, f64::from(FRAMES_PER_SECOND), scale);
        for _ in 0..frames {
            on_budget.record(settings.qp_max, on_budget.bits_per_frame, false);
            assert!(!on_budget.first_stuck_at_qp_max(&settings));
        }
        let mut below_qp_max = BitrateControl::new(&settings, f64::from(FRAMES_PER_SECOND), scale);
        for _ in 0..frames {
            below_qp_max.record(
                settings.qp_max - 1,
                2.0 * below_qp_max.bits_per_frame,
                false,
            );
            assert!(!below_qp_max.first_stuck_at_qp_max(&settings));
        }
    }

    #[test]
    fn keyframes_follow_the_interval() {
        let mut encoder = V4l2StatelessEnc::<V4l2Vp8Encoder>::new();
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
