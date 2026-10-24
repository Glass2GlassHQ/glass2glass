//! Streaming-decoder adapter: presents g2g's Vulkan Video hardware decoders in
//! the chunk-at-a-time, CPU-frame shape a wgpu-based viewer's async video
//! backend consumes (the wgpu-texture wedge, see the wedge-strategy note).
//!
//! g2g depends on no viewer crates; this module exposes the conventional
//! decode-output layout such consumers use:
//!
//! * `DecodedVideoFrame::data` is I420 planar (Y, then U, then V, packed with no
//!   stride padding), the layout software decode backends (dav1d/ffmpeg)
//!   conventionally emit for 4:2:0.
//! * [`VideoPixelLayout`] / [`VideoColorRange`] / [`VideoMatrixCoefficients`]
//!   carry the colour metadata a renderer needs for its own YUV->RGB.
//!
//! A consumer wraps [`VulkanStreamDecoder`] in one async-decoder impl:
//! `submit_chunk` -> [`VulkanStreamDecoder::submit_chunk`], mapping each
//! [`DecodedVideoFrame`] onto its own frame type; `reset` ->
//! [`VulkanStreamDecoder::reset`]. Because the decode is real hardware Vulkan
//! Video, this is the "Tier A" readback path (GPU decode -> CPU I420 -> the
//! renderer uploads + does YUV->RGB on the GPU).
//!
//! ## Container ingestion (real MP4 / CMAF, not just elementary streams)
//!
//! [`VulkanStreamDecoder::new`] takes a raw Annex-B / OBU elementary stream with
//! the parameter sets in-band. Real container content (what a viewer demuxes from
//! an MP4) is different: the parameter sets live out of band in the sample-entry
//! box (`avcC` / `hvcC` / `av1C`)
//! and the samples are length-prefixed (AVCC), not Annex-B.
//! [`VulkanStreamDecoder::from_config`] handles that shape: it builds the session
//! from a [`CodecConfig`] and records the NAL length size, so `submit_chunk` /
//! `submit_chunk_texture` reframe each length-prefixed sample to Annex-B
//! automatically (AV1 OBU samples need no reframing). A decoder built with
//! [`VulkanStreamDecoder::new`] instead passes samples through untouched.
//!
//! ## Tier B (zero-copy): GPU-texture output
//!
//! [`VulkanStreamDecoder::new_gpu`] builds the same decoder in GPU-texture mode:
//! [`VulkanStreamDecoder::submit_chunk_texture`] returns the decoded frame as a
//! GPU-resident RGBA
//! [`wgpu::Texture`] (YUV->RGB already applied on the GPU by g2g's
//! `VkSamplerYcbcrConversion` compute pass), so the frame never leaves the
//! decode device. A consumer hands the texture
//! straight to its renderer, skipping both the CPU readback and the renderer's
//! upload + YUV->RGB.
//!
//! The one integration constraint is device identity: the texture is bound to the
//! decode device, which g2g creates itself (it must enable Vulkan video-decode
//! queue families the render device would not). So zero-copy requires
//! the renderer to run on the decode device, not the reverse: a consumer takes
//! [`VulkanStreamDecoder::gpu_context`] (instance / adapter / device / queue) and
//! builds its
//! renderer on it. On a single-GPU host that is the display GPU too; on a split
//! decode/display host (decode dGPU, present iGPU) a cross-device copy is
//! unavoidable and Tier A is the honest path.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::annexb::{length_prefixed_to_annexb, parse_avcc, parse_hvcc, prepend_param_sets};
use crate::vulkanvideo::{
    extract_av1_sequence_header, extract_h264_parameter_sets, extract_h265_parameter_sets,
    to_std_av1_seq_header, to_std_h265_params, Av1DecodeSession, Av1DpbDecoder, Av1SequenceHeader,
    H264DecodeSession, H264DpbDecoder, H265DecodeSession, H265DpbDecoder, Nv12Frame,
    VulkanVideoDevice, VulkanVideoError,
};

/// Which codec a [`VulkanStreamDecoder`] decodes (the codecs g2g's Vulkan Video
/// path supports).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    H265,
    Av1,
}

/// Out-of-band codec configuration from a container sample-entry box, the shape
/// an MP4 / CMAF demuxer provides: `avcC` for H.264,
/// `hvcC` for H.265, `av1C` for AV1. Build with [`VulkanStreamDecoder::from_config`]
/// when the samples are the container's stored form (length-prefixed NALs with the
/// parameter sets carried out of band) rather than a raw Annex-B / OBU elementary
/// stream. Each variant is the config record body (the box payload, no box header).
#[derive(Debug, Clone, Copy)]
pub enum CodecConfig<'a> {
    /// H.264 `avcC` (AVCDecoderConfigurationRecord).
    Avcc(&'a [u8]),
    /// H.265 `hvcC` (HEVCDecoderConfigurationRecord).
    Hvcc(&'a [u8]),
    /// AV1 `av1C` (AV1CodecConfigurationRecord); the config OBUs carry the
    /// sequence header.
    Av1c(&'a [u8]),
}

/// Chroma layout of [`DecodedVideoFrame::data`]. The Vulkan decoders emit 4:2:0,
/// so only [`Self::Y_U_V420`] is produced today; the other variants name the
/// layouts a consumer's renderer conventionally distinguishes.
#[expect(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoPixelLayout {
    Y_U_V444,
    Y_U_V422,
    Y_U_V420,
    Y400,
}

/// YUV value range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoColorRange {
    Limited,
    Full,
}

/// YUV->RGB matrix coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoMatrixCoefficients {
    /// YUV is actually GBR.
    Identity,
    Bt601,
    Bt709,
}

/// One decoded frame in the conventional CPU layout: packed I420 planar YUV.
#[derive(Debug, Clone)]
pub struct DecodedVideoFrame {
    /// Packed planar YUV: Y (`width*height`), then U, then V (each
    /// `ceil(width/2) * ceil(height/2)` for 4:2:0). No per-row stride padding.
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub layout: VideoPixelLayout,
    pub range: VideoColorRange,
    pub coefficients: VideoMatrixCoefficients,
}

/// Per-codec decode state (session + DPB decoder). The decoder is listed before
/// the session so it drops first (it holds copies of the session's handles); both
/// drop before the device (see [`VulkanStreamDecoder`]). The `_session` is kept
/// only to outlive the decoder, never read directly.
enum Inner {
    H264 {
        // boxed: the H.264 decoder is larger than the other variants
        decoder: Box<H264DpbDecoder>,
        _session: H264DecodeSession,
    },
    H265 {
        decoder: H265DpbDecoder,
        _session: H265DecodeSession,
    },
    Av1 {
        decoder: Av1DpbDecoder,
        _session: Av1DecodeSession,
    },
}

/// A chunk-fed hardware video decoder producing viewer-shaped CPU frames.
///
/// Feed one coded sample (access unit for H.26x, temporal unit for AV1) per
/// [`Self::submit_chunk`]; DPB state carries across calls, so P/B frames decode
/// against their references. Seek by calling [`Self::reset`] then feeding from a
/// keyframe. The codec parameters come from the `init` stream passed to
/// [`Self::new`] (SPS/PPS for H.26x, sequence header for AV1).
#[derive(Debug)]
pub struct VulkanStreamDecoder {
    inner: Inner,
    width: u32,
    height: u32,
    range: VideoColorRange,
    coefficients: VideoMatrixCoefficients,
    // Output mode chosen at construction: `false` = CPU I420 (`submit_chunk`),
    // `true` = GPU-resident RGBA texture (`submit_chunk_texture`). The two are
    // mutually exclusive; the DPB images are allocated differently for each.
    gpu_mode: bool,
    // NAL length-prefix size (1..=4) when the decoder was built from a container
    // config ([`Self::from_config`]): incoming samples are the MP4/CMAF stored
    // form (length-prefixed NALs, AVCC), reframed to Annex-B before decode. `None`
    // for a raw Annex-B / OBU elementary stream ([`Self::new`] / [`Self::new_gpu`]),
    // where samples pass through untouched. Ignored for AV1 (OBU, never reframed).
    nal_length_size: Option<usize>,
    // Drop last: the decoder + session (in `inner`) destroy Vulkan objects that
    // live on this device, so the VkDevice must outlive them. The device also
    // owns the wgpu device/queue a GPU-mode texture is bound to, so it must
    // outlive any texture handed out by `submit_chunk_texture`.
    _device: VulkanVideoDevice,
}

/// One decoded frame kept GPU-resident (Tier B): an RGBA [`wgpu::Texture`] on the
/// decode device, YUV->RGB already applied by g2g's compute pass. This is the
/// zero-copy analog of [`DecodedVideoFrame`]; a consumer maps it onto its own
/// GPU-texture frame type and hands it straight to the renderer.
#[derive(Debug)]
pub struct DecodedVideoTexture {
    /// RGBA8 texture on the decode device. Bind it with the same wgpu device the
    /// decoder exposes via [`VulkanStreamDecoder::gpu_context`].
    pub texture: wgpu::Texture,
    pub width: u32,
    pub height: u32,
}

impl core::fmt::Debug for Inner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::H264 { .. } => "H264",
            Self::H265 { .. } => "H265",
            Self::Av1 { .. } => "Av1",
        };
        f.debug_struct("Inner")
            .field("codec", &name)
            .finish_non_exhaustive()
    }
}

impl VulkanStreamDecoder {
    /// Build a decoder for `codec` on an already-opened `device`, taking codec
    /// parameters from `init` (a prefix of the elementary stream that carries the
    /// SPS/PPS for H.26x or the sequence header OBU for AV1; the whole stream is
    /// fine too). The `device` must have been opened for the matching codec
    /// (`open_h264_decode_device` / `open_h265_decode_device` /
    /// `open_av1_decode_device`).
    pub fn new(
        device: VulkanVideoDevice,
        codec: VideoCodec,
        init: &[u8],
    ) -> Result<Self, VulkanVideoError> {
        match codec {
            VideoCodec::H264 => {
                let ps =
                    extract_h264_parameter_sets(init).ok_or(VulkanVideoError::UnsupportedStream)?;
                let (coded_width, coded_height) = ps.sps.coded_extent();
                let (range, coefficients) =
                    h26x_colorimetry(ps.sps.matrix_coefficients, ps.sps.video_full_range_flag);
                let session = device.create_h264_session(&ps, coded_width, coded_height)?;
                let (width, height) = session.output_extent;
                let decoder = device.create_h264_dpb_decoder(&session, &ps)?;
                Ok(Self {
                    inner: Inner::H264 {
                        decoder: Box::new(decoder),
                        _session: session,
                    },
                    width,
                    height,
                    range,
                    coefficients,
                    gpu_mode: false,
                    nal_length_size: None,
                    _device: device,
                })
            }
            VideoCodec::H265 => {
                let ps =
                    extract_h265_parameter_sets(init).ok_or(VulkanVideoError::UnsupportedStream)?;
                let std = to_std_h265_params(&ps);
                let (range, coefficients) =
                    h26x_colorimetry(ps.sps.matrix_coefficients, ps.sps.video_full_range_flag);
                let session = device.create_h265_session(
                    &std,
                    ps.sps.pic_width_in_luma_samples,
                    ps.sps.pic_height_in_luma_samples,
                )?;
                let (width, height) = session.output_extent;
                let decoder = device.create_h265_dpb_decoder(&session, &ps)?;
                Ok(Self {
                    inner: Inner::H265 {
                        decoder,
                        _session: session,
                    },
                    width,
                    height,
                    range,
                    coefficients,
                    gpu_mode: false,
                    nal_length_size: None,
                    _device: device,
                })
            }
            VideoCodec::Av1 => {
                let seq =
                    extract_av1_sequence_header(init).ok_or(VulkanVideoError::UnsupportedStream)?;
                let width = seq.max_frame_width_minus_1 + 1;
                let height = seq.max_frame_height_minus_1 + 1;
                let (range, coefficients) = av1_colorimetry(&seq);
                let std = to_std_av1_seq_header(&seq);
                let session = device.create_av1_session(&std, width, height)?;
                let decoder = device.create_av1_dpb_decoder(&session, &seq)?;
                Ok(Self {
                    inner: Inner::Av1 {
                        decoder,
                        _session: session,
                    },
                    width,
                    height,
                    range,
                    coefficients,
                    gpu_mode: false,
                    nal_length_size: None,
                    _device: device,
                })
            }
        }
    }

    /// Build a decoder in GPU-texture (Tier B / zero-copy) mode: each decoded
    /// frame is produced as a GPU-resident RGBA [`wgpu::Texture`] via g2g's ycbcr
    /// compute pass, retrieved with [`Self::submit_chunk_texture`]. Same codec /
    /// `init` contract as [`Self::new`]. Requires the decode device to have a
    /// distinct compute queue; returns [`VulkanVideoError::NoComputeQueue`]
    /// otherwise (fall back to [`Self::new`] + [`Self::submit_chunk`]).
    pub fn new_gpu(
        device: VulkanVideoDevice,
        codec: VideoCodec,
        init: &[u8],
    ) -> Result<Self, VulkanVideoError> {
        match codec {
            VideoCodec::H264 => {
                let ps =
                    extract_h264_parameter_sets(init).ok_or(VulkanVideoError::UnsupportedStream)?;
                let (coded_width, coded_height) = ps.sps.coded_extent();
                let (range, coefficients) =
                    h26x_colorimetry(ps.sps.matrix_coefficients, ps.sps.video_full_range_flag);
                let session = device.create_h264_session(&ps, coded_width, coded_height)?;
                let (width, height) = session.output_extent;
                let decoder = device.create_h264_dpb_decoder_gpu(&session, &ps)?;
                Ok(Self {
                    inner: Inner::H264 {
                        decoder: Box::new(decoder),
                        _session: session,
                    },
                    width,
                    height,
                    range,
                    coefficients,
                    gpu_mode: true,
                    nal_length_size: None,
                    _device: device,
                })
            }
            VideoCodec::H265 => {
                let ps =
                    extract_h265_parameter_sets(init).ok_or(VulkanVideoError::UnsupportedStream)?;
                let std = to_std_h265_params(&ps);
                let (range, coefficients) =
                    h26x_colorimetry(ps.sps.matrix_coefficients, ps.sps.video_full_range_flag);
                let session = device.create_h265_session(
                    &std,
                    ps.sps.pic_width_in_luma_samples,
                    ps.sps.pic_height_in_luma_samples,
                )?;
                let (width, height) = session.output_extent;
                let decoder = device.create_h265_dpb_decoder_gpu(&session, &ps)?;
                Ok(Self {
                    inner: Inner::H265 {
                        decoder,
                        _session: session,
                    },
                    width,
                    height,
                    range,
                    coefficients,
                    gpu_mode: true,
                    nal_length_size: None,
                    _device: device,
                })
            }
            VideoCodec::Av1 => {
                let seq =
                    extract_av1_sequence_header(init).ok_or(VulkanVideoError::UnsupportedStream)?;
                let width = seq.max_frame_width_minus_1 + 1;
                let height = seq.max_frame_height_minus_1 + 1;
                let (range, coefficients) = av1_colorimetry(&seq);
                let std = to_std_av1_seq_header(&seq);
                let session = device.create_av1_session(&std, width, height)?;
                let decoder = device.create_av1_dpb_decoder_gpu(&session, &seq)?;
                Ok(Self {
                    inner: Inner::Av1 {
                        decoder,
                        _session: session,
                    },
                    width,
                    height,
                    range,
                    coefficients,
                    gpu_mode: true,
                    nal_length_size: None,
                    _device: device,
                })
            }
        }
    }

    /// Build a decoder from an out-of-band container config ([`CodecConfig`], the
    /// shape an MP4 / CMAF demuxer carries). The
    /// parameter sets come from the config, not scavenged from the first sample, so
    /// this works on real container streams whose samples carry no in-band SPS/PPS.
    /// For H.26x it also records the NAL length-prefix size, so subsequent
    /// length-prefixed (AVCC) samples are reframed to Annex-B automatically by
    /// [`Self::submit_chunk`] / [`Self::submit_chunk_texture`]. `gpu` selects
    /// GPU-texture output (as [`Self::new_gpu`]) over CPU I420 (as [`Self::new`]).
    /// Returns [`VulkanVideoError::UnsupportedStream`] if the config record is
    /// malformed or carries no parameter set.
    pub fn from_config(
        device: VulkanVideoDevice,
        config: CodecConfig<'_>,
        gpu: bool,
    ) -> Result<Self, VulkanVideoError> {
        let (codec, init, nal_length_size) = match config {
            CodecConfig::Avcc(b) => {
                let record = parse_avcc(b).map_err(|_| VulkanVideoError::UnsupportedStream)?;
                let params =
                    prepend_param_sets(&[], &record.parameter_sets, g2g_core::VideoCodec::H264);
                (VideoCodec::H264, params, Some(record.nal_length_size))
            }
            CodecConfig::Hvcc(b) => {
                let record = parse_hvcc(b).map_err(|_| VulkanVideoError::UnsupportedStream)?;
                let params =
                    prepend_param_sets(&[], &record.parameter_sets, g2g_core::VideoCodec::H265);
                (VideoCodec::H265, params, Some(record.nal_length_size))
            }
            CodecConfig::Av1c(b) => {
                let params = parse_av1c_config(b).ok_or(VulkanVideoError::UnsupportedStream)?;
                (VideoCodec::Av1, params, None)
            }
        };
        let mut dec = if gpu {
            Self::new_gpu(device, codec, &init)?
        } else {
            Self::new(device, codec, &init)?
        };
        dec.nal_length_size = nal_length_size;
        Ok(dec)
    }

    /// Width of the decoded frames in luma samples, after the stream's crop.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height of the decoded frames in luma samples, after the stream's crop.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Decode one coded sample and return every frame it produced (usually one)
    /// as packed I420. `is_sync` (keyframe) is accepted for API parity with
    /// a demuxed sample; the DPB decoders derive reference management from the
    /// bitstream itself, so it is currently advisory.
    pub fn submit_chunk(
        &mut self,
        data: &[u8],
        _is_sync: bool,
    ) -> Result<Vec<DecodedVideoFrame>, VulkanVideoError> {
        if self.gpu_mode {
            return Err(VulkanVideoError::WrongOutputMode);
        }
        let reframed = self.reframe(data)?;
        let input = reframed.as_deref().unwrap_or(data);
        let nv12 = match &mut self.inner {
            Inner::H264 { decoder, .. } => decoder.decode_all(input)?,
            Inner::H265 { decoder, .. } => decoder.decode_all(input)?,
            Inner::Av1 { decoder, .. } => decoder.decode_all(input)?,
        };
        Ok(self.to_i420_frames(nv12))
    }

    /// Streaming (pipelined) variant of [`Self::submit_chunk`]: submit the sample
    /// into the decode ring WITHOUT draining, returning only the frames that have
    /// already retired. Output lags submission by up to `DECODE_RING_DEPTH - 1`
    /// frames; this is the low-latency streaming form a chunk-at-a-time consumer
    /// (an async chunk decoder) wants, one sample in per call, frames out as
    /// they retire, rather than a full drain per sample. Pair with [`Self::flush`],
    /// which must be called at end of stream to emit the pipelined tail.
    pub fn submit_chunk_push(
        &mut self,
        data: &[u8],
        _is_sync: bool,
    ) -> Result<Vec<DecodedVideoFrame>, VulkanVideoError> {
        if self.gpu_mode {
            return Err(VulkanVideoError::WrongOutputMode);
        }
        let reframed = self.reframe(data)?;
        let input = reframed.as_deref().unwrap_or(data);
        // Only the retired frames are needed here; the submitted-picture count is
        // for callers that pair a per-picture side channel (see the element).
        let nv12 = match &mut self.inner {
            Inner::H264 { decoder, .. } => decoder.decode_push(input)?.1,
            Inner::H265 { decoder, .. } => decoder.decode_push(input)?.1,
            Inner::Av1 { decoder, .. } => decoder.decode_push(input)?.1,
        };
        Ok(self.to_i420_frames(nv12))
    }

    /// Drain the decode ring at end of stream, returning the pipelined tail frames
    /// held back by [`Self::submit_chunk_push`]. After this the ring is empty.
    pub fn flush(&mut self) -> Result<Vec<DecodedVideoFrame>, VulkanVideoError> {
        if self.gpu_mode {
            return Err(VulkanVideoError::WrongOutputMode);
        }
        let nv12 = match &mut self.inner {
            Inner::H264 { decoder, .. } => decoder.decode_flush()?,
            Inner::H265 { decoder, .. } => decoder.decode_flush()?,
            Inner::Av1 { decoder, .. } => decoder.decode_flush()?,
        };
        Ok(self.to_i420_frames(nv12))
    }

    /// Map decoder NV12 output to packed-I420 [`DecodedVideoFrame`]s with this
    /// decoder's colorimetry. Shared by [`Self::submit_chunk`] /
    /// [`Self::submit_chunk_push`] / [`Self::flush`].
    fn to_i420_frames(&self, nv12: Vec<Nv12Frame>) -> Vec<DecodedVideoFrame> {
        let range = self.range;
        let coefficients = self.coefficients;
        nv12.into_iter()
            .map(|f| DecodedVideoFrame {
                width: f.width,
                height: f.height,
                data: nv12_to_i420(f),
                layout: VideoPixelLayout::Y_U_V420,
                range,
                coefficients,
            })
            .collect()
    }

    /// Decode one coded sample (Tier B / zero-copy) and return every frame it
    /// produced as a GPU-resident RGBA [`DecodedVideoTexture`], no CPU readback.
    /// The decoder must have been built with [`Self::new_gpu`]; returns
    /// [`VulkanVideoError::WrongOutputMode`] on a CPU-mode decoder. `is_sync` is
    /// advisory (see [`Self::submit_chunk`]).
    pub fn submit_chunk_texture(
        &mut self,
        data: &[u8],
        _is_sync: bool,
    ) -> Result<Vec<DecodedVideoTexture>, VulkanVideoError> {
        if !self.gpu_mode {
            return Err(VulkanVideoError::WrongOutputMode);
        }
        let reframed = self.reframe(data)?;
        let input = reframed.as_deref().unwrap_or(data);
        let textures = match &mut self.inner {
            Inner::H264 { decoder, .. } => decoder.decode_all_to_textures(input)?,
            Inner::H265 { decoder, .. } => decoder.decode_all_to_textures(input)?,
            Inner::Av1 { decoder, .. } => decoder.decode_all_to_textures(input)?,
        };
        Ok(textures
            .into_iter()
            .map(|texture| DecodedVideoTexture {
                width: texture.width(),
                height: texture.height(),
                texture,
            })
            .collect())
    }

    /// The decode device's wgpu context (instance / adapter / device / queue). A
    /// GPU-mode texture from [`Self::submit_chunk_texture`] is bound to this
    /// device, so a zero-copy consumer (a viewer renderer, a `WgpuSink`) must build
    /// on this context. See the Tier B note in the module docs: g2g owns the
    /// decode device, so the renderer runs on it, not the other way round.
    pub fn gpu_context(&self) -> crate::gpu::GpuContext {
        self._device.gpu_context()
    }

    /// Read a GPU-mode texture back to a packed RGBA8 `Vec` on the CPU. This is a
    /// verification / debug helper, NOT part of the zero-copy path (it defeats the
    /// point); a real consumer keeps the texture on the GPU. Returns
    /// `width * height * 4` bytes.
    pub fn read_rgba_texture(&self, texture: &wgpu::Texture) -> Vec<u8> {
        self._device.read_rgba_texture(texture)
    }

    /// If this decoder was built from a container config with a NAL length-prefix
    /// size ([`Self::from_config`]), reframe a length-prefixed (AVCC) `sample` to
    /// Annex-B; return `None` (decode `sample` as-is) for a raw Annex-B / OBU
    /// stream ([`Self::new`] / [`Self::new_gpu`], `nal_length_size == None`).
    ///
    /// When a length size is set the sample is ALWAYS reframed: it is the
    /// container's stored form, which is length-prefixed by definition. Do not try
    /// to sniff Annex-B here, a 4-byte NAL length in 256..=511 is `00 00 01 xx`,
    /// which looks exactly like an Annex-B start code and would be mis-detected.
    fn reframe(&self, sample: &[u8]) -> Result<Option<Vec<u8>>, VulkanVideoError> {
        let Some(n) = self.nal_length_size else {
            return Ok(None);
        };
        length_prefixed_to_annexb(sample, n)
            .map(Some)
            .ok_or(VulkanVideoError::UnsupportedStream)
    }

    /// Reset the decoder for a backward seek or large forward jump: the DPB
    /// reference state is cleared and the session `RESET` control re-armed, so the
    /// next chunk must be a keyframe. Cheap (no reallocation); reuses the session
    /// and device. This is what a fork maps `AsyncDecoder::reset` onto.
    ///
    /// Note: rebuilding the decoder over the same session is NOT a clean reset on
    /// the NVIDIA driver (stale session state corrupts even an intra keyframe);
    /// re-issuing the session `RESET` control is the correct reset.
    pub fn reset(&mut self) -> Result<(), VulkanVideoError> {
        match &mut self.inner {
            Inner::H264 { decoder, .. } => decoder.reset(),
            Inner::H265 { decoder, .. } => decoder.reset(),
            Inner::Av1 { decoder, .. } => decoder.reset(),
        }
        Ok(())
    }
}

/// Deinterleave a decoder NV12 frame (Y plane + interleaved Cb/Cr) into packed
/// I420 (Y, then U, then V), the conventional planar layout, in the frame's own
/// buffer.
fn nv12_to_i420(nv12: Nv12Frame) -> Vec<u8> {
    let interleaved = nv12.chroma().to_vec();
    let luma_len = nv12.luma().len();
    let mut i420 = nv12.into_data().into_vec();
    // NV12 chroma is Cb,Cr interleaved (U then V per pair) -> split into planes.
    let (u, v) = i420[luma_len..].split_at_mut(interleaved.len() / 2);
    for ((pair, u), v) in interleaved.as_chunks::<2>().0.iter().zip(u).zip(v) {
        *u = pair[0];
        *v = pair[1];
    }
    i420
}

/// Parse an `av1C` (AV1CodecConfigurationRecord) body into its config OBUs (the
/// sequence header). Layout: [0] `marker` (1) + `version` (7, must be 1), [1..4]
/// seq profile / level / tier / bit-depth / chroma flags, then the config OBUs to
/// the end. AV1 samples are already OBU framed, so only the sequence header is
/// needed for init and no per-sample reframing happens. `None` on a bad marker /
/// version or empty config OBUs.
fn parse_av1c_config(av1c: &[u8]) -> Option<Vec<u8>> {
    let b0 = *av1c.first()?;
    if (b0 >> 7) != 1 || (b0 & 0x7F) != 1 {
        return None;
    }
    let obus = av1c.get(4..)?;
    if obus.is_empty() {
        return None;
    }
    Some(obus.to_vec())
}

/// Map an AV1 sequence header's color config to (range, coefficients). Matrix
/// coefficients follow the AV1 / ISO 23091-2 codepoints; unspecified (2) is
/// guessed as BT.709, matching what players (mpv/VLC) do.
/// Colorimetry for an H.264 / H.265 stream from its VUI colour description
/// (parsed into the SPS as CICP codepoints). A concrete matrix is required
/// here (the converter has to pick one), so an unknown or unmodeled codepoint
/// falls back to BT.601 limited, the value this adapter always used before the
/// VUI was consulted.
fn h26x_colorimetry(
    matrix_coefficients: u8,
    video_full_range_flag: bool,
) -> (VideoColorRange, VideoMatrixCoefficients) {
    let range = if video_full_range_flag {
        VideoColorRange::Full
    } else {
        VideoColorRange::Limited
    };
    let coefficients = match g2g_core::MatrixCoefficients::from_cicp(matrix_coefficients) {
        g2g_core::MatrixCoefficients::Identity => VideoMatrixCoefficients::Identity,
        g2g_core::MatrixCoefficients::Bt709 => VideoMatrixCoefficients::Bt709,
        g2g_core::MatrixCoefficients::Bt601 => VideoMatrixCoefficients::Bt601,
        _ => VideoMatrixCoefficients::Bt601,
    };
    (range, coefficients)
}

fn av1_colorimetry(seq: &Av1SequenceHeader) -> (VideoColorRange, VideoMatrixCoefficients) {
    let range = if seq.color.color_range {
        VideoColorRange::Full
    } else {
        VideoColorRange::Limited
    };
    let coefficients = match seq.color.matrix_coefficients {
        0 => VideoMatrixCoefficients::Identity,
        1 => VideoMatrixCoefficients::Bt709,
        // BT.470M (4), BT.470BG/PAL (5), SMPTE 170M/NTSC (6) -> BT.601.
        4..=6 => VideoMatrixCoefficients::Bt601,
        // Unspecified (2), reserved, and HD/HDR standards -> best-guess BT.709.
        _ => VideoMatrixCoefficients::Bt709,
    };
    (range, coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn parse_av1c_extracts_config_obus() {
        let obus: &[u8] = &[0x0a, 0x0b, 0x0c];
        let mut av1c = vec![0x81u8, 0x00, 0x00, 0x00]; // marker=1, version=1, 3 bytes
        av1c.extend_from_slice(obus);
        assert_eq!(parse_av1c_config(&av1c).as_deref(), Some(obus));

        // Bad marker or version -> None.
        let mut bad = av1c.clone();
        bad[0] = 0x01; // marker = 0
        assert!(parse_av1c_config(&bad).is_none());
        bad[0] = 0x82; // version = 2
        assert!(parse_av1c_config(&bad).is_none());
    }
}
