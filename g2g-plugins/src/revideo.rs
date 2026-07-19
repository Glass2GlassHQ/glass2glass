//! Fork-ready adapter: presents g2g's Vulkan Video hardware decoders in the
//! chunk-at-a-time, CPU-frame shape that Rerun's `re_video::decode::AsyncDecoder`
//! backend consumes (the wgpu-texture wedge, see the wedge-strategy note).
//!
//! g2g does NOT depend on `re_video` (its dependency graph is the whole Rerun
//! workspace: `cros-codecs`, `dav1d`, ~8 sibling `re_*` crates). Instead this
//! module mirrors re_video's native decode-output layout exactly:
//!
//! * `DecodedVideoFrame::data` is I420 planar (Y, then U, then V, packed with no
//!   stride padding), the same layout re_video's dav1d/ffmpeg backends emit for
//!   `PixelFormat::Yuv { layout: Y_U_V420, .. }`.
//! * [`VideoPixelLayout`] / [`VideoColorRange`] / [`VideoMatrixCoefficients`]
//!   are 1:1 with re_video's `YuvPixelLayout` / `YuvRange` / `YuvMatrixCoefficients`.
//!
//! So a small re_video fork wraps [`VulkanStreamDecoder`] in ONE
//! `impl AsyncDecoder`: `submit_chunk` -> [`VulkanStreamDecoder::submit_chunk`],
//! mapping each [`DecodedVideoFrame`] onto `re_video::decode::Frame` and sending
//! it on the output channel; `reset` -> [`VulkanStreamDecoder::reset`]. Because
//! the decode is real hardware Vulkan Video, this is the "Tier A" readback path
//! (GPU decode -> CPU I420 -> re_renderer uploads + does YUV->RGB on the GPU).
//!
//! ## Container ingestion (real MP4 / CMAF, not just elementary streams)
//!
//! [`VulkanStreamDecoder::new`] takes a raw Annex-B / OBU elementary stream with
//! the parameter sets in-band. Real container content (what re_video demuxes from
//! an MP4) is different: the parameter sets live out of band in the sample-entry
//! box (`avcC` / `hvcC` / `av1C`, carried by re_video's `VideoDataDescription`)
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
//! [`Self::submit_chunk_texture`] returns the decoded frame as a GPU-resident RGBA
//! [`wgpu::Texture`] (YUV->RGB already applied on the GPU by g2g's
//! `VkSamplerYcbcrConversion` compute pass, M494), so the frame never leaves the
//! decode device. A re_video fork wraps this in a GPU-texture `FrameContent`
//! variant and hands the texture straight to `re_renderer`, skipping both the
//! CPU readback and re_renderer's upload + YUV->RGB.
//!
//! The one integration constraint is device identity: the texture is bound to the
//! decode device, which g2g creates itself (it must enable Vulkan video-decode
//! queue families the render device would not). So zero-copy requires
//! `re_renderer` to run on the decode device, not the reverse: a consumer takes
//! [`Self::gpu_context`] (instance / adapter / device / queue) and builds its
//! renderer on it. On a single-GPU host that is the display GPU too; on a split
//! decode/display host (decode dGPU, present iGPU) a cross-device copy is
//! unavoidable and Tier A is the honest path.

use alloc::vec::Vec;

use crate::vulkanvideo::{
    extract_av1_sequence_header, extract_h264_parameter_sets, extract_h265_parameter_sets,
    to_std_av1_seq_header, to_std_h265_params, Av1DecodeSession, Av1DpbDecoder, Av1SequenceHeader,
    H264DecodeSession, H264DpbDecoder, H265DecodeSession, H265DpbDecoder, Nv12Frame,
    VulkanVideoDevice, VulkanVideoError,
};

/// Which codec a [`VulkanStreamDecoder`] decodes. Mirrors the subset of
/// `re_video::VideoCodec` g2g's Vulkan Video path supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    H265,
    Av1,
}

/// Out-of-band codec configuration from a container sample-entry box, the shape
/// re_video's `VideoDataDescription` provides for MP4 / CMAF: `avcC` for H.264,
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
/// so only [`Self::Y_U_V420`] is produced today; the enum matches re_video's
/// `YuvPixelLayout` so a fork maps it with no lookup.
#[expect(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoPixelLayout {
    Y_U_V444,
    Y_U_V422,
    Y_U_V420,
    Y400,
}

/// YUV value range. 1:1 with re_video's `YuvRange`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoColorRange {
    Limited,
    Full,
}

/// YUV->RGB matrix coefficients. 1:1 with re_video's `YuvMatrixCoefficients`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoMatrixCoefficients {
    /// YUV is actually GBR.
    Identity,
    Bt601,
    Bt709,
}

/// One decoded frame in re_video's native CPU layout: packed I420 planar YUV.
#[derive(Debug, Clone)]
pub struct DecodedVideoFrame {
    /// Packed planar YUV: Y (`width*height`), then U, then V (each
    /// `width/2 * height/2` for 4:2:0). No per-row stride padding.
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
        decoder: H264DpbDecoder,
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

/// A chunk-fed hardware video decoder producing re_video-shaped CPU frames.
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
    nal_length_size: Option<u8>,
    // Drop last: the decoder + session (in `inner`) destroy Vulkan objects that
    // live on this device, so the VkDevice must outlive them. The device also
    // owns the wgpu device/queue a GPU-mode texture is bound to, so it must
    // outlive any texture handed out by `submit_chunk_texture`.
    _device: VulkanVideoDevice,
}

/// One decoded frame kept GPU-resident (Tier B): an RGBA [`wgpu::Texture`] on the
/// decode device, YUV->RGB already applied by g2g's compute pass. This is the
/// zero-copy analog of [`DecodedVideoFrame`]; a re_video fork maps it onto a
/// GPU-texture `FrameContent` and hands it straight to `re_renderer`.
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
                let width = (ps.sps.pic_width_in_mbs_minus1 + 1) * 16;
                let height = (ps.sps.pic_height_in_map_units_minus1 + 1) * 16;
                let session = device.create_h264_session(&ps, width, height)?;
                let decoder = device.create_h264_dpb_decoder(&session, &ps)?;
                Ok(Self {
                    inner: Inner::H264 {
                        decoder,
                        _session: session,
                    },
                    width,
                    height,
                    // H.264/H.265 VUI colorimetry is not parsed yet; default to
                    // BT.601 limited, which matches the decoder's NV12 output and
                    // is the safe SD default. A fork can override from container
                    // metadata (`re_video::VideoDataDescription`).
                    range: VideoColorRange::Limited,
                    coefficients: VideoMatrixCoefficients::Bt601,
                    gpu_mode: false,
                    nal_length_size: None,
                    _device: device,
                })
            }
            VideoCodec::H265 => {
                let ps =
                    extract_h265_parameter_sets(init).ok_or(VulkanVideoError::UnsupportedStream)?;
                let width = ps.sps.pic_width_in_luma_samples;
                let height = ps.sps.pic_height_in_luma_samples;
                let std = to_std_h265_params(&ps);
                let session = device.create_h265_session(&std, width, height)?;
                let decoder = device.create_h265_dpb_decoder(&session, &ps)?;
                Ok(Self {
                    inner: Inner::H265 {
                        decoder,
                        _session: session,
                    },
                    width,
                    height,
                    range: VideoColorRange::Limited,
                    coefficients: VideoMatrixCoefficients::Bt601,
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
                let width = (ps.sps.pic_width_in_mbs_minus1 + 1) * 16;
                let height = (ps.sps.pic_height_in_map_units_minus1 + 1) * 16;
                let session = device.create_h264_session(&ps, width, height)?;
                let decoder = device.create_h264_dpb_decoder_gpu(&session, &ps)?;
                Ok(Self {
                    inner: Inner::H264 {
                        decoder,
                        _session: session,
                    },
                    width,
                    height,
                    range: VideoColorRange::Limited,
                    coefficients: VideoMatrixCoefficients::Bt601,
                    gpu_mode: true,
                    nal_length_size: None,
                    _device: device,
                })
            }
            VideoCodec::H265 => {
                let ps =
                    extract_h265_parameter_sets(init).ok_or(VulkanVideoError::UnsupportedStream)?;
                let width = ps.sps.pic_width_in_luma_samples;
                let height = ps.sps.pic_height_in_luma_samples;
                let std = to_std_h265_params(&ps);
                let session = device.create_h265_session(&std, width, height)?;
                let decoder = device.create_h265_dpb_decoder_gpu(&session, &ps)?;
                Ok(Self {
                    inner: Inner::H265 {
                        decoder,
                        _session: session,
                    },
                    width,
                    height,
                    range: VideoColorRange::Limited,
                    coefficients: VideoMatrixCoefficients::Bt601,
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
    /// shape re_video's `VideoDataDescription` carries for MP4 / CMAF). The
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
                let (params, n) =
                    parse_avcc_config(b).ok_or(VulkanVideoError::UnsupportedStream)?;
                (VideoCodec::H264, params, Some(n))
            }
            CodecConfig::Hvcc(b) => {
                let (params, n) =
                    parse_hvcc_config(b).ok_or(VulkanVideoError::UnsupportedStream)?;
                (VideoCodec::H265, params, Some(n))
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

    /// Coded width in luma samples.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Coded height in luma samples.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Decode one coded sample and return every frame it produced (usually one)
    /// as packed I420. `is_sync` (keyframe) is accepted for API parity with
    /// re_video's `Chunk`; the DPB decoders derive reference management from the
    /// bitstream itself, so it is currently advisory.
    pub fn submit_chunk(
        &mut self,
        data: &[u8],
        _is_sync: bool,
    ) -> Result<Vec<DecodedVideoFrame>, VulkanVideoError> {
        if self.gpu_mode {
            return Err(VulkanVideoError::WrongOutputMode);
        }
        let reframed = self.reframe(data);
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
    /// (re_video's `AsyncDecoder`) wants, one sample in per call, frames out as
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
        let reframed = self.reframe(data);
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
                data: nv12_to_i420(&f),
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
        let reframed = self.reframe(data);
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
    /// device, so a zero-copy consumer (a `re_renderer`, a `WgpuSink`) must build
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
    fn reframe(&self, sample: &[u8]) -> Option<Vec<u8>> {
        let n = self.nal_length_size?;
        Some(reframe_length_prefixed(sample, n))
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
/// I420 (Y, then U, then V), the layout re_video expects.
fn nv12_to_i420(nv12: &Nv12Frame) -> Vec<u8> {
    let w = nv12.width as usize;
    let h = nv12.height as usize;
    let cw = w / 2;
    let ch = h / 2;
    let mut out = Vec::with_capacity(w * h + 2 * cw * ch);
    out.extend_from_slice(&nv12.luma);
    // NV12 chroma is Cb,Cr interleaved (U then V per pair) -> split into planes.
    let n = cw * ch;
    let mut u = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);
    for px in nv12.chroma.chunks_exact(2) {
        u.push(px[0]);
        v.push(px[1]);
    }
    out.append(&mut u);
    out.append(&mut v);
    out
}

/// Append `nal` to `out` in Annex-B framing (4-byte start code + NAL).
fn push_annexb(out: &mut Vec<u8>, nal: &[u8]) {
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(nal);
}

/// Reframe a length-prefixed (AVCC) sample to Annex-B: each NAL is preceded by an
/// `nal_length_size`-byte big-endian length; rewrite those as start codes. Bounds
/// checked, so a truncated final length/NAL stops the walk rather than panicking
/// (container samples are attacker-controlled).
fn reframe_length_prefixed(sample: &[u8], nal_length_size: u8) -> Vec<u8> {
    let n = nal_length_size as usize;
    let mut out = Vec::with_capacity(sample.len() + 16);
    let mut at = 0usize;
    while at + n <= sample.len() {
        let mut len = 0usize;
        for &b in &sample[at..at + n] {
            len = (len << 8) | b as usize;
        }
        at += n;
        let end = match at.checked_add(len) {
            Some(e) if e <= sample.len() => e,
            _ => break,
        };
        push_annexb(&mut out, &sample[at..end]);
        at = end;
    }
    out
}

/// Parse an `avcC` (AVCDecoderConfigurationRecord) body into (Annex-B SPS+PPS
/// blob, NAL length size). Layout: [0] version, [1..4] profile/compat/level, [4]
/// `111111` + `lengthSizeMinusOne` (2 bits), [5] `111` + `numSPS` (5 bits), then
/// each SPS as u16 length + NAL, then [.] numPPS, then each PPS as u16 length +
/// NAL. Every offset is bounds checked; returns `None` on malformed input or no
/// parameter set (so the caller falls back rather than mis-decoding).
fn parse_avcc_config(avcc: &[u8]) -> Option<(Vec<u8>, u8)> {
    let nal_length_size = (avcc.get(4)? & 0x03) + 1;
    let num_sps = avcc.get(5)? & 0x1F;
    let mut at = 6usize;
    let mut out = Vec::new();
    at = read_param_nals(avcc, at, num_sps as usize, &mut out)?;
    let num_pps = *avcc.get(at)?;
    at += 1;
    read_param_nals(avcc, at, num_pps as usize, &mut out)?;
    if out.is_empty() {
        return None;
    }
    Some((out, nal_length_size))
}

/// Read `count` u16-length-prefixed NALs starting at `at`, appending each in
/// Annex-B framing to `out`. Returns the offset past the last NAL, or `None` on a
/// truncated length/NAL. Shared by the `avcC` SPS and PPS arrays.
fn read_param_nals(buf: &[u8], mut at: usize, count: usize, out: &mut Vec<u8>) -> Option<usize> {
    for _ in 0..count {
        let len = u16::from_be_bytes(buf.get(at..at + 2)?.try_into().ok()?) as usize;
        at += 2;
        let nal = buf.get(at..at.checked_add(len)?)?;
        push_annexb(out, nal);
        at += len;
    }
    Some(at)
}

/// Parse an `hvcC` (HEVCDecoderConfigurationRecord) body into (Annex-B VPS+SPS+PPS
/// blob, NAL length size). Fixed 22-byte prefix (version + 12-byte general PTL +
/// descriptive fields; [21] low 2 bits = `lengthSizeMinusOne`), then [22]
/// `numOfArrays`, then per-array: a type byte, a u16 NAL count, and that many
/// u16-length-prefixed NALs. Bounds checked; `None` on malformed input or no
/// parameter set.
fn parse_hvcc_config(hvcc: &[u8]) -> Option<(Vec<u8>, u8)> {
    let nal_length_size = (hvcc.get(21)? & 0x03) + 1;
    let num_arrays = *hvcc.get(22)?;
    let mut at = 23usize;
    let mut out = Vec::new();
    for _ in 0..num_arrays {
        // array header byte: array_completeness | reserved | NAL_unit_type.
        at += 1;
        let num_nalus = u16::from_be_bytes(hvcc.get(at..at + 2)?.try_into().ok()?) as usize;
        at += 2;
        at = read_param_nals(hvcc, at, num_nalus, &mut out)?;
    }
    if out.is_empty() {
        return None;
    }
    Some((out, nal_length_size))
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
/// guessed as BT.709, matching what re_video (and mpv/VLC) do.
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

    /// Build an `avcC` body from raw SPS/PPS NALs with the given length size.
    fn build_avcc(sps: &[&[u8]], pps: &[&[u8]], nal_length_size: u8) -> Vec<u8> {
        let mut v = vec![1u8, 0x42, 0x00, 0x1e]; // version + profile/compat/level
        v.push(0xFC | (nal_length_size - 1)); // 111111 + lengthSizeMinusOne
        v.push(0xE0 | (sps.len() as u8 & 0x1F)); // 111 + numSPS
        for s in sps {
            v.extend_from_slice(&(s.len() as u16).to_be_bytes());
            v.extend_from_slice(s);
        }
        v.push(pps.len() as u8);
        for p in pps {
            v.extend_from_slice(&(p.len() as u16).to_be_bytes());
            v.extend_from_slice(p);
        }
        v
    }

    #[test]
    fn reframe_avcc_sample_to_annexb() {
        // Two length-prefixed NALs (4-byte size) -> two start-code NALs.
        let mut sample = Vec::new();
        sample.extend_from_slice(&3u32.to_be_bytes());
        sample.extend_from_slice(&[0x65, 0x01, 0x02]);
        sample.extend_from_slice(&2u32.to_be_bytes());
        sample.extend_from_slice(&[0x41, 0x03]);
        let out = reframe_length_prefixed(&sample, 4);
        assert_eq!(
            out,
            vec![0, 0, 0, 1, 0x65, 0x01, 0x02, 0, 0, 0, 1, 0x41, 0x03]
        );

        // 2-byte length size is honoured.
        let mut s2 = Vec::new();
        s2.extend_from_slice(&2u16.to_be_bytes());
        s2.extend_from_slice(&[0x65, 0x09]);
        assert_eq!(
            reframe_length_prefixed(&s2, 2),
            vec![0, 0, 0, 1, 0x65, 0x09]
        );
    }

    #[test]
    fn reframe_stops_on_truncated_length() {
        // A declared length longer than the buffer stops the walk (no panic), so a
        // first valid NAL is still recovered before the truncated tail.
        let mut sample = Vec::new();
        sample.extend_from_slice(&2u32.to_be_bytes());
        sample.extend_from_slice(&[0x65, 0x01]);
        sample.extend_from_slice(&99u32.to_be_bytes()); // bogus length
        sample.extend_from_slice(&[0x00]);
        let out = reframe_length_prefixed(&sample, 4);
        assert_eq!(out, vec![0, 0, 0, 1, 0x65, 0x01]);
    }

    #[test]
    fn parse_avcc_extracts_params_and_length_size() {
        let sps: &[u8] = &[0x67, 0x64, 0x00];
        let pps: &[u8] = &[0x68, 0xee, 0x3c];
        let avcc = build_avcc(&[sps], &[pps], 4);
        let (blob, n) = parse_avcc_config(&avcc).expect("valid avcC");
        assert_eq!(n, 4);
        let mut want = Vec::new();
        push_annexb(&mut want, sps);
        push_annexb(&mut want, pps);
        assert_eq!(blob, want);

        // lengthSizeMinusOne == 1 -> 2-byte prefixes.
        let avcc2 = build_avcc(&[sps], &[pps], 2);
        assert_eq!(parse_avcc_config(&avcc2).unwrap().1, 2);

        // Truncated record -> None (fall back, don't mis-decode).
        assert!(parse_avcc_config(&avcc[..7]).is_none());
    }

    #[test]
    fn parse_hvcc_extracts_all_arrays_in_order() {
        let vps: &[u8] = &[0x40, 0x01, 0x0c];
        let sps: &[u8] = &[0x42, 0x01, 0x01];
        let pps: &[u8] = &[0x44, 0x01];
        let mut hvcc = vec![0u8; 22]; // fixed prefix, indices 0..=21
        hvcc[0] = 1;
        hvcc[21] = 0xFC | 3; // lengthSizeMinusOne = 3 -> size 4
        hvcc.push(3); // numArrays
        for (ty, nal) in [(0x20u8, vps), (0x21, sps), (0x22, pps)] {
            hvcc.push(ty);
            hvcc.extend_from_slice(&1u16.to_be_bytes()); // one NAL
            hvcc.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            hvcc.extend_from_slice(nal);
        }
        let (blob, n) = parse_hvcc_config(&hvcc).expect("valid hvcC");
        assert_eq!(n, 4);
        let mut want = Vec::new();
        push_annexb(&mut want, vps);
        push_annexb(&mut want, sps);
        push_annexb(&mut want, pps);
        assert_eq!(blob, want);
        assert!(parse_hvcc_config(&hvcc[..24]).is_none());
    }

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
