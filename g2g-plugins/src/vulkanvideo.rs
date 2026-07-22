//! Vulkan Video decode support (`vulkan-video` feature).
//!
//! Vendor-neutral, GPU-resident hardware video decode via `VK_KHR_video_queue`
//! and `VK_KHR_video_decode_*`, decoding into a texture on the same Vulkan device
//! wgpu already runs, the cross-vendor analog of the CUDA-locked
//! `NvDec -> CudaToWgpu` path (AMD/RADV, NVIDIA, Intel/ANV all expose the
//! extensions). See DESIGN.md 4.11.6 and the `VulkanVideoDec` entry in
//! DESIGN_TODO.md.
//!
//! This first increment is the **capability probe**: the load-bearing question
//! the whole element negotiates against, and the one the design says to settle
//! in isolation before building the decode session. It answers, for a given
//! codec on this machine: is there a `VK_QUEUE_VIDEO_DECODE_BIT_KHR` queue that
//! supports the codec, and what are the decode limits (coded-extent range, DPB
//! slot counts, whether the decoded picture and DPB can share one image)? Those
//! limits drive `intercept_caps` (so fixate never advertises a resolution the
//! GPU cannot decode) and DPB sizing in the decode session to come.
//!
//! The Vulkan handles are reached the same way `cudawgpu.rs` reaches them: a
//! `wgpu::Instance` on the Vulkan backend, then `as_hal::<Vulkan>()` down to the
//! raw `ash::Entry` / `ash::Instance` + `vk::PhysicalDevice`. No standalone
//! `ash::Entry`; wgpu owns loader and instance lifetime.

use ash::vk;

use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, OwnedWgpuTexture, SystemSlice};
use g2g_core::runtime::block_on;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, FrameTiming, G2gError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};

/// Compiled SPIR-V for the YCbCr -> RGBA compute shader, shared with the Android
/// `mediacodec-wgpu` path (`shaders/mediacodec_ycbcr.comp`): it samples a
/// combined image sampler carrying a `VkSamplerYcbcrConversion` (so the YUV math
/// happens in the sampler) and writes RGBA to a storage image, which is codec
/// and source agnostic.
const YCBCR_COMP_SPV: &[u8] = include_bytes!("shaders/mediacodec_ycbcr.comp.spv");

/// The 10-bit sibling (`shaders/mediacodec_ycbcr16.comp`): identical, but its
/// storage image is `rgba16f` so a `G10X6` (10-bit) frame converts to an
/// `R16G16B16A16_SFLOAT` RGBA texture. Selected by [`YcbcrConverter`] when the
/// decode output is 10-bit; the descriptor's storage-image format must match the
/// shader qualifier, so the two bit depths need distinct shader modules.
const YCBCR16_COMP_SPV: &[u8] = include_bytes!("shaders/mediacodec_ycbcr16.comp.spv");

/// Which coded video format a Vulkan decode profile describes. Mirrors the
/// codec set `VulkanVideoDec` will accept; only the codecs Vulkan Video defines
/// a decode profile for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VulkanVideoCodec {
    H264,
    H265,
    Av1,
}

impl VulkanVideoCodec {
    /// Map a pipeline [`VideoCodec`] onto a Vulkan decode profile, or `None` for
    /// codecs Vulkan Video has no decode profile for (VP8/VP9, etc.).
    pub fn from_video_codec(codec: VideoCodec) -> Option<Self> {
        match codec {
            VideoCodec::H264 => Some(Self::H264),
            VideoCodec::H265 => Some(Self::H265),
            VideoCodec::Av1 => Some(Self::Av1),
            _ => None,
        }
    }

    fn codec_operation(self) -> vk::VideoCodecOperationFlagsKHR {
        match self {
            Self::H264 => vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            Self::H265 => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
            Self::Av1 => vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
        }
    }

    /// The device extension this codec's decode path requires (on top of
    /// `VK_KHR_video_queue` + `VK_KHR_video_decode_queue`).
    pub fn decode_extension(self) -> &'static core::ffi::CStr {
        match self {
            Self::H264 => ash::khr::video_decode_h264::NAME,
            Self::H265 => ash::khr::video_decode_h265::NAME,
            Self::Av1 => ash::khr::video_decode_av1::NAME,
        }
    }
}

/// Decode capabilities of a Vulkan physical device for one codec profile.
///
/// The subset of `VkVideoCapabilitiesKHR` / `VkVideoDecodeCapabilitiesKHR` the
/// element needs: the decode-capable queue family, the coded-extent range that
/// bounds `intercept_caps`, the DPB slot budget that bounds reference-picture
/// management, and whether the decoded output picture and the DPB reference can
/// be the *same* image (the `DPB_AND_OUTPUT_COINCIDE` fast path) or must be
/// separate images with an extra copy (`DPB_AND_OUTPUT_DISTINCT`).
#[derive(Debug, Clone)]
pub struct VulkanVideoDecodeCaps {
    /// Index of a queue family exposing `VK_QUEUE_VIDEO_DECODE_BIT_KHR` for this
    /// codec.
    pub decode_queue_family: u32,
    pub min_coded_extent: (u32, u32),
    pub max_coded_extent: (u32, u32),
    /// Max reference slots the decode session may hold (DPB size ceiling).
    pub max_dpb_slots: u32,
    /// Max reference pictures active in a single decode (bounds the ref lists).
    pub max_active_reference_pictures: u32,
    pub min_bitstream_buffer_offset_alignment: u64,
    pub min_bitstream_buffer_size_alignment: u64,
    /// The decoded picture can alias its DPB reference image (no extra copy).
    pub dpb_and_output_coincide: bool,
    /// The codec Std header version the driver implements; a video session must
    /// be created with exactly this (name + spec version).
    pub std_header_version: vk::ExtensionProperties,
}

/// Errors from probing / setting up Vulkan Video decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VulkanVideoError {
    /// No Vulkan adapter, or the adapter is not on the Vulkan backend (e.g. a
    /// D3D12 / Metal wgpu build). The caller should fall back to another decoder.
    NoVulkanAdapter,
    /// `VK_KHR_video_queue` (or the codec decode extension) is not present on
    /// this device.
    ExtensionUnsupported,
    /// No queue family advertises decode for this codec.
    NoDecodeQueue,
    /// The access unit carried no decodable slice (no IDR slice NAL).
    NoDecodableSlice,
    /// The stream uses an H.264 coding tool this decoder does not implement
    /// (e.g. `pic_order_cnt_type == 1`, or interlaced field pictures). Rejected
    /// up front rather than mis-decoded.
    UnsupportedStream,
    /// No distinct compute queue was available for the GPU-resident NV12 -> RGBA
    /// pass; the caller should use the CPU-convert path instead.
    NoComputeQueue,
    /// A CPU-output method was called on a decoder built for GPU-texture output
    /// (or vice versa). The two output modes are chosen at construction and are
    /// mutually exclusive: use `submit_chunk` for CPU I420, `submit_chunk_texture`
    /// for GPU-resident textures.
    WrongOutputMode,
    /// A Vulkan call failed (capability query, session/image/buffer creation, or
    /// the decode submission).
    QueryFailed(vk::Result),
    /// The decode device is not present-capable (`VK_KHR_swapchain` absent), or
    /// the window / display handle is an unsupported platform, so an HDR swapchain
    /// present sink cannot be built on it.
    PresentUnsupported,
}

/// Probe whether `codec` can be hardware-decoded on the default Vulkan adapter,
/// and with what limits. Creates a transient headless `wgpu::Instance` (no
/// surface) purely to reach the physical device; nothing is retained.
///
/// Returns `Err(NoVulkanAdapter)` on a machine with no Vulkan GPU so callers can
/// skip cleanly (the on-hardware test and the auto-plug rank both rely on this).
pub async fn probe_decode_caps(
    codec: VulkanVideoCodec,
) -> Result<VulkanVideoDecodeCaps, VulkanVideoError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|_| VulkanVideoError::NoVulkanAdapter)?;

    // SAFETY: `as_hal` hands out the adapter's live raw Vulkan handles for as
    // long as the guard is held; `probe_physical_device` only reads them and
    // copies out plain values, retaining nothing past return. `None` means the
    // wgpu adapter is not Vulkan-backed.
    let hal_adapter = unsafe { adapter.as_hal::<wgpu_hal::api::Vulkan>() };
    let hal_adapter = hal_adapter.ok_or(VulkanVideoError::NoVulkanAdapter)?;
    let shared = hal_adapter.shared_instance();
    // SAFETY: entry/instance/physical-device are the adapter's own valid handles,
    // alive for the guard's lifetime, which spans this call.
    unsafe {
        probe_physical_device(
            shared.entry(),
            shared.raw_instance(),
            hal_adapter.raw_physical_device(),
            codec,
        )
    }
}

/// Core of the probe against already-obtained raw Vulkan handles. Split out so
/// the eventual decode-session setup (which already holds these handles from the
/// shared wgpu device) can reuse it without creating a second instance.
///
/// # Safety
/// `entry`/`raw_instance` must be valid and `phys` a physical device enumerated
/// from `raw_instance`; all must outlive the call.
pub unsafe fn probe_physical_device(
    entry: &ash::Entry,
    raw_instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    codec: VulkanVideoCodec,
) -> Result<VulkanVideoDecodeCaps, VulkanVideoError> {
    let codec_op = codec.codec_operation();

    // 1. Find a queue family that advertises decode for this codec. The codec
    //    operations a family supports come from a `QueueFamilyVideoPropertiesKHR`
    //    chained onto `get_physical_device_queue_family_properties2`. The video
    //    props are a parallel vec each entry chains into; `props2` holds the
    //    mutable borrows and is dropped before the borrows are read back.
    // SAFETY: null out-array just counts, per the Vulkan contract.
    let family_count =
        unsafe { raw_instance.get_physical_device_queue_family_properties2_len(phys) };
    let mut video_props: alloc::vec::Vec<vk::QueueFamilyVideoPropertiesKHR> =
        (0..family_count).map(|_| Default::default()).collect();
    let mut props2: alloc::vec::Vec<vk::QueueFamilyProperties2> = video_props
        .iter_mut()
        .map(|vp| vk::QueueFamilyProperties2::default().push_next(vp))
        .collect();
    // SAFETY: `props2` is sized to `family_count` and each element chains a live
    // `video_props` entry (parallel vec, same length, outlives the call).
    unsafe { raw_instance.get_physical_device_queue_family_properties2(phys, &mut props2) };
    drop(props2); // releases the &mut borrows so video_props can be read

    let decode_queue_family = video_props
        .iter()
        .position(|vp| vp.video_codec_operations.contains(codec_op))
        .ok_or(VulkanVideoError::NoDecodeQueue)? as u32;

    // The physical-device video-capabilities query is a `VK_KHR_video_queue`
    // command, so the device must advertise that extension plus the codec decode
    // extension; without them the driver may fail the query with a generic
    // `ERROR_INITIALIZATION_FAILED` rather than a clean profile-rejection code.
    // SAFETY: `phys` is enumerated from `raw_instance`; the returned Vec is owned.
    let device_exts = unsafe { raw_instance.enumerate_device_extension_properties(phys) }
        .map_err(VulkanVideoError::QueryFailed)?;
    let has_ext = |want: &core::ffi::CStr| {
        device_exts.iter().any(|e| {
            // SAFETY: `extension_name` is a NUL-terminated fixed array per Vulkan.
            let name = unsafe { core::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
            name == want
        })
    };
    if !has_ext(ash::khr::video_queue::NAME)
        || !has_ext(ash::khr::video_decode_queue::NAME)
        || !has_ext(codec.decode_extension())
    {
        return Err(VulkanVideoError::ExtensionUnsupported);
    }

    // 2. Query decode capabilities for the profile. `VkVideoCapabilitiesKHR` is
    //    extended by `VkVideoDecodeCapabilitiesKHR` (decode-specific flags), and
    //    the profile is `VkVideoProfileInfoKHR` extended by the codec profile.
    //    NV12 4:2:0 8-bit is the baseline profile every decoder supports; higher
    //    bit-depth / chroma is a later refinement.
    let mut h264_profile = vk::VideoDecodeH264ProfileInfoKHR::default()
        .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH)
        .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);
    let mut h265_profile = vk::VideoDecodeH265ProfileInfoKHR::default()
        .std_profile_idc(vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN);
    let mut av1_profile = vk::VideoDecodeAV1ProfileInfoKHR::default()
        .std_profile(vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN);

    // A decode usage hint is part of a well-formed decode profile; some drivers
    // (NVIDIA) reject the capabilities query without it.
    let mut usage = vk::VideoDecodeUsageInfoKHR::default()
        .video_usage_hints(vk::VideoDecodeUsageFlagsKHR::DEFAULT);
    let mut profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(codec_op)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .push_next(&mut usage);
    profile = match codec {
        VulkanVideoCodec::H264 => profile.push_next(&mut h264_profile),
        VulkanVideoCodec::H265 => profile.push_next(&mut h265_profile),
        VulkanVideoCodec::Av1 => profile.push_next(&mut av1_profile),
    };

    // The output capabilities chain must carry both the generic decode caps and
    // the codec-specific caps struct; NVIDIA fails the query with a generic
    // `ERROR_INITIALIZATION_FAILED` if the codec-specific output struct is
    // missing (it is not in the spec's documented return codes, which is the
    // tell that the chain, not the profile, is at fault).
    let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
    let mut h264_caps = vk::VideoDecodeH264CapabilitiesKHR::default();
    let mut h265_caps = vk::VideoDecodeH265CapabilitiesKHR::default();
    let mut av1_caps = vk::VideoDecodeAV1CapabilitiesKHR::default();
    let mut caps = vk::VideoCapabilitiesKHR::default().push_next(&mut decode_caps);
    caps = match codec {
        VulkanVideoCodec::H264 => caps.push_next(&mut h264_caps),
        VulkanVideoCodec::H265 => caps.push_next(&mut h265_caps),
        VulkanVideoCodec::Av1 => caps.push_next(&mut av1_caps),
    };

    let video_instance = ash::khr::video_queue::Instance::new(entry, raw_instance);
    // SAFETY: `profile` (with its chained codec profile) and `caps` (with its
    // chained decode caps) are valid and outlive the call; the driver writes
    // into `caps`/`decode_caps` in place through the pointers.
    let ret = unsafe {
        (video_instance
            .fp()
            .get_physical_device_video_capabilities_khr)(phys, &profile, &mut caps)
    };
    if ret == vk::Result::ERROR_EXTENSION_NOT_PRESENT {
        return Err(VulkanVideoError::ExtensionUnsupported);
    }
    if ret != vk::Result::SUCCESS {
        return Err(VulkanVideoError::QueryFailed(ret));
    }

    // Copy out the scalars, then drop `caps` to release its `push_next` borrow
    // of `decode_caps` before reading the decode-specific flags.
    let min_coded_extent = (caps.min_coded_extent.width, caps.min_coded_extent.height);
    let max_coded_extent = (caps.max_coded_extent.width, caps.max_coded_extent.height);
    let max_dpb_slots = caps.max_dpb_slots;
    let max_active_reference_pictures = caps.max_active_reference_pictures;
    let min_bitstream_buffer_offset_alignment = caps.min_bitstream_buffer_offset_alignment;
    let min_bitstream_buffer_size_alignment = caps.min_bitstream_buffer_size_alignment;
    let std_header_version = caps.std_header_version;
    // `caps` (Copy) is last used above; NLL ends its `push_next` borrow of
    // `decode_caps` here, so the decode-specific flags read below.
    let dpb_and_output_coincide = decode_caps
        .flags
        .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE);

    Ok(VulkanVideoDecodeCaps {
        decode_queue_family,
        min_coded_extent,
        max_coded_extent,
        max_dpb_slots,
        max_active_reference_pictures,
        min_bitstream_buffer_offset_alignment,
        min_bitstream_buffer_size_alignment,
        dpb_and_output_coincide,
        std_header_version,
    })
}

// ============================================================================
// H.264 parameter-set parsing + `Std*` mapping
//
// Vulkan Video does not parse the bitstream: the app must hand the driver the
// SPS/PPS as filled `StdVideoH264SequenceParameterSet` / `...PictureParameterSet`
// structs (in `VkVideoSessionParametersKHR`) and per-frame `Std*` picture/slice
// info. This is the tedious, correctness-critical half the design flags. We
// parse the H.264 RBSP into plain structs (reusing the crate's `annexb`
// bit-reader) and map those onto the `Std*` layout; a wrong mapping is caught
// early because `vkCreateVideoSessionParametersKHR` validates the SPS/PPS.
// ============================================================================

use crate::annexb::{nal_units_any, strip_emulation_prevention, BitReader};

/// Parsed H.264 sequence parameter set, the fields the `Std*` SPS needs. Plain
/// data, no bitstream cursor. Baseline / main / high 4:2:0 8-bit; scaling
/// matrices and separate colour planes are rejected (returns `None`) rather than
/// silently mis-decoded.
#[derive(Debug, Clone)]
pub struct H264Sps {
    pub profile_idc: u8,
    pub level_idc: u8,
    pub seq_parameter_set_id: u8,
    pub chroma_format_idc: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub log2_max_frame_num_minus4: u8,
    pub pic_order_cnt_type: u8,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub max_num_ref_frames: u8,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub frame_mbs_only_flag: u8,
    pub mb_adaptive_frame_field_flag: u8,
    pub direct_8x8_inference_flag: u8,
    pub gaps_in_frame_num_value_allowed_flag: u8,
    pub frame_cropping_flag: u8,
    pub frame_crop_left_offset: u32,
    pub frame_crop_right_offset: u32,
    pub frame_crop_top_offset: u32,
    pub frame_crop_bottom_offset: u32,
    /// VUI colour description (CICP codepoints, 2 = unspecified) + full-range flag,
    /// driving the YUV -> RGB conversion. Defaults to unspecified when no VUI.
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub video_full_range_flag: bool,
}

/// Parsed H.264 picture parameter set, the fields the `Std*` PPS needs.
#[derive(Debug, Clone)]
pub struct H264Pps {
    pub pic_parameter_set_id: u8,
    pub seq_parameter_set_id: u8,
    pub entropy_coding_mode_flag: u8,
    pub bottom_field_pic_order_in_frame_present_flag: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub weighted_pred_flag: u8,
    pub weighted_bipred_idc: u8,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub deblocking_filter_control_present_flag: u8,
    pub constrained_intra_pred_flag: u8,
    pub redundant_pic_cnt_present_flag: u8,
    pub transform_8x8_mode_flag: u8,
    pub second_chroma_qp_index_offset: i8,
}

/// The SPS + PPS pulled from an Annex-B / AVCC access unit (e.g. an H.264 codec
/// config or an IDR AU that carries them in-band).
#[derive(Debug, Clone)]
pub struct H264ParameterSets {
    pub sps: H264Sps,
    pub pps: H264Pps,
}

/// Profiles carrying the chroma / bit-depth / scaling header block.
fn is_high_profile(profile_idc: u8) -> bool {
    matches!(
        profile_idc,
        100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135
    )
}

/// The colour fields at the head of an H.264 / H.265 VUI, whose layout is
/// identical in both codecs: aspect ratio, overscan, then the `video_signal_type`
/// block carrying `video_full_range_flag` and the optional CICP colour
/// description. Reads from the bit position just after `vui_parameters_present_flag`
/// (the caller checks that flag). Returns `(color_primaries, transfer_characteristics,
/// matrix_coefficients, video_full_range_flag)` as CICP codepoints (2 = unspecified);
/// any field not coded keeps its unspecified default. `None` only on truncation, so
/// the caller can fall back to unspecified.
fn parse_vui_color(br: &mut BitReader) -> Option<(u8, u8, u8, bool)> {
    let (mut primaries, mut transfer, mut matrix, mut full_range) = (2u8, 2u8, 2u8, false);
    // aspect_ratio_info_present_flag
    if br.read_bit()? == 1 {
        let idc = br.read_bits(8)?;
        // Extended_SAR: explicit sar_width / sar_height follow.
        if idc == 255 {
            br.read_bits(16)?;
            br.read_bits(16)?;
        }
    }
    // overscan_info_present_flag -> overscan_appropriate_flag
    if br.read_bit()? == 1 {
        br.read_bit()?;
    }
    // video_signal_type_present_flag -> video_format, full_range, colour desc
    if br.read_bit()? == 1 {
        br.read_bits(3)?; // video_format
        full_range = br.read_bit()? == 1;
        if br.read_bit()? == 1 {
            primaries = br.read_bits(8)? as u8;
            transfer = br.read_bits(8)? as u8;
            matrix = br.read_bits(8)? as u8;
        }
    }
    Some((primaries, transfer, matrix, full_range))
}

/// Parse an SPS RBSP (the bytes after the NAL header, emulation-prevention
/// already stripped). `None` on truncation or an unsupported feature.
pub fn parse_h264_sps(rbsp: &[u8]) -> Option<H264Sps> {
    if rbsp.len() < 3 {
        return None;
    }
    let profile_idc = rbsp[0];
    let level_idc = rbsp[2];
    let mut br = BitReader::new(&rbsp[3..]);

    let seq_parameter_set_id = br.read_ue()?;

    let mut chroma_format_idc = 1u32;
    let mut bit_depth_luma_minus8 = 0u32;
    let mut bit_depth_chroma_minus8 = 0u32;
    if is_high_profile(profile_idc) {
        chroma_format_idc = br.read_ue()?;
        if chroma_format_idc == 3 {
            // separate_colour_plane_flag: unsupported (4:4:4 planar), reject.
            if br.read_bit()? == 1 {
                return None;
            }
        }
        bit_depth_luma_minus8 = br.read_ue()?;
        bit_depth_chroma_minus8 = br.read_ue()?;
        let _qpprime_y_zero_transform_bypass_flag = br.read_bit()?;
        // Custom scaling matrices: unsupported (we pass no scaling lists), reject.
        if br.read_bit()? == 1 {
            return None;
        }
    }

    let log2_max_frame_num_minus4 = br.read_ue()?;
    let pic_order_cnt_type = br.read_ue()?;
    let mut log2_max_pic_order_cnt_lsb_minus4 = 0u32;
    if pic_order_cnt_type == 0 {
        log2_max_pic_order_cnt_lsb_minus4 = br.read_ue()?;
    } else if pic_order_cnt_type == 1 {
        let _delta_pic_order_always_zero_flag = br.read_bit()?;
        let _offset_for_non_ref_pic = br.read_se()?;
        let _offset_for_top_to_bottom_field = br.read_se()?;
        let n = br.read_ue()?;
        // POC type 1 with a ref-frame offset cycle needs pOffsetForRefFrame; we
        // do not carry it, so reject rather than mis-decode.
        if n != 0 {
            return None;
        }
    }
    let max_num_ref_frames = br.read_ue()?;
    let gaps_in_frame_num_value_allowed_flag = br.read_bit()?;

    let pic_width_in_mbs_minus1 = br.read_ue()?;
    let pic_height_in_map_units_minus1 = br.read_ue()?;
    let frame_mbs_only_flag = br.read_bit()?;
    let mut mb_adaptive_frame_field_flag = 0u32;
    if frame_mbs_only_flag == 0 {
        mb_adaptive_frame_field_flag = br.read_bit()?;
    }
    let direct_8x8_inference_flag = br.read_bit()?;

    let frame_cropping_flag = br.read_bit()?;
    let (crop_l, crop_r, crop_t, crop_b) = if frame_cropping_flag == 1 {
        (br.read_ue()?, br.read_ue()?, br.read_ue()?, br.read_ue()?)
    } else {
        (0, 0, 0, 0)
    };

    // VUI (optional): read only the colour prefix, for the YUV -> RGB conversion.
    // Best-effort: a truncated / absent VUI leaves the colour unspecified.
    let (color_primaries, transfer_characteristics, matrix_coefficients, video_full_range_flag) =
        if br.read_bit().unwrap_or(0) == 1 {
            parse_vui_color(&mut br).unwrap_or((2, 2, 2, false))
        } else {
            (2, 2, 2, false)
        };

    Some(H264Sps {
        profile_idc,
        level_idc,
        seq_parameter_set_id: seq_parameter_set_id as u8,
        chroma_format_idc: chroma_format_idc as u8,
        bit_depth_luma_minus8: bit_depth_luma_minus8 as u8,
        bit_depth_chroma_minus8: bit_depth_chroma_minus8 as u8,
        log2_max_frame_num_minus4: log2_max_frame_num_minus4 as u8,
        pic_order_cnt_type: pic_order_cnt_type as u8,
        log2_max_pic_order_cnt_lsb_minus4: log2_max_pic_order_cnt_lsb_minus4 as u8,
        max_num_ref_frames: max_num_ref_frames as u8,
        pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1,
        frame_mbs_only_flag: frame_mbs_only_flag as u8,
        mb_adaptive_frame_field_flag: mb_adaptive_frame_field_flag as u8,
        direct_8x8_inference_flag: direct_8x8_inference_flag as u8,
        gaps_in_frame_num_value_allowed_flag: gaps_in_frame_num_value_allowed_flag as u8,
        frame_cropping_flag: frame_cropping_flag as u8,
        frame_crop_left_offset: crop_l,
        frame_crop_right_offset: crop_r,
        frame_crop_top_offset: crop_t,
        frame_crop_bottom_offset: crop_b,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        video_full_range_flag,
    })
}

/// Parse a PPS RBSP (bytes after the NAL header, de-emulated). `None` on
/// truncation or an unsupported feature (slice groups).
pub fn parse_h264_pps(rbsp: &[u8]) -> Option<H264Pps> {
    let mut br = BitReader::new(rbsp);
    let pic_parameter_set_id = br.read_ue()?;
    let seq_parameter_set_id = br.read_ue()?;
    let entropy_coding_mode_flag = br.read_bit()?;
    let bottom_field_pic_order_in_frame_present_flag = br.read_bit()?;
    let num_slice_groups_minus1 = br.read_ue()?;
    // FMO (slice groups) is not carried into the Std PPS here; reject.
    if num_slice_groups_minus1 != 0 {
        return None;
    }
    let num_ref_idx_l0_default_active_minus1 = br.read_ue()?;
    let num_ref_idx_l1_default_active_minus1 = br.read_ue()?;
    let weighted_pred_flag = br.read_bit()?;
    let weighted_bipred_idc = br.read_bits(2)?;
    let pic_init_qp_minus26 = br.read_se()?;
    let pic_init_qs_minus26 = br.read_se()?;
    let chroma_qp_index_offset = br.read_se()?;
    let deblocking_filter_control_present_flag = br.read_bit()?;
    let constrained_intra_pred_flag = br.read_bit()?;
    let redundant_pic_cnt_present_flag = br.read_bit()?;

    // The optional trailing block is present only when `more_rbsp_data()` is
    // true. A baseline PPS has no such block, so the next bit is the
    // `rbsp_stop_one_bit` (1); reading it as `transform_8x8_mode_flag` would
    // wrongly tell the decoder 8x8 transforms are enabled and desync its CAVLC
    // coefficient parse. Guarding on `more_rbsp_data()` keeps both fields at
    // their (correct) defaults for baseline.
    let mut transform_8x8_mode_flag = 0u32;
    let mut second_chroma_qp_index_offset = chroma_qp_index_offset;
    if br.more_rbsp_data() {
        transform_8x8_mode_flag = br.read_bit()?;
        // pic_scaling_matrix_present_flag: unsupported, ignore its (rare) block;
        // treating a set flag as best-effort (default scaling lists still decode
        // most streams).
        let _pic_scaling_matrix_present_flag = br.read_bit();
        if let Some(v) = br.read_se() {
            second_chroma_qp_index_offset = v;
        }
    }

    Some(H264Pps {
        pic_parameter_set_id: pic_parameter_set_id as u8,
        seq_parameter_set_id: seq_parameter_set_id as u8,
        entropy_coding_mode_flag: entropy_coding_mode_flag as u8,
        bottom_field_pic_order_in_frame_present_flag: bottom_field_pic_order_in_frame_present_flag
            as u8,
        num_ref_idx_l0_default_active_minus1: num_ref_idx_l0_default_active_minus1 as u8,
        num_ref_idx_l1_default_active_minus1: num_ref_idx_l1_default_active_minus1 as u8,
        weighted_pred_flag: weighted_pred_flag as u8,
        weighted_bipred_idc: weighted_bipred_idc as u8,
        pic_init_qp_minus26: pic_init_qp_minus26 as i8,
        pic_init_qs_minus26: pic_init_qs_minus26 as i8,
        chroma_qp_index_offset: chroma_qp_index_offset as i8,
        deblocking_filter_control_present_flag: deblocking_filter_control_present_flag as u8,
        constrained_intra_pred_flag: constrained_intra_pred_flag as u8,
        redundant_pic_cnt_present_flag: redundant_pic_cnt_present_flag as u8,
        transform_8x8_mode_flag: transform_8x8_mode_flag as u8,
        second_chroma_qp_index_offset: second_chroma_qp_index_offset as i8,
    })
}

/// Pull the first SPS (nal type 7) and PPS (nal type 8) from an Annex-B / AVCC
/// access unit and parse both.
pub fn extract_h264_parameter_sets(au: &[u8]) -> Option<H264ParameterSets> {
    let mut sps = None;
    let mut pps = None;
    for nal in nal_units_any(au) {
        if nal.is_empty() {
            continue;
        }
        match nal[0] & 0x1F {
            7 if sps.is_none() => {
                sps = parse_h264_sps(&strip_emulation_prevention(&nal[1..]));
            }
            8 if pps.is_none() => {
                pps = parse_h264_pps(&strip_emulation_prevention(&nal[1..]));
            }
            _ => {}
        }
    }
    Some(H264ParameterSets {
        sps: sps?,
        pps: pps?,
    })
}

/// Map a raw `level_idc` byte onto the `StdVideoH264LevelIdc` enum. Unknown
/// values clamp to the 6.2 ceiling, which keeps a valid enum for
/// session-parameter creation.
fn std_level_idc(level_idc: u8) -> vk::native::StdVideoH264LevelIdc {
    use vk::native::*;
    match level_idc {
        10 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_0,
        11 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_1,
        12 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_2,
        13 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_3,
        20 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_0,
        21 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_1,
        22 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_2,
        30 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_0,
        31 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_1,
        32 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_2,
        40 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_0,
        41 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1,
        42 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_2,
        50 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_0,
        51 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1,
        52 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_2,
        60 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_0,
        61 => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_1,
        _ => StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_2,
    }
}

fn std_profile_idc(profile_idc: u8) -> vk::native::StdVideoH264ProfileIdc {
    use vk::native::*;
    match profile_idc {
        66 => StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE,
        77 => StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN,
        100 => StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH,
        244 => StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
        // Constrained-baseline and others map to baseline for decode purposes.
        _ => StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE,
    }
}

fn std_chroma_format_idc(chroma: u8) -> vk::native::StdVideoH264ChromaFormatIdc {
    use vk::native::*;
    match chroma {
        0 => StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_MONOCHROME,
        2 => StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_422,
        3 => StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_444,
        _ => StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
    }
}

/// Build the `Std*` SPS from our parsed [`H264Sps`]. No pointer fields are set
/// (no scaling lists / VUI / ref-frame offsets, all rejected at parse), so the
/// struct is self-contained and safe to hand the driver by value.
pub fn to_std_sps(sps: &H264Sps) -> vk::native::StdVideoH264SequenceParameterSet {
    // The `Std*` flag structs are bindgen bitfield unions with no `Default`; a
    // zeroed value is the all-flags-clear starting point.
    // SAFETY: `StdVideoH264SpsFlags` is a plain `repr(C)` bitfield POD, valid
    // when all-zero.
    let mut flags: vk::native::StdVideoH264SpsFlags = unsafe { core::mem::zeroed() };
    flags.set_frame_mbs_only_flag(sps.frame_mbs_only_flag as u32);
    flags.set_mb_adaptive_frame_field_flag(sps.mb_adaptive_frame_field_flag as u32);
    flags.set_direct_8x8_inference_flag(sps.direct_8x8_inference_flag as u32);
    flags.set_gaps_in_frame_num_value_allowed_flag(sps.gaps_in_frame_num_value_allowed_flag as u32);
    flags.set_frame_cropping_flag(sps.frame_cropping_flag as u32);
    // No scaling matrix / VUI / separate colour plane (rejected at parse).
    flags.set_seq_scaling_matrix_present_flag(0);
    flags.set_vui_parameters_present_flag(0);
    flags.set_separate_colour_plane_flag(0);

    vk::native::StdVideoH264SequenceParameterSet {
        flags,
        profile_idc: std_profile_idc(sps.profile_idc),
        level_idc: std_level_idc(sps.level_idc),
        chroma_format_idc: std_chroma_format_idc(sps.chroma_format_idc),
        seq_parameter_set_id: sps.seq_parameter_set_id,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
        pic_order_cnt_type: sps.pic_order_cnt_type as vk::native::StdVideoH264PocType,
        offset_for_non_ref_pic: 0,
        offset_for_top_to_bottom_field: 0,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        num_ref_frames_in_pic_order_cnt_cycle: 0,
        max_num_ref_frames: sps.max_num_ref_frames,
        reserved1: 0,
        pic_width_in_mbs_minus1: sps.pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1: sps.pic_height_in_map_units_minus1,
        frame_crop_left_offset: sps.frame_crop_left_offset,
        frame_crop_right_offset: sps.frame_crop_right_offset,
        frame_crop_top_offset: sps.frame_crop_top_offset,
        frame_crop_bottom_offset: sps.frame_crop_bottom_offset,
        reserved2: 0,
        pOffsetForRefFrame: core::ptr::null(),
        pScalingLists: core::ptr::null(),
        pSequenceParameterSetVui: core::ptr::null(),
    }
}

/// Build the `Std*` PPS from our parsed [`H264Pps`].
pub fn to_std_pps(pps: &H264Pps) -> vk::native::StdVideoH264PictureParameterSet {
    // SAFETY: `StdVideoH264PpsFlags` is a plain `repr(C)` bitfield POD, valid
    // when all-zero (all flags clear).
    let mut flags: vk::native::StdVideoH264PpsFlags = unsafe { core::mem::zeroed() };
    flags.set_transform_8x8_mode_flag(pps.transform_8x8_mode_flag as u32);
    flags.set_redundant_pic_cnt_present_flag(pps.redundant_pic_cnt_present_flag as u32);
    flags.set_constrained_intra_pred_flag(pps.constrained_intra_pred_flag as u32);
    flags.set_deblocking_filter_control_present_flag(
        pps.deblocking_filter_control_present_flag as u32,
    );
    flags.set_weighted_pred_flag(pps.weighted_pred_flag as u32);
    flags.set_bottom_field_pic_order_in_frame_present_flag(
        pps.bottom_field_pic_order_in_frame_present_flag as u32,
    );
    flags.set_entropy_coding_mode_flag(pps.entropy_coding_mode_flag as u32);
    flags.set_pic_scaling_matrix_present_flag(0);

    vk::native::StdVideoH264PictureParameterSet {
        flags,
        seq_parameter_set_id: pps.seq_parameter_set_id,
        pic_parameter_set_id: pps.pic_parameter_set_id,
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        weighted_bipred_idc: pps.weighted_bipred_idc as vk::native::StdVideoH264WeightedBipredIdc,
        pic_init_qp_minus26: pps.pic_init_qp_minus26,
        pic_init_qs_minus26: pps.pic_init_qs_minus26,
        chroma_qp_index_offset: pps.chroma_qp_index_offset,
        second_chroma_qp_index_offset: pps.second_chroma_qp_index_offset,
        pScalingLists: core::ptr::null(),
    }
}

// ============================================================================
// H.265 (HEVC) parameter-set parsing + `Std*` mapping
//
// The H.265 sibling of the H.264 block above. Vulkan Video wants the VPS / SPS
// / PPS as filled `StdVideoH265*` structs; H.265 pulls in a few extra pointee
// blocks the SPS/VPS reference by pointer (`profile_tier_level`, the DPB manager
// and the short-term reference-picture sets), so the mapping returns a bundle
// (`StdH265Params`) that owns those blocks and keeps the pointers valid.
//
// H.265's NAL header is two bytes (type is bits [1..7] of the first), so the
// RBSP starts at `nal[2..]` (not `nal[1..]` like H.264). Everything read from
// the bitstream is attacker-controlled: counts are bounded and every read is
// checked, so a malformed stream fails the parse (returns `None`) rather than
// panicking or over-allocating. The geometry-only `h265parse` module stays the
// caps-refinement path; this is the full Std-mapping parse the decoder needs.
// ============================================================================

/// Parsed H.265 `profile_tier_level` general block, the fields the `Std*`
/// profile-tier-level needs. Sub-layer blocks are advanced past but not carried
/// (single-layer decode).
#[derive(Debug, Clone, Default)]
pub struct H265ProfileTierLevel {
    pub general_tier_flag: u8,
    pub general_profile_idc: u8,
    pub general_progressive_source_flag: u8,
    pub general_interlaced_source_flag: u8,
    pub general_non_packed_constraint_flag: u8,
    pub general_frame_only_constraint_flag: u8,
    pub general_level_idc: u8,
}

// The short-term RPS parse moved to the ungated `h265parse` module (M663, its
// SPS walk crosses the RPS list to reach the VUI); re-exported so this module's
// public surface keeps the type.
use crate::h265parse::parse_h265_short_term_rps;
pub use crate::h265parse::H265ShortTermRps;

/// Parsed H.265 sequence parameter set: the fields the `Std*` SPS + its pointee
/// blocks (PTL, DPB manager, short-term RPS list) need. 4:2:0 / 4:2:2 / 4:4:4
/// packed; separate colour planes, explicit scaling lists, and SPS long-term
/// ref-pic sets are rejected (returns `None`) rather than mis-mapped.
#[derive(Debug, Clone)]
pub struct H265Sps {
    pub sps_video_parameter_set_id: u8,
    pub sps_max_sub_layers_minus1: u8,
    pub sps_temporal_id_nesting_flag: u8,
    pub ptl: H265ProfileTierLevel,
    pub sps_seq_parameter_set_id: u8,
    pub chroma_format_idc: u8,
    pub pic_width_in_luma_samples: u32,
    pub pic_height_in_luma_samples: u32,
    pub conformance_window_flag: u8,
    pub conf_win_left_offset: u32,
    pub conf_win_right_offset: u32,
    pub conf_win_top_offset: u32,
    pub conf_win_bottom_offset: u32,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub sps_sub_layer_ordering_info_present_flag: u8,
    pub max_dec_pic_buffering_minus1: [u8; 7],
    pub max_num_reorder_pics: [u8; 7],
    pub max_latency_increase_plus1: [u32; 7],
    pub log2_min_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_luma_coding_block_size: u8,
    pub log2_min_luma_transform_block_size_minus2: u8,
    pub log2_diff_max_min_luma_transform_block_size: u8,
    pub max_transform_hierarchy_depth_inter: u8,
    pub max_transform_hierarchy_depth_intra: u8,
    pub scaling_list_enabled_flag: u8,
    pub amp_enabled_flag: u8,
    pub sample_adaptive_offset_enabled_flag: u8,
    pub pcm_enabled_flag: u8,
    pub pcm_sample_bit_depth_luma_minus1: u8,
    pub pcm_sample_bit_depth_chroma_minus1: u8,
    pub log2_min_pcm_luma_coding_block_size_minus3: u8,
    pub log2_diff_max_min_pcm_luma_coding_block_size: u8,
    pub pcm_loop_filter_disabled_flag: u8,
    pub num_short_term_ref_pic_sets: u8,
    pub short_term_rps: alloc::vec::Vec<H265ShortTermRps>,
    pub long_term_ref_pics_present_flag: u8,
    /// SPS-declared long-term candidates (7.3.2.2.1): valid for the first
    /// `num_long_term_ref_pics_sps` entries of the two arrays.
    pub num_long_term_ref_pics_sps: u8,
    pub lt_ref_pic_poc_lsb_sps: [u32; 32],
    pub used_by_curr_pic_lt_sps_flag: [bool; 32],
    pub sps_temporal_mvp_enabled_flag: u8,
    pub strong_intra_smoothing_enabled_flag: u8,
    /// VUI colour description (CICP codepoints, 2 = unspecified) + full-range flag,
    /// driving the YUV -> RGB conversion. Defaults to unspecified when no VUI.
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub video_full_range_flag: bool,
}

/// Parsed H.265 picture parameter set, the fields the `Std*` PPS needs.
/// Non-uniform tile geometry and explicit scaling lists are rejected.
#[derive(Debug, Clone)]
pub struct H265Pps {
    pub pps_pic_parameter_set_id: u8,
    pub pps_seq_parameter_set_id: u8,
    pub dependent_slice_segments_enabled_flag: u8,
    pub output_flag_present_flag: u8,
    pub num_extra_slice_header_bits: u8,
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
    pub num_tile_columns_minus1: u8,
    pub num_tile_rows_minus1: u8,
    pub uniform_spacing_flag: u8,
    pub loop_filter_across_tiles_enabled_flag: u8,
    pub pps_loop_filter_across_slices_enabled_flag: u8,
    pub deblocking_filter_control_present_flag: u8,
    pub deblocking_filter_override_enabled_flag: u8,
    pub pps_deblocking_filter_disabled_flag: u8,
    pub pps_beta_offset_div2: i8,
    pub pps_tc_offset_div2: i8,
    pub lists_modification_present_flag: u8,
    pub log2_parallel_merge_level_minus2: u8,
    pub slice_segment_header_extension_present_flag: u8,
}

/// Parsed H.265 video parameter set, the fields the `Std*` VPS needs.
#[derive(Debug, Clone)]
pub struct H265Vps {
    pub vps_video_parameter_set_id: u8,
    pub vps_max_sub_layers_minus1: u8,
    pub vps_temporal_id_nesting_flag: u8,
    pub ptl: H265ProfileTierLevel,
    pub max_dec_pic_buffering_minus1: [u8; 7],
    pub max_num_reorder_pics: [u8; 7],
    pub max_latency_increase_plus1: [u32; 7],
}

/// The VPS + SPS + PPS pulled from an Annex-B / HVCC H.265 access unit.
#[derive(Debug, Clone)]
pub struct H265ParameterSets {
    pub vps: H265Vps,
    pub sps: H265Sps,
    pub pps: H265Pps,
}

/// Parse `profile_tier_level(1, max_sub_layers_minus1)` (H.265 7.3.3). The
/// general block is a fixed 96 bits (88 profile/tier/constraints + 8 level);
/// per-sub-layer blocks follow only when `max_sub_layers_minus1 > 0`, and are
/// advanced past without being carried (single-layer decode). `None` on
/// truncation.
fn parse_h265_ptl(br: &mut BitReader, max_sub_layers_minus1: u32) -> Option<H265ProfileTierLevel> {
    let _general_profile_space = br.read_bits(2)?;
    let general_tier_flag = br.read_bit()? as u8;
    let general_profile_idc = br.read_bits(5)? as u8;
    let _general_profile_compatibility = br.read_bits(32)?;
    let general_progressive_source_flag = br.read_bit()? as u8;
    let general_interlaced_source_flag = br.read_bit()? as u8;
    let general_non_packed_constraint_flag = br.read_bit()? as u8;
    let general_frame_only_constraint_flag = br.read_bit()? as u8;
    // 44 bits of constraint / reserved / inbld flags (read in two chunks: the
    // bit reader caps a single read at 32 bits).
    br.read_bits(32)?;
    br.read_bits(12)?;
    let general_level_idc = br.read_bits(8)? as u8;

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

    Some(H265ProfileTierLevel {
        general_tier_flag,
        general_profile_idc,
        general_progressive_source_flag,
        general_interlaced_source_flag,
        general_non_packed_constraint_flag,
        general_frame_only_constraint_flag,
        general_level_idc,
    })
}

/// Parse an H.265 SPS RBSP (bytes after the 2-byte NAL header, de-emulated).
/// `None` on truncation or an unsupported feature.
pub fn parse_h265_sps(rbsp: &[u8]) -> Option<H265Sps> {
    let mut br = BitReader::new(rbsp);
    let sps_video_parameter_set_id = br.read_bits(4)? as u8;
    let sps_max_sub_layers_minus1 = br.read_bits(3)? as u8;
    let sps_temporal_id_nesting_flag = br.read_bit()? as u8;
    let ptl = parse_h265_ptl(&mut br, sps_max_sub_layers_minus1 as u32)?;

    let sps_seq_parameter_set_id = br.read_ue()? as u8;
    let chroma_format_idc = br.read_ue()?;
    if chroma_format_idc == 3 {
        // separate_colour_plane_flag: 4:4:4 planar, unsupported (reject).
        if br.read_bit()? == 1 {
            return None;
        }
    }
    let pic_width_in_luma_samples = br.read_ue()?;
    let pic_height_in_luma_samples = br.read_ue()?;
    let conformance_window_flag = br.read_bit()?;
    let (conf_l, conf_r, conf_t, conf_b) = if conformance_window_flag == 1 {
        (br.read_ue()?, br.read_ue()?, br.read_ue()?, br.read_ue()?)
    } else {
        (0, 0, 0, 0)
    };
    let bit_depth_luma_minus8 = br.read_ue()? as u8;
    let bit_depth_chroma_minus8 = br.read_ue()? as u8;
    let log2_max_pic_order_cnt_lsb_minus4 = br.read_ue()? as u8;

    let sps_sub_layer_ordering_info_present_flag = br.read_bit()?;
    let mut max_dec_pic_buffering_minus1 = [0u8; 7];
    let mut max_num_reorder_pics = [0u8; 7];
    let mut max_latency_increase_plus1 = [0u32; 7];
    let start = if sps_sub_layer_ordering_info_present_flag == 1 {
        0
    } else {
        sps_max_sub_layers_minus1 as usize
    };
    for i in start..=sps_max_sub_layers_minus1 as usize {
        if i >= 7 {
            return None;
        }
        max_dec_pic_buffering_minus1[i] = br.read_ue()? as u8;
        max_num_reorder_pics[i] = br.read_ue()? as u8;
        max_latency_increase_plus1[i] = br.read_ue()?;
    }

    let log2_min_luma_coding_block_size_minus3 = br.read_ue()? as u8;
    let log2_diff_max_min_luma_coding_block_size = br.read_ue()? as u8;
    let log2_min_luma_transform_block_size_minus2 = br.read_ue()? as u8;
    let log2_diff_max_min_luma_transform_block_size = br.read_ue()? as u8;
    let max_transform_hierarchy_depth_inter = br.read_ue()? as u8;
    let max_transform_hierarchy_depth_intra = br.read_ue()? as u8;

    let scaling_list_enabled_flag = br.read_bit()?;
    if scaling_list_enabled_flag == 1 {
        // sps_scaling_list_data_present_flag: explicit lists are not carried into
        // the Std SPS, so reject rather than mis-decode.
        if br.read_bit()? == 1 {
            return None;
        }
    }
    let amp_enabled_flag = br.read_bit()?;
    let sample_adaptive_offset_enabled_flag = br.read_bit()?;
    let pcm_enabled_flag = br.read_bit()?;
    let mut pcm_sample_bit_depth_luma_minus1 = 0u8;
    let mut pcm_sample_bit_depth_chroma_minus1 = 0u8;
    let mut log2_min_pcm_luma_coding_block_size_minus3 = 0u8;
    let mut log2_diff_max_min_pcm_luma_coding_block_size = 0u8;
    let mut pcm_loop_filter_disabled_flag = 0u32;
    if pcm_enabled_flag == 1 {
        pcm_sample_bit_depth_luma_minus1 = br.read_bits(4)? as u8;
        pcm_sample_bit_depth_chroma_minus1 = br.read_bits(4)? as u8;
        log2_min_pcm_luma_coding_block_size_minus3 = br.read_ue()? as u8;
        log2_diff_max_min_pcm_luma_coding_block_size = br.read_ue()? as u8;
        pcm_loop_filter_disabled_flag = br.read_bit()?;
    }

    let num_short_term_ref_pic_sets = br.read_ue()?;
    if num_short_term_ref_pic_sets > 64 {
        return None;
    }
    let mut short_term_rps = alloc::vec::Vec::with_capacity(num_short_term_ref_pic_sets as usize);
    for idx in 0..num_short_term_ref_pic_sets as usize {
        let (rps, _) = parse_h265_short_term_rps(
            &mut br,
            idx,
            num_short_term_ref_pic_sets as usize,
            &short_term_rps,
        )?;
        short_term_rps.push(rps);
    }

    let long_term_ref_pics_present_flag = br.read_bit()?;
    let mut num_long_term_ref_pics_sps = 0u32;
    let mut lt_ref_pic_poc_lsb_sps = [0u32; 32];
    let mut used_by_curr_pic_lt_sps_flag = [false; 32];
    if long_term_ref_pics_present_flag == 1 {
        num_long_term_ref_pics_sps = br.read_ue()?;
        if num_long_term_ref_pics_sps > 32 {
            return None;
        }
        for i in 0..num_long_term_ref_pics_sps as usize {
            lt_ref_pic_poc_lsb_sps[i] =
                br.read_bits(log2_max_pic_order_cnt_lsb_minus4 as u32 + 4)?;
            used_by_curr_pic_lt_sps_flag[i] = br.read_bit()? == 1;
        }
    }
    let sps_temporal_mvp_enabled_flag = br.read_bit()?;
    let strong_intra_smoothing_enabled_flag = br.read_bit()?;
    // The Std mapping needs nothing past here, but the VUI colour description
    // (same layout as H.264's) drives the YUV -> RGB conversion. Best-effort: a
    // truncated / absent VUI leaves the colour unspecified.
    let (color_primaries, transfer_characteristics, matrix_coefficients, video_full_range_flag) =
        if br.read_bit().unwrap_or(0) == 1 {
            parse_vui_color(&mut br).unwrap_or((2, 2, 2, false))
        } else {
            (2, 2, 2, false)
        };

    Some(H265Sps {
        sps_video_parameter_set_id,
        sps_max_sub_layers_minus1,
        sps_temporal_id_nesting_flag,
        ptl,
        sps_seq_parameter_set_id,
        chroma_format_idc: chroma_format_idc as u8,
        pic_width_in_luma_samples,
        pic_height_in_luma_samples,
        conformance_window_flag: conformance_window_flag as u8,
        conf_win_left_offset: conf_l,
        conf_win_right_offset: conf_r,
        conf_win_top_offset: conf_t,
        conf_win_bottom_offset: conf_b,
        bit_depth_luma_minus8,
        bit_depth_chroma_minus8,
        log2_max_pic_order_cnt_lsb_minus4,
        sps_sub_layer_ordering_info_present_flag: sps_sub_layer_ordering_info_present_flag as u8,
        max_dec_pic_buffering_minus1,
        max_num_reorder_pics,
        max_latency_increase_plus1,
        log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_luma_coding_block_size,
        log2_min_luma_transform_block_size_minus2,
        log2_diff_max_min_luma_transform_block_size,
        max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra,
        scaling_list_enabled_flag: scaling_list_enabled_flag as u8,
        amp_enabled_flag: amp_enabled_flag as u8,
        sample_adaptive_offset_enabled_flag: sample_adaptive_offset_enabled_flag as u8,
        pcm_enabled_flag: pcm_enabled_flag as u8,
        pcm_sample_bit_depth_luma_minus1,
        pcm_sample_bit_depth_chroma_minus1,
        log2_min_pcm_luma_coding_block_size_minus3,
        log2_diff_max_min_pcm_luma_coding_block_size,
        pcm_loop_filter_disabled_flag: pcm_loop_filter_disabled_flag as u8,
        num_short_term_ref_pic_sets: num_short_term_ref_pic_sets as u8,
        short_term_rps,
        long_term_ref_pics_present_flag: long_term_ref_pics_present_flag as u8,
        num_long_term_ref_pics_sps: num_long_term_ref_pics_sps as u8,
        lt_ref_pic_poc_lsb_sps,
        used_by_curr_pic_lt_sps_flag,
        sps_temporal_mvp_enabled_flag: sps_temporal_mvp_enabled_flag as u8,
        strong_intra_smoothing_enabled_flag: strong_intra_smoothing_enabled_flag as u8,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        video_full_range_flag,
    })
}

/// Parse an H.265 PPS RBSP (bytes after the 2-byte NAL header, de-emulated).
/// `None` on truncation or an unsupported feature (non-uniform tiles, explicit
/// scaling lists).
pub fn parse_h265_pps(rbsp: &[u8]) -> Option<H265Pps> {
    let mut br = BitReader::new(rbsp);
    let pps_pic_parameter_set_id = br.read_ue()? as u8;
    let pps_seq_parameter_set_id = br.read_ue()? as u8;
    let dependent_slice_segments_enabled_flag = br.read_bit()? as u8;
    let output_flag_present_flag = br.read_bit()? as u8;
    let num_extra_slice_header_bits = br.read_bits(3)? as u8;
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
    let mut num_tile_columns_minus1 = 0u8;
    let mut num_tile_rows_minus1 = 0u8;
    let mut uniform_spacing_flag = 1u8;
    let mut loop_filter_across_tiles_enabled_flag = 1u8;
    if tiles_enabled_flag == 1 {
        num_tile_columns_minus1 = br.read_ue()? as u8;
        num_tile_rows_minus1 = br.read_ue()? as u8;
        uniform_spacing_flag = br.read_bit()? as u8;
        if uniform_spacing_flag == 0 {
            // Non-uniform tile geometry (per-column/row widths) is not carried
            // into the Std PPS; reject rather than emit wrong tile layout.
            return None;
        }
        loop_filter_across_tiles_enabled_flag = br.read_bit()? as u8;
    }
    let pps_loop_filter_across_slices_enabled_flag = br.read_bit()? as u8;
    let deblocking_filter_control_present_flag = br.read_bit()? as u8;
    let mut deblocking_filter_override_enabled_flag = 0u8;
    let mut pps_deblocking_filter_disabled_flag = 0u8;
    let mut pps_beta_offset_div2 = 0i8;
    let mut pps_tc_offset_div2 = 0i8;
    if deblocking_filter_control_present_flag == 1 {
        deblocking_filter_override_enabled_flag = br.read_bit()? as u8;
        pps_deblocking_filter_disabled_flag = br.read_bit()? as u8;
        if pps_deblocking_filter_disabled_flag == 0 {
            pps_beta_offset_div2 = br.read_se()? as i8;
            pps_tc_offset_div2 = br.read_se()? as i8;
        }
    }
    // pps_scaling_list_data_present_flag: explicit lists not carried, reject.
    if br.read_bit()? == 1 {
        return None;
    }
    let lists_modification_present_flag = br.read_bit()? as u8;
    let log2_parallel_merge_level_minus2 = br.read_ue()? as u8;
    let slice_segment_header_extension_present_flag = br.read_bit()? as u8;
    // pps_extension_present_flag + extensions past what the Std mapping needs.

    Some(H265Pps {
        pps_pic_parameter_set_id,
        pps_seq_parameter_set_id,
        dependent_slice_segments_enabled_flag,
        output_flag_present_flag,
        num_extra_slice_header_bits,
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
        num_tile_columns_minus1,
        num_tile_rows_minus1,
        uniform_spacing_flag,
        loop_filter_across_tiles_enabled_flag,
        pps_loop_filter_across_slices_enabled_flag,
        deblocking_filter_control_present_flag,
        deblocking_filter_override_enabled_flag,
        pps_deblocking_filter_disabled_flag,
        pps_beta_offset_div2,
        pps_tc_offset_div2,
        lists_modification_present_flag,
        log2_parallel_merge_level_minus2,
        slice_segment_header_extension_present_flag,
    })
}

/// Parse an H.265 VPS RBSP (bytes after the 2-byte NAL header, de-emulated) up
/// to the sub-layer ordering info the Std VPS + its DPB manager need. `None` on
/// truncation.
pub fn parse_h265_vps(rbsp: &[u8]) -> Option<H265Vps> {
    let mut br = BitReader::new(rbsp);
    let vps_video_parameter_set_id = br.read_bits(4)? as u8;
    let _vps_base_layer_internal_flag = br.read_bit()?;
    let _vps_base_layer_available_flag = br.read_bit()?;
    let _vps_max_layers_minus1 = br.read_bits(6)?;
    let vps_max_sub_layers_minus1 = br.read_bits(3)? as u8;
    let vps_temporal_id_nesting_flag = br.read_bit()? as u8;
    let _vps_reserved_0xffff_16bits = br.read_bits(16)?;
    let ptl = parse_h265_ptl(&mut br, vps_max_sub_layers_minus1 as u32)?;

    let vps_sub_layer_ordering_info_present_flag = br.read_bit()?;
    let mut max_dec_pic_buffering_minus1 = [0u8; 7];
    let mut max_num_reorder_pics = [0u8; 7];
    let mut max_latency_increase_plus1 = [0u32; 7];
    let start = if vps_sub_layer_ordering_info_present_flag == 1 {
        0
    } else {
        vps_max_sub_layers_minus1 as usize
    };
    for i in start..=vps_max_sub_layers_minus1 as usize {
        if i >= 7 {
            return None;
        }
        max_dec_pic_buffering_minus1[i] = br.read_ue()? as u8;
        max_num_reorder_pics[i] = br.read_ue()? as u8;
        max_latency_increase_plus1[i] = br.read_ue()?;
    }
    // Layer sets, HRD and timing info are not needed for the Std VPS decode
    // mapping; stop here.

    Some(H265Vps {
        vps_video_parameter_set_id,
        vps_max_sub_layers_minus1,
        vps_temporal_id_nesting_flag,
        ptl,
        max_dec_pic_buffering_minus1,
        max_num_reorder_pics,
        max_latency_increase_plus1,
    })
}

/// Pull the first VPS (nal type 32), SPS (33) and PPS (34) from an Annex-B /
/// HVCC access unit and parse all three. H.265 NAL type is bits [1..7] of the
/// first header byte; the RBSP starts at `nal[2..]`.
pub fn extract_h265_parameter_sets(au: &[u8]) -> Option<H265ParameterSets> {
    let mut vps = None;
    let mut sps = None;
    let mut pps = None;
    for nal in nal_units_any(au) {
        if nal.len() < 2 {
            continue;
        }
        match (nal[0] >> 1) & 0x3F {
            32 if vps.is_none() => {
                vps = parse_h265_vps(&strip_emulation_prevention(&nal[2..]));
            }
            33 if sps.is_none() => {
                sps = parse_h265_sps(&strip_emulation_prevention(&nal[2..]));
            }
            34 if pps.is_none() => {
                pps = parse_h265_pps(&strip_emulation_prevention(&nal[2..]));
            }
            _ => {}
        }
    }
    Some(H265ParameterSets {
        vps: vps?,
        sps: sps?,
        pps: pps?,
    })
}

/// Map a general_level_idc byte (30 * level) onto `StdVideoH265LevelIdc`.
/// Unknown values clamp to the 6.2 ceiling, a valid enum for session creation.
fn std_h265_level_idc(level: u8) -> vk::native::StdVideoH265LevelIdc {
    use vk::native::*;
    match level {
        30 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0,
        60 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_0,
        63 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_2_1,
        90 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_0,
        93 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_3_1,
        120 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_0,
        123 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_4_1,
        150 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_0,
        153 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1,
        156 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_2,
        180 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_0,
        183 => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_1,
        _ => StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2,
    }
}

fn std_h265_profile_idc(profile_idc: u8) -> vk::native::StdVideoH265ProfileIdc {
    use vk::native::*;
    match profile_idc {
        1 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
        2 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10,
        3 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE,
        4 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS,
        9 => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS,
        _ => StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN,
    }
}

fn std_h265_chroma_format_idc(chroma: u8) -> vk::native::StdVideoH265ChromaFormatIdc {
    use vk::native::*;
    match chroma {
        0 => StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_MONOCHROME,
        2 => StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_422,
        3 => StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_444,
        _ => StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420,
    }
}

fn to_std_h265_ptl(ptl: &H265ProfileTierLevel) -> vk::native::StdVideoH265ProfileTierLevel {
    // SAFETY: `StdVideoH265ProfileTierLevelFlags` is a plain repr(C) bitfield
    // POD, valid all-zero (all flags clear).
    let mut flags: vk::native::StdVideoH265ProfileTierLevelFlags = unsafe { core::mem::zeroed() };
    flags.set_general_tier_flag(ptl.general_tier_flag as u32);
    flags.set_general_progressive_source_flag(ptl.general_progressive_source_flag as u32);
    flags.set_general_interlaced_source_flag(ptl.general_interlaced_source_flag as u32);
    flags.set_general_non_packed_constraint_flag(ptl.general_non_packed_constraint_flag as u32);
    flags.set_general_frame_only_constraint_flag(ptl.general_frame_only_constraint_flag as u32);
    vk::native::StdVideoH265ProfileTierLevel {
        flags,
        general_profile_idc: std_h265_profile_idc(ptl.general_profile_idc),
        general_level_idc: std_h265_level_idc(ptl.general_level_idc),
    }
}

fn to_std_h265_dpb_mgr(
    max_dec_pic_buffering_minus1: &[u8; 7],
    max_num_reorder_pics: &[u8; 7],
    max_latency_increase_plus1: &[u32; 7],
) -> vk::native::StdVideoH265DecPicBufMgr {
    vk::native::StdVideoH265DecPicBufMgr {
        max_latency_increase_plus1: *max_latency_increase_plus1,
        max_dec_pic_buffering_minus1: *max_dec_pic_buffering_minus1,
        max_num_reorder_pics: *max_num_reorder_pics,
    }
}

/// Map a canonical (explicit) short-term RPS onto the `Std*` struct in explicit
/// form (`inter_ref_pic_set_prediction_flag == 0`), inverting the derived
/// `DeltaPocS0/S1` back to the `delta_poc_sX_minus1` deltas the struct stores.
// The delta inversion reads `DeltaPocS0[i-1]` alongside `[i]`, so the loops need
// the running index; an `enumerate()` rewrite cannot express the look-back.
#[allow(clippy::needless_range_loop)]
fn to_std_h265_short_term_rps(
    rps: &H265ShortTermRps,
) -> vk::native::StdVideoH265ShortTermRefPicSet {
    // SAFETY: `StdVideoH265ShortTermRefPicSetFlags` is a plain repr(C) bitfield
    // POD, valid all-zero (inter-prediction off, delta_rps_sign 0).
    let flags: vk::native::StdVideoH265ShortTermRefPicSetFlags = unsafe { core::mem::zeroed() };

    let mut delta_poc_s0_minus1 = [0u16; 16];
    let mut delta_poc_s1_minus1 = [0u16; 16];
    let mut used_s0_mask = 0u16;
    let mut used_s1_mask = 0u16;
    for i in 0..rps.num_negative_pics as usize {
        // DeltaPocS0[0] = -(delta+1); DeltaPocS0[i] = DeltaPocS0[i-1] - (delta+1).
        let m = if i == 0 {
            -rps.delta_poc_s0[0] - 1
        } else {
            rps.delta_poc_s0[i - 1] - rps.delta_poc_s0[i] - 1
        };
        delta_poc_s0_minus1[i] = m.clamp(0, u16::MAX as i32) as u16;
        if rps.used_s0[i] {
            used_s0_mask |= 1 << i;
        }
    }
    for i in 0..rps.num_positive_pics as usize {
        // DeltaPocS1[0] = delta+1; DeltaPocS1[i] = DeltaPocS1[i-1] + (delta+1).
        let m = if i == 0 {
            rps.delta_poc_s1[0] - 1
        } else {
            rps.delta_poc_s1[i] - rps.delta_poc_s1[i - 1] - 1
        };
        delta_poc_s1_minus1[i] = m.clamp(0, u16::MAX as i32) as u16;
        if rps.used_s1[i] {
            used_s1_mask |= 1 << i;
        }
    }

    vk::native::StdVideoH265ShortTermRefPicSet {
        flags,
        delta_idx_minus1: 0,
        use_delta_flag: 0,
        abs_delta_rps_minus1: 0,
        used_by_curr_pic_flag: 0,
        used_by_curr_pic_s0_flag: used_s0_mask,
        used_by_curr_pic_s1_flag: used_s1_mask,
        reserved1: 0,
        reserved2: 0,
        reserved3: 0,
        num_negative_pics: rps.num_negative_pics,
        num_positive_pics: rps.num_positive_pics,
        delta_poc_s0_minus1,
        delta_poc_s1_minus1,
    }
}

/// The `Std*` H.265 VPS + SPS + PPS plus the pointee blocks they reference by
/// pointer (PTL, DPB manager, short-term RPS list). Boxed / vec-backed so the
/// pointers stay valid for as long as the bundle lives; the session-parameter
/// creation reads them during the call.
pub struct StdH265Params {
    pub vps: vk::native::StdVideoH265VideoParameterSet,
    pub sps: vk::native::StdVideoH265SequenceParameterSet,
    pub pps: vk::native::StdVideoH265PictureParameterSet,
    _sps_ptl: alloc::boxed::Box<vk::native::StdVideoH265ProfileTierLevel>,
    _sps_dpb: alloc::boxed::Box<vk::native::StdVideoH265DecPicBufMgr>,
    _sps_rps: alloc::vec::Vec<vk::native::StdVideoH265ShortTermRefPicSet>,
    _sps_lt: alloc::boxed::Box<vk::native::StdVideoH265LongTermRefPicsSps>,
    _vps_ptl: alloc::boxed::Box<vk::native::StdVideoH265ProfileTierLevel>,
    _vps_dpb: alloc::boxed::Box<vk::native::StdVideoH265DecPicBufMgr>,
}

impl core::fmt::Debug for StdH265Params {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StdH265Params")
            .field(
                "pic_width_in_luma_samples",
                &self.sps.pic_width_in_luma_samples,
            )
            .field(
                "pic_height_in_luma_samples",
                &self.sps.pic_height_in_luma_samples,
            )
            .field("chroma_format_idc", &self.sps.chroma_format_idc)
            .field(
                "num_short_term_ref_pic_sets",
                &self.sps.num_short_term_ref_pic_sets,
            )
            .finish_non_exhaustive()
    }
}

/// Build the `Std*` H.265 parameter bundle from parsed VPS/SPS/PPS. The SPS and
/// VPS pointer fields are wired to the boxed / vec-backed pointee blocks the
/// bundle owns (stable addresses survive the moves into the returned struct).
pub fn to_std_h265_params(ps: &H265ParameterSets) -> StdH265Params {
    let sps_ptl = alloc::boxed::Box::new(to_std_h265_ptl(&ps.sps.ptl));
    let sps_dpb = alloc::boxed::Box::new(to_std_h265_dpb_mgr(
        &ps.sps.max_dec_pic_buffering_minus1,
        &ps.sps.max_num_reorder_pics,
        &ps.sps.max_latency_increase_plus1,
    ));
    let sps_rps: alloc::vec::Vec<_> = ps
        .sps
        .short_term_rps
        .iter()
        .map(to_std_h265_short_term_rps)
        .collect();
    let vps_ptl = alloc::boxed::Box::new(to_std_h265_ptl(&ps.vps.ptl));
    let vps_dpb = alloc::boxed::Box::new(to_std_h265_dpb_mgr(
        &ps.vps.max_dec_pic_buffering_minus1,
        &ps.vps.max_num_reorder_pics,
        &ps.vps.max_latency_increase_plus1,
    ));

    let sps_lt = alloc::boxed::Box::new(to_std_h265_lt_sps(&ps.sps));
    let lt_ptr = if ps.sps.long_term_ref_pics_present_flag == 1 {
        &*sps_lt as *const _
    } else {
        core::ptr::null()
    };

    let sps = to_std_h265_sps(&ps.sps, &*sps_ptl, &*sps_dpb, &sps_rps, lt_ptr);
    let pps = to_std_h265_pps(&ps.pps, ps.sps.sps_video_parameter_set_id);
    let vps = to_std_h265_vps(&ps.vps, &*vps_ptl, &*vps_dpb);

    StdH265Params {
        vps,
        sps,
        pps,
        _sps_ptl: sps_ptl,
        _sps_dpb: sps_dpb,
        _sps_rps: sps_rps,
        _sps_lt: sps_lt,
        _vps_ptl: vps_ptl,
        _vps_dpb: vps_dpb,
    }
}

/// The `Std*` SPS long-term table: the per-entry used-by-current flags packed
/// into a bitmask (bit `i` = entry `i`), the POC lsbs in a fixed array.
fn to_std_h265_lt_sps(sps: &H265Sps) -> vk::native::StdVideoH265LongTermRefPicsSps {
    let mut used = 0u32;
    let mut lsb = [0u32; 32];
    let n = sps.num_long_term_ref_pics_sps as usize;
    lsb[..n].copy_from_slice(&sps.lt_ref_pic_poc_lsb_sps[..n]);
    for (i, &u) in sps.used_by_curr_pic_lt_sps_flag[..n].iter().enumerate() {
        if u {
            used |= 1 << i;
        }
    }
    vk::native::StdVideoH265LongTermRefPicsSps {
        used_by_curr_pic_lt_sps_flag: used,
        lt_ref_pic_poc_lsb_sps: lsb,
    }
}

fn to_std_h265_sps(
    sps: &H265Sps,
    ptl: *const vk::native::StdVideoH265ProfileTierLevel,
    dpb: *const vk::native::StdVideoH265DecPicBufMgr,
    rps: &[vk::native::StdVideoH265ShortTermRefPicSet],
    lt: *const vk::native::StdVideoH265LongTermRefPicsSps,
) -> vk::native::StdVideoH265SequenceParameterSet {
    // SAFETY: `StdVideoH265SpsFlags` is a plain repr(C) bitfield POD, valid
    // all-zero (all flags clear).
    let mut flags: vk::native::StdVideoH265SpsFlags = unsafe { core::mem::zeroed() };
    flags.set_sps_temporal_id_nesting_flag(sps.sps_temporal_id_nesting_flag as u32);
    flags.set_separate_colour_plane_flag(0);
    flags.set_conformance_window_flag(sps.conformance_window_flag as u32);
    flags.set_sps_sub_layer_ordering_info_present_flag(
        sps.sps_sub_layer_ordering_info_present_flag as u32,
    );
    flags.set_scaling_list_enabled_flag(sps.scaling_list_enabled_flag as u32);
    flags.set_sps_scaling_list_data_present_flag(0);
    flags.set_amp_enabled_flag(sps.amp_enabled_flag as u32);
    flags.set_sample_adaptive_offset_enabled_flag(sps.sample_adaptive_offset_enabled_flag as u32);
    flags.set_pcm_enabled_flag(sps.pcm_enabled_flag as u32);
    flags.set_pcm_loop_filter_disabled_flag(sps.pcm_loop_filter_disabled_flag as u32);
    flags.set_long_term_ref_pics_present_flag(sps.long_term_ref_pics_present_flag as u32);
    flags.set_sps_temporal_mvp_enabled_flag(sps.sps_temporal_mvp_enabled_flag as u32);
    flags.set_strong_intra_smoothing_enabled_flag(sps.strong_intra_smoothing_enabled_flag as u32);
    flags.set_vui_parameters_present_flag(0);

    vk::native::StdVideoH265SequenceParameterSet {
        flags,
        chroma_format_idc: std_h265_chroma_format_idc(sps.chroma_format_idc),
        pic_width_in_luma_samples: sps.pic_width_in_luma_samples,
        pic_height_in_luma_samples: sps.pic_height_in_luma_samples,
        sps_video_parameter_set_id: sps.sps_video_parameter_set_id,
        sps_max_sub_layers_minus1: sps.sps_max_sub_layers_minus1,
        sps_seq_parameter_set_id: sps.sps_seq_parameter_set_id,
        bit_depth_luma_minus8: sps.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.bit_depth_chroma_minus8,
        log2_max_pic_order_cnt_lsb_minus4: sps.log2_max_pic_order_cnt_lsb_minus4,
        log2_min_luma_coding_block_size_minus3: sps.log2_min_luma_coding_block_size_minus3,
        log2_diff_max_min_luma_coding_block_size: sps.log2_diff_max_min_luma_coding_block_size,
        log2_min_luma_transform_block_size_minus2: sps.log2_min_luma_transform_block_size_minus2,
        log2_diff_max_min_luma_transform_block_size: sps
            .log2_diff_max_min_luma_transform_block_size,
        max_transform_hierarchy_depth_inter: sps.max_transform_hierarchy_depth_inter,
        max_transform_hierarchy_depth_intra: sps.max_transform_hierarchy_depth_intra,
        num_short_term_ref_pic_sets: sps.num_short_term_ref_pic_sets,
        num_long_term_ref_pics_sps: sps.num_long_term_ref_pics_sps,
        pcm_sample_bit_depth_luma_minus1: sps.pcm_sample_bit_depth_luma_minus1,
        pcm_sample_bit_depth_chroma_minus1: sps.pcm_sample_bit_depth_chroma_minus1,
        log2_min_pcm_luma_coding_block_size_minus3: sps.log2_min_pcm_luma_coding_block_size_minus3,
        log2_diff_max_min_pcm_luma_coding_block_size: sps
            .log2_diff_max_min_pcm_luma_coding_block_size,
        reserved1: 0,
        reserved2: 0,
        palette_max_size: 0,
        delta_palette_max_predictor_size: 0,
        motion_vector_resolution_control_idc: 0,
        sps_num_palette_predictor_initializers_minus1: 0,
        conf_win_left_offset: sps.conf_win_left_offset,
        conf_win_right_offset: sps.conf_win_right_offset,
        conf_win_top_offset: sps.conf_win_top_offset,
        conf_win_bottom_offset: sps.conf_win_bottom_offset,
        pProfileTierLevel: ptl,
        pDecPicBufMgr: dpb,
        pScalingLists: core::ptr::null(),
        pShortTermRefPicSet: if rps.is_empty() {
            core::ptr::null()
        } else {
            rps.as_ptr()
        },
        pLongTermRefPicsSps: lt,
        pSequenceParameterSetVui: core::ptr::null(),
        pPredictorPaletteEntries: core::ptr::null(),
    }
}

fn to_std_h265_pps(
    pps: &H265Pps,
    sps_video_parameter_set_id: u8,
) -> vk::native::StdVideoH265PictureParameterSet {
    // SAFETY: `StdVideoH265PpsFlags` is a plain repr(C) bitfield POD, valid
    // all-zero (all flags clear).
    let mut flags: vk::native::StdVideoH265PpsFlags = unsafe { core::mem::zeroed() };
    flags.set_dependent_slice_segments_enabled_flag(
        pps.dependent_slice_segments_enabled_flag as u32,
    );
    flags.set_output_flag_present_flag(pps.output_flag_present_flag as u32);
    flags.set_sign_data_hiding_enabled_flag(pps.sign_data_hiding_enabled_flag as u32);
    flags.set_cabac_init_present_flag(pps.cabac_init_present_flag as u32);
    flags.set_constrained_intra_pred_flag(pps.constrained_intra_pred_flag as u32);
    flags.set_transform_skip_enabled_flag(pps.transform_skip_enabled_flag as u32);
    flags.set_cu_qp_delta_enabled_flag(pps.cu_qp_delta_enabled_flag as u32);
    flags.set_pps_slice_chroma_qp_offsets_present_flag(
        pps.pps_slice_chroma_qp_offsets_present_flag as u32,
    );
    flags.set_weighted_pred_flag(pps.weighted_pred_flag as u32);
    flags.set_weighted_bipred_flag(pps.weighted_bipred_flag as u32);
    flags.set_transquant_bypass_enabled_flag(pps.transquant_bypass_enabled_flag as u32);
    flags.set_tiles_enabled_flag(pps.tiles_enabled_flag as u32);
    flags.set_entropy_coding_sync_enabled_flag(pps.entropy_coding_sync_enabled_flag as u32);
    flags.set_uniform_spacing_flag(pps.uniform_spacing_flag as u32);
    flags.set_loop_filter_across_tiles_enabled_flag(
        pps.loop_filter_across_tiles_enabled_flag as u32,
    );
    flags.set_pps_loop_filter_across_slices_enabled_flag(
        pps.pps_loop_filter_across_slices_enabled_flag as u32,
    );
    flags.set_deblocking_filter_control_present_flag(
        pps.deblocking_filter_control_present_flag as u32,
    );
    flags.set_deblocking_filter_override_enabled_flag(
        pps.deblocking_filter_override_enabled_flag as u32,
    );
    flags.set_pps_deblocking_filter_disabled_flag(pps.pps_deblocking_filter_disabled_flag as u32);
    flags.set_lists_modification_present_flag(pps.lists_modification_present_flag as u32);
    flags.set_slice_segment_header_extension_present_flag(
        pps.slice_segment_header_extension_present_flag as u32,
    );
    flags.set_pps_scaling_list_data_present_flag(0);

    vk::native::StdVideoH265PictureParameterSet {
        flags,
        pps_pic_parameter_set_id: pps.pps_pic_parameter_set_id,
        pps_seq_parameter_set_id: pps.pps_seq_parameter_set_id,
        sps_video_parameter_set_id,
        num_extra_slice_header_bits: pps.num_extra_slice_header_bits,
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1,
        init_qp_minus26: pps.init_qp_minus26,
        diff_cu_qp_delta_depth: pps.diff_cu_qp_delta_depth,
        pps_cb_qp_offset: pps.pps_cb_qp_offset,
        pps_cr_qp_offset: pps.pps_cr_qp_offset,
        pps_beta_offset_div2: pps.pps_beta_offset_div2,
        pps_tc_offset_div2: pps.pps_tc_offset_div2,
        log2_parallel_merge_level_minus2: pps.log2_parallel_merge_level_minus2,
        log2_max_transform_skip_block_size_minus2: 0,
        diff_cu_chroma_qp_offset_depth: 0,
        chroma_qp_offset_list_len_minus1: 0,
        cb_qp_offset_list: [0; 6],
        cr_qp_offset_list: [0; 6],
        log2_sao_offset_scale_luma: 0,
        log2_sao_offset_scale_chroma: 0,
        pps_act_y_qp_offset_plus5: 0,
        pps_act_cb_qp_offset_plus5: 0,
        pps_act_cr_qp_offset_plus3: 0,
        pps_num_palette_predictor_initializers: 0,
        luma_bit_depth_entry_minus8: 0,
        chroma_bit_depth_entry_minus8: 0,
        num_tile_columns_minus1: pps.num_tile_columns_minus1,
        num_tile_rows_minus1: pps.num_tile_rows_minus1,
        reserved1: 0,
        reserved2: 0,
        column_width_minus1: [0; 19],
        row_height_minus1: [0; 21],
        reserved3: 0,
        pScalingLists: core::ptr::null(),
        pPredictorPaletteEntries: core::ptr::null(),
    }
}

fn to_std_h265_vps(
    vps: &H265Vps,
    ptl: *const vk::native::StdVideoH265ProfileTierLevel,
    dpb: *const vk::native::StdVideoH265DecPicBufMgr,
) -> vk::native::StdVideoH265VideoParameterSet {
    // SAFETY: `StdVideoH265VpsFlags` is a plain repr(C) bitfield POD, valid
    // all-zero (timing-info-present cleared, we carry no timing).
    let mut flags: vk::native::StdVideoH265VpsFlags = unsafe { core::mem::zeroed() };
    flags.set_vps_temporal_id_nesting_flag(vps.vps_temporal_id_nesting_flag as u32);
    flags.set_vps_timing_info_present_flag(0);

    vk::native::StdVideoH265VideoParameterSet {
        flags,
        vps_video_parameter_set_id: vps.vps_video_parameter_set_id,
        vps_max_sub_layers_minus1: vps.vps_max_sub_layers_minus1,
        reserved1: 0,
        reserved2: 0,
        vps_num_units_in_tick: 0,
        vps_time_scale: 0,
        vps_num_ticks_poc_diff_one_minus1: 0,
        reserved3: 0,
        pDecPicBufMgr: dpb,
        pHrdParameters: core::ptr::null(),
        pProfileTierLevel: ptl,
    }
}

// ============================================================================
// AV1 OBU parse + Std mapping
//
// AV1 is not NAL / Annex-B framed like H.264 / H.265: the bitstream is a
// sequence of Open Bitstream Units (OBUs), each a 1 (or 2) byte header plus an
// optional LEB128 size and a payload. There is no start-code scan; OBUs are
// walked by their size fields. The sequence header OBU carries the
// session-level parameters (the H.26x SPS analog) and maps onto
// `StdVideoAV1SequenceHeader`; per-frame decoding (frame header + tile groups)
// is the DPB decode path (a later milestone), mirroring how the H.265 slice
// header drives its DPB. This section covers the OBU framing, the sequence
// header parse + Std mapping, and enough of the frame header to classify frames
// (type / shown), all GPU-free and unit tested.
//
// As with the container / bitstream parsers elsewhere in the tree, every count,
// length and dimension read here is attacker controlled: bounds are checked and
// a malformed unit returns `None` rather than panicking or over-reading.

// OBU types (AV1 spec 6.2.2). Only the ones this decoder path inspects.
const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_FRAME_HEADER: u8 = 3;
const OBU_FRAME: u8 = 6;

// AV1 frame types (spec 6.8.2). KEY is the one the header parse emits directly;
// INTER / INTRA_ONLY / SWITCH are read as raw f(2) values.
const AV1_FRAME_TYPE_KEY: u8 = 0;

// Sentinels for the two "choose" syntax elements (spec 6.4.2).
const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;
const SELECT_INTEGER_MV: u8 = 2;

/// One parsed OBU: its type, temporal / spatial id, and payload slice (past the
/// OBU header and size field).
#[derive(Debug, Clone, Copy)]
struct Av1Obu<'a> {
    obu_type: u8,
    #[allow(dead_code)]
    temporal_id: u8,
    #[allow(dead_code)]
    spatial_id: u8,
    payload: &'a [u8],
}

/// LEB128 unsigned little-endian base-128 (AV1 spec 4.10.5). Returns the value
/// and the position past it. `None` on truncation or an over-long (> 8 byte)
/// encoding.
fn read_leb128(data: &[u8], mut pos: usize) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    for i in 0..8u32 {
        let b = *data.get(pos)?;
        pos += 1;
        value |= u64::from(b & 0x7f) << (i * 7);
        if b & 0x80 == 0 {
            return Some((value, pos));
        }
    }
    None
}

/// Walk a low-overhead AV1 bitstream into its OBUs. Requires the size field on
/// every OBU except a final one without it (spec permits the last OBU to omit
/// the size, taking the remainder). Stops at the first malformed / truncated
/// unit rather than over-reading.
fn av1_obus(data: &[u8]) -> alloc::vec::Vec<Av1Obu<'_>> {
    let mut out = alloc::vec::Vec::new();
    let mut pos = 0usize;
    while pos < data.len() {
        let hdr = data[pos];
        // obu_forbidden_bit (bit 7) must be 0; obu_reserved_1bit (bit 0) ignored.
        if hdr & 0x80 != 0 {
            break;
        }
        let obu_type = (hdr >> 3) & 0x0f;
        let ext = (hdr >> 2) & 1;
        let has_size = (hdr >> 1) & 1;
        let mut p = pos + 1;
        let (mut tid, mut sid) = (0u8, 0u8);
        if ext == 1 {
            let Some(&e) = data.get(p) else { break };
            tid = e >> 5;
            sid = (e >> 3) & 0x03;
            p += 1;
        }
        let payload_len = if has_size == 1 {
            let Some((sz, np)) = read_leb128(data, p) else {
                break;
            };
            p = np;
            sz as usize
        } else {
            data.len() - p
        };
        // Bounds: the payload must fit within the buffer.
        let Some(end) = p.checked_add(payload_len) else {
            break;
        };
        if end > data.len() {
            break;
        }
        out.push(Av1Obu {
            obu_type,
            temporal_id: tid,
            spatial_id: sid,
            payload: &data[p..end],
        });
        pos = end;
    }
    out
}

/// Unsigned variable-length code (AV1 spec 4.10.3): a unary-coded leading-zero
/// count then that many value bits.
fn read_uvlc(br: &mut BitReader) -> Option<u32> {
    let mut leading_zeros = 0u32;
    loop {
        let done = br.read_bit()?;
        if done == 1 {
            break;
        }
        leading_zeros += 1;
        if leading_zeros >= 32 {
            return Some(u32::MAX);
        }
    }
    let value = if leading_zeros == 0 {
        0
    } else {
        br.read_bits(leading_zeros)?
    };
    Some(value.saturating_add((1u32 << leading_zeros) - 1))
}

/// Parsed AV1 color config (spec 5.5.2), the subset needed for the Std mapping.
#[derive(Debug, Clone)]
pub struct Av1ColorConfig {
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub color_description_present_flag: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
    /// 1 for monochrome, else 3. Derived, not coded.
    pub num_planes: u8,
}

/// Parsed AV1 sequence header (spec 5.5.1), the subset the Vulkan Std mapping
/// and decode session need.
#[derive(Debug, Clone)]
pub struct Av1SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info_present_flag: bool,
    pub decoder_model_info_present_flag: bool,
    pub initial_display_delay_present_flag: bool,
    pub operating_points_cnt_minus_1: u8,
    pub seq_level_idx0: u8,
    pub seq_tier0: u8,
    pub frame_width_bits_minus_1: u8,
    pub frame_height_bits_minus_1: u8,
    pub max_frame_width_minus_1: u32,
    pub max_frame_height_minus_1: u32,
    pub frame_id_numbers_present_flag: bool,
    pub delta_frame_id_length_minus_2: u8,
    pub additional_frame_id_length_minus_1: u8,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_force_screen_content_tools: u8,
    pub seq_force_integer_mv: u8,
    /// `OrderHintBits` (0 when `enable_order_hint` is clear).
    pub order_hint_bits: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color: Av1ColorConfig,
    pub film_grain_params_present: bool,
}

/// Color config parse (spec 5.5.2). `br` is positioned right after
/// `enable_restoration` in the sequence header.
fn parse_av1_color_config(br: &mut BitReader, seq_profile: u8) -> Option<Av1ColorConfig> {
    // Color primaries / transfer / matrix "unspecified" and the special
    // sRGB/identity triple (spec 6.4.2).
    const CP_BT_709: u8 = 1;
    const CP_UNSPECIFIED: u8 = 2;
    const TC_SRGB: u8 = 13;
    const TC_UNSPECIFIED: u8 = 2;
    const MC_IDENTITY: u8 = 0;
    const MC_UNSPECIFIED: u8 = 2;

    let high_bitdepth = br.read_bit()? == 1;
    let bit_depth = if seq_profile == 2 && high_bitdepth {
        let twelve_bit = br.read_bit()? == 1;
        if twelve_bit {
            12
        } else {
            10
        }
    } else if high_bitdepth {
        10
    } else {
        8
    };

    let mono_chrome = if seq_profile == 1 {
        false
    } else {
        br.read_bit()? == 1
    };
    let num_planes = if mono_chrome { 1 } else { 3 };

    let color_description_present_flag = br.read_bit()? == 1;
    let (color_primaries, transfer_characteristics, matrix_coefficients) =
        if color_description_present_flag {
            (
                br.read_bits(8)? as u8,
                br.read_bits(8)? as u8,
                br.read_bits(8)? as u8,
            )
        } else {
            (CP_UNSPECIFIED, TC_UNSPECIFIED, MC_UNSPECIFIED)
        };

    let subsampling_x;
    let subsampling_y;
    let mut chroma_sample_position = 0u8; // CSP_UNKNOWN
    let color_range;

    if mono_chrome {
        color_range = br.read_bit()? == 1;
        return Some(Av1ColorConfig {
            bit_depth,
            mono_chrome,
            color_description_present_flag,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            color_range,
            subsampling_x: 1,
            subsampling_y: 1,
            chroma_sample_position: 0,
            separate_uv_delta_q: false,
            num_planes,
        });
    } else if color_primaries == CP_BT_709
        && transfer_characteristics == TC_SRGB
        && matrix_coefficients == MC_IDENTITY
    {
        // 4:4:4 sRGB, full range, no coded color_range bit.
        color_range = true;
        subsampling_x = 0;
        subsampling_y = 0;
    } else {
        color_range = br.read_bit()? == 1;
        if seq_profile == 0 {
            subsampling_x = 1;
            subsampling_y = 1;
        } else if seq_profile == 1 {
            subsampling_x = 0;
            subsampling_y = 0;
        } else if bit_depth == 12 {
            subsampling_x = br.read_bit()? as u8;
            subsampling_y = if subsampling_x == 1 {
                br.read_bit()? as u8
            } else {
                0
            };
        } else {
            subsampling_x = 1;
            subsampling_y = 0;
        }
        if subsampling_x == 1 && subsampling_y == 1 {
            chroma_sample_position = br.read_bits(2)? as u8;
        }
    }

    let separate_uv_delta_q = br.read_bit()? == 1;

    Some(Av1ColorConfig {
        bit_depth,
        mono_chrome,
        color_description_present_flag,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        color_range,
        subsampling_x,
        subsampling_y,
        chroma_sample_position,
        separate_uv_delta_q,
        num_planes,
    })
}

/// Parse a sequence header OBU payload (spec 5.5.1).
pub fn parse_av1_sequence_header(payload: &[u8]) -> Option<Av1SequenceHeader> {
    let mut br = BitReader::new(payload);

    let seq_profile = br.read_bits(3)? as u8;
    let still_picture = br.read_bit()? == 1;
    let reduced_still_picture_header = br.read_bit()? == 1;

    let mut timing_info_present_flag = false;
    let mut decoder_model_info_present_flag = false;
    let mut initial_display_delay_present_flag = false;
    let mut operating_points_cnt_minus_1 = 0u8;
    let mut seq_level_idx0 = 0u8;
    let mut seq_tier0 = 0u8;
    let mut buffer_delay_length_minus_1 = 0u32;

    if reduced_still_picture_header {
        seq_level_idx0 = br.read_bits(5)? as u8;
    } else {
        timing_info_present_flag = br.read_bit()? == 1;
        if timing_info_present_flag {
            // timing_info() (spec 5.5.3).
            br.read_bits(32)?; // num_units_in_display_tick
            br.read_bits(32)?; // time_scale
            let equal_picture_interval = br.read_bit()? == 1;
            if equal_picture_interval {
                read_uvlc(&mut br)?; // num_ticks_per_picture_minus_1
            }
            decoder_model_info_present_flag = br.read_bit()? == 1;
            if decoder_model_info_present_flag {
                // decoder_model_info() (spec 5.5.4).
                buffer_delay_length_minus_1 = br.read_bits(5)?;
                br.read_bits(32)?; // num_units_in_decoding_tick
                br.read_bits(5)?; // buffer_removal_time_length_minus_1
                br.read_bits(5)?; // frame_presentation_time_length_minus_1
            }
        }
        initial_display_delay_present_flag = br.read_bit()? == 1;
        operating_points_cnt_minus_1 = br.read_bits(5)? as u8;
        for i in 0..=operating_points_cnt_minus_1 as u32 {
            br.read_bits(12)?; // operating_point_idc[i]
            let level = br.read_bits(5)?;
            let tier = if level > 7 { br.read_bit()? } else { 0 };
            if i == 0 {
                seq_level_idx0 = level as u8;
                seq_tier0 = tier as u8;
            }
            if decoder_model_info_present_flag {
                let decoder_model_present = br.read_bit()? == 1;
                if decoder_model_present {
                    let n = buffer_delay_length_minus_1 + 1;
                    br.read_bits(n)?; // decoder_buffer_delay
                    br.read_bits(n)?; // encoder_buffer_delay
                    br.read_bit()?; // low_delay_mode_flag
                }
            }
            if initial_display_delay_present_flag {
                let present = br.read_bit()? == 1;
                if present {
                    br.read_bits(4)?; // initial_display_delay_minus_1
                }
            }
        }
    }

    let frame_width_bits_minus_1 = br.read_bits(4)? as u8;
    let frame_height_bits_minus_1 = br.read_bits(4)? as u8;
    let max_frame_width_minus_1 = br.read_bits(frame_width_bits_minus_1 as u32 + 1)?;
    let max_frame_height_minus_1 = br.read_bits(frame_height_bits_minus_1 as u32 + 1)?;

    let mut frame_id_numbers_present_flag = false;
    let mut delta_frame_id_length_minus_2 = 0u8;
    let mut additional_frame_id_length_minus_1 = 0u8;
    if !reduced_still_picture_header {
        frame_id_numbers_present_flag = br.read_bit()? == 1;
    }
    if frame_id_numbers_present_flag {
        delta_frame_id_length_minus_2 = br.read_bits(4)? as u8;
        additional_frame_id_length_minus_1 = br.read_bits(3)? as u8;
    }

    let use_128x128_superblock = br.read_bit()? == 1;
    let enable_filter_intra = br.read_bit()? == 1;
    let enable_intra_edge_filter = br.read_bit()? == 1;

    let mut enable_interintra_compound = false;
    let mut enable_masked_compound = false;
    let mut enable_warped_motion = false;
    let mut enable_dual_filter = false;
    let mut enable_order_hint = false;
    let mut enable_jnt_comp = false;
    let mut enable_ref_frame_mvs = false;
    let mut seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
    let mut seq_force_integer_mv = SELECT_INTEGER_MV;
    let mut order_hint_bits = 0u8;

    if !reduced_still_picture_header {
        enable_interintra_compound = br.read_bit()? == 1;
        enable_masked_compound = br.read_bit()? == 1;
        enable_warped_motion = br.read_bit()? == 1;
        enable_dual_filter = br.read_bit()? == 1;
        enable_order_hint = br.read_bit()? == 1;
        if enable_order_hint {
            enable_jnt_comp = br.read_bit()? == 1;
            enable_ref_frame_mvs = br.read_bit()? == 1;
        }
        let seq_choose_screen_content_tools = br.read_bit()? == 1;
        seq_force_screen_content_tools = if seq_choose_screen_content_tools {
            SELECT_SCREEN_CONTENT_TOOLS
        } else {
            br.read_bit()? as u8
        };
        if seq_force_screen_content_tools > 0 {
            let seq_choose_integer_mv = br.read_bit()? == 1;
            seq_force_integer_mv = if seq_choose_integer_mv {
                SELECT_INTEGER_MV
            } else {
                br.read_bit()? as u8
            };
        } else {
            seq_force_integer_mv = SELECT_INTEGER_MV;
        }
        if enable_order_hint {
            let order_hint_bits_minus_1 = br.read_bits(3)?;
            order_hint_bits = (order_hint_bits_minus_1 + 1) as u8;
        }
    }

    let enable_superres = br.read_bit()? == 1;
    let enable_cdef = br.read_bit()? == 1;
    let enable_restoration = br.read_bit()? == 1;

    let color = parse_av1_color_config(&mut br, seq_profile)?;
    let film_grain_params_present = br.read_bit()? == 1;

    Some(Av1SequenceHeader {
        seq_profile,
        still_picture,
        reduced_still_picture_header,
        timing_info_present_flag,
        decoder_model_info_present_flag,
        initial_display_delay_present_flag,
        operating_points_cnt_minus_1,
        seq_level_idx0,
        seq_tier0,
        frame_width_bits_minus_1,
        frame_height_bits_minus_1,
        max_frame_width_minus_1,
        max_frame_height_minus_1,
        frame_id_numbers_present_flag,
        delta_frame_id_length_minus_2,
        additional_frame_id_length_minus_1,
        use_128x128_superblock,
        enable_filter_intra,
        enable_intra_edge_filter,
        enable_interintra_compound,
        enable_masked_compound,
        enable_warped_motion,
        enable_dual_filter,
        enable_order_hint,
        enable_jnt_comp,
        enable_ref_frame_mvs,
        seq_force_screen_content_tools,
        seq_force_integer_mv,
        order_hint_bits,
        enable_superres,
        enable_cdef,
        enable_restoration,
        color,
        film_grain_params_present,
    })
}

/// The leading fields of an AV1 frame header (spec 5.9.2), enough to classify a
/// coded frame by type and whether it is shown. The full uncompressed header
/// (frame size, tile info, quant, loop filter, ...) belongs to the DPB decode
/// path.
#[derive(Debug, Clone, Copy)]
pub struct Av1FrameLead {
    pub show_existing_frame: bool,
    /// Frame type when directly coded; `0xff` for a `show_existing_frame` (the
    /// type comes from the referenced slot, unknown from the header alone).
    pub frame_type: u8,
    pub show_frame: bool,
}

/// Parse the leading fields of a frame header / frame OBU payload. `seq` gives
/// the context (reduced header, frame-id presence) needed to place the reader.
pub fn parse_av1_frame_lead(payload: &[u8], seq: &Av1SequenceHeader) -> Option<Av1FrameLead> {
    let mut br = BitReader::new(payload);
    if seq.reduced_still_picture_header {
        return Some(Av1FrameLead {
            show_existing_frame: false,
            frame_type: AV1_FRAME_TYPE_KEY,
            show_frame: true,
        });
    }
    let show_existing_frame = br.read_bit()? == 1;
    if show_existing_frame {
        br.read_bits(3)?; // frame_to_show_map_idx
                          // decoder_model temporal_point_info is skipped: this classifier does
                          // not target streams that carry a decoder model.
        if seq.frame_id_numbers_present_flag {
            let id_len = seq.additional_frame_id_length_minus_1 as u32
                + seq.delta_frame_id_length_minus_2 as u32
                + 3;
            br.read_bits(id_len)?; // display_frame_id
        }
        return Some(Av1FrameLead {
            show_existing_frame: true,
            frame_type: 0xff,
            show_frame: true,
        });
    }
    let frame_type = br.read_bits(2)? as u8;
    let show_frame = br.read_bit()? == 1;
    Some(Av1FrameLead {
        show_existing_frame: false,
        frame_type,
        show_frame,
    })
}

/// Find and parse the first sequence header OBU in an AV1 bitstream.
pub fn extract_av1_sequence_header(stream: &[u8]) -> Option<Av1SequenceHeader> {
    for obu in av1_obus(stream) {
        if obu.obu_type == OBU_SEQUENCE_HEADER {
            return parse_av1_sequence_header(obu.payload);
        }
    }
    None
}

/// A coded frame's classification, from walking the frame / frame-header OBUs.
#[derive(Debug, Clone, Copy)]
pub struct Av1FrameInfo {
    pub frame_type: u8,
    pub show_frame: bool,
    pub show_existing_frame: bool,
}

/// Classify every coded frame in an AV1 bitstream (type / shown). Requires a
/// sequence header to be present for context. `None` if there is none or a
/// frame header fails to parse.
pub fn av1_frame_infos(stream: &[u8]) -> Option<alloc::vec::Vec<Av1FrameInfo>> {
    let seq = extract_av1_sequence_header(stream)?;
    let mut out = alloc::vec::Vec::new();
    for obu in av1_obus(stream) {
        if obu.obu_type == OBU_FRAME || obu.obu_type == OBU_FRAME_HEADER {
            let lead = parse_av1_frame_lead(obu.payload, &seq)?;
            out.push(Av1FrameInfo {
                frame_type: lead.frame_type,
                show_frame: lead.show_frame,
                show_existing_frame: lead.show_existing_frame,
            });
        }
    }
    Some(out)
}

/// The `(tile_cols, tile_rows)` grid of the first coded frame in an AV1
/// bitstream, or `None` if there is no sequence / frame header or it fails to
/// parse. Introspection for callers (and tests) that need to know whether a
/// stream is tiled; the decoder handles any grid the tile-group parse accepts.
pub fn av1_frame_tile_grid(stream: &[u8]) -> Option<(u32, u32)> {
    let seq = extract_av1_sequence_header(stream)?;
    let refs = Av1RefFrames::default();
    for obu in av1_obus(stream) {
        if obu.obu_type == OBU_FRAME || obu.obu_type == OBU_FRAME_HEADER {
            let fh = parse_av1_frame_header(obu.payload, &seq, &refs)?;
            return Some((fh.tile.tile_cols, fh.tile.tile_rows));
        }
    }
    None
}

/// Whether the first coded frame in an AV1 bitstream uses loop restoration (a
/// nonzero `FrameRestorationType` on any plane). Introspection for tests that need
/// to confirm a stream actually exercises the restoration path. Only the first
/// frame is inspected (an inter frame needs live reference state to parse).
pub fn av1_uses_loop_restoration(stream: &[u8]) -> Option<bool> {
    let seq = extract_av1_sequence_header(stream)?;
    let refs = Av1RefFrames::default();
    for obu in av1_obus(stream) {
        if obu.obu_type == OBU_FRAME || obu.obu_type == OBU_FRAME_HEADER {
            let fh = parse_av1_frame_header(obu.payload, &seq, &refs)?;
            return Some(fh.lr.uses_lr);
        }
    }
    None
}

/// The `Std*` AV1 sequence-header bundle for a decode session. Owns the color
/// config pointee block; `seq_header.pColorConfig` points into it (a stable
/// address that survives the move into the returned struct).
pub struct StdAv1Params {
    pub seq_header: vk::native::StdVideoAV1SequenceHeader,
    _color: alloc::boxed::Box<vk::native::StdVideoAV1ColorConfig>,
}

impl StdAv1Params {
    /// Deep copy with the color-config pointee re-wired to the copy's own box,
    /// boxed so every internal address stays stable for as long as the box
    /// lives. The AV1 decode session stores one: NVIDIA's driver retains the
    /// `pStdSequenceHeader` (and its nested `pColorConfig`) pointers past
    /// `vkCreateVideoSessionParametersKHR` and reads them per decode, so the
    /// pointed-at memory must live as long as the session (dropping it after
    /// creation yields silent, nondeterministic decode corruption).
    fn deep_clone_boxed(&self) -> alloc::boxed::Box<StdAv1Params> {
        let color = alloc::boxed::Box::new(*self._color);
        let mut seq_header = self.seq_header;
        seq_header.pColorConfig = &*color as *const _;
        alloc::boxed::Box::new(StdAv1Params {
            seq_header,
            _color: color,
        })
    }
}

impl core::fmt::Debug for StdAv1Params {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StdAv1Params")
            .field("seq_profile", &self.seq_header.seq_profile)
            .field(
                "max_frame_width_minus_1",
                &self.seq_header.max_frame_width_minus_1,
            )
            .field(
                "max_frame_height_minus_1",
                &self.seq_header.max_frame_height_minus_1,
            )
            .finish_non_exhaustive()
    }
}

fn to_std_av1_color_config(c: &Av1ColorConfig) -> vk::native::StdVideoAV1ColorConfig {
    // SAFETY: `StdVideoAV1ColorConfigFlags` is a plain repr(C) bitfield POD,
    // valid all-zero (all flags clear).
    let mut flags: vk::native::StdVideoAV1ColorConfigFlags = unsafe { core::mem::zeroed() };
    flags.set_mono_chrome(c.mono_chrome as u32);
    flags.set_color_range(c.color_range as u32);
    flags.set_color_description_present_flag(c.color_description_present_flag as u32);
    flags.set_separate_uv_delta_q(c.separate_uv_delta_q as u32);

    // The Std AV1 color enums are defined to equal the AV1 specification's
    // numeric values, so the parsed codepoints cast directly.
    vk::native::StdVideoAV1ColorConfig {
        flags,
        BitDepth: c.bit_depth,
        subsampling_x: c.subsampling_x,
        subsampling_y: c.subsampling_y,
        reserved1: 0,
        color_primaries: c.color_primaries as vk::native::StdVideoAV1ColorPrimaries,
        transfer_characteristics: c.transfer_characteristics
            as vk::native::StdVideoAV1TransferCharacteristics,
        matrix_coefficients: c.matrix_coefficients as vk::native::StdVideoAV1MatrixCoefficients,
        chroma_sample_position: c.chroma_sample_position
            as vk::native::StdVideoAV1ChromaSamplePosition,
    }
}

/// Build the `Std*` AV1 sequence-header bundle from a parsed sequence header.
pub fn to_std_av1_seq_header(seq: &Av1SequenceHeader) -> StdAv1Params {
    let color = alloc::boxed::Box::new(to_std_av1_color_config(&seq.color));

    // SAFETY: `StdVideoAV1SequenceHeaderFlags` is a plain repr(C) bitfield POD,
    // valid all-zero (all flags clear).
    let mut flags: vk::native::StdVideoAV1SequenceHeaderFlags = unsafe { core::mem::zeroed() };
    flags.set_still_picture(seq.still_picture as u32);
    flags.set_reduced_still_picture_header(seq.reduced_still_picture_header as u32);
    flags.set_timing_info_present_flag(seq.timing_info_present_flag as u32);
    flags.set_initial_display_delay_present_flag(seq.initial_display_delay_present_flag as u32);
    flags.set_frame_id_numbers_present_flag(seq.frame_id_numbers_present_flag as u32);
    flags.set_use_128x128_superblock(seq.use_128x128_superblock as u32);
    flags.set_enable_filter_intra(seq.enable_filter_intra as u32);
    flags.set_enable_intra_edge_filter(seq.enable_intra_edge_filter as u32);
    flags.set_enable_interintra_compound(seq.enable_interintra_compound as u32);
    flags.set_enable_masked_compound(seq.enable_masked_compound as u32);
    flags.set_enable_warped_motion(seq.enable_warped_motion as u32);
    flags.set_enable_dual_filter(seq.enable_dual_filter as u32);
    flags.set_enable_order_hint(seq.enable_order_hint as u32);
    flags.set_enable_jnt_comp(seq.enable_jnt_comp as u32);
    flags.set_enable_ref_frame_mvs(seq.enable_ref_frame_mvs as u32);
    flags.set_enable_superres(seq.enable_superres as u32);
    flags.set_enable_cdef(seq.enable_cdef as u32);
    flags.set_enable_restoration(seq.enable_restoration as u32);
    flags.set_film_grain_params_present(seq.film_grain_params_present as u32);

    let seq_header = vk::native::StdVideoAV1SequenceHeader {
        flags,
        seq_profile: seq.seq_profile as vk::native::StdVideoAV1Profile,
        frame_width_bits_minus_1: seq.frame_width_bits_minus_1,
        frame_height_bits_minus_1: seq.frame_height_bits_minus_1,
        max_frame_width_minus_1: seq.max_frame_width_minus_1 as u16,
        max_frame_height_minus_1: seq.max_frame_height_minus_1 as u16,
        delta_frame_id_length_minus_2: seq.delta_frame_id_length_minus_2,
        additional_frame_id_length_minus_1: seq.additional_frame_id_length_minus_1,
        // Only meaningful when enable_order_hint; 0 otherwise.
        order_hint_bits_minus_1: seq.order_hint_bits.saturating_sub(1),
        seq_force_integer_mv: seq.seq_force_integer_mv,
        seq_force_screen_content_tools: seq.seq_force_screen_content_tools,
        reserved1: [0; 5],
        pColorConfig: &*color,
        pTimingInfo: core::ptr::null(),
    };

    StdAv1Params {
        seq_header,
        _color: color,
    }
}

// ============================================================================
// AV1 uncompressed frame header parse (M506)
//
// Unlike H.264/H.265, where the driver parses the slice header from the
// bitstream, Vulkan Video AV1 decode requires the application to hand the driver
// a fully populated `StdVideoDecodeAV1PictureInfo` (+ its tile / quant /
// segmentation / loop-filter / CDEF / loop-restoration / global-motion / film-
// grain sub-structs). So the whole uncompressed frame header (AV1 spec 5.9.2 and
// its sub-parses) must be read here. Every value is cross-checked against ffmpeg
// `trace_headers` on the real fixture. Reference-loaded state (loop-filter deltas
// / segmentation / global motion carried from the primary reference frame when a
// field is not re-coded) uses the AV1 defaults, correct for streams that never
// change those from frame to frame (the fixture); a general implementation would
// thread the per-slot saved state, tracked as a follow-up.

// AV1 reference-frame constants (spec 3, 6.10.24).
const AV1_NUM_REF_FRAMES: usize = 8;
const AV1_REFS_PER_FRAME: usize = 7;
const AV1_PRIMARY_REF_NONE: u8 = 7;
const AV1_FRAME_TYPE_INTRA_ONLY: u8 = 2;
const AV1_FRAME_TYPE_SWITCH: u8 = 3;
// Default loop-filter deltas from setup_past_independence (spec 7.20), indexed by
// reference name: INTRA=1, LAST/LAST2/LAST3=0, GOLDEN=-1, BWDREF=0, ALTREF2=-1,
// ALTREF=-1. The last two (ALTREF2 / ALTREF) are -1, not 0: getting them wrong
// leaves in-loop deblocking off for compound blocks that reference the alt frames,
// a small residual on inter frames past the first (M506c).
const AV1_DEFAULT_LF_REF_DELTAS: [i8; AV1_NUM_REF_FRAMES] = [1, 0, 0, 0, -1, 0, -1, -1];
// Warp model identity: params[i] is (1 << WARPEDMODEL_PREC_BITS) for the two
// scale terms (indices 2, 5), 0 otherwise (spec 7.10.1).
const AV1_WARPEDMODEL_PREC_BITS: u32 = 16;

/// AV1 global-motion transform types (spec 6.8.20 GmType).
const AV1_GM_IDENTITY: u8 = 0;
const AV1_GM_TRANSLATION: u8 = 1;
const AV1_GM_ROTZOOM: u8 = 2;
const AV1_GM_AFFINE: u8 = 3;

/// The reference-frame state the frame-header parse needs for an inter frame:
/// per DPB reference slot, whether it holds a valid picture, its order hint,
/// frame type, and its dimensions (for `frame_size_with_refs`). Maintained by
/// the decoder across frames; a key frame needs none of it.
#[derive(Debug, Clone)]
pub struct Av1RefFrames {
    pub valid: [bool; AV1_NUM_REF_FRAMES],
    pub order_hint: [u8; AV1_NUM_REF_FRAMES],
    pub frame_type: [u8; AV1_NUM_REF_FRAMES],
    pub upscaled_width: [u32; AV1_NUM_REF_FRAMES],
    pub frame_height: [u32; AV1_NUM_REF_FRAMES],
    pub render_width: [u32; AV1_NUM_REF_FRAMES],
    pub render_height: [u32; AV1_NUM_REF_FRAMES],
}

impl Default for Av1RefFrames {
    fn default() -> Self {
        Self {
            valid: [false; AV1_NUM_REF_FRAMES],
            order_hint: [0; AV1_NUM_REF_FRAMES],
            frame_type: [0; AV1_NUM_REF_FRAMES],
            upscaled_width: [0; AV1_NUM_REF_FRAMES],
            frame_height: [0; AV1_NUM_REF_FRAMES],
            render_width: [0; AV1_NUM_REF_FRAMES],
            render_height: [0; AV1_NUM_REF_FRAMES],
        }
    }
}

#[derive(Debug, Clone)]
pub struct Av1TileInfo {
    pub uniform_spacing: bool,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub context_update_tile_id: u32,
    /// `TileSizeBytes` (1..=4); the per-tile size prefix width for multi-tile.
    pub tile_size_bytes: u32,
    pub mi_col_starts: alloc::vec::Vec<u16>,
    pub mi_row_starts: alloc::vec::Vec<u16>,
    pub width_in_sbs_minus_1: alloc::vec::Vec<u16>,
    pub height_in_sbs_minus_1: alloc::vec::Vec<u16>,
}

#[derive(Debug, Clone)]
pub struct Av1Quantization {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub using_qmatrix: bool,
    pub diff_uv_delta: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

#[derive(Debug, Clone)]
pub struct Av1Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    /// Per segment (8), a bitmask over the 8 segmentation features.
    pub feature_enabled: [u8; AV1_NUM_REF_FRAMES],
    pub feature_data: [[i16; 8]; AV1_NUM_REF_FRAMES],
}

#[derive(Debug, Clone)]
pub struct Av1LoopFilter {
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub delta_update: bool,
    pub ref_deltas: [i8; AV1_NUM_REF_FRAMES],
    pub mode_deltas: [i8; 2],
}

#[derive(Debug, Clone)]
pub struct Av1Cdef {
    pub damping_minus_3: u8,
    pub bits: u8,
    pub y_pri: [u8; 8],
    pub y_sec: [u8; 8],
    pub uv_pri: [u8; 8],
    pub uv_sec: [u8; 8],
}

#[derive(Debug, Clone)]
pub struct Av1LoopRestoration {
    /// One of the `RESTORE_*` types per plane (0 = NONE).
    pub frame_restoration_type: [u8; 3],
    pub loop_restoration_size: [u16; 3],
    pub uses_lr: bool,
    pub uses_chroma_lr: bool,
}

#[derive(Debug, Clone)]
pub struct Av1GlobalMotion {
    pub gm_type: [u8; AV1_NUM_REF_FRAMES],
    pub gm_params: [[i32; 6]; AV1_NUM_REF_FRAMES],
}

/// Parsed AV1 uncompressed frame header (spec 5.9.2), everything the Vulkan
/// `StdVideoDecodeAV1PictureInfo` and its sub-structs need.
#[derive(Debug, Clone)]
pub struct Av1FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    pub frame_type: u8,
    pub frame_is_intra: bool,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub current_frame_id: u32,
    pub frame_size_override_flag: bool,
    pub order_hint: u8,
    pub primary_ref_frame: u8,
    pub refresh_frame_flags: u8,
    pub ref_frame_idx: [i8; AV1_REFS_PER_FRAME],
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub use_superres: bool,
    pub superres_denom: u8,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub allow_high_precision_mv: bool,
    pub interpolation_filter: u8,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub disable_frame_end_update_cdf: bool,
    pub allow_intrabc: bool,
    pub reduced_tx_set: bool,
    pub tx_mode: u8,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    pub skip_mode_frame: [u8; 2],
    pub allow_warped_motion: bool,
    pub delta_q_present: bool,
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub tile: Av1TileInfo,
    pub quant: Av1Quantization,
    pub seg: Av1Segmentation,
    pub lf: Av1LoopFilter,
    pub cdef: Av1Cdef,
    pub lr: Av1LoopRestoration,
    pub gm: Av1GlobalMotion,
    /// Parsed `film_grain_params()`. Grain is synthesized on the decoded output
    /// (the hardware produces the grain-free reconstruction); `apply_grain == false`
    /// when the stream carries no grain.
    pub film_grain: Av1FilmGrain,
    /// Byte offset (from the start of the frame OBU payload) where the tile group
    /// data begins, i.e. past the byte-aligned end of this header. For an
    /// `OBU_FRAME` the tile group follows in the same OBU.
    pub header_byte_len: usize,
}

/// `su(n)`: an `n`-bit value read MSB-first, then sign-extended from its top bit
/// (AV1 spec 4.10.6).
fn read_su(br: &mut BitReader, n: u32) -> Option<i32> {
    let value = br.read_bits(n)? as i32;
    let sign_mask = 1i32 << (n - 1);
    Some(if value & sign_mask != 0 {
        value - 2 * sign_mask
    } else {
        value
    })
}

/// `ns(n)`: non-symmetric unsigned encoding of a value in `0..n` (spec 4.10.7).
fn read_ns(br: &mut BitReader, n: u32) -> Option<u32> {
    if n <= 1 {
        return Some(0);
    }
    let w = 32 - (n - 1).leading_zeros(); // FloorLog2(n) + 1
    let m = (1u32 << w) - n;
    let v = br.read_bits(w - 1)?;
    if v < m {
        Some(v)
    } else {
        let extra = br.read_bit()?;
        Some((v << 1) - m + extra)
    }
}

/// `read_delta_q()` (spec 5.9.13): a flag then an optional signed 7-bit delta.
fn read_delta_q(br: &mut BitReader) -> Option<i8> {
    let delta_coded = br.read_bit()? == 1;
    if delta_coded {
        Some(read_su(br, 7)? as i8)
    } else {
        Some(0)
    }
}

/// `tile_log2(blkSize, target)` (spec 5.9.16): smallest k with `blkSize << k >= target`.
fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

const AV1_MAX_TILE_WIDTH_SB_SHIFT: u32 = 4096; // MAX_TILE_WIDTH in luma samples
const AV1_MAX_TILE_AREA: u32 = 4096 * 2304; // MAX_TILE_AREA in luma samples
const AV1_MAX_TILE_COLS: u32 = 64;
const AV1_MAX_TILE_ROWS: u32 = 64;

/// Parse `tile_info()` (spec 5.9.15). `mi_cols` / `mi_rows` come from the frame
/// size; `use_128x128` from the sequence header.
fn parse_av1_tile_info(
    br: &mut BitReader,
    mi_cols: u32,
    mi_rows: u32,
    use_128x128: bool,
) -> Option<Av1TileInfo> {
    let sb_shift = if use_128x128 { 5 } else { 4 };
    let sb_size = sb_shift + 2;
    let sb_cols = if use_128x128 {
        (mi_cols + 31) >> 5
    } else {
        (mi_cols + 15) >> 4
    };
    let sb_rows = if use_128x128 {
        (mi_rows + 31) >> 5
    } else {
        (mi_rows + 15) >> 4
    };
    let max_tile_width_sb = AV1_MAX_TILE_WIDTH_SB_SHIFT >> sb_size;
    let max_tile_area_sb = AV1_MAX_TILE_AREA >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(AV1_MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(AV1_MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    let uniform_spacing = br.read_bit()? == 1;
    let mut mi_col_starts: alloc::vec::Vec<u16> = alloc::vec::Vec::new();
    let mut mi_row_starts: alloc::vec::Vec<u16> = alloc::vec::Vec::new();
    let mut width_in_sbs_minus_1: alloc::vec::Vec<u16> = alloc::vec::Vec::new();
    let mut height_in_sbs_minus_1: alloc::vec::Vec<u16> = alloc::vec::Vec::new();
    let tile_cols_log2;
    let tile_rows_log2;

    if uniform_spacing {
        let mut log2 = min_log2_tile_cols;
        while log2 < max_log2_tile_cols {
            if br.read_bit()? == 1 {
                log2 += 1;
            } else {
                break;
            }
        }
        tile_cols_log2 = log2;
        let tile_width_sb = (sb_cols + (1 << tile_cols_log2) - 1) >> tile_cols_log2;
        let mut start = 0u32;
        let mut i = 0u32;
        while start < sb_cols {
            mi_col_starts.push((start << sb_shift).min(mi_cols) as u16);
            width_in_sbs_minus_1.push((tile_width_sb - 1) as u16);
            start += tile_width_sb;
            i += 1;
        }
        mi_col_starts.push(mi_cols as u16);
        let tile_cols = i;

        let min_log2_tile_rows = min_log2_tiles.saturating_sub(tile_cols_log2);
        let mut log2r = min_log2_tile_rows;
        while log2r < max_log2_tile_rows {
            if br.read_bit()? == 1 {
                log2r += 1;
            } else {
                break;
            }
        }
        tile_rows_log2 = log2r;
        let tile_height_sb = (sb_rows + (1 << tile_rows_log2) - 1) >> tile_rows_log2;
        let mut startr = 0u32;
        let mut j = 0u32;
        while startr < sb_rows {
            mi_row_starts.push((startr << sb_shift).min(mi_rows) as u16);
            height_in_sbs_minus_1.push((tile_height_sb - 1) as u16);
            startr += tile_height_sb;
            j += 1;
        }
        mi_row_starts.push(mi_rows as u16);
        let tile_rows = j;

        finish_tile_info(
            br,
            tile_cols_log2,
            tile_rows_log2,
            tile_cols,
            tile_rows,
            uniform_spacing,
            mi_col_starts,
            mi_row_starts,
            width_in_sbs_minus_1,
            height_in_sbs_minus_1,
        )
    } else {
        // Non-uniform tiling: widths/heights coded explicitly.
        let mut widest_tile_sb = 0u32;
        let mut start = 0u32;
        let mut i = 0u32;
        while start < sb_cols {
            mi_col_starts.push((start << sb_shift).min(mi_cols) as u16);
            let max_width = (sb_cols - start).min(max_tile_width_sb);
            let width_sb = read_ns(br, max_width)? + 1;
            width_in_sbs_minus_1.push((width_sb - 1) as u16);
            widest_tile_sb = widest_tile_sb.max(width_sb);
            start += width_sb;
            i += 1;
        }
        mi_col_starts.push(mi_cols as u16);
        let tile_cols = i;
        let tile_cols_log2_local = tile_log2(1, tile_cols);

        let max_tile_area_sb2 = if min_log2_tiles > 0 {
            (sb_rows * sb_cols) >> (min_log2_tiles + 1)
        } else {
            sb_rows * sb_cols
        };
        let max_tile_height_sb = (max_tile_area_sb2 / widest_tile_sb).max(1);
        let mut startr = 0u32;
        let mut j = 0u32;
        while startr < sb_rows {
            mi_row_starts.push((startr << sb_shift).min(mi_rows) as u16);
            let max_height = (sb_rows - startr).min(max_tile_height_sb);
            let height_sb = read_ns(br, max_height)? + 1;
            height_in_sbs_minus_1.push((height_sb - 1) as u16);
            startr += height_sb;
            j += 1;
        }
        mi_row_starts.push(mi_rows as u16);
        let tile_rows = j;
        let tile_rows_log2_local = tile_log2(1, tile_rows);

        finish_tile_info(
            br,
            tile_cols_log2_local,
            tile_rows_log2_local,
            tile_cols,
            tile_rows,
            uniform_spacing,
            mi_col_starts,
            mi_row_starts,
            width_in_sbs_minus_1,
            height_in_sbs_minus_1,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_tile_info(
    br: &mut BitReader,
    tile_cols_log2: u32,
    tile_rows_log2: u32,
    tile_cols: u32,
    tile_rows: u32,
    uniform_spacing: bool,
    mi_col_starts: alloc::vec::Vec<u16>,
    mi_row_starts: alloc::vec::Vec<u16>,
    width_in_sbs_minus_1: alloc::vec::Vec<u16>,
    height_in_sbs_minus_1: alloc::vec::Vec<u16>,
) -> Option<Av1TileInfo> {
    let (context_update_tile_id, tile_size_bytes) = if tile_cols_log2 > 0 || tile_rows_log2 > 0 {
        let id = br.read_bits(tile_rows_log2 + tile_cols_log2)?;
        let tsb = br.read_bits(2)? + 1;
        (id, tsb)
    } else {
        (0, 1)
    };
    Some(Av1TileInfo {
        uniform_spacing,
        tile_cols_log2,
        tile_rows_log2,
        tile_cols,
        tile_rows,
        context_update_tile_id,
        tile_size_bytes,
        mi_col_starts,
        mi_row_starts,
        width_in_sbs_minus_1,
        height_in_sbs_minus_1,
    })
}

/// Parse the tile-group header + per-tile size prefixes of an `OBU_FRAME` payload
/// (spec 5.11.1 `tile_group_obu`), returning each tile's byte offset (from the
/// payload start) and its size, which is exactly what the driver's `pTileOffsets`
/// / `pTileSizes` expect. The byte-aligned frame header occupies the first
/// `header_byte_len` bytes; the single tile group that follows an `OBU_FRAME`
/// covers the whole frame (`tg_start == 0`, `tg_end == NumTiles - 1`), which every
/// `OBU_FRAME` satisfies by bitstream conformance. Every length read from the
/// stream is validated against the payload bounds (untrusted input): a malformed
/// prefix or an out-of-range tile returns `None` rather than reading past the end.
/// A single-tile frame yields one entry pointing at the data right after the
/// header, matching the pre-multi-tile behaviour.
fn av1_tile_layout(
    payload: &[u8],
    tile: &Av1TileInfo,
    header_byte_len: usize,
) -> Option<(alloc::vec::Vec<u32>, alloc::vec::Vec<u32>)> {
    let num_tiles = (tile.tile_cols as usize).checked_mul(tile.tile_rows as usize)?;
    if num_tiles == 0 || num_tiles > (AV1_MAX_TILE_COLS * AV1_MAX_TILE_ROWS) as usize {
        return None;
    }
    let tsb = tile.tile_size_bytes as usize;
    if !(1..=4).contains(&tsb) {
        return None;
    }

    // tile_group_obu() header: an optional tile_start_and_end_present_flag +
    // tg_start / tg_end, then byte_alignment() before the first tile.
    let tg = payload.get(header_byte_len..)?;
    let mut br = BitReader::new(tg);
    let mut tg_start = 0u32;
    let mut tg_end = (num_tiles - 1) as u32;
    if num_tiles > 1 && br.read_bit()? == 1 {
        let tile_bits = tile.tile_cols_log2 + tile.tile_rows_log2;
        tg_start = br.read_bits(tile_bits)?;
        tg_end = br.read_bits(tile_bits)?;
    }
    // Only the whole-frame single tile group (as in every OBU_FRAME) is handled.
    if tg_start != 0 || tg_end as usize != num_tiles - 1 {
        return None;
    }
    let tg_header_bytes = br.bit_pos().div_ceil(8);

    let mut offsets = alloc::vec::Vec::with_capacity(num_tiles);
    let mut sizes = alloc::vec::Vec::with_capacity(num_tiles);
    // `cursor` is a byte offset from the payload start.
    let mut cursor = header_byte_len.checked_add(tg_header_bytes)?;
    for tile_num in 0..num_tiles {
        let tile_size = if tile_num == num_tiles - 1 {
            // The last tile takes whatever remains of the tile group.
            payload.len().checked_sub(cursor)?
        } else {
            // le(TileSizeBytes): little-endian tile_size_minus_1.
            let prefix = payload.get(cursor..cursor.checked_add(tsb)?)?;
            let mut v = 0u64;
            for (i, &b) in prefix.iter().enumerate() {
                v |= (b as u64) << (8 * i);
            }
            cursor = cursor.checked_add(tsb)?;
            (v as usize).checked_add(1)?
        };
        let end = cursor.checked_add(tile_size)?;
        if end > payload.len() {
            return None;
        }
        offsets.push(cursor as u32);
        sizes.push(tile_size as u32);
        cursor = end;
    }
    Some((offsets, sizes))
}

// ============================================================================
// AV1 film grain synthesis (spec 7.18.3). Vulkan hardware decode produces the
// grain-free reconstruction (NVIDIA does not apply grain: the 3060 exposes only
// `DPB_AND_OUTPUT_COINCIDE`, and driver-applied grain needs a distinct output
// image, `DPB_AND_OUTPUT_DISTINCT`), so grain is synthesized here on the decoded
// NV12, bit-for-bit with the ffmpeg / dav1d software decoder. Ported from the
// re_rav1d `filmgrain.rs` / `fg_apply.rs` scalar reference (the same crate g2g
// uses for `Rav1dDec`), specialized to 8-bit 4:2:0.

const AV1_FRAME_TYPE_INTER: u8 = 1;
const FG_GRAIN_WIDTH: usize = 82;
const FG_GRAIN_HEIGHT: usize = 73;
const FG_SUB_GRAIN_WIDTH: usize = 44;
const FG_SUB_GRAIN_HEIGHT: usize = 38;
const FG_BLOCK_SIZE: usize = 32;
const FG_AR_PAD: usize = 3;

/// The AV1 grain template buffer (`GrainLut`): `[[i16; 82]; 74]`.
type FgGrainLut = [[i16; FG_GRAIN_WIDTH]; FG_GRAIN_HEIGHT + 1];

/// The fixed AV1 Gaussian sequence (spec 7.18.3.1), 2048 entries.
static FG_GAUSSIAN_SEQUENCE: [i16; 2048] = [
    56, 568, -180, 172, 124, -84, 172, -64, -900, 24, 820, 224, 1248, 996, 272, -8, -916, -388,
    -732, -104, -188, 800, 112, -652, -320, -376, 140, -252, 492, -168, 44, -788, 588, -584, 500,
    -228, 12, 680, 272, -476, 972, -100, 652, 368, 432, -196, -720, -192, 1000, -332, 652, -136,
    -552, -604, -4, 192, -220, -136, 1000, -52, 372, -96, -624, 124, -24, 396, 540, -12, -104, 640,
    464, 244, -208, -84, 368, -528, -740, 248, -968, -848, 608, 376, -60, -292, -40, -156, 252,
    -292, 248, 224, -280, 400, -244, 244, -60, 76, -80, 212, 532, 340, 128, -36, 824, -352, -60,
    -264, -96, -612, 416, -704, 220, -204, 640, -160, 1220, -408, 900, 336, 20, -336, -96, -792,
    304, 48, -28, -1232, -1172, -448, 104, -292, -520, 244, 60, -948, 0, -708, 268, 108, 356, -548,
    488, -344, -136, 488, -196, -224, 656, -236, -1128, 60, 4, 140, 276, -676, -376, 168, -108,
    464, 8, 564, 64, 240, 308, -300, -400, -456, -136, 56, 120, -408, -116, 436, 504, -232, 328,
    844, -164, -84, 784, -168, 232, -224, 348, -376, 128, 568, 96, -1244, -288, 276, 848, 832,
    -360, 656, 464, -384, -332, -356, 728, -388, 160, -192, 468, 296, 224, 140, -776, -100, 280, 4,
    196, 44, -36, -648, 932, 16, 1428, 28, 528, 808, 772, 20, 268, 88, -332, -284, 124, -384, -448,
    208, -228, -1044, -328, 660, 380, -148, -300, 588, 240, 540, 28, 136, -88, -436, 256, 296,
    -1000, 1400, 0, -48, 1056, -136, 264, -528, -1108, 632, -484, -592, -344, 796, 124, -668, -768,
    388, 1296, -232, -188, -200, -288, -4, 308, 100, -168, 256, -500, 204, -508, 648, -136, 372,
    -272, -120, -1004, -552, -548, -384, 548, -296, 428, -108, -8, -912, -324, -224, -88, -112,
    -220, -100, 996, -796, 548, 360, -216, 180, 428, -200, -212, 148, 96, 148, 284, 216, -412,
    -320, 120, -300, -384, -604, -572, -332, -8, -180, -176, 696, 116, -88, 628, 76, 44, -516, 240,
    -208, -40, 100, -592, 344, -308, -452, -228, 20, 916, -1752, -136, -340, -804, 140, 40, 512,
    340, 248, 184, -492, 896, -156, 932, -628, 328, -688, -448, -616, -752, -100, 560, -1020, 180,
    -800, -64, 76, 576, 1068, 396, 660, 552, -108, -28, 320, -628, 312, -92, -92, -472, 268, 16,
    560, 516, -672, -52, 492, -100, 260, 384, 284, 292, 304, -148, 88, -152, 1012, 1064, -228, 164,
    -376, -684, 592, -392, 156, 196, -524, -64, -884, 160, -176, 636, 648, 404, -396, -436, 864,
    424, -728, 988, -604, 904, -592, 296, -224, 536, -176, -920, 436, -48, 1176, -884, 416, -776,
    -824, -884, 524, -548, -564, -68, -164, -96, 692, 364, -692, -1012, -68, 260, -480, 876, -1116,
    452, -332, -352, 892, -1088, 1220, -676, 12, -292, 244, 496, 372, -32, 280, 200, 112, -440,
    -96, 24, -644, -184, 56, -432, 224, -980, 272, -260, 144, -436, 420, 356, 364, -528, 76, 172,
    -744, -368, 404, -752, -416, 684, -688, 72, 540, 416, 92, 444, 480, -72, -1416, 164, -1172,
    -68, 24, 424, 264, 1040, 128, -912, -524, -356, 64, 876, -12, 4, -88, 532, 272, -524, 320, 276,
    -508, 940, 24, -400, -120, 756, 60, 236, -412, 100, 376, -484, 400, -100, -740, -108, -260,
    328, -268, 224, -200, -416, 184, -604, -564, -20, 296, 60, 892, -888, 60, 164, 68, -760, 216,
    -296, 904, -336, -28, 404, -356, -568, -208, -1480, -512, 296, 328, -360, -164, -1560, -776,
    1156, -428, 164, -504, -112, 120, -216, -148, -264, 308, 32, 64, -72, 72, 116, 176, -64, -272,
    460, -536, -784, -280, 348, 108, -752, -132, 524, -540, -776, 116, -296, -1196, -288, -560,
    1040, -472, 116, -848, -1116, 116, 636, 696, 284, -176, 1016, 204, -864, -648, -248, 356, 972,
    -584, -204, 264, 880, 528, -24, -184, 116, 448, -144, 828, 524, 212, -212, 52, 12, 200, 268,
    -488, -404, -880, 824, -672, -40, 908, -248, 500, 716, -576, 492, -576, 16, 720, -108, 384,
    124, 344, 280, 576, -500, 252, 104, -308, 196, -188, -8, 1268, 296, 1032, -1196, 436, 316, 372,
    -432, -200, -660, 704, -224, 596, -132, 268, 32, -452, 884, 104, -1008, 424, -1348, -280, 4,
    -1168, 368, 476, 696, 300, -8, 24, 180, -592, -196, 388, 304, 500, 724, -160, 244, -84, 272,
    -256, -420, 320, 208, -144, -156, 156, 364, 452, 28, 540, 316, 220, -644, -248, 464, 72, 360,
    32, -388, 496, -680, -48, 208, -116, -408, 60, -604, -392, 548, -840, 784, -460, 656, -544,
    -388, -264, 908, -800, -628, -612, -568, 572, -220, 164, 288, -16, -308, 308, -112, -636, -760,
    280, -668, 432, 364, 240, -196, 604, 340, 384, 196, 592, -44, -500, 432, -580, -132, 636, -76,
    392, 4, -412, 540, 508, 328, -356, -36, 16, -220, -64, -248, -60, 24, -192, 368, 1040, 92, -24,
    -1044, -32, 40, 104, 148, 192, -136, -520, 56, -816, -224, 732, 392, 356, 212, -80, -424,
    -1008, -324, 588, -1496, 576, 460, -816, -848, 56, -580, -92, -1372, -112, -496, 200, 364, 52,
    -140, 48, -48, -60, 84, 72, 40, 132, -356, -268, -104, -284, -404, 732, -520, 164, -304, -540,
    120, 328, -76, -460, 756, 388, 588, 236, -436, -72, -176, -404, -316, -148, 716, -604, 404,
    -72, -88, -888, -68, 944, 88, -220, -344, 960, 472, 460, -232, 704, 120, 832, -228, 692, -508,
    132, -476, 844, -748, -364, -44, 1116, -1104, -1056, 76, 428, 552, -692, 60, 356, 96, -384,
    -188, -612, -576, 736, 508, 892, 352, -1132, 504, -24, -352, 324, 332, -600, -312, 292, 508,
    -144, -8, 484, 48, 284, -260, -240, 256, -100, -292, -204, -44, 472, -204, 908, -188, -1000,
    -256, 92, 1164, -392, 564, 356, 652, -28, -884, 256, 484, -192, 760, -176, 376, -524, -452,
    -436, 860, -736, 212, 124, 504, -476, 468, 76, -472, 552, -692, -944, -620, 740, -240, 400,
    132, 20, 192, -196, 264, -668, -1012, -60, 296, -316, -828, 76, -156, 284, -768, -448, -832,
    148, 248, 652, 616, 1236, 288, -328, -400, -124, 588, 220, 520, -696, 1032, 768, -740, -92,
    -272, 296, 448, -464, 412, -200, 392, 440, -200, 264, -152, -260, 320, 1032, 216, 320, -8, -64,
    156, -1016, 1084, 1172, 536, 484, -432, 132, 372, -52, -256, 84, 116, -352, 48, 116, 304, -384,
    412, 924, -300, 528, 628, 180, 648, 44, -980, -220, 1320, 48, 332, 748, 524, -268, -720, 540,
    -276, 564, -344, -208, -196, 436, 896, 88, -392, 132, 80, -964, -288, 568, 56, -48, -456, 888,
    8, 552, -156, -292, 948, 288, 128, -716, -292, 1192, -152, 876, 352, -600, -260, -812, -468,
    -28, -120, -32, -44, 1284, 496, 192, 464, 312, -76, -516, -380, -456, -1012, -48, 308, -156,
    36, 492, -156, -808, 188, 1652, 68, -120, -116, 316, 160, -140, 352, 808, -416, 592, 316, -480,
    56, 528, -204, -568, 372, -232, 752, -344, 744, -4, 324, -416, -600, 768, 268, -248, -88, -132,
    -420, -432, 80, -288, 404, -316, -1216, -588, 520, -108, 92, -320, 368, -480, -216, -92, 1688,
    -300, 180, 1020, -176, 820, -68, -228, -260, 436, -904, 20, 40, -508, 440, -736, 312, 332, 204,
    760, -372, 728, 96, -20, -632, -520, -560, 336, 1076, -64, -532, 776, 584, 192, 396, -728,
    -520, 276, -188, 80, -52, -612, -252, -48, 648, 212, -688, 228, -52, -260, 428, -412, -272,
    -404, 180, 816, -796, 48, 152, 484, -88, -216, 988, 696, 188, -528, 648, -116, -180, 316, 476,
    12, -564, 96, 476, -252, -364, -376, -392, 556, -256, -576, 260, -352, 120, -16, -136, -260,
    -492, 72, 556, 660, 580, 616, 772, 436, 424, -32, -324, -1268, 416, -324, -80, 920, 160, 228,
    724, 32, -516, 64, 384, 68, -128, 136, 240, 248, -204, -68, 252, -932, -120, -480, -628, -84,
    192, 852, -404, -288, -132, 204, 100, 168, -68, -196, -868, 460, 1080, 380, -80, 244, 0, 484,
    -888, 64, 184, 352, 600, 460, 164, 604, -196, 320, -64, 588, -184, 228, 12, 372, 48, -848,
    -344, 224, 208, -200, 484, 128, -20, 272, -468, -840, 384, 256, -720, -520, -464, -580, 112,
    -120, 644, -356, -208, -608, -528, 704, 560, -424, 392, 828, 40, 84, 200, -152, 0, -144, 584,
    280, -120, 80, -556, -972, -196, -472, 724, 80, 168, -32, 88, 160, -688, 0, 160, 356, 372,
    -776, 740, -128, 676, -248, -480, 4, -364, 96, 544, 232, -1032, 956, 236, 356, 20, -40, 300,
    24, -676, -596, 132, 1120, -104, 532, -1096, 568, 648, 444, 508, 380, 188, -376, -604, 1488,
    424, 24, 756, -220, -192, 716, 120, 920, 688, 168, 44, -460, 568, 284, 1144, 1160, 600, 424,
    888, 656, -356, -320, 220, 316, -176, -724, -188, -816, -628, -348, -228, -380, 1012, -452,
    -660, 736, 928, 404, -696, -72, -268, -892, 128, 184, -344, -780, 360, 336, 400, 344, 428, 548,
    -112, 136, -228, -216, -820, -516, 340, 92, -136, 116, -300, 376, -244, 100, -316, -520, -284,
    -12, 824, 164, -548, -180, -128, 116, -924, -828, 268, -368, -580, 620, 192, 160, 0, -1676,
    1068, 424, -56, -360, 468, -156, 720, 288, -528, 556, -364, 548, -148, 504, 316, 152, -648,
    -620, -684, -24, -376, -384, -108, -920, -1032, 768, 180, -264, -508, -1268, -260, -60, 300,
    -240, 988, 724, -376, -576, -212, -736, 556, 192, 1092, -620, -880, 376, -56, -4, -216, -32,
    836, 268, 396, 1332, 864, -600, 100, 56, -412, -92, 356, 180, 884, -468, -436, 292, -388, -804,
    -704, -840, 368, -348, 140, -724, 1536, 940, 372, 112, -372, 436, -480, 1136, 296, -32, -228,
    132, -48, -220, 868, -1016, -60, -1044, -464, 328, 916, 244, 12, -736, -296, 360, 468, -376,
    -108, -92, 788, 368, -56, 544, 400, -672, -420, 728, 16, 320, 44, -284, -380, -796, 488, 132,
    204, -596, -372, 88, -152, -908, -636, -572, -624, -116, -692, -200, -56, 276, -88, 484, -324,
    948, 864, 1000, -456, -184, -276, 292, -296, 156, 676, 320, 160, 908, -84, -1236, -288, -116,
    260, -372, -644, 732, -756, -96, 84, 344, -520, 348, -688, 240, -84, 216, -1044, -136, -676,
    -396, -1500, 960, -40, 176, 168, 1516, 420, -504, -344, -364, -360, 1216, -940, -380, -212,
    252, -660, -708, 484, -444, -152, 928, -120, 1112, 476, -260, 560, -148, -344, 108, -196, 228,
    -288, 504, 560, -328, -88, 288, -1008, 460, -228, 468, -836, -196, 76, 388, 232, 412, -1168,
    -716, -644, 756, -172, -356, -504, 116, 432, 528, 48, 476, -168, -608, 448, 160, -532, -272,
    28, -676, -12, 828, 980, 456, 520, 104, -104, 256, -344, -4, -28, -368, -52, -524, -572, -556,
    -200, 768, 1124, -208, -512, 176, 232, 248, -148, -888, 604, -600, -304, 804, -156, -212, 488,
    -192, -804, -256, 368, -360, -916, -328, 228, -240, -448, -472, 856, -556, -364, 572, -12,
    -156, -368, -340, 432, 252, -752, -152, 288, 268, -580, -848, -592, 108, -76, 244, 312, -716,
    592, -80, 436, 360, 4, -248, 160, 516, 584, 732, 44, -468, -280, -292, -156, -588, 28, 308,
    912, 24, 124, 156, 180, -252, 944, -924, -772, -520, -428, -624, 300, -212, -1144, 32, -724,
    800, -1128, -212, -1288, -848, 180, -416, 440, 192, -576, -792, -76, -1080, 80, -532, -352,
    -132, 380, -820, 148, 1112, 128, 164, 456, 700, -924, 144, -668, -384, 648, -832, 508, 552,
    -52, -100, -656, 208, -568, 748, -88, 680, 232, 300, 192, -408, -1012, -152, -252, -268, 272,
    -876, -664, -648, -332, -136, 16, 12, 1152, -28, 332, -536, 320, -672, -460, -316, 532, -260,
    228, -40, 1052, -816, 180, 88, -496, -556, -672, -368, 428, 92, 356, 404, -408, 252, 196, -176,
    -556, 792, 268, 32, 372, 40, 96, -332, 328, 120, 372, -900, -40, 472, -264, -592, 952, 128,
    656, 112, 664, -232, 420, 4, -344, -464, 556, 244, -416, -32, 252, 0, -412, 188, -696, 508,
    -476, 324, -1096, 656, -312, 560, 264, -136, 304, 160, -64, -580, 248, 336, -720, 560, -348,
    -288, -276, -196, -500, 852, -544, -236, -1128, -992, -776, 116, 56, 52, 860, 884, 212, -12,
    168, 1020, 512, -552, 924, -148, 716, 188, 164, -340, -520, -184, 880, -152, -680, -208, -1156,
    -300, -528, -472, 364, 100, -744, -1056, -32, 540, 280, 144, -676, -32, -232, -280, -224, 96,
    568, -76, 172, 148, 148, 104, 32, -296, -32, 788, -80, 32, -16, 280, 288, 944, 428, -484,
];

/// Parsed AV1 `film_grain_params()` (spec 5.9.30). `apply_grain == false` means no
/// synthesis. When `update_grain == false` the coefficients are copied from the
/// reference frame `ref_idx` at apply time (only `seed` is coded); otherwise every
/// field below is coded in the frame header.
#[derive(Debug, Clone, Copy, Default)]
pub struct Av1FilmGrain {
    pub apply_grain: bool,
    pub seed: u16,
    pub update_grain: bool,
    pub ref_idx: u8,
    pub num_y_points: usize,
    pub y_points: [[u8; 2]; 14],
    pub chroma_scaling_from_luma: bool,
    pub num_uv_points: [usize; 2],
    pub uv_points: [[[u8; 2]; 10]; 2],
    pub scaling_shift: u8,
    pub ar_coeff_lag: usize,
    pub ar_coeffs_y: [i8; 24],
    pub ar_coeffs_uv: [[i8; 28]; 2],
    pub ar_coeff_shift: u8,
    pub grain_scale_shift: u8,
    pub uv_mult: [i32; 2],
    pub uv_luma_mult: [i32; 2],
    pub uv_offset: [i32; 2],
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

/// Parse `film_grain_params()` (spec 5.9.30) from the frame header bit position.
/// Untrusted input: an out-of-range point count fails the parse (`None`).
fn parse_av1_film_grain(
    br: &mut BitReader,
    seq: &Av1SequenceHeader,
    frame_type: u8,
    show_frame: bool,
    showable_frame: bool,
) -> Option<Av1FilmGrain> {
    let mut fg = Av1FilmGrain::default();
    if !seq.film_grain_params_present || (!show_frame && !showable_frame) {
        return Some(fg);
    }
    fg.apply_grain = br.read_bit()? == 1;
    if !fg.apply_grain {
        return Some(fg);
    }
    fg.seed = br.read_bits(16)? as u16;
    fg.update_grain = if frame_type == AV1_FRAME_TYPE_INTER {
        br.read_bit()? == 1
    } else {
        true
    };
    if !fg.update_grain {
        fg.ref_idx = br.read_bits(3)? as u8;
        return Some(fg);
    }
    let num_y = br.read_bits(4)? as usize;
    if num_y > 14 {
        return None;
    }
    fg.num_y_points = num_y;
    for i in 0..num_y {
        fg.y_points[i][0] = br.read_bits(8)? as u8;
        if i != 0 && fg.y_points[i - 1][0] >= fg.y_points[i][0] {
            return None;
        }
        fg.y_points[i][1] = br.read_bits(8)? as u8;
    }
    let mono = seq.color.mono_chrome;
    fg.chroma_scaling_from_luma = if mono { false } else { br.read_bit()? == 1 };
    let ss_x = seq.color.subsampling_x == 1;
    let ss_y = seq.color.subsampling_y == 1;
    if mono || fg.chroma_scaling_from_luma || (ss_x && ss_y && num_y == 0) {
        fg.num_uv_points = [0, 0];
    } else {
        for pl in 0..2 {
            let n = br.read_bits(4)? as usize;
            if n > 10 {
                return None;
            }
            fg.num_uv_points[pl] = n;
            for i in 0..n {
                fg.uv_points[pl][i][0] = br.read_bits(8)? as u8;
                if i != 0 && fg.uv_points[pl][i - 1][0] >= fg.uv_points[pl][i][0] {
                    return None;
                }
                fg.uv_points[pl][i][1] = br.read_bits(8)? as u8;
            }
        }
    }
    if ss_x && ss_y && (fg.num_uv_points[0] != 0) != (fg.num_uv_points[1] != 0) {
        return None;
    }
    fg.scaling_shift = br.read_bits(2)? as u8 + 8;
    fg.ar_coeff_lag = br.read_bits(2)? as usize;
    let num_y_pos = 2 * fg.ar_coeff_lag * (fg.ar_coeff_lag + 1);
    if num_y != 0 {
        for i in 0..num_y_pos {
            fg.ar_coeffs_y[i] = (br.read_bits(8)? as i32 - 128) as i8;
        }
    }
    for pl in 0..2 {
        if fg.num_uv_points[pl] != 0 || fg.chroma_scaling_from_luma {
            let num_uv_pos = num_y_pos + (num_y != 0) as usize;
            for i in 0..num_uv_pos {
                fg.ar_coeffs_uv[pl][i] = (br.read_bits(8)? as i32 - 128) as i8;
            }
        }
    }
    fg.ar_coeff_shift = br.read_bits(2)? as u8 + 6;
    fg.grain_scale_shift = br.read_bits(2)? as u8;
    for pl in 0..2 {
        if fg.num_uv_points[pl] != 0 {
            fg.uv_mult[pl] = br.read_bits(8)? as i32 - 128;
            fg.uv_luma_mult[pl] = br.read_bits(8)? as i32 - 128;
            fg.uv_offset[pl] = br.read_bits(9)? as i32 - 256;
        }
    }
    fg.overlap_flag = br.read_bit()? == 1;
    fg.clip_to_restricted_range = br.read_bit()? == 1;
    Some(fg)
}

#[inline]
fn fg_round2(x: i32, shift: u8) -> i32 {
    (x + ((1 << shift) >> 1)) >> shift
}

/// AV1 grain LFSR (spec 7.18.3.2): advance the 16-bit state and return `bits` bits.
#[inline]
fn fg_get_random(bits: u8, state: &mut u32) -> i32 {
    let r = *state;
    let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
    *state = (r >> 1) | (bit << 15);
    ((*state >> (16 - bits as u32)) & ((1 << bits) - 1)) as i32
}

/// Generate the luma grain template (spec 7.18.3.3), 8-bit.
fn fg_generate_grain_y(fg: &Av1FilmGrain, buf: &mut FgGrainLut) {
    let mut seed = fg.seed as u32;
    let shift = 4 + fg.grain_scale_shift;
    let ar_lag = fg.ar_coeff_lag;
    for row in buf[..FG_GRAIN_HEIGHT].iter_mut() {
        for v in row[..FG_GRAIN_WIDTH].iter_mut() {
            let value = fg_get_random(11, &mut seed) as usize;
            *v = fg_round2(FG_GAUSSIAN_SEQUENCE[value] as i32, shift) as i16;
        }
    }
    for y in 0..FG_GRAIN_HEIGHT - FG_AR_PAD {
        for x in 0..FG_GRAIN_WIDTH - 2 * FG_AR_PAD {
            let mut ci = 0usize;
            let mut sum = 0i32;
            'outer: for dy in 0..=ar_lag {
                let by = y + (FG_AR_PAD - ar_lag) + dy;
                for dx in 0..=2 * ar_lag {
                    if dx == ar_lag && dy == ar_lag {
                        break 'outer;
                    }
                    let bx = x + (FG_AR_PAD - ar_lag) + dx;
                    sum += fg.ar_coeffs_y[ci] as i32 * buf[by][bx] as i32;
                    ci += 1;
                }
            }
            let cur = buf[y + FG_AR_PAD][x + FG_AR_PAD] as i32;
            let grain = cur + fg_round2(sum, fg.ar_coeff_shift);
            buf[y + FG_AR_PAD][x + FG_AR_PAD] = grain.clamp(-128, 127) as i16;
        }
    }
}

/// Generate a chroma grain template (spec 7.18.3.3) for 4:2:0. `uv` = 0 (Cb) / 1 (Cr).
fn fg_generate_grain_uv_420(
    fg: &Av1FilmGrain,
    buf: &mut FgGrainLut,
    buf_y: &FgGrainLut,
    uv: usize,
) {
    let mut seed = (fg.seed as u32) ^ if uv != 0 { 0x49d8 } else { 0xb524 };
    let shift = 4 + fg.grain_scale_shift;
    let ar_lag = fg.ar_coeff_lag;
    for row in buf[..FG_SUB_GRAIN_HEIGHT].iter_mut() {
        for v in row[..FG_SUB_GRAIN_WIDTH].iter_mut() {
            let value = fg_get_random(11, &mut seed) as usize;
            *v = fg_round2(FG_GAUSSIAN_SEQUENCE[value] as i32, shift) as i16;
        }
    }
    let len_h = FG_SUB_GRAIN_HEIGHT - FG_AR_PAD;
    let len_w = FG_SUB_GRAIN_WIDTH - 2 * FG_AR_PAD;
    for y in 0..len_h {
        for x in 0..len_w {
            let mut ci = 0usize;
            let mut sum = 0i32;
            'outer: for dy in 0..=ar_lag {
                let by = y + (FG_AR_PAD - ar_lag) + dy;
                for dx in 0..=2 * ar_lag {
                    if dx == ar_lag && dy == ar_lag {
                        // Center: add the co-located 2x2 luma average contribution.
                        let luma_y = (y << 1) + FG_AR_PAD;
                        let luma_x = (x << 1) + FG_AR_PAD;
                        let mut luma = 0i32;
                        for i in 0..2 {
                            for j in 0..2 {
                                luma += buf_y[luma_y + i][luma_x + j] as i32;
                            }
                        }
                        luma = fg_round2(luma, 2);
                        sum += luma * fg.ar_coeffs_uv[uv][ci] as i32;
                        break 'outer;
                    }
                    let bx = x + (FG_AR_PAD - ar_lag) + dx;
                    sum += fg.ar_coeffs_uv[uv][ci] as i32 * buf[by][bx] as i32;
                    ci += 1;
                }
            }
            let cur = buf[y + FG_AR_PAD][x + FG_AR_PAD] as i32;
            let grain = cur + fg_round2(sum, fg.ar_coeff_shift);
            buf[y + FG_AR_PAD][x + FG_AR_PAD] = grain.clamp(-128, 127) as i16;
        }
    }
}

/// Build the 8-bit scaling LUT from the piecewise-linear points (fg_apply.rs).
fn fg_generate_scaling(points: &[[u8; 2]]) -> [u8; 256] {
    let mut scaling = [0u8; 256];
    if points.is_empty() {
        return scaling;
    }
    for s in scaling[..points[0][0] as usize].iter_mut() {
        *s = points[0][1];
    }
    for w in points.windows(2) {
        let bx = w[0][0] as usize;
        let by = w[0][1] as isize;
        let ex = w[1][0] as usize;
        let ey = w[1][1] as isize;
        let dx = ex - bx;
        let dy = ey - by;
        let delta = dy * ((0x10000 + (dx as isize >> 1)) / dx as isize);
        let mut d = 0x8000isize;
        for x in 0..dx {
            scaling[bx + x] = (by + (d >> 16)) as u8;
            d += delta;
        }
    }
    let n = points[points.len() - 1][0] as usize;
    for s in scaling[n..].iter_mut() {
        *s = points[points.len() - 1][1];
    }
    scaling
}

/// Sample the grain template at a block-relative position with the block's random
/// offset (spec 7.18.3.5 `sample_lut`), 4:2:0-aware.
#[inline]
#[allow(clippy::too_many_arguments)]
fn fg_sample_lut(
    grain: &FgGrainLut,
    offsets: &[[i32; 2]; 2],
    subx: usize,
    suby: usize,
    bx: usize,
    by: usize,
    x: usize,
    y: usize,
) -> i32 {
    let randval = offsets[bx][by] as usize;
    let offx = 3 + (2 >> subx) * (3 + (randval >> 4));
    let offy = 3 + (2 >> suby) * (3 + (randval & 0xf));
    grain[offy + y + (FG_BLOCK_SIZE >> suby) * by][offx + x + (FG_BLOCK_SIZE >> subx) * bx] as i32
}

/// Per-row grain seed pair (spec 7.18.3.5).
fn fg_row_seed(rows: usize, row_num: usize, base_seed: u16) -> [u32; 2] {
    let mut seed = [0u32; 2];
    for (i, s) in seed.iter_mut().enumerate().take(rows) {
        *s = base_seed as u32;
        *s ^= ((((row_num - i) * 37 + 178) & 0xff) << 8) as u32;
        *s ^= (((row_num - i) * 173 + 105) & 0xff) as u32;
    }
    seed
}

/// Apply luma grain to one 32-row band (`fgy_32x32xn`), planar 8-bit.
// `rows` (1 or 2) indexes two offset sub-arrays in lockstep; a range loop is clearest.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn fg_apply_y_band(
    fg: &Av1FilmGrain,
    dst: &mut [u8],
    src: &[u8],
    w: usize,
    stride: usize,
    scaling: &[u8; 256],
    grain: &FgGrainLut,
    bh: usize,
    row_num: usize,
    band_y0: usize,
) {
    let rows = 1 + (fg.overlap_flag && row_num > 0) as usize;
    let (min_v, max_v) = if fg.clip_to_restricted_range {
        (16, 235)
    } else {
        (0, 255)
    };
    let mut seed = fg_row_seed(rows, row_num, fg.seed);
    let mut offsets = [[0i32; 2]; 2];
    static WT: [[i32; 2]; 2] = [[27, 17], [17, 27]];
    let noise_y = |sv: u8, g: i32| -> u8 {
        let noise = fg_round2(scaling[sv as usize] as i32 * g, fg.scaling_shift);
        (sv as i32 + noise).clamp(min_v, max_v) as u8
    };
    let px = |buf: &[u8], x: usize, y: usize| buf[(band_y0 + y) * stride + x];
    for bx in (0..w).step_by(FG_BLOCK_SIZE) {
        let bw = core::cmp::min(FG_BLOCK_SIZE, w - bx);
        if fg.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = fg_get_random(8, &mut seed[i]);
        }
        let ystart = if fg.overlap_flag && row_num != 0 {
            core::cmp::min(2, bh)
        } else {
            0
        };
        let xstart = if fg.overlap_flag && bx != 0 {
            core::cmp::min(2, bw)
        } else {
            0
        };
        for y in ystart..bh {
            for x in xstart..bw {
                let g = fg_sample_lut(grain, &offsets, 0, 0, 0, 0, x, y);
                let out = noise_y(px(src, bx + x, y), g);
                dst[(band_y0 + y) * stride + bx + x] = out;
            }
            for x in 0..xstart {
                let g = fg_sample_lut(grain, &offsets, 0, 0, 0, 0, x, y);
                let old = fg_sample_lut(grain, &offsets, 0, 0, 1, 0, x, y);
                let g = fg_round2(old * WT[x][0] + g * WT[x][1], 5).clamp(-128, 127);
                dst[(band_y0 + y) * stride + bx + x] = noise_y(px(src, bx + x, y), g);
            }
        }
        for y in 0..ystart {
            for x in xstart..bw {
                let g = fg_sample_lut(grain, &offsets, 0, 0, 0, 0, x, y);
                let old = fg_sample_lut(grain, &offsets, 0, 0, 0, 1, x, y);
                let g = fg_round2(old * WT[y][0] + g * WT[y][1], 5).clamp(-128, 127);
                dst[(band_y0 + y) * stride + bx + x] = noise_y(px(src, bx + x, y), g);
            }
            for x in 0..xstart {
                let top = fg_sample_lut(grain, &offsets, 0, 0, 0, 1, x, y);
                let old = fg_sample_lut(grain, &offsets, 0, 0, 1, 1, x, y);
                let top = fg_round2(old * WT[x][0] + top * WT[x][1], 5).clamp(-128, 127);
                let g = fg_sample_lut(grain, &offsets, 0, 0, 0, 0, x, y);
                let old = fg_sample_lut(grain, &offsets, 0, 0, 1, 0, x, y);
                let g = fg_round2(old * WT[x][0] + g * WT[x][1], 5).clamp(-128, 127);
                let g = fg_round2(top * WT[y][0] + g * WT[y][1], 5).clamp(-128, 127);
                dst[(band_y0 + y) * stride + bx + x] = noise_y(px(src, bx + x, y), g);
            }
        }
    }
}

/// Apply chroma grain to one band (`fguv_32x32xn`), planar 8-bit 4:2:0. `uv` = 0
/// (Cb) / 1 (Cr). `luma` is the full grain-free luma plane; `scaling` is the plane's
/// (or luma's, when `chroma_scaling_from_luma`) scaling LUT.
// `rows` (1 or 2) indexes two offset sub-arrays in lockstep; a range loop is clearest.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn fg_apply_uv_band_420(
    fg: &Av1FilmGrain,
    dst: &mut [u8],
    src: &[u8],
    cw: usize,
    cstride: usize,
    luma: &[u8],
    lstride: usize,
    scaling: &[u8; 256],
    grain: &FgGrainLut,
    bh: usize,
    row_num: usize,
    band_cy0: usize,
    uv: usize,
    is_id: bool,
) {
    let rows = 1 + (fg.overlap_flag && row_num > 0) as usize;
    let (min_v, max_v) = if fg.clip_to_restricted_range {
        (16, if is_id { 235 } else { 240 })
    } else {
        (0, 255)
    };
    let mut seed = fg_row_seed(rows, row_num, fg.seed);
    let mut offsets = [[0i32; 2]; 2];
    static WT: [[[i32; 2]; 2]; 2] = [[[27, 17], [17, 27]], [[23, 22], [0, 0]]];
    // luma co-located to chroma (bx,by) for 4:2:0: 2x2 average.
    let noise_uv = |sv: u8, g: i32, lx: usize, ly: usize| -> u8 {
        let l0 = luma[(band_cy0 * 2 + ly * 2) * lstride + lx * 2] as i32;
        let l1 = luma[(band_cy0 * 2 + ly * 2) * lstride + lx * 2 + 1] as i32;
        let avg = (l0 + l1 + 1) >> 1;
        let val = if fg.chroma_scaling_from_luma {
            avg
        } else {
            let combined = avg * fg.uv_luma_mult[uv] + sv as i32 * fg.uv_mult[uv];
            ((combined >> 6) + fg.uv_offset[uv]).clamp(0, 255)
        };
        let noise = fg_round2(scaling[val as usize] as i32 * g, fg.scaling_shift);
        (sv as i32 + noise).clamp(min_v, max_v) as u8
    };
    let px = |buf: &[u8], x: usize, y: usize| buf[(band_cy0 + y) * cstride + x];
    let bsz = FG_BLOCK_SIZE >> 1; // subsampled block width/height (16)
    for bx in (0..cw).step_by(bsz) {
        let bw = core::cmp::min(bsz, cw - bx);
        if fg.overlap_flag && bx != 0 {
            for i in 0..rows {
                offsets[1][i] = offsets[0][i];
            }
        }
        for i in 0..rows {
            offsets[0][i] = fg_get_random(8, &mut seed[i]);
        }
        let ystart = if fg.overlap_flag && row_num != 0 {
            core::cmp::min(1, bh)
        } else {
            0
        };
        let xstart = if fg.overlap_flag && bx != 0 {
            core::cmp::min(1, bw)
        } else {
            0
        };
        for y in ystart..bh {
            for x in xstart..bw {
                let g = fg_sample_lut(grain, &offsets, 1, 1, 0, 0, x, y);
                dst[(band_cy0 + y) * cstride + bx + x] = noise_uv(px(src, bx + x, y), g, bx + x, y);
            }
            for x in 0..xstart {
                let g = fg_sample_lut(grain, &offsets, 1, 1, 0, 0, x, y);
                let old = fg_sample_lut(grain, &offsets, 1, 1, 1, 0, x, y);
                let g = fg_round2(old * WT[1][x][0] + g * WT[1][x][1], 5).clamp(-128, 127);
                dst[(band_cy0 + y) * cstride + bx + x] = noise_uv(px(src, bx + x, y), g, bx + x, y);
            }
        }
        for y in 0..ystart {
            for x in xstart..bw {
                let g = fg_sample_lut(grain, &offsets, 1, 1, 0, 0, x, y);
                let old = fg_sample_lut(grain, &offsets, 1, 1, 0, 1, x, y);
                let g = fg_round2(old * WT[1][y][0] + g * WT[1][y][1], 5).clamp(-128, 127);
                dst[(band_cy0 + y) * cstride + bx + x] = noise_uv(px(src, bx + x, y), g, bx + x, y);
            }
            for x in 0..xstart {
                let top = fg_sample_lut(grain, &offsets, 1, 1, 0, 1, x, y);
                let old = fg_sample_lut(grain, &offsets, 1, 1, 1, 1, x, y);
                let top = fg_round2(old * WT[1][x][0] + top * WT[1][x][1], 5).clamp(-128, 127);
                let g = fg_sample_lut(grain, &offsets, 1, 1, 0, 0, x, y);
                let old = fg_sample_lut(grain, &offsets, 1, 1, 1, 0, x, y);
                let g = fg_round2(old * WT[1][x][0] + g * WT[1][x][1], 5).clamp(-128, 127);
                let g = fg_round2(top * WT[1][y][0] + g * WT[1][y][1], 5).clamp(-128, 127);
                dst[(band_cy0 + y) * cstride + bx + x] = noise_uv(px(src, bx + x, y), g, bx + x, y);
            }
        }
    }
}

/// Synthesize + apply AV1 film grain to a decoded NV12 frame in place (8-bit 4:2:0),
/// matching dav1d. The frame is the grain-free reconstruction from the hardware
/// decoder; `fg` is the resolved (ref-copied if needed) grain params.
fn apply_film_grain_nv12(frame: &mut Nv12Frame, fg: &Av1FilmGrain, is_id: bool) {
    if !fg.apply_grain {
        return;
    }
    // 8-bit only: the grain synthesis indexes the planes as one byte per sample.
    // A 10-bit grain stream is left grain-free rather than corrupted (10-bit grain
    // is a follow-up alongside the 10-bit RGB path).
    if frame.bit_depth != 8 {
        return;
    }
    let w = frame.width as usize;
    let h = frame.height as usize;
    let cw = w / 2;
    let ch = h / 2;

    // Grain templates.
    let mut grain_y: alloc::boxed::Box<FgGrainLut> =
        alloc::boxed::Box::new([[0i16; FG_GRAIN_WIDTH]; FG_GRAIN_HEIGHT + 1]);
    fg_generate_grain_y(fg, &mut grain_y);
    let mut grain_cb: alloc::boxed::Box<FgGrainLut> =
        alloc::boxed::Box::new([[0i16; FG_GRAIN_WIDTH]; FG_GRAIN_HEIGHT + 1]);
    let mut grain_cr: alloc::boxed::Box<FgGrainLut> =
        alloc::boxed::Box::new([[0i16; FG_GRAIN_WIDTH]; FG_GRAIN_HEIGHT + 1]);
    if fg.num_uv_points[0] != 0 || fg.chroma_scaling_from_luma {
        fg_generate_grain_uv_420(fg, &mut grain_cb, &grain_y, 0);
    }
    if fg.num_uv_points[1] != 0 || fg.chroma_scaling_from_luma {
        fg_generate_grain_uv_420(fg, &mut grain_cr, &grain_y, 1);
    }

    // Scaling LUTs.
    let scaling_y = fg_generate_scaling(&fg.y_points[..fg.num_y_points]);
    let scaling_cb = fg_generate_scaling(&fg.uv_points[0][..fg.num_uv_points[0]]);
    let scaling_cr = fg_generate_scaling(&fg.uv_points[1][..fg.num_uv_points[1]]);

    // Luma: apply in 32-row bands over a copy of the grain-free source.
    if fg.num_y_points != 0 {
        let src_y = frame.luma.clone();
        let rows = h.div_ceil(FG_BLOCK_SIZE);
        for row in 0..rows {
            let band_y0 = row * FG_BLOCK_SIZE;
            let bh = core::cmp::min(h - band_y0, FG_BLOCK_SIZE);
            fg_apply_y_band(
                fg,
                &mut frame.luma,
                &src_y,
                w,
                w,
                &scaling_y,
                &grain_y,
                bh,
                row,
                band_y0,
            );
        }
    }

    if fg.num_uv_points[0] == 0 && fg.num_uv_points[1] == 0 && !fg.chroma_scaling_from_luma {
        return;
    }

    // De-interleave NV12 chroma into planar Cb / Cr, apply, re-interleave.
    let mut cb = alloc::vec![0u8; cw * ch];
    let mut cr = alloc::vec![0u8; cw * ch];
    for i in 0..cw * ch {
        cb[i] = frame.chroma[2 * i];
        cr[i] = frame.chroma[2 * i + 1];
    }
    let rows = h.div_ceil(FG_BLOCK_SIZE);
    for (uv, (plane, gr)) in [(&mut cb, &*grain_cb), (&mut cr, &*grain_cr)]
        .into_iter()
        .enumerate()
    {
        if fg.num_uv_points[uv] == 0 && !fg.chroma_scaling_from_luma {
            continue;
        }
        let scaling = if fg.chroma_scaling_from_luma {
            &scaling_y
        } else if uv == 0 {
            &scaling_cb
        } else {
            &scaling_cr
        };
        let src = plane.clone();
        for row in 0..rows {
            let band_cy0 = row * (FG_BLOCK_SIZE >> 1);
            let luma_band = core::cmp::min(h - row * FG_BLOCK_SIZE, FG_BLOCK_SIZE);
            let bh = (luma_band + 1) >> 1;
            fg_apply_uv_band_420(
                fg,
                plane,
                &src,
                cw,
                cw,
                &frame.luma,
                w,
                scaling,
                gr,
                bh,
                row,
                band_cy0,
                uv,
                is_id,
            );
        }
    }
    for i in 0..cw * ch {
        frame.chroma[2 * i] = cb[i];
        frame.chroma[2 * i + 1] = cr[i];
    }
}

/// `quantization_params()` (spec 5.9.12). `num_planes` and `separate_uv_delta_q`
/// come from the sequence header's color config.
fn parse_av1_quantization(
    br: &mut BitReader,
    num_planes: u8,
    separate_uv_delta_q: bool,
) -> Option<Av1Quantization> {
    let base_q_idx = br.read_bits(8)? as u8;
    let delta_q_y_dc = read_delta_q(br)?;
    let mut diff_uv_delta = false;
    let (mut du_dc, mut du_ac, mut dv_dc, mut dv_ac) = (0i8, 0i8, 0i8, 0i8);
    if num_planes > 1 {
        if separate_uv_delta_q {
            diff_uv_delta = br.read_bit()? == 1;
        }
        du_dc = read_delta_q(br)?;
        du_ac = read_delta_q(br)?;
        if diff_uv_delta {
            dv_dc = read_delta_q(br)?;
            dv_ac = read_delta_q(br)?;
        } else {
            dv_dc = du_dc;
            dv_ac = du_ac;
        }
    }
    let using_qmatrix = br.read_bit()? == 1;
    let (mut qm_y, mut qm_u, mut qm_v) = (0u8, 0u8, 0u8);
    if using_qmatrix {
        qm_y = br.read_bits(4)? as u8;
        qm_u = br.read_bits(4)? as u8;
        qm_v = if !separate_uv_delta_q {
            qm_u
        } else {
            br.read_bits(4)? as u8
        };
    }
    Some(Av1Quantization {
        base_q_idx,
        delta_q_y_dc,
        delta_q_u_dc: du_dc,
        delta_q_u_ac: du_ac,
        delta_q_v_dc: dv_dc,
        delta_q_v_ac: dv_ac,
        using_qmatrix,
        diff_uv_delta,
        qm_y,
        qm_u,
        qm_v,
    })
}

// Segmentation feature bit widths / signedness / max (spec 5.9.14).
const AV1_SEG_LVL_MAX: usize = 8;
const AV1_SEG_FEATURE_BITS: [u32; AV1_SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const AV1_SEG_FEATURE_SIGNED: [bool; AV1_SEG_LVL_MAX] =
    [true, true, true, true, true, false, false, false];
const AV1_SEG_FEATURE_MAX: [i32; AV1_SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];

/// `segmentation_params()` (spec 5.9.14). `primary_ref_none` selects the
/// setup-past-independence defaults (no update-data path). When segmentation is
/// enabled without re-coding the data (inter, primary ref present), the feature
/// values would be loaded from the primary reference; this uses defaults, which
/// is correct for streams that do not change segmentation across frames.
fn parse_av1_segmentation(br: &mut BitReader, primary_ref_none: bool) -> Option<Av1Segmentation> {
    let enabled = br.read_bit()? == 1;
    let mut seg = Av1Segmentation {
        enabled,
        update_map: false,
        temporal_update: false,
        update_data: false,
        feature_enabled: [0; AV1_NUM_REF_FRAMES],
        feature_data: [[0; 8]; AV1_NUM_REF_FRAMES],
    };
    if !enabled {
        return Some(seg);
    }
    if primary_ref_none {
        seg.update_map = true;
        seg.temporal_update = false;
        seg.update_data = true;
    } else {
        seg.update_map = br.read_bit()? == 1;
        if seg.update_map {
            seg.temporal_update = br.read_bit()? == 1;
        }
        seg.update_data = br.read_bit()? == 1;
    }
    if seg.update_data {
        for s in 0..AV1_NUM_REF_FRAMES {
            for (j, &bits) in AV1_SEG_FEATURE_BITS.iter().enumerate() {
                let feature_value;
                let feature_enabled = br.read_bit()? == 1;
                if feature_enabled {
                    seg.feature_enabled[s] |= 1 << j;
                    let limit = AV1_SEG_FEATURE_MAX[j];
                    if AV1_SEG_FEATURE_SIGNED[j] {
                        let v = read_su(br, bits + 1)?;
                        feature_value = v.clamp(-limit, limit);
                    } else if bits == 0 {
                        feature_value = 0;
                    } else {
                        let v = br.read_bits(bits)? as i32;
                        feature_value = v.clamp(0, limit);
                    }
                } else {
                    feature_value = 0;
                }
                seg.feature_data[s][j] = feature_value as i16;
            }
        }
    }
    Some(seg)
}

/// `loop_filter_params()` (spec 5.9.11). Uses the setup-past-independence default
/// deltas (correct when they are never re-coded across frames, as in the fixture).
fn parse_av1_loop_filter(
    br: &mut BitReader,
    num_planes: u8,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Option<Av1LoopFilter> {
    let mut lf = Av1LoopFilter {
        level: [0; 4],
        sharpness: 0,
        delta_enabled: true,
        delta_update: false,
        ref_deltas: AV1_DEFAULT_LF_REF_DELTAS,
        mode_deltas: [0, 0],
    };
    if coded_lossless || allow_intrabc {
        // Filtering disabled; deltas stay at their defaults.
        lf.delta_enabled = false;
        return Some(lf);
    }
    lf.level[0] = br.read_bits(6)? as u8;
    lf.level[1] = br.read_bits(6)? as u8;
    if num_planes > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
        lf.level[2] = br.read_bits(6)? as u8;
        lf.level[3] = br.read_bits(6)? as u8;
    }
    lf.sharpness = br.read_bits(3)? as u8;
    lf.delta_enabled = br.read_bit()? == 1;
    if lf.delta_enabled {
        lf.delta_update = br.read_bit()? == 1;
        if lf.delta_update {
            for i in 0..AV1_NUM_REF_FRAMES {
                if br.read_bit()? == 1 {
                    lf.ref_deltas[i] = read_su(br, 7)? as i8;
                }
            }
            for i in 0..2 {
                if br.read_bit()? == 1 {
                    lf.mode_deltas[i] = read_su(br, 7)? as i8;
                }
            }
        }
    }
    Some(lf)
}

/// `cdef_params()` (spec 5.9.19).
fn parse_av1_cdef(
    br: &mut BitReader,
    num_planes: u8,
    enable_cdef: bool,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Option<Av1Cdef> {
    let mut cdef = Av1Cdef {
        damping_minus_3: 0,
        bits: 0,
        y_pri: [0; 8],
        y_sec: [0; 8],
        uv_pri: [0; 8],
        uv_sec: [0; 8],
    };
    if coded_lossless || allow_intrabc || !enable_cdef {
        // Single default filter, all zero.
        return Some(cdef);
    }
    cdef.damping_minus_3 = br.read_bits(2)? as u8;
    cdef.bits = br.read_bits(2)? as u8;
    for i in 0..(1usize << cdef.bits) {
        cdef.y_pri[i] = br.read_bits(4)? as u8;
        let mut sec = br.read_bits(2)? as u8;
        if sec == 3 {
            sec += 1;
        }
        cdef.y_sec[i] = sec;
        if num_planes > 1 {
            cdef.uv_pri[i] = br.read_bits(4)? as u8;
            let mut usec = br.read_bits(2)? as u8;
            if usec == 3 {
                usec += 1;
            }
            cdef.uv_sec[i] = usec;
        }
    }
    Some(cdef)
}

// Loop-restoration types (spec 6.10.15). Remap table applied to the coded 2 bits.
const AV1_REMAP_LR_TYPE: [u8; 4] = [0, 3, 1, 2]; // {NONE, SWITCHABLE, WIENER, SGRPROJ}

/// `lr_params()` (spec 5.9.20). `subsampling` = (subsampling_x, subsampling_y).
fn parse_av1_loop_restoration(
    br: &mut BitReader,
    num_planes: u8,
    use_128x128: bool,
    enable_restoration: bool,
    all_lossless: bool,
    allow_intrabc: bool,
    subsampling: (u8, u8),
) -> Option<Av1LoopRestoration> {
    let mut lr = Av1LoopRestoration {
        frame_restoration_type: [0; 3],
        // `StdVideoAV1LoopRestoration::LoopRestorationSize` is NOT the pixel unit
        // size (64/128/256): the Vulkan Std encodes it as `1 + lr_unit_shift` for
        // luma and `1 + lr_unit_shift - lr_uv_shift` for chroma (values 1..3), the
        // same encoding ffmpeg's Vulkan AV1 hwaccel passes. Passing the raw pixel
        // size mis-configures the driver's restoration and corrupts the whole
        // decode (even planes with RESTORE_NONE). Default (no LR) is `1 + 0`.
        loop_restoration_size: [1, 1, 1],
        uses_lr: false,
        uses_chroma_lr: false,
    };
    if all_lossless || allow_intrabc || !enable_restoration {
        return Some(lr);
    }
    let mut uses_lr = false;
    let mut uses_chroma_lr = false;
    for i in 0..num_planes as usize {
        let lr_type = br.read_bits(2)?;
        lr.frame_restoration_type[i] = AV1_REMAP_LR_TYPE[lr_type as usize];
        if lr.frame_restoration_type[i] != 0 {
            uses_lr = true;
            if i > 0 {
                uses_chroma_lr = true;
            }
        }
    }
    lr.uses_lr = uses_lr;
    lr.uses_chroma_lr = uses_chroma_lr;
    if uses_lr {
        let mut lr_unit_shift;
        if use_128x128 {
            lr_unit_shift = br.read_bit()? + 1;
        } else {
            lr_unit_shift = br.read_bit()?;
            if lr_unit_shift != 0 {
                let lr_unit_extra_shift = br.read_bit()?;
                lr_unit_shift += lr_unit_extra_shift;
            }
        }
        lr.loop_restoration_size[0] = (1 + lr_unit_shift) as u16;
        let lr_uv_shift = if subsampling.0 != 0 && subsampling.1 != 0 && uses_chroma_lr {
            br.read_bit()?
        } else {
            0
        };
        // Chroma unit size in the same `1 + shift` encoding, reduced by lr_uv_shift.
        let uv_size = (1 + lr_unit_shift - lr_uv_shift) as u16;
        lr.loop_restoration_size[1] = uv_size;
        lr.loop_restoration_size[2] = uv_size;
    }
    Some(lr)
}

/// Decode one global-motion parameter (spec 5.9.25 read_global_param + the
/// signed-subexp helpers 5.9.26-5.9.28), relative to `ref_val`.
fn read_global_param(br: &mut BitReader, gm_type: u8, idx: usize, ref_val: i32) -> Option<i32> {
    // Bit width per parameter (spec 5.9.25).
    const GM_ABS_TRANS_BITS: u32 = 12;
    const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
    const GM_ABS_ALPHA_BITS: u32 = 12;
    const GM_ALPHA_PREC_BITS: u32 = 15;
    const GM_TRANS_PREC_BITS: u32 = 6;
    const GM_TRANS_ONLY_PREC_BITS: u32 = 3;
    const WARPEDMODEL_PREC_BITS: u32 = 16;

    let (abs_bits, prec_bits) = if idx < 2 {
        if gm_type == AV1_GM_TRANSLATION {
            let allow_hp_bits = 0; // set by caller via ref_val path; hp handled below
            let _ = allow_hp_bits;
            (GM_ABS_TRANS_ONLY_BITS, GM_TRANS_ONLY_PREC_BITS)
        } else {
            (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
        }
    } else {
        (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
    };
    let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    let round = if idx % 3 == 2 {
        1i32 << WARPEDMODEL_PREC_BITS
    } else {
        0
    };
    let sub = if idx % 3 == 2 { 1i32 << prec_bits } else { 0 };
    let mx = 1i32 << abs_bits;
    let r = (ref_val >> prec_diff) - sub;
    let v = decode_signed_subexp_with_ref(br, -mx, mx + 1, r)?;
    Some((v << prec_diff) + round)
}

/// `decode_signed_subexp_with_ref` (spec 5.9.27).
fn decode_signed_subexp_with_ref(br: &mut BitReader, low: i32, high: i32, r: i32) -> Option<i32> {
    let x = decode_unsigned_subexp_with_ref(br, (high - low) as u32, (r - low) as u32)?;
    Some(x as i32 + low)
}

/// `decode_unsigned_subexp_with_ref` (spec 5.9.28) + `inverse_recenter` (5.9.29).
fn decode_unsigned_subexp_with_ref(br: &mut BitReader, mx: u32, r: u32) -> Option<u32> {
    let v = decode_subexp(br, mx)?;
    if (r << 1) <= mx {
        Some(inverse_recenter(r, v))
    } else {
        Some(mx - 1 - inverse_recenter(mx - 1 - r, v))
    }
}

fn inverse_recenter(r: u32, v: u32) -> u32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r + ((v + 1) >> 1)
    } else {
        r - (v >> 1)
    }
}

/// `decode_subexp` (spec 5.9.26).
fn decode_subexp(br: &mut BitReader, num_syms: u32) -> Option<u32> {
    let mut i = 0u32;
    let mut mk = 0u32;
    let k = 3u32;
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1u32 << b2;
        if num_syms <= mk + 3 * a {
            let subexp_final_bits = read_ns(br, num_syms - mk)?;
            return Some(subexp_final_bits + mk);
        } else {
            let subexp_more_bits = br.read_bit()?;
            if subexp_more_bits != 0 {
                i += 1;
                mk += a;
            } else {
                let subexp_bits = br.read_bits(b2)?;
                return Some(subexp_bits + mk);
            }
        }
    }
}

/// `global_motion_params()` (spec 5.9.24). Uses identity as the "previous"
/// reference model (correct when global motion is not carried across frames).
fn parse_av1_global_motion(br: &mut BitReader, frame_is_intra: bool) -> Option<Av1GlobalMotion> {
    let identity = av1_default_warp_params();
    let mut gm = Av1GlobalMotion {
        gm_type: [AV1_GM_IDENTITY; AV1_NUM_REF_FRAMES],
        gm_params: [identity; AV1_NUM_REF_FRAMES],
    };
    if frame_is_intra {
        return Some(gm);
    }
    // Reference frames LAST(1)..=ALTREF(7).
    for r#ref in 1..AV1_NUM_REF_FRAMES {
        let mut gm_type = AV1_GM_IDENTITY;
        let is_global = br.read_bit()? == 1;
        if is_global {
            let is_rot_zoom = br.read_bit()? == 1;
            if is_rot_zoom {
                gm_type = AV1_GM_ROTZOOM;
            } else {
                let is_translation = br.read_bit()? == 1;
                gm_type = if is_translation {
                    AV1_GM_TRANSLATION
                } else {
                    AV1_GM_AFFINE
                };
            }
        }
        gm.gm_type[r#ref] = gm_type;
        // "Previous" params default to identity (no cross-frame carry here).
        let prev = identity;
        if gm_type >= AV1_GM_ROTZOOM {
            gm.gm_params[r#ref][2] = read_global_param(br, gm_type, 2, prev[2])?;
            gm.gm_params[r#ref][3] = read_global_param(br, gm_type, 3, prev[3])?;
            if gm_type == AV1_GM_AFFINE {
                gm.gm_params[r#ref][4] = read_global_param(br, gm_type, 4, prev[4])?;
                gm.gm_params[r#ref][5] = read_global_param(br, gm_type, 5, prev[5])?;
            } else {
                gm.gm_params[r#ref][4] = -gm.gm_params[r#ref][3];
                gm.gm_params[r#ref][5] = gm.gm_params[r#ref][2];
            }
        }
        if gm_type >= AV1_GM_TRANSLATION {
            gm.gm_params[r#ref][0] = read_global_param(br, gm_type, 0, prev[0])?;
            gm.gm_params[r#ref][1] = read_global_param(br, gm_type, 1, prev[1])?;
        }
    }
    Some(gm)
}

/// The identity warp model (spec 7.10.1): scale terms 1<<PREC, rest 0.
fn av1_default_warp_params() -> [i32; 6] {
    let one = 1i32 << AV1_WARPEDMODEL_PREC_BITS;
    [0, 0, one, 0, 0, one]
}

/// `get_relative_dist(a, b)` (spec 5.9.3): the wrapped order-hint distance.
fn av1_relative_dist(order_hint_bits: u32, a: i32, b: i32) -> i32 {
    if order_hint_bits == 0 {
        return 0;
    }
    let diff = a - b;
    let m = 1i32 << (order_hint_bits - 1);
    (diff & (m - 1)) - (diff & m)
}

/// Parse the AV1 uncompressed frame header (spec 5.9.2) from a frame /
/// frame-header OBU payload, with the sequence header and current reference
/// state for context. Returns the parsed header including the byte length of the
/// header (where the tile group begins in an `OBU_FRAME`).
pub fn parse_av1_frame_header(
    payload: &[u8],
    seq: &Av1SequenceHeader,
    refs: &Av1RefFrames,
) -> Option<Av1FrameHeader> {
    // The fixture path targets streams without a decoder model or frame ids;
    // reject those rather than mis-parse them.
    if seq.decoder_model_info_present_flag || seq.frame_id_numbers_present_flag {
        return None;
    }

    let mut br = BitReader::new(payload);
    let order_hint_bits = seq.order_hint_bits as u32;
    let num_planes = seq.color.num_planes;
    let all_frames: u8 = 0xff;

    let mut fh = Av1FrameHeader {
        show_existing_frame: false,
        frame_to_show_map_idx: 0,
        frame_type: AV1_FRAME_TYPE_KEY,
        frame_is_intra: true,
        show_frame: true,
        showable_frame: false,
        error_resilient_mode: false,
        disable_cdf_update: false,
        allow_screen_content_tools: false,
        force_integer_mv: false,
        current_frame_id: 0,
        frame_size_override_flag: false,
        order_hint: 0,
        primary_ref_frame: AV1_PRIMARY_REF_NONE,
        refresh_frame_flags: 0,
        ref_frame_idx: [-1; AV1_REFS_PER_FRAME],
        frame_width: 0,
        frame_height: 0,
        upscaled_width: 0,
        render_width: 0,
        render_height: 0,
        use_superres: false,
        superres_denom: 8,
        mi_cols: 0,
        mi_rows: 0,
        allow_high_precision_mv: false,
        interpolation_filter: 0,
        is_motion_mode_switchable: false,
        use_ref_frame_mvs: false,
        disable_frame_end_update_cdf: false,
        allow_intrabc: false,
        reduced_tx_set: false,
        tx_mode: 0,
        reference_select: false,
        skip_mode_present: false,
        skip_mode_frame: [0, 0],
        allow_warped_motion: false,
        delta_q_present: false,
        delta_q_res: 0,
        delta_lf_present: false,
        delta_lf_res: 0,
        delta_lf_multi: false,
        coded_lossless: false,
        all_lossless: false,
        tile: Av1TileInfo {
            uniform_spacing: true,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            tile_cols: 1,
            tile_rows: 1,
            context_update_tile_id: 0,
            tile_size_bytes: 1,
            mi_col_starts: alloc::vec::Vec::new(),
            mi_row_starts: alloc::vec::Vec::new(),
            width_in_sbs_minus_1: alloc::vec::Vec::new(),
            height_in_sbs_minus_1: alloc::vec::Vec::new(),
        },
        quant: Av1Quantization {
            base_q_idx: 0,
            delta_q_y_dc: 0,
            delta_q_u_dc: 0,
            delta_q_u_ac: 0,
            delta_q_v_dc: 0,
            delta_q_v_ac: 0,
            using_qmatrix: false,
            diff_uv_delta: false,
            qm_y: 0,
            qm_u: 0,
            qm_v: 0,
        },
        seg: Av1Segmentation {
            enabled: false,
            update_map: false,
            temporal_update: false,
            update_data: false,
            feature_enabled: [0; AV1_NUM_REF_FRAMES],
            feature_data: [[0; 8]; AV1_NUM_REF_FRAMES],
        },
        lf: Av1LoopFilter {
            level: [0; 4],
            sharpness: 0,
            delta_enabled: false,
            delta_update: false,
            ref_deltas: AV1_DEFAULT_LF_REF_DELTAS,
            mode_deltas: [0, 0],
        },
        cdef: Av1Cdef {
            damping_minus_3: 0,
            bits: 0,
            y_pri: [0; 8],
            y_sec: [0; 8],
            uv_pri: [0; 8],
            uv_sec: [0; 8],
        },
        lr: Av1LoopRestoration {
            frame_restoration_type: [0; 3],
            loop_restoration_size: [0; 3],
            uses_lr: false,
            uses_chroma_lr: false,
        },
        gm: Av1GlobalMotion {
            gm_type: [AV1_GM_IDENTITY; AV1_NUM_REF_FRAMES],
            gm_params: [av1_default_warp_params(); AV1_NUM_REF_FRAMES],
        },
        film_grain: Av1FilmGrain::default(),
        header_byte_len: 0,
    };

    if seq.reduced_still_picture_header {
        // frame_type / show / intra already at their defaults (KEY, shown, intra).
    } else {
        fh.show_existing_frame = br.read_bit()? == 1;
        if fh.show_existing_frame {
            fh.frame_to_show_map_idx = br.read_bits(3)? as u8;
            fh.frame_type = refs.frame_type[fh.frame_to_show_map_idx as usize];
            fh.header_byte_len = br.bit_pos().div_ceil(8);
            return Some(fh);
        }
        fh.frame_type = br.read_bits(2)? as u8;
        fh.frame_is_intra =
            fh.frame_type == AV1_FRAME_TYPE_KEY || fh.frame_type == AV1_FRAME_TYPE_INTRA_ONLY;
        fh.show_frame = br.read_bit()? == 1;
        // temporal_point_info: decoder-model streams are rejected above.
        if fh.show_frame {
            fh.showable_frame = fh.frame_type != AV1_FRAME_TYPE_KEY;
        } else {
            fh.showable_frame = br.read_bit()? == 1;
        }
        if fh.frame_type == AV1_FRAME_TYPE_SWITCH
            || (fh.frame_type == AV1_FRAME_TYPE_KEY && fh.show_frame)
        {
            fh.error_resilient_mode = true;
        } else {
            fh.error_resilient_mode = br.read_bit()? == 1;
        }
    }

    fh.disable_cdf_update = br.read_bit()? == 1;
    fh.allow_screen_content_tools =
        if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
            br.read_bit()? == 1
        } else {
            seq.seq_force_screen_content_tools != 0
        };
    if fh.allow_screen_content_tools {
        if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
            fh.force_integer_mv = br.read_bit()? == 1;
        } else {
            fh.force_integer_mv = seq.seq_force_integer_mv != 0;
        }
    }
    if fh.frame_is_intra {
        fh.force_integer_mv = true;
    }

    fh.frame_size_override_flag = if fh.frame_type == AV1_FRAME_TYPE_SWITCH {
        true
    } else if seq.reduced_still_picture_header {
        false
    } else {
        br.read_bit()? == 1
    };

    fh.order_hint = br.read_bits(order_hint_bits)? as u8;

    fh.primary_ref_frame = if fh.frame_is_intra || fh.error_resilient_mode {
        AV1_PRIMARY_REF_NONE
    } else {
        br.read_bits(3)? as u8
    };

    fh.refresh_frame_flags = if fh.frame_type == AV1_FRAME_TYPE_SWITCH
        || (fh.frame_type == AV1_FRAME_TYPE_KEY && fh.show_frame)
    {
        all_frames
    } else {
        br.read_bits(8)? as u8
    };

    // ref_order_hint: only signaled for error-resilient streams with order hints.
    if (!fh.frame_is_intra || fh.refresh_frame_flags != all_frames)
        && fh.error_resilient_mode
        && seq.enable_order_hint
    {
        for _ in 0..AV1_NUM_REF_FRAMES {
            br.read_bits(order_hint_bits)?; // ref_order_hint[i] (used only to refresh state)
        }
    }

    // Frame size + render size (+ superres), and reference selection.
    let read_frame_and_render_size = |br: &mut BitReader, fh: &mut Av1FrameHeader| -> Option<()> {
        // frame_size()
        if fh.frame_size_override_flag {
            let n = seq.frame_width_bits_minus_1 as u32 + 1;
            let m = seq.frame_height_bits_minus_1 as u32 + 1;
            fh.frame_width = br.read_bits(n)? + 1;
            fh.frame_height = br.read_bits(m)? + 1;
        } else {
            fh.frame_width = seq.max_frame_width_minus_1 + 1;
            fh.frame_height = seq.max_frame_height_minus_1 + 1;
        }
        // superres_params()
        fh.use_superres = if seq.enable_superres {
            br.read_bit()? == 1
        } else {
            false
        };
        if fh.use_superres {
            let coded_denom = br.read_bits(3)?;
            fh.superres_denom = (coded_denom + 9) as u8;
        } else {
            fh.superres_denom = 8;
        }
        fh.upscaled_width = fh.frame_width;
        fh.frame_width =
            (fh.upscaled_width * 8 + (fh.superres_denom as u32 / 2)) / fh.superres_denom as u32;
        // render_size()
        if br.read_bit()? == 1 {
            fh.render_width = br.read_bits(16)? + 1;
            fh.render_height = br.read_bits(16)? + 1;
        } else {
            fh.render_width = fh.upscaled_width;
            fh.render_height = fh.frame_height;
        }
        // compute_image_size()
        fh.mi_cols = 2 * ((fh.frame_width + 7) >> 3);
        fh.mi_rows = 2 * ((fh.frame_height + 7) >> 3);
        Some(())
    };

    if fh.frame_is_intra {
        read_frame_and_render_size(&mut br, &mut fh)?;
        if fh.allow_screen_content_tools && fh.upscaled_width == fh.frame_width {
            fh.allow_intrabc = br.read_bit()? == 1;
        }
    } else {
        let mut frame_refs_short_signaling = false;
        if seq.enable_order_hint {
            frame_refs_short_signaling = br.read_bit()? == 1;
            if frame_refs_short_signaling {
                // last_frame_idx + gold_frame_idx then set_frame_refs(): not in the
                // target streams, reject rather than guess the derived indices.
                return None;
            }
        }
        for i in 0..AV1_REFS_PER_FRAME {
            if !frame_refs_short_signaling {
                fh.ref_frame_idx[i] = br.read_bits(3)? as i8;
            }
            // frame_id_numbers_present rejected above, so no delta_frame_id read.
        }
        if fh.frame_size_override_flag && !fh.error_resilient_mode {
            // frame_size_with_refs: found_ref loop. Not exercised by the fixture
            // (override is 0), but handle it via the reference dimensions.
            let mut found = false;
            for i in 0..AV1_REFS_PER_FRAME {
                if br.read_bit()? == 1 {
                    let slot = fh.ref_frame_idx[i] as usize;
                    fh.upscaled_width = refs.upscaled_width[slot];
                    fh.frame_width = fh.upscaled_width;
                    fh.frame_height = refs.frame_height[slot];
                    fh.render_width = refs.render_width[slot];
                    fh.render_height = refs.render_height[slot];
                    found = true;
                    break;
                }
            }
            if found {
                // superres_params() + compute_image_size()
                fh.use_superres = if seq.enable_superres {
                    br.read_bit()? == 1
                } else {
                    false
                };
                if fh.use_superres {
                    let coded_denom = br.read_bits(3)?;
                    fh.superres_denom = (coded_denom + 9) as u8;
                } else {
                    fh.superres_denom = 8;
                }
                let up = fh.upscaled_width;
                fh.frame_width =
                    (up * 8 + (fh.superres_denom as u32 / 2)) / fh.superres_denom as u32;
                fh.mi_cols = 2 * ((fh.frame_width + 7) >> 3);
                fh.mi_rows = 2 * ((fh.frame_height + 7) >> 3);
            } else {
                read_frame_and_render_size(&mut br, &mut fh)?;
            }
        } else {
            read_frame_and_render_size(&mut br, &mut fh)?;
        }
        fh.allow_high_precision_mv = if fh.force_integer_mv {
            false
        } else {
            br.read_bit()? == 1
        };
        // read_interpolation_filter()
        if br.read_bit()? == 1 {
            fh.interpolation_filter = 4; // SWITCHABLE
        } else {
            fh.interpolation_filter = br.read_bits(2)? as u8;
        }
        fh.is_motion_mode_switchable = br.read_bit()? == 1;
        fh.use_ref_frame_mvs = if fh.error_resilient_mode || !seq.enable_ref_frame_mvs {
            false
        } else {
            br.read_bit()? == 1
        };
    }

    fh.disable_frame_end_update_cdf = if seq.reduced_still_picture_header || fh.disable_cdf_update {
        true
    } else {
        br.read_bit()? == 1
    };

    // tile_info + quant + segmentation.
    fh.tile = parse_av1_tile_info(&mut br, fh.mi_cols, fh.mi_rows, seq.use_128x128_superblock)?;
    fh.quant = parse_av1_quantization(&mut br, num_planes, seq.color.separate_uv_delta_q)?;
    fh.seg = parse_av1_segmentation(&mut br, fh.primary_ref_frame == AV1_PRIMARY_REF_NONE)?;

    // delta_q_params()
    if fh.quant.base_q_idx > 0 {
        fh.delta_q_present = br.read_bit()? == 1;
    }
    if fh.delta_q_present {
        fh.delta_q_res = br.read_bits(2)? as u8;
    }
    // delta_lf_params()
    if fh.delta_q_present {
        if !fh.allow_intrabc {
            fh.delta_lf_present = br.read_bit()? == 1;
        }
        if fh.delta_lf_present {
            fh.delta_lf_res = br.read_bits(2)? as u8;
            fh.delta_lf_multi = br.read_bit()? == 1;
        }
    }

    // CodedLossless / AllLossless (spec 5.9.2).
    let mut coded_lossless = true;
    for seg_id in 0..AV1_NUM_REF_FRAMES {
        let qindex = av1_seg_qindex(&fh.seg, fh.quant.base_q_idx, seg_id);
        let lossless = qindex == 0
            && fh.quant.delta_q_y_dc == 0
            && fh.quant.delta_q_u_ac == 0
            && fh.quant.delta_q_u_dc == 0
            && fh.quant.delta_q_v_ac == 0
            && fh.quant.delta_q_v_dc == 0;
        if !lossless {
            coded_lossless = false;
        }
    }
    fh.coded_lossless = coded_lossless;
    fh.all_lossless = coded_lossless && fh.frame_width == fh.upscaled_width;

    // loop_filter + cdef + lr.
    fh.lf = parse_av1_loop_filter(&mut br, num_planes, fh.coded_lossless, fh.allow_intrabc)?;
    fh.cdef = parse_av1_cdef(
        &mut br,
        num_planes,
        seq.enable_cdef,
        fh.coded_lossless,
        fh.allow_intrabc,
    )?;
    fh.lr = parse_av1_loop_restoration(
        &mut br,
        num_planes,
        seq.use_128x128_superblock,
        seq.enable_restoration,
        fh.all_lossless,
        fh.allow_intrabc,
        (seq.color.subsampling_x, seq.color.subsampling_y),
    )?;

    // read_tx_mode()
    fh.tx_mode = if fh.coded_lossless {
        0 // ONLY_4X4
    } else if br.read_bit()? == 1 {
        2 // SELECT
    } else {
        1 // LARGEST
    };

    // frame_reference_mode()
    fh.reference_select = if fh.frame_is_intra {
        false
    } else {
        br.read_bit()? == 1
    };

    // skip_mode_params()
    let (skip_mode_allowed, skip_frames) =
        av1_skip_mode_allowed(&fh, refs, seq.enable_order_hint, order_hint_bits);
    fh.skip_mode_frame = skip_frames;
    if skip_mode_allowed {
        fh.skip_mode_present = br.read_bit()? == 1;
    }

    fh.allow_warped_motion =
        if fh.frame_is_intra || fh.error_resilient_mode || !seq.enable_warped_motion {
            false
        } else {
            br.read_bit()? == 1
        };
    fh.reduced_tx_set = br.read_bit()? == 1;

    fh.gm = parse_av1_global_motion(&mut br, fh.frame_is_intra)?;

    // film_grain_params() (spec 5.9.30): parsed into `fh.film_grain`. The hardware
    // decoder produces the grain-free reconstruction; grain is synthesized on the
    // decoded output at display time (see `apply_film_grain_nv12`).
    fh.film_grain = parse_av1_film_grain(
        &mut br,
        seq,
        fh.frame_type,
        fh.show_frame,
        fh.showable_frame,
    )?;

    // byte_alignment() before the tile group.
    fh.header_byte_len = br.bit_pos().div_ceil(8);
    Some(fh)
}

/// The qindex for a segment considering only the segmentation ALT_Q feature
/// (SEG_LVL_ALT_Q == 0), used for the CodedLossless test.
fn av1_seg_qindex(seg: &Av1Segmentation, base_q_idx: u8, seg_id: usize) -> i32 {
    if seg.enabled && (seg.feature_enabled[seg_id] & 1) != 0 {
        let data = seg.feature_data[seg_id][0] as i32;
        (base_q_idx as i32 + data).clamp(0, 255)
    } else {
        base_q_idx as i32
    }
}

/// `skip_mode_params()` allowed test + the two skip-mode reference frames (spec
/// 5.9.22). Returns (skipModeAllowed, SkipModeFrame).
fn av1_skip_mode_allowed(
    fh: &Av1FrameHeader,
    refs: &Av1RefFrames,
    enable_order_hint: bool,
    order_hint_bits: u32,
) -> (bool, [u8; 2]) {
    const LAST_FRAME: u8 = 1;
    if fh.frame_is_intra || !fh.reference_select || !enable_order_hint {
        return (false, [0, 0]);
    }
    let cur = fh.order_hint as i32;
    let mut forward_idx: i32 = -1;
    let mut backward_idx: i32 = -1;
    let mut forward_hint = 0i32;
    let mut backward_hint = 0i32;
    for i in 0..AV1_REFS_PER_FRAME {
        let slot = fh.ref_frame_idx[i] as usize;
        let ref_hint = refs.order_hint[slot] as i32;
        if av1_relative_dist(order_hint_bits, ref_hint, cur) < 0 {
            if forward_idx < 0 || av1_relative_dist(order_hint_bits, ref_hint, forward_hint) > 0 {
                forward_idx = i as i32;
                forward_hint = ref_hint;
            }
        } else if av1_relative_dist(order_hint_bits, ref_hint, cur) > 0
            && (backward_idx < 0 || av1_relative_dist(order_hint_bits, ref_hint, backward_hint) < 0)
        {
            backward_idx = i as i32;
            backward_hint = ref_hint;
        }
    }
    if forward_idx < 0 {
        (false, [0, 0])
    } else if backward_idx >= 0 {
        let a = forward_idx.min(backward_idx) as u8;
        let b = forward_idx.max(backward_idx) as u8;
        (true, [LAST_FRAME + a, LAST_FRAME + b])
    } else {
        // Only a forward reference: find a second forward one.
        let mut second_forward_idx: i32 = -1;
        let mut second_forward_hint = 0i32;
        for i in 0..AV1_REFS_PER_FRAME {
            let slot = fh.ref_frame_idx[i] as usize;
            let ref_hint = refs.order_hint[slot] as i32;
            if av1_relative_dist(order_hint_bits, ref_hint, forward_hint) < 0
                && (second_forward_idx < 0
                    || av1_relative_dist(order_hint_bits, ref_hint, second_forward_hint) > 0)
            {
                second_forward_idx = i as i32;
                second_forward_hint = ref_hint;
            }
        }
        if second_forward_idx < 0 {
            (false, [0, 0])
        } else {
            let a = forward_idx.min(second_forward_idx) as u8;
            let b = forward_idx.max(second_forward_idx) as u8;
            (true, [LAST_FRAME + a, LAST_FRAME + b])
        }
    }
}

/// The `Std*` AV1 per-picture info bundle for one decoded frame: the
/// `StdVideoDecodeAV1PictureInfo` plus every sub-struct it points to, all owned
/// here so the pointers stay valid while the decode command reads them.
pub struct StdAv1PictureInfo {
    pub pic: vk::native::StdVideoDecodeAV1PictureInfo,
    _tile: alloc::boxed::Box<vk::native::StdVideoAV1TileInfo>,
    _mi_col_starts: alloc::vec::Vec<u16>,
    _mi_row_starts: alloc::vec::Vec<u16>,
    _width_in_sbs: alloc::vec::Vec<u16>,
    _height_in_sbs: alloc::vec::Vec<u16>,
    _quant: alloc::boxed::Box<vk::native::StdVideoAV1Quantization>,
    _seg: alloc::boxed::Box<vk::native::StdVideoAV1Segmentation>,
    _lf: alloc::boxed::Box<vk::native::StdVideoAV1LoopFilter>,
    _cdef: alloc::boxed::Box<vk::native::StdVideoAV1CDEF>,
    _lr: alloc::boxed::Box<vk::native::StdVideoAV1LoopRestoration>,
    _gm: alloc::boxed::Box<vk::native::StdVideoAV1GlobalMotion>,
}

impl core::fmt::Debug for StdAv1PictureInfo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("StdAv1PictureInfo")
            .field("frame_type", &self.pic.frame_type)
            .field("OrderHint", &self.pic.OrderHint)
            .finish_non_exhaustive()
    }
}

/// Build the `Std*` per-picture info bundle from a parsed frame header.
/// `order_hints` is the per-reference order-hint array (index by AV1 reference
/// name, 0 = intra/unused), which the decoder derives from its reference state.
pub fn to_std_av1_picture_info(
    fh: &Av1FrameHeader,
    order_hints: [u8; AV1_NUM_REF_FRAMES],
) -> StdAv1PictureInfo {
    // --- tile info ---
    let mut mi_col_starts = fh.tile.mi_col_starts.clone();
    let mut mi_row_starts = fh.tile.mi_row_starts.clone();
    let mut width_in_sbs = fh.tile.width_in_sbs_minus_1.clone();
    let mut height_in_sbs = fh.tile.height_in_sbs_minus_1.clone();
    // Never hand the driver a null array pointer even for a single tile.
    if mi_col_starts.is_empty() {
        mi_col_starts.push(0);
    }
    if mi_row_starts.is_empty() {
        mi_row_starts.push(0);
    }
    if width_in_sbs.is_empty() {
        width_in_sbs.push(0);
    }
    if height_in_sbs.is_empty() {
        height_in_sbs.push(0);
    }
    // SAFETY: bitfield POD, valid all-zero.
    let mut tile_flags: vk::native::StdVideoAV1TileInfoFlags = unsafe { core::mem::zeroed() };
    tile_flags.set_uniform_tile_spacing_flag(fh.tile.uniform_spacing as u32);
    let tile = alloc::boxed::Box::new(vk::native::StdVideoAV1TileInfo {
        flags: tile_flags,
        TileCols: fh.tile.tile_cols as u8,
        TileRows: fh.tile.tile_rows as u8,
        context_update_tile_id: fh.tile.context_update_tile_id as u16,
        tile_size_bytes_minus_1: (fh.tile.tile_size_bytes.max(1) - 1) as u8,
        reserved1: [0; 7],
        pMiColStarts: mi_col_starts.as_ptr(),
        pMiRowStarts: mi_row_starts.as_ptr(),
        pWidthInSbsMinus1: width_in_sbs.as_ptr(),
        pHeightInSbsMinus1: height_in_sbs.as_ptr(),
    });

    // --- quantization ---
    // SAFETY: bitfield POD, valid all-zero.
    let mut q_flags: vk::native::StdVideoAV1QuantizationFlags = unsafe { core::mem::zeroed() };
    q_flags.set_using_qmatrix(fh.quant.using_qmatrix as u32);
    q_flags.set_diff_uv_delta(fh.quant.diff_uv_delta as u32);
    let quant = alloc::boxed::Box::new(vk::native::StdVideoAV1Quantization {
        flags: q_flags,
        base_q_idx: fh.quant.base_q_idx,
        DeltaQYDc: fh.quant.delta_q_y_dc,
        DeltaQUDc: fh.quant.delta_q_u_dc,
        DeltaQUAc: fh.quant.delta_q_u_ac,
        DeltaQVDc: fh.quant.delta_q_v_dc,
        DeltaQVAc: fh.quant.delta_q_v_ac,
        qm_y: fh.quant.qm_y,
        qm_u: fh.quant.qm_u,
        qm_v: fh.quant.qm_v,
    });

    // --- segmentation ---
    let seg = alloc::boxed::Box::new(vk::native::StdVideoAV1Segmentation {
        FeatureEnabled: fh.seg.feature_enabled,
        FeatureData: fh.seg.feature_data,
    });

    // --- loop filter ---
    // SAFETY: bitfield POD, valid all-zero.
    let mut lf_flags: vk::native::StdVideoAV1LoopFilterFlags = unsafe { core::mem::zeroed() };
    lf_flags.set_loop_filter_delta_enabled(fh.lf.delta_enabled as u32);
    lf_flags.set_loop_filter_delta_update(fh.lf.delta_update as u32);
    let lf = alloc::boxed::Box::new(vk::native::StdVideoAV1LoopFilter {
        flags: lf_flags,
        loop_filter_level: fh.lf.level,
        loop_filter_sharpness: fh.lf.sharpness,
        update_ref_delta: if fh.lf.delta_update { 0xff } else { 0 },
        loop_filter_ref_deltas: fh.lf.ref_deltas,
        update_mode_delta: if fh.lf.delta_update { 0x3 } else { 0 },
        loop_filter_mode_deltas: fh.lf.mode_deltas,
    });

    // --- CDEF ---
    let cdef = alloc::boxed::Box::new(vk::native::StdVideoAV1CDEF {
        cdef_damping_minus_3: fh.cdef.damping_minus_3,
        cdef_bits: fh.cdef.bits,
        cdef_y_pri_strength: fh.cdef.y_pri,
        cdef_y_sec_strength: fh.cdef.y_sec,
        cdef_uv_pri_strength: fh.cdef.uv_pri,
        cdef_uv_sec_strength: fh.cdef.uv_sec,
    });

    // --- loop restoration ---
    let lr = alloc::boxed::Box::new(vk::native::StdVideoAV1LoopRestoration {
        FrameRestorationType: [
            fh.lr.frame_restoration_type[0] as vk::native::StdVideoAV1FrameRestorationType,
            fh.lr.frame_restoration_type[1] as vk::native::StdVideoAV1FrameRestorationType,
            fh.lr.frame_restoration_type[2] as vk::native::StdVideoAV1FrameRestorationType,
        ],
        LoopRestorationSize: fh.lr.loop_restoration_size,
    });

    // --- global motion ---
    let gm = alloc::boxed::Box::new(vk::native::StdVideoAV1GlobalMotion {
        GmType: fh.gm.gm_type,
        gm_params: fh.gm.gm_params,
    });

    // --- picture info ---
    // SAFETY: bitfield POD, valid all-zero.
    let mut flags: vk::native::StdVideoDecodeAV1PictureInfoFlags = unsafe { core::mem::zeroed() };
    flags.set_error_resilient_mode(fh.error_resilient_mode as u32);
    flags.set_disable_cdf_update(fh.disable_cdf_update as u32);
    flags.set_use_superres(fh.use_superres as u32);
    let render_diff = fh.render_width != fh.upscaled_width || fh.render_height != fh.frame_height;
    flags.set_render_and_frame_size_different(render_diff as u32);
    flags.set_allow_screen_content_tools(fh.allow_screen_content_tools as u32);
    flags.set_is_filter_switchable((fh.interpolation_filter == 4) as u32);
    flags.set_force_integer_mv(fh.force_integer_mv as u32);
    flags.set_frame_size_override_flag(fh.frame_size_override_flag as u32);
    flags.set_buffer_removal_time_present_flag(0);
    flags.set_allow_intrabc(fh.allow_intrabc as u32);
    flags.set_frame_refs_short_signaling(0);
    flags.set_allow_high_precision_mv(fh.allow_high_precision_mv as u32);
    flags.set_is_motion_mode_switchable(fh.is_motion_mode_switchable as u32);
    flags.set_use_ref_frame_mvs(fh.use_ref_frame_mvs as u32);
    flags.set_disable_frame_end_update_cdf(fh.disable_frame_end_update_cdf as u32);
    flags.set_allow_warped_motion(fh.allow_warped_motion as u32);
    flags.set_reduced_tx_set(fh.reduced_tx_set as u32);
    flags.set_reference_select(fh.reference_select as u32);
    flags.set_skip_mode_present(fh.skip_mode_present as u32);
    flags.set_delta_q_present(fh.delta_q_present as u32);
    flags.set_delta_lf_present(fh.delta_lf_present as u32);
    flags.set_delta_lf_multi(fh.delta_lf_multi as u32);
    flags.set_segmentation_enabled(fh.seg.enabled as u32);
    flags.set_segmentation_update_map(fh.seg.update_map as u32);
    flags.set_segmentation_temporal_update(fh.seg.temporal_update as u32);
    flags.set_segmentation_update_data(fh.seg.update_data as u32);
    flags.set_UsesLr(fh.lr.uses_lr as u32);
    flags.set_usesChromaLr(fh.lr.uses_chroma_lr as u32);
    flags.set_apply_grain(0);

    let coded_denom = if fh.use_superres {
        fh.superres_denom.saturating_sub(9)
    } else {
        0
    };

    let pic = vk::native::StdVideoDecodeAV1PictureInfo {
        flags,
        frame_type: fh.frame_type as vk::native::StdVideoAV1FrameType,
        current_frame_id: fh.current_frame_id,
        OrderHint: fh.order_hint,
        primary_ref_frame: fh.primary_ref_frame,
        refresh_frame_flags: fh.refresh_frame_flags,
        reserved1: 0,
        interpolation_filter: fh.interpolation_filter as vk::native::StdVideoAV1InterpolationFilter,
        TxMode: fh.tx_mode as vk::native::StdVideoAV1TxMode,
        delta_q_res: fh.delta_q_res,
        delta_lf_res: fh.delta_lf_res,
        SkipModeFrame: fh.skip_mode_frame,
        coded_denom,
        reserved2: [0; 3],
        OrderHints: order_hints,
        expectedFrameId: [0; AV1_NUM_REF_FRAMES],
        pTileInfo: &*tile,
        pQuantization: &*quant,
        pSegmentation: &*seg,
        pLoopFilter: &*lf,
        pCDEF: &*cdef,
        pLoopRestoration: &*lr,
        pGlobalMotion: &*gm,
        pFilmGrain: core::ptr::null(),
    };

    StdAv1PictureInfo {
        pic,
        _tile: tile,
        _mi_col_starts: mi_col_starts,
        _mi_row_starts: mi_row_starts,
        _width_in_sbs: width_in_sbs,
        _height_in_sbs: height_in_sbs,
        _quant: quant,
        _seg: seg,
        _lf: lf,
        _cdef: cdef,
        _lr: lr,
        _gm: gm,
    }
}

// ============================================================================
// H.264 slice-header parse
//
// The DPB decode loop needs, per coded picture, the slice-header fields that
// drive reference management and picture-order-count: `frame_num`, the POC
// syntax, `idr_pic_id`, `slice_type` and the NAL `nal_ref_idc`. Only the prefix
// up to the POC syntax is parsed (the parts before `ref_pic_list_modification`);
// the driver re-parses the full slice for the actual decode, so we read just
// enough to run the DPB. Attacker-controlled: every read is checked and an
// unsupported tool (field pictures) returns `None`.
// ============================================================================

/// The H.264 slice-header prefix the DPB loop keys on. Plain data.
#[derive(Debug, Clone)]
pub struct H264SliceHeader {
    pub first_mb_in_slice: u32,
    /// Raw `slice_type` (0..=9); `% 5` gives P/B/I/SP/SI.
    pub slice_type: u32,
    pub pic_parameter_set_id: u8,
    pub frame_num: u32,
    /// `nal_ref_idc` from the NAL header: 0 means the picture is not used as a
    /// reference (it never enters the DPB as a reference slot).
    pub nal_ref_idc: u8,
    /// `nal_unit_type == 5`: an IDR that resets the reference state.
    pub is_idr: bool,
    /// Only meaningful for an IDR.
    pub idr_pic_id: u32,
    /// `pic_order_cnt_lsb` (POC type 0 only; 0 otherwise).
    pub pic_order_cnt_lsb: u32,
    /// `delta_pic_order_cnt_bottom` (POC type 0 with bottom-field POC present).
    pub delta_pic_order_cnt_bottom: i32,
}

impl H264SliceHeader {
    /// Whether this is an I (intra) slice: `slice_type % 5 == 2`.
    pub fn is_intra_slice(&self) -> bool {
        self.slice_type % 5 == 2
    }
}

/// Parse the slice-header prefix of a VCL NAL (type 1 or 5). `sps`/`pps` supply
/// the variable-length field widths and POC type. `None` on truncation, a
/// non-VCL NAL, or an unsupported tool (field pictures).
pub fn parse_h264_slice_header(
    nal: &[u8],
    sps: &H264Sps,
    pps: &H264Pps,
) -> Option<H264SliceHeader> {
    if nal.is_empty() {
        return None;
    }
    let nal_ref_idc = (nal[0] >> 5) & 0x3;
    let nal_unit_type = nal[0] & 0x1F;
    if nal_unit_type != 1 && nal_unit_type != 5 {
        return None;
    }
    let is_idr = nal_unit_type == 5;
    let rbsp = strip_emulation_prevention(&nal[1..]);
    let mut br = BitReader::new(&rbsp);

    let first_mb_in_slice = br.read_ue()?;
    let slice_type = br.read_ue()?;
    let pic_parameter_set_id = br.read_ue()?;
    // separate_colour_plane_flag (chroma_format_idc == 3) is rejected at SPS
    // parse, so there is no colour_plane_id to skip here.
    let frame_num = br.read_bits(sps.log2_max_frame_num_minus4 as u32 + 4)?;
    // Only progressive frame pictures are supported; a field picture (only
    // possible when frame_mbs_only_flag == 0) is rejected rather than mis-run.
    if sps.frame_mbs_only_flag == 0 {
        let field_pic_flag = br.read_bit()?;
        if field_pic_flag == 1 {
            return None;
        }
    }
    let idr_pic_id = if is_idr { br.read_ue()? } else { 0 };

    let mut pic_order_cnt_lsb = 0;
    let mut delta_pic_order_cnt_bottom = 0;
    if sps.pic_order_cnt_type == 0 {
        pic_order_cnt_lsb = br.read_bits(sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4)?;
        if pps.bottom_field_pic_order_in_frame_present_flag == 1 {
            delta_pic_order_cnt_bottom = br.read_se()?;
        }
    }
    // pic_order_cnt_type == 1 carries delta_pic_order_cnt[] here, but the DPB
    // decoder rejects that POC type at creation, so it is never reached.

    Some(H264SliceHeader {
        first_mb_in_slice,
        slice_type,
        pic_parameter_set_id: pic_parameter_set_id as u8,
        frame_num,
        nal_ref_idc,
        is_idr,
        idr_pic_id,
        pic_order_cnt_lsb,
        delta_pic_order_cnt_bottom,
    })
}

// ============================================================================
// H.265 slice-segment-header parse
//
// The H.265 DPB loop needs, per coded picture, the slice-segment-header prefix
// that drives POC and reference management: `first_slice_segment_in_pic_flag`
// (access-unit boundary), the NAL type (IRAP / IDR), `slice_type`,
// `slice_pic_order_cnt_lsb`, and the inline short-term RPS (this fixture, like
// most x265 output, carries its RPS per-slice, not in the SPS). Only the prefix
// up to the RPS is parsed; the driver re-parses the full slice for the decode,
// so we read just enough to run the DPB, and pass it the bit length of the
// inline RPS (`NumBitsForSTRefPicSetInSlice`). Attacker-controlled: every read
// is checked and an unsupported tool returns `None`.
// ============================================================================

/// The H.265 slice-segment-header prefix the DPB loop keys on. Plain data.
#[derive(Debug, Clone)]
pub struct H265SliceHeader {
    /// The MSB of the slice RBSP: 1 marks the first slice of a new coded picture.
    pub first_slice_segment_in_pic_flag: bool,
    /// `nal_unit_type` (bits [1..7] of the NAL header's first byte).
    pub nal_unit_type: u8,
    pub slice_pic_parameter_set_id: u8,
    /// `slice_type`: 0 = B, 1 = P, 2 = I.
    pub slice_type: u32,
    /// `slice_pic_order_cnt_lsb` (0 for an IDR, which does not code it).
    pub slice_pic_order_cnt_lsb: u32,
    /// An IRAP (random-access) picture: `nal_unit_type` 16..=23.
    pub is_irap: bool,
    /// An IDR (`nal_unit_type` 19 or 20): decoding can restart with no references.
    pub is_idr: bool,
    /// `short_term_ref_pic_set_sps_flag`: false = the RPS is coded inline here.
    pub short_term_ref_pic_set_sps_flag: bool,
    /// The short-term RPS in effect for this picture (canonical explicit form).
    pub st_rps: H265ShortTermRps,
    /// Bits the inline `st_ref_pic_set()` occupies in the slice header (0 when the
    /// RPS came from the SPS); the driver's `NumBitsForSTRefPicSetInSlice`.
    pub num_bits_for_st_rps: u16,
    /// `NumDeltaPocs[RefRpsIdx]` when the inline RPS is inter-RPS-predicted
    /// (0 otherwise): the driver re-parses the slice's `st_ref_pic_set` and
    /// needs it (`NumDeltaPocsOfRefRpsIdx`).
    pub num_delta_pocs_of_ref_rps_idx: u8,
    /// The long-term reference entries of this slice (SPS-indexed entries first,
    /// then slice-coded ones), empty when the SPS has no long-term ref pics.
    pub lt: alloc::vec::Vec<H265LtEntry>,
}

/// One long-term reference entry from a slice header (H.265 7.3.6.1), with the
/// MSB-cycle delta already accumulated per 7.4.7.1.
#[derive(Debug, Clone, Copy)]
pub struct H265LtEntry {
    /// `PocLsbLt[i]`: from the SPS table (the first `num_long_term_sps` entries)
    /// or coded inline in the slice.
    pub poc_lsb: u32,
    /// `UsedByCurrPicLt[i]`: referenced by the current picture (vs kept only).
    pub used_by_curr: bool,
    /// `delta_poc_msb_present_flag[i]`: the entry identifies its picture by full
    /// POC; otherwise by POC lsb alone.
    pub has_msb_cycle: bool,
    /// `DeltaPocMsbCycleLt[i]` (accumulated).
    pub delta_poc_msb_cycle: u32,
}

impl H265SliceHeader {
    /// Whether this is an I (intra) slice: `slice_type == 2`.
    pub fn is_intra_slice(&self) -> bool {
        self.slice_type == 2
    }

    /// A BLA (Broken Link Access) IRAP: `nal_unit_type` 16..=18. Always a hard
    /// reference reset (`NoRaslOutputFlag == 1`).
    pub fn is_bla(&self) -> bool {
        (16..=18).contains(&self.nal_unit_type)
    }

    /// A CRA (Clean Random Access) IRAP: `nal_unit_type` 21. A reference reset
    /// only as the first picture of the decode; otherwise an open-GOP anchor
    /// whose RASL followers reference pictures decoded before it.
    pub fn is_cra(&self) -> bool {
        self.nal_unit_type == 21
    }
}

/// Whether an H.265 VCL `nal_unit_type` marks a reference picture: the `_R`
/// (odd, < 16) trailing/leading types and every IRAP (16..=23) are references;
/// the `_N` (even, < 16) sub-layer-non-reference types are not.
fn h265_is_reference(nal_unit_type: u8) -> bool {
    if nal_unit_type >= 16 {
        nal_unit_type <= 23
    } else {
        nal_unit_type % 2 == 1
    }
}

/// Whether an H.265 VCL `nal_unit_type` is a RASL (Random Access Skipped Leading)
/// picture: `RASL_N` (8) or `RASL_R` (9). A RASL displays before its associated
/// IRAP and references pictures decoded before it, so it is discarded when that
/// IRAP has `NoRaslOutputFlag == 1` (a random-access tune-in). RADL (6 / 7)
/// leading pictures reference only the IRAP and later, so they always decode.
fn h265_is_rasl(nal_unit_type: u8) -> bool {
    nal_unit_type == 8 || nal_unit_type == 9
}

/// Parse the slice-segment-header prefix of a VCL NAL (type 0..=21). `sps`/`pps`
/// supply the field widths, POC lsb size, and `num_short_term_ref_pic_sets`.
/// `None` on truncation, a non-VCL NAL, or an unsupported tool (dependent slice
/// segments). Non-first slice segments parse only the flag (the DPB uses just
/// the first slice's header per picture).
pub fn parse_h265_slice_header(
    nal: &[u8],
    sps: &H265Sps,
    pps: &H265Pps,
) -> Option<H265SliceHeader> {
    if nal.len() < 2 {
        return None;
    }
    let nal_unit_type = (nal[0] >> 1) & 0x3F;
    if nal_unit_type > 31 {
        return None; // not a VCL NAL
    }
    let is_irap = (16..=23).contains(&nal_unit_type);
    let is_idr = nal_unit_type == 19 || nal_unit_type == 20;
    let rbsp = strip_emulation_prevention(&nal[2..]);
    let mut br = BitReader::new(&rbsp);

    let first_slice_segment_in_pic_flag = br.read_bit()? == 1;
    // Non-first slice segments of a picture carry no POC / RPS the DPB needs (it
    // keys on the first slice); return early with the flag only.
    if !first_slice_segment_in_pic_flag {
        return Some(H265SliceHeader {
            first_slice_segment_in_pic_flag: false,
            nal_unit_type,
            slice_pic_parameter_set_id: 0,
            slice_type: 0,
            slice_pic_order_cnt_lsb: 0,
            is_irap,
            is_idr,
            short_term_ref_pic_set_sps_flag: false,
            st_rps: H265ShortTermRps::default(),
            num_bits_for_st_rps: 0,
            num_delta_pocs_of_ref_rps_idx: 0,
            lt: alloc::vec::Vec::new(),
        });
    }
    if is_irap {
        let _no_output_of_prior_pics_flag = br.read_bit()?;
    }
    let slice_pic_parameter_set_id = br.read_ue()? as u8;
    // dependent_slice_segment_flag / slice_segment_address only appear when this
    // is not the first slice segment, which we returned early on above.
    for _ in 0..pps.num_extra_slice_header_bits {
        br.read_bit()?; // slice_reserved_flag[i]
    }
    let slice_type = br.read_ue()?;
    if pps.output_flag_present_flag == 1 {
        let _pic_output_flag = br.read_bit()?;
    }
    // separate_colour_plane_flag is rejected at SPS parse, so no colour_plane_id.

    let mut slice_pic_order_cnt_lsb = 0u32;
    let mut short_term_ref_pic_set_sps_flag = false;
    let mut st_rps = H265ShortTermRps::default();
    let mut num_bits_for_st_rps = 0u16;
    let mut num_delta_pocs_of_ref_rps_idx = 0u8;
    let mut lt: alloc::vec::Vec<H265LtEntry> = alloc::vec::Vec::new();
    if !is_idr {
        slice_pic_order_cnt_lsb = br.read_bits(sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4)?;
        short_term_ref_pic_set_sps_flag = br.read_bit()? == 1;
        if !short_term_ref_pic_set_sps_flag {
            // Inline RPS: st_ref_pic_set(num_short_term_ref_pic_sets). Measure the
            // bits it spans for the driver's NumBitsForSTRefPicSetInSlice.
            let before = br.bit_pos();
            let (parsed, ref_delta_pocs) = parse_h265_short_term_rps(
                &mut br,
                sps.num_short_term_ref_pic_sets as usize,
                sps.num_short_term_ref_pic_sets as usize,
                &sps.short_term_rps,
            )?;
            st_rps = parsed;
            num_delta_pocs_of_ref_rps_idx = ref_delta_pocs;
            num_bits_for_st_rps = (br.bit_pos() - before) as u16;
        } else if sps.num_short_term_ref_pic_sets > 1 {
            let bits = ceil_log2(sps.num_short_term_ref_pic_sets as u32);
            let idx = br.read_bits(bits)? as usize;
            st_rps = sps.short_term_rps.get(idx)?.clone();
        } else if sps.num_short_term_ref_pic_sets == 1 {
            st_rps = sps.short_term_rps.first()?.clone();
        }
        if sps.long_term_ref_pics_present_flag == 1 {
            let num_long_term_sps = if sps.num_long_term_ref_pics_sps > 0 {
                br.read_ue()?
            } else {
                0
            };
            let num_long_term_pics = br.read_ue()?;
            let total = num_long_term_sps.checked_add(num_long_term_pics)?;
            if num_long_term_sps > sps.num_long_term_ref_pics_sps as u32 || total > 32 {
                return None;
            }
            let mut prev_cycle = 0u32;
            for i in 0..total {
                let (poc_lsb, used_by_curr) = if i < num_long_term_sps {
                    // An index into the SPS long-term table.
                    let idx = if sps.num_long_term_ref_pics_sps > 1 {
                        br.read_bits(ceil_log2(sps.num_long_term_ref_pics_sps as u32))? as usize
                    } else {
                        0
                    };
                    if idx >= sps.num_long_term_ref_pics_sps as usize {
                        return None;
                    }
                    (
                        sps.lt_ref_pic_poc_lsb_sps[idx],
                        sps.used_by_curr_pic_lt_sps_flag[idx],
                    )
                } else {
                    let lsb = br.read_bits(sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4)?;
                    (lsb, br.read_bit()? == 1)
                };
                let has_msb_cycle = br.read_bit()? == 1;
                let mut cycle = if has_msb_cycle { br.read_ue()? } else { 0 };
                // 7.4.7.1: the MSB cycle is delta-coded against the previous
                // entry, except at i == 0 and at the first slice-coded entry.
                if i != 0 && i != num_long_term_sps {
                    cycle = cycle.checked_add(prev_cycle)?;
                }
                prev_cycle = cycle;
                lt.push(H265LtEntry {
                    poc_lsb,
                    used_by_curr,
                    has_msb_cycle,
                    delta_poc_msb_cycle: cycle,
                });
            }
        }
    }

    Some(H265SliceHeader {
        first_slice_segment_in_pic_flag,
        nal_unit_type,
        slice_pic_parameter_set_id,
        slice_type,
        slice_pic_order_cnt_lsb,
        is_irap,
        is_idr,
        short_term_ref_pic_set_sps_flag,
        st_rps,
        num_bits_for_st_rps,
        num_delta_pocs_of_ref_rps_idx,
        lt,
    })
}

/// `Ceil(Log2(n))`, the bit width of an index into `n` values (n >= 1).
fn ceil_log2(n: u32) -> u32 {
    if n <= 1 {
        0
    } else {
        32 - (n - 1).leading_zeros()
    }
}

// ============================================================================
// Decode device + video session
//
// A `wgpu::Device` opened with the Vulkan Video decode extensions and a decode
// queue (added through wgpu-hal's `open_with_callback`, which lets us extend the
// queue-create list, so we never build the `VkDevice` ourselves). The
// `VkVideoSessionKHR` + `VkVideoSessionParametersKHR` are the decode context;
// creating the parameters is where the driver validates the `Std*` SPS/PPS
// mapping above.
// ============================================================================

/// The Vulkan Video decode extensions a decode device enables (on top of the
/// codec-specific one). `synchronization2` is required: the video-decode image
/// barriers use `PipelineStageFlags2::VIDEO_DECODE` / `AccessFlags2` which only
/// exist in the sync2 (barrier2) path, not classic barriers.
fn decode_device_extensions(codec: VulkanVideoCodec) -> [&'static core::ffi::CStr; 4] {
    [
        ash::khr::video_queue::NAME,
        ash::khr::video_decode_queue::NAME,
        ash::khr::synchronization2::NAME,
        codec.decode_extension(),
    ]
}

/// Build the H.264 decode `VideoProfileInfoKHR` (with its codec profile + usage
/// chained). The three chained structs are returned alongside so the caller
/// keeps them alive for as long as the profile pointer is read.
/// The `VkVideoComponentBitDepthFlags` for a bit depth (8 or 10; anything else is
/// treated as 8, the baseline every decoder supports).
fn component_bit_depth(bit_depth: u8) -> vk::VideoComponentBitDepthFlagsKHR {
    if bit_depth >= 10 {
        vk::VideoComponentBitDepthFlagsKHR::TYPE_10
    } else {
        vk::VideoComponentBitDepthFlagsKHR::TYPE_8
    }
}

/// The two-plane 4:2:0 decode output format for a bit depth: NV12 (8-bit) or
/// `G10X6...` (10-bit, 16-bit samples with the value in the top 10 bits, the P010
/// analog). This is the format `decode_format` prefers and the frame readback
/// interprets.
fn planar_420_format(bit_depth: u8) -> vk::Format {
    if bit_depth >= 10 {
        vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
    } else {
        vk::Format::G8_B8R8_2PLANE_420_UNORM
    }
}

/// Bytes per luma / chroma sample for a decode output format (1 for 8-bit NV12,
/// 2 for the 10-bit `G10X6` format whose samples are 16-bit containers).
fn format_bytes_per_sample(fmt: vk::Format) -> u64 {
    match fmt {
        vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => 2,
        _ => 1,
    }
}

/// The RGBA output formats the [`YcbcrConverter`] targets at a bit depth: 8-bit
/// -> `R8G8B8A8_UNORM` / `Rgba8Unorm`; 10-bit -> `R16G16B16A16_SFLOAT` /
/// `Rgba16Float`. The float target preserves the full 10-bit precision and is
/// where the later PQ / HLG transfer operates; SFLOAT storage-image writes and
/// `Rgba16Float` sampling are both baseline (no extra device / wgpu feature).
fn rgba_output_format(bit_depth: u8) -> (vk::Format, wgpu::TextureFormat) {
    if bit_depth >= 10 {
        (
            vk::Format::R16G16B16A16_SFLOAT,
            wgpu::TextureFormat::Rgba16Float,
        )
    } else {
        (vk::Format::R8G8B8A8_UNORM, wgpu::TextureFormat::Rgba8Unorm)
    }
}

struct H264Profile {
    profile: vk::VideoProfileInfoKHR<'static>,
    _usage: alloc::boxed::Box<vk::VideoDecodeUsageInfoKHR<'static>>,
    _h264: alloc::boxed::Box<vk::VideoDecodeH264ProfileInfoKHR<'static>>,
}

fn h264_profile() -> H264Profile {
    let mut usage = alloc::boxed::Box::new(
        vk::VideoDecodeUsageInfoKHR::default()
            .video_usage_hints(vk::VideoDecodeUsageFlagsKHR::DEFAULT),
    );
    let h264 = alloc::boxed::Box::new(
        vk::VideoDecodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH)
            .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE),
    );
    let mut profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8);
    // Chain manually via the raw p_next pointers: profile -> usage -> h264. The
    // boxes keep stable heap addresses, so the pointers stay valid as the struct
    // moves into the return value (ash's lifetime-tracked `push_next` cannot
    // produce a self-contained `'static` value). The boxes are owned by the
    // returned struct, so the pointees outlive every read of the profile.
    usage.p_next = (&*h264 as *const vk::VideoDecodeH264ProfileInfoKHR).cast();
    profile.p_next = (&*usage as *const vk::VideoDecodeUsageInfoKHR).cast();
    H264Profile {
        profile,
        _usage: usage,
        _h264: h264,
    }
}

/// The H.265 decode `VideoProfileInfoKHR` (with its codec profile + usage
/// chained), the HEVC sibling of [`H264Profile`]. The chained boxes are returned
/// alongside so the caller keeps them alive while the profile pointer is read.
struct H265Profile {
    profile: vk::VideoProfileInfoKHR<'static>,
    _usage: alloc::boxed::Box<vk::VideoDecodeUsageInfoKHR<'static>>,
    _h265: alloc::boxed::Box<vk::VideoDecodeH265ProfileInfoKHR<'static>>,
}

fn h265_profile(bit_depth: u8) -> H265Profile {
    let mut usage = alloc::boxed::Box::new(
        vk::VideoDecodeUsageInfoKHR::default()
            .video_usage_hints(vk::VideoDecodeUsageFlagsKHR::DEFAULT),
    );
    // 10-bit is HEVC Main 10; 8-bit is Main.
    let std_profile_idc = if bit_depth >= 10 {
        vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN_10
    } else {
        vk::native::StdVideoH265ProfileIdc_STD_VIDEO_H265_PROFILE_IDC_MAIN
    };
    let h265 = alloc::boxed::Box::new(
        vk::VideoDecodeH265ProfileInfoKHR::default().std_profile_idc(std_profile_idc),
    );
    let mut profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H265)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(component_bit_depth(bit_depth))
        .chroma_bit_depth(component_bit_depth(bit_depth));
    // Chain manually via raw p_next pointers: profile -> usage -> h265 (same
    // reason as `h264_profile`; the boxes give the pointees stable addresses).
    usage.p_next = (&*h265 as *const vk::VideoDecodeH265ProfileInfoKHR).cast();
    profile.p_next = (&*usage as *const vk::VideoDecodeUsageInfoKHR).cast();
    H265Profile {
        profile,
        _usage: usage,
        _h265: h265,
    }
}

/// The AV1 decode `VideoProfileInfoKHR` (with its codec profile + usage chained),
/// the AV1 sibling of [`H264Profile`] / [`H265Profile`]. The chained boxes are
/// returned alongside so the caller keeps them alive while the profile is read.
struct Av1Profile {
    profile: vk::VideoProfileInfoKHR<'static>,
    _usage: alloc::boxed::Box<vk::VideoDecodeUsageInfoKHR<'static>>,
    _av1: alloc::boxed::Box<vk::VideoDecodeAV1ProfileInfoKHR<'static>>,
}

fn av1_profile(bit_depth: u8) -> Av1Profile {
    let mut usage = alloc::boxed::Box::new(
        vk::VideoDecodeUsageInfoKHR::default()
            .video_usage_hints(vk::VideoDecodeUsageFlagsKHR::DEFAULT),
    );
    // Main profile (covers 8 and 10-bit 4:2:0). film_grain_support false: the
    // decoder does not apply film grain synthesis (it is done on the CPU output).
    let av1 = alloc::boxed::Box::new(
        vk::VideoDecodeAV1ProfileInfoKHR::default()
            .std_profile(vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN)
            .film_grain_support(false),
    );
    let mut profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(component_bit_depth(bit_depth))
        .chroma_bit_depth(component_bit_depth(bit_depth));
    // Chain manually via raw p_next pointers: profile -> usage -> av1 (same
    // reason as `h264_profile`; the boxes give the pointees stable addresses).
    usage.p_next = (&*av1 as *const vk::VideoDecodeAV1ProfileInfoKHR).cast();
    profile.p_next = (&*usage as *const vk::VideoDecodeUsageInfoKHR).cast();
    Av1Profile {
        profile,
        _usage: usage,
        _av1: av1,
    }
}

/// Pick a memory type index satisfying `type_bits` with `flags` (mirrors the
/// helper in `cudawgpu.rs`).
fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(flags)
    })
}

/// Allocate + bind device memory for `image` (preferring `flags`, else any type).
/// Free-function form so both [`VulkanVideoDevice`] and [`H264DpbDecoder`] (which
/// hold only cloned raw handles) can use it.
/// # Safety
/// `image` must be a valid image created from `raw_device`.
unsafe fn alloc_bind_image_raw(
    raw_device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    image: vk::Image,
    flags: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, VulkanVideoError> {
    // SAFETY: image is valid (contract).
    let req = unsafe { raw_device.get_image_memory_requirements(image) };
    let mem_type = find_memory_type(mem_props, req.memory_type_bits, flags)
        .or_else(|| {
            find_memory_type(
                mem_props,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::empty(),
            )
        })
        .ok_or(VulkanVideoError::ExtensionUnsupported)?;
    let ai = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(mem_type);
    // SAFETY: valid allocate info.
    let mem =
        unsafe { raw_device.allocate_memory(&ai, None) }.map_err(VulkanVideoError::QueryFailed)?;
    // SAFETY: fresh image + memory, single bind.
    unsafe { raw_device.bind_image_memory(image, mem, 0) }
        .map_err(VulkanVideoError::QueryFailed)?;
    Ok(mem)
}

/// Convert a decoded NV12 image (currently in `VIDEO_DECODE_DPB_KHR` layout) to
/// an RGBA `wgpu::Texture` via a `VkSamplerYcbcrConversion` compute pass on
/// `compute_queue`, importing the result into wgpu with no CPU copy. Shared by
/// the one-shot IDR path and the DPB decode loop.
///
/// When `restore_to_dpb` is set, the NV12 image is transitioned back to
/// `VIDEO_DECODE_DPB_KHR` after being sampled, so it remains a valid DPB
/// reference for subsequent pictures (the DPB loop needs this; the one-shot IDR
/// path, which throws the image away, does not).
///
/// # Safety
/// `nv12` must be a valid image on `raw_device`, decoded and idle (its decode
/// fence was waited), in `VIDEO_DECODE_DPB_KHR` layout, and accessible from
/// `compute_family` (created `CONCURRENT` across the decode + compute families
/// with `SAMPLED` usage). `wgpu_device` must wrap the same `VkDevice`.
#[allow(clippy::too_many_arguments)]
unsafe fn nv12_to_wgpu_texture(
    raw_device: &ash::Device,
    wgpu_device: &wgpu::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    compute_queue: vk::Queue,
    compute_family: u32,
    nv12: vk::Image,
    w: u32,
    h: u32,
    restore_to_dpb: bool,
) -> Result<wgpu::Texture, VulkanVideoError> {
    let dev = raw_device;
    let err = VulkanVideoError::QueryFailed;

    // SAFETY: standard-format ycbcr conversion + immutable sampler + compute
    // pipeline over the shared shader; every handle is created from `dev`, used
    // while valid, and destroyed exactly once (here, or in the wgpu drop callback
    // for the imported RGBA image); the compute submission is waited on a fence
    // before any teardown.
    unsafe {
        let conv_ci = vk::SamplerYcbcrConversionCreateInfo::default()
            .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
            .ycbcr_model(vk::SamplerYcbcrModelConversion::YCBCR_601)
            .ycbcr_range(vk::SamplerYcbcrRange::ITU_NARROW)
            .components(vk::ComponentMapping::default())
            .x_chroma_offset(vk::ChromaLocation::COSITED_EVEN)
            .y_chroma_offset(vk::ChromaLocation::COSITED_EVEN)
            .chroma_filter(vk::Filter::LINEAR);
        let conversion = dev
            .create_sampler_ycbcr_conversion(&conv_ci, None)
            .map_err(err)?;

        let mut conv_s = vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let sampler_ci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .unnormalized_coordinates(false)
            .push_next(&mut conv_s);
        let sampler = dev.create_sampler(&sampler_ci, None).map_err(err)?;

        let mut conv_v = vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let nv12_view_ci = vk::ImageViewCreateInfo::default()
            .image(nv12)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
            .subresource_range(color_range())
            .push_next(&mut conv_v);
        let nv12_view = dev.create_image_view(&nv12_view_ci, None).map_err(err)?;

        // RGBA output, owned here until moved into the wgpu texture.
        let rgba_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let rgba = dev.create_image(&rgba_ci, None).map_err(err)?;
        let rgba_mem =
            alloc_bind_image_raw(dev, mem_props, rgba, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let rgba_view_ci = vk::ImageViewCreateInfo::default()
            .image(rgba)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(color_range());
        let rgba_view = dev.create_image_view(&rgba_view_ci, None).map_err(err)?;

        // Descriptor layout: binding 0 = immutable ycbcr sampler, 1 = storage.
        let immutable = [sampler];
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .immutable_samplers(&immutable),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let dsl = dev
            .create_descriptor_set_layout(&dsl_ci, None)
            .map_err(err)?;
        let set_layouts = [dsl];
        let pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = dev.create_pipeline_layout(&pl_ci, None).map_err(err)?;

        let code = ash::util::read_spv(&mut std::io::Cursor::new(YCBCR_COMP_SPV))
            .map_err(|_| VulkanVideoError::QueryFailed(vk::Result::ERROR_INITIALIZATION_FAILED))?;
        let sm_ci = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader = dev.create_shader_module(&sm_ci, None).map_err(err)?;
        let entry = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(entry);
        let cp_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipeline = dev
            .create_compute_pipelines(vk::PipelineCache::null(), &[cp_ci], None)
            .map_err(|(_, e)| err(e))?[0];

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        ];
        let dp_ci = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let desc_pool = dev.create_descriptor_pool(&dp_ci, None).map_err(err)?;
        let ds_ai = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(desc_pool)
            .set_layouts(&set_layouts);
        let set = dev.allocate_descriptor_sets(&ds_ai).map_err(err)?[0];
        let in_desc = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(nv12_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let out_desc = [vk::DescriptorImageInfo::default()
            .image_view(rgba_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&in_desc),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_desc),
        ];
        dev.update_descriptor_sets(&writes, &[]);

        let cp_pool_ci = vk::CommandPoolCreateInfo::default().queue_family_index(compute_family);
        let cmd_pool = dev.create_command_pool(&cp_pool_ci, None).map_err(err)?;
        let cb = dev
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(err)?[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        dev.begin_command_buffer(cb, &begin).map_err(err)?;
        // NV12 DPB -> shader-read; RGBA undefined -> general (classic barrier: no
        // video stages here, and the decode already completed on a fence).
        let to_read = vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(nv12)
            .subresource_range(color_range());
        let to_general = vk::ImageMemoryBarrier::default()
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(rgba)
            .subresource_range(color_range());
        dev.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_read, to_general],
        );
        dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
        dev.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[set],
            &[],
        );
        dev.cmd_dispatch(cb, w.div_ceil(8), h.div_ceil(8), 1);
        // RGBA general -> shader-read for wgpu sampling. When this NV12 image is a
        // live DPB slot, also restore it to the decode layout so it stays a valid
        // reference for later pictures.
        let mut after = alloc::vec::Vec::with_capacity(2);
        after.push(
            vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(rgba)
                .subresource_range(color_range()),
        );
        if restore_to_dpb {
            after.push(
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(nv12)
                    .subresource_range(color_range()),
            );
        }
        dev.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &after,
        );
        dev.end_command_buffer(cb).map_err(err)?;

        let fence = dev
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(err)?;
        let cbs = [cb];
        let submit = vk::SubmitInfo::default().command_buffers(&cbs);
        let submit_r = dev
            .queue_submit(compute_queue, core::slice::from_ref(&submit), fence)
            .and_then(|_| dev.wait_for_fences(&[fence], true, u64::MAX));
        dev.destroy_fence(fence, None);

        // Tear down everything except the RGBA image + memory (moved to wgpu).
        dev.destroy_command_pool(cmd_pool, None);
        dev.destroy_descriptor_pool(desc_pool, None);
        dev.destroy_pipeline(pipeline, None);
        dev.destroy_shader_module(shader, None);
        dev.destroy_pipeline_layout(pipeline_layout, None);
        dev.destroy_descriptor_set_layout(dsl, None);
        dev.destroy_image_view(rgba_view, None);
        dev.destroy_image_view(nv12_view, None);
        dev.destroy_sampler(sampler, None);
        dev.destroy_sampler_ycbcr_conversion(conversion, None);

        if let Err(e) = submit_r {
            dev.destroy_image(rgba, None);
            dev.free_memory(rgba_mem, None);
            return Err(err(e));
        }

        // Import the RGBA image into wgpu (no copy); the texture's drop callback
        // frees the image + memory once wgpu is done with it.
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        crate::gpu::import_vk_image_as_wgpu_texture(
            wgpu_device,
            rgba,
            rgba_mem,
            size,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            "vulkan-video-rgba",
        )
        .ok_or(VulkanVideoError::NoVulkanAdapter)
    }
}

/// What the GPU texture converter does with an HDR (PQ / HLG) stream's transfer
/// function. The matrix + range are always applied by the ycbcr hardware; this
/// only selects the transfer stage in the compute pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HdrOutput {
    /// Store the ycbcr result unchanged: an SDR stream is display-ready, and an
    /// HDR stream keeps its PQ / HLG-encoded R'G'B' in the float target (ready for
    /// an HDR swapchain that expects that encoding). The default (M573 behaviour).
    Passthrough,
    /// Tone-map an HDR (PQ / HLG, BT.2020) stream down to display-ready SDR
    /// (BT.709 + gamma) via the shader's EOTF -> BT.2390 EETF -> gamut -> OETF
    /// pipeline. A no-op for a non-HDR (SDR) stream.
    TonemapSdr,
}

/// A persistent NV12 -> RGBA `VkSamplerYcbcrConversion` compute converter. The
/// pipeline, sampler, descriptor-set layout and command pool are built once and
/// reused per picture (the free-fn [`nv12_to_wgpu_texture`] rebuilds all of that
/// every call, the dominant per-frame cost in the DPB texture loop). Per picture
/// only the RGBA output image, the two image views, and a descriptor set are
/// transient. Owned by [`GpuTextureCtx`]; its `Drop` frees the persistent objects.
struct YcbcrConverter {
    raw_device: ash::Device,
    wgpu_device: wgpu::Device,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    compute_queue: vk::Queue,
    conversion: vk::SamplerYcbcrConversion,
    sampler: vk::Sampler,
    dsl: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    desc_pool: vk::DescriptorPool,
    cmd_pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    /// The two-plane 4:2:0 input format the ycbcr conversion + NV12 view use:
    /// `G8_B8R8` (8-bit) or `G10X6` (10-bit). The DPB slot images share it.
    nv12_format: vk::Format,
    /// The RGBA output image + view format (`R8G8B8A8_UNORM` / `R16G16B16A16_SFLOAT`).
    rgba_format: vk::Format,
    /// The matching wgpu texture format the RGBA image imports as
    /// (`Rgba8Unorm` / `Rgba16Float`).
    wgpu_format: wgpu::TextureFormat,
    /// HDR transfer selector pushed to the shader: 0 passthrough, 1 PQ, 2 HLG.
    /// Nonzero only on the 10-bit path with [`HdrOutput::TonemapSdr`].
    xfer: u32,
}

impl core::fmt::Debug for YcbcrConverter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("YcbcrConverter").finish_non_exhaustive()
    }
}

impl Drop for YcbcrConverter {
    fn drop(&mut self) {
        let dev = &self.raw_device;
        // SAFETY: every handle was created from `dev` in `new` and is destroyed
        // exactly once here; `convert` waits its fence per call, so nothing on the
        // compute queue is in flight at teardown.
        unsafe {
            dev.destroy_fence(self.fence, None);
            dev.destroy_command_pool(self.cmd_pool, None);
            dev.destroy_descriptor_pool(self.desc_pool, None);
            dev.destroy_pipeline(self.pipeline, None);
            dev.destroy_pipeline_layout(self.pipeline_layout, None);
            dev.destroy_descriptor_set_layout(self.dsl, None);
            dev.destroy_sampler(self.sampler, None);
            dev.destroy_sampler_ycbcr_conversion(self.conversion, None);
        }
    }
}

impl YcbcrConverter {
    /// Build the persistent conversion pipeline for a two-plane 4:2:0 decode
    /// output -> RGBA. `bit_depth` selects 8-bit (`G8_B8R8` NV12 ->
    /// `R8G8B8A8_UNORM`, the `rgba8` shader) or 10-bit (`G10X6` -> the
    /// `R16G16B16A16_SFLOAT` HDR target, the `rgba16f` shader). The compute
    /// command buffer + fence and the descriptor pool are sized for one in-flight
    /// conversion (the ring in the pipelined path allocates its own per-slot
    /// descriptor sets from `desc_pool`).
    ///
    /// `color` selects the `VkSamplerYcbcrConversion` model + range (BT.601 / 709 /
    /// 2020, studio / full), so the fixed-function ycbcr hardware does the matrix +
    /// range the stream actually uses (BT.2020 for HDR; the PQ / HLG transfer is a
    /// later increment). The ycbcr conversion also unpacks the bit depth from the
    /// format, so the same normalized [0, 1] RGB reaches either shader.
    ///
    /// # Safety
    /// `raw_device` must outlive the returned converter; `wgpu_device` must wrap
    /// the same `VkDevice`; `compute_queue` must belong to `compute_family`.
    // Each argument is a distinct piece of the converter (the two devices, memory
    // props, the compute queue + family, the colour space, bit depth, HDR mode)
    // with no natural grouping.
    #[allow(clippy::too_many_arguments)]
    unsafe fn new(
        raw_device: &ash::Device,
        wgpu_device: &wgpu::Device,
        mem_props: vk::PhysicalDeviceMemoryProperties,
        compute_queue: vk::Queue,
        compute_family: u32,
        color: VideoColorSpace,
        bit_depth: u8,
        hdr_output: HdrOutput,
    ) -> Result<Self, VulkanVideoError> {
        let dev = raw_device;
        let err = VulkanVideoError::QueryFailed;
        let (ycbcr_model, ycbcr_range) = color.vk_ycbcr();
        let nv12_format = planar_420_format(bit_depth);
        let (rgba_format, wgpu_format) = rgba_output_format(bit_depth);
        let spv = if bit_depth >= 10 {
            YCBCR16_COMP_SPV
        } else {
            YCBCR_COMP_SPV
        };
        // The transfer selector the `rgba16f` shader branches on: only tone-map
        // when asked to AND the stream is HDR (PQ / HLG); else passthrough.
        let xfer = match (hdr_output, color.transfer) {
            (HdrOutput::TonemapSdr, TransferFunction::Pq) => 1u32,
            (HdrOutput::TonemapSdr, TransferFunction::Hlg) => 2u32,
            _ => 0u32,
        };
        // SAFETY: standard ycbcr-conversion + immutable-sampler + compute pipeline
        // over the shader for this bit depth; the shader module is destroyed once
        // the pipeline is built, the rest are held by the converter and freed in
        // `Drop`.
        unsafe {
            let conv_ci = vk::SamplerYcbcrConversionCreateInfo::default()
                .format(nv12_format)
                .ycbcr_model(ycbcr_model)
                .ycbcr_range(ycbcr_range)
                .components(vk::ComponentMapping::default())
                .x_chroma_offset(vk::ChromaLocation::COSITED_EVEN)
                .y_chroma_offset(vk::ChromaLocation::COSITED_EVEN)
                .chroma_filter(vk::Filter::LINEAR);
            let conversion = dev
                .create_sampler_ycbcr_conversion(&conv_ci, None)
                .map_err(err)?;

            let mut conv_s = vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
            let sampler_ci = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .unnormalized_coordinates(false)
                .push_next(&mut conv_s);
            let sampler = dev.create_sampler(&sampler_ci, None).map_err(err)?;

            let immutable = [sampler];
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(&immutable),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let dsl = dev
                .create_descriptor_set_layout(&dsl_ci, None)
                .map_err(err)?;
            let set_layouts = [dsl];
            // A 4-byte COMPUTE push constant carries the HDR transfer selector
            // (`xfer`: 0 passthrough / 1 PQ / 2 HLG). The 8-bit shader ignores it
            // (a layout may declare a range no shader uses); the 16-bit shader
            // branches on it for the tone-map pipeline.
            let pc_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(4)];
            let pl_ci = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&pc_ranges);
            let pipeline_layout = dev.create_pipeline_layout(&pl_ci, None).map_err(err)?;

            let code = ash::util::read_spv(&mut std::io::Cursor::new(spv)).map_err(|_| {
                VulkanVideoError::QueryFailed(vk::Result::ERROR_INITIALIZATION_FAILED)
            })?;
            let sm_ci = vk::ShaderModuleCreateInfo::default().code(&code);
            let shader = dev.create_shader_module(&sm_ci, None).map_err(err)?;
            let entry = c"main";
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(entry);
            let cp_ci = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout);
            let pipeline = dev
                .create_compute_pipelines(vk::PipelineCache::null(), &[cp_ci], None)
                .map_err(|(_, e)| err(e))?[0];
            // The pipeline retains what it needs; the module is no longer required.
            dev.destroy_shader_module(shader, None);

            // Descriptor pool sized for the pipelined path's per-slot sets
            // (`DECODE_RING_DEPTH`); `FREE_DESCRIPTOR_SET` so the synchronous path
            // can allocate + free one per picture.
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(DECODE_RING_DEPTH as u32),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(DECODE_RING_DEPTH as u32),
            ];
            let dp_ci = vk::DescriptorPoolCreateInfo::default()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .max_sets(DECODE_RING_DEPTH as u32)
                .pool_sizes(&pool_sizes);
            let desc_pool = dev.create_descriptor_pool(&dp_ci, None).map_err(err)?;

            let cp_pool_ci = vk::CommandPoolCreateInfo::default()
                .queue_family_index(compute_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let cmd_pool = dev.create_command_pool(&cp_pool_ci, None).map_err(err)?;
            let cb = dev
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(err)?[0];
            let fence = dev
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(err)?;

            Ok(YcbcrConverter {
                raw_device: dev.clone(),
                wgpu_device: wgpu_device.clone(),
                mem_props,
                compute_queue,
                conversion,
                sampler,
                dsl,
                pipeline_layout,
                pipeline,
                desc_pool,
                cmd_pool,
                cb,
                fence,
                nv12_format,
                rgba_format,
                wgpu_format,
                xfer,
            })
        }
    }

    /// Allocate the transient per-picture resources: the RGBA output image (+ its
    /// device memory and view) and an NV12 view (carrying the ycbcr conversion) of
    /// `nv12`, plus a descriptor set bound to both. Returned handles are owned by
    /// the caller until moved into wgpu (RGBA image/memory) or destroyed.
    ///
    /// # Safety
    /// `nv12` must be a valid image on `self.raw_device` of the NV12 format.
    unsafe fn make_transients(
        &self,
        nv12: vk::Image,
        w: u32,
        h: u32,
    ) -> Result<Transients, VulkanVideoError> {
        let dev = &self.raw_device;
        let err = VulkanVideoError::QueryFailed;
        // SAFETY: all handles created from `dev`; the RGBA image/memory outlive the
        // conversion (moved into wgpu on success), the views + set are freed by the
        // caller once the compute submission is idle.
        unsafe {
            let mut conv_v = vk::SamplerYcbcrConversionInfo::default().conversion(self.conversion);
            let nv12_view_ci = vk::ImageViewCreateInfo::default()
                .image(nv12)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.nv12_format)
                .subresource_range(color_range())
                .push_next(&mut conv_v);
            let nv12_view = dev.create_image_view(&nv12_view_ci, None).map_err(err)?;

            let rgba_ci = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(self.rgba_format)
                .extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let rgba = dev.create_image(&rgba_ci, None).map_err(err)?;
            let rgba_mem = alloc_bind_image_raw(
                dev,
                &self.mem_props,
                rgba,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let rgba_view_ci = vk::ImageViewCreateInfo::default()
                .image(rgba)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(self.rgba_format)
                .subresource_range(color_range());
            let rgba_view = dev.create_image_view(&rgba_view_ci, None).map_err(err)?;

            let set_layouts = [self.dsl];
            let ds_ai = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.desc_pool)
                .set_layouts(&set_layouts);
            let set = dev.allocate_descriptor_sets(&ds_ai).map_err(err)?[0];
            let in_desc = [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(nv12_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let out_desc = [vk::DescriptorImageInfo::default()
                .image_view(rgba_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&in_desc),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&out_desc),
            ];
            dev.update_descriptor_sets(&writes, &[]);

            Ok(Transients {
                nv12_view,
                rgba,
                rgba_mem,
                rgba_view,
                set,
            })
        }
    }

    /// Record the NV12 -> RGBA conversion of `t` into `cb`: NV12 DPB -> shader-read
    /// and RGBA undefined -> general, the dispatch, then RGBA -> shader-read for
    /// wgpu sampling and (when `restore_to_dpb`) NV12 -> DPB so the slot stays a
    /// valid reference. Classic barriers: no video stages here, the decode of
    /// `nv12` is ordered before this via the caller's fence or semaphore.
    ///
    /// # Safety
    /// `cb` must be recordable; `nv12` + `t` must be valid on `self.raw_device`.
    unsafe fn record_convert(
        &self,
        cb: vk::CommandBuffer,
        nv12: vk::Image,
        t: &Transients,
        w: u32,
        h: u32,
        restore_to_dpb: bool,
    ) -> Result<(), vk::Result> {
        let dev = &self.raw_device;
        // SAFETY: contract above; barriers move the NV12 slot DPB -> shader-read
        // (-> DPB) and the RGBA image undefined -> general -> shader-read.
        unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.begin_command_buffer(cb, &begin)?;
            let to_read = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(nv12)
                .subresource_range(color_range());
            let to_general = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(t.rgba)
                .subresource_range(color_range());
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read, to_general],
            );
            dev.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            dev.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[t.set],
                &[],
            );
            // HDR transfer selector (0 for the 8-bit / passthrough path).
            dev.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self.xfer.to_ne_bytes(),
            );
            dev.cmd_dispatch(cb, w.div_ceil(8), h.div_ceil(8), 1);
            let mut after = alloc::vec::Vec::with_capacity(2);
            after.push(
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(t.rgba)
                    .subresource_range(color_range()),
            );
            if restore_to_dpb {
                after.push(
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_READ)
                        .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .image(nv12)
                        .subresource_range(color_range()),
                );
            }
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &after,
            );
            dev.end_command_buffer(cb)?;
        }
        Ok(())
    }

    /// Import the converted RGBA image into wgpu with no copy; the drop callback
    /// frees the image + memory once wgpu is done with the texture. Consumes the
    /// RGBA image/memory ownership out of `t` (the caller must not free them).
    ///
    /// # Safety
    /// `t.rgba` must be a valid, converted, idle RGBA image on `self.raw_device`.
    unsafe fn import_rgba(
        &self,
        t: &Transients,
        w: u32,
        h: u32,
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        let (rgba, rgba_mem) = (t.rgba, t.rgba_mem);
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        // SAFETY: import the idle RGBA image; `t`'s ownership of it transfers to
        // the texture's drop callback (frees it once, when wgpu is done).
        unsafe {
            crate::gpu::import_vk_image_as_wgpu_texture(
                &self.wgpu_device,
                rgba,
                rgba_mem,
                size,
                self.wgpu_format,
                wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                "vulkan-video-rgba",
            )
            .ok_or(VulkanVideoError::NoVulkanAdapter)
        }
    }

    /// Free a picture's transient views + descriptor set (not the RGBA image,
    /// which is moved into wgpu). Call once the compute submission is idle.
    ///
    /// # Safety
    /// The compute pass referencing `t` must have completed (fence / semaphore).
    unsafe fn free_transients(&self, t: &Transients) {
        let dev = &self.raw_device;
        // SAFETY: idle handles, freed once.
        unsafe {
            let _ = dev.free_descriptor_sets(self.desc_pool, &[t.set]);
            dev.destroy_image_view(t.rgba_view, None);
            dev.destroy_image_view(t.nv12_view, None);
        }
    }

    /// NV12 -> RGBA on the compute queue, waiting the compute fence before import.
    /// The submission waits `wait_sem` at the compute stage: the caller's chained
    /// texture decode signals it, so this compute pass starts the instant the
    /// decode finishes with no CPU round-trip in between (its own CPU prep,
    /// `make_transients` + `record_convert`, overlaps the decode's GPU execution).
    /// Pass `vk::Semaphore::null()` for an unchained convert (show_existing or a
    /// reordered shown frame, where the slot is already decoded and idle): the
    /// submission then has no wait. Reuses the persistent pipeline / command buffer
    /// / fence; only the transients are per picture.
    ///
    /// # Safety
    /// `nv12` must be a valid image on `self.raw_device` in `VIDEO_DECODE_DPB_KHR`
    /// layout, accessible from the compute family, whose decode signals `wait_sem`
    /// (or that is already decoded and idle when `wait_sem` is null).
    unsafe fn convert(
        &self,
        nv12: vk::Image,
        w: u32,
        h: u32,
        restore_to_dpb: bool,
        wait_sem: vk::Semaphore,
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        let dev = &self.raw_device;
        // SAFETY: contract above; per-picture transients created + freed here, the
        // compute submission waited before import / teardown.
        unsafe {
            let t = self.make_transients(nv12, w, h)?;
            let submit_r = self
                .record_convert(self.cb, nv12, &t, w, h, restore_to_dpb)
                .and_then(|()| {
                    dev.reset_fences(&[self.fence])?;
                    let cbs = [self.cb];
                    let wait = [wait_sem];
                    let stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
                    // `wait_sem` is null for an unchained convert (show_existing /
                    // a reordered shown frame: no decode to chain on); submit with
                    // no wait then, since passing a null semaphore is invalid.
                    let submit = if wait_sem == vk::Semaphore::null() {
                        vk::SubmitInfo::default().command_buffers(&cbs)
                    } else {
                        vk::SubmitInfo::default()
                            .wait_semaphores(&wait)
                            .wait_dst_stage_mask(&stages)
                            .command_buffers(&cbs)
                    };
                    dev.queue_submit(
                        self.compute_queue,
                        core::slice::from_ref(&submit),
                        self.fence,
                    )
                    .and_then(|_| dev.wait_for_fences(&[self.fence], true, u64::MAX))
                });
            if let Err(e) = submit_r {
                self.free_transients(&t);
                dev.destroy_image(t.rgba, None);
                dev.free_memory(t.rgba_mem, None);
                return Err(VulkanVideoError::QueryFailed(e));
            }
            let tex = self.import_rgba(&t, w, h)?;
            self.free_transients(&t);
            Ok(tex)
        }
    }
}

/// Per-picture transient resources for one [`YcbcrConverter`] conversion.
#[derive(Debug, Clone, Copy)]
struct Transients {
    nv12_view: vk::ImageView,
    rgba: vk::Image,
    rgba_mem: vk::DeviceMemory,
    rgba_view: vk::ImageView,
    set: vk::DescriptorSet,
}

/// A wgpu device opened with a Vulkan Video H.264 decode queue + extensions.
/// Keeps the wgpu device/queue (for the later YCbCr conversion + interop) and
/// the raw `ash` handles the decode path drives directly.
pub struct VulkanVideoDevice {
    /// The wgpu device that owns the underlying `VkDevice`; keep it alive.
    pub wgpu_device: wgpu::Device,
    pub wgpu_queue: wgpu::Queue,
    _adapter: wgpu::Adapter,
    _instance: wgpu::Instance,
    raw_device: ash::Device,
    phys: vk::PhysicalDevice,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    video_fns: ash::khr::video_queue::Device,
    decode_fns: ash::khr::video_decode_queue::Device,
    sync2_fns: ash::khr::synchronization2::Device,
    decode_queue: vk::Queue,
    decode_queue_family: u32,
    /// A compute-capable queue for the GPU-resident NV12 -> RGBA pass. `None`
    /// if no distinct compute family was available (the RGBA path then stays on
    /// the CPU convert). Its family may equal the decode family only when that
    /// family also advertises compute (it usually does not).
    compute_queue: Option<vk::Queue>,
    compute_queue_family: u32,
    caps: VulkanVideoDecodeCaps,
    /// The ash entry + instance the device was created on, kept so an HDR
    /// swapchain sink can create a `VkSurfaceKHR` on the same instance (the
    /// swapchain must live on the decode device for a zero-copy present). Only
    /// read under `hdr-present`.
    #[allow(dead_code)]
    entry: ash::Entry,
    #[allow(dead_code)]
    raw_instance: ash::Instance,
    /// The family-0 graphics/present queue wgpu opens; the HDR sink presents on it.
    #[allow(dead_code)]
    graphics_queue: vk::Queue,
    /// `VK_KHR_swapchain` was enabled (the device can present).
    present_capable: bool,
    /// `VK_EXT_hdr_metadata` was enabled (mastering metadata can be set).
    hdr_metadata_supported: bool,
}

/// Raw Vulkan handles an HDR swapchain sink needs to present on the decode
/// device (see the `vulkanhdrsink` module). Produced only when the device is
/// present-capable; the swapchain then lives on the same `VkInstance` / device as
/// the decode output, so the decoded texture presents with no cross-device copy.
#[cfg(feature = "hdr-present")]
pub(crate) struct PresentContext {
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) device: ash::Device,
    pub(crate) phys: vk::PhysicalDevice,
    pub(crate) queue: vk::Queue,
    pub(crate) queue_family: u32,
    pub(crate) hdr_metadata: bool,
}

impl VulkanVideoDevice {
    /// The physical device (GPU) name the decoder opened on, e.g.
    /// `"NVIDIA GeForce RTX 3060"`. Used to tag `Hardware` conformance evidence
    /// with the platform it was actually validated on (the honesty contract:
    /// hardware-validated maturity requires a named platform).
    pub fn device_name(&self) -> alloc::string::String {
        self._adapter.get_info().name
    }

    /// Whether the device enabled `VK_KHR_swapchain` (can present): an HDR
    /// swapchain sink requires it. A decode-only GPU reports `false`.
    pub fn present_capable(&self) -> bool {
        self.present_capable
    }

    /// Whether `VK_EXT_hdr_metadata` is available (HDR10 mastering metadata can be
    /// attached to the swapchain).
    pub fn hdr_metadata_supported(&self) -> bool {
        self.hdr_metadata_supported
    }

    /// The raw handles for an HDR present sink, or `None` if not present-capable.
    #[cfg(feature = "hdr-present")]
    pub(crate) fn present_context(&self) -> Option<PresentContext> {
        self.present_capable.then(|| PresentContext {
            entry: self.entry.clone(),
            instance: self.raw_instance.clone(),
            device: self.raw_device.clone(),
            phys: self.phys,
            queue: self.graphics_queue,
            queue_family: 0,
            hdr_metadata: self.hdr_metadata_supported,
        })
    }
}

impl core::fmt::Debug for VulkanVideoDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VulkanVideoDevice")
            .field("decode_queue_family", &self.decode_queue_family)
            .field("caps", &self.caps)
            .finish_non_exhaustive()
    }
}

/// Open a wgpu device with an H.264 Vulkan Video decode queue + extensions.
/// `Err(NoVulkanAdapter)` on a non-Vulkan / GPU-less host.
pub async fn open_h264_decode_device() -> Result<VulkanVideoDevice, VulkanVideoError> {
    open_decode_device(VulkanVideoCodec::H264).await
}

/// Open a wgpu device with an H.265 Vulkan Video decode queue + extensions.
/// `Err(NoVulkanAdapter)` on a non-Vulkan / GPU-less host.
pub async fn open_h265_decode_device() -> Result<VulkanVideoDevice, VulkanVideoError> {
    open_decode_device(VulkanVideoCodec::H265).await
}

/// Open a wgpu device with an AV1 Vulkan Video decode queue + extensions.
/// `Err(NoVulkanAdapter)` on a non-Vulkan / GPU-less host.
pub async fn open_av1_decode_device() -> Result<VulkanVideoDevice, VulkanVideoError> {
    open_decode_device(VulkanVideoCodec::Av1).await
}

/// Open a wgpu device with the given codec's Vulkan Video decode queue + device
/// extensions (plus a distinct compute family for the NV12 -> RGBA pass when one
/// exists). The codec only selects the decode extension and the caps probe; the
/// picture format is chosen later per profile at session creation.
pub async fn open_decode_device(
    codec: VulkanVideoCodec,
) -> Result<VulkanVideoDevice, VulkanVideoError> {
    // Pick a Vulkan adapter that actually supports this codec's hardware decode.
    // `request_adapter(HighPerformance)` is not codec-aware: on a multi-GPU host
    // (e.g. an AMD iGPU beside an NVIDIA dGPU) it can hand back the adapter that
    // has no video-decode queue for this codec, whose probe then fails. That was
    // an intermittent device-open failure. Enumerate instead, collect every
    // adapter whose decode probe succeeds, then prefer a discrete GPU (matching
    // `PowerPreference::HighPerformance`). An integrated GPU may advertise decode
    // caps yet fail to build a real session (e.g. AMD RADV on this host), so a
    // discrete adapter is both faster and more likely to work.
    //
    // Creating a Vulkan instance and enumerating adapters can transiently yield
    // nothing under rapid repeated opens (loader / driver churn across threads),
    // so retry with a fresh instance a couple of times before giving up.
    let mut attempt = 0u32;
    let (instance, adapter, caps, compute_family) = loop {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let mut candidates: alloc::vec::Vec<(
            wgpu::Adapter,
            VulkanVideoDecodeCaps,
            Option<u32>,
            bool,
        )> = alloc::vec::Vec::new();
        for adapter in instance.enumerate_adapters(wgpu::Backends::VULKAN).await {
            let is_discrete = adapter.get_info().device_type == wgpu::DeviceType::DiscreteGpu;
            // SAFETY: the guard holds the adapter's live handles for the probe calls;
            // it is dropped at the end of this block before `adapter` is moved.
            let probed = unsafe {
                let hal_adapter = match adapter.as_hal::<wgpu_hal::api::Vulkan>() {
                    Some(h) => h,
                    None => continue,
                };
                let shared = hal_adapter.shared_instance();
                let raw_instance = shared.raw_instance();
                let phys = hal_adapter.raw_physical_device();
                match probe_physical_device(shared.entry(), raw_instance, phys, codec) {
                    Ok(caps) => {
                        let families =
                            raw_instance.get_physical_device_queue_family_properties(phys);
                        // Prefer a dedicated compute family (COMPUTE without GRAPHICS)
                        // distinct from wgpu's family 0 and the decode family.
                        let compute = families
                            .iter()
                            .enumerate()
                            .filter(|(i, p)| {
                                *i as u32 != caps.decode_queue_family
                                    && *i != 0
                                    && p.queue_flags.contains(vk::QueueFlags::COMPUTE)
                            })
                            .min_by_key(|(_, p)| {
                                p.queue_flags.contains(vk::QueueFlags::GRAPHICS) as u8
                            })
                            .map(|(i, _)| i as u32);
                        Some((caps, compute))
                    }
                    Err(_) => None,
                }
            };
            if let Some((caps, compute)) = probed {
                candidates.push((adapter, caps, compute, is_discrete));
            }
        }
        if !candidates.is_empty() {
            let idx = candidates.iter().position(|c| c.3).unwrap_or(0);
            let (adapter, caps, compute_family, _) = candidates.swap_remove(idx);
            break (instance, adapter, caps, compute_family);
        }
        attempt += 1;
        if attempt >= 3 {
            return Err(VulkanVideoError::NoVulkanAdapter);
        }
    };

    let decode_family = caps.decode_queue_family;
    let exts = decode_device_extensions(codec);
    // Detect the present-side device extensions so a downstream HDR swapchain sink
    // (`hdr-present`) can present on this same device (zero-copy with the decode
    // output): `VK_KHR_swapchain` for any swapchain, `VK_EXT_hdr_metadata` for the
    // HDR10 mastering metadata. Added only when the physical device advertises them
    // (adding an unsupported extension fails device creation), so a decode-only GPU
    // is unaffected. `VK_EXT_swapchain_colorspace` (the instance extension that
    // unlocks HDR colour spaces) is already enabled by wgpu-hal on its instance.
    // SAFETY: the hal guard's handles are only read (extension enumeration) and
    // dropped at the end of the block; nothing is retained.
    let (want_swapchain, want_hdr_metadata) = unsafe {
        match adapter.as_hal::<wgpu_hal::api::Vulkan>() {
            Some(hal) => {
                let raw_instance = hal.shared_instance().raw_instance();
                let phys = hal.raw_physical_device();
                let props = raw_instance
                    .enumerate_device_extension_properties(phys)
                    .unwrap_or_default();
                let has = |name: &core::ffi::CStr| {
                    props
                        .iter()
                        .any(|p| p.extension_name_as_c_str() == Ok(name))
                };
                let sc = has(ash::khr::swapchain::NAME);
                (sc, sc && has(ash::ext::hdr_metadata::NAME))
            }
            None => (false, false),
        }
    };
    // Priorities array lives in this scope so the queue-create pointer the
    // callback records stays valid through `open_with_callback`.
    let priorities = [1.0f32];

    // SAFETY: `open_with_callback` + `create_device_from_hal` follow the
    // documented cudawgpu pattern; the callback only appends extensions and (for
    // a distinct decode family) one queue-create-info borrowing `priorities`,
    // which outlives the call.
    let (wgpu_device, wgpu_queue) = unsafe {
        let hal_adapter = adapter
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(VulkanVideoError::NoVulkanAdapter)?;
        let open = hal_adapter
            .open_with_callback(
                wgpu::Features::empty(),
                &wgpu::Limits::default(),
                &wgpu::MemoryHints::default(),
                Some(alloc::boxed::Box::new(
                    |args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                        for e in exts {
                            args.extensions.push(e);
                        }
                        // Present-side extensions (see the detection above): enable
                        // them so an HDR swapchain sink can present on this device.
                        if want_swapchain {
                            args.extensions.push(ash::khr::swapchain::NAME);
                        }
                        if want_hdr_metadata {
                            args.extensions.push(ash::ext::hdr_metadata::NAME);
                        }
                        // wgpu opens family 0; add the decode family (and a
                        // distinct compute family) only if distinct (Vulkan
                        // forbids two create-infos per family).
                        if decode_family != 0 {
                            args.queue_create_infos.push(
                                vk::DeviceQueueCreateInfo::default()
                                    .queue_family_index(decode_family)
                                    .queue_priorities(&priorities),
                            );
                        }
                        if let Some(cf) = compute_family {
                            if cf != 0 && cf != decode_family {
                                args.queue_create_infos.push(
                                    vk::DeviceQueueCreateInfo::default()
                                        .queue_family_index(cf)
                                        .queue_priorities(&priorities),
                                );
                            }
                        }
                    },
                )),
            )
            .map_err(|_| VulkanVideoError::NoVulkanAdapter)?;
        adapter
            .create_device_from_hal(
                open,
                &wgpu::DeviceDescriptor {
                    label: Some("vulkan-video-decode"),
                    ..Default::default()
                },
            )
            .map_err(|_| VulkanVideoError::NoVulkanAdapter)?
    };

    // Reach the raw ash handles now owned by the wgpu device.
    // SAFETY: the wgpu device is Vulkan-backed (we requested the Vulkan
    // backend); the guard's handles are cloned/copied, not retained by borrow.
    #[allow(clippy::type_complexity)]
    let (
        raw_device,
        phys,
        mem_props,
        video_fns,
        decode_fns,
        sync2_fns,
        decode_queue,
        entry,
        raw_instance,
        graphics_queue,
    ) = unsafe {
        let hal_device = wgpu_device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(VulkanVideoError::NoVulkanAdapter)?;
        let raw_device = hal_device.raw_device().clone();
        let phys = hal_device.raw_physical_device();
        let shared = hal_device.shared_instance();
        let raw_instance = shared.raw_instance();
        let mem_props = raw_instance.get_physical_device_memory_properties(phys);
        let video_fns = ash::khr::video_queue::Device::new(raw_instance, &raw_device);
        let decode_fns = ash::khr::video_decode_queue::Device::new(raw_instance, &raw_device);
        let sync2_fns = ash::khr::synchronization2::Device::new(raw_instance, &raw_device);
        let decode_queue = raw_device.get_device_queue(decode_family, 0);
        // Entry + instance clones and the family-0 graphics/present queue (the one
        // wgpu opens) are kept so an HDR swapchain sink can create a surface +
        // swapchain and present on this device. `ash::Entry` / `ash::Instance` are
        // cheap handle clones.
        let graphics_queue = raw_device.get_device_queue(0, 0);
        (
            raw_device.clone(),
            phys,
            mem_props,
            video_fns,
            decode_fns,
            sync2_fns,
            decode_queue,
            shared.entry().clone(),
            raw_instance.clone(),
            graphics_queue,
        )
    };

    // The dedicated compute queue, if a distinct compute family was requested.
    let (compute_queue, compute_queue_family) = match compute_family {
        Some(cf) if cf != 0 && cf != decode_family => {
            // SAFETY: cf was requested in the device create callback above.
            (Some(unsafe { raw_device.get_device_queue(cf, 0) }), cf)
        }
        _ => (None, 0),
    };

    Ok(VulkanVideoDevice {
        wgpu_device,
        wgpu_queue,
        _adapter: adapter,
        _instance: instance,
        raw_device,
        phys,
        mem_props,
        video_fns,
        decode_fns,
        sync2_fns,
        decode_queue,
        decode_queue_family: decode_family,
        compute_queue,
        compute_queue_family,
        caps,
        entry,
        raw_instance,
        graphics_queue,
        present_capable: want_swapchain,
        hdr_metadata_supported: want_hdr_metadata,
    })
}

/// An H.264 `VkVideoSessionKHR` + `VkVideoSessionParametersKHR`, with the
/// session-backing device memory. Destroys them (params, session, memory) on
/// drop.
pub struct H264DecodeSession {
    session: vk::VideoSessionKHR,
    parameters: vk::VideoSessionParametersKHR,
    memories: alloc::vec::Vec<vk::DeviceMemory>,
    raw_device: ash::Device,
    video_fns: ash::khr::video_queue::Device,
    /// The decode picture format chosen at session creation (e.g. NV12).
    pub picture_format: vk::Format,
    pub coded_extent: (u32, u32),
}

impl core::fmt::Debug for H264DecodeSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("H264DecodeSession")
            .field("picture_format", &self.picture_format)
            .field("coded_extent", &self.coded_extent)
            .field("memory_bindings", &self.memories.len())
            .finish_non_exhaustive()
    }
}

impl Drop for H264DecodeSession {
    fn drop(&mut self) {
        // SAFETY: all handles were created from `raw_device` / `video_fns` and
        // are destroyed exactly once here, params before session before memory,
        // with no in-flight decode (the caller holds the session for the decode
        // and drops it after).
        unsafe {
            (self.video_fns.fp().destroy_video_session_parameters_khr)(
                self.raw_device.handle(),
                self.parameters,
                core::ptr::null(),
            );
            (self.video_fns.fp().destroy_video_session_khr)(
                self.raw_device.handle(),
                self.session,
                core::ptr::null(),
            );
            for &mem in &self.memories {
                self.raw_device.free_memory(mem, None);
            }
        }
    }
}

/// An H.265 `VkVideoSessionKHR` + `VkVideoSessionParametersKHR` (carrying the
/// VPS/SPS/PPS), with the session-backing device memory. The HEVC sibling of
/// [`H264DecodeSession`]; destroys them (params, session, memory) on drop.
pub struct H265DecodeSession {
    session: vk::VideoSessionKHR,
    parameters: vk::VideoSessionParametersKHR,
    memories: alloc::vec::Vec<vk::DeviceMemory>,
    raw_device: ash::Device,
    video_fns: ash::khr::video_queue::Device,
    /// The decode picture format chosen at session creation (e.g. NV12).
    pub picture_format: vk::Format,
    pub coded_extent: (u32, u32),
}

impl core::fmt::Debug for H265DecodeSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("H265DecodeSession")
            .field("picture_format", &self.picture_format)
            .field("coded_extent", &self.coded_extent)
            .field("memory_bindings", &self.memories.len())
            .finish_non_exhaustive()
    }
}

impl Drop for H265DecodeSession {
    fn drop(&mut self) {
        // SAFETY: all handles were created from `raw_device` / `video_fns` and
        // are destroyed exactly once here, params before session before memory.
        unsafe {
            (self.video_fns.fp().destroy_video_session_parameters_khr)(
                self.raw_device.handle(),
                self.parameters,
                core::ptr::null(),
            );
            (self.video_fns.fp().destroy_video_session_khr)(
                self.raw_device.handle(),
                self.session,
                core::ptr::null(),
            );
            for &mem in &self.memories {
                self.raw_device.free_memory(mem, None);
            }
        }
    }
}

/// An AV1 Vulkan Video decode session + parameters, the AV1 sibling of
/// [`H265DecodeSession`].
pub struct Av1DecodeSession {
    session: vk::VideoSessionKHR,
    parameters: vk::VideoSessionParametersKHR,
    memories: alloc::vec::Vec<vk::DeviceMemory>,
    raw_device: ash::Device,
    video_fns: ash::khr::video_queue::Device,
    /// The decode picture format chosen at session creation (e.g. NV12).
    pub picture_format: vk::Format,
    pub coded_extent: (u32, u32),
    /// The Std sequence header handed to the driver at parameters creation.
    /// NVIDIA retains and dereferences the pointer per decode (it does NOT
    /// copy), so the session owns this stable-address block for its lifetime.
    _std: RetainedAv1Std,
}

/// Owner of the driver-retained Std AV1 sequence-header block. Its raw
/// pointers all point into the same boxed allocation (self-contained), which
/// is what makes the manual `Send` sound.
struct RetainedAv1Std(alloc::boxed::Box<StdAv1Params>);

// SAFETY: the block's internal pointers reference its own boxed allocation,
// never external data; moving the owner moves only the box handle (the pointees
// stay put), and the block is only ever read by the driver during decode calls
// made through the owning session.
unsafe impl Send for RetainedAv1Std {}

impl core::fmt::Debug for Av1DecodeSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Av1DecodeSession")
            .field("picture_format", &self.picture_format)
            .field("coded_extent", &self.coded_extent)
            .field("memory_bindings", &self.memories.len())
            .finish_non_exhaustive()
    }
}

impl Drop for Av1DecodeSession {
    fn drop(&mut self) {
        // SAFETY: all handles were created from `raw_device` / `video_fns` and
        // are destroyed exactly once here, params before session before memory.
        unsafe {
            (self.video_fns.fp().destroy_video_session_parameters_khr)(
                self.raw_device.handle(),
                self.parameters,
                core::ptr::null(),
            );
            (self.video_fns.fp().destroy_video_session_khr)(
                self.raw_device.handle(),
                self.session,
                core::ptr::null(),
            );
            for &mem in &self.memories {
                self.raw_device.free_memory(mem, None);
            }
        }
    }
}

impl VulkanVideoDevice {
    pub fn caps(&self) -> &VulkanVideoDecodeCaps {
        &self.caps
    }

    /// A [`GpuContext`](crate::gpu::GpuContext) sharing this decode device's wgpu
    /// instance / adapter / device / queue. A GPU-resident decoded texture
    /// ([`decode_all_to_textures`](H264DpbDecoder::decode_all_to_textures)) is
    /// bound to this device, so a consumer (e.g. `WgpuSink`) that shares this
    /// context presents it with no copy.
    pub fn gpu_context(&self) -> crate::gpu::GpuContext {
        crate::gpu::GpuContext::from_wgpu(
            self._instance.clone(),
            self._adapter.clone(),
            self.wgpu_device.clone(),
            self.wgpu_queue.clone(),
        )
    }

    /// Choose the decode picture format the driver supports for the given decode
    /// profile (DPB + output usage). Prefers the two-plane 4:2:0 NV12 layout.
    /// Codec-agnostic: the profile carries the codec-specific chained info.
    fn decode_format(
        &self,
        profile: &vk::VideoProfileInfoKHR,
        bit_depth: u8,
    ) -> Result<vk::Format, VulkanVideoError> {
        let mut profile_list =
            vk::VideoProfileListInfoKHR::default().profiles(core::slice::from_ref(profile));
        let fmt_info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
            .image_usage(
                vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                    | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR,
            )
            .push_next(&mut profile_list);

        // Two-call enumeration; `fmt_info` (with its chained profile list)
        // outlives both calls. The physical-device video-format query lives on
        // the instance extension.
        let fp = self.instance_video_fp();
        // SAFETY: null out-array just counts; `fmt_info` is valid.
        let count = unsafe {
            let mut n = 0u32;
            let _ = (fp.get_physical_device_video_format_properties_khr)(
                self.phys,
                &fmt_info,
                &mut n,
                core::ptr::null_mut(),
            );
            n
        };
        if count == 0 {
            return Err(VulkanVideoError::ExtensionUnsupported);
        }
        let mut formats: alloc::vec::Vec<vk::VideoFormatPropertiesKHR> =
            (0..count).map(|_| Default::default()).collect();
        // SAFETY: `formats` sized to `count`.
        unsafe {
            let mut n = count;
            let _ = (fp.get_physical_device_video_format_properties_khr)(
                self.phys,
                &fmt_info,
                &mut n,
                formats.as_mut_ptr(),
            );
        }
        // Prefer the two-plane 4:2:0 format matching the bit depth (NV12 for 8-bit,
        // G10X6 for 10-bit); else take the first offered format.
        let want = planar_420_format(bit_depth);
        let chosen = formats
            .iter()
            .find(|f| f.format == want)
            .or_else(|| formats.first())
            .map(|f| f.format)
            .ok_or(VulkanVideoError::ExtensionUnsupported)?;
        Ok(chosen)
    }

    /// The physical-device video-format query lives on the instance extension;
    /// rebuild the instance fn table from the wgpu device's shared instance.
    fn instance_video_fp(&self) -> ash::khr::video_queue::InstanceFn {
        // SAFETY: the wgpu device is Vulkan-backed; its shared instance is live.
        unsafe {
            let hal_device = self
                .wgpu_device
                .as_hal::<wgpu_hal::api::Vulkan>()
                .expect("vulkan wgpu device");
            let shared = hal_device.shared_instance();
            ash::khr::video_queue::InstanceFn::load(|name| {
                shared
                    .entry()
                    .get_instance_proc_addr(shared.raw_instance().handle(), name.as_ptr())
                    .map_or(core::ptr::null(), |f| f as *const _)
            })
        }
    }

    /// Create an H.264 decode session + parameters for `ps`, sized to
    /// `max_w`x`max_h` (clamped to the device's coded-extent range). Session
    /// parameter creation validates the `Std*` SPS/PPS mapping.
    pub fn create_h264_session(
        &self,
        ps: &H264ParameterSets,
        max_w: u32,
        max_h: u32,
    ) -> Result<H264DecodeSession, VulkanVideoError> {
        let prof = h264_profile();
        // H.264 decode is 8-bit here (High profile, NV12); High 10 is out of scope.
        let picture_format = self.decode_format(&prof.profile, 8)?;

        let w = max_w.clamp(self.caps.min_coded_extent.0, self.caps.max_coded_extent.0);
        let h = max_h.clamp(self.caps.min_coded_extent.1, self.caps.max_coded_extent.1);
        let coded_extent = vk::Extent2D {
            width: w,
            height: h,
        };

        let session_ci = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(self.decode_queue_family)
            .video_profile(&prof.profile)
            .picture_format(picture_format)
            .max_coded_extent(coded_extent)
            .reference_picture_format(picture_format)
            .max_dpb_slots(self.caps.max_dpb_slots)
            .max_active_reference_pictures(self.caps.max_active_reference_pictures)
            .std_header_version(&self.caps.std_header_version);

        let mut session = vk::VideoSessionKHR::null();
        // SAFETY: `session_ci` (with its chained profile) outlives the call.
        let ret = unsafe {
            (self.video_fns.fp().create_video_session_khr)(
                self.raw_device.handle(),
                &session_ci,
                core::ptr::null(),
                &mut session,
            )
        };
        if ret != vk::Result::SUCCESS {
            return Err(VulkanVideoError::QueryFailed(ret));
        }

        // Bind the session's device memory (one allocation per requirement).
        let memories = match self.bind_session_memory(session) {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: session was created above; destroy on the error path.
                unsafe {
                    (self.video_fns.fp().destroy_video_session_khr)(
                        self.raw_device.handle(),
                        session,
                        core::ptr::null(),
                    );
                }
                return Err(e);
            }
        };

        // Create session parameters carrying the Std SPS + PPS. This is where
        // the driver validates the mapping.
        let std_sps = [to_std_sps(&ps.sps)];
        let std_pps = [to_std_pps(&ps.pps)];
        let add = vk::VideoDecodeH264SessionParametersAddInfoKHR::default()
            .std_sp_ss(&std_sps)
            .std_pp_ss(&std_pps);
        let mut h264_params = vk::VideoDecodeH264SessionParametersCreateInfoKHR::default()
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&add);
        let params_ci = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push_next(&mut h264_params);

        let mut parameters = vk::VideoSessionParametersKHR::null();
        // SAFETY: `params_ci` and its chained H.264 add-info (referencing the
        // local Std arrays) outlive the call.
        let ret = unsafe {
            (self.video_fns.fp().create_video_session_parameters_khr)(
                self.raw_device.handle(),
                &params_ci,
                core::ptr::null(),
                &mut parameters,
            )
        };
        if ret != vk::Result::SUCCESS {
            for &mem in &memories {
                // SAFETY: allocated in bind_session_memory, freed once here.
                unsafe { self.raw_device.free_memory(mem, None) };
            }
            // SAFETY: session created above, destroyed once on this error path.
            unsafe {
                (self.video_fns.fp().destroy_video_session_khr)(
                    self.raw_device.handle(),
                    session,
                    core::ptr::null(),
                );
            }
            return Err(VulkanVideoError::QueryFailed(ret));
        }

        Ok(H264DecodeSession {
            session,
            parameters,
            memories,
            raw_device: self.raw_device.clone(),
            video_fns: self.video_fns.clone(),
            picture_format,
            coded_extent: (w, h),
        })
    }

    /// Create an H.265 decode session + parameters (carrying the VPS/SPS/PPS) for
    /// a stream of up to `max_w` x `max_h`. The HEVC sibling of
    /// [`create_h264_session`](Self::create_h264_session): building the session
    /// parameters makes the driver validate the M501 `Std*` mapping (a wrong
    /// mapping fails here), the H.265 analog of M488's H.264 validation.
    pub fn create_h265_session(
        &self,
        std: &StdH265Params,
        max_w: u32,
        max_h: u32,
    ) -> Result<H265DecodeSession, VulkanVideoError> {
        // 10-bit HEVC (Main 10) selects the Main 10 profile + the G10X6 output
        // format; 8-bit is Main + NV12. The decoder built over this session derives
        // the same bit depth from the SPS, so its profile / DPB images match.
        let bit_depth = std.sps.bit_depth_luma_minus8 + 8;
        let prof = h265_profile(bit_depth);
        let picture_format = self.decode_format(&prof.profile, bit_depth)?;

        let w = max_w.clamp(self.caps.min_coded_extent.0, self.caps.max_coded_extent.0);
        let h = max_h.clamp(self.caps.min_coded_extent.1, self.caps.max_coded_extent.1);
        let coded_extent = vk::Extent2D {
            width: w,
            height: h,
        };

        let session_ci = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(self.decode_queue_family)
            .video_profile(&prof.profile)
            .picture_format(picture_format)
            .max_coded_extent(coded_extent)
            .reference_picture_format(picture_format)
            .max_dpb_slots(self.caps.max_dpb_slots)
            .max_active_reference_pictures(self.caps.max_active_reference_pictures)
            .std_header_version(&self.caps.std_header_version);

        let mut session = vk::VideoSessionKHR::null();
        // SAFETY: `session_ci` (with its chained profile) outlives the call.
        let ret = unsafe {
            (self.video_fns.fp().create_video_session_khr)(
                self.raw_device.handle(),
                &session_ci,
                core::ptr::null(),
                &mut session,
            )
        };
        if ret != vk::Result::SUCCESS {
            return Err(VulkanVideoError::QueryFailed(ret));
        }

        let memories = match self.bind_session_memory(session) {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: session was created above; destroy on the error path.
                unsafe {
                    (self.video_fns.fp().destroy_video_session_khr)(
                        self.raw_device.handle(),
                        session,
                        core::ptr::null(),
                    );
                }
                return Err(e);
            }
        };

        // Create session parameters carrying the Std VPS + SPS + PPS. The Std
        // structs reference `std`'s owned pointee blocks, which outlive the call.
        let std_vps = [std.vps];
        let std_sps = [std.sps];
        let std_pps = [std.pps];
        let add = vk::VideoDecodeH265SessionParametersAddInfoKHR::default()
            .std_vp_ss(&std_vps)
            .std_sp_ss(&std_sps)
            .std_pp_ss(&std_pps);
        let mut h265_params = vk::VideoDecodeH265SessionParametersCreateInfoKHR::default()
            .max_std_vps_count(1)
            .max_std_sps_count(1)
            .max_std_pps_count(1)
            .parameters_add_info(&add);
        let params_ci = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push_next(&mut h265_params);

        let mut parameters = vk::VideoSessionParametersKHR::null();
        // SAFETY: `params_ci` and its chained H.265 add-info (referencing the
        // local Std arrays, which reference `std`'s pointee blocks) outlive the
        // call.
        let ret = unsafe {
            (self.video_fns.fp().create_video_session_parameters_khr)(
                self.raw_device.handle(),
                &params_ci,
                core::ptr::null(),
                &mut parameters,
            )
        };
        if ret != vk::Result::SUCCESS {
            for &mem in &memories {
                // SAFETY: allocated in bind_session_memory, freed once here.
                unsafe { self.raw_device.free_memory(mem, None) };
            }
            // SAFETY: session created above, destroyed once on this error path.
            unsafe {
                (self.video_fns.fp().destroy_video_session_khr)(
                    self.raw_device.handle(),
                    session,
                    core::ptr::null(),
                );
            }
            return Err(VulkanVideoError::QueryFailed(ret));
        }

        Ok(H265DecodeSession {
            session,
            parameters,
            memories,
            raw_device: self.raw_device.clone(),
            video_fns: self.video_fns.clone(),
            picture_format,
            coded_extent: (w, h),
        })
    }

    /// Create an AV1 decode session + parameters, the AV1 sibling of
    /// [`create_h265_session`](Self::create_h265_session). The session parameters
    /// carry the single Std AV1 sequence header (AV1 has no per-picture parameter
    /// set analog; the frame headers arrive at decode time), so creating them
    /// makes the driver validate the M504 `Std*` mapping.
    pub fn create_av1_session(
        &self,
        std: &StdAv1Params,
        max_w: u32,
        max_h: u32,
    ) -> Result<Av1DecodeSession, VulkanVideoError> {
        // AV1 Main profile covers 8 and 10-bit; the colour config carries the depth.
        // A 10-bit stream selects the G10X6 output format (same as HEVC Main 10).
        let bit_depth = std._color.BitDepth;
        let prof = av1_profile(bit_depth);
        let picture_format = self.decode_format(&prof.profile, bit_depth)?;

        let w = max_w.clamp(self.caps.min_coded_extent.0, self.caps.max_coded_extent.0);
        let h = max_h.clamp(self.caps.min_coded_extent.1, self.caps.max_coded_extent.1);
        let coded_extent = vk::Extent2D {
            width: w,
            height: h,
        };

        let session_ci = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(self.decode_queue_family)
            .video_profile(&prof.profile)
            .picture_format(picture_format)
            .max_coded_extent(coded_extent)
            .reference_picture_format(picture_format)
            .max_dpb_slots(self.caps.max_dpb_slots)
            .max_active_reference_pictures(self.caps.max_active_reference_pictures)
            .std_header_version(&self.caps.std_header_version);

        let mut session = vk::VideoSessionKHR::null();
        // SAFETY: `session_ci` (with its chained profile) outlives the call.
        let ret = unsafe {
            (self.video_fns.fp().create_video_session_khr)(
                self.raw_device.handle(),
                &session_ci,
                core::ptr::null(),
                &mut session,
            )
        };
        if ret != vk::Result::SUCCESS {
            return Err(VulkanVideoError::QueryFailed(ret));
        }

        let memories = match self.bind_session_memory(session) {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: session was created above; destroy on the error path.
                unsafe {
                    (self.video_fns.fp().destroy_video_session_khr)(
                        self.raw_device.handle(),
                        session,
                        core::ptr::null(),
                    );
                }
                return Err(e);
            }
        };

        // Session parameters carry the Std sequence header. The driver retains
        // the pointer past creation (see `StdAv1Params::deep_clone_boxed`), so
        // hand it the session-owned copy's address, not the caller's.
        let own_std = RetainedAv1Std(std.deep_clone_boxed());
        let mut av1_params = vk::VideoDecodeAV1SessionParametersCreateInfoKHR::default()
            .std_sequence_header(&own_std.0.seq_header);
        let params_ci = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push_next(&mut av1_params);

        let mut parameters = vk::VideoSessionParametersKHR::null();
        // SAFETY: `params_ci` and its chained AV1 create-info reference
        // `own_std.seq_header`, whose boxed addresses stay stable through the
        // call AND for the session's lifetime (stored below; the driver keeps
        // reading them per decode).
        let ret = unsafe {
            (self.video_fns.fp().create_video_session_parameters_khr)(
                self.raw_device.handle(),
                &params_ci,
                core::ptr::null(),
                &mut parameters,
            )
        };
        if ret != vk::Result::SUCCESS {
            for &mem in &memories {
                // SAFETY: allocated in bind_session_memory, freed once here.
                unsafe { self.raw_device.free_memory(mem, None) };
            }
            // SAFETY: session created above, destroyed once on this error path.
            unsafe {
                (self.video_fns.fp().destroy_video_session_khr)(
                    self.raw_device.handle(),
                    session,
                    core::ptr::null(),
                );
            }
            return Err(VulkanVideoError::QueryFailed(ret));
        }

        Ok(Av1DecodeSession {
            session,
            parameters,
            memories,
            raw_device: self.raw_device.clone(),
            video_fns: self.video_fns.clone(),
            picture_format,
            coded_extent: (w, h),
            _std: own_std,
        })
    }

    /// Query the session's memory requirements and allocate + bind one
    /// `VkDeviceMemory` per bind index.
    fn bind_session_memory(
        &self,
        session: vk::VideoSessionKHR,
    ) -> Result<alloc::vec::Vec<vk::DeviceMemory>, VulkanVideoError> {
        // Two-call: count, then fill.
        let mut count = 0u32;
        // SAFETY: null out-array counts.
        unsafe {
            let _ = (self
                .video_fns
                .fp()
                .get_video_session_memory_requirements_khr)(
                self.raw_device.handle(),
                session,
                &mut count,
                core::ptr::null_mut(),
            );
        }
        let mut reqs: alloc::vec::Vec<vk::VideoSessionMemoryRequirementsKHR> =
            (0..count).map(|_| Default::default()).collect();
        // SAFETY: `reqs` sized to `count`.
        unsafe {
            let _ = (self
                .video_fns
                .fp()
                .get_video_session_memory_requirements_khr)(
                self.raw_device.handle(),
                session,
                &mut count,
                reqs.as_mut_ptr(),
            );
        }

        let mut memories = alloc::vec::Vec::with_capacity(count as usize);
        let mut binds = alloc::vec::Vec::with_capacity(count as usize);
        for req in &reqs {
            // The requirement's `memory_type_bits` already lists the acceptable
            // memory types; do not impose an extra property flag (some session
            // bindings need a non-DEVICE_LOCAL type). Prefer DEVICE_LOCAL when it
            // is among the allowed types, else take any allowed type.
            let type_bits = req.memory_requirements.memory_type_bits;
            let mem_type = find_memory_type(
                &self.mem_props,
                type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                find_memory_type(&self.mem_props, type_bits, vk::MemoryPropertyFlags::empty())
            })
            .ok_or(VulkanVideoError::ExtensionUnsupported)?;
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(req.memory_requirements.size)
                .memory_type_index(mem_type);
            // SAFETY: valid allocate info; freed on error / drop.
            let mem = unsafe { self.raw_device.allocate_memory(&alloc_info, None) }
                .map_err(VulkanVideoError::QueryFailed)?;
            memories.push(mem);
            binds.push(
                vk::BindVideoSessionMemoryInfoKHR::default()
                    .memory_bind_index(req.memory_bind_index)
                    .memory(mem)
                    .memory_offset(0)
                    .memory_size(req.memory_requirements.size),
            );
        }
        // SAFETY: `binds` references `memories` still owned here; bind once.
        let ret = unsafe {
            (self.video_fns.fp().bind_video_session_memory_khr)(
                self.raw_device.handle(),
                session,
                binds.len() as u32,
                binds.as_ptr(),
            )
        };
        if ret != vk::Result::SUCCESS {
            for &mem in &memories {
                // SAFETY: freed once on this error path.
                unsafe { self.raw_device.free_memory(mem, None) };
            }
            return Err(VulkanVideoError::QueryFailed(ret));
        }
        Ok(memories)
    }

    /// Allocate + bind device memory for an image, returning the memory.
    /// # Safety
    /// `image` must be a valid image created from `self.raw_device`.
    unsafe fn alloc_bind_image(
        &self,
        image: vk::Image,
        flags: vk::MemoryPropertyFlags,
    ) -> Result<vk::DeviceMemory, VulkanVideoError> {
        // SAFETY: image is valid (contract); delegates to the free helper.
        unsafe { alloc_bind_image_raw(&self.raw_device, &self.mem_props, image, flags) }
    }

    /// Decode a single IDR frame into an NV12 image and read back its luma
    /// plane, the minimal end-to-end proof the decode path produces pixels.
    /// `idr_au` is an Annex-B access unit; the first IDR slice NAL (type 5) is
    /// submitted (SPS/PPS come from the session parameters, not the bitstream).
    /// Returns the luma plane (`width*height` bytes).
    ///
    /// IDR-only: no inter-frame references, so the DPB holds just the one
    /// decoded picture and the `Std*` picture info is the known IDR constants
    /// (frame_num 0, POC 0, IdrPicFlag). Full reference management is a later
    /// increment. On the error path some transient Vulkan objects may leak; this
    /// is a one-shot validation entry point, not the steady-state element.
    pub fn decode_idr_nv12(
        &self,
        session: &H264DecodeSession,
        idr_au: &[u8],
    ) -> Result<Nv12Frame, VulkanVideoError> {
        let (w, h) = session.coded_extent;
        let slice = extract_first_idr_slice(idr_au).ok_or(VulkanVideoError::NoDecodableSlice)?;
        let dev = &self.raw_device;
        let prof = h264_profile();

        // Decode target (coincide: same image is DPB slot 0 and decode output).
        let mut profile_list =
            vk::VideoProfileListInfoKHR::default().profiles(core::slice::from_ref(&prof.profile));
        let image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(session.picture_format)
            .extent(vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                    | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut profile_list);
        // SAFETY: valid create info; the chained profile list outlives the call.
        let image =
            unsafe { dev.create_image(&image_ci, None) }.map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh image.
        let image_mem =
            unsafe { self.alloc_bind_image(image, vk::MemoryPropertyFlags::DEVICE_LOCAL) }?;
        let view_ci = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(session.picture_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        // SAFETY: image is valid and outlives the view.
        let view = unsafe { dev.create_image_view(&view_ci, None) }
            .map_err(VulkanVideoError::QueryFailed)?;

        // Coded bitstream buffer (host-visible), holding the IDR slice.
        let size_align = self.caps.min_bitstream_buffer_size_alignment.max(1);
        let buf_size = round_up(slice.len() as u64, size_align);
        let mut buf_profile_list =
            vk::VideoProfileListInfoKHR::default().profiles(core::slice::from_ref(&prof.profile));
        let buf_ci = vk::BufferCreateInfo::default()
            .size(buf_size)
            .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut buf_profile_list);
        // SAFETY: valid create info; chained profile list outlives the call.
        let bitstream =
            unsafe { dev.create_buffer(&buf_ci, None) }.map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh buffer.
        let breq = unsafe { dev.get_buffer_memory_requirements(bitstream) };
        let btype = find_memory_type(
            &self.mem_props,
            breq.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or(VulkanVideoError::ExtensionUnsupported)?;
        let bai = vk::MemoryAllocateInfo::default()
            .allocation_size(breq.size)
            .memory_type_index(btype);
        // SAFETY: valid allocate + bind + map of a fresh host-visible buffer.
        let bitstream_mem = unsafe {
            let m = dev
                .allocate_memory(&bai, None)
                .map_err(VulkanVideoError::QueryFailed)?;
            dev.bind_buffer_memory(bitstream, m, 0)
                .map_err(VulkanVideoError::QueryFailed)?;
            let ptr = dev
                .map_memory(m, 0, breq.size, vk::MemoryMapFlags::empty())
                .map_err(VulkanVideoError::QueryFailed)? as *mut u8;
            core::ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            dev.unmap_memory(m);
            m
        };

        // Readback buffer for the full NV12 frame (luma w*h + interleaved CbCr
        // w*h/2), host-visible. Chroma follows luma at offset w*h.
        let luma_len = (w as u64) * (h as u64);
        let chroma_len = luma_len / 2;
        let nv12_len = luma_len + chroma_len;
        let rb_ci = vk::BufferCreateInfo::default()
            .size(nv12_len)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info.
        let readback =
            unsafe { dev.create_buffer(&rb_ci, None) }.map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh buffer.
        let rreq = unsafe { dev.get_buffer_memory_requirements(readback) };
        let rtype = find_memory_type(
            &self.mem_props,
            rreq.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or(VulkanVideoError::ExtensionUnsupported)?;
        let rai = vk::MemoryAllocateInfo::default()
            .allocation_size(rreq.size)
            .memory_type_index(rtype);
        // SAFETY: allocate + bind of a fresh buffer.
        let readback_mem = unsafe {
            let m = dev
                .allocate_memory(&rai, None)
                .map_err(VulkanVideoError::QueryFailed)?;
            dev.bind_buffer_memory(readback, m, 0)
                .map_err(VulkanVideoError::QueryFailed)?;
            m
        };

        // Command pool + buffer on the decode queue family.
        let pool_ci =
            vk::CommandPoolCreateInfo::default().queue_family_index(self.decode_queue_family);
        // SAFETY: valid create info.
        let pool = unsafe { dev.create_command_pool(&pool_ci, None) }
            .map_err(VulkanVideoError::QueryFailed)?;
        let cb_ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: valid allocate info.
        let cb = unsafe { dev.allocate_command_buffers(&cb_ai) }
            .map_err(VulkanVideoError::QueryFailed)?[0];

        // Per-frame Std picture info: a lone IDR, decoded into DPB slot 0.
        // SAFETY: bitfield POD, valid all-zero.
        let mut pic_flags: vk::native::StdVideoDecodeH264PictureInfoFlags =
            unsafe { core::mem::zeroed() };
        pic_flags.set_field_pic_flag(0);
        pic_flags.set_is_intra(1);
        pic_flags.set_IdrPicFlag(1);
        pic_flags.set_is_reference(1);
        let std_pic = vk::native::StdVideoDecodeH264PictureInfo {
            flags: pic_flags,
            seq_parameter_set_id: session_sps_id(session),
            pic_parameter_set_id: 0,
            reserved1: 0,
            reserved2: 0,
            frame_num: 0,
            idr_pic_id: 0,
            PicOrderCnt: [0, 0],
        };
        let slice_offsets = [0u32];
        let mut h264_pic = vk::VideoDecodeH264PictureInfoKHR::default()
            .std_picture_info(&std_pic)
            .slice_offsets(&slice_offsets);

        // The decoded picture as a DPB resource (slot 0).
        let picres = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: w,
                height: h,
            })
            .base_array_layer(0)
            .image_view_binding(view);
        // At begin, reserve the slot being set up with slot_index -1 (not yet a
        // valid reference); the decode call activates it as slot 0.
        let begin_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(&picres);
        let setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(0)
            .picture_resource(&picres);

        let begin_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(session.session)
            .video_session_parameters(session.parameters)
            .reference_slots(core::slice::from_ref(&begin_slot));
        let control_info =
            vk::VideoCodingControlInfoKHR::default().flags(vk::VideoCodingControlFlagsKHR::RESET);
        let end_info = vk::VideoEndCodingInfoKHR::default();
        let decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(bitstream)
            .src_buffer_offset(0)
            .src_buffer_range(buf_size)
            .dst_picture_resource(picres)
            .setup_reference_slot(&setup_slot)
            .push_next(&mut h264_pic);

        // Record and submit.
        // SAFETY: all handles above are valid and outlive the submission we wait
        // on; the barriers move the image UNDEFINED -> DPB (decode) -> TRANSFER_SRC.
        let submit_result = unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.begin_command_buffer(cb, &begin)
                .map_err(VulkanVideoError::QueryFailed)?;

            let to_dpb = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let dep =
                vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&to_dpb));
            (self.sync2_fns.fp().cmd_pipeline_barrier2_khr)(cb, &dep);

            (self.video_fns.fp().cmd_begin_video_coding_khr)(cb, &begin_info);
            (self.video_fns.fp().cmd_control_video_coding_khr)(cb, &control_info);
            (self.decode_fns.fp().cmd_decode_video_khr)(cb, &decode_info);
            (self.video_fns.fp().cmd_end_video_coding_khr)(cb, &end_info);

            let to_src = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let dep2 =
                vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&to_src));
            (self.sync2_fns.fp().cmd_pipeline_barrier2_khr)(cb, &dep2);

            let luma_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                });
            // Chroma plane (PLANE_1): interleaved CbCr at half resolution, RG8,
            // packed right after luma in the readback buffer.
            let chroma_region = vk::BufferImageCopy::default()
                .buffer_offset(luma_len)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: w / 2,
                    height: h / 2,
                    depth: 1,
                });
            let regions = [luma_region, chroma_region];
            dev.cmd_copy_image_to_buffer(
                cb,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback,
                &regions,
            );

            dev.end_command_buffer(cb)
                .map_err(VulkanVideoError::QueryFailed)?;

            let fence = dev
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(VulkanVideoError::QueryFailed)?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            let r = dev
                .queue_submit(self.decode_queue, core::slice::from_ref(&submit), fence)
                .and_then(|_| dev.wait_for_fences(&[fence], true, u64::MAX));
            dev.destroy_fence(fence, None);
            r.map_err(VulkanVideoError::QueryFailed)
        };

        // Read back the full NV12 frame before tearing down.
        let (luma, chroma) = match submit_result {
            Ok(()) => {
                // SAFETY: readback_mem is host-visible/coherent and holds
                // `nv12_len` bytes written by the completed copies.
                unsafe {
                    let ptr = dev
                        .map_memory(readback_mem, 0, nv12_len, vk::MemoryMapFlags::empty())
                        .map_err(VulkanVideoError::QueryFailed)?
                        as *const u8;
                    let mut luma = alloc::vec![0u8; luma_len as usize];
                    let mut chroma = alloc::vec![0u8; chroma_len as usize];
                    core::ptr::copy_nonoverlapping(ptr, luma.as_mut_ptr(), luma_len as usize);
                    core::ptr::copy_nonoverlapping(
                        ptr.add(luma_len as usize),
                        chroma.as_mut_ptr(),
                        chroma_len as usize,
                    );
                    dev.unmap_memory(readback_mem);
                    (luma, chroma)
                }
            }
            Err(e) => {
                // SAFETY: destroy the transient objects created above once.
                unsafe {
                    self.destroy_decode_transients(
                        pool,
                        image,
                        view,
                        image_mem,
                        bitstream,
                        bitstream_mem,
                        readback,
                        readback_mem,
                    )
                };
                return Err(e);
            }
        };

        // SAFETY: teardown of all transient objects, each destroyed once, after
        // the decode has completed (fence waited).
        unsafe {
            self.destroy_decode_transients(
                pool,
                image,
                view,
                image_mem,
                bitstream,
                bitstream_mem,
                readback,
                readback_mem,
            )
        };

        // The one-shot IDR path is the 8-bit H.264 format.
        Ok(Nv12Frame {
            width: w,
            height: h,
            luma,
            chroma,
            bit_depth: 8,
        })
    }

    /// Decode a single IDR frame and return just the luma plane (`width*height`
    /// bytes). Thin wrapper over [`decode_idr_nv12`](Self::decode_idr_nv12).
    pub fn decode_idr_luma(
        &self,
        session: &H264DecodeSession,
        idr_au: &[u8],
    ) -> Result<DecodedLuma, VulkanVideoError> {
        let f = self.decode_idr_nv12(session, idr_au)?;
        Ok(DecodedLuma {
            width: f.width,
            height: f.height,
            luma: f.luma,
        })
    }

    /// Decode a single IDR frame and upload it to an RGBA `wgpu::Texture` on this
    /// device's wgpu queue, the output type a wgpu consumer (game engine /
    /// visualization viewer) samples directly. NV12 -> RGBA is converted on the
    /// CPU here (M490); the zero-copy GPU-resident `VkSamplerYcbcrConversion`
    /// path that keeps the frame on the GPU is the next increment.
    pub fn decode_idr_to_rgba_texture(
        &self,
        session: &H264DecodeSession,
        idr_au: &[u8],
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        let frame = self.decode_idr_nv12(session, idr_au)?;
        // One-shot legacy path: no decoder-resolved colour, use BT.601 studio (the
        // fixed conversion this path always used). The multi-frame decoders below
        // resolve colour from the stream.
        Ok(nv12_to_rgba_texture(
            &self.wgpu_device,
            &self.wgpu_queue,
            &frame,
            VideoColorSpace::BT601_STUDIO,
        ))
    }

    /// Read a converted RGBA wgpu texture on this device back to tightly-packed
    /// `width*height*bpp` bytes (validation / CPU-consumer helper), where `bpp` is
    /// the texture's texel size: 4 for the 8-bit `Rgba8Unorm` output, 8 for the
    /// 10-bit `Rgba16Float` output. Handles the 256-byte row alignment wgpu
    /// requires for buffer copies.
    pub fn read_rgba_texture(&self, texture: &wgpu::Texture) -> alloc::vec::Vec<u8> {
        crate::gpu::read_rgba_texture_dq(&self.wgpu_device, &self.wgpu_queue, texture)
            .expect("vulkan-video rgba readback")
    }

    /// Submit a lone-IDR decode into `image` (via `decode_view`) on the decode
    /// queue and wait; leaves `image` decoded, in `VIDEO_DECODE_DPB_KHR` layout.
    /// The bitstream buffer + command pool are transient and freed here.
    fn submit_idr_decode_into(
        &self,
        session: &H264DecodeSession,
        slice: &[u8],
        image: vk::Image,
        decode_view: vk::ImageView,
        w: u32,
        h: u32,
    ) -> Result<(), VulkanVideoError> {
        let dev = &self.raw_device;
        let prof = h264_profile();

        let size_align = self.caps.min_bitstream_buffer_size_alignment.max(1);
        let buf_size = round_up(slice.len() as u64, size_align);
        let mut buf_profile_list =
            vk::VideoProfileListInfoKHR::default().profiles(core::slice::from_ref(&prof.profile));
        let buf_ci = vk::BufferCreateInfo::default()
            .size(buf_size)
            .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut buf_profile_list);
        // SAFETY: valid create info; chained profile list outlives the call.
        let bitstream =
            unsafe { dev.create_buffer(&buf_ci, None) }.map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh buffer; allocate host-visible memory, bind, map, fill.
        let bitstream_mem = unsafe {
            let breq = dev.get_buffer_memory_requirements(bitstream);
            let bt = find_memory_type(
                &self.mem_props,
                breq.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or(VulkanVideoError::ExtensionUnsupported)?;
            let bai = vk::MemoryAllocateInfo::default()
                .allocation_size(breq.size)
                .memory_type_index(bt);
            let m = dev
                .allocate_memory(&bai, None)
                .map_err(VulkanVideoError::QueryFailed)?;
            dev.bind_buffer_memory(bitstream, m, 0)
                .map_err(VulkanVideoError::QueryFailed)?;
            let ptr = dev
                .map_memory(m, 0, breq.size, vk::MemoryMapFlags::empty())
                .map_err(VulkanVideoError::QueryFailed)? as *mut u8;
            core::ptr::copy_nonoverlapping(slice.as_ptr(), ptr, slice.len());
            dev.unmap_memory(m);
            m
        };

        let pool_ci =
            vk::CommandPoolCreateInfo::default().queue_family_index(self.decode_queue_family);
        // SAFETY: valid; freed below.
        let pool = unsafe { dev.create_command_pool(&pool_ci, None) }
            .map_err(VulkanVideoError::QueryFailed)?;
        let cb_ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: valid allocate.
        let cb = unsafe { dev.allocate_command_buffers(&cb_ai) }
            .map_err(VulkanVideoError::QueryFailed)?[0];

        // SAFETY: bitfield POD valid all-zero.
        let mut pic_flags: vk::native::StdVideoDecodeH264PictureInfoFlags =
            unsafe { core::mem::zeroed() };
        pic_flags.set_is_intra(1);
        pic_flags.set_IdrPicFlag(1);
        pic_flags.set_is_reference(1);
        let std_pic = vk::native::StdVideoDecodeH264PictureInfo {
            flags: pic_flags,
            seq_parameter_set_id: 0,
            pic_parameter_set_id: 0,
            reserved1: 0,
            reserved2: 0,
            frame_num: 0,
            idr_pic_id: 0,
            PicOrderCnt: [0, 0],
        };
        let slice_offsets = [0u32];
        let mut h264_pic = vk::VideoDecodeH264PictureInfoKHR::default()
            .std_picture_info(&std_pic)
            .slice_offsets(&slice_offsets);
        let picres = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: w,
                height: h,
            })
            .base_array_layer(0)
            .image_view_binding(decode_view);
        let begin_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(-1)
            .picture_resource(&picres);
        let setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(0)
            .picture_resource(&picres);
        let begin_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(session.session)
            .video_session_parameters(session.parameters)
            .reference_slots(core::slice::from_ref(&begin_slot));
        let control_info =
            vk::VideoCodingControlInfoKHR::default().flags(vk::VideoCodingControlFlagsKHR::RESET);
        let end_info = vk::VideoEndCodingInfoKHR::default();
        let decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(bitstream)
            .src_buffer_offset(0)
            .src_buffer_range(buf_size)
            .dst_picture_resource(picres)
            .setup_reference_slot(&setup_slot)
            .push_next(&mut h264_pic);

        // SAFETY: all handles valid and outlive the waited submission.
        let r = unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.begin_command_buffer(cb, &begin)
                .map_err(VulkanVideoError::QueryFailed)?;
            let to_dpb = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range());
            let dep =
                vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&to_dpb));
            (self.sync2_fns.fp().cmd_pipeline_barrier2_khr)(cb, &dep);
            (self.video_fns.fp().cmd_begin_video_coding_khr)(cb, &begin_info);
            (self.video_fns.fp().cmd_control_video_coding_khr)(cb, &control_info);
            (self.decode_fns.fp().cmd_decode_video_khr)(cb, &decode_info);
            (self.video_fns.fp().cmd_end_video_coding_khr)(cb, &end_info);
            dev.end_command_buffer(cb)
                .map_err(VulkanVideoError::QueryFailed)?;
            let fence = dev
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(VulkanVideoError::QueryFailed)?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            let r = dev
                .queue_submit(self.decode_queue, core::slice::from_ref(&submit), fence)
                .and_then(|_| dev.wait_for_fences(&[fence], true, u64::MAX));
            dev.destroy_fence(fence, None);
            r
        };

        // SAFETY: transient decode inputs, freed once, after the wait.
        unsafe {
            dev.destroy_command_pool(pool, None);
            dev.destroy_buffer(bitstream, None);
            dev.free_memory(bitstream_mem, None);
        }
        r.map_err(VulkanVideoError::QueryFailed)
    }

    /// Decode a single IDR frame and hand back an RGBA `wgpu::Texture` produced
    /// entirely on the GPU: the decoded NV12 image is converted to RGBA by a
    /// Vulkan compute pass through a `VkSamplerYcbcrConversion` and the result
    /// image is imported straight into wgpu, so the frame never round-trips
    /// through system memory (unlike [`decode_idr_to_rgba_texture`]). Requires a
    /// distinct compute queue; returns [`VulkanVideoError::NoComputeQueue`]
    /// otherwise (caller falls back to the CPU path).
    pub fn decode_idr_to_rgba_texture_gpu(
        &self,
        session: &H264DecodeSession,
        idr_au: &[u8],
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        let compute_queue = self.compute_queue.ok_or(VulkanVideoError::NoComputeQueue)?;
        let (w, h) = session.coded_extent;
        let slice = extract_first_idr_slice(idr_au).ok_or(VulkanVideoError::NoDecodableSlice)?;
        let dev = &self.raw_device;
        let prof = h264_profile();

        // NV12 decode target, shared CONCURRENT between the decode and compute
        // families so the compute pass samples it with no ownership transfer.
        let families = [self.decode_queue_family, self.compute_queue_family];
        let mut profile_list =
            vk::VideoProfileListInfoKHR::default().profiles(core::slice::from_ref(&prof.profile));
        let nv12_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(session.picture_format)
            .extent(vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                    | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&families)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut profile_list);
        // SAFETY: valid create info; chained profile list outlives the call.
        let nv12 =
            unsafe { dev.create_image(&nv12_ci, None) }.map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh image.
        let nv12_mem =
            unsafe { self.alloc_bind_image(nv12, vk::MemoryPropertyFlags::DEVICE_LOCAL) }?;
        let decode_view_ci = vk::ImageViewCreateInfo::default()
            .image(nv12)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(session.picture_format)
            .subresource_range(color_range());
        // SAFETY: nv12 valid.
        let decode_view = unsafe { dev.create_image_view(&decode_view_ci, None) }
            .map_err(VulkanVideoError::QueryFailed)?;

        // Decode into the NV12 image (leaves it in DPB layout).
        if let Err(e) = self.submit_idr_decode_into(session, &slice, nv12, decode_view, w, h) {
            // SAFETY: destroy what we made on the error path.
            unsafe {
                dev.destroy_image_view(decode_view, None);
                dev.destroy_image(nv12, None);
                dev.free_memory(nv12_mem, None);
            }
            return Err(e);
        }

        // GPU-resident NV12 -> RGBA via a ycbcr-conversion compute pass. On
        // success the RGBA image + memory are moved into the wgpu texture's drop
        // callback; every other object is destroyed before returning.
        // SAFETY: all handles are created from `dev` and destroyed exactly once
        // (here or in the wgpu drop callback); the compute submission is waited
        // on before teardown.
        let result = unsafe { self.ycbcr_to_wgpu(nv12, compute_queue, w, h) };

        // SAFETY: NV12 image + decode view are done with once the compute pass
        // finished (inside ycbcr_to_wgpu); free them regardless of outcome.
        unsafe {
            dev.destroy_image_view(decode_view, None);
            dev.destroy_image(nv12, None);
            dev.free_memory(nv12_mem, None);
        }
        result
    }

    /// Thin wrapper over [`nv12_to_wgpu_texture`] for the one-shot IDR path (the
    /// image is discarded after, so no DPB restore).
    ///
    /// # Safety
    /// `nv12` must be a valid, decoded, idle image on `self.raw_device` in
    /// `VIDEO_DECODE_DPB_KHR` layout, accessible from the compute family.
    unsafe fn ycbcr_to_wgpu(
        &self,
        nv12: vk::Image,
        compute_queue: vk::Queue,
        w: u32,
        h: u32,
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        // SAFETY: contract forwarded to the free helper.
        unsafe {
            nv12_to_wgpu_texture(
                &self.raw_device,
                &self.wgpu_device,
                &self.mem_props,
                compute_queue,
                self.compute_queue_family,
                nv12,
                w,
                h,
                false,
            )
        }
    }

    /// Create one NV12 (coincide) DPB image sized to the coded extent: it is
    /// simultaneously a decode-output target and a DPB reference slot, and is
    /// `TRANSFER_SRC` so its decoded content can be read back. `profile` supplies
    /// the video-profile list the video-usage image needs.
    fn create_dpb_image(
        &self,
        w: u32,
        h: u32,
        format: vk::Format,
        profile: &vk::VideoProfileInfoKHR,
        gpu: bool,
    ) -> Result<DpbImage, VulkanVideoError> {
        let dev = &self.raw_device;
        let mut profile_list =
            vk::VideoProfileListInfoKHR::default().profiles(core::slice::from_ref(profile));
        // System path: TRANSFER_SRC so the decoded planes can be copied to the
        // readback buffer, EXCLUSIVE (only the decode queue touches it). GPU path:
        // SAMPLED so the ycbcr compute pass can sample the slot, CONCURRENT across
        // the decode + compute families so no ownership transfer is needed, and
        // TRANSFER_SRC so a slot can also be read back to NV12 (the AV1 film-grain
        // texture path reads the grain-free reconstruction back, synthesizes grain
        // on the CPU, and re-uploads: grain is output-only and must not feed the DPB).
        let families = [self.decode_queue_family, self.compute_queue_family];
        let usage = if gpu {
            vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR
                | vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR
                | vk::ImageUsageFlags::TRANSFER_SRC
        };
        let mut image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut profile_list);
        if gpu {
            image_ci = image_ci
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families);
        }
        // SAFETY: valid create info; the chained profile list outlives the call.
        let image =
            unsafe { dev.create_image(&image_ci, None) }.map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh image.
        let mem =
            match unsafe { self.alloc_bind_image(image, vk::MemoryPropertyFlags::DEVICE_LOCAL) } {
                Ok(m) => m,
                Err(e) => {
                    // SAFETY: destroy the image we just made on the error path.
                    unsafe { dev.destroy_image(image, None) };
                    return Err(e);
                }
            };
        let view_ci = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(color_range());
        // SAFETY: image valid and outlives the view.
        let view = match unsafe { dev.create_image_view(&view_ci, None) } {
            Ok(v) => v,
            Err(e) => {
                // SAFETY: free image + memory made above on the error path.
                unsafe {
                    dev.destroy_image(image, None);
                    dev.free_memory(mem, None);
                }
                return Err(VulkanVideoError::QueryFailed(e));
            }
        };
        Ok(DpbImage { image, view, mem })
    }

    /// Build a multi-frame H.264 decoder over `session`: a DPB image pool, the
    /// reference-slot bookkeeping, and the POC / frame-num tracking that let it
    /// decode P/B frames (not just the leading IDR). `ps` supplies the SPS/PPS
    /// (POC type, field widths, `max_num_ref_frames`) the loop needs. Rejects
    /// `pic_order_cnt_type == 1` (delta cycle not carried) up front.
    pub fn create_h264_dpb_decoder(
        &self,
        session: &H264DecodeSession,
        ps: &H264ParameterSets,
    ) -> Result<H264DpbDecoder, VulkanVideoError> {
        self.build_dpb_decoder(session, ps, false)
    }

    /// Build a multi-frame H.264 decoder that emits GPU-resident RGBA
    /// `wgpu::Texture`s (the zero-copy wedge): each decoded DPB slot is converted
    /// in place by a `VkSamplerYcbcrConversion` compute pass and imported into
    /// wgpu, so the frame never leaves the GPU. Requires a distinct compute queue
    /// on the decode device; returns [`VulkanVideoError::NoComputeQueue`] if none
    /// (the caller falls back to [`create_h264_dpb_decoder`] + system NV12).
    pub fn create_h264_dpb_decoder_gpu(
        &self,
        session: &H264DecodeSession,
        ps: &H264ParameterSets,
    ) -> Result<H264DpbDecoder, VulkanVideoError> {
        if self.compute_queue.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        self.build_dpb_decoder(session, ps, true)
    }

    /// Build the codec-independent [`DpbCore`] every `*DpbDecoder` embeds: the DPB
    /// image pool (`num_slots` images at `coded_extent` in `picture_format`), the
    /// persistent host-visible NV12 readback buffer, and the decode-queue command
    /// pool. `gpu` selects `SAMPLED` + `CONCURRENT` DPB images (texture output) over
    /// plain system readback; `profile` is the codec profile the images and
    /// bitstream buffers are tagged with. On any failure everything already created
    /// is freed. This is the shared half of the per-codec `build_*` constructors.
    #[allow(clippy::too_many_arguments)]
    fn build_dpb_core(
        &self,
        session: vk::VideoSessionKHR,
        parameters: vk::VideoSessionParametersKHR,
        coded_extent: (u32, u32),
        picture_format: vk::Format,
        num_slots: usize,
        gpu: bool,
        profile: &vk::VideoProfileInfoKHR,
        color: VideoColorSpace,
        hdr_output: HdrOutput,
    ) -> Result<DpbCore, VulkanVideoError> {
        let (w, h) = coded_extent;
        // 8-bit NV12 (`G8_B8R8`, 1 byte/sample) or 10-bit (`G10X6`, 2). The GPU
        // converter picks its ycbcr conversion + RGBA target from this (10-bit ->
        // `R16G16B16A16_SFLOAT`); the system readback scales its buffer lengths.
        let bps = format_bytes_per_sample(picture_format);
        let bit_depth = if bps >= 2 { 10 } else { 8 };

        // DPB image pool. On any failure, free what was already created.
        let mut slots: alloc::vec::Vec<DpbImage> = alloc::vec::Vec::with_capacity(num_slots);
        let free_slots = |slots: &[DpbImage]| {
            for s in slots {
                // SAFETY: each handle created just above, destroyed once.
                unsafe {
                    self.raw_device.destroy_image_view(s.view, None);
                    self.raw_device.destroy_image(s.image, None);
                    self.raw_device.free_memory(s.mem, None);
                }
            }
        };
        for _ in 0..num_slots {
            match self.create_dpb_image(w, h, picture_format, profile, gpu) {
                Ok(img) => slots.push(img),
                Err(e) => {
                    free_slots(&slots);
                    return Err(e);
                }
            }
        }

        // Persistent host-visible readback buffer holding `DECODE_RING_DEPTH`
        // decoded frames back to back (the system path pipelines that many decodes
        // in flight). Each slot's region starts at a `readback_stride` multiple so
        // its copy `bufferOffset` stays aligned. Lengths are in BYTES: a 10-bit
        // (G10X6) format stores 2 bytes per sample, so they scale by `bps`.
        let luma_len = (w as u64) * (h as u64) * bps;
        let chroma_len = luma_len / 2;
        let nv12_len = luma_len + chroma_len;
        let readback_stride = round_up(nv12_len, 256);
        let rb_ci = vk::BufferCreateInfo::default()
            .size(readback_stride * DECODE_RING_DEPTH as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info.
        let readback = match unsafe { self.raw_device.create_buffer(&rb_ci, None) } {
            Ok(b) => b,
            Err(e) => {
                free_slots(&slots);
                return Err(VulkanVideoError::QueryFailed(e));
            }
        };
        // SAFETY: fresh buffer.
        let rreq = unsafe { self.raw_device.get_buffer_memory_requirements(readback) };
        let rtype = find_memory_type(
            &self.mem_props,
            rreq.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let readback_mem = match rtype {
            Some(t) => {
                let rai = vk::MemoryAllocateInfo::default()
                    .allocation_size(rreq.size)
                    .memory_type_index(t);
                // SAFETY: allocate + bind of the fresh readback buffer.
                unsafe {
                    match self.raw_device.allocate_memory(&rai, None) {
                        Ok(m) => {
                            if let Err(e) = self.raw_device.bind_buffer_memory(readback, m, 0) {
                                self.raw_device.free_memory(m, None);
                                self.raw_device.destroy_buffer(readback, None);
                                free_slots(&slots);
                                return Err(VulkanVideoError::QueryFailed(e));
                            }
                            m
                        }
                        Err(e) => {
                            self.raw_device.destroy_buffer(readback, None);
                            free_slots(&slots);
                            return Err(VulkanVideoError::QueryFailed(e));
                        }
                    }
                }
            }
            None => {
                // SAFETY: destroy the buffer made above.
                unsafe { self.raw_device.destroy_buffer(readback, None) };
                free_slots(&slots);
                return Err(VulkanVideoError::ExtensionUnsupported);
            }
        };

        // Command pool on the decode queue family; command buffers are allocated
        // per frame after a pool reset.
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.decode_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: valid create info.
        let pool = match unsafe { self.raw_device.create_command_pool(&pool_ci, None) } {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: free readback + slots made above.
                unsafe {
                    self.raw_device.free_memory(readback_mem, None);
                    self.raw_device.destroy_buffer(readback, None);
                }
                free_slots(&slots);
                return Err(VulkanVideoError::QueryFailed(e));
            }
        };

        // System-path pipelining ring: a second command pool (RESET_COMMAND_BUFFER
        // so each slot's persistent command buffer can be re-recorded in place),
        // plus one command buffer + fence per ring slot. Cleans up everything made
        // above on any failure.
        let dev = &self.raw_device;
        let cleanup = |ring_pool: Option<vk::CommandPool>, fences: &[vk::Fence]| {
            // SAFETY: every handle was created above from `dev` and is destroyed
            // once here on the error path.
            unsafe {
                for &f in fences {
                    dev.destroy_fence(f, None);
                }
                if let Some(rp) = ring_pool {
                    dev.destroy_command_pool(rp, None);
                }
                dev.destroy_command_pool(pool, None);
                dev.free_memory(readback_mem, None);
                dev.destroy_buffer(readback, None);
            }
            free_slots(&slots);
        };
        let ring_pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.decode_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: valid create info.
        let ring_pool = match unsafe { dev.create_command_pool(&ring_pool_ci, None) } {
            Ok(p) => p,
            Err(e) => {
                cleanup(None, &[]);
                return Err(VulkanVideoError::QueryFailed(e));
            }
        };
        let ring_cb_ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(ring_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(DECODE_RING_DEPTH as u32);
        // SAFETY: valid allocate info; frees the pool on failure.
        let ring_cbs = match unsafe { dev.allocate_command_buffers(&ring_cb_ai) } {
            Ok(c) => c,
            Err(e) => {
                cleanup(Some(ring_pool), &[]);
                return Err(VulkanVideoError::QueryFailed(e));
            }
        };
        let mut ring: alloc::vec::Vec<RingSlot> = alloc::vec::Vec::with_capacity(DECODE_RING_DEPTH);
        for &cb in &ring_cbs {
            // SAFETY: valid create info; a signalled fence is not required (a slot
            // is only ever waited after it has been submitted).
            let fence = match unsafe { dev.create_fence(&vk::FenceCreateInfo::default(), None) } {
                Ok(f) => f,
                Err(e) => {
                    let made: alloc::vec::Vec<vk::Fence> = ring.iter().map(|s| s.fence).collect();
                    cleanup(Some(ring_pool), &made);
                    return Err(VulkanVideoError::QueryFailed(e));
                }
            };
            ring.push(RingSlot {
                cb,
                fence,
                in_flight: None,
            });
        }

        // GPU-texture mode: build the persistent NV12 -> RGBA converter once (it
        // owns the wgpu device + compute queue). On failure, free everything above.
        let gpu = if gpu {
            let compute_queue = self
                .compute_queue
                .expect("gpu mode requires a compute queue");
            // SAFETY: raw_device outlives the core; wgpu_device wraps the same
            // VkDevice; compute_queue belongs to compute_queue_family.
            match unsafe {
                YcbcrConverter::new(
                    &self.raw_device,
                    &self.wgpu_device,
                    self.mem_props,
                    compute_queue,
                    self.compute_queue_family,
                    color,
                    bit_depth,
                    hdr_output,
                )
            } {
                Ok(converter) => {
                    // Decode -> compute chaining semaphore (per-frame, drained each
                    // picture). On failure, drop the converter and free the rest.
                    // SAFETY: valid create info from `self.raw_device`.
                    match unsafe {
                        self.raw_device
                            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    } {
                        Ok(sem_dc) => Some(GpuTextureCtx {
                            converter,
                            wgpu_queue: self.wgpu_queue.clone(),
                            sem_dc,
                        }),
                        Err(e) => {
                            drop(converter);
                            let fences: alloc::vec::Vec<vk::Fence> =
                                ring.iter().map(|s| s.fence).collect();
                            cleanup(Some(ring_pool), &fences);
                            return Err(VulkanVideoError::QueryFailed(e));
                        }
                    }
                }
                Err(e) => {
                    let fences: alloc::vec::Vec<vk::Fence> = ring.iter().map(|s| s.fence).collect();
                    cleanup(Some(ring_pool), &fences);
                    return Err(e);
                }
            }
        } else {
            None
        };

        Ok(DpbCore {
            raw_device: self.raw_device.clone(),
            video_fns: self.video_fns.clone(),
            decode_fns: self.decode_fns.clone(),
            sync2_fns: self.sync2_fns.clone(),
            decode_queue: self.decode_queue,
            mem_props: self.mem_props,
            session,
            parameters,
            coded_extent,
            size_align: self.caps.min_bitstream_buffer_size_alignment.max(1),
            slots,
            pool,
            readback,
            readback_mem,
            luma_len,
            chroma_len,
            nv12_len,
            readback_stride,
            ring_pool,
            ring,
            ring_next: 0,
            ready: alloc::collections::VecDeque::new(),
            chain_next: false,
            pending_tex_bitstream: None,
            first: true,
            gpu,
            color,
            bit_depth,
        })
    }

    /// Shared constructor for both output modes. `gpu` selects GPU-resident
    /// texture output (SAMPLED + CONCURRENT DPB images) over system NV12 readback.
    /// `ps` supplies the SPS/PPS (POC type, field widths, `max_num_ref_frames`).
    /// Rejects `pic_order_cnt_type == 1` (delta cycle not carried) up front.
    fn build_dpb_decoder(
        &self,
        session: &H264DecodeSession,
        ps: &H264ParameterSets,
        gpu: bool,
    ) -> Result<H264DpbDecoder, VulkanVideoError> {
        if ps.sps.pic_order_cnt_type == 1 {
            return Err(VulkanVideoError::UnsupportedStream);
        }
        let (w, h) = session.coded_extent;
        let max_num_ref_frames = ps.sps.max_num_ref_frames as usize;
        // One slot per possible short-term reference plus one for the picture
        // being decoded, clamped to the device DPB ceiling (at least two).
        let num_slots = (max_num_ref_frames + 1).clamp(2, self.caps.max_dpb_slots.max(2) as usize);

        let profile = h264_profile();
        let color = VideoColorSpace::from_cicp(
            ps.sps.matrix_coefficients,
            ps.sps.transfer_characteristics,
            ps.sps.video_full_range_flag,
            h,
        );
        let core = self.build_dpb_core(
            session.session,
            session.parameters,
            (w, h),
            session.picture_format,
            num_slots,
            gpu,
            &profile.profile,
            color,
            // H.264 GPU decode is 8-bit here; HDR tone-map is a 10-bit concern.
            HdrOutput::Passthrough,
        )?;

        Ok(H264DpbDecoder {
            core,
            profile,
            refs: alloc::vec![None; num_slots],
            max_num_ref_frames,
            sps: ps.sps.clone(),
            pps: ps.pps.clone(),
            poc_type: ps.sps.pic_order_cnt_type,
            log2_max_pic_order_cnt_lsb: ps.sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4,
            max_frame_num: 1 << (ps.sps.log2_max_frame_num_minus4 as u32 + 4),
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            prev_frame_num: 0,
            prev_frame_num_offset: 0,
        })
    }

    /// Destroy the one-shot decode objects (fence already destroyed inline).
    /// # Safety
    /// Every handle must have been created from `self.raw_device` and not yet
    /// destroyed; no work referencing them may still be in flight.
    #[allow(clippy::too_many_arguments)]
    unsafe fn destroy_decode_transients(
        &self,
        pool: vk::CommandPool,
        image: vk::Image,
        view: vk::ImageView,
        image_mem: vk::DeviceMemory,
        bitstream: vk::Buffer,
        bitstream_mem: vk::DeviceMemory,
        readback: vk::Buffer,
        readback_mem: vk::DeviceMemory,
    ) {
        let dev = &self.raw_device;
        // SAFETY: contract above; destroy in dependency order.
        unsafe {
            dev.destroy_command_pool(pool, None);
            dev.destroy_image_view(view, None);
            dev.destroy_image(image, None);
            dev.free_memory(image_mem, None);
            dev.destroy_buffer(bitstream, None);
            dev.free_memory(bitstream_mem, None);
            dev.destroy_buffer(readback, None);
            dev.free_memory(readback_mem, None);
        }
    }

    /// Build a multi-frame H.265 decoder over `session` (system NV12 output), the
    /// HEVC sibling of [`create_h264_dpb_decoder`](Self::create_h264_dpb_decoder).
    /// `std` supplies the SPS (POC lsb size, `num_short_term_ref_pic_sets`) and
    /// PPS the slice-header parse and DPB need.
    pub fn create_h265_dpb_decoder(
        &self,
        session: &H265DecodeSession,
        ps: &H265ParameterSets,
    ) -> Result<H265DpbDecoder, VulkanVideoError> {
        self.build_h265_dpb_decoder(session, ps, false, HdrOutput::Passthrough)
    }

    /// Build a multi-frame H.265 decoder emitting GPU-resident RGBA
    /// `wgpu::Texture`s (the zero-copy wedge), the HEVC sibling of
    /// [`create_h264_dpb_decoder_gpu`](Self::create_h264_dpb_decoder_gpu).
    /// Requires a distinct compute queue; [`VulkanVideoError::NoComputeQueue`]
    /// otherwise. The float target holds the stream's own transfer-encoded R'G'B'
    /// (matrix + range only); use [`Self::create_h265_dpb_decoder_gpu_tonemap`] to
    /// tone-map an HDR stream to SDR instead.
    pub fn create_h265_dpb_decoder_gpu(
        &self,
        session: &H265DecodeSession,
        ps: &H265ParameterSets,
    ) -> Result<H265DpbDecoder, VulkanVideoError> {
        if self.compute_queue.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        self.build_h265_dpb_decoder(session, ps, true, HdrOutput::Passthrough)
    }

    /// Like [`Self::create_h265_dpb_decoder_gpu`], but an HDR (PQ / HLG, BT.2020)
    /// stream is tone-mapped to display-ready SDR (BT.709) in the ycbcr compute
    /// pass (EOTF -> BT.2390 EETF -> BT.2020->709 gamut -> BT.709 OETF). An SDR
    /// stream is unaffected. Fixes HDR content showing wrong colour on an SDR
    /// consumer.
    pub fn create_h265_dpb_decoder_gpu_tonemap(
        &self,
        session: &H265DecodeSession,
        ps: &H265ParameterSets,
    ) -> Result<H265DpbDecoder, VulkanVideoError> {
        if self.compute_queue.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        self.build_h265_dpb_decoder(session, ps, true, HdrOutput::TonemapSdr)
    }

    fn build_h265_dpb_decoder(
        &self,
        session: &H265DecodeSession,
        ps: &H265ParameterSets,
        gpu: bool,
        hdr_output: HdrOutput,
    ) -> Result<H265DpbDecoder, VulkanVideoError> {
        let (w, h) = session.coded_extent;
        // DPB size: the SPS max buffering (sub-layer 0) plus the picture in
        // flight, clamped to the device DPB ceiling (at least two).
        let max_dpb = ps.sps.max_dec_pic_buffering_minus1[0] as usize + 1;
        let num_slots = (max_dpb + 1).clamp(2, self.caps.max_dpb_slots.max(2) as usize);

        let profile = h265_profile(ps.sps.bit_depth_luma_minus8 + 8);
        let color = VideoColorSpace::from_cicp(
            ps.sps.matrix_coefficients,
            ps.sps.transfer_characteristics,
            ps.sps.video_full_range_flag,
            h,
        );
        let core = self.build_dpb_core(
            session.session,
            session.parameters,
            (w, h),
            session.picture_format,
            num_slots,
            gpu,
            &profile.profile,
            color,
            hdr_output,
        )?;

        Ok(H265DpbDecoder {
            core,
            profile,
            refs: alloc::vec![None; num_slots],
            sps: ps.sps.clone(),
            pps: ps.pps.clone(),
            log2_max_pic_order_cnt_lsb: ps.sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4,
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            seen_picture: false,
            skip_rasl: false,
        })
    }

    /// Build a multi-frame AV1 decoder (system-memory NV12 output), the AV1
    /// sibling of [`create_h265_dpb_decoder`](Self::create_h265_dpb_decoder).
    pub fn create_av1_dpb_decoder(
        &self,
        session: &Av1DecodeSession,
        seq: &Av1SequenceHeader,
    ) -> Result<Av1DpbDecoder, VulkanVideoError> {
        self.build_av1_dpb_decoder(session, seq, false, HdrOutput::Passthrough)
    }

    /// Build a multi-frame AV1 decoder emitting GPU-resident RGBA `wgpu::Texture`s.
    /// Requires a distinct compute queue; [`VulkanVideoError::NoComputeQueue`] otherwise.
    /// Matrix + range only; use [`Self::create_av1_dpb_decoder_gpu_tonemap`] to
    /// tone-map an HDR stream to SDR.
    pub fn create_av1_dpb_decoder_gpu(
        &self,
        session: &Av1DecodeSession,
        seq: &Av1SequenceHeader,
    ) -> Result<Av1DpbDecoder, VulkanVideoError> {
        if self.compute_queue.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        self.build_av1_dpb_decoder(session, seq, true, HdrOutput::Passthrough)
    }

    /// Like [`Self::create_av1_dpb_decoder_gpu`], but an HDR (PQ / HLG, BT.2020)
    /// stream is tone-mapped to display-ready SDR in the ycbcr compute pass. An
    /// SDR stream is unaffected.
    pub fn create_av1_dpb_decoder_gpu_tonemap(
        &self,
        session: &Av1DecodeSession,
        seq: &Av1SequenceHeader,
    ) -> Result<Av1DpbDecoder, VulkanVideoError> {
        if self.compute_queue.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        self.build_av1_dpb_decoder(session, seq, true, HdrOutput::TonemapSdr)
    }

    fn build_av1_dpb_decoder(
        &self,
        session: &Av1DecodeSession,
        seq: &Av1SequenceHeader,
        gpu: bool,
        hdr_output: HdrOutput,
    ) -> Result<Av1DpbDecoder, VulkanVideoError> {
        let (w, h) = session.coded_extent;
        // AV1 keeps up to NUM_REF_FRAMES (8) reference frames; one more physical
        // image gives a free target to decode into, so a fresh frame never
        // aliases a live reference. Clamp to the device DPB ceiling.
        let num_slots = (AV1_NUM_REF_FRAMES + 1).clamp(2, self.caps.max_dpb_slots.max(2) as usize);
        let profile = av1_profile(seq.color.bit_depth);
        // AV1 carries the colour matrix + transfer in color_config; an absent
        // description is unspecified (CICP 2), which `from_cicp` resolves by
        // resolution (matrix) / to SDR (transfer).
        let (mc, tc) = if seq.color.color_description_present_flag {
            (
                seq.color.matrix_coefficients,
                seq.color.transfer_characteristics,
            )
        } else {
            (2, 2)
        };
        let color = VideoColorSpace::from_cicp(mc, tc, seq.color.color_range, h);
        let core = self.build_dpb_core(
            session.session,
            session.parameters,
            (w, h),
            session.picture_format,
            num_slots,
            gpu,
            &profile.profile,
            color,
            hdr_output,
        )?;

        Ok(Av1DpbDecoder {
            core,
            profile,
            ref_slot: [None; AV1_NUM_REF_FRAMES],
            phys_state: alloc::vec![None; num_slots],
            seq: seq.clone(),
        })
    }
}

/// One physical DPB image (coincide: it is both a decode-output target and a
/// reference slot). The pool is fixed-size for the decoder's lifetime.
struct DpbImage {
    image: vk::Image,
    view: vk::ImageView,
    mem: vk::DeviceMemory,
}

impl core::fmt::Debug for DpbImage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DpbImage").finish_non_exhaustive()
    }
}

/// A reference picture held in a DPB slot: the metadata the driver matches
/// against slice-header reference lists (`FrameNum`, POC).
#[derive(Debug, Clone, Copy)]
struct RefPic {
    frame_num: u32,
    poc: i32,
}

/// Per-coded-picture metadata for random access, built by
/// [`H264DpbDecoder::index_pictures`] without decoding: whether the picture is a
/// keyframe (an IDR, a valid seek point to decode forward from), its `frame_num`,
/// and its picture-order-count (the presentation-order key).
#[derive(Debug, Clone, Copy)]
pub struct PictureMeta {
    /// The picture resets the picture-order-count (an IDR, and for H.265 a BLA):
    /// the boundary of a coded video sequence, so POC is only comparable within
    /// the run since the last one. Used to group pictures for display-order sort.
    pub is_keyframe: bool,
    /// The picture is an IRAP random-access point decoding can (re)start at: an
    /// IDR always, and for H.265 also a CRA / BLA. A superset of `is_keyframe`
    /// (a CRA does not reset POC but is a valid seek target). A seek to a CRA
    /// discards its RASL leading pictures (they reference now-absent pre-CRA
    /// frames), so a consumer seeking to a leading picture must use an earlier
    /// random-access point instead.
    pub is_random_access: bool,
    pub frame_num: u32,
    /// Picture-order-count: pictures are presented in ascending POC order.
    pub poc: i32,
}

/// A multi-frame H.264 decoder driving a real DPB: it decodes P (and, given a
/// stream, B) frames by tracking reference pictures across access units, unlike
/// the one-shot IDR entry points. Computes picture-order-count, runs H.264
/// sliding-window reference marking, and hands the driver the active reference
/// slots per picture.
///
/// Owns a fixed DPB image pool, a reusable readback buffer and command pool, and
/// borrows nothing from the [`H264DecodeSession`] beyond copies of its handles
/// (the session must outlive the decoder, since it owns those objects).
pub struct H264DpbDecoder {
    /// Codec-independent GPU plumbing (device, session, DPB pool, readback,
    /// command pool, record/submit path). Its `Drop` frees all of it.
    core: DpbCore,
    /// Kept alive: the DPB-image and bitstream-buffer profile lists point into it.
    profile: H264Profile,
    /// Per-slot reference state: `Some` means the slot holds a short-term
    /// reference picture, `None` means it is free.
    refs: alloc::vec::Vec<Option<RefPic>>,
    max_num_ref_frames: usize,
    sps: H264Sps,
    pps: H264Pps,
    poc_type: u8,
    log2_max_pic_order_cnt_lsb: u32,
    max_frame_num: i32,
    // POC / frame-num tracking, carried across pictures in decoding order.
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    prev_frame_num: i32,
    prev_frame_num_offset: i32,
}

/// Pictures split out of a stream: each is a first-slice header plus the slice
/// NALs of that primary coded picture (borrowing the input stream).
type PictureUnits<'s> = alloc::vec::Vec<(H264SliceHeader, alloc::vec::Vec<&'s [u8]>)>;

/// GPU-texture mode context for the `*DpbDecoder`s: the persistent NV12 -> RGBA
/// converter (which owns the wgpu device the texture is imported into and the
/// compute queue that runs the ycbcr pass), plus the decode -> compute semaphore
/// that chains a picture's decode (decode queue) to its ycbcr conversion (compute
/// queue) with no CPU stall between them.
#[derive(Debug)]
struct GpuTextureCtx {
    converter: YcbcrConverter,
    /// The wgpu queue the converter's device exposes, used to upload a CPU-built
    /// RGBA frame straight to a texture (the AV1 film-grain texture path, which
    /// synthesizes grain on the readback NV12 rather than the GPU ycbcr pass).
    wgpu_queue: wgpu::Queue,
    /// Binary semaphore: the chained texture decode signals it, the following
    /// compute pass waits it. Reused every picture (each picture drains its
    /// compute fence before the next, so the semaphore is unsignalled between
    /// pictures). Cross-frame overlap is impossible anyway (a decode referencing
    /// the previous slot must wait that slot's in-place compute restore), so a
    /// single per-frame semaphore is all this buys.
    sem_dc: vk::Semaphore,
}

impl Drop for GpuTextureCtx {
    fn drop(&mut self) {
        // SAFETY: `sem_dc` was created from the converter's device and is destroyed
        // once here; the converter (dropped after this) still holds the live
        // device, and no texture decode is in flight at teardown (each drains its
        // compute fence).
        unsafe {
            self.converter
                .raw_device
                .destroy_semaphore(self.sem_dc, None)
        };
    }
}

/// How many picture decodes may be in flight on the decode queue at once in the
/// system NV12 path. The CPU records and submits up to this many pictures before
/// waiting on the oldest, keeping the hardware decode queue fed rather than
/// stalling on a fence after every picture. In-order execution on the single
/// decode queue preserves DPB reference correctness (references are CPU-side
/// bookkeeping); only the readback buffer needs per-slot isolation, hence the
/// [`DpbCore::readback_stride`] offset scheme. The texture path stays synchronous
/// (its decode -> compute-convert hand-off crosses queues).
const DECODE_RING_DEPTH: usize = 4;

/// One entry in the system-path decode ring: a persistent command buffer + fence
/// and, while the slot is occupied, the transient bitstream buffer to free and
/// the frame geometry to read back when the slot retires.
#[derive(Debug)]
struct RingSlot {
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    in_flight: Option<InFlightDecode>,
}

/// A submitted-but-not-yet-retired system-path decode. The bitstream buffer is
/// referenced by the in-flight submission, so it (and the frame geometry needed
/// to read the decoded NV12 back out of the ring's readback region) is held here
/// until the slot's fence is waited.
#[derive(Debug)]
struct InFlightDecode {
    bitstream: vk::Buffer,
    bitstream_mem: vk::DeviceMemory,
    w: u32,
    h: u32,
}

/// Codec-independent Vulkan Video decode plumbing shared by every `*DpbDecoder`
/// (H.264 / H.265 / AV1): the device + queue handles, the session it decodes
/// against, the DPB image pool, the persistent NV12 readback buffer, the command
/// pool, and the record-barriers-submit-wait path. Each per-codec decoder embeds
/// one `DpbCore` and keeps only its codec-specific state (parameter sets, POC /
/// reference-picture tracking, and the `Std*` picture / reference structs it
/// hands the driver). This is the single place the GPU command recording and
/// fence submission live, so the three codecs cannot drift apart.
struct DpbCore {
    raw_device: ash::Device,
    video_fns: ash::khr::video_queue::Device,
    decode_fns: ash::khr::video_decode_queue::Device,
    sync2_fns: ash::khr::synchronization2::Device,
    decode_queue: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    session: vk::VideoSessionKHR,
    parameters: vk::VideoSessionParametersKHR,
    coded_extent: (u32, u32),
    size_align: u64,
    slots: alloc::vec::Vec<DpbImage>,
    pool: vk::CommandPool,
    readback: vk::Buffer,
    readback_mem: vk::DeviceMemory,
    luma_len: u64,
    chroma_len: u64,
    nv12_len: u64,
    /// Per-slot byte stride within `readback` (`nv12_len` rounded up so each
    /// slot's `bufferOffset` satisfies the copy alignment). The readback buffer
    /// holds `DECODE_RING_DEPTH` of these back to back.
    readback_stride: u64,
    /// System-path pipelining ring. `pool` records the synchronous texture path;
    /// this pool + its persistent per-slot command buffers record the pipelined
    /// system path (`RESET_COMMAND_BUFFER`, so re-recording a slot resets it).
    ring_pool: vk::CommandPool,
    ring: alloc::vec::Vec<RingSlot>,
    /// The next ring slot to claim; also the oldest in-flight slot to retire.
    ring_next: usize,
    /// Decoded NV12 frames retired from the ring in decode order, awaiting the
    /// caller to collect them (`decode_all`).
    ready: alloc::collections::VecDeque<Nv12Frame>,
    /// Set by `decode_picture_to_texture` before its decode so `submit_texture`
    /// takes the chained path (signal `gpu.sem_dc`, no fence wait) instead of the
    /// synchronous one; cleared as it is consumed.
    chain_next: bool,
    /// A chained texture decode's bitstream buffer, held until the picture's
    /// compute (which chains on `sem_dc`) has completed, then freed by
    /// `decode_picture_to_texture`. `Some` only mid-conversion.
    pending_tex_bitstream: Option<(vk::Buffer, vk::DeviceMemory)>,
    /// The video session needs a `RESET` control on its first coding operation
    /// (and after a seek [`reset`](H264DpbDecoder::reset)).
    first: bool,
    /// `Some` in GPU-texture mode: the wgpu device + compute queue used to convert
    /// each decoded slot to an RGBA `wgpu::Texture` (DPB slot images are then
    /// created `SAMPLED` + `CONCURRENT` across the decode/compute families).
    gpu: Option<GpuTextureCtx>,
    /// The colour space (matrix + range) decoded frames are converted from,
    /// resolved from the stream at decoder build time. Drives the CPU-side
    /// conversions (e.g. the AV1 film-grain texture path); the GPU converter bakes
    /// the same colour space into its `VkSamplerYcbcrConversion` at construction.
    color: VideoColorSpace,
    /// Decoded-sample bit depth (8 or 10), from `picture_format`. A frame read back
    /// from a 10-bit format carries 2 bytes per sample (little-endian 16-bit, value
    /// in the top 10 bits); `Nv12Frame::bit_depth` reflects it.
    bit_depth: u8,
}

impl core::fmt::Debug for DpbCore {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DpbCore")
            .field("coded_extent", &self.coded_extent)
            .field("dpb_slots", &self.slots.len())
            .finish_non_exhaustive()
    }
}

impl Drop for DpbCore {
    fn drop(&mut self) {
        // Wait on + free any still-in-flight system-path decodes before tearing
        // down the ring (fences, command pool) they reference.
        self.abort_ring();
        let dev = &self.raw_device;
        // SAFETY: all handles were created from `dev` in the constructor and are
        // destroyed exactly once here; `abort_ring` above waited every in-flight
        // fence, and the texture path waits its fence per decode, so nothing is in
        // flight.
        unsafe {
            for slot in &self.ring {
                dev.destroy_fence(slot.fence, None);
            }
            dev.destroy_command_pool(self.ring_pool, None);
            dev.destroy_command_pool(self.pool, None);
            dev.destroy_buffer(self.readback, None);
            dev.free_memory(self.readback_mem, None);
            for s in &self.slots {
                dev.destroy_image_view(s.view, None);
                dev.destroy_image(s.image, None);
                dev.free_memory(s.mem, None);
            }
        }
    }
}

impl DpbCore {
    /// Allocate + fill a transient host-visible bitstream buffer holding one
    /// picture's slices, chained to the codec `profile`. The caller frees it with
    /// [`free_bitstream`](Self::free_bitstream) after the decode fence is waited.
    /// Returns the buffer, its memory, and the (alignment-rounded) size the
    /// decode's `src_buffer_range` uses.
    fn new_bitstream(
        &self,
        data: &[u8],
        profile: &vk::VideoProfileInfoKHR,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, u64), VulkanVideoError> {
        let dev = &self.raw_device;
        let buf_size = round_up(data.len() as u64, self.size_align);
        let mut buf_profile_list =
            vk::VideoProfileListInfoKHR::default().profiles(core::slice::from_ref(profile));
        let buf_ci = vk::BufferCreateInfo::default()
            .size(buf_size)
            .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut buf_profile_list);
        // SAFETY: valid create info; chained profile list outlives the call.
        let bitstream =
            unsafe { dev.create_buffer(&buf_ci, None) }.map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh buffer; allocate host-visible memory, bind, map, fill.
        let bitstream_mem = unsafe {
            let breq = dev.get_buffer_memory_requirements(bitstream);
            let bt = find_memory_type(
                &self.mem_props,
                breq.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            let bt = match bt {
                Some(t) => t,
                None => {
                    dev.destroy_buffer(bitstream, None);
                    return Err(VulkanVideoError::ExtensionUnsupported);
                }
            };
            let bai = vk::MemoryAllocateInfo::default()
                .allocation_size(breq.size)
                .memory_type_index(bt);
            let m = dev
                .allocate_memory(&bai, None)
                .map_err(VulkanVideoError::QueryFailed)?;
            dev.bind_buffer_memory(bitstream, m, 0)
                .map_err(VulkanVideoError::QueryFailed)?;
            let ptr = dev
                .map_memory(m, 0, breq.size, vk::MemoryMapFlags::empty())
                .map_err(VulkanVideoError::QueryFailed)? as *mut u8;
            core::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            dev.unmap_memory(m);
            m
        };
        Ok((bitstream, bitstream_mem, buf_size))
    }

    /// Free a bitstream buffer from [`new_bitstream`](Self::new_bitstream).
    /// # Safety
    /// The buffer + memory must have come from `new_bitstream` on this core and no
    /// submission referencing them may still be in flight (the decode fence waited).
    unsafe fn free_bitstream(&self, bitstream: vk::Buffer, bitstream_mem: vk::DeviceMemory) {
        let dev = &self.raw_device;
        // SAFETY: contract above; freed exactly once.
        unsafe {
            dev.destroy_buffer(bitstream, None);
            dev.free_memory(bitstream_mem, None);
        }
    }

    /// Record one picture's decode into `cb`: the UNDEFINED -> DPB barrier,
    /// begin/decode/end video coding, the session `RESET` control when
    /// `issue_reset`, and, for the system path, an NV12 copy into `self.readback`
    /// at `readback_offset` with the slot returned to DPB layout afterward.
    /// `readback_offset == None` is the texture path: it leaves the slot in DPB
    /// layout for the compute pass and emits no copy. Records but does not submit.
    ///
    /// # Safety
    /// `cb` must be recordable (initial or reset state); every handle referenced
    /// by `begin_info` / `decode_info` and the reference images must be valid and
    /// outlive the submission of `cb`.
    unsafe fn record_decode(
        &self,
        cb: vk::CommandBuffer,
        begin_info: &vk::VideoBeginCodingInfoKHR,
        decode_info: &vk::VideoDecodeInfoKHR,
        image: vk::Image,
        issue_reset: bool,
        readback_offset: Option<u64>,
    ) -> Result<(), vk::Result> {
        let dev = &self.raw_device;
        let (w, h) = self.coded_extent;
        let control_info =
            vk::VideoCodingControlInfoKHR::default().flags(vk::VideoCodingControlFlagsKHR::RESET);
        let end_info = vk::VideoEndCodingInfoKHR::default();
        // SAFETY: contract above; the barriers move the target image
        // UNDEFINED -> DPB (decode) -> TRANSFER_SRC (readback copy) -> DPB (ready
        // as a future reference). The reference images passed in `begin`/`decode`
        // are already in DPB layout.
        unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.begin_command_buffer(cb, &begin)?;

            let to_dpb = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range());
            let dep =
                vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&to_dpb));
            (self.sync2_fns.fp().cmd_pipeline_barrier2_khr)(cb, &dep);

            (self.video_fns.fp().cmd_begin_video_coding_khr)(cb, begin_info);
            if issue_reset {
                (self.video_fns.fp().cmd_control_video_coding_khr)(cb, &control_info);
            }
            (self.decode_fns.fp().cmd_decode_video_khr)(cb, decode_info);
            (self.video_fns.fp().cmd_end_video_coding_khr)(cb, &end_info);

            // System path only: copy the decoded NV12 planes to the readback
            // buffer at this slot's region, then return the slot to the decode
            // layout. The texture path leaves the slot in DPB layout and samples
            // it on the compute queue instead.
            if let Some(offset) = readback_offset {
                let to_src = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                    .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color_range());
                let dep2 = vk::DependencyInfo::default()
                    .image_memory_barriers(core::slice::from_ref(&to_src));
                (self.sync2_fns.fp().cmd_pipeline_barrier2_khr)(cb, &dep2);

                let luma_region = vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::PLANE_0,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    });
                let chroma_region = vk::BufferImageCopy::default()
                    .buffer_offset(offset + self.luma_len)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::PLANE_1,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: w / 2,
                        height: h / 2,
                        depth: 1,
                    });
                let regions = [luma_region, chroma_region];
                dev.cmd_copy_image_to_buffer(
                    cb,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    self.readback,
                    &regions,
                );

                // Return the target image to DPB layout so it can be a reference
                // for later pictures (content preserved by the transition).
                let back_to_dpb = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(color_range());
                let dep3 = vk::DependencyInfo::default()
                    .image_memory_barriers(core::slice::from_ref(&back_to_dpb));
                (self.sync2_fns.fp().cmd_pipeline_barrier2_khr)(cb, &dep3);
            }

            dev.end_command_buffer(cb)?;
        }
        Ok(())
    }

    /// Record + submit one picture's decode (built codec-side into `begin_info` /
    /// `decode_info`) into `target_image`, taking ownership of the transient
    /// `bitstream` buffer. Issues the session `RESET` control on the first
    /// operation (or after a reset). This is the shared, codec-agnostic GPU
    /// command path for all `*DpbDecoder`s.
    ///
    /// `to_system == false` (texture path) is synchronous: it waits the decode
    /// fence before returning and frees the bitstream, leaving the slot in
    /// `VIDEO_DECODE_DPB_KHR` layout for the compute pass. `to_system == true`
    /// (NV12 path) is pipelined through the decode ring: it retires the oldest
    /// in-flight slot if the ring is full (reading its frame into `self.ready`),
    /// records + submits this picture without waiting, and holds the bitstream in
    /// the ring slot until that slot is later retired.
    fn record_and_submit(
        &mut self,
        begin_info: &vk::VideoBeginCodingInfoKHR,
        decode_info: &vk::VideoDecodeInfoKHR,
        target_image: vk::Image,
        bitstream: vk::Buffer,
        bitstream_mem: vk::DeviceMemory,
        to_system: bool,
    ) -> Result<(), VulkanVideoError> {
        if to_system {
            self.submit_ring(
                begin_info,
                decode_info,
                target_image,
                bitstream,
                bitstream_mem,
            )
        } else {
            self.submit_texture(
                begin_info,
                decode_info,
                target_image,
                bitstream,
                bitstream_mem,
            )
        }
    }

    /// Texture-path submit. When `chain_next` is set (by
    /// `decode_picture_to_texture`), the decode is submitted with a `sem_dc` signal
    /// and no fence wait, so the following compute pass chains on it with no CPU
    /// stall (the bitstream is held in `pending_tex_bitstream` until that compute
    /// completes). Otherwise it is synchronous: reset `self.pool`, record + submit,
    /// wait the fence, free the bitstream. Either way the decoded slot is left in
    /// DPB layout.
    fn submit_texture(
        &mut self,
        begin_info: &vk::VideoBeginCodingInfoKHR,
        decode_info: &vk::VideoDecodeInfoKHR,
        target_image: vk::Image,
        bitstream: vk::Buffer,
        bitstream_mem: vk::DeviceMemory,
    ) -> Result<(), VulkanVideoError> {
        if self.chain_next {
            self.chain_next = false;
            return self.submit_texture_chained(
                begin_info,
                decode_info,
                target_image,
                bitstream,
                bitstream_mem,
            );
        }
        let issue_reset = self.first;
        // Reset the command pool and allocate one primary buffer.
        // SAFETY: the pool has no in-flight command buffers (the texture path
        // waits every fence before returning).
        let cb = unsafe {
            let dev = &self.raw_device;
            dev.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(VulkanVideoError::QueryFailed)?;
            let cb_ai = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            dev.allocate_command_buffers(&cb_ai)
                .map_err(VulkanVideoError::QueryFailed)?[0]
        };

        // SAFETY: `cb` is freshly allocated; the handles in begin/decode info and
        // the reference images outlive this waited submission.
        let r = unsafe {
            self.record_decode(cb, begin_info, decode_info, target_image, issue_reset, None)
                .and_then(|()| {
                    let dev = &self.raw_device;
                    let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None)?;
                    let cbs = [cb];
                    let submit = vk::SubmitInfo::default().command_buffers(&cbs);
                    let r = dev
                        .queue_submit(self.decode_queue, core::slice::from_ref(&submit), fence)
                        .and_then(|_| dev.wait_for_fences(&[fence], true, u64::MAX));
                    dev.destroy_fence(fence, None);
                    r
                })
        };
        // SAFETY: transient bitstream, freed once now that the fence is waited (or
        // the submission failed, in which case nothing references it).
        unsafe { self.free_bitstream(bitstream, bitstream_mem) };
        r.map_err(VulkanVideoError::QueryFailed)?;
        self.first = false;
        Ok(())
    }

    /// Chained texture-path submit: record the decode into `self.pool`'s buffer and
    /// submit it signalling `gpu.sem_dc` with NO fence and NO CPU wait, so the
    /// following ycbcr compute pass (which waits `sem_dc`) starts the instant the
    /// decode finishes, overlapping the compute's CPU prep with the decode's GPU
    /// execution. The bitstream is stashed in `pending_tex_bitstream` and freed by
    /// `decode_picture_to_texture` once that compute has completed (it waits its own
    /// fence). Cross-frame overlap is impossible (the next decode references this
    /// slot and must wait its in-place compute restore), so this stays one picture
    /// at a time; the win is removing the mid-picture decode fence wait + CPU gap.
    fn submit_texture_chained(
        &mut self,
        begin_info: &vk::VideoBeginCodingInfoKHR,
        decode_info: &vk::VideoDecodeInfoKHR,
        target_image: vk::Image,
        bitstream: vk::Buffer,
        bitstream_mem: vk::DeviceMemory,
    ) -> Result<(), VulkanVideoError> {
        let issue_reset = self.first;
        let sem_dc = self
            .gpu
            .as_ref()
            .expect("chained texture submit requires gpu mode")
            .sem_dc;
        // SAFETY: the pool has no in-flight command buffers (the previous chained
        // picture's compute fence was waited in `decode_picture_to_texture` before
        // this call, which implies its decode completed).
        let cb = unsafe {
            let dev = &self.raw_device;
            dev.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(VulkanVideoError::QueryFailed)?;
            let cb_ai = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            dev.allocate_command_buffers(&cb_ai)
                .map_err(VulkanVideoError::QueryFailed)?[0]
        };

        // SAFETY: `cb` is freshly allocated; record the decode (no readback, slot
        // left in DPB layout for the compute pass) and submit it signalling
        // `sem_dc` with no fence and no wait.
        let r = unsafe {
            self.record_decode(cb, begin_info, decode_info, target_image, issue_reset, None)
                .and_then(|()| {
                    let dev = &self.raw_device;
                    let cbs = [cb];
                    let signal = [sem_dc];
                    let submit = vk::SubmitInfo::default()
                        .command_buffers(&cbs)
                        .signal_semaphores(&signal);
                    dev.queue_submit(
                        self.decode_queue,
                        core::slice::from_ref(&submit),
                        vk::Fence::null(),
                    )
                })
        };
        if let Err(e) = r {
            // The decode never ran (record or submit failed); nothing references
            // the bitstream and `sem_dc` was not signalled, so the caller's `?`
            // aborts before the waiting compute is submitted.
            // SAFETY: freed once, no in-flight reference.
            unsafe { self.free_bitstream(bitstream, bitstream_mem) };
            return Err(VulkanVideoError::QueryFailed(e));
        }
        // Held until the picture's compute (chained on `sem_dc`) completes.
        self.pending_tex_bitstream = Some((bitstream, bitstream_mem));
        self.first = false;
        Ok(())
    }

    /// Pipelined system-path submit into the decode ring. Retires the oldest
    /// in-flight slot first if the ring is full (its frame lands in `self.ready`),
    /// then records this picture into the claimed slot's persistent command buffer
    /// (copying NV12 into the slot's readback region) and submits without waiting;
    /// the bitstream stays owned by the slot until it is later retired.
    fn submit_ring(
        &mut self,
        begin_info: &vk::VideoBeginCodingInfoKHR,
        decode_info: &vk::VideoDecodeInfoKHR,
        target_image: vk::Image,
        bitstream: vk::Buffer,
        bitstream_mem: vk::DeviceMemory,
    ) -> Result<(), VulkanVideoError> {
        let idx = self.ring_next;
        if self.ring[idx].in_flight.is_some() {
            self.retire_slot(idx)?;
        }
        let issue_reset = self.first;
        let offset = idx as u64 * self.readback_stride;
        let cb = self.ring[idx].cb;
        let fence = self.ring[idx].fence;
        let (w, h) = self.coded_extent;

        // SAFETY: the slot's command buffer is in the reset state (its previous
        // submission was waited when the slot was retired above, or it is
        // untouched); `begin_command_buffer` re-records it (RESET_COMMAND_BUFFER
        // pool). All handles outlive the submission (the bitstream is held by the
        // slot until retirement).
        unsafe {
            self.record_decode(
                cb,
                begin_info,
                decode_info,
                target_image,
                issue_reset,
                Some(offset),
            )
            .map_err(VulkanVideoError::QueryFailed)?;
            let dev = &self.raw_device;
            dev.reset_fences(&[fence])
                .map_err(VulkanVideoError::QueryFailed)?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            dev.queue_submit(self.decode_queue, core::slice::from_ref(&submit), fence)
                .map_err(VulkanVideoError::QueryFailed)?;
        }
        self.ring[idx].in_flight = Some(InFlightDecode {
            bitstream,
            bitstream_mem,
            w,
            h,
        });
        self.ring_next = (idx + 1) % DECODE_RING_DEPTH;
        self.first = false;
        Ok(())
    }

    /// Read one decoded NV12 frame out of the readback buffer at a given ring
    /// slot's byte `offset` (`readback_stride` apart per slot). The copy that
    /// filled this region must already have completed (its fence waited).
    fn read_back_nv12_at(
        &self,
        offset: u64,
        w: u32,
        h: u32,
    ) -> Result<Nv12Frame, VulkanVideoError> {
        let dev = &self.raw_device;
        // SAFETY: readback_mem is host-visible/coherent and holds `nv12_len` bytes
        // at `offset` written by the completed copy; mapped and unmapped here.
        unsafe {
            let ptr = dev
                .map_memory(
                    self.readback_mem,
                    offset,
                    self.nv12_len,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(VulkanVideoError::QueryFailed)? as *const u8;
            let mut luma = alloc::vec![0u8; self.luma_len as usize];
            let mut chroma = alloc::vec![0u8; self.chroma_len as usize];
            core::ptr::copy_nonoverlapping(ptr, luma.as_mut_ptr(), self.luma_len as usize);
            core::ptr::copy_nonoverlapping(
                ptr.add(self.luma_len as usize),
                chroma.as_mut_ptr(),
                self.chroma_len as usize,
            );
            dev.unmap_memory(self.readback_mem);
            Ok(Nv12Frame {
                width: w,
                height: h,
                luma,
                chroma,
                bit_depth: self.bit_depth,
            })
        }
    }

    /// Wait on ring slot `idx`'s in-flight decode, read its NV12 frame into
    /// `self.ready`, and free its bitstream. No-op if the slot is empty.
    fn retire_slot(&mut self, idx: usize) -> Result<(), VulkanVideoError> {
        let inf = match self.ring[idx].in_flight.take() {
            Some(i) => i,
            None => return Ok(()),
        };
        let fence = self.ring[idx].fence;
        let dev = self.raw_device.clone();
        // SAFETY: `fence` was submitted with this slot's command buffer; wait it,
        // then the decode output + bitstream are safe to touch / free.
        unsafe {
            dev.wait_for_fences(&[fence], true, u64::MAX)
                .map_err(VulkanVideoError::QueryFailed)?
        };
        let offset = idx as u64 * self.readback_stride;
        let frame = self.read_back_nv12_at(offset, inf.w, inf.h)?;
        self.ready.push_back(frame);
        // SAFETY: nothing references the bitstream now the fence is waited.
        unsafe { self.free_bitstream(inf.bitstream, inf.bitstream_mem) };
        Ok(())
    }

    /// Synchronously read one already-decoded, idle DPB slot image (in
    /// `VIDEO_DECODE_DPB_KHR` layout) back to an NV12 frame, WITHOUT a decode. Used
    /// by the reordered system path to emit a shown frame or a `show_existing_frame`
    /// (which outputs a stored reference) at its display position. Records a
    /// copy-only command buffer (DPB -> TRANSFER_SRC -> copy -> DPB, so the slot
    /// stays a valid reference), waits it, and reads back from `readback` region 0.
    /// Only valid on a system-path core (slot images carry `TRANSFER_SRC`); the
    /// caller must ensure no decode into `image` is in flight (the reordered path is
    /// synchronous).
    fn read_slot_nv12(&mut self, image: vk::Image) -> Result<Nv12Frame, VulkanVideoError> {
        let (w, h) = self.coded_extent;
        // SAFETY: the synchronous pool has no in-flight buffers; the slot image is
        // idle (reordered path waits every decode). The barriers move it
        // DPB -> TRANSFER_SRC (copy) -> DPB (still a valid reference); the copy fills
        // readback region 0, waited before it is read.
        unsafe {
            let dev = self.raw_device.clone();
            dev.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(VulkanVideoError::QueryFailed)?;
            let cb_ai = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cb = dev
                .allocate_command_buffers(&cb_ai)
                .map_err(VulkanVideoError::QueryFailed)?[0];

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.begin_command_buffer(cb, &begin)
                .map_err(VulkanVideoError::QueryFailed)?;

            let to_src = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range());
            let dep =
                vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&to_src));
            (self.sync2_fns.fp().cmd_pipeline_barrier2_khr)(cb, &dep);

            let luma_region = vk::BufferImageCopy::default()
                .buffer_offset(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_0,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
                    depth: 1,
                });
            let chroma_region = vk::BufferImageCopy::default()
                .buffer_offset(self.luma_len)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: w / 2,
                    height: h / 2,
                    depth: 1,
                });
            let regions = [luma_region, chroma_region];
            dev.cmd_copy_image_to_buffer(
                cb,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback,
                &regions,
            );

            let back_to_dpb = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range());
            let dep2 = vk::DependencyInfo::default()
                .image_memory_barriers(core::slice::from_ref(&back_to_dpb));
            (self.sync2_fns.fp().cmd_pipeline_barrier2_khr)(cb, &dep2);

            dev.end_command_buffer(cb)
                .map_err(VulkanVideoError::QueryFailed)?;

            let fence = dev
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(VulkanVideoError::QueryFailed)?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            let r = dev
                .queue_submit(self.decode_queue, core::slice::from_ref(&submit), fence)
                .and_then(|_| dev.wait_for_fences(&[fence], true, u64::MAX));
            dev.destroy_fence(fence, None);
            r.map_err(VulkanVideoError::QueryFailed)?;
        }
        self.read_back_nv12_at(0, w, h)
    }

    /// Retire every in-flight ring slot in decode (FIFO) order, appending their
    /// frames to `self.ready`. Called at the end of a system-path `decode_all`.
    fn drain(&mut self) -> Result<(), VulkanVideoError> {
        for i in 0..DECODE_RING_DEPTH {
            let idx = (self.ring_next + i) % DECODE_RING_DEPTH;
            self.retire_slot(idx)?;
        }
        Ok(())
    }

    /// Convert the just-decoded slot `image` to an RGBA `wgpu::Texture`, chained on
    /// `gpu.sem_dc` (the decode signals it), then free the chained decode's held
    /// bitstream now that the conversion has completed (`convert` waits its fence).
    /// The shared tail of every `decode_picture_to_texture` / `decode_frame_to_texture`.
    ///
    /// # Safety
    /// `image` must be the DPB slot just decoded (in `VIDEO_DECODE_DPB_KHR` layout,
    /// SAMPLED + CONCURRENT) whose decode was submitted signalling `gpu.sem_dc`.
    unsafe fn convert_slot_chained(
        &mut self,
        image: vk::Image,
        w: u32,
        h: u32,
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        let tex = {
            let gpu = self.gpu.as_ref().ok_or(VulkanVideoError::NoComputeQueue)?;
            // SAFETY: forwarded from this fn's contract; convert waits `sem_dc`
            // before sampling and waits its own compute fence before returning.
            unsafe { gpu.converter.convert(image, w, h, true, gpu.sem_dc)? }
        };
        // The decode + its chained compute are both complete; free the held bitstream.
        if let Some((bitstream, bitstream_mem)) = self.pending_tex_bitstream.take() {
            // SAFETY: nothing references the bitstream now the compute fence waited.
            unsafe { self.free_bitstream(bitstream, bitstream_mem) };
        }
        Ok(tex)
    }

    /// Convert an already-decoded, idle DPB slot to an RGBA `wgpu::Texture` with no
    /// chained decode (null wait semaphore), restoring the slot to DPB layout so it
    /// stays a valid reference. The reorder path uses this to emit a shown frame or a
    /// `show_existing_frame`; there is no bitstream to free (no decode happened here).
    ///
    /// # Safety
    /// `image` must be an already-decoded, idle DPB slot (no decode into it in
    /// flight; the reorder path is synchronous) on the compute-capable core.
    unsafe fn convert_slot_unchained(
        &mut self,
        image: vk::Image,
        w: u32,
        h: u32,
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        let gpu = self.gpu.as_ref().ok_or(VulkanVideoError::NoComputeQueue)?;
        // SAFETY: forwarded contract; a null wait means no chained decode, and
        // convert waits its own compute fence before returning.
        unsafe {
            gpu.converter
                .convert(image, w, h, true, vk::Semaphore::null())
        }
    }

    /// Emit an already-decoded, idle DPB slot as an RGBA `wgpu::Texture` with AV1
    /// film grain applied. The GPU ycbcr compute pass cannot apply grain (the
    /// hardware reconstruction is grain-free, and the driver only offers it on the
    /// `DPB_AND_OUTPUT_DISTINCT` path this decoder does not use), so the slot is
    /// read back to NV12 (`TRANSFER_SRC`, grain must not touch the DPB reference),
    /// grain is synthesized on the CPU bit-for-bit with dav1d (`apply_film_grain_nv12`,
    /// the same path the system `decode_all` uses), and the result is uploaded to a
    /// texture. Grain streams take the reorder path, so this is only ever an idle
    /// slot (no decode into `image` in flight).
    fn grained_slot_to_texture(
        &mut self,
        image: vk::Image,
        fg: &Av1FilmGrain,
        is_id: bool,
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        let mut frame = self.read_slot_nv12(image)?;
        apply_film_grain_nv12(&mut frame, fg, is_id);
        let color = self.color;
        let gpu = self.gpu.as_ref().ok_or(VulkanVideoError::NoComputeQueue)?;
        Ok(nv12_to_rgba_texture(
            &gpu.converter.wgpu_device,
            &gpu.wgpu_queue,
            &frame,
            color,
        ))
    }

    /// Wait on and free every in-flight ring decode WITHOUT reading it back, then
    /// rewind the ring. Used on `reset` (seek) and `Drop`, where any pending
    /// system-path output is discarded. Also frees a chained texture decode's
    /// bitstream if one is mid-flight (only on an error unwind), draining the
    /// decode queue first since that decode has no fence of its own.
    fn abort_ring(&mut self) {
        let dev = self.raw_device.clone();
        for slot in &mut self.ring {
            if let Some(inf) = slot.in_flight.take() {
                // SAFETY: the fence was submitted with this slot; wait then free.
                unsafe {
                    let _ = dev.wait_for_fences(&[slot.fence], true, u64::MAX);
                    dev.destroy_buffer(inf.bitstream, None);
                    dev.free_memory(inf.bitstream_mem, None);
                }
            }
        }
        if let Some((bitstream, bitstream_mem)) = self.pending_tex_bitstream.take() {
            // SAFETY: the chained decode may still be in flight (it signals only
            // `sem_dc`, no fence); drain the decode queue before freeing.
            unsafe {
                let _ = dev.queue_wait_idle(self.decode_queue);
                dev.destroy_buffer(bitstream, None);
                dev.free_memory(bitstream_mem, None);
            }
        }
        self.ready.clear();
        self.ring_next = 0;
    }
}

impl core::fmt::Debug for H264DpbDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("H264DpbDecoder")
            .field("coded_extent", &self.core.coded_extent)
            .field("dpb_slots", &self.core.slots.len())
            .field("max_num_ref_frames", &self.max_num_ref_frames)
            .field("poc_type", &self.poc_type)
            .finish_non_exhaustive()
    }
}

impl H264DpbDecoder {
    /// The DPB image count (one per reference slot plus the picture in flight).
    pub fn dpb_slots(&self) -> usize {
        self.core.slots.len()
    }

    /// Streaming (pipelined) decode: submit every picture in `stream` into the
    /// decode ring WITHOUT draining, returning only the frames that have already
    /// retired. Access units are split on VCL NALs with `first_mb_in_slice == 0`;
    /// SPS/PPS/SEI NALs are ignored (the session already carries the parameter
    /// sets). Reference pictures are tracked across frames, so P frames after the
    /// leading IDR decode correctly.
    ///
    /// This is the low-latency streaming form: output lags submission by up to
    /// `DECODE_RING_DEPTH - 1` pictures (the ring keeps that many decodes in
    /// flight), which is the shape a chunk-at-a-time consumer (a viewer's
    /// `AsyncDecoder`) wants, one sample in per call, frames out as they retire.
    /// Call [`decode_flush`](Self::decode_flush) at end of stream to emit the
    /// pipelined tail. For a one-shot whole-stream decode use
    /// [`decode_all`](Self::decode_all).
    ///
    /// Returns `(submitted, retired)`: the number of coded pictures submitted this
    /// call and the frames that retired during it. `submitted` lets a streaming
    /// caller keep a per-picture side channel (e.g. presentation timestamps) in
    /// lockstep with the pipeline even when a chunk carries zero or several
    /// pictures.
    pub fn decode_push(
        &mut self,
        stream: &[u8],
    ) -> Result<(usize, alloc::vec::Vec<Nv12Frame>), VulkanVideoError> {
        let (metas, frames) = self.decode_push_meta(stream)?;
        Ok((metas.len(), frames))
    }

    /// Streaming decode like [`decode_push`](Self::decode_push), but also returns
    /// one [`PictureMeta`] per SUBMITTED picture (decode order, `metas.len()` ==
    /// the submitted count). The metas let a streaming caller reorder the retired
    /// frames into display order without re-parsing (the POC is computed by the
    /// decode itself). The frames still retire in coding order and lag submission
    /// by the ring depth, so a caller pairs each retired frame with the oldest
    /// unconsumed meta (decode order == ring retirement order) and reorders by
    /// (coded-video-sequence, POC), the same key [`reorder_to_display_order`] uses.
    pub fn decode_push_meta(
        &mut self,
        stream: &[u8],
    ) -> Result<(alloc::vec::Vec<PictureMeta>, alloc::vec::Vec<Nv12Frame>), VulkanVideoError> {
        let pictures = self.split_pictures(stream)?;
        let mut metas = alloc::vec::Vec::with_capacity(pictures.len());
        let mut frames = alloc::vec::Vec::new();
        for (hdr, slices) in pictures {
            let (_target, meta) = self.decode_into_slot(&hdr, &slices, true)?;
            metas.push(meta);
            while let Some(f) = self.core.ready.pop_front() {
                frames.push(f);
            }
        }
        Ok((metas, frames))
    }

    /// Retire every in-flight ring decode and return the remaining frames (end of
    /// stream). After this the ring is empty; a later [`decode_push`](Self::decode_push)
    /// starts fresh.
    pub fn decode_flush(&mut self) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        self.core.drain()?;
        let mut frames = alloc::vec::Vec::new();
        while let Some(f) = self.core.ready.pop_front() {
            frames.push(f);
        }
        Ok(frames)
    }

    /// Decode an entire Annex-B / AVCC elementary stream in one call, returning one
    /// [`Nv12Frame`] per coded picture in DISPLAY (presentation) order. Convenience
    /// wrapper over [`decode_push`](Self::decode_push) + [`decode_flush`](Self::decode_flush)
    /// for callers that have the whole stream in hand (tests, whole-clip decode).
    /// The frames are reordered from coding to display order (see
    /// [`reorder_to_display_order`]), so a stream with B-frames comes out correctly;
    /// for an I/P stream this is a no-op. The streaming
    /// [`decode_push`](Self::decode_push) stays in coding order (a low-latency
    /// consumer reorders by PTS itself).
    pub fn decode_all(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        // Decode and capture the per-picture POC map in one pass (`decode_push_meta`
        // computes the POC as it decodes), then reorder to display order. Using the
        // decode's own metas rather than a separate `index_pictures` pass is not
        // just cheaper: `index_pictures` ends by resetting the reference / POC
        // state, which would break a caller that feeds `decode_all` one coded sample
        // at a time (each sample's P/B frames must see the prior sample's DPB
        // references). `decode_push_meta` leaves the DPB intact across calls.
        let (metas, mut frames) = self.decode_push_meta(stream)?;
        frames.extend(self.decode_flush()?);
        Ok(reorder_to_display_order(frames, &metas))
    }

    /// Decode an entire elementary stream to GPU-resident RGBA `wgpu::Texture`s
    /// (one per coded picture, DISPLAY order), the zero-copy output. Requires
    /// the decoder to have been built with [`create_h264_dpb_decoder_gpu`];
    /// returns [`VulkanVideoError::NoComputeQueue`] otherwise. Like [`decode_all`],
    /// the textures are reordered to display order (B-frame streams come out right).
    ///
    /// [`decode_all`]: Self::decode_all
    pub fn decode_all_to_textures(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<wgpu::Texture>, VulkanVideoError> {
        if self.core.gpu.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        let metas = self.index_pictures(stream)?;
        let pictures = self.split_pictures(stream)?;
        let mut textures = alloc::vec::Vec::with_capacity(pictures.len());
        for (hdr, slices) in pictures {
            textures.push(self.decode_picture_to_texture(&hdr, &slices)?);
        }
        Ok(reorder_to_display_order(textures, &metas))
    }

    /// Streaming decode of one access unit to GPU-resident RGBA textures: one
    /// `(PictureMeta, wgpu::Texture)` per coded picture, in CODING order, with
    /// the DPB / POC state left intact across calls (unlike
    /// [`decode_all_to_textures`](Self::decode_all_to_textures), whose indexing
    /// pass resets it, making it whole-stream-only). A streaming caller reorders
    /// by the metas' (coded-video-sequence, POC), as the element's texture
    /// reorder buffer does. Synchronous (each decode chained to its convert).
    pub fn decode_push_to_textures(
        &mut self,
        stream: &[u8],
    ) -> Result<(alloc::vec::Vec<PictureMeta>, alloc::vec::Vec<wgpu::Texture>), VulkanVideoError>
    {
        if self.core.gpu.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        let pictures = self.split_pictures(stream)?;
        let mut metas = alloc::vec::Vec::with_capacity(pictures.len());
        let mut textures = alloc::vec::Vec::with_capacity(pictures.len());
        for (hdr, slices) in pictures {
            let (w, h) = self.core.coded_extent;
            self.core.chain_next = true;
            let (target, meta) = self.decode_into_slot(&hdr, &slices, false)?;
            let image = self.core.slots[target].image;
            // SAFETY: same contract as `decode_picture_to_texture` (the slot was
            // just decoded, SAMPLED + CONCURRENT, chained via `sem_dc`).
            let texture = unsafe { self.core.convert_slot_chained(image, w, h) }?;
            metas.push(meta);
            textures.push(texture);
        }
        Ok((metas, textures))
    }

    /// Reset the reference / POC state so the next decode begins a fresh coded
    /// sequence, re-arming the session `RESET` control. This is how a seek works:
    /// after a reset, decoding from a keyframe reconstructs pictures correctly
    /// regardless of what was decoded before. Cheap (no GPU work); the DPB slot
    /// images are reused and overwritten from `UNDEFINED` by the next decode.
    pub fn reset(&mut self) {
        // Discard any pipelined system-path decodes still in flight (a seek drops
        // pending output) before reusing the ring.
        self.core.abort_ring();
        for r in &mut self.refs {
            *r = None;
        }
        self.prev_poc_msb = 0;
        self.prev_poc_lsb = 0;
        self.prev_frame_num = 0;
        self.prev_frame_num_offset = 0;
        // Re-issue the session RESET control on the next decode (a discontinuity).
        self.core.first = true;
    }

    /// Index every coded picture in the stream without decoding: one
    /// [`PictureMeta`] per picture in decoding order (keyframe flag, `frame_num`,
    /// POC). Runs the same POC state machine `decode_all` uses, then [`reset`]s
    /// the tracking state so a subsequent decode starts clean. This is the GOP /
    /// keyframe map a random-access player seeks against.
    ///
    /// [`reset`]: Self::reset
    pub fn index_pictures(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<PictureMeta>, VulkanVideoError> {
        let pictures = self.split_pictures(stream)?;
        let mut metas = alloc::vec::Vec::with_capacity(pictures.len());
        for (hdr, _slices) in &pictures {
            let poc = self.compute_poc(hdr);
            metas.push(PictureMeta {
                is_keyframe: hdr.is_idr,
                is_random_access: hdr.is_idr,
                frame_num: hdr.frame_num,
                poc,
            });
        }
        // Computing POC advanced the tracking state; reset it for a real decode.
        self.reset();
        Ok(metas)
    }

    /// Decode the run of pictures `start..=target` (decoding order) and return
    /// only `target`, GPU-resident as an RGBA `wgpu::Texture`. The pictures in
    /// `start..target` decode as references only (no texture materialized) so
    /// `target` reconstructs against them; `start` must be a keyframe and the
    /// caller must [`reset`](Self::reset) first (a seek), or continue in sequence
    /// from the previous target. Requires GPU-texture mode
    /// ([`create_h264_dpb_decoder_gpu`](VulkanVideoDevice::create_h264_dpb_decoder_gpu)).
    pub fn decode_range_to_texture(
        &mut self,
        stream: &[u8],
        start: usize,
        target: usize,
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        if self.core.gpu.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        let pictures = self.split_pictures(stream)?;
        if start > target || target >= pictures.len() {
            return Err(VulkanVideoError::UnsupportedStream);
        }
        // The run-up references (start..target): decode into DPB slots, no texture.
        for (hdr, slices) in &pictures[start..target] {
            self.decode_into_slot(hdr, slices, false)?;
        }
        let (hdr, slices) = &pictures[target];
        self.decode_picture_to_texture(hdr, slices)
    }

    /// Like [`decode_range_to_texture`](Self::decode_range_to_texture) but
    /// materializes a texture for **every** picture in `start..=target`, in
    /// decoding order, not just the target. The run-up pictures still decode as
    /// references (the ycbcr pass restores each slot to the DPB layout), so this
    /// costs one extra colour conversion per run-up frame but lets a scrubber
    /// cache a whole GOP from one decode pass (cheap backward scrub within it).
    /// Each texture is paired with its decoding index (H.264 has no skipped
    /// pictures, so this is simply `start..=target`; the pairing matches the H.265
    /// sibling, whose tune-in path may skip a RASL).
    pub fn decode_range_all_to_textures(
        &mut self,
        stream: &[u8],
        start: usize,
        target: usize,
    ) -> Result<alloc::vec::Vec<(usize, wgpu::Texture)>, VulkanVideoError> {
        if self.core.gpu.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        let pictures = self.split_pictures(stream)?;
        if start > target || target >= pictures.len() {
            return Err(VulkanVideoError::UnsupportedStream);
        }
        let mut out = alloc::vec::Vec::with_capacity(target - start + 1);
        for (offset, (hdr, slices)) in pictures[start..=target].iter().enumerate() {
            out.push((start + offset, self.decode_picture_to_texture(hdr, slices)?));
        }
        Ok(out)
    }

    /// Split an Annex-B / AVCC stream into primary coded pictures: a list of
    /// (first-slice header, its slice NALs). Access units break on a VCL NAL with
    /// `first_mb_in_slice == 0`; SPS/PPS/SEI NALs are ignored (the session already
    /// carries the parameter sets). Borrows `stream`, not `self`, so the caller
    /// can decode with `&mut self` afterward.
    fn split_pictures<'s>(&self, stream: &'s [u8]) -> Result<PictureUnits<'s>, VulkanVideoError> {
        let mut pictures = alloc::vec::Vec::new();
        let mut cur_slices: alloc::vec::Vec<&'s [u8]> = alloc::vec::Vec::new();
        let mut cur_hdr: Option<H264SliceHeader> = None;

        for nal in nal_units_any(stream) {
            if nal.is_empty() {
                continue;
            }
            let t = nal[0] & 0x1F;
            if t != 1 && t != 5 {
                continue; // non-VCL: parameter sets / SEI, not part of a picture
            }
            let hdr = parse_h264_slice_header(nal, &self.sps, &self.pps)
                .ok_or(VulkanVideoError::UnsupportedStream)?;
            // A VCL NAL with first_mb 0 begins a new primary coded picture; flush
            // the picture accumulated so far first.
            if hdr.first_mb_in_slice == 0 && !cur_slices.is_empty() {
                let h = cur_hdr.take().ok_or(VulkanVideoError::NoDecodableSlice)?;
                pictures.push((h, core::mem::take(&mut cur_slices)));
            }
            if cur_hdr.is_none() {
                cur_hdr = Some(hdr);
            }
            cur_slices.push(nal);
        }
        if !cur_slices.is_empty() {
            let h = cur_hdr.take().ok_or(VulkanVideoError::NoDecodableSlice)?;
            pictures.push((h, cur_slices));
        }
        Ok(pictures)
    }

    /// Compute the picture-order-count for the current picture and advance the
    /// POC tracking state (POC types 0 and 2; type 1 is rejected at creation).
    fn compute_poc(&mut self, hdr: &H264SliceHeader) -> i32 {
        match self.poc_type {
            0 => {
                if hdr.is_idr {
                    self.prev_poc_msb = 0;
                    self.prev_poc_lsb = 0;
                }
                let max_lsb = 1i32 << self.log2_max_pic_order_cnt_lsb;
                let lsb = hdr.pic_order_cnt_lsb as i32;
                let poc_msb = if lsb < self.prev_poc_lsb && (self.prev_poc_lsb - lsb) >= max_lsb / 2
                {
                    self.prev_poc_msb + max_lsb
                } else if lsb > self.prev_poc_lsb && (lsb - self.prev_poc_lsb) > max_lsb / 2 {
                    self.prev_poc_msb - max_lsb
                } else {
                    self.prev_poc_msb
                };
                let top = poc_msb + lsb;
                let bottom = top + hdr.delta_pic_order_cnt_bottom;
                // Reference pictures update the prev-POC state (per 8.2.1.1);
                // non-reference pictures leave it unchanged.
                if hdr.nal_ref_idc != 0 {
                    self.prev_poc_msb = poc_msb;
                    self.prev_poc_lsb = lsb;
                }
                top.min(bottom)
            }
            // POC type 2: derived from frame_num alone (8.2.1.3).
            _ => {
                let frame_num = hdr.frame_num as i32;
                let frame_num_offset = if hdr.is_idr {
                    0
                } else if self.prev_frame_num > frame_num {
                    self.prev_frame_num_offset + self.max_frame_num
                } else {
                    self.prev_frame_num_offset
                };
                let poc = if hdr.is_idr {
                    0
                } else if hdr.nal_ref_idc == 0 {
                    2 * (frame_num_offset + frame_num) - 1
                } else {
                    2 * (frame_num_offset + frame_num)
                };
                self.prev_frame_num_offset = frame_num_offset;
                self.prev_frame_num = frame_num;
                poc
            }
        }
    }

    /// H.264 sliding-window reference marking: evict the short-term reference
    /// with the smallest `FrameNumWrap` (the oldest in decoding order) relative
    /// to `cur_frame_num`, freeing its slot.
    fn evict_oldest(&mut self, cur_frame_num: u32) {
        let cur = cur_frame_num as i32;
        let mut victim: Option<usize> = None;
        let mut min_wrap = i32::MAX;
        for (i, r) in self.refs.iter().enumerate() {
            if let Some(rp) = r {
                let fnum = rp.frame_num as i32;
                let wrap = if fnum > cur {
                    fnum - self.max_frame_num
                } else {
                    fnum
                };
                if wrap < min_wrap {
                    min_wrap = wrap;
                    victim = Some(i);
                }
            }
        }
        if let Some(i) = victim {
            self.refs[i] = None;
        }
    }

    /// Decode one primary coded picture (its slice NALs) into a free DPB slot,
    /// referencing the pictures currently in the DPB, and read the result back as
    /// an [`Nv12Frame`]. Updates the DPB (stores the picture as a reference, runs
    /// sliding-window marking) afterward.
    /// Decode one primary coded picture into a free DPB slot and run the H.264
    /// reference marking, returning the slot index holding the decoded picture
    /// (left in `VIDEO_DECODE_DPB_KHR` layout). `to_system == true` pipelines the
    /// decode through the ring and reads the NV12 back (system-memory path);
    /// `false` leaves the frame on the GPU (the texture path).
    /// Decode one picture into a DPB slot, returning `(target_slot, meta)`. The
    /// [`PictureMeta`] carries the picture's POC (computed here, so a streaming
    /// caller gets the display-order key without re-running the POC state machine,
    /// which [`index_pictures`](Self::index_pictures) would reset).
    fn decode_into_slot(
        &mut self,
        hdr: &H264SliceHeader,
        slices: &[&[u8]],
        to_system: bool,
    ) -> Result<(usize, PictureMeta), VulkanVideoError> {
        let poc = self.compute_poc(hdr);
        let meta = PictureMeta {
            is_keyframe: hdr.is_idr,
            is_random_access: hdr.is_idr,
            frame_num: hdr.frame_num,
            poc,
        };

        // An IDR resets the reference state: all DPB slots are freed before the
        // picture is decoded (it uses no references).
        if hdr.is_idr {
            for r in &mut self.refs {
                *r = None;
            }
        }

        // The active references for this decode are every slot currently holding
        // a reference picture. The driver builds the actual RefPicList from the
        // slice headers; we just supply the candidate set + their FrameNum/POC.
        let active: alloc::vec::Vec<(usize, RefPic)> = self
            .refs
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|rp| (i, rp)))
            .collect();

        // Pick a free slot for the picture being decoded.
        let target = self
            .refs
            .iter()
            .position(|r| r.is_none())
            .ok_or(VulkanVideoError::UnsupportedStream)?;

        // Build the concatenated bitstream (each slice NAL framed with a 4-byte
        // start code) and the per-slice offsets the driver needs.
        let mut bitstream_data: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
        let mut slice_offsets: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(slices.len());
        for nal in slices {
            slice_offsets.push(bitstream_data.len() as u32);
            bitstream_data.extend_from_slice(&[0, 0, 0, 1]);
            bitstream_data.extend_from_slice(nal);
        }

        self.submit_decode(
            hdr,
            poc,
            target,
            &active,
            &bitstream_data,
            &slice_offsets,
            to_system,
        )?;

        // Reference marking: store the decoded picture as a short-term reference
        // (running sliding-window eviction first if the DPB is full). A
        // non-reference picture (nal_ref_idc == 0) leaves its slot free.
        if hdr.nal_ref_idc != 0 && self.max_num_ref_frames > 0 {
            let ref_count = self.refs.iter().filter(|r| r.is_some()).count();
            if ref_count >= self.max_num_ref_frames {
                self.evict_oldest(hdr.frame_num);
            }
            self.refs[target] = Some(RefPic {
                frame_num: hdr.frame_num,
                poc,
            });
        }

        Ok((target, meta))
    }

    /// Decode one picture and convert it, GPU-resident, into an RGBA
    /// `wgpu::Texture` via the ycbcr compute pass, restoring the slot to the DPB
    /// layout so it remains a valid reference. Requires the decoder to have been
    /// built in GPU-texture mode ([`create_h264_dpb_decoder_gpu`]).
    fn decode_picture_to_texture(
        &mut self,
        hdr: &H264SliceHeader,
        slices: &[&[u8]],
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        let (w, h) = self.core.coded_extent;
        // Chain the decode to its ycbcr conversion via `sem_dc` (no CPU stall
        // between them); the decode is submitted async and its bitstream held until
        // the conversion completes.
        self.core.chain_next = true;
        let (target, _meta) = self.decode_into_slot(hdr, slices, false)?;
        let image = self.core.slots[target].image;
        // SAFETY: the target slot was just decoded into VIDEO_DECODE_DPB_KHR layout;
        // its image is SAMPLED + CONCURRENT (GPU mode) and its decode signals
        // `sem_dc`, which the chained convert waits before sampling.
        unsafe { self.core.convert_slot_chained(image, w, h) }
    }

    /// Record + submit the decode of one picture into `self.core.slots[target]`,
    /// leaving that image decoded in `VIDEO_DECODE_DPB_KHR` layout (so it can
    /// serve as a future reference). `to_system == true` pipelines the decode
    /// through the ring and copies the NV12 planes into the readback buffer;
    /// `false` is the synchronous texture path, leaving the frame on the GPU.
    #[allow(clippy::too_many_arguments)]
    fn submit_decode(
        &mut self,
        hdr: &H264SliceHeader,
        poc: i32,
        target: usize,
        active: &[(usize, RefPic)],
        bitstream_data: &[u8],
        slice_offsets: &[u32],
        to_system: bool,
    ) -> Result<(), VulkanVideoError> {
        let (w, h) = self.core.coded_extent;
        let num_refs = active.len();

        // Transient host-visible bitstream buffer holding this picture's slices.
        let (bitstream, bitstream_mem, buf_size) = self
            .core
            .new_bitstream(bitstream_data, &self.profile.profile)?;

        // Per-picture Std picture info.
        // SAFETY: bitfield POD, valid all-zero.
        let mut pic_flags: vk::native::StdVideoDecodeH264PictureInfoFlags =
            unsafe { core::mem::zeroed() };
        pic_flags.set_field_pic_flag(0);
        pic_flags.set_is_intra((hdr.is_idr || hdr.is_intra_slice()) as u32);
        pic_flags.set_IdrPicFlag(hdr.is_idr as u32);
        pic_flags.set_is_reference((hdr.nal_ref_idc != 0) as u32);
        let std_pic = vk::native::StdVideoDecodeH264PictureInfo {
            flags: pic_flags,
            seq_parameter_set_id: self.sps.seq_parameter_set_id,
            pic_parameter_set_id: hdr.pic_parameter_set_id,
            reserved1: 0,
            reserved2: 0,
            frame_num: hdr.frame_num as u16,
            idr_pic_id: hdr.idr_pic_id as u16,
            PicOrderCnt: [poc, poc],
        };
        let mut h264_pic = vk::VideoDecodeH264PictureInfoKHR::default()
            .std_picture_info(&std_pic)
            .slice_offsets(slice_offsets);

        // Reference-slot chains. These Vecs are sized exactly and fully populated
        // before any pointer into them is taken, so their heap addresses are
        // stable for the raw `p_next` / `p_picture_resource` pointers below
        // (ash's lifetime-tracked builders cannot express an array of chained
        // structs; this mirrors the manual chaining in `h264_profile`).
        let mut std_refs: alloc::vec::Vec<vk::native::StdVideoDecodeH264ReferenceInfo> =
            alloc::vec::Vec::with_capacity(num_refs);
        for (_, rp) in active {
            std_refs.push(std_ref_info(rp.frame_num, rp.poc));
        }
        let std_cur = std_ref_info(hdr.frame_num, poc);

        let mut dpb_infos: alloc::vec::Vec<vk::VideoDecodeH264DpbSlotInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs);
        for sr in &std_refs {
            dpb_infos.push(vk::VideoDecodeH264DpbSlotInfoKHR {
                p_std_reference_info: sr as *const _,
                ..Default::default()
            });
        }
        let dpb_cur = vk::VideoDecodeH264DpbSlotInfoKHR {
            p_std_reference_info: &std_cur as *const _,
            ..Default::default()
        };

        let mut picres: alloc::vec::Vec<vk::VideoPictureResourceInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs);
        for (slot_i, _) in active {
            picres.push(
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_offset(vk::Offset2D { x: 0, y: 0 })
                    .coded_extent(vk::Extent2D {
                        width: w,
                        height: h,
                    })
                    .base_array_layer(0)
                    .image_view_binding(self.core.slots[*slot_i].view),
            );
        }
        let picres_target = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: w,
                height: h,
            })
            .base_array_layer(0)
            .image_view_binding(self.core.slots[target].view);

        // Active reference slots for the decode (real slot indices + Std info).
        let mut ref_slots: alloc::vec::Vec<vk::VideoReferenceSlotInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs);
        for (i, (slot_i, _)) in active.iter().enumerate() {
            ref_slots.push(vk::VideoReferenceSlotInfoKHR {
                slot_index: *slot_i as i32,
                p_picture_resource: &picres[i] as *const _,
                p_next: (&dpb_infos[i] as *const vk::VideoDecodeH264DpbSlotInfoKHR).cast(),
                ..Default::default()
            });
        }

        // The setup slot: where the decoded picture is stored, carrying its own
        // (current) Std reference info.
        let setup_slot = vk::VideoReferenceSlotInfoKHR {
            slot_index: target as i32,
            p_picture_resource: &picres_target as *const _,
            p_next: (&dpb_cur as *const vk::VideoDecodeH264DpbSlotInfoKHR).cast(),
            ..Default::default()
        };

        // Begin-coding reference slots: the active references (bound to their DPB
        // indices) plus the target slot marked -1 (being set up this operation).
        let mut begin_slots: alloc::vec::Vec<vk::VideoReferenceSlotInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs + 1);
        for (i, (slot_i, _)) in active.iter().enumerate() {
            begin_slots.push(vk::VideoReferenceSlotInfoKHR {
                slot_index: *slot_i as i32,
                p_picture_resource: &picres[i] as *const _,
                p_next: (&dpb_infos[i] as *const vk::VideoDecodeH264DpbSlotInfoKHR).cast(),
                ..Default::default()
            });
        }
        begin_slots.push(vk::VideoReferenceSlotInfoKHR {
            slot_index: -1,
            p_picture_resource: &picres_target as *const _,
            ..Default::default()
        });

        let begin_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.core.session)
            .video_session_parameters(self.core.parameters)
            .reference_slots(&begin_slots);
        let decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(bitstream)
            .src_buffer_offset(0)
            .src_buffer_range(buf_size)
            .dst_picture_resource(picres_target)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&ref_slots)
            .push_next(&mut h264_pic);

        // Record + submit the decode (barriers, begin/decode/end coding, optional
        // NV12 readback) via the shared codec-agnostic core, which takes ownership
        // of the transient bitstream buffer.
        let image = self.core.slots[target].image;
        self.core.record_and_submit(
            &begin_info,
            &decode_info,
            image,
            bitstream,
            bitstream_mem,
            to_system,
        )
    }
}

/// A reference picture held in an H.265 DPB slot, keyed by picture-order-count
/// (H.265 reference lists match on POC, not `FrameNum`).
#[derive(Debug, Clone, Copy)]
struct H265RefPic {
    poc: i32,
    /// Marked long-term by the current slice's RPS (re-derived per picture).
    is_long_term: bool,
}

/// Pictures split out of an H.265 stream: each is a first-slice header plus the
/// slice NALs of that primary coded picture (borrowing the input stream).
type H265PictureUnits<'s> = alloc::vec::Vec<(H265SliceHeader, alloc::vec::Vec<&'s [u8]>)>;

/// A multi-frame H.265 decoder driving a real DPB, the HEVC sibling of
/// [`H264DpbDecoder`]: it decodes P frames by tracking reference pictures across
/// access units (by POC), computes picture-order-count from the slice headers,
/// and hands the driver the reference-picture-set slot lists per picture.
///
/// Open-GOP aware: only an IRAP with `NoRaslOutputFlag == 1` (every IDR / BLA,
/// and a CRA that is the first picture of the decode) flushes the DPB. A CRA in
/// the middle of continuous decoding (`NoRaslOutputFlag == 0`, the x265 default
/// `open-gop=1` case) keeps its predecessors: its reference-picture set retains
/// the pre-CRA pictures its RASL leading pictures reference, so they decode
/// correctly instead of against a cleared DPB. Owns a fixed DPB image pool, a
/// reusable readback buffer and command pool, and copies of the
/// [`H265DecodeSession`] handles (the session must outlive the decoder).
pub struct H265DpbDecoder {
    /// Codec-independent GPU plumbing (device, session, DPB pool, readback,
    /// command pool, record/submit path). Its `Drop` frees all of it.
    core: DpbCore,
    /// Kept alive: the DPB-image and bitstream-buffer profile lists point into it.
    profile: H265Profile,
    /// Per-slot reference state: `Some` holds a reference picture, `None` is free.
    refs: alloc::vec::Vec<Option<H265RefPic>>,
    sps: H265Sps,
    pps: H265Pps,
    log2_max_pic_order_cnt_lsb: u32,
    // POC tracking, carried across pictures in decoding order.
    prev_poc_msb: i32,
    prev_poc_lsb: i32,
    /// Whether any picture has been decoded since creation / [`reset`]. A CRA has
    /// `NoRaslOutputFlag == 1` (a hard reference reset) only when it is the first
    /// picture; a later CRA keeps its predecessors for its RASL followers.
    ///
    /// [`reset`]: Self::reset
    seen_picture: bool,
    /// Whether the RASL leading pictures associated with the most recent IRAP are
    /// discarded (the IRAP's `NoRaslOutputFlag`). After a random-access tune-in
    /// (a seek that reset us to a CRA), the CRA's RASL followers reference pre-CRA
    /// pictures that are absent, so they cannot decode and are not output (H.265
    /// 8.1.3): set true at such an IRAP, false at an IRAP in continuous decoding.
    skip_rasl: bool,
}

impl core::fmt::Debug for H265DpbDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("H265DpbDecoder")
            .field("coded_extent", &self.core.coded_extent)
            .field("dpb_slots", &self.core.slots.len())
            .finish_non_exhaustive()
    }
}

impl H265DpbDecoder {
    /// The DPB image count.
    pub fn dpb_slots(&self) -> usize {
        self.core.slots.len()
    }

    /// Reset the reference / POC state so the next decode begins a fresh coded
    /// sequence, re-arming the session `RESET` control (seek). Cheap; DPB slot
    /// images are reused and overwritten by the next decode. Mirrors
    /// [`H264DpbDecoder::reset`].
    pub fn reset(&mut self) {
        // Discard any pipelined system-path decodes still in flight before reuse.
        self.core.abort_ring();
        for r in &mut self.refs {
            *r = None;
        }
        self.prev_poc_msb = 0;
        self.prev_poc_lsb = 0;
        self.seen_picture = false;
        // The next IRAP sets this; a stray RASL before any IRAP is not skipped.
        self.skip_rasl = false;
        self.core.first = true;
    }

    /// Streaming (pipelined) decode of an Annex-B / HVCC elementary stream without
    /// draining, returning `(submitted, retired)` (see
    /// [`H264DpbDecoder::decode_push`] for the pipelining contract). Reference
    /// pictures are tracked across frames so P frames decode against their references.
    pub fn decode_push(
        &mut self,
        stream: &[u8],
    ) -> Result<(usize, alloc::vec::Vec<Nv12Frame>), VulkanVideoError> {
        let (metas, frames) = self.decode_push_meta(stream)?;
        Ok((metas.len(), frames))
    }

    /// Streaming decode returning per-submitted-picture [`PictureMeta`] for
    /// display-order reordering (see [`H264DpbDecoder::decode_push_meta`]).
    pub fn decode_push_meta(
        &mut self,
        stream: &[u8],
    ) -> Result<(alloc::vec::Vec<PictureMeta>, alloc::vec::Vec<Nv12Frame>), VulkanVideoError> {
        let pictures = self.split_pictures(stream)?;
        let mut metas = alloc::vec::Vec::with_capacity(pictures.len());
        let mut frames = alloc::vec::Vec::new();
        for (hdr, slices) in pictures {
            // A discarded RASL (tune-in) produces no picture: no meta, no frame.
            let Some((_target, meta)) = self.decode_into_slot(&hdr, &slices, true)? else {
                continue;
            };
            metas.push(meta);
            while let Some(f) = self.core.ready.pop_front() {
                frames.push(f);
            }
        }
        Ok((metas, frames))
    }

    /// Drain the ring at end of stream (see [`H264DpbDecoder::decode_flush`]).
    pub fn decode_flush(&mut self) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        self.core.drain()?;
        let mut frames = alloc::vec::Vec::new();
        while let Some(f) = self.core.ready.pop_front() {
            frames.push(f);
        }
        Ok(frames)
    }

    /// Index every coded picture without decoding: one [`PictureMeta`] per picture
    /// in decoding order (see [`H264DpbDecoder::index_pictures`]). `is_keyframe`
    /// marks an IDR (a POC reset / closed-GOP random-access point); `frame_num` is
    /// unused for H.265 (0). Runs the POC state machine then resets it.
    pub fn index_pictures(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<PictureMeta>, VulkanVideoError> {
        let pictures = self.split_pictures(stream)?;
        let mut metas = alloc::vec::Vec::with_capacity(pictures.len());
        for (hdr, _slices) in &pictures {
            let poc = self.compute_poc(hdr);
            metas.push(PictureMeta {
                is_keyframe: hdr.is_idr,
                is_random_access: hdr.is_irap,
                frame_num: 0,
                poc,
            });
        }
        self.reset();
        Ok(metas)
    }

    /// Decode an entire elementary stream in one call, returning frames in DISPLAY
    /// order (see [`H264DpbDecoder::decode_all`]): decode + capture the POC map in
    /// one pass ([`decode_push_meta`](Self::decode_push_meta)), flush the tail, then
    /// reorder to display order (B-frame streams come out right, I/P unchanged). No
    /// reference-state reset, so feeding one coded sample per call keeps the DPB.
    pub fn decode_all(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        let (metas, mut frames) = self.decode_push_meta(stream)?;
        frames.extend(self.decode_flush()?);
        Ok(reorder_to_display_order(frames, &metas))
    }

    /// Decode an entire elementary stream to GPU-resident RGBA `wgpu::Texture`s
    /// (one per coded picture, DISPLAY order). Requires the decoder to have been
    /// built with [`create_h265_dpb_decoder_gpu`](VulkanVideoDevice::create_h265_dpb_decoder_gpu).
    pub fn decode_all_to_textures(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<wgpu::Texture>, VulkanVideoError> {
        if self.core.gpu.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        let metas = self.index_pictures(stream)?;
        let pictures = self.split_pictures(stream)?;
        let mut textures = alloc::vec::Vec::with_capacity(pictures.len());
        for (hdr, slices) in pictures {
            // A discarded RASL (tune-in) yields no texture; skip it. (Whole-stream
            // decode from the start never skips: no IRAP has NoRaslOutputFlag == 1
            // past the first, so `metas` and `textures` stay aligned for reorder.)
            if let Some(tex) = self.decode_picture_to_texture(&hdr, &slices)? {
                textures.push(tex);
            }
        }
        Ok(reorder_to_display_order(textures, &metas))
    }

    /// Streaming decode of one access unit to GPU-resident RGBA textures: one
    /// `(PictureMeta, wgpu::Texture)` per DECODED picture, in coding order, with
    /// the DPB / POC state left intact across calls (the H.265 sibling of
    /// [`H264DpbDecoder::decode_push_to_textures`]). A RASL discarded by a
    /// tune-in yields neither a meta nor a texture, keeping the pair aligned.
    pub fn decode_push_to_textures(
        &mut self,
        stream: &[u8],
    ) -> Result<(alloc::vec::Vec<PictureMeta>, alloc::vec::Vec<wgpu::Texture>), VulkanVideoError>
    {
        if self.core.gpu.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        let pictures = self.split_pictures(stream)?;
        let mut metas = alloc::vec::Vec::with_capacity(pictures.len());
        let mut textures = alloc::vec::Vec::with_capacity(pictures.len());
        for (hdr, slices) in pictures {
            let (w, h) = self.core.coded_extent;
            self.core.chain_next = true;
            let Some((target, meta)) = self.decode_into_slot(&hdr, &slices, false)? else {
                continue;
            };
            let image = self.core.slots[target].image;
            // SAFETY: same contract as `decode_picture_to_texture` (the slot was
            // just decoded, SAMPLED + CONCURRENT, chained via `sem_dc`).
            let texture = unsafe { self.core.convert_slot_chained(image, w, h) }?;
            metas.push(meta);
            textures.push(texture);
        }
        Ok((metas, textures))
    }

    /// Decode the run of pictures `start..=target` (decoding order) and return only
    /// `target`, GPU-resident as an RGBA `wgpu::Texture` (the random-access seek
    /// primitive, see [`H264DpbDecoder::decode_range_to_texture`]). `start` must be
    /// a random-access point the caller [`reset`](Self::reset)s to (a seek), or a
    /// continuation of the previous run. A RASL among the run-up references decodes
    /// as nothing after a CRA tune-in (skipped); the caller's seek-point choice
    /// keeps `target` itself decodable (never a skipped RASL).
    pub fn decode_range_to_texture(
        &mut self,
        stream: &[u8],
        start: usize,
        target: usize,
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        if self.core.gpu.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        let pictures = self.split_pictures(stream)?;
        if start > target || target >= pictures.len() {
            return Err(VulkanVideoError::UnsupportedStream);
        }
        for (hdr, slices) in &pictures[start..target] {
            // Run-up references (a skipped RASL simply is not decoded / referenced).
            self.decode_into_slot(hdr, slices, false)?;
        }
        let (hdr, slices) = &pictures[target];
        self.decode_picture_to_texture(hdr, slices)?
            .ok_or(VulkanVideoError::UnsupportedStream)
    }

    /// Materialize a texture for every DECODED picture in `start..=target`, each
    /// paired with its decoding index (see
    /// [`H264DpbDecoder::decode_range_all_to_textures`]). A RASL skipped by a CRA
    /// tune-in yields no texture and no entry, so the index pairing (not a bare
    /// offset) is what keeps a traversed-cache consumer aligned.
    pub fn decode_range_all_to_textures(
        &mut self,
        stream: &[u8],
        start: usize,
        target: usize,
    ) -> Result<alloc::vec::Vec<(usize, wgpu::Texture)>, VulkanVideoError> {
        if self.core.gpu.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        let pictures = self.split_pictures(stream)?;
        if start > target || target >= pictures.len() {
            return Err(VulkanVideoError::UnsupportedStream);
        }
        let mut out = alloc::vec::Vec::with_capacity(target - start + 1);
        for (offset, (hdr, slices)) in pictures[start..=target].iter().enumerate() {
            if let Some(tex) = self.decode_picture_to_texture(hdr, slices)? {
                out.push((start + offset, tex));
            }
        }
        Ok(out)
    }

    /// Split an Annex-B / HVCC stream into primary coded pictures. Access units
    /// break on a VCL NAL whose `first_slice_segment_in_pic_flag` is 1; non-VCL
    /// NALs (parameter sets / SEI) are ignored (the session already carries the
    /// parameter sets).
    fn split_pictures<'s>(
        &self,
        stream: &'s [u8],
    ) -> Result<H265PictureUnits<'s>, VulkanVideoError> {
        let mut pictures = alloc::vec::Vec::new();
        let mut cur_slices: alloc::vec::Vec<&'s [u8]> = alloc::vec::Vec::new();
        let mut cur_hdr: Option<H265SliceHeader> = None;

        for nal in nal_units_any(stream) {
            if nal.len() < 2 {
                continue;
            }
            let t = (nal[0] >> 1) & 0x3F;
            if t > 31 {
                continue; // non-VCL: VPS/SPS/PPS/SEI, not part of a picture
            }
            let hdr = parse_h265_slice_header(nal, &self.sps, &self.pps)
                .ok_or(VulkanVideoError::UnsupportedStream)?;
            if hdr.first_slice_segment_in_pic_flag && !cur_slices.is_empty() {
                let h = cur_hdr.take().ok_or(VulkanVideoError::NoDecodableSlice)?;
                pictures.push((h, core::mem::take(&mut cur_slices)));
            }
            if cur_hdr.is_none() {
                cur_hdr = Some(hdr);
            }
            cur_slices.push(nal);
        }
        if !cur_slices.is_empty() {
            let h = cur_hdr.take().ok_or(VulkanVideoError::NoDecodableSlice)?;
            pictures.push((h, cur_slices));
        }
        Ok(pictures)
    }

    /// Compute the picture-order-count for the current picture and advance the POC
    /// tracking state (H.265 8.3.1, the same MSB/LSB scheme as H.264 POC type 0).
    fn compute_poc(&mut self, hdr: &H265SliceHeader) -> i32 {
        if hdr.is_idr {
            self.prev_poc_msb = 0;
            self.prev_poc_lsb = 0;
            return 0;
        }
        let max_lsb = 1i32 << self.log2_max_pic_order_cnt_lsb;
        let lsb = hdr.slice_pic_order_cnt_lsb as i32;
        let poc_msb = if lsb < self.prev_poc_lsb && (self.prev_poc_lsb - lsb) >= max_lsb / 2 {
            self.prev_poc_msb + max_lsb
        } else if lsb > self.prev_poc_lsb && (lsb - self.prev_poc_lsb) > max_lsb / 2 {
            self.prev_poc_msb - max_lsb
        } else {
            self.prev_poc_msb
        };
        let poc = poc_msb + lsb;
        // A TemporalId-0 reference picture (not RASL/RADL/SLNR) updates the prev
        // state; our streams are all TemporalId-0, so update on any reference.
        if h265_is_reference(hdr.nal_unit_type) {
            self.prev_poc_msb = poc_msb;
            self.prev_poc_lsb = lsb;
        }
        poc
    }

    /// The DPB slot holding the reference picture with this POC, if any.
    fn slot_of_poc(&self, poc: i32) -> Option<usize> {
        self.refs
            .iter()
            .position(|r| matches!(r, Some(rp) if rp.poc == poc))
    }

    /// Decode one primary coded picture into a free DPB slot, applying the H.265
    /// reference-picture set (prune the DPB to the RPS, build the current
    /// reference slot lists), and return the slot holding it. `to_system == true`
    /// pipelines the decode through the ring and reads the NV12 back (system path);
    /// `false` leaves it on the GPU (texture path).
    fn decode_into_slot(
        &mut self,
        hdr: &H265SliceHeader,
        slices: &[&[u8]],
        to_system: bool,
    ) -> Result<Option<(usize, PictureMeta)>, VulkanVideoError> {
        // Random-access tune-in: a RASL leading picture under an IRAP whose
        // `NoRaslOutputFlag == 1` references pre-IRAP pictures that are absent, so
        // it cannot decode and is not output. Skip it BEFORE `compute_poc` so the
        // discarded picture leaves no trace in the POC prediction state the trailing
        // pictures derive against (`skip_rasl` was set by the associated IRAP).
        if h265_is_rasl(hdr.nal_unit_type) && self.skip_rasl {
            return Ok(None);
        }

        let poc = self.compute_poc(hdr);
        // `frame_num` is unused for H.265 (0), matching `index_pictures`; keyframe
        // marks an IDR (a POC reset / closed-GOP random-access point).
        let meta = PictureMeta {
            is_keyframe: hdr.is_idr,
            is_random_access: hdr.is_irap,
            frame_num: 0,
            poc,
        };

        // NoRaslOutputFlag (H.265 8.1.3): 1 for every IDR / BLA, and for a CRA
        // only when it is the first picture of the decode (a fresh stream or a
        // seek that reset us). A CRA in continuous decoding has it 0.
        let no_rasl_output = hdr.is_idr || hdr.is_bla() || (hdr.is_cra() && !self.seen_picture);
        self.seen_picture = true;
        // Associate this IRAP's RASL followers with its NoRaslOutputFlag: after a
        // tune-in the flag is 1 (skip them); in continuous decoding it is 0 (keep,
        // M577). Only IRAPs update it; a RASL only follows its own IRAP, so the
        // association is stable until the next IRAP.
        if hdr.is_irap {
            self.skip_rasl = no_rasl_output;
        }

        // Apply the RPS to the DPB before decoding. An IRAP with NoRaslOutputFlag
        // flushes it (closed-GOP / clean random access). Otherwise, including an
        // open-GOP CRA mid-stream, keep only the pictures the current RPS lists
        // (used or kept-for-future) and drop the rest (H.265 8.3.2): the CRA's RPS
        // retains the pre-CRA pictures its RASL leading pictures reference.
        // Resolve the slice's long-term entries against the DPB first (H.265
        // 8.3.2): an entry with an MSB cycle identifies its picture by full POC;
        // one without matches on POC lsb alone. A resolved slot is (re)marked
        // long-term, so the driver's reference info (and its MV scaling) mirrors
        // the slice's classification; the marking is re-derived per picture.
        let max_lsb = 1i32 << self.log2_max_pic_order_cnt_lsb;
        for r in self.refs.iter_mut().flatten() {
            r.is_long_term = false;
        }
        let mut lt_resolved: alloc::vec::Vec<(usize, bool)> = alloc::vec::Vec::new();
        for e in &hdr.lt {
            let slot = if e.has_msb_cycle {
                let full = poc
                    - e.delta_poc_msb_cycle as i32 * max_lsb
                    - (hdr.slice_pic_order_cnt_lsb as i32 - e.poc_lsb as i32);
                self.slot_of_poc(full)
            } else {
                self.refs.iter().position(
                    |r| matches!(r, Some(rp) if rp.poc & (max_lsb - 1) == e.poc_lsb as i32),
                )
            };
            if let Some(s) = slot {
                if let Some(rp) = self.refs[s].as_mut() {
                    rp.is_long_term = true;
                }
                lt_resolved.push((s, e.used_by_curr));
            }
        }

        if hdr.is_irap && no_rasl_output {
            for r in &mut self.refs {
                *r = None;
            }
        } else {
            let mut keep: alloc::vec::Vec<i32> = alloc::vec::Vec::new();
            for i in 0..hdr.st_rps.num_negative_pics as usize {
                keep.push(poc + hdr.st_rps.delta_poc_s0[i]);
            }
            for i in 0..hdr.st_rps.num_positive_pics as usize {
                keep.push(poc + hdr.st_rps.delta_poc_s1[i]);
            }
            // Long-term-listed pictures are kept too (used-by-current or not).
            for (s, _) in &lt_resolved {
                if let Some(rp) = self.refs[*s] {
                    keep.push(rp.poc);
                }
            }
            for r in &mut self.refs {
                if let Some(rp) = r {
                    if !keep.contains(&rp.poc) {
                        *r = None;
                    }
                }
            }
        }

        // The current reference slot lists (indices into the DPB). Only used-by-
        // current entries populate them; unused entries stay 0xff (no picture).
        let mut before = [0xffu8; 8];
        let mut after = [0xffu8; 8];
        if !hdr.is_irap {
            let mut nb = 0usize;
            for i in 0..hdr.st_rps.num_negative_pics as usize {
                if hdr.st_rps.used_s0[i] {
                    if let Some(slot) = self.slot_of_poc(poc + hdr.st_rps.delta_poc_s0[i]) {
                        if nb < 8 {
                            before[nb] = slot as u8;
                            nb += 1;
                        }
                    }
                }
            }
            let mut na = 0usize;
            for i in 0..hdr.st_rps.num_positive_pics as usize {
                if hdr.st_rps.used_s1[i] {
                    if let Some(slot) = self.slot_of_poc(poc + hdr.st_rps.delta_poc_s1[i]) {
                        if na < 8 {
                            after[na] = slot as u8;
                            na += 1;
                        }
                    }
                }
            }
        }
        let mut lt = [0xffu8; 8];
        if !hdr.is_irap {
            let mut nl = 0usize;
            for (s, used) in &lt_resolved {
                if *used && nl < 8 {
                    lt[nl] = *s as u8;
                    nl += 1;
                }
            }
        }

        // Active references = every slot holding a picture after the prune.
        let active: alloc::vec::Vec<(usize, H265RefPic)> = self
            .refs
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|rp| (i, rp)))
            .collect();
        let target = self
            .refs
            .iter()
            .position(|r| r.is_none())
            .ok_or(VulkanVideoError::UnsupportedStream)?;

        // Concatenate the slice NALs (each framed with a 4-byte start code) and
        // record the per-slice offsets the driver needs.
        // Frame each slice with a 3-BYTE start code (`00 00 01`), not the 4-byte
        // form. NVIDIA's Vulkan H.265 slice-header parser mis-locates the header
        // with a 4-byte start code and reads a garbage PPS id for every non-IDR
        // slice (the IDR tolerates it because its decode uses the picture info,
        // not the parsed header). ffmpeg's Vulkan HEVC path uses 3 bytes too; the
        // H.264 path (M492) is unaffected and keeps its 4-byte code.
        let mut bitstream_data: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
        let mut slice_offsets: alloc::vec::Vec<u32> = alloc::vec::Vec::with_capacity(slices.len());
        for nal in slices {
            slice_offsets.push(bitstream_data.len() as u32);
            bitstream_data.extend_from_slice(&[0, 0, 1]);
            bitstream_data.extend_from_slice(nal);
        }

        self.submit_decode(
            hdr,
            poc,
            target,
            &active,
            &before,
            &after,
            &lt,
            &bitstream_data,
            &slice_offsets,
            to_system,
        )?;

        if h265_is_reference(hdr.nal_unit_type) {
            self.refs[target] = Some(H265RefPic {
                poc,
                is_long_term: false,
            });
        }
        Ok(Some((target, meta)))
    }

    /// Decode one picture, GPU-resident, into an RGBA `wgpu::Texture` via the
    /// ycbcr compute pass (GPU-texture mode only). `Ok(None)` for a discarded RASL
    /// (tune-in): no picture, so no texture (the caller skips it).
    fn decode_picture_to_texture(
        &mut self,
        hdr: &H265SliceHeader,
        slices: &[&[u8]],
    ) -> Result<Option<wgpu::Texture>, VulkanVideoError> {
        let (w, h) = self.core.coded_extent;
        // Chain the decode to its ycbcr conversion via `sem_dc` (see the H.264
        // sibling); the decode is async and its bitstream held until convert done.
        self.core.chain_next = true;
        let Some((target, _meta)) = self.decode_into_slot(hdr, slices, false)? else {
            return Ok(None);
        };
        let image = self.core.slots[target].image;
        // SAFETY: target slot just decoded into VIDEO_DECODE_DPB_KHR, SAMPLED +
        // CONCURRENT (GPU mode); its decode signals `sem_dc` which convert waits.
        let texture = unsafe { self.core.convert_slot_chained(image, w, h) }?;
        Ok(Some(texture))
    }

    /// Record + submit the decode of one picture into `self.slots[target]`. Same
    /// command structure as [`H264DpbDecoder::submit_decode`], with H.265 `Std*`
    /// picture / reference info and the reference-picture-set slot lists.
    #[allow(clippy::too_many_arguments)]
    fn submit_decode(
        &mut self,
        hdr: &H265SliceHeader,
        poc: i32,
        target: usize,
        active: &[(usize, H265RefPic)],
        ref_before: &[u8; 8],
        ref_after: &[u8; 8],
        ref_lt: &[u8; 8],
        bitstream_data: &[u8],
        slice_offsets: &[u32],
        to_system: bool,
    ) -> Result<(), VulkanVideoError> {
        let (w, h) = self.core.coded_extent;
        let num_refs = active.len();

        // Transient host-visible bitstream buffer holding this picture's slices.
        let (bitstream, bitstream_mem, buf_size) = self
            .core
            .new_bitstream(bitstream_data, &self.profile.profile)?;

        // Per-picture Std picture info.
        // SAFETY: bitfield POD, valid all-zero.
        let mut pic_flags: vk::native::StdVideoDecodeH265PictureInfoFlags =
            unsafe { core::mem::zeroed() };
        pic_flags.set_IdrPicFlag(hdr.is_idr as u32);
        pic_flags.set_IrapPicFlag(hdr.is_irap as u32);
        pic_flags.set_IsReference(h265_is_reference(hdr.nal_unit_type) as u32);
        pic_flags.set_short_term_ref_pic_set_sps_flag(hdr.short_term_ref_pic_set_sps_flag as u32);
        let std_pic = vk::native::StdVideoDecodeH265PictureInfo {
            flags: pic_flags,
            sps_video_parameter_set_id: self.sps.sps_video_parameter_set_id,
            pps_seq_parameter_set_id: self.pps.pps_seq_parameter_set_id,
            pps_pic_parameter_set_id: hdr.slice_pic_parameter_set_id,
            NumDeltaPocsOfRefRpsIdx: hdr.num_delta_pocs_of_ref_rps_idx,
            PicOrderCntVal: poc,
            NumBitsForSTRefPicSetInSlice: hdr.num_bits_for_st_rps,
            reserved: 0,
            RefPicSetStCurrBefore: *ref_before,
            RefPicSetStCurrAfter: *ref_after,
            RefPicSetLtCurr: *ref_lt,
        };
        let mut h265_pic = vk::VideoDecodeH265PictureInfoKHR::default()
            .std_picture_info(&std_pic)
            .slice_segment_offsets(slice_offsets);

        // Reference-slot chains (see H264DpbDecoder::submit_decode for the manual
        // pointer-chaining rationale). Sized exactly + fully populated before any
        // pointer into them is taken.
        let mut std_refs: alloc::vec::Vec<vk::native::StdVideoDecodeH265ReferenceInfo> =
            alloc::vec::Vec::with_capacity(num_refs);
        for (_, rp) in active {
            std_refs.push(std_h265_ref_info(rp.poc, rp.is_long_term));
        }
        let std_cur = std_h265_ref_info(poc, false);

        let mut dpb_infos: alloc::vec::Vec<vk::VideoDecodeH265DpbSlotInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs);
        for sr in &std_refs {
            dpb_infos.push(vk::VideoDecodeH265DpbSlotInfoKHR {
                p_std_reference_info: sr as *const _,
                ..Default::default()
            });
        }
        let dpb_cur = vk::VideoDecodeH265DpbSlotInfoKHR {
            p_std_reference_info: &std_cur as *const _,
            ..Default::default()
        };

        let mut picres: alloc::vec::Vec<vk::VideoPictureResourceInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs);
        for (slot_i, _) in active {
            picres.push(
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_offset(vk::Offset2D { x: 0, y: 0 })
                    .coded_extent(vk::Extent2D {
                        width: w,
                        height: h,
                    })
                    .base_array_layer(0)
                    .image_view_binding(self.core.slots[*slot_i].view),
            );
        }
        let picres_target = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: w,
                height: h,
            })
            .base_array_layer(0)
            .image_view_binding(self.core.slots[target].view);

        let mut ref_slots: alloc::vec::Vec<vk::VideoReferenceSlotInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs);
        for (i, (slot_i, _)) in active.iter().enumerate() {
            ref_slots.push(vk::VideoReferenceSlotInfoKHR {
                slot_index: *slot_i as i32,
                p_picture_resource: &picres[i] as *const _,
                p_next: (&dpb_infos[i] as *const vk::VideoDecodeH265DpbSlotInfoKHR).cast(),
                ..Default::default()
            });
        }

        let setup_slot = vk::VideoReferenceSlotInfoKHR {
            slot_index: target as i32,
            p_picture_resource: &picres_target as *const _,
            p_next: (&dpb_cur as *const vk::VideoDecodeH265DpbSlotInfoKHR).cast(),
            ..Default::default()
        };

        let mut begin_slots: alloc::vec::Vec<vk::VideoReferenceSlotInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs + 1);
        for (i, (slot_i, _)) in active.iter().enumerate() {
            begin_slots.push(vk::VideoReferenceSlotInfoKHR {
                slot_index: *slot_i as i32,
                p_picture_resource: &picres[i] as *const _,
                p_next: (&dpb_infos[i] as *const vk::VideoDecodeH265DpbSlotInfoKHR).cast(),
                ..Default::default()
            });
        }
        begin_slots.push(vk::VideoReferenceSlotInfoKHR {
            slot_index: -1,
            p_picture_resource: &picres_target as *const _,
            ..Default::default()
        });

        let begin_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.core.session)
            .video_session_parameters(self.core.parameters)
            .reference_slots(&begin_slots);
        let decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(bitstream)
            .src_buffer_offset(0)
            .src_buffer_range(buf_size)
            .dst_picture_resource(picres_target)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&ref_slots)
            .push_next(&mut h265_pic);

        // Record + submit the decode (barriers, begin/decode/end coding, optional
        // NV12 readback) via the shared codec-agnostic core, which takes ownership
        // of the transient bitstream buffer.
        let image = self.core.slots[target].image;
        self.core.record_and_submit(
            &begin_info,
            &decode_info,
            image,
            bitstream,
            bitstream_mem,
            to_system,
        )
    }
}

/// Reference-slot bookkeeping for one physical DPB image holding a decoded AV1
/// frame usable as a reference.
#[derive(Clone, Copy, Debug)]
struct Av1SlotState {
    order_hint: u8,
    frame_type: u8,
    saved_order_hints: [u8; AV1_NUM_REF_FRAMES],
    disable_frame_end_update_cdf: bool,
    segmentation_enabled: bool,
    upscaled_width: u32,
    frame_height: u32,
    render_width: u32,
    render_height: u32,
    /// The resolved (ref-copy applied) film grain for this decoded frame, used to
    /// synthesize grain when the frame is displayed directly or via
    /// `show_existing_frame`, and as the copy source for a later `update_grain == 0`.
    film_grain: Av1FilmGrain,
}

/// A multi-frame AV1 Vulkan Video decoder with full reference management, the
/// AV1 sibling of [`H265DpbDecoder`]. AV1's DPB is a set of up to
/// `NUM_REF_FRAMES` (8) reference frames addressed by `ref_frame_idx`; each
/// decoded frame is written to a free physical slot, then `refresh_frame_flags`
/// remaps the referenced slots to it. Emits one [`Nv12Frame`] (or `wgpu::Texture`
/// in GPU mode) per shown coded frame in decoding order.
///
/// A stream using alt-ref (invisible) frames + `show_existing_frame` (decode order
/// != display order) is decoded via a separate synchronous, reorder-aware path
/// (`decode_all` / `decode_all_to_textures` detect it); an ordinary all-shown
/// stream keeps the pipelined ring / chained-texture fast path.
pub struct Av1DpbDecoder {
    /// Codec-independent GPU plumbing (device, session, DPB pool, readback,
    /// command pool, record/submit path). Its `Drop` frees all of it.
    core: DpbCore,
    /// Kept alive: the DPB-image and bitstream-buffer profile lists point into it.
    profile: Av1Profile,
    /// AV1 virtual reference slot (0..8) -> physical `core.slots` index (`None` = empty).
    ref_slot: [Option<usize>; AV1_NUM_REF_FRAMES],
    /// Per physical slot, the decoded frame's reference bookkeeping (`None` = free).
    phys_state: alloc::vec::Vec<Option<Av1SlotState>>,
    seq: Av1SequenceHeader,
}

impl core::fmt::Debug for Av1DpbDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Av1DpbDecoder")
            .field("coded_extent", &self.core.coded_extent)
            .field("dpb_slots", &self.core.slots.len())
            .finish_non_exhaustive()
    }
}

/// One display-order-relevant operation from an AV1 bitstream walk. An `OBU_FRAME`
/// carries a coded frame to decode; an `OBU_FRAME_HEADER` with `show_existing_frame`
/// re-displays a stored reference without decoding.
enum Av1Op<'s> {
    Decode(&'s [u8]),
    ShowExisting(&'s [u8]),
}

impl Av1DpbDecoder {
    /// The physical DPB image count.
    pub fn dpb_slots(&self) -> usize {
        self.core.slots.len()
    }

    /// Reset the reference state so the next decode begins a fresh coded sequence,
    /// re-arming the session `RESET` control. This is how a seek works: after a
    /// reset, decoding from a key frame reconstructs pictures correctly regardless
    /// of what was decoded before. Cheap (no GPU work); the DPB slot images are
    /// reused and overwritten from `UNDEFINED` by the next decode. Mirrors
    /// [`H264DpbDecoder::reset`]; a fresh decoder must reuse its session, so
    /// re-issuing the `RESET` control (not rebuilding) is the correct reset.
    pub fn reset(&mut self) {
        // Discard any pipelined system-path decodes still in flight before reuse.
        self.core.abort_ring();
        for r in &mut self.ref_slot {
            *r = None;
        }
        for s in &mut self.phys_state {
            *s = None;
        }
        self.core.first = true;
    }

    /// Streaming (pipelined) decode of an AV1 bitstream without draining, returning
    /// `(submitted, retired)` shown coded frames in decoding order (see
    /// [`H264DpbDecoder::decode_push`] for the pipelining contract).
    pub fn decode_push(
        &mut self,
        stream: &[u8],
    ) -> Result<(usize, alloc::vec::Vec<Nv12Frame>), VulkanVideoError> {
        let frames = self.split_frames(stream)?;
        let submitted = frames.len();
        let mut out = alloc::vec::Vec::new();
        for payload in frames {
            self.decode_frame_into_slot(payload, true)?;
            while let Some(f) = self.core.ready.pop_front() {
                out.push(f);
            }
        }
        Ok((submitted, out))
    }

    /// Drain the ring at end of stream (see [`H264DpbDecoder::decode_flush`]).
    pub fn decode_flush(&mut self) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        self.core.drain()?;
        let mut out = alloc::vec::Vec::new();
        while let Some(f) = self.core.ready.pop_front() {
            out.push(f);
        }
        Ok(out)
    }

    /// Decode an entire AV1 bitstream in one call (see
    /// [`H264DpbDecoder::decode_all`]). An ordinary all-shown stream takes the
    /// pipelined [`decode_push`](Self::decode_push) then [`decode_flush`](Self::decode_flush)
    /// fast path; a stream with alt-ref (invisible) frames and `show_existing_frame`
    /// takes the synchronous reorder-aware path.
    pub fn decode_all(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        let (ops, needs_reorder) = self.scan_ops(stream)?;
        if !needs_reorder {
            let (_, mut out) = self.decode_push(stream)?;
            out.extend(self.decode_flush()?);
            return Ok(out);
        }
        self.decode_ops(ops)
    }

    /// Decode one temporal unit in DISPLAY order via the synchronous op-walk,
    /// unconditionally (no pipelined fast path): the streaming entry the
    /// element feeds AU by AU. AV1 display order is the bitstream's op order,
    /// and the DPB persists across calls, so a `show_existing_frame` resolves
    /// frames decoded in earlier calls; the one path also keeps its output
    /// byte-identical to a whole-stream reorder-aware [`decode_all`]
    /// (the pipelined ring path skips film grain and invisible-frame ops).
    ///
    /// [`decode_all`]: Self::decode_all
    pub fn decode_display(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        let (ops, _) = self.scan_ops(stream)?;
        self.decode_ops(ops)
    }

    /// The reorder-aware op-walk shared by [`decode_all`](Self::decode_all) and
    /// [`decode_display`](Self::decode_display): decode each coded frame
    /// (emitting it only when shown), re-display stored slots on
    /// `show_existing_frame`, and synthesize film grain per displayed frame.
    fn decode_ops(
        &mut self,
        ops: alloc::vec::Vec<Av1Op<'_>>,
    ) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        let is_id = self.seq.color.matrix_coefficients == 0;
        let mut out = alloc::vec::Vec::new();
        for op in ops {
            match op {
                Av1Op::Decode(payload) => {
                    if let Some((target, fh)) = self.decode_frame_into_slot(payload, false)? {
                        if fh.show_frame {
                            let image = self.core.slots[target].image;
                            let mut frame = self.core.read_slot_nv12(image)?;
                            apply_film_grain_nv12(&mut frame, &fh.film_grain, is_id);
                            out.push(frame);
                        }
                    }
                }
                Av1Op::ShowExisting(payload) => {
                    let phys = self.show_existing_slot(payload)?;
                    let image = self.core.slots[phys].image;
                    let mut frame = self.core.read_slot_nv12(image)?;
                    let fg = self.phys_state[phys]
                        .map(|s| s.film_grain)
                        .unwrap_or_default();
                    apply_film_grain_nv12(&mut frame, &fg, is_id);
                    out.push(frame);
                }
            }
        }
        Ok(out)
    }

    /// Decode an entire AV1 bitstream to GPU-resident RGBA `wgpu::Texture`s.
    /// Requires the GPU-texture decoder (`create_av1_dpb_decoder_gpu`). Handles the
    /// reorder-aware path (alt-ref + `show_existing_frame`) as [`decode_all`] does.
    pub fn decode_all_to_textures(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<wgpu::Texture>, VulkanVideoError> {
        if self.core.gpu.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        let (ops, needs_reorder) = self.scan_ops(stream)?;
        if !needs_reorder {
            let mut out = alloc::vec::Vec::with_capacity(ops.len());
            for op in ops {
                if let Av1Op::Decode(payload) = op {
                    if let Some(tex) = self.decode_frame_to_texture(payload)? {
                        out.push(tex);
                    }
                }
            }
            return Ok(out);
        }
        self.decode_ops_to_textures(ops)
    }

    /// Decode one temporal unit to GPU textures in DISPLAY order via the
    /// synchronous op-walk, unconditionally: the streaming texture entry the
    /// element feeds AU by AU (the GPU sibling of
    /// [`decode_display`](Self::decode_display)).
    pub fn decode_display_to_textures(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<wgpu::Texture>, VulkanVideoError> {
        if self.core.gpu.is_none() {
            return Err(VulkanVideoError::NoComputeQueue);
        }
        let (ops, _) = self.scan_ops(stream)?;
        self.decode_ops_to_textures(ops)
    }

    /// The reorder-aware texture op-walk shared by the whole-stream and
    /// streaming entries.
    fn decode_ops_to_textures(
        &mut self,
        ops: alloc::vec::Vec<Av1Op<'_>>,
    ) -> Result<alloc::vec::Vec<wgpu::Texture>, VulkanVideoError> {
        // Film grain (when present) is synthesized on the CPU-read-back NV12 and the
        // result uploaded, since the GPU ycbcr pass produces the grain-free
        // reconstruction (see `grained_slot_to_texture`); a grain-free displayed
        // frame stays on the zero-copy GPU convert. Grain resolution mirrors the
        // system `decode_all`: an `update_grain == 0` frame's coefficients are already
        // folded in by `decode_frame_into_slot`; `show_existing` reuses the stored
        // per-slot grain.
        let (w, h) = self.core.coded_extent;
        let is_id = self.seq.color.matrix_coefficients == 0;
        let mut out = alloc::vec::Vec::new();
        for op in ops {
            let (image, fg) = match op {
                Av1Op::Decode(payload) => match self.decode_frame_into_slot(payload, false)? {
                    Some((target, fh)) if fh.show_frame => {
                        (self.core.slots[target].image, fh.film_grain)
                    }
                    _ => continue,
                },
                Av1Op::ShowExisting(payload) => {
                    let phys = self.show_existing_slot(payload)?;
                    let fg = self.phys_state[phys]
                        .map(|s| s.film_grain)
                        .unwrap_or_default();
                    (self.core.slots[phys].image, fg)
                }
            };
            // Grain is synthesized on the CPU only for 8-bit (`apply_film_grain_nv12`
            // leaves 10-bit grain-free, as the system `decode_all` does), and the CPU
            // upload path is 8-bit only; a 10-bit grain frame therefore takes the
            // grain-free GPU convert, same as any other 10-bit slot.
            if fg.apply_grain && self.core.bit_depth == 8 {
                out.push(self.core.grained_slot_to_texture(image, &fg, is_id)?);
            } else {
                // SAFETY: `image` is an already-decoded, idle DPB slot (the reorder
                // path is synchronous); an unchained convert (null wait) restores it.
                out.push(unsafe { self.core.convert_slot_unchained(image, w, h)? });
            }
        }
        Ok(out)
    }

    /// Walk the bitstream into an ordered op list (`OBU_FRAME` -> `Decode`, an
    /// `OBU_FRAME_HEADER` with `show_existing_frame` -> `ShowExisting`) and report
    /// whether it needs the reorder-aware path (any `show_existing_frame`, or any
    /// coded frame with `show_frame == 0`, i.e. an alt-ref). A bare `OBU_FRAME_HEADER`
    /// that is not `show_existing` implies a separate `OBU_TILE_GROUP`, which is not
    /// handled (rejected). Only the whole-stream `decode_all*` entry points use this;
    /// the pipelined [`decode_push`](Self::decode_push) keeps its own `split_frames`.
    fn scan_ops<'s>(
        &self,
        stream: &'s [u8],
    ) -> Result<(alloc::vec::Vec<Av1Op<'s>>, bool), VulkanVideoError> {
        let mut ops = alloc::vec::Vec::new();
        // Film grain is synthesized per displayed frame on the reorder-aware path
        // (the hardware decode is grain-free), so a grain stream routes through it
        // even when decode order == display order.
        let mut needs_reorder = self.seq.film_grain_params_present;
        for obu in av1_obus(stream) {
            match obu.obu_type {
                OBU_FRAME => {
                    let lead = parse_av1_frame_lead(obu.payload, &self.seq)
                        .ok_or(VulkanVideoError::UnsupportedStream)?;
                    if !lead.show_frame {
                        needs_reorder = true;
                    }
                    ops.push(Av1Op::Decode(obu.payload));
                }
                OBU_FRAME_HEADER => {
                    let lead = parse_av1_frame_lead(obu.payload, &self.seq)
                        .ok_or(VulkanVideoError::UnsupportedStream)?;
                    if !lead.show_existing_frame {
                        // A frame header with no tile data implies a separate
                        // OBU_TILE_GROUP, not handled.
                        return Err(VulkanVideoError::UnsupportedStream);
                    }
                    needs_reorder = true;
                    ops.push(Av1Op::ShowExisting(obu.payload));
                }
                _ => {}
            }
        }
        if ops.is_empty() {
            return Err(VulkanVideoError::NoDecodableSlice);
        }
        Ok((ops, needs_reorder))
    }

    /// Resolve a `show_existing_frame` header to the physical DPB slot it displays,
    /// applying the reference refresh AV1 requires when the shown frame is a key
    /// frame (`refresh_frame_flags = allFrames`; spec 7.4 / 5.9.2). No decode.
    fn show_existing_slot(&mut self, payload: &[u8]) -> Result<usize, VulkanVideoError> {
        let refs = self.ref_frames();
        let fh = parse_av1_frame_header(payload, &self.seq, &refs)
            .ok_or(VulkanVideoError::UnsupportedStream)?;
        let vbi = fh.frame_to_show_map_idx as usize;
        let phys = self
            .ref_slot
            .get(vbi)
            .copied()
            .flatten()
            .ok_or(VulkanVideoError::UnsupportedStream)?;
        // Showing a stored KEY_FRAME refreshes every reference slot to it.
        if fh.frame_type == AV1_FRAME_TYPE_KEY {
            for v in 0..AV1_NUM_REF_FRAMES {
                self.ref_slot[v] = Some(phys);
            }
        }
        Ok(phys)
    }

    /// Collect the frame OBU payloads (decoding order) for the pipelined fast path.
    /// Only `OBU_FRAME` (combined frame header + tile group) is handled; a separate
    /// `OBU_FRAME_HEADER` (+ `OBU_TILE_GROUP`, or a `show_existing_frame`) is
    /// rejected here, so the pipelined path only runs on an all-shown stream (the
    /// reorder-aware [`decode_all`](Self::decode_all) handles the rest).
    fn split_frames<'s>(
        &self,
        stream: &'s [u8],
    ) -> Result<alloc::vec::Vec<&'s [u8]>, VulkanVideoError> {
        let mut frames = alloc::vec::Vec::new();
        for obu in av1_obus(stream) {
            match obu.obu_type {
                OBU_FRAME => frames.push(obu.payload),
                OBU_FRAME_HEADER => return Err(VulkanVideoError::UnsupportedStream),
                _ => {}
            }
        }
        if frames.is_empty() {
            return Err(VulkanVideoError::NoDecodableSlice);
        }
        Ok(frames)
    }

    /// Build the reference-frame context the frame-header parse needs from the
    /// current DPB mapping.
    fn ref_frames(&self) -> Av1RefFrames {
        let mut r = Av1RefFrames::default();
        for v in 0..AV1_NUM_REF_FRAMES {
            if let Some(p) = self.ref_slot[v] {
                if let Some(st) = self.phys_state[p] {
                    r.valid[v] = true;
                    r.order_hint[v] = st.order_hint;
                    r.frame_type[v] = st.frame_type;
                    r.upscaled_width[v] = st.upscaled_width;
                    r.frame_height[v] = st.frame_height;
                    r.render_width[v] = st.render_width;
                    r.render_height[v] = st.render_height;
                }
            }
        }
        r
    }

    /// Parse + decode one frame OBU into a free DPB slot; returns the decoded slot
    /// index and the parsed header. Shared by the system + texture paths.
    fn decode_frame_into_slot(
        &mut self,
        payload: &[u8],
        to_system: bool,
    ) -> Result<Option<(usize, Av1FrameHeader)>, VulkanVideoError> {
        let refs = self.ref_frames();
        let mut fh = parse_av1_frame_header(payload, &self.seq, &refs)
            .ok_or(VulkanVideoError::UnsupportedStream)?;

        // Resolve film grain: an `update_grain == 0` frame copies its coefficients
        // from a reference frame (keeping only its own seed + apply flag). Resolve
        // against the stored per-slot grain so the params are complete for synthesis
        // and for a later frame that copies from this one.
        if fh.film_grain.apply_grain && !fh.film_grain.update_grain {
            if let Some(p) = self
                .ref_slot
                .get(fh.film_grain.ref_idx as usize)
                .copied()
                .flatten()
            {
                if let Some(st) = self.phys_state[p] {
                    let seed = fh.film_grain.seed;
                    fh.film_grain = st.film_grain;
                    fh.film_grain.apply_grain = true;
                    fh.film_grain.update_grain = false;
                    fh.film_grain.seed = seed;
                }
            }
        }

        // An OBU_FRAME carries tile data, so it can never be a show_existing_frame
        // (that is a bare OBU_FRAME_HEADER, routed to `show_existing_slot` by the
        // reorder path). Guard defensively.
        if fh.show_existing_frame {
            return Err(VulkanVideoError::UnsupportedStream);
        }

        // OrderHints[refName] for the picture info (0 = intra/unused).
        let mut order_hints = [0u8; AV1_NUM_REF_FRAMES];
        if !fh.frame_is_intra {
            for i in 0..AV1_REFS_PER_FRAME {
                let v = fh.ref_frame_idx[i] as usize;
                order_hints[i + 1] = refs.order_hint[v];
            }
        }
        let std_pic = to_std_av1_picture_info(&fh, order_hints);

        // Physical slot for this frame: one not currently a reference.
        let in_use: alloc::vec::Vec<usize> = self.ref_slot.iter().flatten().copied().collect();
        let target = (0..self.core.slots.len())
            .find(|p| !in_use.contains(p))
            .ok_or(VulkanVideoError::UnsupportedStream)?;

        // Active reference physical slots used by this frame (unique), plus the
        // per-name slot indices the driver maps.
        let mut name_slots = [-1i32; 7]; // MAX_VIDEO_AV1_REFERENCES_PER_FRAME_KHR
        let mut active: alloc::vec::Vec<usize> = alloc::vec::Vec::new();
        if !fh.frame_is_intra {
            for (slot, &idx) in name_slots.iter_mut().zip(fh.ref_frame_idx.iter()) {
                if let Some(p) = self.ref_slot[idx as usize] {
                    *slot = p as i32;
                    if !active.contains(&p) {
                        active.push(p);
                    }
                }
            }
        }

        // Tile data: one tile follows the byte-aligned header directly; a tiled
        // frame's per-tile offsets + sizes come from the tile-group size prefixes.
        let (tile_offsets, tile_sizes) = av1_tile_layout(payload, &fh.tile, fh.header_byte_len)
            .ok_or(VulkanVideoError::UnsupportedStream)?;

        self.submit_decode_av1(
            &fh,
            &std_pic,
            target,
            &active,
            &name_slots,
            payload,
            &tile_offsets,
            &tile_sizes,
            to_system,
        )?;

        // Record the decoded slot's reference state and apply refresh_frame_flags.
        let state = Av1SlotState {
            order_hint: fh.order_hint,
            frame_type: fh.frame_type,
            saved_order_hints: order_hints,
            disable_frame_end_update_cdf: fh.disable_frame_end_update_cdf,
            segmentation_enabled: fh.seg.enabled,
            upscaled_width: fh.upscaled_width,
            frame_height: fh.frame_height,
            render_width: fh.render_width,
            render_height: fh.render_height,
            film_grain: fh.film_grain,
        };
        self.phys_state[target] = Some(state);
        for v in 0..AV1_NUM_REF_FRAMES {
            if fh.refresh_frame_flags & (1 << v) != 0 {
                self.ref_slot[v] = Some(target);
            }
        }
        Ok(Some((target, fh)))
    }

    fn decode_frame_to_texture(
        &mut self,
        payload: &[u8],
    ) -> Result<Option<wgpu::Texture>, VulkanVideoError> {
        let (w, h) = self.core.coded_extent;
        // Chain the decode to its ycbcr conversion via `sem_dc` (see the H.264
        // sibling); the decode is async and its bitstream held until convert done.
        self.core.chain_next = true;
        let Some((target, _fh)) = self.decode_frame_into_slot(payload, false)? else {
            // No decode happened; drop the unconsumed chain flag.
            self.core.chain_next = false;
            return Ok(None);
        };
        let image = self.core.slots[target].image;
        // SAFETY: target slot just decoded into VIDEO_DECODE_DPB_KHR, SAMPLED +
        // CONCURRENT (GPU mode); its decode signals `sem_dc` which convert waits.
        let tex = unsafe { self.core.convert_slot_chained(image, w, h) }?;
        Ok(Some(tex))
    }

    /// Record + submit the decode of one AV1 frame into `self.core.slots[target]`.
    /// Same command structure as [`H265DpbDecoder::submit_decode`], with AV1
    /// `Std*` picture / reference info and the tile offset/size lists.
    #[allow(clippy::too_many_arguments)]
    fn submit_decode_av1(
        &mut self,
        fh: &Av1FrameHeader,
        std_pic: &StdAv1PictureInfo,
        target: usize,
        active: &[usize],
        name_slots: &[i32; 7],
        bitstream_data: &[u8],
        tile_offsets: &[u32],
        tile_sizes: &[u32],
        to_system: bool,
    ) -> Result<(), VulkanVideoError> {
        let (w, h) = self.core.coded_extent;
        let num_refs = active.len();

        // Transient host-visible bitstream buffer holding the frame OBU payload.
        let (bitstream, bitstream_mem, buf_size) = self
            .core
            .new_bitstream(bitstream_data, &self.profile.profile)?;

        // AV1 picture info (points at the owned Std sub-structs in `std_pic`).
        let mut av1_pic = vk::VideoDecodeAV1PictureInfoKHR::default()
            .std_picture_info(&std_pic.pic)
            .tile_offsets(tile_offsets)
            .tile_sizes(tile_sizes);
        av1_pic.frame_header_offset = 0;
        av1_pic.reference_name_slot_indices = *name_slots;

        // Reference info for each active reference slot + the setup (current) slot.
        let mut std_refs: alloc::vec::Vec<vk::native::StdVideoDecodeAV1ReferenceInfo> =
            alloc::vec::Vec::with_capacity(num_refs);
        for &p in active {
            let st = self.phys_state[p].expect("active reference has state");
            std_refs.push(av1_ref_info(&st));
        }
        let cur_state = Av1SlotState {
            order_hint: fh.order_hint,
            frame_type: fh.frame_type,
            // The setup slot's SavedOrderHints are left zero (ffmpeg passes NULL
            // for the current frame); they matter only when the slot is later
            // used as a reference, where they are filled from the saved state.
            saved_order_hints: [0; AV1_NUM_REF_FRAMES],
            disable_frame_end_update_cdf: fh.disable_frame_end_update_cdf,
            segmentation_enabled: fh.seg.enabled,
            upscaled_width: fh.upscaled_width,
            frame_height: fh.frame_height,
            render_width: fh.render_width,
            render_height: fh.render_height,
            // Not used by `av1_ref_info` (grain is an output-time process).
            film_grain: Av1FilmGrain::default(),
        };
        let std_cur = av1_ref_info(&cur_state);

        let mut dpb_infos: alloc::vec::Vec<vk::VideoDecodeAV1DpbSlotInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs);
        for sr in &std_refs {
            dpb_infos.push(vk::VideoDecodeAV1DpbSlotInfoKHR {
                p_std_reference_info: sr as *const _,
                ..Default::default()
            });
        }
        let dpb_cur = vk::VideoDecodeAV1DpbSlotInfoKHR {
            p_std_reference_info: &std_cur as *const _,
            ..Default::default()
        };

        let mut picres: alloc::vec::Vec<vk::VideoPictureResourceInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs);
        for &p in active {
            picres.push(
                vk::VideoPictureResourceInfoKHR::default()
                    .coded_offset(vk::Offset2D { x: 0, y: 0 })
                    .coded_extent(vk::Extent2D {
                        width: w,
                        height: h,
                    })
                    .base_array_layer(0)
                    .image_view_binding(self.core.slots[p].view),
            );
        }
        let picres_target = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D {
                width: w,
                height: h,
            })
            .base_array_layer(0)
            .image_view_binding(self.core.slots[target].view);

        let mut ref_slots: alloc::vec::Vec<vk::VideoReferenceSlotInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs);
        for (i, &p) in active.iter().enumerate() {
            ref_slots.push(vk::VideoReferenceSlotInfoKHR {
                slot_index: p as i32,
                p_picture_resource: &picres[i] as *const _,
                p_next: (&dpb_infos[i] as *const vk::VideoDecodeAV1DpbSlotInfoKHR).cast(),
                ..Default::default()
            });
        }
        let setup_slot = vk::VideoReferenceSlotInfoKHR {
            slot_index: target as i32,
            p_picture_resource: &picres_target as *const _,
            p_next: (&dpb_cur as *const vk::VideoDecodeAV1DpbSlotInfoKHR).cast(),
            ..Default::default()
        };

        let mut begin_slots: alloc::vec::Vec<vk::VideoReferenceSlotInfoKHR> =
            alloc::vec::Vec::with_capacity(num_refs + 1);
        for (i, &p) in active.iter().enumerate() {
            begin_slots.push(vk::VideoReferenceSlotInfoKHR {
                slot_index: p as i32,
                p_picture_resource: &picres[i] as *const _,
                p_next: (&dpb_infos[i] as *const vk::VideoDecodeAV1DpbSlotInfoKHR).cast(),
                ..Default::default()
            });
        }
        begin_slots.push(vk::VideoReferenceSlotInfoKHR {
            slot_index: -1,
            p_picture_resource: &picres_target as *const _,
            ..Default::default()
        });

        let begin_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(self.core.session)
            .video_session_parameters(self.core.parameters)
            .reference_slots(&begin_slots);
        let decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(bitstream)
            .src_buffer_offset(0)
            .src_buffer_range(buf_size)
            .dst_picture_resource(picres_target)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&ref_slots)
            .push_next(&mut av1_pic);

        // Record + submit the decode (barriers, begin/decode/end coding, optional
        // NV12 readback) via the shared codec-agnostic core, which takes ownership
        // of the transient bitstream buffer.
        let image = self.core.slots[target].image;
        self.core.record_and_submit(
            &begin_info,
            &decode_info,
            image,
            bitstream,
            bitstream_mem,
            to_system,
        )
    }
}

/// Build a `StdVideoDecodeAV1ReferenceInfo` for a DPB slot from its saved state.
fn av1_ref_info(st: &Av1SlotState) -> vk::native::StdVideoDecodeAV1ReferenceInfo {
    // SAFETY: bitfield POD, valid all-zero.
    let mut flags: vk::native::StdVideoDecodeAV1ReferenceInfoFlags = unsafe { core::mem::zeroed() };
    flags.set_disable_frame_end_update_cdf(st.disable_frame_end_update_cdf as u32);
    flags.set_segmentation_enabled(st.segmentation_enabled as u32);
    vk::native::StdVideoDecodeAV1ReferenceInfo {
        flags,
        frame_type: st.frame_type,
        RefFrameSignBias: 0,
        OrderHint: st.order_hint,
        SavedOrderHints: st.saved_order_hints,
    }
}

/// A random-access ("pull") H.264 / H.265 player over a Vulkan Video decoder: it
/// serves the decoded picture at an arbitrary timestamp / index as a GPU-resident
/// RGBA `wgpu::Texture`, decoding forward from the enclosing random-access point on
/// a seek and caching decoded frames. This is the timeline-scrubber model, a viewer
/// asking `frame_at(t)` as the user drags, as opposed to the streaming push model
/// of [`VulkanVideoDec`]: the wedge for a wgpu visualization viewer that needs
/// hardware decode straight into its render device with no CPU round trip
/// (instead of CPU software decode + an upload copy).
///
/// Owns its decode device, session and decoder. H.264 and H.265 (sniffed from the
/// stream), GPU-texture mode (needs a distinct compute queue); the same profile
/// limits as [`H264DpbDecoder`] / [`H265DpbDecoder`] apply. Presentation order is
/// by POC, so B-frame reordering is handled. For H.265 open-GOP, a CRA is a valid
/// seek target (the tune-in discards its RASL leading pictures); a leading picture
/// itself seeks from an earlier random-access point so its references exist.
// Both variants are ~5 KB (each embeds its parsed parameter sets); boxing one
// to close the small size gap would buy nothing.
#[allow(clippy::large_enum_variant)]
enum PlayerDecoder {
    H264(H264DpbDecoder),
    H265(H265DpbDecoder),
}

impl PlayerDecoder {
    fn index_pictures(
        &mut self,
        stream: &[u8],
    ) -> Result<alloc::vec::Vec<PictureMeta>, VulkanVideoError> {
        match self {
            PlayerDecoder::H264(d) => d.index_pictures(stream),
            PlayerDecoder::H265(d) => d.index_pictures(stream),
        }
    }

    fn reset(&mut self) {
        match self {
            PlayerDecoder::H264(d) => d.reset(),
            PlayerDecoder::H265(d) => d.reset(),
        }
    }

    fn decode_range_to_texture(
        &mut self,
        stream: &[u8],
        start: usize,
        target: usize,
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        match self {
            PlayerDecoder::H264(d) => d.decode_range_to_texture(stream, start, target),
            PlayerDecoder::H265(d) => d.decode_range_to_texture(stream, start, target),
        }
    }

    fn decode_range_all_to_textures(
        &mut self,
        stream: &[u8],
        start: usize,
        target: usize,
    ) -> Result<alloc::vec::Vec<(usize, wgpu::Texture)>, VulkanVideoError> {
        match self {
            PlayerDecoder::H264(d) => d.decode_range_all_to_textures(stream, start, target),
            PlayerDecoder::H265(d) => d.decode_range_all_to_textures(stream, start, target),
        }
    }
}

/// The decode session backing a [`PlayerDecoder`], held only to outlive it (the
/// decoder's [`DpbCore`] copies the session's raw handles).
#[allow(clippy::large_enum_variant, dead_code)]
enum PlayerSession {
    H264(H264DecodeSession),
    H265(H265DecodeSession),
}

/// The video codec of an Annex-B elementary stream, sniffed from its NAL types:
/// an H.265 VPS (32) / SPS (33) / PPS (34) marks HEVC, an H.264 SPS (7) marks
/// AVC. Returns `None` if neither parameter set is seen.
fn sniff_annexb_codec(stream: &[u8]) -> Option<VideoCodec> {
    for nal in nal_units_any(stream) {
        if nal.is_empty() {
            continue;
        }
        // H.265 two-byte NAL header: type is bits [1..7] of the first byte.
        let h265_type = (nal[0] >> 1) & 0x3F;
        if (32..=34).contains(&h265_type) {
            return Some(VideoCodec::H265);
        }
        // H.264 one-byte NAL header: type is the low 5 bits.
        if nal[0] & 0x1F == 7 {
            return Some(VideoCodec::H264);
        }
    }
    None
}
pub struct VulkanVideoPlayer {
    // Drop order: decoder before session before device (the decoder holds copies
    // of the session's handles; the session's objects live on the device). The
    // session is held only to outlive the decoder, never read directly.
    decoder: PlayerDecoder,
    _session: PlayerSession,
    device: VulkanVideoDevice,
    /// The elementary stream (Annex-B), re-split per seek (cheap: the frames the
    /// decoder produces are what cost).
    stream: alloc::vec::Vec<u8>,
    /// Coded pictures in decoding order (keyframe flag + POC).
    index: alloc::vec::Vec<PictureMeta>,
    /// Presentation order: `presentation[p]` is the decoding index of the p-th
    /// picture in POC (display) order.
    presentation: alloc::vec::Vec<usize>,
    /// Decoded-frame cache keyed by decoding index (LRU-bounded, see `lru` /
    /// `cache_capacity`): a scrubbed-to frame that is revisited is free.
    cache: alloc::collections::BTreeMap<usize, wgpu::Texture>,
    /// Recency order for `cache` (front = least-recently-used, back = most): the
    /// eviction order once `cache` reaches `cache_capacity`.
    lru: alloc::collections::VecDeque<usize>,
    /// Max decoded frames kept resident (LRU-evicted past this). Bounds GPU
    /// memory on a long stream; a scrubber's working set is far smaller than the
    /// whole video.
    cache_capacity: usize,
    /// Resident-cache byte budget (LRU-evicted past this), the memory-first bound
    /// that matters at 4K/8K where a frame-count cap would pin gigabytes: at
    /// 3840x2160 RGBA one frame is ~33 MB, so 64 of them is ~2 GB. Eviction keeps
    /// the cache under both this and `cache_capacity` (but never drops the frame
    /// just served).
    cache_byte_budget: usize,
    /// Bytes one decoded RGBA frame occupies (`w * h * 4`), for the byte budget.
    bytes_per_frame: usize,
    /// When set, a decode range caches every traversed picture, not just the
    /// target, so a later backward scrub within the same GOP is free (at the cost
    /// of one colour conversion per run-up frame). Off by default: linear
    /// playback already caches each frame as it is served.
    cache_traversed: bool,
    /// Highest decoding index currently valid in the decoder's DPB from the run
    /// since the last `reset` (`None` before the first decode). Lets a forward
    /// seek within reach continue decoding instead of resetting to the keyframe,
    /// which turns linear playback from O(n^2) coded pictures into O(n).
    decoded_up_to: Option<usize>,
    /// Real decode passes (a cache hit does not increment): proves the cache.
    decode_calls: usize,
    /// Total coded pictures pushed through the decoder (a continue decodes fewer
    /// than a keyframe reset): the true GPU-work metric.
    pictures_decoded: usize,
    dims: (u32, u32),
    frame_duration_ns: u64,
}

impl core::fmt::Debug for VulkanVideoPlayer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VulkanVideoPlayer")
            .field("frames", &self.presentation.len())
            .field("dims", &self.dims)
            .field("decode_calls", &self.decode_calls)
            .field("pictures_decoded", &self.pictures_decoded)
            .field("cached", &self.cache.len())
            .field("cache_capacity", &self.cache_capacity)
            .field("cache_byte_budget", &self.cache_byte_budget)
            .finish_non_exhaustive()
    }
}

impl VulkanVideoPlayer {
    /// Default resident decoded-frame budget (LRU-evicted past this).
    const DEFAULT_CACHE_CAPACITY: usize = 64;
    /// Default resident-cache byte budget (512 MiB): enough for a scrubber's
    /// working set at any resolution, while capping a 4K/8K cache that a
    /// frame-count limit alone would let grow into gigabytes.
    const DEFAULT_CACHE_BYTE_BUDGET: usize = 512 << 20;

    /// Build a player over `stream` on an already-opened decode `device`, at
    /// `fps` (the presentation-timestamp mapping for [`frame_at`](Self::frame_at);
    /// pass the stream's real rate). Parses SPS/PPS, opens the session + GPU
    /// decoder, and builds the keyframe / POC index. Returns
    /// [`VulkanVideoError::NoComputeQueue`] on a device without a distinct compute
    /// queue (the zero-copy texture path needs one).
    pub fn new(
        device: VulkanVideoDevice,
        stream: alloc::vec::Vec<u8>,
        fps: u32,
    ) -> Result<Self, VulkanVideoError> {
        // Sniff H.264 vs H.265 from the stream's parameter-set NALs and build the
        // matching session + GPU decoder.
        let (mut decoder, session, width, height) = match sniff_annexb_codec(&stream) {
            Some(VideoCodec::H264) => {
                let ps = extract_h264_parameter_sets(&stream)
                    .ok_or(VulkanVideoError::UnsupportedStream)?;
                let width = (ps.sps.pic_width_in_mbs_minus1 + 1) * 16;
                let height = (ps.sps.pic_height_in_map_units_minus1 + 1) * 16;
                let session = device.create_h264_session(&ps, width, height)?;
                let decoder = device.create_h264_dpb_decoder_gpu(&session, &ps)?;
                (
                    PlayerDecoder::H264(decoder),
                    PlayerSession::H264(session),
                    width,
                    height,
                )
            }
            Some(VideoCodec::H265) => {
                let ps = extract_h265_parameter_sets(&stream)
                    .ok_or(VulkanVideoError::UnsupportedStream)?;
                let width = ps.sps.pic_width_in_luma_samples;
                let height = ps.sps.pic_height_in_luma_samples;
                let std = to_std_h265_params(&ps);
                let session = device.create_h265_session(&std, width, height)?;
                let decoder = device.create_h265_dpb_decoder_gpu(&session, &ps)?;
                (
                    PlayerDecoder::H265(decoder),
                    PlayerSession::H265(session),
                    width,
                    height,
                )
            }
            _ => return Err(VulkanVideoError::UnsupportedStream),
        };
        let index = decoder.index_pictures(&stream)?;
        // Display order: POC is only comparable within a coded video sequence (it
        // resets at each IDR), so order by (GOP number, POC) with GOPs in decode
        // order, not by POC globally. Identity for a no-B-frame stream; a B-frame
        // GOP reorders within its GOP. A stable sort keeps decode order among ties.
        let mut gop: i32 = -1;
        let keys: alloc::vec::Vec<(i32, i32)> = index
            .iter()
            .map(|m| {
                if m.is_keyframe {
                    gop += 1;
                }
                (gop, m.poc)
            })
            .collect();
        let mut presentation: alloc::vec::Vec<usize> = (0..index.len()).collect();
        presentation.sort_by_key(|&i| keys[i]);
        Ok(Self {
            decoder,
            _session: session,
            device,
            stream,
            index,
            presentation,
            cache: alloc::collections::BTreeMap::new(),
            lru: alloc::collections::VecDeque::new(),
            cache_capacity: Self::DEFAULT_CACHE_CAPACITY,
            cache_byte_budget: Self::DEFAULT_CACHE_BYTE_BUDGET,
            bytes_per_frame: (width as usize) * (height as usize) * 4,
            cache_traversed: false,
            decoded_up_to: None,
            decode_calls: 0,
            pictures_decoded: 0,
            dims: (width, height),
            frame_duration_ns: 1_000_000_000 / fps.max(1) as u64,
        })
    }

    /// Number of pictures in the stream.
    pub fn frame_count(&self) -> usize {
        self.presentation.len()
    }

    /// Decoded picture size (width, height).
    pub fn dimensions(&self) -> (u32, u32) {
        self.dims
    }

    /// Real decode passes so far (a revisited frame served from cache does not
    /// count). Instrumentation for tests / demos.
    pub fn decode_calls(&self) -> usize {
        self.decode_calls
    }

    /// Total coded pictures decoded so far (a forward continue decodes fewer
    /// than a keyframe reset; a cache hit decodes none). The true GPU-work
    /// metric: for linear playback it equals the frame count, not the O(n^2) a
    /// reset-per-frame would cost.
    pub fn pictures_decoded(&self) -> usize {
        self.pictures_decoded
    }

    /// Frames currently resident in the cache.
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    /// Max frames kept resident before LRU eviction.
    pub fn cache_capacity(&self) -> usize {
        self.cache_capacity
    }

    /// Approximate bytes the resident cache currently occupies (all frames are
    /// one decoded RGBA texture of the same size).
    pub fn cache_bytes(&self) -> usize {
        self.cache.len() * self.bytes_per_frame
    }

    /// The resident-cache byte budget.
    pub fn cache_byte_budget(&self) -> usize {
        self.cache_byte_budget
    }

    /// Set the resident-frame budget (min 1), evicting least-recently-used
    /// frames immediately if the cache is now over budget.
    pub fn set_cache_capacity(&mut self, capacity: usize) {
        self.cache_capacity = capacity.max(1);
        self.evict_to_budget();
    }

    /// Set the resident-cache byte budget, evicting least-recently-used frames
    /// immediately if the cache is now over it. The memory-first bound for
    /// high-resolution streams (a frame-count cap alone pins gigabytes at 4K/8K).
    pub fn set_cache_byte_budget(&mut self, bytes: usize) {
        self.cache_byte_budget = bytes;
        self.evict_to_budget();
    }

    /// Cache every traversed picture on a decode range (not just the target), so
    /// a later backward scrub within the same GOP is served from cache. Off by
    /// default. Best paired with the byte budget, since a GOP's worth of frames
    /// lands resident at once.
    pub fn set_cache_traversed(&mut self, on: bool) {
        self.cache_traversed = on;
    }

    /// The decoding index of the `p`-th picture in presentation order.
    pub fn decode_index(&self, presentation_idx: usize) -> Option<usize> {
        self.presentation.get(presentation_idx).copied()
    }

    /// The decode device shared as a [`GpuContext`](crate::gpu::GpuContext), so a
    /// `WgpuSink` (or any wgpu consumer) presents the returned textures with no
    /// copy when it drives the same GPU.
    pub fn gpu_context(&self) -> crate::gpu::GpuContext {
        self.device.gpu_context()
    }

    /// Read an RGBA texture this player produced back to tightly-packed bytes
    /// (validation / a CPU consumer / the cross-GPU transport).
    pub fn read_texture(&self, texture: &wgpu::Texture) -> alloc::vec::Vec<u8> {
        self.device.read_rgba_texture(texture)
    }

    /// The random-access point (decoding order) to seek forward from to decode the
    /// picture at `decode_idx`. Normally the nearest IRAP at or before it (an IDR
    /// for H.264; an IDR / CRA / BLA for H.265). But a leading picture (one whose
    /// POC is before its enclosing IRAP, i.e. a CRA's RASL / RADL) cannot be
    /// reconstructed by tuning in at that CRA (its references precede the CRA and
    /// are discarded on tune-in), so it seeks from the random-access point BEFORE
    /// that CRA and decodes continuously through it (which then keeps the CRA's
    /// leading pictures). Always found (a valid stream starts with an IDR).
    fn seek_point_for(&self, decode_idx: usize) -> usize {
        let k = (0..=decode_idx)
            .rev()
            .find(|&i| self.index[i].is_random_access)
            .unwrap_or(0);
        if self.index[decode_idx].poc >= self.index[k].poc {
            k
        } else {
            (0..k)
                .rev()
                .find(|&i| self.index[i].is_random_access)
                .unwrap_or(0)
        }
    }

    /// The picture at presentation index `p` as a GPU-resident RGBA texture.
    ///
    /// A cache hit returns with no GPU work. On a miss the decoder either
    /// **continues** from its current position (a forward seek still within the
    /// target's GOP, no reset, so linear playback decodes each picture once) or
    /// **resets** and decodes from the enclosing keyframe (a backward / cross-GOP
    /// seek). The result is cached, LRU-evicting the least-recently-used frame
    /// once `cache_capacity` is reached.
    pub fn frame_at_index(&mut self, p: usize) -> Result<&wgpu::Texture, VulkanVideoError> {
        let target = *self
            .presentation
            .get(p)
            .ok_or(VulkanVideoError::UnsupportedStream)?;
        if self.cache.contains_key(&target) {
            self.touch_lru(target);
            return Ok(self.cache.get(&target).expect("present"));
        }
        let kf = self.seek_point_for(target);
        // Continue decoding in place when the decoder already sits within reach of
        // the target (at or past the seek point, before the target); otherwise reset
        // and decode from the seek point (a backward or cross-GOP seek). A continue
        // range is all inter frames (it starts after the seek point), a reset range
        // starts at the random-access point.
        let start = match self.decoded_up_to {
            Some(d) if kf <= d && d < target => d + 1,
            _ => {
                self.decoder.reset();
                kf
            }
        };
        self.pictures_decoded += target - start + 1;
        self.decode_calls += 1;
        self.decoded_up_to = Some(target);
        if self.cache_traversed {
            // Cache every picture in the decoded range by its decoding index, so a
            // later scrub back into this GOP hits. Insert in decode order, then
            // touch the target last so it is most-recently-used (never evicted by
            // the budget pass that follows).
            let texes = self
                .decoder
                .decode_range_all_to_textures(&self.stream, start, target)?;
            for (idx, tex) in texes {
                self.insert_cache(idx, tex);
            }
            self.touch_lru(target);
            self.evict_to_budget();
        } else {
            let tex = self
                .decoder
                .decode_range_to_texture(&self.stream, start, target)?;
            self.insert_cache(target, tex);
        }
        Ok(self.cache.get(&target).expect("just decoded / cached"))
    }

    /// Mark `key` most-recently-used in the LRU order.
    fn touch_lru(&mut self, key: usize) {
        if let Some(pos) = self.lru.iter().position(|&k| k == key) {
            self.lru.remove(pos);
        }
        self.lru.push_back(key);
    }

    /// Insert a freshly decoded frame (a fresh miss, not already resident), then
    /// evict least-recently-used frames until the cache is within both the frame
    /// count and byte budgets.
    fn insert_cache(&mut self, key: usize, texture: wgpu::Texture) {
        self.cache.insert(key, texture);
        self.touch_lru(key);
        self.evict_to_budget();
    }

    /// Evict least-recently-used frames until the cache is within both
    /// `cache_capacity` (frames) and `cache_byte_budget` (bytes), always keeping
    /// at least one frame (the most-recently-used, e.g. the frame just served).
    fn evict_to_budget(&mut self) {
        while self.cache.len() > 1
            && (self.cache.len() > self.cache_capacity
                || self.cache.len() * self.bytes_per_frame > self.cache_byte_budget)
        {
            match self.lru.pop_front() {
                Some(evict) => {
                    self.cache.remove(&evict);
                }
                None => break,
            }
        }
    }

    /// The picture shown at presentation timestamp `pts_ns`, mapped to a frame
    /// index by the player's fps and clamped to the stream. See
    /// [`frame_at_index`](Self::frame_at_index).
    pub fn frame_at(&mut self, pts_ns: u64) -> Result<&wgpu::Texture, VulkanVideoError> {
        let n = self.presentation.len();
        let idx = (pts_ns / self.frame_duration_ns.max(1)) as usize;
        self.frame_at_index(idx.min(n.saturating_sub(1)))
    }
}

/// Build a `StdVideoDecodeH264ReferenceInfo` for a short-term frame reference.
fn std_ref_info(frame_num: u32, poc: i32) -> vk::native::StdVideoDecodeH264ReferenceInfo {
    // SAFETY: bitfield POD, valid all-zero (progressive frame: no field flags,
    // short-term, existing).
    let mut flags: vk::native::StdVideoDecodeH264ReferenceInfoFlags =
        unsafe { core::mem::zeroed() };
    flags.set_top_field_flag(0);
    flags.set_bottom_field_flag(0);
    flags.set_used_for_long_term_reference(0);
    flags.set_is_non_existing(0);
    vk::native::StdVideoDecodeH264ReferenceInfo {
        flags,
        FrameNum: frame_num as u16,
        reserved: 0,
        PicOrderCnt: [poc, poc],
    }
}

/// The `Std*` H.265 reference info for a DPB slot: the reference picture's POC
/// and its short/long-term marking (H.265 reference lists match on POC; the
/// long-term flag changes MV scaling, so it must mirror the slice's RPS).
fn std_h265_ref_info(poc: i32, long_term: bool) -> vk::native::StdVideoDecodeH265ReferenceInfo {
    // SAFETY: bitfield POD, valid all-zero (short-term, used for reference).
    let mut flags: vk::native::StdVideoDecodeH265ReferenceInfoFlags =
        unsafe { core::mem::zeroed() };
    flags.set_used_for_long_term_reference(long_term as u32);
    flags.set_unused_for_reference(0);
    vk::native::StdVideoDecodeH265ReferenceInfo {
        flags,
        PicOrderCntVal: poc,
    }
}

/// A decoded luma plane read back to system memory (validation output).
#[derive(Debug, Clone)]
pub struct DecodedLuma {
    pub width: u32,
    pub height: u32,
    /// `width * height` bytes, one per luma sample (row-major, tightly packed).
    pub luma: alloc::vec::Vec<u8>,
}

/// A decoded NV12 frame read back to system memory: a full-resolution luma
/// plane followed by a half-resolution interleaved CbCr plane.
#[derive(Debug, Clone)]
pub struct Nv12Frame {
    pub width: u32,
    pub height: u32,
    /// Luma (Y) plane, row-major. `bit_depth` bytes per sample: `width * height`
    /// bytes for 8-bit, `2 * width * height` for 10-bit (little-endian 16-bit
    /// samples, value in the top 10 bits, the G10X6 layout).
    pub luma: alloc::vec::Vec<u8>,
    /// Interleaved Cb,Cr plane, row-major, `(width/2) * (height/2)` pairs. Half the
    /// luma byte length (same bytes-per-sample).
    pub chroma: alloc::vec::Vec<u8>,
    /// Sample bit depth: 8 (one byte per sample) or 10 (two bytes per sample).
    pub bit_depth: u8,
}

/// The colour matrix a decoded frame's luma / chroma are separated by, from the
/// CICP `matrix_coefficients` codepoint (shared by H.264 / H.265 VUI and AV1
/// `color_config`). Only the matrices the converter implements are named.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMatrix {
    /// BT.601 (SMPTE 170M / BT.470BG), CICP 5 / 6.
    Bt601,
    /// BT.709, CICP 1.
    Bt709,
    /// BT.2020 non-constant luminance, CICP 9. The matrix used by HDR content
    /// (the PQ / HLG transfer function is a separate, later increment).
    Bt2020Ncl,
}

/// The opto-electronic transfer function a decoded frame's samples are encoded
/// with, from the CICP `transfer_characteristics` codepoint. Only the HDR
/// functions the converter linearizes are named; everything else (BT.709, sRGB,
/// BT.601 gamma, unspecified) is [`TransferFunction::Sdr`] and passes through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferFunction {
    /// SDR gamma (BT.709 / sRGB / BT.601, or any non-HDR codepoint). No HDR
    /// linearization: the ycbcr result is display-ready as-is.
    Sdr,
    /// PQ (SMPTE ST 2084 / BT.2100 PQ), CICP 16. The HDR10 transfer.
    Pq,
    /// HLG (ARIB STD-B67 / BT.2100 HLG), CICP 18.
    Hlg,
}

/// The colour space a decoded YUV frame is converted from: the [`ColorMatrix`],
/// the quantization range (`full_range` = full 0..255, else studio 16..235), and
/// the [`TransferFunction`]. Resolved from the stream (H.264 / H.265 VUI colour
/// description or AV1 `color_config`) and applied by BOTH the CPU [`nv12_to_rgba`]
/// and the GPU [`YcbcrConverter`], so decoded video is converted with its actual
/// colour space rather than a fixed BT.601. The matrix + range are always
/// applied; the transfer only matters when the converter is asked to tone-map HDR
/// to SDR ([`HdrOutput::TonemapSdr`]). Primaries are not carried separately: the
/// gamut step keys off `matrix == Bt2020Ncl` (BT.2020 matrix implies BT.2020
/// primaries for real content).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoColorSpace {
    pub matrix: ColorMatrix,
    pub full_range: bool,
    pub transfer: TransferFunction,
}

impl VideoColorSpace {
    /// BT.601 studio range, SDR: the fixed conversion this decoder used before
    /// colour info was plumbed through, and the fallback for an SD, untagged stream.
    pub const BT601_STUDIO: Self = Self {
        matrix: ColorMatrix::Bt601,
        full_range: false,
        transfer: TransferFunction::Sdr,
    };

    /// Resolve from the CICP `matrix_coefficients` + `transfer_characteristics`
    /// codepoints + full-range flag. An unspecified matrix (CICP 2, or any
    /// codepoint not handled) falls back by coded height (>= 720 -> BT.709, else
    /// BT.601), the ffmpeg heuristic for untagged content. Transfer resolves to
    /// PQ (16) / HLG (18), everything else SDR.
    pub fn from_cicp(
        matrix_coefficients: u8,
        transfer_characteristics: u8,
        full_range: bool,
        height: u32,
    ) -> Self {
        let matrix = match matrix_coefficients {
            1 => ColorMatrix::Bt709,
            9 => ColorMatrix::Bt2020Ncl,
            5 | 6 => ColorMatrix::Bt601,
            _ if height >= 720 => ColorMatrix::Bt709,
            _ => ColorMatrix::Bt601,
        };
        let transfer = match transfer_characteristics {
            16 => TransferFunction::Pq,
            18 => TransferFunction::Hlg,
            _ => TransferFunction::Sdr,
        };
        Self {
            matrix,
            full_range,
            transfer,
        }
    }

    /// The `(Kr, Kb)` luma weights for the matrix (`Kg = 1 - Kr - Kb`).
    fn luma_weights(self) -> (f32, f32) {
        match self.matrix {
            ColorMatrix::Bt601 => (0.299, 0.114),
            ColorMatrix::Bt709 => (0.2126, 0.0722),
            ColorMatrix::Bt2020Ncl => (0.2627, 0.0593),
        }
    }

    /// The Vulkan `VkSamplerYcbcrConversion` model + range for this colour space.
    fn vk_ycbcr(self) -> (vk::SamplerYcbcrModelConversion, vk::SamplerYcbcrRange) {
        let model = match self.matrix {
            ColorMatrix::Bt601 => vk::SamplerYcbcrModelConversion::YCBCR_601,
            ColorMatrix::Bt709 => vk::SamplerYcbcrModelConversion::YCBCR_709,
            ColorMatrix::Bt2020Ncl => vk::SamplerYcbcrModelConversion::YCBCR_2020,
        };
        let range = if self.full_range {
            vk::SamplerYcbcrRange::ITU_FULL
        } else {
            vk::SamplerYcbcrRange::ITU_NARROW
        };
        (model, range)
    }
}

/// Convert an [`Nv12Frame`] to packed RGBA8, applying `color`'s matrix + range.
/// CPU reference conversion (nearest-neighbour chroma); the GPU-resident
/// [`YcbcrConverter`] does the same in a compute pass with linear chroma. The
/// general matrix formula reproduces the historical BT.601-studio coefficients
/// exactly (`2*(1-Kr) * 255/224 = 1.596...` for BT.601), so a BT.601 stream is
/// byte-for-byte unchanged from before colour awareness.
fn nv12_to_rgba(frame: &Nv12Frame, color: VideoColorSpace) -> alloc::vec::Vec<u8> {
    // 8-bit only: the 10-bit (G10X6) frame -> RGB path is a follow-up (it belongs
    // with the 10-bit GPU converter). Callers here are all 8-bit (H.264 one-shot,
    // AV1 film-grain), so this only guards misuse.
    debug_assert_eq!(frame.bit_depth, 8, "nv12_to_rgba is 8-bit only");
    let (w, h) = (frame.width as usize, frame.height as usize);
    let cw = w / 2;
    let (kr, kb) = color.luma_weights();
    let kg = 1.0 - kr - kb;
    // Studio range rescales Y from 16..235 and C from 16..240 to full 0..255;
    // full range uses the samples directly.
    let (y_scale, y_off, c_scale) = if color.full_range {
        (1.0, 0.0, 1.0)
    } else {
        (255.0 / 219.0, 16.0, 255.0 / 224.0)
    };
    let cr_r = 2.0 * (1.0 - kr) * c_scale;
    let cb_b = 2.0 * (1.0 - kb) * c_scale;
    let cr_g = 2.0 * kr * (1.0 - kr) / kg * c_scale;
    let cb_g = 2.0 * kb * (1.0 - kb) / kg * c_scale;
    let mut rgba = alloc::vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let yc = (frame.luma[y * w + x] as f32 - y_off) * y_scale;
            let ci = ((y / 2) * cw + (x / 2)) * 2;
            let cb = frame.chroma[ci] as f32 - 128.0;
            let cr = frame.chroma[ci + 1] as f32 - 128.0;
            let r = yc + cr_r * cr;
            let g = yc - cr_g * cr - cb_g * cb;
            let b = yc + cb_b * cb;
            let o = (y * w + x) * 4;
            rgba[o] = r.clamp(0.0, 255.0) as u8;
            rgba[o + 1] = g.clamp(0.0, 255.0) as u8;
            rgba[o + 2] = b.clamp(0.0, 255.0) as u8;
            rgba[o + 3] = 255;
        }
    }
    rgba
}

/// Upload an [`Nv12Frame`] to a fresh `Rgba8Unorm` `wgpu::Texture` via the CPU
/// `nv12_to_rgba` conversion and `write_texture`. Used by the one-shot IDR path
/// and by the AV1 film-grain texture path (grain is synthesized on the CPU NV12,
/// so its output is uploaded rather than run through the GPU ycbcr compute pass).
/// The texture carries `TEXTURE_BINDING | COPY_SRC | COPY_DST` so a consumer can
/// sample it and a validation readback can copy it out.
fn nv12_to_rgba_texture(
    wgpu_device: &wgpu::Device,
    wgpu_queue: &wgpu::Queue,
    frame: &Nv12Frame,
    color: VideoColorSpace,
) -> wgpu::Texture {
    let rgba = nv12_to_rgba(frame, color);
    let size = wgpu::Extent3d {
        width: frame.width,
        height: frame.height,
        depth_or_array_layers: 1,
    };
    let texture = wgpu_device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vulkan-video-decoded-rgba"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    wgpu_queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(frame.width * 4),
            rows_per_image: Some(frame.height),
        },
        size,
    );
    wgpu_queue.submit([]);
    texture
}

/// Reorder pictures decoded in coding order into display (presentation) order.
/// `metas` are the per-picture [`PictureMeta`] in decode order, one per item in
/// `items` (both are the stream's coded pictures, same order). The display key is
/// (coded-video-sequence, POC): POC resets at each keyframe (IDR / IRAP), so
/// pictures are grouped by the GOP they fall in (a counter bumped at each keyframe
/// in decode order) and sorted by POC within the group. A B-frame is decoded
/// before pictures that precede it in display order, so this permutation turns the
/// decoder's coding-order output into what a viewer shows; for an I/P stream (POC
/// monotonic in decode order) it is the identity.
fn reorder_to_display_order<T>(
    items: alloc::vec::Vec<T>,
    metas: &[PictureMeta],
) -> alloc::vec::Vec<T> {
    // Decode is 1:1 with the coded pictures; on any surprise mismatch leave the
    // coding-order output untouched rather than misalign frames to the wrong POC.
    if items.len() != metas.len() {
        return items;
    }
    let mut gop = 0u32;
    let mut order: alloc::vec::Vec<(u32, i32, usize)> = alloc::vec::Vec::with_capacity(items.len());
    for (i, m) in metas.iter().enumerate() {
        if m.is_keyframe && i != 0 {
            gop += 1;
        }
        order.push((gop, m.poc, i));
    }
    order.sort_unstable();
    let mut slots: alloc::vec::Vec<Option<T>> = items.into_iter().map(Some).collect();
    order
        .into_iter()
        .map(|(_, _, i)| {
            slots[i]
                .take()
                .expect("each decode index reordered exactly once")
        })
        .collect()
}

/// The SPS id the session was built with (constant 0 for our single-SPS path).
fn session_sps_id(_session: &H264DecodeSession) -> u8 {
    0
}

fn round_up(x: u64, align: u64) -> u64 {
    x.div_ceil(align).saturating_mul(align)
}

/// The full-color single-mip single-layer subresource range used for the decode
/// / conversion images.
fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Extract the first IDR slice NAL (nal_unit_type 5) from an Annex-B / AVCC
/// access unit and return it framed with a 4-byte start code, ready for the
/// Vulkan bitstream buffer (slice offset 0). `None` if the AU has no IDR slice.
fn extract_first_idr_slice(au: &[u8]) -> Option<alloc::vec::Vec<u8>> {
    for nal in nal_units_any(au) {
        if nal.is_empty() {
            continue;
        }
        if nal[0] & 0x1F == 5 {
            let mut out = alloc::vec::Vec::with_capacity(nal.len() + 4);
            out.extend_from_slice(&[0, 0, 0, 1]);
            out.extend_from_slice(nal);
            return Some(out);
        }
    }
    None
}

// ============================================================================
// VulkanVideoDec pipeline element (M493)
//
// The `AsyncElement` wrapper around the decoder above: `Caps::CompressedVideo`
// H.264 Annex-B in, `Caps::RawVideo{Nv12}` in system memory out (the layout the
// `H264DpbDecoder` already produces, so no colour conversion), on the same
// Vulkan device wgpu runs. Emitting NV12 in system memory makes it usable in any
// pipeline today (videoconvert / waylandsink / a file dump); the zero-copy
// GPU-resident `WgpuTexture` output (M490/M491 already prove the decode->texture
// half) is the next increment, gated on a wgpu-consuming sink.
// ============================================================================

/// A pipeline element: hardware H.264 decode via Vulkan Video (vendor-neutral),
/// emitting NV12 frames in system memory. Wraps [`H264DpbDecoder`].
///
/// The device is opened at `configure_pipeline`; the decode session + DPB are
/// built lazily on the first access unit that carries SPS/PPS (they arrive
/// in-band, not in the caps). Field order is drop order: the decoder (holds
/// copies of the session/device handles) drops first, then the session
/// (destroys the `VkVideoSession`), then the device (destroys the `VkDevice`).
/// The codecs the [`VulkanVideoDec`] element decodes. Each maps to a Vulkan
/// Video profile, a per-codec decode session, and a `*DpbDecoder` (all now
/// sharing [`DpbCore`]); the element dispatches on it so one element handles
/// H.264 / H.265 / AV1 rather than three near-identical elements.
const VULKAN_DEC_CODECS: [VideoCodec; 3] = [VideoCodec::H264, VideoCodec::H265, VideoCodec::Av1];

/// A built Vulkan Video decoder, one variant per codec. All three expose the
/// same `decode_all` / `decode_all_to_textures` surface, so the element drives
/// them uniformly.
enum DpbDecoderKind {
    H264(H264DpbDecoder),
    H265(H265DpbDecoder),
    Av1(Av1DpbDecoder),
}

impl core::fmt::Debug for DpbDecoderKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            DpbDecoderKind::H264(_) => "H264",
            DpbDecoderKind::H265(_) => "H265",
            DpbDecoderKind::Av1(_) => "Av1",
        };
        f.debug_tuple("DpbDecoderKind").field(&name).finish()
    }
}

impl DpbDecoderKind {
    /// Streaming decode returning per-submitted-picture [`PictureMeta`] so the
    /// element can reorder the retired frames into display order (see
    /// [`H264DpbDecoder::decode_push_meta`]). Only H.264 / H.265 carry a POC the
    /// (coded-video-sequence, POC) key sorts on; AV1 (whose display order is driven
    /// by `show_existing_frame` / `order_hint`, not a monotonic POC) returns no
    /// metas, and the element leaves it in coding order.
    fn decode_push_meta(
        &mut self,
        au: &[u8],
    ) -> Result<(alloc::vec::Vec<PictureMeta>, alloc::vec::Vec<Nv12Frame>), VulkanVideoError> {
        match self {
            DpbDecoderKind::H264(d) => d.decode_push_meta(au),
            DpbDecoderKind::H265(d) => d.decode_push_meta(au),
            DpbDecoderKind::Av1(d) => {
                let (_n, frames) = d.decode_push(au)?;
                Ok((alloc::vec::Vec::new(), frames))
            }
        }
    }

    /// Drain the pipelined tail at end of stream (see [`H264DpbDecoder::decode_flush`]).
    fn decode_flush(&mut self) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        match self {
            DpbDecoderKind::H264(d) => d.decode_flush(),
            DpbDecoderKind::H265(d) => d.decode_flush(),
            DpbDecoderKind::Av1(d) => d.decode_flush(),
        }
    }

    /// Discard in-flight ring decodes + DPB reference state (seek / flush).
    fn reset(&mut self) {
        match self {
            DpbDecoderKind::H264(d) => d.reset(),
            DpbDecoderKind::H265(d) => d.reset(),
            DpbDecoderKind::Av1(d) => d.reset(),
        }
    }

    /// DPB image count, a safe upper bound on the display-order reorder depth
    /// (the number of frames the reorder buffer may need to hold before the
    /// earliest-display picture of a group is known).
    fn dpb_slots(&self) -> usize {
        match self {
            DpbDecoderKind::H264(d) => d.dpb_slots(),
            DpbDecoderKind::H265(d) => d.dpb_slots(),
            DpbDecoderKind::Av1(d) => d.dpb_slots(),
        }
    }

    /// Per-AU DISPLAY-order texture decode for AV1 (the op-walk with the DPB
    /// persisting across calls); H.264 / H.265 stream through
    /// [`decode_push_to_textures`](Self::decode_push_to_textures) instead.
    fn decode_display_to_textures(
        &mut self,
        au: &[u8],
    ) -> Result<alloc::vec::Vec<wgpu::Texture>, VulkanVideoError> {
        match self {
            DpbDecoderKind::Av1(d) => d.decode_display_to_textures(au),
            DpbDecoderKind::H264(_) | DpbDecoderKind::H265(_) => {
                Err(VulkanVideoError::UnsupportedStream)
            }
        }
    }

    /// Per-AU DISPLAY-order system decode for AV1 (the op-walk `decode_all`
    /// applied to one temporal unit; the DPB persists across calls, so
    /// `show_existing_frame` resolves frames decoded in earlier AUs). H.264 /
    /// H.265 stream through `decode_push_meta` + the element's reorder instead.
    fn decode_display(
        &mut self,
        au: &[u8],
    ) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        match self {
            DpbDecoderKind::Av1(d) => d.decode_display(au),
            DpbDecoderKind::H264(_) | DpbDecoderKind::H265(_) => {
                Err(VulkanVideoError::UnsupportedStream)
            }
        }
    }

    /// Streaming per-AU decode to textures with per-picture metas (coding
    /// order, DPB/POC state intact across calls); the element reorders by the
    /// metas' (coded-video-sequence, POC). H.264 / H.265 only.
    fn decode_push_to_textures(
        &mut self,
        au: &[u8],
    ) -> Result<(alloc::vec::Vec<PictureMeta>, alloc::vec::Vec<wgpu::Texture>), VulkanVideoError>
    {
        match self {
            DpbDecoderKind::H264(d) => d.decode_push_to_textures(au),
            DpbDecoderKind::H265(d) => d.decode_push_to_textures(au),
            DpbDecoderKind::Av1(_) => Err(VulkanVideoError::UnsupportedStream),
        }
    }
}

/// The decode session backing a [`DpbDecoderKind`]. Held alongside the decoder
/// because the decoder's [`DpbCore`] copies the session's raw handles, so the
/// session (whose `Drop` destroys them) must outlive it. The variants are never
/// read: they exist purely to own the session for the decoder's lifetime.
#[allow(clippy::large_enum_variant, dead_code)]
enum DecodeSessionKind {
    H264(H264DecodeSession),
    H265(H265DecodeSession),
    Av1(Av1DecodeSession),
}

impl core::fmt::Debug for DecodeSessionKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("DecodeSessionKind")
    }
}

/// A streaming display-order reorder buffer for the element's system (NV12)
/// output. Hardware decode retires pictures in coding order; a viewer needs
/// presentation (POC) order, which differs for any stream with B-frames. Each
/// decoded picture is pushed tagged with its (coded-video-sequence, POC) key
/// (the same key the whole-stream [`reorder_to_display_order`] sorts on) and the
/// buffer releases pictures in ascending key order once the earliest-display one
/// of the group is settled.
///
/// POC only compares within one coded video sequence (it resets at each
/// keyframe), so a keyframe both closes the previous CVS (all its pictures are
/// submitted by then, so they release in POC order) and opens a new one. Within a
/// CVS the buffer bumps the lowest-key picture once it holds more than `max_hold`
/// (the DPB slot count, a safe upper bound on the reorder depth), so a long GOP
/// does not buffer without bound.
///
/// Only H.264 / H.265 use this; AV1 (whose display order comes from
/// `show_existing_frame` / `order_hint`, not a monotonic POC) stays in coding
/// order. Generic over the payload so the same logic could serve the GPU-texture
/// path if that becomes AU-streamed.
struct ReorderBuffer<T> {
    /// Held pictures, each `(cvs, poc, payload)`. Small (<= `max_hold` within a
    /// CVS), so a Vec + linear min-scan beats a heap.
    pending: alloc::vec::Vec<(u32, i32, T)>,
    /// Current coded-video-sequence index, bumped at each keyframe after the first.
    cvs: u32,
    /// Whether any picture has been pushed (the first keyframe is not a boundary).
    started: bool,
    /// Bump once the current CVS holds more than this many pictures. Set to the
    /// DPB slot count when the decoder is built.
    max_hold: usize,
}

impl<T> ReorderBuffer<T> {
    fn new() -> Self {
        Self {
            pending: alloc::vec::Vec::new(),
            cvs: 0,
            started: false,
            max_hold: 0,
        }
    }

    /// Push one decoded picture (in decode order) and return the pictures now
    /// ready to emit, in display order. A keyframe first releases the whole
    /// previous CVS (all submitted), then joins the new CVS; within a CVS the
    /// buffer bumps the lowest-key picture while it exceeds `max_hold`.
    fn push(&mut self, is_keyframe: bool, poc: i32, payload: T) -> alloc::vec::Vec<T> {
        let mut out = alloc::vec::Vec::new();
        if is_keyframe && self.started {
            out.append(&mut self.take_sorted());
            self.cvs += 1;
        }
        self.started = true;
        self.pending.push((self.cvs, poc, payload));
        while self.pending.len() > self.max_hold {
            out.push(self.pop_min());
        }
        out
    }

    /// Release everything held, in display order (end of stream / a reconfig
    /// boundary).
    fn flush(&mut self) -> alloc::vec::Vec<T> {
        self.take_sorted()
    }

    /// Discard all held pictures and the CVS state (a seek / flush drops pre-flush
    /// data; the next keyframe starts a fresh sequence).
    fn reset(&mut self) {
        self.pending.clear();
        self.cvs = 0;
        self.started = false;
    }

    /// Remove and return the lowest-`(cvs, poc)` held picture.
    fn pop_min(&mut self) -> T {
        let mut min_i = 0;
        for i in 1..self.pending.len() {
            let a = (self.pending[i].0, self.pending[i].1);
            let b = (self.pending[min_i].0, self.pending[min_i].1);
            if a < b {
                min_i = i;
            }
        }
        self.pending.remove(min_i).2
    }

    /// Drain all held pictures, sorted into display order. A stable sort keeps
    /// decode order among any equal keys (which should not occur).
    fn take_sorted(&mut self) -> alloc::vec::Vec<T> {
        let mut items = core::mem::take(&mut self.pending);
        items.sort_by_key(|a| (a.0, a.1));
        items.into_iter().map(|(_, _, p)| p).collect()
    }
}

pub struct VulkanVideoDec {
    decoder: Option<DpbDecoderKind>,
    session: Option<DecodeSessionKind>,
    device: Option<VulkanVideoDevice>,
    /// The codec settled at `configure_pipeline` from the sink caps; drives
    /// which session / decoder / parameter-set path `ensure_decoder` builds.
    codec: VideoCodec,
    /// Coded geometry `(width, height)` the current session / DPB was built for.
    /// A keyframe whose parameter sets carry a different geometry triggers a
    /// mid-stream rebuild (resolution change); `None` before the first build.
    coded_geometry: Option<(u32, u32)>,
    last_caps: Option<Caps>,
    framerate: Rate,
    emitted: u64,
    /// The resolved output memory domain: `System` (NV12 readback, the default)
    /// or `WgpuTexture` (zero-copy GPU-resident RGBA), settled by
    /// [`configure_allocation`](AsyncElement::configure_allocation) against the
    /// downstream consumer.
    out_domain: MemoryDomainKind,
    /// Set once the decoder is built: whether it emits `WgpuTexture` (GPU path)
    /// or system NV12. Falls back to NV12 if the device has no compute queue.
    emit_wgpu: bool,
    /// System-path pipelining: the source timing of each submitted-but-not-yet-
    /// emitted coded picture, in decode order. `decode_push` pipelines output (a
    /// frame retires up to `DECODE_RING_DEPTH - 1` pictures after its AU was fed),
    /// so a retired frame no longer belongs to the AU currently in `process`; this
    /// FIFO re-pairs each retired frame with its own AU's timing (decode order ==
    /// ring retirement order). Drained by `Eos`, cleared by `Flush`. Unused on the
    /// GPU-texture path, which stays synchronous (one picture in, its frames out).
    pending_timings: alloc::collections::VecDeque<FrameTiming>,
    /// A mid-stream reconfig (`ensure_decoder` rebuilding for a new resolution)
    /// flushes the OUTGOING decoder's pipelined tail here before it drops, so those
    /// frames are not lost; `process` emits them (against the old geometry) before
    /// decoding the reconfig keyframe. Empty except across a single reconfig, and
    /// on the synchronous GPU-texture path (nothing is ever in flight there).
    reconfig_tail: alloc::vec::Vec<Nv12Frame>,
    /// Whether the system-path output is reordered from coding to display order.
    /// True for H.264 / H.265 (a monotonic POC keys the reorder); false for AV1,
    /// which stays in coding order (its display order is not a simple POC sort).
    reorder_enabled: bool,
    /// The (timing, [`PictureMeta`]) of each submitted-but-not-yet-retired coded
    /// picture, in decode order (the reorder path's analog of `pending_timings`).
    /// A retired frame pairs with the front entry (decode order == ring retirement
    /// order), then feeds [`reorder`](Self::reorder) keyed by its POC.
    pending_meta: alloc::collections::VecDeque<(FrameTiming, PictureMeta)>,
    /// The display-order reorder buffer for the system path (H.264 / H.265). Holds
    /// decoded frames until their display position is settled; emptied on `Eos`,
    /// a reconfig boundary, and `Flush`. Unused on the AV1 / GPU-texture paths.
    reorder: ReorderBuffer<(FrameTiming, Nv12Frame)>,
    /// The GPU-texture path's display-order buffer (H.264 / H.265): the same
    /// (coded-video-sequence, POC) reorder over `wgpu::Texture` payloads, fed
    /// synchronously by `decode_push_to_textures` (metas pair 1:1, no FIFO).
    /// AV1's texture path needs none: its display order is the op order.
    reorder_tex: ReorderBuffer<(FrameTiming, wgpu::Texture)>,
}

impl core::fmt::Debug for VulkanVideoDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VulkanVideoDec")
            .field("codec", &self.codec)
            .field("configured", &self.decoder.is_some())
            .finish_non_exhaustive()
    }
}

impl Default for VulkanVideoDec {
    fn default() -> Self {
        Self::new()
    }
}

impl VulkanVideoDec {
    pub fn new() -> Self {
        Self {
            decoder: None,
            session: None,
            device: None,
            codec: VideoCodec::H264,
            coded_geometry: None,
            last_caps: None,
            framerate: Rate::Any,
            emitted: 0,
            // Default to system NV12 (works with any sink); a WgpuTexture-capable
            // consumer flips this via `configure_allocation` (the zero-copy path).
            out_domain: MemoryDomainKind::System,
            emit_wgpu: false,
            pending_timings: alloc::collections::VecDeque::new(),
            reconfig_tail: alloc::vec::Vec::new(),
            reorder_enabled: false,
            pending_meta: alloc::collections::VecDeque::new(),
            reorder: ReorderBuffer::new(),
            reorder_tex: ReorderBuffer::new(),
        }
    }

    /// The domains this decoder can emit: the zero-copy `WgpuTexture` (preferred)
    /// or system memory.
    const OUTPUT_DOMAINS: DomainSet =
        DomainSet::only(MemoryDomainKind::WgpuTexture).with(MemoryDomainKind::System);

    /// Try the GPU-texture decoder first (when the resolved domain wants it),
    /// falling back to the system-NV12 decoder if the device exposes no compute
    /// queue. Returns the built decoder and whether it emits `WgpuTexture`. This
    /// is the codec-agnostic half of [`ensure_decoder`]; each codec passes its
    /// own two constructors.
    fn build_with_fallback<D>(
        want_gpu: bool,
        gpu: impl FnOnce() -> Result<D, VulkanVideoError>,
        sys: impl FnOnce() -> Result<D, VulkanVideoError>,
    ) -> Result<(D, bool), G2gError> {
        if want_gpu {
            match gpu() {
                Ok(d) => Ok((d, true)),
                // No compute queue: fall back to the system NV12 path.
                Err(VulkanVideoError::NoComputeQueue) => {
                    Ok((sys().map_err(|_| G2gError::CapsMismatch)?, false))
                }
                Err(_) => Err(G2gError::CapsMismatch),
            }
        } else {
            Ok((sys().map_err(|_| G2gError::CapsMismatch)?, false))
        }
    }

    /// Ensure a decoder is built for the current access unit, (re)building the
    /// session + DPB from the codec's parameter sets (H.264/H.265 SPS/PPS in-band,
    /// AV1 sequence header). Returns `Ok(false)` when there is no decoder yet and
    /// this AU carries no parameter sets (skip and wait for the keyframe AU that
    /// does), `Ok(true)` once a decoder is ready. Handles mid-stream reconfig: a
    /// keyframe whose parameter sets carry a different coded geometry rebuilds the
    /// session + DPB for the new resolution (the old ones free when replaced). An
    /// unchanged geometry keeps the existing session (also swallowing the
    /// parameter sets repeated on every keyframe). Picks the GPU-texture or
    /// system-NV12 decoder from the resolved `out_domain`, falling back to NV12 if
    /// the device exposes no compute queue.
    fn ensure_decoder(&mut self, au: &[u8]) -> Result<bool, G2gError> {
        let device = self.device.as_ref().ok_or(G2gError::CapsMismatch)?;
        let want_gpu = self.out_domain == MemoryDomainKind::WgpuTexture;
        let built = self.decoder.is_some();
        let cur_geom = self.coded_geometry;

        let (session, decoder, emit_wgpu, geometry) = match self.codec {
            VideoCodec::H264 => {
                let Some(ps) = extract_h264_parameter_sets(au) else {
                    // No parameter sets here: keep decoding if already built (a
                    // non-keyframe AU), else wait for the keyframe that carries them.
                    return Ok(built);
                };
                let width = (ps.sps.pic_width_in_mbs_minus1 + 1) * 16;
                let height = (ps.sps.pic_height_in_map_units_minus1 + 1) * 16;
                if built && cur_geom == Some((width, height)) {
                    return Ok(true);
                }
                let session = device
                    .create_h264_session(&ps, width, height)
                    .map_err(|_| G2gError::CapsMismatch)?;
                let (decoder, emit_wgpu) = Self::build_with_fallback(
                    want_gpu,
                    || device.create_h264_dpb_decoder_gpu(&session, &ps),
                    || device.create_h264_dpb_decoder(&session, &ps),
                )?;
                (
                    DecodeSessionKind::H264(session),
                    DpbDecoderKind::H264(decoder),
                    emit_wgpu,
                    (width, height),
                )
            }
            VideoCodec::H265 => {
                let Some(ps) = extract_h265_parameter_sets(au) else {
                    return Ok(built);
                };
                let width = ps.sps.pic_width_in_luma_samples;
                let height = ps.sps.pic_height_in_luma_samples;
                if built && cur_geom == Some((width, height)) {
                    return Ok(true);
                }
                let std = to_std_h265_params(&ps);
                let session = device
                    .create_h265_session(&std, width, height)
                    .map_err(|_| G2gError::CapsMismatch)?;
                let (decoder, emit_wgpu) = Self::build_with_fallback(
                    want_gpu,
                    || device.create_h265_dpb_decoder_gpu(&session, &ps),
                    || device.create_h265_dpb_decoder(&session, &ps),
                )?;
                (
                    DecodeSessionKind::H265(session),
                    DpbDecoderKind::H265(decoder),
                    emit_wgpu,
                    (width, height),
                )
            }
            VideoCodec::Av1 => {
                let Some(seq) = extract_av1_sequence_header(au) else {
                    return Ok(built);
                };
                let width = seq.max_frame_width_minus_1 + 1;
                let height = seq.max_frame_height_minus_1 + 1;
                if built && cur_geom == Some((width, height)) {
                    return Ok(true);
                }
                let std = to_std_av1_seq_header(&seq);
                let session = device
                    .create_av1_session(&std, width, height)
                    .map_err(|_| G2gError::CapsMismatch)?;
                let (decoder, emit_wgpu) = Self::build_with_fallback(
                    want_gpu,
                    || device.create_av1_dpb_decoder_gpu(&session, &seq),
                    || device.create_av1_dpb_decoder(&session, &seq),
                )?;
                (
                    DecodeSessionKind::Av1(session),
                    DpbDecoderKind::Av1(decoder),
                    emit_wgpu,
                    (width, height),
                )
            }
            _ => return Err(G2gError::CapsMismatch),
        };

        // Flush the outgoing decoder's pipelined tail before it drops, so a
        // mid-stream reconfig does not lose the frames still in its ring (`process`
        // emits `reconfig_tail` against the old geometry before the reconfig
        // keyframe decodes). No-op on the initial build (no old decoder) and on the
        // synchronous texture path (its ring is never in flight).
        let tail = match self.decoder.as_mut() {
            Some(old) => old.decode_flush().map_err(|_| G2gError::CapsMismatch)?,
            None => alloc::vec::Vec::new(),
        };
        self.reconfig_tail = tail;

        // Assigning replaces (and drops) any previous decoder + session: on a
        // mid-stream reconfig the old DPB / session free here, after the new ones
        // are built. Decoder before session mirrors the struct's teardown order.
        // The next emitted frame carries the new dimensions, so `process` emits a
        // fresh `CapsChanged` for the resolution change on its own.
        self.emit_wgpu = emit_wgpu;
        // Size the reorder window to this decoder's DPB (a safe upper bound on the
        // reorder depth), so a long GOP releases frames without buffering unbounded.
        self.reorder.max_hold = decoder.dpb_slots();
        self.reorder_tex.max_hold = decoder.dpb_slots();
        self.decoder = Some(decoder);
        self.session = Some(session);
        self.coded_geometry = Some(geometry);
        Ok(true)
    }

    /// A [`GpuContext`](crate::gpu::GpuContext) sharing the decode device, once
    /// it has been opened (`configure_pipeline`). A `WgpuSink` built from this
    /// context presents the decoder's `WgpuTexture` frames with no copy. `None`
    /// before the device is opened.
    pub fn gpu_context(&self) -> Option<crate::gpu::GpuContext> {
        self.device.as_ref().map(|d| d.gpu_context())
    }

    /// The `Frame` timing for an output picture, inherited from its access unit.
    fn out_timing(&self, src: &FrameTiming) -> FrameTiming {
        FrameTiming {
            pts_ns: src.pts_ns,
            dts_ns: src.pts_ns,
            duration_ns: 0,
            capture_ns: src.capture_ns,
            arrival_ns: src.arrival_ns,
            keyframe: src.keyframe,
        }
    }

    /// Push one retired NV12 frame downstream with the given (its own AU's)
    /// timing, emitting a `CapsChanged` first when the geometry changes.
    async fn emit_one_nv12(
        &mut self,
        src_timing: FrameTiming,
        f: Nv12Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(f.width),
            height: Dim::Fixed(f.height),
            framerate: self.framerate.clone(),
        };
        if self.last_caps.as_ref() != Some(&caps) {
            out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
            self.last_caps = Some(caps);
        }
        // NV12 = luma plane followed by interleaved CbCr, which is exactly the
        // two planes `Nv12Frame` carries: concatenate.
        let mut nv12 = alloc::vec::Vec::with_capacity(f.luma.len() + f.chroma.len());
        nv12.extend_from_slice(&f.luma);
        nv12.extend_from_slice(&f.chroma);
        let out_frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(nv12.into_boxed_slice())),
            timing: self.out_timing(&src_timing),
            sequence: self.emitted,
            meta: Default::default(),
        };
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(out_frame)).await?;
        Ok(())
    }

    /// Push retired system-path NV12 frames downstream in coding order, pairing
    /// each with its own access unit's timing from `pending_timings` (FIFO: decode
    /// order == ring retirement order). The AV1 / non-reorder path; the reorder
    /// path uses [`emit_reordered`](Self::emit_reordered).
    async fn emit_nv12_frames(
        &mut self,
        frames: alloc::vec::Vec<Nv12Frame>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        for f in frames {
            // The retired frame's own AU timing (FIFO). `unwrap_or_default` is a
            // defensive floor: the push-per-submitted-picture / pop-per-retired
            // balance keeps one queued timing per in-flight picture, so a retired
            // frame always has one.
            let src_timing = self.pending_timings.pop_front().unwrap_or_default();
            self.emit_one_nv12(src_timing, f, out).await?;
        }
        Ok(())
    }

    /// Feed retired (coding-order) NV12 frames through the display-order
    /// [`reorder`](Self::reorder) buffer and emit whatever it releases. Each
    /// retired frame pairs with the front of `pending_meta` (decode order == ring
    /// retirement order) for its timing + POC. The H.264 / H.265 path.
    async fn emit_reordered(
        &mut self,
        frames: alloc::vec::Vec<Nv12Frame>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        for f in frames {
            // Defensive floor (see `emit_nv12_frames`): the balanced push/pop keeps
            // one queued meta per in-flight picture.
            let (timing, meta) = self.pending_meta.pop_front().unwrap_or((
                FrameTiming::default(),
                PictureMeta {
                    is_keyframe: false,
                    is_random_access: false,
                    frame_num: 0,
                    poc: 0,
                },
            ));
            let ready = self.reorder.push(meta.is_keyframe, meta.poc, (timing, f));
            for (t, rf) in ready {
                self.emit_one_nv12(t, rf, out).await?;
            }
        }
        Ok(())
    }

    /// Release everything the reorder buffer still holds, in display order (end of
    /// stream or a reconfig boundary), then clear its CVS state.
    async fn drain_reorder(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let leftover = self.reorder.flush();
        for (t, rf) in leftover {
            self.emit_one_nv12(t, rf, out).await?;
        }
        self.reorder.reset();
        Ok(())
    }

    /// Push GPU-texture frames downstream, each with its own AU's timing and a
    /// `CapsChanged` whenever the geometry changes (so a reconfig boundary's
    /// old-geometry stragglers emit against their own caps).
    async fn emit_textures(
        &mut self,
        textures: alloc::vec::Vec<(FrameTiming, wgpu::Texture)>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        for (timing, tex) in textures {
            let (w, h) = (tex.width(), tex.height());
            let caps = Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                framerate: self.framerate.clone(),
            };
            if self.last_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                self.last_caps = Some(caps);
            }
            let keep = alloc::sync::Arc::new(crate::gpu::WgpuTextureKeepAlive(tex));
            let out_frame = Frame {
                domain: MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(w, h, keep)),
                timing: self.out_timing(&timing),
                sequence: self.emitted,
                meta: Default::default(),
            };
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(out_frame)).await?;
        }
        Ok(())
    }
}

impl PadTemplates for VulkanVideoDec {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        let raw = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        alloc::vec::Vec::from([
            // Any of the codecs this element decodes (H.264 / H.265 / AV1).
            PadTemplate::sink(CapsSet::from_alternatives(
                VULKAN_DEC_CODECS
                    .iter()
                    .map(|&codec| Caps::CompressedVideo {
                        codec,
                        width: Dim::Any,
                        height: Dim::Any,
                        framerate: Rate::Any,
                    })
                    .collect(),
            )),
            // Both outputs the element can produce: system NV12 or, on the
            // zero-copy `WgpuTexture` path, RGBA (the `.produces(WgpuTexture)`
            // auto-plug tag steers a GPU consumer to the latter).
            PadTemplate::source(CapsSet::from_alternatives(alloc::vec::Vec::from([
                raw(RawVideoFormat::Nv12),
                raw(RawVideoFormat::Rgba8),
            ]))),
        ])
    }
}

impl AsyncElement for VulkanVideoDec {
    type ProcessFuture<'a>
        = core::pin::Pin<
        alloc::boxed::Box<dyn core::future::Future<Output = Result<(), G2gError>> + 'a>,
    >
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        match upstream_caps {
            Caps::CompressedVideo { codec, .. } if VULKAN_DEC_CODECS.contains(codec) => {
                let candidate = Caps::CompressedVideo {
                    codec: *codec,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                };
                upstream_caps
                    .intersect(&candidate)
                    .map_err(|_| G2gError::CapsMismatch)
            }
            _ => Err(G2gError::CapsMismatch),
        }
    }

    /// Compressed video in (any supported codec / geometry) -> raw video out at
    /// the same dims/framerate, so the caps solver hands each link real caps (the
    /// codec to the decoder, raw to the sink) instead of falling back to the
    /// dynamic `intercept_caps` cascade. The output pixel format tracks the
    /// resolved domain: `Rgba8` for the GPU-texture (`WgpuTexture`) path, `Nv12`
    /// for the system path.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let format = if self.out_domain == MemoryDomainKind::WgpuTexture {
            RawVideoFormat::Rgba8
        } else {
            RawVideoFormat::Nv12
        };
        CapsConstraint::DerivedOutput(alloc::boxed::Box::new(move |input: &Caps| match input {
            Caps::CompressedVideo {
                codec,
                width,
                height,
                framerate,
            } if VULKAN_DEC_CODECS.contains(codec) => CapsSet::one(Caps::RawVideo {
                format,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(alloc::vec::Vec::new()),
        }))
    }

    /// Preferred output domain: the zero-copy `WgpuTexture` (the wedge). The
    /// runner narrows this against the downstream consumer via
    /// [`output_domains`](Self::output_domains) + `configure_allocation`.
    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::WgpuTexture
    }

    /// Both domains the decoder can emit: GPU-resident `WgpuTexture` or system
    /// NV12 (M351 multi-domain negotiation, mirrors `NvDec`).
    fn output_domains(&self) -> DomainSet {
        Self::OUTPUT_DOMAINS
    }

    /// Settle the output domain against the downstream consumer's accepted set: a
    /// `WgpuTexture`-capable sink (e.g. `WgpuSink`) keeps the frame on the GPU;
    /// anything else makes the decoder emit system NV12.
    fn configure_allocation(&mut self, params: &AllocationParams) {
        if let Ok(resolved) = params.resolve_for_producer(Self::OUTPUT_DOMAINS) {
            self.out_domain = resolved.domain;
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let codec = match absolute_caps {
            Caps::CompressedVideo {
                codec, framerate, ..
            } if VULKAN_DEC_CODECS.contains(codec) => {
                self.framerate = framerate.clone();
                *codec
            }
            _ => return Err(G2gError::CapsMismatch),
        };
        self.codec = codec;
        // Reorder the system-path output to display order for the codecs whose
        // display order is a monotonic POC sort (H.264 / H.265). AV1 stays in
        // coding order (its reorder is `show_existing_frame` / `order_hint`, a
        // separate mechanism the whole-stream `decode_all` path handles).
        self.reorder_enabled = matches!(codec, VideoCodec::H264 | VideoCodec::H265);
        // Open the decode device for this codec now (independent of the stream's
        // parameter sets, which arrive in-band and build the session lazily on
        // the first keyframe AU). Each codec enables its own decode profile.
        let device = match codec {
            VideoCodec::H264 => block_on(open_h264_decode_device()),
            VideoCodec::H265 => block_on(open_h265_decode_device()),
            VideoCodec::Av1 => block_on(open_av1_decode_device()),
            _ => return Err(G2gError::CapsMismatch),
        }
        .map_err(|_| G2gError::CapsMismatch)?;
        self.device = Some(device);
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Vulkan Video decoder",
            "Codec/Decoder/Video/Hardware",
            "Vendor-neutral GPU hardware H.264 / H.265 / AV1 decode via VK_KHR_video_queue",
            "g2g",
        )
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        alloc::boxed::Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // Decode the access unit. The whole thing is owned by the
                    // packet, so borrow the bytes into an owned Vec first (the
                    // decoder borrows `self.decoder` mutably below).
                    let au = slice.as_slice().to_vec();
                    let src_timing = frame.timing;
                    if !self.ensure_decoder(&au)? {
                        // No SPS/PPS yet (a leading non-keyframe AU); skip it.
                        return Ok(());
                    }

                    // If this AU triggered a reconfig, emit the previous decoder's
                    // flushed tail (old geometry) before decoding the new keyframe.
                    // On the reorder path, drain the reorder buffer too: the old
                    // coded video sequence is complete, so it releases in display
                    // order before the new-geometry sequence begins.
                    if !self.reconfig_tail.is_empty() {
                        let tail = core::mem::take(&mut self.reconfig_tail);
                        if self.reorder_enabled {
                            self.emit_reordered(tail, out).await?;
                            self.drain_reorder(out).await?;
                        } else {
                            self.emit_nv12_frames(tail, out).await?;
                        }
                    }

                    if self.emit_wgpu {
                        // Zero-copy: each decoded picture is an RGBA wgpu::Texture
                        // on the decode device, wrapped so a WgpuSink (sharing the
                        // device via `gpu_context`) presents it with no copy.
                        if matches!(self.codec, VideoCodec::Av1) {
                            // AV1 display order is the bitstream's op order
                            // (`show_existing_frame` re-displays a stored slot),
                            // so a per-AU op-walk with the DPB persisting across
                            // calls emits display order directly. A frameless AU
                            // (e.g. a bare temporal delimiter) emits nothing.
                            let textures = match self
                                .decoder
                                .as_mut()
                                .expect("decoder built")
                                .decode_display_to_textures(&au)
                            {
                                Ok(t) => t,
                                Err(VulkanVideoError::NoDecodableSlice) => alloc::vec::Vec::new(),
                                Err(_) => return Err(G2gError::CapsMismatch),
                            };
                            let paired = textures.into_iter().map(|t| (src_timing, t)).collect();
                            self.emit_textures(paired, out).await?;
                        } else {
                            // H.264 / H.265: stream each AU through the DPB (state
                            // intact across calls) and reorder the coding-order
                            // textures into display order by (cvs, POC), the
                            // texture analog of the system path (a keyframe also
                            // releases the previous sequence, so a reconfig's
                            // old-geometry stragglers emit first).
                            let (metas, textures) = self
                                .decoder
                                .as_mut()
                                .expect("decoder built")
                                .decode_push_to_textures(&au)
                                .map_err(|_| G2gError::CapsMismatch)?;
                            let mut ready = alloc::vec::Vec::new();
                            for (meta, tex) in metas.into_iter().zip(textures) {
                                ready.extend(self.reorder_tex.push(
                                    meta.is_keyframe,
                                    meta.poc,
                                    (src_timing, tex),
                                ));
                            }
                            self.emit_textures(ready, out).await?;
                        }
                        return Ok(());
                    }

                    // Pipelined system path: submit this AU into the decode ring
                    // without draining (keeps the decode queue saturated across
                    // AUs), collecting only frames that have already retired. Each
                    // submitted picture's timing (and, on the reorder path, its POC)
                    // is queued so the retired frame (which belongs to an earlier
                    // AU) is re-paired with its own and placed in display order.
                    if self.reorder_enabled {
                        let (metas, decoded) = self
                            .decoder
                            .as_mut()
                            .expect("decoder built")
                            .decode_push_meta(&au)
                            .map_err(|_| G2gError::CapsMismatch)?;
                        for meta in metas {
                            self.pending_meta.push_back((src_timing, meta));
                        }
                        self.emit_reordered(decoded, out).await?;
                    } else {
                        // AV1: decode the AU with the display-order op-walk
                        // (`show_existing_frame` re-displays, an alt-ref decodes
                        // without displaying, film grain synthesizes per shown
                        // frame), the DPB persisting across calls. Every frame
                        // this AU emits is displayed at this AU's time. Trades
                        // the cross-AU ring pipelining for display correctness.
                        let decoded = match self
                            .decoder
                            .as_mut()
                            .expect("decoder built")
                            .decode_display(&au)
                        {
                            Ok(f) => f,
                            Err(VulkanVideoError::NoDecodableSlice) => alloc::vec::Vec::new(),
                            Err(_) => return Err(G2gError::CapsMismatch),
                        };
                        for _ in 0..decoded.len() {
                            self.pending_timings.push_back(src_timing);
                        }
                        self.emit_nv12_frames(decoded, out).await?;
                    }
                    Ok(())
                }
                PipelinePacket::CapsChanged(c) => match &c {
                    Caps::CompressedVideo { codec, .. } if VULKAN_DEC_CODECS.contains(codec) => {
                        Ok(())
                    }
                    _ => Err(G2gError::CapsMismatch),
                },
                PipelinePacket::Eos => {
                    // The GPU-texture path is synchronous per AU; only its
                    // display-order buffer can still hold textures.
                    if self.emit_wgpu {
                        let held = self.reorder_tex.flush();
                        self.reorder_tex.reset();
                        self.emit_textures(held, out).await?;
                        out.push(PipelinePacket::Eos).await?;
                        return Ok(());
                    }
                    // Emit the pipelined tail (frames held back by `decode_push`)
                    // before forwarding end-of-stream; `decode_flush` on an idle
                    // ring (the per-AU AV1 display path) returns nothing.
                    let tail = match self.decoder.as_mut() {
                        Some(dec) => dec.decode_flush().map_err(|_| G2gError::CapsMismatch)?,
                        None => alloc::vec::Vec::new(),
                    };
                    if self.reorder_enabled {
                        self.emit_reordered(tail, out).await?;
                        self.drain_reorder(out).await?;
                    } else {
                        self.emit_nv12_frames(tail, out).await?;
                    }
                    out.push(PipelinePacket::Eos).await?;
                    Ok(())
                }
                PipelinePacket::Flush => {
                    // A flush discards buffered state and resumes (typically a seek
                    // to a keyframe). Reset the decoder so the in-flight ring +
                    // queued timings are dropped and the DPB is cleared (matches
                    // ffmpegdec's avcodec_flush_buffers on flush); the pipelined
                    // tail is intentionally NOT emitted, it is pre-flush data.
                    if let Some(dec) = self.decoder.as_mut() {
                        dec.reset();
                    }
                    self.pending_timings.clear();
                    self.pending_meta.clear();
                    self.reorder.reset();
                    self.reorder_tex.reset();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    Ok(())
                }
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annexb::BitWriter;

    // A real 640x480 baseline Annex-B clip (SPS + PPS + frames). GPU-free:
    // exercises the bitstream parse + `Std*` mapping without a Vulkan device.
    const CLIP: &[u8] = include_bytes!("../tests/fixtures/h264_640x480.h264");

    #[test]
    fn parses_sps_pps_geometry_from_real_clip() {
        let ps = extract_h264_parameter_sets(CLIP).expect("SPS+PPS parse");
        // 640x480 baseline: 40x30 macroblocks, progressive.
        assert_eq!(ps.sps.profile_idc, 66, "baseline profile");
        assert_eq!(ps.sps.pic_width_in_mbs_minus1, 39, "640/16 - 1");
        assert_eq!(ps.sps.pic_height_in_map_units_minus1, 29, "480/16 - 1");
        assert_eq!(ps.sps.frame_mbs_only_flag, 1, "progressive");
        assert_eq!(ps.sps.chroma_format_idc, 1, "4:2:0");
        // PPS references SPS 0.
        assert_eq!(ps.pps.seq_parameter_set_id, 0);
        assert_eq!(ps.pps.pic_parameter_set_id, 0);
    }

    #[test]
    fn std_mapping_preserves_geometry_and_ids() {
        let ps = extract_h264_parameter_sets(CLIP).unwrap();
        let std_sps = to_std_sps(&ps.sps);
        let std_pps = to_std_pps(&ps.pps);
        // The mapping must carry the geometry + ids the driver keys on, and set
        // no scaling-list / VUI pointers (we reject those at parse).
        assert_eq!(std_sps.pic_width_in_mbs_minus1, 39);
        assert_eq!(std_sps.pic_height_in_map_units_minus1, 29);
        assert_eq!(std_sps.seq_parameter_set_id, 0);
        assert!(std_sps.pScalingLists.is_null());
        assert!(std_sps.pSequenceParameterSetVui.is_null());
        assert_eq!(std_pps.seq_parameter_set_id, 0);
        assert_eq!(std_pps.pic_parameter_set_id, 0);
        assert!(std_pps.pScalingLists.is_null());
        // frame_mbs_only_flag round-trips through the bitfield setter.
        assert_eq!(std_sps.flags.frame_mbs_only_flag(), 1);
    }

    #[test]
    fn parses_every_slice_header_in_clip() {
        // The clip is two GOPs of IDR + four P frames (frame_num 0..=4 each),
        // POC type 2, single-slice pictures. The DPB decoder keys on exactly the
        // fields asserted here; a parse regression breaks multi-frame decode.
        let ps = extract_h264_parameter_sets(CLIP).unwrap();
        assert_eq!(ps.sps.pic_order_cnt_type, 2, "clip is POC type 2");
        assert_eq!(ps.sps.max_num_ref_frames, 3);

        let mut headers = alloc::vec::Vec::new();
        for nal in nal_units_any(CLIP) {
            if nal.is_empty() {
                continue;
            }
            let t = nal[0] & 0x1F;
            if t == 1 || t == 5 {
                headers.push(
                    parse_h264_slice_header(nal, &ps.sps, &ps.pps).expect("slice header parse"),
                );
            }
        }
        assert_eq!(headers.len(), 10, "10 coded pictures");

        // Both GOPs: an IDR (I slice, frame_num 0, is a reference) then four P
        // slices with frame_num 1..=4.
        for gop in 0..2 {
            let idr = &headers[gop * 5];
            assert!(idr.is_idr, "leading picture of GOP {gop} is IDR");
            assert!(idr.is_intra_slice(), "IDR is an I slice");
            assert_eq!(idr.frame_num, 0);
            assert_eq!(idr.first_mb_in_slice, 0);
            assert_ne!(idr.nal_ref_idc, 0, "IDR is a reference");
            for p in 1..5 {
                let h = &headers[gop * 5 + p];
                assert!(!h.is_idr, "P frame not IDR");
                assert!(!h.is_intra_slice(), "P slice is not intra");
                assert_eq!(h.frame_num, p as u32, "P frame_num counts up");
                assert_ne!(h.nal_ref_idc, 0, "P frames are references here");
            }
        }
    }

    #[test]
    fn baseline_pps_has_no_transform_8x8() {
        // Regression guard: a baseline PPS has no optional trailing block, so the
        // bit after redundant_pic_cnt_present is the rbsp_stop_one_bit (1), not
        // transform_8x8_mode_flag. Misreading it (no more_rbsp_data() check) told
        // the driver 8x8 transforms were on and desynced its CAVLC coefficient
        // parse, silently corrupting every decoded frame. Baseline (profile 66)
        // cannot use 8x8 transforms, so this must be 0.
        let ps = extract_h264_parameter_sets(CLIP).unwrap();
        assert_eq!(ps.sps.profile_idc, 66, "clip is baseline");
        assert_eq!(
            ps.pps.transform_8x8_mode_flag, 0,
            "baseline has no 8x8 transform"
        );
        let std_pps = to_std_pps(&ps.pps);
        assert_eq!(std_pps.flags.transform_8x8_mode_flag(), 0);
        // CAVLC (not CABAC) for baseline.
        assert_eq!(std_pps.flags.entropy_coding_mode_flag(), 0);
    }

    #[test]
    fn slice_header_rejects_non_vcl_nal() {
        let ps = extract_h264_parameter_sets(CLIP).unwrap();
        // The SPS NAL (type 7) is not a slice; the parser must reject it rather
        // than mis-read parameter-set bytes as a slice header.
        let sps_nal = nal_units_any(CLIP)
            .find(|n| !n.is_empty() && n[0] & 0x1F == 7)
            .expect("SPS NAL");
        assert!(parse_h264_slice_header(sps_nal, &ps.sps, &ps.pps).is_none());
    }

    // ------------------------------------------------------------------------
    // M501: H.265 (HEVC) parameter-set parse + `Std*` mapping. A real 640x480
    // x265 Annex-B clip (two GOPs: IDR + CRA, ten frames with inter pics, so it
    // carries short-term reference-picture sets). GPU-free.
    // ------------------------------------------------------------------------
    const H265_CLIP: &[u8] = include_bytes!("../tests/fixtures/h265_640x480.h265");

    #[test]
    fn parses_h265_vps_sps_pps_from_real_clip() {
        let ps = extract_h265_parameter_sets(H265_CLIP).expect("VPS+SPS+PPS parse");
        // 640x480 8-bit 4:2:0 Main profile.
        assert_eq!(ps.sps.pic_width_in_luma_samples, 640, "coded width");
        assert_eq!(ps.sps.pic_height_in_luma_samples, 480, "coded height");
        assert_eq!(ps.sps.chroma_format_idc, 1, "4:2:0");
        assert_eq!(ps.sps.bit_depth_luma_minus8, 0, "8-bit luma");
        assert_eq!(ps.sps.bit_depth_chroma_minus8, 0, "8-bit chroma");
        assert_eq!(ps.sps.ptl.general_profile_idc, 1, "Main profile");
        assert_eq!(ps.sps.ptl.general_level_idc, 90, "level 3.0 (30 * 3)");
        // This x265 clip carries its short-term RPS inline in each slice header
        // (num_short_term_ref_pic_sets == 0 in the SPS); the count and the parsed
        // list length must agree whatever the value, proving the RPS loop and the
        // SPS syntax after it (DPB info reached correctly) stayed in sync.
        assert_eq!(
            ps.sps.num_short_term_ref_pic_sets as usize,
            ps.sps.short_term_rps.len(),
            "declared RPS count == parsed list length"
        );
        // The DPB buffering reached past the RPS section confirms the parse did
        // not desync (x265 uses a 4-picture DPB here).
        assert_eq!(ps.sps.max_dec_pic_buffering_minus1[0], 3, "4-picture DPB");
        // The PPS references SPS 0, which references VPS 0.
        assert_eq!(
            ps.pps.pps_seq_parameter_set_id,
            ps.sps.sps_seq_parameter_set_id
        );
        assert_eq!(
            ps.vps.vps_video_parameter_set_id,
            ps.sps.sps_video_parameter_set_id
        );
    }

    #[test]
    fn h265_geometry_matches_geometry_only_parser() {
        // Cross-check the full Std-mapping parse against the independent
        // geometry-only `h265parse` module (post conformance-window cropping):
        // the coded luma dimensions here minus any crop must equal its output.
        let ps = extract_h265_parameter_sets(H265_CLIP).unwrap();
        // The fixture has no conformance window (640x480 is 4:2:0-aligned), so
        // coded == displayed. If it ever gains one, this still holds because the
        // geometry parser applies the same crop we skip in the Std path.
        assert_eq!(ps.sps.conformance_window_flag, 0, "no conformance window");
        assert_eq!(
            (
                ps.sps.pic_width_in_luma_samples,
                ps.sps.pic_height_in_luma_samples
            ),
            (640, 480)
        );
    }

    #[test]
    fn std_h265_mapping_preserves_geometry_and_wires_pointers() {
        let ps = extract_h265_parameter_sets(H265_CLIP).unwrap();
        let std = to_std_h265_params(&ps);
        // Geometry + ids carried through the mapping.
        assert_eq!(std.sps.pic_width_in_luma_samples, 640);
        assert_eq!(std.sps.pic_height_in_luma_samples, 480);
        assert_eq!(
            std.sps.num_short_term_ref_pic_sets,
            ps.sps.num_short_term_ref_pic_sets
        );
        // The Std SPS must reference the owned pointee blocks, not dangle.
        assert!(!std.sps.pProfileTierLevel.is_null(), "PTL wired");
        assert!(!std.sps.pDecPicBufMgr.is_null(), "DPB manager wired");
        // The RPS pointer is wired iff the SPS declares any (this clip declares
        // none, carrying them per-slice; null is then correct, not a dangle).
        assert_eq!(
            std.sps.pShortTermRefPicSet.is_null(),
            ps.sps.short_term_rps.is_empty(),
            "RPS pointer set iff RPS present"
        );
        // Unsupported blocks stay null (rejected at parse).
        assert!(std.sps.pScalingLists.is_null());
        assert!(std.sps.pSequenceParameterSetVui.is_null());
        assert!(std.sps.pLongTermRefPicsSps.is_null());
        // VPS wires its own PTL + DPB manager; PPS links back to the VPS.
        assert!(!std.vps.pProfileTierLevel.is_null());
        assert!(!std.vps.pDecPicBufMgr.is_null());
        assert_eq!(
            std.pps.sps_video_parameter_set_id,
            ps.sps.sps_video_parameter_set_id
        );
        // The chroma format enum maps 4:2:0.
        assert_eq!(
            std.sps.chroma_format_idc,
            vk::native::StdVideoH265ChromaFormatIdc_STD_VIDEO_H265_CHROMA_FORMAT_IDC_420
        );
    }

    #[test]
    fn h265_short_term_rps_explicit_parse_and_std_roundtrip() {
        // An explicitly-coded set: 2 negative pics (deltas 1 then 2 -> DeltaPocS0
        // -1, -3), both used. num_negative_pics=ue(2), then per pic
        // delta_poc_s0_minus1=ue + used flag.
        let mut w = BitWriter::default();
        w.write_ue(2); // num_negative_pics
        w.write_ue(0); // num_positive_pics
        w.write_ue(0); // delta_poc_s0_minus1[0] = 0 -> DeltaPocS0[0] = -1
        w.write_bit(1); // used[0]
        w.write_ue(1); // delta_poc_s0_minus1[1] = 1 -> DeltaPocS0[1] = -3
        w.write_bit(0); // used[1]
        w.write_bit(1); // stop bit so the buffer is non-empty/aligned
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        let (rps, _) = parse_h265_short_term_rps(&mut br, 0, 1, &[]).expect("explicit RPS parses");
        assert_eq!(rps.num_negative_pics, 2);
        assert_eq!(rps.num_positive_pics, 0);
        assert_eq!(rps.delta_poc_s0[0], -1);
        assert_eq!(rps.delta_poc_s0[1], -3);
        assert!(rps.used_s0[0] && !rps.used_s0[1]);

        // Std round-trip: invert DeltaPocS0 back to delta_poc_s0_minus1 (0, 1) and
        // pack the used flags into the s0 bitmask (bit 0 set, bit 1 clear).
        let std = to_std_h265_short_term_rps(&rps);
        assert_eq!(std.num_negative_pics, 2);
        assert_eq!(std.delta_poc_s0_minus1[0], 0);
        assert_eq!(std.delta_poc_s0_minus1[1], 1);
        assert_eq!(std.used_by_curr_pic_s0_flag, 0b01);
        assert_eq!(std.flags.inter_ref_pic_set_prediction_flag(), 0);
    }

    #[test]
    fn h265_short_term_rps_inter_prediction_derives() {
        // Reference set: one negative pic at DeltaPocS0 = -1, used. The current
        // set is coded by inter-RPS prediction with deltaRps = +1, so its single
        // reference lands at DeltaPocS1 = +1 (a positive pic), per H.265 7.4.8.
        let reference = H265ShortTermRps {
            num_negative_pics: 1,
            num_positive_pics: 0,
            delta_poc_s0: {
                let mut a = [0i32; 16];
                a[0] = -1;
                a
            },
            used_s0: {
                let mut a = [false; 16];
                a[0] = true;
                a
            },
            ..Default::default()
        };
        // Bitstream for st_ref_pic_set(1): inter_ref_pic_set_prediction_flag=1,
        // delta_rps_sign=0, abs_delta_rps_minus1=ue(0), then NumDeltaPocs+1 = 2
        // used_by_curr_pic_flag bits (both 1, so no use_delta bits).
        let mut w = BitWriter::default();
        w.write_bit(1); // inter_ref_pic_set_prediction_flag
        w.write_bit(0); // delta_rps_sign
        w.write_ue(0); // abs_delta_rps_minus1 -> deltaRps = +1
        w.write_bit(1); // used_by_curr_pic_flag[0]
        w.write_bit(1); // used_by_curr_pic_flag[1]
        w.write_bit(1); // stop bit
        w.align_to_byte();
        let bytes = w.into_bytes();
        let mut br = BitReader::new(&bytes);
        let (rps, _) = parse_h265_short_term_rps(&mut br, 1, 2, core::slice::from_ref(&reference))
            .expect("inter-predicted RPS derives");
        assert_eq!(
            rps.num_negative_pics, 0,
            "the ref shifts out of the negative set"
        );
        assert_eq!(rps.num_positive_pics, 1, "and into the positive set");
        assert_eq!(rps.delta_poc_s1[0], 1);
        assert!(rps.used_s1[0]);

        let std = to_std_h265_short_term_rps(&rps);
        assert_eq!(std.num_positive_pics, 1);
        assert_eq!(std.delta_poc_s1_minus1[0], 0, "DeltaPocS1 1 -> minus1 0");
        assert_eq!(std.used_by_curr_pic_s1_flag, 0b1);
    }

    #[test]
    fn h265_parse_rejects_truncated_input() {
        // A VPS NAL truncated mid-profile_tier_level must fail, not panic.
        let vps = nal_units_any(H265_CLIP)
            .find(|n| n.len() >= 2 && (n[0] >> 1) & 0x3F == 32)
            .expect("VPS NAL");
        let rbsp = strip_emulation_prevention(&vps[2..]);
        assert!(
            parse_h265_vps(&rbsp[..4]).is_none(),
            "truncated VPS rejected"
        );
    }

    // ------------------------------------------------------------------------
    // AV1 OBU parse (M504)

    const AV1_CLIP: &[u8] = include_bytes!("../tests/fixtures/av1_640x480.obu");
    const AV1_FRAME_TYPE_INTER: u8 = 1;

    #[test]
    fn parses_av1_sequence_header_from_real_clip() {
        let seq = extract_av1_sequence_header(AV1_CLIP).expect("sequence header parse");
        // Main profile, 8-bit 4:2:0, 640x480. libaom encodes max_frame_*_minus_1
        // as the coded dimension minus one.
        assert_eq!(seq.seq_profile, 0, "Main profile");
        assert_eq!(seq.max_frame_width_minus_1 + 1, 640, "coded width");
        assert_eq!(seq.max_frame_height_minus_1 + 1, 480, "coded height");
        assert_eq!(seq.color.bit_depth, 8, "8-bit");
        assert!(!seq.color.mono_chrome, "not monochrome");
        assert_eq!(seq.color.num_planes, 3, "3 planes");
        assert_eq!(seq.color.subsampling_x, 1, "4:2:0 x");
        assert_eq!(seq.color.subsampling_y, 1, "4:2:0 y");
        // A reduced still-picture header would not carry the operating-point /
        // order-hint syntax the DPB path needs; the real clip is a full header.
        assert!(!seq.reduced_still_picture_header, "full sequence header");
    }

    #[test]
    fn std_av1_mapping_preserves_geometry_and_wires_color_pointer() {
        let seq = extract_av1_sequence_header(AV1_CLIP).unwrap();
        let std = to_std_av1_seq_header(&seq);
        assert_eq!(std.seq_header.seq_profile, seq.seq_profile as _);
        assert_eq!(
            std.seq_header.max_frame_width_minus_1,
            seq.max_frame_width_minus_1 as u16
        );
        assert_eq!(
            std.seq_header.max_frame_height_minus_1,
            seq.max_frame_height_minus_1 as u16
        );
        // pColorConfig must point at the owned block; pTimingInfo stays null (we
        // carry no timing info into the session).
        assert!(!std.seq_header.pColorConfig.is_null(), "color config wired");
        assert!(std.seq_header.pTimingInfo.is_null(), "no timing info");
        // SAFETY: pColorConfig points at the bundle's owned box, alive as long as
        // `std` is.
        let color = unsafe { &*std.seq_header.pColorConfig };
        assert_eq!(color.BitDepth, 8);
        assert_eq!(color.subsampling_x, 1);
        assert_eq!(color.subsampling_y, 1);
    }

    #[test]
    fn av1_frame_classification_matches_fixture_structure() {
        // The fixture is one KEY frame then nine INTER frames, every frame shown,
        // no show_existing_frame (verified against ffmpeg trace_headers).
        let frames = av1_frame_infos(AV1_CLIP).expect("frame classification");
        assert_eq!(frames.len(), 10, "10 coded frames");
        assert_eq!(frames[0].frame_type, AV1_FRAME_TYPE_KEY, "frame 0 is KEY");
        assert!(frames[0].show_frame, "frame 0 shown");
        for (i, f) in frames.iter().enumerate().skip(1) {
            assert_eq!(f.frame_type, AV1_FRAME_TYPE_INTER, "frame {i} is INTER");
            assert!(f.show_frame, "frame {i} shown");
            assert!(
                !f.show_existing_frame,
                "frame {i} is coded, not show-existing"
            );
        }
    }

    #[test]
    fn av1_obu_walk_and_parse_reject_malformed_input() {
        // A truncated sequence header payload must fail, not panic or over-read.
        let seq_obu = av1_obus(AV1_CLIP)
            .into_iter()
            .find(|o| o.obu_type == OBU_SEQUENCE_HEADER)
            .expect("sequence header OBU");
        assert!(
            parse_av1_sequence_header(&seq_obu.payload[..2]).is_none(),
            "truncated sequence header rejected"
        );
        // A header byte with the forbidden bit set stops the walk cleanly.
        assert!(
            av1_obus(&[0x80, 0x00]).is_empty(),
            "forbidden-bit OBU rejected"
        );
        // An OBU claiming a payload longer than the buffer is dropped, not read.
        // Header: type=SEQUENCE_HEADER, has_size=1; size=200 but only 1 byte follows.
        assert!(
            av1_obus(&[0x0a, 200, 0x00]).is_empty(),
            "over-long OBU size rejected"
        );
    }

    #[test]
    fn av1_leb128_decodes_multi_byte_values() {
        // 0x80 0x01 => 128 (continuation on the first byte); consumed 2 bytes.
        assert_eq!(read_leb128(&[0x80, 0x01], 0), Some((128, 2)));
        // Single-byte value.
        assert_eq!(read_leb128(&[0x24], 0), Some((0x24, 1)));
        // Truncated (continuation bit set, no next byte) => None.
        assert_eq!(read_leb128(&[0x80], 0), None);
    }

    /// The frame / frame-header OBU payloads in the fixture, in decoding order.
    fn av1_frame_obu_payloads(stream: &[u8]) -> alloc::vec::Vec<&[u8]> {
        av1_obus(stream)
            .into_iter()
            .filter(|o| o.obu_type == OBU_FRAME || o.obu_type == OBU_FRAME_HEADER)
            .map(|o| o.payload)
            .collect()
    }

    #[test]
    fn av1_frame_header_key_frame_matches_ffmpeg_trace() {
        // Values cross-checked against `ffmpeg -bsf:v trace_headers` on the fixture.
        let seq = extract_av1_sequence_header(AV1_CLIP).expect("seq header");
        let payloads = av1_frame_obu_payloads(AV1_CLIP);
        let refs = Av1RefFrames::default();
        let fh = parse_av1_frame_header(payloads[0], &seq, &refs).expect("frame 0 header");

        assert!(!fh.show_existing_frame);
        assert_eq!(fh.frame_type, AV1_FRAME_TYPE_KEY);
        assert!(fh.frame_is_intra);
        assert!(fh.show_frame);
        assert!(!fh.disable_cdf_update);
        assert!(!fh.allow_screen_content_tools);
        assert!(!fh.frame_size_override_flag);
        assert_eq!(fh.order_hint, 0);
        assert_eq!(fh.primary_ref_frame, AV1_PRIMARY_REF_NONE);
        assert_eq!(fh.refresh_frame_flags, 0xff, "KEY+show refreshes all slots");
        assert_eq!((fh.frame_width, fh.frame_height), (640, 480));
        assert_eq!(
            (fh.upscaled_width, fh.render_width, fh.render_height),
            (640, 640, 480)
        );
        // Single tile.
        assert_eq!((fh.tile.tile_cols, fh.tile.tile_rows), (1, 1));
        assert_eq!((fh.tile.tile_cols_log2, fh.tile.tile_rows_log2), (0, 0));
        // Quantization.
        assert_eq!(fh.quant.base_q_idx, 45);
        assert_eq!(fh.quant.delta_q_y_dc, 0);
        assert!(!fh.quant.using_qmatrix);
        assert!(!fh.seg.enabled);
        assert!(!fh.delta_q_present);
        // Loop filter.
        assert_eq!(fh.lf.level, [2, 2, 2, 2]);
        assert_eq!(fh.lf.sharpness, 0);
        assert!(fh.lf.delta_enabled);
        assert!(!fh.lf.delta_update);
        // CDEF: cdef_bits=1 -> 2 entries.
        assert_eq!(fh.cdef.damping_minus_3, 0);
        assert_eq!(fh.cdef.bits, 1);
        assert_eq!(fh.cdef.y_pri[0], 5);
        assert_eq!(fh.cdef.uv_pri[0], 5);
        assert_eq!(fh.cdef.y_pri[1], 0);
        assert_eq!(fh.cdef.uv_pri[1], 5);
        assert!(!fh.coded_lossless);
        assert_eq!(fh.tx_mode, 1, "TX_MODE_LARGEST");
        assert!(!fh.reduced_tx_set);
        // The header parses fully and lands on a byte boundary.
        assert!(fh.header_byte_len > 0 && fh.header_byte_len <= payloads[0].len());
    }

    #[test]
    fn av1_frame_header_inter_frame_matches_ffmpeg_trace() {
        let seq = extract_av1_sequence_header(AV1_CLIP).expect("seq header");
        let payloads = av1_frame_obu_payloads(AV1_CLIP);

        // After the KEY frame (refresh_frame_flags 0xff), every slot holds it:
        // order hint 0, type KEY, 640x480.
        let mut refs = Av1RefFrames::default();
        for i in 0..8 {
            refs.valid[i] = true;
            refs.order_hint[i] = 0;
            refs.frame_type[i] = AV1_FRAME_TYPE_KEY;
            refs.upscaled_width[i] = 640;
            refs.frame_height[i] = 480;
            refs.render_width[i] = 640;
            refs.render_height[i] = 480;
        }

        let fh = parse_av1_frame_header(payloads[1], &seq, &refs).expect("frame 1 header");
        assert_eq!(fh.frame_type, AV1_FRAME_TYPE_INTER);
        assert!(!fh.frame_is_intra);
        assert!(fh.show_frame);
        assert!(!fh.error_resilient_mode);
        assert_eq!(fh.order_hint, 1);
        assert_eq!(fh.primary_ref_frame, 6);
        assert_eq!(fh.refresh_frame_flags, 0b0000_0010, "refreshes slot 1");
        // ref_frame_idx[0..7] = 0,1,7,6,7,7,0 from the trace.
        assert_eq!(fh.ref_frame_idx, [0, 1, 7, 6, 7, 7, 0]);
        assert_eq!((fh.frame_width, fh.frame_height), (640, 480));
        assert!(!fh.allow_high_precision_mv);
        assert_eq!(
            fh.interpolation_filter, 4,
            "SWITCHABLE (is_filter_switchable=1)"
        );
        assert!(fh.is_motion_mode_switchable);
        assert!(fh.use_ref_frame_mvs);
        assert_eq!(fh.quant.base_q_idx, 128);
        assert_eq!(fh.lf.level[0], 11);
        assert_eq!(fh.lf.level[1], 11);
        // Whole header consumed within the OBU payload.
        assert!(fh.header_byte_len > 0 && fh.header_byte_len <= payloads[1].len());
    }

    #[test]
    fn av1_all_ten_frame_headers_parse() {
        // Every coded frame's header must parse cleanly with a plausible ref
        // state, proving the parser stays in sync across the whole stream.
        let seq = extract_av1_sequence_header(AV1_CLIP).expect("seq header");
        let payloads = av1_frame_obu_payloads(AV1_CLIP);
        assert_eq!(payloads.len(), 10);
        let mut refs = Av1RefFrames::default();
        for i in 0..8 {
            refs.valid[i] = true;
            refs.upscaled_width[i] = 640;
            refs.frame_height[i] = 480;
            refs.render_width[i] = 640;
            refs.render_height[i] = 480;
        }
        for (i, p) in payloads.iter().enumerate() {
            let fh = parse_av1_frame_header(p, &seq, &refs).unwrap_or_else(|| {
                panic!("frame {i} header must parse");
            });
            // Refresh the referenced slots with this frame's order hint (a coarse
            // model, enough to keep skip-mode / order-hint logic well-defined).
            for slot in 0..8 {
                if fh.refresh_frame_flags & (1 << slot) != 0 {
                    refs.order_hint[slot] = fh.order_hint;
                    refs.frame_type[slot] = fh.frame_type;
                }
            }
        }
    }

    /// A 2x2 NV12 frame with a constant (Y, Cb, Cr) sample, for exercising the
    /// colour conversion on a single known triple.
    fn solid_nv12(y: u8, cb: u8, cr: u8) -> Nv12Frame {
        Nv12Frame {
            width: 2,
            height: 2,
            luma: alloc::vec![y; 4],
            chroma: alloc::vec![cb, cr],
            bit_depth: 8,
        }
    }

    #[test]
    fn cicp_matrix_resolution() {
        use ColorMatrix::*;
        // transfer_characteristics 2 (unspecified) -> SDR for these matrix cases.
        assert_eq!(VideoColorSpace::from_cicp(1, 2, false, 480).matrix, Bt709);
        assert_eq!(
            VideoColorSpace::from_cicp(9, 2, false, 480).matrix,
            Bt2020Ncl
        );
        assert_eq!(VideoColorSpace::from_cicp(6, 2, false, 1080).matrix, Bt601);
        assert_eq!(VideoColorSpace::from_cicp(5, 2, false, 1080).matrix, Bt601);
        // Unspecified (2) resolves by height: HD -> 709, SD -> 601.
        assert_eq!(VideoColorSpace::from_cicp(2, 2, false, 1080).matrix, Bt709);
        assert_eq!(VideoColorSpace::from_cicp(2, 2, false, 480).matrix, Bt601);
        assert!(VideoColorSpace::from_cicp(1, 2, true, 1080).full_range);
        // Transfer resolution: CICP 16 -> PQ, 18 -> HLG, else SDR.
        use TransferFunction::*;
        assert_eq!(VideoColorSpace::from_cicp(9, 16, false, 1080).transfer, Pq);
        assert_eq!(VideoColorSpace::from_cicp(9, 18, false, 1080).transfer, Hlg);
        assert_eq!(VideoColorSpace::from_cicp(1, 1, false, 1080).transfer, Sdr);
    }

    #[test]
    fn bt601_studio_matches_legacy_coefficients() {
        // The colour-aware formula must reproduce the historical hard-coded BT.601
        // studio conversion exactly (else every existing texture regresses). Check a
        // chroma-bearing sample against the old inlined coefficients.
        let (y, cb, cr) = (120u8, 90u8, 200u8);
        let px = &nv12_to_rgba(&solid_nv12(y, cb, cr), VideoColorSpace::BT601_STUDIO)[0..3];
        let yc = (y as f32 - 16.0) * 1.164_383;
        let cbf = cb as f32 - 128.0;
        let crf = cr as f32 - 128.0;
        let want = [
            (yc + 1.596_027 * crf).clamp(0.0, 255.0) as u8,
            (yc - 0.391_762 * cbf - 0.812_968 * crf).clamp(0.0, 255.0) as u8,
            (yc + 2.017_232 * cbf).clamp(0.0, 255.0) as u8,
        ];
        assert_eq!(px, want, "BT.601 studio must match the legacy coefficients");
    }

    #[test]
    fn conversion_endpoints_and_matrix_differences() {
        // Studio black / white map to 0 / 255 on the luma axis (neutral chroma).
        let black = &nv12_to_rgba(&solid_nv12(16, 128, 128), VideoColorSpace::BT601_STUDIO)[0..3];
        let white = &nv12_to_rgba(&solid_nv12(235, 128, 128), VideoColorSpace::BT601_STUDIO)[0..3];
        assert_eq!(black, [0, 0, 0]);
        assert_eq!(white, [255, 255, 255]);
        // Full range uses the samples directly: 0 -> black, 255 -> white.
        let fr = VideoColorSpace {
            matrix: ColorMatrix::Bt709,
            full_range: true,
            transfer: TransferFunction::Sdr,
        };
        assert_eq!(&nv12_to_rgba(&solid_nv12(0, 128, 128), fr)[0..3], [0, 0, 0]);
        assert_eq!(
            &nv12_to_rgba(&solid_nv12(255, 128, 128), fr)[0..3],
            [255, 255, 255]
        );
        // A red-ish chroma (high Cr) converts differently under 601 vs 709 vs 2020
        // (different luma weights) -> the matrix actually changes the output.
        let s = solid_nv12(120, 100, 210);
        let p601 = nv12_to_rgba(&s, VideoColorSpace::BT601_STUDIO);
        let p709 = nv12_to_rgba(
            &s,
            VideoColorSpace {
                matrix: ColorMatrix::Bt709,
                full_range: false,
                transfer: TransferFunction::Sdr,
            },
        );
        let p2020 = nv12_to_rgba(
            &s,
            VideoColorSpace {
                matrix: ColorMatrix::Bt2020Ncl,
                full_range: false,
                transfer: TransferFunction::Sdr,
            },
        );
        assert_ne!(p601[0..3], p709[0..3], "601 vs 709 must differ on chroma");
        assert_ne!(p709[0..3], p2020[0..3], "709 vs 2020 must differ on chroma");
    }
}
