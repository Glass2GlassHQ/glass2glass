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

use g2g_core::VideoCodec;

/// Compiled SPIR-V for the YCbCr -> RGBA compute shader, shared with the Android
/// `mediacodec-wgpu` path (`shaders/mediacodec_ycbcr.comp`): it samples a
/// combined image sampler carrying a `VkSamplerYcbcrConversion` (so the YUV math
/// happens in the sampler) and writes RGBA to a storage image, which is codec
/// and source agnostic.
const YCBCR_COMP_SPV: &[u8] = include_bytes!("shaders/mediacodec_ycbcr.comp.spv");

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
    /// A Vulkan call failed (capability query, session/image/buffer creation, or
    /// the decode submission).
    QueryFailed(vk::Result),
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
        (video_instance.fp().get_physical_device_video_capabilities_khr)(
            phys,
            &profile,
            &mut caps,
        )
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
    Some(H264ParameterSets { sps: sps?, pps: pps? })
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
    H264Profile { profile, _usage: usage, _h264: h264 }
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
            && props.memory_types[i as usize].property_flags.contains(flags)
    })
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
    let codec = VulkanVideoCodec::H264;
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

    // Probe first (also yields the decode queue family we must request), and
    // pick a compute family for the GPU-resident NV12 -> RGBA pass.
    // SAFETY: the guard holds the adapter's live handles for the calls.
    let (caps, compute_family) = unsafe {
        let hal_adapter = adapter
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(VulkanVideoError::NoVulkanAdapter)?;
        let shared = hal_adapter.shared_instance();
        let raw_instance = shared.raw_instance();
        let phys = hal_adapter.raw_physical_device();
        let caps = probe_physical_device(shared.entry(), raw_instance, phys, codec)?;
        let families = raw_instance.get_physical_device_queue_family_properties(phys);
        // Prefer a dedicated compute family (COMPUTE without GRAPHICS) distinct
        // from wgpu's family 0 and the decode family, so we own a clean queue.
        let compute = families
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                *i as u32 != caps.decode_queue_family
                    && *i != 0
                    && p.queue_flags.contains(vk::QueueFlags::COMPUTE)
            })
            .min_by_key(|(_, p)| p.queue_flags.contains(vk::QueueFlags::GRAPHICS) as u8)
            .map(|(i, _)| i as u32);
        (caps, compute)
    };

    let decode_family = caps.decode_queue_family;
    let exts = decode_device_extensions(codec);
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
    let (raw_device, phys, mem_props, video_fns, decode_fns, sync2_fns, decode_queue) = unsafe {
        let hal_device = wgpu_device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(VulkanVideoError::NoVulkanAdapter)?;
        let raw_device = hal_device.raw_device().clone();
        let phys = hal_device.raw_physical_device();
        let raw_instance = hal_device.shared_instance().raw_instance();
        let mem_props = raw_instance.get_physical_device_memory_properties(phys);
        let video_fns = ash::khr::video_queue::Device::new(raw_instance, &raw_device);
        let decode_fns = ash::khr::video_decode_queue::Device::new(raw_instance, &raw_device);
        let sync2_fns = ash::khr::synchronization2::Device::new(raw_instance, &raw_device);
        let decode_queue = raw_device.get_device_queue(decode_family, 0);
        (raw_device, phys, mem_props, video_fns, decode_fns, sync2_fns, decode_queue)
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

impl VulkanVideoDevice {
    pub fn caps(&self) -> &VulkanVideoDecodeCaps {
        &self.caps
    }

    /// Choose the decode picture format the driver supports for the H.264
    /// profile (DPB + output usage). Prefers the two-plane 4:2:0 NV12 layout.
    fn h264_decode_format(&self, profile: &H264Profile) -> Result<vk::Format, VulkanVideoError> {
        let mut profile_list = vk::VideoProfileListInfoKHR::default()
            .profiles(core::slice::from_ref(&profile.profile));
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
        // Prefer NV12 (two-plane 4:2:0); else take the first offered format.
        let chosen = formats
            .iter()
            .find(|f| f.format == vk::Format::G8_B8R8_2PLANE_420_UNORM)
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
        let picture_format = self.h264_decode_format(&prof)?;

        let w = max_w.clamp(self.caps.min_coded_extent.0, self.caps.max_coded_extent.0);
        let h = max_h.clamp(self.caps.min_coded_extent.1, self.caps.max_coded_extent.1);
        let coded_extent = vk::Extent2D { width: w, height: h };

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
            let _ = (self.video_fns.fp().get_video_session_memory_requirements_khr)(
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
            let _ = (self.video_fns.fp().get_video_session_memory_requirements_khr)(
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
            .or_else(|| find_memory_type(&self.mem_props, type_bits, vk::MemoryPropertyFlags::empty()))
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
        // SAFETY: image is valid (contract).
        let req = unsafe { self.raw_device.get_image_memory_requirements(image) };
        let mem_type = find_memory_type(&self.mem_props, req.memory_type_bits, flags)
            .or_else(|| {
                find_memory_type(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::empty())
            })
            .ok_or(VulkanVideoError::ExtensionUnsupported)?;
        let ai = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        // SAFETY: valid allocate info.
        let mem = unsafe { self.raw_device.allocate_memory(&ai, None) }
            .map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh image + memory, single bind.
        unsafe { self.raw_device.bind_image_memory(image, mem, 0) }
            .map_err(VulkanVideoError::QueryFailed)?;
        Ok(mem)
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
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
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
        let image = unsafe { dev.create_image(&image_ci, None) }
            .map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh image.
        let image_mem = unsafe { self.alloc_bind_image(image, vk::MemoryPropertyFlags::DEVICE_LOCAL) }?;
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
        let bitstream = unsafe { dev.create_buffer(&buf_ci, None) }
            .map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh buffer.
        let breq = unsafe { dev.get_buffer_memory_requirements(bitstream) };
        let btype = find_memory_type(
            &self.mem_props,
            breq.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or(VulkanVideoError::ExtensionUnsupported)?;
        let bai = vk::MemoryAllocateInfo::default().allocation_size(breq.size).memory_type_index(btype);
        // SAFETY: valid allocate + bind + map of a fresh host-visible buffer.
        let bitstream_mem = unsafe {
            let m = dev.allocate_memory(&bai, None).map_err(VulkanVideoError::QueryFailed)?;
            dev.bind_buffer_memory(bitstream, m, 0).map_err(VulkanVideoError::QueryFailed)?;
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
        let readback = unsafe { dev.create_buffer(&rb_ci, None) }
            .map_err(VulkanVideoError::QueryFailed)?;
        // SAFETY: fresh buffer.
        let rreq = unsafe { dev.get_buffer_memory_requirements(readback) };
        let rtype = find_memory_type(
            &self.mem_props,
            rreq.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or(VulkanVideoError::ExtensionUnsupported)?;
        let rai = vk::MemoryAllocateInfo::default().allocation_size(rreq.size).memory_type_index(rtype);
        // SAFETY: allocate + bind of a fresh buffer.
        let readback_mem = unsafe {
            let m = dev.allocate_memory(&rai, None).map_err(VulkanVideoError::QueryFailed)?;
            dev.bind_buffer_memory(readback, m, 0).map_err(VulkanVideoError::QueryFailed)?;
            m
        };

        // Command pool + buffer on the decode queue family.
        let pool_ci = vk::CommandPoolCreateInfo::default().queue_family_index(self.decode_queue_family);
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
            .coded_extent(vk::Extent2D { width: w, height: h })
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
        let control_info = vk::VideoCodingControlInfoKHR::default()
            .flags(vk::VideoCodingControlFlagsKHR::RESET);
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
            dev.begin_command_buffer(cb, &begin).map_err(VulkanVideoError::QueryFailed)?;

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
            let dep = vk::DependencyInfo::default()
                .image_memory_barriers(core::slice::from_ref(&to_dpb));
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
            let dep2 = vk::DependencyInfo::default()
                .image_memory_barriers(core::slice::from_ref(&to_src));
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
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
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
                .image_extent(vk::Extent3D { width: w / 2, height: h / 2, depth: 1 });
            let regions = [luma_region, chroma_region];
            dev.cmd_copy_image_to_buffer(
                cb,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback,
                &regions,
            );

            dev.end_command_buffer(cb).map_err(VulkanVideoError::QueryFailed)?;

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
                unsafe { self.destroy_decode_transients(pool, image, view, image_mem, bitstream, bitstream_mem, readback, readback_mem) };
                return Err(e);
            }
        };

        // SAFETY: teardown of all transient objects, each destroyed once, after
        // the decode has completed (fence waited).
        unsafe { self.destroy_decode_transients(pool, image, view, image_mem, bitstream, bitstream_mem, readback, readback_mem) };

        Ok(Nv12Frame { width: w, height: h, luma, chroma })
    }

    /// Decode a single IDR frame and return just the luma plane (`width*height`
    /// bytes). Thin wrapper over [`decode_idr_nv12`](Self::decode_idr_nv12).
    pub fn decode_idr_luma(
        &self,
        session: &H264DecodeSession,
        idr_au: &[u8],
    ) -> Result<DecodedLuma, VulkanVideoError> {
        let f = self.decode_idr_nv12(session, idr_au)?;
        Ok(DecodedLuma { width: f.width, height: f.height, luma: f.luma })
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
        let rgba = nv12_to_rgba(&frame);
        let texture = self.wgpu_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vulkan-video-decoded-rgba"),
            size: wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.wgpu_queue.write_texture(
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
            wgpu::Extent3d {
                width: frame.width,
                height: frame.height,
                depth_or_array_layers: 1,
            },
        );
        self.wgpu_queue.submit([]);
        Ok(texture)
    }

    /// Read an `Rgba8Unorm` wgpu texture on this device back to tightly-packed
    /// `width*height*4` bytes (validation / CPU-consumer helper). Handles the
    /// 256-byte row alignment wgpu requires for buffer copies.
    pub fn read_rgba_texture(&self, texture: &wgpu::Texture) -> alloc::vec::Vec<u8> {
        let (w, h) = (texture.width(), texture.height());
        let unpadded = (w * 4) as usize;
        let padded = unpadded.div_ceil(256) * 256;
        let buffer = self.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vulkan-video-readback"),
            size: (padded * h as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.wgpu_queue.submit([enc.finish()]);
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.wgpu_device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        // Strip the row padding back to tightly-packed rows.
        let mut out = alloc::vec::Vec::with_capacity(unpadded * h as usize);
        for row in 0..h as usize {
            let start = row * padded;
            out.extend_from_slice(&mapped[start..start + unpadded]);
        }
        drop(mapped);
        buffer.unmap();
        out
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
            let bai =
                vk::MemoryAllocateInfo::default().allocation_size(breq.size).memory_type_index(bt);
            let m = dev.allocate_memory(&bai, None).map_err(VulkanVideoError::QueryFailed)?;
            dev.bind_buffer_memory(bitstream, m, 0).map_err(VulkanVideoError::QueryFailed)?;
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
        let pool =
            unsafe { dev.create_command_pool(&pool_ci, None) }.map_err(VulkanVideoError::QueryFailed)?;
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
            .coded_extent(vk::Extent2D { width: w, height: h })
            .base_array_layer(0)
            .image_view_binding(decode_view);
        let begin_slot =
            vk::VideoReferenceSlotInfoKHR::default().slot_index(-1).picture_resource(&picres);
        let setup_slot =
            vk::VideoReferenceSlotInfoKHR::default().slot_index(0).picture_resource(&picres);
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
            dev.begin_command_buffer(cb, &begin).map_err(VulkanVideoError::QueryFailed)?;
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
            dev.end_command_buffer(cb).map_err(VulkanVideoError::QueryFailed)?;
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
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
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

    /// Convert a decoded NV12 image (in `VIDEO_DECODE_DPB_KHR` layout) to an
    /// RGBA `wgpu::Texture` via a compute pass on `compute_queue`, importing the
    /// RGBA image into wgpu with no CPU copy.
    ///
    /// # Safety
    /// `nv12` must be a valid image on `self.raw_device`, decoded and idle
    /// (the decode fence was waited), accessible from the compute family.
    unsafe fn ycbcr_to_wgpu(
        &self,
        nv12: vk::Image,
        compute_queue: vk::Queue,
        w: u32,
        h: u32,
    ) -> Result<wgpu::Texture, VulkanVideoError> {
        let dev = &self.raw_device;
        let err = VulkanVideoError::QueryFailed;

        // SAFETY: standard-format ycbcr conversion + immutable sampler + compute
        // pipeline over the shared shader; every handle is created from `dev`,
        // used while valid, and destroyed exactly once (here, or in the wgpu drop
        // callback for the imported RGBA image); the compute submission is waited
        // on a fence before any teardown.
        unsafe {
            let conv_ci = vk::SamplerYcbcrConversionCreateInfo::default()
                .format(vk::Format::G8_B8R8_2PLANE_420_UNORM)
                .ycbcr_model(vk::SamplerYcbcrModelConversion::YCBCR_601)
                .ycbcr_range(vk::SamplerYcbcrRange::ITU_NARROW)
                .components(vk::ComponentMapping::default())
                .x_chroma_offset(vk::ChromaLocation::COSITED_EVEN)
                .y_chroma_offset(vk::ChromaLocation::COSITED_EVEN)
                .chroma_filter(vk::Filter::LINEAR);
            let conversion = dev.create_sampler_ycbcr_conversion(&conv_ci, None).map_err(err)?;

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
                .extent(vk::Extent3D { width: w, height: h, depth: 1 })
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
            let rgba_mem = self.alloc_bind_image(rgba, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
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
            let dsl = dev.create_descriptor_set_layout(&dsl_ci, None).map_err(err)?;
            let set_layouts = [dsl];
            let pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            let pipeline_layout = dev.create_pipeline_layout(&pl_ci, None).map_err(err)?;

            let code = ash::util::read_spv(&mut std::io::Cursor::new(YCBCR_COMP_SPV)).map_err(|_| {
                VulkanVideoError::QueryFailed(vk::Result::ERROR_INITIALIZATION_FAILED)
            })?;
            let sm_ci = vk::ShaderModuleCreateInfo::default().code(&code);
            let shader = dev.create_shader_module(&sm_ci, None).map_err(err)?;
            let entry = c"main";
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(entry);
            let cp_ci =
                vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout);
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
            let dp_ci =
                vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
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

            let cp_pool_ci =
                vk::CommandPoolCreateInfo::default().queue_family_index(self.compute_queue_family);
            let cmd_pool = dev.create_command_pool(&cp_pool_ci, None).map_err(err)?;
            let cb = dev.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .map_err(err)?[0];

            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.begin_command_buffer(cb, &begin).map_err(err)?;
            // NV12 DPB -> shader-read; RGBA undefined -> general (classic barrier:
            // no video stages here, and the decode already completed on a fence).
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
            // RGBA general -> shader-read for wgpu sampling.
            let to_sampled = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(rgba)
                .subresource_range(color_range());
            dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_sampled],
            );
            dev.end_command_buffer(cb).map_err(err)?;

            let fence = dev.create_fence(&vk::FenceCreateInfo::default(), None).map_err(err)?;
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

            // Import the RGBA image into wgpu (no copy). The drop callback frees
            // the image + memory once wgpu is done with the texture.
            let hal_device = self
                .wgpu_device
                .as_hal::<wgpu_hal::api::Vulkan>()
                .ok_or(VulkanVideoError::NoVulkanAdapter)?;
            let raw_for_drop = dev.clone();
            let drop_cb: wgpu_hal::DropCallback = alloc::boxed::Box::new(move || {
                raw_for_drop.destroy_image(rgba, None);
                raw_for_drop.free_memory(rgba_mem, None);
            });
            let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
            let hal_desc = wgpu_hal::TextureDescriptor {
                label: Some("vulkan-video-rgba"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
                memory_flags: wgpu_hal::MemoryFlags::empty(),
                view_formats: alloc::vec::Vec::new(),
            };
            let hal_tex = hal_device.texture_from_raw(
                rgba,
                &hal_desc,
                Some(drop_cb),
                wgpu_hal::vulkan::TextureMemory::External,
            );
            let wgpu_desc = wgpu::TextureDescriptor {
                label: Some("vulkan-video-rgba"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            };
            Ok(self
                .wgpu_device
                .create_texture_from_hal::<wgpu_hal::api::Vulkan>(hal_tex, &wgpu_desc))
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
        profile: &H264Profile,
    ) -> Result<DpbImage, VulkanVideoError> {
        let dev = &self.raw_device;
        let mut profile_list =
            vk::VideoProfileListInfoKHR::default().profiles(core::slice::from_ref(&profile.profile));
        let image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
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
        let mem = match unsafe { self.alloc_bind_image(image, vk::MemoryPropertyFlags::DEVICE_LOCAL) }
        {
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
        if ps.sps.pic_order_cnt_type == 1 {
            return Err(VulkanVideoError::UnsupportedStream);
        }
        let (w, h) = session.coded_extent;
        let max_num_ref_frames = ps.sps.max_num_ref_frames as usize;
        // One slot per possible short-term reference plus one for the picture
        // being decoded, clamped to the device DPB ceiling (at least two).
        let num_slots = (max_num_ref_frames + 1).clamp(2, self.caps.max_dpb_slots.max(2) as usize);

        let profile = h264_profile();

        // DPB image pool. On any failure, free what was already created.
        let mut slots: alloc::vec::Vec<DpbImage> = alloc::vec::Vec::with_capacity(num_slots);
        for _ in 0..num_slots {
            match self.create_dpb_image(w, h, session.picture_format, &profile) {
                Ok(img) => slots.push(img),
                Err(e) => {
                    for s in &slots {
                        // SAFETY: each handle created just above, destroyed once.
                        unsafe {
                            self.raw_device.destroy_image_view(s.view, None);
                            self.raw_device.destroy_image(s.image, None);
                            self.raw_device.free_memory(s.mem, None);
                        }
                    }
                    return Err(e);
                }
            }
        }

        // Persistent host-visible readback buffer for the decoded NV12 frame.
        let luma_len = (w as u64) * (h as u64);
        let chroma_len = luma_len / 2;
        let nv12_len = luma_len + chroma_len;
        let free_slots = |slots: &[DpbImage]| {
            for s in slots {
                // SAFETY: created above, destroyed once on this error path.
                unsafe {
                    self.raw_device.destroy_image_view(s.view, None);
                    self.raw_device.destroy_image(s.image, None);
                    self.raw_device.free_memory(s.mem, None);
                }
            }
        };
        let rb_ci = vk::BufferCreateInfo::default()
            .size(nv12_len)
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

        Ok(H264DpbDecoder {
            raw_device: self.raw_device.clone(),
            video_fns: self.video_fns.clone(),
            decode_fns: self.decode_fns.clone(),
            sync2_fns: self.sync2_fns.clone(),
            decode_queue: self.decode_queue,
            mem_props: self.mem_props,
            session: session.session,
            parameters: session.parameters,
            coded_extent: (w, h),
            size_align: self.caps.min_bitstream_buffer_size_alignment.max(1),
            profile,
            slots,
            refs: alloc::vec![None; num_slots],
            max_num_ref_frames,
            pool,
            readback,
            readback_mem,
            luma_len,
            chroma_len,
            nv12_len,
            sps: ps.sps.clone(),
            pps: ps.pps.clone(),
            poc_type: ps.sps.pic_order_cnt_type,
            log2_max_pic_order_cnt_lsb: ps.sps.log2_max_pic_order_cnt_lsb_minus4 as u32 + 4,
            max_frame_num: 1 << (ps.sps.log2_max_frame_num_minus4 as u32 + 4),
            prev_poc_msb: 0,
            prev_poc_lsb: 0,
            prev_frame_num: 0,
            prev_frame_num_offset: 0,
            first: true,
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
    /// Kept alive: the DPB-image and bitstream-buffer profile lists point into it.
    profile: H264Profile,
    slots: alloc::vec::Vec<DpbImage>,
    /// Per-slot reference state: `Some` means the slot holds a short-term
    /// reference picture, `None` means it is free.
    refs: alloc::vec::Vec<Option<RefPic>>,
    max_num_ref_frames: usize,
    pool: vk::CommandPool,
    readback: vk::Buffer,
    readback_mem: vk::DeviceMemory,
    luma_len: u64,
    chroma_len: u64,
    nv12_len: u64,
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
    /// The video session needs a `RESET` control on its first coding operation.
    first: bool,
}

impl core::fmt::Debug for H264DpbDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("H264DpbDecoder")
            .field("coded_extent", &self.coded_extent)
            .field("dpb_slots", &self.slots.len())
            .field("max_num_ref_frames", &self.max_num_ref_frames)
            .field("poc_type", &self.poc_type)
            .finish_non_exhaustive()
    }
}

impl Drop for H264DpbDecoder {
    fn drop(&mut self) {
        let dev = &self.raw_device;
        // SAFETY: all handles were created from `dev` in the constructor and are
        // destroyed exactly once here; the caller has finished decoding (every
        // decode waits a fence), so nothing is in flight.
        unsafe {
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

impl H264DpbDecoder {
    /// The DPB image count (one per reference slot plus the picture in flight).
    pub fn dpb_slots(&self) -> usize {
        self.slots.len()
    }

    /// Decode an entire Annex-B / AVCC elementary stream, returning one
    /// [`Nv12Frame`] per coded picture in decoding order. Access units are split
    /// on VCL NALs with `first_mb_in_slice == 0`; SPS/PPS/SEI NALs are ignored
    /// (the session already carries the parameter sets). Reference pictures are
    /// tracked across frames, so P frames after the leading IDR decode correctly.
    pub fn decode_all(&mut self, stream: &[u8]) -> Result<alloc::vec::Vec<Nv12Frame>, VulkanVideoError> {
        let mut frames = alloc::vec::Vec::new();
        let mut cur_slices: alloc::vec::Vec<&[u8]> = alloc::vec::Vec::new();
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
                frames.push(self.decode_picture(&h, &cur_slices)?);
                cur_slices.clear();
            }
            if cur_hdr.is_none() {
                cur_hdr = Some(hdr);
            }
            cur_slices.push(nal);
        }
        if !cur_slices.is_empty() {
            let h = cur_hdr.take().ok_or(VulkanVideoError::NoDecodableSlice)?;
            frames.push(self.decode_picture(&h, &cur_slices)?);
        }
        Ok(frames)
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
                let poc_msb = if lsb < self.prev_poc_lsb
                    && (self.prev_poc_lsb - lsb) >= max_lsb / 2
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
                let wrap = if fnum > cur { fnum - self.max_frame_num } else { fnum };
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
    fn decode_picture(
        &mut self,
        hdr: &H264SliceHeader,
        slices: &[&[u8]],
    ) -> Result<Nv12Frame, VulkanVideoError> {
        let (w, h) = self.coded_extent;
        let poc = self.compute_poc(hdr);

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

        self.submit_decode(hdr, poc, target, &active, &bitstream_data, &slice_offsets)?;

        let frame = self.read_back_nv12(w, h)?;

        // Reference marking: store the decoded picture as a short-term reference
        // (running sliding-window eviction first if the DPB is full). A
        // non-reference picture (nal_ref_idc == 0) leaves its slot free.
        if hdr.nal_ref_idc != 0 && self.max_num_ref_frames > 0 {
            let ref_count = self.refs.iter().filter(|r| r.is_some()).count();
            if ref_count >= self.max_num_ref_frames {
                self.evict_oldest(hdr.frame_num);
            }
            self.refs[target] = Some(RefPic { frame_num: hdr.frame_num, poc });
        }

        Ok(frame)
    }

    /// Record + submit the decode of one picture into `self.slots[target]`,
    /// leaving that image decoded and back in `VIDEO_DECODE_DPB_KHR` layout (so
    /// it can serve as a future reference), and copy it into the readback buffer.
    fn submit_decode(
        &mut self,
        hdr: &H264SliceHeader,
        poc: i32,
        target: usize,
        active: &[(usize, RefPic)],
        bitstream_data: &[u8],
        slice_offsets: &[u32],
    ) -> Result<(), VulkanVideoError> {
        let dev = &self.raw_device;
        let (w, h) = self.coded_extent;
        let num_refs = active.len();

        // Transient host-visible bitstream buffer holding this picture's slices.
        let buf_size = round_up(bitstream_data.len() as u64, self.size_align);
        let mut buf_profile_list = vk::VideoProfileListInfoKHR::default()
            .profiles(core::slice::from_ref(&self.profile.profile));
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
            let bai =
                vk::MemoryAllocateInfo::default().allocation_size(breq.size).memory_type_index(bt);
            let m = dev.allocate_memory(&bai, None).map_err(VulkanVideoError::QueryFailed)?;
            dev.bind_buffer_memory(bitstream, m, 0).map_err(VulkanVideoError::QueryFailed)?;
            let ptr = dev
                .map_memory(m, 0, breq.size, vk::MemoryMapFlags::empty())
                .map_err(VulkanVideoError::QueryFailed)? as *mut u8;
            core::ptr::copy_nonoverlapping(bitstream_data.as_ptr(), ptr, bitstream_data.len());
            dev.unmap_memory(m);
            m
        };

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
                    .coded_extent(vk::Extent2D { width: w, height: h })
                    .base_array_layer(0)
                    .image_view_binding(self.slots[*slot_i].view),
            );
        }
        let picres_target = vk::VideoPictureResourceInfoKHR::default()
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .coded_extent(vk::Extent2D { width: w, height: h })
            .base_array_layer(0)
            .image_view_binding(self.slots[target].view);

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
            .video_session(self.session)
            .video_session_parameters(self.parameters)
            .reference_slots(&begin_slots);
        let control_info =
            vk::VideoCodingControlInfoKHR::default().flags(vk::VideoCodingControlFlagsKHR::RESET);
        let end_info = vk::VideoEndCodingInfoKHR::default();
        let decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(bitstream)
            .src_buffer_offset(0)
            .src_buffer_range(buf_size)
            .dst_picture_resource(picres_target)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&ref_slots)
            .push_next(&mut h264_pic);

        // Reset the command pool and record this picture's decode + readback.
        // SAFETY: the pool has no in-flight command buffers (the previous decode
        // waited its fence); reset then allocate one primary buffer.
        let cb = unsafe {
            dev.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(VulkanVideoError::QueryFailed)?;
            let cb_ai = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            dev.allocate_command_buffers(&cb_ai).map_err(VulkanVideoError::QueryFailed)?[0]
        };

        let image = self.slots[target].image;
        let issue_reset = self.first;
        // SAFETY: every handle above is valid and outlives the waited submission;
        // the barriers move the target image UNDEFINED -> DPB (decode) ->
        // TRANSFER_SRC (readback copy) -> DPB (ready as a future reference). The
        // reference images passed in `begin`/`decode` are already in DPB layout.
        let r = unsafe {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            dev.begin_command_buffer(cb, &begin).map_err(VulkanVideoError::QueryFailed)?;

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
            if issue_reset {
                (self.video_fns.fp().cmd_control_video_coding_khr)(cb, &control_info);
            }
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
                .subresource_range(color_range());
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
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            let chroma_region = vk::BufferImageCopy::default()
                .buffer_offset(self.luma_len)
                .buffer_row_length(0)
                .buffer_image_height(0)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::PLANE_1,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D { width: w / 2, height: h / 2, depth: 1 });
            let regions = [luma_region, chroma_region];
            dev.cmd_copy_image_to_buffer(
                cb,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.readback,
                &regions,
            );

            // Return the target image to DPB layout so it can be a reference for
            // subsequent pictures (its content is preserved by the transition).
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

            dev.end_command_buffer(cb).map_err(VulkanVideoError::QueryFailed)?;

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

        // SAFETY: transient bitstream buffer + memory, freed once after the wait.
        unsafe {
            dev.destroy_buffer(bitstream, None);
            dev.free_memory(bitstream_mem, None);
        }
        r.map_err(VulkanVideoError::QueryFailed)?;
        self.first = false;
        Ok(())
    }

    /// Read the decoded NV12 frame out of the persistent readback buffer.
    fn read_back_nv12(&self, w: u32, h: u32) -> Result<Nv12Frame, VulkanVideoError> {
        let dev = &self.raw_device;
        // SAFETY: readback_mem is host-visible/coherent and holds `nv12_len` bytes
        // written by the completed copy; mapped and unmapped within this call.
        unsafe {
            let ptr = dev
                .map_memory(self.readback_mem, 0, self.nv12_len, vk::MemoryMapFlags::empty())
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
            Ok(Nv12Frame { width: w, height: h, luma, chroma })
        }
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
    /// `width * height` luma (Y) samples, row-major.
    pub luma: alloc::vec::Vec<u8>,
    /// `width * height / 2` bytes: `(width/2) * (height/2)` interleaved Cb,Cr
    /// pairs, row-major.
    pub chroma: alloc::vec::Vec<u8>,
}

/// Convert an [`Nv12Frame`] to packed RGBA8 (BT.601 limited-range), the layout a
/// wgpu `Rgba8Unorm` texture expects. CPU reference conversion; the GPU-resident
/// `VkSamplerYcbcrConversion` path is a later increment.
fn nv12_to_rgba(frame: &Nv12Frame) -> alloc::vec::Vec<u8> {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let cw = w / 2;
    let mut rgba = alloc::vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let yv = frame.luma[y * w + x] as f32;
            let ci = ((y / 2) * cw + (x / 2)) * 2;
            let cb = frame.chroma[ci] as f32 - 128.0;
            let cr = frame.chroma[ci + 1] as f32 - 128.0;
            // BT.601 limited-range YCbCr -> RGB.
            let yc = (yv - 16.0) * 1.164_383;
            let r = yc + 1.596_027 * cr;
            let g = yc - 0.391_762 * cb - 0.812_968 * cr;
            let b = yc + 2.017_232 * cb;
            let o = (y * w + x) * 4;
            rgba[o] = r.clamp(0.0, 255.0) as u8;
            rgba[o + 1] = g.clamp(0.0, 255.0) as u8;
            rgba[o + 2] = b.clamp(0.0, 255.0) as u8;
            rgba[o + 3] = 255;
        }
    }
    rgba
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(ps.pps.transform_8x8_mode_flag, 0, "baseline has no 8x8 transform");
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
}
