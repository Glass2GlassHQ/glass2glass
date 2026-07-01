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
    /// The driver rejected the capability query.
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

    Ok(VulkanVideoDecodeCaps {
        decode_queue_family,
        min_coded_extent: (caps.min_coded_extent.width, caps.min_coded_extent.height),
        max_coded_extent: (caps.max_coded_extent.width, caps.max_coded_extent.height),
        max_dpb_slots: caps.max_dpb_slots,
        max_active_reference_pictures: caps.max_active_reference_pictures,
        min_bitstream_buffer_offset_alignment: caps.min_bitstream_buffer_offset_alignment,
        min_bitstream_buffer_size_alignment: caps.min_bitstream_buffer_size_alignment,
        dpb_and_output_coincide: decode_caps
            .flags
            .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE),
    })
}
