//! M486: Vulkan Video decode capability probe (`vulkan-video` feature).
//!
//! First increment of `VulkanVideoDec` (DESIGN.md 4.11.6): confirm the machine's
//! Vulkan driver actually exposes a `VK_QUEUE_VIDEO_DECODE_BIT_KHR` queue for
//! H.264 and report the decode limits `intercept_caps` and DPB sizing will use.
//! Runs for real on a Vulkan GPU (the RTX 3060 dev host); skips cleanly with no
//! adapter / no Vulkan-video support so CI and non-GPU hosts stay green.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{probe_decode_caps, VulkanVideoCodec, VulkanVideoError};

#[test]
fn probe_h264_decode_caps() {
    probe_and_check(VulkanVideoCodec::H264);
}

#[test]
fn probe_h265_decode_caps() {
    probe_and_check(VulkanVideoCodec::H265);
}

#[test]
fn probe_av1_decode_caps() {
    probe_and_check(VulkanVideoCodec::Av1);
}

fn probe_and_check(codec: VulkanVideoCodec) {
    let caps = match block_on(probe_decode_caps(codec)) {
        Ok(c) => c,
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("skipping {codec:?}: no Vulkan adapter");
            return;
        }
        Err(VulkanVideoError::ExtensionUnsupported) | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping {codec:?}: GPU has no Vulkan decode support for it");
            return;
        }
        Err(e) => panic!("unexpected probe error for {codec:?}: {e:?}"),
    };

    // If the driver reported decode support, the limits must be self-consistent:
    // a real coded-extent envelope and at least one DPB slot, or the decode
    // session we build on top of this would be nonsense.
    assert!(
        caps.max_coded_extent.0 >= caps.min_coded_extent.0
            && caps.max_coded_extent.1 >= caps.min_coded_extent.1,
        "max coded extent {:?} below min {:?}",
        caps.max_coded_extent,
        caps.min_coded_extent,
    );
    assert!(
        caps.max_coded_extent.0 >= 16 && caps.max_coded_extent.1 >= 16,
        "implausible max coded extent {:?}",
        caps.max_coded_extent,
    );
    // H.264 decode needs reference pictures, so a real decoder advertises DPB
    // slots and at least one active reference.
    assert!(caps.max_dpb_slots >= 1, "no DPB slots: {caps:?}");
    assert!(
        caps.max_active_reference_pictures >= 1,
        "no active reference pictures: {caps:?}",
    );
    // Bitstream buffer alignments are powers of two (Vulkan requires it); 0 would
    // break the offset math in the decode session.
    assert!(
        caps.min_bitstream_buffer_offset_alignment.is_power_of_two(),
        "offset alignment not a power of two: {caps:?}",
    );
    assert!(
        caps.min_bitstream_buffer_size_alignment.is_power_of_two(),
        "size alignment not a power of two: {caps:?}",
    );

    eprintln!(
        "Vulkan {codec:?} decode: queue family {}, coded {:?}..={:?}, dpb_slots {}, active_refs {}, coincide {}",
        caps.decode_queue_family,
        caps.min_coded_extent,
        caps.max_coded_extent,
        caps.max_dpb_slots,
        caps.max_active_reference_pictures,
        caps.dpb_and_output_coincide,
    );
}
