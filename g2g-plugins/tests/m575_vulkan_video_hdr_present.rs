//! M575: HDR swapchain present - the device-side plumbing, on real hardware.
//!
//! The on-screen HDR present (`VulkanHdrSink`) is display + compositor dependent
//! and is validated live via `examples/vulkan_video_hdr_on_screen.rs`; the
//! surface-format / colour-space selection and `VkHdrMetadataEXT` construction have
//! unit tests in the module. What is checkable headlessly here is that the decode
//! device now opens with the present-side extensions (`VK_KHR_swapchain`, and
//! `VK_EXT_hdr_metadata` when available), so a swapchain sink can be built on it -
//! and that adding those extensions did not regress opening the device.
//!
//! Runs on the RTX 3060 (which supports both); skips with no Vulkan adapter.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "hdr-present"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{open_h265_decode_device, VulkanVideoError};

#[test]
fn decode_device_opens_present_capable() {
    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m575: no Vulkan H.265 decode adapter");
            return;
        }
        Err(e) => panic!("open h265 device: {e:?}"),
    };
    // The device must present (VK_KHR_swapchain enabled) for an HDR swapchain sink
    // to be buildable on it. Any GPU that drives a display supports this; the 3060
    // does. (If a headless-only GPU ever reports false, the sink returns
    // PresentUnsupported rather than mis-presenting.)
    assert!(
        device.present_capable(),
        "decode device did not enable VK_KHR_swapchain; HDR present sink cannot be built"
    );
    eprintln!(
        "m575: decode device present-capable; VK_EXT_hdr_metadata = {}",
        device.hdr_metadata_supported()
    );
}
