//! M491: Vulkan Video decode -> wgpu RGBA texture, GPU-resident (no CPU copy).
//!
//! Fifth increment of `VulkanVideoDec` (DESIGN.md 4.11.6): the frame never
//! leaves the GPU. The decoded NV12 image is converted to RGBA by a Vulkan
//! compute pass through a `VkSamplerYcbcrConversion` on a dedicated compute
//! queue, and the RGBA image is imported straight into wgpu (`texture_from_raw`)
//! -- unlike M490's CPU NV12->RGBA round-trip. Decodes the fixture's IDR, then
//! reads the wgpu texture back and asserts real content. Runs on the RTX 3060;
//! skips with no adapter or no distinct compute queue.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

#[test]
fn decodes_idr_into_wgpu_texture_gpu_resident() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("skipping: no Vulkan adapter");
            return;
        }
        Err(VulkanVideoError::ExtensionUnsupported) | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping: GPU has no Vulkan H.264 decode support");
            return;
        }
        Err(e) => panic!("failed to open decode device: {e:?}"),
    };

    let ps = extract_h264_parameter_sets(CLIP).expect("parse SPS+PPS");
    let session = device
        .create_h264_session(&ps, 640, 480)
        .expect("create session");

    let texture = match device.decode_idr_to_rgba_texture_gpu(&session, CLIP) {
        Ok(t) => t,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skipping: no distinct compute queue for the GPU-resident path");
            return;
        }
        Err(e) => panic!("GPU-resident decode failed: {e:?}"),
    };
    assert_eq!(texture.width(), 640);
    assert_eq!(texture.height(), 480);

    // Read the imported texture back through wgpu and confirm the GPU-side
    // ycbcr conversion produced a real picture.
    let rgba = device.read_rgba_texture(&texture);
    assert_eq!(rgba.len(), 640 * 480 * 4);
    let min = *rgba.iter().min().unwrap();
    let max = *rgba.iter().max().unwrap();
    assert!(
        max > min,
        "GPU-converted RGBA texture is uniform ({min}=={max})"
    );

    let sum: u64 = rgba.iter().map(|&b| b as u64).sum();
    let mean = sum / rgba.len() as u64;
    eprintln!("GPU-resident decode -> wgpu Rgba8Unorm 640x480: range {min}..={max}, mean {mean}");
}
