//! M490: Vulkan Video H.264 decode -> wgpu RGBA texture on real hardware.
//!
//! Fourth increment of `VulkanVideoDec` (DESIGN.md 4.11.6) and the wedge
//! payload: decode a frame and land it in a `wgpu::Texture` the way a wgpu
//! consumer (game engine / visualization viewer) samples it. Decodes the
//! fixture's IDR, converts NV12 -> RGBA, uploads to an `Rgba8Unorm` texture on
//! the decode device's wgpu queue, then reads the texture back through wgpu and
//! asserts real (non-uniform) content. Proves the whole decode -> wgpu-texture
//! path end to end. Runs on the RTX 3060; skips with no adapter.
//!
//! NV12 -> RGBA is a CPU conversion in this increment; the zero-copy GPU-resident
//! `VkSamplerYcbcrConversion` path is the next step.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

#[test]
fn decodes_idr_into_wgpu_rgba_texture() {
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
    let session = device.create_h264_session(&ps, 640, 480).expect("create session");

    // The wedge payload: a decoded frame as a wgpu texture on the decode device.
    let texture = device
        .decode_idr_to_rgba_texture(&session, CLIP)
        .expect("decode IDR to wgpu RGBA texture");
    assert_eq!(texture.width(), 640);
    assert_eq!(texture.height(), 480);

    // Read the texture back through wgpu (same device) and confirm it holds a
    // real decoded picture, not a cleared/garbage buffer.
    let rgba = device.read_rgba_texture(&texture);
    assert_eq!(rgba.len(), 640 * 480 * 4);

    let min = *rgba.iter().min().unwrap();
    let max = *rgba.iter().max().unwrap();
    assert!(max > min, "decoded RGBA texture is uniform ({min}=={max})");
    // The converter sets alpha to 255 (fully opaque).
    assert!(rgba.iter().skip(3).step_by(4).all(|&a| a == 255), "alpha not opaque");

    let sum: u64 = rgba.iter().map(|&b| b as u64).sum();
    let mean = sum / rgba.len() as u64;
    eprintln!("Decoded IDR -> wgpu Rgba8Unorm 640x480: range {min}..={max}, mean {mean}");
}
