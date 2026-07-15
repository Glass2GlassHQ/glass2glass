//! M494: full-DPB H.264 decode straight to GPU-resident RGBA `wgpu::Texture`s.
//!
//! The zero-copy wedge path: `H264DpbDecoder::decode_all_to_textures` decodes
//! every picture (with DPB reference management, M492) and converts each decoded
//! slot in place via the `VkSamplerYcbcrConversion` compute pass into a wgpu
//! texture, so the frame never leaves the GPU (unlike the system-NV12 path).
//! Reads each texture back through wgpu and asserts real, per-frame-distinct
//! content. Runs on the RTX 3060; skips with no adapter / no compute queue.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

#[test]
fn decodes_whole_stream_to_wgpu_textures() {
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
    let mut decoder = match device.create_h264_dpb_decoder_gpu(&session, &ps) {
        Ok(d) => d,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skipping: no distinct compute queue for the GPU-resident path");
            return;
        }
        Err(e) => panic!("create GPU DPB decoder: {e:?}"),
    };

    let textures = decoder.decode_all_to_textures(CLIP).expect("decode to textures");
    assert_eq!(textures.len(), 10, "one texture per coded picture");

    let mut readbacks = Vec::new();
    for (i, tex) in textures.iter().enumerate() {
        assert_eq!(tex.width(), 640);
        assert_eq!(tex.height(), 480);
        assert_eq!(tex.format(), wgpu::TextureFormat::Rgba8Unorm);
        let rgba = device.read_rgba_texture(tex);
        assert_eq!(rgba.len(), 640 * 480 * 4);
        // Real content: the test card has near-black and bright regions; a failed
        // GPU convert would be uniform / flat.
        let min = *rgba.iter().min().unwrap();
        let max = *rgba.iter().max().unwrap();
        assert!(min <= 20 && max >= 200, "frame {i} RGBA range {min}..={max} not a real picture");
        readbacks.push(rgba);
    }

    // P frames differ from their GOP's IDR (inter prediction ran through the DPB,
    // and the GPU ycbcr restored each slot as a valid reference). GOPs at 0 / 5.
    for gop_start in [0usize, 5] {
        for p in 1..5 {
            assert_ne!(
                readbacks[gop_start + p], readbacks[gop_start],
                "texture {} is identical to its IDR; DPB reference decode failed",
                gop_start + p
            );
        }
    }
    eprintln!("decoded {} GPU-resident RGBA textures (640x480), all real content", textures.len());
}
