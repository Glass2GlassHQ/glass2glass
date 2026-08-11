//! M1027: a small coded extent still decodes on the GPU.
//!
//! The video session used to be created with `maxCodedExtent` set to the
//! picture's own size. The NVIDIA driver refused whole geometries outright for
//! that (64x48 and 64x64 did, 64x96 did not), failing with
//! `ERROR_INVALID_VIDEO_STD_PARAMETERS_KHR` before a single frame came out. The
//! session now declares the device's maximum, which is all that bound is, while
//! each picture resource still carries its real extent.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError,
};

/// 64x48, x264 `-bf 2`: a geometry the driver used to refuse outright.
const CLIP: &[u8] = include_bytes!("fixtures/h264_64x48_bframes.h264");

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
/// Coded pictures the clip carries.
const FRAMES: usize = 12;

// One test per file: two decode devices opened concurrently crash the driver on
// this host, and cargo runs test functions in parallel.
#[test]
fn a_64x48_clip_decodes() {
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
        .create_h264_session(&ps, WIDTH, HEIGHT)
        .expect("create session");
    let mut decoder = device
        .create_h264_dpb_decoder(&session, &ps)
        .expect("create DPB decoder");

    let frames = decoder
        .decode_all(CLIP)
        .expect("the driver accepts a 64x48 decode");

    assert_eq!(frames.len(), FRAMES, "one frame per coded picture");
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width, WIDTH);
        assert_eq!(f.height, HEIGHT);
        assert_eq!(f.luma.len(), (WIDTH * HEIGHT) as usize);
        assert!(
            f.luma.iter().any(|&p| p != f.luma[0]),
            "frame {i} decoded to real content, not a flat field"
        );
    }
}
