//! M488: Vulkan Video H.264 decode session + parameters on real hardware.
//!
//! Second increment of `VulkanVideoDec` (DESIGN.md 4.11.6): open a wgpu device
//! with a Vulkan Video decode queue, then create a `VkVideoSessionKHR` +
//! `VkVideoSessionParametersKHR` from the real fixture's SPS/PPS. Creating the
//! session parameters makes the driver validate the `Std*` SPS/PPS mapping
//! (M487), so a green run here proves that mapping is correct end to end on the
//! GPU, not just self-consistent. Runs on the RTX 3060; skips with no adapter.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

#[test]
fn creates_h264_decode_session_from_real_sps_pps() {
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

    let ps = extract_h264_parameter_sets(CLIP).expect("parse SPS+PPS from fixture");

    // The driver validates the Std SPS/PPS mapping here; a failure means the
    // mapping (M487) is wrong, not just that the GPU lacks support.
    let session = device
        .create_h264_session(&ps, 640, 480)
        .expect("create H.264 video session + parameters");

    // Session sized within the device's coded-extent envelope, decoding into a
    // real (non-undefined) picture format.
    assert!(session.coded_extent.0 >= 640 && session.coded_extent.1 >= 480);
    // Format::UNDEFINED is 0; a concrete decode format is non-zero. (Compared
    // via the inherent `as_raw` so the test needs no direct ash dependency.)
    assert_ne!(session.picture_format.as_raw(), 0, "decode picture format must be concrete");
    eprintln!(
        "Vulkan H.264 session created: picture_format={:?}, coded_extent={:?}",
        session.picture_format, session.coded_extent,
    );
}
