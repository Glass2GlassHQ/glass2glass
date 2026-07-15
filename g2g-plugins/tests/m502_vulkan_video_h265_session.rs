//! M502: Vulkan Video H.265 decode session + parameters on real hardware.
//!
//! The HEVC sibling of M488. Open a wgpu device with a Vulkan Video H.265 decode
//! queue, parse the VPS/SPS/PPS from the real fixture (M501), map them onto the
//! `StdVideoH265*` layout, then create a `VkVideoSessionKHR` +
//! `VkVideoSessionParametersKHR`. Creating the session parameters makes the
//! driver validate the M501 `Std*` mapping (a wrong mapping fails here), so a
//! green run proves the H.265 parse + mapping is correct end to end on the GPU,
//! not just self-consistent. Runs on the RTX 3060; skips with no adapter.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h265_parameter_sets, open_h265_decode_device, to_std_h265_params, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h265_640x480.h265");

#[test]
fn creates_h265_decode_session_from_real_vps_sps_pps() {
    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("skipping: no Vulkan adapter");
            return;
        }
        Err(VulkanVideoError::ExtensionUnsupported) | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping: GPU has no Vulkan H.265 decode support");
            return;
        }
        Err(e) => panic!("failed to open H.265 decode device: {e:?}"),
    };

    let ps = extract_h265_parameter_sets(CLIP).expect("parse VPS+SPS+PPS from fixture");
    assert_eq!(ps.sps.pic_width_in_luma_samples, 640);
    assert_eq!(ps.sps.pic_height_in_luma_samples, 480);

    // Map onto the StdVideoH265* layout (owns the PTL / DPB-manager / RPS pointee
    // blocks the session reads by pointer), then let the driver validate it.
    let std = to_std_h265_params(&ps);
    let session = device
        .create_h265_session(&std, 640, 480)
        .expect("create H.265 video session + parameters");

    assert!(session.coded_extent.0 >= 640 && session.coded_extent.1 >= 480);
    // Format::UNDEFINED is 0; a concrete decode format is non-zero.
    assert_ne!(session.picture_format.as_raw(), 0, "decode picture format must be concrete");
    eprintln!(
        "Vulkan H.265 session created: picture_format={:?}, coded_extent={:?}",
        session.picture_format, session.coded_extent,
    );
}
