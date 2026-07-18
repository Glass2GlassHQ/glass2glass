//! M505: Vulkan Video AV1 decode session + parameters on real hardware.
//!
//! The AV1 sibling of M488 / M502. Open a wgpu device with a Vulkan Video AV1
//! decode queue, parse the sequence header from the real fixture (M504), map it
//! onto the `StdVideoAV1SequenceHeader` layout, then create a video session and
//! its session parameters. Creating the session parameters makes the driver
//! validate the M504 `Std*` mapping (a wrong mapping fails here), so a green run
//! proves the AV1 parse + mapping is correct end to end on the GPU, not just
//! self-consistent. Runs on the RTX 3060; skips with no adapter.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, open_av1_decode_device, to_std_av1_seq_header, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480.obu");

#[test]
fn creates_av1_decode_session_from_real_sequence_header() {
    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("skipping: no Vulkan adapter");
            return;
        }
        Err(VulkanVideoError::ExtensionUnsupported) | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping: GPU has no Vulkan AV1 decode support");
            return;
        }
        Err(e) => panic!("failed to open AV1 decode device: {e:?}"),
    };

    let seq = extract_av1_sequence_header(CLIP).expect("parse sequence header from fixture");
    assert_eq!(seq.max_frame_width_minus_1 + 1, 640);
    assert_eq!(seq.max_frame_height_minus_1 + 1, 480);

    // Map onto the StdVideoAV1SequenceHeader layout (owns the color-config pointee
    // block the session reads by pointer), then let the driver validate it.
    let std = to_std_av1_seq_header(&seq);
    let session = device
        .create_av1_session(&std, 640, 480)
        .expect("create AV1 video session + parameters");

    assert!(session.coded_extent.0 >= 640 && session.coded_extent.1 >= 480);
    // Format::UNDEFINED is 0; a concrete decode format is non-zero.
    assert_ne!(
        session.picture_format.as_raw(),
        0,
        "decode picture format must be concrete"
    );
    eprintln!(
        "Vulkan AV1 session created: picture_format={:?}, coded_extent={:?}",
        session.picture_format, session.coded_extent,
    );
}
