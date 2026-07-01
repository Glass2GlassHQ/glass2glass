//! M489: Vulkan Video H.264 IDR frame decode on real hardware.
//!
//! Third increment of `VulkanVideoDec` (DESIGN.md 4.11.6): actually decode a
//! frame. Opens the decode device, creates the session from the fixture's
//! SPS/PPS, submits the first IDR slice through `vkCmdDecodeVideoKHR`, and reads
//! back the decoded NV12 luma plane. Asserts the luma is non-uniform, i.e. the
//! GPU produced real image content, not a cleared/garbage buffer. Runs on the
//! RTX 3060; skips with no adapter / no support.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

#[test]
fn decodes_idr_frame_to_luma() {
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

    let frame = device
        .decode_idr_luma(&session, CLIP)
        .expect("decode IDR frame");

    assert_eq!(frame.width, 640);
    assert_eq!(frame.height, 480);
    assert_eq!(frame.luma.len(), 640 * 480, "one luma byte per sample");

    // A real decoded picture has varied luma; a cleared / undecoded buffer would
    // be uniform. This fails if the decode silently produced nothing.
    let min = *frame.luma.iter().min().unwrap();
    let max = *frame.luma.iter().max().unwrap();
    assert!(max > min, "decoded luma is uniform ({min}=={max}); no real content");
    // The fixture is a test card with near-black (16) and near-white regions.
    // A mid-slice CAVLC desync (e.g. a mis-parsed PPS) conceals as flat mid-grey
    // that clips well below white, so require the bright content to be present:
    // this catches a decode that is non-uniform but still wrong.
    assert!(min <= 20, "no near-black content (min {min}); decode likely wrong");
    assert!(max >= 200, "no bright content (max {max}); decode likely desynced");

    // Report the mean so the run shows a plausible picture, not just non-uniform.
    let sum: u64 = frame.luma.iter().map(|&b| b as u64).sum();
    let mean = sum / frame.luma.len() as u64;
    eprintln!(
        "Decoded IDR: {}x{} luma, range {}..={}, mean {}",
        frame.width, frame.height, min, max, mean
    );
}
