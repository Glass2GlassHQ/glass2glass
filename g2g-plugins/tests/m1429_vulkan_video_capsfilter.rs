#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::streamdec::{VideoCodec, VulkanStreamDecoder};
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoError};

mod vulkan_nv12_common;
use vulkan_nv12_common::{launch_frames, H264_CLIP};

const FIXTURE: &str = "h264_640x480.h264";

// the stream's picture size, or `None` on a host without Vulkan H.264 decode
fn stream_size() -> Option<(u32, u32)> {
    let device = match block_on(open_h264_decode_device()) {
        Ok(device) => device,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip: no Vulkan H.264 decode adapter");
            return None;
        }
        Err(error) => panic!("open H.264 decode device: {error:?}"),
    };
    let decoder = VulkanStreamDecoder::new(device, VideoCodec::H264, H264_CLIP)
        .expect("build stream decoder");
    Some((decoder.width(), decoder.height()))
}

#[test]
fn a_capsfilter_on_the_stream_size_decodes_every_frame() {
    let Some((width, height)) = stream_size() else {
        return;
    };
    let decode = format!(
        "filesrc location={}/tests/fixtures/{FIXTURE} ! h264parse ! vulkanvideodec",
        env!("CARGO_MANIFEST_DIR"),
    );
    let unfiltered = launch_frames(&decode, "m1429_unfiltered").expect("the unfiltered line runs");
    let filtered = launch_frames(
        &format!("{decode} ! video/x-raw,width={width},height={height}"),
        "m1429_filtered",
    )
    .expect("the filtered line runs");
    assert!(!unfiltered.is_empty(), "no frames decoded");
    assert_eq!(filtered, unfiltered);
}
