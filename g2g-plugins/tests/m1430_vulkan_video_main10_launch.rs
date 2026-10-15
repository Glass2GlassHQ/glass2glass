#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_core::G2gError;
use g2g_plugins::vulkanvideo::{
    extract_h265_parameter_sets, open_h265_decode_device, to_std_h265_params, VulkanVideoError,
};

mod vulkan_nv12_common;
use vulkan_nv12_common::launch_frames;

// ffmpeg -f lavfi -i testsrc=size=320x192:rate=30 -frames:v 10 -pix_fmt yuv420p10le -c:v libx265 -x265-params log-level=error -b:v 200k -f hevc h265_320x192_main10.hevc
const FIXTURE: &str = "h265_320x192_main10.hevc";
const CLIP: &[u8] = include_bytes!("fixtures/h265_320x192_main10.hevc");
const MAIN_10_BIT_DEPTH: u8 = 10;

// every picture of the clip through the decoder API, or `None` on a host without Vulkan H.265 decode
fn decoder_api_frames() -> Option<Vec<Vec<u8>>> {
    let device = match block_on(open_h265_decode_device()) {
        Ok(device) => device,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip: no Vulkan H.265 decode adapter");
            return None;
        }
        Err(error) => panic!("open H.265 decode device: {error:?}"),
    };
    let ps = extract_h265_parameter_sets(CLIP).expect("parse vps, sps and pps");
    let session = device
        .create_h265_session(
            &to_std_h265_params(&ps),
            ps.sps.pic_width_in_luma_samples,
            ps.sps.pic_height_in_luma_samples,
        )
        .expect("session");
    let mut decoder = device
        .create_h265_dpb_decoder(&session, &ps)
        .expect("decoder");
    let frames = decoder
        .decode_all(CLIP)
        .expect("decode")
        .into_iter()
        .map(|frame| {
            assert_eq!(frame.bit_depth, MAIN_10_BIT_DEPTH, "the fixture is Main 10");
            frame.into_data().into_vec()
        })
        .collect();
    Some(frames)
}

#[test]
fn a_launch_line_decodes_every_main_10_frame() {
    let Some(expected) = decoder_api_frames() else {
        return;
    };
    let decode = format!(
        "filesrc location={}/tests/fixtures/{FIXTURE} ! h265parse ! vulkanvideodec",
        env!("CARGO_MANIFEST_DIR"),
    );
    let frames = launch_frames(&decode, "m1430_main10").expect("the launch line runs");
    assert!(!expected.is_empty(), "the decoder API decoded no frames");
    assert_eq!(frames.len(), expected.len(), "frame count");
    assert!(
        frames == expected,
        "the launch line's frames differ from the decoder API's"
    );

    let pinned_8_bit = launch_frames(
        &format!("{decode} ! video/x-raw,format=NV12"),
        "m1430_nv12_pin",
    );
    assert_eq!(
        pinned_8_bit,
        Err(G2gError::CapsMismatch),
        "an NV12 pin on a Main 10 stream must fail, not truncate"
    );
}
