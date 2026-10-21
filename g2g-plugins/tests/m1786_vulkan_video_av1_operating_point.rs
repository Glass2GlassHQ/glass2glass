#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_core::RawVideoFormat;
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, open_av1_decode_device, to_std_av1_seq_header, VulkanVideoDevice,
    VulkanVideoError,
};

mod vulkan_ref;
use vulkan_ref::{assert_frames_match_reference, planar_frames};

#[path = "common/ivf.rs"]
mod ivf;

// tools/av1-operating-point-fixture.sh: a 320x256 stream of two spatial and three
// temporal layers whose operating point 0 is spatial layer 0 at temporal layers 0 and 1
const OPERATING_POINT_FIXTURE: &str = "av1_320x256_l2t3_op0_base.ivf";
const OPERATING_POINT_IVF: &[u8] = include_bytes!("fixtures/av1_320x256_l2t3_op0_base.ivf");
const BASE_LAYER_SIZE: (u32, u32) = (160, 128);
// operating point 0 leaves every other temporal unit without a frame
const OPERATING_POINT_PICTURES: usize = 3;

// libtest runs these in parallel and concurrent instance creation SIGSEGVs the NVIDIA loader
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn open_device() -> Option<VulkanVideoDevice> {
    match block_on(open_av1_decode_device()) {
        Ok(device) => Some(device),
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip: no Vulkan AV1 decode adapter");
            None
        }
        Err(error) => panic!("open AV1 decode device: {error:?}"),
    }
}

#[test]
fn layers_outside_operating_point_zero_are_dropped() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let units = ivf::ivf_frames(OPERATING_POINT_IVF);
    let stream = units.concat();
    let sequence = extract_av1_sequence_header(&stream).expect("parse sequence header");
    let session = device
        .create_av1_session(
            &to_std_av1_seq_header(&sequence),
            sequence.max_frame_width_minus_1 + 1,
            sequence.max_frame_height_minus_1 + 1,
        )
        .expect("create AV1 session");

    let mut whole = device
        .create_av1_dpb_decoder(&session, &sequence)
        .expect("AV1 decoder");
    let frames = planar_frames(
        whole.decode_all(&stream).expect("decode the stream"),
        BASE_LAYER_SIZE,
    );
    assert_eq!(frames.len(), OPERATING_POINT_PICTURES);
    assert_frames_match_reference(
        &frames,
        RawVideoFormat::I420,
        BASE_LAYER_SIZE,
        OPERATING_POINT_FIXTURE,
    );
    drop(whole);

    // the element's per-temporal-unit entry drops the same OBUs
    let mut streaming = device
        .create_av1_dpb_decoder(&session, &sequence)
        .expect("AV1 decoder");
    let mut per_unit = Vec::new();
    for unit in &units {
        match streaming.decode_display(unit) {
            Ok(decoded) => per_unit.extend(planar_frames(decoded, BASE_LAYER_SIZE)),
            // every OBU of the unit lies outside operating point 0
            Err(VulkanVideoError::NoDecodableSlice) => {}
            Err(error) => panic!("decode a temporal unit: {error:?}"),
        }
    }
    assert!(
        per_unit == frames,
        "per-unit decode differs from the whole-stream decode"
    );
}
