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

// Rotzoom global motion on LAST (later frames code it against their primary
// reference), a hidden ALTREF and CDEF secondary strengths coded as 3:
// ffmpeg -f lavfi -i testsrc=size=640x360:rate=30 -vf "rotate=0.03*n:ow=640:oh=360:c=gray,scale=w='320+4*n':h=-2:eval=frame,crop=320:180,format=yuv420p" -frames:v 4 -c:v libaom-av1 -cpu-used 1 -b:v 400k -aom-params enable-global-motion=1 -f obu av1_320x180_globalmotion.obu
const GLOBAL_MOTION_CLIP: &[u8] = include_bytes!("fixtures/av1_320x180_globalmotion.obu");
const GLOBAL_MOTION_FIXTURE: &str = "av1_320x180_globalmotion.obu";
// ffmpeg -f lavfi -i testsrc=size=320x180:rate=30 -frames:v 3 -pix_fmt yuv420p -c:v libaom-av1 -cpu-used 8 -b:v 200k -aom-params film-grain-test=1 -f obu av1_320x180_filmgrain_test1.obu
const GRAIN_CLIP: &[u8] = include_bytes!("fixtures/av1_320x180_filmgrain_test1.obu");
const GRAIN_FIXTURE: &str = "av1_320x180_filmgrain_test1.obu";
// the same with -pix_fmt yuv420p10le
const GRAIN_10BIT_CLIP: &[u8] = include_bytes!("fixtures/av1_320x180_10bit_filmgrain_test1.obu");
const GRAIN_10BIT_FIXTURE: &str = "av1_320x180_10bit_filmgrain_test1.obu";

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

fn picture_size(clip: &[u8]) -> (u32, u32) {
    let sequence = extract_av1_sequence_header(clip).expect("parse sequence header");
    (
        sequence.max_frame_width_minus_1 + 1,
        sequence.max_frame_height_minus_1 + 1,
    )
}

// decodes the whole clip on the system-memory path, `None` when the device
// cannot decode the clip's bit depth
fn decode_to_planar(device: &VulkanVideoDevice, clip: &[u8]) -> Option<Vec<Vec<u8>>> {
    let sequence = extract_av1_sequence_header(clip).expect("parse sequence header");
    let (width, height) = picture_size(clip);
    let session = device
        .create_av1_session(&to_std_av1_seq_header(&sequence), width, height)
        .expect("create AV1 session");
    let mut decoder = match device.create_av1_dpb_decoder(&session, &sequence) {
        Ok(decoder) => decoder,
        Err(VulkanVideoError::UnsupportedStream) | Err(VulkanVideoError::ExtensionUnsupported) => {
            eprintln!("skip: this device does not decode the fixture's bit depth");
            return None;
        }
        Err(error) => panic!("build AV1 decoder: {error:?}"),
    };
    let frames = decoder.decode_all(clip).expect("decode the clip");
    assert!(frames
        .iter()
        .all(|frame| frame.bit_depth == sequence.color.bit_depth));
    Some(planar_frames(frames, (width, height)))
}

#[test]
fn global_motion_and_secondary_cdef_strength_three_match_dav1d() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let Some(frames) = decode_to_planar(&device, GLOBAL_MOTION_CLIP) else {
        return;
    };
    assert_frames_match_reference(
        &frames,
        RawVideoFormat::I420,
        picture_size(GLOBAL_MOTION_CLIP),
        GLOBAL_MOTION_FIXTURE,
    );
}

#[test]
fn libaom_film_grain_test_vector_matches_dav1d() {
    let sequence = extract_av1_sequence_header(GRAIN_CLIP).expect("parse sequence header");
    assert!(
        sequence.film_grain_params_present,
        "fixture carries no film grain"
    );
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let Some(frames) = decode_to_planar(&device, GRAIN_CLIP) else {
        return;
    };
    assert_frames_match_reference(
        &frames,
        RawVideoFormat::I420,
        picture_size(GRAIN_CLIP),
        GRAIN_FIXTURE,
    );
}

#[test]
fn ten_bit_film_grain_matches_dav1d() {
    let sequence = extract_av1_sequence_header(GRAIN_10BIT_CLIP).expect("parse sequence header");
    assert!(
        sequence.film_grain_params_present,
        "fixture carries no film grain"
    );
    assert_eq!(sequence.color.bit_depth, 10, "fixture is not 10-bit");
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let Some(frames) = decode_to_planar(&device, GRAIN_10BIT_CLIP) else {
        return;
    };
    assert_frames_match_reference(
        &frames,
        RawVideoFormat::I420p10,
        picture_size(GRAIN_10BIT_CLIP),
        GRAIN_10BIT_FIXTURE,
    );
}
