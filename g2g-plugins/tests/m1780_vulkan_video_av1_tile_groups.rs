#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_core::RawVideoFormat;
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, open_av1_decode_device, to_std_av1_seq_header, Av1DecodeSession,
    Av1DpbDecoder, VulkanVideoDevice, VulkanVideoError,
};

mod vulkan_ref;
use vulkan_ref::{assert_frames_match_reference, planar_frames};

#[path = "common/ivf.rs"]
mod ivf;

// av1-1-b8-22-svc-L2T2 cut to two temporal units: a 640x360 base layer and a
// 1280x720 enhancement layer predicted from it, each frame coded as a frame
// header OBU plus tile group OBUs
const SPATIAL_LAYERS_FIXTURE: &str = "av1-1-b8-22-svc-L2T2_2tu.ivf";
const SPATIAL_LAYERS_IVF: &[u8] = include_bytes!("fixtures/av1-1-b8-22-svc-L2T2_2tu.ivf");

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

fn picture_size(stream: &[u8]) -> (u32, u32) {
    let sequence = extract_av1_sequence_header(stream).expect("parse sequence header");
    (
        sequence.max_frame_width_minus_1 + 1,
        sequence.max_frame_height_minus_1 + 1,
    )
}

fn temporal_units() -> Vec<&'static [u8]> {
    ivf::ivf_frames(SPATIAL_LAYERS_IVF)
}

enum Output {
    System,
    Texture,
}

// the decoder only copies the session's handles, so the session must outlive it
fn create_session(device: &VulkanVideoDevice, stream: &[u8]) -> Av1DecodeSession {
    let sequence = extract_av1_sequence_header(stream).expect("parse sequence header");
    let (width, height) = picture_size(stream);
    device
        .create_av1_session(&to_std_av1_seq_header(&sequence), width, height)
        .expect("create AV1 session")
}

// `None` when the device has no compute queue for the texture path
fn build_decoder(
    device: &VulkanVideoDevice,
    session: &Av1DecodeSession,
    stream: &[u8],
    output: Output,
) -> Option<Av1DpbDecoder> {
    let sequence = extract_av1_sequence_header(stream).expect("parse sequence header");
    let built = match output {
        Output::System => device.create_av1_dpb_decoder(session, &sequence),
        Output::Texture => device.create_av1_dpb_decoder_gpu(session, &sequence),
    };
    match built {
        Ok(decoder) => Some(decoder),
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skip: no compute queue for the texture path");
            None
        }
        Err(error) => panic!("build AV1 decoder: {error:?}"),
    }
}

#[test]
fn split_frames_decode_to_their_highest_spatial_layer() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let units = temporal_units();
    let stream = units.concat();
    let size = picture_size(&stream);

    let session = create_session(&device, &stream);
    let mut whole =
        build_decoder(&device, &session, &stream, Output::System).expect("system decoder");
    let frames = planar_frames(whole.decode_all(&stream).expect("decode the stream"), size);
    assert_eq!(frames.len(), units.len(), "one picture per temporal unit");
    assert_frames_match_reference(&frames, RawVideoFormat::I420, size, SPATIAL_LAYERS_FIXTURE);
    drop(whole);

    // the element's per-temporal-unit entry produces the same pictures
    let mut streaming =
        build_decoder(&device, &session, &stream, Output::System).expect("system decoder");
    let mut per_unit = Vec::new();
    for unit in &units {
        let decoded = streaming
            .decode_display(unit)
            .expect("decode a temporal unit");
        assert_eq!(decoded.len(), 1, "one picture per temporal unit");
        per_unit.extend(planar_frames(decoded, size));
    }
    assert!(
        per_unit == frames,
        "per-unit decode differs from the whole-stream decode"
    );
}

#[test]
fn split_frames_reach_the_texture_path_once_per_temporal_unit() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let units = temporal_units();
    let stream = units.concat();
    let (width, height) = picture_size(&stream);
    let session = create_session(&device, &stream);
    let Some(mut decoder) = build_decoder(&device, &session, &stream, Output::Texture) else {
        return;
    };
    for unit in &units {
        let textures = decoder
            .decode_display_to_textures(unit)
            .expect("decode a temporal unit to textures");
        assert_eq!(textures.len(), 1, "one texture per temporal unit");
        assert_eq!((textures[0].width(), textures[0].height()), (width, height));
    }
}
