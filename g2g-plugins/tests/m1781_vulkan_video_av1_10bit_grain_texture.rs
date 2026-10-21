#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_core::RawVideoFormat;
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, open_av1_decode_device, to_std_av1_seq_header, Av1SequenceHeader,
    TextureOutput, VideoColorSpace, VulkanVideoDevice, VulkanVideoError,
};

mod vulkan_ref;
use vulkan_ref::{
    assert_frames_match_reference, luma_weights, reference_yuv, rgba16f_to_rgb, two_plane_to_planar,
};

mod vulkan_nv12_common;
use vulkan_nv12_common::read_two_plane;

// ffmpeg -f lavfi -i testsrc=size=320x180:rate=30 -frames:v 3 -pix_fmt yuv420p10le -c:v libaom-av1 -cpu-used 8 -b:v 200k -aom-params film-grain-test=1 -f obu av1_320x180_10bit_filmgrain_test1.obu
const GRAIN_10BIT_CLIP: &[u8] = include_bytes!("fixtures/av1_320x180_10bit_filmgrain_test1.obu");
const GRAIN_10BIT_FIXTURE: &str = "av1_320x180_10bit_filmgrain_test1.obu";

// AV1 codes a missing colour description as CICP 2, unspecified
const UNSPECIFIED_CICP: u8 = 2;

// the 10-bit studio range codes black at 64 and spans 876 values to white
const STUDIO_BLACK_10BIT: f32 = 64.0;
const STUDIO_LUMA_SPAN_10BIT: f32 = 876.0;

// above the half-float rounding of 0.0002, below the 0.009 the clip's grain adds
const MAX_MEAN_LUMA_ERROR: f64 = 0.002;

// libtest runs these in parallel and concurrent instance creation SIGSEGVs the NVIDIA loader
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn grain_sequence() -> Av1SequenceHeader {
    let sequence = extract_av1_sequence_header(GRAIN_10BIT_CLIP).expect("parse sequence header");
    assert!(
        sequence.film_grain_params_present,
        "fixture carries no film grain"
    );
    assert_eq!(sequence.color.bit_depth, 10, "fixture is not 10-bit");
    sequence
}

fn picture_size(sequence: &Av1SequenceHeader) -> (u32, u32) {
    (
        sequence.max_frame_width_minus_1 + 1,
        sequence.max_frame_height_minus_1 + 1,
    )
}

// the clip's textures in `output`, `None` when this host cannot produce them
fn decode_to_textures(
    device: &VulkanVideoDevice,
    output: TextureOutput,
) -> Option<Vec<wgpu::Texture>> {
    let sequence = grain_sequence();
    let (width, height) = picture_size(&sequence);
    let session = device
        .create_av1_session(&to_std_av1_seq_header(&sequence), width, height)
        .expect("create AV1 session");
    let mut decoder = match device.create_av1_dpb_decoder_gpu(&session, &sequence) {
        Ok(decoder) => decoder,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skip: no compute queue for the texture path");
            return None;
        }
        Err(VulkanVideoError::UnsupportedStream) | Err(VulkanVideoError::ExtensionUnsupported) => {
            eprintln!("skip: this device does not decode 10-bit AV1");
            return None;
        }
        Err(error) => panic!("build AV1 decoder: {error:?}"),
    };
    decoder.set_texture_output(output);
    let textures = decoder
        .decode_all_to_textures(GRAIN_10BIT_CLIP)
        .expect("decode the clip to textures");
    assert!(!textures.is_empty(), "no textures decoded");
    Some(textures)
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
fn ten_bit_film_grain_reaches_the_p010_texture_bit_exact_with_dav1d() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    if !device
        .wgpu_device
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_P010)
    {
        eprintln!("skip: the wgpu device lacks TEXTURE_FORMAT_P010");
        return;
    }
    let Some(textures) = decode_to_textures(&device, TextureOutput::Nv12) else {
        return;
    };
    let size = picture_size(&grain_sequence());
    let frames: Vec<Vec<u8>> = textures
        .iter()
        .map(|texture| {
            assert_eq!(texture.format(), wgpu::TextureFormat::P010);
            assert_eq!((texture.width(), texture.height()), size);
            let bytes = read_two_plane(&device.wgpu_device, &device.wgpu_queue, texture);
            two_plane_to_planar(&bytes, 10, size)
        })
        .collect();
    assert_frames_match_reference(&frames, RawVideoFormat::I420p10, size, GRAIN_10BIT_FIXTURE);
}

#[test]
fn ten_bit_film_grain_reaches_the_rgba_texture() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let Some(textures) = decode_to_textures(&device, TextureOutput::Rgba) else {
        return;
    };
    let sequence = grain_sequence();
    let (width, height) = picture_size(&sequence);
    for texture in &textures {
        assert_eq!(texture.format(), wgpu::TextureFormat::Rgba16Float);
        assert_eq!((texture.width(), texture.height()), (width, height));
    }
    let Some(reference) = reference_yuv(GRAIN_10BIT_FIXTURE) else {
        return;
    };
    assert!(!sequence.color.color_description_present_flag);
    let color = VideoColorSpace::from_cicp(
        UNSPECIFIED_CICP,
        UNSPECIFIED_CICP,
        sequence.color.color_range,
        height,
    );
    assert!(!color.full_range, "the studio-range constants above apply");
    let (kr, kb) = luma_weights(color.matrix);
    let kg = 1.0 - kr - kb;
    let pixels = (width * height) as usize;
    let frame_bytes = RawVideoFormat::I420p10
        .unpadded_frame_bytes(width, height)
        .expect("frame size fits") as usize;
    assert_eq!(reference.len(), frame_bytes * textures.len());
    // the luma the texture's RGB carries, compared against the grained reference
    // luma, is independent of how the converter filters chroma
    for (index, (texture, expected)) in textures
        .iter()
        .zip(reference.chunks_exact(frame_bytes))
        .enumerate()
    {
        let rgb = rgba16f_to_rgb(
            &device.read_rgba_texture(texture),
            (width as usize, height as usize),
        );
        let expected_luma = expected[..pixels * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&sample| {
                (f32::from(u16::from_le_bytes(sample)) - STUDIO_BLACK_10BIT)
                    / STUDIO_LUMA_SPAN_10BIT
            });
        let total_error: f64 = rgb
            .iter()
            .zip(expected_luma)
            .map(|([r, g, b], luma)| f64::from((kr * r + kg * g + kb * b - luma).abs()))
            .sum();
        let mean_error = total_error / pixels as f64;
        assert!(
            mean_error < MAX_MEAN_LUMA_ERROR,
            "frame {index}: mean luma error {mean_error:.5} against the grained reference"
        );
    }
}
