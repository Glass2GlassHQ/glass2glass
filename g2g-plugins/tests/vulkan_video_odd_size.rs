#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::memory::MemoryDomainKind;
use g2g_core::runtime::block_on;
use g2g_core::{Caps, Dim, RawVideoFormat, VideoCodec};
use g2g_plugins::streamdec::{VideoCodec as StreamCodec, VulkanStreamDecoder};
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, open_av1_decode_device, to_std_av1_seq_header, Av1DecodeSession,
    Av1DpbDecoder, Nv12Frame, TextureOutput, VulkanVideoDevice, VulkanVideoError,
};

mod vulkan_nv12_common;
mod vulkan_ref;
use vulkan_ref::reference_yuv;

// ffmpeg -f lavfi -i testsrc=size=321x181:rate=30 -frames:v 10 -pix_fmt yuv420p -c:v libaom-av1 -cpu-used 8 -b:v 200k -f obu av1_321x181.obu
const CLIP: &[u8] = include_bytes!("fixtures/av1_321x181.obu");
const CLIP_FIXTURE: &str = "av1_321x181.obu";
// ffmpeg -f lavfi -i testsrc=size=321x181:rate=30 -frames:v 5 -pix_fmt yuv420p10le -c:v libaom-av1 -cpu-used 8 -b:v 200k -f obu av1_321x181_10bit.obu
const CLIP_10BIT: &[u8] = include_bytes!("fixtures/av1_321x181_10bit.obu");
const CLIP_10BIT_FIXTURE: &str = "av1_321x181_10bit.obu";
// ffmpeg -f lavfi -i testsrc=size=321x181:rate=30 -frames:v 9 -pix_fmt yuv420p -c:v libsvtav1 -svtav1-params film-grain=8 -f obu av1_321x181_filmgrain.obu
const CLIP_GRAIN: &[u8] = include_bytes!("fixtures/av1_321x181_filmgrain.obu");
const CLIP_GRAIN_FIXTURE: &str = "av1_321x181_filmgrain.obu";

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
    let size = (
        sequence.max_frame_width_minus_1 + 1,
        sequence.max_frame_height_minus_1 + 1,
    );
    assert!(
        !size.0.is_multiple_of(2) && !size.1.is_multiple_of(2),
        "fixture {size:?} is not odd in both directions"
    );
    size
}

fn session(device: &VulkanVideoDevice, clip: &[u8]) -> Av1DecodeSession {
    let sequence = extract_av1_sequence_header(clip).expect("parse sequence header");
    let (width, height) = picture_size(clip);
    device
        .create_av1_session(&to_std_av1_seq_header(&sequence), width, height)
        .expect("an odd-size AV1 session opens")
}

fn system_decoder(
    device: &VulkanVideoDevice,
    session: &Av1DecodeSession,
    clip: &[u8],
) -> Option<Av1DpbDecoder> {
    let sequence = extract_av1_sequence_header(clip).expect("parse sequence header");
    match device.create_av1_dpb_decoder(session, &sequence) {
        Ok(decoder) => Some(decoder),
        Err(VulkanVideoError::UnsupportedStream) | Err(VulkanVideoError::ExtensionUnsupported) => {
            eprintln!("skip: this device does not decode the fixture's bit depth");
            None
        }
        Err(error) => panic!("build AV1 decoder: {error:?}"),
    }
}

fn gpu_decoder(
    device: &VulkanVideoDevice,
    session: &Av1DecodeSession,
    clip: &[u8],
) -> Option<Av1DpbDecoder> {
    let sequence = extract_av1_sequence_header(clip).expect("parse sequence header");
    match device.create_av1_dpb_decoder_gpu(session, &sequence) {
        Ok(decoder) => Some(decoder),
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skip: no distinct compute queue for the GPU-texture path");
            None
        }
        Err(error) => panic!("build GPU AV1 decoder: {error:?}"),
    }
}

// g2g's tight NV12 / P010 frame rewritten as ffmpeg's planar dump of the same picture
fn two_plane_to_planar(frame: &[u8], bit_depth: u8, (width, height): (u32, u32)) -> Vec<u8> {
    let two_plane = if bit_depth > 8 {
        RawVideoFormat::P010
    } else {
        RawVideoFormat::Nv12
    };
    let luma_bytes = two_plane.plane_bytes(0, width, height).expect("plane fits") as usize;
    let chroma_bytes = two_plane.plane_bytes(1, width, height).expect("plane fits") as usize;
    assert_eq!(frame.len(), luma_bytes + chroma_bytes, "frame is not tight");
    let sample_bytes = two_plane.bytes_per_sample();
    let unused_low_bits = u16::BITS - u32::from(bit_depth);
    let planar_sample = |sample: &[u8]| -> Vec<u8> {
        match sample {
            [byte] => vec![*byte],
            [low, high] => (u16::from_le_bytes([*low, *high]) >> unused_low_bits)
                .to_le_bytes()
                .to_vec(),
            _ => unreachable!("one or two bytes per sample"),
        }
    };
    let (luma, chroma) = frame.split_at(luma_bytes);
    let pairs = chroma.chunks_exact(2 * sample_bytes);
    let cb = pairs.clone().map(|pair| &pair[..sample_bytes]);
    let cr = pairs.map(|pair| &pair[sample_bytes..]);
    luma.chunks_exact(sample_bytes)
        .chain(cb)
        .chain(cr)
        .flat_map(planar_sample)
        .collect()
}

fn planar_frames(frames: Vec<Nv12Frame>, size: (u32, u32)) -> Vec<Vec<u8>> {
    frames
        .into_iter()
        .map(|frame| {
            assert_eq!((frame.width, frame.height), size);
            two_plane_to_planar(&[frame.luma, frame.chroma].concat(), frame.bit_depth, size)
        })
        .collect()
}

// sizes always, every sample against the ffmpeg dump when one is configured
fn assert_frames_match_reference(
    frames: &[Vec<u8>],
    planar: RawVideoFormat,
    (width, height): (u32, u32),
    fixture: &str,
) {
    assert!(!frames.is_empty(), "no frames decoded");
    let frame_bytes = planar
        .unpadded_frame_bytes(width, height)
        .expect("frame size fits") as usize;
    for (index, frame) in frames.iter().enumerate() {
        assert_eq!(frame.len(), frame_bytes, "frame {index} has the wrong size");
    }
    let Some(reference) = reference_yuv(fixture) else {
        return;
    };
    assert_eq!(
        reference.len(),
        frame_bytes * frames.len(),
        "the reference holds a different frame count"
    );
    for (index, (frame, expected)) in frames
        .iter()
        .zip(reference.chunks_exact(frame_bytes))
        .enumerate()
    {
        let differing = frame.iter().zip(expected).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "frame {index}: {differing} bytes differ");
    }
    eprintln!("{fixture}: {} frames bit-exact", frames.len());
}

#[test]
fn an_odd_size_av1_stream_decodes_to_system_memory() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let size = picture_size(CLIP);
    let session = session(&device, CLIP);
    let Some(mut decoder) = system_decoder(&device, &session, CLIP) else {
        return;
    };
    let frames = decoder.decode_all(CLIP).expect("decode odd-size stream");
    assert!(frames.iter().all(|frame| frame.bit_depth == 8));
    assert_frames_match_reference(
        &planar_frames(frames, size),
        RawVideoFormat::I420,
        size,
        CLIP_FIXTURE,
    );
}

#[test]
fn an_odd_size_10bit_av1_stream_decodes_to_p010() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let size = picture_size(CLIP_10BIT);
    let session = session(&device, CLIP_10BIT);
    let Some(mut decoder) = system_decoder(&device, &session, CLIP_10BIT) else {
        return;
    };
    let frames = decoder
        .decode_all(CLIP_10BIT)
        .expect("decode odd-size 10-bit stream");
    assert!(frames.iter().all(|frame| frame.bit_depth == 10));
    assert_frames_match_reference(
        &planar_frames(frames, size),
        RawVideoFormat::I420p10,
        size,
        CLIP_10BIT_FIXTURE,
    );
}

#[test]
fn odd_size_film_grain_matches_dav1d() {
    let sequence = extract_av1_sequence_header(CLIP_GRAIN).expect("parse sequence header");
    assert!(
        sequence.film_grain_params_present,
        "fixture carries no film grain"
    );
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let size = picture_size(CLIP_GRAIN);
    let session = session(&device, CLIP_GRAIN);
    let Some(mut decoder) = system_decoder(&device, &session, CLIP_GRAIN) else {
        return;
    };
    let frames = decoder
        .decode_all(CLIP_GRAIN)
        .expect("decode odd-size film-grain stream");
    assert_frames_match_reference(
        &planar_frames(frames, size),
        RawVideoFormat::I420,
        size,
        CLIP_GRAIN_FIXTURE,
    );
}

#[test]
fn the_element_emits_odd_size_nv12_caps_and_frames() {
    let _gpu = gpu_lock();
    if open_device().is_none() {
        return;
    }
    let size = picture_size(CLIP);
    let (_element, collect, decoded) =
        vulkan_nv12_common::decode_stream(VideoCodec::Av1, CLIP, MemoryDomainKind::System, None);
    decoded.expect("decode through the element");
    let caps = collect.caps_changes();
    let Some(Caps::RawVideo {
        format,
        width,
        height,
        ..
    }) = caps.last()
    else {
        panic!("no raw video caps emitted: {caps:?}");
    };
    assert_eq!(*format, RawVideoFormat::Nv12);
    assert_eq!((width, height), (&Dim::Fixed(size.0), &Dim::Fixed(size.1)));
    let frames: Vec<Vec<u8>> = vulkan_nv12_common::system_bytes(&collect)
        .iter()
        .map(|frame| two_plane_to_planar(frame, 8, size))
        .collect();
    assert_frames_match_reference(&frames, RawVideoFormat::I420, size, CLIP_FIXTURE);
}

#[test]
fn the_stream_decoder_splits_odd_size_chroma_into_i420() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let size = picture_size(CLIP);
    let mut decoder =
        VulkanStreamDecoder::new(device, StreamCodec::Av1, CLIP).expect("build stream decoder");
    let frames: Vec<Vec<u8>> = decoder
        .submit_chunk(CLIP, true)
        .expect("decode odd-size stream")
        .into_iter()
        .map(|frame| frame.data)
        .collect();
    assert_frames_match_reference(&frames, RawVideoFormat::I420, size, CLIP_FIXTURE);
}

#[test]
fn an_odd_size_picture_converts_to_an_rgba_texture() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    let (width, height) = picture_size(CLIP);
    let session = session(&device, CLIP);
    let Some(mut decoder) = gpu_decoder(&device, &session, CLIP) else {
        return;
    };
    let textures = decoder
        .decode_all_to_textures(CLIP)
        .expect("decode odd-size stream to textures");
    assert!(!textures.is_empty(), "no textures decoded");
    for (index, texture) in textures.iter().enumerate() {
        assert_eq!((texture.width(), texture.height()), (width, height));
        assert_eq!(texture.format(), wgpu::TextureFormat::Rgba8Unorm);
        let rgba = device.read_rgba_texture(texture);
        let last_column: Vec<&[u8]> = rgba
            .chunks_exact(4)
            .skip(width as usize - 1)
            .step_by(width as usize)
            .collect();
        assert_eq!(last_column.len(), height as usize);
        let distinct = last_column
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        assert!(distinct > 1, "texture {index} has a flat last column");
    }
}

#[test]
fn the_two_plane_texture_output_refuses_an_odd_size() {
    let _gpu = gpu_lock();
    let Some(device) = open_device() else {
        return;
    };
    if !device
        .wgpu_device
        .features()
        .contains(wgpu::Features::TEXTURE_FORMAT_NV12)
    {
        eprintln!("skip: wgpu device lacks TEXTURE_FORMAT_NV12");
        return;
    }
    let session = session(&device, CLIP);
    let Some(mut decoder) = gpu_decoder(&device, &session, CLIP) else {
        return;
    };
    decoder.set_texture_output(TextureOutput::Nv12);
    assert!(matches!(
        decoder.decode_all_to_textures(CLIP),
        Err(VulkanVideoError::UnsupportedStream)
    ));
}
