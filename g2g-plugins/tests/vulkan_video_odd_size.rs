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
    Av1DpbDecoder, VulkanVideoDevice, VulkanVideoError,
};

mod vulkan_nv12_common;
mod vulkan_ref;
use vulkan_ref::{assert_frames_match_reference, planar_frames, two_plane_to_planar};

// ffmpeg -f lavfi -i testsrc=size=321x181:rate=30 -frames:v 10 -pix_fmt yuv420p -c:v libaom-av1 -cpu-used 8 -b:v 200k -f obu av1_321x181.obu
const CLIP: &[u8] = include_bytes!("fixtures/av1_321x181.obu");
const CLIP_FIXTURE: &str = "av1_321x181.obu";
// ffmpeg -f lavfi -i testsrc=size=321x181:rate=30 -frames:v 5 -pix_fmt yuv420p10le -c:v libaom-av1 -cpu-used 8 -b:v 200k -f obu av1_321x181_10bit.obu
const CLIP_10BIT: &[u8] = include_bytes!("fixtures/av1_321x181_10bit.obu");
const CLIP_10BIT_FIXTURE: &str = "av1_321x181_10bit.obu";
// ffmpeg -f lavfi -i testsrc=size=321x181:rate=30 -frames:v 9 -pix_fmt yuv420p -c:v libsvtav1 -svtav1-params film-grain=8 -f obu av1_321x181_filmgrain.obu
const CLIP_GRAIN: &[u8] = include_bytes!("fixtures/av1_321x181_filmgrain.obu");
const CLIP_GRAIN_FIXTURE: &str = "av1_321x181_filmgrain.obu";

const RGBA_PIXEL_BYTES: usize = 4;

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
        let last_column: Vec<&[u8; RGBA_PIXEL_BYTES]> = rgba
            .as_chunks::<RGBA_PIXEL_BYTES>()
            .0
            .iter()
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

/// The two-plane texture of an odd picture is one column and one row larger,
/// since Vulkan and wgpu refuse an odd 4:2:0 image, and the frame carries the
/// picture size beside it: `wgpudownload` reads the planes at that size, which
/// is the tight NV12 layout the reference compares against.
#[cfg(feature = "wgpu-sink")]
#[test]
fn the_two_plane_texture_output_carries_an_odd_picture_in_an_even_texture() {
    use std::sync::Arc;

    use g2g_plugins::vulkanvideo::TextureOutput;

    use g2g_core::frame::FrameTiming;
    use g2g_core::memory::OwnedWgpuTexture;
    use g2g_core::{
        AsyncElement, Frame, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome, Rate,
    };
    use g2g_plugins::gpu::WgpuNv12Texture;
    use g2g_plugins::wgpudownload::WgpuDownload;

    #[derive(Default)]
    struct CollectSink {
        frames: Vec<Frame>,
    }

    impl OutputSink for CollectSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
                self.frames.push(frame);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

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
    let (width, height) = picture_size(CLIP);
    let session = session(&device, CLIP);
    let Some(mut decoder) = gpu_decoder(&device, &session, CLIP) else {
        return;
    };
    decoder.set_texture_output(TextureOutput::Nv12);
    let textures = decoder
        .decode_all_to_textures(CLIP)
        .expect("decode odd-size stream to two-plane textures");
    assert!(!textures.is_empty(), "no textures decoded");

    let mut download = WgpuDownload::new();
    download
        .configure_pipeline(&Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })
        .expect("download configures");
    let mut sink = CollectSink::default();
    for (sequence, texture) in textures.into_iter().enumerate() {
        assert_eq!(texture.format(), wgpu::TextureFormat::NV12);
        assert_eq!(
            (texture.width(), texture.height()),
            (width + 1, height + 1),
            "an odd picture gets an even-rounded texture"
        );
        let keep = Arc::new(WgpuNv12Texture::new(
            device.wgpu_device.clone(),
            device.wgpu_queue.clone(),
            texture,
        ));
        let frame = Frame {
            domain: MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(width, height, keep)),
            timing: FrameTiming::default(),
            sequence: sequence as u64,
            meta: Default::default(),
        };
        block_on(download.process(PipelinePacket::DataFrame(frame), &mut sink))
            .expect("download reads the picture");
    }
    let frames: Vec<Vec<u8>> = sink
        .frames
        .iter()
        .map(|frame| {
            let bytes = frame
                .domain
                .as_system_slice()
                .expect("downloaded frame is system memory");
            two_plane_to_planar(bytes, 8, (width, height))
        })
        .collect();
    assert_frames_match_reference(&frames, RawVideoFormat::I420, (width, height), CLIP_FIXTURE);
}
