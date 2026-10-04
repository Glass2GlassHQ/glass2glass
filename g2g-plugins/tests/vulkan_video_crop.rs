#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::memory::MemoryDomainKind;
use g2g_core::runtime::{block_on, parse_launch, run_graph};
use g2g_core::{Caps, Dim, PipelineClock, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::appsink::register_appsink_pull;
use g2g_plugins::registry::default_registry;
use g2g_plugins::streamdec::{VideoCodec as StreamCodec, VulkanStreamDecoder};
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, extract_h265_parameter_sets, open_h264_decode_device,
    open_h265_decode_device, to_std_h265_params, Nv12Frame, VulkanVideoDevice, VulkanVideoError,
    VulkanVideoPlayer,
};

mod vulkan_nv12_common;
mod vulkan_ref;
use vulkan_nv12_common::{decode_stream, read_two_plane, system_bytes, two_plane_frame};
use vulkan_ref::{assert_frames_match_reference, planar_frames, two_plane_to_planar};

#[derive(Clone, Copy)]
struct Clip {
    file: &'static str,
    bytes: &'static [u8],
    codec: VideoCodec,
    bit_depth: u8,
    size: (u32, u32),
}

struct Crop {
    left: u32,
    right: u32,
    top: u32,
    bottom: u32,
}

// the coded picture of the crop fixtures and their uncropped twins
const CODED_SIZE: (u32, u32) = (320, 192);
// the offsets the h264_metadata / hevc_metadata commands below write, in luma samples
const CROP: Crop = Crop {
    left: 6,
    right: 10,
    top: 4,
    bottom: 8,
};
const CROPPED_SIZE: (u32, u32) = (
    CODED_SIZE.0 - CROP.left - CROP.right,
    CODED_SIZE.1 - CROP.top - CROP.bottom,
);

// ffmpeg -f lavfi -i testsrc=size=320x180:rate=30 -frames:v 10 -pix_fmt yuv420p -c:v libx264 -preset veryfast -b:v 200k -f h264 h264_320x180.h264
const H264_BOTTOM_CROP: Clip = Clip {
    file: "h264_320x180.h264",
    bytes: include_bytes!("fixtures/h264_320x180.h264"),
    codec: VideoCodec::H264,
    bit_depth: 8,
    size: (320, 180),
};
// ffmpeg -f lavfi -i testsrc=size=320x192:rate=30 -frames:v 10 -pix_fmt yuv420p -c:v libx264 -preset veryfast -b:v 200k -bsf:v h264_metadata=crop_left=6:crop_right=10:crop_top=4:crop_bottom=8 -f h264 h264_304x180_crop.h264
const H264_ALL_SIDES_CROP: Clip = Clip {
    file: "h264_304x180_crop.h264",
    bytes: include_bytes!("fixtures/h264_304x180_crop.h264"),
    codec: VideoCodec::H264,
    bit_depth: 8,
    size: CROPPED_SIZE,
};
// ffmpeg -i h264_304x180_crop.h264 -c copy -bsf:v h264_metadata=crop_left=0:crop_right=0:crop_top=0:crop_bottom=0 -f h264 h264_320x192.h264
const H264_UNCROPPED: &[u8] = include_bytes!("fixtures/h264_320x192.h264");
// ffmpeg -f lavfi -i testsrc=size=320x180:rate=30 -frames:v 10 -pix_fmt yuv420p -c:v libx265 -x265-params log-level=error -b:v 200k -f hevc h265_320x180.hevc
const H265_CONFORMANCE_WINDOW: Clip = Clip {
    file: "h265_320x180.hevc",
    bytes: include_bytes!("fixtures/h265_320x180.hevc"),
    codec: VideoCodec::H265,
    bit_depth: 8,
    size: (320, 180),
};
// ffmpeg -f lavfi -i testsrc=size=320x192:rate=30 -frames:v 10 -pix_fmt yuv420p10le -c:v libx265 -x265-params log-level=error -b:v 200k -bsf:v hevc_metadata=crop_left=6:crop_right=10:crop_top=4:crop_bottom=8 -f hevc h265_304x180_main10_crop.hevc
const H265_10BIT_ALL_SIDES_CROP: Clip = Clip {
    file: "h265_304x180_main10_crop.hevc",
    bytes: include_bytes!("fixtures/h265_304x180_main10_crop.hevc"),
    codec: VideoCodec::H265,
    bit_depth: 10,
    size: CROPPED_SIZE,
};
// ffmpeg -i h265_304x180_main10_crop.hevc -c copy -bsf:v hevc_metadata=crop_left=0:crop_right=0:crop_top=0:crop_bottom=0 -f hevc h265_320x192_main10.hevc
const H265_10BIT_UNCROPPED: &[u8] = include_bytes!("fixtures/h265_320x192_main10.hevc");

const CLIPS: [Clip; 4] = [
    H264_BOTTOM_CROP,
    H264_ALL_SIDES_CROP,
    H265_CONFORMANCE_WINDOW,
    H265_10BIT_ALL_SIDES_CROP,
];

// the rate in the generating commands
const FRAMES_PER_SECOND: u32 = 30;
const LINK_CAPACITY: usize = 4;

// libtest runs these in parallel and concurrent instance creation SIGSEGVs the NVIDIA loader
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn open_device(codec: VideoCodec) -> Option<VulkanVideoDevice> {
    let opened = match codec {
        VideoCodec::H264 => block_on(open_h264_decode_device()),
        VideoCodec::H265 => block_on(open_h265_decode_device()),
        other => panic!("no crop fixture for {other:?}"),
    };
    match opened {
        Ok(device) => Some(device),
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip: no Vulkan {codec:?} decode adapter");
            None
        }
        Err(error) => panic!("open {codec:?} decode device: {error:?}"),
    }
}

fn planar_format(bit_depth: u8) -> RawVideoFormat {
    if bit_depth > 8 {
        RawVideoFormat::I420p10
    } else {
        RawVideoFormat::I420
    }
}

fn two_plane_format(bit_depth: u8) -> RawVideoFormat {
    if bit_depth > 8 {
        RawVideoFormat::P010
    } else {
        RawVideoFormat::Nv12
    }
}

// every frame of `bytes` through the codec's DPB decoder, read back to system memory
fn decode_system(device: &VulkanVideoDevice, codec: VideoCodec, bytes: &[u8]) -> Vec<Nv12Frame> {
    match codec {
        VideoCodec::H264 => {
            let ps = extract_h264_parameter_sets(bytes).expect("parse sps and pps");
            let coded = (
                (ps.sps.pic_width_in_mbs_minus1 + 1) * 16,
                (ps.sps.pic_height_in_map_units_minus1 + 1) * 16,
            );
            let session = device
                .create_h264_session(&ps, coded.0, coded.1)
                .expect("session");
            let mut decoder = device
                .create_h264_dpb_decoder(&session, &ps)
                .expect("decoder");
            decoder.decode_all(bytes).expect("decode")
        }
        VideoCodec::H265 => {
            let ps = extract_h265_parameter_sets(bytes).expect("parse vps, sps and pps");
            let std = to_std_h265_params(&ps);
            let session = device
                .create_h265_session(
                    &std,
                    ps.sps.pic_width_in_luma_samples,
                    ps.sps.pic_height_in_luma_samples,
                )
                .expect("session");
            let mut decoder = device
                .create_h265_dpb_decoder(&session, &ps)
                .expect("decoder");
            decoder.decode_all(bytes).expect("decode")
        }
        other => panic!("no crop fixture for {other:?}"),
    }
}

// every frame of `bytes` through the codec's ycbcr compute pass, read back as RGBA
fn decode_rgba(device: &VulkanVideoDevice, codec: VideoCodec, bytes: &[u8]) -> Vec<RgbaPicture> {
    let textures = match codec {
        VideoCodec::H264 => {
            let ps = extract_h264_parameter_sets(bytes).expect("parse sps and pps");
            let coded = (
                (ps.sps.pic_width_in_mbs_minus1 + 1) * 16,
                (ps.sps.pic_height_in_map_units_minus1 + 1) * 16,
            );
            let session = device
                .create_h264_session(&ps, coded.0, coded.1)
                .expect("session");
            let mut decoder = device
                .create_h264_dpb_decoder_gpu(&session, &ps)
                .expect("gpu decoder");
            decoder.decode_all_to_textures(bytes).expect("decode")
        }
        VideoCodec::H265 => {
            let ps = extract_h265_parameter_sets(bytes).expect("parse vps, sps and pps");
            let std = to_std_h265_params(&ps);
            let session = device
                .create_h265_session(
                    &std,
                    ps.sps.pic_width_in_luma_samples,
                    ps.sps.pic_height_in_luma_samples,
                )
                .expect("session");
            let mut decoder = device
                .create_h265_dpb_decoder_gpu(&session, &ps)
                .expect("gpu decoder");
            decoder.decode_all_to_textures(bytes).expect("decode")
        }
        other => panic!("no crop fixture for {other:?}"),
    };
    textures
        .iter()
        .map(|texture| RgbaPicture::read(device, texture))
        .collect()
}

struct RgbaPicture {
    size: (u32, u32),
    pixel_bytes: usize,
    bytes: Vec<u8>,
}

impl RgbaPicture {
    fn read(device: &VulkanVideoDevice, texture: &wgpu::Texture) -> Self {
        Self::new(texture, device.read_rgba_texture(texture))
    }

    fn new(texture: &wgpu::Texture, bytes: Vec<u8>) -> Self {
        Self {
            size: (texture.width(), texture.height()),
            pixel_bytes: texture
                .format()
                .block_copy_size(None)
                .expect("an RGBA format has a texel size") as usize,
            bytes,
        }
    }

    // the `size` rectangle at `(left, top)` of this picture
    fn crop(&self, (left, top): (u32, u32), (width, height): (u32, u32)) -> Vec<u8> {
        let row_bytes = self.size.0 as usize * self.pixel_bytes;
        (top..top + height)
            .flat_map(|row| {
                let start = row as usize * row_bytes + left as usize * self.pixel_bytes;
                &self.bytes[start..start + width as usize * self.pixel_bytes]
            })
            .copied()
            .collect()
    }
}

fn assert_rgba_is_the_crop_of(cropped: &[RgbaPicture], uncropped: &[RgbaPicture]) {
    assert!(!cropped.is_empty(), "no pictures decoded");
    assert_eq!(cropped.len(), uncropped.len(), "frame counts differ");
    for (index, (picture, whole)) in cropped.iter().zip(uncropped).enumerate() {
        assert_eq!(picture.size, CROPPED_SIZE, "picture {index} size");
        assert_eq!(whole.size, CODED_SIZE, "uncropped picture {index} size");
        assert!(
            picture.bytes == whole.crop((CROP.left, CROP.top), CROPPED_SIZE),
            "picture {index} is not the crop rectangle of the uncropped picture"
        );
    }
}

fn reference_check(clip: Clip, two_plane_frames: &[Vec<u8>]) {
    let planar: Vec<Vec<u8>> = two_plane_frames
        .iter()
        .map(|frame| two_plane_to_planar(frame, clip.bit_depth, clip.size))
        .collect();
    assert_frames_match_reference(&planar, planar_format(clip.bit_depth), clip.size, clip.file);
}

#[test]
fn the_dpb_decoders_read_back_the_crop_rectangle() {
    let _gpu = gpu_lock();
    for clip in CLIPS {
        let Some(device) = open_device(clip.codec) else {
            return;
        };
        let frames = decode_system(&device, clip.codec, clip.bytes);
        assert!(frames.iter().all(|frame| frame.bit_depth == clip.bit_depth));
        assert_frames_match_reference(
            &planar_frames(frames, clip.size),
            planar_format(clip.bit_depth),
            clip.size,
            clip.file,
        );
    }
}

#[test]
fn the_stream_decoder_reports_and_returns_the_crop_rectangle() {
    let _gpu = gpu_lock();
    for clip in [
        H264_BOTTOM_CROP,
        H264_ALL_SIDES_CROP,
        H265_CONFORMANCE_WINDOW,
    ] {
        let Some(device) = open_device(clip.codec) else {
            return;
        };
        let codec = match clip.codec {
            VideoCodec::H264 => StreamCodec::H264,
            _ => StreamCodec::H265,
        };
        let mut decoder =
            VulkanStreamDecoder::new(device, codec, clip.bytes).expect("build stream decoder");
        assert_eq!(
            (decoder.width(), decoder.height()),
            clip.size,
            "{}",
            clip.file
        );
        let frames: Vec<Vec<u8>> = decoder
            .submit_chunk(clip.bytes, true)
            .expect("decode")
            .into_iter()
            .map(|frame| {
                assert_eq!((frame.width, frame.height), clip.size, "{}", clip.file);
                frame.data
            })
            .collect();
        assert_frames_match_reference(&frames, RawVideoFormat::I420, clip.size, clip.file);
    }
}

#[test]
fn the_element_emits_cropped_caps_and_frames() {
    let _gpu = gpu_lock();
    for clip in CLIPS {
        if open_device(clip.codec).is_none() {
            return;
        }
        let (_element, collect, decoded) =
            decode_stream(clip.codec, clip.bytes, MemoryDomainKind::System, None);
        decoded.expect("decode through the element");
        let caps = collect.caps_changes();
        assert!(!caps.is_empty(), "no caps emitted");
        for announced in &caps {
            let Caps::RawVideo {
                format,
                width,
                height,
                ..
            } = announced
            else {
                panic!("not raw video caps: {announced:?}");
            };
            assert_eq!(*format, two_plane_format(clip.bit_depth));
            assert_eq!(
                (width, height),
                (&Dim::Fixed(clip.size.0), &Dim::Fixed(clip.size.1)),
                "{}",
                clip.file
            );
        }
        reference_check(clip, &system_bytes(&collect));
    }
}

#[test]
fn the_elements_two_plane_texture_is_the_crop_rectangle() {
    let _gpu = gpu_lock();
    for clip in [H264_ALL_SIDES_CROP, H265_10BIT_ALL_SIDES_CROP] {
        let Some(device) = open_device(clip.codec) else {
            return;
        };
        let format = two_plane_format(clip.bit_depth);
        let texture_format = match format {
            RawVideoFormat::P010 => wgpu::TextureFormat::P010,
            _ => wgpu::TextureFormat::NV12,
        };
        if !device
            .wgpu_device
            .features()
            .contains(texture_format.required_features())
        {
            eprintln!("skip: wgpu device lacks {texture_format:?}");
            return;
        }
        drop(device);
        let pinned = Caps::RawVideo {
            format,
            width: Dim::Fixed(clip.size.0),
            height: Dim::Fixed(clip.size.1),
            framerate: Rate::Fixed(FRAMES_PER_SECOND << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let (_element, collect, decoded) = decode_stream(
            clip.codec,
            clip.bytes,
            MemoryDomainKind::WgpuTexture,
            Some(pinned),
        );
        decoded.expect("decode through the element");
        let frames: Vec<Vec<u8>> = collect
            .frames()
            .iter()
            .map(|frame| {
                let owner = two_plane_frame(frame);
                let texture = owner.texture();
                assert_eq!(texture.format(), texture_format);
                assert_eq!((texture.width(), texture.height()), clip.size);
                read_two_plane(owner.device(), owner.queue(), texture)
            })
            .collect();
        reference_check(clip, &frames);
    }
}

#[test]
fn the_rgba_texture_is_the_crop_rectangle_of_the_uncropped_picture() {
    let _gpu = gpu_lock();
    for (clip, uncropped) in [
        (H264_ALL_SIDES_CROP, H264_UNCROPPED),
        (H265_10BIT_ALL_SIDES_CROP, H265_10BIT_UNCROPPED),
    ] {
        let Some(device) = open_device(clip.codec) else {
            return;
        };
        assert_rgba_is_the_crop_of(
            &decode_rgba(&device, clip.codec, clip.bytes),
            &decode_rgba(&device, clip.codec, uncropped),
        );
    }
}

#[test]
fn the_player_reports_and_renders_the_crop_rectangle() {
    let _gpu = gpu_lock();
    let clip = H264_ALL_SIDES_CROP;
    let render_first = |bytes: &[u8]| {
        let device = open_device(clip.codec)?;
        let mut player = VulkanVideoPlayer::new(device, bytes.to_vec(), FRAMES_PER_SECOND)
            .expect("build player");
        let texture = player.frame_at_index(0).expect("render frame 0").clone();
        assert_eq!((texture.width(), texture.height()), player.dimensions());
        Some(RgbaPicture::new(&texture, player.read_texture(&texture)))
    };
    let Some(cropped) = render_first(clip.bytes) else {
        return;
    };
    let uncropped = render_first(H264_UNCROPPED).expect("device opened once already");
    assert_rgba_is_the_crop_of(&[cropped], &[uncropped]);
}

#[test]
fn the_one_shot_idr_decodes_are_the_crop_rectangle() {
    let _gpu = gpu_lock();
    let clip = H264_ALL_SIDES_CROP;
    let Some(device) = open_device(clip.codec) else {
        return;
    };
    let idr = |bytes: &[u8]| {
        let ps = extract_h264_parameter_sets(bytes).expect("parse sps and pps");
        let session = device
            .create_h264_session(&ps, CODED_SIZE.0, CODED_SIZE.1)
            .expect("session");
        let frame = device.decode_idr_nv12(&session, bytes).expect("idr decode");
        let texture = device
            .decode_idr_to_rgba_texture_gpu(&session, bytes)
            .expect("idr decode to a texture");
        (frame, RgbaPicture::read(&device, &texture))
    };
    let (frame, cropped) = idr(clip.bytes);
    let (_, uncropped) = idr(H264_UNCROPPED);
    assert_rgba_is_the_crop_of(&[cropped], &[uncropped]);
    let planar = planar_frames(vec![frame], clip.size);
    if let Some(reference) = vulkan_ref::reference_yuv(clip.file) {
        assert!(
            reference.starts_with(&planar[0]),
            "the IDR differs from the reference's first frame"
        );
    }
}

#[test]
fn a_launch_line_decodes_the_crop_rectangle() {
    let _gpu = gpu_lock();
    for (clip, parser) in [
        (H264_ALL_SIDES_CROP, "h264parse"),
        (H265_CONFORMANCE_WINDOW, "h265parse"),
    ] {
        if open_device(clip.codec).is_none() {
            return;
        }
        let channel = format!("crop_{}", clip.file);
        let pull = register_appsink_pull(&channel);
        let line = format!(
            "filesrc location={}/tests/fixtures/{} ! {parser} ! vulkanvideodec ! appsink channel={channel}",
            env!("CARGO_MANIFEST_DIR"),
            clip.file,
        );
        let graph = parse_launch(&default_registry(), &line).expect("parses");
        // the appsink holds only a few frames, so they are pulled while the graph runs
        let run = std::thread::spawn(move || block_on(run_graph(graph, &ZeroClock, LINK_CAPACITY)));
        let mut frames = Vec::new();
        while let Some(frame) = block_on(pull.pull()) {
            frames.push(
                frame
                    .domain
                    .as_system_slice()
                    .expect("system frame")
                    .to_vec(),
            );
        }
        run.join()
            .expect("the graph thread finishes")
            .unwrap_or_else(|error| panic!("{line} runs, got {error:?}"));
        reference_check(clip, &frames);
    }
}

#[test]
fn a_crop_outside_the_coded_picture_fails_the_session() {
    let _gpu = gpu_lock();
    let Some(h264_device) = open_device(VideoCodec::H264) else {
        return;
    };
    let ps = extract_h264_parameter_sets(H264_ALL_SIDES_CROP.bytes).expect("parse sps and pps");
    for oversized in [CODED_SIZE.1, u32::MAX] {
        let mut bad = ps.clone();
        bad.sps.frame_crop_bottom_offset = oversized;
        assert!(matches!(
            h264_device.create_h264_session(&bad, CODED_SIZE.0, CODED_SIZE.1),
            Err(VulkanVideoError::UnsupportedStream)
        ));
    }
    drop(h264_device);
    let Some(h265_device) = open_device(VideoCodec::H265) else {
        return;
    };
    let ps = extract_h265_parameter_sets(H265_10BIT_ALL_SIDES_CROP.bytes)
        .expect("parse vps, sps and pps");
    for oversized in [CODED_SIZE.0, u32::MAX] {
        let mut std = to_std_h265_params(&ps);
        std.sps.conf_win_right_offset = oversized;
        assert!(matches!(
            h265_device.create_h265_session(&std, CODED_SIZE.0, CODED_SIZE.1),
            Err(VulkanVideoError::UnsupportedStream)
        ));
    }
}
