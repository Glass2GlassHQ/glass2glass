#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::block_on;
use g2g_core::{AllocationParams, AsyncElement, Caps, Dim, PipelinePacket, RawVideoFormat};
use g2g_plugins::ivfdemux::IvfDemux;
use g2g_plugins::vulkanvideo::{open_av1_decode_device, VulkanVideoDec, VulkanVideoError};

mod vulkan_nv12_common;
use vulkan_nv12_common::{
    read_two_plane, skip_reason, system_bytes, system_frame, two_plane_frame, Collect,
};
mod vulkan_ref;
use vulkan_ref::{reference_yuv, two_plane_to_planar};

#[path = "common/ivf.rs"]
mod ivf;

// SVT-AV1 resize-mode=1 resize-denom=16: a 192x192 sequence whose frames after the key frame are coded at 96x96
const HALF_CODED_FIXTURE: &str = "av1_192x192_coded_96x96.ivf";
const HALF_CODED_IVF: &[u8] = include_bytes!("fixtures/av1_192x192_coded_96x96.ivf");
const HALF_CODED_SIZE: (u32, u32) = (96, 96);
// the second pass starts on the key frame again, so the size returns to the sequence's
const HALF_CODED_PASSES: usize = 2;

const AV1_BIT_DEPTH: u8 = 8;

// libtest runs these in parallel and concurrent instance creation SIGSEGVs the NVIDIA loader
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// the smallest AV1 frame the default decode GPU decodes, `None` without one
fn min_coded_extent() -> Option<(u32, u32)> {
    match block_on(open_av1_decode_device()) {
        Ok(device) => Some(device.caps().min_coded_extent),
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip: no Vulkan AV1 decode adapter");
            None
        }
        Err(error) => panic!("open AV1 decode device: {error:?}"),
    }
}

// m1433 covers a frame below the device minimum, which fails the decode
fn decodable(size: (u32, u32)) -> bool {
    let Some((min_width, min_height)) = min_coded_extent() else {
        return false;
    };
    let decodable = size.0 >= min_width && size.1 >= min_height;
    if !decodable {
        eprintln!("skip: {size:?} is below the device minimum {min_width}x{min_height}");
    }
    decodable
}

fn sequence_size(caps: &Caps) -> (u32, u32) {
    let Caps::CompressedVideo {
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        ..
    } = caps
    else {
        panic!("the IVF header carries the sequence size: {caps:?}");
    };
    (*width, *height)
}

fn pinned_two_plane_caps(caps: &Caps) -> Caps {
    let Caps::CompressedVideo {
        width,
        height,
        framerate,
        colorimetry,
        ..
    } = caps.clone()
    else {
        panic!("compressed video caps: {caps:?}");
    };
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width,
        height,
        framerate,
        interlace: g2g_core::Interlace::Any,
        colorimetry,
    }
}

// everything `vulkanvideodec` pushes for `units`, one temporal unit per frame as ivfdemux delivers them
fn decode_units(
    caps: &Caps,
    units: &[&[u8]],
    domain: MemoryDomainKind,
    pinned_caps: Option<Caps>,
) -> Collect {
    let mut decoder = VulkanVideoDec::new();
    decoder.configure_allocation(&AllocationParams {
        size_bytes: 0,
        min_buffers: 1,
        align: 1,
        domain,
        accepts: DomainSet::only(domain),
        ..Default::default()
    });
    decoder
        .configure_pipeline(caps)
        .expect("configure opens the decode device");
    let mut collect = Collect::default();
    if let Some(pinned) = pinned_caps {
        block_on(decoder.process(PipelinePacket::CapsChanged(pinned), &mut collect))
            .expect("pre-fixed output caps accepted");
    }
    for unit in units {
        block_on(decoder.process(PipelinePacket::DataFrame(system_frame(unit)), &mut collect))
            .expect("decode a temporal unit");
    }
    block_on(decoder.process(PipelinePacket::Eos, &mut collect)).expect("end of stream");
    collect
}

fn caps_sizes(collect: &Collect) -> Vec<(u32, u32)> {
    collect
        .caps_changes()
        .iter()
        .map(|caps| match caps {
            Caps::RawVideo {
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                ..
            } => (*width, *height),
            other => panic!("fixed raw video caps, got {other:?}"),
        })
        .collect()
}

// each frame is announced by the latest caps before it
fn frame_sizes(collect: &Collect) -> Vec<(u32, u32)> {
    let mut size = None;
    let mut sizes = Vec::new();
    for packet in &collect.packets {
        match packet {
            PipelinePacket::CapsChanged(Caps::RawVideo {
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                ..
            }) => size = Some((*width, *height)),
            PipelinePacket::DataFrame(_) => {
                sizes.push(size.expect("caps announce the first frame"));
            }
            _ => {}
        }
    }
    sizes
}

// the frames rewritten as one planar dump, compared against ffmpeg's played `passes` times
fn assert_planar_dump_matches(
    frames: &[Vec<u8>],
    sizes: &[(u32, u32)],
    fixture: &str,
    passes: usize,
) {
    let dump: Vec<u8> = frames
        .iter()
        .zip(sizes)
        .flat_map(|(frame, &size)| two_plane_to_planar(frame, AV1_BIT_DEPTH, size))
        .collect();
    let Some(reference) = reference_yuv(fixture) else {
        return;
    };
    assert!(
        dump == reference.repeat(passes),
        "{fixture}: the decode differs from the reference"
    );
    eprintln!("{fixture}: {} frames bit-exact", frames.len());
}

// the system decode against the reference, then the two-plane texture decode against the system one
fn assert_frames_keep_their_size(
    file: &[u8],
    fixture: &str,
    passes: usize,
    expected_caps_sizes: &[(u32, u32)],
    expected_frame_sizes: &[(u32, u32)],
) {
    let caps = IvfDemux::stream_caps(file).expect("an IVF header");
    let units = ivf::ivf_frames(file).repeat(passes);

    let system = decode_units(&caps, &units, MemoryDomainKind::System, None);
    assert_eq!(caps_sizes(&system), expected_caps_sizes, "system path caps");
    assert_eq!(frame_sizes(&system), expected_frame_sizes);
    let system_frames = system_bytes(&system);
    for (index, (frame, &(width, height))) in
        system_frames.iter().zip(expected_frame_sizes).enumerate()
    {
        let tight = RawVideoFormat::Nv12
            .unpadded_frame_bytes(width, height)
            .expect("frame size fits") as usize;
        assert_eq!(
            frame.len(),
            tight,
            "frame {index} is not read at its own size"
        );
    }
    assert_planar_dump_matches(&system_frames, expected_frame_sizes, fixture, passes);

    if let Some(reason) = skip_reason() {
        eprintln!("skip the texture path: {reason}");
        return;
    }
    let texture = decode_units(
        &caps,
        &units,
        MemoryDomainKind::WgpuTexture,
        Some(pinned_two_plane_caps(&caps)),
    );
    // the pinned caps go out first, and a colorimetry refinement repeats their size
    let mut expected_texture_caps_sizes = vec![sequence_size(&caps)];
    expected_texture_caps_sizes.extend(expected_caps_sizes);
    expected_texture_caps_sizes.dedup();
    let mut texture_caps_sizes = caps_sizes(&texture);
    texture_caps_sizes.dedup();
    assert_eq!(
        texture_caps_sizes, expected_texture_caps_sizes,
        "texture path caps"
    );
    let textures = texture.frames();
    assert_eq!(textures.len(), system_frames.len());
    for (index, ((frame, system_frame), &size)) in textures
        .iter()
        .zip(&system_frames)
        .zip(expected_frame_sizes)
        .enumerate()
    {
        let owner = two_plane_frame(frame);
        let texture_size = (owner.texture().width(), owner.texture().height());
        assert_eq!(texture_size, size, "frame {index} texture size");
        let bytes = read_two_plane(owner.device(), owner.queue(), owner.texture());
        assert!(
            bytes == *system_frame,
            "frame {index}: the texture differs from the system decode"
        );
    }
}

#[test]
fn frames_coded_below_the_sequence_size_go_out_at_their_own_size() {
    let _gpu = gpu_lock();
    if !decodable(HALF_CODED_SIZE) {
        return;
    }
    let sequence = sequence_size(&IvfDemux::stream_caps(HALF_CODED_IVF).expect("an IVF header"));
    let units_per_pass = ivf::ivf_frames(HALF_CODED_IVF).len();
    let mut expected_frame_sizes = Vec::new();
    let mut expected_caps_sizes = Vec::new();
    for _ in 0..HALF_CODED_PASSES {
        expected_frame_sizes.push(sequence);
        expected_frame_sizes.extend(std::iter::repeat_n(HALF_CODED_SIZE, units_per_pass - 1));
        expected_caps_sizes.extend([sequence, HALF_CODED_SIZE]);
    }
    assert_frames_keep_their_size(
        HALF_CODED_IVF,
        HALF_CODED_FIXTURE,
        HALF_CODED_PASSES,
        &expected_caps_sizes,
        &expected_frame_sizes,
    );
}
