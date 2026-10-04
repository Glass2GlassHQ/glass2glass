//! The software reference dumps the Vulkan decode tests compare against:
//! `G2G_VULKAN_REF_DIR` names a directory of raw planar dumps, one per fixture,
//! named after the fixture with a `.yuv` extension. `tools/vulkan-refs.sh`
//! generates them with ffmpeg.
#![allow(dead_code)] // no one test file uses every helper here

use g2g_core::RawVideoFormat;
use g2g_plugins::vulkanvideo::Nv12Frame;

const REFERENCE_DIRECTORY_VAR: &str = "G2G_VULKAN_REF_DIR";
const REFERENCE_EXTENSION: &str = "yuv";

// The dump for `fixture_file_name`, or `None` when the variable is unset (the
// caller then checks geometry only). Panics when the variable is set and the
// dump is missing, so a misnamed dump cannot pass vacuously.
pub(crate) fn reference_yuv(fixture_file_name: &str) -> Option<Vec<u8>> {
    let directory = std::env::var(REFERENCE_DIRECTORY_VAR).ok()?;
    let stem = fixture_file_name
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(fixture_file_name);
    let path = std::path::Path::new(&directory).join(format!("{stem}.{REFERENCE_EXTENSION}"));
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(error) => panic!(
            "{REFERENCE_DIRECTORY_VAR} is set but {} is unreadable: {error}",
            path.display()
        ),
    }
}

// g2g's tight NV12 / P010 frame rewritten as ffmpeg's planar dump of the same picture
pub(crate) fn two_plane_to_planar(
    frame: &[u8],
    bit_depth: u8,
    (width, height): (u32, u32),
) -> Vec<u8> {
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

pub(crate) fn planar_frames(frames: Vec<Nv12Frame>, size: (u32, u32)) -> Vec<Vec<u8>> {
    frames
        .into_iter()
        .map(|frame| {
            assert_eq!((frame.width, frame.height), size);
            two_plane_to_planar(&[frame.luma, frame.chroma].concat(), frame.bit_depth, size)
        })
        .collect()
}

// sizes always, every sample against the ffmpeg dump when one is configured
pub(crate) fn assert_frames_match_reference(
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
