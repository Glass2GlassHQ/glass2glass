//! M498: random-access ("pull") scrubbing over the Vulkan Video decoder.
//!
//! The wedge for a timeline viewer: instead of the streaming push
//! path (`VulkanVideoDec`), a `VulkanVideoPlayer` serves the frame at an
//! arbitrary index straight as a GPU-resident RGBA `wgpu::Texture`, decoding
//! forward from the enclosing keyframe on each seek and caching results.
//!
//! This proves the two claims that turn a decode primitive into a scrubber:
//!
//! 1. **Random access is correct.** We scrub out of order (forward and backward
//!    across GOP boundaries, hitting mid-GOP P frames cold) and assert each
//!    returned frame is bit-identical to a straight linear decode of the same
//!    picture. A P frame served cold only matches if the player decoded it from
//!    its keyframe with the right references, so SAD == 0 *is* the
//!    decode-from-keyframe proof.
//! 2. **The cache works.** Each distinct frame decodes exactly once; revisiting a
//!    frame does no GPU decode.
//!
//! Runs on the RTX 3060; skips with no Vulkan H.264 adapter / no compute queue.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use std::collections::BTreeSet;

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError, VulkanVideoPlayer,
};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

/// Mean absolute per-byte difference between two equal-length RGBA buffers.
fn sad_per_byte(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "frame sizes differ ({} vs {})",
        a.len(),
        b.len()
    );
    if a.is_empty() {
        return 0.0;
    }
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.len() as f64
}

#[test]
fn m498_vulkan_video_scrub() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m498: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("open decode device: {e:?}"),
    };

    let ps = extract_h264_parameter_sets(CLIP).expect("parse SPS/PPS");
    let width = (ps.sps.pic_width_in_mbs_minus1 + 1) * 16;
    let height = (ps.sps.pic_height_in_map_units_minus1 + 1) * 16;

    // Ground truth: a straight linear decode of the whole clip (decoding order),
    // each frame read back to RGBA bytes. Same decode path the player uses, so a
    // correct random-access decode must match it exactly.
    let reference: Vec<Vec<u8>> = {
        let session = device
            .create_h264_session(&ps, width, height)
            .expect("reference session");
        let mut dec = match device.create_h264_dpb_decoder_gpu(&session, &ps) {
            Ok(d) => d,
            Err(VulkanVideoError::NoComputeQueue) => {
                eprintln!("skip m498: no distinct compute queue for the RGBA path");
                return;
            }
            Err(e) => panic!("build reference decoder: {e:?}"),
        };
        let texs = dec.decode_all_to_textures(CLIP).expect("linear decode");
        texs.iter().map(|t| device.read_rgba_texture(t)).collect()
        // dec + session drop here, before `device` moves into the player below.
    };
    let n = reference.len();
    assert!(n >= 6, "fixture should be >= 2 GOPs; got {n} frames");

    // The player takes ownership of the device and builds its keyframe/POC index.
    let mut player = VulkanVideoPlayer::new(device, CLIP.to_vec(), 30).expect("build player");
    assert_eq!(player.frame_count(), n, "player sees the same frame count");
    assert_eq!(
        player.dimensions(),
        (width, height),
        "player reports the coded size"
    );

    // Out-of-order scrub with repeats: jump forward/backward across the GOP
    // boundary and land on mid-GOP P frames cold.
    let scrub = [0usize, 7, 3, 9, 1, 5, 8, 2, 7, 0, 9];
    let mut distinct = BTreeSet::new();
    for &raw in &scrub {
        let p = raw.min(n - 1);
        let d = player
            .decode_index(p)
            .expect("decode index for a valid frame");
        distinct.insert(d);

        // Clone releases the &mut borrow so we can read it back on &player.
        let tex = player.frame_at_index(p).expect("frame_at_index").clone();
        let got = player.read_texture(&tex);
        let sad = sad_per_byte(&got, &reference[d]);
        assert!(
            sad == 0.0,
            "scrub to presentation frame {p} (decode {d}) must equal the linear \
             decode (SAD/byte {sad}) -- decode-from-keyframe failed",
        );
    }

    // Each distinct frame decoded exactly once; the repeated targets were served
    // from cache (no GPU decode).
    assert_eq!(
        player.decode_calls(),
        distinct.len(),
        "each distinct frame decodes once; revisits hit the cache",
    );

    // Re-request the whole sequence: a pure cache pass, zero further decodes.
    let before = player.decode_calls();
    for &raw in &scrub {
        let _ = player
            .frame_at_index(raw.min(n - 1))
            .expect("cached frame_at_index");
    }
    assert_eq!(
        player.decode_calls(),
        before,
        "revisiting cached frames does no decode"
    );
}
