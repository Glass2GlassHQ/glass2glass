//! M499: `VulkanVideoPlayer` caching -- forward-continue fast path + LRU bound.
//!
//! Two follow-ups to the M498 scrubber:
//!
//! 1. **Forward-continue.** A cache miss no longer always resets and re-decodes
//!    from the keyframe: when the decoder already sits within the target's GOP at
//!    or before it (a forward seek in reach), it continues decoding in place. So
//!    linear playback decodes each coded picture exactly **once** (O(n)), not the
//!    O(n^2) a reset-per-frame would cost. We assert `pictures_decoded == n` for a
//!    straight play-through, and that every frame is still bit-identical to a
//!    linear reference decode.
//! 2. **LRU bound.** The decoded-frame cache is capped; past the cap the
//!    least-recently-used frame is evicted. We drive the cache to capacity 1 and
//!    show a resident frame hits (no decode) while an evicted one re-decodes, and
//!    that the cache never exceeds its capacity.
//!
//! Runs on the RTX 3060; skips with no Vulkan H.264 adapter / no compute queue.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError, VulkanVideoPlayer,
};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

/// Mean absolute per-byte difference between two equal-length RGBA buffers.
fn sad_per_byte(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "frame sizes differ ({} vs {})", a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let sum: u64 =
        a.iter().zip(b).map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64).sum();
    sum as f64 / a.len() as f64
}

#[test]
fn m499_vulkan_video_cache() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m499: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("open decode device: {e:?}"),
    };

    let ps = extract_h264_parameter_sets(CLIP).expect("parse SPS/PPS");
    let width = (ps.sps.pic_width_in_mbs_minus1 + 1) * 16;
    let height = (ps.sps.pic_height_in_map_units_minus1 + 1) * 16;

    // Linear-decode ground truth (decoding order), read back to RGBA bytes.
    let reference: Vec<Vec<u8>> = {
        let session = device.create_h264_session(&ps, width, height).expect("reference session");
        let mut dec = match device.create_h264_dpb_decoder_gpu(&session, &ps) {
            Ok(d) => d,
            Err(VulkanVideoError::NoComputeQueue) => {
                eprintln!("skip m499: no distinct compute queue for the RGBA path");
                return;
            }
            Err(e) => panic!("build reference decoder: {e:?}"),
        };
        let texs = dec.decode_all_to_textures(CLIP).expect("linear decode");
        texs.iter().map(|t| device.read_rgba_texture(t)).collect()
    };
    let n = reference.len();
    assert!(n >= 8, "fixture should be >= 2 GOPs; got {n} frames");

    let mut player = VulkanVideoPlayer::new(device, CLIP.to_vec(), 30).expect("build player");

    // --- 1. forward-continue: linear play decodes each picture exactly once ---
    for p in 0..n {
        let tex = player.frame_at_index(p).expect("frame_at_index").clone();
        let got = player.read_texture(&tex);
        let d = player.decode_index(p).expect("decode index");
        assert!(
            sad_per_byte(&got, &reference[d]) == 0.0,
            "linear frame {p} (decode {d}) must match the reference decode",
        );
    }
    assert_eq!(
        player.pictures_decoded(),
        n,
        "linear playback must decode each coded picture exactly once (forward-continue), \
         not O(n^2) via a keyframe reset per frame",
    );
    assert_eq!(player.decode_calls(), n, "n distinct frames = n cache misses");

    // A full re-play is served entirely from cache: no further pictures decoded.
    let pics = player.pictures_decoded();
    for p in 0..n {
        let _ = player.frame_at_index(p).expect("cached frame_at_index");
    }
    assert_eq!(player.pictures_decoded(), pics, "a cached re-play decodes nothing");

    // --- 2. LRU bound: cap to 1 resident frame ---
    player.set_cache_capacity(1);
    assert_eq!(player.cache_len(), 1, "shrinking capacity evicts down to the bound");

    let base = player.decode_calls();
    // Frame 3 was evicted by the trim (only the most-recent frame survives), so
    // this is a miss.
    let _ = player.frame_at_index(3).expect("frame 3");
    assert_eq!(player.decode_calls(), base + 1, "an unresident frame decodes");
    assert_eq!(player.cache_len(), 1, "cache never exceeds capacity");

    // Immediately re-request it: resident, a cache hit (no decode).
    let _ = player.frame_at_index(3).expect("frame 3 again");
    assert_eq!(player.decode_calls(), base + 1, "a resident frame hits the cache");

    // Visit another frame (evicts 3), then request 3 again: a miss, proving the
    // LRU actually evicted it.
    let _ = player.frame_at_index(7).expect("frame 7");
    assert_eq!(player.cache_len(), 1, "still bounded");
    let _ = player.frame_at_index(3).expect("frame 3 re-decoded");
    assert_eq!(
        player.decode_calls(),
        base + 3,
        "an evicted frame re-decodes; the LRU bound is enforced",
    );
    assert_eq!(player.cache_len(), 1);

    // Correctness survives the eviction churn.
    let tex = player.frame_at_index(3).expect("frame 3 final").clone();
    let got = player.read_texture(&tex);
    let d = player.decode_index(3).expect("decode index");
    assert!(sad_per_byte(&got, &reference[d]) == 0.0, "re-decoded frame still matches reference");
}
