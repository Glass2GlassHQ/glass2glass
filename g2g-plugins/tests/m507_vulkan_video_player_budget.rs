//! M507: `VulkanVideoPlayer` hardening -- memory-budgeted LRU + cache-traversed.
//!
//! Two follow-ups to the M499 frame-count LRU:
//!
//! 1. **Byte budget.** A frame-count cap alone pins gigabytes at 4K/8K (one
//!    3840x2160 RGBA frame is ~33 MB, so 64 of them is ~2 GB). The cache now also
//!    honors a byte budget, evicting least-recently-used frames to stay under it.
//!    We set a budget of three frames' worth and show the resident set never
//!    exceeds it, that a frame outside the window re-decodes, and that a
//!    re-decoded frame is still correct.
//! 2. **Cache-traversed (opt-in).** With it on, decoding a range caches every
//!    traversed picture, not just the target, so a backward scrub within the same
//!    GOP is a cache hit (no decode). Off by default (validated by m498/m499).
//!    We decode to a mid-GOP frame, then scrub back within the GOP and assert the
//!    revisit does no decode and is bit-identical to a reference.
//!
//! Runs on the RTX 3060; skips with no Vulkan H.264 adapter / no compute queue.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError, VulkanVideoPlayer,
};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

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
fn m507_vulkan_video_player_budget() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m507: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("open decode device: {e:?}"),
    };

    let ps = extract_h264_parameter_sets(CLIP).expect("parse SPS/PPS");
    let width = (ps.sps.pic_width_in_mbs_minus1 + 1) * 16;
    let height = (ps.sps.pic_height_in_map_units_minus1 + 1) * 16;
    let bytes_per_frame = width as usize * height as usize * 4;

    // Linear-decode ground truth (decoding order), read back to RGBA bytes.
    let reference: Vec<Vec<u8>> = {
        let session = device
            .create_h264_session(&ps, width, height)
            .expect("reference session");
        let mut dec = match device.create_h264_dpb_decoder_gpu(&session, &ps) {
            Ok(d) => d,
            Err(VulkanVideoError::NoComputeQueue) => {
                eprintln!("skip m507: no distinct compute queue for the RGBA path");
                return;
            }
            Err(e) => panic!("build reference decoder: {e:?}"),
        };
        let texs = dec.decode_all_to_textures(CLIP).expect("linear decode");
        texs.iter().map(|t| device.read_rgba_texture(t)).collect()
    };
    let n = reference.len();
    assert!(n >= 8, "fixture should be >= 2 GOPs; got {n} frames");

    // ----- 1. byte budget bounds the resident set -----
    let mut player = VulkanVideoPlayer::new(device, CLIP.to_vec(), 30).expect("build player");
    // Budget for exactly three frames; frame count stays generous so bytes bind.
    player.set_cache_byte_budget(3 * bytes_per_frame);
    for p in 0..n {
        let _ = player.frame_at_index(p).expect("linear frame");
        assert!(
            player.cache_len() <= 3,
            "resident frames stay within the byte budget"
        );
        assert!(
            player.cache_bytes() <= player.cache_byte_budget(),
            "cache stays under budget"
        );
    }
    assert_eq!(player.cache_len(), 3, "the last three frames are resident");

    // Frame 0 was evicted long ago: requesting it re-decodes (a miss).
    let before = player.decode_calls();
    let tex0 = player.frame_at_index(0).expect("frame 0").clone();
    assert_eq!(
        player.decode_calls(),
        before + 1,
        "an evicted frame re-decodes"
    );
    // ...and is still correct after the eviction churn.
    let got0 = player.read_texture(&tex0);
    let d0 = player.decode_index(0).expect("decode index 0");
    assert!(
        sad_per_byte(&got0, &reference[d0]) == 0.0,
        "re-decoded frame 0 matches reference"
    );

    // ----- 2. cache-traversed makes a backward scrub within a GOP free -----
    // A second decode device (the first player consumed the one passed to `new`).
    let device2 = block_on(open_h264_decode_device()).expect("second decode device");
    let mut player = VulkanVideoPlayer::new(device2, CLIP.to_vec(), 30).expect("build player 2");
    player.set_cache_traversed(true);
    // Decode to a mid-GOP frame in the first GOP (frame 4 is the last P of GOP 0
    // in this fixture). This resets to the IDR and, with cache-traversed on,
    // caches the whole run 0..=4.
    let mid = 4usize.min(n - 1);
    let _ = player.frame_at_index(mid).expect("mid frame");
    let after_mid = player.decode_calls();
    assert!(
        player.cache_len() > mid,
        "the traversed GOP prefix is cached"
    );

    // Scrub backward to an earlier frame in the same GOP: a cache hit, no decode.
    let back = mid / 2;
    let tex_back = player.frame_at_index(back).expect("backward scrub").clone();
    assert_eq!(
        player.decode_calls(),
        after_mid,
        "a backward scrub within a cached GOP does no decode (cache-traversed)",
    );
    // And it is the correct frame.
    let got_back = player.read_texture(&tex_back);
    let d_back = player.decode_index(back).expect("decode index");
    assert!(
        sad_per_byte(&got_back, &reference[d_back]) == 0.0,
        "the cache-traversed frame is bit-identical to the reference decode",
    );
}
