//! M588: the random-access `VulkanVideoPlayer` scrubs H.265, tuning in at a CRA.
//!
//! The pull-based player (`frame_at(t)`) was H.264-only. It now sniffs the codec
//! and drives an `H265DpbDecoder` too, and its seek picks the nearest IRAP
//! random-access point (an IDR *or*, for open-GOP HEVC, a CRA) rather than only an
//! IDR. Seeking into a late GOP therefore tunes in at that GOP's CRA (M587 discards
//! the CRA's RASL followers) instead of decoding the whole stream from the leading
//! IDR: hardware random access for the wgpu-texture wedge on the open-GOP content
//! that HEVC in the wild actually is.
//!
//! Oracle: a whole-stream `decode_all_to_textures` (display order, M577/M569
//! validated) on a second decoder. The player's `frame_at_index(p)` must be
//! bit-exact to oracle frame `p` for every `p` (which also exercises the
//! leading-picture case: a CRA's RASL/RADL seek from an earlier IRAP so their
//! references exist). Meaningfulness: a cold seek to the last frame decodes far
//! fewer than the whole stream (it tuned in at the last CRA, not the IDR).
//!
//! Runs on the RTX 3060; skips with no adapter / no decode support.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h265_parameter_sets, open_h265_decode_device, to_std_h265_params, VulkanVideoError,
    VulkanVideoPlayer,
};

const CLIP: &[u8] = include_bytes!("fixtures/h265_640x480_opengop.hevc");

#[test]
fn h265_player_scrubs_and_tunes_in_at_cra() {
    // Skip early if the GPU has no H.265 decode.
    match block_on(open_h265_decode_device()) {
        Ok(_) => {}
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m588: no Vulkan H.265 decode adapter");
            return;
        }
        Err(e) => panic!("open h265 device: {e:?}"),
    }

    // Oracle: whole-stream display-order decode on a dedicated device/decoder.
    let ref_dev = block_on(open_h265_decode_device()).expect("reference device");
    let ps = extract_h265_parameter_sets(CLIP).expect("vps/sps/pps");
    let std = to_std_h265_params(&ps);
    let (w, h) = (ps.sps.pic_width_in_luma_samples, ps.sps.pic_height_in_luma_samples);
    let ref_session = ref_dev.create_h265_session(&std, w, h).expect("ref session");
    let mut ref_dec =
        match ref_dev.create_h265_dpb_decoder_gpu(&ref_session, &ps) {
            Ok(d) => d,
            Err(VulkanVideoError::NoComputeQueue) => {
                eprintln!("skip m588: no distinct compute queue (GPU-texture player needs one)");
                return;
            }
            Err(e) => panic!("ref decoder: {e:?}"),
        };
    let ref_textures = ref_dec.decode_all_to_textures(CLIP).expect("reference decode");
    let reference: Vec<Vec<u8>> = ref_textures.iter().map(|t| ref_dev.read_rgba_texture(t)).collect();
    assert!(reference.len() >= 20, "open-GOP fixture decodes its whole timeline");

    // The player under test (codec sniffed as H.265).
    let dev = block_on(open_h265_decode_device()).expect("player device");
    let mut player = match VulkanVideoPlayer::new(dev, CLIP.to_vec(), 30) {
        Ok(p) => p,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skip m588: no distinct compute queue");
            return;
        }
        Err(e) => panic!("build H.265 player: {e:?}"),
    };
    let n = player.frame_count();
    assert_eq!(n, reference.len(), "player frame count matches the whole-stream decode");
    assert_eq!(player.dimensions(), (w, h));

    // Meaningfulness (first, cold): seeking the LAST display frame tunes in at the
    // last GOP's CRA, so it decodes far fewer pictures than the whole stream would
    // from the leading IDR. (Open-GOP keyint here is 12; a whole-stream decode of
    // the last frame from the IDR would be ~n pictures.)
    {
        let tex = player.frame_at_index(n - 1).expect("scrub to last frame").clone();
        let got = player.read_texture(&tex);
        assert_eq!(got, reference[n - 1], "last frame not bit-exact after CRA tune-in");
    }
    let cold = player.pictures_decoded();
    assert!(
        cold < n,
        "cold seek to the last frame decoded {cold} of {n} pictures; a CRA tune-in must decode fewer than the whole stream"
    );
    assert!(
        cold <= 16,
        "cold seek to the last frame decoded {cold} pictures; expected roughly one GOP (tuned in at the last CRA, not the IDR)"
    );

    // Every displayed frame is bit-exact to the oracle, in any order. This also
    // exercises the leading-picture seek path (a CRA's RASL/RADL, whose POC is
    // before its CRA, must seek from an earlier random-access point). Scrub a mix
    // of forward, backward and jump patterns.
    let order: Vec<usize> = (0..n)
        .chain((0..n).rev())
        .chain((0..n).step_by(3))
        .collect();
    for p in order {
        let tex = player.frame_at_index(p).expect("scrub").clone();
        let got = player.read_texture(&tex);
        assert_eq!(got, reference[p], "scrubbed frame {p} not bit-exact to the oracle");
    }

    eprintln!(
        "m588 h265 player: {n} frames scrubbed bit-exact; cold last-frame seek decoded {cold} pictures (of {n})"
    );
}
