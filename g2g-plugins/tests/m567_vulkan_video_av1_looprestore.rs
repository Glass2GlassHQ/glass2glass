//! M567: Vulkan Video AV1 loop-restoration decode regression on real hardware.
//!
//! A libaom `-cpu-used 4` encode enables loop restoration (Wiener / SGR), which
//! decoded WRONG (~62 SAD/px, even on intra frames and even planes with
//! `RESTORE_NONE`): `StdVideoAV1LoopRestoration::LoopRestorationSize` is NOT the
//! restoration unit's pixel size (64/128/256, as the spec's "5.9.20" reference
//! suggests) but `1 + lr_unit_shift` for luma and `1 + lr_unit_shift - lr_uv_shift`
//! for chroma (values 1..3), the encoding ffmpeg's Vulkan AV1 hwaccel passes.
//! Passing the raw pixel size mis-configured the driver's restoration and corrupted
//! the whole frame. Fixed in `parse_av1_loop_restoration`.
//!
//! The fixture is a 640x480 libaom clip that uses loop restoration on its first
//! frame. The structural assertion (the stream genuinely exercises loop
//! restoration) runs always; bit-exactness vs the ffmpeg / dav1d software decoder
//! (luma AND chroma) is checked when `G2G_AV1_REF` points at a raw `yuv420p` dump
//! of the same clip, verified out of band: every plane SAD/px 0.
//!
//! Runs on the RTX 3060; skips with no adapter / no AV1 decode / no compute queue.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    av1_uses_loop_restoration, extract_av1_sequence_header, open_av1_decode_device,
    to_std_av1_seq_header, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480_looprestore.obu");
const W: usize = 640;
const H: usize = 480;

#[test]
fn decodes_looprestore_av1_stream() {
    assert_eq!(
        av1_uses_loop_restoration(CLIP),
        Some(true),
        "fixture's first frame does not use loop restoration; regression untested"
    );

    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m567: no Vulkan AV1 decode adapter");
            return;
        }
        Err(e) => panic!("open AV1 decode device: {e:?}"),
    };

    let seq = extract_av1_sequence_header(CLIP).expect("parse sequence header");
    let std = to_std_av1_seq_header(&seq);
    let session = device
        .create_av1_session(&std, W as u32, H as u32)
        .expect("create AV1 session");
    let mut dec = device
        .create_av1_dpb_decoder(&session, &seq)
        .expect("build AV1 decoder");

    let frames = dec
        .decode_all(CLIP)
        .expect("decode loop-restoration stream");
    assert!(!frames.is_empty(), "no frames decoded");

    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width, W as u32);
        assert_eq!(f.height, H as u32);
        let min = *f.luma.iter().min().unwrap();
        let max = *f.luma.iter().max().unwrap();
        assert!(
            max > min,
            "frame {i} luma is uniform ({min}=={max}); no real content"
        );
    }

    if let Ok(path) = std::env::var("G2G_AV1_REF") {
        let ref_yuv = std::fs::read(&path).expect("read G2G_AV1_REF");
        let cw = W / 2;
        let ch = H / 2;
        let frame_bytes = W * H + 2 * cw * ch;
        assert!(
            ref_yuv.len() >= frame_bytes * frames.len(),
            "reference too short"
        );
        for (i, f) in frames.iter().enumerate() {
            let base = i * frame_bytes;
            let ref_y = &ref_yuv[base..base + W * H];
            let ref_u = &ref_yuv[base + W * H..base + W * H + cw * ch];
            let ref_v = &ref_yuv[base + W * H + cw * ch..base + frame_bytes];
            let ysad: u64 = f
                .luma
                .iter()
                .zip(ref_y)
                .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
                .sum();
            let mut usad = 0u64;
            let mut vsad = 0u64;
            for k in 0..cw * ch {
                usad += (f.chroma[2 * k] as i32 - ref_u[k] as i32).unsigned_abs() as u64;
                vsad += (f.chroma[2 * k + 1] as i32 - ref_v[k] as i32).unsigned_abs() as u64;
            }
            eprintln!(
                "frame {i}: Y SAD/px={:.6} U SAD/px={:.6} V SAD/px={:.6}",
                ysad as f64 / (W * H) as f64,
                usad as f64 / (cw * ch) as f64,
                vsad as f64 / (cw * ch) as f64
            );
            assert_eq!(ysad, 0, "frame {i} luma not bit-exact (loop restoration)");
            assert_eq!(usad, 0, "frame {i} Cb not bit-exact (loop restoration)");
            assert_eq!(vsad, 0, "frame {i} Cr not bit-exact (loop restoration)");
        }
    }
}
