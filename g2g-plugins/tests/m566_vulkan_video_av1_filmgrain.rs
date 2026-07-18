//! M566: Vulkan Video AV1 film grain synthesis on real hardware.
//!
//! AV1 film grain is applied at output time on the grain-free reconstruction. The
//! Vulkan hardware decoder does not apply it (the RTX 3060 exposes only
//! `DPB_AND_OUTPUT_COINCIDE`; driver grain needs a distinct output image,
//! `DPB_AND_OUTPUT_DISTINCT`), so `Av1DpbDecoder` synthesizes grain on the decoded
//! NV12 (`apply_film_grain_nv12`, ported from the re_rav1d scalar reference, the
//! same crate g2g uses for `Rav1dDec`), matching the ffmpeg / dav1d software
//! decoder bit-for-bit. A film-grain stream routes through the synchronous
//! reorder-aware path so grain is applied per displayed frame.
//!
//! The fixture is a 640x480 SVT-AV1 clip encoded with film grain (9 frames). The
//! structural assertion (the stream actually carries film grain) runs always;
//! bit-exactness vs the software decoder (luma AND chroma) is checked when
//! `G2G_AV1_REF` points at a raw `yuv420p` dump of the same clip (`ffmpeg -i clip
//! -f rawvideo -pix_fmt yuv420p ref.yuv`), verified out of band: every plane SAD/px 0.
//!
//! Runs on the RTX 3060; skips with no adapter / no AV1 decode / no compute queue.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, open_av1_decode_device, to_std_av1_seq_header, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480_filmgrain.obu");
const W: usize = 640;
const H: usize = 480;

#[test]
fn decodes_filmgrain_av1_stream() {
    let seq = extract_av1_sequence_header(CLIP).expect("parse sequence header");
    assert!(
        seq.film_grain_params_present,
        "fixture carries no film grain; feature untested"
    );

    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m566: no Vulkan AV1 decode adapter");
            return;
        }
        Err(e) => panic!("open AV1 decode device: {e:?}"),
    };

    let std = to_std_av1_seq_header(&seq);
    let session = device
        .create_av1_session(&std, W as u32, H as u32)
        .expect("create AV1 session");
    let mut dec = device
        .create_av1_dpb_decoder(&session, &seq)
        .expect("build AV1 decoder");

    let frames = dec.decode_all(CLIP).expect("decode film-grain stream");
    assert!(!frames.is_empty(), "no frames decoded");

    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width, W as u32);
        assert_eq!(f.height, H as u32);
        assert_eq!(f.luma.len(), W * H);
        let min = *f.luma.iter().min().unwrap();
        let max = *f.luma.iter().max().unwrap();
        assert!(
            max > min,
            "frame {i} luma is uniform ({min}=={max}); no real content"
        );
    }

    // Optional bit-exact check vs an ffmpeg/dav1d yuv420p reference (grain applied).
    // NV12 chroma is interleaved CbCr; the reference is planar I420, so de-interleave
    // to compare the Cb (U) and Cr (V) planes.
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
            assert_eq!(ysad, 0, "frame {i} luma not bit-exact (grain mismatch)");
            assert_eq!(usad, 0, "frame {i} Cb not bit-exact (grain mismatch)");
            assert_eq!(vsad, 0, "frame {i} Cr not bit-exact (grain mismatch)");
        }
    }
}
