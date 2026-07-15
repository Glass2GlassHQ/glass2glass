//! M571: 10-bit HEVC (Main 10) hardware decode on the system path, on real hardware.
//!
//! HDR content is 10-bit, but the decoder hardcoded 8-bit (NV12), so Main 10 /
//! 10-bit streams could not decode. The session now derives its bit depth from the
//! SPS: a 10-bit SPS selects the HEVC Main 10 profile and the `G10X6` two-plane
//! 4:2:0 output format (16-bit samples, value in the top 10 bits), and the DPB /
//! readback sizing scales to 2 bytes per sample. `Nv12Frame::bit_depth` reports 10
//! and its planes carry little-endian 16-bit samples (`sample = u16 >> 6`).
//!
//! The fixture is a 640x480 x265 Main 10 clip (BT.2020 / PQ tagged, 5 frames).
//! Structural assertions run always (bit depth 10, 2-byte samples, real content);
//! bit-exactness vs the ffmpeg / libde265 software decoder is checked when
//! `G2G_H265_10BIT_REF` points at a raw `yuv420p10le` dump (`ffmpeg -i clip
//! -f rawvideo -pix_fmt yuv420p10le ref.yuv`): every luma and chroma sample SAD 0.
//! (This is the first HDR-precision decode layer; PQ/HLG tone mapping and the
//! 10-bit GPU-texture path are later increments - the GPU-texture path rejects
//! 10-bit today with `UnsupportedStream`.)
//!
//! Runs on the RTX 3060; skips with no adapter / no 10-bit HEVC decode support.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h265_parameter_sets, open_h265_decode_device, to_std_h265_params, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h265_640x480_main10.hevc");
const W: usize = 640;
const H: usize = 480;

/// Little-endian 16-bit sample `i` from a byte plane.
fn le16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[2 * i], b[2 * i + 1]])
}

#[test]
fn decodes_hevc_main10_10bit() {
    let ps = extract_h265_parameter_sets(CLIP).expect("vps/sps/pps");
    assert_eq!(ps.sps.bit_depth_luma_minus8, 2, "fixture is not 10-bit; feature untested");

    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m571: no Vulkan H.265 decode adapter");
            return;
        }
        Err(e) => panic!("open h265 device: {e:?}"),
    };
    let std = to_std_h265_params(&ps);
    let session = device.create_h265_session(&std, W as u32, H as u32).expect("session");
    let mut dec = match device.create_h265_dpb_decoder(&session, &ps) {
        Ok(d) => d,
        // A device without 10-bit HEVC decode fails session/decoder creation.
        Err(VulkanVideoError::UnsupportedStream) | Err(VulkanVideoError::ExtensionUnsupported) => {
            eprintln!("skip m571: no 10-bit HEVC decode on this device");
            return;
        }
        Err(e) => panic!("decoder: {e:?}"),
    };

    let frames = dec.decode_all(CLIP).expect("decode 10-bit stream");
    assert!(!frames.is_empty(), "no frames decoded");
    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (W as u32, H as u32));
        assert_eq!(f.bit_depth, 10, "frame {i} not reported 10-bit");
        assert_eq!(f.luma.len(), W * H * 2, "10-bit luma is 2 bytes/sample");
        assert_eq!(f.chroma.len(), W * H, "10-bit chroma is 2 bytes/sample, half the luma count");
        // Real content: samples span a range (a failed decode is flat). Values are
        // the top-10-bit G10X6 packing, so shift down to the 0..=1023 range.
        let (mut lo, mut hi) = (u16::MAX, 0u16);
        for p in 0..W * H {
            let v = le16(&f.luma, p) >> 6;
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(hi > lo + 64, "frame {i} luma nearly uniform ({lo}..={hi}); decode likely failed");
    }

    // Optional bit-exact check vs an ffmpeg yuv420p10le reference (display order).
    if let Ok(path) = std::env::var("G2G_H265_10BIT_REF") {
        let ref_yuv = std::fs::read(&path).expect("read G2G_H265_10BIT_REF");
        let cw = W / 2;
        let ch = H / 2;
        let fb = (W * H + 2 * cw * ch) * 2; // planar 10-bit: Y, U, V, 2 bytes each
        assert!(ref_yuv.len() >= fb * frames.len(), "reference too short");
        for (i, f) in frames.iter().enumerate() {
            let base = i * fb;
            let ry = &ref_yuv[base..base + W * H * 2];
            let ru = &ref_yuv[base + W * H * 2..base + W * H * 2 + cw * ch * 2];
            let rv = &ref_yuv[base + W * H * 2 + cw * ch * 2..base + fb];
            let mut ysad = 0u64;
            for p in 0..W * H {
                ysad += ((le16(&f.luma, p) >> 6) as i32 - le16(ry, p) as i32).unsigned_abs() as u64;
            }
            // NV12 chroma is interleaved Cb,Cr; the reference is planar U then V.
            let mut usad = 0u64;
            let mut vsad = 0u64;
            for k in 0..cw * ch {
                usad += ((le16(&f.chroma, 2 * k) >> 6) as i32 - le16(ru, k) as i32).unsigned_abs() as u64;
                vsad += ((le16(&f.chroma, 2 * k + 1) >> 6) as i32 - le16(rv, k) as i32).unsigned_abs() as u64;
            }
            eprintln!(
                "frame {i}: Y SAD/px={:.6} U={:.6} V={:.6}",
                ysad as f64 / (W * H) as f64,
                usad as f64 / (cw * ch) as f64,
                vsad as f64 / (cw * ch) as f64
            );
            assert_eq!(ysad, 0, "frame {i} luma not bit-exact");
            assert_eq!(usad, 0, "frame {i} Cb not bit-exact");
            assert_eq!(vsad, 0, "frame {i} Cr not bit-exact");
        }
        eprintln!("m571: {} HEVC Main 10 frames bit-exact (10-bit)", frames.len());
    }
}
