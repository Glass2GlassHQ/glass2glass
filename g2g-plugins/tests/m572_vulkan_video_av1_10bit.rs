//! M572: 10-bit AV1 hardware decode on the system path, on real hardware.
//!
//! The AV1 sibling of M571 (10-bit HEVC). AV1 Main profile covers 8 and 10-bit
//! 4:2:0; the colour config carries the depth. `create_av1_session` /
//! `build_av1_dpb_decoder` now derive it from `color_config.BitDepth` and select
//! the AV1 Main profile at `TYPE_10` plus the `G10X6` two-plane output format
//! (reusing the shared `h265` machinery: `av1_profile(bit_depth)`,
//! `planar_420_format`, `format_bytes_per_sample`, `Nv12Frame::bit_depth`). 10-bit
//! samples are little-endian 16-bit with the value in the top 10 bits
//! (`sample = u16 >> 6`).
//!
//! The fixture is a 640x480 libaom AV1 Main clip encoded 10-bit (BT.2020 / PQ
//! tagged, 5 frames, no film grain). Structural assertions run always; bit-exactness
//! vs the dav1d software decoder is checked when `G2G_AV1_10BIT_REF` points at a raw
//! `yuv420p10le` dump: every luma and chroma sample SAD 0.
//!
//! Runs on the RTX 3060; skips with no adapter / no 10-bit AV1 decode support.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, open_av1_decode_device, to_std_av1_seq_header, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480_10bit.obu");
const W: usize = 640;
const H: usize = 480;

fn le16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[2 * i], b[2 * i + 1]])
}

#[test]
fn decodes_av1_10bit() {
    let seq = extract_av1_sequence_header(CLIP).expect("sequence header");
    assert_eq!(
        seq.color.bit_depth, 10,
        "fixture is not 10-bit; feature untested"
    );

    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m572: no Vulkan AV1 decode adapter");
            return;
        }
        Err(e) => panic!("open AV1 device: {e:?}"),
    };
    let std = to_std_av1_seq_header(&seq);
    let session = device
        .create_av1_session(&std, W as u32, H as u32)
        .expect("session");
    let mut dec = match device.create_av1_dpb_decoder(&session, &seq) {
        Ok(d) => d,
        Err(VulkanVideoError::UnsupportedStream) | Err(VulkanVideoError::ExtensionUnsupported) => {
            eprintln!("skip m572: no 10-bit AV1 decode on this device");
            return;
        }
        Err(e) => panic!("decoder: {e:?}"),
    };

    let frames = dec.decode_all(CLIP).expect("decode 10-bit AV1");
    assert!(!frames.is_empty(), "no frames decoded");
    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (W as u32, H as u32));
        assert_eq!(f.bit_depth, 10, "frame {i} not reported 10-bit");
        assert_eq!(f.luma.len(), W * H * 2, "10-bit luma is 2 bytes/sample");
        assert_eq!(f.chroma.len(), W * H, "10-bit chroma is 2 bytes/sample");
        let (mut lo, mut hi) = (u16::MAX, 0u16);
        for p in 0..W * H {
            let v = le16(&f.luma, p) >> 6;
            lo = lo.min(v);
            hi = hi.max(v);
        }
        assert!(
            hi > lo + 64,
            "frame {i} luma nearly uniform ({lo}..={hi}); decode likely failed"
        );
    }

    if let Ok(path) = std::env::var("G2G_AV1_10BIT_REF") {
        let ref_yuv = std::fs::read(&path).expect("read G2G_AV1_10BIT_REF");
        let cw = W / 2;
        let ch = H / 2;
        let fb = (W * H + 2 * cw * ch) * 2;
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
            let mut usad = 0u64;
            let mut vsad = 0u64;
            for k in 0..cw * ch {
                usad += ((le16(&f.chroma, 2 * k) >> 6) as i32 - le16(ru, k) as i32).unsigned_abs()
                    as u64;
                vsad += ((le16(&f.chroma, 2 * k + 1) >> 6) as i32 - le16(rv, k) as i32)
                    .unsigned_abs() as u64;
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
        eprintln!("m572: {} AV1 Main 10-bit frames bit-exact", frames.len());
    }
}
