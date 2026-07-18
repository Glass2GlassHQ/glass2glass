//! M569: Vulkan Video H.264 / H.265 B-frame decode with display-order output.
//!
//! B-frames make decode order differ from display (presentation) order: a frame is
//! coded after pictures that precede it on screen, so the decoder emits pictures in
//! coding order while a viewer needs POC order. The hardware decode itself handles
//! B-frames correctly (the driver builds the L0/L1 reference lists from the DPB +
//! per-picture POC that `*DpbDecoder` supplies); the missing piece was reordering
//! the whole-stream output. `decode_all` / `decode_all_to_textures` now index the
//! stream's POCs and reorder coding-order frames into display order
//! (`reorder_to_display_order`), keyed by (coded-video-sequence, POC). For an I/P
//! stream this is the identity; the streaming `decode_push` stays in coding order.
//!
//! The fixtures are 640x480 clips encoded with consecutive B-frames (libx264
//! `-bf 2`; libx265 `bframes=2`, single closed GOP). The structural + reorder
//! assertions run always: the stream must actually carry a non-monotonic POC
//! sequence (else the reorder is untested), and `decode_all` must return one frame
//! per coded picture. Bit-exactness against the software decoder's DISPLAY-order
//! output is checked when `G2G_H264_BF_REF` / `G2G_H265_BF_REF` points at a raw
//! `yuv420p` dump (`ffmpeg -i clip -f rawvideo -pix_fmt yuv420p ref.yuv`): every
//! frame at its display index is SAD/px 0, which only holds if the ordering is right.
//!
//! Runs on the RTX 3060; skips with no adapter / no decode support.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, extract_h265_parameter_sets, open_h264_decode_device,
    open_h265_decode_device, to_std_h265_params, Nv12Frame, VulkanVideoError,
};

const H264: &[u8] = include_bytes!("fixtures/h264_640x480_bframes.h264");
const H265: &[u8] = include_bytes!("fixtures/h265_640x480_bframes.h265");
const W: usize = 640;
const H: usize = 480;

/// The per-frame luma SAD/px between the decoded frames (display order) and a
/// planar `yuv420p` display-order reference; panics on length mismatch.
fn assert_bit_exact(frames: &[Nv12Frame], ref_path: &str) {
    let ref_yuv = std::fs::read(ref_path).expect("read reference");
    let cw = W / 2;
    let ch = H / 2;
    let fb = W * H + 2 * cw * ch;
    assert!(ref_yuv.len() >= fb * frames.len(), "reference too short");
    for (i, f) in frames.iter().enumerate() {
        let base = i * fb;
        let ry = &ref_yuv[base..base + W * H];
        let ru = &ref_yuv[base + W * H..base + W * H + cw * ch];
        let rv = &ref_yuv[base + W * H + cw * ch..base + fb];
        let ysad: u64 = f
            .luma
            .iter()
            .zip(ry)
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
            .sum();
        let mut usad = 0u64;
        let mut vsad = 0u64;
        for k in 0..cw * ch {
            usad += (f.chroma[2 * k] as i32 - ru[k] as i32).unsigned_abs() as u64;
            vsad += (f.chroma[2 * k + 1] as i32 - rv[k] as i32).unsigned_abs() as u64;
        }
        eprintln!(
            "display frame {i}: Y {:.4} U {:.4} V {:.4}",
            ysad as f64 / (W * H) as f64,
            usad as f64 / (cw * ch) as f64,
            vsad as f64 / (cw * ch) as f64
        );
        assert_eq!(
            ysad, 0,
            "frame {i} luma not bit-exact (ordering or decode wrong)"
        );
        assert_eq!(usad, 0, "frame {i} Cb not bit-exact");
        assert_eq!(vsad, 0, "frame {i} Cr not bit-exact");
    }
}

#[test]
fn h264_bframes_decode_in_display_order() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m569 h264: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("open h264 device: {e:?}"),
    };
    let ps = extract_h264_parameter_sets(H264).expect("sps/pps");
    let session = device
        .create_h264_session(&ps, W as u32, H as u32)
        .expect("session");
    let mut dec = device
        .create_h264_dpb_decoder(&session, &ps)
        .expect("decoder");

    // The stream must actually reorder (POC non-monotonic in decode order), else the
    // display-order path is not exercised.
    let metas = dec.index_pictures(H264).expect("index");
    let pocs: Vec<i32> = metas.iter().map(|m| m.poc).collect();
    assert!(
        pocs.windows(2).any(|w| w[1] < w[0]),
        "fixture has no B-frame reorder (POC monotonic)"
    );

    let frames = dec.decode_all(H264).expect("decode");
    assert_eq!(frames.len(), metas.len(), "one frame per coded picture");
    for f in &frames {
        assert_eq!((f.width, f.height), (W as u32, H as u32));
    }
    if let Ok(p) = std::env::var("G2G_H264_BF_REF") {
        assert_bit_exact(&frames, &p);
        eprintln!(
            "m569 h264: {} B-frame frames bit-exact in display order",
            frames.len()
        );
    }
}

#[test]
fn h265_bframes_decode_in_display_order() {
    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m569 h265: no Vulkan H.265 decode adapter");
            return;
        }
        Err(e) => panic!("open h265 device: {e:?}"),
    };
    let ps = extract_h265_parameter_sets(H265).expect("vps/sps/pps");
    let std = to_std_h265_params(&ps);
    let session = device
        .create_h265_session(&std, W as u32, H as u32)
        .expect("session");
    let mut dec = device
        .create_h265_dpb_decoder(&session, &ps)
        .expect("decoder");

    let metas = dec.index_pictures(H265).expect("index");
    let pocs: Vec<i32> = metas.iter().map(|m| m.poc).collect();
    assert!(
        pocs.windows(2).any(|w| w[1] < w[0]),
        "fixture has no B-frame reorder (POC monotonic)"
    );

    let frames = dec.decode_all(H265).expect("decode");
    assert_eq!(frames.len(), metas.len(), "one frame per coded picture");
    if let Ok(p) = std::env::var("G2G_H265_BF_REF") {
        assert_bit_exact(&frames, &p);
        eprintln!(
            "m569 h265: {} B-frame frames bit-exact in display order",
            frames.len()
        );
    }
}
