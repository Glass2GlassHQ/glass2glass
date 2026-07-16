//! M565: Vulkan Video AV1 alt-ref + show_existing_frame decode on real hardware.
//!
//! The earlier fixtures are all-shown (decode order == display order). Real AV1
//! streams use alt-ref (invisible) frames: a frame coded with `show_frame == 0` is
//! decoded and stored but not displayed, then a later `OBU_FRAME_HEADER` with
//! `show_existing_frame` re-displays it at its true position (so decode order !=
//! display order). `Av1DpbDecoder` now detects such a stream and takes a
//! synchronous, reorder-aware path: it decodes every coded frame into its DPB slot,
//! emits only the shown ones, and emits a stored reference for each
//! `show_existing_frame`, so the output is in display order.
//!
//! The fixture is a 640x480 libaom clip with 2 alt-ref frames + 1
//! `show_existing_frame` (17 coded frame OBUs -> 15 displayed frames), no film
//! grain. Structural assertions (including that the fixture genuinely uses alt-ref
//! and show_existing) run always; bit-exactness vs the ffmpeg (dav1d) software
//! decoder is checked when `G2G_AV1_REF` points at a raw `yuv420p` dump of the same
//! clip, verified out of band as for M506: all 15 display frames SAD/px 0.
//!
//! Runs on the RTX 3060; skips with no adapter / no AV1 decode / no compute queue.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    av1_frame_infos, extract_av1_sequence_header, open_av1_decode_device, to_std_av1_seq_header,
    VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480_showexisting.obu");
const W: usize = 640;
const H: usize = 480;

#[test]
fn decodes_showexisting_av1_stream() {
    // The fixture must genuinely exercise the reorder path.
    let infos = av1_frame_infos(CLIP).expect("classify frames");
    let n_show_existing = infos.iter().filter(|f| f.show_existing_frame).count();
    let n_altref = infos.iter().filter(|f| !f.show_frame && !f.show_existing_frame).count();
    let n_display = infos.iter().filter(|f| f.show_frame).count(); // shown + show_existing
    assert!(n_show_existing > 0, "fixture has no show_existing_frame; reorder path untested");
    assert!(n_altref > 0, "fixture has no alt-ref (show_frame==0) frame");
    assert_eq!(n_display, 15, "expected 15 displayed frames");

    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m565: no Vulkan AV1 decode adapter");
            return;
        }
        Err(e) => panic!("open AV1 decode device: {e:?}"),
    };

    let seq = extract_av1_sequence_header(CLIP).expect("parse sequence header");
    let std = to_std_av1_seq_header(&seq);
    let session = device.create_av1_session(&std, W as u32, H as u32).expect("create AV1 session");
    let mut dec = device.create_av1_dpb_decoder(&session, &seq).expect("build AV1 decoder");

    let frames = dec.decode_all(CLIP).expect("decode alt-ref / show_existing stream");
    assert_eq!(frames.len(), 15, "one frame per displayed picture (shown + show_existing)");

    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width, W as u32);
        assert_eq!(f.height, H as u32);
        assert_eq!(f.luma.len(), W * H, "one luma byte per sample");
        let min = *f.luma.iter().min().unwrap();
        let max = *f.luma.iter().max().unwrap();
        assert!(max > min, "frame {i} luma is uniform ({min}=={max}); no real content");
    }

    // Optional bit-exact check vs an ffmpeg yuv420p reference dump (display order).
    if let Ok(path) = std::env::var("G2G_AV1_REF") {
        let ref_yuv = std::fs::read(&path).expect("read G2G_AV1_REF");
        let frame_bytes = W * H * 3 / 2;
        assert!(ref_yuv.len() >= frame_bytes * frames.len(), "reference too short");
        for (i, f) in frames.iter().enumerate() {
            let y0 = i * frame_bytes;
            let ref_y = &ref_yuv[y0..y0 + W * H];
            let sad: u64 = f
                .luma
                .iter()
                .zip(ref_y)
                .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
                .sum();
            let sad_per_px = sad as f64 / (W * H) as f64;
            eprintln!("display frame {i}: SAD/px = {sad_per_px:.6}");
            assert!(sad_per_px == 0.0, "display frame {i} must be bit-exact (SAD/px {sad_per_px})");
        }
    }
}
