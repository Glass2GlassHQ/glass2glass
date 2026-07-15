//! M564: Vulkan Video AV1 multi-tile full-DPB decode on real hardware.
//!
//! The M506 fixture is single-tile; real AV1 streams split each frame into a
//! tile grid for parallel decode. `Av1DpbDecoder` now parses the `OBU_FRAME`
//! tile-group header + the per-tile `TileSizeBytes` size prefixes
//! (`av1_tile_layout`) into the driver's `pTileOffsets` / `pTileSizes`, so a
//! tiled frame decodes. The fixture is a 640x480 libaom clip with a 2x2 tile
//! grid (4 tiles per frame, 1 KEY + INTER frames, no film grain, all shown).
//!
//! Structural assertions run always (including that the fixture is genuinely
//! multi-tile, so the multi-tile path is exercised). Bit-exactness vs the ffmpeg
//! software (dav1d) decoder is checked when `G2G_AV1_REF` points at a raw
//! `yuv420p` dump of the same clip (`ffmpeg -i clip -f rawvideo -pix_fmt
//! yuv420p ref.yuv`), verified out of band as for M506: all 11 frames SAD/px 0.
//!
//! Runs on the RTX 3060; skips with no adapter / no AV1 decode / no compute queue.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    av1_frame_tile_grid, extract_av1_sequence_header, open_av1_decode_device,
    to_std_av1_seq_header, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480_tiles2x2.obu");
const W: usize = 640;
const H: usize = 480;

#[test]
fn decodes_multitile_av1_stream() {
    // The fixture must genuinely be tiled, else the multi-tile path is untested.
    let (cols, rows) = av1_frame_tile_grid(CLIP).expect("parse tile grid");
    assert!(cols * rows > 1, "fixture is single-tile ({cols}x{rows}); not exercising tiles");
    assert_eq!((cols, rows), (2, 2), "fixture should be a 2x2 tile grid");

    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m507: no Vulkan AV1 decode adapter");
            return;
        }
        Err(e) => panic!("open AV1 decode device: {e:?}"),
    };

    let seq = extract_av1_sequence_header(CLIP).expect("parse sequence header");
    let std = to_std_av1_seq_header(&seq);
    let session = device.create_av1_session(&std, W as u32, H as u32).expect("create AV1 session");
    let mut dec = device.create_av1_dpb_decoder(&session, &seq).expect("build AV1 decoder");

    let frames = dec.decode_all(CLIP).expect("decode whole tiled stream");
    assert_eq!(frames.len(), 11, "one frame per shown coded picture");

    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width, W as u32);
        assert_eq!(f.height, H as u32);
        assert_eq!(f.luma.len(), W * H, "one luma byte per sample");
        let min = *f.luma.iter().min().unwrap();
        let max = *f.luma.iter().max().unwrap();
        assert!(max > min, "frame {i} luma is uniform ({min}=={max}); no real content");
    }

    // Inter prediction happened, and the animated content moved.
    let key = &frames[0].luma;
    assert!(frames[1].luma != *key, "frame 1 (INTER) must differ from the key frame");
    for i in 1..frames.len() {
        assert!(frames[i].luma != frames[i - 1].luma, "frame {i} must differ from {}", i - 1);
    }

    // Optional bit-exact check vs an ffmpeg yuv420p reference dump.
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
            eprintln!("frame {i}: SAD/px = {sad_per_px:.6}");
            assert!(sad_per_px == 0.0, "tiled frame {i} must be bit-exact (SAD/px {sad_per_px})");
        }
    }
}
