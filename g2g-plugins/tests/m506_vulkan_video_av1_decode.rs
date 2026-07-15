//! M506: Vulkan Video AV1 full-DPB decode on real hardware, the AV1 sibling of
//! M492 / M503.
//!
//! `Av1DpbDecoder` parses each frame's uncompressed header (M506a), maps it onto
//! the `StdVideoDecodeAV1PictureInfo` + sub-structs, manages AV1's 8-slot
//! reference model (`ref_frame_idx` -> physical DPB slot, `refresh_frame_flags`
//! remap), and issues `vkCmdDecodeVideoKHR` with the per-tile offsets, so the
//! INTER frames decode against their references. The fixture is a 640x480 libaom
//! clip (1 KEY + 9 INTER, single tile, no film grain).
//!
//! Structural + content assertions run always. Bit-exactness against the ffmpeg
//! software decoder is checked when `G2G_AV1_REF` points at a raw `yuv420p` dump
//! of the same clip (`ffmpeg -i clip -f rawvideo -pix_fmt yuv420p ref.yuv`);
//! that comparison is verified out of band, as for M503.
//!
//! Runs on the RTX 3060; skips with no adapter / no AV1 decode / no compute queue.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, open_av1_decode_device, to_std_av1_seq_header, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480.obu");
const W: usize = 640;
const H: usize = 480;

#[test]
fn decodes_whole_av1_stream_with_references() {
    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m506: no Vulkan AV1 decode adapter");
            return;
        }
        Err(e) => panic!("open AV1 decode device: {e:?}"),
    };

    let seq = extract_av1_sequence_header(CLIP).expect("parse sequence header");
    let std = to_std_av1_seq_header(&seq);
    let session = device.create_av1_session(&std, W as u32, H as u32).expect("create AV1 session");
    let mut dec = device.create_av1_dpb_decoder(&session, &seq).expect("build AV1 decoder");

    let frames = dec.decode_all(CLIP).expect("decode whole stream");
    assert_eq!(frames.len(), 10, "one frame per shown coded picture");

    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width, W as u32);
        assert_eq!(f.height, H as u32);
        assert_eq!(f.luma.len(), W * H, "one luma byte per sample");
        let min = *f.luma.iter().min().unwrap();
        let max = *f.luma.iter().max().unwrap();
        assert!(max > min, "frame {i} luma is uniform ({min}=={max}); no real content");
    }

    // An INTER frame must differ from the key frame (inter prediction happened),
    // and consecutive frames must differ (the animated content moved).
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
            let ndiff = f.luma.iter().zip(ref_y).filter(|(a, b)| a != b).count();
            eprintln!("frame {i}: SAD/px = {sad_per_px:.6}  diff_px = {ndiff}");
            // Every frame is bit-exact vs the ffmpeg software decoder, including
            // the compound / temporal-MV inter frames. M506c chased what was once a
            // tiny residual on frames 2+ to its cause: the default loop-filter
            // reference deltas for ALTREF2 / ALTREF were 0 instead of the spec's
            // -1, so in-loop deblocking was mis-configured for compound blocks
            // referencing the alt frames. (Found by dumping the picture-info
            // sub-structs the driver receives from ffmpeg's Vulkan hwaccel vs ours
            // with a capture layer: everything matched except the loop-filter
            // ref deltas.)
            assert!(sad_per_px == 0.0, "frame {i} must be bit-exact (SAD/px {sad_per_px})");
        }
    }

    // Real GPU decode of an AV1 stream with references: persist `Hardware` evidence
    // tagged with the GPU (adds an av1 codec row to vulkanvideo).
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "vulkanvideo",
        &Evidence::new(ConformanceDimension::Hardware)
            .platform(device.device_name())
            .codec("av1")
            .detail("Vulkan Video AV1 decode (KEY + INTER) with references"),
    )
    .expect("record hardware evidence");
}
