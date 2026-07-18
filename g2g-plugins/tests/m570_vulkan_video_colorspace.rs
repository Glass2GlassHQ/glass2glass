//! M570: colour-space-aware YUV -> RGB conversion (the HDR foundation).
//!
//! The decoder converted every stream with a fixed BT.601 matrix, so BT.709 (all
//! HD) and BT.2020 (HDR) content came out with wrong colours. The conversion is now
//! driven by the stream's colour space: the H.264 / H.265 VUI colour description
//! and AV1 `color_config` resolve to a `VideoColorSpace` (matrix + range) that both
//! the CPU `nv12_to_rgba` and the GPU `VkSamplerYcbcrConversion` apply (BT.601 /
//! 709 / 2020, studio / full range). (Primaries and the PQ / HLG transfer function
//! are not applied yet: BT.2020 gets the right matrix here, tone mapping is a later
//! increment.)
//!
//! Two 640x480 clips are encoded from the SAME source, one tagged BT.601
//! (smpte170m), one BT.709. The VUI-parse assertion is GPU-free and always runs:
//! the parser must recover CICP matrix 6 (601) and 1 (709). The reconstruction
//! assertion needs the GPU: decoding both to RGBA textures must recover ~the same
//! image (each YUV was derived from the same RGB with its own matrix, so applying
//! the matching matrix inverts it) - which only holds if the decoder honours each
//! stream's tag. If it ignored the tag and used one matrix for both, the mismatched
//! clip would diverge by tens of levels; honouring it keeps the mean diff tiny.
//!
//! Runs on the RTX 3060 for the reconstruction check; the parse check runs anywhere.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, ColorMatrix, VideoColorSpace,
    VulkanVideoError,
};

const BT601: &[u8] = include_bytes!("fixtures/h264_640x480_bt601.h264");
const BT709: &[u8] = include_bytes!("fixtures/h264_640x480_bt709.h264");
const W: u32 = 640;
const H: u32 = 480;

#[test]
fn vui_colour_description_parses_and_resolves() {
    let ps601 = extract_h264_parameter_sets(BT601).expect("601 sps");
    let ps709 = extract_h264_parameter_sets(BT709).expect("709 sps");
    // CICP matrix codepoints: smpte170m = 6 (BT.601), bt709 = 1.
    assert_eq!(ps601.sps.matrix_coefficients, 6, "601 clip VUI matrix");
    assert_eq!(ps709.sps.matrix_coefficients, 1, "709 clip VUI matrix");
    assert!(!ps601.sps.video_full_range_flag && !ps709.sps.video_full_range_flag);
    // And the resolver maps them to the right matrix (height is only a fallback).
    assert_eq!(
        VideoColorSpace::from_cicp(
            ps601.sps.matrix_coefficients,
            ps601.sps.transfer_characteristics,
            false,
            H
        )
        .matrix,
        ColorMatrix::Bt601
    );
    assert_eq!(
        VideoColorSpace::from_cicp(
            ps709.sps.matrix_coefficients,
            ps709.sps.transfer_characteristics,
            false,
            H
        )
        .matrix,
        ColorMatrix::Bt709
    );
}

#[test]
fn gpu_conversion_honours_stream_colour_matrix() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m570: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("open device: {e:?}"),
    };

    let decode = |clip: &[u8]| -> Option<Vec<u8>> {
        let ps = extract_h264_parameter_sets(clip).expect("sps");
        let session = device.create_h264_session(&ps, W, H).expect("session");
        let mut dec = match device.create_h264_dpb_decoder_gpu(&session, &ps) {
            Ok(d) => d,
            Err(VulkanVideoError::NoComputeQueue) => return None,
            Err(e) => panic!("gpu decoder: {e:?}"),
        };
        let tex = dec.decode_all_to_textures(clip).expect("decode");
        assert_eq!(tex.len(), 1);
        Some(device.read_rgba_texture(&tex[0]))
    };

    let (Some(a), Some(b)) = (decode(BT601), decode(BT709)) else {
        eprintln!("skip m570: no distinct compute queue for the GPU-texture path");
        return;
    };
    assert_eq!(a.len(), (W * H * 4) as usize);

    // Real content in each (not a flat / failed convert).
    let amax = *a.iter().max().unwrap();
    assert!(
        amax >= 200 && *a.iter().min().unwrap() <= 20,
        "601 texture not a real picture"
    );

    // Both must reconstruct ~the same source: mean abs RGB diff tiny. A decoder that
    // ignored the tag (decoding the 709 clip as 601) would diverge by tens of levels.
    let (mut sum, mut n) = (0u64, 0u64);
    for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
        if i % 4 == 3 {
            continue; // alpha
        }
        sum += (x as i32 - y as i32).unsigned_abs() as u64;
        n += 1;
    }
    let mean = sum as f64 / n as f64;
    eprintln!("m570: BT.601 vs BT.709 reconstruction mean abs diff = {mean:.3}");
    assert!(
        mean < 4.0,
        "reconstructions diverge ({mean:.3}); a stream's colour matrix was ignored"
    );
}
