//! M503: Vulkan Video H.265 full-DPB decode on real hardware.
//!
//! The HEVC sibling of M492. M501/M502 built the H.265 parse + `Std*` mapping +
//! decode session; this decodes the *whole* elementary stream: per-picture
//! slice-segment-header parse, picture-order-count, and reference-picture-set
//! DPB management so the P frames decode against their references (not just the
//! IRAP keyframes).
//!
//! The fixture is two GOPs (IDR + four P frames, then an intra CRA + four P
//! frames; 10 pictures, POC 0..9, low-delay). The test asserts every frame
//! decodes to real, non-uniform content, that the P frames differ from their
//! GOP's keyframe (inter prediction ran) and that consecutive frames differ
//! (motion decoded). Runs on the RTX 3060; skips with no adapter / no Vulkan
//! H.265 decode support.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h265_parameter_sets, open_h265_decode_device, to_std_h265_params, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h265_640x480.h265");

#[test]
fn decodes_whole_h265_stream_with_references() {
    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("skipping: no Vulkan adapter");
            return;
        }
        Err(VulkanVideoError::ExtensionUnsupported) | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping: GPU has no Vulkan H.265 decode support");
            return;
        }
        Err(e) => panic!("failed to open H.265 decode device: {e:?}"),
    };

    let ps = extract_h265_parameter_sets(CLIP).expect("parse VPS+SPS+PPS");
    let std = to_std_h265_params(&ps);
    let session = device.create_h265_session(&std, 640, 480).expect("create session");
    let mut decoder = device
        .create_h265_dpb_decoder(&session, &ps)
        .expect("create H.265 DPB decoder");

    let frames = decoder.decode_all(CLIP).expect("decode whole stream");

    // Two GOPs of a keyframe + four P frames.
    assert_eq!(frames.len(), 10, "one frame per coded picture");

    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width, 640);
        assert_eq!(f.height, 480);
        assert_eq!(f.luma.len(), 640 * 480, "one luma byte per sample");
        // A real decoded picture has varied luma; a cleared / failed decode would
        // be uniform. Per-frame proof pixels came out. (Every frame here is also
        // bit-exact against the ffmpeg software decoder, verified out of band.)
        let min = *f.luma.iter().min().unwrap();
        let max = *f.luma.iter().max().unwrap();
        assert!(max > min, "frame {i} luma is uniform ({min}=={max}); no real content");
    }

    let sad = |a: &[u8], b: &[u8]| -> u64 {
        a.iter().zip(b).map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64).sum()
    };

    // The P frames must differ from their GOP's keyframe: if references were
    // ignored (or the decode silently reused the keyframe) they would be
    // identical. GOPs start at index 0 (IDR) and 5 (CRA).
    for gop_start in [0usize, 5] {
        for p in 1..5 {
            assert!(
                sad(&frames[gop_start + p].luma, &frames[gop_start].luma) > 0,
                "P frame {} is byte-identical to its keyframe; inter prediction did not run",
                gop_start + p
            );
        }
    }
    // Consecutive frames differ (motion decoded, not frozen).
    for i in 1..frames.len() {
        assert!(
            sad(&frames[i].luma, &frames[i - 1].luma) > 0,
            "frame {i} is identical to frame {}; no motion decoded",
            i - 1
        );
    }

    // GPU-resident output: the same stream decoded straight to RGBA
    // `wgpu::Texture`s via the ycbcr compute pass (the zero-copy wedge), if a
    // distinct compute queue is available. One texture per coded picture.
    match device.create_h265_dpb_decoder_gpu(&session, &ps) {
        Ok(mut gpu_decoder) => {
            let textures = gpu_decoder
                .decode_all_to_textures(CLIP)
                .expect("decode whole stream to GPU textures");
            assert_eq!(textures.len(), 10, "one GPU texture per coded picture");
            for (i, t) in textures.iter().enumerate() {
                assert_eq!(t.width(), 640);
                assert_eq!(t.height(), 480);
                assert_eq!(t.format(), wgpu::TextureFormat::Rgba8Unorm);
                // Real content through the ycbcr compute pass (near-black + bright).
                let rgba = device.read_rgba_texture(t);
                let min = *rgba.iter().min().unwrap();
                let max = *rgba.iter().max().unwrap();
                assert!(min <= 20 && max >= 200, "GPU frame {i} RGBA range {min}..={max} not real");
            }
        }
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skipping GPU-texture path: no distinct compute queue");
        }
        Err(e) => panic!("failed to create GPU H.265 decoder: {e:?}"),
    }

    // Real GPU decode of a multi-GOP H.265 stream with references: persist
    // `Hardware` evidence tagged with the GPU (adds an h265 codec row to vulkanvideo).
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "vulkanvideo",
        &Evidence::new(ConformanceDimension::Hardware)
            .platform(device.device_name())
            .codec("h265")
            .detail("Vulkan Video multi-GOP H.265 decode with inter references"),
    )
    .expect("record hardware evidence");
}
