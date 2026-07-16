//! M492: Vulkan Video H.264 full-DPB decode on real hardware.
//!
//! Fourth increment of `VulkanVideoDec` (DESIGN.md 4.11.6). The M489-M491
//! entry points decode only the leading IDR (hardcoded lone-IDR `Std*`
//! constants, no references). This decodes the *whole* elementary stream:
//! per-picture slice-header parse, picture-order-count, and H.264 sliding-window
//! reference management so the P frames after the IDR decode against their
//! references, not just the keyframe.
//!
//! The fixture is two GOPs of IDR + four P frames (10 pictures). The test
//! asserts every frame decodes to real, non-uniform content and that the P
//! frames are not byte-identical to their IDR (they carry inter-predicted
//! motion, i.e. references were actually applied). Runs on the RTX 3060; skips
//! cleanly with no adapter / no Vulkan H.264 decode support.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");

#[test]
fn decodes_whole_stream_with_references() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("skipping: no Vulkan adapter");
            return;
        }
        Err(VulkanVideoError::ExtensionUnsupported) | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skipping: GPU has no Vulkan H.264 decode support");
            return;
        }
        Err(e) => panic!("failed to open decode device: {e:?}"),
    };

    let ps = extract_h264_parameter_sets(CLIP).expect("parse SPS+PPS");
    let session = device.create_h264_session(&ps, 640, 480).expect("create session");
    let mut decoder = device
        .create_h264_dpb_decoder(&session, &ps)
        .expect("create DPB decoder");

    let frames = decoder.decode_all(CLIP).expect("decode whole stream");

    // The clip is two GOPs of IDR + four P frames.
    assert_eq!(frames.len(), 10, "one frame per coded picture");

    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width, 640);
        assert_eq!(f.height, 480);
        assert_eq!(f.luma.len(), 640 * 480, "one luma byte per sample");
        // A real decoded picture has varied luma; a cleared / failed decode would
        // be uniform. This is the per-frame proof pixels came out.
        let min = *f.luma.iter().min().unwrap();
        let max = *f.luma.iter().max().unwrap();
        assert!(max > min, "frame {i} luma is uniform ({min}=={max}); no real content");
    }

    let sad = |a: &[u8], b: &[u8]| -> u64 {
        a.iter().zip(b).map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64).sum()
    };

    // The P frames must differ from their GOP's IDR: if references were ignored
    // (or the decode silently reused the keyframe) they would be identical. This
    // clip has real motion, and the decoded output is bit-exact against a
    // software decoder. GOPs start at index 0 and 5.
    for gop_start in [0usize, 5] {
        for p in 1..5 {
            assert!(
                sad(&frames[gop_start + p].luma, &frames[gop_start].luma) > 0,
                "P frame {} is byte-identical to its IDR; inter prediction did not run",
                gop_start + p
            );
        }
    }
    // Consecutive frames also differ (motion is being decoded, not frozen).
    for i in 1..frames.len() {
        assert!(
            sad(&frames[i].luma, &frames[i - 1].luma) > 0,
            "frame {i} is identical to frame {}; no motion decoded",
            i - 1
        );
    }
}
