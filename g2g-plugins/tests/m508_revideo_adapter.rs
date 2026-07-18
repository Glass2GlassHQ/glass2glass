//! M508: the fork-ready re_video adapter ([`g2g_plugins::revideo`]).
//!
//! Drives [`VulkanStreamDecoder`] the way a Rerun `re_video::decode::AsyncDecoder`
//! backend would: one coded sample per `submit_chunk`, DPB state carried across
//! calls, output as packed I420 (re_video's native CPU frame layout). Proves the
//! adapter satisfies that contract on real hardware, so a small re_video fork can
//! wrap it in one `impl AsyncDecoder` (the wgpu-texture wedge, Tier A readback).
//!
//! Runs on the RTX 3060; skips with no AV1 decode adapter. Optional bit-exact
//! check vs an ffmpeg `yuv420p` (== I420) dump via `G2G_AV1_REF`.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::revideo::{
    VideoCodec, VideoColorRange, VideoMatrixCoefficients, VideoPixelLayout, VulkanStreamDecoder,
};
use g2g_plugins::vulkanvideo::{open_av1_decode_device, VulkanVideoError};

const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480.obu");
const W: usize = 640;
const H: usize = 480;

/// LEB128 unsigned (AV1 4.10.5). Returns (value, bytes_read).
fn leb128(d: &[u8], mut p: usize) -> Option<(u64, usize)> {
    let start = p;
    let mut v = 0u64;
    for i in 0..8 {
        let b = *d.get(p)?;
        p += 1;
        v |= ((b & 0x7f) as u64) << (7 * i);
        if b & 0x80 == 0 {
            return Some((v, p - start));
        }
    }
    None
}

/// Split a low-overhead AV1 OBU stream into one chunk per `OBU_FRAME` (type 6),
/// each chunk being that OBU's full bytes (header + size field + payload). This
/// is the per-sample chunking a demuxer feeds an `AsyncDecoder`.
fn split_obu_frames(stream: &[u8]) -> Vec<&[u8]> {
    const OBU_FRAME: u8 = 6;
    let mut out = Vec::new();
    let mut p = 0;
    while p < stream.len() {
        let obu_start = p;
        let b = stream[p];
        let obu_type = (b >> 3) & 0xf;
        let ext = (b >> 2) & 1;
        let has_size = (b >> 1) & 1;
        p += 1;
        if ext == 1 {
            p += 1;
        }
        let payload_len = if has_size == 1 {
            let (sz, n) = leb128(stream, p).expect("valid leb128 size");
            p += n;
            sz as usize
        } else {
            stream.len() - p
        };
        let end = p + payload_len;
        if obu_type == OBU_FRAME {
            out.push(&stream[obu_start..end]);
        }
        p = end;
    }
    out
}

#[test]
fn revideo_adapter_streams_i420_frames() {
    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m508: no Vulkan AV1 decode adapter");
            return;
        }
        Err(e) => panic!("open AV1 decode device: {e:?}"),
    };

    let mut dec =
        VulkanStreamDecoder::new(device, VideoCodec::Av1, CLIP).expect("build re_video adapter");
    assert_eq!(dec.width(), W as u32);
    assert_eq!(dec.height(), H as u32);

    let chunks = split_obu_frames(CLIP);
    assert_eq!(chunks.len(), 10, "fixture is 1 KEY + 9 INTER frames");

    // Feed one coded sample at a time; collect the decoded I420 frames.
    let mut frames = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let out = dec.submit_chunk(chunk, i == 0).expect("submit chunk");
        frames.extend(out);
    }
    assert_eq!(frames.len(), 10, "one I420 frame per coded sample");

    let i420_len = W * H + 2 * (W / 2) * (H / 2);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width, W as u32);
        assert_eq!(f.height, H as u32);
        assert_eq!(f.layout, VideoPixelLayout::Y_U_V420);
        assert_eq!(f.data.len(), i420_len, "frame {i} packed I420 size");
        // The fixture is BT.601-ish content, sequence header matrix_coefficients
        // is unspecified -> guessed BT.709; range limited. Assert the mapping ran.
        assert!(matches!(
            f.coefficients,
            VideoMatrixCoefficients::Bt709 | VideoMatrixCoefficients::Bt601
        ));
        assert_eq!(f.range, VideoColorRange::Limited);
        let y = &f.data[..W * H];
        let min = *y.iter().min().unwrap();
        let max = *y.iter().max().unwrap();
        assert!(max > min, "frame {i} luma is uniform ({min}=={max})");
    }

    // Consecutive frames must differ (animated content decoded against refs).
    for i in 1..frames.len() {
        assert!(
            frames[i].data != frames[i - 1].data,
            "frame {i} == {}",
            i - 1
        );
    }

    // reset() then re-feed from the keyframe reproduces the first frame, proving
    // the seek path rebuilds a clean DPB and re-decodes a coherent picture.
    //
    // The comparison is TOLERANT, not byte-exact, and only for AV1: this NVIDIA
    // driver's AV1 decode is run-to-run nondeterministic (the same coded bytes
    // decode to a slightly different result across runs, a few tenths of a percent
    // of samples off by <=3, even for an intra keyframe on the main thread), so an
    // exact assertion is flaky. See `vulkan_thread_teardown::av1_decode_matches_
    // across_threads` (ignored) for the same driver behaviour cross-thread; the
    // H.264 / H.265 paths ARE byte-exact (see `m503` and the reset checks there).
    // A broken reset would return a stale or garbage frame (most bytes differ, or
    // large deltas), which this still catches.
    dec.reset().expect("reset");
    let f0_again = dec
        .submit_chunk(chunks[0], true)
        .expect("re-decode frame 0");
    assert_eq!(f0_again.len(), 1);
    let (a, b) = (&f0_again[0].data, &frames[0].data);
    assert_eq!(a.len(), b.len());
    let mut n_diff = 0u64;
    let mut max_delta = 0i32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i32 - *y as i32).abs();
        if d != 0 {
            n_diff += 1;
            max_delta = max_delta.max(d);
        }
    }
    // < 2% of samples differ and none by more than 16: a coherent re-decode within
    // the AV1 driver's nondeterminism, not a broken reset (which differs wholesale).
    assert!(
        n_diff * 100 < a.len() as u64 * 2 && max_delta <= 16,
        "reset + re-decode diverged too far: {n_diff}/{} bytes differ, max delta {max_delta}",
        a.len()
    );

    // Bit-exactness of every frame vs an ffmpeg yuv420p (I420) reference: full
    // I420 (Y + U + V), so this also checks the NV12 -> I420 chroma deinterleave.
    if let Ok(path) = std::env::var("G2G_AV1_REF") {
        let ref_yuv = std::fs::read(&path).expect("read G2G_AV1_REF");
        assert!(
            ref_yuv.len() >= i420_len * frames.len(),
            "reference too short"
        );
        for (i, f) in frames.iter().enumerate() {
            let r = &ref_yuv[i * i420_len..(i + 1) * i420_len];
            assert_eq!(&f.data, r, "frame {i} must be bit-exact I420 vs ffmpeg");
        }
        eprintln!(
            "m508: all {} frames bit-exact I420 vs ffmpeg reference",
            frames.len()
        );
    }
}
