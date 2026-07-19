//! M512: Tier B (zero-copy) re_video adapter output.
//!
//! Drives [`VulkanStreamDecoder::new_gpu`] the way a Rerun `re_video` GPU-texture
//! backend would: one coded sample per `submit_chunk_texture`, DPB state carried
//! across calls, each frame handed back as a GPU-resident RGBA `wgpu::Texture`
//! (YUV->RGB already applied by g2g's compute pass) with NO CPU readback in the
//! decode path. This is the wedge Tier B relies on: a re_video fork wraps this in
//! a GPU-texture `FrameContent` and passes the texture straight to `re_renderer`,
//! skipping both the readback and re_renderer's upload + colour convert.
//!
//! Uses H.264, the codec the Tier-A PoC settled on (bit-exact across threads, the
//! common Rerun codec). Runs on the RTX 3060; skips with no adapter / no compute
//! queue. Correctness is anchored to the CPU adapter path (Tier A, already
//! bit-exact vs ffmpeg): the GPU RGBA readback must match a from-I420 reference
//! reconstructed with the same BT.601-limited matrix the compute pass uses.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::revideo::{VideoCodec, VideoPixelLayout, VulkanStreamDecoder};
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoError};

const CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const W: usize = 640;
const H: usize = 480;

/// BT.601 limited-range YUV -> RGB, the conversion g2g's ycbcr compute pass runs
/// on the decoder's NV12 output. Matches `revideo`'s default colorimetry for
/// H.264 (`Bt601` / `Limited`). Used only to reconstruct a reference from the
/// already-validated CPU I420 path, so the GPU pass can be checked without an
/// external RGB fixture.
fn i420_to_rgba_bt601(i420: &[u8], w: usize, h: usize) -> Vec<u8> {
    let y_plane = &i420[..w * h];
    let cw = w / 2;
    let u_plane = &i420[w * h..w * h + cw * (h / 2)];
    let v_plane = &i420[w * h + cw * (h / 2)..];
    let mut out = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let yy = y_plane[y * w + x] as f32;
            let uu = u_plane[(y / 2) * cw + x / 2] as f32;
            let vv = v_plane[(y / 2) * cw + x / 2] as f32;
            // Studio-swing: Y in [16,235], C centred at 128 with [16,240] swing.
            let yl = (yy - 16.0) * (255.0 / 219.0);
            let cb = uu - 128.0;
            let cr = vv - 128.0;
            let r = yl + 1.596 * cr;
            let g = yl - 0.391 * cb - 0.813 * cr;
            let b = yl + 2.018 * cb;
            let o = (y * w + x) * 4;
            out[o] = r.clamp(0.0, 255.0) as u8;
            out[o + 1] = g.clamp(0.0, 255.0) as u8;
            out[o + 2] = b.clamp(0.0, 255.0) as u8;
            out[o + 3] = 255;
        }
    }
    out
}

#[test]
fn revideo_adapter_streams_gpu_textures() {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m512: no Vulkan H.264 decode adapter");
            return;
        }
        Err(e) => panic!("open H.264 decode device: {e:?}"),
    };

    let mut dec = match VulkanStreamDecoder::new_gpu(device, VideoCodec::H264, CLIP) {
        Ok(d) => d,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skip m512: decode device has no distinct compute queue for the GPU path");
            return;
        }
        Err(e) => panic!("build GPU-mode re_video adapter: {e:?}"),
    };
    assert_eq!(dec.width(), W as u32);
    assert_eq!(dec.height(), H as u32);

    // Feed the whole elementary stream as one chunk (the H.264 splitter inside the
    // decoder frames it into access units), collecting GPU-resident textures.
    let textures = dec
        .submit_chunk_texture(CLIP, true)
        .expect("submit chunk (GPU)");
    assert_eq!(textures.len(), 10, "one GPU texture per coded picture");

    // A CPU-mode method on a GPU-mode decoder is rejected, not silently misdecoded.
    assert!(matches!(
        dec.submit_chunk(CLIP, true),
        Err(VulkanVideoError::WrongOutputMode)
    ));

    let mut readbacks = Vec::new();
    for (i, t) in textures.iter().enumerate() {
        assert_eq!(t.width, W as u32);
        assert_eq!(t.height, H as u32);
        assert_eq!(t.texture.format(), wgpu::TextureFormat::Rgba8Unorm);
        assert_eq!(t.texture.width(), W as u32);
        assert_eq!(t.texture.height(), H as u32);
        // Readback is the test's verification only, never the pipeline path.
        // Exercise both the adapter helper and the shared-context free helper
        // (the one a re_renderer-style consumer uses on `gpu_context()`).
        let rgba = dec.read_rgba_texture(&t.texture);
        if i == 0 {
            let via_ctx = g2g_plugins::gpu::read_rgba_texture(&dec.gpu_context(), &t.texture);
            assert_eq!(
                via_ctx, rgba,
                "gpu::read_rgba_texture matches the adapter readback"
            );
        }
        assert_eq!(rgba.len(), W * H * 4);
        let min = *rgba.iter().min().unwrap();
        let max = *rgba.iter().max().unwrap();
        assert!(
            min <= 20 && max >= 200,
            "frame {i} RGBA range {min}..={max} not a real picture"
        );
        readbacks.push(rgba);
    }

    // Inter frames differ from their GOP's IDR (DPB reference decode ran on the
    // GPU and the ycbcr pass restored each slot). GOPs start at 0 and 5.
    for gop_start in [0usize, 5] {
        for p in 1..5 {
            assert_ne!(
                readbacks[gop_start + p],
                readbacks[gop_start],
                "texture {} identical to its IDR; GPU DPB reference decode failed",
                gop_start + p
            );
        }
    }

    // Correctness anchor: the GPU compute-pass RGBA must match the CPU adapter's
    // I420 (Tier A, bit-exact vs ffmpeg) converted with the same BT.601 matrix.
    // A fresh CPU-mode decoder over the same stream gives the reference I420.
    let device2 = block_on(open_h264_decode_device()).expect("re-open decode device");
    let mut cpu =
        VulkanStreamDecoder::new(device2, VideoCodec::H264, CLIP).expect("build CPU-mode adapter");
    let cpu_frames = cpu.submit_chunk(CLIP, true).expect("CPU decode");
    assert_eq!(cpu_frames.len(), textures.len());

    let mut worst = 0i32;
    for (i, f) in cpu_frames.iter().enumerate() {
        assert_eq!(f.layout, VideoPixelLayout::Y_U_V420);
        let expected = i420_to_rgba_bt601(&f.data, W, H);
        // Ignore alpha (index %4 == 3); compare colour channels with a small
        // tolerance for the shader's fixed-point vs this f32 reference.
        let mut sum = 0u64;
        let mut n = 0u64;
        for (j, (&g, &e)) in readbacks[i].iter().zip(expected.iter()).enumerate() {
            if j % 4 == 3 {
                continue;
            }
            let d = (g as i32 - e as i32).abs();
            worst = worst.max(d);
            sum += d as u64;
            n += 1;
        }
        let mean = sum as f64 / n as f64;
        assert!(
            mean < 3.0,
            "frame {i}: GPU RGBA mean abs diff {mean:.2} vs BT.601(I420) too large"
        );
    }
    eprintln!(
        "m512: {} GPU-resident RGBA textures, zero-copy path; match CPU I420 (worst channel diff {worst})",
        textures.len()
    );
}
