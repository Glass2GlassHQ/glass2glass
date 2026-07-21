//! M552: Tier B (zero-copy) streaming-adapter GPU-texture output for H.265 and
//! AV1, closing the codec gap in m512 (which validated only H.264).
//!
//! Drives [`VulkanStreamDecoder::new_gpu`] / `submit_chunk_texture` the way a
//! wgpu viewer's GPU-texture backend would, for the other two codecs the
//! adapter supports. Each decoded picture comes back as a GPU-resident RGBA
//! `wgpu::Texture` (YUV->RGB applied by g2g's fixed BT.601 compute pass) with no
//! CPU readback in the decode path. m535 exercised these `decode_all_to_textures`
//! paths through the pipeline *element*; this exercises the consumer-facing
//! `streamdec` adapter API for H.265 + AV1, which had only ever run for H.264.
//!
//! H.265 is asserted strictly (this driver's HEVC decode is byte-exact): the GPU
//! RGBA must match the already-bit-exact CPU I420 path converted with the same
//! BT.601-limited matrix the compute pass uses. AV1 is asserted loosely (this
//! NVIDIA driver's AV1 decode is run-to-run nondeterministic, see m508): real
//! reference-dependent textures, but no strict CPU anchor.
//!
//! Both run in ONE test function, sequentially, because parallel Vulkan device
//! creation SIGSEGVs (see m504 / m535). Runs on the RTX 3060; skips with no
//! Vulkan decode support / no distinct compute queue.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::streamdec::{VideoCodec, VideoPixelLayout, VulkanStreamDecoder};
use g2g_plugins::vulkanvideo::{
    open_av1_decode_device, open_h265_decode_device, VulkanVideoDevice, VulkanVideoError,
};

const H265: &[u8] = include_bytes!("fixtures/h265_640x480.h265");
const AV1: &[u8] = include_bytes!("fixtures/av1_640x480.obu");
const W: usize = 640;
const H: usize = 480;

/// BT.601 limited-range YUV -> RGB, the fixed conversion g2g's ycbcr compute
/// pass runs on the decoder's NV12 output (YCBCR_601, narrow range) for every
/// codec. Reconstructs a reference from the already-validated CPU I420 path so
/// the GPU pass can be checked without an external RGB fixture. Same matrix as
/// m512.
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

/// LEB128 as used by AV1 OBU sizes.
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

/// Each AV1 `OBU_FRAME` as its own chunk (the seq header is parsed by `new_gpu`
/// from the whole stream, so a per-frame chunk need not carry it). Mirrors m508.
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

/// Common checks over a decoded texture: right dims / format, real picture.
fn assert_real_texture(
    t: &g2g_plugins::streamdec::DecodedVideoTexture,
    dec: &VulkanStreamDecoder,
) -> Vec<u8> {
    assert_eq!(t.width, W as u32);
    assert_eq!(t.height, H as u32);
    assert_eq!(t.texture.format(), wgpu::TextureFormat::Rgba8Unorm);
    assert_eq!(t.texture.width(), W as u32);
    assert_eq!(t.texture.height(), H as u32);
    let rgba = dec.read_rgba_texture(&t.texture);
    assert_eq!(rgba.len(), W * H * 4);
    let min = *rgba.iter().min().unwrap();
    let max = *rgba.iter().max().unwrap();
    assert!(
        min <= 20 && max >= 200,
        "RGBA range {min}..={max} not a real picture"
    );
    rgba
}

/// Skip helper: returns None on a decode-support gap (no adapter / codec / compute queue).
fn open_or_skip(
    dev: Result<VulkanVideoDevice, VulkanVideoError>,
    which: &str,
) -> Option<VulkanVideoDevice> {
    match dev {
        Ok(d) => Some(d),
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m552: no Vulkan {which} decode adapter");
            None
        }
        Err(e) => panic!("open {which} decode device: {e:?}"),
    }
}

#[test]
fn streamdec_streams_h265_and_av1_gpu_textures() {
    // ---- H.265: strict CPU-I420 BT.601 anchor (byte-exact driver path) ----
    if let Some(device) = open_or_skip(block_on(open_h265_decode_device()), "H.265") {
        let mut dec = match VulkanStreamDecoder::new_gpu(device, VideoCodec::H265, H265) {
            Ok(d) => d,
            Err(VulkanVideoError::NoComputeQueue) => {
                eprintln!("skip m552 H.265: no distinct compute queue for the GPU path");
                return;
            }
            Err(e) => panic!("build GPU-mode H.265 adapter: {e:?}"),
        };
        assert_eq!(dec.width(), W as u32);
        assert_eq!(dec.height(), H as u32);

        // Whole elementary stream in one call (internal DPB decodes every picture).
        let textures = dec
            .submit_chunk_texture(H265, true)
            .expect("H.265 submit chunk (GPU)");
        assert_eq!(
            textures.len(),
            10,
            "one GPU texture per H.265 coded picture"
        );
        // A CPU-mode method on a GPU-mode decoder is rejected, not misdecoded.
        assert!(matches!(
            dec.submit_chunk(H265, true),
            Err(VulkanVideoError::WrongOutputMode)
        ));

        let readbacks: Vec<Vec<u8>> = textures
            .iter()
            .map(|t| assert_real_texture(t, &dec))
            .collect();
        // Inter frames differ from their GOP's IDR (GPU DPB reference decode ran).
        // The fixture is two GOPs starting at 0 and 5 (IDR/CRA), as in m503/m517.
        for gop_start in [0usize, 5] {
            for p in 1..5 {
                assert_ne!(
                    readbacks[gop_start + p],
                    readbacks[gop_start],
                    "H.265 texture {} identical to its IRAP; GPU reference decode failed",
                    gop_start + p
                );
            }
        }

        // Anchor: a fresh CPU-mode H.265 decoder gives bit-exact I420; the GPU
        // RGBA must match it under the compute pass's BT.601 matrix.
        let device2 = open_or_skip(block_on(open_h265_decode_device()), "H.265").expect("re-open");
        let mut cpu = VulkanStreamDecoder::new(device2, VideoCodec::H265, H265)
            .expect("build CPU-mode H.265 adapter");
        let cpu_frames = cpu.submit_chunk(H265, true).expect("H.265 CPU decode");
        assert_eq!(cpu_frames.len(), textures.len());
        let mut worst = 0i32;
        for (i, f) in cpu_frames.iter().enumerate() {
            assert_eq!(f.layout, VideoPixelLayout::Y_U_V420);
            let expected = i420_to_rgba_bt601(&f.data, W, H);
            let (mut sum, mut n) = (0u64, 0u64);
            for (j, (&g, &e)) in readbacks[i].iter().zip(expected.iter()).enumerate() {
                if j % 4 == 3 {
                    continue; // ignore alpha
                }
                let d = (g as i32 - e as i32).abs();
                worst = worst.max(d);
                sum += d as u64;
                n += 1;
            }
            let mean = sum as f64 / n as f64;
            assert!(
                mean < 3.0,
                "H.265 frame {i}: GPU RGBA mean abs diff {mean:.2} vs BT.601(I420)"
            );
        }
        eprintln!(
            "m552 H.265: 10 GPU-resident RGBA textures match CPU I420 (worst channel diff {worst})"
        );
    }

    // ---- AV1: loose anchor (driver AV1 decode is run-to-run nondeterministic) ----
    if let Some(device) = open_or_skip(block_on(open_av1_decode_device()), "AV1") {
        let mut dec = match VulkanStreamDecoder::new_gpu(device, VideoCodec::Av1, AV1) {
            Ok(d) => d,
            Err(VulkanVideoError::NoComputeQueue) => {
                eprintln!("skip m552 AV1: no distinct compute queue for the GPU path");
                return;
            }
            Err(e) => panic!("build GPU-mode AV1 adapter: {e:?}"),
        };
        assert_eq!(dec.width(), W as u32);
        assert_eq!(dec.height(), H as u32);

        // One OBU frame per call, DPB carried across calls (the seq header was
        // parsed by new_gpu from the whole stream).
        let chunks = split_obu_frames(AV1);
        assert_eq!(chunks.len(), 10, "fixture is 1 KEY + 9 INTER frames");
        let mut textures = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            textures.extend(
                dec.submit_chunk_texture(chunk, i == 0)
                    .expect("AV1 submit chunk (GPU)"),
            );
        }
        assert_eq!(textures.len(), 10, "one GPU texture per AV1 coded picture");

        let readbacks: Vec<Vec<u8>> = textures
            .iter()
            .map(|t| assert_real_texture(t, &dec))
            .collect();
        // Inter frames differ from the keyframe: the GPU DPB reference decode ran
        // (this holds despite AV1's small nondeterminism, which is <=few % of
        // samples off by <=3, far below a whole-frame difference).
        for p in 1..10 {
            assert_ne!(
                readbacks[p], readbacks[0],
                "AV1 texture {p} identical to the keyframe; GPU reference decode failed"
            );
        }
        eprintln!("m552 AV1: 10 GPU-resident RGBA textures, zero-copy path (loose anchor)");
    }
}
