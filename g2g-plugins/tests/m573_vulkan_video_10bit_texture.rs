//! M573: 10-bit YCbCr -> RGBA GPU-texture path (zero-copy HDR decode), on real
//! hardware.
//!
//! M571 / M572 landed 10-bit HEVC Main 10 and AV1 decode on the *system* readback
//! path; the GPU-texture path (the Rerun / Bevy zero-copy wedge) still rejected
//! 10-bit with `UnsupportedStream` because its `VkSamplerYcbcrConversion` +
//! compute pass were hardcoded to 8-bit NV12 -> `Rgba8Unorm`. The converter now
//! picks its formats from the decode bit depth: a `G10X6` (10-bit) frame samples
//! through a 10-bit ycbcr conversion and stores into an `R16G16B16A16_SFLOAT`
//! image (the `rgba16f` shader), imported as a `Rgba16Float` `wgpu::Texture`. The
//! float target preserves the full 10-bit precision and is where the later PQ /
//! HLG transfer will operate.
//!
//! HEVC Main 10 is asserted strictly: each GPU-resident RGBA texture must match a
//! CPU reference built from the already-bit-exact system 10-bit NV12 decode,
//! converted with the exact colour matrix + range the stream carries (BT.2020
//! narrow for this HDR clip). This catches a bit-depth mishandling: a converter
//! that read `G10X6` as 8-bit would produce grossly wrong colour, not a <2% diff.
//! AV1 10-bit is asserted loosely (this NVIDIA driver's AV1 decode is run-to-run
//! nondeterministic, see m508): real `Rgba16Float` textures, correct dims, inter
//! frames differ from the keyframe (the 10-bit GPU DPB reference decode ran).
//!
//! Both codecs run in ONE test function, sequentially: parallel Vulkan device
//! creation SIGSEGVs (see m504 / m535 / m552). Runs on the RTX 3060; skips with no
//! Vulkan decode support / no distinct compute queue / no 10-bit decode.
#![cfg(all(any(target_os = "linux", target_os = "windows"), feature = "vulkan-video"))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, extract_h265_parameter_sets, open_av1_decode_device,
    open_h265_decode_device, to_std_av1_seq_header, to_std_h265_params, ColorMatrix,
    VideoColorSpace, VulkanVideoDevice, VulkanVideoError,
};

const HEVC: &[u8] = include_bytes!("fixtures/h265_640x480_main10.hevc");
const AV1: &[u8] = include_bytes!("fixtures/av1_640x480_10bit.obu");
const W: usize = 640;
const H: usize = 480;

/// Little-endian 16-bit sample `i` from a byte plane.
fn le16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[2 * i], b[2 * i + 1]])
}

/// IEEE half -> f32 (the `Rgba16Float` readback is packed halfs).
fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = match exp {
        0 if mant == 0 => (sign as u32) << 31,
        0 => {
            // Subnormal: normalize.
            let mut e = -14i32;
            let mut m = mant as u32;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            ((sign as u32) << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
        0x1f => ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13),
        _ => ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | ((mant as u32) << 13),
    };
    f32::from_bits(bits)
}

/// The `(Kr, Kb)` luma weights of a matrix (`Kg = 1 - Kr - Kb`), mirroring the
/// private `VideoColorSpace::luma_weights` the converter uses.
fn luma_weights(m: ColorMatrix) -> (f32, f32) {
    match m {
        ColorMatrix::Bt601 => (0.299, 0.114),
        ColorMatrix::Bt709 => (0.2126, 0.0722),
        ColorMatrix::Bt2020Ncl => (0.2627, 0.0593),
    }
}

/// CPU reference: a 10-bit (`G10X6`) NV12 luma+chroma pair -> per-pixel RGB in
/// `[0, 1]`, matching what the fixed-function `VkSamplerYcbcrConversion` + compute
/// pass produce (nearest-neighbour chroma here; the hardware uses a linear
/// filter, so a small mean diff is expected). Narrow / full range and the matrix
/// come from `color`, exactly as the converter resolves them. Samples are the
/// top-10-bit G10X6 packing, so `>> 6` to the 0..=1023 value.
fn nv12_10bit_to_rgb(luma: &[u8], chroma: &[u8], color: VideoColorSpace) -> Vec<[f32; 3]> {
    let cw = W / 2;
    let (kr, kb) = luma_weights(color.matrix);
    let kg = 1.0 - kr - kb;
    // 10-bit studio: Y in 64..940 (219*4), C centred 512 spanning 896 (224*4).
    let (y_off, y_span, c_off, c_span) =
        if color.full_range { (0.0, 1023.0, 512.0, 1023.0) } else { (64.0, 876.0, 512.0, 896.0) };
    let cr_r = 2.0 * (1.0 - kr);
    let cb_b = 2.0 * (1.0 - kb);
    let cr_g = 2.0 * kr * (1.0 - kr) / kg;
    let cb_g = 2.0 * kb * (1.0 - kb) / kg;
    let mut out = Vec::with_capacity(W * H);
    for y in 0..H {
        for x in 0..W {
            let yy = ((le16(luma, y * W + x) >> 6) as f32 - y_off) / y_span;
            let ci = (y / 2) * cw + (x / 2);
            let cb = ((le16(chroma, 2 * ci) >> 6) as f32 - c_off) / c_span;
            let cr = ((le16(chroma, 2 * ci + 1) >> 6) as f32 - c_off) / c_span;
            let r = yy + cr_r * cr;
            let g = yy - cr_g * cr - cb_g * cb;
            let b = yy + cb_b * cb;
            out.push([r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)]);
        }
    }
    out
}

/// Decode an `Rgba16Float` readback (8 bytes/texel, 4 halfs) to per-pixel RGB.
fn rgba16f_to_rgb(bytes: &[u8]) -> Vec<[f32; 3]> {
    assert_eq!(bytes.len(), W * H * 8, "Rgba16Float readback is 8 bytes/pixel");
    (0..W * H)
        .map(|p| {
            let o = p * 8;
            [
                half_to_f32(le16(bytes, o / 2)),
                half_to_f32(le16(bytes, o / 2 + 1)),
                half_to_f32(le16(bytes, o / 2 + 2)),
            ]
        })
        .collect()
}

/// `Some(device)` or `None` on a decode-support gap (no adapter / codec / queue).
fn open_or_skip(
    dev: Result<VulkanVideoDevice, VulkanVideoError>,
    which: &str,
) -> Option<VulkanVideoDevice> {
    match dev {
        Ok(d) => Some(d),
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m573: no Vulkan {which} decode adapter");
            None
        }
        Err(e) => panic!("open {which} decode device: {e:?}"),
    }
}

#[test]
fn ten_bit_gpu_texture_hevc_and_av1() {
    // ---- HEVC Main 10: strict CPU-reference anchor (bit-exact system decode) ----
    let ps = extract_h265_parameter_sets(HEVC).expect("vps/sps/pps");
    assert_eq!(ps.sps.bit_depth_luma_minus8, 2, "fixture is not 10-bit; feature untested");
    // The exact colour space the converter resolves for this stream.
    let color = VideoColorSpace::from_cicp(
        ps.sps.matrix_coefficients,
        ps.sps.transfer_characteristics,
        ps.sps.video_full_range_flag,
        H as u32,
    );

    // CPU reference frames (system 10-bit NV12 -> RGB), then drop that device
    // before opening the GPU one (never two Vulkan devices live at once).
    let cpu_rgb: Option<Vec<Vec<[f32; 3]>>> =
        open_or_skip(block_on(open_h265_decode_device()), "H.265").map(|device| {
            let std = to_std_h265_params(&ps);
            let session = device.create_h265_session(&std, W as u32, H as u32).expect("session");
            let mut cpu = device.create_h265_dpb_decoder(&session, &ps).expect("cpu decoder");
            let frames = cpu.decode_all(HEVC).expect("cpu 10-bit decode");
            frames.iter().map(|f| nv12_10bit_to_rgb(&f.luma, &f.chroma, color)).collect()
        });
    let Some(cpu_rgb) = cpu_rgb else { return };
    assert!(!cpu_rgb.is_empty(), "no CPU reference frames");

    // GPU-texture path over a fresh device.
    if let Some(device) = open_or_skip(block_on(open_h265_decode_device()), "H.265")
    {
        let std = to_std_h265_params(&ps);
        let session = device.create_h265_session(&std, W as u32, H as u32).expect("session");
        let mut dec = match device.create_h265_dpb_decoder_gpu(&session, &ps) {
            Ok(d) => d,
            Err(VulkanVideoError::NoComputeQueue) => {
                eprintln!("skip m573 HEVC: no distinct compute queue for the GPU path");
                return;
            }
            Err(VulkanVideoError::UnsupportedStream) => {
                eprintln!("skip m573 HEVC: no 10-bit HEVC decode on this device");
                return;
            }
            Err(e) => panic!("gpu decoder: {e:?}"),
        };
        let texes = dec.decode_all_to_textures(HEVC).expect("10-bit HEVC -> textures");
        assert_eq!(texes.len(), cpu_rgb.len(), "one GPU texture per coded picture");

        let mut readbacks = Vec::new();
        for (i, t) in texes.iter().enumerate() {
            assert_eq!(t.format(), wgpu::TextureFormat::Rgba16Float, "frame {i} not 10-bit RGBA16F");
            assert_eq!((t.width(), t.height()), (W as u32, H as u32));
            let rgb = rgba16f_to_rgb(&device.read_rgba_texture(t));
            // Real content: luma spans a range (a failed decode is flat).
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for px in &rgb {
                let l = px[0] + px[1] + px[2];
                lo = lo.min(l);
                hi = hi.max(l);
            }
            assert!(hi - lo > 0.25, "frame {i} nearly uniform ({lo:.3}..={hi:.3}); decode failed");
            readbacks.push(rgb);
        }

        // Strict anchor: GPU RGBA matches the CPU reference under the same matrix +
        // range. Tolerance covers linear-vs-nearest chroma + f16 quantization, but
        // is far below what an 8-bit misread of a 10-bit frame would produce.
        let mut worst_mean = 0.0f64;
        for (i, (gpu, cpu)) in readbacks.iter().zip(cpu_rgb.iter()).enumerate() {
            let (mut sum, mut n, mut worst) = (0.0f64, 0u64, 0.0f32);
            for (g, c) in gpu.iter().zip(cpu.iter()) {
                for k in 0..3 {
                    let d = (g[k] - c[k]).abs();
                    sum += d as f64;
                    worst = worst.max(d);
                    n += 1;
                }
            }
            let mean = sum / n as f64;
            worst_mean = worst_mean.max(mean);
            assert!(
                mean < 0.02,
                "HEVC frame {i}: GPU 10-bit RGBA mean abs diff {mean:.4} (worst {worst:.3}) vs CPU reference"
            );
        }
        // Inter frames differ from their GOP's IRAP (the 10-bit GPU DPB reference
        // decode ran). The fixture is a short closed-GOP clip starting on an IRAP.
        for p in 1..readbacks.len() {
            assert_ne!(readbacks[p], readbacks[0], "HEVC texture {p} identical to frame 0");
        }
        eprintln!(
            "m573 HEVC Main 10: {} GPU Rgba16Float textures match CPU reference (worst frame mean {worst_mean:.4})",
            readbacks.len()
        );
    }

    // ---- AV1 10-bit: loose anchor (driver AV1 decode is nondeterministic) ----
    let seq = extract_av1_sequence_header(AV1).expect("sequence header");
    assert_eq!(seq.color.bit_depth, 10, "fixture is not 10-bit; feature untested");
    if let Some(device) = open_or_skip(block_on(open_av1_decode_device()), "AV1") {
        let std = to_std_av1_seq_header(&seq);
        let session = device.create_av1_session(&std, W as u32, H as u32).expect("session");
        let mut dec = match device.create_av1_dpb_decoder_gpu(&session, &seq) {
            Ok(d) => d,
            Err(VulkanVideoError::NoComputeQueue) => {
                eprintln!("skip m573 AV1: no distinct compute queue for the GPU path");
                return;
            }
            Err(VulkanVideoError::UnsupportedStream) => {
                eprintln!("skip m573 AV1: no 10-bit AV1 decode on this device");
                return;
            }
            Err(e) => panic!("gpu AV1 decoder: {e:?}"),
        };
        let texes = dec.decode_all_to_textures(AV1).expect("10-bit AV1 -> textures");
        assert!(!texes.is_empty(), "no AV1 textures");
        let mut readbacks = Vec::new();
        for (i, t) in texes.iter().enumerate() {
            assert_eq!(t.format(), wgpu::TextureFormat::Rgba16Float, "AV1 frame {i} not RGBA16F");
            assert_eq!((t.width(), t.height()), (W as u32, H as u32));
            let rgb = rgba16f_to_rgb(&device.read_rgba_texture(t));
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for px in &rgb {
                let l = px[0] + px[1] + px[2];
                lo = lo.min(l);
                hi = hi.max(l);
            }
            assert!(hi - lo > 0.25, "AV1 frame {i} nearly uniform; decode failed");
            readbacks.push(rgb);
        }
        for p in 1..readbacks.len() {
            assert_ne!(readbacks[p], readbacks[0], "AV1 texture {p} identical to the keyframe");
        }
        eprintln!("m573 AV1 10-bit: {} GPU Rgba16Float textures, real content", readbacks.len());
    }
}
