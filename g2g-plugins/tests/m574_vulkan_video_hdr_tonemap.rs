//! M574: PQ / HLG HDR -> SDR tone-mapping in the GPU-texture converter, on real
//! hardware.
//!
//! The fixed-function `VkSamplerYcbcrConversion` applies the matrix + range but
//! NOT the transfer function, so an HDR (PQ / HLG, BT.2020) stream came out of
//! the M573 10-bit converter as its raw transfer-encoded R'G'B' (too dark / wrong
//! colour on an SDR consumer). The `create_*_dpb_decoder_gpu_tonemap`
//! constructors enable a transfer stage in the `rgba16f` compute pass: EOTF (PQ
//! ST 2084 or HLG B67) -> BT.2390 EETF display mapping (maxRGB, 1000 -> 100 nits)
//! -> BT.2020 -> BT.709 gamut -> BT.709 OETF, producing display-ready SDR.
//!
//! Verified two ways. (1) Unit tests pin the transfer math to spec anchor values
//! (PQ / HLG EOTF endpoints, round-trips) with no GPU. (2) On the RTX 3060, a PQ
//! and an HLG clip decode through the tone-mapping converter; each GPU-resident
//! `Rgba16Float` texture matches a CPU reference that runs the identical pipeline
//! on the bit-exact system 10-bit decode, and differs from the passthrough
//! (non-tone-mapped) output (so the transfer stage demonstrably ran).
//!
//! Fixtures are 640x480 x265 Main 10 clips tagged BT.2020 + PQ (`smpte2084`) and
//! BT.2020 + HLG (`arib-std-b67`). Both codecs' math is validated; the fixtures
//! are HEVC (the transfer stage is codec-independent, keyed only off the parsed
//! `transfer_characteristics`). Runs on the RTX 3060; skips with no adapter / no
//! compute queue / no 10-bit HEVC decode.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]
// The transfer constants are written to the full spec precision so they read
// identically to the GLSL shader literals (both round to the same f32).
#![allow(clippy::excessive_precision)]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h265_parameter_sets, open_h265_decode_device, to_std_h265_params, ColorMatrix,
    TransferFunction, VideoColorSpace, VulkanVideoDevice, VulkanVideoError,
};

const PQ: &[u8] = include_bytes!("fixtures/h265_640x480_pq.hevc");
const HLG: &[u8] = include_bytes!("fixtures/h265_640x480_hlg.hevc");
const W: usize = 640;
const H: usize = 480;

// ---- Transfer math: an exact Rust port of mediacodec_ycbcr16.comp. The GPU
// runs the GLSL; these run the same formulas so the CPU reference and the unit
// anchors validate one shared definition. ----

const SRC_PEAK: f32 = 1000.0;
const DST_PEAK: f32 = 100.0;
const PQ_M1: f32 = 0.1593017578125;
const PQ_M2: f32 = 78.84375;
const PQ_C1: f32 = 0.8359375;
const PQ_C2: f32 = 18.8515625;
const PQ_C3: f32 = 18.6875;

fn pq_eotf(e: f32) -> f32 {
    let ep = e.max(0.0).powf(1.0 / PQ_M2);
    let num = (ep - PQ_C1).max(0.0);
    let den = (PQ_C2 - PQ_C3 * ep).max(1e-6);
    (num / den).powf(1.0 / PQ_M1)
}

fn pq_oetf(y: f32) -> f32 {
    let ym = y.max(0.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * ym) / (1.0 + PQ_C3 * ym)).powf(PQ_M2)
}

const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.55991073;

fn hlg_inv_oetf(e: f32) -> f32 {
    if e <= 0.5 {
        (e * e) / 3.0
    } else {
        (((e - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

fn bt2390_eetf(e: f32, pq_w: f32, pq_max: f32) -> f32 {
    let e1 = e / pq_w;
    let max_lum = pq_max / pq_w;
    let ks = 1.5 * max_lum - 0.5;
    if e1 > ks && ks < 1.0 {
        let t = (e1 - ks) / (1.0 - ks);
        let (t2, t3) = (t * t, t * t * t);
        let e2 = (2.0 * t3 - 3.0 * t2 + 1.0) * ks
            + (t3 - 2.0 * t2 + t) * (1.0 - ks)
            + (-2.0 * t3 + 3.0 * t2) * max_lum;
        e2 * pq_w
    } else {
        e1 * pq_w
    }
}

fn bt2020_to_709(c: [f32; 3]) -> [f32; 3] {
    [
        1.660491 * c[0] - 0.587641 * c[1] - 0.072850 * c[2],
        -0.124550 * c[0] + 1.132900 * c[1] - 0.008349 * c[2],
        -0.018151 * c[0] - 0.100579 * c[1] + 1.118730 * c[2],
    ]
}

fn bt709_oetf(l: f32) -> f32 {
    let l = l.clamp(0.0, 1.0);
    if l < 0.018 {
        4.5 * l
    } else {
        1.099 * l.powf(0.45) - 0.099
    }
}

/// The HDR->SDR pipeline `tonemap_hdr` from the shader, `xfer` = 1 (PQ) / 2 (HLG).
fn tonemap_hdr(mut rgb: [f32; 3], xfer: u32) -> [f32; 3] {
    if xfer == 2 {
        let s = [
            hlg_inv_oetf(rgb[0]),
            hlg_inv_oetf(rgb[1]),
            hlg_inv_oetf(rgb[2]),
        ];
        let yscene = 0.2627 * s[0] + 0.6780 * s[1] + 0.0593 * s[2];
        let g = yscene.max(1e-6).powf(0.2);
        rgb = [
            pq_oetf(s[0] * g * SRC_PEAK / 10000.0),
            pq_oetf(s[1] * g * SRC_PEAK / 10000.0),
            pq_oetf(s[2] * g * SRC_PEAK / 10000.0),
        ];
    }
    let pq_w = pq_oetf(SRC_PEAK / 10000.0);
    let pq_max = pq_oetf(DST_PEAK / 10000.0);
    let m = rgb[0].max(rgb[1]).max(rgb[2]);
    if m > 1e-6 {
        let mt = bt2390_eetf(m, pq_w, pq_max);
        let s = mt / m;
        rgb = [rgb[0] * s, rgb[1] * s, rgb[2] * s];
    }
    let norm = pq_eotf(pq_max).max(1e-6);
    let lin = bt2020_to_709([
        pq_eotf(rgb[0]) / norm,
        pq_eotf(rgb[1]) / norm,
        pq_eotf(rgb[2]) / norm,
    ]);
    [bt709_oetf(lin[0]), bt709_oetf(lin[1]), bt709_oetf(lin[2])]
}

// ---- Shared YUV / readback helpers (as in m573) ----

fn le16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[2 * i], b[2 * i + 1]])
}

fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = match exp {
        0 if mant == 0 => (sign as u32) << 31,
        0 => {
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
        _ => {
            ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | ((mant as u32) << 13)
        }
    };
    f32::from_bits(bits)
}

fn rgba16f_to_rgb(bytes: &[u8]) -> Vec<[f32; 3]> {
    assert_eq!(bytes.len(), W * H * 8);
    (0..W * H)
        .map(|p| {
            let o = p * 4;
            [
                half_to_f32(le16(bytes, o)),
                half_to_f32(le16(bytes, o + 1)),
                half_to_f32(le16(bytes, o + 2)),
            ]
        })
        .collect()
}

/// 10-bit `G10X6` NV12 -> the ycbcr conversion result (nonlinear R'G'B' in [0,1]),
/// matching the fixed-function narrow-range BT.2020 conversion (nearest chroma).
fn nv12_10bit_to_rgb(luma: &[u8], chroma: &[u8], color: VideoColorSpace) -> Vec<[f32; 3]> {
    let cw = W / 2;
    let (kr, kb) = match color.matrix {
        ColorMatrix::Bt601 => (0.299f32, 0.114f32),
        ColorMatrix::Bt709 => (0.2126, 0.0722),
        ColorMatrix::Bt2020Ncl => (0.2627, 0.0593),
    };
    let kg = 1.0 - kr - kb;
    let (y_off, y_span, c_off, c_span) = if color.full_range {
        (0.0, 1023.0, 512.0, 1023.0)
    } else {
        (64.0, 876.0, 512.0, 896.0)
    };
    let (cr_r, cb_b) = (2.0 * (1.0 - kr), 2.0 * (1.0 - kb));
    let (cr_g, cb_g) = (2.0 * kr * (1.0 - kr) / kg, 2.0 * kb * (1.0 - kb) / kg);
    let mut out = Vec::with_capacity(W * H);
    for y in 0..H {
        for x in 0..W {
            let yy = ((le16(luma, y * W + x) >> 6) as f32 - y_off) / y_span;
            let ci = (y / 2) * cw + (x / 2);
            let cb = ((le16(chroma, 2 * ci) >> 6) as f32 - c_off) / c_span;
            let cr = ((le16(chroma, 2 * ci + 1) >> 6) as f32 - c_off) / c_span;
            out.push([
                (yy + cr_r * cr).clamp(0.0, 1.0),
                (yy - cr_g * cr - cb_g * cb).clamp(0.0, 1.0),
                (yy + cb_b * cb).clamp(0.0, 1.0),
            ]);
        }
    }
    out
}

fn open_or_skip(dev: Result<VulkanVideoDevice, VulkanVideoError>) -> Option<VulkanVideoDevice> {
    match dev {
        Ok(d) => Some(d),
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m574: no Vulkan H.265 decode adapter");
            None
        }
        Err(e) => panic!("open H.265 decode device: {e:?}"),
    }
}

// ---- Unit tests: the transfer math against spec anchors (no GPU) ----

fn approx(a: f32, b: f32, tol: f32, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
}

#[test]
fn pq_transfer_anchors() {
    // PQ EOTF endpoints (normalized, 1.0 == 10000 nits).
    approx(pq_eotf(0.0), 0.0, 1e-5, "PQ(0)");
    approx(pq_eotf(1.0), 1.0, 1e-3, "PQ(1) = 10000 nits");
    // Code 0.5 ~= 92.2 nits (ST 2084 table) -> normalized ~0.00922.
    approx(pq_eotf(0.5), 0.00922, 5e-4, "PQ(0.5) ~ 92 nits");
    // OETF is the inverse: round-trips across the range.
    for &y in &[0.001f32, 0.01, 0.1, 0.5, 1.0] {
        approx(pq_eotf(pq_oetf(y)), y, 1e-3, "PQ round-trip");
    }
    // 100 nits (SDR peak) -> PQ code ~0.5081.
    approx(
        pq_oetf(100.0 / 10000.0),
        0.5081,
        2e-3,
        "PQ code for 100 nits",
    );
}

#[test]
fn hlg_transfer_anchors() {
    // HLG inverse OETF: 0 -> 0, 0.5 -> 1/12, 1.0 -> ~1.0 (scene peak).
    approx(hlg_inv_oetf(0.0), 0.0, 1e-6, "HLG(0)");
    approx(hlg_inv_oetf(0.5), 1.0 / 12.0, 1e-6, "HLG(0.5) = 1/12");
    approx(hlg_inv_oetf(1.0), 1.0, 3e-3, "HLG(1) ~ scene peak");
}

#[test]
fn eetf_endpoints_and_monotonic() {
    let pq_w = pq_oetf(SRC_PEAK / 10000.0);
    let pq_max = pq_oetf(DST_PEAK / 10000.0);
    // Black maps to black; the source peak maps to the target peak.
    approx(bt2390_eetf(0.0, pq_w, pq_max), 0.0, 1e-4, "EETF(0)");
    approx(
        bt2390_eetf(pq_w, pq_w, pq_max),
        pq_max,
        2e-3,
        "EETF(peak) = target peak",
    );
    // Monotonic non-decreasing across the source range.
    let mut prev = -1.0;
    for i in 0..=64 {
        let v = bt2390_eetf(pq_w * i as f32 / 64.0, pq_w, pq_max);
        assert!(v + 1e-4 >= prev, "EETF must be monotonic ({v} < {prev})");
        prev = v;
    }
    // A highlight above the target peak is rolled off below the source peak's raw
    // value (compression actually happened).
    assert!(
        bt2390_eetf(pq_w, pq_w, pq_max) < pq_w,
        "EETF must compress the peak"
    );
}

// ---- GPU integration: tone-mapped output matches the CPU pipeline ----

fn run_codec(stream: &[u8], xfer: u32, label: &str) {
    let ps = extract_h265_parameter_sets(stream).expect("vps/sps/pps");
    assert_eq!(
        ps.sps.bit_depth_luma_minus8, 2,
        "{label} fixture not 10-bit"
    );
    let color = VideoColorSpace::from_cicp(
        ps.sps.matrix_coefficients,
        ps.sps.transfer_characteristics,
        ps.sps.video_full_range_flag,
        H as u32,
    );
    let expect_xfer = matches!(color.transfer, TransferFunction::Pq | TransferFunction::Hlg);
    assert!(
        expect_xfer,
        "{label} fixture is not HDR-tagged (transfer {:?})",
        color.transfer
    );

    let Some(device) = open_or_skip(block_on(open_h265_decode_device())) else {
        return;
    };
    let std = to_std_h265_params(&ps);

    // CPU reference: system 10-bit decode -> ycbcr R'G'B' -> the shader pipeline.
    let session = device
        .create_h265_session(&std, W as u32, H as u32)
        .expect("session");
    let mut cpu = device
        .create_h265_dpb_decoder(&session, &ps)
        .expect("cpu decoder");
    let cpu_ref: Vec<Vec<[f32; 3]>> = cpu
        .decode_all(stream)
        .expect("cpu decode")
        .iter()
        .map(|f| {
            nv12_10bit_to_rgb(&f.luma, &f.chroma, color)
                .iter()
                .map(|&c| tonemap_hdr(c, xfer))
                .collect()
        })
        .collect();
    assert!(!cpu_ref.is_empty());

    // Passthrough textures (no transfer stage) to prove the tone-map changed pixels.
    let pass_session = device
        .create_h265_session(&std, W as u32, H as u32)
        .expect("session");
    let mut pass = match device.create_h265_dpb_decoder_gpu(&pass_session, &ps) {
        Ok(d) => d,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skip m574 {label}: no distinct compute queue");
            return;
        }
        Err(VulkanVideoError::UnsupportedStream) => {
            eprintln!("skip m574 {label}: no 10-bit HEVC decode");
            return;
        }
        Err(e) => panic!("passthrough decoder: {e:?}"),
    };
    let pass_rb: Vec<Vec<[f32; 3]>> = pass
        .decode_all_to_textures(stream)
        .expect("passthrough textures")
        .iter()
        .map(|t| rgba16f_to_rgb(&device.read_rgba_texture(t)))
        .collect();

    // Tone-mapped textures.
    let tm_session = device
        .create_h265_session(&std, W as u32, H as u32)
        .expect("session");
    let mut tm = device
        .create_h265_dpb_decoder_gpu_tonemap(&tm_session, &ps)
        .expect("tonemap decoder");
    let texes = tm.decode_all_to_textures(stream).expect("tonemap textures");
    assert_eq!(texes.len(), cpu_ref.len());

    let mut worst_mean = 0.0f64;
    for (i, t) in texes.iter().enumerate() {
        assert_eq!(t.format(), wgpu::TextureFormat::Rgba16Float);
        assert_eq!((t.width(), t.height()), (W as u32, H as u32));
        let gpu = rgba16f_to_rgb(&device.read_rgba_texture(t));
        // Matches the CPU pipeline (tolerance: linear-vs-nearest chroma + f16,
        // amplified through the tone curve).
        let (mut sum, mut n, mut worst) = (0.0f64, 0u64, 0.0f32);
        for (g, c) in gpu.iter().zip(cpu_ref[i].iter()) {
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
            mean < 0.03,
            "{label} frame {i}: GPU tone-map mean abs diff {mean:.4} (worst {worst:.3})"
        );
        // The tone-map demonstrably changed the image vs passthrough.
        let changed = gpu
            .iter()
            .zip(pass_rb[i].iter())
            .any(|(g, p)| (g[0] - p[0]).abs() > 0.02);
        assert!(
            changed,
            "{label} frame {i}: tone-map output equals passthrough (transfer stage did not run)"
        );
    }
    eprintln!(
        "m574 {label}: {} tone-mapped textures match CPU pipeline (worst mean {worst_mean:.4})",
        texes.len()
    );
}

#[test]
fn hdr_tonemap_pq_and_hlg() {
    run_codec(PQ, 1, "PQ");
    run_codec(HLG, 2, "HLG");
}
