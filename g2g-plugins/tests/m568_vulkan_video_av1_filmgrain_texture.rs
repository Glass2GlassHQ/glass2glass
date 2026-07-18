//! M568: AV1 film grain on the GPU-texture decode path, on real hardware.
//!
//! The zero-copy `decode_all_to_textures` path (M494's AV1 sibling) produces the
//! grain-free hardware reconstruction: the RTX 3060 exposes only
//! `DPB_AND_OUTPUT_COINCIDE`, so the driver cannot apply grain, and the GPU ycbcr
//! compute pass has no grain stage. For a grain stream the reorder-aware texture
//! path now reads each displayed slot back to NV12 (`TRANSFER_SRC`), synthesizes
//! grain on the CPU bit-for-bit with dav1d (`apply_film_grain_nv12`, the same path
//! M566 proved on the system output), and uploads the result to the texture. Grain
//! is output-only, so the read-back leaves the DPB reference untouched.
//!
//! Structural assertions run always. Bit-exactness is checked when `G2G_AV1_REF`
//! points at a raw `yuv420p` dump of the same clip (`ffmpeg -i clip -f rawvideo
//! -pix_fmt yuv420p ref.yuv`, grain applied): each RGBA texture is compared against
//! the BT.601-limited conversion of the reference frame, which equals the texture
//! path's own `nv12_to_rgba` of the (bit-exact) grained NV12, so the SAD is 0.
//!
//! Runs on the RTX 3060; skips with no adapter / no AV1 decode / no compute queue.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_av1_sequence_header, open_av1_decode_device, to_std_av1_seq_header, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/av1_640x480_filmgrain.obu");
const W: usize = 640;
const H: usize = 480;

/// BT.601 studio-range NV12 -> RGBA8, matching the decoder's `nv12_to_rgba`
/// conversion exactly (BT.601 luma weights Kr=0.299 / Kb=0.114, the exact studio
/// range ratios 255/219 and 255/224, clamp and `as u8` truncation), with
/// nearest-neighbour chroma at (x/2, y/2). The reference is planar I420, so its
/// U/V planes are the grained NV12 chroma de-interleaved. (The fixture is untagged
/// SD, so it resolves to BT.601 studio.)
fn ref_pixel_rgb(y: u8, u: u8, v: u8) -> [u8; 3] {
    let (kr, kb) = (0.299f32, 0.114f32);
    let kg = 1.0 - kr - kb;
    let c_scale = 255.0 / 224.0;
    let cb = u as f32 - 128.0;
    let cr = v as f32 - 128.0;
    let yc = (y as f32 - 16.0) * (255.0 / 219.0);
    let r = yc + 2.0 * (1.0 - kr) * c_scale * cr;
    let g =
        yc - 2.0 * kr * (1.0 - kr) / kg * c_scale * cr - 2.0 * kb * (1.0 - kb) / kg * c_scale * cb;
    let b = yc + 2.0 * (1.0 - kb) * c_scale * cb;
    [
        r.clamp(0.0, 255.0) as u8,
        g.clamp(0.0, 255.0) as u8,
        b.clamp(0.0, 255.0) as u8,
    ]
}

#[test]
fn decodes_filmgrain_av1_to_textures() {
    let seq = extract_av1_sequence_header(CLIP).expect("parse sequence header");
    assert!(
        seq.film_grain_params_present,
        "fixture carries no film grain; feature untested"
    );

    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m568: no Vulkan AV1 decode adapter");
            return;
        }
        Err(e) => panic!("open AV1 decode device: {e:?}"),
    };

    let std = to_std_av1_seq_header(&seq);
    let session = device
        .create_av1_session(&std, W as u32, H as u32)
        .expect("create AV1 session");
    let mut dec = match device.create_av1_dpb_decoder_gpu(&session, &seq) {
        Ok(d) => d,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("skip m568: no distinct compute queue for the GPU-texture path");
            return;
        }
        Err(e) => panic!("build GPU AV1 decoder: {e:?}"),
    };

    let textures = dec
        .decode_all_to_textures(CLIP)
        .expect("decode film-grain stream to textures");
    assert!(!textures.is_empty(), "no textures decoded");

    let mut readbacks = Vec::new();
    for (i, tex) in textures.iter().enumerate() {
        assert_eq!(tex.width(), W as u32);
        assert_eq!(tex.height(), H as u32);
        assert_eq!(tex.format(), wgpu::TextureFormat::Rgba8Unorm);
        let rgba = device.read_rgba_texture(tex);
        assert_eq!(rgba.len(), W * H * 4);
        // Real content, not a flat / failed convert.
        let min = *rgba.iter().min().unwrap();
        let max = *rgba.iter().max().unwrap();
        assert!(
            min <= 20 && max >= 200,
            "texture {i} RGBA range {min}..={max} not a real picture"
        );
        readbacks.push(rgba);
    }
    // Animated content + per-frame grain: consecutive textures must differ.
    for i in 1..readbacks.len() {
        assert_ne!(
            readbacks[i],
            readbacks[i - 1],
            "texture {i} identical to {}",
            i - 1
        );
    }

    // Optional bit-exact check vs an ffmpeg/dav1d yuv420p reference (grain applied).
    if let Ok(path) = std::env::var("G2G_AV1_REF") {
        let ref_yuv = std::fs::read(&path).expect("read G2G_AV1_REF");
        let cw = W / 2;
        let ch = H / 2;
        let frame_bytes = W * H + 2 * cw * ch;
        assert!(
            ref_yuv.len() >= frame_bytes * readbacks.len(),
            "reference too short"
        );
        for (i, rgba) in readbacks.iter().enumerate() {
            let base = i * frame_bytes;
            let ref_y = &ref_yuv[base..base + W * H];
            let ref_u = &ref_yuv[base + W * H..base + W * H + cw * ch];
            let ref_v = &ref_yuv[base + W * H + cw * ch..base + frame_bytes];
            let mut sad = 0u64;
            for y in 0..H {
                for x in 0..W {
                    let ci = (y / 2) * cw + (x / 2);
                    let want = ref_pixel_rgb(ref_y[y * W + x], ref_u[ci], ref_v[ci]);
                    let o = (y * W + x) * 4;
                    for c in 0..3 {
                        sad += (rgba[o + c] as i32 - want[c] as i32).unsigned_abs() as u64;
                    }
                }
            }
            eprintln!(
                "texture {i}: RGB SAD/px={:.6}",
                sad as f64 / (W * H * 3) as f64
            );
            assert_eq!(sad, 0, "texture {i} not bit-exact vs grained reference");
        }
        eprintln!(
            "m568: {} grained AV1 textures bit-exact vs reference",
            readbacks.len()
        );
    }
}
