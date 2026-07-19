//! Visual smoke test for the Vulkan Video H.264 decoder (M486-M492).
//!
//! Decodes an H.264 Annex-B elementary stream on the GPU via `VK_KHR_video_*`
//! (vendor-neutral, the `NvDec` analog that also runs on AMD/Intel) and writes
//! every decoded frame to a `.ppm` you can open, so you can SEE the decoder
//! produce real pixels, not just pass an assertion.
//!
//! Run (needs a Vulkan H.264 decode GPU, e.g. the RTX 3060):
//!
//! ```sh
//! cargo run --release -p g2g-plugins --features vulkan-video \
//!     --example vulkan_video_smoke                 # decodes the bundled 640x480 clip
//! cargo run --release -p g2g-plugins --features vulkan-video \
//!     --example vulkan_video_smoke -- my.h264 /tmp/frames   # a file of your own
//! ```
//!
//! Output: `<outdir>/frame_000.ppm` .. one per coded picture, plus a per-frame
//! luma-range/mean line so a wrong (e.g. flat) decode is obvious at a glance.
//! View with any image viewer, or `ffplay -loop 1 frame_000.ppm`.

use std::path::PathBuf;

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, Nv12Frame, VulkanVideoError,
};

/// The 640x480 baseline clip the tests use, embedded so the demo runs with no
/// arguments (two GOPs of IDR + four P frames, i.e. it exercises the DPB).
const BUNDLED_CLIP: &[u8] = include_bytes!("../tests/fixtures/h264_640x480.h264");

fn main() {
    let mut args = std::env::args().skip(1);
    let clip_path = args.next();
    let out_dir = PathBuf::from(
        args.next()
            .unwrap_or_else(|| "vulkan_video_frames".to_string()),
    );

    let clip: Vec<u8> = match &clip_path {
        Some(p) => std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}")),
        None => BUNDLED_CLIP.to_vec(),
    };
    println!(
        "decoding {} ({} bytes)",
        clip_path.as_deref().unwrap_or("<bundled 640x480 clip>"),
        clip.len()
    );

    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("no Vulkan adapter; this demo needs a GPU with Vulkan H.264 decode.");
            return;
        }
        Err(VulkanVideoError::ExtensionUnsupported) | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("this GPU has no Vulkan H.264 decode support.");
            return;
        }
        Err(e) => panic!("failed to open decode device: {e:?}"),
    };

    let ps = extract_h264_parameter_sets(&clip).expect("parse SPS+PPS from the stream");
    // Size the session to the stream's coded dimensions (macroblock geometry).
    let width = (ps.sps.pic_width_in_mbs_minus1 + 1) * 16;
    let height = (ps.sps.pic_height_in_map_units_minus1 + 1) * 16;
    let session = device
        .create_h264_session(&ps, width, height)
        .expect("create decode session");
    let mut decoder = device
        .create_h264_dpb_decoder(&session, &ps)
        .expect("create DPB decoder");

    let frames = decoder.decode_all(&clip).expect("decode the stream");
    println!("decoded {} frames at {width}x{height}", frames.len());

    std::fs::create_dir_all(&out_dir).expect("create output dir");
    for (i, f) in frames.iter().enumerate() {
        let path = out_dir.join(format!("frame_{i:03}.ppm"));
        write_ppm(&path, f);
        let min = *f.luma.iter().min().unwrap();
        let max = *f.luma.iter().max().unwrap();
        let mean = f.luma.iter().map(|&b| b as u64).sum::<u64>() / f.luma.len() as u64;
        println!(
            "  frame {i:>3}: luma {min:>3}..={max:<3} mean {mean:>3}  -> {}",
            path.display()
        );
    }
    println!("wrote {} frames to {}/", frames.len(), out_dir.display());
}

/// Convert an [`Nv12Frame`] to RGB (BT.601 limited range) and write a binary PPM.
fn write_ppm(path: &std::path::Path, frame: &Nv12Frame) {
    let (w, h) = (frame.width as usize, frame.height as usize);
    let cw = w / 2;
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let yv = frame.luma[y * w + x] as f32;
            let ci = ((y / 2) * cw + (x / 2)) * 2;
            let cb = frame.chroma[ci] as f32 - 128.0;
            let cr = frame.chroma[ci + 1] as f32 - 128.0;
            let yc = (yv - 16.0) * 1.164_383;
            let r = yc + 1.596_027 * cr;
            let g = yc - 0.391_762 * cb - 0.812_968 * cr;
            let b = yc + 2.017_232 * cb;
            rgb.push(r.clamp(0.0, 255.0) as u8);
            rgb.push(g.clamp(0.0, 255.0) as u8);
            rgb.push(b.clamp(0.0, 255.0) as u8);
        }
    }
    let mut data = format!("P6\n{w} {h}\n255\n").into_bytes();
    data.extend_from_slice(&rgb);
    std::fs::write(path, data).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
}
