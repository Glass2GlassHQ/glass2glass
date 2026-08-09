#![cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
//! M990: a `MemoryDomain::DmaBuf` NV12 frame binds into `WgpuPreprocess`'s compute
//! pass through a Vulkan external-memory import, with no CPU round trip.
//!
//! `WgpuToDmaBuf` stands in for the capture / decode producer, so the import is
//! exercised on a machine with no VAAPI decoder, and it is the only producer whose
//! dma-buf a *discrete* GPU will bind (a CPU-backed one needs an integrated GPU:
//! `m993_camera_dmabuf_preprocess` is the live-camera case). The tensor must match
//! the host BT.601 reference and, bit for bit, the same pixels through the System
//! upload path. The third case gives the dma-buf a padded row stride (what a v4l2
//! capture produces) and asserts the shader reads it in place, with no repack.
//!
//! Needs a GPU with Vulkan dma-buf export + import; CI-excluded.
//!   cargo test -p g2g-ml --features dmabuf-wgpu --test m990_dmabuf_preprocess -- --nocapture

use g2g_ml::wgpupreprocess::{gpu_available, nv12_to_rgb_tensor, WgpuBufferOwner};

include!("util/dmabuf_preprocess.rs");

/// Frame geometry the tensor is built at.
const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
/// Extra bytes per row for the padded-stride case.
const ROW_PADDING: u32 = 6;

fn nv12_caps(w: u32, h: u32) -> Caps {
    raw_caps(RawVideoFormat::Nv12, w, h)
}

/// An NV12 frame laid out with row stride `stride`: `height` luma rows, then
/// `height / 2` interleaved Cb,Cr rows, each `stride` bytes with only the first
/// `width` meaningful. Padding is 0xFF, a value that shifts the output if the
/// shader ever reads it. Chroma varies per block, so a swapped Cb/Cr or a wrong
/// row would change the tensor.
fn padded_nv12(stride: u32, width: u32, height: u32) -> Vec<u8> {
    let (stride, width, height) = (stride as usize, width as usize, height as usize);
    let mut out = vec![0xFFu8; stride * (height + height / 2)];
    for y in 0..height {
        for x in 0..width {
            out[y * stride + x] = (16 + (x * 13 + y * 29) % 220) as u8;
        }
    }
    let chroma = height * stride;
    for cy in 0..height / 2 {
        for cx in (0..width).step_by(2) {
            out[chroma + cy * stride + cx] = ((cx * 7 + cy * 11) % 256) as u8;
            out[chroma + cy * stride + cx + 1] = ((cx * 23 + cy * 5) % 256) as u8;
        }
    }
    out
}

/// The same pixels repacked at `stride == width`, the layout the host reference
/// and the System upload path take.
fn tight_nv12(padded: &[u8], stride: u32, width: u32, height: u32) -> Vec<u8> {
    let rows = (height + height / 2) as usize;
    repack_tight(padded, stride as usize, width as usize, rows)
}

/// The tensor must match the host BT.601 reference (float tolerance) and be
/// non-grayscale, proving the chroma rows were read.
fn assert_matches_reference(got: &[f32], nv12: &[u8]) {
    let area = (WIDTH * HEIGHT) as usize;
    let expect = nv12_to_rgb_tensor(nv12, WIDTH as usize, HEIGHT as usize);
    assert_eq!(got.len(), 3 * area, "NCHW tensor length");
    for (i, (g, e)) in got.iter().zip(&expect).enumerate() {
        assert!(
            (g - e).abs() < 1e-3,
            "tensor[{i}] = {g}, reference {e}: imported pixels differ"
        );
    }
    assert!(
        (0..area).any(|px| (got[px] - got[area + px]).abs() > 1e-3),
        "coloured chroma must break R == G, else the chroma rows were not read"
    );
}

#[tokio::test]
async fn dmabuf_import_matches_system_upload() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let caps = nv12_caps(WIDTH, HEIGHT);
    let nv12 = padded_nv12(WIDTH, WIDTH, HEIGHT);
    let Some(frame) = export_dmabuf(&nv12, &caps).await else {
        eprintln!("skipping: no GPU dma-buf export on this host");
        return;
    };
    assert_eq!(frame.timing.pts_ns, 7_000);
    let Some(tensor) = preprocess_dmabuf(frame, &caps, ImportAdapter::default(), false).await
    else {
        return;
    };
    assert_eq!(tensor.timing.pts_ns, 7_000, "tensor inherits source timing");

    let got = frame_f32(&tensor);
    assert_matches_reference(&got, &nv12);
    assert_eq!(
        got,
        preprocess_system(&caps, nv12).await,
        "imported dma-buf and uploaded system memory give the same tensor, bit for bit"
    );
    eprintln!(
        "PASS: dma-buf import == system upload, {} values",
        got.len()
    );
}

#[tokio::test]
async fn dmabuf_import_with_gpu_output_stays_resident() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let caps = nv12_caps(WIDTH, HEIGHT);
    let nv12 = padded_nv12(WIDTH, WIDTH, HEIGHT);
    let Some(frame) = export_dmabuf(&nv12, &caps).await else {
        eprintln!("skipping: no GPU dma-buf export on this host");
        return;
    };
    let Some(tensor) = preprocess_dmabuf(frame, &caps, ImportAdapter::default(), true).await else {
        return;
    };

    let MemoryDomain::WgpuBuffer(owned) = &tensor.domain else {
        panic!("gpu-output mode keeps the tensor in a GPU buffer");
    };
    let owner = owned
        .keep_alive()
        .as_any()
        .downcast_ref::<WgpuBufferOwner>()
        .expect("tensor buffer owner");
    let got: Vec<f32> = owner
        .read_back()
        .expect("read back the GPU tensor")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_matches_reference(&got, &nv12);
    eprintln!("PASS: dma-buf in, GPU-resident tensor out, never touched the CPU");
}

#[tokio::test]
async fn padded_stride_dmabuf_needs_no_repack() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let stride = WIDTH + ROW_PADDING;
    let padded = padded_nv12(stride, WIDTH, HEIGHT);
    // The export derives the stride from the caps width, so a wider caps pads
    // every row; the preprocess still runs at the real WIDTH x HEIGHT.
    let Some(frame) = export_dmabuf(&padded, &nv12_caps(stride, HEIGHT)).await else {
        eprintln!("skipping: no GPU dma-buf export on this host");
        return;
    };
    let caps = nv12_caps(WIDTH, HEIGHT);
    let Some(tensor) = preprocess_dmabuf(frame, &caps, ImportAdapter::default(), false).await
    else {
        return;
    };

    let got = frame_f32(&tensor);
    let tight = tight_nv12(&padded, stride, WIDTH, HEIGHT);
    assert_matches_reference(&got, &tight);
    assert_eq!(
        got,
        preprocess_system(&caps, tight).await,
        "a padded dma-buf gives the same tensor as the repacked pixels"
    );
    eprintln!("PASS: stride {stride} for width {WIDTH} read in place, no repack");
}
