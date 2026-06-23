#![cfg(all(target_os = "linux", feature = "cuda-wgpu"))]
//! M220 Stage 1 spike: prove the Vulkan-side external-memory plumbing works on
//! this NVIDIA driver before the CUDA import and the full `CudaToWgpu` element
//! land. Creates a wgpu device with `VK_KHR_external_memory_fd`, allocates an
//! exportable NV12 image, and exports its memory as an opaque FD.
//!
//! ```sh
//! cargo test -p g2g-plugins --features cuda-wgpu --test cudawgpu_spike -- --nocapture
//! ```
//!
//! Skips when no Vulkan adapter is present.

use g2g_plugins::cudawgpu::{
    create_interop_device, cuda_fill_xor_pattern, cuda_roundtrip_check, export_nv12_image,
    wrap_as_texture,
};

/// Allocate a Vulkan external-memory image, export its FD, import it into CUDA,
/// and confirm an NV12 pattern written and read back through the CUDA array
/// round-trips. This is the shared-memory core: Vulkan owns the allocation, CUDA
/// reaches the same bytes, with no wgpu (and no PCIe) involved.
#[tokio::test]
async fn cuda_shares_vulkan_external_memory() {
    let dev = match create_interop_device().await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no Vulkan interop device ({e:?})");
            return;
        }
    };

    // SAFETY: `dev.device` is the interop device just created, with the FD
    // external-memory extension enabled.
    let shared = unsafe { export_nv12_image(&dev.device, 320, 240) }
        .expect("allocate + export an NV12 external-memory image");

    eprintln!(
        "exported NV12 image {}x{} (tex rows {}), {} bytes, fd {}",
        shared.width,
        shared.height,
        shared.height + shared.height / 2,
        shared.size,
        shared.fd,
    );
    assert!(shared.fd >= 0, "a valid opaque FD");
    assert!(shared.size as usize >= (320 * 360) as usize, "image spans at least the packed NV12");

    // SAFETY: `shared` came from the exporter above and has not been imported.
    // The call consumes the FD (CUDA owns an imported FD), so we must not close it.
    let matched = unsafe { cuda_roundtrip_check(&shared) }
        .expect("CUDA import + array map + 2D copy round-trip");
    assert!(matched, "the NV12 pattern read back through the CUDA array must match");
    eprintln!("CUDA round-trip through the shared image matched");

    // SAFETY: same device; the image is idle (the CUDA work was synchronized).
    unsafe { shared.destroy(&dev.device) };
}

/// The end-to-end half of the bridge transport: CUDA writes a pattern into the
/// shared image, it is wrapped as a `wgpu::Texture`, and wgpu copies it back
/// out. If wgpu reads what CUDA wrote, the Vulkan<->CUDA layout / visibility
/// works and the M217 surface-import path can sample these textures directly.
#[tokio::test]
async fn wgpu_reads_what_cuda_wrote() {
    let dev = match create_interop_device().await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("skipping: no Vulkan interop device ({e:?})");
            return;
        }
    };
    let (w, h) = (320u32, 240u32);
    let tex_h = h + h / 2;

    // SAFETY: `dev.device` is the interop device.
    let shared = unsafe { export_nv12_image(&dev.device, w, h) }.expect("export image");
    // SAFETY: freshly exported, not yet imported.
    unsafe { cuda_fill_xor_pattern(&shared) }.expect("CUDA fill the shared image");
    // SAFETY: same device; consumes `shared` (freed when the texture drops).
    let texture = unsafe { wrap_as_texture(&dev.device, shared) };

    // Copy the texture to a buffer and read it back. bytes_per_row must be
    // 256-aligned, so pad 320 -> 512.
    let padded_bpr = w.div_ceil(256) * 256;
    let buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded_bpr * tex_h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = dev.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(tex_h),
            },
        },
        wgpu::Extent3d { width: w, height: tex_h, depth_or_array_layers: 1 },
    );
    dev.queue.submit([enc.finish()]);

    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    dev.device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .expect("poll");
    let data = slice.get_mapped_range();

    let mut mismatches = 0u64;
    for y in 0..tex_h as usize {
        for x in 0..w as usize {
            let got = data[y * padded_bpr as usize + x];
            if got != (x ^ y) as u8 {
                mismatches += 1;
            }
        }
    }
    eprintln!("wgpu read back {}x{} texels, {mismatches} mismatches", w, tex_h);
    assert_eq!(mismatches, 0, "wgpu must read exactly the (x^y) pattern CUDA wrote");
}
