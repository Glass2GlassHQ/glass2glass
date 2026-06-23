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

use g2g_plugins::cudawgpu::{create_interop_device, cuda_roundtrip_check, export_nv12_image};

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
