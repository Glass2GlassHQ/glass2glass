//! Feasibility probe for zero-copy libcamera -> GPU: capture one frame and
//! inspect the buffer's plane file descriptor. Tells us whether libcamera hands
//! out a real dma-buf for this camera (and what backs it), which decides whether
//! a zero-copy Vulkan import into the GPU is even possible.
//!
//! `cargo test -p g2g-plugins --features libcamera --test libcamera_dmabuf_probe -- --ignored --nocapture`

#![cfg(all(target_os = "linux", feature = "libcamera"))]

use std::time::Duration;

use libcamera::camera::CameraConfigurationStatus;
use libcamera::camera_manager::CameraManager;
use libcamera::framebuffer::AsFrameBuffer;
use libcamera::framebuffer_allocator::FrameBufferAllocator;
use libcamera::geometry::Size;
use libcamera::pixel_format::PixelFormat;
use libcamera::stream::StreamRole;

const PF_YUYV: PixelFormat = PixelFormat::new(u32::from_le_bytes([b'Y', b'U', b'Y', b'V']), 0);

/// Resolve what a file descriptor points at via /proc/self/fd (a dma-buf shows
/// up as "/dmabuf:..." or "anon_inode:dmabuf").
fn fd_target(fd: i32) -> String {
    std::fs::read_link(format!("/proc/self/fd/{fd}"))
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|e| format!("<unreadable: {e}>"))
}

#[test]
#[ignore = "needs a real camera; feasibility probe, prints findings"]
fn inspect_buffer_fd() {
    let mgr = CameraManager::new().unwrap();
    let cameras = mgr.cameras();
    let cam = cameras.get(0).expect("camera present");
    let cam = cam.acquire().expect("acquire");

    let mut cfgs = cam
        .generate_configuration(&[StreamRole::ViewFinder])
        .unwrap();
    {
        let mut cfg = cfgs.get_mut(0).unwrap();
        cfg.set_pixel_format(PF_YUYV);
        cfg.set_size(Size {
            width: 640,
            height: 480,
        });
    }
    assert!(!matches!(
        cfgs.validate(),
        CameraConfigurationStatus::Invalid
    ));
    let mut cam_mut = cam;
    cam_mut.configure(&mut cfgs).unwrap();

    let cfg = cfgs.get(0).unwrap();
    let stream = cfg.stream().unwrap();
    let mut alloc = FrameBufferAllocator::new(&cam_mut);
    let buffers = alloc.alloc(&stream).unwrap();
    println!("allocated {} buffers", buffers.len());

    // Inspect each plane's fd before any capture, the allocator already owns the
    // backing dma-bufs.
    for (bi, buf) in buffers.iter().enumerate() {
        for (pi, plane) in buf.planes().into_iter().enumerate() {
            let fd = plane.fd();
            println!(
                "buffer {bi} plane {pi}: fd={fd} len={} offset={:?} target={}",
                plane.len(),
                plane.offset(),
                fd_target(fd),
            );
        }
        if bi == 0 {
            break; // one buffer is enough to see the backing
        }
    }

    let _ = Duration::from_secs(0);
    println!(
        "VERDICT: a target of 'anon_inode:dmabuf' means a real dma-buf fd is \
         available; whether a discrete GPU can import it depends on the backing \
         allocator (USB/vmalloc dma-bufs are CPU-only and a dGPU cannot map them \
         zero-copy)."
    );
}

/// The decisive experiment: ask the real GPU whether it can import the camera's
/// dma-buf, via `vkGetMemoryFdPropertiesKHR`. A non-zero memory-type mask means
/// zero-copy is viable on this hardware; `0` means it is not (CPU upload only).
#[cfg(feature = "libcamera-dmabuf")]
#[tokio::test]
#[ignore = "needs a real camera + a Vulkan GPU; feasibility experiment"]
async fn gpu_can_import_camera_dmabuf() {
    let mgr = CameraManager::new().unwrap();
    let cameras = mgr.cameras();
    let cam = cameras.get(0).expect("camera present");
    let cam = cam.acquire().expect("acquire");
    let mut cfgs = cam
        .generate_configuration(&[StreamRole::ViewFinder])
        .unwrap();
    {
        let mut cfg = cfgs.get_mut(0).unwrap();
        cfg.set_pixel_format(PF_YUYV);
        cfg.set_size(Size {
            width: 640,
            height: 480,
        });
    }
    assert!(!matches!(
        cfgs.validate(),
        CameraConfigurationStatus::Invalid
    ));
    let mut cam_mut = cam;
    cam_mut.configure(&mut cfgs).unwrap();
    let cfg = cfgs.get(0).unwrap();
    let stream = cfg.stream().unwrap();
    let mut alloc = FrameBufferAllocator::new(&cam_mut);
    // `buffers` must stay alive: it owns the dma-buf the fd refers to.
    let buffers = alloc.alloc(&stream).unwrap();
    let (fd, len) = {
        let planes = buffers[0].planes();
        let plane = planes.into_iter().next().unwrap();
        (plane.fd(), plane.len() as u64)
    };
    println!("importing camera dma-buf fd={fd} len={len}");

    match g2g_plugins::libcamera_dmabuf::import_dmabuf_to_gpu_buffer(fd, len).await {
        Ok(r) if r.bound => println!(
            "RESULT: memory_type_bits={:#x}, VkBuffer BOUND to the imported \
             memory -> a GPU-usable buffer aliases the camera dma-buf with no \
             upload. Zero-copy import works on this hardware.",
            r.memory_type_bits
        ),
        Ok(r) if r.memory_type_bits == 0 => println!(
            "RESULT: memory_type_bits=0 -> the GPU CANNOT import this dma-buf. \
             Zero-copy is not possible here; the M315 CPU-upload path is correct."
        ),
        Ok(r) => println!(
            "RESULT: importable (bits={:#x}) but bind failed -> partial support; \
             use the CPU-upload path.",
            r.memory_type_bits
        ),
        Err(e) => println!(
            "RESULT: import failed ({e:?}) -> driver lacks \
             VK_EXT_external_memory_dma_buf or rejected the fd; use CPU upload."
        ),
    }
    // Keep the buffers alive until after the import.
    drop(buffers);
}
