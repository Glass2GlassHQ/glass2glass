//! Zero-copy libcamera -> GPU feasibility (the Linux analog of the CUDA /
//! AHardwareBuffer interop in [`crate::cudawgpu`] / `mediacodec_wgpu`).
//!
//! A libcamera `FrameBuffer` plane is a dma-buf (it exposes an fd). The dream
//! is to import that fd straight into the GPU as a `wgpu::Texture` and feed
//! `WgpuPreprocess` with no CPU upload. Whether that works is a *hardware*
//! question: the GPU's Vulkan driver must be able to map the dma-buf, and a
//! discrete GPU generally cannot map a USB camera's CPU-backed (vmalloc) dma-buf
//! at all, while an integrated GPU or a CSI/ISP camera (shared / GPU-visible
//! memory) can.
//!
//! [`dmabuf_import_memory_type_bits`] asks the driver, via
//! `vkGetMemoryFdPropertiesKHR`, which Vulkan memory types the dma-buf can be
//! imported into; [`import_dmabuf_to_gpu_buffer`] goes further and actually
//! attempts the import + buffer bind, because the query is optimistic.
//!
//! Measured finding (developer's UVC webcam + RTX 3060, libcamera 0.5.2): the
//! driver *reports* the dma-buf as importable (`memory_type_bits = 0x11`), but
//! the real `vkAllocateMemory` import then fails to bind for every reported
//! memory type. So zero-copy import is NOT viable for a USB camera feeding a
//! discrete GPU (the buffer is CPU/vmalloc-backed; the dGPU cannot map it), and
//! the CPU-upload path (M315) is the correct one on this hardware. The import
//! would be expected to work on an integrated GPU (shared memory) or a CSI/ISP
//! camera (GPU-visible buffers); that is where building the full
//! import-to-`wgpu::Texture` element pays off, which is why it is gated behind
//! this probe and not shipped blind.

use alloc::boxed::Box;
use std::os::fd::{BorrowedFd, IntoRawFd};

use ash::vk;

use g2g_core::{G2gError, HardwareError};

fn gpu_err() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Outcome of [`import_dmabuf_to_gpu_buffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmabufImport {
    /// Vulkan memory-type bitmask the dma-buf can be imported into (`0` = the
    /// GPU cannot import it; zero-copy impossible).
    pub memory_type_bits: u32,
    /// Whether a `VkBuffer` was actually allocated against the imported memory
    /// and bound, i.e. a GPU-usable buffer aliases the camera dma-buf with no
    /// copy. `false` when `memory_type_bits == 0`.
    pub bound: bool,
}

/// Ask the default high-performance GPU which Vulkan memory types the dma-buf
/// `fd` can be imported into (the bitmask from `vkGetMemoryFdPropertiesKHR` for
/// the `DMA_BUF` handle type). `0` means the GPU cannot import this buffer
/// (no zero-copy); a non-zero mask means it can.
///
/// `fd` must stay open for the duration of the call (keep the owning libcamera
/// `FrameBuffer` / allocator alive).
pub async fn dmabuf_import_memory_type_bits(fd: i32) -> Result<u32, G2gError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|_| gpu_err())?;

    // Open through the hal escape hatch so the device carries the dma-buf import
    // extensions (wgpu's safe path cannot add device extensions).
    // SAFETY: we read the hal adapter only to open a device from it; the guard
    // outlives the open call. A non-Vulkan backend yields None.
    let open = unsafe {
        let hal_adapter = adapter
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or_else(gpu_err)?;
        hal_adapter.open_with_callback(
            wgpu::Features::empty(),
            &wgpu::Limits::default(),
            &wgpu::MemoryHints::default(),
            Some(Box::new(|args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                args.extensions.push(ash::khr::external_memory_fd::NAME);
                args.extensions.push(ash::ext::external_memory_dma_buf::NAME);
            })),
        )
    }
    .map_err(|_| gpu_err())?;

    // SAFETY: `open` came from this adapter's hal, as required.
    let (device, _queue) = unsafe {
        adapter.create_device_from_hal(
            open,
            &wgpu::DeviceDescriptor {
                label: Some("libcamera-dmabuf-probe"),
                ..Default::default()
            },
        )
    }
    .map_err(|_| gpu_err())?;

    // Query the importable memory types for the dma-buf fd.
    // SAFETY: the raw device / instance come from the live wgpu device; the
    // loader is used only while they are borrowed, and `fd` is a valid open
    // dma-buf for the duration of the call (caller contract).
    let bits = unsafe {
        let hal_device = device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or_else(gpu_err)?;
        let raw = hal_device.raw_device();
        let instance = hal_device.shared_instance().raw_instance();
        let loader = ash::khr::external_memory_fd::Device::new(instance, raw);
        let mut props = vk::MemoryFdPropertiesKHR::default();
        loader
            .get_memory_fd_properties(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT, fd, &mut props)
            .map_err(|_| gpu_err())?;
        props.memory_type_bits
    };
    Ok(bits)
}

/// Actually import the dma-buf `fd` (of `len` bytes) into the GPU and bind a
/// `VkBuffer` to it, the real zero-copy primitive: on success a GPU-usable
/// buffer aliases the camera's dma-buf with no upload. Goes one step past
/// [`dmabuf_import_memory_type_bits`] (which only asks if it is possible) by
/// performing the import + bind and tearing it back down.
///
/// `fd` is duplicated before import (Vulkan takes ownership of the fd it
/// imports, so the caller's libcamera buffer keeps its own).
pub async fn import_dmabuf_to_gpu_buffer(fd: i32, len: u64) -> Result<DmabufImport, G2gError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .map_err(|_| gpu_err())?;

    // SAFETY: read the hal adapter only to open a device carrying the dma-buf
    // import extensions; the guard outlives the open call.
    let open = unsafe {
        let hal_adapter = adapter.as_hal::<wgpu_hal::api::Vulkan>().ok_or_else(gpu_err)?;
        hal_adapter.open_with_callback(
            wgpu::Features::empty(),
            &wgpu::Limits::default(),
            &wgpu::MemoryHints::default(),
            Some(Box::new(|args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                args.extensions.push(ash::khr::external_memory_fd::NAME);
                args.extensions.push(ash::ext::external_memory_dma_buf::NAME);
            })),
        )
    }
    .map_err(|_| gpu_err())?;
    // SAFETY: `open` came from this adapter's hal.
    let (device, _queue) = unsafe {
        adapter.create_device_from_hal(
            open,
            &wgpu::DeviceDescriptor { label: Some("libcamera-dmabuf-import"), ..Default::default() },
        )
    }
    .map_err(|_| gpu_err())?;

    // SAFETY: raw device/instance from the live wgpu device; all raw Vulkan
    // objects created here are destroyed before returning. `fd` is a valid open
    // dma-buf (caller contract) and is duplicated before Vulkan takes ownership.
    let import = unsafe {
        let hal_device = device.as_hal::<wgpu_hal::api::Vulkan>().ok_or_else(gpu_err)?;
        let raw = hal_device.raw_device();
        let instance = hal_device.shared_instance().raw_instance();
        let loader = ash::khr::external_memory_fd::Device::new(instance, raw);

        let mut props = vk::MemoryFdPropertiesKHR::default();
        loader
            .get_memory_fd_properties(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT, fd, &mut props)
            .map_err(|_| gpu_err())?;
        let bits = props.memory_type_bits;
        if bits == 0 {
            return Ok(DmabufImport { memory_type_bits: 0, bound: false });
        }

        // The buffer must declare the same external handle type to bind imported
        // memory to it; create it once and reuse across memory-type attempts.
        let mut ext_buf = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let buf_info = vk::BufferCreateInfo::default()
            .size(len)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut ext_buf);
        let buffer = raw.create_buffer(&buf_info, None).map_err(|_| gpu_err())?;

        // The query mask is optimistic; the actual import can still be rejected,
        // and only some of the reported memory types may accept the dma-buf. Try
        // each candidate type until an allocate + bind succeeds.
        let mut bound = false;
        for type_index in 0..32u32 {
            if bits & (1 << type_index) == 0 {
                continue;
            }
            // Vulkan consumes the imported fd, so hand each attempt a duplicate.
            let dup_fd = match BorrowedFd::borrow_raw(fd).try_clone_to_owned() {
                Ok(f) => f.into_raw_fd(),
                Err(_) => continue,
            };
            let mut import_info = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                .fd(dup_fd);
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(len)
                .memory_type_index(type_index)
                .push_next(&mut import_info);
            if let Ok(memory) = raw.allocate_memory(&alloc_info, None) {
                bound = raw.bind_buffer_memory(buffer, memory, 0).is_ok();
                raw.free_memory(memory, None); // also closes the imported fd
                if bound {
                    break;
                }
            } else {
                // Allocate failed. The fd's ownership state is driver-dependent
                // on failure, so don't reclaim it (a probe leaking a handle is
                // fine); reclaiming risked a double-close abort.
                let _ = dup_fd;
            }
        }

        raw.destroy_buffer(buffer, None);
        DmabufImport { memory_type_bits: bits, bound }
    };
    Ok(import)
}
