//! Probe: does this host's Vulkan driver support dma-buf *export*
//! (`vkGetMemoryFdKHR` with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT`)?
//! Import is already exercised by `dmabuftowgpu`; export is the missing half a
//! `WgpuToDmaBuf` element needs. Run locally: needs a GPU + the extension.
//!   cargo test -p g2g-plugins --features dmabuf-wgpu --test dmabuf_export_probe -- --nocapture
#![cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]

use ash::vk;
use std::os::fd::{FromRawFd, OwnedFd};

async fn export_device() -> Option<(wgpu::Device, wgpu::Queue, wgpu::Adapter)> {
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
        .ok()?;
    // SAFETY: read the hal adapter only to open a device carrying the export
    // extensions; the guard outlives the open call.
    let open = unsafe {
        let hal = adapter.as_hal::<wgpu_hal::api::Vulkan>()?;
        hal.open_with_callback(
            wgpu::Features::empty(),
            &wgpu::Limits::default(),
            &wgpu::MemoryHints::default(),
            Some(Box::new(
                |args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                    args.extensions.push(ash::khr::external_memory_fd::NAME);
                    args.extensions
                        .push(ash::ext::external_memory_dma_buf::NAME);
                },
            )),
        )
        .ok()?
    };
    // SAFETY: `open` came from this adapter's hal.
    let (device, queue) = unsafe {
        adapter
            .create_device_from_hal(
                open,
                &wgpu::DeviceDescriptor {
                    label: Some("export-probe"),
                    ..Default::default()
                },
            )
            .ok()?
    };
    Some((device, queue, adapter))
}

#[tokio::test]
async fn dmabuf_export_supported() {
    let Some((device, _queue, adapter)) = export_device().await else {
        eprintln!("SKIP: no Vulkan adapter / extensions");
        return;
    };
    let info = adapter.get_info();
    eprintln!("adapter: {} ({:?})", info.name, info.backend);

    const SIZE: u64 = 4096;
    // SAFETY: raw Vulkan on the live device; all raw handles created here are freed
    // before return.
    let result = unsafe {
        let hal = device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .expect("vulkan device");
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();
        let phys = hal.raw_physical_device();

        // Exportable buffer with the dma-buf handle type.
        let mut ext = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let buf_info = vk::BufferCreateInfo::default()
            .size(SIZE)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut ext);
        let buffer = raw.create_buffer(&buf_info, None).expect("create_buffer");

        let reqs = raw.get_buffer_memory_requirements(buffer);
        let props = instance.get_physical_device_memory_properties(phys);
        let mem_type = (0..props.memory_type_count)
            .find(|&i| {
                (reqs.memory_type_bits & (1 << i)) != 0
                    && props.memory_types[i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .expect("device-local memory type");

        // Export as dma-buf, dedicated to this buffer.
        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_type)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        let memory = match raw.allocate_memory(&alloc, None) {
            Ok(m) => m,
            Err(e) => {
                raw.destroy_buffer(buffer, None);
                eprintln!("allocate_memory(dma_buf export) FAILED: {e:?}");
                return;
            }
        };
        raw.bind_buffer_memory(buffer, memory, 0).expect("bind");

        let loader = ash::khr::external_memory_fd::Device::new(instance, raw);
        let fd = loader.get_memory_fd(
            &vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
        );
        let outcome = match fd {
            Ok(fd) if fd >= 0 => {
                // Own it so it is closed on drop.
                let _owned = OwnedFd::from_raw_fd(fd);
                Ok(reqs.size)
            }
            Ok(fd) => Err(format!("bad fd {fd}")),
            Err(e) => Err(format!("get_memory_fd(DMA_BUF) -> {e:?}")),
        };
        raw.free_memory(memory, None);
        raw.destroy_buffer(buffer, None);
        outcome
    };

    match result {
        Ok(size) => eprintln!("DMA-BUF EXPORT SUPPORTED: got an fd for a {size}-byte buffer"),
        Err(e) => eprintln!("DMA-BUF EXPORT NOT SUPPORTED: {e}"),
    }
}
