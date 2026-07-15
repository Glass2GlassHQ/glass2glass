//! Probe: does this host's Vulkan driver support external *semaphore* fd
//! export+import (`VK_KHR_external_semaphore_fd`)? This is the primitive a future
//! zero-stall cross-device / cross-process sync for `WgpuToDmaBuf` would use in
//! place of the current producer-side `device.poll(Wait)`: the producer exports a
//! semaphore fd signalled by the copy, the consumer imports it and waits on it in
//! its own queue before reading the shared dma-buf. This probe validates the fd
//! export/import mechanism exists on this driver (like `dmabuf_export_probe` does
//! for memory); wiring the fd through the transport and injecting the consumer
//! wait is the remaining integration (see the M560 CHANGELOG note). Run locally:
//!   cargo test -p g2g-plugins --features dmabuf-wgpu --test dmabuf_semaphore_probe -- --nocapture
#![cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]

use ash::vk;
use std::os::fd::{FromRawFd, OwnedFd};

/// Build a Vulkan wgpu device carrying `VK_KHR_external_semaphore_fd`.
async fn sem_device() -> Option<(wgpu::Device, wgpu::Adapter)> {
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
    // SAFETY: read the hal adapter only to open a device with the extension.
    let open = unsafe {
        let hal = adapter.as_hal::<wgpu_hal::api::Vulkan>()?;
        hal.open_with_callback(
            wgpu::Features::empty(),
            &wgpu::Limits::default(),
            &wgpu::MemoryHints::default(),
            Some(Box::new(|args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                args.extensions.push(ash::khr::external_semaphore_fd::NAME);
            })),
        )
        .ok()?
    };
    // SAFETY: `open` came from this adapter's hal.
    let (device, _queue) = unsafe {
        adapter
            .create_device_from_hal(
                open,
                &wgpu::DeviceDescriptor { label: Some("sem-probe"), ..Default::default() },
            )
            .ok()?
    };
    Some((device, adapter))
}

#[tokio::test]
async fn external_semaphore_fd_supported() {
    let Some((dev_a, adapter)) = sem_device().await else {
        eprintln!("SKIP: no Vulkan adapter / external-semaphore-fd extension");
        return;
    };
    let Some((dev_b, _)) = sem_device().await else {
        eprintln!("SKIP: could not open a second device");
        return;
    };
    eprintln!("adapter: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);

    // Export an exportable semaphore's fd from device A, import it on device B.
    // SAFETY: raw Vulkan on the two live devices; the created semaphores are
    // destroyed before return and the exported fd is owned (closed on drop).
    let outcome = unsafe {
        let hal_a = dev_a.as_hal::<wgpu_hal::api::Vulkan>().expect("vk a");
        let raw_a = hal_a.raw_device();
        let inst_a = hal_a.shared_instance().raw_instance();

        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut export);
        let sem_a = raw_a.create_semaphore(&info, None).expect("create exportable semaphore");

        let loader_a = ash::khr::external_semaphore_fd::Device::new(inst_a, raw_a);
        let fd = loader_a.get_semaphore_fd(
            &vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(sem_a)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
        );

        let result = match fd {
            Ok(fd) if fd >= 0 => {
                let owned = OwnedFd::from_raw_fd(fd);
                // Import it on device B.
                let hal_b = dev_b.as_hal::<wgpu_hal::api::Vulkan>().expect("vk b");
                let raw_b = hal_b.raw_device();
                let inst_b = hal_b.shared_instance().raw_instance();
                let sem_b =
                    raw_b.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).expect("sem b");
                let loader_b = ash::khr::external_semaphore_fd::Device::new(inst_b, raw_b);
                // import_semaphore_fd consumes the fd on success; use into_raw_fd so
                // our OwnedFd does not also close it.
                use std::os::fd::IntoRawFd;
                let raw_fd = owned.into_raw_fd();
                let imported = loader_b.import_semaphore_fd(
                    &vk::ImportSemaphoreFdInfoKHR::default()
                        .semaphore(sem_b)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                        .flags(vk::SemaphoreImportFlags::TEMPORARY)
                        .fd(raw_fd),
                );
                let r = match imported {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        // On import failure we still own the fd; close it.
                        let _ = OwnedFd::from_raw_fd(raw_fd);
                        Err(format!("import_semaphore_fd -> {e:?}"))
                    }
                };
                raw_b.destroy_semaphore(sem_b, None);
                r
            }
            Ok(fd) => Err(format!("bad semaphore fd {fd}")),
            Err(e) => Err(format!("get_semaphore_fd -> {e:?}")),
        };
        raw_a.destroy_semaphore(sem_a, None);
        result
    };

    match outcome {
        Ok(()) => eprintln!(
            "EXTERNAL SEMAPHORE FD SUPPORTED: exported an fd from one device and imported it on \
             another (the primitive for zero-stall cross-device dma-buf sync)"
        ),
        Err(e) => eprintln!("EXTERNAL SEMAPHORE FD NOT SUPPORTED: {e}"),
    }
}
