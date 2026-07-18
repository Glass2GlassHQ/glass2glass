//! Probe: can a *timeline* semaphore be signalled on one wgpu device's queue
//! (via `wgpu_hal::vulkan::Queue::add_signal_semaphore`) and observed by a host
//! `vkWaitSemaphores` on a *second* device that imported the same semaphore as an
//! `OPAQUE_FD`? This is the exact primitive the zero-stall `WgpuToDmaBuf` export
//! uses: the producer signals a per-frame timeline value on the copy submit (no
//! `device.poll(Wait)`), exports the timeline fd once, and the consumer host-waits
//! that value before reading the shared dma-buf. `dmabuf_semaphore_probe` proved
//! *binary* fd export/import exists; this proves the *timeline* signal + host wait
//! works across devices, which the integration relies on. Run locally:
//!   cargo test -p g2g-plugins --features dmabuf-wgpu --test dmabuf_timeline_probe -- --nocapture
#![cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]

use ash::vk;
use std::os::fd::{FromRawFd, IntoRawFd, OwnedFd};

/// Build a Vulkan wgpu device carrying `VK_KHR_external_semaphore_fd`. wgpu itself
/// enables the timeline-semaphore feature (it uses timeline semaphores
/// internally), so a timeline semaphore can be created without extra plumbing.
async fn sem_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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
            Some(Box::new(
                |args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                    args.extensions.push(ash::khr::external_semaphore_fd::NAME);
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
                    label: Some("timeline-probe"),
                    ..Default::default()
                },
            )
            .ok()?
    };
    Some((device, queue))
}

#[tokio::test]
async fn cross_device_timeline_semaphore_signal_and_host_wait() {
    let Some((dev_a, queue_a)) = sem_device().await else {
        eprintln!("SKIP: no Vulkan adapter / external-semaphore-fd extension");
        return;
    };
    let Some((dev_b, _queue_b)) = sem_device().await else {
        eprintln!("SKIP: could not open a second device");
        return;
    };

    // SAFETY: raw Vulkan on two live devices. Both semaphores are created up front
    // and destroyed unconditionally at the end; the exported fd is owned until
    // consumed by a successful import.
    let outcome: Result<(), String> = unsafe {
        let hal_a = dev_a.as_hal::<wgpu_hal::api::Vulkan>().expect("vk a");
        let raw_a = hal_a.raw_device().clone();
        let inst_a = hal_a.shared_instance().raw_instance().clone();
        let hal_b = dev_b.as_hal::<wgpu_hal::api::Vulkan>().expect("vk b");
        let raw_b = hal_b.raw_device().clone();
        let inst_b = hal_b.shared_instance().raw_instance().clone();

        // Exportable timeline semaphore on A (initial 0); plain timeline on B.
        let mut type_a = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let mut type_b = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let sem_a = raw_a.create_semaphore(
            &vk::SemaphoreCreateInfo::default()
                .push_next(&mut type_a)
                .push_next(&mut export),
            None,
        );
        let sem_b = raw_b.create_semaphore(
            &vk::SemaphoreCreateInfo::default().push_next(&mut type_b),
            None,
        );

        let result = match (sem_a, sem_b) {
            (Ok(sem_a), Ok(sem_b)) => {
                let r = run_probe(
                    &dev_a, &queue_a, &raw_a, &inst_a, sem_a, &raw_b, &inst_b, sem_b,
                );
                raw_a.destroy_semaphore(sem_a, None);
                raw_b.destroy_semaphore(sem_b, None);
                r
            }
            (a, b) => {
                if let Ok(s) = a {
                    raw_a.destroy_semaphore(s, None);
                }
                if let Ok(s) = b {
                    raw_b.destroy_semaphore(s, None);
                }
                Err(format!(
                    "create timeline semaphore(s) failed: a={a:?} b={b:?}"
                ))
            }
        };
        result
    };

    match outcome {
        Ok(()) => eprintln!(
            "CROSS-DEVICE TIMELINE SEMAPHORE OK: device A signalled value 1 on its queue and \
             device B observed it via an imported OPAQUE_FD timeline (the zero-stall sync primitive)"
        ),
        Err(e) => eprintln!("CROSS-DEVICE TIMELINE SEMAPHORE UNSUPPORTED: {e}"),
    }
}

/// Export A's timeline fd, import it on B, signal value 1 from A's queue (no
/// producer wait), then host-wait value 1 on B.
///
/// # Safety
/// The two raw devices / instances are live and the two semaphores were created on
/// them (A exportable). Caller destroys the semaphores.
#[allow(clippy::too_many_arguments)]
unsafe fn run_probe(
    dev_a: &wgpu::Device,
    queue_a: &wgpu::Queue,
    raw_a: &ash::Device,
    inst_a: &ash::Instance,
    sem_a: vk::Semaphore,
    raw_b: &ash::Device,
    inst_b: &ash::Instance,
    sem_b: vk::Semaphore,
) -> Result<(), String> {
    // Export A's fd, import it (permanently) onto B's timeline semaphore.
    let loader_a = ash::khr::external_semaphore_fd::Device::new(inst_a, raw_a);
    // SAFETY: `sem_a` is a live exportable semaphore on `raw_a`.
    let fd = unsafe {
        loader_a.get_semaphore_fd(
            &vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(sem_a)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
        )
    };
    let owned_fd = match fd {
        // SAFETY: a fresh, owned OPAQUE_FD.
        Ok(fd) if fd >= 0 => unsafe { OwnedFd::from_raw_fd(fd) },
        other => return Err(format!("get_semaphore_fd -> {other:?}")),
    };
    let loader_b = ash::khr::external_semaphore_fd::Device::new(inst_b, raw_b);
    let raw_fd = owned_fd.into_raw_fd();
    // SAFETY: `sem_b` is a live timeline semaphore on `raw_b`; `raw_fd` is a valid
    // OPAQUE_FD owned by this process (consumed by a successful import).
    let imported = unsafe {
        loader_b.import_semaphore_fd(
            &vk::ImportSemaphoreFdInfoKHR::default()
                .semaphore(sem_b)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                .flags(vk::SemaphoreImportFlags::empty()) // permanent
                .fd(raw_fd),
        )
    };
    if let Err(e) = imported {
        // SAFETY: import failed, so we still own the fd; close it.
        let _ = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        return Err(format!("import_semaphore_fd -> {e:?}"));
    }

    // Signal timeline value 1 from A's queue via the wgpu-hal injection hook, on a
    // trivial submit. No device.poll(Wait): the point is B observes it without A
    // blocking on completion.
    {
        // SAFETY: `queue_a` is a live Vulkan-backend wgpu queue; `sem_a` is a
        // timeline semaphore on its device, signalled on the submit below.
        let hal_q = unsafe { queue_a.as_hal::<wgpu_hal::api::Vulkan>() }.expect("hal queue a");
        hal_q.add_signal_semaphore(sem_a, Some(1));
    }
    let enc = dev_a.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    queue_a.submit([enc.finish()]);

    // Host-wait value 1 on B (5 s). SUCCESS proves the cross-device timeline share.
    let sems = [sem_b];
    let vals = [1u64];
    let wait = vk::SemaphoreWaitInfo::default()
        .semaphores(&sems)
        .values(&vals);
    // SAFETY: `wait` references live locals for the call; `raw_b` is a live device.
    match unsafe { raw_b.wait_semaphores(&wait, 5_000_000_000) } {
        Ok(()) => Ok(()),
        Err(vk::Result::TIMEOUT) => Err("TIMEOUT: value 1 not observed on device B".into()),
        Err(e) => Err(format!("wait_semaphores -> {e:?}")),
    }
}
