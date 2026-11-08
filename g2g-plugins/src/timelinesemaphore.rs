use ash::vk;

use g2g_core::{G2gError, HardwareError};

/// Create an exportable *timeline* semaphore on `device` (initial value 0) and
/// export its `OPAQUE_FD`. Cross-device / cross-process timeline share is
/// validated on the RTX 3060 by `dmabuf_timeline_probe`.
///
/// # Safety
/// `device` must be a Vulkan-backend wgpu device carrying
/// `VK_KHR_external_semaphore_fd`.
pub(crate) unsafe fn create_export_semaphore(
    device: &wgpu::Device,
) -> Result<(vk::Semaphore, i32), G2gError> {
    // SAFETY: caller guarantees the export device; the hal guard is held for the
    // whole creation + export.
    unsafe {
        let hal = device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let raw = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();

        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let info = vk::SemaphoreCreateInfo::default()
            .push_next(&mut type_info)
            .push_next(&mut export);
        let sem = raw
            .create_semaphore(&info, None)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        let loader = ash::khr::external_semaphore_fd::Device::new(instance, raw);
        let fd = loader.get_semaphore_fd(
            &vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(sem)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
        );
        match fd {
            Ok(fd) if fd >= 0 => Ok((sem, fd)),
            _ => {
                raw.destroy_semaphore(sem, None);
                Err(G2gError::Hardware(HardwareError::Other))
            }
        }
    }
}
