//! M304: Android MediaCodec -> wgpu/Vulkan zero-copy bridge.
//!
//! The Android analog of [`cudawgpu`](crate::cudawgpu): keep the decoded frame
//! on the GPU (no CPU NV12 pack) by importing the `AImage`'s backing
//! `AHardwareBuffer` into a wgpu Vulkan device, then doing a device-local copy
//! into a plain sampled texture `WgpuPreprocess` consumes, no PCIe / CPU
//! readback.
//!
//! ## The wall, and the probe
//!
//! Unlike the CUDA path (which owns a plain `R8_UINT` image and does YUV->RGB in
//! the shader), the decoded `AHardwareBuffer` is *vendor YUV*. Sampling it in
//! Vulkan in the general case needs a [`VkSamplerYcbcrConversion`] baked into an
//! immutable sampler in the descriptor layout, which wgpu's bind-group / pipeline
//! API cannot express. The device-local-copy plan is only viable if the imported
//! buffer reports a *known multi-planar `VkFormat`* (e.g.
//! `G8_B8R8_2PLANE_420_UNORM`): then we can `vkCmdCopyImage` per plane
//! (`VK_IMAGE_ASPECT_PLANE_0/1_BIT`) into our own `R8`/`R8G8` images and skip
//! ycbcr entirely. If instead the buffer is *opaque* (`external_format != 0`,
//! `format == UNDEFINED`), there is no per-plane access and ycbcr is forced.
//!
//! Everything downstream depends on which of those the device reports, so this
//! module's first job is the on-device probe: [`create_android_interop_device`]
//! opens a Vulkan-backed wgpu device with the AHB external-memory extension, and
//! [`ahb_format_info`] runs `vkGetAndroidHardwareBufferPropertiesANDROID` against
//! a real decoded buffer (captured via
//! [`MediaCodecDec::captured_hardware_buffer`](crate::mediacodecdec::MediaCodecDec::captured_hardware_buffer)).
//!
//! Android only (`mediacodec-wgpu` feature).

use alloc::boxed::Box;
use ash::vk;

use g2g_core::{G2gError, HardwareError};

/// Map any wgpu / Vulkan failure to a structured hardware error.
fn gpu_err<E>(_e: E) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// A wgpu device on the Vulkan backend with the Android AHardwareBuffer
/// external-memory extension enabled, plus the instance / adapter it was opened
/// from (kept alive alongside it). Mirrors `cudawgpu::InteropDevice`.
#[derive(Debug)]
pub struct InteropDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    // Drop order: device before adapter before instance.
    pub adapter: wgpu::Adapter,
    pub instance: wgpu::Instance,
}

/// Create a wgpu device that can import Android hardware buffers.
///
/// Forces the Vulkan backend (the only one with an AHB import path) and enables
/// `VK_ANDROID_external_memory_android_hardware_buffer` +
/// `VK_EXT_queue_family_foreign` via wgpu-hal's create-device callback
/// (`VK_KHR_sampler_ycbcr_conversion` / external-memory are Vulkan 1.1 core).
/// Fails loud if the adapter is not Vulkan or the extension is absent.
pub async fn create_android_interop_device() -> Result<InteropDevice, G2gError> {
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
        .map_err(gpu_err)?;

    let features = wgpu::Features::empty();
    let limits = wgpu::Limits::default();
    let memory_hints = wgpu::MemoryHints::default();

    // Open via the hal escape hatch so we can add the AHB device extensions;
    // wgpu-hal fills in the rest of the feature / queue chain.
    // SAFETY: we only read the hal adapter and immediately open a device from
    // it; the guard outlives the open call. Backend mismatch yields None.
    let open = unsafe {
        let hal_adapter = adapter
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        hal_adapter.open_with_callback(
            features,
            &limits,
            &memory_hints,
            Some(Box::new(|args: wgpu_hal::vulkan::CreateDeviceCallbackArgs| {
                args.extensions
                    .push(ash::android::external_memory_android_hardware_buffer::NAME);
                args.extensions.push(ash::ext::queue_family_foreign::NAME);
            })),
        )
    }
    .map_err(gpu_err)?;

    // SAFETY: `open` was produced by this adapter's hal, as required.
    let (device, queue) = unsafe {
        adapter.create_device_from_hal(
            open,
            &wgpu::DeviceDescriptor {
                label: Some("mediacodec-wgpu-interop"),
                required_features: features,
                required_limits: limits,
                ..Default::default()
            },
        )
    }
    .map_err(gpu_err)?;

    Ok(InteropDevice { device, queue, adapter, instance })
}

/// The Vulkan view of an imported `AHardwareBuffer`, as reported by
/// `vkGetAndroidHardwareBufferPropertiesANDROID`. This is the linchpin probe
/// result: [`vk_format`](Self::vk_format) vs [`external_format`](Self::external_format)
/// decides whether the device-local per-plane copy is viable.
#[derive(Debug, Clone)]
pub struct AhbFormatInfo {
    /// The buffer's `VkFormat`. A concrete value (e.g.
    /// `G8_B8R8_2PLANE_420_UNORM`) means per-plane image access is possible and
    /// the device-local copy is viable; `UNDEFINED` means opaque (see
    /// [`external_format`](Self::external_format)).
    pub vk_format: vk::Format,
    /// Nonzero opaque vendor format id, set iff [`vk_format`](Self::vk_format) is
    /// `UNDEFINED`. Nonzero forces a ycbcr conversion in an immutable sampler:
    /// no per-plane access, so the simple copy plan does not apply.
    pub external_format: u64,
    /// Bytes a dedicated import allocation must span.
    pub allocation_size: u64,
    /// Bitmask of memory types the import allocation may use.
    pub memory_type_bits: u32,
    /// Vulkan format-feature flags the buffer supports (sampling, ycbcr, etc.).
    pub format_features: vk::FormatFeatureFlags,
    /// The driver's suggested ycbcr model / range (BT.601 vs BT.709, narrow vs
    /// full), needed to set up a conversion if the opaque path is forced.
    pub suggested_ycbcr_model: vk::SamplerYcbcrModelConversion,
    pub suggested_ycbcr_range: vk::SamplerYcbcrRange,
}

impl AhbFormatInfo {
    /// True when the buffer exposes a concrete multi-planar / single `VkFormat`
    /// (per-plane image access possible, device-local copy viable). False when
    /// the buffer is opaque and a ycbcr conversion is forced.
    pub fn is_importable_format(&self) -> bool {
        self.vk_format != vk::Format::UNDEFINED && self.external_format == 0
    }
}

/// Query the Vulkan format properties of an `AHardwareBuffer` on an interop
/// device. `ahb` is the raw `*mut AHardwareBuffer` from the decoder, cast to
/// `*const vk::AHardwareBuffer`.
///
/// # Safety
/// `ahb` must point to a live `AHardwareBuffer` (e.g. an acquired reference held
/// for the duration of the call), and `dev` must be a Vulkan-backend interop
/// device from [`create_android_interop_device`].
pub unsafe fn ahb_format_info(
    dev: &InteropDevice,
    ahb: *const vk::AHardwareBuffer,
) -> Result<AhbFormatInfo, G2gError> {
    // SAFETY: caller guarantees a Vulkan device; we hold the hal guard for the
    // whole call and never retain raw handles past it.
    unsafe {
        let hal_device = dev
            .device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let raw: &ash::Device = hal_device.raw_device();
        let instance: &ash::Instance = hal_device.shared_instance().raw_instance();

        let ext =
            ash::android::external_memory_android_hardware_buffer::Device::new(instance, raw);

        let mut fmt = vk::AndroidHardwareBufferFormatPropertiesANDROID::default();
        let mut props =
            vk::AndroidHardwareBufferPropertiesANDROID::default().push_next(&mut fmt);
        ext.get_android_hardware_buffer_properties(ahb, &mut props).map_err(gpu_err)?;

        // `props` holds a mutable borrow of `fmt` (the push_next chain), so read
        // its fields out first; after this its borrow of `fmt` ends and `fmt`'s
        // own fields can be read below.
        let allocation_size = props.allocation_size;
        let memory_type_bits = props.memory_type_bits;

        Ok(AhbFormatInfo {
            vk_format: fmt.format,
            external_format: fmt.external_format,
            allocation_size,
            memory_type_bits,
            format_features: fmt.format_features,
            suggested_ycbcr_model: fmt.suggested_ycbcr_model,
            suggested_ycbcr_range: fmt.suggested_ycbcr_range,
        })
    }
}
