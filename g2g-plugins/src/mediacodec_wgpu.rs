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
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::CStr;

use ash::vk;

use g2g_core::{G2gError, HardwareError};

/// Compiled SPIR-V for the YCbCr -> RGBA compute shader
/// (`shaders/mediacodec_ycbcr.comp`, glslc, Vulkan 1.1).
const YCBCR_COMP_SPV: &[u8] = include_bytes!("shaders/mediacodec_ycbcr.comp.spv");

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
    /// Component swizzle the conversion must use for an external-format buffer
    /// (the driver dictates it; the conversion create-info must echo it back).
    pub sampler_ycbcr_conversion_components: vk::ComponentMapping,
    /// Suggested chroma sample location (cosited vs midpoint) per axis.
    pub suggested_x_chroma_offset: vk::ChromaLocation,
    pub suggested_y_chroma_offset: vk::ChromaLocation,
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
            sampler_ycbcr_conversion_components: fmt.sampler_ycbcr_conversion_components,
            suggested_x_chroma_offset: fmt.suggested_x_chroma_offset,
            suggested_y_chroma_offset: fmt.suggested_y_chroma_offset,
        })
    }
}

/// Pick a memory type index satisfying `type_bits` with the requested property
/// flags. Mirrors the helper in [`cudawgpu`](crate::cudawgpu).
fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    props.memory_types[..props.memory_type_count as usize]
        .iter()
        .enumerate()
        .position(|(i, t)| (type_bits & (1 << i)) != 0 && t.property_flags.contains(flags))
        .map(|i| i as u32)
}

/// The lowest memory type index allowed by `type_bits` (imported AHB memory
/// dictates the type bits and needs no extra property flags).
fn first_memory_type(type_bits: u32) -> Option<u32> {
    (0..32).find(|i| type_bits & (1 << i) != 0)
}

/// Build the immutable `VkSamplerYcbcrConversion` for an opaque (external-format)
/// AHardwareBuffer from its probed properties. Returns the conversion handle; the
/// caller owns it and must `destroy_sampler_ycbcr_conversion` it.
///
/// # Safety
/// `d` must be the interop device's raw `ash::Device`.
unsafe fn create_ycbcr_conversion(
    d: &ash::Device,
    info: &AhbFormatInfo,
) -> Result<vk::SamplerYcbcrConversion, G2gError> {
    let chroma_filter = if info
        .format_features
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER)
    {
        vk::Filter::LINEAR
    } else {
        vk::Filter::NEAREST
    };
    // For an external format the create-info must echo the driver's dictated
    // format (UNDEFINED + externalFormat), model / range / components / offsets.
    let mut ext_fmt = vk::ExternalFormatANDROID::default().external_format(info.external_format);
    let create = vk::SamplerYcbcrConversionCreateInfo::default()
        .format(vk::Format::UNDEFINED)
        .ycbcr_model(info.suggested_ycbcr_model)
        .ycbcr_range(info.suggested_ycbcr_range)
        .components(info.sampler_ycbcr_conversion_components)
        .x_chroma_offset(info.suggested_x_chroma_offset)
        .y_chroma_offset(info.suggested_y_chroma_offset)
        .chroma_filter(chroma_filter)
        .force_explicit_reconstruction(false)
        .push_next(&mut ext_fmt);
    // SAFETY: `d` is the interop device; create-info is fully populated.
    unsafe { d.create_sampler_ycbcr_conversion(&create, None) }.map_err(gpu_err)
}

/// Convert one decoded `AHardwareBuffer` (opaque vendor YCbCr) to RGBA on the
/// GPU and read the result back to host memory, for on-device validation of the
/// M304 conversion path. Returns `width * height * 4` bytes (R8G8B8A8).
///
/// The whole pipeline is raw Vulkan because wgpu's bind-group API cannot express
/// the immutable `VkSamplerYcbcrConversion` an opaque buffer requires: import the
/// AHB as a sampled image, sample it through a ycbcr-conversion sampler in a
/// compute shader, write RGBA to a self-allocated storage image, then
/// `vkCmdCopyImageToBuffer` into a host-visible buffer. No CPU/PCIe readback of
/// the decoded frame is involved; the readback here is only the validation tap.
///
/// This is the bring-up / probe entry point: on any error it returns early and
/// leaks the Vulkan objects created so far (the one-shot test process then
/// exits). The steady-state element path will reuse the pipeline objects and
/// hand the RGBA image to `WgpuPreprocess` as a wgpu texture instead.
///
/// # Safety
/// `ahb` must point to a live `AHardwareBuffer` (hold an acquired reference for
/// the call); `info` must be its [`ahb_format_info`] for an opaque buffer; `dev`
/// must be the interop device from [`create_android_interop_device`].
pub unsafe fn ahb_to_rgba_readback(
    dev: &InteropDevice,
    ahb: *const vk::AHardwareBuffer,
    info: &AhbFormatInfo,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, G2gError> {
    // SAFETY: caller guarantees a Vulkan interop device and a live AHB; we hold
    // the hal guard for the whole call and submit to the device's own queue,
    // waiting on a fence before reading back. Every handle created is destroyed
    // on the success path (errors abort the one-shot probe; see the doc note).
    unsafe {
        let hal = dev
            .device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let d: &ash::Device = hal.raw_device();
        let inst: &ash::Instance = hal.shared_instance().raw_instance();
        let phys = hal.raw_physical_device();
        let queue = hal.raw_queue();
        let qfi = hal.queue_family_index();
        let mem_props = inst.get_physical_device_memory_properties(phys);

        let extent = vk::Extent3D { width, height, depth: 1 };
        let color_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        // ---- ycbcr conversion + immutable sampler ----
        let conversion = create_ycbcr_conversion(d, info)?;
        let mut conv_for_sampler = vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .unnormalized_coordinates(false)
            .push_next(&mut conv_for_sampler);
        let sampler = d.create_sampler(&sampler_info, None).map_err(gpu_err)?;

        // ---- import the AHB as a sampled image (external format) ----
        let mut ext_fmt_img =
            vk::ExternalFormatANDROID::default().external_format(info.external_format);
        let mut ext_mem_img = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);
        let in_image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::UNDEFINED)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut ext_mem_img)
            .push_next(&mut ext_fmt_img);
        let in_image = d.create_image(&in_image_info, None).map_err(gpu_err)?;

        // Import the AHB's memory, dedicated to this image.
        let mem_type = first_memory_type(info.memory_type_bits)
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let mut import =
            vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(ahb as *mut _);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(in_image);
        let in_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(info.allocation_size)
            .memory_type_index(mem_type)
            .push_next(&mut dedicated)
            .push_next(&mut import);
        let in_mem = d.allocate_memory(&in_alloc, None).map_err(gpu_err)?;
        d.bind_image_memory(in_image, in_mem, 0).map_err(gpu_err)?;

        let mut conv_for_view = vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
        let in_view_info = vk::ImageViewCreateInfo::default()
            .image(in_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::UNDEFINED)
            .subresource_range(color_range)
            .push_next(&mut conv_for_view);
        let in_view = d.create_image_view(&in_view_info, None).map_err(gpu_err)?;

        // ---- output RGBA storage image (device-local, we own it) ----
        let out_image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let out_image = d.create_image(&out_image_info, None).map_err(gpu_err)?;
        let out_reqs = d.get_image_memory_requirements(out_image);
        let out_mt = find_memory_type(
            &mem_props,
            out_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let out_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(out_reqs.size)
            .memory_type_index(out_mt);
        let out_mem = d.allocate_memory(&out_alloc, None).map_err(gpu_err)?;
        d.bind_image_memory(out_image, out_mem, 0).map_err(gpu_err)?;
        let out_view_info = vk::ImageViewCreateInfo::default()
            .image(out_image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(color_range);
        let out_view = d.create_image_view(&out_view_info, None).map_err(gpu_err)?;

        // ---- host-visible readback buffer ----
        let byte_len = (width as u64) * (height as u64) * 4;
        let buf_info = vk::BufferCreateInfo::default()
            .size(byte_len)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = d.create_buffer(&buf_info, None).map_err(gpu_err)?;
        let buf_reqs = d.get_buffer_memory_requirements(buffer);
        let buf_mt = find_memory_type(
            &mem_props,
            buf_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let buf_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(buf_reqs.size)
            .memory_type_index(buf_mt);
        let buf_mem = d.allocate_memory(&buf_alloc, None).map_err(gpu_err)?;
        d.bind_buffer_memory(buffer, buf_mem, 0).map_err(gpu_err)?;

        // ---- descriptor layout (binding 0 = immutable ycbcr sampler) ----
        let immutable = [sampler];
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .immutable_samplers(&immutable),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let dsl = d.create_descriptor_set_layout(&dsl_info, None).map_err(gpu_err)?;
        let set_layouts = [dsl];
        let pl_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let pipeline_layout = d.create_pipeline_layout(&pl_info, None).map_err(gpu_err)?;

        // ---- compute pipeline ----
        let code = ash::util::read_spv(&mut std::io::Cursor::new(YCBCR_COMP_SPV)).map_err(gpu_err)?;
        let sm_info = vk::ShaderModuleCreateInfo::default().code(&code);
        let shader = d.create_shader_module(&sm_info, None).map_err(gpu_err)?;
        let entry = CStr::from_bytes_with_nul(b"main\0").map_err(gpu_err)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(entry);
        let cp_info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout);
        let pipelines = d
            .create_compute_pipelines(vk::PipelineCache::null(), &[cp_info], None)
            .map_err(|(_, e)| gpu_err(e))?;
        let pipeline = pipelines[0];

        // ---- descriptor pool + set ----
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
        let pool = d.create_descriptor_pool(&pool_info, None).map_err(gpu_err)?;
        let set_alloc =
            vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts);
        let set = d.allocate_descriptor_sets(&set_alloc).map_err(gpu_err)?[0];

        let in_desc = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(in_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let out_desc = [vk::DescriptorImageInfo::default()
            .image_view(out_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&in_desc),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_desc),
        ];
        d.update_descriptor_sets(&writes, &[]);

        // ---- record + submit ----
        let cmd_pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(qfi);
        let cmd_pool = d.create_command_pool(&cmd_pool_info, None).map_err(gpu_err)?;
        let cb_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cb = d.allocate_command_buffers(&cb_alloc).map_err(gpu_err)?[0];
        let begin =
            vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        d.begin_command_buffer(cb, &begin).map_err(gpu_err)?;

        // Imported image UNDEFINED -> SHADER_READ_ONLY; output UNDEFINED ->
        // GENERAL. Queue families IGNORED (no ownership transfer): the AHB import
        // makes the contents available without a foreign-queue release to wait on.
        let to_read = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(in_image)
            .subresource_range(color_range);
        let to_general = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(out_image)
            .subresource_range(color_range);
        d.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_read, to_general],
        );

        d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
        d.cmd_bind_descriptor_sets(
            cb,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[set],
            &[],
        );
        d.cmd_dispatch(cb, width.div_ceil(8), height.div_ceil(8), 1);

        // Output GENERAL -> TRANSFER_SRC for the copy-to-buffer.
        let to_copy = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(out_image)
            .subresource_range(color_range);
        d.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_copy],
        );

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(extent);
        d.cmd_copy_image_to_buffer(
            cb,
            out_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[region],
        );
        d.end_command_buffer(cb).map_err(gpu_err)?;

        let cbs = [cb];
        let submit = vk::SubmitInfo::default().command_buffers(&cbs);
        let fence = d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(gpu_err)?;
        d.queue_submit(queue, &[submit], fence).map_err(gpu_err)?;
        d.wait_for_fences(&[fence], true, u64::MAX).map_err(gpu_err)?;

        // ---- read back ----
        let ptr = d
            .map_memory(buf_mem, 0, byte_len, vk::MemoryMapFlags::empty())
            .map_err(gpu_err)? as *const u8;
        let mut out = vec![0u8; byte_len as usize];
        core::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), byte_len as usize);
        d.unmap_memory(buf_mem);

        // ---- cleanup (GPU idle past the fence) ----
        d.destroy_fence(fence, None);
        d.destroy_command_pool(cmd_pool, None);
        d.destroy_descriptor_pool(pool, None);
        d.destroy_pipeline(pipeline, None);
        d.destroy_shader_module(shader, None);
        d.destroy_pipeline_layout(pipeline_layout, None);
        d.destroy_descriptor_set_layout(dsl, None);
        d.destroy_buffer(buffer, None);
        d.free_memory(buf_mem, None);
        d.destroy_image_view(out_view, None);
        d.destroy_image(out_image, None);
        d.free_memory(out_mem, None);
        d.destroy_image_view(in_view, None);
        d.destroy_image(in_image, None);
        d.free_memory(in_mem, None);
        d.destroy_sampler(sampler, None);
        d.destroy_sampler_ycbcr_conversion(conversion, None);

        Ok(out)
    }
}

/// A converted RGBA frame living in a `wgpu::Texture`, with the device / queue it
/// lives on. Boxed as the [`WgpuKeepAlive`](g2g_core::WgpuKeepAlive) of a
/// [`MemoryDomain::WgpuTexture`](g2g_core::MemoryDomain::WgpuTexture): a consumer
/// that links wgpu (e.g. a future RGBA import path in `WgpuPreprocess`) downcasts
/// via [`as_any`](g2g_core::WgpuKeepAlive::as_any) to recover the texture and
/// adopt the device (a texture is bindable only on its own device). The RGBA
/// analog of g2g-ml's `WgpuNv12Texture`: the ycbcr conversion already happened in
/// [`YcbcrToRgba`], so the consumer samples RGBA directly with no YUV math.
pub struct WgpuRgbaTexture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    texture: wgpu::Texture,
}

impl core::fmt::Debug for WgpuRgbaTexture {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuRgbaTexture").field("texture", &self.texture).finish_non_exhaustive()
    }
}

impl WgpuRgbaTexture {
    /// Wrap an RGBA texture with the device / queue it lives on.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, texture: wgpu::Texture) -> Self {
        Self { device, queue, texture }
    }

    /// The backing RGBA (`Rgba8Unorm`) texture, for the importer to sample.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The device the texture lives on; the importer adopts it to bind the
    /// texture rather than uploading to its own device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue paired with [`device`](Self::device).
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}

impl g2g_core::WgpuKeepAlive for WgpuRgbaTexture {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// A reusable YCbCr -> RGBA converter for a fixed external format + geometry: the
/// steady-state form of [`ahb_to_rgba_readback`]. Builds the ycbcr conversion,
/// immutable sampler, descriptor layout, and compute pipeline once, then
/// [`convert`](Self::convert) imports each decoded `AHardwareBuffer`, runs the
/// compute pass into a fresh RGBA `wgpu::Texture`, and hands it downstream. No
/// CPU / PCIe readback of the decoded frame is involved.
///
/// The conversion is synchronous: each [`convert`](Self::convert) submits to the
/// device's queue and waits on a fence before returning, so the imported buffer
/// can be released back to the decoder immediately and the texture is ready for
/// the consumer. Because it submits to the wgpu device's queue directly (raw
/// Vulkan, bypassing wgpu's queue mutex), the owning element must run on a
/// single-thread executor, the same contract `MediaCodecDec` already documents.
/// Depth of the conversion ring: how many conversions may be outstanding (the
/// GPU still reading the imported buffer / writing the output) before the next
/// `convert` blocks on the oldest. Small: just enough to overlap a conversion
/// with the downstream consume of the previous frame.
const RING_DEPTH: usize = 3;

/// Transient Vulkan objects of one in-flight conversion, kept alive until its
/// fence signals (the GPU is done reading the imported buffer and the views).
/// The output image / memory are not here: they transfer to the wgpu texture.
#[derive(Debug)]
struct InFlight {
    in_image: vk::Image,
    in_mem: vk::DeviceMemory,
    in_view: vk::ImageView,
    out_view: vk::ImageView,
}

/// One slot of the conversion ring: its own fence / command buffer / descriptor
/// set, so a conversion can be in flight here while other slots run.
#[derive(Debug)]
struct ConvSlot {
    fence: vk::Fence,
    cmd_buf: vk::CommandBuffer,
    desc_set: vk::DescriptorSet,
    /// The transient objects of the conversion currently running on this slot,
    /// reclaimed (after a fence wait) when the slot is reused. `None` when idle.
    in_flight: Option<InFlight>,
}

pub struct YcbcrToRgba {
    device: wgpu::Device,
    raw: ash::Device,
    queue_raw: vk::Queue,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
    /// Probed once: the opaque buffer's external format and the import allocation
    /// size / memory-type bits (constant for a fixed format + geometry stream).
    external_format: u64,
    allocation_size: u64,
    import_mem_type_bits: u32,
    conversion: vk::SamplerYcbcrConversion,
    sampler: vk::Sampler,
    dsl: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    shader: vk::ShaderModule,
    cmd_pool: vk::CommandPool,
    desc_pool: vk::DescriptorPool,
    /// Round-robin ring of `RING_DEPTH` slots; `next` is the slot the next
    /// conversion uses (reclaiming whatever it last held).
    slots: Vec<ConvSlot>,
    next: usize,
}

impl core::fmt::Debug for YcbcrToRgba {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("YcbcrToRgba")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

impl YcbcrToRgba {
    /// Build the reusable conversion pipeline for an opaque AHardwareBuffer of
    /// `info`'s external format and the given geometry.
    ///
    /// # Safety
    /// `dev` must be the interop device from [`create_android_interop_device`];
    /// `info` must be the [`ahb_format_info`] of the buffers this will convert.
    pub unsafe fn new(
        dev: &InteropDevice,
        info: &AhbFormatInfo,
        width: u32,
        height: u32,
    ) -> Result<Self, G2gError> {
        // SAFETY: caller guarantees a Vulkan interop device; the hal guard is held
        // only to read the raw handles, which we clone / copy out.
        unsafe {
            let hal = dev
                .device
                .as_hal::<wgpu_hal::api::Vulkan>()
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            let d: ash::Device = hal.raw_device().clone();
            let inst: &ash::Instance = hal.shared_instance().raw_instance();
            let phys = hal.raw_physical_device();
            let qfi = hal.queue_family_index();
            let queue_raw = hal.raw_queue();
            let mem_props = inst.get_physical_device_memory_properties(phys);

            let conversion = create_ycbcr_conversion(&d, info)?;
            let mut conv_for_sampler =
                vk::SamplerYcbcrConversionInfo::default().conversion(conversion);
            let sampler_info = vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .unnormalized_coordinates(false)
                .push_next(&mut conv_for_sampler);
            let sampler = d.create_sampler(&sampler_info, None).map_err(gpu_err)?;

            let immutable = [sampler];
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .immutable_samplers(&immutable),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let dsl = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(gpu_err)?;
            let set_layouts = [dsl];
            let pipeline_layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
                .map_err(gpu_err)?;

            let code =
                ash::util::read_spv(&mut std::io::Cursor::new(YCBCR_COMP_SPV)).map_err(gpu_err)?;
            let shader = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
                .map_err(gpu_err)?;
            let entry = CStr::from_bytes_with_nul(b"main\0").map_err(gpu_err)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(entry);
            let pipeline = d
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| gpu_err(e))?[0];

            // The ring needs RING_DEPTH descriptor sets (one per slot, so a slot's
            // bindings stay valid while another slot's conversion runs).
            let ring = RING_DEPTH as u32;
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(ring),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(ring),
            ];
            let desc_pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(ring).pool_sizes(&pool_sizes),
                    None,
                )
                .map_err(gpu_err)?;
            let all_layouts = [dsl; RING_DEPTH];
            let desc_sets = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(&all_layouts),
                )
                .map_err(gpu_err)?;

            // RESET_COMMAND_BUFFER: each slot re-records its own buffer per reuse
            // (a whole-pool reset would clobber buffers still in flight).
            let cmd_pool = d
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(qfi)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(gpu_err)?;
            let cmd_bufs = d
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(ring),
                )
                .map_err(gpu_err)?;

            let mut slots = Vec::with_capacity(RING_DEPTH);
            for i in 0..RING_DEPTH {
                let fence =
                    d.create_fence(&vk::FenceCreateInfo::default(), None).map_err(gpu_err)?;
                slots.push(ConvSlot {
                    fence,
                    cmd_buf: cmd_bufs[i],
                    desc_set: desc_sets[i],
                    in_flight: None,
                });
            }

            Ok(Self {
                device: dev.device.clone(),
                raw: d,
                queue_raw,
                mem_props,
                width,
                height,
                external_format: info.external_format,
                allocation_size: info.allocation_size,
                import_mem_type_bits: info.memory_type_bits,
                conversion,
                sampler,
                dsl,
                pipeline_layout,
                pipeline,
                shader,
                cmd_pool,
                desc_pool,
                slots,
                next: 0,
            })
        }
    }

    /// The geometry this converter was built for.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Import one decoded `AHardwareBuffer`, convert it to RGBA on the GPU, and
    /// return the result as a `wgpu::Texture` (`Rgba8Unorm`), ready to sample.
    ///
    /// Pipelined: the conversion is submitted but not waited on, so up to
    /// `RING_DEPTH` conversions can overlap downstream consumption. The returned
    /// texture's writes complete on the same queue before any later wgpu use of
    /// it, so a consumer on this device sees correct data without an explicit
    /// wait. The caller may release `ahb` as soon as this returns: importing it
    /// gave Vulkan its own reference (held until the slot frees the memory).
    ///
    /// # Safety
    /// `ahb` must point to a live `AHardwareBuffer` of this converter's external
    /// format and geometry (hold an acquired reference across the call).
    pub unsafe fn convert(
        &mut self,
        ahb: *const vk::AHardwareBuffer,
    ) -> Result<wgpu::Texture, G2gError> {
        // Cheap handle clone so we can use the device while mutating `self.slots`.
        let d = self.raw.clone();
        let extent = vk::Extent3D { width: self.width, height: self.height, depth: 1 };
        let color_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        // SAFETY: persistent objects were built on `d` in `new`; `ahb` is a live
        // buffer per the contract; each conversion's transient objects are kept on
        // its ring slot and reclaimed (after a fence wait) when the slot is reused;
        // the output image+memory transfer into the returned texture's drop cb.
        unsafe {
            // ---- import the AHB as a sampled image ----
            let mut ext_fmt_img =
                vk::ExternalFormatANDROID::default().external_format(self.external_format);
            let mut ext_mem_img = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::ANDROID_HARDWARE_BUFFER_ANDROID);
            let in_image = d
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::UNDEFINED)
                        .extent(extent)
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(vk::ImageUsageFlags::SAMPLED)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .push_next(&mut ext_mem_img)
                        .push_next(&mut ext_fmt_img),
                    None,
                )
                .map_err(gpu_err)?;

            // Import allocation size / memory type were probed once at build time
            // (constant for this stream's format + geometry).
            let mem_type = first_memory_type(self.import_mem_type_bits)
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            let mut import =
                vk::ImportAndroidHardwareBufferInfoANDROID::default().buffer(ahb as *mut _);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(in_image);
            let in_mem = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(self.allocation_size)
                        .memory_type_index(mem_type)
                        .push_next(&mut dedicated)
                        .push_next(&mut import),
                    None,
                )
                .map_err(gpu_err)?;
            d.bind_image_memory(in_image, in_mem, 0).map_err(gpu_err)?;

            let mut conv_for_view =
                vk::SamplerYcbcrConversionInfo::default().conversion(self.conversion);
            let in_view = d
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(in_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::UNDEFINED)
                        .subresource_range(color_range)
                        .push_next(&mut conv_for_view),
                    None,
                )
                .map_err(gpu_err)?;

            // ---- output RGBA image (handed to wgpu) ----
            let out_image = d
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .extent(extent)
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .tiling(vk::ImageTiling::OPTIMAL)
                        .usage(
                            vk::ImageUsageFlags::STORAGE
                                | vk::ImageUsageFlags::SAMPLED
                                | vk::ImageUsageFlags::TRANSFER_SRC,
                        )
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    None,
                )
                .map_err(gpu_err)?;
            let out_reqs = d.get_image_memory_requirements(out_image);
            let out_mt = find_memory_type(
                &self.mem_props,
                out_reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
            let out_mem = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(out_reqs.size)
                        .memory_type_index(out_mt),
                    None,
                )
                .map_err(gpu_err)?;
            d.bind_image_memory(out_image, out_mem, 0).map_err(gpu_err)?;
            let out_view = d
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(out_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .subresource_range(color_range),
                    None,
                )
                .map_err(gpu_err)?;

            // ---- claim a ring slot, reclaiming its previous conversion ----
            let idx = self.next;
            self.next = (self.next + 1) % RING_DEPTH;
            let (cb, desc_set, fence) = {
                let slot = &mut self.slots[idx];
                if let Some(old) = slot.in_flight.take() {
                    d.wait_for_fences(&[slot.fence], true, u64::MAX).map_err(gpu_err)?;
                    d.destroy_image_view(old.in_view, None);
                    d.destroy_image(old.in_image, None);
                    d.free_memory(old.in_mem, None);
                    d.destroy_image_view(old.out_view, None);
                }
                d.reset_fences(&[slot.fence]).map_err(gpu_err)?;
                (slot.cmd_buf, slot.desc_set, slot.fence)
            };

            // ---- bind + record on this slot ----
            let in_desc = [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(in_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let out_desc = [vk::DescriptorImageInfo::default()
                .image_view(out_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            d.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&in_desc),
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_set)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&out_desc),
                ],
                &[],
            );

            d.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty()).map_err(gpu_err)?;
            d.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(gpu_err)?;

            let to_read = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(in_image)
                .subresource_range(color_range);
            let to_general = vk::ImageMemoryBarrier::default()
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(out_image)
                .subresource_range(color_range);
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read, to_general],
            );
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            d.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            d.cmd_dispatch(cb, self.width.div_ceil(8), self.height.div_ceil(8), 1);
            // Leave the output in SHADER_READ_ONLY_OPTIMAL: a tidy layout for the
            // wgpu consumer (which still re-transitions from UNDEFINED on first use).
            let to_sampled = vk::ImageMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(out_image)
                .subresource_range(color_range);
            d.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_sampled],
            );
            d.end_command_buffer(cb).map_err(gpu_err)?;

            // Submit without waiting: this conversion now runs in flight on `idx`.
            let cbs = [cb];
            d.queue_submit(
                self.queue_raw,
                &[vk::SubmitInfo::default().command_buffers(&cbs)],
                fence,
            )
            .map_err(gpu_err)?;
            self.slots[idx].in_flight = Some(InFlight { in_image, in_mem, in_view, out_view });

            // ---- wrap the output image as a wgpu texture ----
            Ok(self.wrap_rgba_texture(out_image, out_mem))
        }
    }

    /// Wrap a self-allocated RGBA image as a `wgpu::Texture` owning its memory via
    /// a drop callback (the cudawgpu `texture_from_raw` pattern).
    fn wrap_rgba_texture(&self, image: vk::Image, memory: vk::DeviceMemory) -> wgpu::Texture {
        let size = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };
        let raw = self.raw.clone();
        // SAFETY: `image` / `memory` were allocated on this device and are
        // transferred into the drop callback (fired once when wgpu drops the
        // texture, after the GPU is done with it).
        let hal_texture = unsafe {
            let hal_device =
                self.device.as_hal::<wgpu_hal::api::Vulkan>().expect("vulkan wgpu device");
            let drop_cb: wgpu_hal::DropCallback = Box::new(move || {
                raw.destroy_image(image, None);
                raw.free_memory(memory, None);
            });
            hal_device.texture_from_raw(
                image,
                &wgpu_hal::TextureDescriptor {
                    label: Some("mediacodec-rgba"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
                    memory_flags: wgpu_hal::MemoryFlags::empty(),
                    view_formats: Vec::new(),
                },
                Some(drop_cb),
                wgpu_hal::vulkan::TextureMemory::External,
            )
        };
        // SAFETY: `hal_texture` was just produced by this device's hal.
        unsafe {
            self.device.create_texture_from_hal::<wgpu_hal::api::Vulkan>(
                hal_texture,
                &wgpu::TextureDescriptor {
                    label: Some("mediacodec-rgba"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                },
            )
        }
    }
}

impl Drop for YcbcrToRgba {
    fn drop(&mut self) {
        // SAFETY: every handle was created on `self.raw`; we drain the device so
        // no conversion is still reading the in-flight transient objects, then
        // free each slot's resources and the shared pipeline objects.
        unsafe {
            let _ = self.raw.device_wait_idle();
            for slot in &self.slots {
                if let Some(f) = &slot.in_flight {
                    self.raw.destroy_image_view(f.in_view, None);
                    self.raw.destroy_image(f.in_image, None);
                    self.raw.free_memory(f.in_mem, None);
                    self.raw.destroy_image_view(f.out_view, None);
                }
                self.raw.destroy_fence(slot.fence, None);
            }
            self.raw.destroy_command_pool(self.cmd_pool, None);
            self.raw.destroy_descriptor_pool(self.desc_pool, None);
            self.raw.destroy_pipeline(self.pipeline, None);
            self.raw.destroy_shader_module(self.shader, None);
            self.raw.destroy_pipeline_layout(self.pipeline_layout, None);
            self.raw.destroy_descriptor_set_layout(self.dsl, None);
            self.raw.destroy_sampler(self.sampler, None);
            self.raw.destroy_sampler_ycbcr_conversion(self.conversion, None);
        }
    }
}


/// Read a converted [`WgpuRgbaTexture`] back to host memory through wgpu, for
/// on-device validation of the zero-copy output path: a normal
/// `copy_texture_to_buffer` + map on the texture's own device proves wgpu can
/// consume the externally-written texture (and that its content survived the
/// adoption). Returns `width * height * 4` bytes (`Rgba8Unorm`), de-padded from
/// the 256-byte row alignment wgpu requires.
pub fn readback_rgba_texture(owner: &WgpuRgbaTexture) -> Result<Vec<u8>, G2gError> {
    let device = owner.device();
    let queue = owner.queue();
    let texture = owner.texture();
    let w = texture.width();
    let h = texture.height();
    let unpadded = w * 4;
    let padded = unpadded.div_ceil(256) * 256;

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rgba-readback"),
        size: (padded as u64) * (h as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rgba-readback") });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit([enc.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .map_err(gpu_err)?;
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded as usize) * (h as usize));
    for row in 0..h as usize {
        let start = row * padded as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    buffer.unmap();
    Ok(out)
}
