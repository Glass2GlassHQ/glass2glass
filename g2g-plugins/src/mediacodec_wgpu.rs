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
