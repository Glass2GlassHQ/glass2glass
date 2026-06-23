//! CUDA<->wgpu zero-copy interop (M220, the GPU keep-on-GPU pillar).
//!
//! Joins the NVDEC decode side ([`MemoryDomain::Cuda`](g2g_core::MemoryDomain),
//! NV12 in CUDA device memory) to the wgpu preprocess / inference side
//! ([`MemoryDomain::WgpuTexture`], consumed by `WgpuPreprocess`'s M217
//! surface-import path), so decode -> preprocess -> infer stay on the GPU with
//! no PCIe round-trip.
//!
//! ## Mechanism
//!
//! There is no portable "share this CUDA pointer with wgpu" call; the bridge is
//! built on Vulkan external memory (`VK_KHR_external_memory_fd`), which both
//! sides speak:
//!
//! 1. Create a wgpu device on the Vulkan backend with `VK_KHR_external_memory_fd`
//!    enabled. wgpu's safe `request_device` can't add device extensions, so we
//!    drop to wgpu-hal's `open_with_callback` and append the extension in the
//!    callback (wgpu-hal still builds the full feature / queue chain).
//! 2. Allocate an exportable `VkImage` + `VkDeviceMemory` ourselves via ash
//!    (R8Uint, `width x (height + height/2)`, the packed-NV12 layout M217
//!    samples), export the memory as an opaque FD, and wrap the image as a
//!    `wgpu::Texture` via `texture_from_raw` with `TextureMemory::External` (we
//!    own the memory, shared with CUDA).
//! 3. CUDA imports that FD (`cuImportExternalMemory`), maps it as a CUDA array,
//!    and copies the NVDEC NV12 planes device->device into it (`cuMemcpy2D`,
//!    the same copy `CudaGlSink` does into a GL texture).
//!
//! The wgpu device created here is carried on the output frame's keep-alive, so
//! `WgpuPreprocess` adopts it (the M217 device-identity pattern) and samples the
//! shared image directly.
//!
//! Linux + NVIDIA only (`cuda-wgpu` feature). This module is COMPILE/RUN-pending
//! incrementally on the RTX 3060 host; the spike below proves the Vulkan-side
//! external-memory plumbing before the CUDA import and the full element land.

use alloc::boxed::Box;
use ash::vk;

use g2g_core::{G2gError, HardwareError};

/// Map any wgpu request/poll error to a structured hardware failure.
fn gpu_err<E>(_e: E) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// A wgpu device on the Vulkan backend with `VK_KHR_external_memory_fd` enabled,
/// plus the instance / adapter it was opened from (kept alive alongside it).
#[derive(Debug)]
pub struct InteropDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    // Order matters for Drop: device before adapter before instance.
    _adapter: wgpu::Adapter,
    _instance: wgpu::Instance,
}

/// Create a wgpu device that can import / export external memory by FD.
///
/// Forces the Vulkan backend (the only one with an FD external-memory path on
/// Linux) and enables `VK_KHR_external_memory_fd` via wgpu-hal's create-device
/// callback. Fails loud if the adapter is not Vulkan or the extension is absent.
pub async fn create_interop_device() -> Result<InteropDevice, G2gError> {
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

    // Open via the hal escape hatch so we can add the FD external-memory device
    // extension. wgpu-hal fills in the rest of the feature / queue chain.
    let features = wgpu::Features::empty();
    let limits = wgpu::Limits::default();
    let memory_hints = wgpu::MemoryHints::default();

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
                args.extensions.push(ash::khr::external_memory_fd::NAME);
            })),
        )
    }
    .map_err(gpu_err)?;

    // SAFETY: `open` was produced by this adapter's hal, as required.
    let (device, queue) = unsafe {
        adapter.create_device_from_hal(
            open,
            &wgpu::DeviceDescriptor {
                label: Some("cuda-wgpu-interop"),
                required_features: features,
                required_limits: limits,
                ..Default::default()
            },
        )
    }
    .map_err(gpu_err)?;

    Ok(InteropDevice { device, queue, _adapter: adapter, _instance: instance })
}

/// The packed-NV12 texture geometry M217 samples: one R8Uint plane holding the
/// Y rows then the interleaved CbCr rows.
fn nv12_texture_height(height: u32) -> u32 {
    height + height / 2
}

/// Pick a memory type index satisfying `type_bits` with the requested property
/// flags (here `DEVICE_LOCAL`).
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

/// A self-allocated, FD-exportable Vulkan image holding a packed-NV12 frame,
/// shared with CUDA. The raw handles are owned here (not by wgpu), so Drop frees
/// them; the exported FD is owned by CUDA after import.
#[derive(Debug)]
pub struct SharedNv12Image {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    /// Opaque FD of the backing memory, for `cuImportExternalMemory`.
    pub fd: i32,
    /// Total bytes the allocation spans (the import descriptor needs it).
    pub size: u64,
    pub width: u32,
    pub height: u32,
}

/// Allocate an exportable R8Uint NV12 image on `device`'s Vulkan device and
/// export its memory as an opaque FD.
///
/// # Safety
/// `device` must be a Vulkan-backend wgpu device with `VK_KHR_external_memory_fd`
/// enabled (use [`create_interop_device`]). The returned handles are valid until
/// freed via [`SharedNv12Image::destroy`].
pub unsafe fn export_nv12_image(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> Result<SharedNv12Image, G2gError> {
    let tex_h = nv12_texture_height(height);

    // SAFETY: caller guarantees a Vulkan device; we hold the hal guard for the
    // whole allocation and never retain raw handles past the ash device.
    unsafe {
        let hal_device = device
            .as_hal::<wgpu_hal::api::Vulkan>()
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let raw: &ash::Device = hal_device.raw_device();
        let phys = hal_device.raw_physical_device();
        let instance: &ash::Instance = hal_device.shared_instance().raw_instance();

        let mut ext_img = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8_UINT)
            .extent(vk::Extent3D { width, height: tex_h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            // SAMPLED: WgpuPreprocess's textureLoad. TRANSFER_SRC: lets wgpu copy
            // the image out (the spike's readback verification). TRANSFER_DST:
            // valid-usage completeness.
            .usage(
                vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut ext_img);
        let image = raw.create_image(&image_info, None).map_err(gpu_err)?;

        let reqs = raw.get_image_memory_requirements(image);
        let props = instance.get_physical_device_memory_properties(phys);
        let Some(mem_type) =
            find_memory_type(&props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        else {
            raw.destroy_image(image, None);
            return Err(G2gError::Hardware(HardwareError::Other));
        };

        // Export the FD, and use a dedicated allocation: CUDA's import of a
        // Vulkan image requires the memory be dedicated to that image.
        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(mem_type)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        let memory = match raw.allocate_memory(&alloc_info, None) {
            Ok(m) => m,
            Err(e) => {
                raw.destroy_image(image, None);
                return Err(gpu_err(e));
            }
        };
        if let Err(e) = raw.bind_image_memory(image, memory, 0) {
            raw.free_memory(memory, None);
            raw.destroy_image(image, None);
            return Err(gpu_err(e));
        }

        let ext_fd = ash::khr::external_memory_fd::Device::new(instance, raw);
        let fd = match ext_fd.get_memory_fd(
            &vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
        ) {
            Ok(fd) => fd,
            Err(e) => {
                raw.free_memory(memory, None);
                raw.destroy_image(image, None);
                return Err(gpu_err(e));
            }
        };

        Ok(SharedNv12Image { image, memory, fd, size: reqs.size, width, height })
    }
}

impl SharedNv12Image {
    /// Free the Vulkan image and memory. The FD is owned by CUDA after import;
    /// if it was never imported, the caller closes it.
    ///
    /// # Safety
    /// `device` must be the same Vulkan wgpu device the image was created on, and
    /// the image must no longer be in use by the GPU.
    pub unsafe fn destroy(self, device: &wgpu::Device) {
        // SAFETY: per the contract, same device and no in-flight use.
        unsafe {
            if let Some(hal_device) = device.as_hal::<wgpu_hal::api::Vulkan>() {
                let raw = hal_device.raw_device();
                raw.free_memory(self.memory, None);
                raw.destroy_image(self.image, None);
            }
        }
    }
}

/// Wrap the shared image as a `wgpu::Texture` on `device` (which must be the
/// interop device the image was allocated on). wgpu does not own the backing
/// memory (`TextureMemory::External`); the image and its memory are freed by the
/// texture's drop callback when wgpu drops the texture. The result is a normal
/// R8Uint sampled texture that `WgpuPreprocess`'s M217 path consumes.
///
/// # Safety
/// `device` must be the Vulkan wgpu device `shared` was created on. Consumes
/// `shared`: its image and memory must not be freed by any other path.
pub unsafe fn wrap_as_texture(device: &wgpu::Device, shared: SharedNv12Image) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: shared.width,
        height: nv12_texture_height(shared.height),
        depth_or_array_layers: 1,
    };
    let image = shared.image;
    let memory = shared.memory;

    // SAFETY: `device` is the interop Vulkan device; `image` / `memory` are
    // valid and owned here, transferred into the drop callback (fired once when
    // wgpu drops the texture, after the GPU is done with it).
    let hal_texture = unsafe {
        let hal_device = device.as_hal::<wgpu_hal::api::Vulkan>().expect("vulkan wgpu device");
        let raw = hal_device.raw_device().clone();
        // SAFETY (closure): fired once when wgpu drops the texture, when the
        // image is idle; covered by the enclosing unsafe block.
        let drop_cb: wgpu_hal::DropCallback = Box::new(move || {
            raw.destroy_image(image, None);
            raw.free_memory(memory, None);
        });
        let hal_desc = wgpu_hal::TextureDescriptor {
            label: Some("cuda-nv12"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
            memory_flags: wgpu_hal::MemoryFlags::empty(),
            view_formats: alloc::vec::Vec::new(),
        };
        hal_device.texture_from_raw(
            image,
            &hal_desc,
            Some(drop_cb),
            wgpu_hal::vulkan::TextureMemory::External,
        )
    };

    let wgpu_desc = wgpu::TextureDescriptor {
        label: Some("cuda-nv12"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };
    // SAFETY: `hal_texture` was just produced by this device's hal.
    unsafe { device.create_texture_from_hal::<wgpu_hal::api::Vulkan>(hal_texture, &wgpu_desc) }
}

/// Import the shared image's memory into CUDA as a CUDA array, write a test NV12
/// pattern into it, read it back, and confirm it round-trips. Proves the
/// Vulkan<->CUDA external-memory sharing works on this driver, with no wgpu
/// involvement. Consumes `shared.fd` (CUDA takes ownership of an imported FD).
///
/// Returns `Ok(true)` when the bytes read back match what was written.
///
/// # Safety
/// `shared` must come from [`export_nv12_image`] and not yet have been imported.
pub unsafe fn cuda_roundtrip_check(shared: &SharedNv12Image) -> Result<bool, G2gError> {
    use cuda_ffi as c;

    let w = shared.width as usize;
    let tex_h = nv12_texture_height(shared.height) as usize;

    // SAFETY: all calls follow the CUDA Driver API contract; handles are checked
    // before use and destroyed before return.
    unsafe {
        check(c::cuInit(0))?;
        let mut dev: i32 = 0;
        check(c::cuDeviceGet(&mut dev, 0))?;
        let mut ctx: c::CuContext = core::ptr::null_mut();
        check(c::cuDevicePrimaryCtxRetain(&mut ctx, dev))?;
        check(c::cuCtxPushCurrent(ctx))?;

        let import_desc = c::CudaExternalMemoryHandleDesc {
            type_: c::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            _pad: 0,
            handle_fd: shared.fd,
            _handle_rest: [0; 12],
            size: shared.size,
            // The Vulkan allocation was dedicated to the image, so CUDA must be
            // told so or the import fails.
            flags: c::CUDA_EXTERNAL_MEMORY_DEDICATED,
            reserved: [0; 16],
        };
        let mut ext_mem: c::CuExternalMemory = core::ptr::null_mut();
        check(c::cuImportExternalMemory(&mut ext_mem, &import_desc))?;

        let array_desc = c::CudaArray3dDescriptor {
            width: w,
            height: tex_h,
            depth: 0,
            format: c::CU_AD_FORMAT_UNSIGNED_INT8,
            num_channels: 1,
            flags: 0,
        };
        let mipmap_desc = c::CudaExternalMemoryMipmappedArrayDesc {
            offset: 0,
            array_desc,
            num_levels: 1,
            reserved: [0; 16],
        };
        let mut mipmap: c::CuMipmappedArray = core::ptr::null_mut();
        let mut array: c::CuArray = core::ptr::null_mut();
        let mut ok = false;
        let mut result =
            check(c::cuExternalMemoryGetMappedMipmappedArray(&mut mipmap, ext_mem, &mipmap_desc));
        if result.is_ok() {
            result = check(c::cuMipmappedArrayGetLevel(&mut array, mipmap, 0));
        }
        if result.is_ok() {
            // A recognizable pattern: byte (x ^ y) per texel.
            let mut src = alloc::vec![0u8; w * tex_h];
            for y in 0..tex_h {
                for x in 0..w {
                    src[y * w + x] = (x ^ y) as u8;
                }
            }
            let to_array = c::CudaMemcpy2D {
                src_x_in_bytes: 0,
                src_y: 0,
                src_memory_type: c::CU_MEMORYTYPE_HOST,
                src_host: src.as_ptr().cast(),
                src_device: 0,
                src_array: core::ptr::null_mut(),
                src_pitch: w,
                dst_x_in_bytes: 0,
                dst_y: 0,
                dst_memory_type: c::CU_MEMORYTYPE_ARRAY,
                dst_host: core::ptr::null_mut(),
                dst_device: 0,
                dst_array: array,
                dst_pitch: 0,
                width_in_bytes: w,
                height: tex_h,
            };
            result = check(c::cu_memcpy_2d(&to_array));
            if result.is_ok() {
                result = check(c::cuCtxSynchronize());
            }
            if result.is_ok() {
                let mut back = alloc::vec![0u8; w * tex_h];
                let from_array = c::CudaMemcpy2D {
                    src_x_in_bytes: 0,
                    src_y: 0,
                    src_memory_type: c::CU_MEMORYTYPE_ARRAY,
                    src_host: core::ptr::null(),
                    src_device: 0,
                    src_array: array,
                    src_pitch: 0,
                    dst_x_in_bytes: 0,
                    dst_y: 0,
                    dst_memory_type: c::CU_MEMORYTYPE_HOST,
                    dst_host: back.as_mut_ptr().cast(),
                    dst_device: 0,
                    dst_array: core::ptr::null_mut(),
                    dst_pitch: w,
                    width_in_bytes: w,
                    height: tex_h,
                };
                result = check(c::cu_memcpy_2d(&from_array));
                if result.is_ok() {
                    let _ = check(c::cuCtxSynchronize());
                    ok = back == src;
                }
            }
        }

        if !mipmap.is_null() {
            c::cuMipmappedArrayDestroy(mipmap);
        }
        c::cuDestroyExternalMemory(ext_mem);
        let mut popped: c::CuContext = core::ptr::null_mut();
        let _ = c::cuCtxPopCurrent(&mut popped);
        c::cuDevicePrimaryCtxRelease(dev);

        result?;
        Ok(ok)
    }
}

/// Import the shared image into CUDA and fill it with the `(x ^ y)` test
/// pattern (no readback). Used by the spike that proves wgpu samples what CUDA
/// wrote. Consumes `shared.fd` (CUDA owns an imported FD).
///
/// # Safety
/// `shared` must come from [`export_nv12_image`] and not yet have been imported.
pub unsafe fn cuda_fill_xor_pattern(shared: &SharedNv12Image) -> Result<(), G2gError> {
    use cuda_ffi as c;
    let w = shared.width as usize;
    let tex_h = nv12_texture_height(shared.height) as usize;

    // SAFETY: standard CUDA Driver API sequence; handles destroyed before return.
    unsafe {
        check(c::cuInit(0))?;
        let mut dev: i32 = 0;
        check(c::cuDeviceGet(&mut dev, 0))?;
        let mut ctx: c::CuContext = core::ptr::null_mut();
        check(c::cuDevicePrimaryCtxRetain(&mut ctx, dev))?;
        check(c::cuCtxPushCurrent(ctx))?;

        let import_desc = c::CudaExternalMemoryHandleDesc {
            type_: c::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            _pad: 0,
            handle_fd: shared.fd,
            _handle_rest: [0; 12],
            size: shared.size,
            flags: c::CUDA_EXTERNAL_MEMORY_DEDICATED,
            reserved: [0; 16],
        };
        let mut ext_mem: c::CuExternalMemory = core::ptr::null_mut();
        check(c::cuImportExternalMemory(&mut ext_mem, &import_desc))?;

        let mipmap_desc = c::CudaExternalMemoryMipmappedArrayDesc {
            offset: 0,
            array_desc: c::CudaArray3dDescriptor {
                width: w,
                height: tex_h,
                depth: 0,
                format: c::CU_AD_FORMAT_UNSIGNED_INT8,
                num_channels: 1,
                flags: 0,
            },
            num_levels: 1,
            reserved: [0; 16],
        };
        let mut mipmap: c::CuMipmappedArray = core::ptr::null_mut();
        let mut array: c::CuArray = core::ptr::null_mut();
        let mut result =
            check(c::cuExternalMemoryGetMappedMipmappedArray(&mut mipmap, ext_mem, &mipmap_desc));
        if result.is_ok() {
            result = check(c::cuMipmappedArrayGetLevel(&mut array, mipmap, 0));
        }
        if result.is_ok() {
            let mut src = alloc::vec![0u8; w * tex_h];
            for y in 0..tex_h {
                for x in 0..w {
                    src[y * w + x] = (x ^ y) as u8;
                }
            }
            let to_array = c::CudaMemcpy2D {
                src_x_in_bytes: 0,
                src_y: 0,
                src_memory_type: c::CU_MEMORYTYPE_HOST,
                src_host: src.as_ptr().cast(),
                src_device: 0,
                src_array: core::ptr::null_mut(),
                src_pitch: w,
                dst_x_in_bytes: 0,
                dst_y: 0,
                dst_memory_type: c::CU_MEMORYTYPE_ARRAY,
                dst_host: core::ptr::null_mut(),
                dst_device: 0,
                dst_array: array,
                dst_pitch: 0,
                width_in_bytes: w,
                height: tex_h,
            };
            result = check(c::cu_memcpy_2d(&to_array));
            if result.is_ok() {
                result = check(c::cuCtxSynchronize());
            }
        }

        if !mipmap.is_null() {
            c::cuMipmappedArrayDestroy(mipmap);
        }
        c::cuDestroyExternalMemory(ext_mem);
        let mut popped: c::CuContext = core::ptr::null_mut();
        let _ = c::cuCtxPopCurrent(&mut popped);
        c::cuDevicePrimaryCtxRelease(dev);
        result
    }
}

/// Copy a decoded NV12 frame's two planes from CUDA device memory into the
/// shared image's CUDA array, device->device (no PCIe). The Y plane fills rows
/// `0..height`, the interleaved CbCr plane rows `height..height + height/2`,
/// matching the packed-NV12 layout `WgpuPreprocess` samples.
///
/// Runs in `context` (the decoder's `CUcontext`, where the plane pointers are
/// valid), imports the shared FD there, maps the array, copies, synchronizes,
/// and tears down the CUDA import (the Vulkan allocation persists). Consumes
/// `shared.fd`.
///
/// # Safety
/// `shared` must come from [`export_nv12_image`] (matching `width`/`height`) and
/// not yet be imported. The plane pointers / pitches must describe valid NV12
/// device memory in `context`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cuda_copy_nv12_planes(
    shared: &SharedNv12Image,
    context: u64,
    luma_ptr: u64,
    luma_pitch: u32,
    chroma_ptr: u64,
    chroma_pitch: u32,
    width: u32,
    height: u32,
) -> Result<(), G2gError> {
    use cuda_ffi as c;
    let w = width as usize;
    let h = height as usize;

    // SAFETY: CUDA Driver API sequence in the decoder's context; the import and
    // mapping are destroyed before return.
    unsafe {
        check(c::cuInit(0))?;
        check(c::cuCtxPushCurrent(context as c::CuContext))?;

        let import_desc = c::CudaExternalMemoryHandleDesc {
            type_: c::CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
            _pad: 0,
            handle_fd: shared.fd,
            _handle_rest: [0; 12],
            size: shared.size,
            flags: c::CUDA_EXTERNAL_MEMORY_DEDICATED,
            reserved: [0; 16],
        };
        let mut ext_mem: c::CuExternalMemory = core::ptr::null_mut();
        let mut result = check(c::cuImportExternalMemory(&mut ext_mem, &import_desc));

        let tex_h = nv12_texture_height(height) as usize;
        let mipmap_desc = c::CudaExternalMemoryMipmappedArrayDesc {
            offset: 0,
            array_desc: c::CudaArray3dDescriptor {
                width: w,
                height: tex_h,
                depth: 0,
                format: c::CU_AD_FORMAT_UNSIGNED_INT8,
                num_channels: 1,
                flags: 0,
            },
            num_levels: 1,
            reserved: [0; 16],
        };
        let mut mipmap: c::CuMipmappedArray = core::ptr::null_mut();
        let mut array: c::CuArray = core::ptr::null_mut();
        if result.is_ok() {
            result =
                check(c::cuExternalMemoryGetMappedMipmappedArray(&mut mipmap, ext_mem, &mipmap_desc));
        }
        if result.is_ok() {
            result = check(c::cuMipmappedArrayGetLevel(&mut array, mipmap, 0));
        }
        if result.is_ok() {
            // Y plane -> array rows 0..h.
            let luma = c::CudaMemcpy2D {
                src_x_in_bytes: 0,
                src_y: 0,
                src_memory_type: c::CU_MEMORYTYPE_DEVICE,
                src_host: core::ptr::null(),
                src_device: luma_ptr,
                src_array: core::ptr::null_mut(),
                src_pitch: luma_pitch as usize,
                dst_x_in_bytes: 0,
                dst_y: 0,
                dst_memory_type: c::CU_MEMORYTYPE_ARRAY,
                dst_host: core::ptr::null_mut(),
                dst_device: 0,
                dst_array: array,
                dst_pitch: 0,
                width_in_bytes: w,
                height: h,
            };
            result = check(c::cu_memcpy_2d(&luma));
            if result.is_ok() {
                // Interleaved CbCr plane (w bytes/row, h/2 rows) -> array rows h..
                let chroma = c::CudaMemcpy2D {
                    src_x_in_bytes: 0,
                    src_y: 0,
                    src_memory_type: c::CU_MEMORYTYPE_DEVICE,
                    src_host: core::ptr::null(),
                    src_device: chroma_ptr,
                    src_array: core::ptr::null_mut(),
                    src_pitch: chroma_pitch as usize,
                    dst_x_in_bytes: 0,
                    dst_y: h,
                    dst_memory_type: c::CU_MEMORYTYPE_ARRAY,
                    dst_host: core::ptr::null_mut(),
                    dst_device: 0,
                    dst_array: array,
                    dst_pitch: 0,
                    width_in_bytes: w,
                    height: h / 2,
                };
                result = check(c::cu_memcpy_2d(&chroma));
            }
            if result.is_ok() {
                result = check(c::cuCtxSynchronize());
            }
        }

        if !mipmap.is_null() {
            c::cuMipmappedArrayDestroy(mipmap);
        }
        if !ext_mem.is_null() {
            c::cuDestroyExternalMemory(ext_mem);
        }
        let mut popped: c::CuContext = core::ptr::null_mut();
        let _ = c::cuCtxPopCurrent(&mut popped);
        result
    }
}

/// Map a `CUresult` to a `Result`, carrying the raw code on failure.
fn check(code: i32) -> Result<(), G2gError> {
    if code == 0 {
        Ok(())
    } else {
        Err(G2gError::Hardware(HardwareError::Cuda(code)))
    }
}

/// CUDA Driver API external-memory FFI for the interop bridge. Extends the
/// surface in [`crate::cuda`] (which has the download path) with the
/// import-by-FD + mapped-array calls. Struct layouts are field-for-field from
/// `cuda.h` (64-bit); the `_v2` suffixed names match the symbols `libcuda`
/// exports for the `#define`d unsuffixed entry points.
#[allow(non_snake_case)]
mod cuda_ffi {
    use core::ffi::c_void;

    pub type CuContext = *mut c_void;
    pub type CuExternalMemory = *mut c_void;
    pub type CuMipmappedArray = *mut c_void;
    pub type CuArray = *mut c_void;

    pub const CU_MEMORYTYPE_HOST: u32 = 0x01;
    pub const CU_MEMORYTYPE_DEVICE: u32 = 0x02;
    pub const CU_MEMORYTYPE_ARRAY: u32 = 0x03;
    pub const CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD: u32 = 1;
    pub const CUDA_EXTERNAL_MEMORY_DEDICATED: u32 = 0x1;
    pub const CU_AD_FORMAT_UNSIGNED_INT8: u32 = 0x01;

    /// `CUDA_MEMCPY2D` (a.k.a. `_v2`), field-for-field from `cuda.h`. Mirrors the
    /// copy of this struct in [`crate::cuda`]; redeclared here so the interop
    /// module is self-contained (the original is private to that module).
    #[repr(C)]
    pub struct CudaMemcpy2D {
        pub src_x_in_bytes: usize,
        pub src_y: usize,
        pub src_memory_type: u32,
        pub src_host: *const c_void,
        pub src_device: u64,
        pub src_array: *mut c_void,
        pub src_pitch: usize,
        pub dst_x_in_bytes: usize,
        pub dst_y: usize,
        pub dst_memory_type: u32,
        pub dst_host: *mut c_void,
        pub dst_device: u64,
        pub dst_array: *mut c_void,
        pub dst_pitch: usize,
        pub width_in_bytes: usize,
        pub height: usize,
    }

    /// `CUDA_EXTERNAL_MEMORY_HANDLE_DESC`. The 16-byte `handle` union is modelled
    /// as the leading `int fd` plus 12 padding bytes; `_pad` aligns the union to
    /// 8 (it contains pointers).
    #[repr(C)]
    pub struct CudaExternalMemoryHandleDesc {
        pub type_: u32,
        pub _pad: u32,
        pub handle_fd: i32,
        pub _handle_rest: [u8; 12],
        pub size: u64,
        pub flags: u32,
        pub reserved: [u32; 16],
    }

    /// `CUDA_ARRAY3D_DESCRIPTOR`.
    #[repr(C)]
    pub struct CudaArray3dDescriptor {
        pub width: usize,
        pub height: usize,
        pub depth: usize,
        pub format: u32,
        pub num_channels: u32,
        pub flags: u32,
    }

    /// `CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC`.
    #[repr(C)]
    pub struct CudaExternalMemoryMipmappedArrayDesc {
        pub offset: u64,
        pub array_desc: CudaArray3dDescriptor,
        pub num_levels: u32,
        pub reserved: [u32; 16],
    }

    #[link(name = "cuda")]
    extern "C" {
        pub fn cuInit(flags: u32) -> i32;
        pub fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
        #[link_name = "cuDevicePrimaryCtxRetain"]
        pub fn cuDevicePrimaryCtxRetain(pctx: *mut CuContext, dev: i32) -> i32;
        #[link_name = "cuDevicePrimaryCtxRelease_v2"]
        pub fn cuDevicePrimaryCtxRelease(dev: i32) -> i32;
        #[link_name = "cuCtxPushCurrent_v2"]
        pub fn cuCtxPushCurrent(ctx: CuContext) -> i32;
        #[link_name = "cuCtxPopCurrent_v2"]
        pub fn cuCtxPopCurrent(pctx: *mut CuContext) -> i32;
        pub fn cuCtxSynchronize() -> i32;
        pub fn cuImportExternalMemory(
            ext_mem_out: *mut CuExternalMemory,
            mem_handle_desc: *const CudaExternalMemoryHandleDesc,
        ) -> i32;
        pub fn cuExternalMemoryGetMappedMipmappedArray(
            mipmap: *mut CuMipmappedArray,
            ext_mem: CuExternalMemory,
            mipmap_desc: *const CudaExternalMemoryMipmappedArrayDesc,
        ) -> i32;
        pub fn cuMipmappedArrayGetLevel(
            level_array: *mut CuArray,
            mipmap: CuMipmappedArray,
            level: u32,
        ) -> i32;
        pub fn cuMipmappedArrayDestroy(mipmap: CuMipmappedArray) -> i32;
        pub fn cuDestroyExternalMemory(ext_mem: CuExternalMemory) -> i32;
        #[link_name = "cuMemcpy2D_v2"]
        pub fn cu_memcpy_2d(pcopy: *const CudaMemcpy2D) -> i32;
    }
}
