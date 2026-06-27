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
//! Linux + NVIDIA only (`cuda-wgpu` feature). Validated on an RTX 3060 (M251):
//! the `cudawgpu_spike` tests prove the Vulkan<->CUDA external-memory round-trip
//! in both directions (CUDA writes / wgpu reads back, and the reverse, 0 byte
//! mismatches), and the cross-crate `cuda_wgpu_e2e` test runs the full
//! NVDEC -> `CudaToWgpu` -> `WgpuPreprocess` -> inference chain against a CPU
//! reference (all frames match, no PCIe download).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use ash::vk;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    MemoryDomain, OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

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
    // Order matters for Drop: device before adapter before instance. Public so a
    // host renderer (the Bevy demo's `RenderCreation::Manual`) can clone all four
    // handles to adopt this device; the field order still governs drop order.
    pub adapter: wgpu::Adapter,
    pub instance: wgpu::Instance,
}

/// Create a wgpu device that can import / export external memory by FD.
///
/// Forces the Vulkan backend (the only one with an FD external-memory path on
/// Linux) and enables `VK_KHR_external_memory_fd` via wgpu-hal's create-device
/// callback. Fails loud if the adapter is not Vulkan or the extension is absent.
pub async fn create_interop_device() -> Result<InteropDevice, G2gError> {
    create_interop_device_inner(false).await
}

/// Like [`create_interop_device`], but opens the device with the adapter's full
/// feature set and limits instead of the minimal default. Use this when the
/// interop device is also driving a full renderer (e.g. handed to a Bevy app via
/// `RenderCreation::Manual` for the M278 zero-copy render -> NVENC path): a render
/// engine needs more than the bridge's bare NV12 / RGBA copy path, and a device
/// opened with `Features::empty()` would fail its pipeline setup. The CUDA interop
/// itself is unaffected (more features never hurt the copy).
pub async fn create_interop_device_full() -> Result<InteropDevice, G2gError> {
    create_interop_device_inner(true).await
}

/// Shared body: `full` requests the adapter's whole feature set + limits (for a
/// renderer driving this device), else the minimal default (the bridge's own use).
async fn create_interop_device_inner(full: bool) -> Result<InteropDevice, G2gError> {
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
    let (features, limits) = if full {
        (adapter.features(), adapter.limits())
    } else {
        (wgpu::Features::empty(), wgpu::Limits::default())
    };
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

    Ok(InteropDevice { device, queue, adapter, instance })
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

/// A persistent CUDA import of a [`SharedNv12Image`], mapped as a CUDA array
/// ready for `cuMemcpy2D`. Created once per pooled image and reused across frames
/// via [`cuda_copy_planes_into`], so the per-frame `cuImportExternalMemory` + map
/// + teardown a non-pooled import would pay is amortized.
///
/// Handles are stored as integers (not raw pointers), so the mapping is `Send`,
/// the same contract as [`g2g_core::memory::OwnedCudaBuffer`]: the CUDA context
/// is thread-floating and pushed current per use, so the handles carry no thread
/// affinity. Dropping it destroys the import in its context.
#[derive(Debug)]
pub struct CudaImageMapping {
    ext_mem: usize,
    mipmap: usize,
    array: usize,
    context: u64,
}

impl Drop for CudaImageMapping {
    fn drop(&mut self) {
        use cuda_ffi as c;
        if self.ext_mem == 0 {
            return;
        }
        // SAFETY: the handles came from `import_image_into_cuda` in `context`; we
        // push it current for the destroy and pop after. Only reached once (the
        // mapping is owned, not copied).
        unsafe {
            if c::cuCtxPushCurrent(self.context as c::CuContext) == 0 {
                if self.mipmap != 0 {
                    c::cuMipmappedArrayDestroy(self.mipmap as c::CuMipmappedArray);
                }
                c::cuDestroyExternalMemory(self.ext_mem as c::CuExternalMemory);
                let mut popped: c::CuContext = core::ptr::null_mut();
                let _ = c::cuCtxPopCurrent(&mut popped);
            }
        }
    }
}

/// Import `shared` into CUDA once and map it as a CUDA array, in `context` (the
/// decoder's `CUcontext`). Consumes `shared.fd`. The returned mapping persists
/// until dropped; frames copy into the same array with [`cuda_copy_planes_into`]
/// without re-importing.
///
/// # Safety
/// `shared` must come from [`export_nv12_image`] and not yet be imported; the
/// plane copies later issued against the mapping must target `context`.
pub unsafe fn import_image_into_cuda(
    shared: &SharedNv12Image,
    context: u64,
) -> Result<CudaImageMapping, G2gError> {
    use cuda_ffi as c;
    let w = shared.width as usize;
    let tex_h = nv12_texture_height(shared.height) as usize;

    // SAFETY: CUDA Driver API import sequence in `context`. On error every handle
    // created so far is destroyed before the context is popped, so nothing leaks.
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

        if let Err(e) = result {
            // Destroy whatever was created, while still current.
            if !mipmap.is_null() {
                c::cuMipmappedArrayDestroy(mipmap);
            }
            if !ext_mem.is_null() {
                c::cuDestroyExternalMemory(ext_mem);
            }
            let mut popped: c::CuContext = core::ptr::null_mut();
            let _ = c::cuCtxPopCurrent(&mut popped);
            return Err(e);
        }

        let mut popped: c::CuContext = core::ptr::null_mut();
        let _ = c::cuCtxPopCurrent(&mut popped);
        Ok(CudaImageMapping {
            ext_mem: ext_mem as usize,
            mipmap: mipmap as usize,
            array: array as usize,
            context,
        })
    }
}

/// Copy a decoded NV12 frame's two planes device->device into a persistently
/// imported [`CudaImageMapping`]'s array (the pooled-reuse counterpart of a
/// per-frame import path, which would import + tear down every call). The Y plane
/// fills array rows `0..height`, the interleaved CbCr rows `height..`.
///
/// # Safety
/// `mapping` must come from [`import_image_into_cuda`] (matching `width`/`height`)
/// and the plane pointers / pitches must describe valid NV12 device memory in
/// `mapping`'s context.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cuda_copy_planes_into(
    mapping: &CudaImageMapping,
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
    let array = mapping.array as c::CuArray;

    // SAFETY: `array` is the persistent mapped array from `import_image_into_cuda`
    // in `mapping.context`; the plane pointers are valid device NV12 there. The
    // context is pushed for the copies and popped before return.
    unsafe {
        check(c::cuCtxPushCurrent(mapping.context as c::CuContext))?;
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
        let mut result = check(c::cu_memcpy_2d(&luma));
        if result.is_ok() {
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
        let mut popped: c::CuContext = core::ptr::null_mut();
        let _ = c::cuCtxPopCurrent(&mut popped);
        result
    }
}

/// One pooled shared image: the `wgpu::Texture` (whose drop frees the backing
/// Vulkan image/memory) and the persistent CUDA import that writes into it. Field
/// order matters: `mapping` is declared first so its Drop (destroy the CUDA
/// import) runs before `texture`'s drop frees the Vulkan memory the import aliased.
#[derive(Debug)]
pub struct PoolEntry {
    mapping: CudaImageMapping,
    texture: wgpu::Texture,
}

impl PoolEntry {
    /// The backing texture, to clone for a downstream frame.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Copy this frame's NV12 planes into the entry's persistent CUDA array.
    ///
    /// # Safety
    /// The plane pointers / pitches must describe valid NV12 device memory in the
    /// entry's CUDA context (see [`cuda_copy_planes_into`]).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn copy_planes(
        &self,
        luma_ptr: u64,
        luma_pitch: u32,
        chroma_ptr: u64,
        chroma_pitch: u32,
        width: u32,
        height: u32,
    ) -> Result<(), G2gError> {
        // SAFETY: forwarded to the documented contract of `cuda_copy_planes_into`.
        unsafe {
            cuda_copy_planes_into(&self.mapping, luma_ptr, luma_pitch, chroma_ptr, chroma_pitch, width, height)
        }
    }
}

/// A reuse pool of shared CUDA<->wgpu NV12 images, keyed externally by geometry
/// (the owner rebuilds the pool on a size change). It is just a free list:
/// [`take_free`](Self::take_free) pops a recycled entry (or `None` to build a
/// fresh one), and [`in_flight`](Self::in_flight) wraps an entry so it returns to
/// the free list when the downstream frame is released. v1 allocated + imported a
/// fresh image per frame; this amortizes both across frames.
///
/// Cross-API safety: a recycled entry's image may still be sampled by an
/// in-flight wgpu submission, so the owner must drain the device
/// (`Device::poll(Wait)`) before overwriting a `take_free` entry.
#[derive(Debug, Clone, Default)]
pub struct CudaWgpuPool {
    free: alloc::sync::Arc<std::sync::Mutex<alloc::vec::Vec<PoolEntry>>>,
}

impl CudaWgpuPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a fresh pooled entry: allocate an exportable Vulkan NV12 image, import
    /// it into CUDA once, and wrap it as a `wgpu::Texture`. Used when the free list
    /// is empty.
    ///
    /// # Safety
    /// `device` must be a `VK_KHR_external_memory_fd` interop device (see
    /// [`create_interop_device`]); `context` must be the decoder's CUDA context.
    pub unsafe fn build_entry(
        device: &wgpu::Device,
        context: u64,
        width: u32,
        height: u32,
    ) -> Result<PoolEntry, G2gError> {
        // SAFETY: `device` is an interop device per the contract; `shared` is freshly
        // exported and imported exactly once before being wrapped as a texture (the
        // FD is consumed by the import, the image/memory by the texture's drop).
        unsafe {
            let shared = export_nv12_image(device, width, height)?;
            let mapping = import_image_into_cuda(&shared, context)?;
            let texture = wrap_as_texture(device, shared);
            Ok(PoolEntry { mapping, texture })
        }
    }

    /// Pop a recycled entry, or `None` if the pool is empty. A returned entry has
    /// been written before, so the caller must drain prior GPU reads first.
    pub fn take_free(&self) -> Option<PoolEntry> {
        self.free.lock().unwrap().pop()
    }

    /// Wrap an in-flight entry in a drop guard that returns it to the free list
    /// when dropped (i.e. when the downstream frame's keep-alive is released).
    pub fn in_flight(&self, entry: PoolEntry) -> PoolReturn {
        PoolReturn { free: alloc::sync::Arc::clone(&self.free), entry: Some(entry) }
    }
}

/// Drop guard that recycles a [`PoolEntry`] back into its [`CudaWgpuPool`]. Stored
/// (type-erased) in the downstream frame's keep-alive, so dropping the frame frees
/// the entry for reuse rather than destroying the image.
#[derive(Debug)]
pub struct PoolReturn {
    free: alloc::sync::Arc<std::sync::Mutex<alloc::vec::Vec<PoolEntry>>>,
    entry: Option<PoolEntry>,
}

impl Drop for PoolReturn {
    fn drop(&mut self) {
        if let Some(entry) = self.entry.take() {
            if let Ok(mut free) = self.free.lock() {
                free.push(entry);
            }
        }
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

// ===========================================================================
// wgpu -> CUDA (the encode direction, M271): the reverse of the NV12 import
// above. A renderer writes a packed-RGBA `wgpu::Texture` backed by exportable
// Vulkan memory; CUDA reads the same memory as an array and copies it
// device->device into a linear `CUdeviceptr`, which `NvEnc` registers as an
// `ABGR` surface (NVENC color converts to H.264 internally). The pixels reach
// the encoder with no device->host read-back, the moat the M267 Bevy demo pays.
// ===========================================================================

/// A self-allocated, FD-exportable Vulkan RGBA8 image shared with CUDA, the
/// encode-side counterpart of [`SharedNv12Image`]. Raw handles owned here; the
/// `wgpu::Texture` that wraps it frees the image/memory on drop, and CUDA owns
/// the exported FD after import.
#[derive(Debug)]
pub struct SharedRgbaImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    fd: i32,
    size: u64,
    width: u32,
    height: u32,
}

/// Allocate an FD-exportable `R8G8B8A8_UNORM` Vulkan image (`width` x `height`),
/// usable as a wgpu render target and importable into CUDA. Mirror of
/// [`export_nv12_image`].
///
/// # Safety
/// `device` must be a Vulkan-backend wgpu device with `VK_KHR_external_memory_fd`
/// (see [`create_interop_device`]).
pub unsafe fn export_rgba_image(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> Result<SharedRgbaImage, G2gError> {
    // SAFETY: caller guarantees a Vulkan device; the hal guard is held for the
    // whole allocation and no raw handle outlives the ash device.
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
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            // COLOR_ATTACHMENT: a renderer draws into it. TRANSFER_DST: the test
            // (and `queue.write_texture`) uploads a pattern. TRANSFER_SRC/SAMPLED:
            // completeness for copies / sampling.
            .usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
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

        // Dedicated allocation: CUDA's import of a Vulkan image requires it.
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

        Ok(SharedRgbaImage { image, memory, fd, size: reqs.size, width, height })
    }
}

/// Wrap a [`SharedRgbaImage`] as a `wgpu::Texture` (an `Rgba8Unorm` render
/// target). The texture's drop frees the backing Vulkan image/memory. Mirror of
/// [`wrap_as_texture`].
///
/// # Safety
/// `device` must be the interop device the image was created on; `shared`'s image
/// and memory must not be freed by any other path.
pub unsafe fn wrap_rgba_as_texture(
    device: &wgpu::Device,
    shared: SharedRgbaImage,
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: shared.width,
        height: shared.height,
        depth_or_array_layers: 1,
    };
    let image = shared.image;
    let memory = shared.memory;
    // SAFETY: `device` is the interop Vulkan device; `image` / `memory` are valid
    // and owned here, moved into the drop callback (fired once when wgpu drops the
    // texture, after the GPU is idle).
    let hal_texture = unsafe {
        let hal_device = device.as_hal::<wgpu_hal::api::Vulkan>().expect("vulkan wgpu device");
        let raw = hal_device.raw_device().clone();
        let drop_cb: wgpu_hal::DropCallback = Box::new(move || {
            raw.destroy_image(image, None);
            raw.free_memory(memory, None);
        });
        let hal_desc = wgpu_hal::TextureDescriptor {
            label: Some("wgpu-cuda-rgba"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUses::COLOR_TARGET
                | wgpu::TextureUses::COPY_DST
                | wgpu::TextureUses::COPY_SRC
                | wgpu::TextureUses::RESOURCE,
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
        label: Some("wgpu-cuda-rgba"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    };
    // SAFETY: `hal_texture` was just produced by this device's hal.
    unsafe { device.create_texture_from_hal::<wgpu_hal::api::Vulkan>(hal_texture, &wgpu_desc) }
}

/// Import a [`SharedRgbaImage`] into CUDA once and map it as a 4-channel uint8
/// CUDA array, in `context`. Consumes `shared.fd`. Mirror of
/// [`import_image_into_cuda`] with the RGBA (4-channel, full-height) array shape.
///
/// # Safety
/// `shared` must come from [`export_rgba_image`] and not yet be imported.
pub unsafe fn import_rgba_into_cuda(
    shared: &SharedRgbaImage,
    context: u64,
) -> Result<CudaImageMapping, G2gError> {
    use cuda_ffi as c;
    // SAFETY: CUDA Driver API import sequence in `context`; on error every handle
    // created so far is destroyed before the context is popped.
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

        let mipmap_desc = c::CudaExternalMemoryMipmappedArrayDesc {
            offset: 0,
            array_desc: c::CudaArray3dDescriptor {
                width: shared.width as usize,
                height: shared.height as usize,
                depth: 0,
                format: c::CU_AD_FORMAT_UNSIGNED_INT8,
                num_channels: 4,
                flags: 0,
            },
            num_levels: 1,
            reserved: [0; 16],
        };
        let mut mipmap: c::CuMipmappedArray = core::ptr::null_mut();
        let mut array: c::CuArray = core::ptr::null_mut();
        if result.is_ok() {
            result = check(c::cuExternalMemoryGetMappedMipmappedArray(
                &mut mipmap,
                ext_mem,
                &mipmap_desc,
            ));
        }
        if result.is_ok() {
            result = check(c::cuMipmappedArrayGetLevel(&mut array, mipmap, 0));
        }
        if let Err(e) = result {
            if !mipmap.is_null() {
                c::cuMipmappedArrayDestroy(mipmap);
            }
            if !ext_mem.is_null() {
                c::cuDestroyExternalMemory(ext_mem);
            }
            let mut popped: c::CuContext = core::ptr::null_mut();
            let _ = c::cuCtxPopCurrent(&mut popped);
            return Err(e);
        }
        let mut popped: c::CuContext = core::ptr::null_mut();
        let _ = c::cuCtxPopCurrent(&mut popped);
        Ok(CudaImageMapping {
            ext_mem: ext_mem as usize,
            mipmap: mipmap as usize,
            array: array as usize,
            context,
        })
    }
}

/// Bridges a GPU-rendered RGBA `wgpu::Texture` to a CUDA-resident frame `NvEnc`
/// can encode, with no device->host read-back. Owns an exportable RGBA texture
/// (the render target) and its persistent CUDA import; [`to_cuda_frame`] copies
/// the shared array device->device into a fresh linear `CUdeviceptr` and hands it
/// downstream as a `MemoryDomain::Cuda` `Rgba8` frame.
///
/// `mapping` is declared before `texture` so the CUDA import is destroyed before
/// the texture's drop frees the Vulkan memory it aliased.
pub struct WgpuToCuda {
    device: wgpu::Device,
    queue: wgpu::Queue,
    mapping: CudaImageMapping,
    texture: wgpu::Texture,
    context: u64,
    width: u32,
    height: u32,
    /// Free list of linear output buffers, recycled across frames.
    pool: LinearBufferPool,
    /// Set once [`configure_pipeline`](AsyncElement::configure_pipeline) accepts
    /// (the element path); the standalone struct API ignores it.
    configured: bool,
    /// Frames bridged wgpu -> CUDA (the element path). Atomic so `to_cuda_frame`
    /// can count on `&self` (the standalone struct API takes `&self`).
    frames: core::sync::atomic::AtomicU64,
    // Last field: releases the retained CUDA primary context, so it runs *after*
    // `mapping` (the CUDA import) and `texture` (the Vulkan memory) have dropped.
    _ctx_release: PrimaryCtxRelease,
}

/// Releases a retained CUDA primary context on drop (the device ordinal it was
/// retained on). Held as the last field of [`WgpuToCuda`] so the release is
/// ordered after the CUDA import and Vulkan image teardown.
#[derive(Debug)]
struct PrimaryCtxRelease(i32);

impl Drop for PrimaryCtxRelease {
    fn drop(&mut self) {
        // SAFETY: balances the `cuDevicePrimaryCtxRetain` in `WgpuToCuda::new`;
        // released once. Best-effort.
        unsafe {
            let _ = cuda_ffi::cuDevicePrimaryCtxRelease(self.0);
        }
    }
}

impl core::fmt::Debug for WgpuToCuda {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuToCuda")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}

impl WgpuToCuda {
    /// Build the bridge on an interop `device`: retain the CUDA primary context on
    /// device 0 (the NVIDIA GPU the interop device also selects), allocate an
    /// exportable RGBA image, import it into CUDA, and wrap it as the render
    /// target texture. The retained context is the one `NvEnc` runs in (the
    /// bridge's frames are valid there) and is released on drop, so the bridge
    /// must outlive any frame it produced.
    ///
    /// # Safety
    /// `device` must be a `VK_KHR_external_memory_fd` interop device (see
    /// [`create_interop_device`]).
    pub unsafe fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        width: u32,
        height: u32,
    ) -> Result<Self, G2gError> {
        use cuda_ffi as c;
        let device_ordinal = 0i32;
        // SAFETY: retain the primary context on device 0; released in `Drop`.
        let context = unsafe {
            check(c::cuInit(0))?;
            let mut dev = 0i32;
            check(c::cuDeviceGet(&mut dev, device_ordinal))?;
            let mut ctx: c::CuContext = core::ptr::null_mut();
            check(c::cuDevicePrimaryCtxRetain(&mut ctx, dev))?;
            ctx as u64
        };
        // SAFETY: interop device per the contract; `shared` is exported, imported
        // exactly once (FD consumed), then wrapped (image/memory owned by the texture).
        let built = unsafe {
            export_rgba_image(&device, width, height).and_then(|shared| {
                import_rgba_into_cuda(&shared, context)
                    .map(|mapping| (mapping, wrap_rgba_as_texture(&device, shared)))
            })
        };
        match built {
            Ok((mapping, texture)) => Ok(Self {
                device,
                queue,
                mapping,
                texture,
                context,
                width,
                height,
                pool: LinearBufferPool::new(),
                configured: false,
                frames: core::sync::atomic::AtomicU64::new(0),
                _ctx_release: PrimaryCtxRelease(device_ordinal),
            }),
            Err(e) => {
                // SAFETY: balance the retain above on the build failure path.
                unsafe {
                    let _ = c::cuDevicePrimaryCtxRelease(device_ordinal);
                }
                Err(e)
            }
        }
    }

    /// The CUDA context the bridge (and `NvEnc`) run in.
    pub fn context(&self) -> u64 {
        self.context
    }

    /// The exportable RGBA render target. A renderer draws into this (or the
    /// caller `queue.write_texture`s it); the pixels are then visible to CUDA.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// The interop wgpu device / queue, for the caller to submit its render and
    /// drain it before [`to_cuda_frame`].
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Copy the current texture contents (the shared CUDA array) device->device
    /// into a linear `CUdeviceptr` and return it as a `MemoryDomain::Cuda` `Rgba8`
    /// frame stamped with `pts_ns`. The caller must have completed and drained the
    /// render (`device.poll(Wait)`) before calling this.
    ///
    /// The linear buffer is drawn from the bridge's [`LinearBufferPool`]: a
    /// recycled buffer of the right size is reused, else one is allocated, and it
    /// returns to the pool when the emitted frame's keep-alive is released. v1
    /// `cuMemAlloc`d / `cuMemFree`d every frame; this amortizes both.
    pub fn to_cuda_frame(&self, pts_ns: u64) -> Result<g2g_core::frame::Frame, G2gError> {
        use cuda_ffi as c;
        let pitch = (self.width as usize) * 4; // packed RGBA, NVENC pitch (mult of 4)
        let size = pitch * self.height as usize;
        // Reuse a pooled buffer of this size, or allocate one. A recycled buffer's
        // previous contents are fully overwritten by the copy below.
        let buf = match self.pool.take(size) {
            Some(buf) => buf,
            None => LinearBuf::alloc(self.context, size)?,
        };
        let dptr = buf.dptr;
        // SAFETY: array->linear copy in `self.context`, pushed current.
        let copied = unsafe {
            check(c::cuCtxPushCurrent(self.context as c::CuContext))?;
            let copy = c::CudaMemcpy2D {
                src_x_in_bytes: 0,
                src_y: 0,
                src_memory_type: c::CU_MEMORYTYPE_ARRAY,
                src_host: core::ptr::null(),
                src_device: 0,
                src_array: self.mapping.array as *mut core::ffi::c_void,
                src_pitch: 0,
                dst_x_in_bytes: 0,
                dst_y: 0,
                dst_memory_type: c::CU_MEMORYTYPE_DEVICE,
                dst_host: core::ptr::null_mut(),
                dst_device: dptr,
                dst_array: core::ptr::null_mut(),
                dst_pitch: pitch,
                width_in_bytes: pitch,
                height: self.height as usize,
            };
            let result = check(c::cu_memcpy_2d(&copy)).and_then(|()| check(c::cuCtxSynchronize()));
            let mut popped: c::CuContext = core::ptr::null_mut();
            let _ = c::cuCtxPopCurrent(&mut popped);
            result
        };
        // On copy failure the buffer drops here (freeing its CUdeviceptr), not
        // returned to the pool.
        copied?;

        // The frame's keep-alive recycles `buf` into the pool when released.
        let keep_alive = alloc::sync::Arc::new(RecycledLinearBuffer {
            free: alloc::sync::Arc::clone(&self.pool.free),
            buf: std::sync::Mutex::new(Some(buf)),
        });
        let buffer = g2g_core::memory::OwnedCudaBuffer::new(
            dptr,
            0,
            pitch as u32,
            0,
            self.width,
            self.height,
            self.context,
            keep_alive,
        );
        self.frames.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        Ok(g2g_core::frame::Frame::new(
            g2g_core::MemoryDomain::Cuda(buffer),
            g2g_core::FrameTiming { pts_ns, dts_ns: pts_ns, ..Default::default() },
            0,
        ))
    }

    /// Frames bridged wgpu -> CUDA so far (the element path). Useful in tests.
    pub fn frames(&self) -> u64 {
        self.frames.load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Copy an upstream `src` RGBA texture (which must live on this bridge's
    /// interop [`device`](Self::device)) into the bridge's exportable render
    /// target, then drain the device so CUDA sees the result. The element path
    /// calls this before [`to_cuda_frame`](Self::to_cuda_frame); a renderer that
    /// already draws straight into [`texture`](Self::texture) skips it. Public so a
    /// host (e.g. the Bevy demo) that renders on the interop device can push its
    /// target texture in directly, then call `to_cuda_frame`, without the
    /// `WgpuTexture`-frame `process` wrapper. `src` and the bridge texture must be
    /// copy-compatible RGBA (the srgb / non-srgb pair is allowed).
    pub fn ingest_texture(&self, src: &wgpu::Texture) -> Result<(), G2gError> {
        // Copy only the overlap, so a slightly mismatched upstream size is clamped
        // rather than triggering a wgpu validation panic.
        let w = src.width().min(self.width);
        let h = src.height().min(self.height);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("wgpu-to-cuda") });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);
        // CUDA reads the shared memory directly (no wgpu fence), so the copy must
        // be complete before to_cuda_frame's device->device copy runs.
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(gpu_err)?;
        Ok(())
    }
}

/// RGBA8 with open geometry: the bridge element's identity caps set. Only the
/// memory domain changes (WgpuTexture -> Cuda); caps do not encode the domain.
fn rgba_any() -> CapsSet {
    CapsSet::one(Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    })
}

/// `WgpuToCuda` as a pipeline transform (M275): a `MemoryDomain::WgpuTexture`
/// RGBA frame (rendered on this bridge's interop [`device`](WgpuToCuda::device))
/// in, a `MemoryDomain::Cuda` RGBA frame `NvEnc` can encode out, with no
/// device->host read-back. Caps are `Identity(RGBA)`; only the domain changes.
///
/// Unlike the M220 CUDA -> wgpu (`CudaToWgpu`) bridge, whose interop device is
/// built lazily on the first frame, this element owns its device up
/// front (in [`WgpuToCuda::new`]) and exposes it: the upstream renderer must draw
/// onto it (clone [`device`](WgpuToCuda::device) / [`queue`](WgpuToCuda::queue)
/// in), because a `wgpu::Texture` is bound to the device that made it and the
/// CUDA import only sees this device's memory. The application-wired, render-side
/// counterpart of the [`WgpuSink`](crate::wgpusink) pattern.
impl AsyncElement for WgpuToCuda {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for alt in rgba_any().alternatives() {
            if let Ok(narrowed) = upstream_caps.intersect(alt) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(rgba_any())
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !rgba_any().accepts(absolute_caps) {
            return Err(G2gError::CapsMismatch);
        }
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
                        // GPU-input only: a System frame is the CPU encoder's job.
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // A frame from a foreign GPU producer (other keep-alive type)
                    // or a different device is not bridgeable here.
                    let src = crate::gpu::texture_of(owned).ok_or(G2gError::UnsupportedDomain)?;
                    self.ingest_texture(src)?;
                    let out_frame = self.to_cuda_frame(frame.timing.pts_ns)?;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

/// One linear `CUdeviceptr` output buffer for [`WgpuToCuda::to_cuda_frame`], with
/// its byte size (the pool keys reuse on it) and the context it lives in. Drop
/// frees the pointer, so a buffer that is never recycled (pool teardown, or a
/// copy failure) is reclaimed.
#[derive(Debug)]
struct LinearBuf {
    dptr: u64,
    size: usize,
    context: u64,
}

impl LinearBuf {
    /// Allocate a `size`-byte linear buffer in `context`.
    fn alloc(context: u64, size: usize) -> Result<Self, G2gError> {
        use cuda_ffi as c;
        // SAFETY: `cuMemAlloc` in `context`, pushed current and popped before return.
        unsafe {
            check(c::cuCtxPushCurrent(context as c::CuContext))?;
            let mut dptr = 0u64;
            let result = check(c::cuMemAlloc(&mut dptr, size));
            let mut popped: c::CuContext = core::ptr::null_mut();
            let _ = c::cuCtxPopCurrent(&mut popped);
            result?;
            Ok(LinearBuf { dptr, size, context })
        }
    }
}

impl Drop for LinearBuf {
    fn drop(&mut self) {
        use cuda_ffi as c;
        // SAFETY: `dptr` came from `cuMemAlloc` in `context`; freed once, context
        // pushed. Best-effort.
        unsafe {
            if c::cuCtxPushCurrent(self.context as c::CuContext) == 0 {
                c::cuMemFree(self.dptr);
                let mut popped: c::CuContext = core::ptr::null_mut();
                let _ = c::cuCtxPopCurrent(&mut popped);
            }
        }
    }
}

/// Free list of recycled [`LinearBuf`] output buffers, shared (`Arc`) between the
/// bridge and each emitted frame's keep-alive. A buffer returns here when its
/// frame is released and is handed back out by the next [`WgpuToCuda::to_cuda_frame`]
/// of the same size; remaining buffers are freed when the pool is dropped.
#[derive(Debug, Clone, Default)]
pub struct LinearBufferPool {
    free: alloc::sync::Arc<std::sync::Mutex<alloc::vec::Vec<LinearBuf>>>,
}

impl LinearBufferPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pop a recycled buffer of exactly `size` bytes, or `None` to allocate one.
    fn take(&self, size: usize) -> Option<LinearBuf> {
        let mut free = self.free.lock().unwrap();
        let idx = free.iter().position(|b| b.size == size)?;
        Some(free.swap_remove(idx))
    }
}

/// Keep-alive for one [`WgpuToCuda::to_cuda_frame`] output frame: holds its
/// [`LinearBuf`] and returns it to the [`LinearBufferPool`] on drop (when the
/// downstream consumer releases the frame), rather than freeing it. The recycling
/// counterpart of the old per-frame-free `LinearCudaBuffer`.
#[derive(Debug)]
struct RecycledLinearBuffer {
    free: alloc::sync::Arc<std::sync::Mutex<alloc::vec::Vec<LinearBuf>>>,
    buf: std::sync::Mutex<Option<LinearBuf>>,
}

impl g2g_core::memory::CudaKeepAlive for RecycledLinearBuffer {}

impl Drop for RecycledLinearBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.lock().ok().and_then(|mut b| b.take()) {
            if let Ok(mut free) = self.free.lock() {
                free.push(buf);
            }
            // If the pool lock is poisoned, `buf` drops here and frees itself.
        }
    }
}

/// CUDA Driver API external-memory FFI for the interop bridge. Extends the
/// surface in [`crate::cuda`] (which has the download path) with the
/// import-by-FD + mapped-array calls. Struct layouts are field-for-field from
/// `cuda.h` (64-bit); the `_v2` suffixed names match the symbols `libcuda`
/// exports for the `#define`d unsuffixed entry points.
#[allow(non_snake_case, unreachable_pub)]
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
        #[link_name = "cuMemAlloc_v2"]
        pub fn cuMemAlloc(dptr: *mut u64, bytesize: usize) -> i32;
        #[link_name = "cuMemFree_v2"]
        pub fn cuMemFree(dptr: u64) -> i32;
    }
}
