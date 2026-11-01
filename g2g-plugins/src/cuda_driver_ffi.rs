// libcuda exports the `_v2` symbols that cuda.h `#define`s the unsuffixed names to.

use core::ffi::c_void;

use g2g_core::{G2gError, HardwareError};

// `CUresult` is an int-sized C enum.
pub(crate) type CuResult = i32;
// `CUdevice` is an int handle.
pub(crate) type CuDevice = i32;
pub(crate) type CuContext = *mut c_void;

pub(crate) const CUDA_SUCCESS: CuResult = 0;

pub(crate) const CU_MEMORYTYPE_HOST: u32 = 0x01;
pub(crate) const CU_MEMORYTYPE_DEVICE: u32 = 0x02;
#[cfg(any(feature = "cuda-wgpu", feature = "cuda-gl", feature = "cuda-kms"))]
pub(crate) const CU_MEMORYTYPE_ARRAY: u32 = 0x03;

// `CUDA_MEMCPY2D`, field for field from cuda.h (64-bit).
#[repr(C)]
#[derive(Debug)]
pub(crate) struct CudaMemcpy2D {
    pub(crate) src_x_in_bytes: usize,
    pub(crate) src_y: usize,
    pub(crate) src_memory_type: u32,
    pub(crate) src_host: *const c_void,
    pub(crate) src_device: u64,
    pub(crate) src_array: *mut c_void,
    pub(crate) src_pitch: usize,
    pub(crate) dst_x_in_bytes: usize,
    pub(crate) dst_y: usize,
    pub(crate) dst_memory_type: u32,
    pub(crate) dst_host: *mut c_void,
    pub(crate) dst_device: u64,
    pub(crate) dst_array: *mut c_void,
    pub(crate) dst_pitch: usize,
    pub(crate) width_in_bytes: usize,
    pub(crate) height: usize,
}

pub(crate) const LIBCUDA: &str = "libcuda.so.1";
// cuda.h names this "shared object initialization failed"
pub(crate) const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: CuResult = 303;

crate::shared_library::shared_library! {
    pub(crate) struct LibCuda = LIBCUDA;
    when missing return CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;
    pub(crate) fn cu_init = "cuInit"(flags: u32) -> CuResult;
    pub(crate) fn cu_device_get = "cuDeviceGet"(device: *mut CuDevice, ordinal: i32) -> CuResult;
    pub(crate) fn cu_ctx_create = "cuCtxCreate_v2"(context: *mut CuContext, flags: u32, device: CuDevice) -> CuResult;
    pub(crate) fn cu_ctx_destroy = "cuCtxDestroy_v2"(context: CuContext) -> CuResult;
    pub(crate) fn cu_ctx_push_current = "cuCtxPushCurrent_v2"(context: CuContext) -> CuResult;
    pub(crate) fn cu_ctx_pop_current = "cuCtxPopCurrent_v2"(context: *mut CuContext) -> CuResult;
    #[cfg(any(
        feature = "local-ipc",
        feature = "cuda-wgpu",
        all(test, feature = "nvenc")
    ))]
    pub(crate) fn cu_mem_alloc = "cuMemAlloc_v2"(device_pointer: *mut u64, size: usize) -> CuResult;
    // pitched rows, aligned so NVENC's input-resource registration accepts them
    pub(crate) fn cu_mem_alloc_pitch = "cuMemAllocPitch_v2"(
        device_pointer: *mut u64,
        pitch: *mut usize,
        width_in_bytes: usize,
        height: usize,
        element_size: u32,
    ) -> CuResult;
    pub(crate) fn cu_mem_free = "cuMemFree_v2"(device_pointer: u64) -> CuResult;
    #[cfg(any(feature = "local-ipc", all(test, feature = "nvenc")))]
    pub(crate) fn cu_memcpy_htod = "cuMemcpyHtoD_v2"(destination: u64, source: *const c_void, size: usize) -> CuResult;
    #[cfg(any(feature = "local-ipc", all(test, feature = "nvdec")))]
    pub(crate) fn cu_memcpy_dtoh = "cuMemcpyDtoH_v2"(destination: *mut c_void, source: u64, size: usize) -> CuResult;
    pub(crate) fn cu_memcpy_2d = "cuMemcpy2D_v2"(copy: *const CudaMemcpy2D) -> CuResult;
    #[cfg(feature = "nvdec")]
    pub(crate) fn cu_get_error_name = "cuGetErrorName"(
        error: CuResult,
        name: *mut *const core::ffi::c_char,
    ) -> CuResult;
}

pub(crate) fn init_driver() -> Result<(), G2gError> {
    if LibCuda::get().is_none() {
        return Err(G2gError::Hardware(HardwareError::LibraryNotFound(LIBCUDA)));
    }
    // SAFETY: cuInit takes only a flags word, and 0 is its one legal value.
    match unsafe { cu_init(0) } {
        CUDA_SUCCESS => Ok(()),
        code => Err(G2gError::Hardware(HardwareError::Cuda(code))),
    }
}

// CUDA-GL interop, the GL entry points live in libcuda itself
#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
pub(crate) use gl_interop::*;

#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
mod gl_interop {
    use super::{c_void, CuResult};

    pub(crate) type CuGraphicsResource = *mut c_void;
    pub(crate) type CuArray = *mut c_void;
    // null is the default stream
    pub(crate) type CuStream = *mut c_void;
    pub(crate) type GlUint = u32;
    pub(crate) type GlEnum = u32;

    // CUDA fully overwrites the texture each frame, so the driver may skip preserving it
    pub(crate) const CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: u32 = 0x02;
    pub(crate) const GL_TEXTURE_2D: GlEnum = 0x0DE1;

    crate::shared_library::shared_library! {
        pub(crate) struct LibCudaGraphics = super::LIBCUDA;
        when missing return super::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;
        pub(crate) fn cu_graphics_gl_register_image = "cuGraphicsGLRegisterImage"(
            resource: *mut CuGraphicsResource,
            image: GlUint,
            target: GlEnum,
            flags: u32,
        ) -> CuResult;
        pub(crate) fn cu_graphics_unregister_resource = "cuGraphicsUnregisterResource"(resource: CuGraphicsResource) -> CuResult;
        pub(crate) fn cu_graphics_map_resources = "cuGraphicsMapResources"(
            count: u32,
            resources: *mut CuGraphicsResource,
            stream: CuStream,
        ) -> CuResult;
        pub(crate) fn cu_graphics_unmap_resources = "cuGraphicsUnmapResources"(
            count: u32,
            resources: *mut CuGraphicsResource,
            stream: CuStream,
        ) -> CuResult;
        pub(crate) fn cu_graphics_sub_resource_get_mapped_array = "cuGraphicsSubResourceGetMappedArray"(
            array: *mut CuArray,
            resource: CuGraphicsResource,
            array_index: u32,
            mip_level: u32,
        ) -> CuResult;
    }
}
