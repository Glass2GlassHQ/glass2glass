// libcuda exports the `_v2` symbols that cuda.h `#define`s the unsuffixed names to.

use core::ffi::c_void;

use g2g_core::{G2gError, HardwareError};

// `CUresult` is an int-sized C enum.
pub(crate) type CuResult = i32;
// `CUdevice` is an int handle.
pub(crate) type CuDevice = i32;
pub(crate) type CuContext = *mut c_void;
pub(crate) type CuStream = *mut c_void;
pub(crate) type CuEvent = *mut c_void;
pub(crate) const CU_STREAM_NON_BLOCKING: u32 = 0x1;
pub(crate) const CU_EVENT_DISABLE_TIMING: u32 = 0x2;
#[cfg(feature = "hip")]
pub(crate) type CuModule = *mut c_void;
#[cfg(feature = "hip")]
pub(crate) type CuFunction = *mut c_void;
#[cfg(feature = "hip")]
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
#[cfg(feature = "hip")]
pub(crate) const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

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

#[cfg(target_os = "linux")]
pub(crate) const LIBCUDA: &str = "libcuda.so.1";
#[cfg(windows)]
pub(crate) const LIBCUDA: &str = "nvcuda.dll";
// cuda.h names this "shared object initialization failed"
pub(crate) const CUDA_ERROR_SHARED_OBJECT_INIT_FAILED: CuResult = 303;

crate::shared_library::shared_library! {
    pub(crate) struct LibCuda = LIBCUDA;
    when missing return CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;
    pub(crate) fn cu_init = "cuInit"(flags: u32) -> CuResult;
    pub(crate) fn cu_device_get = "cuDeviceGet"(device: *mut CuDevice, ordinal: i32) -> CuResult;
    pub(crate) fn cu_device_get_name = "cuDeviceGetName"(name: *mut core::ffi::c_char, len: i32, device: CuDevice) -> CuResult;
    #[cfg(any(feature = "nvdec", feature = "local-ipc", all(test, feature = "nvenc")))]
    pub(crate) fn cu_ctx_create = "cuCtxCreate_v2"(context: *mut CuContext, flags: u32, device: CuDevice) -> CuResult;
    #[cfg(any(feature = "nvdec", feature = "local-ipc", all(test, feature = "nvenc")))]
    pub(crate) fn cu_ctx_destroy = "cuCtxDestroy_v2"(context: CuContext) -> CuResult;
    pub(crate) fn cu_ctx_push_current = "cuCtxPushCurrent_v2"(context: CuContext) -> CuResult;
    pub(crate) fn cu_ctx_pop_current = "cuCtxPopCurrent_v2"(context: *mut CuContext) -> CuResult;
    #[cfg(any(
        feature = "local-ipc",
        feature = "cuda-wgpu",
        feature = "hip",
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
    #[cfg(any(feature = "local-ipc", all(test, any(feature = "nvdec", feature = "cuda-wgpu"))))]
    pub(crate) fn cu_memcpy_dtoh = "cuMemcpyDtoH_v2"(destination: *mut c_void, source: u64, size: usize) -> CuResult;
    #[cfg(any(feature = "cuda-wgpu", feature = "cuda-gl", feature = "cuda-kms"))]
    pub(crate) fn cu_memcpy_2d = "cuMemcpy2D_v2"(copy: *const CudaMemcpy2D) -> CuResult;
    pub(crate) fn cu_memcpy_2d_async = "cuMemcpy2DAsync_v2"(copy: *const CudaMemcpy2D, stream: CuStream) -> CuResult;
    pub(crate) fn cu_device_primary_ctx_retain = "cuDevicePrimaryCtxRetain"(context: *mut CuContext, device: CuDevice) -> CuResult;
    pub(crate) fn cu_device_primary_ctx_release = "cuDevicePrimaryCtxRelease_v2"(device: CuDevice) -> CuResult;
    pub(crate) fn cu_stream_create = "cuStreamCreate"(stream: *mut CuStream, flags: u32) -> CuResult;
    pub(crate) fn cu_stream_destroy = "cuStreamDestroy_v2"(stream: CuStream) -> CuResult;
    pub(crate) fn cu_stream_synchronize = "cuStreamSynchronize"(stream: CuStream) -> CuResult;
    pub(crate) fn cu_stream_wait_event = "cuStreamWaitEvent"(stream: CuStream, event: CuEvent, flags: u32) -> CuResult;
    pub(crate) fn cu_event_create = "cuEventCreate"(event: *mut CuEvent, flags: u32) -> CuResult;
    pub(crate) fn cu_event_record = "cuEventRecord"(event: CuEvent, stream: CuStream) -> CuResult;
    pub(crate) fn cu_event_synchronize = "cuEventSynchronize"(event: CuEvent) -> CuResult;
    pub(crate) fn cu_event_destroy = "cuEventDestroy_v2"(event: CuEvent) -> CuResult;
    #[cfg(test)]
    pub(crate) fn cu_launch_host_func = "cuLaunchHostFunc"(
        stream: CuStream,
        function: unsafe extern "C" fn(*mut c_void),
        data: *mut c_void,
    ) -> CuResult;
    #[cfg(feature = "hip")]
    pub(crate) fn cu_memcpy_htod_async = "cuMemcpyHtoDAsync_v2"(destination: u64, source: *const c_void, size: usize, stream: CuStream) -> CuResult;
    #[cfg(feature = "hip")]
    pub(crate) fn cu_memcpy_dtoh_async = "cuMemcpyDtoHAsync_v2"(destination: *mut c_void, source: u64, size: usize, stream: CuStream) -> CuResult;
    pub(crate) fn cu_mem_alloc_host = "cuMemAllocHost_v2"(pointer: *mut *mut c_void, size: usize) -> CuResult;
    pub(crate) fn cu_mem_free_host = "cuMemFreeHost"(pointer: *mut c_void) -> CuResult;
    #[cfg(feature = "hip")]
    pub(crate) fn cu_device_get_attribute = "cuDeviceGetAttribute"(value: *mut i32, attribute: i32, device: CuDevice) -> CuResult;
    #[cfg(feature = "hip")]
    pub(crate) fn cu_module_load_data = "cuModuleLoadData"(module: *mut CuModule, image: *const c_void) -> CuResult;
    #[cfg(feature = "hip")]
    pub(crate) fn cu_module_unload = "cuModuleUnload"(module: CuModule) -> CuResult;
    #[cfg(feature = "hip")]
    pub(crate) fn cu_module_get_function = "cuModuleGetFunction"(function: *mut CuFunction, module: CuModule, name: *const core::ffi::c_char) -> CuResult;
    #[cfg(feature = "hip")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn cu_launch_kernel = "cuLaunchKernel"(
        function: CuFunction,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        shared_memory_bytes: u32,
        stream: CuStream,
        parameters: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CuResult;
    #[cfg(feature = "nvdec")]
    pub(crate) fn cu_get_error_name = "cuGetErrorName"(
        error: CuResult,
        name: *mut *const core::ffi::c_char,
    ) -> CuResult;
}

pub(crate) fn check(code: CuResult) -> Result<(), G2gError> {
    match code {
        CUDA_SUCCESS => Ok(()),
        code => Err(G2gError::Hardware(HardwareError::Cuda(code))),
    }
}

pub(crate) fn init_driver() -> Result<(), G2gError> {
    if LibCuda::get().is_none() {
        return Err(G2gError::Hardware(HardwareError::LibraryNotFound(LIBCUDA)));
    }
    // SAFETY: cuInit takes only a flags word, and 0 is its one legal value.
    check(unsafe { cu_init(0) })
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
