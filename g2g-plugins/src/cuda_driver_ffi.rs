// libcuda exports the `_v2` symbols that cuda.h `#define`s the unsuffixed names to.

use core::ffi::c_void;

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

#[link(name = "cuda")]
extern "C" {
    #[link_name = "cuInit"]
    pub(crate) fn cu_init(flags: u32) -> CuResult;
    #[link_name = "cuDeviceGet"]
    pub(crate) fn cu_device_get(device: *mut CuDevice, ordinal: i32) -> CuResult;
    #[link_name = "cuCtxCreate_v2"]
    pub(crate) fn cu_ctx_create(context: *mut CuContext, flags: u32, device: CuDevice) -> CuResult;
    #[link_name = "cuCtxDestroy_v2"]
    pub(crate) fn cu_ctx_destroy(context: CuContext) -> CuResult;
    #[link_name = "cuCtxPushCurrent_v2"]
    pub(crate) fn cu_ctx_push_current(context: CuContext) -> CuResult;
    #[link_name = "cuCtxPopCurrent_v2"]
    pub(crate) fn cu_ctx_pop_current(context: *mut CuContext) -> CuResult;
    #[cfg(any(
        feature = "local-ipc",
        feature = "cuda-wgpu",
        all(test, feature = "nvenc")
    ))]
    #[link_name = "cuMemAlloc_v2"]
    pub(crate) fn cu_mem_alloc(device_pointer: *mut u64, size: usize) -> CuResult;
    // pitched rows, aligned so NVENC's input-resource registration accepts them
    #[link_name = "cuMemAllocPitch_v2"]
    pub(crate) fn cu_mem_alloc_pitch(
        device_pointer: *mut u64,
        pitch: *mut usize,
        width_in_bytes: usize,
        height: usize,
        element_size: u32,
    ) -> CuResult;
    #[link_name = "cuMemFree_v2"]
    pub(crate) fn cu_mem_free(device_pointer: u64) -> CuResult;
    #[cfg(any(feature = "local-ipc", all(test, feature = "nvenc")))]
    #[link_name = "cuMemcpyHtoD_v2"]
    pub(crate) fn cu_memcpy_htod(destination: u64, source: *const c_void, size: usize) -> CuResult;
    #[cfg(any(feature = "local-ipc", all(test, feature = "nvdec")))]
    #[link_name = "cuMemcpyDtoH_v2"]
    pub(crate) fn cu_memcpy_dtoh(destination: *mut c_void, source: u64, size: usize) -> CuResult;
    #[link_name = "cuMemcpy2D_v2"]
    pub(crate) fn cu_memcpy_2d(copy: *const CudaMemcpy2D) -> CuResult;
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

    #[link(name = "cuda")]
    extern "C" {
        #[link_name = "cuGraphicsGLRegisterImage"]
        pub(crate) fn cu_graphics_gl_register_image(
            resource: *mut CuGraphicsResource,
            image: GlUint,
            target: GlEnum,
            flags: u32,
        ) -> CuResult;
        #[link_name = "cuGraphicsUnregisterResource"]
        pub(crate) fn cu_graphics_unregister_resource(resource: CuGraphicsResource) -> CuResult;
        #[link_name = "cuGraphicsMapResources"]
        pub(crate) fn cu_graphics_map_resources(
            count: u32,
            resources: *mut CuGraphicsResource,
            stream: CuStream,
        ) -> CuResult;
        #[link_name = "cuGraphicsUnmapResources"]
        pub(crate) fn cu_graphics_unmap_resources(
            count: u32,
            resources: *mut CuGraphicsResource,
            stream: CuStream,
        ) -> CuResult;
        #[link_name = "cuGraphicsSubResourceGetMappedArray"]
        pub(crate) fn cu_graphics_sub_resource_get_mapped_array(
            array: *mut CuArray,
            resource: CuGraphicsResource,
            array_index: u32,
            mip_level: u32,
        ) -> CuResult;
    }
}
