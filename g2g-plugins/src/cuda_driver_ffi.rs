// libcuda exports the `_v2` symbols that cuda.h `#define`s the unsuffixed names to.

#[cfg(any(
    feature = "local-ipc",
    feature = "nvdec",
    feature = "nvenc",
    feature = "cuda-wgpu"
))]
use core::ffi::c_void;

// `CUresult` is an int-sized C enum.
pub(crate) type CuResult = i32;
// `CUdevice` is an int handle.
pub(crate) type CuDevice = i32;
#[cfg(any(
    feature = "local-ipc",
    feature = "nvdec",
    feature = "nvenc",
    feature = "cuda-wgpu"
))]
pub(crate) type CuContext = *mut c_void;

pub(crate) const CUDA_SUCCESS: CuResult = 0;

#[cfg(feature = "cuda-wgpu")]
pub(crate) const CU_MEMORYTYPE_HOST: u32 = 0x01;
#[cfg(feature = "cuda-wgpu")]
pub(crate) const CU_MEMORYTYPE_DEVICE: u32 = 0x02;
#[cfg(feature = "cuda-wgpu")]
pub(crate) const CU_MEMORYTYPE_ARRAY: u32 = 0x03;

// `CUDA_MEMCPY2D`, field for field from cuda.h (64-bit).
#[cfg(feature = "cuda-wgpu")]
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
    #[cfg(any(feature = "local-ipc", feature = "nvdec", all(test, feature = "nvenc")))]
    #[link_name = "cuCtxCreate_v2"]
    pub(crate) fn cu_ctx_create(context: *mut CuContext, flags: u32, device: CuDevice) -> CuResult;
    #[cfg(any(feature = "local-ipc", feature = "nvdec", all(test, feature = "nvenc")))]
    #[link_name = "cuCtxDestroy_v2"]
    pub(crate) fn cu_ctx_destroy(context: CuContext) -> CuResult;
    #[cfg(any(feature = "nvdec", feature = "nvenc", feature = "cuda-wgpu"))]
    #[link_name = "cuCtxPushCurrent_v2"]
    pub(crate) fn cu_ctx_push_current(context: CuContext) -> CuResult;
    #[cfg(any(feature = "nvdec", feature = "nvenc", feature = "cuda-wgpu"))]
    #[link_name = "cuCtxPopCurrent_v2"]
    pub(crate) fn cu_ctx_pop_current(context: *mut CuContext) -> CuResult;
    #[cfg(any(
        feature = "local-ipc",
        feature = "cuda-wgpu",
        all(test, feature = "nvenc")
    ))]
    #[link_name = "cuMemAlloc_v2"]
    pub(crate) fn cu_mem_alloc(device_pointer: *mut u64, size: usize) -> CuResult;
    #[cfg(any(
        feature = "local-ipc",
        feature = "cuda-wgpu",
        all(test, feature = "nvenc")
    ))]
    #[link_name = "cuMemFree_v2"]
    pub(crate) fn cu_mem_free(device_pointer: u64) -> CuResult;
    #[cfg(any(feature = "local-ipc", all(test, feature = "nvenc")))]
    #[link_name = "cuMemcpyHtoD_v2"]
    pub(crate) fn cu_memcpy_htod(destination: u64, source: *const c_void, size: usize) -> CuResult;
    #[cfg(any(feature = "local-ipc", all(test, feature = "nvdec")))]
    #[link_name = "cuMemcpyDtoH_v2"]
    pub(crate) fn cu_memcpy_dtoh(destination: *mut c_void, source: u64, size: usize) -> CuResult;
    #[cfg(feature = "cuda-wgpu")]
    #[link_name = "cuMemcpy2D_v2"]
    pub(crate) fn cu_memcpy_2d(copy: *const CudaMemcpy2D) -> CuResult;
}
