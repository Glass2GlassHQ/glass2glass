//! Local zero-copy IPC over CUDA IPC memory handles (M556, `local-ipc` feature).
//!
//! The wire codec ([`g2g_core::wire`]) and its transports (`remote`, `remote-ws`)
//! only carry CPU memory: a device-resident frame must be downloaded first, so a
//! GPU producer feeding a GPU consumer in *another process* pays a full
//! device->host->device round trip. On the same machine that copy is avoidable:
//! two processes can map the *same* VRAM. This module is the CUDA path
//! (NVIDIA-only, Linux via the `cuda` feature's gate).
//!
//! The mechanism is CUDA IPC: [`ipc_export`] turns a `CUdeviceptr` into a 64-byte
//! [`CudaIpcHandle`] that another process passes to [`ipc_open`] to get a pointer
//! to the *same* device allocation (no copy; the importer reads the producer's
//! VRAM directly). Crucially the handle is **plain bytes**, unlike a DMABUF file
//! descriptor (which needs `SCM_RIGHTS` fd-passing over a Unix socket): so it
//! rides *any* byte transport, even the existing wire codec, with the only
//! constraint that the two ends share a machine and a GPU. That is why a
//! [`CudaIpcDescriptor`] (the handle plus the buffer size) serializes to a flat
//! byte buffer this module frames and reads back.
//!
//! Contract / caveats the caller owns:
//! - **Same device.** Both processes must select the same CUDA device; a handle
//!   from device 0 is meaningless on device 1.
//! - **Lifetime.** The exporting allocation (and its context) must stay alive
//!   until the importer has called [`ipc_open`]; the importer must
//!   [`ipc_close`] before the exporter frees. The producer graph keeps the frame
//!   (and its keep-alive) pinned while it is in flight, which covers this.
//! - **DMABUF is different.** The vendor-neutral path (a `dma_buf` fd imported via
//!   Vulkan external memory) needs real fd-passing (`SCM_RIGHTS`) and is a
//!   separate follow-up; only CUDA's byte-handle model fits a plain transport.

use alloc::vec::Vec;

use g2g_core::{G2gError, HardwareError};

/// Size in bytes of a `CUipcMemHandle` (`CU_IPC_HANDLE_SIZE`, cuda.h).
pub const CUDA_IPC_HANDLE_SIZE: usize = 64;

/// A CUDA IPC memory handle: 64 opaque bytes identifying a device allocation
/// across processes on the same machine + device. Plain bytes, so it crosses any
/// transport.
pub type CudaIpcHandle = [u8; CUDA_IPC_HANDLE_SIZE];

/// What crosses the transport to share one device allocation: the IPC handle and
/// the allocation size (so the importer knows how much it may read). Real
/// per-frame geometry (NV12 plane offsets / pitches / dims) is layered on top in
/// the graph-element phase; this is the transport-agnostic core.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaIpcDescriptor {
    pub handle: CudaIpcHandle,
    pub size: u64,
}

impl CudaIpcDescriptor {
    /// Serialized length: the 64-byte handle followed by the `u64` LE size.
    pub const WIRE_LEN: usize = CUDA_IPC_HANDLE_SIZE + 8;

    /// Flatten to bytes for the transport (handle, then size LE).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::WIRE_LEN);
        out.extend_from_slice(&self.handle);
        out.extend_from_slice(&self.size.to_le_bytes());
        out
    }

    /// Parse from bytes produced by [`to_bytes`](Self::to_bytes). Returns `None`
    /// on a short buffer (never trust the transport).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::WIRE_LEN {
            return None;
        }
        let mut handle = [0u8; CUDA_IPC_HANDLE_SIZE];
        handle.copy_from_slice(&bytes[..CUDA_IPC_HANDLE_SIZE]);
        let size = u64::from_le_bytes(
            bytes[CUDA_IPC_HANDLE_SIZE..Self::WIRE_LEN]
                .try_into()
                .ok()?,
        );
        Some(Self { handle, size })
    }
}

/// Map a `CUresult` to a `Result`, carrying the raw code on failure (mirrors
/// `cuda.rs`).
fn check(code: ffi::CuResult) -> Result<(), G2gError> {
    if code == ffi::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(G2gError::Hardware(HardwareError::Cuda(code)))
    }
}

/// Initialise the CUDA driver API and create a context on device `ordinal`,
/// leaving it current on the calling thread. Returns the `CUcontext` as a `u64`.
/// Destroy it with [`destroy_context`] when done.
pub fn init_context(ordinal: i32) -> Result<u64, G2gError> {
    // SAFETY: plain driver-API init/create calls; a host without an NVIDIA GPU
    // fails here (mapped to a hardware error) rather than proceeding.
    unsafe {
        check(ffi::cu_init(0))?;
        let mut dev: ffi::CuDevice = 0;
        check(ffi::cu_device_get(&mut dev, ordinal))?;
        let mut ctx: ffi::CuContext = core::ptr::null_mut();
        check(ffi::cu_ctx_create(&mut ctx, 0, dev))?;
        Ok(ctx as u64)
    }
}

/// Destroy a context from [`init_context`]. Best-effort (a failure is
/// unactionable at drop time).
pub fn destroy_context(ctx: u64) {
    // SAFETY: `ctx` came from `cu_ctx_create`; destroyed exactly once by the caller.
    unsafe {
        let _ = ffi::cu_ctx_destroy(ctx as ffi::CuContext);
    }
}

/// Allocate `size` bytes of linear device memory in the current context.
pub fn alloc(size: usize) -> Result<u64, G2gError> {
    let mut dptr: u64 = 0;
    // SAFETY: `dptr` is a valid out-pointer; the driver writes the allocation.
    unsafe { check(ffi::cu_mem_alloc(&mut dptr, size))? };
    Ok(dptr)
}

/// Free device memory from [`alloc`].
///
/// # Safety
/// `dptr` must be a live allocation from [`alloc`] in the current context, freed
/// exactly once.
pub unsafe fn free(dptr: u64) -> Result<(), G2gError> {
    // SAFETY: the caller guarantees `dptr` is a live allocation freed once.
    unsafe { check(ffi::cu_mem_free(dptr)) }
}

/// Copy `src` host bytes into device allocation `dst` (host->device).
///
/// # Safety
/// `dst` must be a live device allocation of at least `src.len()` bytes in the
/// current context.
pub unsafe fn htod(dst: u64, src: &[u8]) -> Result<(), G2gError> {
    // SAFETY: `src` is a valid slice of `src.len()` bytes; `dst` has room per the
    // caller's contract.
    unsafe {
        check(ffi::cu_memcpy_htod(
            dst,
            src.as_ptr() as *const core::ffi::c_void,
            src.len(),
        ))
    }
}

/// Copy `src` device allocation into `dst` host bytes (device->host).
///
/// # Safety
/// `src` must be a live device allocation of at least `dst.len()` bytes in the
/// current context.
pub unsafe fn dtoh(dst: &mut [u8], src: u64) -> Result<(), G2gError> {
    // SAFETY: `dst` is a valid mutable slice of `dst.len()` bytes; `src` has that
    // many bytes per the caller's contract.
    unsafe {
        check(ffi::cu_memcpy_dtoh(
            dst.as_mut_ptr() as *mut core::ffi::c_void,
            src,
            dst.len(),
        ))
    }
}

/// Copy `size` bytes device->device (`src` -> `dst`), an on-GPU copy with no
/// host / PCIe round trip. Used by the receive side of the local transport to
/// take a private copy of a mapped IPC allocation, so the producer may free the
/// original as soon as the copy completes (decoupling the two processes'
/// lifetimes without a device->host->device round trip).
///
/// # Safety
/// `src` and `dst` must be live device allocations of at least `size` bytes,
/// both valid in the current context.
pub unsafe fn dtod(dst: u64, src: u64, size: usize) -> Result<(), G2gError> {
    // SAFETY: both pointers address `size`-byte allocations in the current
    // context per the caller's contract.
    unsafe { check(ffi::cu_memcpy_dtod(dst, src, size)) }
}

/// Return the base pointer and total size of the allocation containing `dptr`
/// (`cuMemGetAddressRange`). CUDA IPC exports whole allocations, so a frame whose
/// plane pointer is a *sub-allocation* of a larger pool (e.g. an NVDEC decode
/// pool) must export this base, not the plane pointer, and carry the plane's
/// offset within it.
///
/// # Safety
/// `dptr` must be a live device pointer in the current context.
pub unsafe fn address_range(dptr: u64) -> Result<(u64, u64), G2gError> {
    let mut base: u64 = 0;
    let mut size: usize = 0;
    // SAFETY: `base` / `size` are valid out-pointers; `dptr` is live per the
    // caller's contract.
    unsafe { check(ffi::cu_mem_get_address_range(&mut base, &mut size, dptr))? };
    Ok((base, size as u64))
}

/// Export a device allocation as an IPC handle another process can [`ipc_open`].
///
/// # Safety
/// `dptr` must be the base of a live `cuMemAlloc` allocation in the current
/// context (CUDA IPC exports whole allocations, not sub-ranges).
pub unsafe fn ipc_export(dptr: u64) -> Result<CudaIpcHandle, G2gError> {
    let mut handle = ffi::CuIpcMemHandle {
        reserved: [0u8; CUDA_IPC_HANDLE_SIZE],
    };
    // SAFETY: `handle` is a valid out-struct; `dptr` is an allocation base per the
    // caller's contract.
    unsafe { check(ffi::cu_ipc_get_mem_handle(&mut handle, dptr))? };
    Ok(handle.reserved)
}

/// Open an IPC handle exported by another process, returning a device pointer to
/// the *same* allocation (no copy). Pair with [`ipc_close`].
///
/// # Safety
/// A CUDA context on the *same device* as the exporter must be current; the
/// exporting allocation must still be live.
pub unsafe fn ipc_open(handle: &CudaIpcHandle) -> Result<u64, G2gError> {
    let mut dptr: u64 = 0;
    let h = ffi::CuIpcMemHandle { reserved: *handle };
    // SAFETY: `dptr` is a valid out-pointer; `h` is a 64-byte handle passed by
    // value per the C ABI; LAZY_ENABLE_PEER_ACCESS is the documented default flag.
    unsafe {
        check(ffi::cu_ipc_open_mem_handle(
            &mut dptr,
            h,
            ffi::CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
        ))?
    };
    Ok(dptr)
}

/// Close a mapping from [`ipc_open`] (does not free the exporter's allocation).
///
/// # Safety
/// `dptr` must be a pointer returned by [`ipc_open`], closed exactly once, with
/// the same context current.
pub unsafe fn ipc_close(dptr: u64) -> Result<(), G2gError> {
    // SAFETY: `dptr` came from `ipc_open` per the caller's contract.
    unsafe { check(ffi::cu_ipc_close_mem_handle(dptr)) }
}

/// Thin hand-rolled CUDA Driver API FFI for the IPC path, linking `libcuda`
/// directly (the `cuda` feature's gate guarantees Linux + NVIDIA). Kept separate
/// from `cuda.rs`'s FFI so the `local-ipc` feature is self-contained; the driver
/// `#define`s the unsuffixed names to their `_v2` symbols, so we name the `_v2`
/// exports the shared object provides.
mod ffi {
    use core::ffi::c_void;

    pub(super) type CuContext = *mut c_void;
    pub(super) type CuResult = i32;
    pub(super) type CuDevice = i32;

    pub(super) const CUDA_SUCCESS: CuResult = 0;
    /// `CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS`: enable peer access lazily on first
    /// use, the documented default flag for `cuIpcOpenMemHandle`.
    pub(super) const CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS: u32 = 0x1;

    /// `CUipcMemHandle` (a.k.a. `CUipcMemHandle_st`): `char reserved[64]`.
    #[repr(C)]
    #[derive(Debug)]
    pub(super) struct CuIpcMemHandle {
        pub(super) reserved: [u8; super::CUDA_IPC_HANDLE_SIZE],
    }

    #[link(name = "cuda")]
    extern "C" {
        #[link_name = "cuInit"]
        pub(super) fn cu_init(flags: u32) -> CuResult;
        #[link_name = "cuDeviceGet"]
        pub(super) fn cu_device_get(device: *mut CuDevice, ordinal: i32) -> CuResult;
        #[link_name = "cuCtxCreate_v2"]
        pub(super) fn cu_ctx_create(pctx: *mut CuContext, flags: u32, dev: CuDevice) -> CuResult;
        #[link_name = "cuCtxDestroy_v2"]
        pub(super) fn cu_ctx_destroy(ctx: CuContext) -> CuResult;
        #[link_name = "cuMemAlloc_v2"]
        pub(super) fn cu_mem_alloc(dptr: *mut u64, size: usize) -> CuResult;
        #[link_name = "cuMemFree_v2"]
        pub(super) fn cu_mem_free(dptr: u64) -> CuResult;
        #[link_name = "cuMemGetAddressRange_v2"]
        pub(super) fn cu_mem_get_address_range(
            pbase: *mut u64,
            psize: *mut usize,
            dptr: u64,
        ) -> CuResult;
        #[link_name = "cuMemcpyHtoD_v2"]
        pub(super) fn cu_memcpy_htod(dst: u64, src: *const c_void, size: usize) -> CuResult;
        #[link_name = "cuMemcpyDtoH_v2"]
        pub(super) fn cu_memcpy_dtoh(dst: *mut c_void, src: u64, size: usize) -> CuResult;
        #[link_name = "cuMemcpyDtoD_v2"]
        pub(super) fn cu_memcpy_dtod(dst: u64, src: u64, size: usize) -> CuResult;
        /// Export an allocation as a 64-byte IPC handle.
        #[link_name = "cuIpcGetMemHandle"]
        pub(super) fn cu_ipc_get_mem_handle(handle: *mut CuIpcMemHandle, dptr: u64) -> CuResult;
        /// Open a handle from another process; the handle is passed by value.
        #[link_name = "cuIpcOpenMemHandle_v2"]
        pub(super) fn cu_ipc_open_mem_handle(
            pdptr: *mut u64,
            handle: CuIpcMemHandle,
            flags: u32,
        ) -> CuResult;
        /// Close a mapping from `cuIpcOpenMemHandle` (does not free the source).
        #[link_name = "cuIpcCloseMemHandle"]
        pub(super) fn cu_ipc_close_mem_handle(dptr: u64) -> CuResult;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn descriptor_round_trips_through_bytes() {
        let mut handle = [0u8; CUDA_IPC_HANDLE_SIZE];
        for (i, b) in handle.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(3).wrapping_add(1);
        }
        let desc = CudaIpcDescriptor {
            handle,
            size: 1920 * 1080 * 3 / 2,
        };
        let bytes = desc.to_bytes();
        assert_eq!(bytes.len(), CudaIpcDescriptor::WIRE_LEN);
        assert_eq!(CudaIpcDescriptor::from_bytes(&bytes), Some(desc));
    }

    #[test]
    fn short_buffer_is_rejected() {
        // A truncated descriptor must not parse (never trust the transport).
        let short = vec![0u8; CudaIpcDescriptor::WIRE_LEN - 1];
        assert_eq!(CudaIpcDescriptor::from_bytes(&short), None);
    }

    #[test]
    fn trailing_bytes_are_ignored() {
        let desc = CudaIpcDescriptor {
            handle: [7u8; CUDA_IPC_HANDLE_SIZE],
            size: 42,
        };
        let mut bytes = desc.to_bytes();
        bytes.extend_from_slice(&[0xFF; 8]); // transport may frame with padding
        assert_eq!(CudaIpcDescriptor::from_bytes(&bytes), Some(desc));
    }
}
