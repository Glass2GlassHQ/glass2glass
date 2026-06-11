//! CUDA device-memory consumers (C3 Phase 3).
//!
//! [`CudaDownload`] is the low-risk Phase 3 bring-up element
//! (DESIGN-C3-cuda.md §3.4): a transform that copies a
//! [`Backend::NvdecCuda`](crate::ffmpegdec::Backend) frame
//! ([`MemoryDomain::Cuda`]) back to system memory (NV12, device->host
//! `cuMemcpy2D`) so a CUDA-resident stream can reach the existing CPU sinks
//! (`WaylandSink` / `KmsSink`). It negates the zero-copy latency win, but it
//! makes the `NvdecCuda` decode path end-to-end usable and testable (frame
//! counts, geometry) before the real `CudaGlSink` exists (§4 step 2).
//!
//! Caps surface is `Identity(NV12)`: input and output are the same NV12
//! description (caps do not encode the memory domain), so the element drops
//! into any `NvdecCuda -> sink` chain without changing negotiation. Only the
//! frame's domain changes, `Cuda -> System`. A frame that is already in
//! system memory passes through untouched, so the element is a safe no-op on
//! the software / cuvid backends.
//!
//! CUDA bindings are a thin hand-rolled FFI linking `libcuda` directly (the
//! `cuda` feature's gate guarantees Linux + NVIDIA), matching the decision in
//! DESIGN-C3-cuda.md §6: `cudarc` has no GL-interop wrappers and fights the
//! foreign-`CUcontext` ownership (the context is created and owned by
//! ffmpeg's hwdevice and carried on the [`OwnedCudaBuffer`], so the consumer
//! pushes a context it does not own).
//!
//! Per the transform contract (see `run_source_transform_sink`), this element
//! does NOT emit `Eos` itself: the runner forwards the EOS sentinel after
//! `process(Eos)` returns.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use g2g_core::memory::OwnedCudaBuffer;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, Frame, G2gError,
    HardwareError, MemoryDomain, OutputSink, PipelinePacket, Rate, RawVideoFormat, SystemSlice,
};

/// Pass-through transform that copies CUDA device-memory NV12 frames to
/// system memory. See the module docs.
#[derive(Debug, Default)]
pub struct CudaDownload {
    configured: bool,
    /// Frames copied device->host (the `MemoryDomain::Cuda` inputs).
    downloaded: u64,
    /// Frames forwarded untouched (already system-memory inputs).
    forwarded: u64,
}

impl CudaDownload {
    pub fn new() -> Self {
        Self::default()
    }

    /// Frames copied out of CUDA device memory so far.
    pub fn downloaded(&self) -> u64 {
        self.downloaded
    }

    /// Frames that were already in system memory and passed through untouched.
    pub fn forwarded(&self) -> u64 {
        self.forwarded
    }
}

/// The element's NV12 caps set with open geometry. The `Identity` constraint
/// couples input and output to this set; any concrete NV12 geometry the
/// solver fixates is accepted (caps do not encode the memory domain).
fn nv12_any() -> CapsSet {
    CapsSet::one(Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    })
}

impl AsyncElement for CudaDownload {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Legacy / mixed-cascade path: the download keeps the caps unchanged
        // (only the domain changes), so narrow upstream against NV12. The
        // native solver uses the `Identity` constraint below instead.
        for alt in nv12_any().alternatives() {
            if let Ok(narrowed) = upstream_caps.intersect(alt) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Identity(nv12_any())
    }

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError> {
        // The solver should only ever hand us NV12; fail loud otherwise (a
        // negotiation bug, not a runtime state).
        if !nv12_any().accepts(absolute_caps) {
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
                    let out_frame = match frame.domain {
                        MemoryDomain::Cuda(ref buf) => {
                            // SAFETY: `buf` came from a `NvdecCuda` decode, so
                            // its plane pointers are valid CUDA device memory
                            // in `buf.context` for the life of the frame (the
                            // keep-alive owner pins them); `frame` outlives
                            // this copy.
                            let bytes = unsafe { download_nv12(buf)? };
                            self.downloaded += 1;
                            Frame {
                                domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
                                timing: frame.timing,
                                sequence: frame.sequence,
                            }
                        }
                        // Already in system memory (software / cuvid backend);
                        // forward untouched.
                        _ => {
                            self.forwarded += 1;
                            frame
                        }
                    };
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

/// Per-plane parameters for the device->host `cuMemcpy2D` of one NV12 plane.
/// Pure geometry, so the layout is unit-testable without a GPU.
#[derive(Debug, PartialEq, Eq)]
struct PlaneCopy {
    /// Source row pitch in bytes (the decoder's device-side alignment).
    src_pitch: usize,
    /// Byte offset of this plane in the packed destination buffer.
    dst_offset: usize,
    /// Destination row pitch in bytes (packed: equals `width_bytes`).
    dst_pitch: usize,
    /// Bytes copied per row.
    width_bytes: usize,
    /// Number of rows.
    height: usize,
}

/// Compute the two plane copies (luma, then interleaved chroma) and the total
/// packed NV12 buffer length for a frame of `width` x `height` whose device
/// planes have the given pitches.
///
/// Packed NV12 layout: luma plane (`width` x `height`) at offset 0, then the
/// interleaved chroma plane (`2*ceil(width/2)` x `ceil(height/2)`). For even
/// dimensions the total is `width * height * 3 / 2`.
fn nv12_plane_copies(
    width: u32,
    height: u32,
    luma_pitch: u32,
    chroma_pitch: u32,
) -> (PlaneCopy, PlaneCopy, usize) {
    let w = width as usize;
    let h = height as usize;
    // ceil division so odd dimensions still cover the last partial chroma
    // column / row.
    let chroma_w_bytes = 2 * w.div_ceil(2);
    let chroma_h = h.div_ceil(2);

    let luma = PlaneCopy {
        src_pitch: luma_pitch as usize,
        dst_offset: 0,
        dst_pitch: w,
        width_bytes: w,
        height: h,
    };
    let chroma = PlaneCopy {
        src_pitch: chroma_pitch as usize,
        dst_offset: w * h,
        dst_pitch: chroma_w_bytes,
        width_bytes: chroma_w_bytes,
        height: chroma_h,
    };
    let total = w * h + chroma_w_bytes * chroma_h;
    (luma, chroma, total)
}

/// Copy both NV12 planes of `buf` from CUDA device memory into a freshly
/// allocated packed system buffer (device->host).
///
/// # Safety
/// `buf`'s plane pointers must be valid device memory in `buf.context`, and
/// the backing allocation must stay alive for the duration of the call (its
/// keep-alive owner guarantees this while the [`OwnedCudaBuffer`] is held).
unsafe fn download_nv12(buf: &OwnedCudaBuffer) -> Result<Box<[u8]>, G2gError> {
    let (luma, chroma, total) =
        nv12_plane_copies(buf.width, buf.height, buf.luma_pitch, buf.chroma_pitch);
    let mut dst = vec![0u8; total].into_boxed_slice();
    let dst_base = dst.as_mut_ptr();

    // SAFETY: push the foreign `CUcontext` the pointers are valid in, run both
    // plane copies, then always pop it (even on copy failure) so we leave the
    // thread's context stack as we found it. The copies run unconditionally
    // and the first error is surfaced after the pop via `Result::and`.
    unsafe {
        check(ffi::cu_ctx_push_current(buf.context as ffi::CuContext))?;
        let luma_result = copy_plane(&luma, buf.luma_ptr, dst_base);
        let chroma_result = copy_plane(&chroma, buf.chroma_ptr, dst_base);
        let mut popped: ffi::CuContext = core::ptr::null_mut();
        let pop_result = check(ffi::cu_ctx_pop_current(&mut popped));
        luma_result.and(chroma_result).and(pop_result)?;
    }
    Ok(dst)
}

/// Issue one device->host `cuMemcpy2D` for a single plane.
///
/// # Safety
/// `src_device` must point to valid device memory of at least
/// `plane.src_pitch * plane.height` bytes in the current CUDA context;
/// `dst_base + plane.dst_offset` must have room for
/// `plane.dst_pitch * plane.height` bytes.
unsafe fn copy_plane(
    plane: &PlaneCopy,
    src_device: u64,
    dst_base: *mut u8,
) -> Result<(), G2gError> {
    let copy = ffi::CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: ffi::CU_MEMORYTYPE_DEVICE,
        src_host: core::ptr::null(),
        src_device,
        src_array: core::ptr::null_mut(),
        src_pitch: plane.src_pitch,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: ffi::CU_MEMORYTYPE_HOST,
        // SAFETY: `dst_offset` is within the buffer `download_nv12` sized to
        // hold both planes (see `nv12_plane_copies`).
        dst_host: unsafe { dst_base.add(plane.dst_offset) } as *mut core::ffi::c_void,
        dst_device: 0,
        dst_array: core::ptr::null_mut(),
        dst_pitch: plane.dst_pitch,
        width_in_bytes: plane.width_bytes,
        height: plane.height,
    };
    // SAFETY: `copy` is a fully-initialised CUDA_MEMCPY2D describing a
    // device->host copy; the driver only reads through it.
    check(unsafe { ffi::cu_memcpy_2d(&copy) })
}

/// Map a `CUresult` to a `Result`, carrying the raw code on failure.
fn check(code: ffi::CuResult) -> Result<(), G2gError> {
    if code == ffi::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(G2gError::Hardware(HardwareError::Cuda(code)))
    }
}

/// Thin hand-rolled CUDA Driver API FFI: exactly the surface `CudaDownload`
/// needs (foreign-context push/pop + a 2D device->host copy), linking
/// `libcuda` directly. The driver `#define`s the unsuffixed `cuMemcpy2D` /
/// `cuCtxPushCurrent` etc. to their `_v2` symbols, so we name the `_v2`
/// exports the shared object actually provides.
mod ffi {
    use core::ffi::c_void;

    /// `CUcontext` is an opaque handle (`struct CUctx_st *`).
    pub type CuContext = *mut c_void;
    /// `CUresult` is a C `enum` (int-sized).
    pub type CuResult = i32;

    pub const CUDA_SUCCESS: CuResult = 0;

    /// `CUmemorytype` values used here (`cuda.h`).
    pub const CU_MEMORYTYPE_HOST: u32 = 0x01;
    pub const CU_MEMORYTYPE_DEVICE: u32 = 0x02;

    /// `CUDA_MEMCPY2D` (a.k.a. `CUDA_MEMCPY2D_v2`), field-for-field from
    /// `cuda.h`. `size_t` -> `usize`, `CUdeviceptr` -> `u64`,
    /// `CUmemorytype` -> `u32` (int-sized C enum), `CUarray` -> opaque
    /// pointer.
    #[repr(C)]
    #[derive(Debug)]
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

    #[link(name = "cuda")]
    extern "C" {
        /// Push `ctx` onto the calling thread's current-context stack.
        #[link_name = "cuCtxPushCurrent_v2"]
        pub fn cu_ctx_push_current(ctx: CuContext) -> CuResult;
        /// Pop the current context, returning it through `pctx`.
        #[link_name = "cuCtxPopCurrent_v2"]
        pub fn cu_ctx_pop_current(pctx: *mut CuContext) -> CuResult;
        /// 2D memory copy described entirely by `*pcopy`.
        #[link_name = "cuMemcpy2D_v2"]
        pub fn cu_memcpy_2d(pcopy: *const CudaMemcpy2D) -> CuResult;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    #[test]
    fn caps_constraint_is_identity_nv12() {
        let d = CudaDownload::new();
        let CapsConstraint::Identity(set) = d.caps_constraint_as_transform() else {
            panic!("expected Identity");
        };
        // Open-geometry NV12: the solver fixates the concrete dims.
        assert!(set.accepts(&nv12(1920, 1080)));
        assert!(set.accepts(&nv12(640, 480)));
    }

    #[test]
    fn configure_rejects_non_nv12() {
        let mut d = CudaDownload::new();
        assert!(d.configure_pipeline(&nv12(1280, 720)).is_ok());

        let mut e = CudaDownload::new();
        let i420 = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
        };
        assert_eq!(
            e.configure_pipeline(&i420).err(),
            Some(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn intercept_narrows_nv12_and_rejects_other() {
        let d = CudaDownload::new();
        assert_eq!(d.intercept_caps(&nv12(1280, 720)), Ok(nv12(1280, 720)));

        let i420 = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(d.intercept_caps(&i420), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn plane_copies_even_dims_pack_to_3_2() {
        // 1920x1080 with the decoder aligning the device pitch up to 2048.
        let (luma, chroma, total) = nv12_plane_copies(1920, 1080, 2048, 2048);
        assert_eq!(
            luma,
            PlaneCopy {
                src_pitch: 2048,
                dst_offset: 0,
                dst_pitch: 1920,
                width_bytes: 1920,
                height: 1080,
            }
        );
        assert_eq!(
            chroma,
            PlaneCopy {
                src_pitch: 2048,
                dst_offset: 1920 * 1080,
                dst_pitch: 1920,
                width_bytes: 1920,
                height: 540,
            }
        );
        // Packed: no row padding, total is the standard NV12 size.
        assert_eq!(total, 1920 * 1080 * 3 / 2);
    }

    #[test]
    fn plane_copies_odd_dims_round_chroma_up() {
        // Odd dims: chroma must cover the partial last column and row.
        let (luma, chroma, total) = nv12_plane_copies(3, 3, 256, 256);
        assert_eq!(luma.width_bytes, 3);
        assert_eq!(luma.height, 3);
        assert_eq!(luma.dst_pitch, 3);
        // ceil(3/2)=2 -> 2*2=4 bytes wide, 2 rows tall.
        assert_eq!(chroma.width_bytes, 4);
        assert_eq!(chroma.height, 2);
        assert_eq!(chroma.dst_offset, 9);
        assert_eq!(total, 9 + 8);
    }
}
