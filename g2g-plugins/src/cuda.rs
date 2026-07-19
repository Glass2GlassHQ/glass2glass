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
use alloc::sync::Arc;
use alloc::vec;

use g2g_core::memory::{CudaKeepAlive, MemoryDomainKind, OwnedCudaBuffer};
use g2g_core::runtime::{auto_plug_domain_converters, GraphNode};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, Frame, G2gError, Graph,
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
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
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

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
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
                                meta: Default::default(),
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
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Owns a CUDA context created by [`CudaUpload`]. Destroyed on drop (the last
/// `Arc` ref, after every uploaded frame's [`DevAlloc`] is gone, since each holds
/// one). `Send + Sync`: the context is used single-threaded by the element and
/// only read-only by downstream GPU consumers, the same contract as the other
/// CUDA owners (M213).
#[derive(Debug)]
struct UploadCtx(u64);

// SAFETY: the contained CUcontext is driven through `&mut CudaUpload` (never
// concurrently) and downstream frames reference its device memory read-only.
unsafe impl Send for UploadCtx {}
// SAFETY: see above.
unsafe impl Sync for UploadCtx {}

impl Drop for UploadCtx {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a context created by `cu_ctx_create`, destroyed once.
        unsafe {
            let _ = ffi::cu_ctx_destroy(self.0 as ffi::CuContext);
        }
    }
}

/// Keep-alive owner for an uploaded device buffer: frees the allocation on drop
/// and pins the context (`Arc<UploadCtx>`) for the frame's lifetime so the
/// context outlives any frame still in flight downstream.
#[derive(Debug)]
struct DevAlloc {
    dptr: u64,
    ctx: Arc<UploadCtx>,
}

impl Drop for DevAlloc {
    fn drop(&mut self) {
        // SAFETY: push the owning context, free the allocation made under it, pop.
        unsafe {
            if ffi::cu_ctx_push_current(self.ctx.0 as ffi::CuContext) == ffi::CUDA_SUCCESS {
                let _ = ffi::cu_mem_free(self.dptr);
                let mut popped: ffi::CuContext = core::ptr::null_mut();
                let _ = ffi::cu_ctx_pop_current(&mut popped);
            }
        }
    }
}

impl CudaKeepAlive for DevAlloc {}

/// Pass-through transform that uploads system-memory NV12 frames to CUDA device
/// memory (`System -> Cuda`), the mirror of [`CudaDownload`]. It is the
/// host->device converter the auto-plug registry uses, and on its own unblocks
/// feeding a CPU-side NV12 stream into [`NvEnc`](crate::nvenc::NvEnc), which
/// ingests `MemoryDomain::Cuda` only. A frame already in CUDA memory passes
/// through untouched, so the element is a safe no-op on a GPU-resident path.
///
/// Unlike `CudaDownload` (whose Cuda input carries the foreign context), an
/// upload's System input has no context, so this element creates and owns its
/// own CUDA context at configure and tears it down once every uploaded frame is
/// released.
#[derive(Debug, Default)]
pub struct CudaUpload {
    configured: bool,
    width: u32,
    height: u32,
    /// Created at configure; cloned into every uploaded frame's keep-alive.
    context: Option<Arc<UploadCtx>>,
    uploaded: u64,
    forwarded: u64,
}

impl CudaUpload {
    pub fn new() -> Self {
        Self::default()
    }

    /// Frames copied host->device into CUDA memory so far.
    pub fn uploaded(&self) -> u64 {
        self.uploaded
    }

    /// Frames already in CUDA memory, passed through untouched.
    pub fn forwarded(&self) -> u64 {
        self.forwarded
    }
}

impl AsyncElement for CudaUpload {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
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

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width,
            height,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if let Dim::Fixed(w) = width {
            self.width = *w;
        }
        if let Dim::Fixed(h) = height {
            self.height = *h;
        }
        // Create our own CUDA context (the System input carries none).
        // SAFETY: plain driver-API init/create calls; failures map to a hardware
        // error so a host without an NVIDIA GPU fails the configure gracefully.
        let ctx = unsafe {
            check(ffi::cu_init(0))?;
            let mut dev: ffi::CuDevice = 0;
            check(ffi::cu_device_get(&mut dev, 0))?;
            let mut ctx: ffi::CuContext = core::ptr::null_mut();
            check(ffi::cu_ctx_create(&mut ctx, 0, dev))?;
            // `cu_ctx_create` leaves the new context current on this thread; pop
            // it so per-frame uploads push/pop explicitly (the element may run on
            // a different worker thread).
            let mut popped: ffi::CuContext = core::ptr::null_mut();
            let _ = ffi::cu_ctx_pop_current(&mut popped);
            ctx as u64
        };
        self.context = Some(Arc::new(UploadCtx(ctx)));
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// Emits CUDA NV12 (the uploaded frames; a passed-through Cuda frame keeps its
    /// own domain), so a downstream link is a GPU link (M285).
    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Cuda
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
                        MemoryDomain::System(ref slice) => {
                            let ctx = self.context.clone().ok_or(G2gError::NotConfigured)?;
                            // SAFETY: `slice` holds at least a packed NV12 frame of
                            // the configured geometry (checked in `upload_nv12`);
                            // `ctx` is a live context owned by this element.
                            let buf = unsafe {
                                upload_nv12(&ctx, slice.as_slice(), self.width, self.height)?
                            };
                            self.uploaded += 1;
                            Frame {
                                domain: MemoryDomain::Cuda(buf),
                                timing: frame.timing,
                                sequence: frame.sequence,
                                meta: Default::default(),
                            }
                        }
                        // Already on the GPU: forward untouched.
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
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Allocate a pitched device NV12 buffer in `ctx` and copy `src` (a packed system
/// NV12 frame) into it host->device. One pitched allocation holds the luma plane
/// (`height` rows) followed by the interleaved chroma plane (`ceil(height/2)`
/// rows), both at the hardware pitch `cuMemAllocPitch` returns, so NVENC's input
/// registration accepts the device pointer (a tight pitch is rejected).
///
/// # Safety
/// `ctx` must own a live CUcontext; `src` must hold at least
/// [`nv12_byte_size`]`(width, height)` bytes.
unsafe fn upload_nv12(
    ctx: &Arc<UploadCtx>,
    src: &[u8],
    width: u32,
    height: u32,
) -> Result<OwnedCudaBuffer, G2gError> {
    let total = nv12_byte_size(width, height);
    if src.len() < total {
        return Err(G2gError::CapsMismatch);
    }
    let w = width as usize;
    let h = height as usize;
    let chroma_w = 2 * w.div_ceil(2); // interleaved CbCr row, = width for even w
    let chroma_h = h.div_ceil(2);

    // SAFETY: push the element's context, allocate a pitched buffer for both
    // planes stacked, copy each plane host->device honoring the device pitch, then
    // pop. On any failure free the allocation (context still current) so we never
    // leak; on success the buffer's `DevAlloc` owns the free.
    unsafe {
        check(ffi::cu_ctx_push_current(ctx.0 as ffi::CuContext))?;
        let mut dptr: u64 = 0;
        let mut pitch: usize = 0;
        // Widest row across the two planes (chroma_w == w here) sizes the pitch;
        // element size 16 matches NVENC's preferred alignment.
        let mut result = check(ffi::cu_mem_alloc_pitch(
            &mut dptr,
            &mut pitch,
            w.max(chroma_w),
            h + chroma_h,
            16,
        ));
        if result.is_ok() {
            let chroma_dst = dptr + (pitch * h) as u64;
            result = htod_plane(src.as_ptr(), 0, w, w, h, dptr, pitch).and_then(|()| {
                htod_plane(
                    src.as_ptr(),
                    w * h,
                    chroma_w,
                    w,
                    chroma_h,
                    chroma_dst,
                    pitch,
                )
            });
            if result.is_err() {
                let _ = ffi::cu_mem_free(dptr);
            }
        }
        let mut popped: ffi::CuContext = core::ptr::null_mut();
        let _ = ffi::cu_ctx_pop_current(&mut popped);
        result?;
        Ok(OwnedCudaBuffer::new(
            dptr,
            dptr + (pitch * h) as u64,
            pitch as u32,
            pitch as u32,
            width,
            height,
            ctx.0,
            Arc::new(DevAlloc {
                dptr,
                ctx: Arc::clone(ctx),
            }),
        ))
    }
}

/// One host->device `cuMemcpy2D` of an NV12 plane: `width_bytes` x `height` rows
/// from `src_base + src_off` (tight `src_pitch`) into `dst_device` at `dst_pitch`.
///
/// # Safety
/// The current CUDA context must own `dst_device`, which must have room for
/// `dst_pitch * height` bytes; `src_base + src_off` must have `width_bytes` per
/// row for `height` rows.
unsafe fn htod_plane(
    src_base: *const u8,
    src_off: usize,
    width_bytes: usize,
    src_pitch: usize,
    height: usize,
    dst_device: u64,
    dst_pitch: usize,
) -> Result<(), G2gError> {
    let copy = ffi::CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: ffi::CU_MEMORYTYPE_HOST,
        // SAFETY: `src_off` is within the caller's packed NV12 buffer.
        src_host: unsafe { src_base.add(src_off) } as *const core::ffi::c_void,
        src_device: 0,
        src_array: core::ptr::null_mut(),
        src_pitch,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: ffi::CU_MEMORYTYPE_DEVICE,
        dst_host: core::ptr::null_mut(),
        dst_device,
        dst_array: core::ptr::null_mut(),
        dst_pitch,
        width_in_bytes: width_bytes,
        height,
    };
    // SAFETY: `copy` fully describes a host->device 2D copy; the driver reads it.
    check(unsafe { ffi::cu_memcpy_2d(&copy) })
}

/// The g2g memory-domain converter for a `(from, to)` pair, or `None` when g2g
/// has none (the auto-plug then leaves the edge to fail loud). Covers the
/// CUDA<->System pair: [`CudaDownload`] (`Cuda -> System`) and [`CudaUpload`]
/// (`System -> Cuda`). The factory the M354 auto-plug calls; see
/// [`auto_plug_cuda_converters`].
pub fn cuda_domain_converter(from: MemoryDomainKind, to: MemoryDomainKind) -> Option<GraphNode> {
    match (from, to) {
        (MemoryDomainKind::Cuda, MemoryDomainKind::System) => {
            Some(GraphNode::element(CudaDownload::new()))
        }
        (MemoryDomainKind::System, MemoryDomainKind::Cuda) => {
            Some(GraphNode::element(CudaUpload::new()))
        }
        _ => None,
    }
}

/// Auto-plug CUDA<->System domain converters into `graph` (M354): splices a
/// [`CudaUpload`] / [`CudaDownload`] on any edge whose producer and consumer
/// cannot agree on a memory domain. E.g. a System NV12 source feeding `NvEnc`
/// (CUDA-only input) gains a `CudaUpload`; a CUDA decoder fanned out to a System
/// sink gains a `CudaDownload` on that branch. A no-op where domains already
/// agree, so it is safe to call on any graph before `run_graph`.
pub fn auto_plug_cuda_converters(graph: Graph<GraphNode>) -> Graph<GraphNode> {
    auto_plug_domain_converters(graph, &cuda_domain_converter)
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
/// Packed NV12 buffer size in bytes for a `width` x `height` frame
/// (`width*height` luma + `2*ceil(w/2)*ceil(h/2)` interleaved chroma; for even
/// dims the familiar `width*height*3/2`). Used by `CudaGlSink` to size its M12
/// allocation proposal.
pub fn nv12_byte_size(width: u32, height: u32) -> usize {
    // The pitches do not affect the packed size; pass width as a tight pitch.
    let (_, _, total) = nv12_plane_copies(width, height, width, width);
    total
}

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
/// allocated packed system buffer (device->host). `pub(crate)` so `NvDec`'s
/// download-on-demand path (M352) reuses the same copy when negotiation settles
/// its output on `System`.
///
/// # Safety
/// `buf`'s plane pointers must be valid device memory in `buf.context`, and
/// the backing allocation must stay alive for the duration of the call (its
/// keep-alive owner guarantees this while the [`OwnedCudaBuffer`] is held).
pub(crate) unsafe fn download_nv12(buf: &OwnedCudaBuffer) -> Result<Box<[u8]>, G2gError> {
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

/// Push `context` onto the calling thread's CUDA current-context stack and
/// leave it current. The `CudaGlSink` worker calls this once on its GL thread
/// (with the ffmpeg-owned context from the first decoded frame) so subsequent
/// CUDA-GL interop and copies run in that context.
///
/// # Safety
/// `context` must be a valid `CUcontext` and the calling thread must own the
/// stack it is pushed onto (a dedicated worker thread).
#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
pub unsafe fn make_context_current(context: u64) -> Result<(), G2gError> {
    // SAFETY: `context` is a valid CUcontext per the contract.
    unsafe { check(ffi::cu_ctx_push_current(context as ffi::CuContext)) }
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
    pub(super) type CuContext = *mut c_void;
    /// `CUresult` is a C `enum` (int-sized).
    pub(super) type CuResult = i32;

    pub(super) const CUDA_SUCCESS: CuResult = 0;

    /// `CUmemorytype` values used here (`cuda.h`).
    pub(super) const CU_MEMORYTYPE_HOST: u32 = 0x01;
    pub(super) const CU_MEMORYTYPE_DEVICE: u32 = 0x02;

    /// `CUDA_MEMCPY2D` (a.k.a. `CUDA_MEMCPY2D_v2`), field-for-field from
    /// `cuda.h`. `size_t` -> `usize`, `CUdeviceptr` -> `u64`,
    /// `CUmemorytype` -> `u32` (int-sized C enum), `CUarray` -> opaque
    /// pointer.
    #[repr(C)]
    #[derive(Debug)]
    pub(super) struct CudaMemcpy2D {
        pub(super) src_x_in_bytes: usize,
        pub(super) src_y: usize,
        pub(super) src_memory_type: u32,
        pub(super) src_host: *const c_void,
        pub(super) src_device: u64,
        pub(super) src_array: *mut c_void,
        pub(super) src_pitch: usize,
        pub(super) dst_x_in_bytes: usize,
        pub(super) dst_y: usize,
        pub(super) dst_memory_type: u32,
        pub(super) dst_host: *mut c_void,
        pub(super) dst_device: u64,
        pub(super) dst_array: *mut c_void,
        pub(super) dst_pitch: usize,
        pub(super) width_in_bytes: usize,
        pub(super) height: usize,
    }

    /// `CUdevice` is an int handle.
    pub(super) type CuDevice = i32;

    #[link(name = "cuda")]
    extern "C" {
        /// Push `ctx` onto the calling thread's current-context stack.
        #[link_name = "cuCtxPushCurrent_v2"]
        pub(super) fn cu_ctx_push_current(ctx: CuContext) -> CuResult;
        /// Pop the current context, returning it through `pctx`.
        #[link_name = "cuCtxPopCurrent_v2"]
        pub(super) fn cu_ctx_pop_current(pctx: *mut CuContext) -> CuResult;
        /// 2D memory copy described entirely by `*pcopy`.
        #[link_name = "cuMemcpy2D_v2"]
        pub(super) fn cu_memcpy_2d(pcopy: *const CudaMemcpy2D) -> CuResult;

        // --- Context + device-memory lifecycle for `CudaUpload` (M353). Unlike
        // `CudaDownload`, whose input frame carries the foreign CUcontext, an
        // upload's System input has none, so the element creates and owns its own
        // context and device allocations. ---

        /// Initialise the CUDA driver API (flags must be 0). Idempotent.
        #[link_name = "cuInit"]
        pub(super) fn cu_init(flags: u32) -> CuResult;
        /// Get a device handle by ordinal.
        #[link_name = "cuDeviceGet"]
        pub(super) fn cu_device_get(device: *mut CuDevice, ordinal: i32) -> CuResult;
        /// Create a context on `dev` and leave it current on the calling thread.
        #[link_name = "cuCtxCreate_v2"]
        pub(super) fn cu_ctx_create(pctx: *mut CuContext, flags: u32, dev: CuDevice) -> CuResult;
        /// Destroy a context created by [`cu_ctx_create`].
        #[link_name = "cuCtxDestroy_v2"]
        pub(super) fn cu_ctx_destroy(ctx: CuContext) -> CuResult;
        /// Allocate pitched device memory (`*pitch` rows of >= `width_bytes`),
        /// hardware-aligned so NVENC's input-resource registration accepts it.
        #[link_name = "cuMemAllocPitch_v2"]
        pub(super) fn cu_mem_alloc_pitch(
            dptr: *mut u64,
            pitch: *mut usize,
            width_bytes: usize,
            height: usize,
            element_size: u32,
        ) -> CuResult;
        /// Free device memory from [`cu_mem_alloc`] / [`cu_mem_alloc_pitch`].
        #[link_name = "cuMemFree_v2"]
        pub(super) fn cu_mem_free(dptr: u64) -> CuResult;
    }

    // --- CUDA-GL interop (C3 Phase 3, step 2 / `CudaGlSink`) ---
    //
    // Staged ahead of its consumer (the EGL/GL windowing in step 2 phase B):
    // the per-frame path is map -> get the mapped `cudaArray` -> `cuMemcpy2D`
    // the NV12 plane device->array -> unmap, presenting via a fragment shader
    // (DESIGN-C3-cuda.md §3.2, Appendix A). Signatures are verified against
    // the CUDA Driver API docs (`CUDA_GL` / `CUDA_GRAPHICS` groups). These GL
    // entry points live in `libcuda` itself, so no extra link is needed.
    // `#[allow(dead_code)]` until phase B calls them; `non_snake_case` keeps
    // the C names so they map 1:1 to the docs for the phase-B implementer.

    /// `CUgraphicsResource` is an opaque handle (`struct CUgraphicsResource_st *`).
    #[allow(dead_code)]
    pub(super) type CuGraphicsResource = *mut c_void;
    /// `CUarray` is an opaque handle (`struct CUarray_st *`).
    #[allow(dead_code)]
    pub(super) type CuArray = *mut c_void;
    /// `CUstream`; the default stream is null.
    #[allow(dead_code)]
    pub(super) type CuStream = *mut c_void;
    /// `GLuint` (OpenGL texture name).
    #[allow(dead_code)]
    pub(super) type GlUint = u32;
    /// `GLenum` (OpenGL enumerant).
    #[allow(dead_code)]
    pub(super) type GlEnum = u32;

    /// `CUmemorytype::CU_MEMORYTYPE_ARRAY`: a `cuMemcpy2D` destination that is
    /// a mapped `CUarray` (the GL texture) rather than host/device memory.
    #[allow(dead_code)]
    pub(super) const CU_MEMORYTYPE_ARRAY: u32 = 0x03;
    /// `CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD`: CUDA fully overwrites the
    /// resource each frame (the decoder plane is its sole writer), so the
    /// driver may skip preserving prior contents.
    #[allow(dead_code)]
    pub(super) const CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD: u32 = 0x02;
    /// `GL_TEXTURE_2D` target (OpenGL spec constant).
    #[allow(dead_code)]
    pub(super) const GL_TEXTURE_2D: GlEnum = 0x0DE1;

    #[allow(dead_code, non_snake_case)]
    #[link(name = "cuda")]
    extern "C" {
        /// Register a GL texture object for CUDA access.
        pub(super) fn cuGraphicsGLRegisterImage(
            pCudaResource: *mut CuGraphicsResource,
            image: GlUint,
            target: GlEnum,
            Flags: u32,
        ) -> CuResult;
        /// Unregister a previously-registered graphics resource.
        pub(super) fn cuGraphicsUnregisterResource(resource: CuGraphicsResource) -> CuResult;
        /// Map graphics resources for access by CUDA.
        pub(super) fn cuGraphicsMapResources(
            count: u32,
            resources: *mut CuGraphicsResource,
            hStream: CuStream,
        ) -> CuResult;
        /// Unmap graphics resources.
        pub(super) fn cuGraphicsUnmapResources(
            count: u32,
            resources: *mut CuGraphicsResource,
            hStream: CuStream,
        ) -> CuResult;
        /// Get the `CUarray` through which to access a mapped resource.
        pub(super) fn cuGraphicsSubResourceGetMappedArray(
            pArray: *mut CuArray,
            resource: CuGraphicsResource,
            arrayIndex: u32,
            mipLevel: u32,
        ) -> CuResult;
    }
}

/// Per-plane extent of the NV12 -> GL-texture upload (CUDA device memory ->
/// mapped `cudaArray`), in bytes per row and rows. Pure geometry, so it is
/// unit-testable without a GPU.
///
/// Per DESIGN-C3-cuda.md Appendix A the NV12 frame is two GL textures: a
/// full-res `R8` luma plane (1 byte / texel) and a half-res `RG8` interleaved
/// CbCr chroma plane (2 bytes / texel). The source row pitch comes from the
/// [`OwnedCudaBuffer`] at upload time; the destination is a `cudaArray`, which
/// carries no pitch of its own.
#[derive(Debug, PartialEq, Eq)]
pub struct GlUpload {
    /// Bytes copied per row into the texture's array.
    pub width_bytes: usize,
    /// Number of rows.
    pub height: usize,
}

/// Luma then chroma upload extents for a `width` x `height` NV12 frame.
pub fn nv12_gl_uploads(width: u32, height: u32) -> (GlUpload, GlUpload) {
    let w = width as usize;
    let h = height as usize;
    let luma = GlUpload {
        width_bytes: w,
        height: h,
    };
    let chroma = GlUpload {
        // RG8 half-res: ceil(w/2) texels * 2 bytes, ceil(h/2) rows.
        width_bytes: 2 * w.div_ceil(2),
        height: h.div_ceil(2),
    };
    (luma, chroma)
}

/// GLSL ES 1.00 vertex shader: pass the texcoords through and position a
/// fullscreen quad. Paired with [`FRAGMENT_SHADER_NV12`].
pub const VERTEX_SHADER: &str = "\
attribute vec2 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = vec4(a_pos, 0.0, 1.0);
}
";

/// GLSL ES 1.00 fragment shader: sample the NV12 luma (`R8`) and interleaved
/// chroma (`RG8`) textures and convert BT.601 limited-range YCbCr -> RGB.
/// Verbatim from DESIGN-C3-cuda.md Appendix A (swap the matrix for BT.709 on
/// HD sources once a colour-metadata field exists on `Caps`).
pub const FRAGMENT_SHADER_NV12: &str = "\
precision mediump float;
varying vec2 v_uv;
uniform sampler2D y_tex;
uniform sampler2D uv_tex;
void main() {
    float y = texture2D(y_tex, v_uv).r;
    vec2  c = texture2D(uv_tex, v_uv).rg;
    y = 1.1643 * (y - 0.0625);
    float cb = c.x - 0.5;
    float cr = c.y - 0.5;
    float r = y + 1.5958 * cr;
    float g = y - 0.3917 * cb - 0.8129 * cr;
    float b = y + 2.0170 * cb;
    gl_FragColor = vec4(r, g, b, 1.0);
}
";

/// CUDA side of the NV12 -> GL-texture presentation (`CudaGlSink`). Registers
/// the two GL textures (full-res `R8` luma, half-res `RG8` chroma) with CUDA
/// once, then per frame maps them, copies each decoded NV12 plane
/// device->`cudaArray` (`cuMemcpy2D`), and unmaps. Unregisters on drop.
///
/// The textures must already be allocated at the plane dimensions and a GL
/// context current on the calling thread, and the ffmpeg CUDA context must be
/// current (pushed) on that same thread. The sink worker owns this on its GL
/// thread, so the raw resource handles never cross threads.
#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
#[derive(Debug)]
pub struct CudaGlInterop {
    y_res: ffi::CuGraphicsResource,
    uv_res: ffi::CuGraphicsResource,
}

#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
impl CudaGlInterop {
    /// Register the luma (`y_tex`) and chroma (`uv_tex`) GL textures with CUDA
    /// (write-discard: CUDA is their sole writer).
    ///
    /// # Safety
    /// `y_tex`/`uv_tex` must be live `GL_TEXTURE_2D` names allocated at the
    /// luma/chroma plane dimensions in the GL context current on this thread;
    /// the ffmpeg CUDA context must be current (pushed) on this thread.
    pub unsafe fn register(y_tex: u32, uv_tex: u32) -> Result<Self, G2gError> {
        let mut y_res: ffi::CuGraphicsResource = core::ptr::null_mut();
        let mut uv_res: ffi::CuGraphicsResource = core::ptr::null_mut();
        // SAFETY: the textures are live GL_TEXTURE_2D names per the contract.
        unsafe {
            check(ffi::cuGraphicsGLRegisterImage(
                &mut y_res,
                y_tex,
                ffi::GL_TEXTURE_2D,
                ffi::CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
            ))?;
            if let Err(e) = check(ffi::cuGraphicsGLRegisterImage(
                &mut uv_res,
                uv_tex,
                ffi::GL_TEXTURE_2D,
                ffi::CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
            )) {
                // Roll back the luma registration so we don't leak it.
                let _ = ffi::cuGraphicsUnregisterResource(y_res);
                return Err(e);
            }
        }
        Ok(Self { y_res, uv_res })
    }

    /// Copy the decoded NV12 planes of `buf` into the registered GL textures
    /// (device->array), honouring the source pitch.
    ///
    /// # Safety
    /// `buf`'s pointers must be valid device memory in the CUDA context
    /// current on this thread; the registered textures must match the frame
    /// geometry.
    pub unsafe fn upload(&self, buf: &OwnedCudaBuffer) -> Result<(), G2gError> {
        let (luma, chroma) = nv12_gl_uploads(buf.width, buf.height);
        let mut resources = [self.y_res, self.uv_res];
        // SAFETY: both resources are registered; the array handles obtained
        // below are valid only between map and unmap, so the copies sit
        // strictly inside that window and the unmap always runs.
        unsafe {
            check(ffi::cuGraphicsMapResources(
                2,
                resources.as_mut_ptr(),
                core::ptr::null_mut(),
            ))?;
            let y_copy = copy_plane_to_array(self.y_res, buf.luma_ptr, buf.luma_pitch, &luma);
            let uv_copy =
                copy_plane_to_array(self.uv_res, buf.chroma_ptr, buf.chroma_pitch, &chroma);
            let unmap = check(ffi::cuGraphicsUnmapResources(
                2,
                resources.as_mut_ptr(),
                core::ptr::null_mut(),
            ));
            y_copy.and(uv_copy).and(unmap)?;
        }
        Ok(())
    }
}

#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
impl Drop for CudaGlInterop {
    fn drop(&mut self) {
        // SAFETY: both resources were registered in `register` and not yet
        // unregistered; best-effort cleanup, a failure here is unactionable.
        unsafe {
            let _ = ffi::cuGraphicsUnregisterResource(self.y_res);
            let _ = ffi::cuGraphicsUnregisterResource(self.uv_res);
        }
    }
}

/// `cuMemcpy2D` one NV12 plane from device memory into a mapped `cudaArray`.
///
/// # Safety
/// `resource` must be currently mapped; `src_device` must be valid device
/// memory of at least `src_pitch * upload.height` bytes in the current context.
#[cfg(any(feature = "cuda-gl", feature = "cuda-kms"))]
unsafe fn copy_plane_to_array(
    resource: ffi::CuGraphicsResource,
    src_device: u64,
    src_pitch: u32,
    upload: &GlUpload,
) -> Result<(), G2gError> {
    let mut array: ffi::CuArray = core::ptr::null_mut();
    // SAFETY: `resource` is mapped; array index 0 / mip 0 is the base image.
    unsafe {
        check(ffi::cuGraphicsSubResourceGetMappedArray(
            &mut array, resource, 0, 0,
        ))?;
    }
    let copy = ffi::CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: ffi::CU_MEMORYTYPE_DEVICE,
        src_host: core::ptr::null(),
        src_device,
        src_array: core::ptr::null_mut(),
        src_pitch: src_pitch as usize,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: ffi::CU_MEMORYTYPE_ARRAY,
        dst_host: core::ptr::null_mut(),
        dst_device: 0,
        dst_array: array,
        // Pitch is ignored for an array destination.
        dst_pitch: 0,
        width_in_bytes: upload.width_bytes,
        height: upload.height,
    };
    // SAFETY: fully-initialised array-destination CUDA_MEMCPY2D; the driver
    // only reads through it.
    unsafe { check(ffi::cu_memcpy_2d(&copy)) }
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

    #[test]
    fn nv12_byte_size_matches_three_halves_for_even_dims() {
        assert_eq!(nv12_byte_size(1920, 1080), 1920 * 1080 * 3 / 2);
        // Odd dims round the chroma planes up.
        assert_eq!(nv12_byte_size(3, 3), 9 + 8);
    }

    #[test]
    fn gl_uploads_even_dims() {
        // Luma R8 is full-res (1 byte/texel); chroma RG8 is half-res
        // (2 bytes/texel).
        let (luma, chroma) = nv12_gl_uploads(1920, 1080);
        assert_eq!(
            luma,
            GlUpload {
                width_bytes: 1920,
                height: 1080
            }
        );
        assert_eq!(
            chroma,
            GlUpload {
                width_bytes: 1920,
                height: 540
            }
        );
    }

    #[test]
    fn gl_uploads_odd_dims_round_chroma_up() {
        let (luma, chroma) = nv12_gl_uploads(3, 3);
        assert_eq!(luma.width_bytes, 3);
        assert_eq!(luma.height, 3);
        // ceil(3/2)=2 texels -> 4 bytes wide (RG8), 2 rows tall.
        assert_eq!(chroma.width_bytes, 4);
        assert_eq!(chroma.height, 2);
    }

    #[test]
    fn shaders_declare_the_nv12_sampler_pair() {
        // Lock the Appendix A contract the CUDA upload side relies on: a
        // full-res luma sampler and a half-res interleaved chroma sampler.
        assert!(FRAGMENT_SHADER_NV12.contains("uniform sampler2D y_tex"));
        assert!(FRAGMENT_SHADER_NV12.contains("uniform sampler2D uv_tex"));
        // Vertex shader feeds the fragment shader's texcoord varying.
        assert!(VERTEX_SHADER.contains("varying vec2 v_uv"));
        assert!(FRAGMENT_SHADER_NV12.contains("varying vec2 v_uv"));
    }
}
