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
use alloc::vec::Vec;
use std::sync::{Mutex, OnceLock, PoisonError};

use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{CudaKeepAlive, GpuDevice, GpuVendor, MemoryDomainKind, OwnedCudaBuffer};
use g2g_core::runtime::{auto_plug_domain_converters, GraphNode};
use g2g_core::{
    g2g_warn, AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, Frame, G2gError, Graph, MemoryDomain, OutputSink, PipelinePacket, PropError,
    PropValue, PropertySpec, Rate, RawVideoFormat, SystemSlice,
};

use crate::cuda_driver_ffi as ffi;
use crate::cuda_driver_ffi::check;
use crate::cudadeviceid::{
    get_cuda_device_id, set_cuda_device_id, CUDA_DEVICE_ID_PROP, DEFAULT_CUDA_DEVICE_ID,
};
use crate::gpurelease::{release_after, GpuFence};
use crate::pinnedframes::PinnedAllocation;

#[derive(Debug)]
pub(crate) struct CurrentContext;

impl CurrentContext {
    pub(crate) fn push(context: u64) -> Result<Self, G2gError> {
        // SAFETY: the caller's context is live, a frame's or a retained primary context.
        check(unsafe { ffi::cu_ctx_push_current(context as ffi::CuContext) })?;
        Ok(Self)
    }
}

impl Drop for CurrentContext {
    fn drop(&mut self) {
        let mut popped: ffi::CuContext = core::ptr::null_mut();
        // SAFETY: `push` pushed a context on this thread.
        unsafe { ffi::cu_ctx_pop_current(&mut popped) };
    }
}

#[derive(Debug)]
pub(crate) struct CudaDevice {
    ordinal: i32,
    device: ffi::CuDevice,
    context: u64,
}

impl CudaDevice {
    pub(crate) fn open(ordinal: i32) -> Result<Arc<Self>, G2gError> {
        ffi::init_driver()?;
        let mut device: ffi::CuDevice = 0;
        let mut context: ffi::CuContext = core::ptr::null_mut();
        // SAFETY: both out pointers are valid, and cuDeviceGet validates the ordinal.
        unsafe {
            check(ffi::cu_device_get(&mut device, ordinal))?;
            check(ffi::cu_device_primary_ctx_retain(&mut context, device))?;
        }
        Ok(Arc::new(Self {
            ordinal,
            device,
            context: context as u64,
        }))
    }

    pub(crate) fn bind(&self) -> Result<CurrentContext, G2gError> {
        CurrentContext::push(self.context)
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // SAFETY: balances the retain in `open`.
        unsafe { ffi::cu_device_primary_ctx_release(self.device) };
    }
}

#[derive(Debug)]
pub(crate) struct CudaEvent {
    device: Arc<CudaDevice>,
    handle: usize,
}

impl CudaEvent {
    pub(crate) fn new(device: &Arc<CudaDevice>) -> Result<Self, G2gError> {
        let _bound = device.bind()?;
        let mut handle: ffi::CuEvent = core::ptr::null_mut();
        // SAFETY: `handle` is a valid out pointer and the context is current.
        check(unsafe { ffi::cu_event_create(&mut handle, ffi::CU_EVENT_DISABLE_TIMING) })?;
        Ok(Self {
            device: Arc::clone(device),
            handle: handle as usize,
        })
    }

    pub(crate) fn synchronize(&self) -> Result<(), G2gError> {
        let _bound = self.device.bind()?;
        // SAFETY: the event lives as long as `self`.
        check(unsafe { ffi::cu_event_synchronize(self.handle as ffi::CuEvent) })
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let Ok(_bound) = self.device.bind() else {
            return;
        };
        // SAFETY: nothing uses the event after this, and cuEventDestroy defers the release past pending work.
        unsafe { ffi::cu_event_destroy(self.handle as ffi::CuEvent) };
    }
}

impl GpuFence for CudaEvent {
    fn wait(&self) -> Result<(), G2gError> {
        self.synchronize()
    }
}

// drops `buffer` once the work queued so far on `stream` in its device's primary context completes
pub fn release_after_stream(buffer: OwnedCudaBuffer, stream: u64) -> Result<(), G2gError> {
    let device = CudaDevice::open(buffer.device_ordinal)?;
    let event = CudaEvent::new(&device)?;
    {
        let _bound = device.bind()?;
        // SAFETY: the event belongs to the current context, and `stream` names a stream of that context.
        check(unsafe {
            ffi::cu_event_record(event.handle as ffi::CuEvent, stream as ffi::CuStream)
        })?;
    }
    release_after(event, Box::new(buffer));
    Ok(())
}

#[derive(Debug)]
pub(crate) struct CudaStream {
    device: Arc<CudaDevice>,
    handle: usize,
}

impl CudaStream {
    pub(crate) fn new(device: &Arc<CudaDevice>) -> Result<Self, G2gError> {
        let _bound = device.bind()?;
        let mut handle: ffi::CuStream = core::ptr::null_mut();
        // SAFETY: `handle` is a valid out pointer and the context is current.
        check(unsafe { ffi::cu_stream_create(&mut handle, ffi::CU_STREAM_NON_BLOCKING) })?;
        Ok(Self {
            device: Arc::clone(device),
            handle: handle as usize,
        })
    }

    pub(crate) fn wait_for(&self, buffer: &OwnedCudaBuffer) -> Result<(), G2gError> {
        let Some(event) = buffer.keep_alive().ready_event() else {
            return Ok(());
        };
        let _bound = self.device.bind()?;
        // SAFETY: the frame keeps its event alive, and cuStreamWaitEvent takes events from other contexts.
        check(unsafe {
            ffi::cu_stream_wait_event(self.handle as ffi::CuStream, event as ffi::CuEvent, 0)
        })
    }

    pub(crate) fn synchronize(&self) -> Result<(), G2gError> {
        let _bound = self.device.bind()?;
        // SAFETY: the stream belongs to the current context.
        check(unsafe { ffi::cu_stream_synchronize(self.handle as ffi::CuStream) })
    }

    pub(crate) fn record(&self, event: &CudaEvent) -> Result<(), G2gError> {
        let _bound = self.device.bind()?;
        // SAFETY: the stream and the event belong to the current context.
        check(unsafe {
            ffi::cu_event_record(event.handle as ffi::CuEvent, self.handle as ffi::CuStream)
        })
    }

    unsafe fn copy_2d_async(&self, copy: &ffi::CudaMemcpy2D) -> Result<(), G2gError> {
        let _bound = self.device.bind()?;
        // SAFETY: the caller keeps both ends of the copy alive until the stream passes it.
        check(unsafe { ffi::cu_memcpy_2d_async(copy, self.handle as ffi::CuStream) })
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let Ok(_bound) = self.device.bind() else {
            return;
        };
        // SAFETY: nothing queues work on the stream after this.
        unsafe { ffi::cu_stream_destroy(self.handle as ffi::CuStream) };
    }
}

#[derive(Debug)]
pub(crate) struct CudaPinnedMemory {
    device: Arc<CudaDevice>,
    pointer: usize,
    size: usize,
}

impl PinnedAllocation for CudaPinnedMemory {
    type Device = Arc<CudaDevice>;

    fn allocate(device: &Arc<CudaDevice>, size: usize) -> Result<Self, G2gError> {
        let _bound = device.bind()?;
        let mut pointer: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: `pointer` is a valid out pointer and the context is current.
        check(unsafe { ffi::cu_mem_alloc_host(&mut pointer, size) })?;
        Ok(Self {
            device: Arc::clone(device),
            pointer: pointer as usize,
            size,
        })
    }

    fn size(&self) -> usize {
        self.size
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: the allocation is `size` bytes and `&mut self` makes this the only view of it.
        unsafe { core::slice::from_raw_parts_mut(self.pointer as *mut u8, self.size) }
    }
}

impl Drop for CudaPinnedMemory {
    fn drop(&mut self) {
        let Ok(_bound) = self.device.bind() else {
            return;
        };
        // SAFETY: the allocation came from `allocate` in this context and nothing uses it after this.
        unsafe { ffi::cu_mem_free_host(self.pointer as *mut core::ffi::c_void) };
    }
}

type PinnedFrames = crate::pinnedframes::PinnedFrames<CudaPinnedMemory>;
type PinnedOffer = crate::pinnedframes::PinnedOffer<CudaPinnedMemory>;

/// Pass-through transform that copies CUDA device-memory NV12 frames to
/// system memory. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::cuda::CudaDownload;
///
/// let download = CudaDownload::new();
/// assert_eq!(download.downloaded(), 0);
/// ```
#[derive(Debug, Default)]
pub struct CudaDownload {
    configured: bool,
    /// Frames copied device->host (the `MemoryDomain::Cuda` inputs).
    downloaded: u64,
    /// Frames forwarded untouched (already system-memory inputs).
    forwarded: u64,
    to_host: CudaToHostCopier,
}

#[derive(Debug, Default)]
pub(crate) struct CudaToHostCopier {
    gpu: Option<DownloadGpu>,
}

#[derive(Debug)]
struct DownloadGpu {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    pinned: Arc<PinnedFrames>,
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

impl CudaToHostCopier {
    fn gpu_for(&mut self, ordinal: i32) -> Result<&DownloadGpu, G2gError> {
        let gpu = match self.gpu.take() {
            Some(gpu) if gpu.device.ordinal == ordinal => gpu,
            _ => {
                let device = CudaDevice::open(ordinal)?;
                DownloadGpu {
                    stream: CudaStream::new(&device)?,
                    pinned: PinnedOffer::new(Arc::clone(&device)).frames,
                    device,
                }
            }
        };
        Ok(self.gpu.insert(gpu))
    }

    pub(crate) async fn download(
        &mut self,
        buffer: OwnedCudaBuffer,
        format: RawVideoFormat,
    ) -> Result<SystemSlice, G2gError> {
        let (luma, chroma, total) = nv12_plane_copies(
            buffer.width,
            buffer.height,
            buffer.luma_pitch,
            buffer.chroma_pitch,
            format,
        )
        .ok_or(G2gError::CapsMismatch)?;
        let gpu = self.gpu_for(buffer.device_ordinal)?;
        let mut output = gpu.pinned.lease(total)?;
        let host = output.as_mut_slice().as_mut_ptr();
        gpu.stream.wait_for(&buffer)?;
        // SAFETY: the copies stay inside the frame's planes and the leased slice, and both stay alive until the stream passes them below.
        let queued = unsafe {
            gpu.stream
                .copy_2d_async(&device_to_host(&luma, buffer.luma_ptr, host))
                .and_then(|()| {
                    gpu.stream
                        .copy_2d_async(&device_to_host(&chroma, buffer.chroma_ptr, host))
                })
        };
        let copied = queued.and_then(|()| {
            let copied = CudaEvent::new(&gpu.device)?;
            gpu.stream.record(&copied)?;
            Ok(copied)
        });
        let copied = match copied {
            Ok(copied) => copied,
            Err(error) => {
                // a copy queued before the failure still writes the leased slice
                let _ = gpu.stream.synchronize();
                return Err(error);
            }
        };
        g2g_core::offload::run_blocking(move || {
            let _source = buffer;
            copied.synchronize().map(|()| output)
        })
        .await
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
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
    })
}

impl AsyncElement for CudaDownload {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "CUDA download",
            "Filter/Converter/Video/GPU",
            "Copies a CUDA device frame back to system memory",
            crate::AARON_BOXER,
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // The download keeps the caps unchanged (only the domain changes), so
        // narrow upstream against NV12. The solver uses the `Identity`
        // constraint below instead.
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
                    let Frame {
                        domain,
                        timing,
                        sequence,
                        meta,
                    } = frame;
                    let domain = match domain {
                        MemoryDomain::Cuda(buffer) => {
                            let bytes = self.to_host.download(buffer, RawVideoFormat::Nv12).await?;
                            self.downloaded += 1;
                            MemoryDomain::System(bytes)
                        }
                        // Already in system memory (software / cuvid backend);
                        // forward untouched.
                        domain => {
                            self.forwarded += 1;
                            domain
                        }
                    };
                    let out_frame = Frame {
                        domain,
                        timing,
                        sequence,
                        meta,
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

type FreeAllocations = Arc<Mutex<Vec<PitchedAllocation>>>;

#[derive(Debug)]
struct PitchedAllocation {
    device: Arc<CudaDevice>,
    pointer: u64,
    pitch: usize,
}

impl Drop for PitchedAllocation {
    fn drop(&mut self) {
        let Ok(_bound) = self.device.bind() else {
            return;
        };
        // SAFETY: the allocation came from `FramePool::acquire` in this context and nothing uses it after this.
        unsafe { ffi::cu_mem_free(self.pointer) };
    }
}

#[derive(Debug)]
struct FramePool {
    width: u32,
    height: u32,
    free: FreeAllocations,
}

impl FramePool {
    fn acquire(&self, device: &Arc<CudaDevice>) -> Result<PitchedAllocation, G2gError> {
        let recycled = self
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        if let Some(allocation) = recycled {
            return Ok(allocation);
        }
        let width = self.width as usize;
        let rows = self.height as usize + self.height.div_ceil(2) as usize;
        // an odd width makes the interleaved chroma row one byte wider than the luma row
        let widest_row = 2 * width.div_ceil(2);
        let _bound = device.bind()?;
        let mut pointer: u64 = 0;
        let mut pitch: usize = 0;
        // nvenc rejects a tight pitch, and an element size of 16 gives the alignment it takes
        // SAFETY: both out pointers are valid and the context is current.
        check(unsafe { ffi::cu_mem_alloc_pitch(&mut pointer, &mut pitch, widest_row, rows, 16) })?;
        Ok(PitchedAllocation {
            device: Arc::clone(device),
            pointer,
            pitch,
        })
    }
}

#[derive(Debug)]
struct UploadedFrame {
    allocation: Option<PitchedAllocation>,
    recycle_into: FreeAllocations,
    ready: CudaEvent,
}

impl CudaKeepAlive for UploadedFrame {
    fn ready_event(&self) -> Option<u64> {
        Some(self.ready.handle as u64)
    }

    fn wait_until_ready(&self) -> Result<(), G2gError> {
        self.ready.synchronize()
    }
}

impl Drop for UploadedFrame {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            self.recycle_into
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(allocation);
        }
    }
}

#[derive(Debug)]
struct UploadGpu {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    pool: Option<FramePool>,
}

fn upload_pinned(
    gpu: &mut UploadGpu,
    staging: SystemSlice,
    width: u32,
    height: u32,
) -> Result<OwnedCudaBuffer, G2gError> {
    if gpu
        .pool
        .as_ref()
        .is_none_or(|pool| (pool.width, pool.height) != (width, height))
    {
        gpu.pool = Some(FramePool {
            width,
            height,
            free: Arc::default(),
        });
    }
    let pool = gpu.pool.as_ref().ok_or(G2gError::NotConfigured)?;
    let allocation = pool.acquire(&gpu.device)?;
    let pitch = allocation.pitch;
    let pitch_u32 = u32::try_from(pitch).map_err(|_| G2gError::CapsMismatch)?;
    let (luma, chroma, total) =
        nv12_plane_copies(width, height, pitch_u32, pitch_u32, RawVideoFormat::Nv12)
            .ok_or(G2gError::CapsMismatch)?;
    if staging.as_slice().len() < total {
        return Err(G2gError::CapsMismatch);
    }
    let luma_pointer = allocation.pointer;
    let chroma_pointer = luma_pointer + (pitch * height as usize) as u64;
    let host = staging.as_slice().as_ptr();
    // SAFETY: the copies stay inside the staging slice and the allocation, `release_after` keeps the slice alive until the stream passes them, and the frame keeps the allocation.
    let queued = unsafe {
        gpu.stream
            .copy_2d_async(&host_to_device(&luma, host, luma_pointer))
            .and_then(|()| {
                gpu.stream
                    .copy_2d_async(&host_to_device(&chroma, host, chroma_pointer))
            })
    };
    let copied = queued.and_then(|()| {
        let copied = CudaEvent::new(&gpu.device)?;
        gpu.stream.record(&copied)?;
        Ok(copied)
    });
    match copied {
        Ok(copied) => release_after(copied, Box::new(staging)),
        Err(error) => {
            // a copy queued before the failure still reads the staging slice
            let _ = gpu.stream.synchronize();
            return Err(error);
        }
    }
    let ready = CudaEvent::new(&gpu.device)?;
    gpu.stream.record(&ready)?;
    Ok(OwnedCudaBuffer::new(
        luma_pointer,
        chroma_pointer,
        pitch_u32,
        pitch_u32,
        width,
        height,
        gpu.device.context,
        gpu.device.ordinal,
        Arc::new(UploadedFrame {
            allocation: Some(allocation),
            recycle_into: Arc::clone(&pool.free),
            ready,
        }),
    ))
}

/// Pass-through transform that uploads system-memory NV12 frames to CUDA device
/// memory (`System -> Cuda`), the mirror of [`CudaDownload`]. It is the
/// host->device converter the auto-plug registry uses, and on its own unblocks
/// feeding a CPU-side NV12 stream into [`NvEnc`](crate::nvenc::NvEnc), which
/// ingests `MemoryDomain::Cuda` only. A frame already in CUDA memory passes
/// through untouched, so the element is a safe no-op on a GPU-resident path.
///
/// The copy runs on the element's own stream in the device's primary context,
/// from pinned host memory: either a buffer the element offered upstream through
/// `propose_allocation`, or a staging buffer the frame is copied into. The frame
/// is pushed once the copy is queued, and its keep-alive carries the event the
/// copy records, so the next frame's copy overlaps the consumer's work.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::cuda::CudaUpload;
///
/// let upload = CudaUpload::new();
/// assert_eq!(upload.uploaded(), 0);
/// ```
#[derive(Debug)]
pub struct CudaUpload {
    configured: bool,
    width: u32,
    height: u32,
    pinned: OnceLock<PinnedOffer>,
    gpu: Option<UploadGpu>,
    uploaded: u64,
    forwarded: u64,
    /// Device the upload runs on, carried onto every uploaded frame so a
    /// consumer knows which GPU the surface lives on.
    cuda_device_id: i32,
}

impl Default for CudaUpload {
    fn default() -> Self {
        Self {
            configured: false,
            width: 0,
            height: 0,
            pinned: OnceLock::new(),
            gpu: None,
            uploaded: 0,
            forwarded: 0,
            cuda_device_id: DEFAULT_CUDA_DEVICE_ID,
        }
    }
}

impl CudaUpload {
    pub fn new() -> Self {
        Self::default()
    }

    /// Upload onto CUDA device `ordinal` instead of device 0. Read when the
    /// device opens, at configure or at the first allocation query; a later
    /// change is refused.
    pub fn with_cuda_device_id(mut self, ordinal: i32) -> Self {
        self.cuda_device_id = ordinal;
        self
    }

    /// Frames copied host->device into CUDA memory so far.
    pub fn uploaded(&self) -> u64 {
        self.uploaded
    }

    /// Frames already in CUDA memory, passed through untouched.
    pub fn forwarded(&self) -> u64 {
        self.forwarded
    }

    fn pinned_offer(&self) -> Result<&PinnedOffer, G2gError> {
        if let Some(offer) = self.pinned.get() {
            return Ok(offer);
        }
        let offer = PinnedOffer::new(CudaDevice::open(self.cuda_device_id)?);
        Ok(self.pinned.get_or_init(|| offer))
    }

    fn gpu(&mut self) -> Result<&mut UploadGpu, G2gError> {
        let gpu = match self.gpu.take() {
            Some(gpu) => gpu,
            None => {
                let device = Arc::clone(&self.pinned_offer()?.frames.device);
                UploadGpu {
                    stream: CudaStream::new(&device)?,
                    device,
                    pool: None,
                }
            }
        };
        Ok(self.gpu.insert(gpu))
    }
}

/// The upload context's device, so a `gst-launch` line can pin the GPU.
static CUDA_UPLOAD_PROPS: &[PropertySpec] = &[CUDA_DEVICE_ID_PROP];

impl AsyncElement for CudaUpload {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "CUDA upload",
            "Filter/Converter/Video/GPU",
            "Copies a system frame into CUDA device memory",
            crate::AARON_BOXER,
        )
    }

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
        // opening the device here fails the configure on a host without an NVIDIA GPU
        self.pinned_offer()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        CUDA_UPLOAD_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "cuda-device-id" => set_cuda_device_id(
                &mut self.cuda_device_id,
                self.pinned.get().is_some(),
                &value,
            ),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "cuda-device-id" => Some(get_cuda_device_id(self.cuda_device_id)),
            _ => None,
        }
    }

    /// Emits CUDA NV12 (the uploaded frames; a passed-through Cuda frame keeps its
    /// own domain), so a downstream link is a GPU link.
    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Cuda
    }

    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        let offer = match self.pinned_offer() {
            Ok(offer) => offer,
            Err(error) => {
                g2g_warn!(
                    Target::category(short_type_name::<Self>()),
                    "no pinned host buffers to offer upstream, opening the CUDA device failed: {error:?}"
                );
                return None;
            }
        };
        Some(AllocationParams {
            domain: MemoryDomainKind::System,
            accepts: self.input_domains(),
            host_allocator: Some(offer.allocator.clone()),
            ..AllocationParams::default()
        })
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
                    let Frame {
                        domain,
                        timing,
                        sequence,
                        meta,
                    } = frame;
                    let domain = match domain {
                        MemoryDomain::System(slice) => {
                            let frames = Arc::clone(&self.pinned_offer()?.frames);
                            let staging = if frames.allocated(slice.as_slice()) {
                                slice
                            } else {
                                let mut staging = frames.lease(slice.as_slice().len())?;
                                staging.as_mut_slice().copy_from_slice(slice.as_slice());
                                staging
                            };
                            let (width, height) = (self.width, self.height);
                            let buffer = upload_pinned(self.gpu()?, staging, width, height)?;
                            self.uploaded += 1;
                            MemoryDomain::Cuda(buffer)
                        }
                        // Already on the GPU: forward untouched.
                        domain => {
                            self.forwarded += 1;
                            domain
                        }
                    };
                    let out_frame = Frame {
                        domain,
                        timing,
                        sequence,
                        meta,
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

/// The g2g memory-domain converter for a `(from, to)` pair, or `None` when g2g
/// has none (the auto-plug then leaves the edge to fail loud). Covers the
/// CUDA<->System pair: [`CudaDownload`] (`Cuda -> System`) and [`CudaUpload`]
/// (`System -> Cuda`), plus `Cuda -> WgpuTexture` through the zero-copy
/// external-memory bridge where that is built in. The factory the domain auto-plug
/// calls; see [`auto_plug_cuda_converters`].
pub fn cuda_domain_converter(
    from: MemoryDomainKind,
    to: MemoryDomainKind,
    device: Option<GpuDevice>,
) -> Option<GraphNode> {
    match (from, to) {
        (MemoryDomainKind::Cuda, MemoryDomainKind::System) => {
            Some(GraphNode::element(CudaDownload::new()))
        }
        (MemoryDomainKind::System, MemoryDomainKind::Cuda) => {
            let ordinal = device
                .filter(|device| device.vendor == GpuVendor::Nvidia)
                .map_or(DEFAULT_CUDA_DEVICE_ID, |device| device.ordinal);
            Some(GraphNode::element(
                CudaUpload::new().with_cuda_device_id(ordinal),
            ))
        }
        // Keeps a decoded NVDEC frame on the GPU all the way to a wgpu consumer
        // (preprocess, present), instead of the PCIe download a System hop costs.
        #[cfg(all(target_os = "linux", feature = "cuda-wgpu"))]
        (MemoryDomainKind::Cuda, MemoryDomainKind::WgpuTexture) => {
            Some(GraphNode::element(crate::cudatowgpu::CudaToWgpu::new()))
        }
        _ => None,
    }
}

/// Auto-plug CUDA<->System domain converters into `graph`: splices a
/// [`CudaUpload`] / [`CudaDownload`] on any edge whose producer and consumer
/// cannot agree on a memory domain. E.g. a System NV12 source feeding `NvEnc`
/// (CUDA-only input) gains a `CudaUpload`; a CUDA decoder fanned out to a System
/// sink gains a `CudaDownload` on that branch. A no-op where domains already
/// agree, so it is safe to call on any graph before `run_graph`.
pub fn auto_plug_cuda_converters(graph: Graph<GraphNode>) -> Graph<GraphNode> {
    auto_plug_domain_converters(graph, &cuda_domain_converter)
}

/// Per-plane parameters for the `cuMemcpy2D` of one NV12 plane between device
/// memory and a packed host buffer. Pure geometry, so the layout is
/// unit-testable without a GPU.
#[derive(Debug, PartialEq, Eq)]
struct PlaneCopy {
    /// Row pitch in bytes in device memory (the decoder's or allocator's alignment).
    device_pitch: usize,
    /// Byte offset of this plane in the packed host buffer.
    host_offset: usize,
    /// Row pitch in bytes in the packed host buffer (equals `width_bytes`).
    host_pitch: usize,
    /// Bytes copied per row.
    width_bytes: usize,
    /// Number of rows.
    height: usize,
}

/// Packed NV12 buffer size in bytes for a `width` x `height` frame. Used by
/// `CudaGlSink` to size its allocation proposal.
pub fn nv12_byte_size(width: u32, height: u32) -> usize {
    crate::pixel::frame_byte_size(RawVideoFormat::Nv12, width, height)
}

/// The two plane copies (luma, then interleaved chroma) and the total packed
/// buffer length for a semi-planar `format` frame (NV12 or P010) of `width` x
/// `height` whose device planes have the given pitches. `None` on overflow.
fn nv12_plane_copies(
    width: u32,
    height: u32,
    luma_pitch: u32,
    chroma_pitch: u32,
    format: RawVideoFormat,
) -> Option<(PlaneCopy, PlaneCopy, usize)> {
    let (w, h) = (width as usize, height as usize);
    let copy = |plane: usize, device_pitch: u32| {
        let (host_offset, width_bytes, rows) = crate::pixel::tight_plane(format, plane, w, h)?;
        Some(PlaneCopy {
            device_pitch: device_pitch as usize,
            host_offset,
            host_pitch: width_bytes,
            width_bytes,
            height: rows,
        })
    };
    let total = format.unpadded_frame_bytes(width, height)?;
    Some((
        copy(0, luma_pitch)?,
        copy(1, chroma_pitch)?,
        usize::try_from(total).ok()?,
    ))
}

fn device_to_host(plane: &PlaneCopy, device: u64, host_base: *mut u8) -> ffi::CudaMemcpy2D {
    ffi::CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: ffi::CU_MEMORYTYPE_DEVICE,
        src_host: core::ptr::null(),
        src_device: device,
        src_array: core::ptr::null_mut(),
        src_pitch: plane.device_pitch,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: ffi::CU_MEMORYTYPE_HOST,
        dst_host: host_base.wrapping_add(plane.host_offset).cast(),
        dst_device: 0,
        dst_array: core::ptr::null_mut(),
        dst_pitch: plane.host_pitch,
        width_in_bytes: plane.width_bytes,
        height: plane.height,
    }
}

fn host_to_device(plane: &PlaneCopy, host_base: *const u8, device: u64) -> ffi::CudaMemcpy2D {
    ffi::CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: ffi::CU_MEMORYTYPE_HOST,
        src_host: host_base.wrapping_add(plane.host_offset).cast(),
        src_device: 0,
        src_array: core::ptr::null_mut(),
        src_pitch: plane.host_pitch,
        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: ffi::CU_MEMORYTYPE_DEVICE,
        dst_host: core::ptr::null_mut(),
        dst_device: device,
        dst_array: core::ptr::null_mut(),
        dst_pitch: plane.device_pitch,
        width_in_bytes: plane.width_bytes,
        height: plane.height,
    }
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
            check(ffi::cu_graphics_gl_register_image(
                &mut y_res,
                y_tex,
                ffi::GL_TEXTURE_2D,
                ffi::CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
            ))?;
            if let Err(e) = check(ffi::cu_graphics_gl_register_image(
                &mut uv_res,
                uv_tex,
                ffi::GL_TEXTURE_2D,
                ffi::CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
            )) {
                // Roll back the luma registration so we don't leak it.
                let _ = ffi::cu_graphics_unregister_resource(y_res);
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
            check(ffi::cu_graphics_map_resources(
                2,
                resources.as_mut_ptr(),
                core::ptr::null_mut(),
            ))?;
            let y_copy = copy_plane_to_array(self.y_res, buf.luma_ptr, buf.luma_pitch, &luma);
            let uv_copy =
                copy_plane_to_array(self.uv_res, buf.chroma_ptr, buf.chroma_pitch, &chroma);
            let unmap = check(ffi::cu_graphics_unmap_resources(
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
            let _ = ffi::cu_graphics_unregister_resource(self.y_res);
            let _ = ffi::cu_graphics_unregister_resource(self.uv_res);
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
        check(ffi::cu_graphics_sub_resource_get_mapped_array(
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
    use g2g_core::{FrameTiming, PushOutcome};

    #[derive(Debug, Default)]
    struct StreamGate {
        open: Mutex<bool>,
        opened: std::sync::Condvar,
    }

    unsafe extern "C" fn wait_for_gate(gate: *mut core::ffi::c_void) {
        // SAFETY: the test keeps the gate alive until the stream passes this callback.
        let gate = unsafe { &*(gate as *const StreamGate) };
        let mut open = gate.open.lock().unwrap();
        while !*open {
            open = gate.opened.wait(open).unwrap();
        }
    }

    #[derive(Debug)]
    struct DropFlag(Arc<std::sync::atomic::AtomicBool>);

    impl CudaKeepAlive for DropFlag {}

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn a_frame_released_after_a_stream_outlives_the_work_queued_on_it() {
        let device = match CudaDevice::open(0) {
            Ok(device) => device,
            Err(error) => {
                std::eprintln!("skipping: no CUDA device 0: {error:?}");
                return;
            }
        };
        let stream = CudaStream::new(&device).unwrap();
        let gate = Arc::new(StreamGate::default());
        {
            let _bound = device.bind().unwrap();
            // SAFETY: `gate` outlives the callback, which the test unblocks before it ends.
            check(unsafe {
                ffi::cu_launch_host_func(
                    stream.handle as ffi::CuStream,
                    wait_for_gate,
                    Arc::as_ptr(&gate) as *mut core::ffi::c_void,
                )
            })
            .unwrap();
        }
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let frame =
            OwnedCudaBuffer::new(0, 0, 0, 0, 0, 0, 0, 0, Arc::new(DropFlag(dropped.clone())));
        release_after_stream(frame, stream.handle as u64).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(50));
        let dropped_while_blocked = dropped.load(std::sync::atomic::Ordering::SeqCst);
        *gate.open.lock().unwrap() = true;
        gate.opened.notify_all();
        stream.synchronize().unwrap();
        crate::gpurelease::wait_until_released();

        assert!(!dropped_while_blocked);
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[derive(Default)]
    struct FrameSink {
        frames: Vec<Frame>,
    }

    impl OutputSink for FrameSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            if let Some(PipelinePacket::DataFrame(frame)) = packet.take() {
                self.frames.push(frame);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn run_one<E: AsyncElement>(element: &mut E, domain: MemoryDomain) -> MemoryDomain {
        let mut sink = FrameSink::default();
        let frame = Frame::new(domain, FrameTiming::default(), 0);
        g2g_core::runtime::block_on(element.process(PipelinePacket::DataFrame(frame), &mut sink))
            .unwrap();
        let [frame] = <[Frame; 1]>::try_from(sink.frames).unwrap();
        frame.domain
    }

    #[test]
    fn a_frame_in_offered_pinned_memory_uploads_without_a_staging_copy() {
        const WIDTH: u32 = 64;
        const HEIGHT: u32 = 48;
        let caps = nv12(WIDTH, HEIGHT);
        let mut upload = CudaUpload::new();
        if let Err(error) = upload.configure_pipeline(&caps) {
            std::eprintln!("skipping: no CUDA device 0: {error:?}");
            return;
        }
        let len = nv12_byte_size(WIDTH, HEIGHT);
        let bytes: Vec<u8> = (0..len).map(|index| (index * 7) as u8).collect();
        let offered = upload
            .propose_allocation(&caps)
            .and_then(|params| params.host_allocator)
            .expect("cudaupload offers pinned buffers upstream");
        let mut pinned = offered.allocate(len).unwrap();
        pinned.as_mut_slice().copy_from_slice(&bytes);

        let on_gpu = run_one(&mut upload, MemoryDomain::System(pinned));
        let mut download = CudaDownload::new();
        download.configure_pipeline(&caps).unwrap();
        let back = run_one(&mut download, on_gpu);

        assert_eq!(back.as_system_slice(), Some(bytes.as_slice()));
        let allocations = upload
            .pinned
            .get()
            .unwrap()
            .frames
            .allocations
            .lock()
            .unwrap()
            .len();
        assert_eq!(allocations, 1, "no staging buffer");
    }

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
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
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
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
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        };
        assert_eq!(d.intercept_caps(&i420), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn plane_copies_even_dims_pack_to_3_2() {
        // 1920x1080 with the decoder aligning the device pitch up to 2048.
        let (luma, chroma, total) =
            nv12_plane_copies(1920, 1080, 2048, 2048, RawVideoFormat::Nv12).unwrap();
        assert_eq!(
            luma,
            PlaneCopy {
                device_pitch: 2048,
                host_offset: 0,
                host_pitch: 1920,
                width_bytes: 1920,
                height: 1080,
            }
        );
        assert_eq!(
            chroma,
            PlaneCopy {
                device_pitch: 2048,
                host_offset: 1920 * 1080,
                host_pitch: 1920,
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
        let (luma, chroma, total) =
            nv12_plane_copies(3, 3, 256, 256, RawVideoFormat::Nv12).unwrap();
        assert_eq!(luma.width_bytes, 3);
        assert_eq!(luma.height, 3);
        assert_eq!(luma.host_pitch, 3);
        // ceil(3/2)=2 -> 2*2=4 bytes wide, 2 rows tall.
        assert_eq!(chroma.width_bytes, 4);
        assert_eq!(chroma.height, 2);
        assert_eq!(chroma.host_offset, 9);
        assert_eq!(total, 9 + 8);
    }

    #[test]
    fn plane_copies_scale_for_ten_bit_samples() {
        // P010: the same sample counts at 2 bytes each (NVDEC pitch in bytes).
        let (luma, chroma, total) =
            nv12_plane_copies(640, 480, 1536, 1536, RawVideoFormat::P010).unwrap();
        assert_eq!(luma.width_bytes, 1280);
        assert_eq!(luma.host_pitch, 1280);
        assert_eq!(chroma.width_bytes, 1280);
        assert_eq!(chroma.height, 240);
        assert_eq!(chroma.host_offset, 640 * 480 * 2);
        assert_eq!(total, 640 * 480 * 3);
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
}
