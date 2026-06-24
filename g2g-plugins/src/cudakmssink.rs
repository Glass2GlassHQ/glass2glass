//! CUDA-GL display sink on DRM/KMS (`CudaKmsSink`): the tty / no-compositor
//! counterpart of [`crate::cudaglsink`].
//!
//! Keeps `Backend::NvdecCuda` decoded NV12 resident on the GPU and presents it
//! with no PCIe round-trip or CPU colour convert, the same CUDA-GL interop +
//! NV12->RGB shader as `CudaGlSink` (shared via [`crate::glnv12`]), but driving a
//! bare DRM/KMS display instead of a Wayland compositor: EGL renders into a GBM
//! surface whose buffers are scanned out via DRM page-flips. This is the
//! production path for an embedded / headless box with no compositor.
//!
//! ## Pipeline shape
//!
//! ```text
//! RtspSrc ─► H264Parse ─► FfmpegVideoDec(NvdecCuda) ─► CudaKmsSink
//!                                                          │
//!                                                          └─► GBM ─► DRM CRTC
//! ```
//!
//! ## Threading
//!
//! DRM, GBM and EGL are all thread-affine, so (like `CudaGlSink`) the whole stack
//! lives on a dedicated worker thread spun up at `configure_pipeline`. The sink
//! struct holds only `Send` handles (an mpsc sender plus shared atomics). The
//! decoded `OwnedCudaBuffer` is `Send`, so the frame crosses to the worker and the
//! device memory stays pinned until the worker drops it after upload.
//!
//! ## Verification status
//!
//! `cuda-kms` + Linux + NVIDIA-gated. **Authored, compiles + lints clean, but the
//! on-display run is owed**: KMS needs DRM master, which a running Wayland/X11
//! compositor holds, so it must be exercised from a bare tty (or a DRM lease),
//! not the dev session. The render half is the validated `CudaGlSink` path
//! (`glnv12`); the new, unexercised half is the GBM/EGL/DRM present, whose
//! crate-API spots are flagged with `// VERIFY:` (the EGL GBM-platform display,
//! the GBM surface pointer as the EGL native window, and the GBM bo gem-handle
//! union access for the DRM framebuffer).
//!
//! ## Constraints (v1)
//!
//! - NV12 in CUDA device memory only (`MemoryDomain::Cuda`); a system frame is
//!   rejected loud (use `CudaDownload` + `KmsSink` for that).
//! - First connected connector, its first mode, the first CRTC (same discovery as
//!   `KmsSink`); the GBM surface opens at the *video* dimensions, so the mode
//!   should match or the image is not scaled.
//! - BT.601 limited range (the `glnv12` shader); BT.709 awaits colour metadata.

use core::future::Future;
use core::num::NonZeroU32;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use std::os::fd::{AsFd, BorrowedFd};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::String;

use drm::buffer::{Buffer, DrmFourcc, Handle as BufferHandle};
use drm::control::{connector, framebuffer, Device as ControlDevice, Event, Mode, PageFlipFlags};
use drm::Device;
use gbm::{AsRaw, BufferObject, BufferObjectFlags, Device as GbmDevice, Format as GbmFormat, Surface as GbmSurface};
use khronos_egl as egl;

use g2g_core::memory::OwnedCudaBuffer;
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ClockCandidate, ClockPriority,
    ConfigureOutcome, Dim, Frame, G2gError, HardwareError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};

use crate::cuda::nv12_byte_size;
use crate::glnv12::GlState;

/// `EGL_PLATFORM_GBM_KHR` (khronos-egl 6 has no named constant for it).
const EGL_PLATFORM_GBM_KHR: egl::Enum = 0x31D7;

/// Device-buffer pool headroom the sink asks the producer to keep resident.
const CUDA_POOL_HEADROOM: usize = 3;
/// GPU upload alignment the sink requests (256 bytes, the common CUDA align).
const CUDA_ALIGN: usize = 256;

/// Worker-thread command. `Frame` carries the decoded CUDA buffer plus the
/// source-side `arrival_ns` for latency and a one-shot `ack` the worker signals
/// once the frame is presented (compositor-free, paced by the page flip).
enum WorkerCmd {
    Frame {
        buf: OwnedCudaBuffer,
        arrival_ns: u64,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    Shutdown,
}

/// Thin wrapper over `/dev/dri/cardN` implementing the `drm` device traits via
/// its owned file's borrowed fd (same pattern as `KmsSink`).
#[derive(Debug)]
struct Card(std::fs::File);

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}
impl Device for Card {}
impl ControlDevice for Card {}

impl Card {
    fn open<P: AsRef<Path>>(path: P) -> Result<Self, G2gError> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        Ok(Card(file))
    }
}

/// A single-plane XRGB8888 framebuffer view over a GBM buffer object's gem
/// handle, so `add_framebuffer` registers a DRM fb pointing at the bo (we cannot
/// use gbm's own `drm-support` impl: it targets drm 0.12, not our 0.15).
struct GbmFb {
    handle: BufferHandle,
    width: u32,
    height: u32,
    pitch: u32,
}

impl Buffer for GbmFb {
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    fn format(&self) -> DrmFourcc {
        DrmFourcc::Xrgb8888
    }
    fn pitch(&self) -> u32 {
        self.pitch
    }
    fn handle(&self) -> BufferHandle {
        self.handle
    }
}

/// Sink-side handle set. Only `Send + Sync` state lives here so the multi-thread
/// runner can move the sink between tasks.
pub struct CudaKmsSink {
    device_path: PathBuf,
    cmd_tx: Option<Sender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
    width: u32,
    height: u32,
    frames_presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
}

impl core::fmt::Debug for CudaKmsSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaKmsSink")
            .field("device_path", &self.device_path)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("frames_presented", &self.frames_presented.load(Ordering::Relaxed))
            .finish()
    }
}

impl Default for CudaKmsSink {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaKmsSink {
    pub fn new() -> Self {
        Self {
            device_path: PathBuf::from("/dev/dri/card0"),
            cmd_tx: None,
            worker: None,
            width: 0,
            height: 0,
            frames_presented: Arc::new(AtomicU64::new(0)),
            latency: Arc::new(LatencyHistogram::new()),
        }
    }

    /// Select the DRM device node (default `/dev/dri/card0`).
    pub fn with_device<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.device_path = path.into();
        self
    }

    pub fn frames_presented(&self) -> u64 {
        self.frames_presented.load(Ordering::Relaxed)
    }

    /// Glass-to-glass latency snapshot: source-side `arrival_ns` to the page
    /// flip that presents the frame. Untimed pipelines report `count = 0`.
    pub fn latency_snapshot(&self) -> LatencySnapshot {
        self.latency.snapshot()
    }

    fn shutdown(&mut self) {
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(WorkerCmd::Shutdown);
        }
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
    }
}

impl Drop for CudaKmsSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Monotonic clock the sink offers, matching the source-side `arrival_ns` epoch
/// so the latency histogram is meaningful. Same role as `KmsClock`.
#[derive(Debug)]
struct CudaKmsClock;
impl PipelineClock for CudaKmsClock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

impl AsyncElement for CudaKmsSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(ClockPriority::Provider, alloc::sync::Arc::new(CudaKmsClock)))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Pass-through at negotiation; NV12 is enforced in configure_pipeline.
        Ok(upstream_caps.clone())
    }

    /// Native NV12-only sink constraint (mirrors `CudaGlSink` / `KmsSink`): the
    /// solver lands fixed NV12 on the link; the CUDA-vs-system memory-domain check
    /// stays per-frame in `process`.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    /// Ask the producer to keep buffers in CUDA device memory so the `NvdecCuda`
    /// -> sink handoff stays on the GPU (same proposal as `CudaGlSink`).
    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        let (w, h, _) = caps.dims()?;
        let (&Dim::Fixed(w), &Dim::Fixed(h)) = (w, h) else {
            return None;
        };
        Some(AllocationParams::cuda(nv12_byte_size(w, h), CUDA_POOL_HEADROOM, CUDA_ALIGN))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = match absolute_caps {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => (*w, *h),
            _ => return Err(G2gError::CapsMismatch),
        };
        if w % 2 != 0 || h % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }

        // Mid-stream geometry change: same dims is a no-op; new dims respawn.
        if self.worker.is_some() {
            if w == self.width && h == self.height {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (tx, rx) = channel::<WorkerCmd>();
        let (ready_tx, ready_rx) = channel::<Result<(), ()>>();
        let presented = Arc::clone(&self.frames_presented);
        let latency = Arc::clone(&self.latency);
        let device_path = self.device_path.clone();

        let join = thread::Builder::new()
            .name(String::from("g2g-cudakmssink"))
            .spawn(move || {
                if let Err(e) = worker_main(device_path, w, h, rx, presented, latency, &ready_tx) {
                    std::eprintln!("g2g-cudakmssink worker error: {e:?}");
                    let _ = ready_tx.send(Err(()));
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // Wait for the worker to finish DRM/GBM/EGL/GL setup before accepting.
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => {}
            _ => {
                let _ = tx.send(WorkerCmd::Shutdown);
                let _ = join.join();
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }

        self.cmd_tx = Some(tx);
        self.worker = Some(join);
        self.width = w;
        self.height = h;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame { domain, timing, .. }) => {
                    // CUDA device memory only; a system frame means the chain
                    // forgot the NvdecCuda backend.
                    let MemoryDomain::Cuda(buf) = domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                    tx.send(WorkerCmd::Frame { buf, arrival_ns: timing.arrival_ns, ack: ack_tx })
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    // Block until the worker presents this frame (page-flip paced).
                    ack_rx.await.map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    Ok(())
                }
                PipelinePacket::CapsChanged(_) | PipelinePacket::Flush | PipelinePacket::Segment(_) => {
                    Ok(())
                }
                PipelinePacket::Eos => {
                    self.shutdown();
                    Ok(())
                }
            }
        })
    }
}

// =================================================================
// Worker thread: DRM/KMS + GBM + EGL/GL + CUDA-GL upload
// =================================================================

/// Everything the worker owns. Field declaration order is drop order, which
/// matters for the FFI teardown: the GL/CUDA interop and the locked front buffer
/// drop first, then the GBM surface (which holds a weak ref to the device), then
/// the GBM device, then the card. The EGL handles are pointer wrappers with no
/// `Drop` (EGL teardown would be manual, same as `CudaGlSink`), so their order
/// is immaterial.
struct WorkerState {
    gl: GlState,
    /// The currently-displayed bo, kept locked until the next flip retires it;
    /// dropped before `gbm_surface` since releasing it needs the surface.
    front_bo: Option<BufferObject<()>>,
    egl_surface: egl::Surface,
    _egl_context: egl::Context,
    egl_display: egl::Display,
    egl: egl::Instance<egl::Static>,
    gbm_surface: GbmSurface<()>,
    // Keep-alive: the GBM surface holds only a weak ref to the device, so the
    // device must outlive it (declared after, so dropped after).
    _gbm: GbmDevice<Card>,
    card: Card,
    crtc: drm::control::crtc::Handle,
    connector: connector::Handle,
    mode: Mode,
    /// DRM framebuffers, cached by the GBM bo's gem handle (bos recur).
    fbs: BTreeMap<u32, framebuffer::Handle>,
    crtc_set: bool,
    flip_pending: bool,
    width: u32,
    height: u32,
}

#[allow(clippy::too_many_arguments)]
fn worker_main(
    device_path: PathBuf,
    width: u32,
    height: u32,
    rx: Receiver<WorkerCmd>,
    presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    ready_tx: &Sender<Result<(), ()>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = match WorkerState::setup(&device_path, width, height) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(()));
            return Err(e);
        }
    };
    // Setup done: let configure_pipeline proceed.
    let _ = ready_tx.send(Ok(()));

    loop {
        // Block for the next frame; a long idle is not an error (live sources
        // pace themselves), so just keep waiting.
        match rx.recv_timeout(Duration::from_secs(3600)) {
            Ok(WorkerCmd::Frame { buf, arrival_ns, ack }) => {
                if let Err(e) = state.draw(&buf) {
                    std::eprintln!("g2g-cudakmssink draw error: {e:?}");
                    // Release the producer so a transient error doesn't deadlock.
                    let _ = ack.send(());
                    continue;
                }
                presented.fetch_add(1, Ordering::Relaxed);
                if arrival_ns != 0 {
                    let now = monotonic_ns();
                    if now >= arrival_ns {
                        latency.record(now - arrival_ns);
                    }
                }
                let _ = ack.send(());
            }
            Ok(WorkerCmd::Shutdown) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => continue,
        }
    }
    state.teardown();
    Ok(())
}

impl WorkerState {
    fn setup(device_path: &Path, width: u32, height: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let card = Card::open(device_path).map_err(|_| "open DRM card")?;

        // Discover the first connected connector, its first mode, the first CRTC
        // (same forgiving discovery as KmsSink).
        let res = card.resource_handles()?;
        let connector = res
            .connectors()
            .iter()
            .copied()
            .find_map(|h| {
                let info = card.get_connector(h, true).ok()?;
                (info.state() == connector::State::Connected).then_some(h)
            })
            .ok_or("no connected DRM connector")?;
        let con_info = card.get_connector(connector, true)?;
        let mode = *con_info.modes().first().ok_or("connector has no modes")?;
        let crtc = *res.crtcs().first().ok_or("no DRM CRTC")?;

        // GBM device on a second handle to the same node, and a scanout surface.
        let gbm = GbmDevice::new(Card::open(device_path).map_err(|_| "open DRM card for GBM")?)?;
        let gbm_surface = gbm.create_surface::<()>(
            width,
            height,
            GbmFormat::Xrgb8888,
            BufferObjectFlags::SCANOUT | BufferObjectFlags::RENDERING,
        )?;

        // EGL on the GBM platform.
        let egl = egl::Instance::new(egl::Static);
        // VERIFY: EGL native display is the raw gbm_device pointer under
        // EGL_PLATFORM_GBM_KHR; cast to the c_void EGL expects.
        let gbm_ptr = gbm.as_raw() as *mut core::ffi::c_void;
        // SAFETY: `gbm_ptr` is the live gbm_device (owned by `gbm`, which outlives
        // the display); GBM-platform displays only record the handle here.
        let egl_display = unsafe {
            egl.get_platform_display(EGL_PLATFORM_GBM_KHR, gbm_ptr, &[egl::ATTRIB_NONE])
        }?;
        egl.initialize(egl_display)?;
        egl.bind_api(egl::OPENGL_ES_API)?;

        let config_attribs = [
            egl::SURFACE_TYPE,
            egl::WINDOW_BIT,
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES3_BIT,
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::NONE,
        ];
        let config = egl.choose_first_config(egl_display, &config_attribs)?.ok_or("no EGL config")?;
        let context_attribs = [egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE];
        let egl_context = egl.create_context(egl_display, config, None, &context_attribs)?;

        // VERIFY: the EGL native window is the raw gbm_surface pointer; the GBM
        // surface must outlive the EGL surface (both held in WorkerState).
        let surf_ptr = gbm_surface.as_raw() as *mut core::ffi::c_void;
        // SAFETY: `surf_ptr` is the live gbm_surface for this display/config.
        let egl_surface =
            unsafe { egl.create_window_surface(egl_display, config, surf_ptr, None) }?;
        egl.make_current(egl_display, Some(egl_surface), Some(egl_surface), Some(egl_context))?;

        // glow loads GL ES entry points through eglGetProcAddress.
        // SAFETY: the loader resolves GL ES symbols against the current context.
        let gl = unsafe {
            glow::Context::from_loader_function(|s| match egl.get_proc_address(s) {
                Some(p) => p as *const core::ffi::c_void,
                None => core::ptr::null(),
            })
        };
        // SAFETY: `gl` wraps the GL ES 3 context made current above.
        let gl_state = unsafe { GlState::build(gl, width, height) }?;

        Ok(WorkerState {
            gl: gl_state,
            front_bo: None,
            egl_surface,
            _egl_context: egl_context,
            egl_display,
            egl,
            gbm_surface,
            _gbm: gbm,
            card,
            crtc,
            connector,
            mode,
            fbs: BTreeMap::new(),
            crtc_set: false,
            flip_pending: false,
            width,
            height,
        })
    }

    /// Upload + draw the NV12 frame (shared `glnv12` path), swap the GBM surface,
    /// and scan out the new buffer via `set_crtc` (first frame) or `page_flip`.
    fn draw(&mut self, buf: &OwnedCudaBuffer) -> Result<(), G2gError> {
        // CUDA upload + NV12->RGB draw (shared with the Wayland sink).
        self.gl.upload_and_draw(buf)?;

        // Present: swap renders into the GBM surface's back buffer, then lock it
        // as the new front buffer to scan out.
        self.egl
            .swap_buffers(self.egl_display, self.egl_surface)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // Drain a still-pending flip before retiring its buffer.
        if self.flip_pending {
            self.wait_for_flip()?;
        }

        // SAFETY: a buffer was just rendered, so a front buffer is available.
        let bo = unsafe {
            self.gbm_surface
                .lock_front_buffer()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?
        };
        let fb = self.framebuffer_for(&bo)?;

        if !self.crtc_set {
            self.card
                .set_crtc(self.crtc, Some(fb), (0, 0), &[self.connector], Some(self.mode))
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            self.crtc_set = true;
        } else {
            self.card
                .page_flip(self.crtc, fb, PageFlipFlags::EVENT, None)
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            self.flip_pending = true;
        }

        // The new bo is now (being) scanned out; dropping the previous front bo
        // releases it back to the GBM surface for reuse.
        self.front_bo = Some(bo);
        Ok(())
    }

    /// Get or create the DRM framebuffer pointing at this GBM bo, cached by its
    /// gem handle (the surface recycles a small set of bos).
    fn framebuffer_for(&mut self, bo: &BufferObject<()>) -> Result<framebuffer::Handle, G2gError> {
        // VERIFY: `gbm_bo_handle` is a union; the gem handle is its `u32_` arm.
        // SAFETY: NVIDIA / Mesa GBM bos use the 32-bit gem handle arm.
        let gem = unsafe {
            bo.handle().map_err(|_| G2gError::Hardware(HardwareError::Other))?.u32_
        };
        if let Some(&fb) = self.fbs.get(&gem) {
            return Ok(fb);
        }
        let pitch = bo.stride().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let handle = BufferHandle::from(
            NonZeroU32::new(gem).ok_or(G2gError::Hardware(HardwareError::Other))?,
        );
        let view = GbmFb { handle, width: self.width, height: self.height, pitch };
        let fb = self
            .card
            .add_framebuffer(&view, 24, 32)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.fbs.insert(gem, fb);
        Ok(fb)
    }

    /// Block until the kernel reports the in-flight page flip completed (drains
    /// events up to and including a `PageFlip`); same pattern as `KmsSink`.
    fn wait_for_flip(&mut self) -> Result<(), G2gError> {
        let mut empty_reads = 0u32;
        loop {
            let events = self
                .card
                .receive_events()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            let mut saw_flip = false;
            let mut saw_any = false;
            for ev in events {
                saw_any = true;
                if matches!(ev, Event::PageFlip(_)) {
                    saw_flip = true;
                }
            }
            if saw_flip {
                self.flip_pending = false;
                return Ok(());
            }
            if saw_any {
                empty_reads = 0;
            } else {
                empty_reads += 1;
                if empty_reads >= 8 {
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
            }
        }
    }

    fn teardown(&mut self) {
        if self.flip_pending {
            let _ = self.wait_for_flip();
        }
        let fbs = core::mem::take(&mut self.fbs);
        for (_, fb) in fbs {
            let _ = self.card.destroy_framebuffer(fb);
        }
        self.front_bo = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::VideoCodec;

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo { format: RawVideoFormat::Nv12, width: Dim::Fixed(w), height: Dim::Fixed(h), framerate: Rate::Any }
    }

    #[test]
    fn intercept_passes_through() {
        let sink = CudaKmsSink::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert_eq!(sink.intercept_caps(&h264), Ok(h264));
    }

    #[test]
    fn configure_rejects_non_nv12() {
        let mut sink = CudaKmsSink::new();
        let i420 = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert_eq!(sink.configure_pipeline(&i420).err(), Some(G2gError::CapsMismatch));
        assert!(sink.worker.is_none());
    }

    #[test]
    fn configure_rejects_odd_dims() {
        let mut sink = CudaKmsSink::new();
        match sink.configure_pipeline(&nv12(641, 480)) {
            Err(G2gError::CapsMismatch) => {}
            other => panic!("expected CapsMismatch on odd dims, got {other:?}"),
        }
    }

    #[test]
    fn proposes_cuda_device_memory() {
        use g2g_core::MemoryDomainKind;
        let sink = CudaKmsSink::new();
        let p = sink.propose_allocation(&nv12(1920, 1080)).expect("fixed-geometry NV12 yields a proposal");
        assert_eq!(p.domain, MemoryDomainKind::Cuda);
        assert_eq!(p.size_bytes, 1920 * 1080 * 3 / 2);
        assert_eq!(p.align, CUDA_ALIGN);
        assert_eq!(p.min_buffers, CUDA_POOL_HEADROOM);
    }
}
