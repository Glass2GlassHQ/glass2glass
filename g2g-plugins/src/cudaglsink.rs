//! CUDA-GL display sink (C3 Phase 3, step 2): the zero-copy-ish payoff.
//!
//! Keeps `Backend::NvdecCuda` decoded NV12 resident on the GPU and presents it
//! without a PCIe round-trip or CPU colour convert. Per frame: CUDA copies the
//! two NV12 planes device->`cudaArray` into two registered GL textures
//! (`CudaGlInterop`), then a fragment shader converts NV12->RGB on the GPU and
//! presents via `eglSwapBuffers` (DESIGN-C3-cuda.md §3.2, Appendix A). Not
//! literally zero-copy (one device->device copy into the texture), but it
//! removes the device->host copy `CudaDownload` pays and the CPU convert
//! `WaylandSink` pays.
//!
//! ## Pipeline shape
//!
//! ```text
//! RtspSrc ─► H264Parse ─► FfmpegH264Dec(NvdecCuda) ─► CudaGlSink
//!                                                          │
//!                                                          └─► EGL/GL window
//! ```
//!
//! ## Threading
//!
//! GL and Wayland are both single-thread-affine, so (like [`WaylandSink`]) all
//! of it lives on a dedicated worker thread spun up at `configure_pipeline`.
//! The sink struct holds only `Send` handles (a `calloop` channel sender plus
//! shared atomics), so the runner can move it between executor tasks. The
//! decoded `OwnedCudaBuffer` is `Send` (its keep-alive owner is), so the frame
//! crosses to the worker and the device memory stays pinned until the worker
//! drops it after upload.
//!
//! ## Verification status
//!
//! `cuda-gl` + Linux + NVIDIA-gated. Validated on the RTX 3060 host (M252):
//! compiles + lints clean, and the `cudagl_smoke` on-display e2e presents real
//! `NvdecCuda` frames through the CUDA-GL path (60 frames, glass-to-glass
//! p50 ~8 ms on a GNOME Wayland session). The off-host draft needed only two
//! adjustments at first compile, the `khronos-egl` 6 `get_display` now being
//! `unsafe`, and importing `alloc::string::ToString`; the crate-API spots that
//! were flagged `// VERIFY:` (the `wayland-client` 0.31 raw-pointer accessors,
//! glow 0.17's `tex_image_2d` pixel-source parameter, the `eglGetProcAddress`
//! cast) all held. On a hybrid iGPU+NVIDIA host the GL context must be forced
//! onto the NVIDIA GPU (`__NV_PRIME_RENDER_OFFLOAD` / `__EGL_VENDOR_LIBRARY_FILENAMES`)
//! or `cuGraphicsGLRegisterImage` fails. The `cudagl_vs_wayland` A/B benchmark
//! (M253) measured this device-resident path at **10.7x lower present latency**
//! than the `NvdecCuvid -> WaylandSink` baseline at 1080p (p50 ~8 ms vs ~90 ms):
//! the GPU NV12->RGB convert replaces the baseline's per-frame CPU convert +
//! `wl_shm` upload.
//!
//! ## Constraints (v1)
//!
//! - NV12 in CUDA device memory only (`MemoryDomain::Cuda`); a system-memory
//!   frame is rejected loud (use `CudaDownload` + `WaylandSink` for that).
//! - No scaling: the window opens at the video dimensions; the compositor
//!   letterboxes/clips if it resizes us.
//! - BT.601 limited range (Appendix A shader); BT.709 awaits colour metadata.

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;

use smithay_client_toolkit::reexports::calloop::channel::{channel, Sender as CalloopSender};

use crate::worker_ready::Handshake;
use g2g_core::element::QosMessage;
use g2g_core::memory::OwnedCudaBuffer;
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AllocationParams, AsyncElement, BusHandle, Caps, CapsConstraint, CapsSet, ClockCandidate,
    ClockPriority, ClockSync, ConfigureOutcome, Dim, Frame, G2gError, HardwareError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, PresentationPacer, PropError, PropValue,
    PropertySpec, Rate, RawVideoFormat, PACING_PROPERTIES,
};

use crate::clock::wait_to_present;
use crate::cuda::nv12_byte_size;
use crate::glnv12::{GlMode, GlState};
use crate::glwindow::{run_gl_window, FramePresenter, WindowParams, WorkerChannels, WorkerCmd};

/// Device-buffer pool headroom the sink asks the producer to keep resident:
/// the frame in flight on the GL thread plus the one the runner link holds, so
/// the decoder's hwframe pool does not starve under live pacing.
const CUDA_POOL_HEADROOM: usize = 3;

/// GPU upload alignment the sink requests (256 bytes is the common CUDA / NVENC
/// surface alignment).
const CUDA_ALIGN: usize = 256;

/// Worker thread name, also the prefix on the worker's error lines.
const WORKER_NAME: &str = "g2g-cudaglsink";

/// Sink-side handle set. Only `Send + Sync` state lives here so the
/// multi-thread runner can move the sink between tasks. The worker's frame
/// payload is the decoded CUDA buffer, still device-resident.
pub struct CudaGlSink {
    title: String,
    app_id: String,
    cmd_tx: Option<CalloopSender<WorkerCmd<OwnedCudaBuffer>>>,
    worker: Option<JoinHandle<()>>,
    width: u32,
    height: u32,
    frames_presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    /// PTS pacing + QoS late-drop, on top of the worker's swap-paced ack: idle
    /// until the runner hands over a clock, and the default lateness bound never
    /// drops.
    pacer: PresentationPacer,
}

impl core::fmt::Debug for CudaGlSink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaGlSink")
            .field("title", &self.title)
            .field("width", &self.width)
            .field("height", &self.height)
            .field(
                "frames_presented",
                &self.frames_presented.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl Default for CudaGlSink {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaGlSink {
    pub fn new() -> Self {
        Self {
            title: String::from("glass2glass"),
            app_id: String::from("io.glass2glass.CudaGlSink"),
            cmd_tx: None,
            worker: None,
            width: 0,
            height: 0,
            frames_presented: Arc::new(AtomicU64::new(0)),
            latency: Arc::new(LatencyHistogram::new()),
            pacer: PresentationPacer::new(),
        }
    }

    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = title.into();
        self
    }

    pub fn with_app_id<S: Into<String>>(mut self, app_id: S) -> Self {
        self.app_id = app_id.into();
        self
    }

    /// QoS late-drop bound: once PTS pacing is engaged, a frame past its
    /// deadline by more than `ns` is dropped instead of presented late, so the
    /// sink catches up. The default (`u64::MAX`) never drops.
    pub fn with_max_lateness_ns(mut self, ns: u64) -> Self {
        self.pacer.set_max_lateness_ns(ns);
        self
    }

    /// Post a running-stats `Qos` report every `ns` of clock time while frames
    /// present, on top of the per-drop reports. `0` (the default) reports only
    /// drops.
    pub fn with_qos_interval_ns(mut self, ns: u64) -> Self {
        self.pacer.set_report_interval_ns(ns);
        self
    }

    /// Attach the pipeline bus so QoS reports reach the application.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.pacer.set_bus(bus);
        self
    }

    /// Frames dropped by QoS late-drop (past their deadline beyond the bound).
    pub fn late_dropped(&self) -> u64 {
        self.pacer.late_dropped()
    }

    pub fn frames_presented(&self) -> u64 {
        self.frames_presented.load(Ordering::Relaxed)
    }

    /// Glass-to-glass latency snapshot: source-side `arrival_ns` to the
    /// `eglSwapBuffers` that presents the frame. Untimed pipelines report
    /// `count = 0`.
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

impl Drop for CudaGlSink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Monotonic clock the sink offers, matching the source-side `arrival_ns`
/// epoch so the latency histogram is meaningful. Same role as `WaylandClock`.
#[derive(Debug)]
struct CudaGlClock;
impl PipelineClock for CudaGlClock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

impl AsyncElement for CudaGlSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            alloc::sync::Arc::new(CudaGlClock),
        ))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Pass-through at negotiation; NV12 is enforced in configure_pipeline.
        // The native decoder lands NV12 on this link via its DerivedOutput.
        Ok(upstream_caps.clone())
    }

    /// Native NV12-only sink constraint (mirrors `KmsSink` / `WaylandSink`): the
    /// solver intersects this against the upstream decoder's NV12 `DerivedOutput`
    /// and lands fixed NV12 on the link, so an undecoded (non-NV12) chain fails
    /// loud in negotiation. Geometry stays open; the decoder fixates it. The
    /// CUDA-vs-system memory-domain check stays per-frame in `process`.
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    /// Adopt the elected clock + base time so frames present at their PTS
    /// deadline rather than as fast as the EGL swap allows.
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    /// Relay a late drop upstream (M174): the runner forwards it onto the
    /// incoming link, where the producer can shed load.
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PACING_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.pacer
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.pacer.get_property(name)
    }

    /// Presents from CUDA device memory only; declaring it lets the M354 converter
    /// auto-plug splice a `CudaDownload`-free GPU path (or, behind a tee, a
    /// `CudaDownload` only on a sibling System branch).
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::Cuda)
    }

    /// M12 / C3 step 3: ask the producer to keep buffers in CUDA device memory
    /// so the `NvdecCuda` -> sink handoff stays on the GPU. The runner conveys
    /// this `MemoryDomainKind::Cuda` proposal to the decoder's
    /// `configure_allocation`. Returns `None` until the geometry is known (no
    /// proposal to make pre-`configure_pipeline`).
    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        let (w, h, _) = caps.dims()?;
        let (&Dim::Fixed(w), &Dim::Fixed(h)) = (w, h) else {
            return None;
        };
        Some(AllocationParams::cuda(
            nv12_byte_size(w, h),
            CUDA_POOL_HEADROOM,
            CUDA_ALIGN,
        ))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // NV12 only. Caps do not encode the memory domain, so the Cuda-vs-
        // System distinction is checked per frame in `process`.
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

        // Mid-stream geometry change: same dims is a no-op; new dims tear down
        // the worker and respawn (M16 5j), as WaylandSink does.
        if self.worker.is_some() {
            if w == self.width && h == self.height {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (tx, rx) = channel::<WorkerCmd<OwnedCudaBuffer>>();
        let presented = Arc::clone(&self.frames_presented);
        let latency = Arc::clone(&self.latency);
        let params = WindowParams {
            width: w,
            height: h,
            title: self.title.clone(),
            app_id: self.app_id.clone(),
            log_tag: WORKER_NAME,
        };

        let ready = Arc::new(Handshake::new());
        let ready_for_worker = Arc::clone(&ready);

        let join = thread::Builder::new()
            .name(String::from(WORKER_NAME))
            .spawn(move || {
                let channels = WorkerChannels {
                    rx,
                    presented,
                    latency,
                    ready: ready_for_worker,
                };
                if let Err(e) = run_gl_window(CudaPresenter, params, channels) {
                    std::eprintln!("{WORKER_NAME} worker error: {e:?}");
                }
            })
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        if !ready.wait(Duration::from_secs(5)) {
            let _ = tx.send(WorkerCmd::Shutdown);
            let _ = join.join();
            return Err(G2gError::Hardware(HardwareError::Other));
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
                    // This sink consumes CUDA device memory only; a system
                    // frame means the chain forgot the NvdecCuda backend.
                    let MemoryDomain::Cuda(buf) = domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // PTS pacing: hold the frame until its deadline on the
                    // elected clock, or drop it when it is already too late (the
                    // QoS bound) or outside the segment. Unpaced without a clock:
                    // present as fast as the swap allows.
                    let presented = self.frames_presented.load(Ordering::Relaxed);
                    if !wait_to_present(self.pacer.judge(timing.pts_ns, presented)).await {
                        return Ok(());
                    }
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                    tx.send(WorkerCmd::Frame {
                        frame: buf,
                        arrival_ns: timing.arrival_ns,
                        ack: ack_tx,
                    })
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    // Block until the worker presents this frame (vsync-paced
                    // by the compositor's release of the EGL back buffer).
                    ack_rx
                        .await
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    Ok(())
                }
                PipelinePacket::CapsChanged(_) => Ok(()),
                // Track the playback segment so PTS maps to running time
                // (correct across a seek), and re-anchor after a seek flush.
                PipelinePacket::Segment(seg) => {
                    self.pacer.set_segment(seg);
                    Ok(())
                }
                PipelinePacket::Flush => {
                    self.pacer.flush();
                    Ok(())
                }
                PipelinePacket::Eos => {
                    self.shutdown();
                    Ok(())
                }
                // future PipelinePacket variants: no-op (terminal sink).
                _ => Ok(()),
            }
        })
    }
}

// =================================================================
// Worker thread: the CUDA half of the shared Wayland + EGL worker
// =================================================================

/// The decoded planes are already in device memory, so this sink's "upload" is
/// the CUDA-GL interop copy into the NV12 textures; the window, the EGL context
/// and the present are the shared worker's ([`crate::glwindow`]).
struct CudaPresenter;

impl FramePresenter for CudaPresenter {
    type Frame = OwnedCudaBuffer;

    fn mode(&self) -> GlMode {
        GlMode::Nv12
    }

    fn present(&mut self, gl: &mut GlState, frame: &OwnedCudaBuffer) -> Result<(), G2gError> {
        gl.upload_and_draw(frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Rate, VideoCodec};

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
        }
    }

    #[test]
    fn intercept_passes_through() {
        let sink = CudaGlSink::new();
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
        let mut sink = CudaGlSink::new();
        let i420 = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert_eq!(
            sink.configure_pipeline(&i420).err(),
            Some(G2gError::CapsMismatch)
        );
        assert!(sink.worker.is_none());
    }

    #[test]
    fn configure_rejects_odd_dims() {
        let mut sink = CudaGlSink::new();
        match sink.configure_pipeline(&nv12(641, 480)) {
            Err(G2gError::CapsMismatch) => {}
            other => panic!("expected CapsMismatch on odd dims, got {other:?}"),
        }
    }

    #[test]
    fn proposes_cuda_device_memory() {
        use g2g_core::MemoryDomainKind;
        let sink = CudaGlSink::new();
        let p = sink
            .propose_allocation(&nv12(1920, 1080))
            .expect("fixed-geometry NV12 yields a proposal");
        assert_eq!(p.domain, MemoryDomainKind::Cuda);
        assert_eq!(p.size_bytes, 1920 * 1080 * 3 / 2);
        assert_eq!(p.align, CUDA_ALIGN);
        assert_eq!(p.min_buffers, CUDA_POOL_HEADROOM);
    }
}
