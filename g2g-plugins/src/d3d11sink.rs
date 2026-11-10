// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Windows D3D11 present sink (W1 Phase 4): the zero-copy display end of the
//! DXVA decode path, the analog of [`CudaGlSink`](crate::cudaglsink) on Linux.
//!
//! Consumes `MemoryDomain::D3D11Texture` frames (from `MfDecode::with_d3d11()`)
//! and presents them in a Win32 window via a DXGI flip-model swapchain. The
//! NV12 -> RGB colour convert runs on the GPU through a D3D11 video processor
//! (`VideoProcessorBlt`), so the decoded texture never leaves the GPU.
//!
//! ## Pipeline shape
//!
//! ```text
//! RtspSrc -> H264Parse -> MfDecode(with_d3d11) -> D3D11Sink
//!                                                     |
//!                                                     +-> Win32 window
//! ```
//!
//! ## Threading
//!
//! Win32 windows are thread-affine (their messages dispatch on the creating
//! thread) and D3D11 is driven single-threaded here, so (like `WaylandSink` /
//! `CudaGlSink`) all of it lives on a dedicated worker thread spun up at
//! `configure_pipeline`. The sink struct holds only `Send` handles (an mpsc
//! sender plus shared atomics). The decoded `OwnedD3D11Texture` is `Send` (its
//! keep-alive owns the `IMFSample`), so it crosses to the worker and the
//! texture stays valid until the worker drops it after presenting.
//!
//! The swapchain and video processor are created lazily on the first frame
//! using that frame's `ID3D11Device` (the decoder's device): a D3D11 resource
//! and the views over it must live on the same device, so the sink reuses the
//! decoder's rather than creating a second device and sharing textures.
//!
//! ## Verification status
//!
//! `d3d11-sink` + Windows-gated. The module COMPILES on the Windows dev host
//! (the COM surface is compiler-checked); the actual present (a real GPU
//! decoding into textures shown in a window) is owed as a user-side run on a
//! machine with a GPU, which the dev host can do.

use core::future::Future;
use core::mem::ManuallyDrop;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use windows::core::{w, Interface};
use windows::Win32::Foundation::{HMODULE, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, ID3D11VideoContext,
    ID3D11VideoContext1, ID3D11VideoDevice, ID3D11VideoProcessor, ID3D11VideoProcessorEnumerator,
    ID3D11VideoProcessorInputView, ID3D11VideoProcessorOutputView, D3D11_BIND_RENDER_TARGET,
    D3D11_BIND_SHADER_RESOURCE, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_FLAG,
    D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_SDK_VERSION, D3D11_TEX2D_VPIV, D3D11_TEX2D_VPOV,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
    D3D11_VIDEO_PROCESSOR_CAPS, D3D11_VIDEO_PROCESSOR_CONTENT_DESC,
    D3D11_VIDEO_PROCESSOR_FEATURE_CAPS, D3D11_VIDEO_PROCESSOR_FEATURE_CAPS_MIRROR,
    D3D11_VIDEO_PROCESSOR_FEATURE_CAPS_ROTATION, D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC,
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC,
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_ROTATION,
    D3D11_VIDEO_PROCESSOR_ROTATION_180, D3D11_VIDEO_PROCESSOR_ROTATION_270,
    D3D11_VIDEO_PROCESSOR_ROTATION_90, D3D11_VIDEO_PROCESSOR_ROTATION_IDENTITY,
    D3D11_VIDEO_PROCESSOR_STREAM, D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
    D3D11_VPIV_DIMENSION_TEXTURE2D, D3D11_VPOV_DIMENSION_TEXTURE2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_FORMAT_UNKNOWN,
    DXGI_RATIONAL, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIFactory2, IDXGISwapChain1, DXGI_PRESENT, DXGI_SCALING_STRETCH,
    DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_CHAIN_FLAG, DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
    DXGI_USAGE_RENDER_TARGET_OUTPUT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, PeekMessageW,
    PostQuitMessage, RegisterClassExW, SetWindowPos, TranslateMessage, CW_USEDEFAULT, MSG,
    PM_REMOVE, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, WM_DESTROY, WM_QUIT, WNDCLASSEXW,
    WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

use crate::videoconvert::display::{display_sink_caps, DisplayInput};
use crate::videoflip::frame_orientation;
use crate::worker_ready::Handshake;
use g2g_core::element::{PresentationStats, QosMessage};
use g2g_core::frame::Frame;
use g2g_core::memory::{DomainSet, MemoryDomainKind, SystemSlice};
use g2g_core::meta::{Orientation, OrientationSet};
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AllocationParams, AsyncElement, BusHandle, Caps, CapsConstraint, ClockCandidate, ClockPriority,
    ClockSync, ConfigureOutcome, Dim, ElementMetadata, G2gError, HardwareError, MemoryDomain,
    OutputSink, OwnedD3D11Texture, PadTemplate, PadTemplates, PipelineClock, PipelinePacket,
    PresentationPacer, PropError, PropKind, PropValue, PropertySpec, RawVideoFormat,
    MAX_LATENESS_PROPERTY, QOS_INTERVAL_PROPERTY,
};

use crate::clock::wait_to_present;

/// Texture-pool headroom the sink asks the decoder to keep resident: the frame
/// in flight on the present thread plus the one the runner link holds, so the
/// MFT's output texture pool does not starve under live pacing.
const D3D11_POOL_HEADROOM: usize = 3;

/// GPU surface alignment the sink requests (256 bytes is the common D3D11 /
/// DXVA texture alignment).
const D3D11_ALIGN: usize = 256;

/// Worker-thread command. `Frame` carries the frame's memory (a decoded D3D11
/// texture, still GPU-resident, or packed system NV12) plus the source-side
/// `arrival_ns` and a one-shot `ack` signalled once the frame is presented
/// (compositor-paced backpressure).
enum WorkerCmd {
    Frame {
        domain: MemoryDomain,
        arrival_ns: u64,
        orientation: Orientation,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    Shutdown,
}

/// Sink-side handle set. Only `Send` state lives here so the runner can move
/// the sink between executor tasks.
///
/// # Example
///
/// ```ignore
/// use g2g_plugins::d3d11sink::D3D11Sink;
///
/// let sink = D3D11Sink::new()
///     .with_title("preview")
///     .with_max_lateness_ns(20_000_000);
/// ```
pub struct D3D11Sink {
    title: String,
    cmd_tx: Option<Sender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
    width: u32,
    height: u32,
    input: Option<DisplayInput>,
    frames_presented: Arc<AtomicU64>,
    window_closed: Arc<AtomicBool>,
    latency: Arc<LatencyHistogram>,
    /// PTS pacing + QoS late-drop, on top of the worker's present-paced ack: idle
    /// until the runner hands over a clock, and the default lateness bound never
    /// drops.
    pacer: PresentationPacer,
}

impl core::fmt::Debug for D3D11Sink {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("D3D11Sink")
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

impl Default for D3D11Sink {
    fn default() -> Self {
        Self::new()
    }
}

impl D3D11Sink {
    pub fn new() -> Self {
        Self {
            title: String::from("glass2glass"),
            cmd_tx: None,
            worker: None,
            width: 0,
            height: 0,
            input: None,
            frames_presented: Arc::new(AtomicU64::new(0)),
            window_closed: Arc::new(AtomicBool::new(false)),
            latency: Arc::new(LatencyHistogram::new()),
            pacer: PresentationPacer::for_video_sink(),
        }
    }

    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = title.into();
        self
    }

    /// QoS late-drop bound: once PTS pacing is engaged, a frame past its
    /// deadline by more than `ns` is dropped instead of presented late, so the
    /// sink catches up. The default is 20 ms, `u64::MAX` never drops.
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

    fn worker_gone_error(&self) -> G2gError {
        if self.window_closed.load(Ordering::Acquire) {
            G2gError::WindowClosed
        } else {
            G2gError::Hardware(HardwareError::Other)
        }
    }

    pub fn frames_presented(&self) -> u64 {
        self.frames_presented.load(Ordering::Relaxed)
    }

    /// Glass-to-glass latency snapshot: source-side `arrival_ns` to `Present`.
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

impl Drop for D3D11Sink {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Debug)]
struct D3D11Clock;
impl PipelineClock for D3D11Clock {
    fn now_ns(&self) -> u64 {
        monotonic_ns()
    }
}

impl AsyncElement for D3D11Sink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    // the processor's rotation and mirror caps are read at the first frame, after the runner asks
    fn absorbed_orientations(&self) -> OrientationSet {
        if cfg!(feature = "metadata") {
            OrientationSet::ALL
        } else {
            OrientationSet::EMPTY
        }
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            alloc::sync::Arc::new(D3D11Clock),
        ))
    }

    /// Adopt the elected clock + base time so textures present at their PTS
    /// deadline rather than as fast as the swapchain allows.
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    /// Relay a late drop upstream: the runner forwards it onto the
    /// incoming link, where the producer can shed load.
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn presentation_stats(&self) -> Option<PresentationStats> {
        Some(PresentationStats {
            presented: self.frames_presented(),
            dropped: 0,
            late_dropped: self.late_dropped(),
        })
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Direct3D11 video sink",
            "Sink/Video",
            "Presents raw video in a window through a D3D11 video processor",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("title", PropKind::Str, "window title").with_default("glass2glass"),
            MAX_LATENESS_PROPERTY,
            QOS_INTERVAL_PROPERTY,
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "title" => {
                self.title = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            _ => self
                .pacer
                .set_property(name, &value)
                .unwrap_or(Err(PropError::Unknown)),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "title" => Some(PropValue::Str(self.title.clone())),
            _ => self.pacer.get_property(name),
        }
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(display_sink_caps(NATIVE_FORMATS))
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System).with(MemoryDomainKind::D3D11Texture)
    }

    /// Ask the decoder to keep buffers in D3D11 textures so the
    /// `MfDecode(with_d3d11)` -> sink handoff stays on the GPU. The runner
    /// conveys this `MemoryDomainKind::D3D11Texture` proposal to the decoder's
    /// `configure_allocation`. Returns `None` until the geometry is fixed.
    fn propose_allocation(&self, caps: &Caps) -> Option<AllocationParams> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } = caps
        else {
            return None;
        };
        let size = crate::pixel::frame_byte_size(*format, *w, *h);
        let mut params = AllocationParams::d3d11(size, D3D11_POOL_HEADROOM, D3D11_ALIGN);
        // a producer that cannot allocate D3D11 textures hands over system memory instead
        params.accepts = self.input_domains();
        Some(params)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let input = DisplayInput::from_caps(absolute_caps, NATIVE_FORMATS)?;
        let (w, h) = (input.width, input.height);
        if w % 2 != 0 || h % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }
        self.input = Some(input);

        if self.worker.is_some() {
            if w == self.width && h == self.height {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (tx, rx) = std::sync::mpsc::channel::<WorkerCmd>();
        let reports = WorkerReports {
            presented: Arc::clone(&self.frames_presented),
            window_closed: Arc::clone(&self.window_closed),
            latency: Arc::clone(&self.latency),
        };
        let title = self.title.clone();
        let ready = Arc::new(Handshake::new());
        let ready_for_worker = Arc::clone(&ready);

        let join = thread::Builder::new()
            .name(String::from("g2g-d3d11sink"))
            .spawn(move || {
                if let Err(e) = worker_main(w, h, title, rx, reports, ready_for_worker) {
                    std::eprintln!("g2g-d3d11sink worker error: {e:?}");
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
                PipelinePacket::DataFrame(frame) => {
                    let orientation = frame_orientation(&frame);
                    let Frame { domain, timing, .. } = frame;
                    if !matches!(domain, MemoryDomain::D3D11Texture(_))
                        && domain.as_system_slice().is_none()
                    {
                        return Err(G2gError::UnsupportedDomain);
                    }
                    // PTS pacing: hold the frame until its deadline on the
                    // elected clock, or drop it when it is already too late (the
                    // QoS bound) or outside the segment. Unpaced without a clock:
                    // present as fast as the swapchain allows.
                    let presented = self.frames_presented.load(Ordering::Relaxed);
                    if !wait_to_present(self.pacer.judge(timing.pts_ns, presented)).await {
                        return Ok(());
                    }
                    let converted = match domain.as_system_slice() {
                        Some(slice) => self
                            .input
                            .ok_or(G2gError::NotConfigured)?
                            .convert_frame(slice)?,
                        None => None,
                    };
                    let domain = match converted {
                        Some(native) => MemoryDomain::System(SystemSlice::from_boxed(native)),
                        None => domain,
                    };
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                    tx.send(WorkerCmd::Frame {
                        domain,
                        arrival_ns: timing.arrival_ns,
                        orientation,
                        ack: ack_tx,
                    })
                    .map_err(|_| self.worker_gone_error())?;
                    ack_rx.await.map_err(|_| self.worker_gone_error())?;
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

const NATIVE_FORMATS: &[RawVideoFormat] = &[RawVideoFormat::Nv12];

impl PadTemplates for D3D11Sink {
    /// A terminal sink: accepts NV12, and every other format `videoconvert`
    /// reads in system memory, at any geometry (the memory domain D3D11Texture
    /// is not encoded in caps). No source pad.
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(display_sink_caps(NATIVE_FORMATS))])
    }
}

// =================================================================
// Worker thread: Win32 window + DXGI swapchain + D3D11 video processor
// =================================================================

/// GPU objects on the sink's own device, rebuilt on the device of a decoded
/// D3D11 texture when one arrives from another.
struct GpuState {
    context: ID3D11DeviceContext,
    // NV12 texture system-memory frames are copied into, made on the first one
    upload: Option<ID3D11Texture2D>,
    device: ID3D11Device,
    // `None` on a device without D3D11 video (WARP, a GPU-less VM or CI runner):
    // system-memory frames are then converted to BGRA on the CPU
    video: Option<VideoPath>,
    swapchain: IDXGISwapChain1,
    // the turn the processor applies to the stream
    orientation: Orientation,
}

/// The D3D11 video processor that converts NV12 to the RGB backbuffer.
struct VideoPath {
    video_device: ID3D11VideoDevice,
    video_context: ID3D11VideoContext,
    vp_enum: ID3D11VideoProcessorEnumerator,
    processor: ID3D11VideoProcessor,
}

struct WorkerReports {
    presented: Arc<AtomicU64>,
    window_closed: Arc<AtomicBool>,
    latency: Arc<LatencyHistogram>,
}

fn worker_main(
    width: u32,
    height: u32,
    title: String,
    rx: Receiver<WorkerCmd>,
    reports: WorkerReports,
    ready: Arc<Handshake>,
) -> Result<(), G2gError> {
    let hwnd = create_window(&title, width, height)?;
    // d3d11 device and swapchain creation takes about 300 ms, enough to make the opening frames late when it waits for the first one
    let mut gpu = Some(init_gpu(create_device()?, width, height, hwnd)?);
    ready.notify();

    loop {
        if !pump_messages() {
            std::eprintln!("g2g-d3d11sink: the window was closed");
            reports.window_closed.store(true, Ordering::Release);
            break;
        }
        match rx.recv_timeout(Duration::from_millis(8)) {
            Ok(WorkerCmd::Frame {
                domain,
                arrival_ns,
                orientation,
                ack,
            }) => {
                if let Err(e) = present_frame(&mut gpu, hwnd, &domain, width, height, orientation) {
                    std::eprintln!("g2g-d3d11sink present error: {e:?}");
                } else {
                    reports.presented.fetch_add(1, Ordering::Relaxed);
                    if arrival_ns != 0 {
                        let now = monotonic_ns();
                        if now >= arrival_ns {
                            reports.latency.record(now - arrival_ns);
                        }
                    }
                }
                // Drop the texture (release the IMFSample) before acking, then
                // release the producer.
                drop(domain);
                let _ = ack.send(());
            }
            Ok(WorkerCmd::Shutdown) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => {}
        }
    }

    // SAFETY: the window was created on this thread and is not used after.
    unsafe {
        let _ = DestroyWindow(hwnd);
    }
    Ok(())
}

/// Drain pending window messages so the window stays responsive. False once
/// the window has been closed.
fn pump_messages() -> bool {
    let mut msg = MSG::default();
    // SAFETY: standard message-pump calls; `msg` is a valid local.
    unsafe {
        while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
            if msg.message == WM_QUIT {
                return false;
            }
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    true
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    // win32 sends WM_DESTROY when the user closes the window
    if msg == WM_DESTROY {
        // SAFETY: called on the thread that owns the window's message queue.
        unsafe { PostQuitMessage(0) };
        return LRESULT(0);
    }
    // SAFETY: forwarded straight to the default handler.
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
}

/// A hardware D3D11 device for system-memory frames that bring no device of
/// their own: one with video support when the driver has it, else a plain one
/// (WARP refuses the video flag with `DXGI_ERROR_UNSUPPORTED`).
fn create_device() -> Result<ID3D11Device, G2gError> {
    let base = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    create_device_with(base | D3D11_CREATE_DEVICE_VIDEO_SUPPORT)
        .or_else(|_| create_device_with(base))
}

fn create_device_with(flags: D3D11_CREATE_DEVICE_FLAG) -> Result<ID3D11Device, G2gError> {
    let mut device: Option<ID3D11Device> = None;
    // SAFETY: the out-param is a valid local, initialised by the call before it is read.
    unsafe {
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            flags,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            None,
        )
        .map_err(win_err)?;
    }
    device.ok_or(G2gError::Hardware(HardwareError::Other))
}

/// Register the window class (idempotent) and create a visible window at the
/// video dimensions.
fn create_window(title: &str, width: u32, height: u32) -> Result<HWND, G2gError> {
    // SAFETY: standard Win32 window creation on the worker thread.
    unsafe {
        let hinstance = GetModuleHandleW(None).map_err(win_err)?;
        let class_name = w!("g2g_d3d11sink");
        let wc = WNDCLASSEXW {
            cbSize: core::mem::size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(wndproc),
            hInstance: hinstance.into(),
            lpszClassName: class_name,
            ..Default::default()
        };
        // RegisterClassExW returns 0 on failure; a duplicate registration
        // (second sink) also "fails" but is harmless, so we ignore the result
        // and let CreateWindowExW surface a real problem.
        let _ = RegisterClassExW(&wc);

        let title_w: alloc::vec::Vec<u16> =
            title.encode_utf16().chain(core::iter::once(0)).collect();
        let hwnd = CreateWindowExW(
            Default::default(),
            class_name,
            windows::core::PCWSTR(title_w.as_ptr()),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            width as i32,
            height as i32,
            None,
            None,
            Some(hinstance.into()),
            None,
        )
        .map_err(win_err)?;
        Ok(hwnd)
    }
}

/// Build the swapchain, plus the video processor when `device` has D3D11 video,
/// for `device` on `hwnd`.
fn init_gpu(
    device: ID3D11Device,
    width: u32,
    height: u32,
    hwnd: HWND,
) -> Result<GpuState, G2gError> {
    // SAFETY: COM object creation on the worker thread, every object from `device`.
    unsafe {
        let context = device.GetImmediateContext().map_err(win_err)?;
        let video = init_video(&device, &context, width, height).ok();

        let factory: IDXGIFactory2 = CreateDXGIFactory1().map_err(win_err)?;
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: width,
            Height: height,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: 2,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
            AlphaMode: DXGI_ALPHA_MODE_UNSPECIFIED,
            Flags: 0,
        };
        let swapchain = factory
            .CreateSwapChainForHwnd(&device, hwnd, &desc, None, None)
            .map_err(win_err)?;

        Ok(GpuState {
            context,
            upload: None,
            device,
            video,
            swapchain,
            orientation: Orientation::Identity,
        })
    }
}

/// The NV12 -> RGB video processor for `device`, which fails on a device
/// created without video support.
fn init_video(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    width: u32,
    height: u32,
) -> Result<VideoPath, G2gError> {
    // SAFETY: COM object creation on the worker thread, every object from `device`.
    unsafe {
        let video_device: ID3D11VideoDevice = device.cast().map_err(win_err)?;
        let video_context: ID3D11VideoContext = context.cast().map_err(win_err)?;

        let rate = DXGI_RATIONAL {
            Numerator: 60,
            Denominator: 1,
        };
        let content = D3D11_VIDEO_PROCESSOR_CONTENT_DESC {
            InputFrameFormat: D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            InputFrameRate: rate,
            InputWidth: width,
            InputHeight: height,
            OutputFrameRate: rate,
            OutputWidth: width,
            OutputHeight: height,
            Usage: D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
        };
        let vp_enum = video_device
            .CreateVideoProcessorEnumerator(&content)
            .map_err(win_err)?;
        let processor = video_device
            .CreateVideoProcessor(&vp_enum, 0)
            .map_err(win_err)?;
        Ok(VideoPath {
            video_device,
            video_context,
            vp_enum,
            processor,
        })
    }
}

/// The D3D11 objects a decoded frame carries, as owned references.
fn decoded_texture(
    texture: &OwnedD3D11Texture,
) -> Result<(ID3D11Device, ID3D11Texture2D), G2gError> {
    // SAFETY: both pointers come from a live decoded frame whose keep-alive holds
    // the owning IMFSample, and cloning takes a reference of our own.
    unsafe {
        let device_raw = texture.device as *mut core::ffi::c_void;
        let texture_raw = texture.texture as *mut core::ffi::c_void;
        let device = ID3D11Device::from_raw_borrowed(&device_raw)
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let input = ID3D11Texture2D::from_raw_borrowed(&texture_raw)
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        Ok((device.clone(), input.clone()))
    }
}

/// Copy a packed system-memory NV12 frame into the sink's own texture.
fn upload_system(
    gpu: &mut GpuState,
    bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<ID3D11Texture2D, G2gError> {
    if bytes.len() < crate::pixel::frame_byte_size(RawVideoFormat::Nv12, width, height) {
        return Err(G2gError::CapsMismatch);
    }
    let texture = match &gpu.upload {
        Some(texture) => texture.clone(),
        None => {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_NV12,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                // a video processor input view rejects a shader-resource-only NV12 texture (E_INVALIDARG)
                BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let mut texture: Option<ID3D11Texture2D> = None;
            // SAFETY: `desc` and the out-param are valid locals for the call.
            unsafe {
                gpu.device
                    .CreateTexture2D(&desc, None, Some(&mut texture))
                    .map_err(win_err)?;
            }
            let texture = texture.ok_or(G2gError::Hardware(HardwareError::Other))?;
            gpu.upload = Some(texture.clone());
            texture
        }
    };
    // d3d11 reads an NV12 upload as the luma rows followed by the chroma rows, both at the source row pitch
    // SAFETY: `bytes` holds a whole packed frame at `width` row pitch, checked above.
    unsafe {
        gpu.context
            .UpdateSubresource(&texture, 0, None, bytes.as_ptr().cast(), width, 0);
    }
    Ok(texture)
}

/// Present one NV12 frame, a decoded D3D11 texture or packed system memory:
/// blit it (NV12 -> RGB) to the swapchain backbuffer through the video processor
/// and `Present`.
fn present_frame(
    gpu: &mut Option<GpuState>,
    hwnd: HWND,
    domain: &MemoryDomain,
    width: u32,
    height: u32,
    orientation: Orientation,
) -> Result<(), G2gError> {
    let decoded = match domain {
        MemoryDomain::D3D11Texture(texture) => Some((texture, decoded_texture(texture)?)),
        _ => None,
    };
    let foreign_device = match (&decoded, gpu.as_ref()) {
        (Some((_, (device, _))), Some(current)) => *device != current.device,
        _ => false,
    };
    if let Some(old) = gpu.take_if(|_| foreign_device) {
        let context = old.context.clone();
        drop(old);
        // dxgi frees a released flip-model swapchain, and lets another bind to the window, only once its device context flushes
        // SAFETY: the context belongs to this worker thread's device and nothing else uses it now.
        unsafe {
            context.ClearState();
            context.Flush();
        }
    }
    if gpu.is_none() {
        *gpu = Some(match &decoded {
            Some((texture, (device, _))) => {
                init_gpu(device.clone(), texture.width, texture.height, hwnd)?
            }
            None => init_gpu(create_device()?, width, height, hwnd)?,
        });
    }
    let gpu = gpu.as_mut().ok_or(G2gError::NotConfigured)?;
    if gpu.video.is_none() {
        // a decoded texture's device always has video, so only system memory lands here
        let bytes = domain
            .as_system_slice()
            .ok_or(G2gError::UnsupportedDomain)?;
        return present_converted(gpu, bytes, width, height, orientation);
    }
    let (input_tex, subresource, picture_width, picture_height) = match decoded {
        Some((texture, (_, input))) => (input, texture.subresource, texture.width, texture.height),
        None => {
            let bytes = domain
                .as_system_slice()
                .ok_or(G2gError::UnsupportedDomain)?;
            (upload_system(gpu, bytes, width, height)?, 0, width, height)
        }
    };
    apply_orientation(gpu, hwnd, picture_width, picture_height, orientation)?;
    let video = gpu.video.as_ref().ok_or(G2gError::NotConfigured)?;

    // SAFETY: `input_tex` holds its own reference for this call, and a decoded
    // frame's keep-alive in `domain` outlives it; all COM objects belong to the same device.
    unsafe {
        let in_desc = D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC {
            FourCC: 0,
            ViewDimension: D3D11_VPIV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPIV {
                    MipSlice: 0,
                    ArraySlice: subresource,
                },
            },
        };
        let mut input_view: Option<ID3D11VideoProcessorInputView> = None;
        video
            .video_device
            .CreateVideoProcessorInputView(
                &input_tex,
                &video.vp_enum,
                &in_desc,
                Some(&mut input_view),
            )
            .map_err(win_err)?;
        let input_view = input_view.ok_or(G2gError::Hardware(HardwareError::Other))?;

        let backbuffer: ID3D11Texture2D = gpu.swapchain.GetBuffer(0).map_err(win_err)?;
        let out_desc = D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
            ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPOV { MipSlice: 0 },
            },
        };
        let mut output_view: Option<ID3D11VideoProcessorOutputView> = None;
        video
            .video_device
            .CreateVideoProcessorOutputView(
                &backbuffer,
                &video.vp_enum,
                &out_desc,
                Some(&mut output_view),
            )
            .map_err(win_err)?;
        let output_view = output_view.ok_or(G2gError::Hardware(HardwareError::Other))?;

        let mut streams = [D3D11_VIDEO_PROCESSOR_STREAM {
            Enable: true.into(),
            OutputIndex: 0,
            InputFrameOrField: 0,
            PastFrames: 0,
            FutureFrames: 0,
            ppPastSurfaces: core::ptr::null_mut(),
            pInputSurface: ManuallyDrop::new(Some(input_view)),
            ppFutureSurfaces: core::ptr::null_mut(),
            ppPastSurfacesRight: core::ptr::null_mut(),
            pInputSurfaceRight: ManuallyDrop::new(None),
            ppFutureSurfacesRight: core::ptr::null_mut(),
        }];
        let blt =
            video
                .video_context
                .VideoProcessorBlt(&video.processor, &output_view, 0, &streams);
        // Release the input-view refs we placed in the stream struct.
        drop(ManuallyDrop::into_inner(core::mem::replace(
            &mut streams[0].pInputSurface,
            ManuallyDrop::new(None),
        )));
        drop(ManuallyDrop::into_inner(core::mem::replace(
            &mut streams[0].pInputSurfaceRight,
            ManuallyDrop::new(None),
        )));
        blt.map_err(win_err)?;

        gpu.swapchain
            .Present(1, DXGI_PRESENT(0))
            .ok()
            .map_err(win_err)?;
    }
    Ok(())
}

/// Present a packed system-memory NV12 frame without a video processor: convert
/// it to BGRA on the CPU, copy that into the swapchain backbuffer, `Present`.
fn present_converted(
    gpu: &mut GpuState,
    bytes: &[u8],
    width: u32,
    height: u32,
    orientation: Orientation,
) -> Result<(), G2gError> {
    if bytes.len() < crate::pixel::frame_byte_size(RawVideoFormat::Nv12, width, height) {
        return Err(G2gError::CapsMismatch);
    }
    // turning or mirroring needs the video processor this device lacks
    if orientation != Orientation::Identity {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let bgra = crate::videoconvert::convert(
        bytes,
        RawVideoFormat::Nv12,
        RawVideoFormat::Bgra8,
        width as usize,
        height as usize,
        g2g_core::Colorimetry::UNKNOWN,
    );
    // SAFETY: `bgra` holds `width * height` tightly packed BGRA pixels, the
    // backbuffer's format and size, and every object belongs to one device.
    unsafe {
        let backbuffer: ID3D11Texture2D = gpu.swapchain.GetBuffer(0).map_err(win_err)?;
        gpu.context
            .UpdateSubresource(&backbuffer, 0, None, bgra.as_ptr().cast(), width * 4, 0);
        gpu.swapchain
            .Present(1, DXGI_PRESENT(0))
            .ok()
            .map_err(win_err)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StreamOrientation {
    rotation: D3D11_VIDEO_PROCESSOR_ROTATION,
    flip_horizontal: bool,
    flip_vertical: bool,
}

// The processor rotates clockwise, assumed here to happen after the mirror.
fn stream_orientation(orientation: Orientation) -> StreamOrientation {
    let (rotation, flip_horizontal, flip_vertical) = match orientation {
        Orientation::Identity => (D3D11_VIDEO_PROCESSOR_ROTATION_IDENTITY, false, false),
        Orientation::Rotate90Cw => (D3D11_VIDEO_PROCESSOR_ROTATION_90, false, false),
        Orientation::Rotate180 => (D3D11_VIDEO_PROCESSOR_ROTATION_180, false, false),
        Orientation::Rotate90Ccw => (D3D11_VIDEO_PROCESSOR_ROTATION_270, false, false),
        Orientation::HorizontalMirror => (D3D11_VIDEO_PROCESSOR_ROTATION_IDENTITY, true, false),
        Orientation::VerticalMirror => (D3D11_VIDEO_PROCESSOR_ROTATION_IDENTITY, false, true),
        Orientation::Transverse => (D3D11_VIDEO_PROCESSOR_ROTATION_90, true, false),
        Orientation::Transpose => (D3D11_VIDEO_PROCESSOR_ROTATION_270, true, false),
    };
    StreamOrientation {
        rotation,
        flip_horizontal,
        flip_vertical,
    }
}

impl StreamOrientation {
    fn rotates(self) -> bool {
        self.rotation != D3D11_VIDEO_PROCESSOR_ROTATION_IDENTITY
    }

    fn mirrors(self) -> bool {
        self.flip_horizontal || self.flip_vertical
    }
}

fn has_feature(
    caps: &D3D11_VIDEO_PROCESSOR_CAPS,
    feature: D3D11_VIDEO_PROCESSOR_FEATURE_CAPS,
) -> bool {
    (caps.FeatureCaps & feature.0 as u32) != 0
}

fn apply_orientation(
    gpu: &mut GpuState,
    hwnd: HWND,
    picture_width: u32,
    picture_height: u32,
    orientation: Orientation,
) -> Result<(), G2gError> {
    if orientation == gpu.orientation {
        return Ok(());
    }
    let previous = stream_orientation(gpu.orientation);
    let next = stream_orientation(orientation);
    // only the video processor turns or mirrors the picture
    let video = gpu
        .video
        .as_ref()
        .ok_or(G2gError::Hardware(HardwareError::Other))?;
    // SAFETY: the enumerator, context and processor come from one device and stay on this thread.
    unsafe {
        let mut caps = D3D11_VIDEO_PROCESSOR_CAPS::default();
        video
            .vp_enum
            .GetVideoProcessorCaps(&mut caps)
            .map_err(win_err)?;
        let missing_rotation =
            next.rotates() && !has_feature(&caps, D3D11_VIDEO_PROCESSOR_FEATURE_CAPS_ROTATION);
        let missing_mirror =
            next.mirrors() && !has_feature(&caps, D3D11_VIDEO_PROCESSOR_FEATURE_CAPS_MIRROR);
        if missing_rotation || missing_mirror {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        video.video_context.VideoProcessorSetStreamRotation(
            &video.processor,
            0,
            next.rotates(),
            next.rotation,
        );
        if next.mirrors() || previous.mirrors() {
            let video_context1: ID3D11VideoContext1 =
                video.video_context.cast().map_err(win_err)?;
            video_context1.VideoProcessorSetStreamMirror(
                &video.processor,
                0,
                next.mirrors(),
                next.flip_horizontal,
                next.flip_vertical,
            );
        }
    }
    if orientation.swaps_dims() != gpu.orientation.swaps_dims() {
        let (width, height) = if orientation.swaps_dims() {
            (picture_height, picture_width)
        } else {
            (picture_width, picture_height)
        };
        // SAFETY: `present_frame` drops its buffer views, and `hwnd` was created on this thread.
        unsafe {
            gpu.swapchain
                .ResizeBuffers(
                    0,
                    width,
                    height,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
                .map_err(win_err)?;
            SetWindowPos(
                hwnd,
                None,
                0,
                0,
                width as i32,
                height as i32,
                SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
            )
            .map_err(win_err)?;
        }
    }
    gpu.orientation = orientation;
    Ok(())
}

fn win_err(e: windows::core::Error) -> G2gError {
    // D3D / DXGI / Win32 errors are COM HRESULTs, same carrier as the MF path.
    G2gError::Hardware(HardwareError::MediaFoundation(e.code().0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::{Interlace, Rate, VideoCodec};

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        }
    }

    #[test]
    fn intercept_passes_through() {
        let sink = D3D11Sink::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(sink.intercept_caps(&h264), Ok(h264));
    }

    #[test]
    fn caps_constraint_takes_nv12_first_and_converted_formats() {
        let sink = D3D11Sink::new();
        let CapsConstraint::Accepts(set) = sink.caps_constraint_as_sink() else {
            panic!("expected Accepts");
        };
        let alternatives = set.alternatives();
        assert_eq!(
            alternatives[0],
            crate::videofx::any_geometry(RawVideoFormat::Nv12)
        );
        assert!(alternatives.contains(&crate::videofx::any_geometry(RawVideoFormat::I420)));
    }

    #[test]
    fn configure_rejects_a_format_videoconvert_cannot_read() {
        let mut sink = D3D11Sink::new();
        let planar_rgb = Caps::RawVideo {
            format: RawVideoFormat::Rgbp8,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        };
        assert_eq!(
            sink.configure_pipeline(&planar_rgb).err(),
            Some(G2gError::CapsMismatch)
        );
        assert!(sink.worker.is_none());
    }

    #[test]
    fn configure_rejects_odd_dims() {
        let mut sink = D3D11Sink::new();
        match sink.configure_pipeline(&nv12(641, 480)) {
            Err(G2gError::CapsMismatch) => {}
            other => panic!("expected CapsMismatch on odd dims, got {other:?}"),
        }
    }

    #[test]
    fn pad_template_links_from_an_nv12_source() {
        use g2g_core::{types_can_link, CapsSet, PadCaps, PadDirection, PadTemplate, PadTemplates};

        // The sink exposes one NV12 sink pad and no source pad.
        let t = D3D11Sink::pad_templates();
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].direction, PadDirection::Sink);
        assert!(matches!(t[0].caps, PadCaps::Fixed(ref s) if s.accepts(&nv12(1920, 1080))));

        // A NV12 source links into it; an RGBA one does not.
        struct Nv12Source;
        impl PadTemplates for Nv12Source {
            fn pad_templates() -> Vec<PadTemplate> {
                Vec::from([PadTemplate::source(CapsSet::one(Caps::RawVideo {
                    format: RawVideoFormat::Nv12,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                    interlace: Interlace::Any,
                    colorimetry: g2g_core::Colorimetry::UNKNOWN,
                    pixel_aspect: g2g_core::PixelAspect::Any,
                }))])
            }
        }
        struct RgbaSource;
        impl PadTemplates for RgbaSource {
            fn pad_templates() -> Vec<PadTemplate> {
                Vec::from([PadTemplate::source(CapsSet::one(Caps::RawVideo {
                    format: RawVideoFormat::Rgba8,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                    interlace: Interlace::Any,
                    colorimetry: g2g_core::Colorimetry::UNKNOWN,
                    pixel_aspect: g2g_core::PixelAspect::Any,
                }))])
            }
        }
        assert!(types_can_link::<Nv12Source, D3D11Sink>());
        assert!(!types_can_link::<RgbaSource, D3D11Sink>());
    }

    // The turn the processor shows for `stream`, the mirror applied before the clockwise turns.
    fn shown_orientation(stream: StreamOrientation) -> Orientation {
        let mirrored = match (stream.flip_horizontal, stream.flip_vertical) {
            (false, false) => Orientation::Identity,
            (true, false) => Orientation::HorizontalMirror,
            (false, true) => Orientation::VerticalMirror,
            (true, true) => Orientation::Rotate180,
        };
        let quarter_turns = match stream.rotation {
            D3D11_VIDEO_PROCESSOR_ROTATION_IDENTITY => 0,
            D3D11_VIDEO_PROCESSOR_ROTATION_90 => 1,
            D3D11_VIDEO_PROCESSOR_ROTATION_180 => 2,
            D3D11_VIDEO_PROCESSOR_ROTATION_270 => 3,
            other => panic!("not a rotation: {other:?}"),
        };
        (0..quarter_turns).fold(mirrored, |shown, _| shown.compose(Orientation::Rotate90Cw))
    }

    #[test]
    fn stream_orientation_shows_the_descriptor() {
        for orientation in [
            Orientation::Identity,
            Orientation::Rotate90Cw,
            Orientation::Rotate180,
            Orientation::Rotate90Ccw,
            Orientation::HorizontalMirror,
            Orientation::VerticalMirror,
            Orientation::Transpose,
            Orientation::Transverse,
        ] {
            assert_eq!(
                shown_orientation(stream_orientation(orientation)),
                orientation,
                "{orientation:?}"
            );
        }
    }

    #[test]
    fn absorbs_orientation_with_metadata() {
        let expected = if cfg!(feature = "metadata") {
            OrientationSet::ALL
        } else {
            OrientationSet::EMPTY
        };
        assert_eq!(D3D11Sink::new().absorbed_orientations(), expected);
    }

    #[test]
    fn proposes_d3d11_texture_memory() {
        use g2g_core::MemoryDomainKind;
        let sink = D3D11Sink::new();
        let p = sink
            .propose_allocation(&nv12(1920, 1080))
            .expect("fixed-geometry NV12 yields a proposal");
        assert_eq!(p.domain, MemoryDomainKind::D3D11Texture);
        assert_eq!(p.size_bytes, 1920 * 1080 * 3 / 2);
        assert_eq!(p.align, D3D11_ALIGN);
        assert_eq!(p.min_buffers, D3D11_POOL_HEADROOM);
    }
}
