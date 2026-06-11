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
use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::boxed::Box;
use alloc::string::String;

use windows::core::{w, Interface};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11Texture2D, ID3D11VideoContext, ID3D11VideoDevice, ID3D11VideoProcessor,
    ID3D11VideoProcessorEnumerator, ID3D11VideoProcessorInputView, ID3D11VideoProcessorOutputView,
    D3D11_TEX2D_VPIV, D3D11_TEX2D_VPOV, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC, D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC,
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC,
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0, D3D11_VIDEO_PROCESSOR_STREAM,
    D3D11_VIDEO_USAGE_PLAYBACK_NORMAL, D3D11_VPIV_DIMENSION_TEXTURE2D, D3D11_VPOV_DIMENSION_TEXTURE2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_ALPHA_MODE_UNSPECIFIED, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_RATIONAL, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIFactory2, IDXGISwapChain1, DXGI_PRESENT, DXGI_SCALING_STRETCH,
    DXGI_SWAP_CHAIN_DESC1, DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL, DXGI_USAGE_RENDER_TARGET_OUTPUT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, PeekMessageW,
    RegisterClassExW, TranslateMessage, CW_USEDEFAULT, MSG, PM_REMOVE, WNDCLASSEXW,
    WS_OVERLAPPEDWINDOW, WS_VISIBLE,
};

use g2g_core::frame::Frame;
use g2g_core::metrics::{monotonic_ns, LatencyHistogram, LatencySnapshot};
use g2g_core::{
    AsyncElement, Caps, ClockCandidate, ClockPriority, ConfigureOutcome, Dim, G2gError,
    HardwareError, MemoryDomain, OutputSink, OwnedD3D11Texture, PipelineClock, PipelinePacket,
    RawVideoFormat,
};

/// Worker-thread command. `Frame` carries the decoded D3D11 texture (still
/// GPU-resident) plus the source-side `arrival_ns` and a one-shot `ack`
/// signalled once the frame is presented (compositor-paced backpressure).
enum WorkerCmd {
    Frame {
        texture: OwnedD3D11Texture,
        arrival_ns: u64,
        ack: tokio::sync::oneshot::Sender<()>,
    },
    Shutdown,
}

/// Sink-side handle set. Only `Send` state lives here so the runner can move
/// the sink between executor tasks.
pub struct D3D11Sink {
    title: String,
    cmd_tx: Option<Sender<WorkerCmd>>,
    worker: Option<JoinHandle<()>>,
    width: u32,
    height: u32,
    frames_presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
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
            frames_presented: Arc::new(AtomicU64::new(0)),
            latency: Arc::new(LatencyHistogram::new()),
        }
    }

    pub fn with_title<S: Into<String>>(mut self, title: S) -> Self {
        self.title = title.into();
        self
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
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::Provider,
            alloc::sync::Arc::new(D3D11Clock),
        ))
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
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

        if self.worker.is_some() {
            if w == self.width && h == self.height {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.shutdown();
        }

        let (tx, rx) = std::sync::mpsc::channel::<WorkerCmd>();
        let presented = Arc::clone(&self.frames_presented);
        let latency = Arc::clone(&self.latency);
        let title = self.title.clone();
        let ready = Arc::new(Handshake::new());
        let ready_for_worker = Arc::clone(&ready);

        let join = thread::Builder::new()
            .name(String::from("g2g-d3d11sink"))
            .spawn(move || {
                if let Err(e) = worker_main(w, h, title, rx, presented, latency, ready_for_worker) {
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
                PipelinePacket::DataFrame(Frame { domain, timing, .. }) => {
                    let MemoryDomain::D3D11Texture(texture) = domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let tx = self.cmd_tx.as_ref().ok_or(G2gError::NotConfigured)?;
                    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
                    tx.send(WorkerCmd::Frame {
                        texture,
                        arrival_ns: timing.arrival_ns,
                        ack: ack_tx,
                    })
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    ack_rx
                        .await
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    Ok(())
                }
                PipelinePacket::CapsChanged(_) | PipelinePacket::Flush => Ok(()),
                PipelinePacket::Eos => {
                    self.shutdown();
                    Ok(())
                }
            }
        })
    }
}

// =================================================================
// Worker thread: Win32 window + DXGI swapchain + D3D11 video processor
// =================================================================

/// GPU objects, created lazily on the first frame from that frame's device.
struct GpuState {
    video_device: ID3D11VideoDevice,
    video_context: ID3D11VideoContext,
    vp_enum: ID3D11VideoProcessorEnumerator,
    processor: ID3D11VideoProcessor,
    swapchain: IDXGISwapChain1,
}

fn worker_main(
    width: u32,
    height: u32,
    title: String,
    rx: Receiver<WorkerCmd>,
    presented: Arc<AtomicU64>,
    latency: Arc<LatencyHistogram>,
    ready: Arc<Handshake>,
) -> Result<(), G2gError> {
    let hwnd = create_window(&title, width, height)?;
    // The window is mappable immediately; signal readiness so the producer can
    // start.
    ready.notify();

    let mut gpu: Option<GpuState> = None;

    loop {
        pump_messages();
        match rx.recv_timeout(Duration::from_millis(8)) {
            Ok(WorkerCmd::Frame { texture, arrival_ns, ack }) => {
                if let Err(e) = present_frame(&mut gpu, hwnd, &texture) {
                    std::eprintln!("g2g-d3d11sink present error: {e:?}");
                } else {
                    presented.fetch_add(1, Ordering::Relaxed);
                    if arrival_ns != 0 {
                        let now = monotonic_ns();
                        if now >= arrival_ns {
                            latency.record(now - arrival_ns);
                        }
                    }
                }
                // Drop the texture (release the IMFSample) before acking, then
                // release the producer.
                drop(texture);
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

/// Drain pending window messages so the window stays responsive.
fn pump_messages() {
    let mut msg = MSG::default();
    // SAFETY: standard message-pump calls; `msg` is a valid local.
    unsafe {
        while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

/// Minimal window procedure: default handling for everything (the worker loop
/// owns the lifecycle via the command channel).
unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    // SAFETY: forwarded straight to the default handler.
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
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

/// Build the swapchain + video processor for `device_ptr` on `hwnd`.
fn init_gpu(device_ptr: u64, width: u32, height: u32, hwnd: HWND) -> Result<GpuState, G2gError> {
    // SAFETY: the device pointer comes from a live decoded frame (its
    // keep-alive holds the owning IMFSample, which refs the device); we only
    // borrow it to create owned video/​swapchain objects that take their own
    // refs.
    unsafe {
        let dev_raw = device_ptr as *mut core::ffi::c_void;
        let device = ID3D11Device::from_raw_borrowed(&dev_raw)
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let video_device: ID3D11VideoDevice = device.cast().map_err(win_err)?;
        let context = device.GetImmediateContext().map_err(win_err)?;
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
            .CreateSwapChainForHwnd(device, hwnd, &desc, None, None)
            .map_err(win_err)?;

        Ok(GpuState {
            video_device,
            video_context,
            vp_enum,
            processor,
            swapchain,
        })
    }
}

/// Present one decoded NV12 texture: blit it (NV12 -> RGB) to the swapchain
/// backbuffer through the video processor and `Present`.
fn present_frame(
    gpu: &mut Option<GpuState>,
    hwnd: HWND,
    texture: &OwnedD3D11Texture,
) -> Result<(), G2gError> {
    if gpu.is_none() {
        *gpu = Some(init_gpu(texture.device, texture.width, texture.height, hwnd)?);
    }
    let gpu = gpu.as_ref().unwrap();

    // SAFETY: the input texture is kept alive by `texture`'s keep-alive for the
    // duration of this call; all COM objects belong to the same device.
    unsafe {
        let tex_raw = texture.texture as *mut core::ffi::c_void;
        let input_tex = ID3D11Texture2D::from_raw_borrowed(&tex_raw)
            .ok_or(G2gError::Hardware(HardwareError::Other))?;

        let in_desc = D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC {
            FourCC: 0,
            ViewDimension: D3D11_VPIV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPIV {
                    MipSlice: 0,
                    ArraySlice: texture.subresource,
                },
            },
        };
        let mut input_view: Option<ID3D11VideoProcessorInputView> = None;
        gpu.video_device
            .CreateVideoProcessorInputView(input_tex, &gpu.vp_enum, &in_desc, Some(&mut input_view))
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
        gpu.video_device
            .CreateVideoProcessorOutputView(
                &backbuffer,
                &gpu.vp_enum,
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
        let blt = gpu
            .video_context
            .VideoProcessorBlt(&gpu.processor, &output_view, 0, &streams);
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

        gpu.swapchain.Present(1, DXGI_PRESENT(0)).ok().map_err(win_err)?;
    }
    Ok(())
}

fn win_err(e: windows::core::Error) -> G2gError {
    // D3D / DXGI / Win32 errors are COM HRESULTs, same carrier as the MF path.
    G2gError::Hardware(HardwareError::MediaFoundation(e.code().0))
}

// =================================================================
// Worker-readiness handshake (same primitive as WaylandSink)
// =================================================================

struct Handshake {
    flag: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

impl Handshake {
    fn new() -> Self {
        Self {
            flag: std::sync::Mutex::new(false),
            cv: std::sync::Condvar::new(),
        }
    }
    fn notify(&self) {
        *self.flag.lock().unwrap() = true;
        self.cv.notify_all();
    }
    fn wait(&self, timeout: Duration) -> bool {
        let guard = self.flag.lock().unwrap();
        let (guard, _) = self
            .cv
            .wait_timeout_while(guard, timeout, |notified| !*notified)
            .unwrap();
        *guard
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
        let sink = D3D11Sink::new();
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
        let mut sink = D3D11Sink::new();
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
        let mut sink = D3D11Sink::new();
        match sink.configure_pipeline(&nv12(641, 480)) {
            Err(G2gError::CapsMismatch) => {}
            other => panic!("expected CapsMismatch on odd dims, got {other:?}"),
        }
    }
}
