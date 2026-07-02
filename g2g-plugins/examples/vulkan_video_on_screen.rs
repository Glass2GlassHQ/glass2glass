//! On-screen live demo of the Vulkan Video wedge (M497): H.264 decoded on the
//! GPU via `VK_KHR_video_*`, presented in a real window by [`WgpuSink`].
//!
//! Two present paths, chosen at runtime by what the hardware allows:
//!
//! - **Zero-copy** (the wedge): when the GPU that decodes also drives the
//!   display, each decoded picture is an RGBA `wgpu::Texture` that `WgpuSink`
//!   blits straight to the swapchain, with NO GPU->CPU readback. The on-screen
//!   sibling of the `m495` test (offscreen target read back to assert) and of
//!   `vulkan_video_smoke` (dumps PPMs).
//! - **Cross-GPU fallback**: on a hybrid / PRIME machine only the discrete GPU
//!   exposes `VK_KHR_video_decode_h264` while a different (integrated) GPU drives
//!   the display, so a single-device present is impossible. The demo decodes on
//!   the discrete GPU, reads each frame back once, and re-uploads + presents it on
//!   the display GPU. Not zero-copy (the transport a real playbin uses on a PRIME
//!   box), but you can watch the hardware decode live.
//!
//! Window + event loop ownership belongs to the application (a wgpu surface is
//! built from a window handle and driven by the app's event loop), which is why
//! this is an example, not a self-checking test.
//!
//! Run (needs a Vulkan H.264 decode GPU, e.g. the RTX 3060, and a display):
//!
//! ```sh
//! cargo run --release -p g2g-plugins --features vulkan-video,wgpu-sink \
//!     --example vulkan_video_on_screen                 # the bundled 640x480 clip
//! cargo run --release -p g2g-plugins --features vulkan-video,wgpu-sink \
//!     --example vulkan_video_on_screen -- my.h264      # a stream of your own
//! ```
//!
//! The bundled clip is two GOPs (IDR + P frames) of animated content, so it
//! visibly moves; it loops. Close the window (or Esc) to quit.

use std::sync::Arc;
use std::time::{Duration, Instant};

use g2g_core::frame::Frame;
use g2g_core::memory::OwnedWgpuTexture;
use g2g_core::runtime::block_on;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PushOutcome, RawVideoFormat, Rate,
};
use g2g_plugins::gpu::{GpuContext, WgpuTextureKeepAlive};
use g2g_plugins::vulkanvideo::{
    extract_h264_parameter_sets, open_h264_decode_device, VulkanVideoError,
};
use g2g_plugins::wgpusink::WgpuSink;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// The 640x480 baseline clip the tests use, embedded so the demo runs with no
/// arguments (two GOPs of IDR + P frames, so it exercises the DPB and moves).
const BUNDLED_CLIP: &[u8] = include_bytes!("../tests/fixtures/h264_640x480.h264");

/// How long each decoded frame stays on screen before advancing (playback rate).
/// The clip is authored at 30 fps; 66 ms (~15 fps) makes the animation easy to
/// follow while it loops.
const FRAME_INTERVAL: Duration = Duration::from_millis(66);

/// The decoded RGBA texture layout (what the GPU decoder converts NV12 into).
const RGBA: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn main() {
    let clip_path = std::env::args().nth(1);
    let clip: Vec<u8> = match &clip_path {
        Some(p) => std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}")),
        None => BUNDLED_CLIP.to_vec(),
    };
    println!(
        "decoding {} ({} bytes)",
        clip_path.as_deref().unwrap_or("<bundled 640x480 clip>"),
        clip.len()
    );

    // Open the Vulkan Video decode device (the GPU that does the hardware decode).
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter) => {
            eprintln!("no Vulkan adapter; this demo needs a GPU with Vulkan H.264 decode.");
            return;
        }
        Err(VulkanVideoError::ExtensionUnsupported) | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("this GPU has no Vulkan H.264 decode support.");
            return;
        }
        Err(e) => panic!("failed to open decode device: {e:?}"),
    };

    let ps = extract_h264_parameter_sets(&clip).expect("parse SPS+PPS from the stream");
    let width = (ps.sps.pic_width_in_mbs_minus1 + 1) * 16;
    let height = (ps.sps.pic_height_in_map_units_minus1 + 1) * 16;
    let session = device.create_h264_session(&ps, width, height).expect("create decode session");

    // GPU-resident decode: each picture converts in place to an RGBA texture on
    // the decode device (no NV12 readback). Needs a distinct compute queue.
    let mut decoder = match device.create_h264_dpb_decoder_gpu(&session, &ps) {
        Ok(d) => d,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!(
                "this GPU exposes no separate compute queue for the NV12->RGBA pass; \
                 the present path needs one. (Decode-to-NV12 still works, see the \
                 vulkan_video_smoke example.)"
            );
            return;
        }
        Err(e) => panic!("failed to build GPU decoder: {e:?}"),
    };

    let decode_ctx = device.gpu_context();
    let textures = decoder.decode_all_to_textures(&clip).expect("decode the stream to textures");
    let decode_gpu = decode_ctx.adapter.get_info().name;
    println!("decoded {} frames at {width}x{height} on {decode_gpu}", textures.len());
    if textures.is_empty() {
        eprintln!("stream produced no frames; nothing to show.");
        return;
    }

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        decode_ctx,
        decode_textures: textures,
        width,
        height,
        window: None,
        present: None,
        idx: 0,
        last_advance: Instant::now(),
    };
    if let Err(e) = event_loop.run_app(&mut app) {
        // A compositor dropping the connection (e.g. window closed abruptly) is
        // not a decode failure; report it without aborting.
        eprintln!("event loop ended: {e}");
    }
    let shown = app.present.as_ref().map(|p| p.sink.presented_count()).unwrap_or(0);
    println!("presented {shown} frames; bye.");
}

/// The chosen present path: a sink and the textures to present through it (both
/// bound to the sink's device, whichever GPU that turned out to be).
struct Present {
    sink: WgpuSink,
    textures: Vec<wgpu::Texture>,
}

struct App {
    decode_ctx: GpuContext,
    /// Decoded RGBA textures on the decode GPU (the zero-copy source).
    decode_textures: Vec<wgpu::Texture>,
    width: u32,
    height: u32,
    // Created once the event loop resumes and a window exists.
    window: Option<Arc<Window>>,
    present: Option<Present>,
    idx: usize,
    last_advance: Instant,
}

impl App {
    /// Choose and build the present path for `window`.
    ///
    /// The GPU that actually drives the display is found first, on a clean wgpu
    /// instance. This matters on Wayland: creating a surface on the discrete
    /// NVIDIA instance can poison the whole window (a compositor DRM-syncobj
    /// error), so we must not touch the decode instance unless it is also the
    /// display GPU. If the display GPU and the decode GPU are the same physical
    /// device (a single-GPU host), present the decoded textures with no copy;
    /// otherwise (PRIME) read each frame back once and re-upload it to the display
    /// GPU.
    fn build_present(&self, window: &Arc<Window>, width: u32, height: u32) -> Present {
        let (display_ctx, surface, config) = open_display_gpu(window, width, height)
            .expect("no GPU on this host can present to the window");
        let decode_gpu = self.decode_ctx.adapter.get_info();
        let display_gpu = display_ctx.adapter.get_info();

        if same_physical_device(&decode_gpu, &display_gpu) {
            // Single-GPU host: the decode GPU drives the display. Release the probe
            // surface and rebuild it on the decode device's own instance, which is
            // required for the decoded textures to be bindable (true zero copy).
            drop(surface);
            drop(display_ctx);
            if let Some((surface, config)) = configured_surface(&self.decode_ctx, window, width, height)
            {
                println!("present: zero-copy on the decode GPU {} (no readback)", decode_gpu.name);
                let sink = self.make_sink(self.decode_ctx.clone(), surface, config);
                let textures = self.decode_textures.to_vec();
                return Present { sink, textures };
            }
            // Unexpected (same GPU yet could not configure): reopen and cross-copy.
            let (display_ctx, surface, config) = open_display_gpu(window, width, height)
                .expect("no GPU on this host can present to the window");
            return self.cross_gpu_present(display_ctx, surface, config);
        }

        println!(
            "present: cross-GPU fallback (PRIME) -- decode on {}, present on {} \
             (one readback per frame, not zero-copy)",
            decode_gpu.name, display_gpu.name
        );
        self.cross_gpu_present(display_ctx, surface, config)
    }

    /// Present the decoded frames on a *different* GPU than the one that decoded
    /// them: read each back from the decode GPU once, re-upload to the display GPU.
    fn cross_gpu_present(
        &self,
        display_ctx: GpuContext,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    ) -> Present {
        let textures = self
            .decode_textures
            .iter()
            .map(|t| readback_rgba(&self.decode_ctx, t, self.width, self.height))
            .map(|bytes| upload_rgba(&display_ctx, self.width, self.height, &bytes))
            .collect();
        let sink = self.make_sink(display_ctx, surface, config);
        Present { sink, textures }
    }

    /// Build a `WgpuSink` presenting to an already-configured surface on `ctx`.
    fn make_sink(
        &self,
        ctx: GpuContext,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
    ) -> WgpuSink {
        let mut sink = WgpuSink::with_surface(ctx, surface, config);
        let rgba = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(30 << 16),
        };
        sink.configure_pipeline(&rgba).expect("sink configure");
        sink
    }

    /// Build the present path once, when a real (non-zero) window size is first
    /// known. A no-op if already built or the size is still zero.
    fn ensure_present(&mut self, width: u32, height: u32) {
        if self.present.is_some() || width == 0 || height == 0 {
            return;
        }
        let Some(window) = self.window.clone() else { return };
        self.present = Some(self.build_present(&window, width, height));
        self.last_advance = Instant::now();
        window.request_redraw();
    }

    /// Present the current texture, advancing the source frame once
    /// `FRAME_INTERVAL` has elapsed so playback runs at a watchable rate and loops.
    fn present_current(&mut self) {
        let Some(present) = self.present.as_mut() else { return };
        let tex = &present.textures[self.idx];
        let frame = Frame::new(
            MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                self.width,
                self.height,
                Arc::new(WgpuTextureKeepAlive(tex.clone())),
            )),
            FrameTiming::default(),
            self.idx as u64,
        );
        block_on(present.sink.process(PipelinePacket::DataFrame(frame), &mut NullSink))
            .expect("present");

        // A periodic heartbeat so the live present is visible in the terminal too.
        let n = present.sink.presented_count();
        if n % 120 == 0 {
            eprintln!("  presented {n} frames (looping the {}-frame clip)", present.textures.len());
        }

        if self.last_advance.elapsed() >= FRAME_INTERVAL {
            self.idx = (self.idx + 1) % present.textures.len();
            self.last_advance = Instant::now();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("g2g: Vulkan Video -> WgpuSink")
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let size = window.inner_size();
        self.window = Some(window);
        // Build now if the compositor already gave us a real size, else wait for
        // the first Resized. Only ONE surface is ever created per window: creating
        // a second on the same Wayland surface is a protocol error.
        self.ensure_present(size.width, size.height);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }
            WindowEvent::Resized(new) => {
                // Build once, the first time we learn a non-zero size (Wayland
                // often reports 0x0 at creation). Later resizes are handled by the
                // sink's own surface-reconfigure on the next present.
                self.ensure_present(new.width, new.height);
            }
            WindowEvent::RedrawRequested => {
                self.present_current();
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

/// Create a surface on `ctx` for `window` and configure it, returning the
/// configured surface if `ctx`'s adapter can actually drive it. `None` (rather
/// than a process-aborting validation panic) when it cannot, which is how the
/// PRIME case is detected: `is_surface_supported` can report a false positive, so
/// the actual check is whether `configure` succeeds, caught in an error scope.
fn configured_surface(
    ctx: &GpuContext,
    window: &Arc<Window>,
    width: u32,
    height: u32,
) -> Option<(wgpu::Surface<'static>, wgpu::SurfaceConfiguration)> {
    let surface = ctx.instance.create_surface(window.clone()).ok()?;
    if !ctx.adapter.is_surface_supported(&surface) {
        return None;
    }
    let config = surface_config(&surface, &ctx.adapter, width, height)?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    surface.configure(&ctx.device, &config);
    if block_on(scope.pop()).is_some() {
        return None;
    }
    Some((surface, config))
}

/// Whether two adapters (from different wgpu instances) are the same physical
/// GPU, by PCI vendor + device id. On a single-GPU host the decode and display
/// adapters match; on a PRIME laptop they differ.
fn same_physical_device(a: &wgpu::AdapterInfo, b: &wgpu::AdapterInfo) -> bool {
    a.vendor == b.vendor && a.device == b.device
}

/// Enumerate the host's GPUs and return the first hardware one that can present
/// to a fresh surface for `window` (skips the CPU/llvmpipe fallback). On a PRIME
/// host this picks the integrated GPU that drives the display, since the discrete
/// (decode) GPU is the one that failed `configured_surface`.
fn open_display_gpu(
    window: &Arc<Window>,
    width: u32,
    height: u32,
) -> Option<(GpuContext, wgpu::Surface<'static>, wgpu::SurfaceConfiguration)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let surface = instance.create_surface(window.clone()).ok()?;
    for adapter in block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN)) {
        if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
            continue; // llvmpipe: correct but too slow to watch
        }
        if !adapter.is_surface_supported(&surface) {
            continue;
        }
        let Some(config) = surface_config(&surface, &adapter, width, height) else {
            continue;
        };
        let Ok((device, queue)) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("g2g-display-gpu"),
            required_limits: adapter.limits(),
            ..Default::default()
        })) else {
            continue;
        };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        surface.configure(&device, &config);
        if block_on(scope.pop()).is_some() {
            continue; // claims support but cannot actually configure a swapchain
        }
        let ctx = GpuContext::from_wgpu(instance.clone(), adapter, device, queue);
        return Some((ctx, surface, config));
    }
    None
}

/// A surface config for `adapter` at `width` x `height`, preferring a plain
/// (non-sRGB) format so decoded RGBA presents without a regamma. `None` if the
/// adapter advertises no formats for the surface.
fn surface_config(
    surface: &wgpu::Surface<'static>,
    adapter: &wgpu::Adapter,
    width: u32,
    height: u32,
) -> Option<wgpu::SurfaceConfiguration> {
    let caps = surface.get_capabilities(adapter);
    if caps.formats.is_empty() {
        return None;
    }
    let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
    Some(wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: width.max(1),
        height: height.max(1),
        present_mode: wgpu::PresentMode::AutoVsync,
        alpha_mode: caps.alpha_modes[0],
        view_formats: Vec::new(),
        desired_maximum_frame_latency: 2,
    })
}

/// Read an RGBA8 texture on `ctx` back to a tightly-packed byte buffer (one-time,
/// for the cross-GPU transport). Handles the 256-byte row alignment wgpu requires.
fn readback_rgba(ctx: &GpuContext, tex: &wgpu::Texture, width: u32, height: u32) -> Vec<u8> {
    let unpadded = (width * 4) as usize;
    let padded = unpadded.next_multiple_of(256);
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * height as usize) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .expect("poll readback");
    rx.recv().expect("readback channel").expect("map readback");

    let mapped = slice.get_mapped_range();
    let mut out = Vec::with_capacity(unpadded * height as usize);
    for row in 0..height as usize {
        let start = row * padded;
        out.extend_from_slice(&mapped[start..start + unpadded]);
    }
    drop(mapped);
    buffer.unmap();
    out
}

/// Upload tightly-packed RGBA8 bytes to a sampleable texture on `ctx`.
fn upload_rgba(ctx: &GpuContext, width: u32, height: u32, data: &[u8]) -> wgpu::Texture {
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("display-gpu-frame"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: RGBA,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    tex
}

/// A discarding sink for the terminal `WgpuSink` (it forwards nothing).
struct NullSink;
impl OutputSink for NullSink {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> core::pin::Pin<Box<dyn core::future::Future<Output = Result<PushOutcome, G2gError>> + 'a>>
    {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}
