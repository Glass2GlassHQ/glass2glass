//! On-screen interactive scrubber for the Vulkan Video wedge (M498): the
//! random-access ("pull") counterpart of the linear `vulkan_video_on_screen`
//! demo. Instead of pre-decoding the clip and playing it forward, this drives a
//! [`VulkanVideoPlayer`]: as you scrub the timeline the app calls
//! `frame_at_index` and the player decodes forward from the enclosing keyframe
//! on demand, straight into a GPU-resident RGBA `wgpu::Texture` that `WgpuSink`
//! presents. This is the model a timeline viewer (e.g. Rerun) needs, and the
//! thing Rerun's native pipeline lacks: hardware decode into the render device
//! with no CPU round-trip.
//!
//! The title bar shows the current frame and the running **decode count** so the
//! cache is visible: scrub forward and the count climbs; scrub back over frames
//! you have already visited and it stays flat (served from cache, no GPU work).
//!
//! Controls:
//!   Left / Right arrows : step one frame back / forward
//!   Down / Up arrows    : jump ~one GOP (5 frames) back / forward
//!   Home / End          : first / last frame
//!   Left-click / drag   : scrub to a position along the window width
//!   Space               : toggle auto-play
//!   Esc                 : quit
//!
//! Present paths mirror `vulkan_video_on_screen`: **zero-copy** when the decode
//! GPU also drives the display; a **cross-GPU** readback+upload fallback on a
//! hybrid / PRIME host (the case on this laptop: decode on the RTX 3060, present
//! on the AMD iGPU).
//!
//! Run (needs a Vulkan H.264 decode GPU and a display):
//!
//! ```sh
//! cargo run --release -p g2g-plugins --features vulkan-video,wgpu-sink \
//!     --example vulkan_video_scrubber                  # the bundled 640x480 clip
//! cargo run --release -p g2g-plugins --features vulkan-video,wgpu-sink \
//!     --example vulkan_video_scrubber -- my.h264       # a stream of your own
//! ```

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
use g2g_plugins::vulkanvideo::{open_h264_decode_device, VulkanVideoError, VulkanVideoPlayer};
use g2g_plugins::wgpusink::WgpuSink;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// The 640x480 baseline clip the tests use, embedded so the demo runs with no
/// arguments (two GOPs of IDR + P frames, so scrubbing crosses a GOP boundary).
const BUNDLED_CLIP: &[u8] = include_bytes!("../tests/fixtures/h264_640x480.h264");

/// Auto-play frame interval (Space toggles it); ~15 fps is easy to follow.
const FRAME_INTERVAL: Duration = Duration::from_millis(66);

/// The decoded RGBA texture layout (what the GPU decoder converts NV12 into).
const RGBA: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn main() {
    let clip_path = std::env::args().nth(1);
    let clip: Vec<u8> = match &clip_path {
        Some(p) => std::fs::read(p).unwrap_or_else(|e| panic!("read {p}: {e}")),
        None => BUNDLED_CLIP.to_vec(),
    };

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

    // The random-access player: owns the device, builds the keyframe/POC index.
    let player = match VulkanVideoPlayer::new(device, clip, 30) {
        Ok(p) => p,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!(
                "this GPU exposes no separate compute queue for the NV12->RGBA pass; \
                 the texture path needs one."
            );
            return;
        }
        Err(e) => panic!("failed to build player: {e:?}"),
    };
    let (width, height) = player.dimensions();
    let n = player.frame_count();
    if n == 0 {
        eprintln!("stream produced no frames; nothing to show.");
        return;
    }
    let decode_ctx = player.gpu_context();
    println!(
        "loaded {n} frames at {width}x{height} on {}",
        decode_ctx.adapter.get_info().name
    );
    println!(
        "scrub: Left/Right = +/-1 frame, Down/Up = +/-1 GOP, Home/End, click/drag, Space = play, Esc"
    );

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        player,
        decode_ctx,
        width,
        height,
        frames: n,
        window: None,
        present: None,
        shown: None,
        cur: 0,
        playing: false,
        dragging: false,
        last_advance: Instant::now(),
    };
    if let Err(e) = event_loop.run_app(&mut app) {
        eprintln!("event loop ended: {e}");
    }
}

/// The chosen present path: a sink plus how a decode-GPU texture reaches it.
struct Present {
    sink: WgpuSink,
    mode: Mode,
}

/// How a decoded texture (always on the decode GPU) gets to the display.
enum Mode {
    /// Decode GPU drives the display: present the decoded texture directly.
    ZeroCopy,
    /// Hybrid/PRIME: read each frame back off the decode GPU and re-upload it to
    /// this display-GPU context (the transport a real playbin uses on PRIME).
    CrossGpu(GpuContext),
}

struct App {
    player: VulkanVideoPlayer,
    /// The decode GPU shared context (source of the decoded textures).
    decode_ctx: GpuContext,
    width: u32,
    height: u32,
    frames: usize,
    window: Option<Arc<Window>>,
    present: Option<Present>,
    /// The display-GPU texture for the currently shown frame, cached so repeated
    /// redraws of the same frame do not re-decode or re-transfer.
    shown: Option<(usize, wgpu::Texture)>,
    cur: usize,
    playing: bool,
    dragging: bool,
    last_advance: Instant,
}

impl App {
    /// Move to frame `i` (clamped), pausing auto-play (a manual scrub).
    fn seek_to(&mut self, i: usize) {
        self.cur = i.min(self.frames - 1);
        self.playing = false;
    }

    /// Step `delta` frames with wraparound, pausing auto-play.
    fn step(&mut self, delta: i64) {
        let n = self.frames as i64;
        self.cur = (self.cur as i64 + delta).rem_euclid(n) as usize;
        self.playing = false;
    }

    /// Scrub to the frame under a window-relative x (0..width) fraction.
    fn scrub_to_x(&mut self, x: f64) {
        let w = self.window.as_ref().map(|w| w.inner_size().width).unwrap_or(self.width).max(1);
        let frac = (x / w as f64).clamp(0.0, 1.0);
        let i = (frac * (self.frames as f64 - 1.0)).round() as usize;
        self.seek_to(i);
    }

    /// Ensure `shown` holds the display-GPU texture for the current frame. Decodes
    /// via the player (cache hit if revisited) and, on the cross-GPU path, reads
    /// back + re-uploads. Returns false on a decode error.
    fn ensure_shown(&mut self) -> bool {
        if self.shown.as_ref().map(|(i, _)| *i) == Some(self.cur) {
            return true;
        }
        let cur = self.cur;
        let decode_tex = match self.player.frame_at_index(cur) {
            Ok(t) => t.clone(),
            Err(e) => {
                eprintln!("decode frame {cur}: {e:?}");
                return false;
            }
        };
        let (w, h) = (self.width, self.height);
        let decode_ctx = self.decode_ctx.clone();
        let display_tex = match self.present.as_ref().map(|p| &p.mode) {
            Some(Mode::CrossGpu(display_ctx)) => {
                let bytes = readback_rgba(&decode_ctx, &decode_tex, w, h);
                upload_rgba(display_ctx, w, h, &bytes)
            }
            // Zero-copy (or not yet built): the decode texture is presented as is.
            _ => decode_tex,
        };
        self.shown = Some((cur, display_tex));
        self.update_title();
        true
    }

    /// Present the current frame, advancing when auto-play is on.
    fn present_current(&mut self) {
        if self.present.is_none() || !self.ensure_shown() {
            return;
        }
        let cur = self.cur;
        let (w, h) = (self.width, self.height);
        let tex = self.shown.as_ref().expect("ensured").1.clone();
        let present = self.present.as_mut().expect("checked");
        let frame = Frame::new(
            MemoryDomain::WgpuTexture(OwnedWgpuTexture::new(
                w,
                h,
                Arc::new(WgpuTextureKeepAlive(tex)),
            )),
            FrameTiming::default(),
            cur as u64,
        );
        block_on(present.sink.process(PipelinePacket::DataFrame(frame), &mut NullSink))
            .expect("present");

        if self.playing && self.last_advance.elapsed() >= FRAME_INTERVAL {
            self.cur = (self.cur + 1) % self.frames;
            self.last_advance = Instant::now();
        }
    }

    fn update_title(&self) {
        if let Some(window) = self.window.as_ref() {
            let mode = match self.present.as_ref().map(|p| &p.mode) {
                Some(Mode::ZeroCopy) => "zero-copy",
                Some(Mode::CrossGpu(_)) => "cross-GPU",
                None => "...",
            };
            window.set_title(&format!(
                "g2g scrubber [{}]  frame {}/{}  decodes: {}{}",
                mode,
                self.cur + 1,
                self.frames,
                self.player.decode_calls(),
                if self.playing { "  (playing)" } else { "" },
            ));
        }
    }

    /// Choose and build the present path (zero-copy vs cross-GPU), once a real
    /// window size is known.
    fn ensure_present(&mut self, width: u32, height: u32) {
        if self.present.is_some() || width == 0 || height == 0 {
            return;
        }
        let Some(window) = self.window.clone() else { return };
        let (display_ctx, surface, config) = open_display_gpu(&window, width, height)
            .expect("no GPU on this host can present to the window");
        let decode_gpu = self.decode_ctx.adapter.get_info();
        let display_gpu = display_ctx.adapter.get_info();

        let present = if same_physical_device(&decode_gpu, &display_gpu) {
            // Single-GPU host: rebuild the surface on the decode device's own
            // instance so its textures are bindable (true zero copy).
            drop(surface);
            drop(display_ctx);
            match configured_surface(&self.decode_ctx, &window, width, height) {
                Some((surface, config)) => {
                    println!("present: zero-copy on the decode GPU {} (no readback)", decode_gpu.name);
                    Present { sink: self.make_sink(self.decode_ctx.clone(), surface, config), mode: Mode::ZeroCopy }
                }
                None => {
                    let (display_ctx, surface, config) = open_display_gpu(&window, width, height)
                        .expect("no GPU on this host can present to the window");
                    println!("present: cross-GPU fallback on {} (one readback per frame)", display_ctx.adapter.get_info().name);
                    let sink = self.make_sink(display_ctx.clone(), surface, config);
                    Present { sink, mode: Mode::CrossGpu(display_ctx) }
                }
            }
        } else {
            println!(
                "present: cross-GPU (PRIME) -- decode on {}, present on {} (one readback per frame)",
                decode_gpu.name, display_gpu.name
            );
            let sink = self.make_sink(display_ctx.clone(), surface, config);
            Present { sink, mode: Mode::CrossGpu(display_ctx) }
        };
        self.present = Some(present);
        self.shown = None; // rebuild the shown texture on the sink's device
        self.last_advance = Instant::now();
        self.update_title();
        window.request_redraw();
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
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("g2g scrubber")
            .with_inner_size(winit::dpi::PhysicalSize::new(self.width, self.height));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let size = window.inner_size();
        self.window = Some(window);
        self.ensure_present(size.width, size.height);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(new) => self.ensure_present(new.width, new.height),
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key, state: ElementState::Pressed, .. },
                ..
            } => {
                let last = self.frames - 1;
                match logical_key {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::ArrowRight) => self.step(1),
                    Key::Named(NamedKey::ArrowLeft) => self.step(-1),
                    Key::Named(NamedKey::ArrowUp) => self.step(5),
                    Key::Named(NamedKey::ArrowDown) => self.step(-5),
                    Key::Named(NamedKey::Home) => self.seek_to(0),
                    Key::Named(NamedKey::End) => self.seek_to(last),
                    Key::Named(NamedKey::Space) => {
                        self.playing = !self.playing;
                        self.last_advance = Instant::now();
                        self.update_title();
                    }
                    _ => {}
                }
                if let Some(w) = self.window.as_ref() {
                    w.request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = state == ElementState::Pressed;
                if let Some(w) = self.window.as_ref() {
                    w.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } if self.dragging => {
                self.scrub_to_x(position.x);
                if let Some(w) = self.window.as_ref() {
                    w.request_redraw();
                }
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

/// Create + configure a surface on `ctx` for `window`, returning it only if the
/// adapter can actually drive it (the PRIME check: `configure` in an error scope).
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

/// Whether two adapters are the same physical GPU (PCI vendor + device id).
fn same_physical_device(a: &wgpu::AdapterInfo, b: &wgpu::AdapterInfo) -> bool {
    a.vendor == b.vendor && a.device == b.device
}

/// The first hardware GPU that can present to a fresh surface for `window`.
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
            continue;
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
            continue;
        }
        let ctx = GpuContext::from_wgpu(instance.clone(), adapter, device, queue);
        return Some((ctx, surface, config));
    }
    None
}

/// A surface config for `adapter`, preferring a non-sRGB format so decoded RGBA
/// presents without a regamma.
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

/// Read an RGBA8 texture on `ctx` back to tightly-packed bytes (cross-GPU
/// transport). Handles wgpu's 256-byte row alignment.
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

/// A discarding sink for the terminal `WgpuSink`.
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
