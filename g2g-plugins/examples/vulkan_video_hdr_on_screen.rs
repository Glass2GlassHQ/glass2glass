//! On-screen live demo of the HDR swapchain present (M575): a 10-bit HDR10 (PQ,
//! BT.2020) clip is decoded on the GPU via `VK_KHR_video_*` into an `Rgba16Float`
//! texture holding the stream's PQ-encoded R'G'B' (the decoder's passthrough
//! output), and presented by [`VulkanHdrSink`] to a raw Vulkan swapchain with the
//! best HDR colour space the surface offers (`HDR10_ST2084` PQ, falling back to
//! scRGB linear, then SDR) plus BT.2020 mastering metadata.
//!
//! wgpu 29 cannot express a swapchain colour space, so the sink owns a raw
//! `VK_KHR_swapchain` on the decode device (window + event loop belong to the app,
//! which is why this is an example, not a self-checking test). Whether the picture
//! is actually HDR on screen depends on the display (an HDR monitor in HDR mode)
//! and the compositor (Wayland colour-management / Windows Advanced Color); the
//! terminal prints the negotiated colour space so you can see what was picked.
//!
//! Run (needs a Vulkan H.265 decode + present GPU, e.g. the RTX 3060, and a
//! display; ideally an HDR display in HDR mode):
//!
//! ```sh
//! cargo run --release -p g2g-plugins --features hdr-present \
//!     --example vulkan_video_hdr_on_screen                 # bundled PQ clip
//! cargo run --release -p g2g-plugins --features hdr-present \
//!     --example vulkan_video_hdr_on_screen -- my_hdr10.hevc
//! ```
//!
//! Close the window (or Esc) to quit.

use std::sync::Arc;
use std::time::{Duration, Instant};

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanhdrsink::{HdrMasteringDisplay, VulkanHdrSink};
use g2g_plugins::vulkanvideo::{
    extract_h265_parameter_sets, open_h265_decode_device, to_std_h265_params, VulkanVideoError,
};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Bundled 640x480 HDR10 (PQ / BT.2020) clip so the demo runs with no arguments.
const BUNDLED_CLIP: &[u8] = include_bytes!("../tests/fixtures/h265_640x480_pq.hevc");
const W: u32 = 640;
const H: u32 = 480;

/// Playback pacing (~15 fps) so the loop is easy to follow.
const FRAME_INTERVAL: Duration = Duration::from_millis(66);

struct App {
    // The decode device must outlive the textures + sink.
    device: g2g_plugins::vulkanvideo::VulkanVideoDevice,
    textures: Vec<wgpu::Texture>,
    window: Option<Arc<Window>>,
    sink: Option<VulkanHdrSink>,
    idx: usize,
    last_advance: Instant,
}

impl App {
    fn ensure_sink(&mut self, width: u32, height: u32) {
        if self.sink.is_some() || width == 0 || height == 0 {
            return;
        }
        let window = self.window.as_ref().expect("window set before ensure_sink");
        let display = window.display_handle().expect("display handle").as_raw();
        let win = window.window_handle().expect("window handle").as_raw();
        // SAFETY: the window (held in `self.window`) outlives the sink; the handles
        // are valid for the running window system.
        match unsafe {
            VulkanHdrSink::new(
                &self.device,
                display,
                win,
                width,
                height,
                HdrMasteringDisplay::default(),
            )
        } {
            Ok(sink) => {
                eprintln!(
                    "HDR present: colour space = {:?}, format = {:?}",
                    sink.color_space(),
                    sink.format()
                );
                self.sink = Some(sink);
            }
            Err(e) => eprintln!("could not build HDR present sink: {e:?}"),
        }
    }

    fn present_current(&mut self) {
        let Some(sink) = self.sink.as_mut() else {
            return;
        };
        if self.textures.is_empty() {
            return;
        }
        // SAFETY: the texture is a live decode-device texture in SHADER_READ_ONLY
        // layout (the decoder's GPU-texture output); no other GPU work runs now.
        if let Err(e) = unsafe { sink.present(&self.textures[self.idx]) } {
            eprintln!("present error: {e:?}");
        }
        let n = sink.presented_count();
        if n % 120 == 0 && n > 0 {
            eprintln!(
                "  presented {n} frames (looping {} textures)",
                self.textures.len()
            );
        }
        if self.last_advance.elapsed() >= FRAME_INTERVAL {
            self.idx = (self.idx + 1) % self.textures.len();
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
            .with_title("g2g: Vulkan Video HDR10 present")
            .with_inner_size(winit::dpi::PhysicalSize::new(W, H));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let size = window.inner_size();
        self.window = Some(window);
        self.ensure_sink(size.width, size.height);
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
            } => event_loop.exit(),
            WindowEvent::Resized(new) => {
                if self.sink.is_none() {
                    self.ensure_sink(new.width, new.height);
                } else if let Some(sink) = self.sink.as_mut() {
                    if new.width > 0 && new.height > 0 {
                        let _ = sink.resize(new.width, new.height);
                    }
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

fn main() {
    let clip: Vec<u8> = match std::env::args().nth(1) {
        Some(path) => std::fs::read(&path).expect("read clip"),
        None => BUNDLED_CLIP.to_vec(),
    };

    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no Vulkan H.265 decode device: {e:?}");
            return;
        }
    };
    if !device.present_capable() {
        eprintln!("decode device cannot present (VK_KHR_swapchain absent); on a PRIME laptop the display GPU differs from the decode GPU, which HDR present does not bridge");
        return;
    }

    let ps = extract_h265_parameter_sets(&clip).expect("vps/sps/pps");
    let std_ps = to_std_h265_params(&ps);
    let session = device.create_h265_session(&std_ps, W, H).expect("session");
    // Passthrough GPU decode: the Rgba16Float texture keeps the stream's PQ
    // encoding, which the HDR10 swapchain presents directly.
    let mut dec = match device.create_h265_dpb_decoder_gpu(&session, &ps) {
        Ok(d) => d,
        Err(VulkanVideoError::NoComputeQueue) => {
            eprintln!("no distinct compute queue for the GPU-texture path");
            return;
        }
        Err(e) => {
            eprintln!("build GPU decoder: {e:?}");
            return;
        }
    };
    let textures = dec
        .decode_all_to_textures(&clip)
        .expect("decode to textures");
    drop(dec);
    drop(session);
    eprintln!("decoded {} HDR textures; opening window", textures.len());

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        device,
        textures,
        window: None,
        sink: None,
        idx: 0,
        last_advance: Instant::now(),
    };
    event_loop.run_app(&mut app).expect("run app");
}
