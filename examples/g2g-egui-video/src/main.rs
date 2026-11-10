// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::thread::{self, JoinHandle};

use eframe::egui;
use eframe::egui_wgpu::{RenderState, WgpuConfiguration, WgpuSetup, WgpuSetupExisting};
use eframe::wgpu;
use g2g_core::clock::MonotonicClock;
use g2g_core::element::DynAsyncElement;
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{block_on, run_linear_chain};
use g2g_core::{
    BitstreamAlignment, Caps, Colorimetry, Dim, G2gError, Rate, RawVideoFormat, VideoCodec,
};
use g2g_plugins::appsink::{register_appsink_pull, AppSink, AppSinkPull, Pull};
use g2g_plugins::clocksync::ClockSyncTransform;
use g2g_plugins::ffmpegdec::FfmpegH264Dec;
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::gpu::{texture_of, GpuContext};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::vellooverlay::VelloAnalyticsOverlay;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};

const APPSINK_CHANNEL: &str = "g2g-egui-video";
const LINK_CAPACITY: usize = 2;
const TEST_WIDTH: u32 = 1280;
const TEST_HEIGHT: u32 = 720;
const TEST_FPS: u32 = 30;

fn main() -> eframe::Result {
    let clip = std::env::args().nth(1);
    let gpu = block_on(GpuContext::headless()).expect("open a wgpu device");
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("g2g egui video"),
        wgpu_options: WgpuConfiguration {
            wgpu_setup: WgpuSetup::Existing(WgpuSetupExisting {
                instance: gpu.instance.clone(),
                adapter: gpu.adapter.clone(),
                device: gpu.device.clone(),
                queue: gpu.queue.clone(),
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    // registered before the pipeline builds its appsink, which looks the channel up
    let frames = register_appsink_pull(APPSINK_CHANNEL);
    let source_label = clip.clone().unwrap_or_else(|| "videotestsrc".to_owned());
    let pipeline = thread::spawn(move || run_pipeline(gpu, clip));
    eframe::run_native(
        "g2g egui video",
        options,
        Box::new(|creation| {
            let render_state = creation
                .wgpu_render_state
                .clone()
                .ok_or("eframe started without the wgpu renderer")?;
            Ok(Box::new(VideoApp {
                render_state,
                frames,
                pipeline: Some(pipeline),
                outcome: None,
                source_label,
                video: None,
                frames_shown: 0,
            }))
        }),
    )
}

// the overlay renders each RGBA frame into a new texture on `gpu`, the device egui draws with
fn run_pipeline(gpu: GpuContext, clip: Option<String>) -> Result<u64, G2gError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("build the pipeline thread's tokio runtime");
    let mut convert = VideoConvert::new(RawVideoFormat::Rgba8);
    let mut to_texture = VelloAnalyticsOverlay::new().with_context(gpu);
    let mut pace = ClockSyncTransform::new();
    let mut sink = AppSink::new().with_channel(APPSINK_CHANNEL);
    let stats = match clip {
        Some(clip) => {
            let mut source = FileSrc::new(clip, h264_placeholder_caps());
            let mut parse = H264Parse::reframing();
            let mut decode = FfmpegH264Dec::new();
            let transforms: Vec<&mut dyn DynAsyncElement> = vec![
                &mut parse,
                &mut decode,
                &mut convert,
                &mut to_texture,
                &mut pace,
            ];
            runtime.block_on(run_linear_chain(
                &mut source,
                transforms,
                &mut sink,
                &MonotonicClock,
                LINK_CAPACITY,
            ))
        }
        None => {
            let mut source = VideoTestSrc::new(TEST_WIDTH, TEST_HEIGHT, TEST_FPS, u64::MAX)
                .with_pattern(Pattern::Ball);
            let transforms: Vec<&mut dyn DynAsyncElement> =
                vec![&mut convert, &mut to_texture, &mut pace];
            runtime.block_on(run_linear_chain(
                &mut source,
                transforms,
                &mut sink,
                &MonotonicClock,
                LINK_CAPACITY,
            ))
        }
    }?;
    Ok(stats.frames_consumed)
}

// h264parse replaces the geometry and rate from the SPS before any frame is decoded
fn h264_placeholder_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: Colorimetry::UNKNOWN,
        pixel_aspect: g2g_core::PixelAspect::Any,
        alignment: BitstreamAlignment::Unspecified,
    }
}

struct VideoApp {
    render_state: RenderState,
    frames: AppSinkPull,
    pipeline: Option<JoinHandle<Result<u64, G2gError>>>,
    outcome: Option<String>,
    source_label: String,
    video: Option<(egui::TextureId, egui::Vec2)>,
    frames_shown: u64,
}

impl VideoApp {
    fn show_latest_frame(&mut self) {
        let mut latest = None;
        while let Pull::Frame(frame) = self.frames.try_pull() {
            latest = Some(frame);
        }
        let Some(frame) = latest else {
            return;
        };
        let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
            panic!("the pipeline ends in a wgpu texture, got another memory domain");
        };
        let texture = texture_of(owned).expect("a texture the overlay rendered");
        // the pixels are sRGB encoded, and egui samples its own textures through sRGB views
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        });
        let size = egui::vec2(texture.width() as f32, texture.height() as f32);
        let device = &self.render_state.device;
        let mut renderer = self.render_state.renderer.write();
        let id = match self.video {
            Some((id, _)) => {
                renderer.update_egui_texture_from_wgpu_texture(
                    device,
                    &view,
                    wgpu::FilterMode::Linear,
                    id,
                );
                id
            }
            None => renderer.register_native_texture(device, &view, wgpu::FilterMode::Linear),
        };
        self.video = Some((id, size));
        self.frames_shown += 1;
    }

    fn collect_outcome(&mut self) {
        let finished = self.pipeline.as_ref().is_some_and(JoinHandle::is_finished);
        if !finished {
            return;
        }
        let handle = self.pipeline.take().expect("checked above");
        self.outcome = Some(match handle.join() {
            Ok(Ok(frames)) => format!("ended after {frames} frames"),
            Ok(Err(error)) => format!("failed: {error:?}"),
            Err(_) => "the pipeline thread panicked".to_owned(),
        });
    }
}

impl eframe::App for VideoApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.show_latest_frame();
        self.collect_outcome();
        if self.pipeline.is_some() {
            ctx.request_repaint();
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::left("stream").show(ui, |ui| {
            ui.heading("g2g pipeline");
            ui.label(format!("source: {}", self.source_label));
            ui.label(format!("frames shown: {}", self.frames_shown));
            if let Some((_, size)) = self.video {
                ui.label(format!("size: {} x {}", size.x, size.y));
            }
            ui.label(self.outcome.as_deref().unwrap_or("playing"));
        });
        egui::CentralPanel::default().show(ui, |ui| match self.video {
            Some(texture) => {
                ui.add(egui::Image::new(texture).shrink_to_fit());
            }
            None => {
                ui.spinner();
            }
        });
    }
}
