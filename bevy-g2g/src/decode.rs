//! Bring-your-own-device decode (from the M741 demo): a stock windowed Bevy
//! app keeps its own wgpu device; g2g joins it via `GpuContext::from_wgpu` and
//! decodes video straight onto it. Every decoded frame is a `wgpu::Texture`
//! Bevy binds in its own render graph: no second device, no readback, no copy.
//!
//! [`VideoPlayerPlugin`] runs the pipeline (`filesrc -> h264parse ->
//! ffmpegdec -> videoconvert -> vello overlay -> appsink`) on its own thread
//! and binds the current frame to the material of every mesh tagged
//! [`VideoScreen`]. While the stream is live the latest frame shows; once
//! decode ends playback loops at `loop_fps`.
//!
//! Device handles are cloned in `Plugin::finish`, which Bevy calls after the
//! renderer's own finish (resources installed) and before `cleanup()` moves
//! the render world to the pipelined-rendering thread, the only window where
//! both the main-world handles and the render-world `RenderInstance` are
//! reachable.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use bevy::asset::AssetId;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::renderer::{RenderAdapter, RenderDevice, RenderInstance, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};

use g2g_core::element::DynAsyncElement;
use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{block_on, run_linear_chain};
use g2g_core::{Caps, Dim, PipelineClock, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::appsink::{register_appsink_pull, AppSink, AppSinkPull, Pull};
use g2g_plugins::ffmpegdec::FfmpegH264Dec;
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::gpu::{texture_of, GpuContext};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::vellooverlay::VelloAnalyticsOverlay;
use g2g_plugins::videoconvert::VideoConvert;

/// appsink delivery channel shared by the pipeline thread and the app.
const CHANNEL: &str = "bevy-g2g-decode";

/// The overlay's output texture is created with this extra view format
/// (sRGB), so the app's view samples the video with correct gamma in a lit
/// scene.
const SRGB_VIEW: &[wgpu::TextureFormat] = &[wgpu::TextureFormat::Rgba8UnormSrgb];

/// Decodes an H.264 Annex-B clip onto Bevy's own wgpu device and plays it on
/// every [`VideoScreen`] mesh. Add after `DefaultPlugins`.
#[derive(Debug)]
pub struct VideoPlayerPlugin {
    /// Path to the H.264 Annex-B stream to play.
    pub source: String,
    /// Loop rate once the clip has fully decoded.
    pub loop_fps: u32,
}

impl VideoPlayerPlugin {
    pub fn new(source: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            loop_fps: 30,
        }
    }
}

/// Tag a mesh entity with this and its `StandardMaterial` plays the video.
#[derive(Component, Debug)]
pub struct VideoScreen;

/// Decode + playback state, readable by the app (e.g. a smoke-run exit).
#[derive(Resource)]
pub struct VideoPlayback {
    /// One reserved `Handle<Image>` per decoded frame, in decode order. The
    /// strong handles keep the reserved asset ids (and render-world entries)
    /// alive.
    frames: Vec<Handle<Image>>,
    ended: bool,
    idx: usize,
    last_advance: Instant,
    bound: u64,
    frame_interval_ms: u128,
}

impl VideoPlayback {
    pub fn frames_decoded(&self) -> usize {
        self.frames.len()
    }

    /// Distinct frames bound to a screen material so far.
    pub fn frames_bound(&self) -> u64 {
        self.bound
    }

    pub fn ended(&self) -> bool {
        self.ended
    }
}

impl Plugin for VideoPlayerPlugin {
    fn build(&self, app: &mut App) {
        let pending = PendingTextures::default();
        app.insert_resource(pending.clone())
            .insert_resource(VideoPlayback {
                frames: Vec::new(),
                ended: false,
                idx: 0,
                last_advance: Instant::now(),
                bound: 0,
                frame_interval_ms: 1_000 / self.loop_fps.max(1) as u128,
            })
            .add_systems(Update, (ingest_frames, show_current));
        app.sub_app_mut(RenderApp)
            .insert_resource(pending)
            .add_systems(
                Render,
                register_textures.in_set(RenderSystems::PrepareAssets),
            );
    }

    fn finish(&self, app: &mut App) {
        // The embedder handoff (M263): clone Bevy's own wgpu handles into g2g.
        // Everything g2g's GPU elements produce now lives on Bevy's device.
        let device = app.world().resource::<RenderDevice>().wgpu_device().clone();
        let queue = (**app.world().resource::<RenderQueue>().0).clone();
        let adapter = (**app.world().resource::<RenderAdapter>().0).clone();
        // Bevy keeps the instance in the render world only.
        let instance = (**app
            .sub_app(RenderApp)
            .world()
            .resource::<RenderInstance>()
            .0)
            .clone();
        let ctx = GpuContext::from_wgpu(instance, adapter, device, queue);

        let pull = register_appsink_pull(CHANNEL);
        app.insert_resource(VideoPull(pull));
        let source = self.source.clone();
        std::thread::spawn(move || run_decode(ctx, source));
    }
}

/// The g2g pipeline, on its own thread: decode the clip and hand each frame
/// to the app as a `wgpu::Texture` on Bevy's device. The appsink pull channel
/// is bounded, so a slow app back-pressures the decode instead of piling
/// frames.
fn run_decode(ctx: GpuContext, clip: String) {
    // Placeholder geometry: negotiation fixates before data flows, and the
    // parser re-fixes from the SPS, so any clip geometry works.
    let caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    };
    let mut src = FileSrc::new(&clip, caps);
    let mut parse = H264Parse::reframing();
    let mut dec = FfmpegH264Dec::new();
    let mut convert = VideoConvert::new(RawVideoFormat::Rgba8);
    // The System -> WgpuTexture hop: renders the RGBA frame into a texture on
    // the shared (= Bevy's) device.
    let mut overlay = VelloAnalyticsOverlay::new().with_context(ctx);
    let mut sink = AppSink::new().with_channel(CHANNEL);
    let transforms: Vec<&mut dyn DynAsyncElement> =
        vec![&mut parse, &mut dec, &mut convert, &mut overlay];
    let clock = ZeroClock;
    match block_on(run_linear_chain(&mut src, transforms, &mut sink, &clock, 4)) {
        Ok(stats) => info!("decode pipeline done: {} frames", stats.frames_consumed),
        Err(e) => error!("decode pipeline failed: {e:?}"),
    }
}

/// The sinks here do not pace to a clock; the runner only needs a `now_ns`.
struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Pull handle for the appsink channel.
#[derive(Resource)]
struct VideoPull(AppSinkPull);

/// Textures pulled in the main world, awaiting registration as render-world
/// [`GpuImage`]s under their reserved asset ids. Shared with the render app.
#[derive(Resource, Clone, Default)]
struct PendingTextures(Arc<Mutex<Vec<PendingTexture>>>);

type PendingTexture = (AssetId<Image>, wgpu::Texture);

/// Drain the appsink: each pulled frame carries a `wgpu::Texture` already on
/// Bevy's device. Reserve an image handle for it and queue the render-world
/// registration; no pixel data crosses the CPU here.
fn ingest_frames(
    pull: Option<Res<VideoPull>>,
    images: Res<Assets<Image>>,
    pending: Res<PendingTextures>,
    mut playback: ResMut<VideoPlayback>,
) {
    let Some(pull) = pull else {
        return;
    };
    loop {
        match pull.0.try_pull() {
            Pull::Frame(frame) => {
                let MemoryDomain::WgpuTexture(owned) = &frame.domain else {
                    warn!("non-GPU frame from the pipeline; dropping");
                    continue;
                };
                let Some(texture) = texture_of(owned) else {
                    warn!("foreign keep-alive on a WgpuTexture frame; dropping");
                    continue;
                };
                // Clones share the refcounted wgpu texture; it outlives the
                // g2g frame we drop at the end of this iteration.
                let texture = texture.clone();
                let handle = images.reserve_handle();
                pending.0.lock().unwrap().push((handle.id(), texture));
                if playback.frames.is_empty() {
                    info!("first decoded frame arrived on Bevy's device (zero-copy)");
                }
                playback.frames.push(handle);
            }
            Pull::Empty => break,
            Pull::Ended => {
                if !playback.ended {
                    playback.ended = true;
                    info!(
                        "decode ended: {} frames; looping playback",
                        playback.frames.len()
                    );
                }
                break;
            }
        }
    }
}

/// Bind the current frame's texture to every screen material: the latest
/// frame while the stream is live, a fixed-rate loop once it ends. Binding is
/// a material asset change, so Bevy rebuilds the bind group onto the g2g
/// texture.
fn show_current(
    mut playback: ResMut<VideoPlayback>,
    screens: Query<&MeshMaterial3d<StandardMaterial>, With<VideoScreen>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if playback.frames.is_empty() {
        return;
    }
    if playback.ended {
        if playback.last_advance.elapsed().as_millis() >= playback.frame_interval_ms {
            playback.idx = (playback.idx + 1) % playback.frames.len();
            playback.last_advance = Instant::now();
        }
    } else {
        playback.idx = playback.frames.len() - 1;
    }
    let want = playback.frames[playback.idx].clone();
    for screen in &screens {
        // Only touch the material on an actual frame change: `get_mut` marks
        // the asset modified, and an unconditional touch would rebuild every
        // frame.
        let unchanged = materials
            .get(&screen.0)
            .is_some_and(|m| m.base_color_texture.as_ref() == Some(&want));
        if unchanged {
            continue;
        }
        if let Some(mut mat) = materials.get_mut(&screen.0) {
            mat.base_color_texture = Some(want.clone());
            playback.bound += 1;
            if playback.bound == 1 {
                info!("sampling the g2g-decoded texture on a VideoScreen");
            }
        }
    }
}

/// Registers pulled textures with the render world: wraps each
/// `wgpu::Texture` as a [`GpuImage`] under its reserved asset id, before
/// material bind groups prepare, so a material can reference the handle the
/// same frame.
fn register_textures(
    pending: Res<PendingTextures>,
    device: Res<RenderDevice>,
    mut images: ResMut<RenderAssets<GpuImage>>,
) {
    let mut list = pending.0.lock().unwrap();
    if list.is_empty() {
        return;
    }
    for (id, texture) in list.drain(..) {
        // Sample through the sRGB view so the video's gamma survives Bevy's
        // lighting + tonemapping (the texture itself is Rgba8Unorm).
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("bevy-g2g-frame-srgb"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            // The texture also carries STORAGE_BINDING (Vello writes it); an
            // sRGB view cannot, so narrow this view to sampling only.
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bevy-g2g-frame"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let texture_descriptor = wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: texture.width(),
                height: texture.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture.format(),
            usage: texture.usage(),
            view_formats: SRGB_VIEW,
        };
        images.insert(
            id,
            GpuImage {
                texture: texture.into(),
                texture_view: view.into(),
                sampler,
                texture_descriptor,
                texture_view_descriptor: None,
                had_data: true,
            },
        );
    }
}
