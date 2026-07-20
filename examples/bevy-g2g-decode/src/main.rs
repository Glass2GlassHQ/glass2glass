//! Bevy + g2g bring-your-own-device decode demo (M741): the embedder owns the
//! wgpu device, g2g joins it.
//!
//! A stock Bevy app opens its window and render device the ordinary way (no
//! custom `RenderCreation`), then hands its `RenderDevice` / `RenderQueue` /
//! `RenderAdapter` / `RenderInstance` clones to [`GpuContext::from_wgpu`]
//! (M263). A g2g pipeline (`filesrc -> h264parse -> ffmpegdec -> videoconvert
//! -> vello overlay -> appsink`) then decodes an H.264 clip and lands every
//! frame in a `wgpu::Texture` created **on Bevy's device**, so the app binds it
//! directly in its own render graph: no second GPU device, no readback, no
//! copy. Bevy 0.19 pins the same wgpu 29 as g2g, so the handles are the same
//! types.
//!
//! The frames stream in over an `appsink` pull channel while the pipeline runs
//! on its own thread; the app samples the latest frame on a spinning cube and
//! loops the clip once decode ends. The inverse of `bevy-g2g-stream` (Bevy
//! renders, g2g encodes); here g2g decodes and Bevy consumes.
//!
//! Run (needs a display and a wgpu adapter):
//!
//! ```sh
//! cargo run --release                 # the bundled 640x480 test clip
//! cargo run --release -- my.h264      # any Annex-B H.264 stream
//! ```
//!
//! `G2G_EXIT_AFTER_SECS=5 cargo run --release` exits by itself (a smoke run).
//! Close the window or press Esc to quit.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use bevy::app::PluginsState;
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

/// The 640x480 baseline clip the g2g tests use (two GOPs of IDR + P frames).
const BUNDLED_CLIP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../g2g-plugins/tests/fixtures/h264_640x480.h264"
);

/// appsink delivery channel shared by the pipeline thread and the app.
const CHANNEL: &str = "bevy-decode";

/// The clip is authored at 30 fps; loop playback at that rate once decoded.
const FRAME_INTERVAL_MS: u128 = 33;

/// The overlay's output texture is created with this extra view format (sRGB),
/// so the app's view samples the video with correct gamma in a lit scene.
const SRGB_VIEW: &[wgpu::TextureFormat] = &[wgpu::TextureFormat::Rgba8UnormSrgb];

fn main() {
    let pending = PendingTextures::default();
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "g2g decode -> Bevy (shared wgpu device)".into(),
            resolution: (960, 720).into(),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(TextureBridgePlugin {
        pending: pending.clone(),
    })
    .insert_resource(pending)
    .insert_resource(Playback {
        frames: Vec::new(),
        ended: false,
        idx: 0,
        last_advance: Instant::now(),
        bound: 0,
    })
    .insert_resource(SmokeExit {
        started: Instant::now(),
        exit_after_secs: std::env::var("G2G_EXIT_AFTER_SECS")
            .ok()
            .and_then(|s| s.parse::<f64>().ok()),
    })
    .add_systems(Startup, setup)
    .add_systems(Update, (ingest_frames, show_current, spin_cube, smoke_exit));

    // The renderer initializes asynchronously inside DefaultPlugins; wait for
    // it, then finish the app so the device resources exist to be adopted.
    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
    }
    app.finish();

    // The embedder handoff (M263): clone Bevy's own wgpu handles into g2g.
    // Everything g2g's GPU elements produce now lives on Bevy's device.
    // Read them between finish() (which installs the render resources) and
    // cleanup() (which moves the render sub-app to the pipelined thread).
    let device = app.world().resource::<RenderDevice>().wgpu_device().clone();
    let queue = (**app.world().resource::<RenderQueue>().0).clone();
    let adapter = (**app.world().resource::<RenderAdapter>().0).clone();
    // Bevy keeps the instance in the render world only.
    let instance = (**app.sub_app(RenderApp).world().resource::<RenderInstance>().0).clone();
    let ctx = GpuContext::from_wgpu(instance, adapter, device, queue);

    app.cleanup();

    let pull = register_appsink_pull(CHANNEL);
    app.insert_resource(VideoPull(pull));
    std::thread::spawn(move || run_decode(ctx));

    app.run();
}

/// The g2g pipeline, on its own thread: decode the clip and hand each frame to
/// the app as a `wgpu::Texture` on Bevy's device. The appsink pull channel is
/// bounded, so a slow app back-pressures the decode instead of piling frames.
fn run_decode(ctx: GpuContext) {
    let clip = std::env::args()
        .nth(1)
        .unwrap_or_else(|| BUNDLED_CLIP.to_string());
    eprintln!("decoding {clip}");
    // Concrete caps: negotiation fixates before data flows, so the source
    // advertises the fixture's real geometry, and the parser re-fixes from the
    // SPS for a caller-supplied clip.
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
    // The System -> WgpuTexture hop: renders the RGBA frame (plus any analytics
    // boxes; none here) into a texture on the shared (= Bevy's) device.
    let mut overlay = VelloAnalyticsOverlay::new().with_context(ctx);
    let mut sink = AppSink::new().with_channel(CHANNEL);
    let transforms: Vec<&mut dyn DynAsyncElement> =
        vec![&mut parse, &mut dec, &mut convert, &mut overlay];
    let clock = ZeroClock;
    match block_on(run_linear_chain(&mut src, transforms, &mut sink, &clock, 4)) {
        Ok(stats) => eprintln!("decode pipeline done: {} frames", stats.frames_consumed),
        Err(e) => eprintln!("decode pipeline failed: {e:?}"),
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
struct PendingTextures(Arc<Mutex<Vec<(AssetId<Image>, wgpu::Texture)>>>);

/// Decoded-frame handles and playback state.
#[derive(Resource)]
struct Playback {
    /// One reserved `Handle<Image>` per decoded frame, in decode order. The
    /// strong handles keep the reserved asset ids (and render-world entries)
    /// alive.
    frames: Vec<Handle<Image>>,
    ended: bool,
    idx: usize,
    last_advance: Instant,
    /// Distinct frames bound to the material so far (the smoke-run evidence).
    bound: u64,
}

/// Optional self-exit for unattended smoke runs (`G2G_EXIT_AFTER_SECS`).
#[derive(Resource)]
struct SmokeExit {
    started: Instant,
    exit_after_secs: Option<f64>,
}

/// The material the video frames are bound to.
#[derive(Resource)]
struct ScreenMaterial(Handle<StandardMaterial>);

#[derive(Component)]
struct Spin;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // The cube the video plays on. Starts white until the first frame arrives.
    let screen = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 1.5, 2.0))),
        MeshMaterial3d(screen.clone()),
        Transform::from_xyz(0.0, 1.0, 0.0),
        Spin,
    ));
    commands.insert_resource(ScreenMaterial(screen));

    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.3, 0.35))),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    commands.spawn((
        PointLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
}

/// Drain the appsink: each pulled frame carries a `wgpu::Texture` already on
/// Bevy's device. Reserve an image handle for it and queue the render-world
/// registration; no pixel data crosses the CPU here.
fn ingest_frames(
    pull: Res<VideoPull>,
    images: Res<Assets<Image>>,
    pending: Res<PendingTextures>,
    mut playback: ResMut<Playback>,
) {
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
                    info!("decode ended: {} frames; looping playback", playback.frames.len());
                }
                break;
            }
        }
    }
}

/// Bind the current frame's texture to the cube's material: the latest frame
/// while the stream is live, a 30 fps loop once it ends. Binding is a material
/// asset change, so Bevy rebuilds the bind group onto the g2g texture.
fn show_current(
    mut playback: ResMut<Playback>,
    screen: Res<ScreenMaterial>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if playback.frames.is_empty() {
        return;
    }
    if playback.ended {
        if playback.last_advance.elapsed().as_millis() >= FRAME_INTERVAL_MS {
            playback.idx = (playback.idx + 1) % playback.frames.len();
            playback.last_advance = Instant::now();
        }
    } else {
        playback.idx = playback.frames.len() - 1;
    }
    let want = playback.frames[playback.idx].clone();
    // Only touch the material on an actual frame change: `get_mut` marks the
    // asset modified, and an unconditional touch would rebuild every frame.
    let unchanged = materials
        .get(&screen.0)
        .is_some_and(|m| m.base_color_texture.as_ref() == Some(&want));
    if unchanged {
        return;
    }
    if let Some(mut mat) = materials.get_mut(&screen.0) {
        mat.base_color_texture = Some(want);
        playback.bound += 1;
        if playback.bound == 1 {
            info!("sampling the g2g-decoded texture on the cube");
        }
    }
}

fn spin_cube(time: Res<Time>, mut q: Query<&mut Transform, With<Spin>>) {
    for mut t in &mut q {
        t.rotate_y(time.delta_secs() * 0.8);
    }
}

/// Unattended smoke run: exit success once frames were decoded AND bound.
fn smoke_exit(smoke: Res<SmokeExit>, playback: Res<Playback>, mut exit: MessageWriter<AppExit>) {
    let Some(secs) = smoke.exit_after_secs else {
        return;
    };
    if smoke.started.elapsed().as_secs_f64() < secs {
        return;
    }
    if playback.bound > 0 {
        info!(
            "smoke exit: {} decoded frames, {} bound to the material",
            playback.frames.len(),
            playback.bound
        );
        exit.write(AppExit::Success);
    } else {
        error!("smoke exit: no decoded frame was ever bound");
        exit.write(AppExit::error());
    }
}

/// Registers pulled textures with the render world: wraps each `wgpu::Texture`
/// as a [`GpuImage`] under its reserved asset id, before material bind groups
/// prepare, so a material can reference the handle the same frame.
struct TextureBridgePlugin {
    pending: PendingTextures,
}

impl Plugin for TextureBridgePlugin {
    fn build(&self, app: &mut App) {
        app.sub_app_mut(RenderApp)
            .insert_resource(self.pending.clone())
            .add_systems(
                Render,
                register_textures.in_set(RenderSystems::PrepareAssets),
            );
    }
}

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
            label: Some("g2g-frame-srgb"),
            format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
            // The texture also carries STORAGE_BINDING (Vello writes it); an
            // sRGB view cannot, so narrow this view to sampling only.
            usage: Some(wgpu::TextureUsages::TEXTURE_BINDING),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("g2g-frame"),
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
