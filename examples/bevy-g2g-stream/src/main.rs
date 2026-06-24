//! Bevy + g2g server-side render-and-stream demo (M267).
//!
//! A Bevy app renders a 3D scene **headless** (no window) to an offscreen
//! texture on its own GPU, and g2g — joining Bevy's wgpu device via
//! [`GpuContext::from_wgpu`] — reads that rendered texture back off the *shared*
//! device each frame. This is the server-side / cloud-gaming shape: the engine
//! renders on a server GPU, g2g captures and (phase B) encodes + streams the
//! frames to a thin client.
//!
//! Phase A (this file): the render + zero-copy device handoff + read-back, the
//! load-bearing core. Bevy 0.19 pins the same wgpu 29 as g2g, so a `wgpu::Texture`
//! Bevy creates is bindable on the device g2g wraps. Phase B feeds the read-back
//! frames into `AppSrc -> VideoConvert -> FfmpegH264Enc -> sink`.
//!
//! Headless setup follows Bevy's official `headless_renderer` example: render to
//! a `RenderTarget::Image`, drive the loop with `ScheduleRunnerPlugin` and no
//! `WinitPlugin`. The read-back is a single render-world system that runs after
//! the main render and copies the target texture to a mapped buffer via the g2g
//! `GpuContext` (= Bevy's device), the same `copy_texture_to_buffer` + map the
//! g2g `gpu` module uses.

use std::time::Duration;

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::RenderTarget,
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{
            BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
            TexelCopyBufferInfo, TexelCopyBufferLayout, TextureFormat, TextureUsages,
        },
        renderer::{RenderAdapter, RenderDevice, RenderInstance, RenderQueue},
        texture::GpuImage,
        Render, RenderApp, RenderSystems,
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use crossbeam_channel::{Receiver, Sender};
use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_linear_chain, SourceLoop};
use g2g_core::{PipelineClock, PropValue, RawVideoFormat};
use g2g_plugins::appsrc::{register_appsrc, AppSrc, AppSrcFeed};
use g2g_plugins::ffmpegenc::FfmpegH264Enc;
use g2g_plugins::filesink::FileSink;
use g2g_plugins::gpu::GpuContext;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::webrtcsink::WebRtcSink;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FPS: u32 = 60;
/// Render this many frames, then exit (a demo run, not an endless server).
const FRAMES: u32 = 240;
/// AppSrc feed channel name shared with the encode thread.
const APPSRC_CHANNEL: &str = "bevy";
/// Where the encoded H.264 Annex-B stream is written.
const OUT_PATH: &str = "bevy_g2g.h264";

fn main() {
    let (tx, rx) = crossbeam_channel::unbounded::<Vec<u8>>();

    // The g2g encode pipeline runs on its own thread, fed the read-back RGBA
    // frames through this push handle (claimed by the AppSrc inside the chain by
    // matching channel name). Register before spawning so the source finds it.
    let feed = register_appsrc(APPSRC_CHANNEL);
    let encode = std::thread::spawn(encode_pipeline);

    let max_frames = std::env::var("G2G_FRAMES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(FRAMES);

    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.1)))
        .insert_resource(FrameReceiver(rx))
        .insert_resource(FrameCount(0))
        .insert_resource(MaxFrames(max_frames))
        .insert_resource(EncodeFeed(feed))
        .init_resource::<FirstNonBlank>()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                // No display: ScheduleRunnerPlugin (below) drives the loop, so a
                // window is never created and winit would only panic here.
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(1.0 / FPS as f64)))
        .add_plugins(ReadbackPlugin { sender: tx })
        .add_systems(Startup, setup)
        .add_systems(Update, (spin_cube, drain_frames));
    app.run();

    // The exit system signalled EOS to the feed; wait for the encoder to flush
    // and close the file.
    match encode.join() {
        Ok(Ok(stats)) => info!("encode pipeline finished: {} frames", stats),
        Ok(Err(e)) => error!("encode pipeline failed: {e:?}"),
        Err(_) => error!("encode thread panicked"),
    }
}

/// Drives `AppSrc -> VideoConvert(RGBA->I420) -> FfmpegH264Enc(NVENC H.264) ->
/// sink` to completion on its own thread. The sink is `WebRtcSink` (WHIP egress)
/// when `G2G_WHIP_URL` is set, else `FileSink` (the self-contained default).
/// `AppSrc` blocks on the feed until the main loop pushes frames and finishes on
/// EOS. Returns the number of source frames pushed through.
///
/// Runs inside a tokio runtime: `WebRtcSink`'s WHIP handshake (reqwest) and
/// session (tokio::spawn) need a reactor. `FileSink` is happy under it too.
fn encode_pipeline() -> Result<u64, g2g_core::G2gError> {
    let mut src = AppSrc::new();
    src.set_property("channel", PropValue::Str(APPSRC_CHANNEL.into()))
        .expect("appsrc channel");
    src.set_property(
        "caps",
        PropValue::Str(
            format!("video/x-raw,format=RGBA,width={WIDTH},height={HEIGHT},framerate={FPS}/1")
                .into(),
        ),
    )
    .expect("appsrc caps");

    let mut convert = VideoConvert::new(RawVideoFormat::I420);
    let mut encoder = FfmpegH264Enc::new(); // NVENC by default
    let clock = ZeroClock;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    let stats = match std::env::var("G2G_WHIP_URL") {
        Ok(url) => {
            info!("streaming H.264 to WHIP endpoint: {url}");
            let mut sink = WebRtcSink::new(url);
            let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut convert, &mut encoder];
            rt.block_on(run_linear_chain(&mut src, transforms, &mut sink, &clock, 4))?
        }
        Err(_) => {
            info!("G2G_WHIP_URL unset; writing H.264 to {OUT_PATH} (set it to stream over WHIP)");
            let mut sink = FileSink::new(OUT_PATH);
            let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut convert, &mut encoder];
            rt.block_on(run_linear_chain(&mut src, transforms, &mut sink, &clock, 4))?
        }
    };
    Ok(stats.frames_consumed)
}

/// Trivial clock: the sink (`FileSink`) does not pace to a clock, so the runner
/// needs only a `now_ns`, never advanced.
struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Marks the cube so `spin_cube` can rotate it (visible motion between frames).
#[derive(Component)]
struct Spin;

/// The offscreen texture the camera renders into; extracted to the render world
/// so the read-back system can find its GPU texture.
#[derive(Resource, Clone, ExtractResource)]
struct RenderTargetImage(Handle<Image>);

/// Main-world end of the render-world -> main-world frame channel.
#[derive(Resource)]
struct FrameReceiver(Receiver<Vec<u8>>);

/// Render-world end of that channel.
#[derive(Resource, Clone)]
struct FrameSender(Sender<Vec<u8>>);

/// Push handle into the g2g encode pipeline (the AppSrc feed).
#[derive(Resource)]
struct EncodeFeed(AppSrcFeed);

#[derive(Resource)]
struct FrameCount(u32);

/// Frames to render before exiting; `0` means run forever (for a live stream you
/// watch in a browser). From `G2G_FRAMES`, default `FRAMES`.
#[derive(Resource)]
struct MaxFrames(u32);

/// Sequence of the first non-blank frame seen, or `None` until one arrives.
/// Headless render has a few frames of pre-roll warmup where the target is still
/// transparent, so "the scene rendered" is asserted over the run, not frame 0.
#[derive(Resource, Default)]
struct FirstNonBlank(Option<u32>);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    // The texture the camera renders into. COPY_SRC so g2g can copy it out.
    let mut target = Image::new_target_texture(WIDTH, HEIGHT, TextureFormat::Rgba8UnormSrgb, None);
    target.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target_handle = images.add(target);
    commands.insert_resource(RenderTargetImage(target_handle.clone()));

    // A spinning cube on a ground plane, lit, so successive frames differ.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Spin,
    ));
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    commands.spawn((
        PointLight { shadow_maps_enabled: true, ..default() },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
    commands.spawn((
        Camera3d::default(),
        // The render target is its own component in Bevy 0.19 (not a `Camera`
        // field): point the camera at the offscreen image.
        RenderTarget::Image(target_handle.into()),
        Tonemapping::None,
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn spin_cube(time: Res<Time>, mut q: Query<&mut Transform, With<Spin>>) {
    for mut t in &mut q {
        t.rotate_y(time.delta_secs() * 1.2);
    }
}

/// Drain read-back frames in the main world: count them, sanity-check the first
/// is a real rendered image (not blank), and exit after `FRAMES`.
fn drain_frames(
    receiver: Res<FrameReceiver>,
    feed: Res<EncodeFeed>,
    max: Res<MaxFrames>,
    mut count: ResMut<FrameCount>,
    mut first_non_blank: ResMut<FirstNonBlank>,
    mut exit: MessageWriter<AppExit>,
) {
    while let Ok(rgba) = receiver.0.try_recv() {
        if count.0 == 0 {
            info!(
                "g2g read back Bevy's first frame off the shared device: {} bytes, {WIDTH}x{HEIGHT} RGBA",
                rgba.len()
            );
        }
        // A lit cube + plane means pixels vary well above the dark clear colour.
        // The first frames are blank (headless render pre-roll); record when the
        // scene actually shows up.
        if first_non_blank.0.is_none() {
            let lit = rgba.chunks_exact(4).any(|p| p[0] > 60 || p[1] > 60 || p[2] > 60);
            if lit {
                first_non_blank.0 = Some(count.0);
                info!("scene rendered: first non-blank frame at index {}", count.0);
            }
        }
        // Hand the frame to the g2g encode pipeline (RGBA -> I420 -> NVENC H.264
        // -> file). `push` returns false if the encoder is backed up; at this
        // resolution NVENC keeps up, so a drop would be notable.
        let pts_ns = count.0 as u64 * 1_000_000_000 / FPS as u64;
        if !feed.0.push(&rgba, pts_ns) {
            warn!("encode feed full; dropped frame {}", count.0);
        }
        count.0 += 1;
        // `max == 0` streams forever (watch it live); otherwise stop at the cap.
        if max.0 != 0 && count.0 >= max.0 {
            assert!(
                first_non_blank.0.is_some(),
                "no non-blank frame in {} frames: the scene never rendered onto the target",
                max.0
            );
            // Signal EOS so the encoder flushes and the sink finalises, then
            // exit; `main` joins the encode thread after the loop returns.
            feed.0.end_of_stream();
            info!(
                "captured {} frames off the shared GPU device (first non-blank at {:?}); EOS sent, exiting",
                count.0, first_non_blank.0
            );
            exit.write(AppExit::Success);
        }
    }
}

/// Wires the render-world read-back: extracts the target image handle, stashes
/// the channel sender, and runs `readback_via_g2g` after the main render.
struct ReadbackPlugin {
    sender: Sender<Vec<u8>>,
}

impl Plugin for ReadbackPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<RenderTargetImage>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(FrameSender(self.sender.clone()));
        render_app.add_systems(Render, readback_via_g2g.after(RenderSystems::Render));
    }
}

/// Render-world system: after Bevy renders the scene into the target texture,
/// g2g (wrapping Bevy's device via `GpuContext::from_wgpu`) copies that texture
/// to a mapped buffer and sends the RGBA bytes to the main world. The `GpuContext`
/// is built once and cached in a `Local`. Demonstrates the bring-your-own-device
/// handoff: the texture was created by Bevy, read by g2g, on the *one* device.
#[allow(clippy::too_many_arguments)]
fn readback_via_g2g(
    mut ctx: Local<Option<GpuContext>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    adapter: Res<RenderAdapter>,
    instance: Res<RenderInstance>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    target: Res<RenderTargetImage>,
    sender: Res<FrameSender>,
) {
    // The target's GPU texture is only present once the image asset is prepared.
    let Some(gpu_image) = gpu_images.get(&target.0) else {
        return;
    };

    // Build the g2g context from Bevy's wgpu handles on first run (cheap clones;
    // wgpu handles are reference-counted, so this shares Bevy's GPU, not a copy).
    let ctx = ctx.get_or_insert_with(|| {
        GpuContext::from_wgpu(
            (**instance.0).clone(),
            (**adapter.0).clone(),
            device.wgpu_device().clone(),
            (**queue.0).clone(),
        )
    });

    if let Some(rgba) = read_texture_rgba(ctx, &gpu_image.texture) {
        let _ = sender.0.send(rgba);
    }
}

/// Copy an `Rgba8` `WIDTH x HEIGHT` texture to a buffer on `ctx`'s device, map it,
/// and return tightly-packed RGBA bytes (the wgpu 256-byte row alignment is undone
/// here). This is the g2g side reading a Bevy-created texture on the shared device,
/// the same `copy_texture_to_buffer` + map the g2g `gpu` module uses.
fn read_texture_rgba(ctx: &GpuContext, texture: &wgpu::Texture) -> Option<Vec<u8>> {
    let unpadded = (WIDTH * 4) as usize;
    // wgpu requires each buffer row aligned to 256 bytes for texture->buffer copy.
    let padded = unpadded.div_ceil(256) * 256;
    let buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("g2g-readback"),
        size: (padded * HEIGHT as usize) as u64,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder =
        ctx.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("g2g-readback") });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(HEIGHT),
            },
        },
        Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    // wgpu 29: poll takes a PollType (the same call g2g's `gpu` module uses).
    let _ = ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    rx.recv().ok()?.ok()?;

    let mapped = slice.get_mapped_range();
    let mut rgba = Vec::with_capacity(unpadded * HEIGHT as usize);
    for row in 0..HEIGHT as usize {
        let start = row * padded;
        rgba.extend_from_slice(&mapped[start..start + unpadded]);
    }
    drop(mapped);
    buffer.unmap();
    Some(rgba)
}
