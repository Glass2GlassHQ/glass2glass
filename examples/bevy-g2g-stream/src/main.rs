//! Bevy + g2g server-side render-and-stream demo, zero-copy path (M267 -> M278).
//!
//! A Bevy app renders a 3D scene **headless** (no window) on an interop GPU
//! device, and g2g encodes each rendered frame to H.264 with **no device->host
//! read-back**: the rendered texture is copied device->device into a CUDA surface
//! ([`WgpuToCuda`], M275) and handed straight to the native NVENC encoder
//! ([`NvEnc`]). Only the compact H.264 access units leave the GPU. This is the
//! server-side / cloud-gaming shape: render on a server GPU, encode + stream to a
//! thin client, never paying a full-frame PCIe download.
//!
//! The load-bearing trick is making Bevy render on g2g's *interop* device: g2g
//! creates a Vulkan device with `VK_KHR_external_memory_fd`
//! ([`create_interop_device_full`]) and hands it to Bevy via
//! `RenderCreation::Manual`, so a `wgpu::Texture` Bevy renders is on the exact
//! device the `WgpuToCuda` bridge can export to CUDA. Bevy 0.19 pins the same
//! wgpu 29 as g2g, so the handle types match. The earlier M267 version read the
//! texture back to system memory and encoded with the ffmpeg NVENC backend; this
//! version removes the read-back entirely.
//!
//! Layout:
//! - **Render world** (`encode_via_g2g`): after Bevy renders into the target
//!   texture, copy it through `WgpuToCuda` -> CUDA and encode with `NvEnc`,
//!   emitting H.264 access units to the main world over a channel.
//! - **Main world** (`drain_frames`): push the access units into the g2g sink
//!   pipeline's `AppSrc` feed and stop after the frame cap.
//! - **Sink thread** (`sink_pipeline`): `AppSrc(H.264) -> FileSink` (default) or
//!   `-> WebRtcSink` (WHIP egress when `G2G_WHIP_URL` is set).
//!
//! Headless setup follows Bevy's official `headless_renderer` example: render to
//! a `RenderTarget::Image`, drive the loop with `ScheduleRunnerPlugin`, no
//! `WinitPlugin`.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use bevy::{
    app::{AppExit, ScheduleRunnerPlugin},
    camera::RenderTarget,
    core_pipeline::tonemapping::Tonemapping,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{TextureFormat, TextureUsages},
        renderer::{
            RenderAdapter, RenderAdapterInfo, RenderDevice, RenderInstance, RenderQueue,
            WgpuWrapper,
        },
        settings::{RenderCreation, RenderResources},
        texture::GpuImage,
        Render, RenderApp, RenderPlugin, RenderSystems,
    },
    window::ExitCondition,
    winit::WinitPlugin,
};
use crossbeam_channel::{Receiver, Sender};
use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_linear_chain, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket,
    PropValue, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::appsrc::{register_appsrc, AppSrc, AppSrcFeed};
use g2g_plugins::cudawgpu::{create_interop_device_full, WgpuToCuda};
use g2g_plugins::filesink::FileSink;
use g2g_plugins::nvenc::NvEnc;
use g2g_plugins::webrtcsink::WebRtcSink;

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FPS: u32 = 60;
const KEYFRAME_INTERVAL: u64 = FPS as u64;
/// Render this many frames, then exit (a demo run, not an endless server).
const FRAMES: u32 = 240;
/// AppSrc feed channel name shared with the sink thread.
const APPSRC_CHANNEL: &str = "bevy";
/// Where the encoded H.264 Annex-B stream is written (the no-WHIP default).
const OUT_PATH: &str = "bevy_g2g.h264";

/// One encoded access unit handed render-world -> main-world: the H.264 Annex-B
/// bytes and their presentation timestamp (ns).
type EncodedAu = (Vec<u8>, u64);

enum RenderMessage {
    AccessUnit(EncodedAu),
    Fatal(String),
}

fn main() {
    let (tx, rx) = crossbeam_channel::unbounded::<RenderMessage>();
    let failed = Arc::new(AtomicBool::new(false));

    // The g2g sink pipeline runs on its own thread, fed the encoded H.264 access
    // units through this push handle (claimed by the AppSrc inside the chain by
    // matching channel name). Register before spawning so the source finds it.
    let feed = register_appsrc(APPSRC_CHANNEL);
    let sink = std::thread::spawn(sink_pipeline);

    let max_frames = std::env::var("G2G_FRAMES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(FRAMES);

    // g2g's interop device (Vulkan + VK_KHR_external_memory_fd, opened with the
    // adapter's full features so Bevy's renderer is happy on it). Bevy adopts it
    // via RenderCreation::Manual, so every texture it renders is exportable to
    // CUDA on this exact device, the prerequisite for the zero-copy bridge.
    let interop = pollster::block_on(create_interop_device_full())
        .expect("create interop wgpu device (need a Vulkan + NVIDIA GPU)");
    let render_resources = RenderResources(
        RenderDevice::from(interop.device.clone()),
        RenderQueue(Arc::new(WgpuWrapper::new(interop.queue.clone()))),
        RenderAdapterInfo(WgpuWrapper::new(interop.adapter.get_info())),
        RenderAdapter(Arc::new(WgpuWrapper::new(interop.adapter.clone()))),
        RenderInstance(Arc::new(WgpuWrapper::new(interop.instance.clone()))),
    );
    // Bevy holds its own (reference-counted) clones now; drop our handle.
    drop(interop);

    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.1)))
        .insert_resource(FrameReceiver(rx))
        .insert_resource(FrameCount(0))
        .insert_resource(MaxFrames(max_frames))
        .insert_resource(RunFailed(failed.clone()))
        .insert_resource(EncodeFeed(feed))
        .add_plugins(
            DefaultPlugins
                .set(RenderPlugin {
                    // Render on g2g's interop device instead of letting Bevy open
                    // its own: the load-bearing handoff for zero-copy encode.
                    render_creation: RenderCreation::Manual(render_resources),
                    // Compile pipelines synchronously on the render thread. Bevy's
                    // default async compilation runs Vulkan pipeline creation on a
                    // background task that, on the NVIDIA driver, faults when it
                    // overlaps our CUDA encode work on the same device (Vulkan +
                    // CUDA concurrency on the shared driver). Synchronous compile
                    // serialises it with our after-render encode system.
                    synchronous_pipeline_compilation: true,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                // No display: ScheduleRunnerPlugin (below) drives the loop, so a
                // window is never created and winit would only panic here.
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / FPS as f64,
        )))
        .add_plugins(EncodePlugin { sender: tx })
        .add_systems(Startup, setup)
        .add_systems(Update, (spin_cube, drain_frames));
    app.run();

    // The exit system signalled EOS to the feed; wait for the sink thread to flush
    // and close the file / WHIP session. The H.264 output is complete here.
    let ok = match sink.join() {
        Ok(Ok(frames)) => {
            info!("sink pipeline finished: {frames} access units");
            !failed.load(Ordering::Relaxed)
        }
        Ok(Err(e)) => {
            error!("sink pipeline failed: {e:?}");
            false
        }
        Err(_) => {
            error!("sink thread panicked");
            false
        }
    };

    // Exit now rather than dropping `app`. The render world holds CUDA + wgpu
    // resources (the `WgpuToCuda` bridge, the NVENC session) on Bevy's device;
    // dropping them races Bevy's own render-thread / device teardown, which can
    // segfault in the driver on shutdown (a known GPU-app teardown-order hazard).
    // The work is done and flushed, so skip the destructors and let the OS reclaim
    // the GPU resources, the standard demo-shutdown approach.
    std::process::exit(if ok { 0 } else { 1 });
}

/// Drives `AppSrc(H.264) -> sink` to completion on its own thread. The sink is
/// `WebRtcSink` (WHIP egress) when `G2G_WHIP_URL` is set, else `FileSink` (the
/// self-contained default). `AppSrc` carries the already-encoded H.264 access
/// units from the render world (no `VideoConvert` / encoder in the chain anymore,
/// the GPU did that), blocks on the feed until the main loop pushes, and finishes
/// on EOS. Returns the number of access units pushed through.
///
/// Runs inside a tokio runtime: `WebRtcSink`'s WHIP handshake (reqwest) and
/// session (tokio::spawn) need a reactor. `FileSink` is happy under it too.
fn sink_pipeline() -> Result<u64, G2gError> {
    let mut src = AppSrc::new();
    src.set_property("channel", PropValue::Str(APPSRC_CHANNEL.into()))
        .expect("appsrc channel");
    src.set_property(
        "caps",
        PropValue::Str(
            format!("video/x-h264,width={WIDTH},height={HEIGHT},framerate={FPS}/1").into(),
        ),
    )
    .expect("appsrc caps");

    let clock = ZeroClock;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    // No transforms: the render world already produced H.264, so the sink chain is
    // just source -> sink.
    let stats = match std::env::var("G2G_WHIP_URL") {
        Ok(url) => {
            info!("streaming H.264 to WHIP endpoint: {url}");
            let mut sink = WebRtcSink::new(url);
            let transforms: Vec<&mut dyn DynAsyncElement> = vec![];
            rt.block_on(run_linear_chain(&mut src, transforms, &mut sink, &clock, 4))?
        }
        Err(_) => {
            info!("G2G_WHIP_URL unset; writing H.264 to {OUT_PATH} (set it to stream over WHIP)");
            let mut sink = FileSink::new(OUT_PATH);
            let transforms: Vec<&mut dyn DynAsyncElement> = vec![];
            rt.block_on(run_linear_chain(&mut src, transforms, &mut sink, &clock, 4))?
        }
    };
    Ok(stats.frames_consumed)
}

/// Trivial clock: the sinks here do not pace to a clock, so the runner needs only
/// a `now_ns`, never advanced.
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
/// so the encode system can find its GPU texture.
#[derive(Resource, Clone, ExtractResource)]
struct RenderTargetImage(Handle<Image>);

/// Main-world end of the render-world -> main-world access-unit channel.
#[derive(Resource)]
struct FrameReceiver(Receiver<RenderMessage>);

/// Render-world end of that channel.
#[derive(Resource, Clone)]
struct FrameSender(Sender<RenderMessage>);

/// Push handle into the g2g sink pipeline (the AppSrc feed).
#[derive(Resource)]
struct EncodeFeed(AppSrcFeed);

#[derive(Resource)]
struct FrameCount(u32);

/// Frames to render before exiting; `0` means run forever (for a live stream you
/// watch in a browser). From `G2G_FRAMES`, default `FRAMES`.
#[derive(Resource)]
struct MaxFrames(u32);

#[derive(Resource, Clone)]
struct RunFailed(Arc<AtomicBool>);

/// The render-world zero-copy encoder, behind a `Mutex` so it satisfies the
/// `Send + Sync` resource bound (`NvEnc` is `Send` but not `Sync`; the system
/// takes exclusive access through the lock). Built lazily on the first render
/// frame, once the target's GPU texture exists.
#[derive(Resource)]
struct Encoder(Mutex<Option<EncodeState>>);

/// The render-world encode state: the NVENC encoder and the wgpu->CUDA bridge,
/// both living on Bevy's (= the interop) device, plus the running frame index for
/// presentation timestamps.
///
/// Field order matters for `Drop`: `nvenc` is declared first so it drops first.
/// NVENC's session lives in the CUDA primary context the `bridge` retains, so the
/// session must be destroyed before the bridge releases that context, else
/// teardown destroys a session on a freed context (an intermittent exit segfault).
struct EncodeState {
    nvenc: NvEnc,
    bridge: WgpuToCuda,
    frame_no: u64,
}

impl EncodeState {
    /// Build the bridge + encoder on `device` (Bevy's interop device).
    fn new(device: wgpu::Device, queue: wgpu::Queue) -> Result<Self, G2gError> {
        // SAFETY: `device` is the VK_KHR_external_memory_fd interop device created
        // by `create_interop_device_full` and handed to Bevy, so the bridge's
        // exportable-image allocation and CUDA import are valid on it.
        let bridge = unsafe { WgpuToCuda::new(device, queue, WIDTH, HEIGHT) }?;
        let mut nvenc = NvEnc::new();
        // Disambiguate: both AsyncElement and DynAsyncElement (imported for the
        // sink chain) are in scope.
        AsyncElement::configure_pipeline(&mut nvenc, &rgba_caps())?;
        Ok(Self {
            bridge,
            nvenc,
            frame_no: 0,
        })
    }

    /// Copy `texture` (Bevy's just-rendered target) into the bridge's CUDA surface
    /// and encode it, returning any ready H.264 access units. No device->host copy.
    fn encode(&mut self, texture: &wgpu::Texture) -> Result<Vec<EncodedAu>, G2gError> {
        let pts_ns = self.frame_no * 1_000_000_000 / FPS as u64;
        if self.frame_no % KEYFRAME_INTERVAL == 0 {
            self.nvenc.force_keyframe();
        }
        self.frame_no += 1;
        self.bridge.ingest_texture(texture)?;
        let frame = self.bridge.to_cuda_frame(pts_ns)?;
        let mut cap = CaptureAus::default();
        // NVENC sync-mode encode; the capture sink resolves immediately, so the
        // block_on returns this frame's access unit without a reactor.
        let fut =
            AsyncElement::process(&mut self.nvenc, PipelinePacket::DataFrame(frame), &mut cap);
        pollster::block_on(fut)?;
        Ok(cap.aus)
    }
}

/// RGBA at the render geometry: the caps `NvEnc` is configured for (it color
/// converts ABGR -> H.264 internally).
fn rgba_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FPS << 16),
    }
}

/// Render-world sink that captures `NvEnc`'s emitted H.264 access units (System
/// memory) and their timestamps.
#[derive(Default)]
struct CaptureAus {
    aus: Vec<EncodedAu>,
}
impl OutputSink for CaptureAus {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<PushOutcome, G2gError>> + 'a>>
    {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(s) = f.domain.as_system_slice() {
                    self.aus.push((s.as_slice().to_vec(), f.timing.pts_ns));
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
) {
    // The texture the camera renders into. COPY_SRC so the bridge can copy it into
    // its exportable CUDA surface. Rgba8UnormSrgb is copy-compatible with the
    // bridge's Rgba8Unorm export image (same format ignoring the srgb suffix).
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
        PointLight {
            shadow_maps_enabled: true,
            ..default()
        },
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

/// Drain encoded access units in the main world: push them into the g2g sink
/// pipeline and exit after `FRAMES` have been encoded.
fn drain_frames(
    receiver: Res<FrameReceiver>,
    feed: Res<EncodeFeed>,
    max: Res<MaxFrames>,
    failed: Res<RunFailed>,
    mut count: ResMut<FrameCount>,
    mut exit: MessageWriter<AppExit>,
) {
    while let Ok(message) = receiver.0.try_recv() {
        let RenderMessage::AccessUnit((au, pts_ns)) = message else {
            if let RenderMessage::Fatal(reason) = message {
                failed.0.store(true, Ordering::Relaxed);
                error!("{reason}");
                feed.0.end_of_stream_blocking();
                exit.write(AppExit::error());
            }
            return;
        };
        if count.0 == 0 {
            info!(
                "g2g encoded Bevy's first frame on the GPU with no read-back: {} bytes H.264",
                au.len()
            );
        }
        // Hand the access unit to the g2g sink pipeline (H.264 -> file / WHIP).
        // Backpressure here slows the render loop instead of dropping frames
        // before WebRtcSink can publish them.
        if !feed.0.push_blocking(&au, pts_ns) {
            failed.0.store(true, Ordering::Relaxed);
            error!("sink feed closed before access unit {}", count.0);
            exit.write(AppExit::error());
            return;
        }
        count.0 += 1;
        // `max == 0` streams forever (watch it live); otherwise stop at the cap.
        if max.0 != 0 && count.0 >= max.0 {
            // Signal EOS so the sink flushes and finalises, then exit; `main` joins
            // the sink thread after the loop returns.
            feed.0.end_of_stream_blocking();
            info!(
                "encoded {} frames on the GPU (no read-back); EOS sent, exiting",
                count.0
            );
            exit.write(AppExit::Success);
        }
    }
}

/// Wires the render-world zero-copy encode: extracts the target image handle,
/// stashes the channel sender + the lazily-built [`Encoder`], and runs
/// `encode_via_g2g` after the main render.
struct EncodePlugin {
    sender: Sender<RenderMessage>,
}

impl Plugin for EncodePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<RenderTargetImage>::default());
        let render_app = app.sub_app_mut(RenderApp);
        render_app.insert_resource(FrameSender(self.sender.clone()));
        render_app.insert_resource(Encoder(Mutex::new(None)));
        render_app.add_systems(Render, encode_via_g2g.after(RenderSystems::Render));
    }
}

/// Render-world system: after Bevy renders the scene into the target texture,
/// copy that texture device->device into a CUDA surface (`WgpuToCuda` on Bevy's
/// interop device) and encode it with `NvEnc`, sending the H.264 access units to
/// the main world. No GPU->CPU read-back: the pixels go straight from the render
/// target to the encoder on the one device.
fn encode_via_g2g(
    encoder: Res<Encoder>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    target: Res<RenderTargetImage>,
    sender: Res<FrameSender>,
) {
    // The target's GPU texture is only present once the image asset is prepared.
    let Some(gpu_image) = gpu_images.get(&target.0) else {
        return;
    };

    let mut guard = encoder.0.lock().expect("encoder lock");
    let state = match &mut *guard {
        Some(s) => s,
        none => {
            // Build the bridge + encoder on Bevy's (interop) device on first run.
            match EncodeState::new(device.wgpu_device().clone(), (**queue.0).clone()) {
                Ok(s) => none.insert(s),
                Err(e) => {
                    let _ = sender.0.send(RenderMessage::Fatal(format!(
                        "failed to initialize WgpuToCuda/NvEnc: {e:?}"
                    )));
                    return;
                }
            }
        }
    };

    match state.encode(&gpu_image.texture) {
        Ok(aus) => {
            for au in aus {
                let _ = sender.0.send(RenderMessage::AccessUnit(au));
            }
        }
        Err(e) => {
            let _ = sender.0.send(RenderMessage::Fatal(format!(
                "failed to encode Bevy render target: {e:?}"
            )));
        }
    }
}
