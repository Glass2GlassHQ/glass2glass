//! Headless render-and-stream: Bevy renders, g2g encodes and ships H.264.
//!
//! Two encode paths behind one plugin group:
//! - **Zero-copy** (`nvenc` feature, NVIDIA): Bevy renders on g2g's interop
//!   device (`RenderCreation::Manual`), the target texture is copied
//!   device->device into a CUDA surface (`WgpuToCuda`) and encoded by the
//!   native `NvEnc`. Only H.264 access units reach the CPU.
//! - **Readback** (fallback, any adapter): the target texture is read back to
//!   system memory after render and the sink pipeline encodes it with libx264
//!   (`videoconvert -> ffmpegenc`).
//!
//! The path is chosen at plugin-group build: with `nvenc` compiled in, the
//! interop device is attempted and a failure (no NVIDIA / no CUDA) falls back
//! to readback at runtime.
//!
//! Layout mirrors the validated bevy-g2g-stream demo (M267 -> M278): a
//! render-world system produces frames after `RenderSystems::Render`, a
//! main-world system pushes them into a g2g `AppSrc` feed, and the sink
//! pipeline (`appsrc -> [convert -> encode] -> filesink | webrtcsink`) runs on
//! its own thread.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};
use std::thread::JoinHandle;
use std::time::Duration;

use bevy::{
    app::{AppExit, PluginGroupBuilder, ScheduleRunnerPlugin},
    camera::RenderTarget,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_asset::RenderAssets,
        render_resource::{TextureFormat, TextureUsages},
        renderer::{RenderDevice, RenderQueue},
        settings::RenderCreation,
        texture::GpuImage,
        Render, RenderApp, RenderPlugin, RenderSystems,
    },
    window::{ExitCondition, WindowRef},
    winit::WinitPlugin,
};
use crossbeam_channel::{Receiver, Sender};
use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_linear_chain, SourceLoop};
use g2g_core::{G2gError, HardwareError, PipelineClock, PropValue, RawVideoFormat};
use g2g_plugins::appsrc::{register_appsrc, AppSrc, AppSrcFeed};
use g2g_plugins::ffmpegenc::{Backend, FfmpegH264Enc};
use g2g_plugins::filesink::FileSink;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::webrtcsink::WebRtcSink;

#[cfg(feature = "nvenc")]
mod zerocopy;

/// AppSrc feed channel name shared with the sink thread.
const CHANNEL: &str = "bevy-g2g-stream";

/// What to render and where to send it. Every knob of the streaming plugin.
#[derive(Clone, Debug)]
pub struct StreamSettings {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    /// Target bitrate, bits/second (both encoders).
    pub bitrate: u32,
    /// Frames between forced IDR keyframes (zero-copy path; the software
    /// encoder keyframes on its own GOP and on downstream requests).
    pub keyframe_interval: u32,
    pub output: StreamOutput,
    /// Frames to render before exiting; `0` = run until the app exits.
    pub max_frames: u32,
    /// Serve the viewer-input WebSocket backchannel on this port (see
    /// `RemoteInputPlugin`). `None` = no input backchannel.
    pub input_port: Option<u16>,
}

#[derive(Clone, Debug)]
pub enum StreamOutput {
    /// Publish to a WHIP endpoint (e.g. MediaMTX) over WebRTC.
    Whip(String),
    /// Write the H.264 Annex-B stream to a file.
    File(String),
}

impl Default for StreamSettings {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            fps: 60,
            bitrate: 4_000_000,
            keyframe_interval: 60,
            output: StreamOutput::File("bevy_g2g.h264".into()),
            max_frames: 0,
            input_port: None,
        }
    }
}

impl StreamSettings {
    /// The demo-run environment convention: `G2G_WHIP_URL` selects WHIP egress
    /// (else a file), `G2G_FRAMES` caps the run (default 900, a 15 s clip; `0` = forever),
    /// `G2G_INPUT_PORT` enables the viewer-input backchannel.
    pub fn from_env() -> Self {
        let mut s = Self::default();
        if let Ok(url) = std::env::var("G2G_WHIP_URL") {
            s.output = StreamOutput::Whip(url);
        }
        s.max_frames = std::env::var("G2G_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(900);
        s.input_port = std::env::var("G2G_INPUT_PORT")
            .ok()
            .and_then(|v| v.parse().ok());
        s
    }
}

/// Everything a headless streaming app needs in one `add_plugins` call:
/// `DefaultPlugins` reconfigured for windowless rendering (no winit, schedule
/// runner paced at the stream fps, the render device swapped for g2g's interop
/// device when the zero-copy path is available) plus the streaming plugin
/// itself. The app only spawns its scene and a camera; the camera is
/// retargeted onto the stream automatically.
#[derive(Debug)]
pub struct RemoteRenderPlugins {
    settings: StreamSettings,
    windowed: bool,
}

impl RemoteRenderPlugins {
    pub fn new(settings: StreamSettings) -> Self {
        Self {
            settings,
            windowed: false,
        }
    }

    pub fn from_env() -> Self {
        Self::new(StreamSettings::from_env())
    }

    /// Windowed variant: the app keeps its normal window (winit event loop,
    /// vsync pacing) and streams at the same time. The scene camera still
    /// renders into the stream texture; the window shows that texture through
    /// a fullscreen mirror, so the desktop view and the stream are the same
    /// pixels.
    pub fn windowed(settings: StreamSettings) -> Self {
        Self {
            settings,
            windowed: true,
        }
    }
}

impl PluginGroup for RemoteRenderPlugins {
    fn build(self) -> PluginGroupBuilder {
        let settings = self.settings;
        let (render_creation, zero_copy) = pick_render_creation();
        let mut group = DefaultPlugins.build().set(RenderPlugin {
            render_creation,
            // Compile pipelines synchronously on the render thread. Bevy's
            // default async compilation runs Vulkan pipeline creation on a
            // background task that, on the NVIDIA driver, faults when it
            // overlaps the CUDA encode work on the same device. Harmless
            // (a little startup latency) on the readback path.
            synchronous_pipeline_compilation: true,
            ..default()
        });
        group = if self.windowed {
            // Normal window, sized to the stream so the mirror is 1:1. The
            // winit loop paces the app (vsync), so the stream rate follows
            // the display rate rather than `fps`.
            group.set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: (settings.width, settings.height).into(),
                    title: "bevy-g2g stream".into(),
                    ..default()
                }),
                ..default()
            })
        } else {
            group
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                // No display: the ScheduleRunnerPlugin drives the loop, so a
                // window is never created and winit would only panic here.
                .disable::<WinitPlugin>()
                .add(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
                    1.0 / settings.fps as f64,
                )))
        };
        if let Some(port) = settings.input_port {
            group = group.add(crate::input::RemoteInputPlugin { port });
        }
        group.add(StreamPlugin {
            settings,
            zero_copy,
            windowed: self.windowed,
        })
    }
}

/// With `nvenc`, try g2g's interop device (Vulkan + VK_KHR_external_memory_fd,
/// opened with the adapter's full features so Bevy's renderer is happy on it):
/// Bevy adopting it is what makes every rendered texture exportable to CUDA.
/// On failure (or without the feature) Bevy opens its own device and frames
/// are read back instead.
fn pick_render_creation() -> (RenderCreation, bool) {
    #[cfg(feature = "nvenc")]
    match zerocopy::interop_render_creation() {
        Ok(rc) => return (rc, true),
        Err(e) => {
            warn!("no zero-copy encode path ({e:?}); falling back to GPU readback + libx264");
        }
    }
    (RenderCreation::default(), false)
}

/// Runs the whole app and finishes the stream: sends EOS if the in-app systems
/// have not already, joins the sink thread so the file / WHIP session is
/// flushed, then exits the process.
///
/// The process exit is deliberate: the render world holds GPU resources (on
/// the zero-copy path a CUDA context and an NVENC session on Bevy's device)
/// whose drop order races Bevy's own device teardown and can segfault in the
/// driver. The work is flushed, so skip the destructors and let the OS reclaim
/// the GPU, the standard GPU-demo shutdown approach.
pub fn run(mut app: App) -> ! {
    let exit = app.run();
    let sink = SINK.lock().expect("sink handle lock").take();
    let ok = match sink.map(JoinHandle::join) {
        Some(Ok(Ok(frames))) => {
            info!("stream finished: {frames} frames");
            !FAILED.load(Ordering::Relaxed)
        }
        Some(Ok(Err(e))) => {
            error!("sink pipeline failed: {e:?}");
            false
        }
        Some(Err(_)) => {
            error!("sink thread panicked");
            false
        }
        None => {
            error!("bevy_g2g::run called without RemoteRenderPlugins");
            false
        }
    };
    let code = if ok && exit == AppExit::Success { 0 } else { 1 };
    std::process::exit(code);
}

/// Sink-thread handle, taken by [`run`] after the app loop ends. A static
/// because the `App` is consumed by `run` and never dropped (see above).
static SINK: Mutex<Option<JoinHandle<Result<u64, G2gError>>>> = Mutex::new(None);
static FAILED: AtomicBool = AtomicBool::new(false);

/// One frame handed render-world -> main-world: encoded H.264 on the zero-copy
/// path, raw RGBA on the readback path, plus its presentation timestamp (ns).
pub(crate) enum RenderMessage {
    Frame(Vec<u8>, u64),
    Fatal(String),
}

/// The offscreen texture cameras render into. Inserted at startup; exposed so
/// an app can point extra cameras or UI at the stream explicitly.
#[derive(Resource, Clone, ExtractResource)]
pub struct StreamTarget(pub Handle<Image>);

#[derive(Resource)]
struct FrameReceiver(Receiver<RenderMessage>);

#[derive(Resource, Clone)]
struct FrameSender(pub(crate) Sender<RenderMessage>);

/// Push handle into the g2g sink pipeline (the AppSrc feed).
#[derive(Resource)]
struct EncodeFeed(AppSrcFeed);

#[derive(Resource)]
struct StreamState {
    frames: u32,
    eos_sent: bool,
}

/// Settings snapshot for the systems (main and render world).
#[derive(Resource, Clone, ExtractResource)]
struct Settings(StreamSettings);

/// Which encode path the render world should build.
#[derive(Resource, Clone, Copy)]
struct ZeroCopy(bool);

/// The render-world frame producer, behind a `Mutex` so it satisfies the
/// `Send + Sync` resource bound (`NvEnc` is `Send` but not `Sync`; the system
/// takes exclusive access through the lock). Built lazily on the first render
/// frame, once the target's GPU texture exists.
#[derive(Resource, Default)]
struct Producer(Mutex<Option<PathState>>);

// one instance per app, so the variant size gap is irrelevant
#[allow(clippy::large_enum_variant)]
enum PathState {
    #[cfg(feature = "nvenc")]
    ZeroCopy(zerocopy::EncodeState),
    Readback(ReadbackState),
}

struct StreamPlugin {
    settings: StreamSettings,
    zero_copy: bool,
    windowed: bool,
}

/// The window-mirror 2D camera (windowed mode): shows the stream texture in
/// the app's window. Excluded from camera retargeting by construction.
#[derive(Component)]
struct MirrorCamera;

impl Plugin for StreamPlugin {
    fn build(&self, app: &mut App) {
        let (tx, rx) = crossbeam_channel::unbounded::<RenderMessage>();
        // The sink pipeline runs on its own thread, fed frames through this
        // push handle (claimed by the AppSrc in the chain by matching channel
        // name). Register before spawning so the source finds it.
        let feed = register_appsrc(CHANNEL);
        let settings = self.settings.clone();
        let zero_copy = self.zero_copy;
        let handle = std::thread::spawn({
            let settings = settings.clone();
            move || sink_pipeline(settings, zero_copy)
        });
        *SINK.lock().expect("sink handle lock") = Some(handle);

        app.insert_resource(FrameReceiver(rx))
            .insert_resource(EncodeFeed(feed))
            .insert_resource(StreamState {
                frames: 0,
                eos_sent: false,
            })
            .insert_resource(Settings(settings))
            .add_plugins(ExtractResourcePlugin::<StreamTarget>::default())
            .add_systems(Startup, create_target)
            .add_systems(PreUpdate, retarget_cameras)
            .add_systems(
                Startup,
                spawn_mirror.after(create_target).run_if({
                    let windowed = self.windowed;
                    move || windowed
                }),
            )
            .add_systems(Update, drain_frames)
            // Last: runs after user systems, so an AppExit written anywhere
            // this frame still gets an EOS before the runner stops the loop.
            .add_systems(Last, eos_on_exit);

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(FrameSender(tx))
            .insert_resource(Producer::default())
            .insert_resource(ZeroCopy(self.zero_copy))
            // The settings are static, so a plain clone in the render world
            // beats a per-frame extract.
            .insert_resource(Settings(self.settings.clone()))
            .add_systems(Render, produce_frame.after(RenderSystems::Render));
    }
}

/// The texture the cameras render into. COPY_SRC so either path can copy out
/// of it (the CUDA bridge device->device, the readback path device->buffer).
/// Rgba8UnormSrgb is copy-compatible with the bridge's Rgba8Unorm export image
/// and reads back as sRGB-encoded bytes, which is what the encoder wants.
fn create_target(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    settings: Res<Settings>,
) {
    let mut target = Image::new_target_texture(
        settings.0.width,
        settings.0.height,
        TextureFormat::Rgba8UnormSrgb,
        None,
    );
    target.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    commands.insert_resource(StreamTarget(images.add(target)));
}

/// Point every camera that still targets the (nonexistent) primary window at
/// the stream texture, so a stock scene-with-a-camera app streams with no
/// app-side render-target code. Cameras aimed at another image are left alone.
type UntargetedCamera = (With<Camera>, Without<RenderTarget>, Without<MirrorCamera>);

fn retarget_cameras(
    target: Res<StreamTarget>,
    mut with_target: Query<&mut RenderTarget, (With<Camera>, Without<MirrorCamera>)>,
    without_target: Query<Entity, UntargetedCamera>,
    mut commands: Commands,
) {
    for mut rt in &mut with_target {
        if matches!(*rt, RenderTarget::Window(WindowRef::Primary)) {
            *rt = RenderTarget::Image(target.0.clone().into());
        }
    }
    for entity in &without_target {
        commands
            .entity(entity)
            .insert(RenderTarget::Image(target.0.clone().into()));
    }
}

/// Windowed mode: a 2D mirror camera shows the stream texture fullscreen in
/// the window, so the desktop view and the stream are the same pixels. The
/// mirror is the default UI camera; UI meant for the stream itself goes on
/// the scene camera via `UiTargetCamera`.
fn spawn_mirror(mut commands: Commands, target: Res<StreamTarget>) {
    let camera = commands
        .spawn((
            Camera2d,
            // Above the scene camera's order so same-target warnings cannot
            // arise if an app camera ever ends up on the window too.
            Camera {
                order: 100,
                ..default()
            },
            MirrorCamera,
            IsDefaultUiCamera,
        ))
        .id();
    commands.spawn((
        ImageNode::new(target.0.clone()),
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
        UiTargetCamera(camera),
    ));
}

/// Drain produced frames in the main world: push them into the g2g sink
/// pipeline and exit once `max_frames` is reached (when set).
fn drain_frames(
    receiver: Res<FrameReceiver>,
    feed: Res<EncodeFeed>,
    settings: Res<Settings>,
    mut state: ResMut<StreamState>,
    mut exit: MessageWriter<AppExit>,
) {
    if state.eos_sent {
        return;
    }
    while let Ok(message) = receiver.0.try_recv() {
        let RenderMessage::Frame(data, pts_ns) = message else {
            if let RenderMessage::Fatal(reason) = message {
                error!("{reason}");
                FAILED.store(true, Ordering::Relaxed);
                feed.0.end_of_stream_blocking();
                state.eos_sent = true;
                exit.write(AppExit::error());
            }
            return;
        };
        if state.frames == 0 {
            info!("first frame handed to g2g: {} bytes", data.len());
        }
        // Backpressure here slows the render loop instead of dropping frames
        // before the sink can consume them.
        if !feed.0.push_blocking(&data, pts_ns) {
            FAILED.store(true, Ordering::Relaxed);
            error!("sink feed closed before frame {}", state.frames);
            state.eos_sent = true;
            exit.write(AppExit::error());
            return;
        }
        state.frames += 1;
        if settings.0.max_frames != 0 && state.frames >= settings.0.max_frames {
            feed.0.end_of_stream_blocking();
            state.eos_sent = true;
            info!("streamed {} frames; EOS sent, exiting", state.frames);
            exit.write(AppExit::Success);
            return;
        }
    }
}

/// An app-initiated exit (window logic, a game-over system, ...) must still
/// flush the stream: send EOS when an `AppExit` message is seen.
fn eos_on_exit(
    mut exits: MessageReader<AppExit>,
    feed: Res<EncodeFeed>,
    mut state: ResMut<StreamState>,
) {
    if exits.read().next().is_some() && !state.eos_sent {
        feed.0.end_of_stream_blocking();
        state.eos_sent = true;
    }
}

/// Render-world system: after Bevy renders the scene into the target texture,
/// hand the frame to the sink. Zero-copy path: device->device into CUDA and
/// NVENC-encode (H.264 out). Readback path: copy into a mapped buffer (raw
/// RGBA out; the sink pipeline encodes).
#[allow(clippy::too_many_arguments)] // bevy system: each param is an injected resource
fn produce_frame(
    producer: Res<Producer>,
    zero_copy: Res<ZeroCopy>,
    settings: Res<Settings>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    target: Option<Res<StreamTarget>>,
    sender: Res<FrameSender>,
) {
    // The target's GPU texture is only present once the image asset prepared.
    let Some(target) = target else {
        return;
    };
    let Some(gpu_image) = gpu_images.get(&target.0) else {
        return;
    };

    let mut guard = producer.0.lock().expect("producer lock");
    let state = match &mut *guard {
        Some(s) => s,
        none => {
            let built: Result<PathState, G2gError> = if zero_copy.0 {
                #[cfg(feature = "nvenc")]
                {
                    zerocopy::EncodeState::new(
                        device.wgpu_device().clone(),
                        (**queue.0).clone(),
                        &settings.0,
                    )
                    .map(PathState::ZeroCopy)
                }
                #[cfg(not(feature = "nvenc"))]
                unreachable!("zero_copy set without the nvenc feature")
            } else {
                Ok(PathState::Readback(ReadbackState::new(
                    device.wgpu_device(),
                    &settings.0,
                )))
            };
            match built {
                Ok(s) => none.insert(s),
                Err(e) => {
                    let _ = sender.0.send(RenderMessage::Fatal(format!(
                        "failed to initialize the encode path: {e:?}"
                    )));
                    return;
                }
            }
        }
    };

    let result = match state {
        #[cfg(feature = "nvenc")]
        PathState::ZeroCopy(z) => z.encode(&gpu_image.texture).map(|aus| {
            for (au, pts_ns) in aus {
                let _ = sender.0.send(RenderMessage::Frame(au, pts_ns));
            }
        }),
        PathState::Readback(r) => r
            .read(device.wgpu_device(), &queue.0, &gpu_image.texture)
            .map(|(rgba, pts_ns)| {
                let _ = sender.0.send(RenderMessage::Frame(rgba, pts_ns));
            }),
    };
    if let Err(e) = result {
        let _ = sender
            .0
            .send(RenderMessage::Fatal(format!("frame capture failed: {e:?}")));
    }
}

/// The readback fallback: copy the rendered texture into a persistent
/// MAP_READ buffer, block on the map, and hand the tightly packed RGBA bytes
/// to the sink pipeline (which converts + encodes on the CPU).
struct ReadbackState {
    buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
    width: u32,
    height: u32,
    fps: u32,
    frame_no: u64,
}

impl ReadbackState {
    fn new(device: &wgpu::Device, settings: &StreamSettings) -> Self {
        let unpadded = settings.width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bevy-g2g readback"),
            size: padded as u64 * settings.height as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            padded_bytes_per_row: padded,
            width: settings.width,
            height: settings.height,
            fps: settings.fps,
            frame_no: 0,
        }
    }

    fn read(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Result<(Vec<u8>, u64), G2gError> {
        let pts_ns = self.frame_no * 1_000_000_000 / self.fps as u64;
        self.frame_no += 1;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bevy-g2g readback"),
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| G2gError::Hardware(HardwareError::Wgpu))?;
        rx.recv()
            .map_err(|_| G2gError::Hardware(HardwareError::Wgpu))?
            .map_err(|_| G2gError::Hardware(HardwareError::Wgpu))?;

        let unpadded = (self.width * 4) as usize;
        let mut rgba = Vec::with_capacity(unpadded * self.height as usize);
        {
            let data = slice.get_mapped_range();
            for row in data.chunks_exact(self.padded_bytes_per_row as usize) {
                rgba.extend_from_slice(&row[..unpadded]);
            }
        }
        self.buffer.unmap();
        Ok((rgba, pts_ns))
    }
}

/// Drives the sink chain to completion on its own thread. Zero-copy path:
/// `AppSrc(H.264) -> sink` (the GPU already encoded). Readback path:
/// `AppSrc(RGBA) -> videoconvert(I420) -> ffmpegenc(libx264) -> sink`. The
/// sink is `WebRtcSink` (WHIP) or `FileSink` per the settings. Returns the
/// number of frames consumed.
///
/// Runs inside a tokio runtime: `WebRtcSink`'s WHIP handshake (reqwest) and
/// session (tokio::spawn) need a reactor. `FileSink` is happy under it too.
fn sink_pipeline(settings: StreamSettings, encoded: bool) -> Result<u64, G2gError> {
    let (width, height, fps) = (settings.width, settings.height, settings.fps);
    let mut src = AppSrc::new();
    src.set_property("channel", PropValue::Str(CHANNEL.into()))
        .expect("appsrc channel");
    let caps = if encoded {
        format!("video/x-h264,width={width},height={height},framerate={fps}/1")
    } else {
        format!("video/x-raw,format=RGBA,width={width},height={height},framerate={fps}/1")
    };
    src.set_property("caps", PropValue::Str(caps))
        .expect("appsrc caps");

    let mut convert = VideoConvert::new(RawVideoFormat::I420);
    let mut enc = FfmpegH264Enc::new()
        .with_backend(Backend::Software)
        .with_bitrate(settings.bitrate as usize);
    let transforms: Vec<&mut dyn DynAsyncElement> = if encoded {
        vec![]
    } else {
        vec![&mut convert, &mut enc]
    };

    let clock = ZeroClock;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let stats = match settings.output {
        StreamOutput::Whip(url) => {
            info!("streaming H.264 to WHIP endpoint: {url}");
            let mut sink = WebRtcSink::new(url);
            rt.block_on(run_linear_chain(&mut src, transforms, &mut sink, &clock, 4))?
        }
        StreamOutput::File(path) => {
            info!("writing H.264 to {path}");
            let mut sink = FileSink::new(&path);
            rt.block_on(run_linear_chain(&mut src, transforms, &mut sink, &clock, 4))?
        }
    };
    Ok(stats.frames_consumed)
}

/// Trivial clock: the sinks here do not pace to a clock, so the runner needs
/// only a `now_ns`, never advanced.
struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}
