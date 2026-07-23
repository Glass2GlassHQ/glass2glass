//! M742: true on-screen Android present. The APK harness the bare-binary
//! probes cannot be: a `NativeActivity` owns a real on-screen window (the
//! `SurfaceView` case), `MediaCodec` decodes the bundled H.264 clip zero-copy
//! onto the GPU (`mediacodec-wgpu`), and `WgpuSink` presents each frame to a
//! `wgpu::Surface` built over the activity's `ANativeWindow`. The on-screen
//! sibling of `android_surface_present_probe` (which stands the window in with
//! a headless `ImageReader`).
//!
//! Frames present as they decode (decode -> present, the probe's shape); at end
//! of stream the decoder is rebuilt and the clip loops. The decoder stays alive
//! while its output textures are in flight: a decoded GPU frame borrows codec
//! output resources, so presenting retained textures after dropping the codec
//! is not sound.
//!
//! Build + install + run: `tools/android-apk-present-smoke.sh` (needs the NDK,
//! cargo-ndk, and SDK build-tools; see the script header). The device must be
//! unlocked: behind the keyguard the activity is stopped and its window taken
//! away before any frame presents.
//!
//! Unlike the `/data/local/tmp` probes there is no binder-threadpool shim here:
//! an APK process forks from zygote with its binder threadpool already running,
//! so Codec2's buffer allocation just works.

#![cfg(target_os = "android")]

use std::time::{Duration, Instant};

use android_activity::{AndroidApp, MainEvent, PollEvent};
use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::block_on;
use g2g_core::{Caps, ConfigureOutcome, Dim, G2gError, Rate, RawVideoFormat, VideoCodec};
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::mediacodec_wgpu::{
    create_android_interop_device, create_android_surface, InteropDevice,
};
use g2g_plugins::mediacodecdec::MediaCodecDec;
use g2g_plugins::wgpusink::WgpuSink;

/// The 640x480 baseline clip the g2g tests use (two GOPs of IDR + P frames,
/// with a scrolling gradient band), embedded so the APK is self-contained.
const H264: &[u8] = include_bytes!("../../../g2g-plugins/tests/fixtures/h264_640x480.h264");

/// The clip is authored at 30 fps; play it back at that rate.
const FRAME_INTERVAL: Duration = Duration::from_millis(33);

const FRAME_DURATION_NS: u64 = 33_366_700;

/// Collects an element's output packets.
#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Discards output (the present sink is the pipeline tail).
struct Discard;

impl OutputSink for Discard {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move { Ok(PushOutcome::Accepted) })
    }
}

/// Forwards the decoder's output straight into the present sink, so each
/// decoded frame is presented as it is produced.
struct PresentRelay<'s> {
    sink: &'s mut WgpuSink,
}

impl OutputSink for PresentRelay<'_> {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            let mut nil = Discard;
            self.sink.process(packet, &mut nil).await?;
            Ok(PushOutcome::Accepted)
        })
    }
}

fn h264_caps(w: u32, h: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

fn rgba_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

/// The wgpu handles are refcounted; a field-wise clone shares the one device,
/// so a rebuilt decoder emits textures the existing sink can bind.
fn clone_dev(dev: &InteropDevice) -> InteropDevice {
    InteropDevice {
        device: dev.device.clone(),
        queue: dev.queue.clone(),
        adapter: dev.adapter.clone(),
        instance: dev.instance.clone(),
    }
}

/// Parse the Annex-B clip into access units once (the first AU carries the
/// parameter sets alongside the IDR, as `MediaCodecDec` requires).
fn parse_access_units() -> Result<Vec<Vec<u8>>, G2gError> {
    let mut parse = H264Parse::reframing();
    parse.configure_pipeline(&h264_caps(640, 480))?;
    let mut sink = Collect::default();
    block_on(async {
        for (i, chunk) in H264.chunks(4096).enumerate() {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(chunk.to_vec().into_boxed_slice())),
                FrameTiming::default(),
                i as u64,
            );
            parse
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await?;
        }
        parse.process(PipelinePacket::Eos, &mut sink).await
    })?;
    Ok(sink
        .packets
        .into_iter()
        .filter_map(|p| match p {
            PipelinePacket::DataFrame(f) => f.domain.as_system_slice().map(<[u8]>::to_vec),
            _ => None,
        })
        .collect())
}

fn build_decoder(dev: &InteropDevice) -> Result<MediaCodecDec, G2gError> {
    let mut dec = MediaCodecDec::h264().with_gpu_device(clone_dev(dev));
    let narrowed = dec.intercept_caps(&h264_caps(640, 480))?;
    if !matches!(
        dec.configure_pipeline(&narrowed)?,
        ConfigureOutcome::Accepted
    ) {
        return Err(G2gError::CapsMismatch);
    }
    Ok(dec)
}

/// The live present state. Dropped when the window goes away
/// (`TerminateWindow`); the surface must not outlive the `ANativeWindow`.
struct Presenter {
    dev: InteropDevice,
    aus: Vec<Vec<u8>>,
    dec: MediaCodecDec,
    sink: WgpuSink,
    au_idx: usize,
    pts_ns: u64,
    last_feed: Instant,
    loops: u64,
    // Declared last so the sink (and its surface) drop first: the surface was
    // built over this window and must not outlive it.
    _window: ndk::native_window::NativeWindow,
}

impl Presenter {
    fn new(window: ndk::native_window::NativeWindow) -> Result<Self, G2gError> {
        // One interop device for decode and present: the decoded texture binds
        // only to the device that made it.
        let dev = block_on(create_android_interop_device())?;
        let ctx = dev.gpu_context();
        let (sw, sh) = (window.width() as u32, window.height() as u32);
        let (surface, config) = create_android_surface(&dev, &window, sw, sh)?;
        log::info!(
            "surface configured: {:?} {}x{}",
            config.format,
            config.width,
            config.height
        );
        let aus = parse_access_units()?;
        log::info!("parsed {} access units", aus.len());
        if aus.is_empty() {
            return Err(G2gError::CapsMismatch);
        }
        let dec = build_decoder(&dev)?;
        let mut sink = WgpuSink::with_surface(ctx, surface, config);
        // Input caps are the video geometry; the fullscreen-triangle blit
        // scales it to the window.
        sink.configure_pipeline(&rgba_caps(640, 480))?;
        Ok(Self {
            dev,
            aus,
            dec,
            sink,
            au_idx: 0,
            pts_ns: 0,
            last_feed: Instant::now(),
            loops: 0,
            _window: window,
        })
    }

    /// Feed the next access unit once `FRAME_INTERVAL` has elapsed; the decoder
    /// pushes each decoded GPU frame straight into the present sink. At end of
    /// stream the codec is drained and rebuilt, looping the clip.
    fn tick(&mut self) {
        if self.last_feed.elapsed() < FRAME_INTERVAL {
            return;
        }
        self.last_feed = Instant::now();
        let au = self.aus[self.au_idx].clone();
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns: self.pts_ns,
                dts_ns: self.pts_ns,
                capture_ns: self.pts_ns,
                ..FrameTiming::default()
            },
            sequence: 0,
            meta: Default::default(),
        };
        self.pts_ns += FRAME_DURATION_NS;
        let mut relay = PresentRelay {
            sink: &mut self.sink,
        };
        if let Err(e) = block_on(
            self.dec
                .process(PipelinePacket::DataFrame(frame), &mut relay),
        ) {
            log::error!("decode+present failed: {e:?}");
            return;
        }
        self.au_idx += 1;
        if self.au_idx == self.aus.len() {
            // Drain the codec's tail through the sink, then rebuild for the
            // next loop of the clip.
            if let Err(e) = block_on(self.dec.process(PipelinePacket::Eos, &mut relay)) {
                log::error!("Eos drain failed: {e:?}");
            }
            drop(relay);
            match build_decoder(&self.dev) {
                Ok(d) => self.dec = d,
                Err(e) => log::error!("decoder rebuild failed: {e:?}"),
            }
            self.au_idx = 0;
            self.loops += 1;
            if self.loops % 10 == 1 {
                log::info!(
                    "presented {} frames over {} loops of the clip",
                    self.sink.presented_count(),
                    self.loops
                );
            }
        }
    }
}

#[no_mangle]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("g2g-present"),
    );
    log::info!("g2g on-screen present harness starting");

    let mut presenter: Option<Presenter> = None;
    let mut window_ready = false;
    let mut quit = false;
    while !quit {
        // Only flag lifecycle changes inside the callback: Android's main
        // (Java) thread blocks until some events are acknowledged, and codec /
        // window framework calls can wait on that thread in turn, so doing the
        // multi-second init inside the callback risks a deadlock.
        app.poll_events(Some(Duration::from_millis(8)), |event| {
            if let PollEvent::Main(main) = event {
                match main {
                    MainEvent::InitWindow { .. } => {
                        window_ready = true;
                    }
                    MainEvent::TerminateWindow { .. } => {
                        // The lockscreen / backgrounding takes the window; the
                        // presenter (surface, then its window) goes with it. A
                        // later InitWindow rebuilds both.
                        log::info!("window gone; presenter torn down");
                        window_ready = false;
                        presenter = None;
                    }
                    MainEvent::Destroy => {
                        quit = true;
                    }
                    _ => {}
                }
            }
        });
        if window_ready && presenter.is_none() {
            // The window the OS just gave the activity: the real on-screen
            // surface the bare-binary probes cannot own.
            if let Some(window) = app.native_window() {
                match Presenter::new(window) {
                    Ok(p) => presenter = Some(p),
                    Err(e) => {
                        log::error!("failed to start presenter: {e:?}");
                        window_ready = false;
                    }
                }
            }
        }
        if let Some(p) = presenter.as_mut() {
            p.tick();
        }
    }
    log::info!("g2g present harness exiting");
}
