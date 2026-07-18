//! End-to-end smoke test for the libcamera capture source.
//!
//! Pipeline: `LibCameraSrc -> FakeSink`. The source negotiates NV12 (else
//! YUYV) with the camera, so FakeSink (format-agnostic) just counts frames.
//!
//! Ignored by default: needs a real camera libcamera can open and that the
//! running user can access (a local desktop session grants this via a device
//! ACL on `/dev/videoN`; otherwise join the `video` group). Select a non-default
//! camera with `G2G_LIBCAMERA_INDEX`.
//!
//! ```sh
//! cargo test -p g2g-plugins --features libcamera \
//!     --test libcamera_smoke -- --ignored --nocapture
//!
//! # visual confirmation in a window (needs a Wayland session):
//! cargo test -p g2g-plugins --features "libcamera wayland-sink videoconvert" \
//!     --test libcamera_smoke libcamera_capture_displays -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "linux", feature = "libcamera"))]

use g2g_core::runtime::{run_simple_pipeline, LatencyProfile};
use g2g_core::PipelineClock;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::libcamerasrc::LibCameraSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Sink that averages the luma (Y) of each captured frame. The camera emits
/// YUYV (Y at even byte offsets), so this tracks overall image brightness,
/// used to prove the `brightness` control actually changes the pixels.
#[derive(Default)]
struct LumaSink {
    sum_of_means: f64,
    frames: u64,
}

impl LumaSink {
    fn mean_luma(&self) -> f64 {
        if self.frames == 0 {
            0.0
        } else {
            self.sum_of_means / self.frames as f64
        }
    }
}

impl g2g_core::element::OutputSink for LumaSink {
    fn push<'a>(
        &'a mut self,
        packet: g2g_core::frame::PipelinePacket,
    ) -> g2g_core::element::BoxFuture<'a, Result<g2g_core::element::PushOutcome, g2g_core::G2gError>>
    {
        Box::pin(async move {
            if let g2g_core::frame::PipelinePacket::DataFrame(f) = &packet {
                if let g2g_core::memory::MemoryDomain::System(s) = &f.domain {
                    let bytes = s.as_slice();
                    let (mut sum, mut n) = (0u64, 0u64);
                    let mut i = 0;
                    while i < bytes.len() {
                        sum += bytes[i] as u64; // Y in YUYV
                        n += 1;
                        i += 2;
                    }
                    if n > 0 {
                        self.sum_of_means += sum as f64 / n as f64;
                        self.frames += 1;
                    }
                }
            }
            Ok(g2g_core::element::PushOutcome::Accepted)
        })
    }
}

fn camera_index() -> usize {
    std::env::var("G2G_LIBCAMERA_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

#[tokio::test]
#[ignore = "needs a real camera libcamera can open (set G2G_LIBCAMERA_INDEX)"]
async fn libcamera_capture_to_fakesink_yields_frames() {
    let target: u64 = 30;
    let mut src = LibCameraSrc::new()
        .with_camera(camera_index())
        .with_size(640, 480)
        .with_fps(30)
        .with_frame_limit(target);
    let mut sink = FakeSink::new();
    let clock = ZeroClock;

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("capture should finish within 30s")
    .expect("libcamera capture pipeline should succeed");

    eprintln!(
        "emitted={} received={} last_seq={:?} last_bytes={:?}",
        stats.frames_emitted,
        sink.received(),
        sink.last_sequence(),
        sink.last_view_bytes().map(|b| b.len()),
    );
    assert_eq!(
        stats.frames_emitted, target,
        "source should emit the requested frame count"
    );
    assert!(sink.received() > 0, "sink received no frames");
    assert_eq!(
        sink.last_sequence(),
        Some(target - 1),
        "frames arrive in order"
    );
}

/// Prove `FrameDurationLimits` actually throttles: at a forced 8 fps (below the
/// camera's mode ceiling, so achievable) the measured rate tracks 8 fps, well
/// under the camera's free-running rate. A pure consumer-side cap would not
/// slow the source, so the wall time bracketing the requested rate shows
/// libcamera held the interval. (Note: a cap *above* a mode's max fps cannot
/// raise it, e.g. uncompressed YUYV at higher resolutions is USB-bandwidth
/// limited; use MJPEG for high frame rates.)
#[tokio::test]
#[ignore = "needs a real camera libcamera can open (set G2G_LIBCAMERA_INDEX)"]
async fn libcamera_fps_limit_is_enforced() {
    let fps: u32 = std::env::var("G2G_LIBCAMERA_FPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let target: u64 = std::env::var("G2G_LIBCAMERA_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let mjpeg = std::env::var("G2G_LIBCAMERA_MJPEG").is_ok();
    let w: u32 = std::env::var("G2G_LIBCAMERA_W")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(640);
    let h: u32 = std::env::var("G2G_LIBCAMERA_H")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(480);
    let mut src = LibCameraSrc::new()
        .with_camera(camera_index())
        .with_size(w, h)
        .with_fps(fps)
        .with_mjpeg(mjpeg)
        .with_frame_limit(target);
    // Optional manual exposure (us) / gain to lift the auto-exposure fps cap.
    if let Some(e) = std::env::var("G2G_LIBCAMERA_EXPOSURE")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        src = src.with_exposure(e);
    }
    if let Some(g) = std::env::var("G2G_LIBCAMERA_GAIN")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        src = src.with_gain(g);
    }
    let mut sink = FakeSink::new();
    let clock = ZeroClock;

    let start = std::time::Instant::now();
    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("capture should finish within 30s")
    .expect("libcamera capture pipeline should succeed");
    let elapsed = start.elapsed();

    let actual_fps = stats.frames_emitted as f64 / elapsed.as_secs_f64();
    eprintln!(
        "captured {} frames {w}x{h} {} at cap={fps} in {:.2}s (actual {:.1} fps)",
        stats.frames_emitted,
        if mjpeg { "mjpeg" } else { "raw" },
        elapsed.as_secs_f64(),
        actual_fps,
    );
    assert_eq!(stats.frames_emitted, target);
    // fps == 0 is a diagnostic free-run (no cap): just report, no bounds.
    if fps == 0 {
        return;
    }
    // Ceiling: never faster than the requested cap (a frame or two of slack).
    assert!(
        actual_fps <= fps as f64 + 2.0,
        "captured faster ({actual_fps:.1} fps) than the {fps} fps cap"
    );
    // Floor (only when the rate is achievable for the mode): the cap genuinely
    // paced the stream rather than the rate collapsing. 8 fps is within YUYV
    // 640x480's reach on typical UVC cams.
    if fps <= 10 {
        assert!(
            actual_fps >= fps as f64 * 0.6,
            "rate collapsed to {actual_fps:.1} fps under a {fps} fps cap"
        );
    }
}

/// Prove the `brightness` control changes the pixels: at a fixed exposure (so
/// the camera cannot auto-compensate), a high brightness yields a clearly
/// brighter average luma than a low one. This is the low-light lever that
/// brightens *without* lowering the frame rate.
#[tokio::test]
#[ignore = "needs a real camera that supports the Brightness control"]
async fn libcamera_brightness_changes_luma() {
    async fn mean_luma(brightness: f32) -> f64 {
        use g2g_core::runtime::SourceLoop as _;
        let mut src = LibCameraSrc::new()
            .with_camera(camera_index())
            .with_size(640, 480)
            .with_fps(15)
            .with_exposure(8_000) // fixed exposure: isolate the brightness effect
            .with_brightness(brightness)
            .with_frame_limit(10);
        // Drive the source directly (LumaSink is an OutputSink, not a full
        // AsyncElement), the minimal source -> sink path.
        let caps = src.intercept_caps().await.expect("negotiate");
        src.configure_pipeline(&caps).expect("configure");
        let mut sink = LumaSink::default();
        tokio::time::timeout(std::time::Duration::from_secs(20), src.run(&mut sink))
            .await
            .expect("finishes")
            .expect("capture succeeds");
        sink.mean_luma()
    }

    let dark = mean_luma(-0.8).await;
    let bright = mean_luma(0.9).await;
    eprintln!("mean luma: brightness -0.8 -> {dark:.1}, brightness +0.9 -> {bright:.1}");
    assert!(
        bright > dark + 5.0,
        "brightness control had no visible effect: dark={dark:.1} bright={bright:.1}"
    );
}

/// Select the camera by an id substring instead of index: a matching substring
/// captures, a non-matching one fails negotiation (no camera selected).
#[tokio::test]
#[ignore = "needs a real camera; set G2G_LIBCAMERA_ID to a substring of its id"]
async fn libcamera_camera_id_selects() {
    // Default is this developer's webcam USB VID:PID; override per device.
    let id = std::env::var("G2G_LIBCAMERA_ID").unwrap_or_else(|_| "30c9:0057".to_string());

    let mut src = LibCameraSrc::new()
        .with_camera_id(&id)
        .with_size(640, 480)
        .with_fps(15)
        .with_frame_limit(5);
    let mut sink = FakeSink::new();
    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        run_simple_pipeline(
            &mut src,
            &mut sink,
            &ZeroClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("finishes")
    .expect("by-id selection should capture");
    assert_eq!(
        stats.frames_emitted, 5,
        "selected camera by id and captured"
    );

    // A bogus id matches no camera, so negotiation must fail (not pick #0).
    let mut bad = LibCameraSrc::new()
        .with_camera_id("no-such-camera-zzz")
        .with_frame_limit(1);
    let mut sink2 = FakeSink::new();
    let r = run_simple_pipeline(
        &mut bad,
        &mut sink2,
        &ZeroClock,
        LatencyProfile::Live.link_capacity(),
    )
    .await;
    assert!(
        r.is_err(),
        "a non-matching camera id must fail, not fall back to index 0"
    );
}

/// Prove manual exposure lifts the frame rate that auto-exposure caps in low
/// light. Two back-to-back captures at a 30 fps request: one with AE on
/// (default), one with a fixed short exposure (AE off). A short exposure
/// removes the per-frame exposure time as the bottleneck, so it reaches a
/// rate the AE run cannot in a dim room. (Unsupported controls like
/// `AnalogueGain` on this UVC cam are skipped, not set, so this never aborts.)
#[tokio::test]
#[ignore = "needs a real camera; the effect is clearest in a dim room"]
async fn libcamera_manual_exposure_lifts_fps() {
    async fn measure(exposure_us: Option<i32>) -> f64 {
        let target: u64 = 40;
        let mut src = LibCameraSrc::new()
            .with_camera(camera_index())
            .with_size(640, 480)
            .with_fps(30)
            .with_frame_limit(target);
        if let Some(e) = exposure_us {
            src = src.with_exposure(e);
        }
        let mut sink = FakeSink::new();
        let clock = ZeroClock;
        let start = std::time::Instant::now();
        let stats = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            run_simple_pipeline(
                &mut src,
                &mut sink,
                &clock,
                LatencyProfile::Live.link_capacity(),
            ),
        )
        .await
        .expect("finishes in 30s")
        .expect("capture succeeds");
        stats.frames_emitted as f64 / start.elapsed().as_secs_f64()
    }

    let rate_ae = measure(None).await;
    let rate_manual = measure(Some(8_000)).await; // 8 ms exposure -> up to 125 fps
    eprintln!("auto-exposure: {rate_ae:.1} fps, manual 8ms exposure: {rate_manual:.1} fps");

    // A fixed short exposure is not exposure-bound, so it reaches a high rate
    // regardless of lighting; in a dim room it clearly beats the AE rate.
    assert!(
        rate_manual > 15.0,
        "manual exposure did not lift the rate: {rate_manual:.1} fps"
    );
    assert!(
        rate_manual >= rate_ae - 1.0,
        "manual exposure ({rate_manual:.1}) slower than auto ({rate_ae:.1})"
    );
}

/// MJPEG path: the source negotiates `CompressedVideo{Mjpeg}` and `MjpegDec`
/// decodes the camera's real JPEGs to raw frames end to end. (MJPEG's frame-rate
/// benefit over uncompressed YUYV is real but only shows when the camera is not
/// otherwise limited, e.g. by auto-exposure in low light, which caps this
/// developer's webcam to ~9 fps in every format, so the fps is reported, not
/// asserted.)
#[cfg(feature = "mjpeg")]
#[tokio::test]
#[ignore = "needs a real camera that offers MJPEG (most UVC webcams do)"]
async fn libcamera_mjpeg_capture_decodes() {
    use g2g_core::runtime::{run_source_transform_sink, SourceLoop as _};
    use g2g_core::{Caps, VideoCodec};
    use g2g_plugins::mjpegdec::MjpegDec;

    // The source must advertise MJPEG (compressed) caps in MJPEG mode.
    let mut probe = LibCameraSrc::new()
        .with_camera(camera_index())
        .with_size(640, 480)
        .with_mjpeg(true);
    let caps = probe.intercept_caps().await.expect("negotiate mjpeg");
    assert!(
        matches!(
            caps,
            Caps::CompressedVideo {
                codec: VideoCodec::Mjpeg,
                ..
            }
        ),
        "expected CompressedVideo(Mjpeg), got {caps:?}"
    );

    let target: u64 = 30;
    let mut src = LibCameraSrc::new()
        .with_camera(camera_index())
        .with_size(640, 480)
        .with_fps(30)
        .with_mjpeg(true)
        .with_frame_limit(target);
    let mut dec = MjpegDec::new();
    let mut sink = FakeSink::new();
    let clock = ZeroClock;

    let start = std::time::Instant::now();
    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        run_source_transform_sink(
            &mut src,
            &mut dec,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("capture should finish within 30s")
    .expect("libcamera mjpeg -> mjpegdec pipeline should succeed");
    let actual_fps = stats.frames_emitted as f64 / start.elapsed().as_secs_f64();

    eprintln!(
        "mjpeg: {} frames captured + decoded, sink received {} ({:.1} fps)",
        stats.frames_emitted,
        sink.received(),
        actual_fps,
    );
    assert_eq!(stats.frames_emitted, target, "all MJPEG frames captured");
    assert_eq!(sink.received(), target, "all frames decoded to the sink");
}

#[cfg(feature = "wayland-sink")]
#[tokio::test]
#[ignore = "needs a camera + a Wayland session"]
async fn libcamera_capture_displays_in_a_window() {
    use g2g_core::runtime::run_source_transform_sink;
    use g2g_core::RawVideoFormat;
    use g2g_plugins::videoconvert::VideoConvert;
    use g2g_plugins::waylandsink::WaylandSink;

    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping: no WAYLAND_DISPLAY (run under a Wayland session)");
        return;
    }
    let fps: u32 = std::env::var("G2G_LIBCAMERA_FPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    // ~10 s of capture by default; override with G2G_LIBCAMERA_FRAMES.
    let target: u64 = std::env::var("G2G_LIBCAMERA_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or((fps as u64) * 10);
    let mut src = LibCameraSrc::new()
        .with_camera(camera_index())
        .with_size(640, 480)
        .with_fps(fps)
        .with_frame_limit(target);
    // Optional manual exposure (us) to keep the rate up in low light.
    if let Some(e) = std::env::var("G2G_LIBCAMERA_EXPOSURE")
        .ok()
        .and_then(|s| s.parse().ok())
    {
        src = src.with_exposure(e);
    }
    // libcamera gives YUYV on this UVC cam; WaylandSink wants NV12.
    let mut conv = VideoConvert::new(RawVideoFormat::Nv12);
    let mut sink = WaylandSink::new().with_title("glass2glass libcamera capture");
    let clock = ZeroClock;

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        run_source_transform_sink(
            &mut src,
            &mut conv,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("capture should finish within 60s")
    .expect("libcamera -> wayland pipeline should succeed");

    eprintln!(
        "emitted={} presented={}",
        stats.frames_emitted,
        sink.frames_presented()
    );
    assert!(stats.frames_emitted > 0, "no frames captured");
}
