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
        run_simple_pipeline(&mut src, &mut sink, &clock, LatencyProfile::Live.link_capacity()),
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
    assert_eq!(stats.frames_emitted, target, "source should emit the requested frame count");
    assert!(sink.received() > 0, "sink received no frames");
    assert_eq!(sink.last_sequence(), Some(target - 1), "frames arrive in order");
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
    // ~10 s at 30 fps by default; override with G2G_LIBCAMERA_FRAMES.
    let target: u64 = std::env::var("G2G_LIBCAMERA_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);
    let mut src = LibCameraSrc::new()
        .with_camera(camera_index())
        .with_size(640, 480)
        .with_fps(30)
        .with_frame_limit(target);
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

    eprintln!("emitted={} presented={}", stats.frames_emitted, sink.frames_presented());
    assert!(stats.frames_emitted > 0, "no frames captured");
}
