//! End-to-end smoke test for the V4L2 capture source.
//!
//! Pipeline: `V4l2Src -> VideoConvert(Yuyv -> Nv12) -> FakeSink`, with an
//! optional Wayland display variant when the `wayland-sink` feature is on.
//!
//! Ignored by default: needs a real `/dev/videoN` UVC device the running user
//! can open (a local desktop session grants this via a device ACL; otherwise
//! join the `video` group). Override the device with `G2G_V4L2_DEVICE`.
//!
//! ```sh
//! cargo test -p g2g-plugins --features "v4l2 ffmpeg" \
//!     --test v4l2_smoke -- --ignored --nocapture
//!
//! # visual confirmation in a window (needs a Wayland session):
//! cargo test -p g2g-plugins --features "v4l2 wayland-sink" \
//!     --test v4l2_smoke v4l2_capture_displays -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "linux", feature = "v4l2"))]

use g2g_core::runtime::{run_source_transform_sink, LatencyProfile};
use g2g_core::PipelineClock;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::v4l2src::V4l2Src;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_core::RawVideoFormat;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn device() -> String {
    std::env::var("G2G_V4L2_DEVICE").unwrap_or_else(|_| "/dev/video0".to_string())
}

#[tokio::test]
#[ignore = "needs a real /dev/videoN device (set G2G_V4L2_DEVICE)"]
async fn v4l2_capture_to_fakesink_yields_frames() {
    let dev = device();
    eprintln!("capturing from {dev}");

    let target: u64 = 30;
    let mut src = V4l2Src::new(dev).with_size(640, 480).with_fps(30).with_frame_limit(target);
    let mut conv = VideoConvert::new(RawVideoFormat::Nv12);
    let mut sink = FakeSink::new();
    let clock = ZeroClock;

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(20),
        run_source_transform_sink(
            &mut src,
            &mut conv,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("capture should finish within 20s")
    .expect("v4l2 capture pipeline should succeed");

    eprintln!(
        "emitted={} received={} last_seq={:?}",
        stats.frames_emitted,
        sink.received(),
        sink.last_sequence()
    );
    assert_eq!(stats.frames_emitted, target, "source should emit the requested frame count");
    assert!(sink.received() > 0, "sink received no converted frames");
    // The convert step turns YUYV (w*h*2) into NV12 (w*h*3/2); reaching the
    // sink at all proves the YUYV unpack negotiated and ran on real data.
    assert_eq!(sink.last_sequence(), Some(target - 1), "frames arrive in order");
}

#[cfg(feature = "wayland-sink")]
#[tokio::test]
#[ignore = "needs a /dev/videoN device + a Wayland session"]
async fn v4l2_capture_displays_in_a_window() {
    use g2g_plugins::waylandsink::WaylandSink;

    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping: no WAYLAND_DISPLAY (run under a Wayland session)");
        return;
    }
    let dev = device();
    let target: u64 = 120;
    let mut src = V4l2Src::new(dev).with_size(640, 480).with_fps(30).with_frame_limit(target);
    let mut conv = VideoConvert::new(RawVideoFormat::Nv12);
    let mut sink = WaylandSink::new().with_title("glass2glass v4l2 capture");
    let clock = ZeroClock;

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        run_source_transform_sink(
            &mut src,
            &mut conv,
            &mut sink,
            &clock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("capture should finish within 30s")
    .expect("v4l2 -> wayland pipeline should succeed");

    eprintln!("emitted={} presented={}", stats.frames_emitted, sink.frames_presented());
    assert!(stats.frames_emitted > 0, "no frames captured");
}
