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
use g2g_core::RawVideoFormat;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::v4l2src::V4l2Src;
use g2g_plugins::videoconvert::VideoConvert;

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
    let mut src = V4l2Src::new(dev)
        .with_size(640, 480)
        .with_fps(30)
        .with_frame_limit(target);
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
    assert_eq!(
        stats.frames_emitted, target,
        "source should emit the requested frame count"
    );
    assert!(sink.received() > 0, "sink received no converted frames");
    // The convert step turns YUYV (w*h*2) into NV12 (w*h*3/2); reaching the
    // sink at all proves the YUYV unpack negotiated and ran on real data.
    assert_eq!(
        sink.last_sequence(),
        Some(target - 1),
        "frames arrive in order"
    );
}

/// Sink that records the PTS of the first and last frame plus the wall-clock
/// instants they arrived, so a capture's media timeline can be compared with
/// the time it really took.
#[derive(Default)]
struct PtsSpanSink {
    first: Option<(u64, std::time::Instant)>,
    last: Option<(u64, std::time::Instant)>,
    frames: u64,
}

impl g2g_core::element::OutputSink for PtsSpanSink {
    fn push<'a>(
        &'a mut self,
        packet: g2g_core::frame::PipelinePacket,
    ) -> g2g_core::element::BoxFuture<'a, Result<g2g_core::element::PushOutcome, g2g_core::G2gError>>
    {
        Box::pin(async move {
            if let g2g_core::frame::PipelinePacket::DataFrame(f) = &packet {
                let now = (f.timing.pts_ns, std::time::Instant::now());
                self.first.get_or_insert(now);
                self.last = Some(now);
                self.frames += 1;
            }
            Ok(g2g_core::element::PushOutcome::Accepted)
        })
    }
}

/// The media timeline must track the wall clock: the PTS span of a capture has
/// to match the time that capture actually took. The source stamps PTS from the
/// driver's buffer timestamps, so this holds whatever rate the camera ends up
/// running at, including a rate it was never asked for. Requesting 5 fps from a
/// UVC cam that ignores it and free-runs is the sharpest version: a PTS
/// synthesized from the *request* would stretch a two-second capture into a
/// twelve-second timeline, and the recording would play back in slow motion.
#[tokio::test]
#[ignore = "needs a real /dev/videoN device (set G2G_V4L2_DEVICE)"]
async fn v4l2_pts_span_tracks_wall_clock() {
    use g2g_core::runtime::SourceLoop as _;

    let target: u64 = 60;
    let mut src = V4l2Src::new(device())
        .with_size(640, 480)
        .with_fps(5)
        .with_frame_limit(target);
    let caps = src.intercept_caps().await.expect("negotiate");
    src.configure_pipeline(&caps).expect("configure");
    let mut sink = PtsSpanSink::default();
    tokio::time::timeout(std::time::Duration::from_secs(60), src.run(&mut sink))
        .await
        .expect("capture finishes within 60s")
        .expect("capture succeeds");

    assert_eq!(sink.frames, target, "captured the requested frame count");
    let (first_pts, first_at) = sink.first.expect("a first frame");
    let (last_pts, last_at) = sink.last.expect("a last frame");
    assert_eq!(first_pts, 0, "the timeline starts at zero");
    let pts_span = (last_pts - first_pts) as f64 / 1e9;
    let wall_span = (last_at - first_at).as_secs_f64();
    eprintln!(
        "pts span {pts_span:.2}s over {wall_span:.2}s of wall clock ({:.1} fps measured)",
        (target - 1) as f64 / wall_span
    );
    // Generous: the arrival instants trail the driver timestamps by a variable
    // amount, so only a timeline built from the wrong clock (off by the ratio
    // of requested to actual rate) is caught.
    assert!(
        (pts_span - wall_span).abs() < wall_span * 0.25 + 0.1,
        "media timeline ({pts_span:.2}s) does not track the wall clock ({wall_span:.2}s)"
    );
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
    let mut src = V4l2Src::new(dev)
        .with_size(640, 480)
        .with_fps(30)
        .with_frame_limit(target);
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

    eprintln!(
        "emitted={} presented={}",
        stats.frames_emitted,
        sink.frames_presented()
    );
    assert!(stats.frames_emitted > 0, "no frames captured");
}
