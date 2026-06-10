//! End-to-end smoke test for the Wayland display sink.
//!
//! Pipeline: RtspSrc -> FfmpegH264Dec(Nv12) -> WaylandSink.
//!
//! Ignored by default. Requires:
//! - A running Wayland session (`WAYLAND_DISPLAY` set in the environment).
//! - An RTSP feed at `G2G_RTSP_TEST_URL`, or the rtsp.stream default.
//!
//! Unlike `kms_smoke` this test runs *inside* your normal desktop session,
//! so the easiest setup is to leave the MediaMTX + ffmpeg recipe from
//! `rtsp_ffmpeg_e2e.rs` running in two terminals, then:
//!
//! ```sh
//! G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
//!     cargo test -p g2g-plugins \
//!     --features "rtsp ffmpeg wayland-sink" \
//!     --test wayland_smoke -- --ignored --nocapture
//! ```
//!
//! A window titled "glass2glass" should appear on the active output
//! showing the test pattern for ~2 seconds.

#![cfg(all(
    target_os = "linux",
    feature = "rtsp",
    feature = "ffmpeg",
    feature = "wayland-sink"
))]

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::PipelineClock;
use g2g_plugins::ffmpegdec::{FfmpegH264Dec, OutputFormat};
use g2g_plugins::rtspsrc::RtspSrc;
use g2g_plugins::waylandsink::WaylandSink;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
#[ignore = "needs a Wayland session + an RTSP feed (set G2G_RTSP_TEST_URL)"]
async fn wayland_sink_shows_rtsp_h264_in_a_window() {
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping: no WAYLAND_DISPLAY in env (run under a Wayland session)");
        return;
    }

    let url = std::env::var("G2G_RTSP_TEST_URL")
        .unwrap_or_else(|_| "rtsp://localhost:8554/pattern".to_string());
    eprintln!("connecting to {url}");

    const TARGET: u64 = 60;

    let mut src = RtspSrc::new(url).with_frame_limit(TARGET);
    let mut dec = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
    let mut snk = WaylandSink::new().with_title("glass2glass smoke test");
    let clock = ZeroClock;

    let start = std::time::Instant::now();
    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        run_source_transform_sink(&mut src, &mut dec, &mut snk, &clock, 8),
    )
    .await
    .expect("pipeline should complete within 60s")
    .expect("end-to-end Wayland pipeline should succeed");
    let elapsed = start.elapsed();

    let fps = stats.frames_emitted as f64 / elapsed.as_secs_f64();
    eprintln!(
        "stats: emitted={} decoded={} presented={} elapsed={:.2}s effective_fps={:.1}",
        stats.frames_emitted,
        dec.decoded_count(),
        snk.frames_presented(),
        elapsed.as_secs_f64(),
        fps,
    );

    assert!(dec.decoded_count() > 0, "decoder produced no NV12 frames");
    // We don't assert `presented > 0` because the compositor's frame-
    // callback cadence and the EOS-driven shutdown can race: the last
    // few frames in flight may not paint before the worker exits.

    // Pacing assertions: with compositor frame-callback gating, the
    // producer is throttled to refresh. On a 60 Hz output, 60 frames
    // should take ~1s. If pacing regresses (process() returns without
    // waiting for the frame callback), fps will be hundreds — the
    // decoder runs faster than display refresh by an order of magnitude.
    // The lower bound catches the opposite regression (stall on a
    // never-arriving callback) that the outer timeout would otherwise
    // mask as "passed, just slow".
    assert!(
        fps < 200.0,
        "pacing regression: {fps:.1} fps suggests process() is not waiting on the frame callback"
    );
    assert!(
        fps > 10.0,
        "pacing stall: only {fps:.1} fps — frame callback may not be firing"
    );
}
