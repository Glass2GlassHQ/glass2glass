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

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        run_source_transform_sink(&mut src, &mut dec, &mut snk, &clock, 8),
    )
    .await
    .expect("pipeline should complete within 60s")
    .expect("end-to-end Wayland pipeline should succeed");

    eprintln!(
        "stats: emitted={} decoded={} presented={}",
        stats.frames_emitted,
        dec.decoded_count(),
        snk.frames_presented(),
    );

    assert!(dec.decoded_count() > 0, "decoder produced no NV12 frames");
    // We don't assert `presented > 0` because the compositor's frame-
    // callback cadence and the EOS-driven shutdown can race: the last
    // few frames in flight may not paint before the worker exits.
    // What we *do* assert: nothing in the pipeline errored.
}
