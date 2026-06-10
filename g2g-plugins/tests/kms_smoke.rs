//! End-to-end smoke test for the KMS display sink.
//!
//! Pipeline: RtspSrc -> FfmpegH264Dec(Nv12) -> KmsSink.
//!
//! Ignored by default because it needs all of:
//! - Linux running outside a Wayland/X11 session (drop to a tty with
//!   `Ctrl+Alt+F3`, or run with a DRM lease). A compositor will hold
//!   DRM master and `set_crtc` returns `PermissionDenied`.
//! - Access to `/dev/dri/card0` (group `video` on most distros).
//! - An RTSP feed at `G2G_RTSP_TEST_URL` (or the rtsp.stream default).
//!
//! Run with:
//!
//! ```sh
//! G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
//!     cargo test -p g2g-plugins --features "rtsp ffmpeg kms-sink" \
//!     --test kms_smoke -- --ignored --nocapture
//! ```
//!
//! See the MediaMTX + ffmpeg recipe in `rtsp_ffmpeg_e2e.rs` for a local
//! deterministic source. **Important:** the rtsp publisher's resolution
//! must match (or be smaller than) your display's active mode — v1 does
//! no scaling. The recipe's `testsrc=size=640x480:rate=30` is fine on any
//! desktop, but the visible result will be 640x480 at the top-left of
//! the screen with stale framebuffer around it.

#![cfg(all(
    target_os = "linux",
    feature = "rtsp",
    feature = "ffmpeg",
    feature = "kms-sink"
))]

use g2g_core::runtime::run_source_transform_sink;
use g2g_core::PipelineClock;
use g2g_plugins::ffmpegdec::{FfmpegH264Dec, OutputFormat};
use g2g_plugins::kmssink::KmsSink;
use g2g_plugins::rtspsrc::RtspSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
#[ignore = "needs tty/lease DRM access + an RTSP feed (set G2G_RTSP_TEST_URL)"]
async fn kms_sink_displays_rtsp_h264() {
    let url = std::env::var("G2G_RTSP_TEST_URL")
        .unwrap_or_else(|_| "rtsp://localhost:8554/pattern".to_string());
    eprintln!("connecting to {url}");

    // 60 frames @ 30fps = 2 seconds on screen. Enough to verify the
    // image arrives and the vsync wait isn't stalling.
    const TARGET: u64 = 60;

    let mut src = RtspSrc::new(url).with_frame_limit(TARGET);
    let mut dec = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
    let mut snk = KmsSink::new();
    let clock = ZeroClock;

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        run_source_transform_sink(&mut src, &mut dec, &mut snk, &clock, 8),
    )
    .await
    .expect("pipeline should complete within 60s")
    .expect("end-to-end KMS pipeline should succeed");

    eprintln!(
        "stats: source emitted={} decoded={} presented={}",
        stats.frames_emitted,
        dec.decoded_count(),
        snk.frames_presented(),
    );

    assert!(dec.decoded_count() > 0, "decoder produced no NV12 frames");
    assert!(snk.frames_presented() > 0, "KMS sink presented no frames");
}
