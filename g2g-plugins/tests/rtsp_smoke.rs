//! M5: smoke tests for the RTSP source element.
//!
//! Construction test runs always (no network). The live-server integration
//! test is `#[ignore]` and only runs with `cargo test -- --ignored`. Override
//! the URL via env var `G2G_RTSP_TEST_URL` when running locally:
//!
//! ```sh
//! G2G_RTSP_TEST_URL=rtsp://my.camera/stream cargo test -p g2g-plugins \
//!     --features rtsp -- --ignored
//! ```

#![cfg(feature = "rtsp")]

use g2g_core::runtime::run_simple_pipeline;
use g2g_core::{Caps, Dim, G2gError, PipelineClock, Rate, VideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::rtspsrc::RtspSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[test]
fn rtspsrc_intercept_caps_returns_h264_with_any_geometry() {
    // intercept_caps is called by the runner during Phase 1 negotiation,
    // before we connect to the server. Until the SPS lands (M6), geometry
    // and framerate are advertised as Any.
    use g2g_core::runtime::SourceLoop as _;
    let src = RtspSrc::new("rtsp://invalid.example/never-reached");
    let caps = src.intercept_caps().expect("intercept_caps should succeed");
    match caps {
        Caps::Video {
            format,
            width,
            height,
            framerate,
        } => {
            assert_eq!(format, VideoFormat::H264);
            assert_eq!(width, Dim::Any);
            assert_eq!(height, Dim::Any);
            assert_eq!(framerate, Rate::Any);
        }
        other => panic!("expected Caps::Video, got {other:?}"),
    }
}

#[tokio::test]
async fn rtspsrc_bad_url_returns_hardware_error_or_caps_mismatch() {
    // Hitting an unreachable address must fail fast rather than hang.
    let url = "rtsp://127.0.0.1:1/no-such-server";
    let mut src = RtspSrc::new(url).with_frame_limit(1);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let result =
        tokio::time::timeout(std::time::Duration::from_secs(10),
            run_simple_pipeline(&mut src, &mut snk, &clock, 4),
        )
        .await
        .expect("connect attempt should not hang");

    let err = result.expect_err("connecting to a nonexistent server must fail");
    assert!(
        matches!(err, G2gError::Hardware(_) | G2gError::CapsMismatch),
        "unexpected error: {err:?}"
    );
}

#[tokio::test]
#[ignore = "requires live RTSP server; set G2G_RTSP_TEST_URL or use the default rtsp.stream feed"]
async fn rtspsrc_pulls_h264_from_live_server() {
    let url = std::env::var("G2G_RTSP_TEST_URL")
        .unwrap_or_else(|_| "rtsp://rtsp.stream/pattern".to_string());

    let mut src = RtspSrc::new(url).with_frame_limit(10);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats =
        tokio::time::timeout(std::time::Duration::from_secs(30),
            run_simple_pipeline(&mut src, &mut snk, &clock, 4),
        )
        .await
        .expect("live pull should complete in 30s")
        .expect("live RTSP pull should succeed");

    assert_eq!(stats.frames_consumed, 10);
    assert_eq!(snk.last_sequence(), Some(9));
    assert!(snk.eos_seen());
}
