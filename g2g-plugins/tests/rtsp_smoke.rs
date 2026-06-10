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
fn rtspsrc_intercept_caps_advertises_fixate_friendly_h264_range() {
    // intercept_caps runs during Phase 1 before we touch the network. The
    // real dims live in the SDP and only land inside `run`, so we advertise
    // a wide `Range` (not `Any`, which `Caps::fixate()` rejects) and rely
    // on the SDP-derived `CapsChanged` to refine to a fixed geometry.
    use g2g_core::runtime::SourceLoop as _;
    let src = RtspSrc::new("rtsp://invalid.example/never-reached");
    let caps = src.intercept_caps().expect("intercept_caps should succeed");
    // Critical contract: the advertised caps must be fixable, else the
    // runner aborts with `CapsMismatch` before any frames flow.
    caps.fixate().expect("RtspSrc caps must be fixate-friendly");
    match &caps {
        Caps::Video {
            format,
            width,
            height,
            framerate,
        } => {
            assert_eq!(*format, VideoFormat::H264);
            assert!(matches!(width, Dim::Range { .. }));
            assert!(matches!(height, Dim::Range { .. }));
            assert!(matches!(framerate, Rate::Range { .. }));
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

    // M7: RtspSrc must emit a refined CapsChanged before the first frame
    // (assuming the server's SDP carries an in-band SPS, which it should
    // for any conformant H.264 RTSP source). The first caps change must
    // arrive ahead of any DataFrame (`frames_before == 0`) and carry
    // fixed pixel dimensions.
    let changes = snk.caps_changes();
    assert!(!changes.is_empty(), "RtspSrc must emit at least one CapsChanged");
    let first = &changes[0];
    assert_eq!(first.frames_before, 0, "CapsChanged must precede first frame");
    match &first.caps {
        Caps::Video { format, width, height, .. } => {
            assert_eq!(*format, VideoFormat::H264);
            assert!(matches!(width, Dim::Fixed(_)), "width should be Fixed, got {width:?}");
            assert!(matches!(height, Dim::Fixed(_)), "height should be Fixed, got {height:?}");
        }
        other => panic!("expected Caps::Video, got {other:?}"),
    }
}
