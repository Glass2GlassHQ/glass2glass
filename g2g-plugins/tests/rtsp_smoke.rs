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
use g2g_core::{Caps, Dim, G2gError, PipelineClock, VideoCodec, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::rtspsrc::RtspSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn rtspsrc_intercept_caps_with_expected_dims_skips_probe_and_fixates() {
    // M18 item 5: `intercept_caps` is async and probes the server by
    // default. `with_expected_dims` is the offline fast path: caps come
    // from the caller, no network touch. The advertised caps must still
    // be fixable so Phase 2 doesn't abort with `CapsMismatch`.
    use g2g_core::runtime::SourceLoop as _;
    let mut src = RtspSrc::new("rtsp://invalid.example/never-reached")
        .with_expected_dims(1920, 1080);
    let caps = src.intercept_caps().await.expect("intercept_caps");
    caps.fixate().expect("expected-dims caps must be fixate-friendly");
    match &caps {
        Caps::CompressedVideo {
            codec,
            width,
            height,
            ..
        } => {
            assert_eq!(*codec, VideoCodec::H264);
            assert_eq!(*width, Dim::Fixed(1920));
            assert_eq!(*height, Dim::Fixed(1080));
        }
        other => panic!("expected Caps::CompressedVideo, got {other:?}"),
    }
}

#[tokio::test]
async fn rtspsrc_intercept_caps_probes_and_fails_on_unreachable_url() {
    // Without `with_expected_dims`, intercept_caps attempts DESCRIBE +
    // SETUP. An unreachable URL must surface a `Hardware` error during
    // negotiation rather than (as before M18 item 5) silently advertising
    // a placeholder Range and exploding inside `run`.
    use g2g_core::runtime::SourceLoop as _;
    let mut src = RtspSrc::new("rtsp://127.0.0.1:1/never-listens");
    let res = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        src.intercept_caps(),
    )
    .await
    .expect("probe must terminate within 5s");
    assert!(matches!(res, Err(G2gError::Hardware(_))), "got {res:?}");
}

#[tokio::test]
async fn rtspsrc_with_reconnect_retries_then_fails() {
    // Unreachable address + reconnect with a short backoff. The source
    // must try max_attempts+1 times (the first connect plus the retries)
    // then surface a Hardware error. The whole loop must complete well
    // under our timeout so a hang regression is visible.
    let url = "rtsp://127.0.0.1:1/never-listens";
    let mut src = RtspSrc::new(url)
        .with_reconnect(3)
        .with_reconnect_backoff(10, 50);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let started = std::time::Instant::now();
    let result =
        tokio::time::timeout(std::time::Duration::from_secs(10),
            run_simple_pipeline(&mut src, &mut snk, &clock, 4),
        )
        .await
        .expect("reconnect retries should complete within 10s");
    let elapsed = started.elapsed();

    // 3 retries with backoff 10, 20, 40 ms = 70 ms minimum of sleeps;
    // realistic network/connect overhead pushes total to ~hundreds of ms.
    // The point of the assertion: we didn't return on the first failure.
    assert!(
        elapsed >= std::time::Duration::from_millis(50),
        "must have slept between retries; actual elapsed = {elapsed:?}",
    );

    let err = result.expect_err("exhausted retries must surface as error");
    assert!(
        matches!(err, G2gError::Hardware(_) | G2gError::CapsMismatch),
        "unexpected error: {err:?}",
    );
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
        Caps::CompressedVideo { codec, width, height, .. } => {
            assert_eq!(*codec, VideoCodec::H264);
            assert!(matches!(width, Dim::Fixed(_)), "width should be Fixed, got {width:?}");
            assert!(matches!(height, Dim::Fixed(_)), "height should be Fixed, got {height:?}");
        }
        other => panic!("expected Caps::CompressedVideo, got {other:?}"),
    }
}
