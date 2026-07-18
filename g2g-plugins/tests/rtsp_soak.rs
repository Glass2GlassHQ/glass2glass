//! Long-running stability soak for the RTSP + ffmpeg decode path.
//!
//! Pulls H.264 from a local RTSP server, decodes through `FfmpegH264Dec`,
//! and runs for N seconds (default 30) with reconnect enabled. Asserts:
//!
//! - At least `expected_min_frames` reach the sink — proves the pipeline
//!   keeps producing across the soak window, not just the first second.
//! - Frame `sequence` is monotonically increasing — no duplicates, no
//!   out-of-order, no overflows.
//! - Frame `pts_ns` is monotonically non-decreasing — the per-session
//!   PTS continuation logic (and the reconnect-gap insertion) didn't
//!   regress.
//! - No `Err` from the runner.
//!
//! Ignored by default — it needs an RTSP feed (the MediaMTX + ffmpeg
//! recipe from `rtsp_ffmpeg_e2e.rs` is the canonical local fixture).
//!
//! Run with:
//!
//! ```sh
//! G2G_RTSP_TEST_URL=rtsp://localhost:8554/pattern \
//!     cargo test -p g2g-plugins --features "rtsp ffmpeg" \
//!     --test rtsp_soak -- --ignored --nocapture
//! ```
//!
//! Override duration via `G2G_SOAK_SECONDS` (defaults to 30). Override
//! the expected frame floor via `G2G_SOAK_MIN_FRAMES` (defaults to
//! `seconds * 20`, ie ~20 fps min — well under MediaMTX's 30 fps so the
//! test tolerates a slow CI host).
//!
//! ## Manual reconnect exercise
//!
//! To actually exercise the reconnect path, leave this test running and
//! `Ctrl-C` the ffmpeg publisher process in another terminal. The source
//! should log `rtsp: session ended (...) ; reconnect 1/N after ...ms`,
//! sleep, and resume once you restart the publisher. PTS should stay
//! monotonic across the gap (the soak asserts this).

#![cfg(all(target_os = "linux", feature = "rtsp", feature = "ffmpeg"))]

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink};
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::run_source_transform_sink;
use g2g_core::{Caps, ConfigureOutcome, G2gError, PipelineClock};
use g2g_plugins::ffmpegdec::FfmpegH264Dec;
use g2g_plugins::rtspsrc::RtspSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Asserting sink: panics on the first monotonicity violation it sees,
/// counts frames into a shared atomic so the test thread can read it.
/// Implements `AsyncElement` directly (rather than wrapping `FakeSink`)
/// because `FakeSink`'s `sequence` check already aborts on regressions,
/// but we want richer PTS assertions and richer diagnostics.
struct AssertingSink {
    count: Arc<AtomicU64>,
    last_seq: Option<u64>,
    last_pts: Option<u64>,
}

impl AssertingSink {
    fn new(count: Arc<AtomicU64>) -> Self {
        Self {
            count,
            last_seq: None,
            last_pts: None,
        }
    }
}

impl AsyncElement for AssertingSink {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(prev) = self.last_seq {
                        assert!(
                            f.sequence > prev,
                            "sequence regression: prev={prev}, now={}",
                            f.sequence
                        );
                    }
                    if let Some(prev) = self.last_pts {
                        assert!(
                            f.timing.pts_ns >= prev,
                            "pts regression: prev={prev}, now={}",
                            f.timing.pts_ns
                        );
                    }
                    self.last_seq = Some(f.sequence);
                    self.last_pts = Some(f.timing.pts_ns);
                    self.count.fetch_add(1, Ordering::Relaxed);
                }
                PipelinePacket::Flush => {
                    // Reconnect-discontinuity boundary. sequence keeps
                    // counting (so the prev>now check still holds), but
                    // PTS just jumped forward by 1s+ which is fine for
                    // the monotonic-non-decreasing check.
                }
                PipelinePacket::CapsChanged(_)
                | PipelinePacket::Segment(_)
                | PipelinePacket::Eos => {}
                _ => {}
            }
            Ok(())
        })
    }
}

#[tokio::test]
#[ignore = "long-running stability soak; needs an RTSP feed (set G2G_RTSP_TEST_URL)"]
async fn rtsp_ffmpeg_soak_keeps_pts_monotonic_for_duration() {
    let url = std::env::var("G2G_RTSP_TEST_URL")
        .unwrap_or_else(|_| "rtsp://localhost:8554/pattern".to_string());
    let seconds: u64 = std::env::var("G2G_SOAK_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let min_frames: u64 = std::env::var("G2G_SOAK_MIN_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(seconds * 20);

    eprintln!("soak target: {url} for {seconds}s, expecting at least {min_frames} decoded frames");

    // Frame budget: enough to keep going through `seconds`. At ~30 fps,
    // 30 * seconds is the ideal; we cap higher so the source doesn't
    // self-terminate before the test deadline does.
    let frame_budget = seconds.saturating_mul(60).max(min_frames * 2);

    let mut src = RtspSrc::new(url)
        .with_frame_limit(frame_budget)
        .with_reconnect(20)
        .with_reconnect_backoff(100, 2_000);
    let mut dec = FfmpegH264Dec::new();
    let count = Arc::new(AtomicU64::new(0));
    let mut snk = AssertingSink::new(Arc::clone(&count));
    let clock = ZeroClock;

    // Outer wall-clock deadline. We size it slightly above the soak
    // budget so we observe a clean pipeline shutdown via frame_limit
    // rather than a timeout panic.
    let deadline = Duration::from_secs(seconds + 10);

    let result = tokio::time::timeout(
        deadline,
        run_source_transform_sink(&mut src, &mut dec, &mut snk, &clock, 2),
    )
    .await;

    let final_count = count.load(Ordering::Relaxed);
    eprintln!("soak finished: {final_count} frames reached sink");

    match result {
        Ok(Ok(_stats)) => {}
        Ok(Err(e)) => panic!("pipeline errored mid-soak: {e:?}"),
        Err(_) => {
            // Hitting the wall-clock deadline is acceptable as long as
            // we got enough frames. Useful when the publisher rate is
            // slow and the frame_limit doesn't trip in time.
            eprintln!("soak hit wall-clock deadline (expected for slow publishers)");
        }
    }

    assert!(
        final_count >= min_frames,
        "soak underdelivered: {final_count} < {min_frames}",
    );
}
