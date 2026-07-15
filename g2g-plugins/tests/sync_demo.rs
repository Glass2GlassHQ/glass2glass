//! "Test source + sync" milestone (M3): verify the SyncSink genuinely
//! paces presentation to PTS, and that backpressure makes the upstream
//! VideoTestSrc track the sink's rate without any explicit pacing.

use std::time::{Duration, Instant};

use g2g_core::runtime::run_simple_pipeline;
use g2g_plugins::clock::WallClock;
use g2g_plugins::syncsink::SyncSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

const FRAMES: u64 = 5;
const FPS: u32 = 100;
/// 1 frame at 100 fps = 10 ms.
const FRAME_INTERVAL_MS: u64 = 1000 / FPS as u64;
/// Last PTS = (FRAMES - 1) * interval. With FRAMES=5, FPS=100 -> 40 ms.
const EXPECTED_MIN_MS: u64 = (FRAMES - 1) * FRAME_INTERVAL_MS - 5;
/// Generous upper bound for slow CI / scheduler jitter.
const EXPECTED_MAX_MS: u64 = (FRAMES - 1) * FRAME_INTERVAL_MS + 200;

#[tokio::test]
async fn sync_sink_paces_presentation_to_pts() {
    let clock = WallClock::new();
    let mut src = VideoTestSrc::new(16, 16, FPS, FRAMES);
    let mut snk = SyncSink::new(clock);

    let t0 = Instant::now();
    let stats = run_simple_pipeline(&mut src, &mut snk, &clock, 8)
        .await
        .expect("sync pipeline should complete");
    let elapsed = t0.elapsed();

    assert_eq!(stats.frames_consumed, FRAMES);
    assert_eq!(snk.last_sequence(), Some(FRAMES - 1));
    assert!(snk.eos_seen());

    let elapsed_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
    assert!(
        elapsed_ms >= EXPECTED_MIN_MS,
        "pipeline finished too fast ({elapsed_ms}ms < {EXPECTED_MIN_MS}ms): sink is not actually waiting for PTS"
    );
    assert!(
        elapsed_ms <= EXPECTED_MAX_MS,
        "pipeline took too long ({elapsed_ms}ms > {EXPECTED_MAX_MS}ms): excess scheduler overhead"
    );

    // Drift = how late we presented each frame vs. its PTS. Should be small
    // (single-digit ms per frame on a quiet machine). Allow 50ms for CI noise.
    assert!(
        snk.max_drift_ns() < Duration::from_millis(50).as_nanos() as u64,
        "max drift {}ns exceeds 50ms",
        snk.max_drift_ns()
    );
}
