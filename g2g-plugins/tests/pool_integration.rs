//! M4: 100 frames flow through a 4-buffer pool. The fakesink drops each
//! frame immediately, returning its buffer to the pool, so the pool must
//! never be exhausted and end at full capacity.

use g2g_core::runtime::run_simple_pipeline;
use g2g_core::{BufferPool, PipelineClock};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

const FRAME_BYTES: usize = 16 * 16 * 4;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn pool_recycles_through_100_frames_without_exhaustion() {
    let pool = BufferPool::new_byte_pool(4, FRAME_BYTES);
    assert_eq!(pool.available(), 4);

    let mut src = VideoTestSrc::with_pool(16, 16, 30, 100, pool.clone());
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut snk, &clock, 8)
        .await
        .expect("pool-backed pipeline should complete");

    assert_eq!(stats.frames_consumed, 100);
    assert_eq!(snk.last_sequence(), Some(99));
    assert!(snk.eos_seen());

    // All buffers must be back in the pool after the pipeline tears down.
    assert_eq!(
        pool.available(),
        4,
        "pool leaked buffers: {} of {} returned",
        pool.available(),
        pool.capacity()
    );
    assert_eq!(pool.outstanding(), 0);
}

#[tokio::test]
async fn pool_visible_outstanding_during_run_is_bounded() {
    // Channel capacity 2 + sink processing one frame + source holding one
    // buffer should never exceed pool capacity 4. We can't sample mid-run
    // from outside without instrumentation, so this test is a sanity check
    // that the pipeline completes with capacity = 4 and channel = 2.
    let pool = BufferPool::new_byte_pool(4, FRAME_BYTES);
    let mut src = VideoTestSrc::with_pool(16, 16, 30, 50, pool.clone());
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut snk, &clock, 2)
        .await
        .expect("tight pool + tight link should drain via backpressure");

    assert_eq!(stats.frames_consumed, 50);
    assert_eq!(pool.available(), 4);
}
