use g2g_core::runtime::run_simple_pipeline;
use g2g_core::PipelineClock;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn videotestsrc_to_fakesink_30_frames_round_trip() {
    let mut src = VideoTestSrc::new(64, 64, 30, 30);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut snk, &clock, 32)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.frames_emitted, 30, "source must report 30 emitted");
    assert_eq!(stats.frames_consumed, 30, "sink must report 30 consumed");
    assert_eq!(snk.received(), 30, "sink internal counter must match");
    assert_eq!(snk.last_sequence(), Some(29), "sequence must reach 29");
    assert!(snk.eos_seen(), "sink must observe EOS");
}

#[tokio::test]
async fn small_pipeline_with_eos_marker() {
    // Capacity must accommodate `target_frames + 1` (EOS occupies one slot).
    // M1 source uses sync `OutputSink::push`; backpressure-aware send lands in M2/M3.
    let mut src = VideoTestSrc::new(16, 16, 60, 8);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let stats = run_simple_pipeline(&mut src, &mut snk, &clock, 9)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.frames_consumed, 8);
    assert!(snk.eos_seen());
}

#[tokio::test]
async fn pipeline_undersized_link_fails_fast() {
    // Documenting the M1 limitation: undersized link surfaces as PoolExhausted
    // rather than blocking. Fix lands with async OutputSink in M2/M3.
    let mut src = VideoTestSrc::new(16, 16, 60, 8);
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let err = run_simple_pipeline(&mut src, &mut snk, &clock, 4)
        .await
        .expect_err("undersized link should not block");

    assert!(matches!(err, g2g_core::G2gError::PoolExhausted));
}
