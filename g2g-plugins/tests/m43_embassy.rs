//! M43: the no_std core runs under an Embassy executor primitive. Drives a real
//! `VideoTestSrc -> FakeSink` pipeline to completion with
//! `embassy_futures::block_on` (the future runner an embedded app uses on bare
//! metal), proving the executor-agnostic runner works off the Embassy executor,
//! not just tokio.

use g2g_core::runtime::run_simple_pipeline;
use g2g_core::PipelineClock;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

/// Trivial clock so the run needs no time driver (the executor path is what is
/// under test here, not timing). `EmbassyClock` carries the embassy-time path.
struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[test]
fn pipeline_runs_to_eos_under_embassy_block_on() {
    let mut src = VideoTestSrc::new(16, 8, 30, 5);
    let mut sink = FakeSink::new();

    embassy_futures::block_on(run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 4))
        .expect("pipeline completes under embassy block_on");

    assert!(sink.eos_seen(), "EOS must reach the sink");
    assert_eq!(sink.received(), 5, "all frames delivered under the embassy executor");
}
