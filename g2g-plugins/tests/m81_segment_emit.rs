//! M81 - the runner emits the opening SEGMENT, and prefers a substantive arm
//! error over a secondary `Shutdown`.
//!
//! `run_simple_pipeline` / `run_graph` now open every stream with an
//! open-ended, normal-rate SEGMENT ahead of the source's data, so a sink maps
//! frame timestamps to running time from the first frame. Re-landing that emit
//! needed the error-priority fix: an extra packet can make the source block on
//! a full link, so when a downstream element returns a real error the source's
//! pending push fails with `Shutdown`; the runner must report the real error,
//! not that secondary `Shutdown`.

use g2g_core::element::{AsyncElement, BoxFuture, ConfigureOutcome, OutputSink};
use g2g_core::runtime::run_simple_pipeline;
use g2g_core::{Caps, G2gError, PipelineClock, PipelinePacket, Segment};

use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// `run_simple_pipeline` opens the stream with an open-ended, normal-rate
/// SEGMENT that reaches the sink ahead of the frames.
#[tokio::test]
async fn run_simple_emits_opening_segment() {
    let target = 4u64;
    let mut src = VideoTestSrc::new(32, 32, 30, target);
    let mut sink = FakeSink::new();

    let stats = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 4)
        .await
        .expect("pipeline runs");

    assert_eq!(stats.frames_consumed, target);
    assert_eq!(sink.segments(), 1, "exactly one opening SEGMENT");
    assert_eq!(sink.last_segment(), Some(Segment::new()));
    // running time is computable: open segment from 0 at rate 1.
    assert_eq!(
        sink.last_segment().unwrap().to_running_time(5_000),
        Some(5_000)
    );
}

/// Sink that errors on its first `DataFrame`. It drops its link end, so the
/// source's next (blocked) push fails with `Shutdown` — the runner must report
/// this sink's real error instead.
struct FailingSink;

impl AsyncElement for FailingSink {
    type ProcessFuture<'a> = BoxFuture<'a, Result<(), G2gError>>;

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
                PipelinePacket::DataFrame(_) => Err(G2gError::CapsMismatch),
                _ => Ok(()),
            }
        })
    }
}

/// A real sink error is reported even though the source's blocked push then
/// fails with the secondary `Shutdown` (capacity 1 forces the source to block).
#[tokio::test]
async fn substantive_error_preferred_over_shutdown() {
    let mut src = VideoTestSrc::new(16, 16, 30, 8);
    let mut sink = FailingSink;

    let result = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 1).await;

    assert_eq!(
        result,
        Err(G2gError::CapsMismatch),
        "the sink's real error must surface, not the source's secondary Shutdown"
    );
}
