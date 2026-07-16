//! M18 item 4 follow-up — `run_linear_chain` folds clock + latency from its
//! `dyn` interior elements.
//!
//! Before, only the statically-typed source and sink contributed to
//! `RunStats.latency` / `clock_priority`; the erased interior transforms were
//! skipped because `DynAsyncElement` didn't expose `latency` / `provide_clock`.
//! A buffering interior element (jitter buffer, reorder queue) under-reported
//! pipeline latency, and an interior clock provider was ignored. The dyn-safe
//! mirrors close that gap.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::Arc;

use g2g_core::clock::{ClockCandidate, ClockPriority};
use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::run_linear_chain;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, LatencyReport, OutputSink,
    PipelineClock, PipelinePacket,
};

use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Pass-through interior element that both buffers (5 ms..10 ms of latency)
/// and offers a `Provider`-priority clock, so a single chain proves both
/// contributions are folded.
struct BufferingClockTransform;

impl AsyncElement for BufferingClockTransform {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn latency(&self) -> LatencyReport {
        LatencyReport::buffered(5_000_000, Some(10_000_000))
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(ClockPriority::Provider, Arc::new(ZeroClock)))
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

/// `VideoTestSrc -> BufferingClockTransform -> FakeSink`: source and sink both
/// contribute zero latency and no clock (defaults), so the aggregate latency
/// and elected clock priority come entirely from the interior element.
#[tokio::test]
async fn interior_dyn_element_contributes_latency_and_clock() {
    let mut src = VideoTestSrc::new(32, 32, 30, 4);
    let mut mid = BufferingClockTransform;
    let mut sink = FakeSink::new();
    let clock = ZeroClock;

    let transforms: Vec<&mut dyn DynAsyncElement> = std::vec![&mut mid];
    let stats = run_linear_chain(&mut src, transforms, &mut sink, &clock, 4)
        .await
        .expect("chain runs");

    assert_eq!(stats.latency.min_ns, 5_000_000, "interior buffering folds into path min");
    assert_eq!(stats.latency.max_ns, Some(10_000_000), "and into path max");
    assert!(!stats.latency.live, "no live element on the path");
    assert_eq!(
        stats.clock_priority,
        ClockPriority::Provider,
        "the interior element's clock won election over the system fallback"
    );
}
