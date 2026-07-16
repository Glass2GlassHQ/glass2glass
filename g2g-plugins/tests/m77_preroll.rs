//! M77 - preroll + live/non-live state changes, over the stateful runner.
//!
//! Extends M76's flow gate with GStreamer-style preroll: a *non-live* pipeline
//! in `Paused` admits exactly one buffer (the preroll frame) and holds it, the
//! `set_state(Paused)` reports `Async`, and the change completes (bus
//! `AsyncDone`, `await_prerolled()` resolves) once that frame lands. A *live*
//! pipeline produces no preroll buffer, reports `NoPreroll`, and full-holds.
//! The gate internals are unit-tested in `g2g-core`; here we drive the wiring
//! end-to-end on tokio.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::element::{AsyncElement, BoxFuture, ConfigureOutcome, OutputSink};
use g2g_core::runtime::{run_simple_pipeline_stateful, StateController};
use g2g_core::{
    Bus, BusMessage, Caps, G2gError, PipelineClock, PipelinePacket, PipelineState,
    StateChangeReturn,
};

use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Sink that bumps a shared counter on every `DataFrame`, readable mid-run.
struct CountingSink {
    seen: Arc<AtomicU64>,
}

impl AsyncElement for CountingSink {
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
            if let PipelinePacket::DataFrame(_) = packet {
                self.seen.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })
    }
}

/// Non-live `Paused` prerolls exactly one frame and holds; `await_prerolled`
/// resolves and `AsyncDone` is posted once it lands; `Playing` drains the rest.
#[tokio::test]
async fn nonlive_paused_prerolls_one_frame_then_plays() {
    let target = 5u64;
    let mut src = VideoTestSrc::new(64, 64, 30, target);
    let seen = Arc::new(AtomicU64::new(0));
    let mut sink = CountingSink { seen: seen.clone() };
    let clock = ZeroClock;

    let (bus, handle) = Bus::new(16);
    let ctrl = StateController::with_bus(PipelineState::Ready, handle); // non-live default

    // capacity 4 so the preroll hold (1 frame) is well inside the link depth.
    let pipeline = run_simple_pipeline_stateful(&mut src, &mut sink, &clock, 4, &ctrl);

    let seen_for_driver = seen.clone();
    let ctrl_for_driver = ctrl.clone();
    let driver = async move {
        // Non-live Paused is async: the change completes when the sink prerolls.
        assert_eq!(
            ctrl_for_driver.set_state(PipelineState::Paused),
            StateChangeReturn::Async
        );
        // Await preroll rather than spin a fixed yield count: deterministic.
        ctrl_for_driver.await_prerolled().await;
        assert!(ctrl_for_driver.is_prerolled());
        // Exactly the one preroll frame crossed; the gate now holds the rest.
        assert_eq!(
            seen_for_driver.load(Ordering::SeqCst),
            1,
            "exactly one preroll buffer crosses in non-live Paused"
        );
        // Give the held pipeline room to (not) advance, then confirm it held.
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            seen_for_driver.load(Ordering::SeqCst),
            1,
            "the gate holds after the preroll frame"
        );
        ctrl_for_driver.set_state(PipelineState::Playing);
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    let stats = res.expect("pipeline runs to completion");
    assert_eq!(
        stats.frames_consumed, target,
        "all frames drained once Playing"
    );
    assert_eq!(seen.load(Ordering::SeqCst), target);

    // AsyncDone is posted once, when preroll completes.
    let async_dones = std::iter::from_fn(|| bus.try_recv())
        .filter(|m| matches!(m, BusMessage::AsyncDone))
        .count();
    assert_eq!(async_dones, 1, "exactly one AsyncDone for the preroll");
}

/// A live pipeline reports `NoPreroll` for `Paused` and admits no buffer until
/// `Playing` (full hold, no preroll frame).
#[tokio::test]
async fn live_paused_reports_no_preroll_and_full_holds() {
    let target = 4u64;
    let mut src = VideoTestSrc::new(32, 32, 30, target);
    let seen = Arc::new(AtomicU64::new(0));
    let mut sink = CountingSink { seen: seen.clone() };
    let clock = ZeroClock;
    let ctrl = StateController::new(PipelineState::Ready);
    ctrl.set_live(true);

    let pipeline = run_simple_pipeline_stateful(&mut src, &mut sink, &clock, 4, &ctrl);

    let seen_for_driver = seen.clone();
    let ctrl_for_driver = ctrl.clone();
    let driver = async move {
        assert_eq!(
            ctrl_for_driver.set_state(PipelineState::Paused),
            StateChangeReturn::NoPreroll
        );
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            seen_for_driver.load(Ordering::SeqCst),
            0,
            "live Paused admits no preroll buffer"
        );
        ctrl_for_driver.set_state(PipelineState::Playing);
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    let stats = res.expect("pipeline runs");
    assert_eq!(stats.frames_consumed, target);
    assert_eq!(seen.load(Ordering::SeqCst), target);
}
