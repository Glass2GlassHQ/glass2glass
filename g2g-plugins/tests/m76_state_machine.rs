//! M76 - pipeline state machine, integration over a real runner.
//!
//! Drives `run_simple_pipeline_stateful` through the `NULL → READY → PAUSED →
//! PLAYING` ladder with a `StateController`. Proves the sink-side flow gate:
//! data only crosses the link once the controller reaches `Playing`, the run
//! completes through the transition, and each transition is observable on the
//! bus as a `StateChanged`. The gate's lost-wakeup-free internals are unit-
//! tested in `g2g-core`; here we exercise the wiring end-to-end on tokio.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use g2g_core::element::{AsyncElement, BoxFuture, ConfigureOutcome, OutputSink};
use g2g_core::runtime::{run_simple_pipeline_stateful, StateController};
use g2g_core::{Bus, BusMessage, Caps, G2gError, PipelineClock, PipelinePacket, PipelineState};

use g2g_plugins::videotestsrc::VideoTestSrc;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Sink that bumps a shared counter on every `DataFrame`, so a test can read
/// how many frames have crossed the link *while the run is still in flight*.
struct CountingSink {
    seen: Arc<AtomicU64>,
    eos: Arc<std::sync::atomic::AtomicBool>,
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
            match packet {
                PipelinePacket::DataFrame(_) => {
                    self.seen.fetch_add(1, Ordering::SeqCst);
                }
                PipelinePacket::Eos => {
                    self.eos.store(true, Ordering::SeqCst);
                }
                _ => {}
            }
            Ok(())
        })
    }
}

/// Start `Paused` on a *live* pipeline (full hold, no preroll), let it stall,
/// prove no frames crossed, then `Playing` drains all of them and the run
/// completes through the transition. (The non-live preroll path is in
/// `m77_preroll`.)
#[tokio::test]
async fn paused_gates_flow_until_playing() {
    let target = 5u64;
    let mut src = VideoTestSrc::new(64, 64, 30, target);
    let seen = Arc::new(AtomicU64::new(0));
    let eos = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut sink = CountingSink {
        seen: seen.clone(),
        eos: eos.clone(),
    };
    let clock = ZeroClock;

    let (bus, handle) = Bus::new(16);
    let ctrl = StateController::with_bus(PipelineState::Ready, handle);
    // Live: `Paused` is a full hold (no preroll buffer), so the sink consumes
    // nothing until `Playing`.
    ctrl.set_live(true);

    // capacity 2: in Paused the source can fill at most 2 in-flight before it
    // backpressures, and the sink consumes 0.
    let pipeline = run_simple_pipeline_stateful(&mut src, &mut sink, &clock, 2, &ctrl);

    let seen_for_driver = seen.clone();
    let ctrl_for_driver = ctrl.clone();
    let driver = async move {
        // Move into Paused and let the pipeline poll until it parks: source
        // blocked on a full link, sink parked on the gate.
        ctrl_for_driver.set_state(PipelineState::Paused);
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        // The gate held: the sink consumed nothing despite the source running.
        assert_eq!(
            seen_for_driver.load(Ordering::SeqCst),
            0,
            "no frame may cross the link while below Playing"
        );
        ctrl_for_driver.set_state(PipelineState::Playing);
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    let stats = res.expect("stateful pipeline runs to completion");

    assert_eq!(stats.frames_emitted, target);
    assert_eq!(
        stats.frames_consumed, target,
        "all frames drained once Playing"
    );
    assert_eq!(seen.load(Ordering::SeqCst), target);
    assert!(eos.load(Ordering::SeqCst), "EOS reaches the sink");

    // The bus records the ladder we walked.
    let mut transitions = Vec::new();
    while let Some(m) = bus.try_recv() {
        if let BusMessage::StateChanged { old, new } = m {
            transitions.push((old, new));
        }
    }
    assert_eq!(
        transitions,
        vec![
            (PipelineState::Ready, PipelineState::Paused),
            (PipelineState::Paused, PipelineState::Playing),
        ],
        "every effective transition posts exactly one StateChanged"
    );
}

/// Starting already `Playing` flows immediately, with no gating stall.
#[tokio::test]
async fn playing_from_the_start_flows_immediately() {
    let target = 4u64;
    let mut src = VideoTestSrc::new(32, 32, 30, target);
    let seen = Arc::new(AtomicU64::new(0));
    let eos = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut sink = CountingSink {
        seen: seen.clone(),
        eos,
    };
    let clock = ZeroClock;
    let ctrl = StateController::new(PipelineState::Playing);

    let stats = run_simple_pipeline_stateful(&mut src, &mut sink, &clock, 4, &ctrl)
        .await
        .expect("pipeline runs");

    assert_eq!(stats.frames_consumed, target);
    assert_eq!(seen.load(Ordering::SeqCst), target);
}
