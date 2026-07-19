//! M176: presentation base time is stamped at the `Playing` transition, not at
//! runner startup. Under a `StateController`, the runner arms a `PlayAnchor` on
//! the elected clock and hands each sink a `ClockSync::with_play_anchor`; the
//! controller stamps it with `clock.now_ns()` when `set_state(Playing)` fires.
//! So a sink anchors presentation to when streaming actually began, even if the
//! pipeline sat in `Paused` (prerolled) for a while first.

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline_stateful, SourceLoop, StateController};
use g2g_core::{
    AsyncElement, Caps, ClockCandidate, ClockPriority, ClockSync, ConfigureOutcome, Dim, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, PipelineState, Rate, RawVideoFormat,
};

/// A clock whose instant the test advances, so the value stamped at `Playing`
/// differs from the one read at startup (proving anchoring is at the play edge).
struct AdvancingClock(Arc<AtomicU64>);
impl PipelineClock for AdvancingClock {
    fn now_ns(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(8),
        height: Dim::Fixed(8),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Source emitting `target` frames then EOS, offering the shared clock so it is
/// elected (and is the one the controller stamps at `Playing`).
struct EmitSrc {
    clock: Arc<AtomicU64>,
    target: u64,
}

impl SourceLoop for EmitSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn provide_clock(&self) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(
            ClockPriority::LiveSource,
            Arc::new(AdvancingClock(self.clock.clone())),
        ))
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.target {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new(
                        [0u8; 8 * 8 * 4],
                    ))),
                    timing: FrameTiming::default(),
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.target)
        })
    }
}

/// Sink that captures a clone of the `ClockSync` it is handed; because the clone
/// shares the `PlayAnchor`, the test reads `base_time()` / `play_anchored()`
/// after the run to observe the play-edge stamp.
struct RecordingSink {
    sync: Arc<Mutex<Option<ClockSync>>>,
    seen: Arc<AtomicU64>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn set_clock_sync(&mut self, sync: ClockSync) {
        *self.sync.lock().unwrap() = Some(sync);
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let seen = self.seen.clone();
        Box::pin(async move {
            if let PipelinePacket::DataFrame(_) = packet {
                seen.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn base_time_is_stamped_at_the_playing_transition() {
    // Clock starts at 1_000 (the startup / eager base time).
    let clock = Arc::new(AtomicU64::new(1_000));
    let target = 4u64;
    let mut src = EmitSrc {
        clock: clock.clone(),
        target,
    };
    let sync_cell = Arc::new(Mutex::new(None));
    let seen = Arc::new(AtomicU64::new(0));
    let mut sink = RecordingSink {
        sync: sync_cell.clone(),
        seen: seen.clone(),
    };

    let ctrl = StateController::new(PipelineState::Ready); // non-live, prerolls

    let fallback = AdvancingClock(clock.clone());
    let pipeline = run_simple_pipeline_stateful(&mut src, &mut sink, &fallback, 4, &ctrl);

    let ctrl_for_driver = ctrl.clone();
    let sync_for_driver = sync_cell.clone();
    let clock_for_driver = clock.clone();
    let driver = async move {
        ctrl_for_driver.set_state(PipelineState::Paused);
        ctrl_for_driver.await_prerolled().await;

        // Before Playing: the sink has a ClockSync, but it is not yet
        // play-anchored, so it reports the eager startup base time (1_000).
        {
            let guard = sync_for_driver.lock().unwrap();
            let sync = guard.as_ref().expect("sink received a ClockSync");
            assert!(!sync.play_anchored(), "not anchored before Playing");
            assert_eq!(
                sync.base_time(),
                1_000,
                "eager startup base time until Playing"
            );
        }

        // Time advances during the pause; the play edge stamps THIS instant.
        clock_for_driver.store(9_000, Ordering::SeqCst);
        ctrl_for_driver.set_state(PipelineState::Playing);
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    let stats = res.expect("pipeline runs to completion");
    assert_eq!(stats.frames_consumed, target);
    assert_eq!(seen.load(Ordering::SeqCst), target);

    // After Playing: the anchor is stamped with the clock reading at the play
    // edge (9_000), superseding the startup base time (1_000).
    let guard = sync_cell.lock().unwrap();
    let sync = guard.as_ref().unwrap();
    assert!(
        sync.play_anchored(),
        "anchored once Playing stamped the base time"
    );
    assert_eq!(
        sync.base_time(),
        9_000,
        "base time is the play-edge instant, not startup"
    );
    assert_eq!(
        sync.base_time_ns, 1_000,
        "eager field still records the startup instant"
    );
}
