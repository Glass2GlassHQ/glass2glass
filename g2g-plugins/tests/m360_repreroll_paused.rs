//! M360 - re-preroll after a flushing seek while paused.
//!
//! A non-live pipeline in `Paused` prerolls one frame and holds, backpressuring
//! the source. A flushing seek issued now cannot otherwise take effect (the held
//! sink never drains, so the source is stuck and never observes the seek). The
//! app calls `StateController::request_repreroll()` alongside the seek: the sink
//! arm drains the stale pre-seek frames (discarding them, not presenting), waits
//! for the `Flush`, then prerolls the post-flush *target* frame and re-completes
//! preroll (a fresh `AsyncDone`). So scrubbing a paused pipeline updates the
//! displayed frame, exactly as GStreamer's flushing seek re-prerolls.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use core::future::Future;

use g2g_core::element::{AsyncElement, BoxFuture, ConfigureOutcome, OutputSink};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    run_simple_pipeline_stateful, SeekController, SourceLoop, StateController,
};
use g2g_core::{
    Caps, Dim, FrameTiming, G2gError, MemoryDomain, PipelineClock, PipelinePacket, PipelineState,
    Rate, RawVideoFormat, Seek, Segment, StateChangeReturn,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(64),
        height: Dim::Fixed(64),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Seek-aware source: emits one frame per `step_ns` from `position`, polling the
/// `SeekController` each iteration; a flushing seek emits `Flush` + a reset
/// `Segment` and jumps to the target. Stops after `total` frames.
struct SeekableSrc {
    position: u64,
    step_ns: u64,
    total: u64,
    sequence: u64,
    seek_ctl: SeekController,
}

impl SourceLoop for SeekableSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let _ = out.push(PipelinePacket::CapsChanged(caps())).await?;
            let mut emitted = 0u64;
            while emitted < self.total {
                if let Some(seek) = self.seek_ctl.take_pending() {
                    if seek.is_flush() {
                        let _ = out.push(PipelinePacket::Flush).await?;
                        self.position = seek.start;
                        let seg = Segment::for_flush_seek(&seek, None);
                        let _ = out.push(PipelinePacket::Segment(seg)).await?;
                    }
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        vec![0u8; 4].into_boxed_slice(),
                    )),
                    timing: FrameTiming {
                        pts_ns: self.position,
                        ..FrameTiming::default()
                    },
                    sequence: self.sequence,
                    meta: Default::default(),
                };
                let _ = out.push(PipelinePacket::DataFrame(frame)).await?;
                self.position += self.step_ns;
                self.sequence += 1;
                emitted += 1;
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

/// Sink recording, via shared atomics the driver can read mid-run, how many
/// `DataFrame`s it presented and the PTS of the last one.
struct RecordSink {
    presented: Arc<AtomicU64>,
    last_pts: Arc<AtomicU64>,
}

impl AsyncElement for RecordSink {
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
            if let PipelinePacket::DataFrame(f) = packet {
                self.last_pts.store(f.timing.pts_ns, Ordering::SeqCst);
                self.presented.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn flushing_seek_while_paused_re_prerolls_the_target_frame() {
    let seek_target = 1_000_000u64;
    let presented = Arc::new(AtomicU64::new(0));
    let last_pts = Arc::new(AtomicU64::new(u64::MAX));
    let seek_ctl = SeekController::new();

    let mut src = SeekableSrc {
        position: 0,
        step_ns: 1_000,
        total: 40,
        sequence: 0,
        seek_ctl: seek_ctl.clone(),
    };
    let mut sink = RecordSink {
        presented: presented.clone(),
        last_pts: last_pts.clone(),
    };
    let clock = ZeroClock;
    let ctrl = StateController::new(PipelineState::Ready); // non-live default

    // Small link so only a couple of stale pre-seek frames buffer behind the hold.
    let pipeline = run_simple_pipeline_stateful(&mut src, &mut sink, &clock, 2, &ctrl);

    let ctrl_d = ctrl.clone();
    let driver = async move {
        // Non-live Paused prerolls one frame (pts 0) and holds.
        assert_eq!(
            ctrl_d.set_state(PipelineState::Paused),
            StateChangeReturn::Async
        );
        ctrl_d.await_prerolled().await;
        assert_eq!(presented.load(Ordering::SeqCst), 1, "one preroll frame");
        assert_eq!(
            last_pts.load(Ordering::SeqCst),
            0,
            "the preroll frame is the first one"
        );

        // Flushing seek while still paused, with the re-preroll request. The
        // target frame must become the new preroll without leaving Paused.
        seek_ctl.seek(Seek::flush_to(seek_target));
        ctrl_d.request_repreroll();
        ctrl_d.await_prerolled().await; // re-completes when the target prerolls

        assert_eq!(
            last_pts.load(Ordering::SeqCst),
            seek_target,
            "the post-flush target frame is now the visible preroll"
        );
        assert_eq!(
            presented.load(Ordering::SeqCst),
            2,
            "exactly two frames presented (original preroll + re-preroll target); \
             the stale pre-seek frames were drained, not shown"
        );

        // Still paused: the gate holds again after the re-preroll frame.
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            presented.load(Ordering::SeqCst),
            2,
            "held after the re-preroll"
        );

        ctrl_d.set_state(PipelineState::Playing);
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    let stats = res.expect("pipeline runs to completion");
    // After Playing, post-target frames flow; the run ends on Eos. The first
    // presented post-seek frame was the target, so playback resumed there.
    assert!(stats.frames_consumed >= 2);
}
