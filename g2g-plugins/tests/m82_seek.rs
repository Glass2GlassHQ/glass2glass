//! M82 - end-to-end flush seek. A seek-aware source polls a `SeekController`
//! between frames; on a flushing seek it emits `Flush`, repositions, emits the
//! post-flush `Segment`, and resumes from the new position. The runner already
//! forwards `Flush` and `Segment`, so seek works through the existing data
//! plane; the sink observes the flush, the new segment, and the repositioned
//! frames.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use core::future::Future;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, SeekController, SourceLoop};
use g2g_core::{
    Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat, Seek, Segment,
};

use g2g_plugins::fakesink::FakeSink;

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

/// Seek-aware test source. Emits one frame per `step_ns` from `position`,
/// polling the `SeekController` each iteration. On a flushing seek it emits
/// `Flush`, jumps `position` to the seek target, emits the post-flush
/// `Segment`, and continues. Stops after `total` frames have been emitted
/// (across the whole run, seek included). Bumps a shared counter so a test
/// can wait for the source to make progress before seeking.
struct SeekableSrc {
    position: u64,
    step_ns: u64,
    total: u64,
    sequence: u64,
    seek_ctl: SeekController,
    emitted_observable: Arc<AtomicU64>,
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
                // Poll for a seek before producing the next frame.
                if let Some(seek) = self.seek_ctl.take_pending() {
                    let _ = out.push(PipelinePacket::Flush).await?;
                    self.position = seek.start;
                    let seg = Segment::for_flush_seek(&seek, None);
                    let _ = out.push(PipelinePacket::Segment(seg)).await?;
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
                };
                let _ = out.push(PipelinePacket::DataFrame(frame)).await?;
                self.position += self.step_ns;
                self.sequence += 1;
                emitted += 1;
                self.emitted_observable.fetch_add(1, Ordering::SeqCst);
                // Yield so a concurrent seek can land mid-stream.
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

/// A mid-stream flushing seek: the sink sees a `Flush`, a second `Segment`
/// (the post-flush one, starting at the seek target), and resumes; the opening
/// SEGMENT plus the post-flush one make two.
#[tokio::test]
async fn flush_seek_repositions_and_emits_new_segment() {
    let seek_target = 1_000_000u64;
    let emitted = Arc::new(AtomicU64::new(0));
    let seek_ctl = SeekController::new();

    let mut src = SeekableSrc {
        position: 0,
        step_ns: 1_000,
        total: 8,
        sequence: 0,
        seek_ctl: seek_ctl.clone(),
        emitted_observable: emitted.clone(),
    };
    let mut sink = FakeSink::new();

    let pipeline = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 4);

    let emitted_for_driver = emitted.clone();
    let driver = async move {
        // Let a few frames flow, then seek.
        while emitted_for_driver.load(Ordering::SeqCst) < 3 {
            tokio::task::yield_now().await;
        }
        seek_ctl.seek(Seek::flush_to(seek_target));
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    let stats = res.expect("pipeline runs");

    // Every emitted frame reached the sink (8 total: pre + post seek).
    assert_eq!(stats.frames_consumed, 8);
    assert_eq!(sink.received(), 8);
    assert!(sink.eos_seen());

    // The seek produced exactly one Flush downstream.
    assert_eq!(sink.flushes(), 1, "the flushing seek flushed the sink once");

    // Two segments: the runner's opening one, then the post-flush one.
    assert_eq!(sink.segments(), 2);
    let seg = sink.last_segment().expect("post-flush segment recorded");
    assert_eq!(
        seg.start, seek_target,
        "post-flush segment starts at the target"
    );
    assert_eq!(seg.base, 0, "a flushing seek restarts running time");
    // The first post-seek frame (pts == target) maps to running time 0.
    assert_eq!(seg.to_running_time(seek_target), Some(0));
}

/// With no seek requested, the source runs straight through: one opening
/// segment, no flush.
#[tokio::test]
async fn no_seek_runs_straight_through() {
    let emitted = Arc::new(AtomicU64::new(0));
    let mut src = SeekableSrc {
        position: 0,
        step_ns: 1_000,
        total: 5,
        sequence: 0,
        seek_ctl: SeekController::new(),
        emitted_observable: emitted,
    };
    let mut sink = FakeSink::new();

    let stats = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 4)
        .await
        .expect("pipeline runs");

    assert_eq!(stats.frames_consumed, 5);
    assert_eq!(sink.flushes(), 0, "no seek, no flush");
    assert_eq!(sink.segments(), 1, "only the opening segment");
}
