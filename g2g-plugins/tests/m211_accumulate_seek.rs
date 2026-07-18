//! M211 - non-flushing (accumulating) seek, end to end. Unlike a flushing seek
//! (M82), a non-flushing seek does NOT reset the running-time clock or flush the
//! pipeline: the source emits a new `Segment` whose `base` is the running time
//! already played, so downstream running time stays monotonic across the seek
//! (the gapless / segment-seek / loop case). The runner forwards `Segment`
//! through the existing data plane, so no flush reaches the sink.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use core::future::Future;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, SeekController, SourceLoop};
use g2g_core::{
    Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat, Seek, SeekFlags, SeekType, Segment,
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

/// Seek-aware source that handles both seek kinds. On a **flushing** seek it
/// emits `Flush` + a reset segment (base 0); on a **non-flushing** seek it emits
/// only the accumulating segment (base = running time reached so far), no flush.
/// It tracks its current `Segment` so `accumulate_seek` can read the running time
/// playback has reached.
struct SeekableSrc {
    position: u64,
    step_ns: u64,
    total: u64,
    sequence: u64,
    segment: Segment,
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
                if let Some(seek) = self.seek_ctl.take_pending() {
                    // Record where playback has reached so the segment math is
                    // anchored to the current position.
                    self.segment.position = self.position;
                    let seg = if seek.is_flush() {
                        let _ = out.push(PipelinePacket::Flush).await?;
                        Segment::for_flush_seek(&seek, None)
                    } else {
                        // No flush: running time continues from where it is.
                        self.segment.accumulate_seek(&seek, None)
                    };
                    self.position = seek.start;
                    self.segment = seg;
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
                    meta: Default::default(),
                };
                let _ = out.push(PipelinePacket::DataFrame(frame)).await?;
                self.position += self.step_ns;
                self.sequence += 1;
                emitted += 1;
                self.emitted_observable.fetch_add(1, Ordering::SeqCst);
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(emitted)
        })
    }
}

fn non_flush_seek_to(start: u64) -> Seek {
    Seek {
        rate: 1.0,
        flags: SeekFlags::NONE,
        start_type: SeekType::Set,
        start,
        stop_type: SeekType::None,
        stop: 0,
    }
}

#[tokio::test]
async fn non_flush_seek_accumulates_base_and_does_not_flush() {
    let seek_target = 1_000_000u64;
    let emitted = Arc::new(AtomicU64::new(0));
    let seek_ctl = SeekController::new();

    let mut src = SeekableSrc {
        position: 0,
        step_ns: 1_000,
        total: 8,
        sequence: 0,
        segment: Segment::new(),
        seek_ctl: seek_ctl.clone(),
        emitted_observable: emitted.clone(),
    };
    let mut sink = FakeSink::new();

    let pipeline = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 4);
    let emitted_for_driver = emitted.clone();
    let driver = async move {
        while emitted_for_driver.load(Ordering::SeqCst) < 3 {
            tokio::task::yield_now().await;
        }
        seek_ctl.seek(non_flush_seek_to(seek_target));
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    let stats = res.expect("pipeline runs");

    assert_eq!(
        stats.frames_consumed, 8,
        "all frames (pre + post seek) reached the sink"
    );
    assert!(sink.eos_seen());

    // The defining contrast with a flushing seek: NO flush downstream.
    assert_eq!(
        sink.flushes(),
        0,
        "a non-flushing seek does not flush the pipeline"
    );
    // Opening segment + the accumulating one.
    assert_eq!(sink.segments(), 2);

    let seg = sink.last_segment().expect("post-seek segment recorded");
    assert_eq!(seg.start, seek_target, "repositioned to the target");
    assert!(
        seg.base > 0,
        "non-flushing seek accumulates running time, does not reset it"
    );
    // The first post-seek frame continues at the accumulated running time, so the
    // running-time line is monotonic across the seek (gapless).
    assert_eq!(seg.to_running_time(seek_target), Some(seg.base));
}
