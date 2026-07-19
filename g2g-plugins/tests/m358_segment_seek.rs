//! M358 - SEGMENT seek (segment playback / gapless looping), end to end.
//!
//! A `SeekFlags::SEGMENT` seek asks the source to play `[start, stop]` and, on
//! reaching `stop`, report **segment-done** instead of running to `Eos`. The app
//! observes the segment-done on the `SeekController` and loops with a
//! *non-flushing* accumulating seek, so the running-time clock advances across
//! every iteration (gapless) and no flush reaches the sink. After N loops the
//! app calls `shutdown()` and the source emits `Eos`.
//!
//! This is the seek track's analog of M82 (flush seek) and M211 (accumulating
//! seek): the mechanism is proven with a controlled source and the existing data
//! plane, no runner/packet changes. `SeekFlags::SEGMENT` is the decision point
//! the source consumes: with it, a `stop` boundary means segment-done + loop;
//! without it a bounded seek would just clip and end.

use std::pin::Pin;

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

/// A flushing SEGMENT seek over `[start, stop]`: starts segment playback,
/// resetting the running-time clock (base 0).
fn flush_segment_seek(start: u64, stop: u64) -> Seek {
    Seek {
        rate: 1.0,
        flags: SeekFlags::FLUSH | SeekFlags::SEGMENT,
        start_type: SeekType::Set,
        start,
        stop_type: SeekType::Set,
        stop,
    }
}

/// A non-flushing SEGMENT seek over `[start, stop]`: the loop iteration. No
/// flush; the running-time clock keeps advancing (accumulating base).
fn loop_segment_seek(start: u64, stop: u64) -> Seek {
    Seek {
        rate: 1.0,
        flags: SeekFlags::SEGMENT,
        start_type: SeekType::Set,
        start,
        stop_type: SeekType::Set,
        stop,
    }
}

/// Segment-aware source. Plays the active segment, clipping to `stop`; on a
/// `SEGMENT` segment it reports segment-done at the boundary and parks (polling)
/// for the app's loop seek or a shutdown, instead of running to `Eos`.
struct SegmentLoopSrc {
    position: u64,
    step_ns: u64,
    sequence: u64,
    segment: Segment,
    /// Whether the active segment is a `SEGMENT` (looping) segment.
    segment_mode: bool,
    seek_ctl: SeekController,
}

impl SegmentLoopSrc {
    /// Apply a seek: emit `Flush` (flushing) or accumulate (non-flushing), then
    /// the new `Segment`, repositioning playback. Anchors the running-time
    /// accumulation at the segment `stop` (playback is clipped there).
    async fn apply_seek(&mut self, out: &mut dyn OutputSink, seek: Seek) -> Result<(), G2gError> {
        // The running time reached is measured at the segment end, not the
        // overshot `position`, so the accumulating base is exact.
        let anchor = self
            .segment
            .stop
            .map_or(self.position, |s| self.position.min(s));
        self.segment.position = anchor;
        let seg = if seek.is_flush() {
            out.push(PipelinePacket::Flush).await?;
            Segment::for_flush_seek(&seek, None)
        } else {
            self.segment.accumulate_seek(&seek, None)
        };
        self.segment_mode = seek.flags.contains(SeekFlags::SEGMENT);
        self.position = seg.start;
        self.segment = seg;
        out.push(PipelinePacket::Segment(seg)).await?;
        Ok(())
    }
}

impl SourceLoop for SegmentLoopSrc {
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
            loop {
                // Pick up an app-requested (re)position before the next frame.
                if let Some(seek) = self.seek_ctl.take_pending() {
                    self.apply_seek(out, seek).await?;
                }

                // Segment boundary: clip at `stop`. On a SEGMENT segment, report
                // done and park for the loop seek / shutdown rather than Eos.
                if let Some(stop) = self.segment.stop {
                    if self.position > stop {
                        if self.segment_mode {
                            self.seek_ctl.notify_segment_done(stop);
                            loop {
                                if let Some(seek) = self.seek_ctl.take_pending() {
                                    self.apply_seek(out, seek).await?;
                                    break;
                                }
                                if self.seek_ctl.is_shutdown() {
                                    out.push(PipelinePacket::Eos).await?;
                                    return Ok(emitted);
                                }
                                tokio::task::yield_now().await;
                            }
                            continue;
                        }
                        // A non-SEGMENT bounded segment just ends.
                        out.push(PipelinePacket::Eos).await?;
                        return Ok(emitted);
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
        })
    }
}

#[tokio::test]
async fn segment_seek_loops_gaplessly_then_shuts_down() {
    let stop = 5_000u64;
    let step = 1_000u64;
    let n_loops = 3u64; // segment-dones before shutdown
                        // Frames per segment: pts 0, 1000, .., 5000 (stop inclusive) = 6.
    let frames_per_segment = stop / step + 1;

    let seek_ctl = SeekController::new();
    // Arm the initial flushing SEGMENT seek before the run starts, so the source
    // enters segment mode on its first loop iteration (no default-segment frames
    // leak out first).
    seek_ctl.seek(flush_segment_seek(0, stop));

    let mut src = SegmentLoopSrc {
        position: 0,
        step_ns: step,
        sequence: 0,
        segment: Segment::new(),
        segment_mode: false,
        seek_ctl: seek_ctl.clone(),
    };
    let mut sink = FakeSink::new();

    let pipeline = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 4);

    let driver_ctl = seek_ctl.clone();
    let driver = async move {
        // After each completed segment, loop with a non-flushing SEGMENT seek;
        // after `n_loops` completions, shut the source down.
        loop {
            let before = driver_ctl.segment_done_count();
            while driver_ctl.segment_done_count() == before {
                tokio::task::yield_now().await;
            }
            // A new segment-done is available to consume.
            assert_eq!(driver_ctl.take_segment_done(), Some(stop));
            if driver_ctl.segment_done_count() >= n_loops {
                driver_ctl.shutdown();
                break;
            }
            driver_ctl.seek(loop_segment_seek(0, stop));
        }
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    let stats = res.expect("pipeline runs");

    assert!(sink.eos_seen(), "shutdown ends the looping source with Eos");
    // The source completed `n_loops` segments (segment-done reported each time).
    assert_eq!(seek_ctl.segment_done_count(), n_loops);

    // Every loop's frames reached the sink: n_loops segments, each fully played.
    assert_eq!(stats.frames_consumed, n_loops * frames_per_segment);
    assert_eq!(sink.received(), n_loops * frames_per_segment);

    // The defining contrast: only the *initial* seek flushed; the loop seeks are
    // non-flushing (gapless).
    assert_eq!(sink.flushes(), 1, "only the initial segment seek flushes");
    // Opening segment + initial + (n_loops - 1) loop segments. The last segment
    // is the (n_loops-1)th loop seek; the final completion triggers shutdown
    // before another segment is armed.
    assert_eq!(sink.segments(), 1 + 1 + (n_loops - 1));

    // Running time is monotonic across loops: each loop's base accumulates by the
    // segment span (stop - start), so the last loop segment starts at
    // (n_loops-1) * stop and its first frame (pts 0) continues there.
    let last = sink.last_segment().expect("a loop segment was recorded");
    assert_eq!(
        last.base,
        (n_loops - 1) * stop,
        "base accumulates one span per loop"
    );
    assert_eq!(
        last.to_running_time(0),
        Some((n_loops - 1) * stop),
        "gapless across the loop"
    );
}

/// A bounded SEGMENT seek with no app loop still terminates cleanly: the source
/// reports the single segment-done, the app shuts it down, and it emits `Eos`.
#[tokio::test]
async fn single_segment_then_shutdown_emits_eos() {
    let stop = 3_000u64;
    let seek_ctl = SeekController::new();
    seek_ctl.seek(flush_segment_seek(0, stop));

    let mut src = SegmentLoopSrc {
        position: 0,
        step_ns: 1_000,
        sequence: 0,
        segment: Segment::new(),
        segment_mode: false,
        seek_ctl: seek_ctl.clone(),
    };
    let mut sink = FakeSink::new();

    let pipeline = run_simple_pipeline(&mut src, &mut sink, &ZeroClock, 4);
    let driver_ctl = seek_ctl.clone();
    let driver = async move {
        while driver_ctl.segment_done_count() == 0 {
            tokio::task::yield_now().await;
        }
        driver_ctl.shutdown();
    };

    let (res, ()) = tokio::join!(pipeline, driver);
    let stats = res.expect("pipeline runs");

    assert!(sink.eos_seen());
    assert_eq!(seek_ctl.segment_done_count(), 1);
    // pts 0..=3000 step 1000 = 4 frames.
    assert_eq!(stats.frames_consumed, 4);
    assert_eq!(sink.flushes(), 1, "the initial segment seek flushed once");
}
