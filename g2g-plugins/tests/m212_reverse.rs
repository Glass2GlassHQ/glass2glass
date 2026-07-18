//! M212 - reverse playback at the sink. A reverse-rate segment (`rate < 0`)
//! plays a stream from `stop` down to `start`: the source emits frames in
//! descending PTS order, and the sink maps each to ascending running time
//! (`Segment::to_running_time` measures reverse from `stop`). The key finding is
//! that `SyncSink` needs no reverse-specific code: it already schedules by the
//! segment's running time and clips via `contains`, so the `Segment` abstraction
//! generalizes presentation to negative rate transparently. These tests prove
//! that path end to end.
//!
//! Trick-mode KEY_UNIT frame selection (present only keyframes for fast scrub) is
//! a separate milestone: it needs a per-frame keyframe flag, which the codec
//! parsers can detect (`h264_au_is_keyframe`, vp8 `parse_keyframe`) but do not
//! yet surface as a frame property.

use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use core::future::{Future, Ready};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, SourceLoop};
use g2g_core::{
    AsyncClock, AsyncElement, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat, Seek, Segment,
};
use g2g_plugins::syncsink::SyncSink;

/// A clock fixed at 0 (so nothing is ever "late") that records every deadline it
/// is asked to sleep until, and resolves the sleep immediately. The recorded
/// deadlines are the running-time order in which frames were presented.
#[derive(Clone)]
struct RecordingClock {
    deadlines: Arc<Mutex<Vec<u64>>>,
}
impl PipelineClock for RecordingClock {
    fn now_ns(&self) -> u64 {
        0
    }
}
impl AsyncClock for RecordingClock {
    type SleepFuture<'a> = Ready<()>;
    fn sleep_until_ns(&self, deadline_ns: u64) -> Ready<()> {
        self.deadlines.lock().unwrap().push(deadline_ns);
        core::future::ready(())
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

fn frame(pts_ns: u64, sequence: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(std::boxed::Box::new([0u8]))),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        sequence,
    ))
}

struct NullSink;
impl OutputSink for NullSink {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<g2g_core::PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(g2g_core::PushOutcome::Accepted) })
    }
}

#[tokio::test]
async fn syncsink_presents_a_reverse_segment_in_ascending_running_time() {
    let deadlines = Arc::new(Mutex::new(Vec::new()));
    let mut sink = SyncSink::new(RecordingClock {
        deadlines: deadlines.clone(),
    });
    sink.configure_pipeline(&caps()).unwrap();
    let mut out = NullSink;

    // Reverse over [0, 100ms]: source emits frames newest-PTS-first.
    let seg = Segment::for_flush_seek(&Seek::reverse(0, 100_000_000), None);
    sink.process(PipelinePacket::Segment(seg), &mut out)
        .await
        .unwrap();

    // Descending PTS (reverse emission order); the last one is outside the
    // segment (above stop) and must be clipped.
    for (i, pts) in [100_000_000u64, 75_000_000, 50_000_000, 25_000_000, 0]
        .into_iter()
        .enumerate()
    {
        sink.process(frame(pts, i as u64), &mut out).await.unwrap();
    }
    sink.process(frame(150_000_000, 99), &mut out)
        .await
        .unwrap(); // outside: clipped

    assert_eq!(
        sink.received(),
        5,
        "every in-range reverse frame is presented"
    );
    assert_eq!(sink.clipped(), 1, "the above-stop frame is clipped");
    // Reverse maps descending PTS to ASCENDING running-time deadlines: the sink
    // presented them in increasing running-time order, the correct visual order.
    let got = deadlines.lock().unwrap().clone();
    assert_eq!(
        got,
        vec![0, 25_000_000, 50_000_000, 75_000_000, 100_000_000]
    );
}

/// Source that emits `count` frames in descending PTS order over `[0, top]`
/// after announcing a reverse segment, then EOS, the synthetic upstream of
/// reverse playback.
struct ReverseSrc {
    top: u64,
    step: u64,
    count: u64,
    emitted: Arc<AtomicU64>,
}

impl SourceLoop for ReverseSrc {
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
            out.push(PipelinePacket::CapsChanged(caps())).await?;
            // Reverse segment over the played range.
            let seg = Segment::for_flush_seek(&Seek::reverse(0, self.top), None);
            out.push(PipelinePacket::Segment(seg)).await?;
            for i in 0..self.count {
                let pts = self.top - i * self.step; // descending
                let f = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(
                        std::vec![0u8; 4].into_boxed_slice(),
                    )),
                    FrameTiming {
                        pts_ns: pts,
                        ..FrameTiming::default()
                    },
                    i,
                );
                out.push(PipelinePacket::DataFrame(f)).await?;
                self.emitted.fetch_add(1, Ordering::SeqCst);
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.count)
        })
    }
}

#[tokio::test]
async fn reverse_pipeline_runs_end_to_end() {
    let deadlines = Arc::new(Mutex::new(Vec::new()));
    let emitted = Arc::new(AtomicU64::new(0));
    let mut src = ReverseSrc {
        top: 40_000_000,
        step: 10_000_000,
        count: 5,
        emitted: emitted.clone(),
    };
    let mut sink = SyncSink::new(RecordingClock {
        deadlines: deadlines.clone(),
    });

    let stats = run_simple_pipeline(
        &mut src,
        &mut sink,
        &RecordingClock {
            deadlines: deadlines.clone(),
        },
        4,
    )
    .await
    .expect("reverse pipeline runs");

    // All 5 frames (pts 40,30,20,10,0 ms) reached the sink, presented in
    // ascending running time (0,10,20,30,40 ms), i.e. correct reverse order.
    assert_eq!(stats.frames_consumed, 5);
    let got = deadlines.lock().unwrap().clone();
    assert_eq!(got, vec![0, 10_000_000, 20_000_000, 30_000_000, 40_000_000]);
}
