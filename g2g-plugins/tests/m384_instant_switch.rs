//! M384 - instant (flushing) URI switch on `GaplessSrc`, the `instant-uri` analog
//! (vs the M383 gapless `enqueue`). `GaplessController::switch_now` preempts the
//! *currently playing* item: the source races the item's `run` against
//! `wait_instant`, and when the switch wins it drops the run future (cancelling
//! the inner source mid-stream), pushes a `Flush`, resets the timeline to 0, and
//! plays the requested source. The abandoned item's remaining frames never reach
//! the sink.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{GaplessController, SourceLoop};
use g2g_core::{
    Caps, ConfigureOutcome, Dim, G2gError, OutputSink, PipelinePacket, PushOutcome, Rate,
    RawVideoFormat,
};

use g2g_plugins::gaplesssrc::GaplessSrc;

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// A source that emits `frames` DataFrames (pts = i*period) and yields to the
/// executor after each one, so a concurrent `switch_now` gets a chance to preempt
/// it between frames (a non-yielding source would run to completion in one poll).
#[derive(Debug)]
struct YieldSrc {
    frames: u64,
    period_ns: u64,
    configured: bool,
}
impl YieldSrc {
    fn new(frames: u64, period_ns: u64) -> Self {
        Self { frames, period_ns, configured: false }
    }
}
impl SourceLoop for YieldSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;
    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(caps()))
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            for i in 0..self.frames {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8]) as Box<[u8]>)),
                    FrameTiming {
                        pts_ns: i * self.period_ns,
                        dts_ns: i * self.period_ns,
                        duration_ns: self.period_ns,
                        ..FrameTiming::default()
                    },
                    i,
                );
                out.push(PipelinePacket::DataFrame(frame)).await?;
                tokio::task::yield_now().await;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.frames)
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Event {
    Frame(u64),
    Flush,
    Eos,
}

/// Records the ordered packet stream and a live frame counter the app polls.
#[derive(Clone)]
struct Collect {
    events: Arc<Mutex<Vec<Event>>>,
    frames: Arc<AtomicUsize>,
}
impl Collect {
    fn new() -> Self {
        Self { events: Arc::new(Mutex::new(Vec::new())), frames: Arc::new(AtomicUsize::new(0)) }
    }
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    self.events.lock().unwrap().push(Event::Frame(f.timing.pts_ns));
                    self.frames.fetch_add(1, Ordering::SeqCst);
                }
                PipelinePacket::Flush => self.events.lock().unwrap().push(Event::Flush),
                PipelinePacket::Eos => self.events.lock().unwrap().push(Event::Eos),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

#[tokio::test]
async fn switch_now_preempts_the_current_item_with_a_flush() {
    let ctl = GaplessController::new();
    let app = ctl.clone();
    // clip1 is long (100 frames) so the switch lands well before its natural end.
    let mut src = GaplessSrc::new(Box::new(YieldSrc::new(100, 500)), ctl);
    let mut sink = Collect::new();
    let frames = sink.frames.clone();
    let events = sink.events.clone();

    let app_fut = async {
        // Wait until clip1 has shown a couple of frames, then switch instantly to
        // a short clip2 and finish (so playback ends after clip2's clean EOS).
        while frames.load(Ordering::SeqCst) < 2 {
            tokio::task::yield_now().await;
        }
        app.switch_now(Box::new(YieldSrc::new(3, 500)));
        app.finish();
    };

    let run_fut = async {
        let c = src.intercept_caps().await.unwrap();
        src.configure_pipeline(&c).unwrap();
        src.run(&mut sink).await.unwrap()
    };

    let (total, ()) = tokio::join!(run_fut, app_fut);

    let evs = events.lock().unwrap().clone();
    let flush_at = evs.iter().position(|e| *e == Event::Flush).expect("a flush on the instant switch");
    assert_eq!(evs.iter().filter(|e| **e == Event::Flush).count(), 1, "exactly one flush");

    // Before the flush: a clip1 prefix, preempted well before its 100 frames.
    let before: Vec<u64> =
        evs[..flush_at].iter().filter_map(|e| if let Event::Frame(p) = e { Some(*p) } else { None }).collect();
    assert!(before.len() >= 2, "clip1 showed the frames the app waited for: {before:?}");
    assert!(before.len() < 100, "clip1 was preempted, not played to its end: {before:?}");
    assert!(before.windows(2).all(|w| w[1] > w[0]), "clip1 frames are monotonic: {before:?}");

    // After the flush: clip2 from a reset timeline (offset 0), then one Eos.
    let after: Vec<Event> = evs[flush_at + 1..].to_vec();
    assert_eq!(
        after,
        vec![Event::Frame(0), Event::Frame(500), Event::Frame(1000), Event::Eos],
        "clip2 plays from 0 after the flush, then a single terminal Eos"
    );

    // `total` counts only fully-pushed frames (clip1 prefix + clip2's 3).
    assert_eq!(total as usize, before.len() + 3, "frame count = clip1 prefix + clip2");
}
