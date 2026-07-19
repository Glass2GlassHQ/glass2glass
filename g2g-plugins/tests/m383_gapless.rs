//! M383 - gapless playback. `GaplessSrc` concatenates a playlist of sources into
//! one continuous, monotonically-timed stream (the playbin `about-to-finish` +
//! next-`uri` analog): each item's PTS is rebased onto the running timeline, the
//! inner items' EOS are swallowed, and only the finished playlist emits a terminal
//! `Eos`. The downstream decode chain is reused across items (no flush, no gap).
//!
//! Two checks: a pre-loaded playlist concatenates deterministically (rebased
//! timestamps, single Eos), and the dynamic `about-to-finish` path works (the app
//! enqueues each successor in response to the source's signal, then finishes).

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

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

/// A source that emits `frames` one-byte DataFrames at `period_ns` spacing
/// (pts = i*period, duration = period) on its own zero-based timeline, then Eos.
/// The unit a gapless playlist concatenates.
#[derive(Debug)]
struct CountedSrc {
    frames: u64,
    period_ns: u64,
    configured: bool,
}
impl CountedSrc {
    fn new(frames: u64, period_ns: u64) -> Self {
        Self {
            frames,
            period_ns,
            configured: false,
        }
    }
}
impl SourceLoop for CountedSrc {
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
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.frames)
        })
    }
}

/// Records the PTS of every forwarded DataFrame and counts terminal Eos packets.
#[derive(Default)]
struct Collect {
    pts: Vec<u64>,
    eos: usize,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => self.pts.push(f.timing.pts_ns),
                PipelinePacket::Eos => self.eos += 1,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

async fn drive(src: &mut GaplessSrc, sink: &mut Collect) -> u64 {
    let c = src.intercept_caps().await.expect("first item caps");
    src.configure_pipeline(&c).expect("configure");
    src.run(sink).await.expect("gapless run")
}

#[tokio::test]
async fn preloaded_playlist_concatenates_with_rebased_timestamps() {
    let ctl = GaplessController::new();
    let mut src = GaplessSrc::new(Box::new(CountedSrc::new(3, 1000)), ctl.clone());
    // Pre-load the playlist: clip2 is queued before clip1 even starts.
    ctl.enqueue(Box::new(CountedSrc::new(2, 1000)));
    ctl.finish();

    let mut sink = Collect::default();
    let n = drive(&mut src, &mut sink).await;

    assert_eq!(n, 5, "three frames from clip1 + two from clip2");
    assert_eq!(sink.eos, 1, "a single terminal Eos for the whole playlist");
    // clip1 plays at 0,1000,2000; clip2 rebased onto the timeline at 3000,4000.
    assert_eq!(
        sink.pts,
        vec![0, 1000, 2000, 3000, 4000],
        "monotonic, gapless"
    );
    // Nothing queued behind clip1 was unknown (clip2 was pre-loaded), so the
    // source never had to ask: no about-to-finish for a pre-loaded playlist.
    assert_eq!(
        ctl.about_to_finish_count(),
        0,
        "no signal needed when pre-loaded"
    );
}

#[tokio::test]
async fn dynamic_about_to_finish_drives_the_playlist() {
    let ctl = GaplessController::new();
    let app = ctl.clone();
    let mut src = GaplessSrc::new(Box::new(CountedSrc::new(2, 500)), ctl);
    let mut sink = Collect::default();

    // The app reacts to each about-to-finish: enqueue clip2, then on the next
    // signal finish the playlist (no clip3). Runs concurrently with the source.
    let app_fut = async {
        // First signal fires at clip1 start (empty queue) -> enqueue clip2.
        while !app.take_about_to_finish() {
            tokio::task::yield_now().await;
        }
        app.enqueue(Box::new(CountedSrc::new(3, 500)));
        // Second signal fires at clip2 start (still empty queue) -> finish.
        while !app.take_about_to_finish() {
            tokio::task::yield_now().await;
        }
        app.finish();
    };

    let (n, ()) = tokio::join!(drive(&mut src, &mut sink), app_fut);

    assert_eq!(n, 5, "two frames from clip1 + three from clip2");
    assert_eq!(sink.eos, 1, "a single terminal Eos");
    // clip1 at 0,500; clip2 rebased at 1000,1500,2000.
    assert_eq!(
        sink.pts,
        vec![0, 500, 1000, 1500, 2000],
        "gapless across the dynamic swap"
    );
    assert_eq!(
        app.about_to_finish_count(),
        2,
        "one signal per item with an empty queue"
    );
}
