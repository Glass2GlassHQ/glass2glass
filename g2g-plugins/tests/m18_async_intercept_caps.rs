//! M18 item 5: `SourceLoop::intercept_caps` is async.
//!
//! Proves the runner genuinely awaits the source's caps future during
//! negotiation, not just polls it once on a sync stub. The source
//! `tokio::time::sleep`s for a real interval inside `intercept_caps`
//! and only then returns its caps; the pipeline must complete in at
//! least that interval. With the old sync trait, this source would not
//! compile; with a `Ready`-wrapped sync impl, the sleep wouldn't run at
//! all and the test would finish in microseconds.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::time::{Duration, Instant};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, SourceLoop};
use g2g_core::{
    Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink, PipelineClock, PipelinePacket,
    Rate, RawVideoFormat,
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
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(16),
        height: Dim::Fixed(16),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Source whose `intercept_caps` sleeps for a real interval before
/// returning. Models the RTSP case: caps come from an async network
/// probe (DESCRIBE/SETUP), not from a constant.
struct AsyncCapsSrc {
    probe_delay: Duration,
    probed: bool,
    configured: bool,
}

impl SourceLoop for AsyncCapsSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        let delay = self.probe_delay;
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            self.probed = true;
            Ok(caps())
        })
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            assert!(self.configured, "configure_pipeline must precede run");
            assert!(self.probed, "intercept_caps must have been awaited");
            out.push(PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                timing: FrameTiming::default(),
                sequence: 0,
            }))
            .await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

#[tokio::test]
async fn runner_awaits_async_intercept_caps_before_starting_run() {
    let probe_delay = Duration::from_millis(50);
    let mut src = AsyncCapsSrc {
        probe_delay,
        probed: false,
        configured: false,
    };
    let mut snk = FakeSink::new();
    let clock = ZeroClock;

    let started = Instant::now();
    let stats = run_simple_pipeline(&mut src, &mut snk, &clock, 4)
        .await
        .expect("pipeline must complete");
    let elapsed = started.elapsed();

    assert_eq!(stats.frames_emitted, 1);
    // The runner must have actually awaited the probe future. If it
    // polled `intercept_caps` synchronously (the old sync trait, or a
    // `Ready` impl that ignored the sleep), elapsed would be in
    // microseconds.
    assert!(
        elapsed >= probe_delay,
        "runner must await intercept_caps; elapsed = {elapsed:?}, probe = {probe_delay:?}"
    );
}
