//! M12: pipeline clock distribution (provider election).
//!
//! A pipeline runs against one clock. When an element offers one (a live
//! source's capture clock, an audio sink's DAC clock), the runner elects the
//! highest-priority offer over the supplied system clock and reports it on
//! `RunStats`. These tests drive the real runners and assert the elected
//! clock's priority and base time.

use core::future::Future;
use core::pin::Pin;
use std::sync::Arc;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_simple_pipeline, run_source_transform_sink, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ClockCandidate, ClockPriority, ConfigureOutcome, Dim, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

/// A clock pinned to a fixed instant, so a test can assert the exact base time
/// the runner read from the elected clock.
struct FixedClock(u64);
impl PipelineClock for FixedClock {
    fn now_ns(&self) -> u64 {
        self.0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(640),
        height: Dim::Fixed(480),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn make_frame() -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

/// Source that optionally provides a clock at a chosen priority/instant.
struct EmitSrc {
    provide: Option<(ClockPriority, u64)>,
}

impl SourceLoop for EmitSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        self.provide
            .map(|(p, now)| ClockCandidate::new(p, Arc::new(FixedClock(now))))
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::DataFrame(make_frame())).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Sink that optionally provides a clock at a chosen priority/instant.
struct EmitSink {
    provide: Option<(ClockPriority, u64)>,
}

impl AsyncElement for EmitSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn provide_clock(&self) -> Option<ClockCandidate> {
        self.provide
            .map(|(p, now)| ClockCandidate::new(p, Arc::new(FixedClock(now))))
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// Identity transform with no clock to provide.
struct PassTransform;

impl AsyncElement for PassTransform {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::Eos => Ok(()),
                other => {
                    out.push(other).await?;
                    Ok(())
                }
            }
        })
    }
}

#[tokio::test]
async fn live_source_clock_is_elected_over_fallback() {
    let mut src = EmitSrc { provide: Some((ClockPriority::LiveSource, 5_000)) };
    let mut sink = EmitSink { provide: None };
    // Fallback clock reads a different instant; election must ignore it.
    let fallback = FixedClock(999);

    let stats = run_simple_pipeline(&mut src, &mut sink, &fallback, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.clock_priority, ClockPriority::LiveSource);
    assert_eq!(stats.base_time_ns, 5_000, "base time read from the elected source clock");
}

#[tokio::test]
async fn fallback_clock_used_when_no_element_provides_one() {
    let mut src = EmitSrc { provide: None };
    let mut sink = EmitSink { provide: None };
    let fallback = FixedClock(777);

    let stats = run_simple_pipeline(&mut src, &mut sink, &fallback, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.clock_priority, ClockPriority::SystemFallback);
    assert_eq!(stats.base_time_ns, 777, "base time read from the supplied fallback clock");
}

#[tokio::test]
async fn live_source_outranks_a_sink_provider() {
    let mut src = EmitSrc { provide: Some((ClockPriority::LiveSource, 5_000)) };
    let mut tx = PassTransform;
    let mut sink = EmitSink { provide: Some((ClockPriority::Provider, 8_000)) };
    let fallback = FixedClock(1);

    let stats = run_source_transform_sink(&mut src, &mut tx, &mut sink, &fallback, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.clock_priority, ClockPriority::LiveSource);
    assert_eq!(stats.base_time_ns, 5_000, "the higher-priority live source clock wins");
}

#[tokio::test]
async fn sink_provider_beats_fallback_when_source_has_none() {
    let mut src = EmitSrc { provide: None };
    let mut tx = PassTransform;
    let mut sink = EmitSink { provide: Some((ClockPriority::Provider, 8_000)) };
    let fallback = FixedClock(1);

    let stats = run_source_transform_sink(&mut src, &mut tx, &mut sink, &fallback, 4)
        .await
        .expect("pipeline should complete");

    assert_eq!(stats.clock_priority, ClockPriority::Provider);
    assert_eq!(stats.base_time_ns, 8_000);
}
