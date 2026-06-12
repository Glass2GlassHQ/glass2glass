//! M18 item 7 — structured negotiation failures on the bus.
//!
//! When startup caps negotiation fails, the runner returns the opaque
//! `G2gError::CapsMismatch` to its caller (the error type can't carry the
//! detail). `run_source_transform_sink_with_bus` additionally posts a
//! `BusMessage::NegotiationFailed(NegotiationFailure)` so the application
//! learns *which* link conflicted on *what* (DESIGN-M16-caps-nego.md §13.4
//! item 7; §13.3 "structured caps-failure messages aren't routed through
//! the bus").
//!
//! Here an RGBA source feeds an NV12-only sink: the formats don't intersect,
//! so the solver returns `EmptyLink`. The bus carries that, while the run
//! still errors with `CapsMismatch`.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_source_transform_sink_with_bus, SourceLoop};
use g2g_core::{
    AsyncElement, Bus, BusMessage, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, MemoryDomain, NegotiationFailure, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Source advertising RGBA. It never produces frames here: negotiation
/// fails before `run` is reached.
struct RgbaSource;

impl SourceLoop for RgbaSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(rgba(640, 480)))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let _ = out
                .push(PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                    timing: FrameTiming::default(),
                    sequence: 0,
                }))
                .await;
            Ok(0)
        })
    }
}

/// Pass-through transform (legacy bridge: no native constraint).
struct PassThrough;

impl AsyncElement for PassThrough {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move { out.push(packet).await.map(|_| ()) })
    }
}

/// NV12-only sink: its native constraint never overlaps an RGBA producer.
struct Nv12Sink;

impl AsyncElement for Nv12Sink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        _packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

#[tokio::test]
async fn startup_negotiation_failure_posts_structured_message_to_bus() {
    let (bus, handle) = Bus::new(4);
    let mut src = RgbaSource;
    let mut tx = PassThrough;
    let mut snk = Nv12Sink;
    let clock = ZeroClock;

    let result =
        run_source_transform_sink_with_bus(&mut src, &mut tx, &mut snk, &clock, 4, &handle).await;

    // The caller still receives the opaque error.
    assert_eq!(result.err(), Some(G2gError::CapsMismatch));

    // The bus carries the structured detail: an empty link between the
    // RGBA-producing side and the NV12-only sink.
    match bus.try_recv() {
        Some(BusMessage::NegotiationFailed(NegotiationFailure::EmptyLink {
            upstream,
            downstream,
        })) => {
            assert!(
                downstream == upstream + 1,
                "EmptyLink must name an adjacent pair, got {upstream}->{downstream}"
            );
        }
        other => panic!("expected NegotiationFailed(EmptyLink), got {other:?}"),
    }
    assert!(bus.try_recv().is_none(), "exactly one failure posted");
}

#[tokio::test]
async fn successful_negotiation_posts_nothing() {
    // An NV12 source through the same NV12 sink negotiates cleanly: the bus
    // stays empty, so the failure post above is not spurious.
    struct Nv12Source;
    impl SourceLoop for Nv12Source {
        type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
        type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
        where
            Self: 'a;
        fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
            core::future::ready(Ok(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
            }))
        }
        fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }
        fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
            Box::pin(async move {
                out.push(PipelinePacket::Eos).await?;
                Ok(0)
            })
        }
    }

    let (bus, handle) = Bus::new(4);
    let mut src = Nv12Source;
    let mut tx = PassThrough;
    let mut snk = Nv12Sink;
    let clock = ZeroClock;

    run_source_transform_sink_with_bus(&mut src, &mut tx, &mut snk, &clock, 4, &handle)
        .await
        .expect("NV12 source -> NV12 sink negotiates");

    assert!(
        bus.try_recv().is_none(),
        "a successful negotiation posts no failure"
    );
}
