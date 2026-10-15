#![cfg(all(
    target_os = "linux",
    feature = "std",
    any(feature = "cuda-gl", feature = "cuda-kms")
))]

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;

use g2g_core::runtime::solver::solve_linear;
use g2g_core::runtime::{run_simple_pipeline_with_bus, SourceLoop};
use g2g_core::{
    AsyncElement, Bus, BusMessage, Caps, CapsConstraint, CapsSet, Colorimetry, ConfigureOutcome,
    Dim, G2gError, Interlace, NegotiationFailure, OutputSink, PipelineClock, Rate, RawVideoFormat,
};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FRAMERATE_Q16: u32 = 30 << 16;
const LINK_CAPACITY: usize = 4;
const BUS_CAPACITY: usize = 4;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn raw_video(format: RawVideoFormat) -> Caps {
    Caps::RawVideo {
        format,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FRAMERATE_Q16),
        interlace: Interlace::Progressive,
        colorimetry: Colorimetry::UNKNOWN,
    }
}

// never runs: a refused negotiation ends the pipeline before `run`
struct FixedFormatSource(RawVideoFormat);

impl SourceLoop for FixedFormatSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(raw_video(self.0)))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(raw_video(
            self.0,
        )))))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, _out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async { Ok(0) })
    }
}

async fn assert_refuses_i420_at_negotiation<Sink: AsyncElement>(sink: &mut Sink) {
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let mut source = FixedFormatSource(RawVideoFormat::I420);
    let result =
        run_simple_pipeline_with_bus(&mut source, sink, &ZeroClock, LINK_CAPACITY, &handle).await;
    assert_eq!(result.err(), Some(G2gError::CapsMismatch));
    // a refusal from configure_pipeline posts no NegotiationFailed
    match bus.try_recv() {
        Some(BusMessage::NegotiationFailed(NegotiationFailure::EmptyLink { .. })) => {}
        other => panic!("expected NegotiationFailed(EmptyLink), got {other:?}"),
    }
}

fn assert_accepts_nv12<Sink: AsyncElement>(sink: &Sink) {
    let source = CapsConstraint::Produces(CapsSet::one(raw_video(RawVideoFormat::Nv12)));
    let links = solve_linear(&[&source, &sink.caps_constraint_as_sink()])
        .expect("NV12 negotiates onto the sink");
    assert_eq!(links, vec![raw_video(RawVideoFormat::Nv12)]);
}

#[cfg(feature = "cuda-gl")]
#[tokio::test]
async fn cudaglsink_refuses_i420_at_negotiation() {
    assert_refuses_i420_at_negotiation(&mut g2g_plugins::cudaglsink::CudaGlSink::new()).await;
}

#[cfg(feature = "cuda-gl")]
#[test]
fn cudaglsink_accepts_nv12() {
    assert_accepts_nv12(&g2g_plugins::cudaglsink::CudaGlSink::new());
}

#[cfg(feature = "cuda-kms")]
#[tokio::test]
async fn cudakmssink_refuses_i420_at_negotiation() {
    assert_refuses_i420_at_negotiation(&mut g2g_plugins::cudakmssink::CudaKmsSink::new()).await;
}

#[cfg(feature = "cuda-kms")]
#[test]
fn cudakmssink_accepts_nv12() {
    assert_accepts_nv12(&g2g_plugins::cudakmssink::CudaKmsSink::new());
}
