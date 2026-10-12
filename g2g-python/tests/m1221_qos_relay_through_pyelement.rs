#![cfg(feature = "analytics")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::element::DynAsyncElement;
use g2g_core::memory::SystemSlice;
use g2g_core::metrics::monotonic_ns;
use g2g_core::runtime::{run_linear_chain, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, Frame, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, QosMessage, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

const WIDTH: u32 = 2;
const HEIGHT: u32 = 1;
const FRAME_COUNT: u64 = 60;
const FRAME_PERIOD_NS: u64 = 33_000_000;
const JITTER_NS: i64 = 10 * FRAME_PERIOD_NS as i64;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

// Ignores QoS, so only a pyelement can shed the late frames.
struct DeafSource;

impl SourceLoop for DeafSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(caps()))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for sequence in 0..FRAME_COUNT {
                let bytes = vec![0u8; (WIDTH * HEIGHT * 4) as usize];
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns: sequence * FRAME_PERIOD_NS,
                        ..FrameTiming::default()
                    },
                    sequence,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(FRAME_COUNT)
        })
    }
}

// Reports every frame it receives as late.
#[derive(Default)]
struct LateSink {
    pending: Option<QosMessage>,
}

impl AsyncElement for LateSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                self.pending = Some(QosMessage {
                    jitter_ns: JITTER_NS,
                    running_time_ns: frame.timing.pts_ns,
                    taken_ns: monotonic_ns(),
                });
            }
            Ok(())
        })
    }
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pending.take()
    }
}

fn counting_element() -> PyTransform {
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );
    PyTransform::new("m1178_element", "CountingTransform")
}

#[tokio::test]
async fn a_late_report_reaches_the_pyelement_upstream_of_another() {
    let mut upstream = counting_element();
    let mut downstream = counting_element();
    let mut sink = LateSink::default();

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut upstream, &mut downstream];
    run_linear_chain(&mut DeafSource, transforms, &mut sink, &ZeroClock, 1)
        .await
        .expect("pipeline runs");

    assert!(
        upstream.late_dropped_count() > 0,
        "the upstream pyelement heard the sink and dropped late frames (downstream dropped {})",
        downstream.late_dropped_count()
    );
}
