//! M18 Caps-α — mid-stream caps re-solve over the downstream subgraph
//! (`run_linear_chain`). See `DESIGN-M18-caps-resolve.md`.
//!
//! On a mid-stream `CapsChanged`, the runner derives an interior element's
//! forwarded output from its declared constraint, steered by the downstream
//! feasibility snapshot, instead of the element's greedy local choice (D3). A
//! format converter is pushed toward a sink-acceptable output it would not
//! pick on its own; a converter that can reach no sink-acceptable output
//! fails loud to the bus, rather than forwarding a doomed caps the sink then
//! rejects.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};
use std::vec::Vec;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{
    run_linear_chain, run_linear_chain_with_bus, run_source_transform_sink, SourceLoop,
};
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

fn video(fmt: RawVideoFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: fmt,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn nv12_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// Source: produces `start` caps, emits one frame, pushes a mid-stream
/// `CapsChanged(change_to)`, then the remaining frames + EOS. Native
/// `Produces` so the whole chain takes the arc-consistency solver path.
struct ConvSource {
    start: Caps,
    change_to: Caps,
    total: u32,
}

impl SourceLoop for ConvSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.start.clone()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.start.clone()))))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for i in 0..self.total {
                out.push(PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
                    timing: FrameTiming::default(),
                    sequence: i as u64,
                    meta: Default::default(),
                }))
                .await?;
                if i == 0 {
                    out.push(PipelinePacket::CapsChanged(self.change_to.clone())).await?;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.total as u64)
        })
    }
}

/// Format converter declared as a `DerivedOutput`: for any input it can pass
/// the format through, and for inputs in `nv12_from` it can additionally emit
/// NV12 at the same geometry. Its `process` forwards whatever caps the runner
/// hands it (D3: apply, don't re-choose).
struct FormatConverter {
    nv12_from: Vec<RawVideoFormat>,
}

impl AsyncElement for FormatConverter {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let nv12_from = self.nv12_from.clone();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            let mut alts = std::vec![input.clone()];
            if let Caps::RawVideo { format, width, height, framerate } = input {
                if nv12_from.contains(format) {
                    alts.push(Caps::RawVideo {
                        format: RawVideoFormat::Nv12,
                        width: width.clone(),
                        height: height.clone(),
                        framerate: framerate.clone(),
                    });
                }
            }
            CapsSet::from_alternatives(alts)
        }))
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

/// NV12-only sink that records the format of every `CapsChanged` it applies,
/// so a test can prove which caps actually reached it.
struct RecordingSink {
    caps_log: Arc<Mutex<Vec<RawVideoFormat>>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(nv12_any()))
    }

    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let log = Arc::clone(&self.caps_log);
        Box::pin(async move {
            if let PipelinePacket::CapsChanged(Caps::RawVideo { format, .. }) = packet {
                log.lock().unwrap().push(format);
            }
            Ok(())
        })
    }
}

/// Source flips RGBA -> I420 mid-stream. The converter can emit NV12 from
/// either, so Caps-α steers its output to NV12 (the only format the sink
/// accepts) instead of forwarding I420 greedily. Without the re-solve the
/// sink would reject I420 and never see NV12 after the change.
#[tokio::test]
async fn midstream_change_steers_converter_to_sink_acceptable_output() {
    let caps_log = Arc::new(Mutex::new(Vec::new()));
    let mut src = ConvSource {
        start: video(RawVideoFormat::Rgba8, 640, 480),
        change_to: video(RawVideoFormat::I420, 640, 480),
        total: 8,
    };
    let mut conv = FormatConverter {
        nv12_from: std::vec![RawVideoFormat::Rgba8, RawVideoFormat::I420],
    };
    let mut sink = RecordingSink { caps_log: Arc::clone(&caps_log) };
    let clock = ZeroClock;

    let transforms: Vec<&mut dyn DynAsyncElement> = std::vec![&mut conv];
    let stats = run_linear_chain(&mut src, transforms, &mut sink, &clock, 4)
        .await
        .expect("chain runs");

    assert_eq!(stats.frames_consumed, 8, "every frame crosses the converter");
    assert_eq!(
        *caps_log.lock().unwrap(),
        std::vec![RawVideoFormat::Nv12],
        "the sink received the runner-steered NV12, not the source's I420"
    );
}

/// Same chain, but the converter can only emit NV12 from RGBA. After the
/// mid-stream switch to I420 it has no sink-acceptable output: Caps-α fails
/// loud (reverse reconfigure + structured `EmptyLink` on the bus) instead of
/// forwarding a doomed I420 the sink silently mishandles.
#[tokio::test]
async fn midstream_change_with_no_acceptable_output_fails_loud_to_bus() {
    let caps_log = Arc::new(Mutex::new(Vec::new()));
    let mut src = ConvSource {
        start: video(RawVideoFormat::Rgba8, 640, 480),
        change_to: video(RawVideoFormat::I420, 640, 480),
        total: 8,
    };
    let mut conv = FormatConverter { nv12_from: std::vec![RawVideoFormat::Rgba8] };
    let mut sink = RecordingSink { caps_log: Arc::clone(&caps_log) };
    let clock = ZeroClock;
    let (bus, handle) = Bus::new(4);

    let transforms: Vec<&mut dyn DynAsyncElement> = std::vec![&mut conv];
    run_linear_chain_with_bus(&mut src, transforms, &mut sink, &clock, 4, &handle)
        .await
        .expect("run completes (the failure is signalled out-of-band, not fatal)");

    match bus.try_recv() {
        Some(BusMessage::NegotiationFailed(NegotiationFailure::EmptyLink { .. })) => {}
        other => panic!("expected NegotiationFailed(EmptyLink), got {other:?}"),
    }
    assert!(
        caps_log.lock().unwrap().is_empty(),
        "no NV12 reached the sink: the infeasible change was rejected, not forwarded"
    );
}

/// The single-transform runner mirrors Caps-α: `run_source_transform_sink`
/// steers the same converter to NV12 on the mid-stream switch, so the legacy
/// fixed-arity path and the N-hop `run_linear_chain` agree.
#[tokio::test]
async fn single_transform_runner_also_steers_to_sink_acceptable_output() {
    let caps_log = Arc::new(Mutex::new(Vec::new()));
    let mut src = ConvSource {
        start: video(RawVideoFormat::Rgba8, 640, 480),
        change_to: video(RawVideoFormat::I420, 640, 480),
        total: 8,
    };
    let mut conv = FormatConverter {
        nv12_from: std::vec![RawVideoFormat::Rgba8, RawVideoFormat::I420],
    };
    let mut sink = RecordingSink { caps_log: Arc::clone(&caps_log) };
    let clock = ZeroClock;

    let stats = run_source_transform_sink(&mut src, &mut conv, &mut sink, &clock, 4)
        .await
        .expect("chain runs");

    assert_eq!(stats.frames_consumed, 8);
    assert_eq!(
        *caps_log.lock().unwrap(),
        std::vec![RawVideoFormat::Nv12],
        "single-transform runner steered the converter to NV12 too"
    );
}
