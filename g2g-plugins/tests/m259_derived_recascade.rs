//! M259 driver: a downstream `DerivedOutput` that genuinely re-derives its
//! output, stacked below another format-changing `DerivedOutput`, on a
//! mid-stream input change. This is the "decoder below another format-changing
//! transform" case the Caps-β forward walk was gated on.
//!
//! Topology: source -> convert(`DerivedOutput`: Rgba8 -> I420, geometry
//! passthrough) -> scale(`DerivedOutput`: I420 WxH -> I420 W/2 x H/2, geometry
//! *re-derived*, format passthrough) -> sink(accepts I420 at any geometry).
//! The source flips geometry mid-stream; the scaler must re-derive its halved
//! output for the new input and the runner must cascade the re-derived caps to
//! the sink. The sink records the full caps it applied so the test can prove
//! the re-derivation reached it under both geometries.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};
use std::vec::Vec;

use g2g_core::element::DynAsyncElement;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_linear_chain, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    MemoryDomain, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn raw(fmt: RawVideoFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: fmt,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn i420_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::I420,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// Source: produces `start`, emits one frame, pushes a mid-stream
/// `CapsChanged(change_to)`, then the rest + EOS.
struct FlipSource {
    start: Caps,
    change_to: Caps,
    total: u32,
}

impl SourceLoop for FlipSource {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.start.clone()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(
            self.start.clone(),
        ))))
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
                    out.push(PipelinePacket::CapsChanged(self.change_to.clone()))
                        .await?;
                }
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.total as u64)
        })
    }
}

/// Format converter: Rgba8 -> I420 at the same geometry, a pure
/// `DerivedOutput` (single output per input). Geometry is passthrough.
struct Converter;

impl AsyncElement for Converter {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                width,
                height,
                framerate,
                ..
            } => CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::I420,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
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

/// Geometry-re-deriving scaler: I420 WxH -> I420 W/2 x H/2, format passthrough.
/// A genuine `DerivedOutput` whose geometry output is *not* a passthrough of its
/// input, so a downstream geometry pin cannot couple back through it: the
/// mid-stream re-derivation is the whole point.
struct Halver;

impl AsyncElement for Halver {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                framerate,
            } => CapsSet::one(Caps::RawVideo {
                format: *format,
                width: Dim::Fixed(w / 2),
                height: Dim::Fixed(h / 2),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
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

/// Sink accepting I420 at any geometry; records the full caps of every
/// `CapsChanged` it applies.
struct RecordingSink {
    caps_log: Arc<Mutex<Vec<Caps>>>,
}

impl AsyncElement for RecordingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(i420_any()))
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
            if let PipelinePacket::CapsChanged(c) = packet {
                log.lock().unwrap().push(c);
            }
            Ok(())
        })
    }
}

/// The downstream `DerivedOutput` (the halver) re-derives its halved output for
/// the new input on the mid-stream geometry flip, and the runner cascades it to
/// the sink. Startup caps reach the sink via `configure_pipeline` (not a
/// `CapsChanged` packet), so the log holds only the mid-stream re-derivation:
/// I420 640x360, the halved 1280x720 the source switched to. Before the snapshot
/// widened non-passthrough fields, the converter's frozen-geometry snapshot
/// rejected the re-derived I420 1280x720 and nothing reached the sink.
#[tokio::test]
async fn midstream_change_redrives_through_stacked_derived_outputs() {
    let caps_log = Arc::new(Mutex::new(Vec::new()));
    let mut src = FlipSource {
        start: raw(RawVideoFormat::Rgba8, 640, 480),
        change_to: raw(RawVideoFormat::Rgba8, 1280, 720),
        total: 8,
    };
    let mut conv = Converter;
    let mut scale = Halver;
    let mut sink = RecordingSink {
        caps_log: Arc::clone(&caps_log),
    };
    let clock = ZeroClock;

    let transforms: Vec<&mut dyn DynAsyncElement> = std::vec![&mut conv, &mut scale];
    let stats = run_linear_chain(&mut src, transforms, &mut sink, &clock, 4)
        .await
        .expect("chain runs");

    assert_eq!(
        stats.frames_consumed, 8,
        "every frame crosses both transforms"
    );
    assert_eq!(
        *caps_log.lock().unwrap(),
        std::vec![raw(RawVideoFormat::I420, 640, 360)],
        "the scaler re-derived its halved output for the new 1280x720 input",
    );
}
