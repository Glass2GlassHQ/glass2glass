//! M452: the one-graph detector. The detection sibling of the M451 runner-driven
//! classifier: a real YOLO and its perception post-processing run as a single
//! negotiated g2g graph through `run_linear_chain`, not hand-driven element by
//! element. This is the "MediaPipe but one typed pipeline" story made tangible -
//! `decode/source -> inference -> NMS/decode -> structured detections`, the runner
//! owning caps negotiation and frame pumping across every tensor hop (unblocked by
//! M451's interior-tensor-node fix).
//!
//! Chain: `TensorSource(f32 [1,3,640,640]) -> OrtInference(YOLO) ->
//! DetectionPostprocess -> DetectSink`. The model's raw `[1, 84, 8400]` tensor
//! becomes an `AnalyticsMeta` of bounding boxes (anchor decode + per-class NMS),
//! attached to the frame the sink reads. Asserts the dog is detected.
//!
//! Uses the gitignored YOLO fixtures from `fixtures/detect/gen.py`
//! (`tools/detect-fixture.sh`); skips when absent. Run:
//!   cargo test -p g2g-ml --features "ort analytics" --test runner_driven_detect -- --nocapture

#![cfg(all(feature = "ort", feature = "analytics"))]

use std::path::PathBuf;
use std::pin::Pin;

use g2g_core::element::{AsyncElement, BoxFuture, DynAsyncElement, OutputSink};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_linear_chain, SourceLoop};
use g2g_core::{
    AnalyticsMeta, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, PipelineClock, TensorDType,
    TensorLayout, TensorShape,
};
use g2g_ml::detect::DetectionPostprocess;
use g2g_ml::ortinfer::OrtInference;

use core::future::Future;

const SIZE: u32 = 640;
const COCO_DOG: u32 = 16;

fn detect_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/detect")
}

fn tensor_caps(dtype: TensorDType, shape: &[u32]) -> Caps {
    Caps::Tensor { dtype, shape: TensorShape::from_slice(shape).unwrap(), layout: TensorLayout::Nchw }
}

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Emits one f32 NCHW tensor frame then EOS, advertising a fixed `Caps::Tensor`.
struct TensorSource {
    caps: Caps,
    data: Option<Vec<u8>>,
}

impl SourceLoop for TensorSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(self.caps.clone()))
    }

    fn caps_constraint(&mut self) -> impl Future<Output = Result<CapsConstraint<'_>, G2gError>> {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps.clone()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let data = self.data.take().ok_or(G2gError::NotConfigured)?;
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
                timing: FrameTiming::default(),
                sequence: 0,
                meta: Default::default(),
            };
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Records the detections off the frame's `AnalyticsMeta` (the terminal sink).
#[derive(Default)]
struct DetectSink {
    dets: Vec<(u32, f32)>,
}

impl AsyncElement for DetectSink {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(&'a mut self, packet: PipelinePacket, _out: &'a mut dyn OutputSink) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = &packet {
                if let Some(a) = f.meta.get::<AnalyticsMeta>() {
                    self.dets = a.detections().map(|d| (d.label, d.confidence)).collect();
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn runner_drives_detect_chain() {
    let dir = detect_dir();
    let model_path = dir.join("model.onnx");
    let input_path = dir.join("input_f32.bin");
    if !model_path.exists() || !input_path.exists() {
        eprintln!("detect fixtures absent ({}); run tools/detect-fixture.sh. skipping.", dir.display());
        return;
    }
    let model = std::fs::read(&model_path).expect("read model");
    let input = std::fs::read(&input_path).expect("read input");

    let mut src = TensorSource { caps: tensor_caps(TensorDType::F32, &[1, 3, SIZE, SIZE]), data: Some(input) };
    let mut infer = OrtInference::from_memory(&model).expect("model loads").with_tensor_input();
    let mut decode = DetectionPostprocess::new(0.25, 0.45).with_input_size(SIZE, SIZE);
    let mut sink = DetectSink::default();

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut infer, &mut decode];
    let stats = run_linear_chain(&mut src, transforms, &mut sink, &NullClock, 4)
        .await
        .expect("runner drives the infer -> detect chain");

    eprintln!(">> one-graph detector: {stats:?}; detections = {:?}", sink.dets);
    assert!(!sink.dets.is_empty(), "expected at least one detection");
    let dog = sink.dets.iter().find(|(label, _)| *label == COCO_DOG);
    let (_, conf) = dog.expect("a 'dog' (COCO class 16) detection through the negotiated graph");
    assert!(*conf > 0.5, "dog detected with confidence > 0.5, got {conf}");
    eprintln!(">> the whole detector ran as one negotiated graph; dog @ {conf:.3}");
}
