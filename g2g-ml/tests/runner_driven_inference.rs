//! M451: a runner-driven quantized-inference Graph. The M444 / M447 probes drive
//! the chain by hand (`element.process(...)` per stage); this hands the *same*
//! chain to the g2g runner, which negotiates caps across it and pumps frames
//! through under backpressure, the gst-style orchestration the probes bypass.
//!
//! Chain: `TensorSource(f32 [1,3,224,224]) -> TensorConvert::quantize(u8) ->
//! OrtInference(uint8 MobileNetV2) -> TensorPostprocess::argmax -> ClassSink`,
//! the M447 Edge-TPU chain run on the host CPU EP through `run_linear_chain`.
//! `TensorConvert` is an *interior* node, so this exercises the mid-stream
//! `CapsChanged` each tensor transform forwards (M451 aligned the ML elements to
//! the videoconvert / videoscale CapsChanged contract). Asserts the
//! runner-produced class matches the known top-1 (258, Samoyed).
//!
//! Uses the gitignored uint8-input model from `fixtures/mobilenet/gen_u8in.py`
//! (run `tools/android-mobilenet-tpu-smoke.sh` or `gen_u8in.py` once); skips when
//! absent. Run: cargo test -p g2g-ml --features ort --test runner_driven_inference

#![cfg(feature = "ort")]

use std::path::PathBuf;
use std::pin::Pin;

use g2g_core::element::{AsyncElement, BoxFuture, DynAsyncElement, OutputSink};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_linear_chain, SourceLoop};
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError, PipelineClock, TensorDType, TensorLayout,
    TensorShape,
};
use g2g_ml::ortinfer::OrtInference;
use g2g_ml::postprocess::TensorPostprocess;
use g2g_plugins::tensorconvert::TensorConvert;

use core::future::Future;

const SIZE: u32 = 224;
const EXPECTED_IDX: usize = 258; // Samoyed; the uint8 MobileNetV2's host top-1 (M447)
const DEFAULT_SCALE: f32 = 0.018658;
const DEFAULT_ZERO_POINT: i32 = 114;

fn mobilenet_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mobilenet")
}

fn tensor_caps(dtype: TensorDType, shape: Vec<u32>) -> Caps {
    Caps::Tensor { dtype, shape: TensorShape(shape), layout: TensorLayout::Nchw }
}

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Emits one f32 NCHW tensor frame (from a byte buffer) then EOS, advertising a
/// fixed `Caps::Tensor` output. The minimal source the runner needs to drive the
/// inference chain (AppSrc can't express a tensor caps, which has no launch-syntax
/// media type).
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

/// Captures the winning class index off the argmax `[1,2]` tensor frame.
#[derive(Default)]
struct ClassSink {
    idx: Option<usize>,
}

impl AsyncElement for ClassSink {
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
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    let b = s.as_slice();
                    if b.len() >= 4 {
                        self.idx = Some(f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
                    }
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn runner_drives_quantize_infer_argmax() {
    let dir = mobilenet_dir();
    let model_path = dir.join("mn_u8in.onnx");
    let input_path = dir.join("mn_input_f32.bin");
    if !model_path.exists() || !input_path.exists() {
        eprintln!(
            "uint8 mobilenet fixtures absent ({}); run g2g-ml/tests/fixtures/mobilenet/gen_u8in.py. skipping.",
            dir.display()
        );
        return;
    }
    let (scale, zp) = std::fs::read_to_string(dir.join("u8in_quant.txt"))
        .ok()
        .and_then(|s| {
            let mut it = s.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .unwrap_or((DEFAULT_SCALE, DEFAULT_ZERO_POINT));

    let model = std::fs::read(&model_path).expect("read model");
    let input = std::fs::read(&input_path).expect("read input");

    // The full M447 chain as a *negotiated* graph: the runner solves caps across
    // it and pumps the frame through, including the mid-stream `CapsChanged` each
    // tensor transform emits (which the probes bypass by re-building frames). The
    // f32 input is quantized in-graph by `TensorConvert`, an interior node.
    let mut src = TensorSource { caps: tensor_caps(TensorDType::F32, vec![1, 3, SIZE, SIZE]), data: Some(input) };
    let mut quant = TensorConvert::quantize(TensorDType::U8, scale, zp);
    let mut infer = OrtInference::from_memory(&model).expect("model loads").with_tensor_input();
    let mut argmax = TensorPostprocess::argmax();
    let mut sink = ClassSink::default();

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut quant, &mut infer, &mut argmax];
    let stats = run_linear_chain(&mut src, transforms, &mut sink, &NullClock, 4)
        .await
        .expect("runner drives the quantize -> infer -> argmax chain");

    eprintln!(">> runner-driven chain: {stats:?}; top-1 class = {:?}", sink.idx);
    assert_eq!(sink.idx, Some(EXPECTED_IDX), "runner-produced class matches the known top-1 (Samoyed)");
}
