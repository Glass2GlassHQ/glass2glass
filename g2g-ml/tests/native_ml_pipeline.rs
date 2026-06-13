//! M61: native end-to-end ML pipeline on real hardware, the native half of the
//! cross-target proof. `VideoTestSrc(RGBA) -> VideoConvert(NV12) ->
//! VideoScale(NV12) -> WgpuPreprocess(GPU) -> OrtInference(tensor-input) ->
//! FakeSink` shows the software transforms compose with the GPU preprocess and
//! the tensor-input inference into one negotiated chain, the same element graph
//! shape the browser pipeline runs (substituting platform source/decode/sink).
//!
//! Run: cargo test -p g2g-ml --features "wgpu ort" --test native_ml_pipeline

#![cfg(all(feature = "wgpu", feature = "ort"))]

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::run_linear_chain;
use g2g_core::{Caps, PipelineClock, RawVideoFormat, TensorDType, TensorLayout, TensorShape};
use g2g_ml::ortinfer::OrtInference;
use g2g_ml::wgpupreprocess::{gpu_available, WgpuPreprocess};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videoscale::VideoScale;
use g2g_plugins::videotestsrc::VideoTestSrc;

// shared hand-encoded ONNX fixture builder (tests/util/onnx_fixture.rs)
mod onnx {
    include!("util/onnx_fixture.rs");
}
use onnx::identity_model;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn native_ml_pipeline_negotiates_and_runs_on_gpu() {
    if !gpu_available().await {
        eprintln!("no wgpu adapter on this host; skipping native ML pipeline test");
        return;
    }

    // identity model at the preprocess output geometry [1, 3, 2, 2].
    let model = identity_model(&[1, 3, 2, 2]);

    let mut src = VideoTestSrc::new(4, 4, 30, 3); // 4x4 RGBA
    let mut convert = VideoConvert::new(RawVideoFormat::Nv12); // -> NV12 4x4
    let mut scale = VideoScale::new(2, 2); // -> NV12 2x2
    let mut preprocess = WgpuPreprocess::new(); // -> f32 tensor [1,3,2,2] on the GPU
    let mut infer = OrtInference::from_memory(&model)
        .expect("model loads")
        .with_tensor_input();
    let mut sink = FakeSink::new();

    let transforms: Vec<&mut dyn DynAsyncElement> =
        vec![&mut convert, &mut scale, &mut preprocess, &mut infer];

    run_linear_chain(&mut src, transforms, &mut sink, &NullClock, 4)
        .await
        .expect("RGBA -> NV12 -> scale -> GPU preprocess -> tensor inference negotiates and flows");

    assert_eq!(sink.received(), 3, "every frame reaches the inference output");
    assert!(sink.eos_seen());
    let changes = sink.caps_changes();
    assert!(
        changes.iter().any(|c| matches!(
            &c.caps,
            Caps::Tensor { dtype: TensorDType::F32, layout: TensorLayout::Nchw, shape }
            if *shape == TensorShape(vec![1, 3, 2, 2])
        )),
        "sink saw the model's [1,3,2,2] tensor caps, got {changes:?}"
    );
}
