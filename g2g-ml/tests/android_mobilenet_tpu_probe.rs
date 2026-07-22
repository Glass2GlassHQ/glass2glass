//! M447: a real (non-toy) quantized vision model on the Edge TPU, the device
//! phase of the M446 host milestone. Where M442 / M444 put a [1,3,4,4] toy conv
//! on the TPU, this runs the real MobileNetV2 (ImageNet, [1,3,224,224] ->
//! [1,1000]) through the g2g graph on the Pixel's Edge TPU (DarwiNN).
//!
//! The model is the uint8-input QDQ MobileNetV2 from `fixtures/mobilenet/gen_u8in.py`
//! (the zoo's QOperator int8 model does not place on NNAPI, so it is re-quantized
//! to QDQ per-tensor with a static batch, then the M442 uint8-input surgery removes
//! the float boundary). It is 3.6 MB so it is not committed / `include_bytes!`d
//! (CI cross-compiles this probe): the smoke script pushes the model + a
//! preprocessed f32 input to the device and the probe reads them at runtime.
//!
//! Two tests:
//!   1. `classifier_runs_through_g2g_chain` - the full g2g chain
//!      `TensorConvert::quantize -> OrtInference(uint8, for_android) ->
//!      TensorPostprocess::argmax` runs on-device and yields a valid class.
//!   2. `nnapi_claims_the_classifier` - build the session with NNAPI + XNNPACK,
//!      profile one run on the uint8 input, and read the per-node provider out of
//!      the profiling JSON. Asserts NNAPI claimed node(s) (the conv body on the
//!      Edge TPU); the full split is printed (the Shape/Gather/Reshape classifier
//!      tail may stay on CPU, unlike the all-conv toy of M442).
//!
//! Runs only on `aarch64-linux-android` with `nnapi` + `xnnpack`. Build with
//! cargo-ndk `--platform 27`, push model + input, run. See
//! `tools/android-mobilenet-tpu-smoke.sh`.

#![cfg(all(target_os = "android", feature = "nnapi"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, G2gError, TensorDType, TensorLayout, TensorShape};
use g2g_ml::ortinfer::OrtInference;
use g2g_ml::postprocess::TensorPostprocess;
use g2g_plugins::tensorconvert::{quantize_f32, TensorConvert};

use ::ort::ep::NNAPI;
use ::ort::session::Session;
use ::ort::value::Tensor;

const SIZE: u32 = 224;
const CLASSES: usize = 1000;
/// The model's input quantization (printed by `gen_u8in.py` / `u8in_quant.txt`):
/// a normalized f32 pixel maps to uint8 with this affine, the same one
/// `TensorConvert::quantize` applies. Overridable via env for a regenerated model.
const DEFAULT_SCALE: f32 = 0.018658;
const DEFAULT_ZERO_POINT: i32 = 114;

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn model_path() -> String {
    std::env::var("G2G_MN_MODEL").unwrap_or_else(|_| "/data/local/tmp/mn_u8in.onnx".to_owned())
}

fn input_path() -> String {
    std::env::var("G2G_MN_INPUT").unwrap_or_else(|_| "/data/local/tmp/mn_input_f32.bin".to_owned())
}

fn quant() -> (f32, i32) {
    (
        env_or("G2G_MN_SCALE", DEFAULT_SCALE),
        env_or("G2G_MN_ZERO_POINT", DEFAULT_ZERO_POINT),
    )
}

/// Start a binder threadpool so the NNAPI vendor HAL (DarwiNN, over binder/AIDL)
/// can bring up the accelerator from this bare native binary. Same shim as the
/// MediaCodec / M439-M442 probes.
fn start_binder_threadpool() {
    use core::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;
    // SAFETY: libbinder_ndk.so is loadable; the dlsym'd symbols match
    // <android/binder_process.h>.
    unsafe {
        let lib = dlopen(b"libbinder_ndk.so\0".as_ptr() as *const c_char, RTLD_NOW);
        if lib.is_null() {
            return;
        }
        let set = dlsym(
            lib,
            b"ABinderProcess_setThreadPoolMaxThreadCount\0".as_ptr() as *const c_char,
        );
        if !set.is_null() {
            let set: extern "C" fn(u32) -> bool = core::mem::transmute(set);
            set(1);
        }
        let start = dlsym(
            lib,
            b"ABinderProcess_startThreadPool\0".as_ptr() as *const c_char,
        );
        if !start.is_null() {
            let start: extern "C" fn() = core::mem::transmute(start);
            start();
        }
    }
}

/// Tally the provider of every node compute event in an ORT profiling JSON
/// (whitespace-tolerant scan, no JSON dep). NNAPI fuses its claimed subgraph into
/// one node tagged `NnapiExecutionProvider`.
fn providers_from_profiling(json: &str) -> std::collections::BTreeMap<String, usize> {
    let mut providers: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    let mut rest = json;
    while let Some(i) = rest.find("\"provider\"") {
        rest = &rest[i + "\"provider\"".len()..];
        let Some(open) = rest.find('"') else { break };
        rest = &rest[open + 1..];
        if let Some(end) = rest.find('"') {
            let prov = &rest[..end];
            if !prov.is_empty() {
                *providers.entry(prov.to_owned()).or_insert(0) += 1;
            }
            rest = &rest[end + 1..];
        }
    }
    providers
}

#[derive(Default)]
struct OneFrame {
    bytes: Option<Vec<u8>>,
}

impl OutputSink for OneFrame {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if self.bytes.is_none() {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.bytes = Some(s.to_vec());
                    }
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame(bytes: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    })
}

fn tensor_caps(dtype: TensorDType, shape: &[u32]) -> Caps {
    Caps::Tensor {
        dtype,
        shape: TensorShape::from_slice(shape).unwrap(),
        layout: TensorLayout::Nchw,
    }
}

fn read_input_f32_bytes() -> Vec<u8> {
    let bytes = std::fs::read(input_path()).expect("read pushed f32 input (mn_input_f32.bin)");
    assert_eq!(
        bytes.len(),
        3 * SIZE as usize * SIZE as usize * 4,
        "f32 NCHW [1,3,224,224]"
    );
    bytes
}

/// The full g2g chain on-device: `TensorConvert::quantize -> OrtInference(uint8)
/// -> TensorPostprocess::argmax` yields a valid ImageNet class.
#[tokio::test]
async fn classifier_runs_through_g2g_chain() {
    start_binder_threadpool();
    let model = std::fs::read(model_path()).expect("read pushed model (mn_u8in.onnx)");
    let input_bytes = read_input_f32_bytes();
    let (scale, zp) = quant();

    // Stage 1: quantize the normalized f32 tensor to the model's uint8 input.
    let mut quantize = TensorConvert::quantize(TensorDType::U8, scale, zp);
    quantize
        .configure_pipeline(&tensor_caps(TensorDType::F32, &[1, 3, SIZE, SIZE]))
        .expect("configure quantize");
    let mut q_out = OneFrame::default();
    quantize
        .process(frame(input_bytes), &mut q_out)
        .await
        .expect("quantize runs");
    let u8_tensor = q_out.bytes.expect("uint8 tensor");
    assert_eq!(
        u8_tensor.len(),
        3 * SIZE as usize * SIZE as usize,
        "one byte per element"
    );

    // Stage 2: the real MobileNetV2 on NNAPI (-> Edge TPU), CPU as the implicit
    // fallback. XNNPACK is intentionally not registered: it rejects the int8 QDQ
    // classifier-tail initializers ("weight_quantized type: 3"), where NNAPI fuses
    // the whole graph onto the TPU (the M442 toy had no tail, so XNNPACK was fine).
    let mut infer = OrtInference::from_memory_with_nnapi(&model)
        .expect("uint8 model loads")
        .with_tensor_input();
    infer
        .configure_pipeline(&tensor_caps(TensorDType::U8, &[1, 3, SIZE, SIZE]))
        .expect("configure inference");
    let mut logits = OneFrame::default();
    infer
        .process(frame(u8_tensor), &mut logits)
        .await
        .expect("inference runs on-device");
    let logit_bytes = logits.bytes.expect("logits tensor");
    assert_eq!(logit_bytes.len(), CLASSES * 4, "1000 f32 logits");

    // Stage 3: classification head -> winning [index, value].
    let mut argmax = TensorPostprocess::argmax();
    argmax
        .configure_pipeline(&tensor_caps(TensorDType::F32, &[1, CLASSES as u32]))
        .expect("configure argmax");
    let mut top = OneFrame::default();
    argmax
        .process(frame(logit_bytes), &mut top)
        .await
        .expect("argmax runs");
    let top_bytes = top.bytes.expect("argmax output");
    let idx = f32::from_le_bytes([top_bytes[0], top_bytes[1], top_bytes[2], top_bytes[3]]) as usize;

    assert!(idx < CLASSES, "class index in range");
    eprintln!(">> MobileNetV2 ran on the Android EP stack via the g2g chain; top-1 = class {idx}");
}

/// Build the session with NNAPI + XNNPACK, profile one run on the uint8 input, and
/// read which EP claimed each node. Asserts NNAPI took node(s): the conv body ran
/// on the Edge TPU accelerator. The classifier tail (Shape/Gather/Reshape/MatMul)
/// may remain on CPU, so the split is reported, not asserted to be fully on NNAPI.
#[tokio::test]
async fn nnapi_claims_the_classifier() {
    start_binder_threadpool();
    let model = std::fs::read(model_path()).expect("read pushed model");
    let (scale, zp) = quant();

    // The uint8 input TensorConvert::quantize would produce.
    let f32_bytes = read_input_f32_bytes();
    let floats: Vec<f32> = f32_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let u8_input = quantize_f32(&floats, TensorDType::U8, scale, zp).expect("quantize to u8");

    let profile_prefix = "/data/local/tmp/g2g_mobilenet_profile";
    let mut session = Session::builder()
        .expect("builder")
        .with_execution_providers([NNAPI::default().build()])
        .expect("register NNAPI")
        .with_profiling(profile_prefix)
        .expect("enable profiling")
        .commit_from_memory(&model)
        .expect("session builds on-device");

    let input_name = session.inputs()[0].name().to_owned();
    let out_len = {
        let t = Tensor::from_array((vec![1i64, 3, SIZE as i64, SIZE as i64], u8_input))
            .expect("u8 tensor");
        let outputs = session
            .run(::ort::inputs![input_name.as_str() => t])
            .expect("run on-device");
        let (_s, out) = outputs[0].try_extract_tensor::<f32>().expect("f32 output");
        out.len()
    };
    assert_eq!(out_len, CLASSES, "[1,1000] logits");

    let profile_path = session.end_profiling().expect("flush profiling");
    let json = std::fs::read_to_string(&profile_path).expect("read profiling json");
    let _ = std::fs::remove_file(&profile_path);

    let providers = providers_from_profiling(&json);
    eprintln!("--- ORT node placement (provider -> node count) ---");
    for (prov, n) in &providers {
        eprintln!("    {prov}: {n}");
    }
    let nnapi: usize = providers
        .iter()
        .filter(|(p, _)| p.contains("Nnapi"))
        .map(|(_, n)| *n)
        .sum();
    let non_nnapi: usize = providers
        .iter()
        .filter(|(p, _)| !p.contains("Nnapi"))
        .map(|(_, n)| *n)
        .sum();
    eprintln!(">> {nnapi} node(s) on NNAPI (Edge TPU), {non_nnapi} off-accelerator");
    assert!(
        nnapi > 0,
        "NNAPI claimed no node, so nothing ran on the Edge TPU; placement: {providers:?}"
    );
}
