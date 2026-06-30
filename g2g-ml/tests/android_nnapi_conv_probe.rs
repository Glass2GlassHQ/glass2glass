//! M440: on-device proof that the Android NNAPI EP (Edge TPU / DarwiNN) actually
//! runs a quantized convolution, the follow-up to the M439 Identity probe (which
//! proved only that the EP path links / registers / runs, not that an accelerator
//! executed anything).
//!
//! The fixture is an int8 QDQ Conv->ReLU model (`fixtures/qconv_relu_int8.onnx`,
//! built by `fixtures/gen_qconv.py`): int8 quantization is what ORT's NNAPI EP
//! folds into a quantized conv the Edge TPU can run, where an fp32 / Identity model
//! is left on the CPU/GPU. Two tests:
//!   1. `conv_runs_through_android_ep_stack` - the quantized conv runs end to end
//!      through `OrtInference::from_memory_for_android` and produces a correct-shape,
//!      ReLU-nonnegative tensor (the element handles a real conv, not just Identity).
//!   2. `nnapi_claims_the_quantized_conv` - build the session with NNAPI + XNNPACK,
//!      enable ORT profiling, run once, and read the per-node provider out of the
//!      profiling JSON. Asserts `NnapiExecutionProvider` claimed node(s): proof the
//!      conv ran on the NNAPI accelerator, not the CPU fallback. The full per-EP
//!      breakdown is printed either way (run with `--nocapture`); pair it with the
//!      `edgetpu` / `darwinn` logcat lines the smoke script greps to confirm the TPU
//!      specifically (NNAPI itself only reports "NNAPI", not which sub-accelerator).
//!
//! Runs only on `aarch64-linux-android` (et al.) with `nnapi` + `xnnpack`; compiles
//! to nothing on the dev host. Build with cargo-ndk `--platform 27`, push, run as a
//! bare native binary. See `tools/android-nnapi-conv-smoke.sh`.

#![cfg(all(target_os = "android", feature = "nnapi", feature = "xnnpack"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Dim, G2gError, RawVideoFormat, Rate};
use g2g_ml::ortinfer::OrtInference;

use ::ort::ep::{NNAPI, XNNPACK};
use ::ort::session::Session;
use ::ort::value::Tensor;

/// The committed int8 QDQ Conv->ReLU fixture (input f32 [1,3,4,4] -> [1,4,4,4]).
const QCONV: &[u8] = include_bytes!("fixtures/qconv_relu_int8.onnx");

/// Start a binder threadpool so the NNAPI vendor HAL (DarwiNN, reached over
/// binder/AIDL) can set up the accelerator from this bare native binary. Same shim
/// as the MediaCodec / M439 probes.
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
        let set = dlsym(lib, b"ABinderProcess_setThreadPoolMaxThreadCount\0".as_ptr() as *const c_char);
        if !set.is_null() {
            let set: extern "C" fn(u32) -> bool = core::mem::transmute(set);
            set(1);
        }
        let start = dlsym(lib, b"ABinderProcess_startThreadPool\0".as_ptr() as *const c_char);
        if !start.is_null() {
            let start: extern "C" fn() = core::mem::transmute(start);
            start();
        }
    }
}

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}
impl OutputSink for Collect {
    fn push<'a>(&'a mut self, packet: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

fn rgba_frame(bytes: Vec<u8>, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence,
        meta: Default::default(),
    }
}

/// The quantized conv runs end-to-end through the Android EP stack and produces a
/// correct-shape, ReLU-nonnegative output (the element handles a real conv).
#[tokio::test]
async fn conv_runs_through_android_ep_stack() {
    start_binder_threadpool();

    let mut inf = OrtInference::from_memory_for_android(QCONV).expect("quantized conv model loads");
    // OrtInference reads the model's static [N,3,H,W] input geometry as W x H RGBA.
    assert_eq!(inf.input_dims(), (4, 4));
    let caps = Caps::RawVideo { format: RawVideoFormat::Rgba8, width: Dim::Fixed(4), height: Dim::Fixed(4), framerate: Rate::Any };
    let narrowed = inf.intercept_caps(&caps).expect("4x4 accepted");
    inf.configure_pipeline(&narrowed).expect("configure");

    let mut sink = Collect::default();
    // 4x4 RGBA, an arbitrary gradient.
    let rgba: Vec<u8> = (0..(4 * 4 * 4) as u16).map(|p| (p * 3) as u8).collect();
    inf.process(PipelinePacket::DataFrame(rgba_frame(rgba, 0)), &mut sink)
        .await
        .expect("conv inference runs on-device");

    let out: Vec<f32> = sink
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                let MemoryDomain::System(slice) = &f.domain else { return None };
                Some(slice.as_slice().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
            }
            _ => None,
        })
        .expect("one tensor frame out");

    // Output is [1,4,4,4] = 64 floats, post-ReLU so non-negative and finite.
    assert_eq!(out.len(), 64, "conv output [1,4,4,4]");
    assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0), "ReLU output finite + non-negative");
    eprintln!("quantized Conv->ReLU ran through the Android EP stack, output [1,4,4,4] ok");
}

/// Build the session with NNAPI + XNNPACK, profile one run, and read which EP
/// claimed each node out of the profiling JSON. Asserts NNAPI took node(s): proof
/// the quantized conv ran on the accelerator, not the CPU fallback.
#[tokio::test]
async fn nnapi_claims_the_quantized_conv() {
    start_binder_threadpool();

    // ORT appends `_<timestamp>.json`; end_profiling returns the full path.
    let profile_prefix = "/data/local/tmp/g2g_qconv_profile";
    let mut session = Session::builder()
        .expect("builder")
        .with_execution_providers([NNAPI::default().build(), XNNPACK::default().build()])
        .expect("register NNAPI + XNNPACK")
        .with_profiling(profile_prefix)
        .expect("enable profiling")
        .commit_from_memory(QCONV)
        .expect("session builds on-device");

    let input_name = session.inputs()[0].name().to_owned();
    let data: Vec<f32> = (0..48).map(|i| (i as f32) / 48.0).collect(); // [1,3,4,4]
    let tensor = Tensor::from_array((vec![1i64, 3, 4, 4], data)).expect("input tensor");
    // Scope the run so `outputs` (which borrows `session`) drops before
    // `end_profiling` takes its own &mut borrow.
    let out_len = {
        let outputs = session.run(::ort::inputs![input_name.as_str() => tensor]).expect("run on-device");
        let (_shape, data) = outputs[0].try_extract_tensor::<f32>().expect("f32 output");
        data.len()
    };
    assert_eq!(out_len, 64, "conv output [1,4,4,4] = 64 floats");

    let profile_path = session.end_profiling().expect("flush profiling");
    let json = std::fs::read_to_string(&profile_path).expect("read profiling json");
    let _ = std::fs::remove_file(&profile_path);

    // Tally the provider of every node compute event (crude scan, no JSON dep, and
    // whitespace-tolerant: ORT writes `"provider": "..."`). NNAPI fuses its claimed
    // subgraph into one node tagged NnapiExecutionProvider.
    let mut providers: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut rest = json.as_str();
    while let Some(i) = rest.find("\"provider\"") {
        rest = &rest[i + "\"provider\"".len()..];
        // skip `:` and whitespace up to the opening quote of the value
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

    if providers.is_empty() {
        // Diagnostic: show the profiling shape so we can fix the scan if the schema
        // differs (the file should be a non-empty Chrome-trace JSON array).
        eprintln!("(no providers parsed; profiling json is {} bytes, head: {})",
            json.len(), &json[..json.len().min(400)]);
    }
    eprintln!("--- ORT node placement (provider -> node count) ---");
    for (prov, n) in &providers {
        eprintln!("    {prov}: {n}");
    }
    let nnapi_nodes = providers.iter().filter(|(p, _)| p.contains("Nnapi")).map(|(_, n)| *n).sum::<usize>();
    if nnapi_nodes > 0 {
        eprintln!(">> NNAPI claimed {nnapi_nodes} node(s): the quantized conv ran on the NNAPI accelerator");
    } else {
        eprintln!(">> NNAPI claimed NO nodes: the conv fell back (see the breakdown above)");
    }
    assert!(
        nnapi_nodes > 0,
        "NNAPI did not claim any node, so the Edge TPU did not run the conv; placement: {providers:?}"
    );
}
