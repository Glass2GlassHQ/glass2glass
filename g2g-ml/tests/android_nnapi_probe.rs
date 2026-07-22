//! M439: on-device probe for the Android NNAPI / XNNPACK ONNX Runtime EPs.
//!
//! Builds an `OrtInference` through `from_memory_for_android` (NNAPI accelerator
//! preferred, then XNNPACK ARM CPU, then ORT's default CPU EP), runs one frame,
//! and asserts the output is byte-exact with the CPU reference (an Identity model,
//! so the output equals the normalized input). This proves the Android ORT build
//! actually carries the NNAPI / XNNPACK symbols, the EPs register, and the session
//! runs on the device without crashing. Whether NNAPI *offloaded* the (trivial)
//! Identity op to an accelerator is reported, not asserted, the same best-effort
//! contract as the CUDA EP test (a Conv/Gemm fixture that the NPU actually runs is
//! a follow-up).
//!
//! Runs only on `aarch64-linux-android` (et al.) with the `nnapi` + `xnnpack`
//! features; compiles to nothing on the dev host (the EP symbols are not in a
//! host ORT, so the features are Android-target only). Build with cargo-ndk
//! `--platform 27`, push, run as a bare native binary. See
//! `tools/android-nnapi-smoke.sh`.

#![cfg(all(target_os = "android", feature = "nnapi", feature = "xnnpack"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Dim, G2gError, Rate, RawVideoFormat};
use g2g_ml::ortinfer::OrtInference;

// Shared hand-encoded ONNX fixture builder (tests/util/onnx_fixture.rs): a single
// Identity node, so the expected output is exactly the normalized input.
mod onnx {
    include!("util/onnx_fixture.rs");
}
use onnx::identity_model;

/// Start a binder threadpool so an NNAPI vendor HAL driver (reached over
/// binder/AIDL) can call back into this process to set up the accelerator. A bare
/// native binary from /data/local/tmp has no threadpool, so an accelerator
/// transaction would stall; the `nnapi-reference` CPU driver does not need it, but
/// the NPU / GPU drivers (the point of NNAPI) do. The `ABinderProcess_*` symbols
/// live in the device's libbinder_ndk.so but not the NDK link stub, so dlsym them.
/// Same shim as the MediaCodec probes.
fn start_binder_threadpool() {
    use core::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;
    // SAFETY: libbinder_ndk.so is loadable; the dlsym'd symbols have the C
    // signatures from <android/binder_process.h>.
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

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

fn rgba_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
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

/// The Android edge EP stack (NNAPI -> XNNPACK -> CPU) loads, registers, and runs
/// a model on the device, producing the byte-exact CPU-reference output.
#[tokio::test]
async fn android_ep_stack_infers_identically() {
    start_binder_threadpool();

    let model = identity_model(&[1, 3, 2, 2]);
    let mut inf = OrtInference::from_memory_for_android(&model)
        .expect("Android ORT carries the NNAPI/XNNPACK symbols and the session builds");
    assert_eq!(inf.input_dims(), (2, 2));
    let narrowed = inf.intercept_caps(&rgba_caps(2, 2)).expect("2x2 accepted");
    inf.configure_pipeline(&narrowed).expect("configure");

    let mut sink = Collect::default();
    inf.process(
        PipelinePacket::DataFrame(rgba_frame((0..16).collect(), 0)),
        &mut sink,
    )
    .await
    .expect("frame runs on-device");

    let values: Vec<f32> = sink
        .packets
        .iter()
        .find_map(|p| match p {
            PipelinePacket::DataFrame(f) => {
                let Some(slice) = f.domain.as_system_slice() else {
                    return None;
                };
                Some(
                    slice
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect(),
                )
            }
            _ => None,
        })
        .expect("one tensor frame out");

    // Identity model: output = RGB planes / 255 in NCHW order.
    let expected: Vec<f32> = [0u8, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14]
        .iter()
        .map(|b| *b as f32 / 255.0)
        .collect();
    assert_eq!(
        values, expected,
        "Android EP output matches the CPU reference"
    );
    eprintln!(
        "android NNAPI/XNNPACK EP stack ran; output byte-exact with CPU \
         (EP node assignment is logged by ORT at verbose level)"
    );
}
