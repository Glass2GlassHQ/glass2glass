//! M443: the live chain - Android camera -> quantize -> Edge TPU inference, the
//! pieces of M308 / M441 / M442 wired end to end on the device.
//!
//! `Camera2Src` captures real NV12 frames; one frame's luminance is downsampled to
//! the model's [1,3,4,4] geometry and normalized to f32 [0,1]; `TensorConvert`
//! quantizes that to the uint8 the quantized model wants; `OrtInference`
//! (`from_memory_for_android` + `with_tensor_input`) runs the uint8 Conv->ReLU,
//! which M442 proved runs entirely on the Edge TPU. So a real camera pixel reaches
//! the NPU through the g2g graph, no float boundary on the CPU.
//!
//! Camera capture needs the `CAMERA` permission; an `adb shell` run (shell uid)
//! has it, so this captures live. If the camera cannot open (a stricter sandbox),
//! the probe falls back to a synthetic frame so the quantize -> TPU chain is still
//! exercised, and reports which input it used.
//!
//! Runs only on `aarch64-linux-android` with the `camera2-tpu` feature. Build with
//! cargo-ndk `--platform 27`, push, run. See `tools/android-camera-tpu-smoke.sh`.

#![cfg(all(target_os = "android", feature = "camera2-tpu"))]

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, G2gError, TensorDType, TensorLayout, TensorShape};
use g2g_ml::ortinfer::OrtInference;
use g2g_plugins::camera2src::Camera2Src;
use g2g_plugins::tensorconvert::TensorConvert;

/// The uint8-input QDQ Conv->ReLU model (M442): runs 100% on the Edge TPU.
const QCONV_U8IN: &[u8] = include_bytes!("fixtures/qconv_relu_u8in.onnx");
/// The model's input quantization (printed by `gen_qconv.py`): a normalized f32
/// pixel maps to uint8 with this affine, so `TensorConvert::quantize` matches it.
const IN_SCALE: f32 = 0.003918;
const IN_ZERO_POINT: i32 = 0;

const CAM_W: u32 = 640;
const CAM_H: u32 = 480;
const MODEL_HW: usize = 4; // the model's input is [1,3,4,4]

fn start_binder_threadpool() {
    use core::ffi::{c_char, c_int, c_void};
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }
    const RTLD_NOW: c_int = 2;
    // SAFETY: libbinder_ndk.so is loadable; symbols match <android/binder_process.h>.
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

/// Captures the first NV12 frame's bytes (the camera source's output).
#[derive(Default)]
struct FrameGrab {
    first: Option<Vec<u8>>,
    count: u64,
}
impl OutputSink for FrameGrab {
    fn push<'a>(&'a mut self, packet: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    if self.first.is_none() {
                        self.first = Some(s.as_slice().to_vec());
                    }
                    self.count += 1;
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Grabs the first DataFrame an element pushes (to chain element outputs by hand).
#[derive(Default)]
struct OneFrame {
    bytes: Option<Vec<u8>>,
}
impl OutputSink for OneFrame {
    fn push<'a>(&'a mut self, packet: PipelinePacket) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if self.bytes.is_none() {
                    if let MemoryDomain::System(s) = &f.domain {
                        self.bytes = Some(s.as_slice().to_vec());
                    }
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn f32_frame(values: &[f32]) -> Frame {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

fn tensor_caps(dtype: TensorDType) -> Caps {
    Caps::Tensor { dtype, shape: TensorShape(vec![1, 3, MODEL_HW as u32, MODEL_HW as u32]), layout: TensorLayout::Nchw }
}

/// Downsample an NV12 frame's Y (luminance) plane to a `[1,3,4,4]` f32 tensor in
/// [0,1], sampling one luma at the center of each 4x4 grid cell and replicating it
/// across the 3 channels (a real camera pixel per cell; the toy model is about the
/// plumbing, not color fidelity).
fn nv12_to_model_tensor(nv12: &[u8]) -> Vec<f32> {
    let (w, h) = (CAM_W as usize, CAM_H as usize);
    let mut chw = vec![0f32; 3 * MODEL_HW * MODEL_HW];
    for gy in 0..MODEL_HW {
        for gx in 0..MODEL_HW {
            let px = gx * (w / MODEL_HW) + (w / MODEL_HW) / 2;
            let py = gy * (h / MODEL_HW) + (h / MODEL_HW) / 2;
            let luma = nv12.get(py * w + px).copied().unwrap_or(0) as f32 / 255.0;
            let cell = gy * MODEL_HW + gx;
            for c in 0..3 {
                chw[c * MODEL_HW * MODEL_HW + cell] = luma;
            }
        }
    }
    chw
}

/// Capture one live NV12 frame, or `None` if the camera could not open.
async fn capture_one() -> Option<Vec<u8>> {
    let mut src = Camera2Src::new(CAM_W, CAM_H, 3);
    let caps = Caps::RawVideo {
        format: g2g_core::RawVideoFormat::Nv12,
        width: g2g_core::Dim::Fixed(CAM_W),
        height: g2g_core::Dim::Fixed(CAM_H),
        framerate: g2g_core::Rate::Any,
    };
    if let Err(e) = src.configure_pipeline(&caps) {
        eprintln!(">>> camera open failed ({e:?}); falling back to a synthetic frame");
        return None;
    }
    let mut grab = FrameGrab::default();
    match src.run(&mut grab).await {
        Ok(_) => {
            eprintln!(">>> captured {} live camera frame(s)", grab.count);
            grab.first
        }
        Err(e) => {
            eprintln!(">>> capture failed after open ({e:?}); synthetic fallback");
            None
        }
    }
}

#[tokio::test]
async fn live_camera_quantize_to_edge_tpu() {
    start_binder_threadpool();

    // 1. A real camera frame, or a synthetic NV12 gradient if the camera is denied.
    let (nv12, live) = match capture_one().await {
        Some(f) if f.len() >= (CAM_W * CAM_H) as usize => (f, true),
        _ => {
            let synth: Vec<u8> = (0..(CAM_W * CAM_H * 3 / 2)).map(|i| (i % 256) as u8).collect();
            (synth, false)
        }
    };

    // 2. Downsample to the model geometry and normalize to f32 [0,1].
    let chw = nv12_to_model_tensor(&nv12);

    // 3. TensorConvert: quantize f32 -> uint8 with the model's input affine.
    let mut quant = TensorConvert::quantize(TensorDType::U8, IN_SCALE, IN_ZERO_POINT);
    quant.configure_pipeline(&tensor_caps(TensorDType::F32)).expect("quantize configure");
    let mut quant_out = OneFrame::default();
    quant
        .process(PipelinePacket::DataFrame(f32_frame(&chw)), &mut quant_out)
        .await
        .expect("quantize runs");
    let u8_tensor = quant_out.bytes.expect("quantized uint8 tensor");
    assert_eq!(u8_tensor.len(), 3 * MODEL_HW * MODEL_HW, "one byte per element");

    // 4. OrtInference on the uint8 model (the M442 100%-on-TPU path).
    let mut inf = OrtInference::from_memory_for_android(QCONV_U8IN)
        .expect("uint8 model loads")
        .with_tensor_input();
    inf.configure_pipeline(&tensor_caps(TensorDType::U8)).expect("inference configure");
    let mut inf_out = OneFrame::default();
    inf.process(
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(u8_tensor.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        }),
        &mut inf_out,
    )
    .await
    .expect("inference runs on-device");

    let out_bytes = inf_out.bytes.expect("inference output tensor");
    let out: Vec<f32> = out_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(out.len(), 64, "conv output [1,4,4,4]");
    assert!(out.iter().all(|v| v.is_finite() && *v >= 0.0), "ReLU output finite + non-negative");

    let src = if live { "LIVE CAMERA" } else { "synthetic (camera denied)" };
    eprintln!(">> {src} frame -> TensorConvert(quantize) -> OrtInference(uint8) -> output [1,4,4,4] on the Edge TPU");
}
