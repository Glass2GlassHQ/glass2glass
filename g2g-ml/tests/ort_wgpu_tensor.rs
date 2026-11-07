//! `ortinfer` on the CUDA or MIGraphX provider takes the tensor `wgpupreprocess`
//! leaves on the GPU and returns what the same model returns on the CPU from the
//! read-back tensor. Each test runs when `wgpupreprocess` opens its device on that
//! vendor's GPU.
//!
//! ```sh
//! cargo test -p g2g-ml --features ort-cuda-wgpu --test ort_wgpu_tensor
//! VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
//!   ORT_LIB_LOCATION=... cargo test -p g2g-ml --features ort-hip-wgpu --test ort_wgpu_tensor
//! ```

#![cfg(all(
    target_os = "linux",
    any(feature = "ort-cuda-wgpu", feature = "ort-hip-wgpu")
))]

use g2g_core::element::PushOutcome;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, Caps, Dim, G2gError, OutputSink, PropValue, Rate, RawVideoFormat, TensorDType,
    TensorLayout, TensorShape,
};
use g2g_ml::ortinfer::OrtInference;
use g2g_ml::wgpupreprocess::{gpu_available, WgpuBufferOwner, WgpuPreprocess};

mod onnx {
    include!("util/onnx_fixture.rs");
}

const WIDTH: u32 = 4;
const HEIGHT: u32 = 2;

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        self.packets
            .push(packet_slot.take().expect("poll_push without a packet"));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

impl Collect {
    fn frame(self) -> Frame {
        self.packets
            .into_iter()
            .find_map(|packet| match packet {
                PipelinePacket::DataFrame(frame) => Some(frame),
                _ => None,
            })
            .expect("a data frame")
    }
}

fn nv12_frame() -> Frame {
    let luma = [16u8, 81, 145, 235, 41, 100, 200, 128];
    let chroma = [128u8, 128, 90, 200];
    let bytes: Vec<u8> = luma.iter().chain(&chroma).copied().collect();
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

async fn preprocess(gpu_output: bool) -> Frame {
    let mut preprocess = WgpuPreprocess::new();
    if gpu_output {
        preprocess = preprocess.with_gpu_output();
    }
    preprocess
        .configure_pipeline(&Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(WIDTH),
            height: Dim::Fixed(HEIGHT),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        })
        .expect("configure preprocess");
    let mut out = Collect::default();
    preprocess
        .process(PipelinePacket::DataFrame(nv12_frame()), &mut out)
        .await
        .expect("preprocess");
    out.frame()
}

async fn infer(provider: &str, tensor: Frame) -> Vec<u8> {
    let model = std::env::temp_dir().join("g2g_ort_wgpu_tensor.onnx");
    std::fs::write(
        &model,
        onnx::identity_model(&[1, 3, HEIGHT as u64, WIDTH as u64]),
    )
    .expect("write model");
    let mut element = OrtInference::new();
    element
        .set_property("execution-provider", PropValue::Str(provider.into()))
        .expect("provider");
    element
        .set_property("model", PropValue::Str(model.to_str().unwrap().into()))
        .expect("load model");
    element
        .set_property("tensor-input", PropValue::Bool(true))
        .expect("tensor input");
    element
        .configure_pipeline(&Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 3, HEIGHT, WIDTH]),
            layout: TensorLayout::Nchw,
        })
        .expect("configure ortinfer");
    let mut out = Collect::default();
    element
        .process(PipelinePacket::DataFrame(tensor), &mut out)
        .await
        .expect("infer");
    out.frame()
        .domain
        .as_system_slice()
        .expect("system output")
        .to_vec()
}

fn vendor(tensor: &Frame) -> u32 {
    let MemoryDomain::WgpuBuffer(buffer) = &tensor.domain else {
        panic!("wgpupreprocess gpu-output must emit a wgpu buffer");
    };
    buffer
        .keep_alive()
        .as_any()
        .downcast_ref::<WgpuBufferOwner>()
        .expect("a wgpupreprocess buffer")
        .device()
        .adapter_info()
        .vendor
}

async fn gpu_matches_cpu(provider: &str, gpu_vendor: u32) {
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    let gpu_tensor = preprocess(true).await;
    if vendor(&gpu_tensor) != gpu_vendor {
        eprintln!("skipping: wgpupreprocess is not on a {gpu_vendor:#06x} GPU");
        return;
    }
    let host_tensor = preprocess(false).await;

    let from_gpu = infer(provider, gpu_tensor).await;
    let from_host = infer("cpu", host_tensor).await;

    assert_eq!(from_gpu.len(), 3 * (WIDTH * HEIGHT) as usize * 4);
    assert!(from_host.iter().any(|&byte| byte != 0));
    assert_eq!(from_gpu, from_host);
}

#[cfg(feature = "ort-cuda-wgpu")]
#[tokio::test]
async fn cuda_runs_on_the_gpu_tensor_and_matches_the_cpu() {
    gpu_matches_cpu("cuda", 0x10de).await;
}

#[cfg(feature = "ort-hip-wgpu")]
#[tokio::test]
async fn migraphx_runs_on_the_gpu_tensor_and_matches_the_cpu() {
    gpu_matches_cpu("migraphx", 0x1002).await;
}
