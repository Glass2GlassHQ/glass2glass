#![cfg(all(target_os = "linux", feature = "ort-hip", feature = "launch"))]

use g2g_core::clock::MonotonicClock;
use g2g_core::memory::GpuVendor;
use g2g_core::runtime::{block_on, parse_launch, run_graph, Registry};
use g2g_plugins::appsink::register_appsink_pull;
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::hip::HipDevice;

mod onnx {
    include!("util/onnx_fixture.rs");
}

const SOURCE_WIDTH: u32 = 1280;
const SOURCE_HEIGHT: u32 = 720;
const MODEL_SIDE: u32 = 320;
fn nv12_frame() -> Vec<u8> {
    let (w, h) = (SOURCE_WIDTH as usize, SOURCE_HEIGHT as usize);
    let luma = (0..w * h).map(|i| (16 + (i % w * 3 + i / w * 5) % 220) as u8);
    let chroma = (0..w * h / 2).map(|i| (64 + (i % w + i / w) % 128) as u8);
    luma.chain(chroma).collect()
}

fn identity_model() -> String {
    let path = std::env::temp_dir().join("g2g_m2106_identity.onnx");
    let side = u64::from(MODEL_SIDE);
    std::fs::write(&path, onnx::identity_model(&[1, 3, side, side])).expect("write model");
    path.to_str().expect("utf-8 path").to_owned()
}

fn registry() -> Registry {
    let mut registry = g2g_plugins::registry::default_registry();
    g2g_ml::register(&mut registry);
    registry
}

fn hip_device_present(vendor: GpuVendor) -> bool {
    match HipDevice::open(vendor, 0) {
        Ok(_) => true,
        Err(error) => {
            eprintln!("skipping: no {vendor:?} HIP device 0: {error:?}");
            false
        }
    }
}

// `hipupload ! hipconvertscale` to the model's input, then `ortinfer` on `provider`
fn infer_on(vendor: &str, provider: &str) -> String {
    let model = identity_model();
    let inference = match provider {
        "cpu" => format!("hipdownload ! ortinfer model={model} execution-provider=cpu"),
        _ => format!("ortinfer model={model} execution-provider={provider}"),
    };
    format!(
        "hipupload vendor={vendor} ! hipconvertscale add-borders=true \
         ! video/x-raw,format=RGBP_F32LE,width={MODEL_SIDE},height={MODEL_SIDE} ! {inference}"
    )
}

// runs `<head> ! <middle> ! appsink` and returns every frame the appsink pulls
fn pulled_frames(registry: &Registry, head: &str, middle: &str, sink: &str) -> Vec<Vec<u8>> {
    let pull = register_appsink_pull(sink);
    let line = format!("{head} ! {middle} ! appsink channel={sink}");
    let graph = parse_launch(registry, &line).expect("parses");
    // appsink stops the graph once its pull queue is full, so frames are pulled while it runs
    let puller = std::thread::spawn(move || {
        let mut frames = Vec::new();
        while let Some(frame) = block_on(pull.pull()) {
            let bytes = frame.domain.as_system_slice().expect("a system frame");
            frames.push(bytes.to_vec());
        }
        frames
    });
    block_on(run_graph(graph, &MonotonicClock, 4)).expect("runs");
    puller.join().expect("the pull thread finishes")
}

fn appsrc_frames(name: &str, middle: &str) -> Vec<Vec<u8>> {
    let source = format!("m2106_{name}_in");
    let feed = register_appsrc(&source);
    assert!(feed.push(&nv12_frame(), 0));
    feed.end_of_stream();
    let head = format!(
        "appsrc channel={source} caps=video/x-raw,format=NV12,width={SOURCE_WIDTH},height={SOURCE_HEIGHT},framerate=30/1"
    );
    pulled_frames(&registry(), &head, middle, &format!("m2106_{name}_out"))
}

fn assert_same_tensors(on_gpu: &[Vec<u8>], on_cpu: &[Vec<u8>]) {
    let tensor_len = 3 * (MODEL_SIDE * MODEL_SIDE) as usize * size_of::<f32>();
    assert!(!on_gpu.is_empty(), "no tensor reached the appsink");
    assert_eq!(on_gpu.len(), on_cpu.len(), "tensor count");
    for (index, (gpu, cpu)) in on_gpu.iter().zip(on_cpu).enumerate() {
        assert_eq!(gpu.len(), tensor_len, "tensor {index}: length");
        assert!(
            cpu.iter().any(|&byte| byte != 0),
            "tensor {index}: all black"
        );
        assert!(gpu == cpu, "tensor {index}: the GPU and CPU outputs differ");
    }
}

fn provider_reads_hip_memory(vendor: GpuVendor, vendor_name: &str, provider: &str) {
    if !hip_device_present(vendor) {
        return;
    }
    let on_gpu = appsrc_frames(provider, &infer_on(vendor_name, provider));
    let on_cpu = appsrc_frames(
        &format!("{provider}_reference"),
        &infer_on(vendor_name, "cpu"),
    );
    assert_same_tensors(&on_gpu, &on_cpu);
}

#[cfg(feature = "migraphx")]
#[test]
fn migraphx_reads_hipconvertscale_output_in_place() {
    provider_reads_hip_memory(GpuVendor::Amd, "amd", "migraphx");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_reads_hipconvertscale_output_in_place() {
    provider_reads_hip_memory(GpuVendor::Nvidia, "nvidia", "cuda");
}

// nvdec's frames live in its own CUDA context, and hipupload wraps them without a copy
#[cfg(feature = "cuda")]
const H264_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../g2g-plugins/tests/fixtures/av_h264_aac.mp4"
);

#[cfg(feature = "cuda")]
#[test]
fn cuda_infers_on_nvdec_frames() {
    let registry = registry();
    if registry.inspect("nvdec").is_none() {
        eprintln!("skipping: built without g2g-plugins/nvdec");
        return;
    }
    if !hip_device_present(GpuVendor::Nvidia) {
        return;
    }
    let head =
        format!("filesrc location={H264_FIXTURE} ! qtdemux name=d d.video_0 ! h264parse ! nvdec");
    let on_gpu = pulled_frames(
        &registry,
        &head,
        &infer_on("nvidia", "cuda"),
        "m2106_nvdec_out",
    );
    let on_cpu = pulled_frames(
        &registry,
        &head,
        &infer_on("nvidia", "cpu"),
        "m2106_nvdec_reference_out",
    );
    assert_same_tensors(&on_gpu, &on_cpu);
}
