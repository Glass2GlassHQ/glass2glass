//! M315 end-to-end: a real camera into the GPU/ML preprocess path.
//! `LibCameraSrc -> VideoConvert(NV12) -> WgpuPreprocess`, producing a
//! normalized f32 NCHW RGB tensor on the GPU from live camera frames, the
//! camera-side entry to the ML tensor graph (`-> inference -> postprocess`).
//!
//! Linux + the `libcamera-wgpu` feature; needs a camera libcamera can open and
//! a wgpu adapter (skips cleanly without one). Run:
//! `cargo test -p g2g-ml --features libcamera-wgpu --test libcamera_preprocess -- --ignored --nocapture`

#![cfg(all(target_os = "linux", feature = "libcamera-wgpu"))]

use g2g_core::element::DynAsyncElement;
use g2g_core::runtime::{run_linear_chain, LatencyProfile};
use g2g_core::{Caps, PipelineClock, RawVideoFormat, TensorDType, TensorLayout, TensorShape};
use g2g_ml::wgpupreprocess::{gpu_available, WgpuPreprocess};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::libcamerasrc::LibCameraSrc;
use g2g_plugins::videoconvert::VideoConvert;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn camera_index() -> usize {
    std::env::var("G2G_LIBCAMERA_INDEX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}

#[tokio::test]
#[ignore = "needs a real camera + a wgpu adapter"]
async fn libcamera_preprocess_to_gpu_tensor() {
    if !gpu_available().await {
        eprintln!("no wgpu adapter on this host; skipping libcamera GPU preprocess test");
        return;
    }

    let (w, h) = (640u32, 480u32);
    let target: u64 = 5;
    let mut src = LibCameraSrc::new()
        .with_camera(camera_index())
        .with_size(w, h)
        .with_fps(15)
        .with_frame_limit(target);
    // The camera emits YUYV (raw); WgpuPreprocess consumes NV12.
    let mut convert = VideoConvert::new(RawVideoFormat::Nv12);
    let mut preprocess = WgpuPreprocess::new();
    let mut sink = FakeSink::new();

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut convert, &mut preprocess];

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        run_linear_chain(
            &mut src,
            transforms,
            &mut sink,
            &NullClock,
            LatencyProfile::Live.link_capacity(),
        ),
    )
    .await
    .expect("pipeline should finish within 30s")
    .expect("camera -> convert -> GPU preprocess should negotiate and flow");

    eprintln!(
        "camera -> GPU preprocess: emitted={} tensors_received={}",
        stats.frames_emitted,
        sink.received()
    );
    assert_eq!(sink.received(), target, "every camera frame became a tensor");
    assert!(sink.eos_seen());

    // The sink must have seen the [1, 3, H, W] f32 NCHW tensor caps the GPU
    // preprocess produces, proof the camera frames were turned into ML tensors.
    let changes = sink.caps_changes();
    assert!(
        changes.iter().any(|c| matches!(
            &c.caps,
            Caps::Tensor { dtype: TensorDType::F32, layout: TensorLayout::Nchw, shape }
            if *shape == TensorShape::new([1u32, 3, h, w])
        )),
        "sink saw the [1,3,{h},{w}] tensor caps, got {changes:?}"
    );
}
