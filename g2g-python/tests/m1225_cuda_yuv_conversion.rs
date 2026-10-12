//! M1225: `g2g_process_cuda` receives the YUV conversion of the frame's current
//! caps as `(kr, kb, full_range)`, including caps a mid-stream `CapsChanged`
//! announced after configure. No device memory is touched, so it runs without a
//! GPU.
#![cfg(feature = "analytics")]

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use g2g_core::memory::{CudaKeepAlive, OwnedCudaBuffer};
use g2g_core::{
    AsyncElement, Caps, Colorimetry, Dim, Frame, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;
/// Never dereferenced: the probe only reads its arguments.
const FAKE_LUMA: u64 = 0xdead_0000;
const BT601_LIMITED: (f32, f32, bool) = (0.299, 0.114, false);
const BT709_LIMITED: (f32, f32, bool) = (0.2126, 0.0722, false);

#[derive(Default)]
struct CollectSink {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.packets.push(packet);
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

#[derive(Debug)]
struct NoOwner;
impl CudaKeepAlive for NoOwner {}

fn nv12_caps(colorimetry: Colorimetry) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry,
    }
}

fn cuda_frame() -> Frame {
    let chroma_ptr = FAKE_LUMA + u64::from(WIDTH) * u64::from(HEIGHT);
    Frame::new(
        MemoryDomain::Cuda(OwnedCudaBuffer::new(
            FAKE_LUMA,
            chroma_ptr,
            WIDTH,
            WIDTH,
            WIDTH,
            HEIGHT,
            0,
            0,
            Arc::new(NoOwner),
        )),
        Default::default(),
        0,
    )
}

/// Register the native `g2g` module and put the fixtures on the import path.
fn prepare_interpreter() {
    g2g_python::init_host();
    let fixtures = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");
    let inherited = std::env::var("PYTHONPATH").unwrap_or_default();
    if inherited.split(':').any(|entry| entry == fixtures) {
        return;
    }
    let path = if inherited.is_empty() {
        fixtures.to_string()
    } else {
        format!("{fixtures}:{inherited}")
    };
    std::env::set_var("PYTHONPATH", path);
}

fn observed_yuv_conversion() -> (f32, f32, bool) {
    Python::attach(|py| {
        let dict: Bound<'_, PyDict> = PyModule::import(py, "cuda_element")
            .expect("fixture importable")
            .getattr("OBSERVED")
            .unwrap()
            .downcast_into()
            .unwrap();
        dict.get_item("yuv_conversion")
            .unwrap()
            .expect("the probe ran")
            .extract()
            .expect("a (float, float, bool) tuple")
    })
}

#[test]
fn hook_receives_the_yuv_conversion_of_the_latest_caps() {
    prepare_interpreter();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut element = PyTransform::new("cuda_element", "YuvConversionProbe")
        .with_accept(nv12_caps(Colorimetry::UNKNOWN))
        .with_cuda_frames(true);
    let mut sink = CollectSink::default();

    element
        .configure_pipeline(&nv12_caps(Colorimetry::UNKNOWN))
        .unwrap();
    runtime
        .block_on(element.process(PipelinePacket::DataFrame(cuda_frame()), &mut sink))
        .expect("the hosted element should complete");
    assert_eq!(
        observed_yuv_conversion(),
        BT601_LIMITED,
        "unknown colorimetry resolves to BT.601 limited"
    );

    runtime
        .block_on(element.process(
            PipelinePacket::CapsChanged(nv12_caps(Colorimetry::BT709)),
            &mut sink,
        ))
        .expect("the refined caps are accepted");
    runtime
        .block_on(element.process(PipelinePacket::DataFrame(cuda_frame()), &mut sink))
        .expect("the hosted element should complete");
    assert_eq!(observed_yuv_conversion(), BT709_LIMITED);
}
