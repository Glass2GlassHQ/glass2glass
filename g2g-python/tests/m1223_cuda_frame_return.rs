//! M1223: `g2g_process_cuda` can hand back a new `(luma, chroma)` pair, which the
//! host sends on in place of the input frame with the input's timing, sequence
//! and metadata. Returning `None` passes the input frame on. The replacement test
//! needs torch and a CUDA device and skips itself without one.
#![cfg(feature = "analytics")]

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use g2g_core::memory::{CudaKeepAlive, OwnedCudaBuffer};
use g2g_core::{
    AsyncElement, BlobMeta, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;
/// Wider than the width, so the input's pitch cannot match the packed planes the
/// fixture returns.
const PITCH: u32 = 128;
/// Never dereferenced: the pass-through test only reads the frame back.
const FAKE_LUMA: u64 = 0xdead_0000;
const INPUT_PTS_NS: u64 = 40_000_000;
const INPUT_SEQUENCE: u64 = 7;
const UPSTREAM_HEADER: &str = "upstream";
const UPSTREAM_PAYLOAD: &[u8] = b"detector";
/// What `TorchFrameReturner` fills its planes with and stages; mirrors the fixture.
const RETURNED_LUMA_VALUE: u32 = 0x3C;
const RETURNED_CHROMA_VALUE: u32 = 0x5A;
const STAGED_HEADER: &str = "returned";

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

/// The input surface is either fake or a torch tensor the fixture module keeps
/// alive, so there is nothing to release.
#[derive(Debug)]
struct NoOwner;
impl CudaKeepAlive for NoOwner {}

fn nv12_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// A pitched CUDA NV12 frame with distinctive timing, sequence and an upstream
/// blob, chroma following luma after `HEIGHT` rows.
fn cuda_frame(luma_ptr: u64, device_ordinal: i32) -> Frame {
    let chroma_ptr = luma_ptr + u64::from(PITCH) * u64::from(HEIGHT);
    let mut frame = Frame {
        domain: MemoryDomain::Cuda(OwnedCudaBuffer::new(
            luma_ptr,
            chroma_ptr,
            PITCH,
            PITCH,
            WIDTH,
            HEIGHT,
            0,
            device_ordinal,
            Arc::new(NoOwner),
        )),
        timing: FrameTiming {
            pts_ns: INPUT_PTS_NS,
            dts_ns: INPUT_PTS_NS,
            ..FrameTiming::default()
        },
        sequence: INPUT_SEQUENCE,
        meta: Default::default(),
    };
    let mut blobs = BlobMeta::new();
    blobs.push(UPSTREAM_HEADER, UPSTREAM_PAYLOAD.to_vec());
    frame.meta.attach(blobs);
    frame
}

/// Register the native `g2g` module and put the fixtures on the import path,
/// keeping any inherited `PYTHONPATH` so a torch install can be pointed at.
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

fn fixture<'py>(py: Python<'py>) -> Bound<'py, PyModule> {
    PyModule::import(py, "cuda_element").expect("fixture importable")
}

fn observed<T: for<'p> pyo3::FromPyObject<'p>>(key: &str) -> T {
    Python::attach(|py| {
        let dict: Bound<'_, PyDict> = fixture(py)
            .getattr("OBSERVED")
            .unwrap()
            .downcast_into()
            .unwrap();
        dict.get_item(key)
            .unwrap()
            .unwrap_or_else(|| panic!("OBSERVED[{key}] missing"))
            .extract()
            .expect("OBSERVED value has the expected type")
    })
}

/// Run one GPU frame through the hosted `class` and return the one frame that
/// came out.
fn run_transform(class: &str, frame: Frame) -> Frame {
    prepare_interpreter();
    let mut element = PyTransform::new("cuda_element", class)
        .with_accept(nv12_caps())
        .with_cuda_frames(true);
    element.configure_pipeline(&nv12_caps()).unwrap();
    let mut sink = CollectSink::default();
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(element.process(PipelinePacket::DataFrame(frame), &mut sink))
        .expect("the hosted element should complete");
    assert_eq!(sink.packets.len(), 1, "one frame in, one frame out");
    let Some(PipelinePacket::DataFrame(frame)) = sink.packets.pop() else {
        panic!("expected a DataFrame downstream");
    };
    frame
}

/// Have the fixture allocate an input NV12 surface with torch. `None` when this
/// host has no torch or no CUDA device.
fn allocate_torch_surface() -> Option<(u64, u32, i32)> {
    prepare_interpreter();
    Python::attach(|py| {
        fixture(py)
            .call_method1("allocate_nv12_torch", (WIDTH, HEIGHT, PITCH))
            .expect("allocate_nv12_torch reports its own failures as None")
            .extract()
            .unwrap()
    })
}

fn read_device_byte(pointer: u64) -> u32 {
    Python::attach(|py| {
        fixture(py)
            .call_method1("read_device_byte", (pointer,))
            .expect("read_device_byte")
            .extract()
            .unwrap()
    })
}

fn upstream_blob(frame: &Frame, header: &str) -> Vec<u8> {
    frame
        .meta
        .get::<BlobMeta>()
        .and_then(|blobs| blobs.get(header))
        .unwrap_or_else(|| panic!("no blob tagged {header} on the frame"))
        .payload
        .clone()
}

#[test]
fn returned_planes_replace_the_input_frame() {
    let Some((input_luma, _, ordinal)) = allocate_torch_surface() else {
        eprintln!("skipping: no torch / no CUDA device on this host");
        return;
    };

    let frame = run_transform("TorchFrameReturner", cuda_frame(input_luma, ordinal));

    let MemoryDomain::Cuda(buffer) = &frame.domain else {
        panic!("the replacement frame must stay in the Cuda domain");
    };
    let returned_luma: u64 = observed("returned_luma_ptr");
    assert_ne!(returned_luma, input_luma);
    assert_eq!(
        buffer.luma_ptr, returned_luma,
        "the returned luma is sent on"
    );
    assert_eq!(buffer.chroma_ptr, observed::<u64>("returned_chroma_ptr"));
    assert_eq!(
        (buffer.width, buffer.height),
        (WIDTH, HEIGHT),
        "geometry unchanged"
    );
    assert_eq!(
        (buffer.luma_pitch, buffer.chroma_pitch),
        (WIDTH, WIDTH),
        "the packed planes' own pitch, not the input's"
    );
    let context: u64 = observed("returned_context");
    assert_ne!(
        context, 0,
        "torch's context is current on the worker thread"
    );
    assert_eq!(
        buffer.context, context,
        "the element's cuda_context is carried"
    );
    assert_eq!(buffer.device_ordinal, observed::<i32>("returned_device"));

    assert_eq!(frame.timing.pts_ns, INPUT_PTS_NS);
    assert_eq!(frame.sequence, INPUT_SEQUENCE);
    assert_eq!(upstream_blob(&frame, UPSTREAM_HEADER), UPSTREAM_PAYLOAD);
    assert_eq!(upstream_blob(&frame, STAGED_HEADER), b"yes");

    // The frame holds the only reference to the planes, so this also shows the
    // keep-alive works.
    assert_eq!(read_device_byte(buffer.luma_ptr), RETURNED_LUMA_VALUE);
    assert_eq!(read_device_byte(buffer.chroma_ptr), RETURNED_CHROMA_VALUE);
}

#[test]
fn returning_none_passes_the_input_frame_on() {
    let frame = run_transform("NoneReturner", cuda_frame(FAKE_LUMA, 0));

    let MemoryDomain::Cuda(buffer) = &frame.domain else {
        panic!("the input frame must stay in the Cuda domain");
    };
    assert_eq!(buffer.luma_ptr, FAKE_LUMA);
    assert_eq!(
        buffer.chroma_ptr,
        FAKE_LUMA + u64::from(PITCH) * u64::from(HEIGHT)
    );
    assert_eq!(buffer.luma_pitch, PITCH);
    assert_eq!(frame.timing.pts_ns, INPUT_PTS_NS);
    assert_eq!(frame.sequence, INPUT_SEQUENCE);
    assert_eq!(upstream_blob(&frame, UPSTREAM_HEADER), UPSTREAM_PAYLOAD);
}

#[test]
fn returning_the_input_planes_fails_the_frame() {
    prepare_interpreter();
    let mut element = PyTransform::new("cuda_element", "InputReturner")
        .with_accept(nv12_caps())
        .with_cuda_frames(true);
    element.configure_pipeline(&nv12_caps()).unwrap();
    let mut sink = CollectSink::default();
    let result = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(element.process(
            PipelinePacket::DataFrame(cuda_frame(FAKE_LUMA, 0)),
            &mut sink,
        ));
    assert!(
        result.is_err(),
        "the input planes dangle once the input frame is replaced, got {result:?}"
    );
    assert!(sink.packets.is_empty());
}
