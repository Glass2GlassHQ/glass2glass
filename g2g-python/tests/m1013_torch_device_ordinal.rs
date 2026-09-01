//! M1013: the CUDA device ordinal the frame carries, and the torch import paths.
//!
//! `OwnedCudaBuffer` names the device its pointers live on, so a plane handed to
//! Python reports that ordinal instead of assuming device 0. The description
//! tests run anywhere; the torch tests need `torch` plus a real device and skip
//! themselves without one (they are independent of the cupy tests: a host with
//! only torch still runs these).
#![cfg(feature = "python")]

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use g2g_core::memory::{CudaKeepAlive, OwnedCudaBuffer};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::{PySource, PyTransform};

/// Small enough for a torch allocation per test, pitched wider than the width.
const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;
const PITCH: u32 = 128;
/// Never dereferenced: the description tests only read what the plane reports.
const FAKE_LUMA: u64 = 0xdead_0000;
/// An ordinal no single-GPU host would default to, so a plane reporting a fixed
/// device rather than the frame's fails.
const OTHER_DEVICE: i32 = 3;
/// What `DescribedCudaSource` reports through its `cuda_device` attribute.
const DESCRIBED_DEVICE: i32 = 5;
/// The byte `TorchDlpackConsumer` writes through the plane; mirrors the fixture.
const MARKER_VALUE: u32 = 0xA5;

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

/// Stands in for the producer that owns the device allocation: either a fake
/// pointer or memory a torch tensor in the fixture module keeps alive.
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

/// A CUDA NV12 frame on `device_ordinal`, chroma following luma after `HEIGHT`
/// pitched rows.
fn cuda_frame(luma_ptr: u64, pitch: u32, device_ordinal: i32) -> Frame {
    let chroma_ptr = luma_ptr + u64::from(pitch) * u64::from(HEIGHT);
    Frame {
        domain: MemoryDomain::Cuda(OwnedCudaBuffer::new(
            luma_ptr,
            chroma_ptr,
            pitch,
            pitch,
            WIDTH,
            HEIGHT,
            0,
            device_ordinal,
            Arc::new(NoOwner),
        )),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

/// Register the native `g2g` module and put the fixtures on the import path,
/// before anything acquires the GIL. Any inherited `PYTHONPATH` is kept, so a
/// torch install can be pointed at from outside.
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

/// Read one entry of the fixture module's `OBSERVED` dict.
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

fn current_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
}

/// Run one GPU frame through a hosted transform.
fn run_transform(class: &str, frame: Frame) -> (Result<(), G2gError>, Vec<PipelinePacket>) {
    prepare_interpreter();
    let mut element = PyTransform::new("cuda_element", class)
        .with_accept(nv12_caps())
        .with_cuda_frames(true);
    element.configure_pipeline(&nv12_caps()).unwrap();
    let mut sink = CollectSink::default();
    let result =
        current_runtime().block_on(element.process(PipelinePacket::DataFrame(frame), &mut sink));
    (result, sink.packets)
}

/// Have the fixture allocate a real NV12 surface with torch, reporting the
/// device it landed on. `None` when this host has no torch or no CUDA device.
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

/// Read one luma byte of the surface at `pointer` through the producer's own
/// torch tensor.
fn read_torch_surface(pointer: u64, row: u32, column: u32) -> u32 {
    Python::attach(|py| {
        fixture(py)
            .call_method1("read_surface_torch", (pointer, row, column))
            .expect("read_surface_torch")
            .extract()
            .unwrap()
    })
}

/// Whether this host has torch and a CUDA device for it.
fn torch_available() -> bool {
    prepare_interpreter();
    Python::attach(|py| {
        fixture(py)
            .call_method0("torch_cuda_available")
            .expect("torch_cuda_available reports its own failures as False")
            .extract()
            .unwrap()
    })
}

#[test]
fn a_plane_reports_the_device_ordinal_its_frame_carries() {
    // No device needed: the DLPack device tuple is pure description.
    let (result, _) = run_transform(
        "DlpackShapeProbe",
        cuda_frame(FAKE_LUMA, PITCH, OTHER_DEVICE),
    );
    result.expect("the probe only reads the capsule's name and the device tuple");
    assert_eq!(
        observed::<(i32, i32)>("dlpack_device"),
        (2, OTHER_DEVICE),
        "kDLCUDA on the device the frame named"
    );
    assert!(
        observed::<bool>("other_device_refused"),
        "the probe asks for device 7, which is not this plane's, so it must be refused"
    );
}

#[test]
fn a_produced_frame_carries_the_device_the_source_reported() {
    prepare_interpreter();
    let mut source = PySource::new("cuda_element", "DescribedCudaSource")
        .with_caps(nv12_caps())
        .with_cuda_frames(true);
    source.configure_pipeline(&nv12_caps()).unwrap();
    let mut sink = CollectSink::default();
    let produced = current_runtime()
        .block_on(source.run(&mut sink))
        .expect("a source of pure descriptions needs no device");
    assert_eq!(produced, 1);

    let Some(PipelinePacket::DataFrame(frame)) = sink.packets.first() else {
        panic!("expected a DataFrame first");
    };
    let MemoryDomain::Cuda(buffer) = &frame.domain else {
        panic!("a cuda-frames source must emit Cuda-domain frames");
    };
    assert_eq!(
        buffer.device_ordinal, DESCRIBED_DEVICE,
        "the ordinal the source reported reaches the frame"
    );
}

#[test]
fn torch_from_dlpack_aliases_the_producers_device_memory() {
    let Some((luma_ptr, pitch, ordinal)) = allocate_torch_surface() else {
        eprintln!("skipping: no torch / no CUDA device on this host");
        return;
    };

    let (result, packets) =
        run_transform("TorchDlpackConsumer", cuda_frame(luma_ptr, pitch, ordinal));
    result.expect("the torch DLPack consumer should complete");
    assert_eq!(packets.len(), 1);

    assert_eq!(observed::<(i32, i32)>("torch_dlpack_device"), (2, ordinal));
    assert_eq!(
        observed::<(String, i32)>("torch_luma_device"),
        ("cuda".to_string(), ordinal),
        "torch put the tensor on the device the plane named"
    );
    assert_eq!(
        observed::<u64>("torch_luma_ptr"),
        luma_ptr,
        "torch's tensor points at the producer's own device memory"
    );
    assert_eq!(
        observed::<u64>("torch_chroma_ptr"),
        luma_ptr + u64::from(pitch) * u64::from(HEIGHT)
    );
    assert_eq!(
        observed::<(usize, usize)>("torch_luma_shape"),
        (HEIGHT as usize, WIDTH as usize)
    );
    assert_eq!(
        observed::<(usize, usize)>("torch_luma_strides"),
        (pitch as usize, 1),
        "the row pitch survives as torch's element stride"
    );
    assert_eq!(
        observed::<(usize, usize, usize)>("torch_chroma_shape"),
        ((HEIGHT / 2) as usize, (WIDTH / 2) as usize, 2)
    );
    assert!(
        observed::<bool>("torch_pattern_matches"),
        "the producer's luma pattern reads back through the torch tensor"
    );
    assert!(observed::<bool>("torch_chroma_matches"), "chroma differed");

    // And the write the element made through its tensor is in the producer's
    // own tensor: one allocation, two views of it.
    assert_eq!(
        read_torch_surface(luma_ptr, 1, 2),
        MARKER_VALUE,
        "write through the plane did not reach the producer's surface"
    );
}

#[test]
fn torch_as_tensor_reads_the_cuda_array_interface() {
    let Some((luma_ptr, pitch, ordinal)) = allocate_torch_surface() else {
        eprintln!("skipping: no torch / no CUDA device on this host");
        return;
    };

    let (result, packets) = run_transform("TorchCaiConsumer", cuda_frame(luma_ptr, pitch, ordinal));
    result.expect("the torch CAI consumer should complete");
    assert_eq!(packets.len(), 1);

    assert!(
        observed::<bool>("torch_cai_read_only_refused"),
        "torch refuses a read-only CAI export, which is why torch's own path is DLPack"
    );
    assert_eq!(observed::<u64>("torch_cai_luma_ptr"), luma_ptr);
    assert_eq!(
        observed::<u64>("torch_cai_chroma_ptr"),
        luma_ptr + u64::from(pitch) * u64::from(HEIGHT)
    );
    assert_eq!(
        observed::<(usize, usize)>("torch_cai_luma_shape"),
        (HEIGHT as usize, WIDTH as usize)
    );
    assert_eq!(
        observed::<(usize, usize)>("torch_cai_luma_strides"),
        (pitch as usize, 1)
    );
    assert_eq!(
        observed::<(String, i32)>("torch_cai_luma_device"),
        ("cuda".to_string(), ordinal),
        "CAI carries no device, so torch resolves it from the pointer"
    );
    assert!(
        observed::<bool>("torch_cai_pattern_matches"),
        "the producer's luma pattern reads back through the torch tensor"
    );
    assert!(
        observed::<bool>("torch_cai_chroma_matches"),
        "chroma differed"
    );
}

#[test]
fn a_torch_source_produces_frames_naming_their_device() {
    if !torch_available() {
        eprintln!("skipping: no torch / no CUDA device on this host");
        return;
    }
    let mut source = PySource::new("cuda_element", "TorchCudaSource")
        .with_caps(nv12_caps())
        .with_cuda_frames(true);
    source.configure_pipeline(&nv12_caps()).unwrap();
    let mut sink = CollectSink::default();
    let produced = current_runtime()
        .block_on(source.run(&mut sink))
        .expect("the torch GPU source should run to EOS");
    assert_eq!(produced, 2, "the fixture ends after two surfaces");

    let pitch = observed::<u32>("torch_produced_pitch");
    let device = Python::attach(|py| {
        py.import("torch")
            .unwrap()
            .getattr("cuda")
            .unwrap()
            .call_method0("current_device")
            .unwrap()
            .extract::<i32>()
            .unwrap()
    });
    let frames: Vec<&Frame> = sink
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect();
    assert_eq!(frames.len(), 2);
    for frame in &frames {
        let MemoryDomain::Cuda(buffer) = &frame.domain else {
            panic!("a cuda-frames source must emit Cuda-domain frames");
        };
        assert_ne!(buffer.luma_ptr, 0);
        assert_eq!(buffer.luma_pitch, pitch, "the producer's pitch is carried");
        assert_eq!(
            buffer.device_ordinal, device,
            "the frame names the device torch allocated on"
        );
    }
    assert!(matches!(sink.packets.last(), Some(PipelinePacket::Eos)));
}
