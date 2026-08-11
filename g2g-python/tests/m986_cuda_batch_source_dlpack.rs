//! M986: the GPU handoff beyond the single transform.
//!
//! Three additions over M984: a Cuda-domain *batch* reaches an aggregator as one
//! `(luma, chroma)` pair per input, a hosted *source* produces device-resident
//! frames by handing back its own CAI surfaces, and every plane also exports
//! DLPack. The routing and validation tests use fake device pointers and run
//! anywhere; the tests that touch memory need cupy and a real device, and skip
//! themselves without one.
#![cfg(feature = "python")]

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use g2g_core::memory::{CudaKeepAlive, MemoryDomainKind, OwnedCudaBuffer};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, MultiInputElement,
    OutputSink, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::{PyAggregator, PySource, PyTransform};

/// Small enough to allocate two surfaces per test, pitched wider than the width.
const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;
const PITCH: u32 = 128;
const FAKE_LUMA: [u64; 2] = [0xdead_0000, 0xbeef_0000];

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

/// Stands in for the decoder that owns the device allocation; the pointers in the
/// fake-pointer tests are never dereferenced.
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
    }
}

fn cuda_frame(luma_ptr: u64, pitch: u32, sequence: u64) -> Frame {
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
            0,
            Arc::new(NoOwner),
        )),
        timing: FrameTiming::default(),
        sequence,
        meta: Default::default(),
    }
}

/// Register the native `g2g` module and put the fixtures on the import path,
/// before anything acquires the GIL. Any inherited `PYTHONPATH` is kept, so a cupy
/// install can be pointed at from outside.
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

/// Whether this host has cupy and a CUDA device to allocate from.
fn cupy_available() -> bool {
    prepare_interpreter();
    Python::attach(|py| {
        py.import("cupy")
            .and_then(|cupy| {
                cupy.getattr("cuda")?
                    .getattr("runtime")?
                    .call_method0("getDeviceCount")?
                    .extract::<i32>()
            })
            .map(|devices| devices > 0)
            .unwrap_or(false)
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

/// Feed one GPU frame per input into a hosted aggregator and drain it.
fn run_batch(class: &str, frames: Vec<Frame>) -> (Result<(), G2gError>, Vec<PipelinePacket>) {
    prepare_interpreter();
    let inputs = frames.len();
    let mut element = PyAggregator::new("cuda_element", class, inputs)
        .with_accept(nv12_caps())
        .with_cuda_frames(true);
    for input in 0..inputs {
        element.configure_pipeline(input, &nv12_caps()).unwrap();
    }
    let mut sink = CollectSink::default();
    let runtime = current_runtime();
    let mut result = Ok(());
    for (input, frame) in frames.into_iter().enumerate() {
        result =
            runtime.block_on(element.process(input, PipelinePacket::DataFrame(frame), &mut sink));
        if result.is_err() {
            break;
        }
    }
    (result, sink.packets)
}

#[test]
fn a_gpu_batch_reaches_the_aggregator_as_one_plane_pair_per_input() {
    let frames = vec![
        cuda_frame(FAKE_LUMA[0], PITCH, 0),
        cuda_frame(FAKE_LUMA[1], PITCH, 0),
    ];
    let (result, packets) = run_batch("CudaBatchProbe", frames);
    result.expect("a Cuda-domain batch should route to g2g_process_cuda_batch");

    // One anchor frame out, still GPU-resident: no readback anywhere.
    assert_eq!(packets.len(), 1);
    let PipelinePacket::DataFrame(frame) = &packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    let MemoryDomain::Cuda(buffer) = &frame.domain else {
        panic!("the anchor should still be in the Cuda domain");
    };
    assert_eq!(buffer.luma_ptr, FAKE_LUMA[0], "input 0 is the anchor");

    assert_eq!(observed::<usize>("batch_size"), 2);
    assert_eq!(observed::<Vec<u64>>("batch_luma_ptrs"), FAKE_LUMA.to_vec());
    assert_eq!(
        observed::<Vec<u64>>("batch_chroma_ptrs"),
        FAKE_LUMA
            .iter()
            .map(|p| p + u64::from(PITCH) * u64::from(HEIGHT))
            .collect::<Vec<_>>()
    );
    assert_eq!(observed::<(u32, u32)>("batch_geometry"), (WIDTH, HEIGHT));
}

#[test]
fn an_aggregator_without_the_cuda_batch_hook_refuses_gpu_frames() {
    let frames = vec![cuda_frame(FAKE_LUMA[0], PITCH, 0)];
    let (result, packets) = run_batch("CpuOnly", frames);
    assert!(
        matches!(result, Err(G2gError::UnsupportedDomain)),
        "got {result:?}"
    );
    assert!(packets.is_empty());
}

#[test]
fn the_aggregator_asks_each_input_branch_for_the_domain_it_reads() {
    let cpu = PyAggregator::new("cuda_element", "CudaBatchProbe", 2).with_accept(nv12_caps());
    let gpu = PyAggregator::new("cuda_element", "CudaBatchProbe", 2)
        .with_accept(nv12_caps())
        .with_cuda_frames(true);
    let nv12_bytes = (WIDTH * HEIGHT * 3 / 2) as usize;
    for input in 0..2 {
        let system = cpu
            .propose_allocation_for_input(input, &nv12_caps())
            .unwrap();
        assert_eq!(system.domain, MemoryDomainKind::System);
        assert_eq!(system.size_bytes, nv12_bytes);
        let cuda = gpu
            .propose_allocation_for_input(input, &nv12_caps())
            .unwrap();
        assert_eq!(cuda.domain, MemoryDomainKind::Cuda);
        assert_eq!(cuda.size_bytes, nv12_bytes);
    }
}

#[test]
fn dlpack_describes_the_same_plane_as_cai() {
    // No device needed: the capsule and the device tuple are pure description.
    let (result, _) = run_transform("DlpackShapeProbe", cuda_frame(FAKE_LUMA[0], PITCH, 0));
    result.expect("the probe only reads the capsule's name and the device tuple");
    assert_eq!(observed::<(i32, i32)>("dlpack_device"), (2, 0));
    assert_eq!(
        observed::<String>("versioned_capsule_name"),
        "dltensor_versioned"
    );
    assert_eq!(observed::<String>("legacy_capsule_name"), "dltensor");
    assert!(
        observed::<bool>("copy_refused"),
        "copy=True must be refused"
    );
    assert!(
        observed::<bool>("other_device_refused"),
        "a request for another device must be refused"
    );
}

#[test]
fn cupy_from_dlpack_aliases_the_producers_device_memory() {
    if !cupy_available() {
        eprintln!("skipping: no cupy / no CUDA device on this host");
        return;
    }
    let (luma_ptr, pitch) = Python::attach(|py| {
        fixture(py)
            .call_method1("allocate_nv12", (WIDTH, HEIGHT, PITCH))
            .expect("allocate_nv12 reports its own failures as None")
            .extract::<Option<(u64, u32)>>()
            .unwrap()
            .expect("cupy is importable, so the allocation should succeed")
    });

    let (result, packets) = run_transform("DlpackConsumer", cuda_frame(luma_ptr, pitch, 0));
    result.expect("the DLPack consumer should complete");
    assert_eq!(packets.len(), 1);

    assert_eq!(observed::<(i32, i32)>("dlpack_device"), (2, 0));
    assert_eq!(
        observed::<u64>("dlpack_luma_ptr"),
        luma_ptr,
        "cupy's array points at the producer's own device memory"
    );
    assert_eq!(
        observed::<u64>("dlpack_chroma_ptr"),
        luma_ptr + u64::from(pitch) * u64::from(HEIGHT)
    );
    assert_eq!(
        observed::<(usize, usize)>("dlpack_luma_shape"),
        (HEIGHT as usize, WIDTH as usize)
    );
    assert_eq!(
        observed::<(usize, usize)>("dlpack_luma_strides"),
        (pitch as usize, 1),
        "the row pitch survives the element-stride conversion"
    );
    assert_eq!(
        observed::<(usize, usize, usize)>("dlpack_chroma_shape"),
        ((HEIGHT / 2) as usize, (WIDTH / 2) as usize, 2)
    );
    assert!(
        observed::<bool>("dlpack_pattern_matches"),
        "the producer's pixels read back through the DLPack tensor"
    );
    assert_eq!(observed::<String>("unconsumed_capsule"), "PyCapsule");
}

#[test]
fn a_hosted_source_produces_gpu_resident_frames() {
    if !cupy_available() {
        eprintln!("skipping: no cupy / no CUDA device on this host");
        return;
    }
    prepare_interpreter();
    let mut source = PySource::new("cuda_element", "CudaSource")
        .with_caps(nv12_caps())
        .with_cuda_frames(true);
    source.configure_pipeline(&nv12_caps()).unwrap();
    assert_eq!(
        source.output_memory(),
        MemoryDomainKind::Cuda,
        "the frames leave in the domain the source allocated them in"
    );

    let mut sink = CollectSink::default();
    let produced = current_runtime()
        .block_on(source.run(&mut sink))
        .expect("the GPU source should run to EOS");
    assert_eq!(
        produced, 2,
        "the fixture ends the stream after two surfaces"
    );

    let frames: Vec<&Frame> = sink
        .packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(frame),
            _ => None,
        })
        .collect();
    assert_eq!(frames.len(), 2);
    let pitch = observed::<u32>("produced_pitch");
    for (sequence, frame) in frames.iter().enumerate() {
        let MemoryDomain::Cuda(buffer) = &frame.domain else {
            panic!("a cuda-frames source must emit Cuda-domain frames");
        };
        assert_ne!(buffer.luma_ptr, 0);
        assert_eq!(buffer.luma_pitch, pitch, "the producer's pitch is carried");
        assert_eq!(
            buffer.chroma_ptr,
            buffer.luma_ptr + u64::from(pitch) * u64::from(HEIGHT),
            "chroma follows luma in the one allocation the fixture made"
        );
        assert_eq!((buffer.width, buffer.height), (WIDTH, HEIGHT));
        // The source stamps timing, not the host.
        assert_eq!(frame.sequence, sequence as u64);
        assert_eq!(frame.timing.pts_ns, sequence as u64 * 33_333_333);
    }
    // The last frame's surface is still the pointer the fixture reported, so the
    // keep-alive really is holding the Python allocation.
    assert_eq!(
        observed::<u64>("produced_ptr"),
        match &frames[1].domain {
            MemoryDomain::Cuda(buffer) => buffer.luma_ptr,
            _ => unreachable!(),
        }
    );
    assert!(matches!(sink.packets.last(), Some(PipelinePacket::Eos)));
}

#[test]
fn a_source_that_produces_a_mis_shaped_plane_is_refused() {
    if !cupy_available() {
        eprintln!("skipping: no cupy / no CUDA device on this host");
        return;
    }
    prepare_interpreter();
    let mut source = PySource::new("cuda_element", "BadCudaSource")
        .with_caps(nv12_caps())
        .with_cuda_frames(true);
    source.configure_pipeline(&nv12_caps()).unwrap();
    let mut sink = CollectSink::default();
    let result = current_runtime().block_on(source.run(&mut sink));
    assert!(
        result.is_err(),
        "a plane whose shape contradicts the caps must not become a frame"
    );
}

#[test]
fn a_source_without_the_produce_hook_is_refused() {
    prepare_interpreter();
    let mut source = PySource::new("cuda_element", "CpuOnly")
        .with_caps(nv12_caps())
        .with_cuda_frames(true);
    source.configure_pipeline(&nv12_caps()).unwrap();
    let mut sink = CollectSink::default();
    let result = current_runtime().block_on(source.run(&mut sink));
    assert!(
        matches!(result, Err(G2gError::UnsupportedDomain)),
        "got {result:?}"
    );
}
