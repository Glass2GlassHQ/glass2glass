//! M984: hand a GPU-resident frame to a hosted Python element with no readback.
//!
//! A `MemoryDomain::Cuda` frame routes to `g2g_process_cuda`, which receives the
//! NV12 planes as `g2g.CudaPlane` objects describing the device pointers through
//! `__cuda_array_interface__`. The routing / layout / lifetime tests never
//! dereference the pointers, so they run with no GPU; the aliasing test needs a
//! real device plus cupy and skips itself when either is missing.
#![cfg(feature = "python")]

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use g2g_core::memory::{CudaKeepAlive, OwnedCudaBuffer};
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

/// 1920x1080 at a decoder-style 2048-byte pitch, so pitch != width throughout.
const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const PITCH: u32 = 2048;
/// Never dereferenced: only the routing and the CAI description are under test.
const FAKE_LUMA: u64 = 0xdead_0000;
/// The byte `CupyConsumer` writes through the plane; mirrors the fixture.
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

/// Stands in for the decoder that owns the device allocation. The pointers here
/// are either fake or owned by a cupy array the fixture module keeps alive, so
/// there is nothing to release.
#[derive(Debug)]
struct NoOwner;
impl CudaKeepAlive for NoOwner {}

/// One plane's `__cuda_array_interface__` as Rust values.
#[derive(Debug)]
struct Cai {
    shape: Vec<usize>,
    strides: Vec<usize>,
    typestr: String,
    data: (u64, bool),
    version: u32,
}

/// Register the native `g2g` module and put the fixtures on the interpreter's
/// import path, keeping any inherited `PYTHONPATH` so a cupy install can be
/// pointed at from outside. Both must happen before the first GIL acquisition,
/// which here can be this test's own `Python::attach` rather than the element's.
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

fn nv12_caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
    }
}

/// A frame whose memory is a CUDA NV12 surface at `luma_ptr`, chroma following
/// after `HEIGHT` pitched rows (the layout NVDEC hands over).
fn cuda_frame(luma_ptr: u64, pitch: u32) -> Frame {
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
        timing: FrameTiming {
            pts_ns: 0,
            dts_ns: 0,
            duration_ns: 0,
            capture_ns: 0,
            arrival_ns: 0,
            keyframe: true,
        },
        sequence: 0,
        meta: Default::default(),
    }
}

/// Run one GPU frame through the hosted `class`, returning the process result and
/// what the element left downstream.
fn run_one(class: &str, frame: Frame) -> (Result<(), G2gError>, Vec<PipelinePacket>) {
    prepare_interpreter();
    let mut element = PyTransform::new("cuda_element", class).with_accept(nv12_caps());
    element.configure_pipeline(&nv12_caps()).unwrap();

    let mut sink = CollectSink::default();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let result = runtime.block_on(element.process(PipelinePacket::DataFrame(frame), &mut sink));
    (result, sink.packets)
}

fn fixture<'py>(py: Python<'py>) -> Bound<'py, PyModule> {
    PyModule::import(py, "cuda_element").expect("fixture importable")
}

/// Have the fixture allocate a real NV12 device surface with cupy. `None` when
/// this host has no cupy or no usable CUDA device.
fn allocate_surface() -> Option<(u64, u32)> {
    Python::attach(|py| {
        fixture(py)
            .call_method1("allocate_nv12", (WIDTH, HEIGHT, PITCH))
            .expect("allocate_nv12 reports its own failures as None")
            .extract()
            .unwrap()
    })
}

/// Read one luma byte through the producer's own cupy array.
fn read_surface(row: u32, column: u32) -> u32 {
    Python::attach(|py| {
        fixture(py)
            .call_method1("read_surface", (row, column))
            .expect("read_surface")
            .extract()
            .unwrap()
    })
}

/// Read one entry of the fixture module's `OBSERVED` dict, recorded during the
/// last `g2g_process_cuda` call.
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

/// The CAI dict the element saw for one plane.
fn observed_cai(key: &str) -> Cai {
    Python::attach(|py| {
        let dict: Py<PyDict> = observed(key);
        let dict = dict.bind(py);
        let get = |name: &str| {
            dict.get_item(name)
                .unwrap()
                .unwrap_or_else(|| panic!("CAI[{name}] missing"))
        };
        assert!(get("stream").is_none(), "no stream synchronization claimed");
        Cai {
            shape: get("shape").extract().unwrap(),
            strides: get("strides").extract().unwrap(),
            typestr: get("typestr").extract().unwrap(),
            data: get("data").extract().unwrap(),
            version: get("version").extract().unwrap(),
        }
    })
}

#[test]
fn cuda_frame_reaches_the_cuda_entry_point_as_cai_planes() {
    let (result, packets) = run_one("CudaProbe", cuda_frame(FAKE_LUMA, PITCH));
    result.expect("a Cuda-domain frame should route to g2g_process_cuda");

    // The frame flowed on untouched, still GPU-resident: no readback happened.
    assert_eq!(packets.len(), 1);
    let PipelinePacket::DataFrame(frame) = &packets[0] else {
        panic!("expected a DataFrame downstream");
    };
    let MemoryDomain::Cuda(buffer) = &frame.domain else {
        panic!("frame should still be in the Cuda domain");
    };
    assert_eq!(buffer.luma_ptr, FAKE_LUMA);
    assert_eq!(observed::<(u32, u32)>("geometry"), (WIDTH, HEIGHT));

    let luma = observed_cai("luma");
    assert_eq!(luma.shape, vec![1080, 1920]);
    assert_eq!(luma.strides, vec![2048, 1], "row stride is the pitch");
    assert_eq!(luma.typestr, "|u1");
    assert_eq!(luma.version, 3);
    assert_eq!(
        luma.data,
        (FAKE_LUMA, true),
        "the luma device pointer, exported read-only"
    );

    let chroma = observed_cai("chroma");
    assert_eq!(chroma.shape, vec![540, 960, 2], "interleaved UV pairs");
    assert_eq!(chroma.strides, vec![2048, 2, 1]);
    assert_eq!(
        chroma.data.0,
        FAKE_LUMA + u64::from(PITCH) * u64::from(HEIGHT)
    );
}

#[test]
fn element_without_the_cuda_entry_point_refuses_a_gpu_frame() {
    let (result, packets) = run_one("CpuOnly", cuda_frame(FAKE_LUMA, PITCH));
    assert!(
        matches!(result, Err(G2gError::UnsupportedDomain)),
        "a CPU-only element must not silently receive a GPU frame, got {result:?}"
    );
    assert!(packets.is_empty(), "nothing should flow downstream");
}

#[test]
fn retaining_a_plane_past_the_call_fails_the_frame() {
    let (result, packets) = run_one("PlaneRetainer", cuda_frame(FAKE_LUMA, PITCH));
    assert!(
        result.is_err(),
        "a plane kept past return would dangle; the host must reject it"
    );
    assert!(packets.is_empty());
}

#[test]
fn cupy_array_aliases_the_producers_device_memory() {
    prepare_interpreter();
    let Some((luma_ptr, pitch)) = allocate_surface() else {
        eprintln!("skipping: no cupy / no CUDA device on this host");
        return;
    };

    let (result, packets) = run_one("CupyConsumer", cuda_frame(luma_ptr, pitch));
    result.expect("the cupy consumer should complete");
    assert_eq!(packets.len(), 1);

    // Aliasing, not a copy: cupy's array data pointer is the producer's device
    // pointer, byte for byte, and the pitched strides survived the handoff.
    assert_eq!(observed::<u64>("cai_ptr"), luma_ptr);
    assert_eq!(observed::<u64>("luma_ptr"), luma_ptr);
    assert_eq!(
        observed::<u64>("chroma_ptr"),
        luma_ptr + u64::from(pitch) * u64::from(HEIGHT)
    );
    assert_eq!(
        observed::<(usize, usize)>("luma_shape"),
        (HEIGHT as usize, WIDTH as usize)
    );
    assert_eq!(observed::<(usize, usize)>("luma_strides"), (2048, 1));
    assert_eq!(
        observed::<(usize, usize, usize)>("chroma_shape"),
        (540, 960, 2)
    );
    // The producer's pixels read back through the CAI arrays.
    assert!(observed::<bool>("pattern_matches"), "luma pattern differed");
    assert!(observed::<bool>("chroma_matches"), "chroma value differed");

    // And a byte written through the CAI array is visible in the producer's own
    // array: one allocation, two views of it.
    let marker = read_surface(1, 2);
    assert_eq!(
        marker, MARKER_VALUE,
        "write through the plane did not reach the producer's surface"
    );
}
