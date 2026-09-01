//! M988: does one worker thread per hosted element actually parallelize on a
//! free-threaded (PEP 703) interpreter?
//!
//! The design claim is that a `PyWorker` OS thread per element is the
//! free-threading unit: the same code GIL-serializes on a stock interpreter and
//! runs in parallel on a free-threaded one. This measures it. `WORKERS` elements
//! each run one compute-bound *pure Python* frame callback; the speedup is the
//! serial cost (`WORKERS` x one call) over the measured wall clock of running them
//! at once. A stock build lands near 1.0, a free-threaded build near `WORKERS`.
//!
//! Timing-dependent, so it is `#[ignore]`d and never runs in a normal suite. To
//! run it, point the build at the interpreter under test (pyo3 picks the
//! interpreter up at build time, so each one needs its own target directory):
//!
//! ```text
//! # stock
//! cargo test -p g2g-python --features python --test m988_gil_offload -- --ignored --nocapture
//!
//! # free-threaded (uv python install 3.14.2+freethreaded)
//! FT=$(uv python find 3.14.2+freethreaded)
//! PYO3_PYTHON=$FT \
//!   LD_LIBRARY_PATH=$(dirname $(dirname $FT))/lib \
//!   CARGO_TARGET_DIR=target/freethreaded \
//!   cargo test -p g2g-python --features python --test m988_gil_offload -- --ignored --nocapture
//! ```
#![cfg(feature = "python")]

use std::time::{Duration, Instant};

use pyo3::prelude::*;

use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, Dim, Frame, FrameTiming, G2gError, MemoryDomain, OutputSink,
    PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_python::PyTransform;

/// Enough workers to show scaling without needing a big machine.
const WORKERS: usize = 4;
/// A free-threaded run must recover at least this much of the ideal `WORKERS`x.
/// Loose on purpose: the point is "parallel, not serialized", and a loaded machine
/// still clears it.
const PARALLEL_SPEEDUP_FLOOR: f64 = 2.0;
/// A GIL build cannot exceed this: the Python work is one lock's worth of
/// interpreter time no matter how many workers hold it.
const SERIALIZED_SPEEDUP_CEILING: f64 = 1.6;

struct CollectSink;

impl OutputSink for CollectSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(2),
        height: Dim::Fixed(1),
        framerate: Rate::Fixed(30),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame() -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 8].into_boxed_slice())),
        timing: FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    }
}

/// A hosted element with its worker thread already spawned.
fn spin_element() -> PyTransform {
    let mut element = PyTransform::new("gil_element", "SpinTransform");
    element
        .configure_pipeline(&caps())
        .expect("the fixture element should instantiate");
    element
}

/// Run one frame through `element` on this thread and return the wall clock. Each
/// caller gets its own single-threaded runtime, so the only concurrency in play is
/// the workers' own OS threads, which is exactly the mechanism under test.
fn one_frame(element: &mut PyTransform) -> Duration {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let mut sink = CollectSink;
    let start = Instant::now();
    runtime
        .block_on(element.process(PipelinePacket::DataFrame(frame()), &mut sink))
        .expect("the hosted element should process the frame");
    start.elapsed()
}

/// Whether the interpreter still has the GIL off, as the fixture saw it from
/// inside a call (so after its `import g2g`), plus the interpreter banner.
fn interpreter_state() -> (bool, String) {
    Python::attach(|py| {
        let state = PyModule::import(py, "gil_element")
            .expect("fixture importable")
            .getattr("STATE")
            .unwrap();
        let gil_enabled = state
            .get_item("gil_enabled")
            .expect("a frame must have run first")
            .extract()
            .unwrap();
        let version: String = state.get_item("version").unwrap().extract().unwrap();
        (gil_enabled, version.replace('\n', " "))
    })
}

#[test]
#[ignore = "timing measurement, run explicitly (see the module docs)"]
fn workers_parallelize_only_without_the_gil() {
    g2g_python::init_host();
    std::env::set_var(
        "PYTHONPATH",
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures"),
    );

    // One element alone: the serial cost of a single frame's Python work. Run it
    // twice and keep the faster, so a cold interpreter or a first-touch page fault
    // does not inflate the baseline.
    let mut solo = spin_element();
    let single = one_frame(&mut solo).min(one_frame(&mut solo));

    let mut elements: Vec<PyTransform> = (0..WORKERS).map(|_| spin_element()).collect();
    let start = Instant::now();
    std::thread::scope(|scope| {
        for element in elements.iter_mut() {
            scope.spawn(move || one_frame(element));
        }
    });
    let concurrent = start.elapsed();

    let (gil_enabled, version) = interpreter_state();
    let speedup = (single.as_secs_f64() * WORKERS as f64) / concurrent.as_secs_f64();
    println!("interpreter: {version}");
    println!("GIL enabled in-process: {gil_enabled}");
    println!(
        "one frame {:?}, {WORKERS} frames on {WORKERS} workers {:?}, speedup {speedup:.2}x of {WORKERS}x ideal",
        single, concurrent
    );

    if gil_enabled {
        assert!(
            speedup < SERIALIZED_SPEEDUP_CEILING,
            "a GIL interpreter cannot run {WORKERS} Python callbacks in parallel, \
             yet the speedup was {speedup:.2}x"
        );
    } else {
        assert!(
            speedup >= PARALLEL_SPEEDUP_FLOOR,
            "on a free-threaded interpreter {WORKERS} worker threads should run in \
             parallel, but the speedup was only {speedup:.2}x"
        );
    }
}
