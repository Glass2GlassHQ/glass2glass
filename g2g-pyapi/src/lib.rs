//! Python bindings to drive glass2glass pipelines (the inverse of `g2g-python`,
//! which hosts Python *elements* inside a pipeline). Sits on the same
//! language-neutral waist as `g2g-capi`: describe a pipeline as a string, run
//! it, watch the bus, and push/pull buffers via `appsrc` / `appsink`.
//!
//! Without the `python` feature the crate is empty, so `cargo check --workspace`
//! needs no libpython. With it, `#[pymodule] g2g` exposes `Pipeline`, `AppSrc`,
//! and `AppSink`.
//!
//! ```python
//! import g2g
//! src  = g2g.AppSrc("cam")
//! sink = g2g.AppSink("out")
//! p = g2g.Pipeline("appsrc channel=cam caps=video/x-raw,format=RGBA,"
//!                   "width=2,height=2,framerate=30/1 ! appsink channel=out")
//! src.push(b"\x00" * 16, 0); src.end_of_stream()
//! view = sink.pull()               # a zero-copy FrameView, None at end of stream
//! frame = np.frombuffer(view, np.uint8)   # no copy; view owns the buffer
//! pts = view.pts_ns
//! p.wait()
//! ```
//!
//! `AppSink.pull()` lends the frame through the buffer protocol (the
//! [`FrameView`] owns the [`Frame`], so the buffer outlives any `memoryview` /
//! numpy array over it). Pass `timeout_ms` for a bounded blocking pull.

#![cfg_attr(not(feature = "python"), allow(unused_crate_dependencies))]

#[cfg(feature = "python")]
mod pymod {
    use pyo3::exceptions::{PyBufferError, PyValueError};
    use pyo3::prelude::*;

    use std::os::raw::c_int;
    use std::thread::JoinHandle;
    use std::time::{Duration, Instant};

    use g2g_core::frame::Frame;
    use g2g_core::runtime::{block_on, parse_launch, run_graph_with_bus, RunStats};
    use g2g_core::{Bus, BusMessage, G2gError, MemoryDomain};
    use g2g_plugins::appsink::{register_appsink_pull, AppSinkPull, Pull};
    use g2g_plugins::appsrc::{register_appsrc, AppSrcFeed};
    use g2g_plugins::clock::WallClock;
    use g2g_plugins::registry::default_registry;

    const LINK_CAPACITY: usize = 4;
    const BUS_CAPACITY: usize = 64;

    /// A running pipeline parsed from a `gst-launch`-style string. Runs on a
    /// background thread; poll the bus and `wait()` for the end of stream.
    #[pyclass]
    struct Pipeline {
        bus: Bus,
        join: Option<JoinHandle<Result<RunStats, G2gError>>>,
        // The joined outcome, cached so a second `wait()` is idempotent. The
        // error arm is a formatted message (not the raw `G2gError`) so it can
        // also carry a run-thread panic, which is not a `G2gError`.
        result: Option<Result<RunStats, String>>,
    }

    /// A human-readable message for each `G2gError` variant, so `Pipeline::wait`
    /// surfaces *why* a run failed instead of a single opaque "pipeline errored".
    fn describe_error(e: &G2gError) -> String {
        match e {
            G2gError::CapsMismatch => {
                "caps negotiation failed (no common format between linked elements)".into()
            }
            G2gError::NotConfigured => {
                "an element received data before configuration completed".into()
            }
            G2gError::FixationFailed => "caps fixation failed".into(),
            G2gError::PoolExhausted => "buffer pool exhausted".into(),
            G2gError::UnsupportedDomain => {
                "an element was handed a memory domain it cannot consume".into()
            }
            G2gError::AllocationConflict => "fan-out branches share no common memory domain".into(),
            G2gError::Hardware(h) => format!("hardware/driver failure: {h:?}"),
            G2gError::Shutdown => "pipeline shut down before completing".into(),
            other => format!("pipeline error: {other:?}"),
        }
    }

    /// Best-effort extraction of a panic payload's message (the common
    /// `&str` / `String` cases a `panic!` / `expect` produces).
    fn panic_message(payload: &(dyn core::any::Any + Send)) -> String {
        if let Some(s) = payload.downcast_ref::<&str>() {
            (*s).into()
        } else if let Some(s) = payload.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic".into()
        }
    }

    #[pymethods]
    impl Pipeline {
        #[new]
        fn new(description: &str) -> PyResult<Self> {
            let reg = default_registry();
            let graph = parse_launch(&reg, description)
                .map_err(|e| PyValueError::new_err(format!("parse error: {e}")))?;
            let (bus, handle) = Bus::new(BUS_CAPACITY);
            let join = std::thread::Builder::new()
                .name("g2g-pyapi-run".into())
                .spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_time()
                        .build()
                        .expect("build tokio runtime");
                    let clock = WallClock::new();
                    rt.block_on(run_graph_with_bus(graph, &clock, LINK_CAPACITY, &handle))
                })
                .map_err(|e| PyValueError::new_err(format!("spawn run thread: {e}")))?;
            Ok(Pipeline {
                bus,
                join: Some(join),
                result: None,
            })
        }

        /// Pop one bus message as `(kind, text_or_None, a, b)`, or `None` if the
        /// bus is empty. `kind` is a lowercase string; `a`/`b` are kind-specific
        /// (see the C header).
        fn bus_poll(&self) -> Option<(String, Option<String>, u64, u64)> {
            self.bus.try_recv().map(|m| project(&m))
        }

        /// True once the run thread has finished (EOS or error).
        fn is_done(&self) -> bool {
            self.join.as_ref().map_or(true, |j| j.is_finished())
        }

        /// Block until the pipeline ends; returns `(emitted, consumed, dropped)`.
        /// Raises on a pipeline error, carrying the underlying cause (a described
        /// `G2gError`, or the run-thread panic message) rather than an opaque
        /// "pipeline errored". Releases the GIL while blocking.
        fn wait(&mut self, py: Python<'_>) -> PyResult<(u64, u64, u64)> {
            if let Some(j) = self.join.take() {
                // Detach the GIL while joining. A clean run returns
                // Ok/Err(G2gError); a panic in the run thread surfaces as the
                // join Err, which we describe rather than swallow.
                self.result = Some(match py.detach(|| j.join()) {
                    Ok(Ok(s)) => Ok(s),
                    Ok(Err(e)) => Err(describe_error(&e)),
                    Err(panic) => Err(format!(
                        "pipeline run thread panicked: {}",
                        panic_message(&*panic)
                    )),
                });
            }
            match &self.result {
                Some(Ok(s)) => Ok((s.frames_emitted, s.frames_consumed, s.frames_dropped)),
                Some(Err(msg)) => Err(PyValueError::new_err(format!("pipeline error: {msg}"))),
                // `wait()` always sets `result` above, so this is unreachable in
                // practice; kept as a non-panicking fallback.
                None => Err(PyValueError::new_err("pipeline error: result unavailable")),
            }
        }
    }

    /// Application push source feeding `appsrc channel=<name>`.
    #[pyclass]
    struct AppSrc {
        feed: AppSrcFeed,
    }

    #[pymethods]
    impl AppSrc {
        #[new]
        #[pyo3(signature = (channel="default"))]
        fn new(channel: &str) -> Self {
            AppSrc {
                feed: register_appsrc(channel),
            }
        }

        /// Push a buffer (copied) with timestamp `pts_ns`. False if the feed is
        /// full (retry) or the pipeline is gone.
        #[pyo3(signature = (data, pts_ns=0))]
        fn push(&self, data: &[u8], pts_ns: u64) -> bool {
            self.feed.push(data, pts_ns)
        }

        /// Signal end-of-stream.
        fn end_of_stream(&self) -> bool {
            self.feed.end_of_stream()
        }
    }

    /// A pulled sample, lent to Python zero-copy through the buffer protocol.
    /// Owns the [`Frame`], so the lent bytes stay valid for the whole life of
    /// the view (and of any `memoryview` / numpy array over it): CPython increfs
    /// this object into the buffer view, so it cannot be dropped while a view is
    /// live. Wrap it with `memoryview(view)` or `np.frombuffer(view, np.uint8)`;
    /// the bytes are lent read-only (the frame is pipeline-owned), so copy if you
    /// need to mutate.
    #[pyclass]
    #[derive(Debug)]
    struct FrameView {
        frame: Frame,
        /// A contiguous copy made once for a strided `SystemView` (so the lent
        /// pointer is stable); `None` for a `System` frame, lent directly with
        /// no copy at all.
        materialized: Option<Box<[u8]>>,
        /// Presentation timestamp, in nanoseconds.
        #[pyo3(get)]
        pts_ns: u64,
    }

    impl FrameView {
        fn new(frame: Frame) -> Self {
            let pts_ns = frame.timing.pts_ns;
            let materialized = match &frame.domain {
                MemoryDomain::SystemView(sv) => Some(sv.materialize()),
                _ => None,
            };
            FrameView {
                frame,
                materialized,
                pts_ns,
            }
        }

        /// The host-visible bytes to lend, or `None` for a non-host (GPU /
        /// foreign) domain that has no CPU-mapped buffer.
        fn host_bytes(&self) -> Option<(*const u8, usize)> {
            if let Some(b) = &self.materialized {
                return Some((b.as_ptr(), b.len()));
            }
            self.frame
                .domain
                .as_system_slice()
                .map(|sl| (sl.as_ptr(), sl.len()))
        }
    }

    #[pymethods]
    impl FrameView {
        /// Number of host-visible bytes (0 for a non-host domain).
        #[getter]
        fn nbytes(&self) -> usize {
            self.host_bytes().map_or(0, |(_, len)| len)
        }

        unsafe fn __getbuffer__(
            slf: PyRef<'_, Self>,
            view: *mut pyo3::ffi::Py_buffer,
            flags: c_int,
        ) -> PyResult<()> {
            if view.is_null() {
                return Err(PyBufferError::new_err("null buffer view"));
            }
            let Some((ptr, len)) = slf.host_bytes() else {
                return Err(PyBufferError::new_err(
                    "frame is not host-visible (GPU / foreign memory domain)",
                ));
            };
            // SAFETY: `view` is a valid out-pointer supplied by CPython.
            // `PyBuffer_FillInfo` increfs the exporter (`slf`) into `view->obj`,
            // so this `FrameView`, the `Frame` it owns, and thus the lent bytes
            // all outlive the view; CPython decrefs on release. Lent read-only
            // (`readonly = 1`), so Python cannot mutate pipeline-owned memory.
            let ret = unsafe {
                pyo3::ffi::PyBuffer_FillInfo(
                    view,
                    slf.as_ptr(),
                    ptr as *mut core::ffi::c_void,
                    len as pyo3::ffi::Py_ssize_t,
                    1, // readonly
                    flags,
                )
            };
            if ret == -1 {
                Err(PyErr::take(slf.py()).unwrap_or_else(|| PyBufferError::new_err("fill failed")))
            } else {
                Ok(())
            }
        }

        unsafe fn __releasebuffer__(&self, _view: *mut pyo3::ffi::Py_buffer) {
            // The lent bytes are owned by `self.frame` / `self.materialized`;
            // CPython manages the exporter refcount, so release is a no-op.
        }
    }

    /// Application pull sink draining `appsink channel=<name>`.
    #[pyclass]
    struct AppSink {
        pull: AppSinkPull,
    }

    #[pymethods]
    impl AppSink {
        #[new]
        #[pyo3(signature = (channel="default"))]
        fn new(channel: &str) -> Self {
            AppSink {
                pull: register_appsink_pull(channel),
            }
        }

        /// Block for the next sample, returning a zero-copy [`FrameView`] or
        /// `None` once the stream ends. With `timeout_ms`, returns `None` if no
        /// sample arrives in that window; poll again, and use the pipeline's
        /// `is_done()` to tell a timeout from a real end of stream. Releases the
        /// GIL while blocking.
        #[pyo3(signature = (timeout_ms=None))]
        fn pull(&self, py: Python<'_>, timeout_ms: Option<u64>) -> Option<FrameView> {
            let frame = py.detach(|| match timeout_ms {
                None => block_on(self.pull.pull()),
                Some(ms) => pull_timeout(&self.pull, ms),
            })?;
            Some(FrameView::new(frame))
        }

        /// Non-blocking: a zero-copy [`FrameView`] if a sample is ready, else
        /// `None` (whether the stream is merely idle or fully ended).
        fn try_pull(&self) -> Option<FrameView> {
            match self.pull.try_pull() {
                Pull::Frame(frame) => Some(FrameView::new(frame)),
                Pull::Empty | Pull::Ended => None,
            }
        }
    }

    /// Poll for a frame until one arrives or `timeout_ms` elapses. `None` on a
    /// timeout or once the stream has ended (the caller distinguishes via the
    /// pipeline's `is_done()`). Runs with the GIL released.
    fn pull_timeout(pull: &AppSinkPull, timeout_ms: u64) -> Option<Frame> {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            match pull.try_pull() {
                Pull::Frame(f) => return Some(f),
                Pull::Ended => return None,
                Pull::Empty => {
                    if Instant::now() >= deadline {
                        return None;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        }
    }

    /// Flatten a [`BusMessage`] to `(kind, text, a, b)`; mirrors the C ABI shape.
    fn project(msg: &BusMessage) -> (String, Option<String>, u64, u64) {
        match msg {
            BusMessage::StreamStart => ("stream-start".into(), None, 0, 0),
            BusMessage::Eos => ("eos".into(), None, 0, 0),
            BusMessage::Info(m) => ("info".into(), Some(m.clone()), 0, 0),
            BusMessage::Error(e) => ("error".into(), Some(format!("{e:?}")), 0, 0),
            BusMessage::Warning(e) => ("warning".into(), Some(format!("{e:?}")), 0, 0),
            BusMessage::Buffering { percent } => ("buffering".into(), None, u64::from(*percent), 0),
            BusMessage::DurationChanged { duration_ns } => {
                ("duration-changed".into(), None, *duration_ns, 0)
            }
            _ => ("other".into(), None, 0, 0),
        }
    }

    #[pymodule]
    fn g2g(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Pipeline>()?;
        m.add_class::<AppSrc>()?;
        m.add_class::<AppSink>()?;
        m.add_class::<FrameView>()?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::g2g;
        use pyo3::append_to_inittab;
        use pyo3::prelude::*;
        use pyo3::types::PyDict;
        use std::ffi::CString;
        use std::sync::Once;

        /// Register the native `g2g` module exactly once across all tests:
        /// `append_to_inittab` must run before the interpreter initializes, and
        /// the test binary shares one interpreter, so a second call panics.
        fn ensure_module() {
            static INIT: Once = Once::new();
            INIT.call_once(|| append_to_inittab!(g2g));
        }

        /// Drives the whole binding from Python through an embedded interpreter:
        /// build a pipeline, push via AppSrc, pull via AppSink, wait for stats.
        #[test]
        fn python_drives_appsrc_through_appsink() {
            // Register the module before the interpreter initializes.
            ensure_module();
            Python::attach(|py| {
                let script = CString::new(
                    r#"
import g2g
src  = g2g.AppSrc("pyc")
sink = g2g.AppSink("pyo")
p = g2g.Pipeline(
    "appsrc channel=pyc caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1"
    " ! appsink channel=pyo"
)
assert src.push(b"\x01" * 16, 0)
assert src.push(b"\x02" * 16, 1000)
src.end_of_stream()

a = sink.pull()
assert a is not None
# Zero-copy: a FrameView (not bytes), lent through the buffer protocol.
assert not isinstance(a, (bytes, tuple)), type(a)
mv = memoryview(a)
assert mv.readonly, "the lent buffer is read-only"
assert a.nbytes == 16 and bytes(mv) == b"\x01" * 16 and a.pts_ns == 0, a
b = sink.pull(timeout_ms=1000)      # bounded blocking pull
assert bytes(memoryview(b)) == b"\x02" * 16 and b.pts_ns == 1000
assert sink.pull() is None          # end of stream

emitted, consumed, dropped = p.wait()
assert consumed == 2, consumed
"#,
                )
                .unwrap();
                // Fresh globals so concurrent tests sharing the one interpreter
                // do not clobber each other's module-level variables.
                let globals = PyDict::new(py);
                py.run(script.as_c_str(), Some(&globals), None)
                    .expect("python script runs");
            });
        }

        /// The zero-copy lend is sound: a `FrameView` owns its frame, so a
        /// `memoryview` (or numpy array) over it stays valid after later pulls
        /// and after the pipeline is torn down, never a use-after-free.
        #[test]
        fn frame_view_outlives_later_pulls_and_pipeline() {
            ensure_module();
            Python::attach(|py| {
                let script = CString::new(
                    r#"
import g2g
src  = g2g.AppSrc("lc")
sink = g2g.AppSink("lo")
p = g2g.Pipeline(
    "appsrc channel=lc caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1"
    " ! appsink channel=lo"
)
src.push(b"\xAA" * 16, 0)
src.push(b"\xBB" * 16, 1)
src.end_of_stream()

a = sink.pull()
held = memoryview(a)            # retain a view over the first frame
b = sink.pull()                 # pull the next frame
assert bytes(memoryview(b)) == b"\xBB" * 16
assert sink.pull() is None
p.wait()                        # tear the pipeline down

# `held` (and `a`) still read the first frame's bytes: the FrameView owns
# the memory, so nothing above invalidated it.
assert bytes(held) == b"\xAA" * 16, bytes(held)

try:
    import numpy as np
    arr = np.frombuffer(a, np.uint8)
    assert arr.flags["OWNDATA"] is False, "numpy must share the buffer, not copy"
    assert int(arr[0]) == 0xAA and arr.nbytes == 16
except ImportError:
    pass
"#,
                )
                .unwrap();
                let globals = PyDict::new(py);
                py.run(script.as_c_str(), Some(&globals), None)
                    .expect("lifetime script runs");
            });
        }

        /// The non-blocking `try_pull()` path: spin until a sample is ready
        /// (the run thread delivers asynchronously), confirm it lends the same
        /// zero-copy `FrameView`, then `None` once the stream has ended.
        #[test]
        fn try_pull_returns_frame_then_none_at_end() {
            ensure_module();
            Python::attach(|py| {
                let script = CString::new(
                    r#"
import g2g
import time
src  = g2g.AppSrc("tc")
sink = g2g.AppSink("to")
p = g2g.Pipeline(
    "appsrc channel=tc caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1"
    " ! appsink channel=to"
)
src.push(b"\xCD" * 16, 7)
src.end_of_stream()

# try_pull is non-blocking and may race the run thread, so spin briefly.
frame = None
for _ in range(1000):
    frame = sink.try_pull()
    if frame is not None:
        break
    time.sleep(0.001)
assert frame is not None, "try_pull never delivered the pushed frame"
assert not isinstance(frame, (bytes, tuple)), type(frame)
assert bytes(memoryview(frame)) == b"\xCD" * 16 and frame.pts_ns == 7, frame

# Once drained and ended, try_pull settles on None (empty or ended both map
# to None); wait() confirms the single frame was consumed.
for _ in range(1000):
    if sink.try_pull() is None and p.is_done():
        break
    time.sleep(0.001)
assert sink.try_pull() is None
_, consumed, _ = p.wait()
assert consumed == 1, consumed
"#,
                )
                .unwrap();
                let globals = PyDict::new(py);
                py.run(script.as_c_str(), Some(&globals), None)
                    .expect("try_pull script runs");
            });
        }

        /// `Pipeline::wait` surfaces *why* a run failed: every `G2gError` variant
        /// maps to a distinct, non-empty message, and never the old opaque
        /// "pipeline errored" placeholder. Guards against a regression to a lossy
        /// error path. No interpreter needed: the formatter is a pure function.
        #[test]
        fn describe_error_is_specific_per_variant() {
            use super::describe_error;
            use g2g_core::{G2gError, HardwareError};

            let cases = [
                G2gError::CapsMismatch,
                G2gError::NotConfigured,
                G2gError::FixationFailed,
                G2gError::PoolExhausted,
                G2gError::UnsupportedDomain,
                G2gError::AllocationConflict,
                G2gError::Hardware(HardwareError::Cuda(2)),
                G2gError::Shutdown,
            ];
            let msgs: Vec<String> = cases.iter().map(describe_error).collect();

            // Every message is informative, not the discarded opaque text.
            for m in &msgs {
                assert!(!m.is_empty());
                assert_ne!(m, "pipeline errored");
            }
            // Variants map to distinct messages (no information collapse).
            for i in 0..msgs.len() {
                for j in (i + 1)..msgs.len() {
                    assert_ne!(msgs[i], msgs[j], "variants {i} and {j} share a message");
                }
            }
            // The hardware arm carries the backend detail (the CUDA code).
            assert!(
                describe_error(&G2gError::Hardware(HardwareError::Cuda(2))).contains('2'),
                "hardware error should carry its backend code"
            );
            assert!(describe_error(&G2gError::CapsMismatch).contains("caps"));
        }

        /// A run-thread panic is surfaced (not swallowed into `Shutdown`): the
        /// `&str` / `String` payloads a `panic!` / `expect` produce come through.
        #[test]
        fn panic_message_extracts_common_payloads() {
            use super::panic_message;
            use core::any::Any;

            let s: Box<dyn Any + Send> = Box::new("boom");
            assert_eq!(panic_message(&*s), "boom");
            let owned: Box<dyn Any + Send> = Box::new(String::from("kaboom"));
            assert_eq!(panic_message(&*owned), "kaboom");
            let other: Box<dyn Any + Send> = Box::new(42i32);
            assert_eq!(panic_message(&*other), "unknown panic");
        }
    }
}
