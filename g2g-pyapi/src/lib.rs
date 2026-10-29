//! Python bindings to drive glass2glass pipelines (the inverse of `g2g-python`,
//! which hosts Python *elements* inside a pipeline).
//!
//! Experimental (Tier 3 in `STABILITY.md`): tracks `g2g-core`, no stability
//! promise of its own. Sits on the same
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
//! `FrameView` owns the [`Frame`](g2g_core::Frame), so the buffer outlives any `memoryview` /
//! numpy array over it). Pass `timeout_ms` for a bounded blocking pull.

#![cfg_attr(not(feature = "python"), allow(unused_crate_dependencies))]

#[cfg(feature = "python")]
mod pymod {
    use pyo3::exceptions::{PyBufferError, PyValueError};
    use pyo3::prelude::*;

    use std::os::raw::c_int;
    use std::time::Duration;

    use g2g_core::frame::Frame;
    use g2g_core::runtime::block_on;
    use g2g_plugins::appsink::{register_appsink_pull, AppSinkPull, HostFrame, Pull};
    use g2g_plugins::appsrc::{register_appsrc, AppSrcFeed};
    use g2g_plugins::launched::{EditError, LaunchedPipeline};

    /// A running pipeline parsed from a `gst-launch`-style string. Runs on a
    /// background thread; poll the bus and `wait()` for the end of stream.
    #[pyclass]
    struct Pipeline {
        launched: LaunchedPipeline,
    }

    fn refused(error: EditError) -> PyErr {
        PyValueError::new_err(error.to_string())
    }

    #[pymethods]
    impl Pipeline {
        #[new]
        fn new(description: &str) -> PyResult<Self> {
            LaunchedPipeline::launch(description, "g2g-pyapi-run")
                .map(|launched| Pipeline { launched })
                .map_err(|error| PyValueError::new_err(error.to_string()))
        }

        fn insert_after(&self, py: Python<'_>, node: &str, description: &str) -> PyResult<String> {
            py.detach(|| self.launched.insert_after(node, description))
                .map_err(refused)
        }

        fn insert_before(&self, py: Python<'_>, node: &str, description: &str) -> PyResult<String> {
            py.detach(|| self.launched.insert_before(node, description))
                .map_err(refused)
        }

        fn insert_chain_after(
            &self,
            py: Python<'_>,
            node: &str,
            description: &str,
        ) -> PyResult<Vec<String>> {
            py.detach(|| self.launched.insert_chain_after(node, description))
                .map_err(refused)
        }

        fn insert_chain_before(
            &self,
            py: Python<'_>,
            node: &str,
            description: &str,
        ) -> PyResult<Vec<String>> {
            py.detach(|| self.launched.insert_chain_before(node, description))
                .map_err(refused)
        }

        fn add_branch(
            &self,
            py: Python<'_>,
            tee: &str,
            description: &str,
        ) -> PyResult<Vec<String>> {
            py.detach(|| self.launched.add_branch(tee, description))
                .map_err(refused)
        }

        fn remove_branch(&self, py: Python<'_>, sink: &str) -> PyResult<()> {
            py.detach(|| self.launched.remove_branch(sink))
                .map_err(refused)
        }

        fn add_input(&self, py: Python<'_>, muxer: &str, description: &str) -> PyResult<String> {
            py.detach(|| self.launched.add_input(muxer, description))
                .map_err(refused)
        }

        fn remove(&self, py: Python<'_>, node: &str) -> PyResult<()> {
            py.detach(|| self.launched.remove(node)).map_err(refused)
        }

        fn replace_sink(&self, py: Python<'_>, node: &str, description: &str) -> PyResult<String> {
            py.detach(|| self.launched.replace_sink(node, description))
                .map_err(refused)
        }

        fn replace_source(
            &self,
            py: Python<'_>,
            node: &str,
            description: &str,
        ) -> PyResult<String> {
            py.detach(|| self.launched.replace_source(node, description))
                .map_err(refused)
        }

        fn set_property(
            &self,
            py: Python<'_>,
            node: &str,
            name: &str,
            value: &str,
        ) -> PyResult<()> {
            py.detach(|| self.launched.set_property(node, name, value))
                .map_err(refused)
        }

        fn get_property(&self, py: Python<'_>, node: &str, name: &str) -> PyResult<Option<String>> {
            py.detach(|| self.launched.get_property(node, name))
                .map_err(refused)
        }

        /// Pop one bus message as `(kind, text_or_None, a, b)`, or `None` if the
        /// bus is empty. `kind` is a lowercase string; `a`/`b` are kind-specific
        /// (see the C header).
        fn bus_poll(&self) -> Option<(String, Option<String>, u64, u64)> {
            self.launched
                .bus_poll()
                .map(|flat| (flat.kind.name().to_string(), flat.text, flat.a, flat.b))
        }

        /// True once the run thread has finished (EOS or error).
        fn is_done(&self) -> bool {
            self.launched.is_done()
        }

        /// Block until the pipeline ends; returns `(emitted, consumed, dropped)`.
        /// Raises on a pipeline error, carrying the underlying cause (a described
        /// `G2gError`, or the run-thread panic message) rather than an opaque
        /// "pipeline errored". Releases the GIL while blocking.
        fn wait(&self, py: Python<'_>) -> PyResult<(u64, u64, u64)> {
            py.detach(|| self.launched.wait())
                .map(|stats| {
                    (
                        stats.frames_emitted,
                        stats.frames_consumed,
                        stats.frames_dropped,
                    )
                })
                .map_err(|failure| PyValueError::new_err(format!("pipeline error: {failure}")))
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
        host: HostFrame,
        /// Presentation timestamp, in nanoseconds.
        #[pyo3(get)]
        pts_ns: u64,
    }

    impl FrameView {
        fn new(frame: Frame) -> Self {
            let pts_ns = frame.timing.pts_ns;
            FrameView {
                host: HostFrame::new(frame),
                pts_ns,
            }
        }
    }

    #[pymethods]
    impl FrameView {
        /// Number of host-visible bytes (0 for a non-host domain).
        #[getter]
        fn nbytes(&self) -> usize {
            self.host.bytes().map_or(0, <[u8]>::len)
        }

        unsafe fn __getbuffer__(
            slf: PyRef<'_, Self>,
            view: *mut pyo3::ffi::Py_buffer,
            flags: c_int,
        ) -> PyResult<()> {
            if view.is_null() {
                return Err(PyBufferError::new_err("null buffer view"));
            }
            let Some(bytes) = slf.host.bytes() else {
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
                    bytes.as_ptr().cast_mut().cast(),
                    bytes.len() as pyo3::ffi::Py_ssize_t,
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
            // The lent bytes are owned by `self.host`;
            // CPython manages the exporter refcount, so release is a no-op.
        }
    }

    /// Application pull sink draining `appsink channel=<name>`.
    #[pyclass]
    struct AppSink {
        pull: AppSinkPull,
        // runs the timer for `pull(timeout_ms=...)`
        timer: tokio::runtime::Runtime,
    }

    #[pymethods]
    impl AppSink {
        #[new]
        #[pyo3(signature = (channel="default"))]
        fn new(channel: &str) -> PyResult<Self> {
            let timer = tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .map_err(|e| PyValueError::new_err(format!("build pull timer: {e}")))?;
            Ok(AppSink {
                pull: register_appsink_pull(channel),
                timer,
            })
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
                Some(ms) => self
                    .timer
                    .block_on(async {
                        tokio::time::timeout(Duration::from_millis(ms), self.pull.pull()).await
                    })
                    .ok()
                    .flatten(),
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
assert sink.pull(timeout_ms=10) is None, "nothing pushed yet, so the pull times out"
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

        #[test]
        fn python_mutates_a_running_pipeline() {
            ensure_module();
            Python::attach(|py| {
                let script = CString::new(
                    r#"
import g2g
FRAMES = 90
p = g2g.Pipeline(
    f"videotestsrc name=src num-buffers={FRAMES} ! clocksync name=pace ! fakesink name=sink"
)

inserted = p.insert_after("src", "valve drop=false")
assert inserted not in ("src", "pace", "sink"), inserted
assert p.get_property(inserted, "drop") == "false"
p.set_property(inserted, "drop-mode", "forward-sticky-events")
assert p.get_property(inserted, "drop-mode") == "forward-sticky-events"
assert p.get_property(inserted, "no-such-property") is None

try:
    p.set_property(inserted, "drop", "maybe")
    raise AssertionError("an unparseable value was accepted")
except ValueError as error:
    assert "PropertyRejected" in str(error), error

try:
    p.insert_after("src", "identity name=extra")
    raise AssertionError("name= was accepted")
except ValueError as error:
    assert "parse error" in str(error), error

p.remove(inserted)

try:
    p.insert_after("no-such-node", "identity")
    raise AssertionError("an unknown node was accepted")
except ValueError as error:
    assert "UnknownNode" in str(error), error

_, consumed, _ = p.wait()
assert consumed == FRAMES, consumed
"#,
                )
                .unwrap();
                let globals = PyDict::new(py);
                py.run(script.as_c_str(), Some(&globals), None)
                    .expect("mutation script runs");
            });
        }

        #[test]
        fn python_adds_chains_and_branches_to_a_running_pipeline() {
            ensure_module();
            Python::attach(|py| {
                let script = CString::new(
                    r#"
import g2g
FRAMES = 90
p = g2g.Pipeline(
    f"videotestsrc name=src num-buffers={FRAMES} ! clocksync ! tee name=t ! fakesink name=main"
)

chain = p.insert_chain_after("src", "identity ! valve drop=false")
assert len(chain) == 2, chain
assert p.get_property(chain[1], "drop") == "false"

branch = p.add_branch("t", "identity ! fakesink")
assert len(branch) == 2, branch
p.remove_branch(branch[-1])

try:
    p.remove_branch("src")
    raise AssertionError("a source was removed as a branch")
except ValueError as error:
    assert "NotMutable" in str(error), error

try:
    p.add_input("main", "videotestsrc")
    raise AssertionError("a sink took an input")
except ValueError as error:
    assert "NotMutable" in str(error), error

try:
    p.add_branch("t", "identity ! video/x-raw ! fakesink")
    raise AssertionError("a caps filter was accepted in a branch")
except ValueError as error:
    assert "parse error" in str(error), error

_, consumed, _ = p.wait()
assert consumed == FRAMES, consumed
"#,
                )
                .unwrap();
                let globals = PyDict::new(py);
                py.run(script.as_c_str(), Some(&globals), None)
                    .expect("branch script runs");
            });
        }

        #[test]
        fn an_insert_that_changes_the_format_gets_a_converter() {
            ensure_module();
            Python::attach(|py| {
                let script = CString::new(
                    r#"
import g2g
FRAMES = 90
p = g2g.Pipeline(
    f"videotestsrc name=src num-buffers={FRAMES} ! video/x-raw,format=RGBA ! clocksync ! fakesink"
)
p.insert_after("src", "videoconvert format=I420")
_, consumed, _ = p.wait()
assert consumed == FRAMES, consumed
"#,
                )
                .unwrap();
                let globals = PyDict::new(py);
                py.run(script.as_c_str(), Some(&globals), None)
                    .expect("converter script runs");
            });
        }

        #[test]
        fn an_edit_on_a_stalled_pipeline_times_out() {
            ensure_module();
            Python::attach(|py| {
                let script = CString::new(
                    r#"
import threading
import g2g
src = g2g.AppSrc("stalled-in")
sink = g2g.AppSink("stalled-out")
p = g2g.Pipeline(
    "appsrc name=src channel=stalled-in caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1"
    " ! appsink channel=stalled-out"
)
assert src.push(b"\x01" * 16, 0)
assert sink.pull(timeout_ms=5000) is not None

# longer than the handle's own timeout, so only a missing timeout reaches it
EDIT_DEADLINE_S = 10
outcome = []
def edit():
    try:
        p.insert_after("src", "identity")
        outcome.append("completed")
    except ValueError as error:
        outcome.append(str(error))
editor = threading.Thread(target=edit)
editor.start()
editor.join(timeout=EDIT_DEADLINE_S)
src.end_of_stream()
editor.join()
assert "TimedOut" in outcome[0], outcome
_, consumed, _ = p.wait()
assert consumed == 1, consumed
"#,
                )
                .unwrap();
                let globals = PyDict::new(py);
                py.run(script.as_c_str(), Some(&globals), None)
                    .expect("stall script runs");
            });
        }
    }
}
