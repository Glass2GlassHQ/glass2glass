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
//! data, pts = sink.pull()          # None once the stream ends
//! p.wait()
//! ```

#![cfg_attr(not(feature = "python"), allow(unused_crate_dependencies))]

#[cfg(feature = "python")]
mod pymod {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use std::thread::JoinHandle;

    use g2g_core::runtime::{parse_launch, run_graph_with_bus, RunStats};
    use g2g_core::{Bus, BusMessage, G2gError, MemoryDomain};
    use g2g_plugins::appsink::{register_appsink_pull, AppSinkPull, Pull};
    use g2g_plugins::appsrc::{register_appsrc, AppSrcFeed};
    use g2g_plugins::clock::WallClock;
    use g2g_plugins::registry::default_registry;

    const LINK_CAPACITY: usize = 4;
    const BUS_CAPACITY: usize = 64;

    /// Drive a future to completion on the calling thread (the runtime channel's
    /// recv future is woken cross-thread by the run thread). The GIL is released
    /// around the call by the caller, so other Python threads keep running.
    fn block_on<F: core::future::Future>(fut: F) -> F::Output {
        use std::sync::Arc;
        use std::task::{Context, Poll, Wake, Waker};

        struct ThreadWaker(std::thread::Thread);
        impl Wake for ThreadWaker {
            fn wake(self: Arc<Self>) {
                self.0.unpark();
            }
            fn wake_by_ref(self: &Arc<Self>) {
                self.0.unpark();
            }
        }

        let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let mut fut = core::pin::pin!(fut);
        loop {
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    /// A running pipeline parsed from a `gst-launch`-style string. Runs on a
    /// background thread; poll the bus and `wait()` for the end of stream.
    #[pyclass]
    struct Pipeline {
        bus: Bus,
        join: Option<JoinHandle<Result<RunStats, G2gError>>>,
        result: Option<Result<RunStats, G2gError>>,
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
            Ok(Pipeline { bus, join: Some(join), result: None })
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
        /// Raises on a pipeline error. Releases the GIL while blocking.
        fn wait(&mut self, py: Python<'_>) -> PyResult<(u64, u64, u64)> {
            if let Some(j) = self.join.take() {
                self.result = Some(py.detach(|| j.join().unwrap_or(Err(G2gError::Shutdown))));
            }
            match &self.result {
                Some(Ok(s)) => Ok((s.frames_emitted, s.frames_consumed, s.frames_dropped)),
                _ => Err(PyValueError::new_err("pipeline errored")),
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
            AppSrc { feed: register_appsrc(channel) }
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
            AppSink { pull: register_appsink_pull(channel) }
        }

        /// Block for the next sample, returning `(bytes, pts_ns)` or `None` once
        /// the stream ends. Releases the GIL while blocking.
        fn pull<'py>(&self, py: Python<'py>) -> Option<(Bound<'py, PyBytes>, u64)> {
            let frame = py.detach(|| block_on(self.pull.pull()))?;
            Some((frame_bytes(py, &frame), frame.timing.pts_ns))
        }

        /// Non-blocking: `(bytes, pts_ns)` if a sample is ready, else `None`
        /// (whether the stream is merely idle or fully ended).
        fn try_pull<'py>(&self, py: Python<'py>) -> Option<(Bound<'py, PyBytes>, u64)> {
            match self.pull.try_pull() {
                Pull::Frame(frame) => Some((frame_bytes(py, &frame), frame.timing.pts_ns)),
                Pull::Empty | Pull::Ended => None,
            }
        }
    }

    /// Copy a frame's host-visible bytes into a Python `bytes` (the one copy at
    /// the language boundary; the pipeline stayed zero-copy up to here).
    fn frame_bytes<'py>(py: Python<'py>, frame: &g2g_core::frame::Frame) -> Bound<'py, PyBytes> {
        match &frame.domain {
            MemoryDomain::System(s) => PyBytes::new(py, s.as_slice()),
            MemoryDomain::SystemView(sv) => PyBytes::new(py, &sv.materialize()),
            _ => PyBytes::new(py, &[]),
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
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::g2g;
        use pyo3::append_to_inittab;
        use pyo3::prelude::*;
        use std::ffi::CString;

        /// Drives the whole binding from Python through an embedded interpreter:
        /// build a pipeline, push via AppSrc, pull via AppSink, wait for stats.
        #[test]
        fn python_drives_appsrc_through_appsink() {
            // Register the module before the interpreter initializes.
            append_to_inittab!(g2g);
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
assert a is not None and a[0] == b"\x01" * 16 and a[1] == 0, a
b = sink.pull()
assert b[0] == b"\x02" * 16 and b[1] == 1000, b
assert sink.pull() is None          # end of stream

emitted, consumed, dropped = p.wait()
assert consumed == 2, consumed
"#,
                )
                .unwrap();
                py.run(script.as_c_str(), None, None).expect("python script runs");
            });
        }
    }
}
