//! Embedded-CPython host for [`PyTransform`] (M198, `python` feature).
//!
//! Bootstraps a single in-process CPython interpreter (pyo3 `auto-initialize`),
//! registers the native `g2g` module that a gst-python-ml `backend/g2g` package
//! imports, and drives a hosted element instance per frame.
//!
//! Contract with the Python side (the `backend/g2g` package the gst-python-ml
//! team writes against `GSTML_BACKEND=g2g`): importing an element module yields
//! a class whose instances expose
//!
//! ```text
//! g2g_process(buf, width: int, height: int, fmt: str) -> list[bytes]
//! ```
//!
//! where `buf` is a **writable buffer-protocol object** over the frame's own
//! System memory. Python wraps it (`memoryview(buf)`, or
//! `np.frombuffer(buf, np.uint8).reshape(h, w, c)`) and reads / overwrites
//! pixels in place, so neither direction copies. It returns a list of opaque
//! metadata blobs. This is the `backend/gst` `GstFrameIO` shape on a g2g
//! [`Frame`]. Step 3 routes the blob list into [`g2g_core::FrameMetaSet`].
//!
//! GIL / threading (step 2b): CPython is single-interpreter and GIL-serialized,
//! and g2g's runtime is a custom cooperative executor (`runtime::join`, not
//! tokio) that polls every node arm on one thread, so an inline `Python::attach`
//! would stall the whole graph for the duration of the Python work. Instead each
//! [`PyWorker`] owns a dedicated OS thread that holds the instance and does all
//! GIL work; [`PyWorker::run`] hands it the owned [`Frame`] over a std channel
//! and awaits the reply over g2g-core's Waker-based channel, so the executor
//! thread is free to poll other arms while Python runs. Multiple hosted elements
//! still serialize on the one GIL (expected); per-element sub-interpreters are a
//! later option (see `DESIGN_TODO.md`).

use std::os::raw::c_int;
use std::sync::mpsc;
use std::sync::Once;
use std::thread::{self, JoinHandle};

use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;

use g2g_core::runtime::{bounded, Receiver};
use g2g_core::{Caps, Dim, Frame, G2gError, HardwareError, MemoryDomain, RawVideoFormat};

use crate::format::format_to_py;

static INIT: Once = Once::new();

/// A frame plus its negotiated geometry, sent to the worker thread.
struct Job {
    frame: Frame,
    width: u32,
    height: u32,
    fmt: RawVideoFormat,
}

/// Worker -> element reply: the (possibly mutated) frame, or an error.
type Reply = Result<Frame, G2gError>;

/// Zero-copy writable view over a frame's System-memory bytes, handed to the
/// hosted element through the Python buffer protocol. Holds a raw pointer into
/// memory the worker thread owns (the `Job`'s frame) for the whole
/// `g2g_process` call; `unsendable` because the pointer is only touched on the
/// worker thread inside that call.
#[pyclass(unsendable)]
#[derive(Debug)]
struct FrameBuffer {
    ptr: *mut u8,
    len: usize,
}

#[pymethods]
impl FrameBuffer {
    unsafe fn __getbuffer__(
        slf: PyRefMut<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("null buffer view"));
        }
        // SAFETY: `view` is a valid out-pointer supplied by CPython.
        // `PyBuffer_FillInfo` increfs the exporter (`slf`) into `view->obj`, so
        // the `FrameBuffer` (and thus the validity guarantee on `ptr`) outlives
        // the view; CPython decrefs on release. `ptr`/`len` describe the worker
        // thread's owned frame slice, alive for the whole call, and the Python
        // contract forbids retaining the buffer past `g2g_process`'s return.
        let ret = unsafe {
            ffi::PyBuffer_FillInfo(
                view,
                slf.as_ptr(),
                slf.ptr as *mut core::ffi::c_void,
                slf.len as ffi::Py_ssize_t,
                0, // writable
                flags,
            )
        };
        if ret == -1 {
            Err(PyErr::take(slf.py()).unwrap_or_else(|| PyBufferError::new_err("fill failed")))
        } else {
            Ok(())
        }
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        // Nothing allocated in `__getbuffer__` beyond the exporter refcount,
        // which CPython manages; release is a no-op.
    }
}

/// Native `g2g` module visible to the embedded interpreter. The FrameIO /
/// AnalyticsBackend surface (M198 step 3) hangs off here; empty for now so the
/// `import g2g` in a `backend/g2g` package resolves.
#[pymodule]
fn g2g(_py: Python<'_>, _m: &Bound<'_, PyModule>) -> PyResult<()> {
    // TODO(M198 step 3): expose FrameIO (read_frame / write_frame /
    // append_blob) and AnalyticsBackend (add_object / add_classification /
    // ...) backed by the live Rust Frame and FrameMetaSet.
    Ok(())
}

/// Register the native `g2g` module and select the g2g backend, before the
/// interpreter initializes. Idempotent; safe to call from every worker spawn.
pub fn init_host() {
    INIT.call_once(|| {
        // Selected before the Python `backend` package is imported so its
        // GSTML_BACKEND branch binds to `backend/g2g`.
        std::env::set_var("GSTML_BACKEND", "g2g");
        // append_to_inittab! must run before the interpreter is initialized;
        // `auto-initialize` defers init to the first `attach`, and `Once`
        // guarantees this runs first.
        pyo3::append_to_inittab!(g2g);
    });
}

/// A hosted Python element running on its own GIL-owning OS thread. Frames are
/// handed over by [`run`](Self::run); the thread is joined on drop.
#[derive(Debug)]
pub(crate) struct PyWorker {
    /// `None` only after [`Drop`] takes it to signal the worker to exit.
    job_tx: Option<mpsc::Sender<Job>>,
    result_rx: Receiver<Reply>,
    handle: Option<JoinHandle<()>>,
}

impl PyWorker {
    /// Spawn the worker, import `module`, instantiate `class`, and block until
    /// it reports readiness (so a construction failure surfaces synchronously
    /// from `configure_pipeline`, not on the first frame).
    pub(crate) fn spawn(
        module: &str,
        class: &str,
        draw_label: bool,
    ) -> Result<Self, G2gError> {
        init_host();
        let (job_tx, jobs) = mpsc::channel::<Job>();
        let (results, result_rx) = bounded::<Reply>(1);
        let (ack_tx, ack_rx) = mpsc::channel::<Result<(), G2gError>>();
        let (m, c) = (module.to_owned(), class.to_owned());

        let handle = thread::Builder::new()
            .name("g2g-pyworker".into())
            .spawn(move || worker_main(m, c, draw_label, ack_tx, jobs, results))
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        match ack_rx.recv() {
            Ok(Ok(())) => {
                Ok(Self { job_tx: Some(job_tx), result_rx, handle: Some(handle) })
            }
            // Construction failed in Python: drain the now-finished thread.
            Ok(Err(e)) => {
                let _ = handle.join();
                Err(e)
            }
            // Thread died before acking (panic during init).
            Err(_) => {
                let _ = handle.join();
                Err(G2gError::Hardware(HardwareError::Other))
            }
        }
    }

    /// Hand one frame to the worker and await the (possibly mutated) frame. The
    /// `send` is non-blocking (unbounded job channel); the `await` parks until
    /// the worker's `try_send`, freeing the executor thread meanwhile.
    pub(crate) async fn run(&self, frame: Frame, caps: &Caps) -> Result<Frame, G2gError> {
        let (width, height, fmt) = raw_video_dims(caps)?;
        self.job_tx
            .as_ref()
            .ok_or(G2gError::Shutdown)?
            .send(Job { frame, width, height, fmt })
            .map_err(|_| G2gError::Shutdown)?;
        self.result_rx.recv().await.unwrap_or(Err(G2gError::Shutdown))
    }
}

impl Drop for PyWorker {
    fn drop(&mut self) {
        // Drop the sender so the worker's `recv` returns `Err` and it exits,
        // then join so the GIL-owning thread is gone before we return.
        drop(self.job_tx.take());
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The worker thread body: import + instantiate, ack, then service jobs until
/// the channel closes. All GIL work happens here, off the executor thread.
fn worker_main(
    module: String,
    class: String,
    draw_label: bool,
    ack: mpsc::Sender<Result<(), G2gError>>,
    jobs: mpsc::Receiver<Job>,
    results: g2g_core::runtime::Sender<Reply>,
) {
    let instance = match Python::attach(|py| instantiate(py, &module, &class, draw_label)) {
        Ok(obj) => {
            let _ = ack.send(Ok(()));
            obj
        }
        Err(e) => {
            let _ = ack.send(Err(e));
            return;
        }
    };

    while let Ok(job) = jobs.recv() {
        let reply = Python::attach(|py| process_frame(py, &instance, job));
        // Capacity-1, and the element awaits each reply before sending the next
        // job, so this never blocks; an error means the element (receiver) is
        // gone, so stop.
        if results.try_send(reply).is_err() {
            break;
        }
    }

    // Release the instance under the GIL before the thread ends.
    Python::attach(|_py| drop(instance));
}

/// Import `module`, instantiate `class`, and set the `draw_label` attribute.
fn instantiate(
    py: Python<'_>,
    module: &str,
    class: &str,
    draw_label: bool,
) -> Result<Py<PyAny>, G2gError> {
    (|| -> PyResult<Py<PyAny>> {
        let m = PyModule::import(py, module)?;
        let obj = m.getattr(class)?.call0()?;
        obj.setattr("draw_label", draw_label)?;
        Ok(obj.unbind())
    })()
    .map_err(|e| py_fail(py, e))
}

/// Run one frame through the hosted element. Python reads / overwrites the
/// frame's System memory in place via the buffer protocol; the same frame flows
/// back, timing and sequence preserved.
fn process_frame(py: Python<'_>, instance: &Py<PyAny>, mut job: Job) -> Reply {
    let (ptr, len) = {
        let MemoryDomain::System(slice) = &mut job.frame.domain else {
            // GPU / DMABUF domains need the download (step 4) before Python
            // sees them; the host requires System memory.
            return Err(G2gError::UnsupportedDomain);
        };
        let bytes = slice.as_mut_slice();
        (bytes.as_mut_ptr(), bytes.len())
    };

    let result = (|| -> PyResult<()> {
        let buffer = Py::new(py, FrameBuffer { ptr, len })?;
        let _blobs = instance.bind(py).call_method1(
            "g2g_process",
            (buffer, job.width, job.height, format_to_py(job.fmt)),
        )?;
        // TODO(M198 step 3): route `_blobs` (the metadata list) into
        // `job.frame.meta` (FrameMetaSet).
        Ok(())
    })();

    match result {
        Ok(()) => Ok(job.frame),
        Err(e) => Err(py_fail(py, e)),
    }
}

/// Pull the fixed `(width, height, format)` out of negotiated raw-video caps.
fn raw_video_dims(caps: &Caps) -> Result<(u32, u32, RawVideoFormat), G2gError> {
    match caps {
        Caps::RawVideo { format, width, height, .. } => {
            Ok((dim_fixed(width)?, dim_fixed(height)?, *format))
        }
        _ => Err(G2gError::CapsMismatch),
    }
}

fn dim_fixed(d: &Dim) -> Result<u32, G2gError> {
    match d {
        Dim::Fixed(v) => Ok(*v),
        _ => Err(G2gError::FixationFailed),
    }
}

/// Surface the Python traceback to stderr (the standard pyo3 path) before
/// collapsing to a structural error; `G2gError` carries no string payload, so a
/// richer error (carrying the traceback) waits on a core enum change.
fn py_fail(py: Python<'_>, e: PyErr) -> G2gError {
    e.print(py);
    G2gError::Hardware(HardwareError::Other)
}
