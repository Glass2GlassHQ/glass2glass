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
//! still serialize on the one GIL (expected) on a standard build. This
//! one-thread-per-element shape is deliberately the free-threaded (PEP 703,
//! `python3.x` `--disable-gil`) unit: on a free-threaded interpreter the workers
//! run truly in parallel with no code change (the `Python::attach` API is the
//! no-GIL model, not "acquire the GIL"). Per-interpreter-GIL sub-interpreters
//! were rejected: numpy / torch / cv2 are not reliably sub-interpreter-safe.

use std::os::raw::c_int;
use std::sync::mpsc;
use std::sync::Once;
use std::thread::{self, JoinHandle};

use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;

use g2g_core::runtime::{bounded, Receiver};
use g2g_core::{Caps, Dim, Frame, G2gError, HardwareError, MemoryDomain, PropValue, RawVideoFormat};

use crate::format::format_to_py;

static INIT: Once = Once::new();

/// A frame plus its negotiated geometry, sent to the worker thread.
/// Which Python entry point a job invokes.
enum JobKind {
    /// `g2g_process(buf, w, h, fmt, meta)` — one frame, mutated in place.
    Transform,
    /// `g2g_process_batch([buf, ...], w, h, fmt, meta)` — N frames.
    Batch,
    /// `g2g_produce(buf, w, h, fmt, meta) -> bool` — fill a blank frame; a
    /// `False` return signals end of stream.
    Produce,
}

struct Job {
    /// One frame for a transform / produce; one per contributing input for an
    /// aggregator batch. Frame 0 is the anchor that carries any metadata.
    frames: Vec<Frame>,
    width: u32,
    height: u32,
    fmt: RawVideoFormat,
    kind: JobKind,
}

/// Worker -> element reply: the (possibly mutated) frames, or an error. An empty
/// vec from a `Produce` job means the Python source signalled EOS.
type Reply = Result<Vec<Frame>, G2gError>;

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

/// One staged analytics result collected from the Python side during a frame.
/// Materialized into an [`g2g_core::AnalyticsMeta`] after the call (under the
/// `analytics` feature); the fields are read only there.
#[cfg_attr(not(feature = "analytics"), allow(dead_code))]
#[derive(Debug, Clone)]
enum Staged {
    Object { label: u32, x: f32, y: f32, w: f32, h: f32, score: f32 },
    Classification { label: u32, score: f32 },
    Blob { header: String, payload: Vec<u8> },
}

/// The analytics sink handed to `g2g_process` as `meta`: the `AnalyticsBackend`
/// mirror. The Python side (a `backend/g2g` element) calls `add_object` /
/// `add_classification`; the host drains the collected results into the frame's
/// metadata after the call. Labels are interned ids (`u32`), as g2g's
/// `ObjectDetection` stores; the Python side maps string classes to ids (the
/// `quark` step) before calling.
#[pyclass(unsendable)]
#[derive(Debug, Default)]
struct MetaSink {
    staged: core::cell::RefCell<Vec<Staged>>,
}

#[pymethods]
impl MetaSink {
    /// Add an object-detection box: class `label` id, pixel `(x, y, w, h)`,
    /// confidence `score` in `[0, 1]`.
    fn add_object(&self, label: u32, x: f32, y: f32, w: f32, h: f32, score: f32) {
        self.staged.borrow_mut().push(Staged::Object { label, x, y, w, h, score });
    }

    /// Add a whole-frame classification: class `label` id and `score`.
    fn add_classification(&self, label: u32, score: f32) {
        self.staged.borrow_mut().push(Staged::Classification { label, score });
    }

    /// Append an opaque tagged blob (the `FrameIO.append_blob` mirror): a
    /// `header` tag and serialized `payload` bytes, e.g. an embedding's f32
    /// bytes or a JSON record. Carried on the frame as a `BlobMeta`.
    fn add_blob(&self, header: String, payload: Vec<u8>) {
        self.staged.borrow_mut().push(Staged::Blob { header, payload });
    }
}

/// Native `g2g` module visible to the embedded interpreter, so the `import g2g`
/// in a `backend/g2g` package resolves. Exposes the analytics sink type; the
/// FrameIO read/write helpers (M198 step 4) hang off here later.
#[pymodule]
fn g2g(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MetaSink>()?;
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
        params: &[(String, PropValue)],
    ) -> Result<Self, G2gError> {
        init_host();
        let (job_tx, jobs) = mpsc::channel::<Job>();
        let (results, result_rx) = bounded::<Reply>(1);
        let (ack_tx, ack_rx) = mpsc::channel::<Result<(), G2gError>>();
        let (m, c) = (module.to_owned(), class.to_owned());
        let params = params.to_vec();

        let handle = thread::Builder::new()
            .name("g2g-pyworker".into())
            .spawn(move || worker_main(m, c, draw_label, params, ack_tx, jobs, results))
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
        let mut out = self
            .dispatch(Job { frames: vec![frame], width, height, fmt, kind: JobKind::Transform })
            .await?;
        out.pop().ok_or(G2gError::Shutdown)
    }

    /// Hand a batch (one frame per contributing input) to the worker and await
    /// the frames back. Frame 0 is the anchor; it carries any metadata the
    /// batch produced. Used by `PyAggregator`.
    pub(crate) async fn run_batch(
        &self,
        frames: Vec<Frame>,
        caps: &Caps,
    ) -> Result<Vec<Frame>, G2gError> {
        let (width, height, fmt) = raw_video_dims(caps)?;
        self.dispatch(Job { frames, width, height, fmt, kind: JobKind::Batch }).await
    }

    /// Hand a blank frame to the Python source to fill. Returns the produced
    /// frame, or `None` when the source signalled EOS. Used by `PySource`.
    pub(crate) async fn run_produce(
        &self,
        frame: Frame,
        caps: &Caps,
    ) -> Result<Option<Frame>, G2gError> {
        let (width, height, fmt) = raw_video_dims(caps)?;
        let mut out = self
            .dispatch(Job { frames: vec![frame], width, height, fmt, kind: JobKind::Produce })
            .await?;
        Ok(out.pop())
    }

    async fn dispatch(&self, job: Job) -> Result<Vec<Frame>, G2gError> {
        self.job_tx
            .as_ref()
            .ok_or(G2gError::Shutdown)?
            .send(job)
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
    params: Vec<(String, PropValue)>,
    ack: mpsc::Sender<Result<(), G2gError>>,
    jobs: mpsc::Receiver<Job>,
    results: g2g_core::runtime::Sender<Reply>,
) {
    let instance = match Python::attach(|py| instantiate(py, &module, &class, draw_label, &params)) {
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
        let reply = Python::attach(|py| process_job(py, &instance, job));
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

/// Import `module`, instantiate `class`, set `draw_label`, and forward the
/// element's properties onto the instance. Each property name is mapped from
/// gst style (`model-name`) to the Python attribute (`model_name`); the value
/// becomes the matching Python scalar. A property the class declares via the g2g
/// backend's `GObject` shim routes through its setter; one it does not declare is
/// set as a plain attribute (harmless if unused).
fn instantiate(
    py: Python<'_>,
    module: &str,
    class: &str,
    draw_label: bool,
    params: &[(String, PropValue)],
) -> Result<Py<PyAny>, G2gError> {
    (|| -> PyResult<Py<PyAny>> {
        let m = PyModule::import(py, module)?;
        let obj = m.getattr(class)?.call0()?;
        obj.setattr("draw_label", draw_label)?;
        for (name, value) in params {
            let attr = name.replace('-', "_");
            obj.setattr(attr.as_str(), propvalue_to_py(py, value)?)?;
        }
        Ok(obj.unbind())
    })()
    .map_err(|e| py_fail(py, e))
}

/// Convert a g2g [`PropValue`] to the Python scalar an element property expects.
fn propvalue_to_py(py: Python<'_>, value: &PropValue) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    match value {
        PropValue::Bool(b) => b.into_py_any(py),
        PropValue::Int(i) => i.into_py_any(py),
        PropValue::Uint(u) => u.into_py_any(py),
        PropValue::Double(d) => d.into_py_any(py),
        // A fraction arrives as a (num, den) tuple, matching gst's fraction props.
        PropValue::Fraction(n, d) => (*n, *d).into_py_any(py),
        PropValue::Str(s) => s.into_py_any(py),
    }
}

/// Run a job (one frame for a transform, a batch for an aggregator) through the
/// hosted element. Python reads / overwrites each frame's System memory in place
/// via the buffer protocol; the frames flow back, timing and sequence preserved.
fn process_job(py: Python<'_>, instance: &Py<PyAny>, mut job: Job) -> Reply {
    // Gather a raw pointer into every frame's System slice first, so the `&mut`
    // borrows end before the frames are moved into the reply. System memory
    // only; GPU / DMABUF domains need the download (step 4) before Python.
    let mut spans = Vec::with_capacity(job.frames.len());
    for frame in &mut job.frames {
        let MemoryDomain::System(slice) = &mut frame.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let bytes = slice.as_mut_slice();
        spans.push((bytes.as_mut_ptr(), bytes.len()));
    }

    let sink = match Py::new(py, MetaSink::default()) {
        Ok(s) => s,
        Err(e) => return Err(py_fail(py, e)),
    };

    // Returns whether a frame was produced: always true for transform / batch;
    // a `Produce` job returns the Python source's bool (false = EOS).
    let produced = (|| -> PyResult<bool> {
        let buffers: Vec<Py<FrameBuffer>> = spans
            .iter()
            .map(|&(ptr, len)| Py::new(py, FrameBuffer { ptr, len }))
            .collect::<PyResult<_>>()?;
        let bound = instance.bind(py);
        let (w, h, fmt) = (job.width, job.height, format_to_py(job.fmt));
        match job.kind {
            JobKind::Batch => {
                let list = pyo3::types::PyList::new(py, buffers.iter().map(|b| b.clone_ref(py)))?;
                bound.call_method1("g2g_process_batch", (list, w, h, fmt, sink.clone_ref(py)))?;
                Ok(true)
            }
            JobKind::Transform => {
                let buffer = buffers.into_iter().next().expect("single job has one frame");
                bound.call_method1("g2g_process", (buffer, w, h, fmt, sink.clone_ref(py)))?;
                Ok(true)
            }
            JobKind::Produce => {
                let buffer = buffers.into_iter().next().expect("produce job has one frame");
                let ret =
                    bound.call_method1("g2g_produce", (buffer, w, h, fmt, sink.clone_ref(py)))?;
                ret.extract::<bool>()
            }
        }
    })();

    // Drain the staged results regardless (so the field is always read);
    // materialize onto the anchor frame (frame 0) only under `analytics`.
    let staged = core::mem::take(&mut *sink.borrow(py).staged.borrow_mut());

    match produced {
        Ok(true) => {
            let (w, h) = (job.width, job.height);
            if let Some(anchor) = job.frames.first_mut() {
                attach_metadata(anchor, staged, w, h);
            }
            Ok(job.frames)
        }
        // Produce EOS: drop the blank frame, signal end with an empty reply.
        Ok(false) => Ok(Vec::new()),
        Err(e) => Err(py_fail(py, e)),
    }
}

/// Materialize staged results onto the frame: detections / classifications into
/// an [`g2g_core::AnalyticsMeta`], opaque blobs into a [`g2g_core::BlobMeta`].
#[cfg(feature = "analytics")]
fn attach_metadata(frame: &mut Frame, staged: Vec<Staged>, frame_w: u32, frame_h: u32) {
    use g2g_core::{AnalyticsMeta, AnalyticsNode, BBox, BlobMeta, Classification, ObjectDetection};

    if staged.is_empty() {
        return;
    }
    // The Python side reports detection boxes in pixels of the processed frame
    // (the gst-python-ml / GstAnalytics convention); g2g's `BBox` is normalized
    // to [0, 1] so it survives a downstream scale / crop. Divide by the frame
    // dims here (the one place that knows them), so an `analyticsoverlay`
    // denormalizes back to the right pixels. Guard against a zero dim.
    let sx = if frame_w > 0 { 1.0 / frame_w as f32 } else { 0.0 };
    let sy = if frame_h > 0 { 1.0 / frame_h as f32 } else { 0.0 };
    let mut analytics = AnalyticsMeta::new();
    let mut blobs = BlobMeta::new();
    for s in staged {
        match s {
            Staged::Object { label, x, y, w, h, score } => {
                analytics.add_detection(ObjectDetection {
                    bbox: BBox { x: x * sx, y: y * sy, w: w * sx, h: h * sy },
                    label,
                    confidence: score,
                });
            }
            Staged::Classification { label, score } => {
                analytics
                    .push(AnalyticsNode::Classification(Classification { label, confidence: score }));
            }
            Staged::Blob { header, payload } => blobs.push(header, payload),
        }
    }
    if !analytics.nodes.is_empty() {
        frame.meta.attach(analytics);
    }
    if !blobs.is_empty() {
        frame.meta.attach(blobs);
    }
}

/// Without the `analytics` feature `FrameMetaSet` is the ZST, so staged results
/// are dropped.
#[cfg(not(feature = "analytics"))]
fn attach_metadata(_frame: &mut Frame, _staged: Vec<Staged>, _frame_w: u32, _frame_h: u32) {}

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
