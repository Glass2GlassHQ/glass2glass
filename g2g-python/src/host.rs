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
//! GPU-resident frames take their own entry points (M984, M986). A
//! [`MemoryDomain::Cuda`] frame has no CPU bytes to wrap, so its two semi-planar
//! planes are handed over as [`CudaPlane`]s instead, one hook per shape:
//!
//! ```text
//! g2g_process_cuda(luma, chroma, width: int, height: int, meta)
//! g2g_process_cuda_batch([(luma, chroma), ...], width: int, height: int, meta)
//! g2g_produce_cuda(width: int, height: int, meta) -> (luma, chroma) | None
//! ```
//!
//! Each plane exposes `__cuda_array_interface__` (CAI v3) and `__dlpack__`, so
//! `cupy.asarray(luma)` / `torch.from_dlpack(luma)` alias the decoder's device
//! memory with no PCIe round-trip. See [`crate::cuda_plane`] for the layout and
//! the CUDA-context caveat. The produce hook runs the other way: this crate links
//! no CUDA and cannot allocate device memory, so the Python source allocates the
//! surface and hands back two CAI-exporting objects, which the frame holds as its
//! keep-alive.
//!
//! An element that does not define the hook for its shape cannot take such a
//! frame: the host fails it with [`G2gError::UnsupportedDomain`] (it links no
//! CUDA, so it cannot read the frame back itself), and the pipeline needs an
//! explicit `cudadownload` ahead of the element. A plane is valid only for the
//! duration of the call, enforced like the System path's buffer views.
//!
//! GIL / threading (step 2b): CPython is single-interpreter and GIL-serialized,
//! and g2g's runtime is a custom cooperative executor (`runtime::join`, not
//! tokio) that polls every node arm on one thread, so an inline `Python::attach`
//! would stall the whole graph for the duration of the Python work. Instead each
//! [`PyWorker`] owns a dedicated OS thread that holds the instance and does all
//! GIL work; [`PyWorker::run`] hands it the owned [`Frame`] over a std channel
//! and awaits the reply over g2g-core's Waker-based channel, so the executor
//! thread is free to poll other arms while Python runs. This
//! one-thread-per-element shape is deliberately the free-threaded (PEP 703,
//! `python3.14t`) unit: on a free-threaded interpreter the workers run truly in
//! parallel with no code change (the `Python::attach` API is the no-GIL model, not
//! "acquire the GIL"). Measured on both (M988, `tests/m988_gil_offload.rs`): four
//! hosted elements running one compute-bound Python callback each recover 3.6x of
//! the ideal 4x on free-threaded 3.14, and 0.9x on stock 3.14.
//! Per-interpreter-GIL sub-interpreters were rejected: numpy / torch / cv2 are not
//! reliably sub-interpreter-safe.
//!
//! Sizing consequence on a stock interpreter: N hosted elements do not overlap, so
//! the *graph's* Python cost is the sum of their per-frame times, not the slowest
//! one. A `link_capacity` chosen for a parallel chain therefore under-buffers a
//! chain with several hosted elements in it: each link's queue has to absorb the
//! wait while the other elements hold the GIL, so raise capacity on those links
//! (or accept the latency, which grows with the sum) rather than assuming the
//! elements pipeline. On a free-threaded interpreter the parallel assumption holds
//! and the usual live-latency sizing applies.

use std::os::raw::c_int;
use std::sync::mpsc;
use std::sync::{Mutex, Once};
use std::thread::{self, JoinHandle};

use pyo3::exceptions::{PyBufferError, PyRuntimeError};
use pyo3::ffi;
use pyo3::prelude::*;

use g2g_core::log::Target;
use g2g_core::runtime::{bounded, Receiver};
use g2g_core::{
    g2g_warn, Caps, Dim, Frame, G2gError, HardwareError, MemoryDomain, PropValue, RawVideoFormat,
};

use crate::cuda_plane::{nv12_planes, produced_cuda_buffer, CudaPlane};
use crate::format::format_to_py;

/// The Python entry points for GPU-resident frames (M984, M986). An element that
/// works on device memory defines the one matching its shape.
const CUDA_HOOK: &str = "g2g_process_cuda";
const CUDA_BATCH_HOOK: &str = "g2g_process_cuda_batch";
const CUDA_PRODUCE_HOOK: &str = "g2g_produce_cuda";
/// Optional attribute a hosted GPU source sets to report the `CUcontext` its
/// surfaces live in, for a downstream consumer that has to push it.
const CUDA_CONTEXT_ATTR: &str = "cuda_context";

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
    /// `g2g_produce_cuda(w, h, meta) -> (luma, chroma) | None` — the source
    /// allocates the device memory itself (g2g-python links no CUDA) and hands
    /// back the two planes as CAI-exporting objects; `None` signals end of
    /// stream. Carries no input frame.
    ProduceCuda,
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
    /// Outstanding buffer-protocol exports: `__getbuffer__` increments,
    /// `__releasebuffer__` decrements. The host checks this is back to 0 after
    /// each `g2g_process` call (see `process_job`): a nonzero count means the
    /// Python element retained a `memoryview` / numpy view past the call, whose
    /// raw pointer would dangle once the frame is freed downstream, so the host
    /// fails the frame loud instead of letting a later access become a
    /// use-after-free. Only touched on the worker thread, so a `Cell` suffices.
    exports: core::cell::Cell<isize>,
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
        // thread's owned frame slice, alive for the whole call; the export
        // counter below enforces that no view survives past the call.
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
            slf.exports.set(slf.exports.get() + 1);
            Ok(())
        }
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {
        // Balance the count `__getbuffer__` raised; nothing else was allocated
        // (the exporter refcount is CPython's to manage).
        self.exports.set(self.exports.get() - 1);
    }
}

/// One staged analytics result collected from the Python side during a frame.
/// Materialized into an [`g2g_core::AnalyticsMeta`] after the call (under the
/// `analytics` feature); the fields are read only there.
#[cfg_attr(not(feature = "analytics"), allow(dead_code))]
#[derive(Debug, Clone)]
enum Staged {
    Object {
        label: u32,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        score: f32,
    },
    Classification {
        label: u32,
        score: f32,
    },
    Blob {
        header: String,
        payload: Vec<u8>,
    },
}

/// The analytics sink handed to `g2g_process` as `meta`: the `AnalyticsBackend`
/// mirror. The Python side (a `backend/g2g` element) calls `add_object` /
/// `add_classification`; the host drains the collected results into the frame's
/// metadata after the call. Labels are interned ids (`u32`), as g2g's
/// `ObjectDetection` stores; the Python side maps string classes to ids (the
/// `quark` step) before calling.
///
/// `staged` is a `Mutex`, not a `RefCell`, so the pyclass is `Sync` and need not
/// be `unsendable`: a Python element that parallelizes post-processing (e.g. a
/// torch worker thread calling `add_object`) reaches the sink from a thread
/// other than the one that created it. On a GIL build an `unsendable` +
/// `RefCell` sink panics the moment any such thread touches it (the affinity
/// check fires before the GIL even matters); on a free-threaded (PEP 703) build
/// it would be an outright data race. The `Mutex` serializes the pushes on
/// either, so the "free-threaded with no code change" claim above actually holds
/// for the sink. Contention is nil in the common single-threaded-element case.
#[pyclass]
#[derive(Debug, Default)]
struct MetaSink {
    staged: Mutex<Vec<Staged>>,
}

impl MetaSink {
    fn stage(&self, item: Staged) {
        self.staged
            .lock()
            .expect("MetaSink staged lock poisoned")
            .push(item);
    }
}

#[pymethods]
impl MetaSink {
    /// Add an object-detection box: class `label` id, pixel `(x, y, w, h)`,
    /// confidence `score` in `[0, 1]`.
    fn add_object(&self, label: u32, x: f32, y: f32, w: f32, h: f32, score: f32) {
        self.stage(Staged::Object {
            label,
            x,
            y,
            w,
            h,
            score,
        });
    }

    /// Add a whole-frame classification: class `label` id and `score`.
    fn add_classification(&self, label: u32, score: f32) {
        self.stage(Staged::Classification { label, score });
    }

    /// Append an opaque tagged blob (the `FrameIO.append_blob` mirror): a
    /// `header` tag and serialized `payload` bytes, e.g. an embedding's f32
    /// bytes or a JSON record. Carried on the frame as a `BlobMeta`.
    fn add_blob(&self, header: String, payload: Vec<u8>) {
        self.stage(Staged::Blob { header, payload });
    }
}

/// Native `g2g` module visible to the embedded interpreter, so the `import g2g`
/// in a `backend/g2g` package resolves. Exposes the analytics sink type and the
/// GPU plane type.
///
/// `gil_used = false` declares the module safe to use without the GIL, which is
/// what keeps a free-threaded interpreter free-threaded: CPython re-enables the
/// GIL for the whole process when a module that does not declare this is imported,
/// and the `import g2g` in a hosted element would do exactly that. The types here
/// hold that claim up: `MetaSink` serializes its staging behind a `Mutex` and
/// `CudaPlane` is frozen and immutable.
#[pymodule(gil_used = false)]
fn g2g(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MetaSink>()?;
    m.add_class::<CudaPlane>()?;
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
            Ok(Ok(())) => Ok(Self {
                job_tx: Some(job_tx),
                result_rx,
                handle: Some(handle),
            }),
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
            .dispatch(Job {
                frames: vec![frame],
                width,
                height,
                fmt,
                kind: JobKind::Transform,
            })
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
        self.dispatch(Job {
            frames,
            width,
            height,
            fmt,
            kind: JobKind::Batch,
        })
        .await
    }

    /// Ask a GPU source for its next surface: it allocates the device memory and
    /// hands back the planes, so there is no blank frame to fill. `None` when the
    /// source signalled EOS. The returned frame carries no timing; the source
    /// stamps it. Used by `PySource` under `cuda-frames`.
    pub(crate) async fn run_produce_cuda(&self, caps: &Caps) -> Result<Option<Frame>, G2gError> {
        let (width, height, fmt) = raw_video_dims(caps)?;
        let mut out = self
            .dispatch(Job {
                frames: Vec::new(),
                width,
                height,
                fmt,
                kind: JobKind::ProduceCuda,
            })
            .await?;
        Ok(out.pop())
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
            .dispatch(Job {
                frames: vec![frame],
                width,
                height,
                fmt,
                kind: JobKind::Produce,
            })
            .await?;
        Ok(out.pop())
    }

    async fn dispatch(&self, job: Job) -> Result<Vec<Frame>, G2gError> {
        self.job_tx
            .as_ref()
            .ok_or(G2gError::Shutdown)?
            .send(job)
            .map_err(|_| G2gError::Shutdown)?;
        self.result_rx
            .recv()
            .await
            .unwrap_or(Err(G2gError::Shutdown))
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
    let instance = match Python::attach(|py| instantiate(py, &module, &class, draw_label, &params))
    {
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
        // `PropValue` is non_exhaustive: a kind added later must be mapped here
        // before a Python element can receive it.
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "unsupported property kind {:?}",
            other.kind()
        ))),
    }
}

/// How a job's frames reach Python: writable CPU bytes over the buffer protocol,
/// or GPU frames' device-pointer planes over CAI.
enum Handoff {
    /// One `(pointer, length)` per frame's System slice.
    System(Vec<(*mut u8, usize)>),
    /// Luma and interleaved chroma of a single Cuda-domain frame.
    Cuda(CudaPlane, CudaPlane),
    /// One plane pair per contributing input of a Cuda-domain batch.
    CudaBatch(Vec<(CudaPlane, CudaPlane)>),
    /// Nothing to hand over: a GPU source allocates its own surface and returns
    /// the planes.
    ProduceCuda,
}

/// Decide how this job's frames reach Python, and reject a frame the hosted
/// element cannot take. The System pointers are gathered here so the `&mut`
/// borrows end before the frames are moved into the reply.
fn handoff(py: Python<'_>, instance: &Py<PyAny>, job: &mut Job) -> Result<Handoff, G2gError> {
    if matches!(job.kind, JobKind::ProduceCuda) {
        require_hook(py, instance, CUDA_PRODUCE_HOOK)?;
        return Ok(Handoff::ProduceCuda);
    }

    if matches!(
        job.frames.first().map(|f| &f.domain),
        Some(MemoryDomain::Cuda(_))
    ) {
        let mut planes = Vec::with_capacity(job.frames.len());
        for frame in &job.frames {
            let MemoryDomain::Cuda(buf) = &frame.domain else {
                // A batch mixing GPU and CPU frames has no single contract.
                return Err(G2gError::UnsupportedDomain);
            };
            planes.push(nv12_planes(job.fmt, buf).ok_or(G2gError::UnsupportedDomain)?);
        }
        return match job.kind {
            JobKind::Transform => {
                require_hook(py, instance, CUDA_HOOK)?;
                let (luma, chroma) = planes.pop().ok_or(G2gError::UnsupportedDomain)?;
                Ok(Handoff::Cuda(luma, chroma))
            }
            JobKind::Batch => {
                require_hook(py, instance, CUDA_BATCH_HOOK)?;
                Ok(Handoff::CudaBatch(planes))
            }
            // A GPU frame handed to the System produce path (or a kind that
            // cannot arise) has nowhere to go.
            JobKind::Produce | JobKind::ProduceCuda => Err(G2gError::UnsupportedDomain),
        };
    }

    let mut spans = Vec::with_capacity(job.frames.len());
    for frame in &mut job.frames {
        let MemoryDomain::System(slice) = &mut frame.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let bytes = slice.as_mut_slice();
        spans.push((bytes.as_mut_ptr(), bytes.len()));
    }
    Ok(Handoff::System(spans))
}

/// Refuse the frame unless the hosted element defines `hook`. There is no
/// readback fallback: `g2g-python` links no CUDA, so a CPU-only element needs an
/// explicit `cudadownload` ahead of it.
fn require_hook(py: Python<'_>, instance: &Py<PyAny>, hook: &str) -> Result<(), G2gError> {
    let defined = instance
        .bind(py)
        .hasattr(hook)
        .map_err(|e| py_fail(py, e))?;
    if !defined {
        g2g_warn!(
            Target::category("pyelement"),
            "hosted element defines no {hook}, so a GPU-resident frame cannot reach it: insert cudadownload upstream"
        );
        return Err(G2gError::UnsupportedDomain);
    }
    Ok(())
}

/// Run a job (one frame for a transform, a batch for an aggregator) through the
/// hosted element. Python reads / overwrites each frame's System memory in place
/// via the buffer protocol, or reads a GPU frame's planes through CAI; the frames
/// flow back, timing and sequence preserved.
fn process_job(py: Python<'_>, instance: &Py<PyAny>, mut job: Job) -> Reply {
    let handoff = handoff(py, instance, &mut job)?;

    let sink = match Py::new(py, MetaSink::default()) {
        Ok(s) => s,
        Err(e) => return Err(py_fail(py, e)),
    };

    // Whether a frame was produced: always true for transform / batch; a
    // `Produce` job returns the Python source's bool (false = EOS).
    let produced = match handoff {
        Handoff::System(spans) => call_system(py, instance, &spans, &job, &sink),
        Handoff::Cuda(luma, chroma) => call_cuda(py, instance, luma, chroma, &job, &sink),
        Handoff::CudaBatch(planes) => call_cuda_batch(py, instance, planes, &job, &sink),
        Handoff::ProduceCuda => call_produce_cuda(py, instance, &mut job, &sink),
    };

    // Drain the staged results regardless (so the field is always read);
    // materialize onto the anchor frame (frame 0) only under `analytics`.
    let staged = core::mem::take(
        &mut *sink
            .borrow(py)
            .staged
            .lock()
            .expect("MetaSink staged lock poisoned"),
    );

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

/// Call the hosted element with writable views over the frames' System bytes.
fn call_system(
    py: Python<'_>,
    instance: &Py<PyAny>,
    spans: &[(*mut u8, usize)],
    job: &Job,
    sink: &Py<MetaSink>,
) -> PyResult<bool> {
    let buffers: Vec<Py<FrameBuffer>> = spans
        .iter()
        .map(|&(ptr, len)| {
            Py::new(
                py,
                FrameBuffer {
                    ptr,
                    len,
                    exports: core::cell::Cell::new(0),
                },
            )
        })
        .collect::<PyResult<_>>()?;
    let bound = instance.bind(py);
    let (w, h, fmt) = (job.width, job.height, format_to_py(job.fmt));
    // Pass cloned handles into the call and keep `buffers` so the export
    // counters can be inspected after it returns.
    let produced = match job.kind {
        JobKind::Batch => {
            let list = pyo3::types::PyList::new(py, buffers.iter().map(|b| b.clone_ref(py)))?;
            bound.call_method1("g2g_process_batch", (list, w, h, fmt, sink.clone_ref(py)))?;
            true
        }
        JobKind::Transform => {
            let buffer = buffers
                .first()
                .expect("single job has one frame")
                .clone_ref(py);
            bound.call_method1("g2g_process", (buffer, w, h, fmt, sink.clone_ref(py)))?;
            true
        }
        JobKind::Produce => {
            let buffer = buffers
                .first()
                .expect("produce job has one frame")
                .clone_ref(py);
            let ret = bound.call_method1("g2g_produce", (buffer, w, h, fmt, sink.clone_ref(py)))?;
            ret.extract::<bool>()?
        }
        // Routed to `call_produce_cuda`, which never builds System views.
        JobKind::ProduceCuda => unreachable!("a GPU produce job carries no System frame"),
    };
    // The zero-copy views must not outlive the call: a retained `memoryview` /
    // numpy view holds a pointer that dangles once the frame is freed
    // downstream. A nonzero export count means the element kept one, so fail
    // this frame loud rather than risk a use-after-free next frame.
    if buffers.iter().any(|b| b.borrow(py).exports.get() != 0) {
        return Err(PyBufferError::new_err(
            "g2g_process retained a frame buffer view past return (use-after-free risk)",
        ));
    }
    Ok(produced)
}

/// Call the hosted element with the GPU frame's two planes described by CAI, so
/// the Python side maps them into cupy / torch without a device->host copy.
fn call_cuda(
    py: Python<'_>,
    instance: &Py<PyAny>,
    luma: CudaPlane,
    chroma: CudaPlane,
    job: &Job,
    sink: &Py<MetaSink>,
) -> PyResult<bool> {
    let planes = [Py::new(py, luma)?, Py::new(py, chroma)?];
    let (w, h) = (job.width, job.height);
    instance.bind(py).call_method1(
        CUDA_HOOK,
        (
            planes[0].clone_ref(py),
            planes[1].clone_ref(py),
            w,
            h,
            sink.clone_ref(py),
        ),
    )?;
    planes_released(py, &planes)
}

/// Call the hosted aggregator with one plane pair per contributing input, as a
/// list of `(luma, chroma)` tuples: the GPU shape of `g2g_process_batch`.
fn call_cuda_batch(
    py: Python<'_>,
    instance: &Py<PyAny>,
    planes: Vec<(CudaPlane, CudaPlane)>,
    job: &Job,
    sink: &Py<MetaSink>,
) -> PyResult<bool> {
    let mut handles = Vec::with_capacity(planes.len() * 2);
    let mut pairs = Vec::with_capacity(planes.len());
    for (luma, chroma) in planes {
        let (luma, chroma) = (Py::new(py, luma)?, Py::new(py, chroma)?);
        pairs.push((luma.clone_ref(py), chroma.clone_ref(py)));
        handles.push(luma);
        handles.push(chroma);
    }
    let list = pyo3::types::PyList::new(py, pairs)?;
    instance.bind(py).call_method1(
        CUDA_BATCH_HOOK,
        (list.clone(), job.width, job.height, sink.clone_ref(py)),
    )?;
    // Release the list's own references before counting, so only what the element
    // kept is left.
    drop(list);
    planes_released(py, &handles)
}

/// Ask a hosted GPU source for its next surface. It allocates the device memory
/// itself (this crate links no CUDA) and returns the two planes as CAI-exporting
/// objects, which become the frame's CUDA buffer with those objects held as its
/// keep-alive. A falsy return is end of stream.
fn call_produce_cuda(
    py: Python<'_>,
    instance: &Py<PyAny>,
    job: &mut Job,
    sink: &Py<MetaSink>,
) -> PyResult<bool> {
    let bound = instance.bind(py);
    let returned = bound.call_method1(
        CUDA_PRODUCE_HOOK,
        (job.width, job.height, sink.clone_ref(py)),
    )?;
    if !returned.is_truthy()? {
        return Ok(false);
    }
    let (luma, chroma): (Bound<'_, PyAny>, Bound<'_, PyAny>) = returned.extract()?;
    let context = reported_cuda_context(bound)?;
    let buffer = produced_cuda_buffer(&luma, &chroma, job.fmt, job.width, job.height, context)?;
    job.frames.push(Frame {
        domain: MemoryDomain::Cuda(buffer),
        // The source stamps timing and sequence on the way out; it owns the clock.
        timing: g2g_core::FrameTiming::default(),
        sequence: 0,
        meta: Default::default(),
    });
    Ok(true)
}

/// The `CUcontext` a hosted GPU source reports through its optional
/// `cuda_context` attribute, or zero when it reports none.
fn reported_cuda_context(instance: &Bound<'_, PyAny>) -> PyResult<u64> {
    match instance.getattr(CUDA_CONTEXT_ATTR) {
        Ok(value) if !value.is_none() => value.extract(),
        _ => Ok(0),
    }
}

/// The device pointers belong to the producer and are freed once the frame is
/// released downstream, so nothing built over a plane may outlive the call: a
/// retained cupy array holds its plane as the array's base, and a consumed DLPack
/// capsule holds it through the tensor's manager context. Our own handle is the
/// only reference left if the element let go, so a higher count means it kept one:
/// fail the frame loud rather than let a later kernel read freed device memory.
fn planes_released(py: Python<'_>, planes: &[Py<CudaPlane>]) -> PyResult<bool> {
    if planes.iter().any(|plane| plane.get_refcnt(py) > 1) {
        return Err(PyRuntimeError::new_err(
            "a hosted element retained a CudaPlane past return (use-after-free risk)",
        ));
    }
    Ok(true)
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
    let sx = if frame_w > 0 {
        1.0 / frame_w as f32
    } else {
        0.0
    };
    let sy = if frame_h > 0 {
        1.0 / frame_h as f32
    } else {
        0.0
    };
    let mut analytics = AnalyticsMeta::new();
    let mut blobs = BlobMeta::new();
    for s in staged {
        match s {
            Staged::Object {
                label,
                x,
                y,
                w,
                h,
                score,
            } => {
                analytics.add_detection(ObjectDetection {
                    bbox: BBox {
                        x: x * sx,
                        y: y * sy,
                        w: w * sx,
                        h: h * sy,
                    },
                    label,
                    confidence: score,
                });
            }
            Staged::Classification { label, score } => {
                analytics.push(AnalyticsNode::Classification(Classification {
                    label,
                    confidence: score,
                }));
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
        Caps::RawVideo {
            format,
            width,
            height,
            ..
        } => Ok((dim_fixed(width)?, dim_fixed(height)?, *format)),
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
