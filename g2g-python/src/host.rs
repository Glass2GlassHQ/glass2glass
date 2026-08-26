//! Embedded-CPython host for [`PyTransform`] (M198, `python` feature).
//!
//! Bootstraps a single in-process CPython interpreter (pyo3 `auto-initialize`),
//! registers the native `g2g` module that a gst-python-ml `backend/g2g` package
//! imports, and drives a hosted element instance per frame.
//!
//! Contract with the Python side (the `backend/g2g` package the gst-python-ml
//! team writes against `PYML_BACKEND=g2g`): importing an element module yields
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
//! A stream that is not raw video has no picture shape, so it reaches Python
//! through the payload hook instead:
//!
//! ```text
//! g2g_process_payload(buffers, caps: str, meta)
//! ```
//!
//! `buffers` are the same writable views and `caps` is the negotiated caps as a
//! `gst-launch` string. Those elements (transcription reading audio and writing
//! text, speech synthesis the other way) rarely produce output the size of their
//! input, so they return bytes through
//! `meta.emit(payload, duration_ns=None, pts_ns=None)`: the host wraps them in a
//! new frame that replaces the one they were given, keeping its timing but for
//! the duration and presentation time the element states. A streaming element
//! (speech synthesized a chunk at a time) passes each chunk's own `pts_ns`,
//! usually the previous chunk's pts plus its duration, so the chunks play one
//! after another; an element whose outputs run in parallel (the separation
//! family's stems) leaves it unset so they all share the anchor's.
//! `g2g.PTS_NONE` means the buffer has no presentation time and a sink presents
//! it on arrival.
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

use pyo3::exceptions::{PyBufferError, PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;

use g2g_core::log::Target;
use g2g_core::runtime::{bounded, Receiver};
use g2g_core::{
    g2g_warn, Caps, Dim, Frame, FrameTiming, G2gError, HardwareError, MemoryDomain, PropValue,
    RawVideoFormat, SystemSlice,
};

use crate::cuda_plane::{nv12_planes, produced_cuda_buffer, CudaPlane};
use crate::format::format_to_py;

/// The Python entry points for GPU-resident frames (M984, M986). An element that
/// works on device memory defines the one matching its shape.
const CUDA_HOOK: &str = "g2g_process_cuda";
const CUDA_BATCH_HOOK: &str = "g2g_process_cuda_batch";
const CUDA_PRODUCE_HOOK: &str = "g2g_produce_cuda";
/// The Python entry point for a stream that is not raw video.
const PAYLOAD_HOOK: &str = "g2g_process_payload";
/// Optional method a hosted class defines to list the properties it declares, so
/// a pipeline naming one it has not is refused instead of silently setting an
/// attribute nothing reads.
const DECLARED_PROPERTIES_HOOK: &str = "g2g_properties";
/// Optional attribute a hosted GPU source sets to report the `CUcontext` its
/// surfaces live in, for a downstream consumer that has to push it.
const CUDA_CONTEXT_ATTR: &str = "cuda_context";
/// Optional attribute a hosted GPU source sets to report which CUDA device it
/// allocated on. Absent means device 0.
const CUDA_DEVICE_ATTR: &str = "cuda_device";

static INIT: Once = Once::new();

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

/// What the job's frames hold: a picture of a known geometry, or a payload of
/// some other media type, which has no geometry to describe.
enum JobCaps {
    RawVideo {
        width: u32,
        height: u32,
        fmt: RawVideoFormat,
    },
    /// The negotiated caps, handed to Python as its `gst-launch` string.
    Payload(Caps),
}

impl JobCaps {
    fn video(&self) -> Option<(u32, u32, RawVideoFormat)> {
        match self {
            JobCaps::RawVideo { width, height, fmt } => Some((*width, *height, *fmt)),
            JobCaps::Payload(_) => None,
        }
    }
}

/// Frames plus their negotiated caps, sent to the worker thread.
struct Job {
    /// One frame for a transform / produce; one per contributing input for an
    /// aggregator batch. Frame 0 is the anchor that carries any metadata.
    frames: Vec<Frame>,
    caps: JobCaps,
    kind: JobKind,
}

/// Worker -> element reply: the buffers to send downstream, or an error. Either
/// the one frame Python was handed, mutated in place, or the ones it emitted in
/// its stead. An empty vec from a `Produce` job means the Python source
/// signalled EOS.
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
    ClassNames {
        names: Vec<String>,
    },
    Blob {
        header: String,
        payload: Vec<u8>,
    },
    Tracking {
        object_id: u64,
    },
    /// A directed edge between two staged records, by their staging index.
    Relation {
        from: usize,
        to: usize,
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
    /// The buffers the element produced, when it emits its own rather than
    /// overwriting the frame it was handed. Locked for the same reason `staged`
    /// is.
    emitted: Mutex<Vec<Emitted>>,
}

/// A buffer a hosted element produced in place of the one it was handed.
#[derive(Debug)]
struct Emitted {
    payload: Vec<u8>,
    /// How long the produced buffer runs, when that is not the anchor's
    /// duration: synthesized speech lasts as long as its samples, not as long as
    /// the text it was generated from. `None` inherits the anchor's.
    duration_ns: Option<u64>,
    /// When the produced buffer is presented, when that is not the anchor's
    /// time: each chunk of streaming speech starts where the previous one ended,
    /// rather than all of them stacking at the instant the text arrived. `None`
    /// inherits the anchor's.
    pts_ns: Option<u64>,
}

impl MetaSink {
    /// Stage one record, returning its staging index (the handle Python relates
    /// records by).
    fn stage(&self, item: Staged) -> usize {
        let mut staged = self.staged.lock().expect("MetaSink staged lock poisoned");
        staged.push(item);
        staged.len() - 1
    }
}

#[pymethods]
impl MetaSink {
    /// Add an object-detection box: class `label` id, pixel `(x, y, w, h)`,
    /// confidence `score` in `[0, 1]`. Returns the record's handle, for
    /// [`relate`](Self::relate).
    fn add_object(&self, label: u32, x: f32, y: f32, w: f32, h: f32, score: f32) -> usize {
        self.stage(Staged::Object {
            label,
            x,
            y,
            w,
            h,
            score,
        })
    }

    /// Add a whole-frame classification: class `label` id and `score`. Returns
    /// the record's handle.
    fn add_classification(&self, label: u32, score: f32) -> usize {
        self.stage(Staged::Classification { label, score })
    }

    /// Publish the class names the staged label ids index into, so a consumer
    /// can show "person" rather than `0`. Names the element already holds, sent
    /// once per frame rather than per detection.
    fn set_class_names(&self, names: Vec<String>) {
        self.stage(Staged::ClassNames { names });
    }

    /// Add a tracking identity that persists across frames. Returns the record's
    /// handle; pair it with the detection it belongs to via
    /// [`relate`](Self::relate).
    fn add_tracking(&self, object_id: u64) -> usize {
        self.stage(Staged::Tracking { object_id })
    }

    /// Relate two staged records by their handles (detection -> tracking), the
    /// `GstAnalytics` `set_relation` mirror. Out-of-range handles are dropped
    /// when the frame's metadata is materialized.
    fn relate(&self, from: usize, to: usize) {
        self.stage(Staged::Relation { from, to });
    }

    /// Append an opaque tagged blob (the `FrameIO.append_blob` mirror): a
    /// `header` tag and serialized `payload` bytes, e.g. an embedding's f32
    /// bytes or a JSON record. Carried on the frame as a `BlobMeta`.
    fn add_blob(&self, header: String, payload: Vec<u8>) {
        self.stage(Staged::Blob { header, payload });
    }

    /// Emit `payload` as a buffer this element produces: it replaces the frame
    /// the element was handed, so the output need not be the size of the input
    /// (audio in, a short transcript out). Where `add_blob` travels alongside the
    /// buffer, this *is* the buffer. Call it more than once to send several
    /// buffers from one input, as chunked speech and source separation do; they
    /// go downstream in call order.
    ///
    /// Every emitted frame inherits the frame's timing, which is right whenever
    /// the output covers the same stretch of the stream as the input. An element
    /// whose output runs for its own length (speech synthesized from a text
    /// buffer) passes `duration_ns` to say how long, keeping the presentation
    /// time it was generated at. An element whose buffers play one after another
    /// (streaming speech, a chunk at a time) passes each one's `pts_ns` too,
    /// usually the previous chunk's pts plus its duration; one whose buffers run
    /// in parallel (the separation family's stems) leaves it unset so they share
    /// the anchor's. `g2g.PTS_NONE` says the buffer has no presentation time, and
    /// a sink presents it on arrival. The frame number is the host's, counted
    /// over what this element emitted, not the number of the buffer it read.
    #[pyo3(signature = (payload, duration_ns = None, pts_ns = None))]
    fn emit(&self, payload: Vec<u8>, duration_ns: Option<u64>, pts_ns: Option<u64>) {
        self.emitted
            .lock()
            .expect("MetaSink emitted lock poisoned")
            .push(Emitted {
                payload,
                duration_ns,
                pts_ns,
            });
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
    // The `pts_ns=` an element passes to emit a buffer with no presentation time.
    m.add("PTS_NONE", FrameTiming::PTS_NONE)?;
    Ok(())
}

/// Register the native `g2g` module and select the g2g backend, before the
/// interpreter initializes. Idempotent; safe to call from every worker spawn.
pub fn init_host() {
    INIT.call_once(|| {
        // Selected before the Python `backend` package is imported so its
        // PYML_BACKEND branch binds to `backend/g2g`.
        std::env::set_var("PYML_BACKEND", "g2g");
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

    /// Hand one frame to the worker and await the buffers back: the same frame
    /// mutated in place, or the ones the element emitted in its stead, which an
    /// element that chunks its output states more than one of. A frame of any
    /// media type but raw video reaches the element through the payload hook.
    /// The `send` is non-blocking (unbounded job channel); the `await` parks until
    /// the worker's `try_send`, freeing the executor thread meanwhile.
    pub(crate) async fn run(&self, frame: Frame, caps: &Caps) -> Result<Vec<Frame>, G2gError> {
        self.dispatch(Job {
            frames: vec![frame],
            caps: job_caps(caps)?,
            kind: JobKind::Transform,
        })
        .await
    }

    /// Hand a batch (one frame per contributing input) to the worker and await
    /// the buffers back: the anchor (frame 0) mutated in place, carrying any
    /// metadata the batch produced, or the ones the element emitted in its
    /// stead. A batch of any media type but raw video reaches the element
    /// through the payload hook instead. Used by `PyAggregator`.
    pub(crate) async fn run_batch(
        &self,
        frames: Vec<Frame>,
        caps: &Caps,
    ) -> Result<Vec<Frame>, G2gError> {
        self.dispatch(Job {
            frames,
            caps: job_caps(caps)?,
            kind: JobKind::Batch,
        })
        .await
    }

    /// Ask a GPU source for its next surface: it allocates the device memory and
    /// hands back the planes, so there is no blank frame to fill. `None` when the
    /// source signalled EOS. The returned frame carries no timing; the source
    /// stamps it. Used by `PySource` under `cuda-frames`.
    pub(crate) async fn run_produce_cuda(&self, caps: &Caps) -> Result<Option<Frame>, G2gError> {
        let mut out = self
            .dispatch(Job {
                frames: Vec::new(),
                caps: raw_video_caps(caps)?,
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
        let mut out = self
            .dispatch(Job {
                frames: vec![frame],
                caps: raw_video_caps(caps)?,
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

    // Numbers the buffers the element emits: one input can produce several, and
    // a sink takes a repeated sequence as a stream fault.
    let mut emitted_sequence = 0u64;

    while let Ok(job) = jobs.recv() {
        let reply = Python::attach(|py| process_job(py, &instance, job, &mut emitted_sequence));
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
        let declared = declared_properties(&obj)?;
        for (name, value) in params {
            let attr = name.replace('-', "_");
            if let Some(declared) = &declared {
                if !declared.contains(&attr) {
                    return Err(PyValueError::new_err(format!(
                        "{class} has no property {name}; it declares {}",
                        declared.join(", ").replace('_', "-")
                    )));
                }
            }
            obj.setattr(attr.as_str(), propvalue_to_py(py, value)?)?;
        }
        Ok(obj.unbind())
    })()
    .map_err(|e| py_fail(py, e))
}

/// The property names the hosted class says it has, so a pipeline naming one it
/// does not is refused here rather than quietly setting an attribute nothing
/// reads. `None` from a class that does not answer, which is every hosted class
/// outside gst-python-ml: nothing to check against, so everything is forwarded.
fn declared_properties(obj: &Bound<'_, PyAny>) -> PyResult<Option<Vec<String>>> {
    if !obj.hasattr(DECLARED_PROPERTIES_HOOK)? {
        return Ok(None);
    }
    Ok(Some(obj.call_method0(DECLARED_PROPERTIES_HOOK)?.extract()?))
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
        require_hook(py, instance, CUDA_PRODUCE_HOOK, CUDA_REMEDY)?;
        return Ok(Handoff::ProduceCuda);
    }

    if matches!(
        job.frames.first().map(|f| &f.domain),
        Some(MemoryDomain::Cuda(_))
    ) {
        // A GPU frame is a surface with a plane layout, which only raw-video
        // caps describes.
        let (_, _, fmt) = job.caps.video().ok_or(G2gError::UnsupportedDomain)?;
        let mut planes = Vec::with_capacity(job.frames.len());
        for frame in &job.frames {
            let MemoryDomain::Cuda(buf) = &frame.domain else {
                // A batch mixing GPU and CPU frames has no single contract.
                return Err(G2gError::UnsupportedDomain);
            };
            planes.push(nv12_planes(fmt, buf).ok_or(G2gError::UnsupportedDomain)?);
        }
        return match job.kind {
            JobKind::Transform => {
                require_hook(py, instance, CUDA_HOOK, CUDA_REMEDY)?;
                let (luma, chroma) = planes.pop().ok_or(G2gError::UnsupportedDomain)?;
                Ok(Handoff::Cuda(luma, chroma))
            }
            JobKind::Batch => {
                require_hook(py, instance, CUDA_BATCH_HOOK, CUDA_REMEDY)?;
                Ok(Handoff::CudaBatch(planes))
            }
            // A GPU frame handed to the System produce path (or a kind that
            // cannot arise) has nowhere to go.
            JobKind::Produce | JobKind::ProduceCuda => Err(G2gError::UnsupportedDomain),
        };
    }

    if matches!(job.caps, JobCaps::Payload(_)) {
        require_hook(py, instance, PAYLOAD_HOOK, PAYLOAD_REMEDY)?;
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

/// There is no readback fallback for a GPU frame: `g2g-python` links no CUDA, so
/// a CPU-only element needs an explicit `cudadownload` ahead of it.
const CUDA_REMEDY: &str = "a GPU-resident frame cannot reach it: insert cudadownload upstream";
const PAYLOAD_REMEDY: &str =
    "a stream that is not raw video cannot reach it: it defines only the picture hooks";

/// Refuse the frame unless the hosted element defines `hook`, logging `remedy`
/// as the way out.
fn require_hook(
    py: Python<'_>,
    instance: &Py<PyAny>,
    hook: &str,
    remedy: &str,
) -> Result<(), G2gError> {
    let defined = instance
        .bind(py)
        .hasattr(hook)
        .map_err(|e| py_fail(py, e))?;
    if !defined {
        g2g_warn!(
            Target::category("pyelement"),
            "hosted element defines no {hook}, so {remedy}"
        );
        return Err(G2gError::UnsupportedDomain);
    }
    Ok(())
}

/// Run a job (one frame for a transform, a batch for an aggregator) through the
/// hosted element. Python reads / overwrites each frame's System memory in place
/// via the buffer protocol, or reads a GPU frame's planes through CAI; the frames
/// flow back, timing and sequence preserved.
fn process_job(
    py: Python<'_>,
    instance: &Py<PyAny>,
    mut job: Job,
    emitted_sequence: &mut u64,
) -> Reply {
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
    let emitted = core::mem::take(
        &mut *sink
            .borrow(py)
            .emitted
            .lock()
            .expect("MetaSink emitted lock poisoned"),
    );

    match produced {
        Ok(true) => {
            // `None` for a payload job: it has no pixels for a detection box to
            // be normalized against.
            let frame_dims = job
                .caps
                .video()
                .map(|(w, h, _)| (w, h))
                .filter(|(w, h)| *w > 0 && *h > 0);
            let Some(anchor) = job.frames.first() else {
                return Ok(Vec::new());
            };
            let anchor_timing = anchor.timing;
            let anchor_sequence = anchor.sequence;
            let anchor_meta = anchor.meta.clone();
            let mut out = if emitted.is_empty() {
                // Nothing emitted: the frame Python was handed, mutated in
                // place. A batch's other inputs have done their work in the
                // call, so only the anchor travels on.
                job.frames.truncate(1);
                job.frames
            } else {
                // Emitted frames number on past the input's rather than from
                // zero: a sink reads a repeated sequence as a stream fault.
                *emitted_sequence = (*emitted_sequence).max(anchor_sequence);
                emitted
                    .into_iter()
                    .map(|emitted| {
                        let mut timing = anchor_timing;
                        if let Some(duration_ns) = emitted.duration_ns {
                            timing.duration_ns = duration_ns;
                        }
                        // An emitted buffer is never reordered, so its decode
                        // time is its presentation time.
                        if let Some(pts_ns) = emitted.pts_ns {
                            timing.pts_ns = pts_ns;
                            timing.dts_ns = pts_ns;
                        }
                        let sequence = *emitted_sequence;
                        *emitted_sequence += 1;
                        let mut frame = Frame::new(
                            MemoryDomain::System(SystemSlice::from_boxed(
                                emitted.payload.into_boxed_slice(),
                            )),
                            timing,
                            sequence,
                        );
                        // What upstream attached describes the stream, not the
                        // one buffer replaced, so it travels with every one sent.
                        frame.meta = anchor_meta.clone();
                        frame
                    })
                    .collect()
            };
            // The staged records describe the one call, so they go on one frame:
            // repeating them would count each detection several times downstream.
            if let Some(first) = out.first_mut() {
                attach_metadata(first, staged, frame_dims);
            }
            Ok(out)
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
    // Pass cloned handles into the call and keep `buffers` so the export
    // counters can be inspected after it returns.
    let produced = match &job.caps {
        // A payload has no geometry to pass, so the element gets the caps
        // description instead and reads the buffers' own lengths.
        JobCaps::Payload(caps) => {
            let list = pyo3::types::PyList::new(py, buffers.iter().map(|b| b.clone_ref(py)))?;
            bound.call_method1(
                PAYLOAD_HOOK,
                (list, caps.to_gst_string(), sink.clone_ref(py)),
            )?;
            true
        }
        JobCaps::RawVideo { width, height, fmt } => {
            let (w, h, fmt) = (*width, *height, format_to_py(*fmt));
            match job.kind {
                JobKind::Batch => {
                    let list =
                        pyo3::types::PyList::new(py, buffers.iter().map(|b| b.clone_ref(py)))?;
                    bound
                        .call_method1("g2g_process_batch", (list, w, h, fmt, sink.clone_ref(py)))?;
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
                    let ret = bound
                        .call_method1("g2g_produce", (buffer, w, h, fmt, sink.clone_ref(py)))?;
                    ret.extract::<bool>()?
                }
                // Routed to `call_produce_cuda`, which never builds System views.
                JobKind::ProduceCuda => unreachable!("a GPU produce job carries no System frame"),
            }
        }
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

/// The geometry the GPU hooks pass to Python. `handoff` refuses a GPU job whose
/// caps is not raw video, so the error never reaches an element.
fn cuda_geometry(job: &Job) -> PyResult<(u32, u32, RawVideoFormat)> {
    job.caps
        .video()
        .ok_or_else(|| PyRuntimeError::new_err("a GPU job needs raw-video caps"))
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
    let (w, h, _) = cuda_geometry(job)?;
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
    let (width, height, _) = cuda_geometry(job)?;
    let list = pyo3::types::PyList::new(py, pairs)?;
    instance.bind(py).call_method1(
        CUDA_BATCH_HOOK,
        (list.clone(), width, height, sink.clone_ref(py)),
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
    let (width, height, fmt) = cuda_geometry(job)?;
    let bound = instance.bind(py);
    let returned = bound.call_method1(CUDA_PRODUCE_HOOK, (width, height, sink.clone_ref(py)))?;
    if !returned.is_truthy()? {
        return Ok(false);
    }
    let (luma, chroma): (Bound<'_, PyAny>, Bound<'_, PyAny>) = returned.extract()?;
    let context = reported_cuda_context(bound)?;
    let device_ordinal = reported_cuda_device(bound)?;
    let buffer = produced_cuda_buffer(&luma, &chroma, fmt, width, height, context, device_ordinal)?;
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

/// The CUDA device ordinal a hosted GPU source reports through its optional
/// `cuda_device` attribute, or zero when it reports none.
fn reported_cuda_device(instance: &Bound<'_, PyAny>) -> PyResult<i32> {
    match instance.getattr(CUDA_DEVICE_ATTR) {
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
fn attach_metadata(frame: &mut Frame, staged: Vec<Staged>, frame_dims: Option<(u32, u32)>) {
    use g2g_core::{
        AnalyticsMeta, AnalyticsNode, BBox, BlobMeta, Classification, ObjectDetection,
        RelationKind, Tracking,
    };

    if staged.is_empty() {
        return;
    }
    // The Python side reports detection boxes in pixels of the processed frame
    // (the gst-python-ml / GstAnalytics convention); g2g's `BBox` is normalized
    // to [0, 1] so it survives a downstream scale / crop. Divide by the frame
    // dims here (the one place that knows them), so an `analyticsoverlay`
    // denormalizes back to the right pixels.
    let (sx, sy) = match frame_dims {
        Some((w, h)) => (1.0 / w as f32, 1.0 / h as f32),
        None => (0.0, 0.0),
    };
    let mut analytics = AnalyticsMeta::new();
    let mut blobs = BlobMeta::new();
    // Python relates records by staging index, but a blob stages a record and
    // adds no analytics node, so the two index spaces drift apart. Keep the map
    // and resolve relations in a second pass, since a relation may also be
    // staged before the record it names.
    let mut node_of_staged: Vec<Option<usize>> = vec![None; staged.len()];
    let mut relations: Vec<(usize, usize)> = Vec::new();
    for (index, s) in staged.into_iter().enumerate() {
        match s {
            Staged::Object {
                label,
                x,
                y,
                w,
                h,
                score,
            } => {
                // A box is in pixels of a frame, and a text or audio buffer has
                // none to divide by. Dropping it says so; keeping it would put a
                // box of zeros on the buffer and look like a detection at the
                // origin.
                if frame_dims.is_none() {
                    g2g_warn!(
                        Target::category("pyelement"),
                        "hosted element staged a detection on a stream with no \
                         pixels, dropping it"
                    );
                    continue;
                }
                node_of_staged[index] = Some(analytics.add_detection(ObjectDetection {
                    bbox: BBox {
                        x: x * sx,
                        y: y * sy,
                        w: w * sx,
                        h: h * sy,
                    },
                    label,
                    confidence: score,
                }));
            }
            Staged::Classification { label, score } => {
                node_of_staged[index] = Some(analytics.push(AnalyticsNode::Classification(
                    Classification {
                        label,
                        confidence: score,
                    },
                )));
            }
            Staged::Tracking { object_id } => {
                node_of_staged[index] =
                    Some(analytics.push(AnalyticsNode::Tracking(Tracking { object_id })));
            }
            Staged::Relation { from, to } => relations.push((from, to)),
            Staged::ClassNames { names } => analytics.set_class_names(names),
            Staged::Blob { header, payload } => blobs.push(header, payload),
        }
    }
    for (from, to) in relations {
        let (Some(Some(from)), Some(Some(to))) = (
            node_of_staged.get(from).copied(),
            node_of_staged.get(to).copied(),
        ) else {
            continue;
        };
        analytics.relate(from, to, RelationKind::Tracks);
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
fn attach_metadata(_frame: &mut Frame, _staged: Vec<Staged>, _frame_dims: Option<(u32, u32)>) {}

/// Describe the negotiated caps for a job: raw video keeps its fixed geometry,
/// anything else travels as an opaque payload.
fn job_caps(caps: &Caps) -> Result<JobCaps, G2gError> {
    match caps {
        Caps::RawVideo { .. } => raw_video_caps(caps),
        other => Ok(JobCaps::Payload(other.clone())),
    }
}

/// Pull the fixed geometry out of negotiated raw-video caps, for the paths that
/// take a picture and nothing else (transform, produce, the GPU hooks).
fn raw_video_caps(caps: &Caps) -> Result<JobCaps, G2gError> {
    match caps {
        Caps::RawVideo {
            format,
            width,
            height,
            ..
        } => Ok(JobCaps::RawVideo {
            width: dim_fixed(width)?,
            height: dim_fixed(height)?,
            fmt: *format,
        }),
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
