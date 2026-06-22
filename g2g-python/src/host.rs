//! Embedded-CPython host for [`PyTransform`] (M198, `python` feature).
//!
//! Bootstraps a single in-process CPython interpreter (pyo3 `auto-initialize`),
//! registers the native `g2g` module that a gst-python-ml `backend/g2g` package
//! imports, and drives a hosted element instance per frame under the GIL.
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
//! metadata blobs. This is the `backend/gst` `GstFrameIO` shape (map the
//! buffer, read a frame, write a frame in place, append a blob) on a g2g
//! [`Frame`]. Step 3 routes the blob list into [`g2g_core::FrameMetaSet`].
//!
//! GIL / threading: CPython is single-interpreter and GIL-serialized. This host
//! calls `g2g_process` **inline** under [`Python::attach`] (GIL acquire), which blocks the
//! runner arm for the duration of the Python work. That is correct but not
//! concurrent: g2g's runtime is a custom cooperative executor (`runtime::join`,
//! not tokio), so `tokio::spawn_blocking` is unavailable and a blocking hop must
//! go to a dedicated OS thread reached over a runtime-agnostic async channel
//! (the `MfDecode` single-thread-affinity model). That offload, and the
//! one-shared-interpreter vs per-element-sub-interpreter GIL-contention choice,
//! are step 2b (see `DESIGN_TODO.md`); doing them before the executor contract
//! is pinned risks the wrong abstraction.

use std::os::raw::c_int;
use std::sync::Once;

use pyo3::exceptions::PyBufferError;
use pyo3::ffi;
use pyo3::prelude::*;

use g2g_core::{Caps, Dim, Frame, G2gError, HardwareError, MemoryDomain, RawVideoFormat};

use crate::format::format_to_py;

static INIT: Once = Once::new();

/// Zero-copy writable view over a frame's System-memory bytes, handed to the
/// hosted element through the Python buffer protocol. Holds a raw pointer into
/// memory the host (`run_transform`) keeps alive and untouched for the whole
/// `g2g_process` call; `unsendable` because the pointer is only ever touched on
/// the GIL thread inside that call.
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
        // the view; CPython decrefs on release. `ptr`/`len` describe a slice
        // the host pins for the whole call, and the Python contract forbids
        // retaining the buffer past `g2g_process`'s return.
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
/// interpreter initializes. Idempotent; safe to call from every element's
/// `configure_pipeline`.
pub fn init_host() {
    INIT.call_once(|| {
        // Selected before the Python `backend` package is imported so its
        // GSTML_BACKEND branch binds to `backend/g2g`.
        std::env::set_var("GSTML_BACKEND", "g2g");
        // append_to_inittab! must run before the interpreter is initialized;
        // `auto-initialize` defers init to the first `with_gil`, and `Once`
        // guarantees this runs first.
        pyo3::append_to_inittab!(g2g);
    });
}

/// Import `module`, instantiate `class`, set the `draw_label` attribute, and
/// return the live instance (a GIL-independent `Py` handle).
pub(crate) fn instantiate(
    module: &str,
    class: &str,
    draw_label: bool,
) -> Result<Py<PyAny>, G2gError> {
    init_host();
    Python::attach(|py| -> Result<Py<PyAny>, G2gError> {
        let obj = (|| -> PyResult<Py<PyAny>> {
            let m = PyModule::import(py, module)?;
            let obj = m.getattr(class)?.call0()?;
            obj.setattr("draw_label", draw_label)?;
            Ok(obj.unbind())
        })();
        obj.map_err(|e| py_fail(py, e))
    })
}

/// Hand one frame to the hosted element. Python reads / overwrites the frame's
/// System memory in place through the buffer protocol; the same frame (now
/// possibly modified) flows on, timing and sequence preserved.
pub(crate) fn run_transform(
    instance: &Py<PyAny>,
    mut frame: Frame,
    caps: &Caps,
) -> Result<Frame, G2gError> {
    let (width, height, fmt) = raw_video_dims(caps)?;

    // Scope the mutable borrow so `frame` can be moved into the return after
    // the call. The raw pointer outlives the borrow; the heap buffer it points
    // at does not move when `frame` does (a `Box` move relocates only the
    // pointer, not the allocation), and Python is done with the buffer before
    // the return.
    let (ptr, len) = {
        let MemoryDomain::System(slice) = &mut frame.domain else {
            // GPU / DMABUF domains need the download (step 4) before Python
            // sees them; the skeleton requires System memory.
            return Err(G2gError::UnsupportedDomain);
        };
        let bytes = slice.as_mut_slice();
        (bytes.as_mut_ptr(), bytes.len())
    };

    Python::attach(|py| -> Result<(), G2gError> {
        (|| -> PyResult<()> {
            let buffer = Py::new(py, FrameBuffer { ptr, len })?;
            let _blobs = instance.bind(py).call_method1(
                "g2g_process",
                (buffer, width, height, format_to_py(fmt)),
            )?;
            // TODO(M198 step 3): route `_blobs` (the metadata list) into
            // `frame.meta` (FrameMetaSet).
            Ok(())
        })()
        .map_err(|e| py_fail(py, e))
    })?;

    Ok(frame)
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
