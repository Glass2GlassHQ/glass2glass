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
//! g2g_process(buf: bytes, width: int, height: int, fmt: str)
//!     -> tuple[bytes | None, list[bytes]]
//! ```
//!
//! returning the (optionally overwritten) frame bytes and a list of opaque
//! metadata blobs. This is the same shape `backend/gst` builds on a
//! `Gst.Buffer`: read a frame, run the task, write a frame and/or append a
//! blob. Step 2 replaces the `bytes` copy with a zero-copy numpy view over the
//! System slice via the buffer protocol; step 3 routes the blob list into
//! [`g2g_core::FrameMetaSet`] through the native `g2g` module.
//!
//! GIL note: CPython is single-interpreter and GIL-serialized; hosted elements
//! contend on one lock. Elements run on the executor's task threads, so a
//! production runner should hand `run_transform` to a Python-affine blocking
//! thread (`tokio::task::spawn_blocking` or a dedicated OS thread, like
//! `MfDecode`'s single-thread contract) rather than calling it inline as this
//! skeleton does. Document the chosen `link_capacity` accordingly: a blocking
//! hop widens the in-flight window.

use std::sync::Once;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

use g2g_core::memory::SystemSlice;
use g2g_core::{Caps, Dim, Frame, G2gError, HardwareError, MemoryDomain, RawVideoFormat};

use crate::format::format_to_py;

static INIT: Once = Once::new();

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
        // SAFETY contract of append_to_inittab!: called before the interpreter
        // is initialized. `auto-initialize` defers init to the first
        // `with_gil`, and `Once` guarantees this runs first.
        pyo3::append_to_inittab!(g2g);
    });
}

/// Import `module`, instantiate `class`, set the `draw-label` property, and
/// return the live instance (a GIL-independent `Py` handle).
pub(crate) fn instantiate(
    module: &str,
    class: &str,
    draw_label: bool,
) -> Result<Py<PyAny>, G2gError> {
    init_host();
    Python::with_gil(|py| -> PyResult<Py<PyAny>> {
        let m = PyModule::import(py, module)?;
        let obj = m.getattr(class)?.call0()?;
        obj.setattr("draw_label", draw_label)?;
        Ok(obj.unbind())
    })
    .map_err(py_err)
}

/// Hand one frame to the hosted element and rebuild the output frame.
///
/// Skeleton: copies the System slice into a `bytes`, calls `g2g_process`, and
/// reads back only the (optional) overwritten frame. The blob list and the
/// zero-copy buffer-protocol path are later steps (see module docs).
pub(crate) fn run_transform(
    instance: &Py<PyAny>,
    frame: Frame,
    caps: &Caps,
) -> Result<Frame, G2gError> {
    let MemoryDomain::System(slice) = &frame.domain else {
        // GPU / DMABUF domains need the download (step 4) before Python sees
        // them; the skeleton requires System memory.
        return Err(G2gError::UnsupportedDomain);
    };
    let (width, height, fmt) = raw_video_dims(caps)?;

    let out_bytes = Python::with_gil(|py| -> PyResult<Option<Vec<u8>>> {
        let buf = PyBytes::new(py, slice.as_slice());
        let ret = instance
            .bind(py)
            .call_method1("g2g_process", (buf, width, height, format_to_py(fmt)))?;
        // ret is (out_bytes | None, [blobs]); the skeleton reads only frame 0.
        // TODO(M198 step 3): route ret[1] (the blob list) into FrameMetaSet.
        let frame_obj = ret.get_item(0)?;
        if frame_obj.is_none() {
            Ok(None)
        } else {
            Ok(Some(frame_obj.extract::<Vec<u8>>()?))
        }
    })
    .map_err(py_err)?;

    Ok(match out_bytes {
        // Python overwrote the frame (e.g. drew the label overlay): rebuild on
        // a fresh System slice, preserving timing and sequence for latency
        // traceability.
        Some(bytes) => Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: frame.timing,
            sequence: frame.sequence,
            meta: Default::default(),
        },
        // No overlay drawn: forward the input frame untouched.
        None => frame,
    })
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

fn py_err(_e: PyErr) -> G2gError {
    // TODO(M198 step 2): carry the Python traceback (e.to_string under the GIL)
    // into a richer error rather than collapsing to Other.
    G2gError::Hardware(HardwareError::Other)
}
