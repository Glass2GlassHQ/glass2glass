//! Host gst-python-ml elements as first-class glass2glass elements (M198).
//!
//! gst-python-ml factored its ML logic away from GStreamer: the inference
//! `tasks/`, the engine `MLEngineMixin`, and the per-frame work all run with no
//! framework types, behind three seams selected by the `GSTML_BACKEND` env var:
//! `FrameIO` (read/write/append-blob a buffer), `AnalyticsBackend` (attach
//! detection metadata), and the element base classes. Today only a `gst`
//! backend exists. This crate is the g2g host those seams target: it embeds
//! CPython in the g2g process, exposes a native `g2g` module that backs
//! `FrameIO` / `AnalyticsBackend` against the live Rust [`Frame`], and wraps a
//! hosted element instance in a g2g [`AsyncElement`] so it negotiates caps and
//! flows frames like any other node.
//!
//! Build layers:
//! - **default (`std`)**: the [`PyTransform`] shell and the pixel-format
//!   mapping ([`format`]) compile. Caps negotiation works; the per-frame Python
//!   call returns [`G2gError::UnsupportedDomain`] because there is no
//!   interpreter. This keeps the crate in `cargo check --workspace` without
//!   libpython.
//! - **`python`**: pulls pyo3 + numpy, embeds CPython, and runs the hosted
//!   element for real. OS-coupled, off the no_std / RTOS baseline.
//!
//! Roadmap (M198 = the skeleton; later steps flesh it out):
//! 1. crate + feature + interpreter bootstrap  (this milestone)
//! 2. `PyTransform` + `g2g` module `FrameIO`, zero-copy over System memory
//! 3. `AnalyticsBackend` -> build out [`g2g_core::FrameMetaSet`] (the M88 defer)
//! 4. launch-registry factory; then aggregator / source variants; then the
//!    GPU zero-copy (DLPack / `__cuda_array_interface__`) path.
//!
//! This crate links `std` unconditionally (embedding CPython is the most
//! OS-coupled thing in the tree). The `std` feature only forwards to
//! `g2g-core/std`; it is not a no_std gate on this crate.

pub mod format;

mod element;
pub use element::PyTransform;

/// Register `pyelement` as a `gst-launch` / autoplug factory on `registry`, so
/// a hosted Python element is instantiable by name like any built-in:
/// `... ! pyelement module=action class=ActionTransform draw-label=true ! ...`.
/// The parser default-constructs it and applies `module=` / `class=` /
/// `draw-label=` via the property system, then negotiation + `configure_pipeline`
/// spawn the worker. Call after [`g2g_plugins::default_registry`] (or any
/// `Registry`). Building the per-frame path still needs the `python` feature.
#[cfg(feature = "std")]
pub fn register(registry: &mut g2g_core::runtime::Registry) {
    use g2g_core::runtime::LaunchFactory;
    registry.register_launch(LaunchFactory::of::<PyTransform>("pyelement", || {
        Box::new(PyTransform::new("", ""))
    }));
}

#[cfg(feature = "python")]
mod host;
#[cfg(feature = "python")]
pub use host::init_host;
