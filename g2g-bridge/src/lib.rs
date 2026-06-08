//! GStreamer C-FFI bridge for `glass2glass`.
//!
//! Compiles as `libgstglass2glass.so`. A dedicated Tokio current-thread
//! runtime drives the embedded async sub-graph on its own OS thread,
//! communicating with the synchronous GStreamer `chain` function via a
//! bounded channel (see DESIGN.md §7).

#![forbid(unsafe_op_in_unsafe_fn)]
