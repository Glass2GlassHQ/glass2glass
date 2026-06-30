//! GStreamer bridge for `glass2glass`.
//!
//! Embeds an isolated `g2g` processing sub-graph inside a legacy GStreamer
//! pipeline (DESIGN.md §7), so an existing GStreamer application can adopt one
//! g2g element (a wgpu filter, an ML inference stage) at a time instead of
//! rewriting the whole pipeline. This is the incremental-migration path for a
//! GStreamer *application* (the hardest port: dynamic pipelines built against
//! the `GstElement` C API, not a `gst-launch` string).
//!
//! # Layers
//!
//! 1. [`BridgeGraph`] (this module): the sync/async impedance matcher. It runs a
//!    user-supplied launch fragment as `appsrc ! <fragment> ! appsink` on a
//!    dedicated OS thread with its own current-thread runtime, and exposes a
//!    **synchronous** push/pull API. GStreamer's streaming thread calls
//!    [`BridgeGraph::push`] from its `chain` function and drains output with
//!    [`BridgeGraph::try_pull`], never blocking the async engine and never being
//!    blocked by it. This layer has no GStreamer dependency, so it is testable
//!    on any host.
//! 2. The GObject `GstBaseTransform` shell (`libgstglass2glass.so`, a follow-up):
//!    a thin C-FFI wrapper that registers `glass2glass` as a GStreamer element,
//!    maps `GstBuffer`/`GstCaps` to/from g2g [`Frame`]/`Caps`, and drives a
//!    `BridgeGraph`. It needs a live `gst-launch` to validate, so it is gated
//!    separately.
//!
//! The reuse of the already-tested [`appsrc`](g2g_plugins::appsrc) /
//! [`appsink`](g2g_plugins::appsink) elements is deliberate: they are exactly the
//! "synchronous external code feeds/drains a running async graph" boundary the
//! bridge needs, including the bounded-channel backpressure DESIGN.md §7 calls
//! for.

#![forbid(unsafe_op_in_unsafe_fn)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
mod bridge;

#[cfg(feature = "std")]
pub use bridge::{frame_bytes, BridgeError, BridgeGraph};

// Re-export the frame / pull types a caller works with, so an embedder depends
// on `g2g-bridge` alone.
#[cfg(feature = "std")]
pub use g2g_core::frame::Frame;
#[cfg(feature = "std")]
pub use g2g_plugins::appsink::Pull;
