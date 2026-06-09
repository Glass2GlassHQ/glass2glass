//! Standard source / sink / transform elements for `glass2glass`.
//!
//! Per the spec (§2), this crate is `no_std + alloc` at baseline. Network
//! and OS-coupled elements (RTSP source via `retina`, V4L2, wgpu sinks)
//! live behind cargo features that imply `std`.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

extern crate alloc;

pub mod fakesink;
pub mod h264parse;
pub mod identity;
pub mod mux;
pub mod videotestsrc;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
pub mod clock;
#[cfg(feature = "std")]
pub mod syncsink;

#[cfg(feature = "rtsp")]
pub mod rtspsrc;
