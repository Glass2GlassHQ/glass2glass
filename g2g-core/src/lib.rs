//! Core types for the `glass2glass` multimedia framework.
//!
//! This crate is `no_std + alloc`. It defines the data carriers (`Frame`,
//! `PipelinePacket`), the memory domain model, capability negotiation types,
//! the `AsyncElement` execution trait, the pipeline clock, the link backpressure
//! policy, and the error enum. It contains no I/O and no executor.
//!
//! See `DESIGN.md` for the full specification.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

extern crate alloc;

pub mod caps;
pub mod clock;
pub mod element;
pub mod error;
pub mod frame;
pub mod link;
pub mod memory;
pub mod pool;

#[cfg(feature = "runtime")]
pub mod runtime;

#[cfg(feature = "dyn-slot")]
pub mod slot;

pub use caps::{AudioFormat, Caps, Dim, Rate, TensorDType, TensorLayout, TensorShape, VideoFormat};
pub use clock::{AsyncClock, PipelineClock};
pub use element::{
    AsyncElement, ConfigureOutcome, ElementBound, OutputSink, PushOutcome, Reconfigure,
};
pub use error::{G2gError, HardwareError};
pub use frame::{Frame, FrameTiming, PipelinePacket};
pub use link::LinkPolicy;
pub use memory::{MemoryDomain, OwnedDmaBuf, OwnedVulkanTexture, OwnedWebGPUBuffer, SystemSlice};

#[cfg(feature = "runtime")]
pub use pool::{BufferPool, PooledBuffer};

#[cfg(feature = "dyn-slot")]
pub use slot::ElementSlot;
