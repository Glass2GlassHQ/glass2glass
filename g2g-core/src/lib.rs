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
pub mod format_element;
pub mod clock;
pub mod element;
pub mod error;
pub mod frame;
pub mod link;
pub mod memory;
pub mod metrics;
pub mod pool;
pub mod query;

#[cfg(feature = "runtime")]
pub mod bus;

#[cfg(feature = "runtime")]
pub mod fanout;

#[cfg(feature = "runtime")]
pub mod runtime;

#[cfg(feature = "runtime")]
pub mod pad_template;

#[cfg(feature = "dyn-slot")]
pub mod slot;

pub use caps::{
    AudioFormat, Caps, CapsSet, Dim, Rate, TensorDType, TensorLayout, TensorShape, VideoCodec, RawVideoFormat,
};
pub use clock::{elect_clock, AsyncClock, ClockCandidate, ClockPriority, PipelineClock};
pub use element::{
    AsyncElement, ConfigureOutcome, ElementBound, OutputSink, PushOutcome, Reconfigure,
};
pub use error::{G2gError, HardwareError};
pub use format_element::{
    legacy_sink_constraint, legacy_transform_constraint, CapsConstraint, CapsPreferences,
    FormatElement,
};
pub use frame::{Frame, FrameTiming, PipelinePacket};
pub use link::LinkPolicy;
pub use memory::{
    CudaKeepAlive, D3D11KeepAlive, MemoryDomain, MemoryDomainKind, OwnedCudaBuffer,
    OwnedD3D11Texture, OwnedDmaBuf, OwnedVulkanTexture, OwnedWebGPUBuffer, SystemSlice,
};
pub use metrics::{LatencyHistogram, LatencySnapshot};
pub use query::{AllocationParams, LatencyReport};

#[cfg(feature = "runtime")]
pub use pool::{BufferPool, PooledBuffer};

#[cfg(feature = "runtime")]
pub use bus::{Bus, BusHandle, BusMessage};

#[cfg(feature = "runtime")]
pub use runtime::{LinkInterceptor, NegotiationFailure, ProbeAction, ProbeSlot};

#[cfg(feature = "runtime")]
pub use pad_template::{
    pad_link, types_can_link, PadCaps, PadDirection, PadTemplate, PadTemplates,
};

#[cfg(feature = "runtime")]
pub use fanout::{
    Gate, GateHandle, Merger, MergerHandle, MultiInputElement, MultiOutputElement, MultiOutputSink,
    MultiSenderSink, Router, RouterHandle,
};

#[cfg(feature = "dyn-slot")]
pub use slot::{ElementSlot, SwapHandle};
