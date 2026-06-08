//! Minimal async pipeline runtime (M1).
//!
//! Provides a bounded SPSC channel, a hand-rolled `Join2` combinator (no
//! external executor dependency), and a `run_simple_pipeline` function that
//! drives a single source → sink topology. Source elements implement
//! [`SourceLoop`] rather than [`crate::AsyncElement`] because they have no
//! input pad and cannot be modeled as packet-in / packet-out.
//!
//! M2 will replace this with a graph builder and full caps negotiation.
//! M4 will replace the `spin::Mutex`-backed channel with a lock-free MPMC
//! implementation.

mod channel;
mod join;
mod runner;

pub use channel::{
    bounded, Receiver, RecvFuture, SendError, SendFuture, Sender, SenderSink,
};
pub use join::Join2;
pub use runner::{run_simple_pipeline, RunStats, SourceLoop};
