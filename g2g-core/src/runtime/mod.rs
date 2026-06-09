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

#[cfg(feature = "std")]
mod fanin;

pub use channel::{
    bounded, link, LinkReceiver, LinkSender, Receiver, ReconfigureSlot, RecvFuture, SendError,
    SendFuture, Sender, SenderSink,
};
pub use join::{join_all, Join2, JoinAll};
pub use runner::{run_simple_pipeline, run_source_transform_sink, RunStats, SourceLoop};

#[cfg(feature = "std")]
pub use runner::run_source_fanout;

#[cfg(feature = "std")]
pub use fanin::{run_fanin_sink, DynSourceLoop};
