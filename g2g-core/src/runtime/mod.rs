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
mod coordinator;
mod join;
mod runner;
pub mod solver;

#[cfg(feature = "std")]
mod fanin;

pub use channel::{
    bounded, link, LinkInterceptor, LinkReceiver, LinkSender, ProbeAction, ProbeSlot, Receiver,
    ReconfigureSlot, RecvFuture, SendError, SendFuture, Sender, SenderSink,
};
pub use coordinator::{coordinator, Coordinator, CoordinatorEvent, CoordinatorHandle};
pub use join::{join_all, select2, Either, Join2, JoinAll, Select2};
pub use runner::{
    run_simple_pipeline, run_source_transform_sink, run_source_transform_sink_with_bus,
    LatencyProfile, LinkCapacity, RunStats, SourceLoop,
};
pub use solver::NegotiationFailure;

#[cfg(feature = "std")]
pub use runner::{run_linear_chain, run_source_fanout};

#[cfg(feature = "std")]
pub use fanin::{run_fanin_sink, run_muxer_sink, DynSourceLoop};
