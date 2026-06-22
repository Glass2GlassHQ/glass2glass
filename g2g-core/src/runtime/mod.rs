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

mod autoplug;
mod channel;
mod coordinator;
mod join;
mod progress;
mod runner;
mod seek;
pub mod solver;
mod state;

#[cfg(feature = "std")]
mod fanin;

#[cfg(feature = "std")]
mod graph_runner;

#[cfg(feature = "std")]
mod launch;

pub use channel::{
    bounded, link, LinkInterceptor, LinkReceiver, LinkSender, ProbeAction, ProbeSlot, QosSlot,
    Receiver, ReconfigureSlot, RecvFuture, SendError, SendFuture, Sender, SenderSink,
};
pub use coordinator::{coordinator, Coordinator, CoordinatorEvent, CoordinatorHandle};
pub use join::{join_all, select2, Either, Join2, JoinAll, Select2};
pub use runner::{
    run_simple_pipeline, run_simple_pipeline_stateful, run_simple_pipeline_with_bus,
    run_source_transform_sink, run_source_transform_sink_with_bus, LatencyProfile, LinkCapacity,
    RunStats, SourceLoop,
};
pub use autoplug::{find_chain, is_raw_audio, is_raw_video, ChainLink, ElementDesc};
pub use progress::PipelineProgress;
pub use seek::SeekController;
pub use state::{Flow, FlowGate, PrerollGate, StateController};
pub use solver::NegotiationFailure;

#[cfg(feature = "std")]
pub use runner::{
    run_linear_chain, run_linear_chain_with_bus, run_source_fanout, run_source_fanout_with_bus,
};

#[cfg(feature = "std")]
pub use fanin::{
    run_fanin_sink, run_muxer_sink, run_muxer_sink_with_bus, DynMultiInputElement, DynSourceLoop,
};

#[cfg(feature = "std")]
pub use graph_runner::{
    run_graph, run_graph_stateful, run_graph_with_bus, run_graph_with_progress, DynMultiOutputElement,
    GraphNode, GraphNodeRef,
};

#[cfg(feature = "std")]
pub use autoplug::{
    declared_source_caps, DecodebinError, DemuxFactory, ElementFactory, LaunchFactory,
    MuxerFactory, PlaybinError, Registry, SourceFactory, Uri, UriError, UriSourceFactory,
};

#[cfg(feature = "std")]
pub use launch::{parse_launch, ParseError};
