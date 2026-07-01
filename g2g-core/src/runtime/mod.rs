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
mod instrument;
mod join;
mod progress;
mod runner;
mod seek;
pub mod solver;
mod state;
mod stream_select;

#[cfg(feature = "std")]
mod blocking;

#[cfg(feature = "std")]
mod fanin;

#[cfg(feature = "std")]
mod gapless;

#[cfg(feature = "std")]
mod graph_runner;

#[cfg(feature = "std")]
mod launch;

pub use channel::{
    bounded, link, BitrateSlot, LinkInterceptor, LinkReceiver, LinkSender, ProbeAction, ProbeSlot,
    QosSlot, Receiver, ReconfigureSlot, RecvFuture, SendError, SendFuture, Sender, SenderSink,
};
pub use coordinator::{coordinator, Coordinator, CoordinatorEvent, CoordinatorHandle};
pub use instrument::{snapshot_all, ElementLatency, ElementProbe, Probe};
pub use join::{join_all, select2, Either, Join2, JoinAll, Select2};
pub use runner::{
    run_simple_pipeline, run_simple_pipeline_stateful, run_simple_pipeline_with_bus,
    run_source_transform_sink, run_source_transform_sink_with_bus, LatencyProfile, LinkCapacity,
    RunStats, SourceLoop,
};
pub use autoplug::{
    find_chain, find_chain_preferring, find_chain_with, is_raw_audio, is_raw_video, Acceleration,
    CapabilityDescriptor, ChainLink, ElementDesc, SelectionContext,
};
pub use progress::PipelineProgress;
pub use seek::{SeekController, WaitEvent};
pub use state::{Flow, FlowGate, PrerollGate, StateController};
pub use stream_select::StreamSelectController;
pub use solver::NegotiationFailure;

#[cfg(feature = "std")]
pub use blocking::block_on;

#[cfg(feature = "std")]
pub use runner::{
    run_fanout_session, run_linear_chain, run_linear_chain_with_bus, run_source_fanout,
    run_source_fanout_with_bus, run_source_router_dynamic, run_source_tee_dynamic,
    DynamicFanoutHandle,
};

#[cfg(feature = "std")]
pub use fanin::{
    run_aggregator_dynamic, run_duplex_session, run_fanin_session, run_fanin_sink,
    run_muxer_sink, run_muxer_sink_dynamic, run_muxer_sink_with_bus, DynMultiInputElement,
    DynamicFaninHandle, DynSourceLoop,
};

#[cfg(feature = "std")]
pub use gapless::{GaplessController, GaplessInstantWait, GaplessWait};

#[cfg(feature = "std")]
pub use graph_runner::{
    auto_plug_domain_converters, negotiate_graph, run_graph, run_graph_stateful,
    run_graph_with_bus, run_graph_with_progress, DynMultiOutputElement, GraphNode, GraphNodeRef,
    GraphTemplate,
};

// Thread-per-arm (opt-in multicore) runner. Needs `multi-thread` so the graph's
// elements + channels are `Send` to cross onto worker threads (the `!Send` arm
// future stays on its thread); `std` for the OS-thread spawner it drives.
#[cfg(all(feature = "std", feature = "multi-thread"))]
pub use graph_runner::{
    run_graph_threaded, run_graph_threaded_with_progress, GraphSpawner, LocalArmFuture,
    ThreadSpawner,
};

// `PadKind` / `PadRequest` are not std-gated: the `no_std` fan-in trait
// (`MultiInputElement::input_pad_index`) references them (M481).
pub use autoplug::{PadKind, PadRequest};

#[cfg(feature = "std")]
pub use autoplug::{
    declared_source_caps, DecodebinError, DecodebinSelectHook, DemuxFactory, DemuxSelectHook,
    ElementFactory,
    LaunchFactory, MuxerFactory, PlaybinGraphError, PlaybinHook, PlaybinPort,
    PlaybinError, Registry, SourceFactory, Uri, UriError, UriSourceFactory,
};

#[cfg(feature = "std")]
pub use launch::{parse_launch, ParseError};
