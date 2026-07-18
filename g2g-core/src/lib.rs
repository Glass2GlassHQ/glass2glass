//! Core types for the `glass2glass` multimedia framework.
//!
//! This crate is `no_std`. It defines the data carriers (`Frame`,
//! `PipelinePacket`), the memory domain model, capability negotiation types,
//! the `AsyncElement` execution trait, the pipeline clock, the link backpressure
//! policy, and the error enum. It contains no I/O and no executor.
//!
//! The default build enables `alloc`. Building `--no-default-features` yields the
//! heap-free MCU / safety subset: it links no allocator and carries only the
//! data-plane types (`Frame`, `Caps`, `System`/`Foreign` memory, the const-generic
//! `StaticLendRing`, the clock / time newtypes). The dynamic graph, caps solver,
//! `parse_launch`, the `dyn` element traits, and the tooling (conformance, dot,
//! copy plan, wire codec) all live behind the `alloc` feature.
//!
//! See `DESIGN.md` for the full specification.

#![no_std]
#![forbid(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "alloc")]
extern crate alloc;

/// The ABI compatibility tag for dynamically loaded (`dlopen`ed) plugins.
///
/// Rust has no stable ABI, so a third-party `.so` built against this crate and
/// the host that loads it must share the same `g2g-core` version, the same
/// `rustc`, and the same layout-affecting features (`metadata` resizes
/// [`Frame`], `multi-thread` changes the `Send` bound on the element trait
/// objects). This string folds all three together; the plugin loader compares
/// the plugin's embedded copy against the host's and refuses a mismatch rather
/// than risk undefined behavior. Computed by `build.rs`. See `g2g-plugin`
/// (`declare_plugin!`) and `g2g_plugins::plugin_loader`.
pub const ABI_VERSION: &str = env!("G2G_ABI_VERSION");

// ---- heap-free data-plane subset (compiles with `--no-default-features`) ----
pub mod caps;
pub mod error;
pub mod frame;
pub mod link;
// ST 2110-10 media clock (M595): PTP/TAI <-> RTP-timestamp mapping. Pure no_std
// arithmetic tying RTP media transport to the pipeline's PTP clock.
pub mod mediaclock;
pub mod memory;
pub mod meta;
pub mod metrics;
pub mod query;
// RFC 3550 RTP fixed header (M643): the one shared header builder for every
// RTP packetizer (MCU packet sink + the std packetizers in g2g-plugins).
pub mod rtp;
pub mod segment;
pub mod state;
// Static (heap-free) element model (M624, Phase 2 of the alloc-optional core):
// generic `async fn`-in-trait source/transform/sink + const-arity runners that
// monomorphize to unboxed futures, so an MCU pipeline needs no `dyn` and no heap.
pub mod spsc;
pub mod staticelem;
pub mod staticpool;
// Concurrency-primitive compat layer so the SpscFrameRing can be model-checked
// under loom (`--cfg loom`); the core primitives in every normal build.
mod sync;
// Runtime fault recovery (M652): a supervisor that turns a returned fault into a
// bounded retry / degrade / reset / escalate action + a watchdog seam, for the
// safety / cert MCU market. In the no-alloc subset.
pub mod supervise;
pub mod tensor;
// Boundary-scoped time newtypes (M618): TaiNs / RtpTs at the clock/PTP/RTP seam.
pub mod time;

// ---- dynamic / build-time / tooling layer (needs the heap) ----
#[cfg(feature = "alloc")]
pub mod aggregator;
// Conformance vocabulary + derived maturity (M614): a maturity level computed from
// evidence produced by passing conformance cases, never hand-authored. Pure.
#[cfg(feature = "alloc")]
pub mod clock;
#[cfg(feature = "alloc")]
pub mod conformance;
#[cfg(feature = "alloc")]
pub mod format_element;
// Copy / allocation plan (M613): static memory-domain path analysis over a
// negotiated graph. Pure (like `dot`); the runner extracts its flat inputs.
#[cfg(feature = "alloc")]
pub mod copyplan;
#[cfg(feature = "alloc")]
pub mod dot;
#[cfg(feature = "alloc")]
pub mod element;
#[cfg(feature = "alloc")]
pub mod graph;
#[cfg(feature = "alloc")]
pub mod log;
#[cfg(feature = "alloc")]
pub mod pool;
#[cfg(feature = "alloc")]
pub mod property;
#[cfg(feature = "alloc")]
pub mod stream;
#[cfg(feature = "alloc")]
pub mod tag;
#[cfg(feature = "alloc")]
pub mod wire;
// PTP clock servo (M593 phase A): disciplines a monotonic reference to a
// grandmaster. Needs the `DriftClock` servo core, hence the `runtime` gate.
#[cfg(feature = "runtime")]
pub mod ptp;

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

#[cfg(feature = "alloc")]
pub use aggregator::InputAggregator;
pub use caps::{
    AudioFormat, ByteStreamEncoding, Caps, Dim, PassthroughFields, Rate, RawVideoFormat,
    TensorDType, TensorLayout, TensorShape, TextFormat, VideoCodec, ANY_CHANNELS, ANY_SAMPLE_RATE,
};
// `CapsSet` (negotiation-time alternatives) needs alloc; `TensorShape` is
// fixed-rank inline (M636) and part of the no-alloc subset above.
#[cfg(feature = "alloc")]
pub use caps::CapsSet;
#[cfg(feature = "runtime")]
pub use clock::DriftClock;
#[cfg(feature = "std")]
pub use clock::MonotonicClock;
#[cfg(feature = "alloc")]
pub use clock::{elect_clock, AsyncClock, ClockCandidate, ClockPriority, ClockSync, PipelineClock};
#[cfg(feature = "alloc")]
pub use conformance::{
    ConformanceDimension, ConformanceReport, Evidence, MaturityLevel, MaturityRecord,
};
#[cfg(feature = "alloc")]
pub use copyplan::{
    classify as classify_transfer, CopyBudgetError, CopyPlan, CopyPolicy, EdgeProfile, Hop,
    NodeProfile, Transfer, TransferKind,
};
#[cfg(feature = "alloc")]
pub use dot::DotAnnotations;
#[cfg(feature = "alloc")]
pub use element::{
    AsyncElement, ConfigureOutcome, ElementBound, OutputSink, PushOutcome, QosMessage, Reconfigure,
};
pub use error::{G2gError, HardwareError};
#[cfg(feature = "alloc")]
pub use format_element::{
    legacy_sink_constraint, legacy_transform_constraint, CapsConstraint, CapsPreferences,
    FormatElement,
};
pub use frame::{Frame, FrameTiming, PipelinePacket};
#[cfg(feature = "alloc")]
pub use graph::{
    Bin, BinInstance, Demux, Edge, Graph, GraphError, Muxer, NodeId, NodeIdOffset, NodeKind,
    PadDir, PadId, Tee, ValidatedGraph,
};
pub use link::LinkPolicy;
pub use mediaclock::MediaClock;
pub use meta::FrameMetaSet;
#[cfg(feature = "metadata")]
pub use meta::{
    AnalyticsMeta, AnalyticsNode, BBox, Blob, BlobMeta, Classification, FrameMeta, ObjectDetection,
    Propagation, Relation, RelationKind, Tracking, Transform,
};
#[cfg(feature = "alloc")]
pub use property::{ElementMetadata, PropError, PropFlags, PropKind, PropValue, PropertySpec};
#[cfg(feature = "runtime")]
pub use ptp::{
    ExchangeResult, PtpClock, PtpHeader, PtpMessageType, PtpServo, PtpSlave, PtpState, SlaveAction,
};
// The heap-free memory subset: the domain enum + its discriminant / set, and the
// `System` slice (whose `Foreign` variant the StaticLendRing lends zero-copy).
pub use memory::{DomainSet, MemoryDomain, MemoryDomainKind, SystemSlice};
// The GPU / shared-CPU domains are heap-backed (Arc/Box keep-alives).
#[cfg(feature = "alloc")]
pub use memory::{
    CudaKeepAlive, D3D11KeepAlive, OwnedCudaBuffer, OwnedD3D11Texture, OwnedDmaBuf,
    OwnedVulkanTexture, OwnedWebGPUBuffer, OwnedWebGPUExternalTexture, OwnedWgpuBuffer,
    OwnedWgpuTexture, SyncFd, SystemView, WebGPUKeepAlive, WgpuBufferKeepAlive, WgpuKeepAlive,
};
pub use metrics::{LatencyHistogram, LatencySnapshot};
pub use query::{AllocationParams, LatencyReport};
pub use rtp::{RtpHeader, RtpParsed, RTP_HEADER_LEN};
pub use segment::{Seek, SeekFlags, SeekType, Segment};
pub use spsc::{Overrun, SpscFrameRing};
// SpscCaptureSrc uses the zero-copy lend, which is not built under loom.
#[cfg(not(loom))]
pub use spsc::SpscCaptureSrc;
pub use state::{PipelineState, StateChangeReturn};
pub use staticelem::{
    drive_ready, run_source_sink, run_source_transform_sink, run_sources_fanin_sink,
    step_source_sink, Chain, SinkChain, SourceChain, StaticFanIn2, StaticSink, StaticSource,
    StaticTransform, Step,
};
pub use staticpool::{RingSlot, StaticAcquire, StaticBufferPool, StaticLendRing, StaticPooled};
#[cfg(feature = "alloc")]
pub use stream::{Stream, StreamCollection, StreamType};
pub use supervise::{
    run_supervised, step_supervised, FaultPolicy, NoWatchdog, Recover, Recovery, RetryThenReset,
    RunOutcome, SkipBounded, Supervised, SupervisorReport, Watchdog, MAX_ATTEMPTS,
};
#[cfg(feature = "alloc")]
pub use tag::{Tag, TagList};
pub use tensor::{TensorView, MAX_TENSOR_RANK};
pub use time::{RefNs, RtpTs, TaiNs};
#[cfg(feature = "alloc")]
pub use wire::{
    decode_packet, encode_packet, raw_format_from_u8, raw_format_to_u8, WireError, WIRE_VERSION,
};

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
    DuplexInbound, Gate, GateHandle, Merger, MergerHandle, MultiDuplexSession, MultiInputElement,
    MultiOutputElement, MultiOutputSink, MultiOutputSource, MultiSenderSink, ReverseChannel,
    Router, RouterHandle,
};

#[cfg(feature = "dyn-slot")]
pub use slot::{ElementSlot, SwapHandle};
