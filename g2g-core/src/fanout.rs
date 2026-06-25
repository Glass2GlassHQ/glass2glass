//! Fan-out primitives for the dynamic graph layer (DESIGN.md §4.8.4).
//!
//! M9 (1→N slice): a multi-output sink abstraction plus the two routing
//! primitives that cover branch enable/disable and A/B switching:
//!
//! - [`Gate`] — 1→1. Forwards or drops each `DataFrame` by an atomic flag.
//!   It is a plain [`AsyncElement`], so it drops into the existing
//!   `run_source_transform_sink` runner unchanged.
//! - [`Router`] — 1→N. Sends each `DataFrame` to exactly one output port
//!   chosen by an atomic discriminator, and broadcasts `CapsChanged` to
//!   every port. It implements [`MultiOutputElement`], driven by the
//!   `run_source_fanout` runner.
//!
//! Both expose a cloneable control handle ([`GateHandle`], [`RouterHandle`]),
//! mirroring `SwapHandle` (`slot.rs`), so application code or another task
//! flips routing mid-stream without stalling the pipeline.
//!
//! The Merger (fan-in) and `BranchSlot` are a later slice. EOS broadcast on
//! the `Router` is the runner's responsibility, matching the existing
//! "runner forwards Eos" transform contract.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::caps::Caps;
use crate::format_element::CapsConstraint;
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, ElementBound, OutputSink, PushOutcome,
};
use crate::error::G2gError;
use crate::frame::PipelinePacket;
use crate::property::{PropError, PropValue, PropertySpec};
use crate::runtime::SenderSink;

/// Downstream output addressing one of N ports. The fan-out analog of
/// [`OutputSink`]: `push_to` selects the destination port. Dyn-safe via a
/// boxed future so [`MultiOutputElement`] can take `&mut dyn MultiOutputSink`.
pub trait MultiOutputSink {
    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>>;

    fn port_count(&self) -> usize;
}

/// [`MultiOutputSink`] backed by one [`SenderSink`] per output link. Built
/// by the fan-out runner from the branch links; `push_to` forwards to the
/// addressed branch.
#[derive(Debug)]
pub struct MultiSenderSink {
    ports: Vec<SenderSink>,
}

impl MultiSenderSink {
    pub fn new(ports: Vec<SenderSink>) -> Self {
        Self { ports }
    }
}

impl MultiOutputSink for MultiSenderSink {
    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        // Port range is an internal invariant: `Router` clamps its selection
        // and broadcasts only over `0..port_count`, so an out-of-range port
        // is a framework bug, not a runtime error.
        let sink = self.ports.get_mut(port).expect("push_to: port out of range");
        sink.push(packet)
    }

    fn port_count(&self) -> usize {
        self.ports.len()
    }
}

/// A terminal multi-output *source*: 0 inputs to N outputs, driven by
/// [`run_fanout_session`](crate::runtime::run_fanout_session). Where
/// [`MultiOutputElement`] demultiplexes an upstream input stream, this generates
/// its outputs itself from an external source (e.g. a WHEP session that receives
/// video + audio over one PeerConnection and emits each on its own pad). It is
/// the fan-out mirror of a [`MultiInputElement`] used as a terminal session sink.
pub trait MultiOutputSource: ElementBound {
    type RunFuture<'a>: core::future::Future<Output = Result<u64, G2gError>> + 'a
    where
        Self: 'a;

    /// Number of output pads (one per produced track).
    fn output_count(&self) -> usize;

    /// The caps this source produces on `output`. The runner fixates each and
    /// configures the matching downstream sink before [`Self::run`]. Geometry the
    /// source only learns later (e.g. H.264 dimensions from the in-band SPS) is
    /// reported as `Any`, exactly as a single-output `SourceLoop` does.
    fn output_caps(&self, output: usize) -> Result<Caps, G2gError>;

    /// Run until EOS / disconnect, pushing frames to outputs via
    /// `out.push_to(port, ..)`. The implementation MUST push a
    /// [`PipelinePacket::Eos`] to every output before returning `Ok`, so no
    /// downstream branch is stranded. Returns the count of `DataFrame`s pushed.
    fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a>;
}

/// Inbound side of a [`MultiDuplexSession`]: the runner hands the session a
/// stream of `(input_index, packet)` drawn from its N send-side sources, the
/// receive-end analog of the [`MultiOutputSink`] it pushes received tracks into.
/// `recv` yields `None` once every send source has ended (all senders dropped),
/// so the session can stop publishing while still draining the peer.
pub trait DuplexInbound {
    fn recv(&mut self) -> BoxFuture<'_, Option<(usize, PipelinePacket)>>;
}

/// A terminal **duplex** session: N send-side inputs **and** M recv-side outputs
/// over one connection, with no external upstream or downstream beyond itself.
/// The union of [`MultiInputElement`] used as a terminal sink
/// ([`run_fanin_session`](crate::runtime::run_fanin_session)) and
/// [`MultiOutputSource`] ([`run_fanout_session`](crate::runtime::run_fanout_session)):
/// a `WebRtcBin`-style sendrecv PeerConnection both publishes local tracks and
/// emits the peer's tracks. Driven by
/// [`run_duplex_session`](crate::runtime::run_duplex_session).
///
/// One `run` loop owns the connection and is the sole holder of `&mut self`, so
/// (unlike the egress session, which spawns a detached task to dodge aliasing)
/// the send and recv halves share state directly: `run` selects over the inbound
/// packets (`inbound.recv()`) and the network, feeding the former into the
/// connection and pushing the latter to `out`.
pub trait MultiDuplexSession: ElementBound {
    type RunFuture<'a>: core::future::Future<Output = Result<u64, G2gError>> + 'a
    where
        Self: 'a;

    /// Number of send-side input pads (local tracks published to the peer).
    fn input_count(&self) -> usize;

    /// Number of recv-side output pads (peer tracks emitted locally).
    fn output_count(&self) -> usize;

    /// Phase 1 for one send-side input pad: narrow that input's proposed caps.
    fn intercept_caps(&self, input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError>;

    /// Phase 2 for one send-side input pad: fixate and configure it (the session
    /// reads the track kind, e.g. H.264 video vs Opus audio, from these caps).
    fn configure_input(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    /// The caps this session produces on one recv-side output pad. Geometry only
    /// learned later (e.g. H.264 dimensions from the in-band SPS) is reported as a
    /// `Range` placeholder, exactly as [`MultiOutputSource`] does.
    fn output_caps(&self, output: usize) -> Result<Caps, G2gError>;

    /// Declare one send input pad's negotiation-time constraint, mirroring
    /// [`MultiInputElement::caps_constraint_as_input`].
    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_>
    where
        Self: Sized,
    {
        CapsConstraint::LegacySink(alloc::boxed::Box::new(move |c: &Caps| {
            <Self as MultiDuplexSession>::intercept_caps(self, input, c)
        }))
    }

    /// Drive the session until the connection ends: drain `inbound` (the send-side
    /// packets, tagged with their input pad) into the connection and push received
    /// frames to `out`. Must push a [`PipelinePacket::Eos`] to every output before
    /// returning `Ok`, so no downstream branch is stranded. Returns the count of
    /// received `DataFrame`s pushed to outputs.
    fn run<'a>(
        &'a mut self,
        inbound: &'a mut dyn DuplexInbound,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::RunFuture<'a>;
}

/// Multi-output element trait variant: identical negotiation to
/// [`AsyncElement`], but `process` emits into a [`MultiOutputSink`] rather
/// than a single downstream. [`Router`] is the first implementor; user code
/// can write others (e.g. a content-based demux).
pub trait MultiOutputElement: ElementBound {
    type ProcessFuture<'a>: core::future::Future<Output = Result<(), G2gError>> + 'a
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError>;

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a>;

    /// M18 step 1: declare the fan-out's input-side negotiation
    /// constraint. Default wraps `intercept_caps(...)` as a
    /// `LegacySink` (the fan-out narrows what it accepts from the
    /// upstream source; downstream branches all receive the narrowed
    /// caps via broadcast). Migrated fan-outs override to return
    /// native variants (typically `AcceptsAny` for pass-through fan-
    /// outs like `Router` whose output broadcasts the input verbatim,
    /// or `Accepts(set)` for fan-outs that filter format on the input
    /// side).
    ///
    /// Phase C FO-2 (per-branch downstream re-solve) will sit on top
    /// of this: once the runner has the fan-out's negotiated input
    /// caps, it broadcasts to each branch and runs Phase B's
    /// `re_solve_downstream_sink` per branch sink.
    fn caps_constraint_as_input(&self) -> CapsConstraint<'_>
    where
        Self: Sized,
    {
        CapsConstraint::LegacySink(alloc::boxed::Box::new(move |c: &Caps| {
            <Self as MultiOutputElement>::intercept_caps(self, c)
        }))
    }

    /// Runtime properties this demux exposes (M104), mirroring
    /// [`AsyncElement::properties`](crate::AsyncElement::properties). Default:
    /// none. A demux overrides this (with `set_property` / `get_property`) to be
    /// settable by name from a `gst-launch` line, the same as a transform.
    fn properties(&self) -> &'static [PropertySpec] {
        &[]
    }

    /// Set a property by name (M104). Default: every name is unknown.
    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    /// Read a property back by name (M104). Default: `None`.
    fn get_property(&self, _name: &str) -> Option<PropValue> {
        None
    }
}

/// Multi-input element trait variant: an N-input, 1-output element (a
/// muxer). The mirror of [`MultiOutputElement`]. Negotiation is **per
/// input** — each input pad narrows and fixates its own caps — and the
/// element exposes a single merged `output_caps`. The fan-in runner
/// (`run_muxer_sink`) aggregates EOS itself, so `process` is only ever
/// handed `DataFrame`/`CapsChanged`, tagged with the originating `input`.
pub trait MultiInputElement: ElementBound {
    type ProcessFuture<'a>: core::future::Future<Output = Result<(), G2gError>> + 'a
    where
        Self: 'a;

    fn input_count(&self) -> usize;

    /// Whether the runner should deliver this element's inputs in global
    /// presentation-timestamp order. Default `false` (arrival-order round-robin,
    /// the historical behavior). When `true`, the runner merges the per-input
    /// streams by `DataFrame` PTS, releasing the globally-earliest only once every
    /// still-open input has one queued, so `process(pad, DataFrame(..))` arrives in
    /// non-decreasing PTS across all pads.
    ///
    /// An element wanting time-aligned input without hand-rolling an
    /// [`InputAggregator`](crate::InputAggregator) (a muxer, a multi-camera grid, a
    /// PTS-synchronized compositor) opts in by returning `true`. Per-input `Eos`
    /// and `CapsChanged` are still delivered as they occur; the merge holds only
    /// `DataFrame`s. Inputs are assumed monotonic in PTS (the merge invariant).
    fn input_pts_ordered(&self) -> bool {
        false
    }

    /// Phase 1 for one input pad: narrow that input's proposed caps.
    fn intercept_caps(&self, input: usize, upstream_caps: &Caps) -> Result<Caps, G2gError>;

    /// Phase 2 for one input pad: fixate and configure that input.
    fn configure_pipeline(
        &mut self,
        input: usize,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    /// The merged-output caps, valid once every input has been configured.
    fn output_caps(&self) -> Result<Caps, G2gError>;

    /// Combine one packet from `input` into the merged output.
    ///
    /// M22: a per-input `Eos` is delivered here when that input ends, so a
    /// stateful muxer (a batcher) can flush per-input state. Implementations
    /// must NOT forward `Eos` downstream: the runner aggregates input ends
    /// and emits the single merged `Eos` itself.
    fn process<'a>(
        &'a mut self,
        input: usize,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a>;

    /// M18 step 1: declare this input pad's negotiation-time
    /// constraint. Default wraps `intercept_caps(input, ...)` as a
    /// `LegacySink` (per-pad legacy bridge). Migrated muxers override
    /// to return native variants (typically `AcceptsAny` for
    /// per-frame-tagged interleave muxers, or `Accepts(set)` for
    /// per-input format-restricted muxers).
    ///
    /// The runner calls this per-input during startup negotiation
    /// (replacing the inline `LegacySink` construction in
    /// `run_muxer_sink`) and during per-input mid-stream re-solve
    /// once Phase C MX-1 lands.
    fn caps_constraint_as_input(&self, input: usize) -> CapsConstraint<'_>
    where
        Self: Sized,
    {
        CapsConstraint::LegacySink(alloc::boxed::Box::new(move |c: &Caps| {
            <Self as MultiInputElement>::intercept_caps(self, input, c)
        }))
    }

    /// M18 step 1: declare the merged output's negotiation-time
    /// constraint, evaluated against the muxer's current configured
    /// inputs. Default eagerly calls `output_caps()` and wraps as
    /// `LegacySource`. Migrated muxers with static or input-derived
    /// output may override with `Produces(set)` or `DerivedOutput(fn)`.
    ///
    /// The runner uses this in place of `output_caps()?.fixate()` so
    /// the downstream sink sees a uniformly-shaped constraint and the
    /// Phase B-style re-solve (workaround #3 §4) extends naturally to
    /// the muxer-output boundary.
    fn caps_constraint_for_output(&self) -> Result<CapsConstraint<'_>, G2gError>
    where
        Self: Sized,
    {
        Ok(CapsConstraint::LegacySource(self.output_caps()?))
    }

    /// Runtime properties this muxer exposes (M104), mirroring
    /// [`AsyncElement::properties`](crate::AsyncElement::properties). Default:
    /// none. A muxer overrides this (with `set_property` / `get_property`) to be
    /// settable by name from a `gst-launch` line, the same as a transform.
    fn properties(&self) -> &'static [PropertySpec] {
        &[]
    }

    /// Set a property by name (M104). Default: every name is unknown.
    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    /// Read a property back by name (M104). Default: `None`.
    fn get_property(&self, _name: &str) -> Option<PropValue> {
        None
    }
}

/// 1→1 enable/disable element. Forwards `CapsChanged` unconditionally and
/// `DataFrame` only while open; `Eos` is forwarded by the runner, never by
/// the element (the transform contract). Drops dropped frames silently —
/// observability of gate drops is a tracing concern for a later milestone.
#[derive(Debug)]
pub struct Gate {
    open: Arc<AtomicBool>,
}

impl Gate {
    pub fn new(open: bool) -> Self {
        Self { open: Arc::new(AtomicBool::new(open)) }
    }

    /// A cloneable handle that flips this gate from another task while the
    /// runner drives it.
    pub fn handle(&self) -> GateHandle {
        GateHandle { open: self.open.clone() }
    }
}

/// Detached control handle for a [`Gate`].
#[derive(Debug, Clone)]
pub struct GateHandle {
    open: Arc<AtomicBool>,
}

impl GateHandle {
    pub fn set_open(&self, open: bool) {
        self.open.store(open, Ordering::SeqCst);
    }

    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::SeqCst)
    }
}

impl AsyncElement for Gate {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        let open = self.open.load(Ordering::SeqCst);
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if open {
                        out.push(PipelinePacket::DataFrame(f)).await?;
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    out.push(PipelinePacket::CapsChanged(c)).await?;
                }
                // Flush is control: forward regardless of open state.
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                // Segment is control: forward regardless of open state.
                PipelinePacket::Segment(s) => {
                    out.push(PipelinePacket::Segment(s)).await?;
                }
                // Runner forwards Eos after process() returns.
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

/// 1→N router. Each `DataFrame` goes to the single port named by an atomic
/// discriminator; `CapsChanged` is broadcast to every port so all branches
/// stay configured. `Eos` is broadcast by the runner.
#[derive(Debug)]
pub struct Router {
    selected: Arc<AtomicUsize>,
    ports: usize,
}

impl Router {
    pub fn new(ports: usize) -> Self {
        assert!(ports > 0, "Router needs at least one output port");
        Self { selected: Arc::new(AtomicUsize::new(0)), ports }
    }

    /// Number of output ports. The fan-out runner allocates one branch link
    /// per port.
    pub fn port_count(&self) -> usize {
        self.ports
    }

    /// A cloneable handle that re-targets this router from another task.
    pub fn handle(&self) -> RouterHandle {
        RouterHandle { selected: self.selected.clone(), ports: self.ports }
    }
}

/// Detached control handle for a [`Router`].
#[derive(Debug, Clone)]
pub struct RouterHandle {
    selected: Arc<AtomicUsize>,
    ports: usize,
}

impl RouterHandle {
    /// Select the output port subsequent `DataFrame`s route to. Panics if
    /// `port >= port_count`.
    pub fn select(&self, port: usize) {
        assert!(port < self.ports, "select: port out of range");
        self.selected.store(port, Ordering::SeqCst);
    }

    pub fn selected(&self) -> usize {
        self.selected.load(Ordering::SeqCst)
    }
}

impl MultiOutputElement for Router {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    /// M18 step 1: pass-through wildcard. `Router` broadcasts the
    /// upstream caps verbatim to every active branch and has no
    /// per-branch format restriction. `AcceptsAny` is the native
    /// shape; skips the dynamic intercept callback on the solver
    /// path.
    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> Self::ProcessFuture<'a> {
        // Clamp defensively so a stale handle write can never index past the
        // port list (the runner allocated exactly `ports` branches).
        let selected = self.selected.load(Ordering::SeqCst).min(self.ports - 1);
        let ports = self.ports;
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    out.push_to(selected, PipelinePacket::DataFrame(f)).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    for port in 0..ports {
                        out.push_to(port, PipelinePacket::CapsChanged(c.clone())).await?;
                    }
                }
                // Flush is broadcast to every branch, like CapsChanged.
                PipelinePacket::Flush => {
                    for port in 0..ports {
                        out.push_to(port, PipelinePacket::Flush).await?;
                    }
                }
                // Segment is broadcast to every branch, like CapsChanged.
                PipelinePacket::Segment(s) => {
                    for port in 0..ports {
                        out.push_to(port, PipelinePacket::Segment(s)).await?;
                    }
                }
                // Runner broadcasts Eos to all ports after process() returns.
                PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

/// N→1 fan-in selector: the control-driven mirror of [`Router`]. An atomic
/// discriminator names the single active input; the fan-in runner forwards
/// that input's frames and drains/discards the rest. The merged stream ends
/// only once every input has reached EOS (see `run_fanin_sink`). `Merger`
/// holds just the selector; the forwarding lives in the runner.
#[derive(Debug)]
pub struct Merger {
    selected: Arc<AtomicUsize>,
    inputs: usize,
}

impl Merger {
    pub fn new(inputs: usize) -> Self {
        assert!(inputs > 0, "Merger needs at least one input");
        Self { selected: Arc::new(AtomicUsize::new(0)), inputs }
    }

    /// Number of input ports. The fan-in runner allocates one branch link
    /// per input.
    pub fn input_count(&self) -> usize {
        self.inputs
    }

    /// A cloneable handle that re-selects the active input from another task.
    pub fn handle(&self) -> MergerHandle {
        MergerHandle { selected: self.selected.clone(), inputs: self.inputs }
    }
}

/// Detached control handle for a [`Merger`].
#[derive(Debug, Clone)]
pub struct MergerHandle {
    selected: Arc<AtomicUsize>,
    inputs: usize,
}

impl MergerHandle {
    /// Select which input feeds the merged output. Panics if
    /// `input >= input_count`.
    pub fn select(&self, input: usize) {
        assert!(input < self.inputs, "select: input out of range");
        self.selected.store(input, Ordering::SeqCst);
    }

    pub fn selected(&self) -> usize {
        self.selected.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Dim, Rate, RawVideoFormat};
    use crate::frame::{Frame, FrameTiming};
    use crate::memory::{MemoryDomain, SystemSlice};
    use core::future::Future;
    use core::pin::Pin;

    fn caps() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(16),
            height: Dim::Fixed(16),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    fn data(seq: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            timing: FrameTiming::default(),
            sequence: seq,
            meta: Default::default(),
        })
    }

    /// Records the kind of every packet pushed, per port, without channels.
    #[derive(Default)]
    struct RecordingMultiSink {
        ports: usize,
        data_seqs: Vec<Vec<u64>>,
        caps_changes: Vec<usize>,
    }

    impl RecordingMultiSink {
        fn new(ports: usize) -> Self {
            Self { ports, data_seqs: alloc::vec![Vec::new(); ports], caps_changes: alloc::vec![0; ports] }
        }
    }

    impl MultiOutputSink for RecordingMultiSink {
        fn push_to<'a>(
            &'a mut self,
            port: usize,
            packet: PipelinePacket,
        ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
            match packet {
                PipelinePacket::DataFrame(f) => self.data_seqs[port].push(f.sequence),
                PipelinePacket::CapsChanged(_) => self.caps_changes[port] += 1,
                PipelinePacket::Eos | PipelinePacket::Flush | PipelinePacket::Segment(_) => {}
            }
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }

        fn port_count(&self) -> usize {
            self.ports
        }
    }

    /// Records every packet a single-output element forwards.
    #[derive(Default)]
    struct RecordingSink {
        data_seqs: Vec<u64>,
        caps_changes: usize,
    }

    impl OutputSink for RecordingSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
            match packet {
                PipelinePacket::DataFrame(f) => self.data_seqs.push(f.sequence),
                PipelinePacket::CapsChanged(_) => self.caps_changes += 1,
                PipelinePacket::Eos | PipelinePacket::Flush | PipelinePacket::Segment(_) => {}
            }
            Box::pin(async { Ok(PushOutcome::Accepted) })
        }
    }

    /// Single-poll block_on; all futures here resolve immediately.
    fn block_on<F: Future>(mut fut: F) -> F::Output {
        use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        static VT: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VT),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: VT's hooks never dereference the data pointer.
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VT)) };
        let mut cx = Context::from_waker(&waker);
        // SAFETY: `fut` is pinned to the stack for the duration of this call.
        let mut pinned = unsafe { Pin::new_unchecked(&mut fut) };
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(v) => v,
            Poll::Pending => panic!("fanout::tests::block_on saw Pending"),
        }
    }

    #[test]
    fn router_input_constraint_is_wildcard() {
        // M18 step 1: Router broadcasts upstream caps verbatim, no
        // per-branch format restriction. AcceptsAny is the native
        // shape; skips the dynamic intercept callback on the solver
        // path.
        let r = Router::new(3);
        let c = r.caps_constraint_as_input();
        assert!(
            matches!(c, CapsConstraint::AcceptsAny),
            "Router input should be AcceptsAny, got {c:?}"
        );
    }

    #[test]
    fn router_sends_each_frame_to_selected_port() {
        let mut router = Router::new(2);
        let handle = router.handle();
        let mut out = RecordingMultiSink::new(2);

        block_on(router.process(data(0), &mut out)).unwrap(); // port 0
        handle.select(1);
        block_on(router.process(data(1), &mut out)).unwrap(); // port 1
        block_on(router.process(data(2), &mut out)).unwrap(); // port 1 (sticky)
        handle.select(0);
        block_on(router.process(data(3), &mut out)).unwrap(); // port 0

        assert_eq!(out.data_seqs[0], alloc::vec![0, 3]);
        assert_eq!(out.data_seqs[1], alloc::vec![1, 2]);
    }

    #[test]
    fn router_broadcasts_caps_changed_to_all_ports() {
        let mut router = Router::new(3);
        let mut out = RecordingMultiSink::new(3);

        block_on(router.process(PipelinePacket::CapsChanged(caps()), &mut out)).unwrap();

        assert_eq!(out.caps_changes, alloc::vec![1, 1, 1]);
    }

    #[test]
    fn gate_open_forwards_data_closed_drops_it() {
        let gate = Gate::new(true);
        let handle = gate.handle();
        let mut gate = gate;
        let mut out = RecordingSink::default();

        block_on(gate.process(data(0), &mut out)).unwrap(); // open -> pass
        handle.set_open(false);
        block_on(gate.process(data(1), &mut out)).unwrap(); // closed -> drop
        handle.set_open(true);
        block_on(gate.process(data(2), &mut out)).unwrap(); // open -> pass

        assert_eq!(out.data_seqs, alloc::vec![0, 2], "frame 1 dropped while closed");
    }

    #[test]
    fn gate_forwards_caps_changed_regardless_of_open_state() {
        let mut gate = Gate::new(false);
        let mut out = RecordingSink::default();

        block_on(gate.process(PipelinePacket::CapsChanged(caps()), &mut out)).unwrap();

        assert_eq!(out.caps_changes, 1, "CapsChanged forwarded even while closed");
    }

    #[test]
    fn merger_handle_selects_active_input() {
        let merger = Merger::new(3);
        let handle = merger.handle();
        assert_eq!(handle.selected(), 0, "defaults to input 0");
        handle.select(2);
        assert_eq!(handle.selected(), 2);
        assert_eq!(merger.input_count(), 3);
    }

    #[test]
    #[should_panic(expected = "input out of range")]
    fn merger_handle_rejects_out_of_range_input() {
        Merger::new(2).handle().select(2);
    }
}
