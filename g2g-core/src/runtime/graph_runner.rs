//! DAG pipeline runner (DESIGN_TODO "DAG runner" D3).
//!
//! [`run_graph`] drives an arbitrary multimedia DAG built with [`Graph`]:
//! whole-graph CSP negotiation via [`solve_graph`] (D2), then one spawned arm
//! per node over per-edge channels, joined with [`join_all`]. It collapses the
//! linear + fan-out runner shapes into one entry point.
//!
//! Scope: source / transform / sink / tee (fan-out) + muxer (fan-in). A tee
//! broadcasts each packet to all its branches via [`MemoryDomain::share`]
//! (M213): a zero-copy refcount bump for the GPU domains and the shared-CPU
//! `SystemView`, a deep copy only for owned-CPU `System` bytes. So a GPU-decoded
//! frame fans out to several consumers (inference + display) with no
//! device-to-host copy. A muxer node
//! runs a [`DynMultiInputElement`]: per-input forwarder arms tag each packet
//! with its pad and feed one muxer arm that combines them, emitting a single
//! `Eos` after every input ends (the `run_muxer_sink` shape).
//!
//! D4 adds the mid-stream re-solve and the β allocation re-cascade over the
//! DAG. Each arm gets a per-edge downstream feasibility snapshot at startup
//! ([`graph_downstream_feasibility`]); on a mid-stream `CapsChanged` a transform
//! steers its forwarded output toward a downstream-acceptable shape (Caps-α),
//! and a sink re-solves its input against its declared constraint. A node-keyed
//! [`GraphCoordinator`] walks the sink's re-derived allocation proposal one hop
//! upstream per reply via [`ValidatedGraph::in_edges`], resolving through
//! structural tee nodes; a source terminates the walk. Tee branches re-solve
//! independently (each broadcast `CapsChanged` lands in its own arm); muxer
//! inputs re-configure per pad. A muxer's per-pad allocation demand
//! (`propose_allocation_for_input`) crosses the boundary both at startup
//! negotiation and mid-stream: a `CapsChanged` on one pad re-cascades the
//! re-derived proposal up that pad's branch alone (the `Recascade::target`
//! override), leaving the other inputs untouched.

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::pin::Pin;

// The thread-per-arm `ThreadSpawner` uses `std::thread`; `std` is otherwise
// unused by this (no_std baseline) module.
#[cfg(all(feature = "std", feature = "multi-thread"))]
extern crate std;

use crate::bus::{BusHandle, BusMessage};
use crate::caps::{Caps, CapsSet};
use crate::clock::{elect_clock, ClockCandidate, ClockPriority, ClockSync, PipelineClock};
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, DynAsyncElement, ElementBound, OutputSink,
    Reconfigure,
};
use crate::error::G2gError;
use crate::aggregator::InputAggregator;
use crate::fanout::{MultiInputElement, MultiOutputElement, MultiOutputSink, MultiSenderSink};
use crate::format_element::CapsConstraint;
use crate::frame::{Frame, PipelinePacket};
use crate::graph::{FanOutPolicy, Graph, NodeId, NodeKind, ValidatedGraph};
use crate::memory::{DomainSet, MemoryDomainKind};
use crate::property::{PropError, PropValue, PropertySpec};
use crate::segment::Segment;
use crate::query::{AllocationParams, LatencyReport};
use crate::runtime::channel::{bounded, link, LinkReceiver, LinkSender, Receiver, Sender, SenderSink};
use crate::runtime::coordinator::{realloc_local_dyn, report_nego_failure, ArmDirective};
use crate::runtime::fanin::{DynMultiInputElement, DynSourceLoop};
use crate::runtime::instrument::{snapshot_all, ElementProbe, Probe};
use crate::runtime::Observer;
use crate::runtime::join::{join_all, select2, Either};
use crate::runtime::progress::PipelineProgress;
use crate::runtime::runner::{
    re_solve_downstream_dyn_sink, LinkCapacity, NullSink, RunStats, SourceLoop,
};
use crate::runtime::solver::{
    graph_downstream_feasibility, resolve_forward_output, solve_graph_labeled, solve_linear,
    ForwardResolve, NegotiationFailure, NodeConstraint,
};
use crate::runtime::state::{Flow, StateController};

/// Element payload for a [`Graph`] driven by [`run_graph`]. Sources,
/// transforms/sinks, and muxers implement different traits (a source has no
/// input pad, a muxer has many), so the payload is an enum the runner matches
/// on per node kind. A tee carries no element (`Graph::add_tee` takes none).
///
/// The `'a` lifetime is the lifetime of the boxed elements. Owned `'static`
/// elements (the common case, [`GraphNode`]) use `source` / `element` / `muxer`;
/// the convenience wrappers build a *borrowing* graph over their `&mut` element
/// references with `source_ref` / `element_ref` / `muxer_ref`, so they can call
/// `run_graph` without taking ownership and the caller keeps its elements.
pub enum GraphNodeRef<'a> {
    Source(Box<dyn DynSourceLoop + 'a>),
    Element(Box<dyn DynAsyncElement + 'a>),
    Muxer(Box<dyn DynMultiInputElement + 'a>),
    /// A content-routing demultiplexer: 1 input, N outputs. Structurally a tee
    /// (its node kind is `Tee(n)`), so it negotiates as a tee at startup, but it
    /// carries a [`MultiOutputElement`] that routes each packet to a chosen
    /// output instead of broadcasting, and emits per-output `CapsChanged` so each
    /// branch retypes from the byte-stream input (M210). `Graph::add_demux`.
    Demux(Box<dyn DynMultiOutputElement + 'a>),
}

/// The owning, `'static` graph payload: what most callers build directly.
pub type GraphNode = GraphNodeRef<'static>;

impl<'a> GraphNodeRef<'a> {
    /// Box an owned source (`add_source`).
    pub fn source<S: SourceLoop + 'static>(source: S) -> Self {
        GraphNodeRef::Source(Box::new(source))
    }

    /// Box an owned transform or sink (`add_transform` / `add_sink`).
    pub fn element<E: AsyncElement + 'static>(element: E) -> Self {
        GraphNodeRef::Element(Box::new(element))
    }

    /// Box an owned fan-in muxer (`add_muxer`).
    pub fn muxer<M: MultiInputElement + 'static>(muxer: M) -> Self {
        GraphNodeRef::Muxer(Box::new(muxer))
    }

    /// Box a borrowed source, for a borrowing graph (the convenience wrappers).
    pub fn source_ref(source: &'a mut (dyn DynSourceLoop + 'a)) -> Self {
        GraphNodeRef::Source(Box::new(source))
    }

    /// Box a borrowed transform or sink.
    pub fn element_ref(element: &'a mut (dyn DynAsyncElement + 'a)) -> Self {
        GraphNodeRef::Element(Box::new(element))
    }

    /// Box a borrowed fan-in muxer.
    pub fn muxer_ref(muxer: &'a mut (dyn DynMultiInputElement + 'a)) -> Self {
        GraphNodeRef::Muxer(Box::new(muxer))
    }

    /// Box an owned fan-out demultiplexer (`add_demux`).
    pub fn demux<D: MultiOutputElement + 'static>(demux: D) -> Self {
        GraphNodeRef::Demux(Box::new(demux))
    }

    /// Box a borrowed fan-out demultiplexer.
    pub fn demux_ref(demux: &'a mut (dyn DynMultiOutputElement + 'a)) -> Self {
        GraphNodeRef::Demux(Box::new(demux))
    }

    /// The element's log category (M179), its short type name, e.g.
    /// `videotestsrc`. The runner uses it to derive instance names
    /// (`<category>N`); a DOT dump uses it as the node label before the run
    /// assigns the suffixed name. Fan-in / fan-out elements don't expose a
    /// category on their dyn trait (the runner doesn't name them either), so
    /// they report their structural role.
    pub fn log_category(&self) -> &'static str {
        match self {
            GraphNodeRef::Source(s) => s.log_category(),
            GraphNodeRef::Element(e) => e.log_category(),
            GraphNodeRef::Muxer(_) => "mux",
            GraphNodeRef::Demux(_) => "demux",
        }
    }

    /// The memory domain of the frames this node emits on its output pad(s)
    /// (M285): the source's / element's `output_memory`, surfaced per edge for
    /// the DOT dump so a GPU / zero-copy link is marked. Fan-in / fan-out
    /// elements are reported as `System` (their domain is the upstream's; the
    /// per-edge derivation does not propagate through them yet).
    pub fn output_memory(&self) -> crate::memory::MemoryDomainKind {
        match self {
            GraphNodeRef::Source(s) => s.output_memory(),
            GraphNodeRef::Element(e) => e.output_memory(),
            GraphNodeRef::Muxer(_) | GraphNodeRef::Demux(_) => {
                crate::memory::MemoryDomainKind::System
            }
        }
    }

    /// The full set of memory domains this node can emit (M351), the
    /// producer-capability half of the two-sided allocation-domain negotiation.
    /// A source's / element's `output_domains`; fan-in / fan-out nodes report a
    /// System singleton (their domain follows the upstream, like
    /// [`output_memory`](Self::output_memory)).
    pub fn output_domains(&self) -> crate::memory::DomainSet {
        match self {
            GraphNodeRef::Source(s) => s.output_domains(),
            GraphNodeRef::Element(e) => e.output_domains(),
            GraphNodeRef::Muxer(_) | GraphNodeRef::Demux(_) => {
                crate::memory::DomainSet::only(crate::memory::MemoryDomainKind::System)
            }
        }
    }

    /// The memory domains this node accepts on its input pad (M354), for the
    /// converter auto-plug. A source has no input; fan-in/out nodes are
    /// domain-transparent here, so both report [`DomainSet::ALL`] (no requirement).
    pub fn input_domains(&self) -> crate::memory::DomainSet {
        match self {
            GraphNodeRef::Element(e) => e.input_domains(),
            GraphNodeRef::Source(_) | GraphNodeRef::Muxer(_) | GraphNodeRef::Demux(_) => {
                crate::memory::DomainSet::ALL
            }
        }
    }
}

impl core::fmt::Debug for GraphNodeRef<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GraphNodeRef::Source(_) => f.write_str("GraphNodeRef::Source(..)"),
            GraphNodeRef::Element(_) => f.write_str("GraphNodeRef::Element(..)"),
            GraphNodeRef::Muxer(_) => f.write_str("GraphNodeRef::Muxer(..)"),
            GraphNodeRef::Demux(_) => f.write_str("GraphNodeRef::Demux(..)"),
        }
    }
}

/// Dyn-safe mirror of [`MultiOutputElement`] for a fan-out demux node in the DAG
/// runner, the transpose of [`DynMultiInputElement`]. Boxes `process`'s future
/// and forwards the `Self: Sized` constraint methods. Only the methods the
/// runner uses are mirrored.
pub trait DynMultiOutputElement: ElementBound {
    fn caps_constraint_as_input(&self) -> CapsConstraint<'_>;
    fn port_output_caps(&self, port: usize) -> Option<Caps>;
    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError>;
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> BoxFuture<'a, Result<(), G2gError>>;
    fn properties(&self) -> &'static [PropertySpec];
    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError>;
    fn get_property(&self, name: &str) -> Option<PropValue>;
}

impl<T: MultiOutputElement> DynMultiOutputElement for T {
    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        MultiOutputElement::caps_constraint_as_input(self)
    }

    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        MultiOutputElement::port_output_caps(self, port)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        MultiOutputElement::configure_pipeline(self, absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> BoxFuture<'a, Result<(), G2gError>> {
        Box::pin(MultiOutputElement::process(self, packet, out))
    }

    fn properties(&self) -> &'static [PropertySpec] {
        MultiOutputElement::properties(self)
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        MultiOutputElement::set_property(self, name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        MultiOutputElement::get_property(self, name)
    }
}

/// Forwarding impl so a borrowed `&mut dyn DynMultiOutputElement` can be boxed
/// into a `Box<dyn DynMultiOutputElement + 'a>` graph node (the borrowing-graph
/// convenience wrappers). Disjoint from the `MultiOutputElement` blanket above.
impl<'b> DynMultiOutputElement for &'b mut (dyn DynMultiOutputElement + 'b) {
    fn caps_constraint_as_input(&self) -> CapsConstraint<'_> {
        (**self).caps_constraint_as_input()
    }

    fn port_output_caps(&self, port: usize) -> Option<Caps> {
        (**self).port_output_caps(port)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        (**self).configure_pipeline(absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn MultiOutputSink,
    ) -> BoxFuture<'a, Result<(), G2gError>> {
        (**self).process(packet, out)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        (**self).properties()
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        (**self).set_property(name, value)
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        (**self).get_property(name)
    }
}

/// A β allocation re-cascade report from an arm to the [`GraphCoordinator`].
/// A sink reports the proposal it re-derived on a mid-stream `CapsChanged`; an
/// interior transform reports the proposal it re-derived after applying an
/// upstream directive. `node` is the reporting node, so the coordinator walks
/// the proposal one hop further upstream through the graph topology.
///
/// `target` overrides that topology walk: a muxer has many inputs but a
/// mid-stream `CapsChanged` arrives on one pad, so it names the single upstream
/// arm feeding that pad's branch rather than re-cascading to all of them (which
/// the node-keyed `upstream_arms` lookup would do). `None` for the ordinary
/// single-input transform / sink path.
#[derive(Debug, Clone)]
struct Recascade {
    node: NodeId,
    target: Option<NodeId>,
    proposal: Option<AllocationParams>,
}

/// Producer end of the graph coordinator's control channel, cloned to each
/// transform and sink arm so it can report a [`Recascade`].
#[derive(Debug, Clone)]
struct GraphCoordHandle {
    tx: Sender<Recascade>,
}

impl GraphCoordHandle {
    async fn report(&self, event: Recascade) {
        let _ = self.tx.send(event).await;
    }
}

/// Node-keyed β coordinator for the DAG (the DAG analog of the linear
/// [`Coordinator`](crate::runtime::coordinator::Coordinator)). It owns one
/// [`ArmDirective`] sender per interruptible interior arm (transforms), keyed by
/// node id, plus `upstream_arms`: for each reporting node, the nearest interior
/// transform arm feeding each of its inputs (resolved through structural tee
/// nodes; a source terminates the walk). On each report it forwards an
/// `ArmDirective::Recascade` one hop upstream to those arms, which re-derive and
/// report again, so the cascade walks the DAG without a global lock. A report
/// carrying an explicit [`Recascade::target`] (a muxer's per-pad re-cascade)
/// forwards to that one arm instead of the node-keyed set. The walk is reactive
/// and non-blocking (`try_send`), so it never wedges the data plane.
#[derive(Debug)]
struct GraphCoordinator {
    rx: Receiver<Recascade>,
    arm_ctrl: Vec<Option<Sender<ArmDirective>>>,
    upstream_arms: Vec<Vec<NodeId>>,
}

impl GraphCoordinator {
    async fn run(self) -> u64 {
        let mut observed = 0u64;
        while let Some(event) = self.rx.recv().await {
            observed += 1;
            if let Some(p) = event.proposal {
                match event.target {
                    // Muxer per-pad path: re-cascade up exactly the one branch
                    // whose pad changed.
                    Some(t) => {
                        if let Some(ctrl) = &self.arm_ctrl[t.0 as usize] {
                            let _ = ctrl.try_send(ArmDirective::Recascade(p));
                        }
                    }
                    // Single-input path: walk to every arm feeding the reporter.
                    None => {
                        for &u in &self.upstream_arms[event.node.0 as usize] {
                            if let Some(ctrl) = &self.arm_ctrl[u.0 as usize] {
                                let _ = ctrl.try_send(ArmDirective::Recascade(p));
                            }
                        }
                    }
                }
            }
        }
        observed
    }
}

/// The nearest upstream interruptible arm feeding `edge_id`: a transform is an
/// arm; a structural tee is skipped to its own single input; a source or muxer
/// terminates the β walk (neither carries a per-element allocation re-cascade).
fn nearest_upstream_arm<E>(vg: &ValidatedGraph<E>, edge_id: usize) -> Option<NodeId> {
    let src = vg.edge(edge_id).src.node;
    match vg.kind(src) {
        NodeKind::Transform => Some(src),
        NodeKind::Tee(_) => nearest_upstream_arm(vg, vg.in_edges(src)[0]),
        _ => None,
    }
}

/// The fan-out policy of the nearest tee on the path from `node` up toward a
/// source, or `None` if `node` is on a single-producer chain. A node behind a
/// tee shares its upstream with sibling branches, so it can't reverse-
/// reconfigure on a rejected mid-stream change: under `FailLoud` it fails the
/// run (the `run_source_fanout` strict default), under `AllowBranchDrop` it
/// drops out. A node on a single-producer chain (`run_linear_chain` /
/// `run_source_transform_sink`) reverse-reconfigures and keeps flowing.
fn behind_tee_policy<E>(vg: &ValidatedGraph<E>, node: NodeId) -> Option<FanOutPolicy> {
    let mut cur = node;
    loop {
        let ins = vg.in_edges(cur);
        if ins.is_empty() {
            return None;
        }
        let src = vg.edge(ins[0]).src.node;
        if matches!(vg.kind(src), NodeKind::Tee(_)) {
            return Some(vg.fanout_policy(src));
        }
        cur = src;
    }
}

/// How a branch arm reacts to a mid-stream `CapsChanged` it cannot negotiate,
/// derived from its position ([`behind_tee_policy`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum BranchMode {
    /// Single-producer chain: post the failure and reverse-reconfigure into the
    /// boundary that emitted the change, then keep flowing.
    Reconfigure,
    /// Behind a `FailLoud` tee: a rejected change fails the whole run loud.
    FailLoud,
    /// Behind an `AllowBranchDrop` tee: a rejected change drops this branch (its
    /// arm ends Ok) while the siblings keep flowing.
    Drop,
}

fn branch_mode<E>(vg: &ValidatedGraph<E>, node: NodeId) -> BranchMode {
    match behind_tee_policy(vg, node) {
        None => BranchMode::Reconfigure,
        Some(FanOutPolicy::FailLoud) => BranchMode::FailLoud,
        Some(FanOutPolicy::AllowBranchDrop) => BranchMode::Drop,
    }
}

/// A reusable recipe for an owned graph: a builder closure that produces a fresh
/// [`Graph<GraphNode>`](Graph) each time it is [`instantiate`](Self::instantiate)d.
///
/// [`run_graph`] consumes the elements it runs (it `take()`s the boxed payloads
/// out of the graph), so a graph cannot be run twice. Seek-and-replay (re-run
/// from the start after a flushing seek), retry-on-error, and A/B benchmarking
/// all need a *fresh* set of elements per run, because real elements carry state
/// (a decoder's reference frames, a source's file offset) that cannot simply be
/// rewound. A template rebuilds them via the closure rather than cloning, which
/// is cleaner than making `Graph` itself reusable: that would force every element
/// to be `Clone` or re-initialisable in place, a contract the element traits
/// deliberately do not impose.
pub struct GraphTemplate {
    build: Box<dyn Fn() -> Graph<GraphNode> + Send + Sync>,
}

impl GraphTemplate {
    /// Wrap a graph-builder closure. The closure must construct the whole graph
    /// (nodes + links) from scratch on each call, so each instance gets its own
    /// elements.
    pub fn new(build: impl Fn() -> Graph<GraphNode> + Send + Sync + 'static) -> Self {
        Self { build: Box::new(build) }
    }

    /// Build a fresh runnable graph. Call once per [`run_graph`] invocation.
    pub fn instantiate(&self) -> Graph<GraphNode> {
        (self.build)()
    }
}

impl core::fmt::Debug for GraphTemplate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GraphTemplate").finish_non_exhaustive()
    }
}

/// Drive an arbitrary DAG to EOS. Negotiates the whole graph at once, then runs
/// one arm per node over per-edge channels. `link_capacity` accepts a
/// [`LatencyProfile`](crate::runtime::LatencyProfile) or a `usize` depth.
///
/// Reports the M12 stats (latency / clock / allocation) folded over the graph,
/// the same as the linear runners. The graph payload may own its elements
/// ([`GraphNode`]) or borrow them ([`GraphNodeRef<'a>`], what the convenience
/// wrappers build). A non-`System` frame in a tee or a negotiation conflict
/// fails loud.
pub async fn run_graph<'a, Clk: PipelineClock>(
    graph: Graph<GraphNodeRef<'a>>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError> {
    run_graph_inner(graph, clock, link_capacity, None, None, None, None, None).await
}

/// As [`run_graph`], but enforces a memory-domain [`CopyPolicy`](crate::copyplan::CopyPolicy)
/// as a graph-level contract (M617). After negotiation resolves every link's memory
/// domain, the copy plan (`crate::copyplan`) is checked against `policy` *before any
/// frame flows*: a pipeline that must stay zero-copy
/// ([`CopyPolicy::DenyAll`](crate::copyplan::CopyPolicy::DenyAll)) refuses to start
/// with [`G2gError::CopyBudget`] if an accidental host round-trip appears, rather than
/// paying it at runtime. This turns "is this pipeline zero-copy?" from a question
/// measured after the fact into a guarantee checked at construction. Use
/// [`copy_plan`] on the same graph for the offending-transfer detail.
pub async fn run_graph_with_copy_policy<'a, Clk: PipelineClock>(
    graph: Graph<GraphNodeRef<'a>>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    policy: crate::copyplan::CopyPolicy,
) -> Result<RunStats, G2gError> {
    run_graph_inner(graph, clock, link_capacity, None, None, None, Some(policy), None).await
}

/// Splice memory-domain converters where a producer and consumer cannot agree on
/// a domain (M354), the structural complement to the M351/M352 in-band domain
/// negotiation: negotiation settles a *shared* domain when one exists, and this
/// inserts a converter when one does not. For each original edge `P -> C`, the
/// producer domain is traced through structural tee/demux nodes back to the real
/// producer ([`output_domains`](GraphNodeRef::output_domains)); if it shares no
/// domain with `C`'s [`input_domains`](GraphNodeRef::input_domains), `factory` is
/// asked for a converter from the producer's preferred domain to one `C` accepts,
/// and it is spliced onto that edge ([`Graph::insert_on_edge`]).
///
/// Converters are caps-transparent (`Identity`), so the subsequent caps solve is
/// unaffected. `factory` returns `None` when it has no converter for a pair, in
/// which case the edge is left as-is and the conflict surfaces later as an
/// [`AllocationConflict`](G2gError::AllocationConflict) (the loud failure M351
/// already gives). The converter elements live in `g2g-plugins`, so the caller
/// supplies the factory (the crate layering keeps `g2g-core` converter-agnostic);
/// `g2g-plugins` provides the CUDA wrapper.
pub fn auto_plug_domain_converters<'a>(
    mut graph: Graph<GraphNodeRef<'a>>,
    factory: &dyn Fn(MemoryDomainKind, MemoryDomainKind) -> Option<GraphNodeRef<'a>>,
) -> Graph<GraphNodeRef<'a>> {
    // Snapshot the original edge count: splicing appends nodes and a `K -> C`
    // edge and only rewires the edge being spliced, so original ids stay valid.
    let original_edges = graph.edges().len();
    for e in 0..original_edges {
        let edge = graph.edges()[e];
        let producer = traced_output_domains(&graph, edge.src.node);
        let consumer =
            graph.element(edge.dst.node).map(|n| n.input_domains()).unwrap_or(DomainSet::ALL);
        if !producer.intersect(consumer).is_empty() {
            continue; // already compatible, no converter needed
        }
        let (Some(from), Some(to)) = (producer.preferred(), consumer.preferred()) else {
            continue;
        };
        if let Some(conv) = factory(from, to) {
            graph.insert_on_edge(e, conv);
        }
    }
    graph
}

/// Domains the node emits, tracing through structural tee/demux nodes (which
/// forward their single input's domain) to the real producer. Used by the
/// converter auto-plug so a tee fed by a GPU decoder reports the GPU domain on
/// every branch rather than the structural `System` default.
fn traced_output_domains<'a>(graph: &Graph<GraphNodeRef<'a>>, node: NodeId) -> DomainSet {
    if let Some(NodeKind::Tee(_)) = graph.node_kind(node) {
        // A tee/demux is domain-transparent: trace its single input's producer.
        return match graph.edges().iter().find(|e| e.dst.node == node) {
            Some(in_edge) => traced_output_domains(graph, in_edge.src.node),
            None => DomainSet::only(MemoryDomainKind::System),
        };
    }
    graph
        .element(node)
        .map(|n| n.output_domains())
        .unwrap_or(DomainSet::only(MemoryDomainKind::System))
}

/// As [`run_graph`], but posts pipeline [`BusMessage`](crate::BusMessage)s to
/// `bus`: a startup `NegotiationFailed`, per sink a `Buffering` level report
/// each time the sink's input link crosses a fill quartile (M87), and a
/// `DurationChanged` when a source first reports its duration (M203), so the app
/// can show a buffering indicator, wait for a full buffer, or size a seek bar.
pub async fn run_graph_with_bus<'a, Clk: PipelineClock>(
    graph: Graph<GraphNodeRef<'a>>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: &BusHandle,
) -> Result<RunStats, G2gError> {
    run_graph_inner(graph, clock, link_capacity, Some(bus), None, None, None, None).await
}

/// As [`run_graph`], but taps live telemetry into `observer` and (optionally)
/// posts events to `bus`, the pairing a dev dashboard consumes: the observer
/// carries the graph topology plus per-element `process()` latency / input-link
/// fill, readable mid-run via [`Observer::snapshot`](crate::runtime::Observer::snapshot)
/// from a concurrent task, while the bus carries the out-of-band events (caps
/// changes surface as `Info`/`NegotiationFailed`, plus `Buffering` / `Qos` /
/// `Eos` / `Error`). Pass `bus: None` for telemetry only.
pub async fn run_graph_observed<'a, Clk: PipelineClock>(
    graph: Graph<GraphNodeRef<'a>>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    observer: &Observer,
    bus: Option<&BusHandle>,
) -> Result<RunStats, G2gError> {
    run_graph_inner(graph, clock, link_capacity, bus, None, None, None, Some(observer)).await
}

/// As [`run_graph`], but publishes playback progress into `progress` (M203): the
/// sink arm publishes the stream-time [`position`](PipelineProgress::position) of
/// every buffer it consumes, and the source arm publishes the
/// [`duration`](PipelineProgress::duration) its source reports
/// ([`SourceLoop::query_duration`]). The application polls the handle while the
/// pipeline runs, the `POSITION` / `DURATION` query analog. Pair with
/// [`run_graph_with_bus`] for the matching `DurationChanged` push notification.
pub async fn run_graph_with_progress<'a, Clk: PipelineClock>(
    graph: Graph<GraphNodeRef<'a>>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    progress: &PipelineProgress,
) -> Result<RunStats, G2gError> {
    run_graph_inner(graph, clock, link_capacity, None, None, Some(progress), None, None).await
}

/// Coarsen a link fill percent into a 0..=4 quartile band, so a sink posts a
/// `Buffering` message on a meaningful level transition (underrun, quarter
/// steps, full) rather than on every packet.
fn buffering_bucket(percent: u8) -> u8 {
    (percent / 25).min(4)
}

/// As [`run_graph`], but driven by a [`StateController`] (M78): every `Sink`
/// arm gates on the controller, so the whole DAG (linear / fan-out / fan-in /
/// diamond) honors `NULL → READY → PAUSED → PLAYING`. Preroll aggregates: the
/// async `Paused` transition completes only when *all* sinks have prerolled
/// (the runner calls [`StateController::expect_prerolls`] with the sink count).
pub async fn run_graph_stateful<'a, Clk: PipelineClock>(
    graph: Graph<GraphNodeRef<'a>>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    state: &StateController,
) -> Result<RunStats, G2gError> {
    run_graph_inner(graph, clock, link_capacity, None, Some(state.clone()), None, None, None).await
}

/// As [`run_graph`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed) to
/// `bus` on a startup negotiation failure. The convenience wrappers' `_with_bus`
/// variants build a borrowing `Graph` and route through here; `run_graph` passes
/// `None`.
/// Result of negotiating and configuring a graph, shared by the cooperative and
/// thread-per-arm drivers: the DAG-wide M12 folds (latency / clock / allocation)
/// plus the solved per-edge caps, computed once before the arms take the elements.
#[derive(Debug)]
struct Prepared {
    solution: Vec<Caps>,
    feasibility: Vec<Option<CapsSet>>,
    latency: LatencyReport,
    allocation: Option<AllocationParams>,
    clock_priority: ClockPriority,
    base_time_ns: u64,
}

/// Phases 1-3.5 of the graph runner: name instances + mint probes, probe source
/// caps, solve + configure the whole DAG, run the allocation cascade, and elect
/// the clock. Mutates `vg` (configure / clock-sync / instance names) and returns
/// the per-node probes plus the folded [`Prepared`] stats. Shared verbatim by
/// [`run_graph_inner`] (cooperative) and [`run_graph_threaded`] (thread-per-arm)
/// so both negotiate identically and differ only in how they run the arms.
async fn prepare_graph<'a>(
    vg: &mut ValidatedGraph<GraphNodeRef<'a>>,
    topo: &[NodeId],
    state: &Option<StateController>,
    bus: Option<&BusHandle>,
    clock: &dyn PipelineClock,
    observer: Option<&Observer>,
) -> Result<(Vec<Probe>, Prepared), G2gError> {
    let n = vg.node_count();
    // M78: tell the controller how many sinks must preroll before the async
    // `Paused` transition completes (aggregated `AsyncDone`).
    if let Some(sc) = state {
        let sinks = topo
            .iter()
            .filter(|&&n| matches!(vg.kind(n), NodeKind::Sink))
            .count();
        sc.expect_prerolls(sinks);
    }

    // M179: give each source / element instance a `<category>N` log name (per
    // category, the GStreamer `videotestsrc0` convention) and log its addition.
    // Done before negotiation so an element's own log lines (e.g. at
    // `configure_pipeline`) already carry the instance name. Naming runs whether
    // or not a sink is installed (it is cheap; the `g2g_info!` is threshold-gated).
    // M399: while naming, mint a measured-latency probe for each interior element
    // (Transform / Sink: the nodes with a `process()`), keyed by its instance name.
    let mut probes: Vec<Probe> = (0..n).map(|_| None).collect();
    // Per-node instance names, captured for the observer tap (empty for unnamed
    // structural tee / muxer nodes). Indexed by `NodeId`, like `probes`.
    let mut names: Vec<alloc::string::String> = alloc::vec![alloc::string::String::new(); n];
    {
        let mut counts: Vec<(&'static str, u32)> = Vec::new();
        for &node in topo {
            let category = match vg.element_mut(node) {
                Some(GraphNodeRef::Source(src)) => src.log_category(),
                Some(GraphNodeRef::Element(elem)) => elem.log_category(),
                _ => continue, // muxer / tee: not named for v1
            };
            let n = match counts.iter_mut().find(|(c, _)| *c == category) {
                Some(e) => {
                    let v = e.1;
                    e.1 += 1;
                    v
                }
                None => {
                    counts.push((category, 1));
                    0
                }
            };
            let name = alloc::format!("{category}{n}");
            names[node.0 as usize] = name.clone();
            match vg.element_mut(node) {
                Some(GraphNodeRef::Source(src)) => src.set_instance_name(name.clone()),
                Some(GraphNodeRef::Element(elem)) => elem.set_instance_name(name.clone()),
                _ => {}
            }
            if matches!(vg.kind(node), NodeKind::Transform | NodeKind::Sink) {
                probes[node.0 as usize] = Some(ElementProbe::new(name.clone()));
            }
            crate::g2g_info!(crate::log::Target::named(category, &name), "added to pipeline");
        }
    }

    // Dev-tooling tap: hand the observer the topology + a clone of every probe
    // `Arc`, so a concurrent task can read live per-element telemetry while the
    // arms run. No-op (and zero cost) when no observer was supplied.
    if let Some(obs) = observer {
        let roles: Vec<crate::runtime::NodeRole> =
            (0..n).map(|i| vg.kind(NodeId(i as u32)).into()).collect();
        let edges: Vec<crate::runtime::EdgeInfo> = vg
            .edges()
            .iter()
            .map(|e| crate::runtime::EdgeInfo { from: e.src.node.0 as usize, to: e.dst.node.0 as usize })
            .collect();
        obs.register(names, roles, probes.clone(), edges);
    }

    // Phase 1: probe each source's caps (async) into an owned map, releasing
    // the mutable borrow before the constraint phase borrows every node.
    let mut source_caps: Vec<Option<Caps>> = (0..n).map(|_| None).collect();
    for &node in topo {
        if matches!(vg.kind(node), NodeKind::Source) {
            let GraphNodeRef::Source(src) = vg.element_mut(node).ok_or(G2gError::CapsMismatch)? else {
                return Err(G2gError::CapsMismatch);
            };
            source_caps[node.0 as usize] = Some(src.intercept_caps().await?);
        }
    }

    // Phase 2: build a per-node constraint and solve the whole DAG, and snapshot
    // each edge's downstream feasibility (D4) for the mid-stream re-solve. The
    // transform/sink constraints borrow their elements immutably (coexisting),
    // so both are computed and the borrows released before configure.
    let (solution, feasibility): (Vec<Caps>, Vec<Option<CapsSet>>) = {
        let constraints = build_node_constraints(vg, &source_caps)?;
        let solution = solve_graph_labeled(vg, &constraints, &|node| caps_label(vg, node))
            .map_err(|f| {
                report_nego_failure(bus, f);
                G2gError::CapsMismatch
            })?;
        let feasibility = graph_downstream_feasibility(vg, &constraints, &solution);
        (solution, feasibility)
    };

    // Phase 3: configure each element with its negotiated caps. Source nodes
    // take their single output edge's caps (no input); transforms and sinks
    // take their input edge's caps.
    for &node in topo {
        match vg.kind(node) {
            NodeKind::Source => {
                let caps = solution[vg.out_edges(node)[0]].clone();
                let GraphNodeRef::Source(src) =
                    vg.element_mut(node).ok_or(G2gError::CapsMismatch)?
                else {
                    return Err(G2gError::CapsMismatch);
                };
                src.configure_pipeline(&caps)?.reject_refixate()?;
            }
            NodeKind::Transform | NodeKind::Sink => {
                let caps = solution[vg.in_edges(node)[0]].clone();
                // A transform also learns its negotiated OUTPUT caps (M185), so a
                // caps-driven transform (videoscale fed by a downstream
                // capsfilter) can take its target from the solve. Sinks have no
                // output edge and skip it.
                let out_caps = vg
                    .out_edges(node)
                    .first()
                    .map(|&eid| solution[eid].clone());
                let GraphNodeRef::Element(elem) =
                    vg.element_mut(node).ok_or(G2gError::CapsMismatch)?
                else {
                    return Err(G2gError::CapsMismatch);
                };
                elem.configure_pipeline(&caps)?.reject_refixate()?;
                if let Some(out_caps) = out_caps {
                    elem.configure_output(&out_caps)?;
                }
            }
            NodeKind::Tee(_) => {
                // A plain (broadcast) tee carries no element. A demux is a
                // tee-shaped node carrying a `MultiOutputElement`, configured
                // with its single input edge's negotiated caps (the byte stream);
                // each branch retypes later via per-output `CapsChanged`.
                if matches!(vg.element(node), Some(GraphNodeRef::Demux(_))) {
                    let caps = solution[vg.in_edges(node)[0]].clone();
                    let GraphNodeRef::Demux(elem) =
                        vg.element_mut(node).ok_or(G2gError::CapsMismatch)?
                    else {
                        return Err(G2gError::CapsMismatch);
                    };
                    elem.configure_pipeline(&caps)?.reject_refixate()?;
                }
            }
            NodeKind::Muxer(_) => {
                // Configure each input pad with its in-edge's negotiated caps.
                let in_edges: Vec<usize> = vg.in_edges(node).to_vec();
                for &eid in &in_edges {
                    let pad = vg.edge(eid).dst.index as usize;
                    let caps = solution[eid].clone();
                    let GraphNodeRef::Muxer(elem) =
                        vg.element_mut(node).ok_or(G2gError::CapsMismatch)?
                    else {
                        return Err(G2gError::CapsMismatch);
                    };
                    elem.configure_pipeline(pad, &caps)?.reject_refixate()?;
                }
            }
        }
    }

    // Phase 3.5: DAG-wide M12 folds, so the runner reports the same latency /
    // clock / allocation the linear runners do (the convenience wrappers reduce
    // to thin builders over this). Done before Phase 4 takes the elements.
    //
    // Allocation cascade in reverse topo order: each element absorbs the
    // proposal arriving on its output edge(s) (`configure_allocation`), then
    // proposes from its output-link caps; the proposal is stored on its input
    // edge(s) for its upstream to absorb. A tee joins its branch proposals
    // (most-restrictive intersection, loud failure on a domain conflict) onto
    // its single input; a muxer proposes its own per-pad demand onto each input
    // edge (the boundary now crosses at startup). The source's absorbed proposal
    // is the reported `allocation`. For a linear chain this is byte-for-byte the
    // linear runner's sink->source fold.
    let nee = vg.edge_count();
    let mut edge_proposal: Vec<Option<AllocationParams>> = (0..nee).map(|_| None).collect();
    let mut allocation: Option<AllocationParams> = None;
    for &node in topo.iter().rev() {
        match vg.kind(node) {
            NodeKind::Sink => {
                let in_e = vg.in_edges(node)[0];
                let caps = solution[in_e].clone();
                edge_proposal[in_e] = element_propose(vg, node, &caps);
            }
            NodeKind::Transform => {
                let in_e = vg.in_edges(node)[0];
                let out_e = vg.out_edges(node)[0];
                // A transform is a memory-domain pass-through here: it forwards the
                // downstream proposal to its own pool and re-proposes upstream
                // unchanged. Domain capability is enforced at the buffer-pool
                // origin (the source) and at the sibling join (the tee), not at
                // every hop, so a GPU proposal merely passing through a plain
                // transform is not rejected against its System default (M351).
                if let Some(p) = edge_proposal[out_e] {
                    element_configure_alloc(vg, node, &p);
                }
                let caps = solution[out_e].clone();
                edge_proposal[in_e] = element_propose(vg, node, &caps);
            }
            NodeKind::Tee(_) => {
                let in_e = vg.in_edges(node)[0];
                let mut joined: Option<AllocationParams> = None;
                for &oe in vg.out_edges(node) {
                    joined = join_alloc(joined, edge_proposal[oe])?;
                }
                edge_proposal[in_e] = joined;
            }
            NodeKind::Source => {
                let out_e = vg.out_edges(node)[0];
                if let Some(p) = edge_proposal[out_e] {
                    // M351: reconcile against the source's emittable domains, the
                    // upstream end of the two-sided negotiation. The reconciled
                    // proposal is what the source allocates and what `RunStats`
                    // reports.
                    let can = node_output_domains(vg, node);
                    let resolved = p.resolve_for_producer(can)?;
                    if let GraphNodeRef::Source(src) = vg.element_mut(node).ok_or(G2gError::CapsMismatch)? {
                        src.configure_allocation(&resolved);
                    }
                    allocation = Some(resolved);
                }
            }
            NodeKind::Muxer(_) => {
                // A muxer asks each input pad for the allocation it wants (most
                // are content-agnostic and propose nothing), storing it on that
                // input edge so the demand crosses the boundary and re-cascades
                // up the branch like any other downstream proposal. The muxer's
                // own output edge proposal is not absorbed here: a container
                // muxer's byte output has no memory-domain tie to its inputs.
                if let Some(GraphNodeRef::Muxer(mux)) = vg.element(node) {
                    for &in_e in vg.in_edges(node) {
                        let pad = vg.edge(in_e).dst.index as usize;
                        let caps = solution[in_e].clone();
                        edge_proposal[in_e] = mux.propose_allocation_for_input(pad, &caps);
                    }
                }
            }
        }
    }

    // Latency fold + clock election over every element node (tee is structural;
    // a muxer contributes neither, like the fan-in runner).
    let mut latencies: Vec<LatencyReport> = Vec::with_capacity(n);
    let mut clocks: Vec<Option<ClockCandidate>> = Vec::with_capacity(n);
    for &node in topo {
        if let Some(l) = element_latency(vg, node) {
            latencies.push(l);
            clocks.push(element_clock(vg, node));
        }
    }
    let latency = LatencyReport::aggregate(latencies);
    let elected = elect_clock(clocks);
    let (clock_priority, base_time_ns) = match &elected {
        Some(c) => (c.priority, c.clock.now_ns()),
        None => (ClockPriority::SystemFallback, clock.now_ns()),
    };

    // Hand the elected clock + base time to every sink so each presents its
    // frames at their running-time deadline (PTS pacing), the same as the linear
    // runners (M169). Only when a clock was elected; without one the sinks present
    // as fast as backpressure allows. A sink node always holds a
    // `GraphNodeRef::Element` (not a `Source`), so the match below covers them.
    if let Some(c) = &elected {
        // M176: under a state controller, arm one Playing-transition anchor
        // (shared across sinks) so each bases presentation on the play edge,
        // not on startup / its preroll frame; without one, the eager base time
        // stands. Armed once outside the loop; the anchor is cheaply cloned.
        let anchor = state.as_ref().map(|sc| sc.arm_play_anchor(c.clock.clone()));
        for &node in topo {
            if matches!(vg.kind(node), NodeKind::Sink) {
                if let Some(GraphNodeRef::Element(elem)) = vg.element_mut(node) {
                    let sync = match &anchor {
                        Some(a) => {
                            ClockSync::with_play_anchor(c.clock.clone(), base_time_ns, a.clone())
                        }
                        None => ClockSync::new(c.clock.clone(), base_time_ns),
                    };
                    elem.set_clock_sync(sync);
                }
            }
        }
    }

    Ok((
        probes,
        Prepared { solution, feasibility, latency, allocation, clock_priority, base_time_ns },
    ))
}

/// Per-edge bounded channels plus the D4 β re-cascade coordinator, shared by the
/// cooperative and thread-per-arm drivers. Returns the edge sender/receiver
/// slots (taken by the arm loop), the shared leaky-link drop counter, the
/// per-transform `ArmDirective` receivers, and the coordinator + its handle.
struct GraphChannels {
    txs: Vec<Option<LinkSender>>,
    rxs: Vec<Option<LinkReceiver>>,
    dropped: alloc::sync::Arc<spin::Mutex<u64>>,
    arm_ctrl_rx: Vec<Option<Receiver<ArmDirective>>>,
    coord_handle: GraphCoordHandle,
    coordinator: GraphCoordinator,
}

fn build_channels<'a>(
    vg: &ValidatedGraph<GraphNodeRef<'a>>,
    topo: &[NodeId],
    link_capacity: usize,
) -> GraphChannels {
    let n = vg.node_count();
    // Phase 4: one bounded channel per edge, then one arm per node. Each arm
    // takes the senders of its outgoing edges and the receivers of its
    // incoming edges (a tee holds n senders, a sink one receiver, etc.).
    let ne = vg.edge_count();
    let mut txs: Vec<Option<LinkSender>> = Vec::with_capacity(ne);
    let mut rxs: Vec<Option<LinkReceiver>> = Vec::with_capacity(ne);
    // Shared drop counter: leaky links (`LinkPolicy::DropOldest`/`DropNewest`)
    // increment it per dropped frame, so the total surfaces in `RunStats`.
    let dropped = alloc::sync::Arc::new(spin::Mutex::new(0u64));
    for eid in 0..ne {
        // A per-edge depth (a `queue max-size-buffers=N`) overrides the graph-wide
        // default; most edges leave it `None` and take `link_capacity`.
        let (mut tx, rx) = link(vg.edge(eid).capacity.unwrap_or(link_capacity));
        let policy = vg.edge(eid).policy;
        tx.set_policy(policy);
        if policy != crate::link::LinkPolicy::Block {
            tx.set_drop_counter(dropped.clone());
        }
        txs.push(Some(tx));
        rxs.push(Some(rx));
    }

    // D4 β coordinator: one `ArmDirective` channel per transform arm (the
    // interruptible interior arms), plus the upstream-arm adjacency the
    // coordinator walks. Sinks and transforms hold a clone of the report handle;
    // when every such arm finishes (EOS-driven), the handles drop and the
    // coordinator ends.
    let mut arm_ctrl: Vec<Option<Sender<ArmDirective>>> = (0..n).map(|_| None).collect();
    let mut arm_ctrl_rx: Vec<Option<Receiver<ArmDirective>>> = (0..n).map(|_| None).collect();
    let mut upstream_arms: Vec<Vec<NodeId>> = (0..n).map(|_| Vec::new()).collect();
    for &node in topo {
        if matches!(vg.kind(node), NodeKind::Transform) {
            let (ctx, crx) = bounded::<ArmDirective>(link_capacity);
            arm_ctrl[node.0 as usize] = Some(ctx);
            arm_ctrl_rx[node.0 as usize] = Some(crx);
        }
    }
    // β reporters are transforms (after a directive) and sinks (on caps change);
    // each forwards to the nearest interior arm feeding its inputs.
    for &node in topo {
        if matches!(vg.kind(node), NodeKind::Transform | NodeKind::Sink) {
            let mut ups: Vec<NodeId> = Vec::new();
            for &ie in vg.in_edges(node) {
                if let Some(u) = nearest_upstream_arm(vg, ie) {
                    if !ups.contains(&u) {
                        ups.push(u);
                    }
                }
            }
            upstream_arms[node.0 as usize] = ups;
        }
    }
    let (coord_tx, coord_rx) = bounded::<Recascade>(link_capacity);
    let coord_handle = GraphCoordHandle { tx: coord_tx };
    let coordinator = GraphCoordinator { rx: coord_rx, arm_ctrl, upstream_arms };

    GraphChannels { txs, rxs, dropped, arm_ctrl_rx, coord_handle, coordinator }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_graph_inner<'a, Clk: PipelineClock>(
    graph: Graph<GraphNodeRef<'a>>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
    state: Option<StateController>,
    progress: Option<&PipelineProgress>,
    copy_policy: Option<crate::copyplan::CopyPolicy>,
    observer: Option<&Observer>,
) -> Result<RunStats, G2gError> {
    let link_capacity: usize = link_capacity.into().get();
    let mut vg = graph.finish().map_err(|_| G2gError::CapsMismatch)?;
    let n = vg.node_count();
    if n < 2 {
        return Err(G2gError::CapsMismatch);
    }
    let topo = vg.topo().to_vec();

    let (probes, Prepared { solution, feasibility, latency, allocation, clock_priority, base_time_ns }) =
        prepare_graph(&mut vg, &topo, &state, bus, clock, observer).await?;

    // Enforce the memory-domain copy budget (M617) before any frame flows: the graph
    // is negotiated, so the per-edge domains are known and the copy plan is exact. A
    // zero-copy pipeline (`CopyPolicy::DenyAll`) refuses to start rather than silently
    // paying a host round-trip at runtime. Configure side effects already ran (bound
    // sockets etc.) and are released when the doomed graph drops.
    if let Some(policy) = copy_policy {
        let edge_memory: Vec<crate::memory::MemoryDomainKind> = (0..vg.edge_count())
            .map(|id| {
                let src = vg.edge(id).src.node;
                vg.element(src)
                    .map(|e| e.output_memory())
                    .unwrap_or(crate::memory::MemoryDomainKind::System)
            })
            .collect();
        let plan = copy_plan(&vg, &solution, &edge_memory);
        if plan.check(policy).is_err() {
            return Err(G2gError::CopyBudget);
        }
    }

    let GraphChannels { mut txs, mut rxs, dropped, mut arm_ctrl_rx, coord_handle, coordinator } =
        build_channels(&vg, &topo, link_capacity);

    let mut arms: Vec<BoxFuture<'a, Result<u64, G2gError>>> = Vec::with_capacity(n + 1);
    let mut arm_kinds: Vec<NodeKind> = Vec::with_capacity(n);

    for &node in &topo {
        let kind = vg.kind(node);
        let in_e: Vec<usize> = vg.in_edges(node).to_vec();
        let out_e: Vec<usize> = vg.out_edges(node).to_vec();
        let mut in_rxs: Vec<LinkReceiver> =
            in_e.iter().map(|&e| rxs[e].take().expect("edge rx present")).collect();
        let mut out_txs: Vec<LinkSender> =
            out_e.iter().map(|&e| txs[e].take().expect("edge tx present")).collect();
        let element = vg.take_element(node);

        // A muxer contributes N+1 arms (one forwarder per input pad plus the
        // muxer arm), so it is built before the single-arm match below.
        if let NodeKind::Muxer(_) = kind {
            let Some(GraphNodeRef::Muxer(mux)) = element else {
                return Err(G2gError::CapsMismatch);
            };
            let out_tx = out_txs.pop().expect("muxer output edge");
            let mux_out_caps = solution[out_e[0]].clone();
            let pads: Vec<usize> =
                in_e.iter().map(|&eid| vg.edge(eid).dst.index as usize).collect();
            // The interior arm feeding each input pad (resolved through structural
            // tees; `None` when a source feeds the pad directly, which has no
            // mid-stream re-cascade channel). Aligned with `pads` / `pad_rxs`, so
            // the muxer arm can re-cascade a per-pad β proposal up just that
            // branch. `in_e` order matches `in_rxs`, so slot k maps to pad k here.
            let pad_upstream: Vec<Option<NodeId>> =
                in_e.iter().map(|&eid| nearest_upstream_arm(&vg, eid)).collect();
            let input_count = in_rxs.len();
            // Each input pad gets its OWN bounded channel feeding the muxer arm,
            // which drains them round-robin. A single shared FIFO would let a
            // fast input (e.g. a free-running background) monopolize the queue
            // and starve a slower real-time input (e.g. a 30 fps camera): the
            // camera's overlay would freeze and, worse, its EOS would never
            // arrive, hanging the all-inputs-EOS aggregation forever. Forwarders
            // are still indexed by pad so a muxer's `process(pad, ..)` keeps its
            // per-input geometry straight even if pads link out of order.
            let mut pad_rxs: Vec<(usize, Receiver<PipelinePacket>)> =
                Vec::with_capacity(input_count);
            for (in_rx, pad) in in_rxs.into_iter().zip(pads) {
                let (pad_tx, pad_rx) = bounded::<PipelinePacket>(link_capacity);
                let fwd: BoxFuture<'a, Result<u64, G2gError>> =
                    Box::pin(muxer_forwarder(in_rx, pad_tx));
                arms.push(fwd);
                arm_kinds.push(kind);
                pad_rxs.push((pad, pad_rx));
            }
            // A muxer can opt into runner-level PTS-ordered delivery (the runner
            // merges its inputs by DataFrame PTS); the default drains round-robin
            // in arrival order.
            let arm: BoxFuture<'a, Result<u64, G2gError>> = if mux.input_pts_ordered() {
                Box::pin(muxer_arm_pts(
                    mux,
                    pad_rxs,
                    out_tx,
                    input_count,
                    mux_out_caps,
                    coord_handle.clone(),
                    node,
                    pad_upstream,
                ))
            } else {
                Box::pin(muxer_arm(
                    mux,
                    pad_rxs,
                    out_tx,
                    input_count,
                    mux_out_caps,
                    coord_handle.clone(),
                    node,
                    pad_upstream,
                ))
            };
            arms.push(arm);
            arm_kinds.push(kind);
            continue;
        }

        let arm: BoxFuture<'a, Result<u64, G2gError>> = match kind {
            NodeKind::Source => {
                let Some(GraphNodeRef::Source(src)) = element else {
                    return Err(G2gError::CapsMismatch);
                };
                let out_tx = out_txs.pop().expect("source output edge");
                Box::pin(source_arm(src, out_tx, bus.cloned(), progress.cloned()))
            }
            NodeKind::Transform => {
                let Some(GraphNodeRef::Element(elem)) = element else {
                    return Err(G2gError::CapsMismatch);
                };
                let in_rx = in_rxs.pop().expect("transform input edge");
                let out_tx = out_txs.pop().expect("transform output edge");
                let out_edge = out_e[0];
                let arm_rx = arm_ctrl_rx[node.0 as usize].take().expect("transform ctrl rx");
                let out_caps = solution[out_edge].clone();
                let downstream_feasible = feasibility[out_edge].clone();
                Box::pin(transform_arm(
                    elem,
                    in_rx,
                    out_tx,
                    arm_rx,
                    coord_handle.clone(),
                    node,
                    out_caps,
                    downstream_feasible,
                    branch_mode(&vg, node),
                    bus.cloned(),
                    probes[node.0 as usize].clone(),
                ))
            }
            NodeKind::Sink => {
                let Some(GraphNodeRef::Element(elem)) = element else {
                    return Err(G2gError::CapsMismatch);
                };
                let in_rx = in_rxs.pop().expect("sink input edge");
                Box::pin(sink_arm(
                    elem,
                    in_rx,
                    coord_handle.clone(),
                    node,
                    branch_mode(&vg, node),
                    bus.cloned(),
                    state.clone(),
                    progress.cloned(),
                    probes[node.0 as usize].clone(),
                ))
            }
            NodeKind::Tee(_) => {
                let in_rx = in_rxs.pop().expect("tee input edge");
                // A tee-shaped node carrying a demux element routes per-output;
                // a plain tee broadcasts. Under `AllowBranchDrop` the broadcast
                // tolerates a branch that has dropped out (closed its channel).
                let branch_drop = vg.fanout_policy(node) == FanOutPolicy::AllowBranchDrop;
                match element {
                    Some(GraphNodeRef::Demux(demux)) => Box::pin(demux_arm(demux, in_rx, out_txs)),
                    _ => Box::pin(tee_arm(in_rx, out_txs, branch_drop)),
                }
            }
            NodeKind::Muxer(_) => unreachable!("muxer handled above"),
        };
        arms.push(arm);
        arm_kinds.push(kind);
    }

    // Drop the template handle so the coordinator can end once every arm's
    // clone drops; append the coordinator as the final arm. Its result is the
    // count of re-cascade events it observed.
    drop(coord_handle);
    let coord_arm_index = arms.len();
    arms.push(Box::pin(async move { Ok(coordinator.run().await) }));

    let results = join_all(arms).await;
    fold_run_stats(
        results,
        &arm_kinds,
        coord_arm_index,
        &dropped,
        &probes,
        latency,
        allocation,
        clock_priority,
        base_time_ns,
    )
}

/// Fold each arm's per-node frame count into the final [`RunStats`], shared by
/// the cooperative ([`run_graph_inner`]) and thread-per-arm
/// ([`run_graph_threaded`]) drivers, which differ only in how the arms are run
/// (one executor vs one OS thread each), not in how their results aggregate.
/// `results` is in arm order (source / transform / sink / muxer arms, then the
/// coordinator last at `coord_arm_index`); `arm_kinds` labels every arm except
/// the coordinator.
#[allow(clippy::too_many_arguments)]
fn fold_run_stats(
    results: Vec<Result<u64, G2gError>>,
    arm_kinds: &[NodeKind],
    coord_arm_index: usize,
    dropped: &alloc::sync::Arc<spin::Mutex<u64>>,
    probes: &[Probe],
    latency: LatencyReport,
    allocation: Option<AllocationParams>,
    clock_priority: ClockPriority,
    base_time_ns: u64,
) -> Result<RunStats, G2gError> {
    // M81: surface a substantive arm error over a secondary `Shutdown` (a real
    // error in one node closes links, which surfaces as `Shutdown` on the
    // others; reporting the first-in-topo-order error would often mask the
    // cause).
    if let Some(e) = results
        .iter()
        .filter_map(|r| r.as_ref().err())
        .find(|e| **e != G2gError::Shutdown)
        .or_else(|| results.iter().filter_map(|r| r.as_ref().err()).next())
    {
        return Err(e.clone());
    }
    let mut counts = Vec::with_capacity(results.len());
    for r in results {
        counts.push(r?);
    }
    let coordinator_events = counts[coord_arm_index];
    let mut emitted = 0u64;
    let mut consumed = 0u64;
    for (kind, &count) in arm_kinds.iter().zip(counts.iter()) {
        match kind {
            NodeKind::Source => emitted += count,
            NodeKind::Sink => consumed += count,
            _ => {}
        }
    }

    let frames_dropped = *dropped.lock();
    // M399: every arm has joined, so its probe is no longer being written; snapshot
    // each interior element's measured latency + fill into the report.
    let per_element = snapshot_all(probes);
    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        frames_dropped,
        latency,
        allocation,
        clock_priority,
        base_time_ns,
        coordinator_events,
        per_element,
    })
}

/// A graph arm's future, built and driven entirely on one worker thread. It is
/// deliberately **not** `Send`: the thread-per-arm runner never moves a future
/// between threads (only the element + channels, all `Send`, cross at setup), so
/// an element whose future is `!Send` (a hardware decoder holding a raw context)
/// runs unchanged, exactly as under the cooperative runner.
#[cfg(all(feature = "std", feature = "multi-thread"))]
pub type LocalArmFuture =
    core::pin::Pin<alloc::boxed::Box<dyn core::future::Future<Output = Result<u64, G2gError>>>>;

/// Executor abstraction for [`run_graph_threaded`]. The runner hands each graph
/// node's arm to `spawn_arm` as a `Send` builder closure; the spawner runs the
/// builder on a dedicated worker thread and drives the [`LocalArmFuture`] it
/// returns to completion there, resolving the returned handle with the arm's
/// frame count (or its error). This is the GStreamer streaming-thread model: one
/// OS thread per element, so CPU-bound stages (software decode/encode) overlap
/// across cores instead of serialising on one cooperative executor.
///
/// `g2g-core` stays executor-agnostic; `g2g-plugins` supplies a tokio-backed
/// `ThreadSpawner`. Only the element and its channels (all `Send`) cross the
/// thread boundary; the future stays put, so elements need no `Send` future.
#[cfg(all(feature = "std", feature = "multi-thread"))]
pub trait GraphSpawner {
    /// Run `build` on a worker thread and drive its future there; the returned
    /// handle resolves (on the caller thread) once that arm finishes.
    fn spawn_arm(
        &self,
        build: alloc::boxed::Box<dyn FnOnce() -> LocalArmFuture + Send>,
    ) -> BoxFuture<'static, Result<u64, G2gError>>;
}

/// Thread-per-arm sibling of [`run_graph_inner`]: negotiates the graph
/// identically (shared [`prepare_graph`] / [`build_channels`] / [`fold_run_stats`]),
/// then hands each arm to `spawner` to run on its own OS thread rather than
/// cooperatively multiplexing them on the caller's executor. The graph must own
/// its elements (`Graph<GraphNode>`, i.e. `'static`) so each arm can move its
/// element onto a worker thread. All `vg`-borrowing / reference-typed inputs are
/// resolved to owned values *before* each arm's builder closure, since the
/// closure runs on another thread and may not borrow `vg`, `bus`, or `progress`.
#[cfg(all(feature = "std", feature = "multi-thread"))]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_graph_threaded_inner<S: GraphSpawner>(
    graph: Graph<GraphNode>,
    clock: &dyn PipelineClock,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
    state: Option<StateController>,
    progress: Option<&PipelineProgress>,
    spawner: &S,
) -> Result<RunStats, G2gError> {
    let link_capacity: usize = link_capacity.into().get();
    let mut vg = graph.finish().map_err(|_| G2gError::CapsMismatch)?;
    let n = vg.node_count();
    if n < 2 {
        return Err(G2gError::CapsMismatch);
    }
    let topo = vg.topo().to_vec();

    let (probes, Prepared { solution, feasibility, latency, allocation, clock_priority, base_time_ns }) =
        prepare_graph(&mut vg, &topo, &state, bus, clock, None).await?;

    let GraphChannels { mut txs, mut rxs, dropped, mut arm_ctrl_rx, coord_handle, coordinator } =
        build_channels(&vg, &topo, link_capacity);

    // One `spawn_arm` handle per arm (mirrors the cooperative `arms` vec). Each
    // handle resolves on this thread once its worker thread finishes.
    let mut handles: Vec<BoxFuture<'static, Result<u64, G2gError>>> = Vec::with_capacity(n + 1);
    let mut arm_kinds: Vec<NodeKind> = Vec::with_capacity(n);

    for &node in &topo {
        let kind = vg.kind(node);
        let in_e: Vec<usize> = vg.in_edges(node).to_vec();
        let out_e: Vec<usize> = vg.out_edges(node).to_vec();
        let mut in_rxs: Vec<LinkReceiver> =
            in_e.iter().map(|&e| rxs[e].take().expect("edge rx present")).collect();
        let mut out_txs: Vec<LinkSender> =
            out_e.iter().map(|&e| txs[e].take().expect("edge tx present")).collect();
        let element = vg.take_element(node);

        if let NodeKind::Muxer(_) = kind {
            let Some(GraphNodeRef::Muxer(mux)) = element else {
                return Err(G2gError::CapsMismatch);
            };
            let out_tx = out_txs.pop().expect("muxer output edge");
            let mux_out_caps = solution[out_e[0]].clone();
            let pads: Vec<usize> =
                in_e.iter().map(|&eid| vg.edge(eid).dst.index as usize).collect();
            let pad_upstream: Vec<Option<NodeId>> =
                in_e.iter().map(|&eid| nearest_upstream_arm(&vg, eid)).collect();
            let input_count = in_rxs.len();
            let mut pad_rxs: Vec<(usize, Receiver<PipelinePacket>)> =
                Vec::with_capacity(input_count);
            for (in_rx, pad) in in_rxs.into_iter().zip(pads) {
                let (pad_tx, pad_rx) = bounded::<PipelinePacket>(link_capacity);
                let build: alloc::boxed::Box<dyn FnOnce() -> LocalArmFuture + Send> =
                    alloc::boxed::Box::new(move || -> LocalArmFuture {
                        Box::pin(muxer_forwarder(in_rx, pad_tx))
                    });
                handles.push(spawner.spawn_arm(build));
                arm_kinds.push(kind);
                pad_rxs.push((pad, pad_rx));
            }
            let pts_ordered = mux.input_pts_ordered();
            let ch = coord_handle.clone();
            let build: alloc::boxed::Box<dyn FnOnce() -> LocalArmFuture + Send> = if pts_ordered {
                alloc::boxed::Box::new(move || -> LocalArmFuture {
                    Box::pin(muxer_arm_pts(
                        mux, pad_rxs, out_tx, input_count, mux_out_caps, ch, node, pad_upstream,
                    ))
                })
            } else {
                alloc::boxed::Box::new(move || -> LocalArmFuture {
                    Box::pin(muxer_arm(
                        mux, pad_rxs, out_tx, input_count, mux_out_caps, ch, node, pad_upstream,
                    ))
                })
            };
            handles.push(spawner.spawn_arm(build));
            arm_kinds.push(kind);
            continue;
        }

        let build: alloc::boxed::Box<dyn FnOnce() -> LocalArmFuture + Send> = match kind {
            NodeKind::Source => {
                let Some(GraphNodeRef::Source(src)) = element else {
                    return Err(G2gError::CapsMismatch);
                };
                let out_tx = out_txs.pop().expect("source output edge");
                let bus_c = bus.cloned();
                let prog_c = progress.cloned();
                alloc::boxed::Box::new(move || -> LocalArmFuture {
                    Box::pin(source_arm(src, out_tx, bus_c, prog_c))
                })
            }
            NodeKind::Transform => {
                let Some(GraphNodeRef::Element(elem)) = element else {
                    return Err(G2gError::CapsMismatch);
                };
                let in_rx = in_rxs.pop().expect("transform input edge");
                let out_tx = out_txs.pop().expect("transform output edge");
                let out_edge = out_e[0];
                let arm_rx = arm_ctrl_rx[node.0 as usize].take().expect("transform ctrl rx");
                let out_caps = solution[out_edge].clone();
                let downstream_feasible = feasibility[out_edge].clone();
                let bm = branch_mode(&vg, node);
                let bus_c = bus.cloned();
                let probe = probes[node.0 as usize].clone();
                let ch = coord_handle.clone();
                alloc::boxed::Box::new(move || -> LocalArmFuture {
                    Box::pin(transform_arm(
                        elem, in_rx, out_tx, arm_rx, ch, node, out_caps, downstream_feasible, bm,
                        bus_c, probe,
                    ))
                })
            }
            NodeKind::Sink => {
                let Some(GraphNodeRef::Element(elem)) = element else {
                    return Err(G2gError::CapsMismatch);
                };
                let in_rx = in_rxs.pop().expect("sink input edge");
                let bm = branch_mode(&vg, node);
                let bus_c = bus.cloned();
                let state_c = state.clone();
                let prog_c = progress.cloned();
                let probe = probes[node.0 as usize].clone();
                let ch = coord_handle.clone();
                alloc::boxed::Box::new(move || -> LocalArmFuture {
                    Box::pin(sink_arm(elem, in_rx, ch, node, bm, bus_c, state_c, prog_c, probe))
                })
            }
            NodeKind::Tee(_) => {
                let in_rx = in_rxs.pop().expect("tee input edge");
                let branch_drop = vg.fanout_policy(node) == FanOutPolicy::AllowBranchDrop;
                match element {
                    Some(GraphNodeRef::Demux(demux)) => {
                        alloc::boxed::Box::new(move || -> LocalArmFuture {
                            Box::pin(demux_arm(demux, in_rx, out_txs))
                        })
                    }
                    _ => alloc::boxed::Box::new(move || -> LocalArmFuture {
                        Box::pin(tee_arm(in_rx, out_txs, branch_drop))
                    }),
                }
            }
            NodeKind::Muxer(_) => unreachable!("muxer handled above"),
        };
        handles.push(spawner.spawn_arm(build));
        arm_kinds.push(kind);
    }

    // Drop the template handle so the coordinator ends once every arm's clone
    // drops; append the coordinator as the final arm on its own thread.
    drop(coord_handle);
    let coord_arm_index = handles.len();
    handles.push(spawner.spawn_arm(alloc::boxed::Box::new(move || -> LocalArmFuture {
        Box::pin(async move { Ok(coordinator.run().await) })
    })));

    let results = join_all(handles).await;
    fold_run_stats(
        results,
        &arm_kinds,
        coord_arm_index,
        &dropped,
        &probes,
        latency,
        allocation,
        clock_priority,
        base_time_ns,
    )
}

/// Run a DAG with one OS thread per arm via `spawner` (opt-in multicore; the
/// GStreamer streaming-thread model). Cooperative [`run_graph`] stays the default
/// for lowest latency and the `no_std` / wasm executors; this trades a per-stage
/// thread handoff for CPU-bound stages overlapping across cores.
#[cfg(all(feature = "std", feature = "multi-thread"))]
pub async fn run_graph_threaded<Clk: PipelineClock, S: GraphSpawner>(
    graph: Graph<GraphNode>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    spawner: &S,
) -> Result<RunStats, G2gError> {
    run_graph_threaded_inner(graph, clock, link_capacity, None, None, None, spawner).await
}

/// As [`run_graph_threaded`], but publishes playback progress (the thread-per-arm
/// analog of [`run_graph_with_progress`]).
#[cfg(all(feature = "std", feature = "multi-thread"))]
pub async fn run_graph_threaded_with_progress<Clk: PipelineClock, S: GraphSpawner>(
    graph: Graph<GraphNode>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    progress: &PipelineProgress,
    spawner: &S,
) -> Result<RunStats, G2gError> {
    run_graph_threaded_inner(graph, clock, link_capacity, None, None, Some(progress), spawner).await
}

/// Zero-dependency [`GraphSpawner`]: each arm runs on its own `std` thread driven
/// by the park-based [`block_on`](crate::runtime::block_on). Dependency-free and
/// sufficient for graphs of pure-core elements (core channels + clock). Elements
/// that need a tokio reactor (network sources) require a tokio-backed spawner
/// instead (`g2g-plugins`' `TokioThreadSpawner`), since `block_on` provides no
/// I/O driver.
#[cfg(all(feature = "std", feature = "multi-thread"))]
#[derive(Debug, Default, Clone, Copy)]
pub struct ThreadSpawner;

#[cfg(all(feature = "std", feature = "multi-thread"))]
impl GraphSpawner for ThreadSpawner {
    fn spawn_arm(
        &self,
        build: alloc::boxed::Box<dyn FnOnce() -> LocalArmFuture + Send>,
    ) -> BoxFuture<'static, Result<u64, G2gError>> {
        // A capacity-1 channel is the handle: the worker delivers its one result,
        // the caller awaits it. Cross-thread wake is via the channel's waker.
        let (tx, rx) = crate::runtime::channel::bounded::<Result<u64, G2gError>>(1);
        std::thread::spawn(move || {
            let result = crate::runtime::block_on(build());
            // Best-effort: a dropped handle (the join aborted after a sibling's
            // error) just discards the result.
            let _ = tx.try_send(result);
        });
        Box::pin(async move { rx.recv().await.unwrap_or(Err(G2gError::Shutdown)) })
    }
}

/// View a node's payload as a transform/sink element. `None` for a source or a
/// muxer (whose constraints the runner builds from their own trait methods).
fn element_ref<'g, 'a>(
    vg: &'g ValidatedGraph<GraphNodeRef<'a>>,
    node: NodeId,
) -> Option<&'g (dyn DynAsyncElement + 'a)> {
    match vg.element(node)? {
        GraphNodeRef::Element(elem) => Some(&**elem),
        GraphNodeRef::Source(_) | GraphNodeRef::Muxer(_) | GraphNodeRef::Demux(_) => None,
    }
}

/// Build the per-node solver constraints for a validated graph, given each
/// source's probed caps (indexed by node id, `None` for non-sources). The
/// constraints borrow their elements immutably, so the returned vec must be
/// dropped before any `&mut` borrow (configure). Shared by the runner's Phase 2
/// and the negotiate-only tooling path ([`negotiate_graph`]).
fn build_node_constraints<'g, 'a>(
    vg: &'g ValidatedGraph<GraphNodeRef<'a>>,
    source_caps: &[Option<Caps>],
) -> Result<Vec<NodeConstraint<'g>>, G2gError> {
    let mut constraints: Vec<NodeConstraint<'g>> = Vec::with_capacity(vg.node_count());
    for (i, src_caps) in source_caps.iter().enumerate() {
        let node = NodeId(i as u32);
        let nc = match vg.kind(node) {
            NodeKind::Source => {
                let caps = src_caps.clone().ok_or(G2gError::CapsMismatch)?;
                NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(caps)))
            }
            NodeKind::Transform => {
                let elem = element_ref(vg, node).ok_or(G2gError::CapsMismatch)?;
                NodeConstraint::Element(elem.caps_constraint_as_transform())
            }
            NodeKind::Sink => {
                let elem = element_ref(vg, node).ok_or(G2gError::CapsMismatch)?;
                NodeConstraint::Element(elem.caps_constraint_as_sink())
            }
            // A plain (broadcast) tee is structural; the solver couples its
            // branches via `IdentityAny`. A demux that declares per-port caps
            // (M380) instead negotiates each branch against its port: build a
            // `Demux` constraint so a downstream decoder configures at startup.
            NodeKind::Tee(n) => {
                let ports: Vec<Option<Caps>> = match vg.element(node) {
                    Some(GraphNodeRef::Demux(elem)) => {
                        (0..n as usize).map(|p| elem.port_output_caps(p)).collect()
                    }
                    _ => Vec::new(),
                };
                if !ports.is_empty() && ports.iter().all(Option::is_some) {
                    let input = match vg.element(node) {
                        Some(GraphNodeRef::Demux(elem)) => elem.caps_constraint_as_input(),
                        _ => CapsConstraint::AcceptsAny,
                    };
                    let ports = ports
                        .into_iter()
                        .map(|c| CapsConstraint::Produces(CapsSet::one(c.expect("all Some"))))
                        .collect();
                    NodeConstraint::Demux { input, ports }
                } else {
                    NodeConstraint::Element(CapsConstraint::IdentityAny)
                }
            }
            NodeKind::Muxer(_) => {
                let GraphNodeRef::Muxer(elem) = vg.element(node).ok_or(G2gError::CapsMismatch)?
                else {
                    return Err(G2gError::CapsMismatch);
                };
                let inputs: Vec<CapsConstraint<'g>> = (0..elem.input_count())
                    .map(|pad| elem.caps_constraint_as_input(pad))
                    .collect();
                let follows = elem.output_follows_input();
                // An identity-passthrough mux derives its output from a pad, so it
                // need not (and may be unable to) declare output caps up front;
                // only ask for them in the independent-output case.
                let output = match follows {
                    Some(_) => CapsConstraint::AcceptsAny,
                    None => elem.caps_constraint_for_output().map_err(|_| G2gError::CapsMismatch)?,
                };
                NodeConstraint::Muxer { inputs, output, follows }
            }
        };
        constraints.push(nc);
    }
    Ok(constraints)
}

/// A node's label for the caps explainer and the DOT dump: the element's log
/// category (e.g. `h264parse`), falling back to the structural kind for a tee
/// (which carries no element).
fn caps_label(vg: &ValidatedGraph<GraphNodeRef<'_>>, node: NodeId) -> alloc::string::String {
    match vg.element(node) {
        Some(e) => e.log_category().to_string(),
        None => match vg.kind(node) {
            NodeKind::Tee(_) => "tee".to_string(),
            k => alloc::format!("{k:?}"),
        },
    }
}

/// Run startup caps negotiation only, without running the pipeline: validate the
/// graph, probe each source's caps (Phase 1, async), and solve the whole-graph
/// CSP (Phase 2), returning the validated graph, the fixated caps per edge, and
/// each edge's memory domain (all indexed by edge id, as
/// [`crate::dot::DotAnnotations`] expects). The per-edge domain is the producing
/// node's [`output_memory`](GraphNodeRef::output_memory) (M285), so a GPU /
/// zero-copy link shows up in the dump. For tooling that wants the *chosen* caps
/// without moving data, e.g. `g2g-launch --dot`. It performs the same
/// source-caps probing the runner does, so a source that connects on
/// `intercept_caps` (a live ingress) will do so here too; a negotiation failure
/// returns `CapsMismatch` (the caller can fall back to a topology-only dump).
pub async fn negotiate_graph<'a>(
    graph: Graph<GraphNodeRef<'a>>,
) -> Result<
    (ValidatedGraph<GraphNodeRef<'a>>, Vec<Caps>, Vec<crate::memory::MemoryDomainKind>),
    G2gError,
> {
    negotiate_graph_explained(graph).await.map_err(|e| match e {
        NegotiateError::Setup(err) => err,
        NegotiateError::Solve(_) => G2gError::CapsMismatch,
    })
}

/// Why [`negotiate_graph_explained`] could not negotiate a graph. `Setup` is a
/// structural / I/O failure before the solve (too few nodes, a bad source, a
/// source caps-probe error); `Solve` carries the structured
/// [`NegotiationFailure`] naming the conflicting link, which the opaque
/// [`negotiate_graph`] flattens to `CapsMismatch`.
#[derive(Debug)]
pub enum NegotiateError {
    Setup(G2gError),
    Solve(NegotiationFailure),
}

/// As [`negotiate_graph`], but preserves the structured [`NegotiationFailure`]
/// on a solve conflict (for the caps-negotiation explainer / `validate`
/// tooling). `negotiate_graph` is the opaque wrapper over this.
pub async fn negotiate_graph_explained<'a>(
    graph: Graph<GraphNodeRef<'a>>,
) -> Result<
    (ValidatedGraph<GraphNodeRef<'a>>, Vec<Caps>, Vec<crate::memory::MemoryDomainKind>),
    NegotiateError,
> {
    let mut vg = graph.finish().map_err(|_| NegotiateError::Setup(G2gError::CapsMismatch))?;
    let n = vg.node_count();
    if n < 2 {
        return Err(NegotiateError::Setup(G2gError::CapsMismatch));
    }
    let topo = vg.topo().to_vec();

    // Phase 1: probe each source's caps (async), releasing the mutable borrow
    // before the constraint phase borrows every node immutably.
    let mut source_caps: Vec<Option<Caps>> = (0..n).map(|_| None).collect();
    for &node in &topo {
        if matches!(vg.kind(node), NodeKind::Source) {
            let GraphNodeRef::Source(src) = vg
                .element_mut(node)
                .ok_or(NegotiateError::Setup(G2gError::CapsMismatch))?
            else {
                return Err(NegotiateError::Setup(G2gError::CapsMismatch));
            };
            source_caps[node.0 as usize] =
                Some(src.intercept_caps().await.map_err(NegotiateError::Setup)?);
        }
    }

    // Phase 2: build constraints and solve. Scope the immutable borrow so `vg`
    // moves out cleanly in the return.
    let solution = {
        let constraints =
            build_node_constraints(&vg, &source_caps).map_err(NegotiateError::Setup)?;
        solve_graph_labeled(&vg, &constraints, &|node| caps_label(&vg, node))
            .map_err(NegotiateError::Solve)?
    };

    // Per-edge memory domain: the domain of the node producing onto that edge.
    let edge_memory: Vec<crate::memory::MemoryDomainKind> = (0..vg.edge_count())
        .map(|id| {
            let src = vg.edge(id).src.node;
            vg.element(src)
                .map(|e| e.output_memory())
                .unwrap_or(crate::memory::MemoryDomainKind::System)
        })
        .collect();

    Ok((vg, solution, edge_memory))
}

/// Build the [`CopyPlan`](crate::copyplan::CopyPlan) for a negotiated graph from the
/// three arrays [`negotiate_graph`] returns (the validated graph, per-edge fixated
/// caps, and per-edge memory domain). Extracts each node's label + output domain and
/// each edge's producer/consumer/domain/caps into the flat profiles the pure
/// analysis works over, so tooling (e.g. `g2g-launch --copy-plan`) or a graph-level
/// copy budget can inspect the memory-domain path before running.
pub fn copy_plan(
    vg: &ValidatedGraph<GraphNodeRef<'_>>,
    edge_caps: &[Caps],
    edge_memory: &[crate::memory::MemoryDomainKind],
) -> crate::copyplan::CopyPlan {
    let nodes: Vec<crate::copyplan::NodeProfile> = (0..vg.node_count())
        .map(|i| {
            let node = NodeId(i as u32);
            crate::copyplan::NodeProfile {
                label: caps_label(vg, node),
                out_domain: vg
                    .element(node)
                    .map(|e| e.output_memory())
                    .unwrap_or(crate::memory::MemoryDomainKind::System),
            }
        })
        .collect();
    let edges: Vec<crate::copyplan::EdgeProfile> = (0..vg.edge_count())
        .map(|id| {
            let e = vg.edge(id);
            crate::copyplan::EdgeProfile {
                src: e.src.node.0 as usize,
                dst: e.dst.node.0 as usize,
                domain: edge_memory[id],
                caps: edge_caps[id].clone(),
            }
        })
        .collect();
    crate::copyplan::CopyPlan::analyze(&nodes, &edges)
}

/// A transform/sink node's allocation proposal from `caps` (its output-link caps
/// for a transform, its input-link caps for a sink). `None` for other kinds.
fn element_propose(
    vg: &ValidatedGraph<GraphNodeRef<'_>>,
    node: NodeId,
    caps: &Caps,
) -> Option<AllocationParams> {
    match vg.element(node) {
        Some(GraphNodeRef::Element(elem)) => elem.propose_allocation(caps),
        _ => None,
    }
}

/// The set of memory domains a node can emit (M351), for reconciling a
/// downstream allocation proposal against the producer's real capability.
/// Missing nodes report a System singleton (the conservative default).
fn node_output_domains(
    vg: &ValidatedGraph<GraphNodeRef<'_>>,
    node: NodeId,
) -> crate::memory::DomainSet {
    vg.element(node)
        .map(|n| n.output_domains())
        .unwrap_or(crate::memory::DomainSet::only(crate::memory::MemoryDomainKind::System))
}

/// Apply a downstream-derived allocation proposal to a transform's own pool.
fn element_configure_alloc(
    vg: &mut ValidatedGraph<GraphNodeRef<'_>>,
    node: NodeId,
    params: &AllocationParams,
) {
    if let Some(GraphNodeRef::Element(elem)) = vg.element_mut(node) {
        elem.configure_allocation(params);
    }
}

/// A node's latency contribution. `None` for structural (tee) and muxer nodes.
fn element_latency(vg: &ValidatedGraph<GraphNodeRef<'_>>, node: NodeId) -> Option<LatencyReport> {
    match vg.element(node) {
        Some(GraphNodeRef::Source(src)) => Some(src.latency()),
        Some(GraphNodeRef::Element(elem)) => Some(elem.latency()),
        _ => None,
    }
}

/// A node's offered clock for the pipeline clock election.
fn element_clock(vg: &ValidatedGraph<GraphNodeRef<'_>>, node: NodeId) -> Option<ClockCandidate> {
    match vg.element(node) {
        Some(GraphNodeRef::Source(src)) => src.provide_clock(),
        Some(GraphNodeRef::Element(elem)) => elem.provide_clock(),
        _ => None,
    }
}

/// Join two allocation proposals at a tee's input. Both branches consume the
/// one upstream producer, so the result is the most-restrictive per-parameter
/// intersection ([`AllocationParams::join`]): the larger size, count, and
/// alignment, with a matching memory domain. Divergent domains are an empty
/// intersection and fail loud with [`G2gError::AllocationConflict`] (no single
/// pool can satisfy, say, a CUDA branch and a D3D11 branch at once).
fn join_alloc(
    a: Option<AllocationParams>,
    b: Option<AllocationParams>,
) -> Result<Option<AllocationParams>, G2gError> {
    match (a, b) {
        (Some(x), Some(y)) => x.join(y).map(Some),
        (Some(x), None) => Ok(Some(x)),
        (None, b) => Ok(b),
    }
}

/// Re-solve one muxer input pad against the boundary's new caps (MX-1).
fn solve_mux_input_dyn(
    new_caps: &Caps,
    mux: &dyn DynMultiInputElement,
    pad: usize,
) -> Result<Caps, G2gError> {
    let src_c = CapsConstraint::LegacySource(new_caps.clone());
    let mux_c = mux.caps_constraint_as_input(pad);
    let links = solve_linear(&[&src_c, &mux_c]).map_err(|_| G2gError::CapsMismatch)?;
    links.last().cloned().ok_or(G2gError::CapsMismatch)
}

/// Re-derive the merged muxer output from its current per-input config (MX-2).
fn solve_mux_output_dyn(mux: &dyn DynMultiInputElement) -> Result<Caps, G2gError> {
    let mux_c = mux.caps_constraint_for_output().map_err(|_| G2gError::CapsMismatch)?;
    let sink_c = CapsConstraint::AcceptsAny;
    let links = solve_linear(&[&mux_c, &sink_c]).map_err(|_| G2gError::CapsMismatch)?;
    links.last().cloned().ok_or(G2gError::CapsMismatch)
}

async fn source_arm<'a>(
    mut src: Box<dyn DynSourceLoop + 'a>,
    out_tx: LinkSender,
    bus: Option<BusHandle>,
    progress: Option<PipelineProgress>,
) -> Result<u64, G2gError> {
    // M206: announce the stream start before any data, one per source, so an
    // application can bracket each stream's lifetime (StreamStart .. Eos).
    if let Some(b) = &bus {
        b.try_post(BusMessage::StreamStart);
    }
    // M203: publish the source's duration (if it knows one) before producing,
    // so a `DURATION` query is answerable from the first poll, and push-notify a
    // change on the bus. Polled once here; a source that discovers its length
    // mid-stream is a follow-up (it would publish through the handle directly).
    if let Some(duration_ns) = src.query_duration() {
        let changed = progress.as_ref().map(|p| p.publish_duration(duration_ns)).unwrap_or(true);
        if changed {
            if let Some(b) = &bus {
                b.try_post(BusMessage::DurationChanged { duration_ns });
            }
        }
    }
    let mut adapter = SenderSink::new(out_tx);
    // M81: open the stream with a SEGMENT ahead of the source's data, so every
    // downstream branch maps timestamps to running time from the first frame.
    let _ = adapter
        .push(PipelinePacket::Segment(Segment::new()))
        .await?;
    src.run(&mut adapter).await
}

/// An interior transform arm. Besides forwarding data, it (D4) selects on a β
/// `ArmDirective` channel alongside its data link so an upstream re-cascade
/// reaches it while parked on data, and on a mid-stream `CapsChanged` it steers
/// its forwarded output toward a downstream-acceptable shape using its
/// `downstream_feasible` snapshot (Caps-α), failing loud via a reverse
/// reconfigure if downstream positively rejects every output it can produce.
#[allow(clippy::too_many_arguments)]
async fn transform_arm<'a>(
    mut elem: Box<dyn DynAsyncElement + 'a>,
    in_rx: LinkReceiver,
    out_tx: LinkSender,
    arm_rx: Receiver<ArmDirective>,
    coord: GraphCoordHandle,
    node: NodeId,
    mut out_caps: Caps,
    downstream_feasible: Option<CapsSet>,
    mode: BranchMode,
    bus: Option<BusHandle>,
    probe: Probe,
) -> Result<u64, G2gError> {
    let mut adapter = SenderSink::new(out_tx);
    // M175: relay a downstream QoS report (seen on this transform's output link)
    // onto its input link, so it reaches the source/decoder one hop at a time
    // through any number of generic transforms, not just the sink's direct
    // upstream. The element's `process` is unaffected.
    adapter.relay_qos_to(in_rx.qos_slot());
    let mut control_open = true;
    loop {
        let packet = if control_open {
            match select2(arm_rx.recv(), in_rx.recv()).await {
                Either::Left(Some(ArmDirective::Recascade(params))) => {
                    // β: absorb the downstream proposal, re-derive our own from
                    // our output caps, and report it so the cascade continues to
                    // our upstream neighbour.
                    elem.configure_allocation(&params);
                    let proposal = elem.propose_allocation(&out_caps);
                    coord.report(Recascade { node, target: None, proposal }).await;
                    continue;
                }
                Either::Left(None) => {
                    control_open = false;
                    continue;
                }
                Either::Right(packet) => packet,
            }
        } else {
            in_rx.recv().await
        };
        match packet {
            Some(PipelinePacket::Eos) => {
                elem.process(PipelinePacket::Eos, &mut adapter).await?;
                adapter.push(PipelinePacket::Eos).await?;
                // Drop our report handle so the coordinator can wind down once
                // every arm exits, then drain any tail-end re-cascade directive
                // still in flight (a β triggered by the final pre-EOS frames).
                // Dropping before the drain decouples wind-down from the drain,
                // so no arm blocks holding the last handle.
                drop(coord);
                while let Some(ArmDirective::Recascade(params)) = arm_rx.recv().await {
                    elem.configure_allocation(&params);
                }
                return Ok(0);
            }
            Some(PipelinePacket::CapsChanged(new_caps)) => {
                // Caps-α: derive the forwarded output from this element's
                // constraint steered by the downstream feasibility snapshot.
                // `Defer` keeps the prior behavior (forward the incoming caps);
                // `Infeasible` surfaces loud as a reverse reconfigure.
                let forward_caps = {
                    let constraint = elem.caps_constraint_as_transform();
                    match resolve_forward_output(&constraint, &new_caps, downstream_feasible.as_ref())
                    {
                        ForwardResolve::Fixed(caps) => caps,
                        ForwardResolve::Defer => new_caps.clone(),
                        ForwardResolve::Infeasible(failure) => {
                            // Behind a tee (shared upstream): can't reverse-
                            // reconfigure. `FailLoud` fails the run; `Drop` ends
                            // this branch (siblings continue). On a single-producer
                            // chain: post the failure and reverse-reconfigure into
                            // the boundary, then keep flowing.
                            match mode {
                                BranchMode::FailLoud => return Err(G2gError::CapsMismatch),
                                BranchMode::Drop => {
                                    report_nego_failure(bus.as_ref(), failure);
                                    return Ok(0);
                                }
                                BranchMode::Reconfigure => {
                                    report_nego_failure(bus.as_ref(), failure);
                                    in_rx.request_reconfigure(Reconfigure::Renegotiate);
                                    continue;
                                }
                            }
                        }
                    }
                };
                match elem.configure_pipeline(&new_caps)? {
                    ConfigureOutcome::Accepted => {
                        // M188: re-resolve a caps-driven transform's output target
                        // on the mid-stream change too (matches startup, line ~421
                        // / the linear coordinator arm). No-op for property-driven
                        // or passthrough elements.
                        elem.configure_output(&forward_caps)?;
                        realloc_local_dyn(&mut *elem, &forward_caps);
                        out_caps = forward_caps.clone();
                        elem.process(PipelinePacket::CapsChanged(forward_caps), &mut adapter)
                            .await?;
                    }
                    ConfigureOutcome::ReFixate(counter) => {
                        in_rx.request_reconfigure(Reconfigure::Propose(counter));
                    }
                }
            }
            Some(packet) => {
                // M399: time the data-frame `process()` and sample input fill;
                // control packets (segment/flush) are excluded so the histogram
                // reflects real per-frame work, not cheap signalling.
                let timed =
                    probe.as_deref().filter(|_| matches!(&packet, PipelinePacket::DataFrame(_)));
                if let Some(p) = timed {
                    p.record_fill(in_rx.fill_percent());
                }
                let t0 = ElementProbe::mark();
                elem.process(packet, &mut adapter).await?;
                if let Some(p) = timed {
                    p.record_proc_since(t0);
                }
            }
            None => return Ok(0),
        }
    }
}

/// A sink arm. On a mid-stream `CapsChanged` (D4) it re-solves its input against
/// its declared constraint; on accept it re-derives its own pool and reports the
/// proposal so the β cascade walks one hop upstream. A re-solve failure surfaces
/// loud as a reverse reconfigure into the boundary that emitted the change.
#[allow(clippy::too_many_arguments)]
async fn sink_arm<'a>(
    mut elem: Box<dyn DynAsyncElement + 'a>,
    in_rx: LinkReceiver,
    coord: GraphCoordHandle,
    node: NodeId,
    mode: BranchMode,
    bus: Option<BusHandle>,
    state: Option<StateController>,
    progress: Option<PipelineProgress>,
    probe: Probe,
) -> Result<u64, G2gError> {
    let mut null = NullSink;
    let mut consumed = 0u64;
    let mut prerolled_self = false;
    let mut last_buffer_bucket: Option<u8> = None;
    // M203: the segment in force, so a buffer's PTS maps to stream-time position.
    let mut current_segment: Option<Segment> = None;
    // M360 re-preroll: generation this arm last prerolled at, and whether it is
    // draining stale pre-seek frames (paused flushing seek) until the `Flush`.
    let mut preroll_gen = state.as_ref().map_or(0, |sc| sc.preroll_generation());
    let mut flushing = false;
    loop {
        // M78 flow gate: below `Playing` the sink parks here, so it stops
        // draining its edge and backpressure stalls the DAG upstream. Non-live
        // `Paused` admits this sink's one preroll buffer; `Null` ends the arm.
        if let Some(sc) = &state {
            if sc.flow_gate(prerolled_self, preroll_gen).await == Flow::Stop {
                return Ok(consumed);
            }
            // M360: a `request_repreroll` (paused flushing seek) bumped the
            // generation; re-arm preroll and drain stale pre-seek frames until
            // the `Flush`, so the post-flush target is the new visible preroll.
            let gen = sc.preroll_generation();
            if gen != preroll_gen {
                preroll_gen = gen;
                prerolled_self = false;
                flushing = true;
            }
        }
        // M87 buffering: sample the input link's fill and post a `Buffering`
        // report when it crosses a quartile band. The first iteration samples
        // an as-yet-unfilled link, so a `bus` always sees at least one report.
        if let Some(b) = &bus {
            let pct = in_rx.fill_percent();
            let bucket = buffering_bucket(pct);
            if last_buffer_bucket != Some(bucket) {
                last_buffer_bucket = Some(bucket);
                b.try_post(BusMessage::Buffering { percent: pct });
            }
        }
        match in_rx.recv().await {
            // M360: discard stale pre-seek buffers while draining toward the
            // `Flush`; control packets fall through (the `Flush` ends drain).
            Some(PipelinePacket::DataFrame(_)) if flushing => continue,
            Some(PipelinePacket::Eos) => {
                elem.process(PipelinePacket::Eos, &mut null).await?;
                // Count this sink toward preroll only if it never took a real
                // preroll buffer (an empty stream). Without the guard a sink
                // that already prerolled double-decrements the shared counter
                // and completes the pipeline preroll prematurely.
                if !prerolled_self {
                    if let Some(sc) = &state {
                        sc.notify_prerolled();
                    }
                }
                return Ok(consumed);
            }
            Some(PipelinePacket::CapsChanged(new_caps)) => {
                // Behind a tee (shared upstream): can't reverse-reconfigure.
                // `FailLoud` fails the run; `Drop` ends this branch (siblings
                // continue). On a single-producer chain: post the failure and
                // reverse-reconfigure into the boundary, then keep flowing.
                let sink_caps = match re_solve_downstream_dyn_sink(&new_caps, &*elem) {
                    Ok(caps) => caps,
                    Err(failure) => match mode {
                        BranchMode::FailLoud => return Err(G2gError::CapsMismatch),
                        BranchMode::Drop => {
                            report_nego_failure(bus.as_ref(), failure);
                            return Ok(consumed);
                        }
                        BranchMode::Reconfigure => {
                            report_nego_failure(bus.as_ref(), failure);
                            in_rx.request_reconfigure(Reconfigure::Renegotiate);
                            continue;
                        }
                    },
                };
                match elem.configure_pipeline(&sink_caps)? {
                    ConfigureOutcome::Accepted => {
                        let proposal = elem.propose_allocation(&sink_caps);
                        if let Some(p) = &proposal {
                            elem.configure_allocation(p);
                        }
                        coord.report(Recascade { node, target: None, proposal }).await;
                        elem.process(PipelinePacket::CapsChanged(sink_caps), &mut null)
                            .await?;
                    }
                    ConfigureOutcome::ReFixate(counter) => {
                        in_rx.request_reconfigure(Reconfigure::Propose(counter));
                    }
                }
            }
            // M360: the `Flush` ends the re-preroll drain; the next (post-flush)
            // DataFrame becomes the new visible preroll.
            Some(PipelinePacket::Flush) => {
                flushing = false;
                elem.process(PipelinePacket::Flush, &mut null).await?;
            }
            Some(packet) => {
                // M203: follow the segment and publish each buffer's stream-time
                // position, so an application POSITION poll is answerable. The
                // sink is the position authority, as in GStreamer (segment + last
                // buffer). Inspect before `process` moves the packet.
                match &packet {
                    PipelinePacket::Segment(seg) => current_segment = Some(*seg),
                    PipelinePacket::DataFrame(frame) => {
                        if let Some(p) = &progress {
                            let pts = frame.timing.pts_ns;
                            let pos = current_segment
                                .as_ref()
                                .and_then(|s| s.to_stream_time(pts))
                                .unwrap_or(pts);
                            p.set_position(pos);
                        }
                    }
                    _ => {}
                }
                let is_buffer = matches!(packet, PipelinePacket::DataFrame(_));
                if is_buffer {
                    consumed += 1;
                }
                // M399: time the data-frame `process()` and sample input fill.
                let timed = probe.as_deref().filter(|_| is_buffer);
                if let Some(p) = timed {
                    p.record_fill(in_rx.fill_percent());
                }
                let t0 = ElementProbe::mark();
                elem.process(packet, &mut null).await?;
                if let Some(p) = timed {
                    p.record_proc_since(t0);
                }
                // M175 upstream QoS: a sink that dropped a late frame asks to
                // shed load; store its report on this sink's input link, where
                // the upstream transform relays it one hop further (or the source
                // observes it directly as `PushOutcome::Qos`).
                if let Some(qos) = elem.take_qos() {
                    in_rx.request_qos(qos);
                }
                // Keyframe-request / renegotiation a sink originates (WebRTC PLI);
                // store it on the input link, where the upstream encoder/transform
                // observes it as `PushOutcome::Reconfigure`.
                if let Some(reconf) = elem.take_reconfigure() {
                    in_rx.request_reconfigure(reconf);
                }
                // Target bitrate (WebRTC BWE) up the reverse channel to the encoder.
                if let Some(bps) = elem.take_bitrate() {
                    in_rx.request_bitrate(bps);
                }
                // M78: the first buffer in non-live `Paused` is this sink's
                // preroll frame; mark this arm prerolled so the gate flips to a
                // hold, and report it so the pipeline preroll aggregates toward
                // a single `AsyncDone`.
                if is_buffer && !prerolled_self {
                    prerolled_self = true;
                    if let Some(sc) = &state {
                        sc.notify_prerolled();
                    }
                }
            }
            None => return Ok(consumed),
        }
    }
}

async fn tee_arm(
    in_rx: LinkReceiver,
    out_txs: Vec<LinkSender>,
    branch_drop: bool,
) -> Result<u64, G2gError> {
    let mut senders: Vec<SenderSink> = out_txs.into_iter().map(SenderSink::new).collect();
    loop {
        match in_rx.recv().await {
            Some(PipelinePacket::Eos) => {
                for s in senders.iter_mut() {
                    match s.push(PipelinePacket::Eos).await {
                        Ok(_) => {}
                        // A dropped branch (`AllowBranchDrop`) has closed its
                        // channel; skip it. Under `FailLoud` a closed branch is a
                        // genuine error and propagates.
                        Err(G2gError::Shutdown) if branch_drop => {}
                        Err(e) => return Err(e),
                    }
                }
                return Ok(0);
            }
            Some(packet) => {
                if branch_drop {
                    broadcast_drop_closed(&mut senders, packet).await?;
                    // Every branch has dropped: the fan-out has no consumers left,
                    // so this tee is done.
                    if senders.is_empty() {
                        return Ok(0);
                    }
                } else {
                    broadcast(&mut senders, packet).await?;
                }
            }
            None => return Ok(0),
        }
    }
}

/// Broadcast like [`broadcast`], but a branch whose receiver has closed
/// (a dropped `AllowBranchDrop` branch) is removed from `senders` instead of
/// failing the fan-out. A genuine downstream error still surfaces through that
/// branch arm's own result, so swallowing the closed channel here is safe.
async fn broadcast_drop_closed(
    senders: &mut Vec<SenderSink>,
    mut packet: PipelinePacket,
) -> Result<(), G2gError> {
    if let PipelinePacket::DataFrame(frame) = &mut packet {
        frame.domain.make_shareable();
    }
    let mut dead: Vec<usize> = Vec::new();
    for (i, s) in senders.iter_mut().enumerate() {
        match s.push(try_clone_packet(&packet)?).await {
            Ok(_) => {}
            Err(G2gError::Shutdown) => dead.push(i),
            Err(e) => return Err(e),
        }
    }
    // Remove dead senders high-index-first so earlier indices stay valid.
    for &i in dead.iter().rev() {
        senders.remove(i);
    }
    Ok(())
}

/// The demux arm: drain the single input edge and let the routing element
/// dispatch each packet to a chosen output port (the transpose of `muxer_arm`).
/// Mirrors the `run_source_fanout` router loop: a packet goes to
/// `MultiOutputElement::process`, which calls `push_to(port, ..)`; on `Eos` the
/// element flushes first, then the arm closes every branch with its own `Eos`
/// (the runner owns the per-branch end, like the tee arm).
async fn demux_arm<'a>(
    mut demux: Box<dyn DynMultiOutputElement + 'a>,
    in_rx: LinkReceiver,
    out_txs: Vec<LinkSender>,
) -> Result<u64, G2gError> {
    let branch_count = out_txs.len();
    let senders: Vec<SenderSink> = out_txs.into_iter().map(SenderSink::new).collect();
    let mut multi = MultiSenderSink::new(senders);
    loop {
        match in_rx.recv().await {
            Some(PipelinePacket::Eos) => {
                demux.process(PipelinePacket::Eos, &mut multi).await?;
                for port in 0..branch_count {
                    multi.push_to(port, PipelinePacket::Eos).await?;
                }
                return Ok(0);
            }
            Some(packet) => {
                demux.process(packet, &mut multi).await?;
            }
            None => return Ok(0),
        }
    }
}

/// Send `packet` to every tee branch. The frame's memory is made shareable once
/// (a zero-copy refcount handle, M250), so the per-branch clones below are
/// refcount bumps, not deep copies; the original is then moved into the last
/// branch. A fan-out of `n` makes zero byte copies of a `System` / GPU frame.
pub(crate) async fn broadcast(
    senders: &mut [SenderSink],
    mut packet: PipelinePacket,
) -> Result<(), G2gError> {
    if let PipelinePacket::DataFrame(frame) = &mut packet {
        frame.domain.make_shareable();
    }
    let last = senders.len() - 1;
    for s in senders[..last].iter_mut() {
        s.push(try_clone_packet(&packet)?).await?;
    }
    senders[last].push(packet).await?;
    Ok(())
}

/// One muxer input: drain its edge and tag every packet with its pad index for
/// the muxer arm. A per-input `Eos` (or a closed edge) is tagged so the muxer
/// arm can aggregate the single merged `Eos`.
async fn muxer_forwarder(
    in_rx: LinkReceiver,
    tagged: Sender<PipelinePacket>,
) -> Result<u64, G2gError> {
    loop {
        match in_rx.recv().await {
            Some(PipelinePacket::Eos) | None => {
                tagged.send(PipelinePacket::Eos).await.map_err(|_| G2gError::Shutdown)?;
                return Ok(0);
            }
            Some(packet) => {
                tagged.send(packet).await.map_err(|_| G2gError::Shutdown)?;
            }
        }
    }
}

/// Block until some open input pad delivers a packet (or closes), scanning
/// round-robin from `start` so the wake-up path stays fair too. Returns the
/// pad's slot index (into `pad_rxs`) and its packet; `None` means that pad's
/// channel closed without an `Eos` (an upstream error), treated as an end. All
/// receivers register the same task waker, so a push on any one wakes us.
async fn muxer_recv_any(
    pad_rxs: &[(usize, Receiver<PipelinePacket>)],
    open: &[bool],
    start: usize,
) -> (usize, Option<PipelinePacket>) {
    let n = pad_rxs.len();
    core::future::poll_fn(|cx| {
        for k in 0..n {
            let slot = (start + k) % n;
            if !open[slot] {
                continue;
            }
            // `RecvFuture` holds only a `&Receiver`, so it is `Unpin`; polling it
            // parks our waker on that channel when pending.
            let mut f = pad_rxs[slot].1.recv();
            if let core::task::Poll::Ready(v) = core::future::Future::poll(Pin::new(&mut f), cx) {
                return core::task::Poll::Ready((slot, v));
            }
        }
        core::task::Poll::Pending
    })
    .await
}

/// The muxer arm: drain the per-input channels round-robin, combine each input's
/// packets via `process(pad, ..)`, and emit a single `Eos` once every input has
/// ended. Round-robin draining keeps a fast input from starving a slow one (a
/// frozen overlay and a hung EOS aggregation). The per-input `Eos` is delivered
/// to the element first (so a stateful muxer can flush) but the element must not
/// forward it; the runner owns the merged one.
#[allow(clippy::too_many_arguments)]
async fn muxer_arm<'a>(
    mut mux: Box<dyn DynMultiInputElement + 'a>,
    pad_rxs: Vec<(usize, Receiver<PipelinePacket>)>,
    out_tx: LinkSender,
    input_count: usize,
    mut current_output: Caps,
    coord: GraphCoordHandle,
    node: NodeId,
    pad_upstream: Vec<Option<NodeId>>,
) -> Result<u64, G2gError> {
    let mut adapter = SenderSink::new(out_tx);
    let mut open = alloc::vec![true; input_count];
    let mut ended = 0usize;
    // Cursor for round-robin fairness across both the try-drain and block paths.
    let mut next = 0usize;
    loop {
        // Take one buffered packet, scanning round-robin from `next`, so no
        // single input can monopolize the muxer while others have data waiting.
        let mut picked: Option<(usize, PipelinePacket)> = None;
        for k in 0..input_count {
            let slot = (next + k) % input_count;
            if !open[slot] {
                continue;
            }
            if let Some(pkt) = pad_rxs[slot].1.try_recv() {
                picked = Some((slot, pkt));
                next = (slot + 1) % input_count;
                break;
            }
        }
        let (slot, packet) = match picked {
            Some(p) => p,
            None => {
                if !open.iter().any(|&o| o) {
                    return Ok(0);
                }
                let (slot, maybe) = muxer_recv_any(&pad_rxs, &open, next).await;
                next = (slot + 1) % input_count;
                // A closed channel with no `Eos` is an upstream end; fold it into
                // the same end-of-input path so aggregation still completes.
                (slot, maybe.unwrap_or(PipelinePacket::Eos))
            }
        };
        let pad = pad_rxs[slot].0;
        match packet {
            PipelinePacket::Eos => {
                mux.process(pad, PipelinePacket::Eos, &mut adapter).await?;
                open[slot] = false;
                ended += 1;
                if ended == input_count {
                    adapter.push(PipelinePacket::Eos).await?;
                    return Ok(0);
                }
            }
            PipelinePacket::CapsChanged(new_caps) => {
                // MX-1: re-solve this input against its pad constraint and
                // reconfigure the pad; the input-side `CapsChanged` is consumed,
                // not forwarded as if it were the merged output.
                let input_caps = solve_mux_input_dyn(&new_caps, &*mux, pad)?;
                mux.configure_pipeline(pad, &input_caps)?.reject_refixate()?;
                // MX-1β: the muxer's per-pad allocation demand may shift with the
                // new input caps; re-cascade it up exactly this pad's branch (the
                // other inputs are untouched). A source feeding the pad directly
                // has no interruptible arm, so there is nothing to walk.
                if let (Some(target), Some(p)) =
                    (pad_upstream[slot], mux.propose_allocation_for_input(pad, &input_caps))
                {
                    coord.report(Recascade { node, target: Some(target), proposal: Some(p) }).await;
                }
                // MX-2: the per-input change may shift the merged output. Emit one
                // downstream `CapsChanged` only when it actually changed.
                let new_output = solve_mux_output_dyn(&*mux)?;
                if new_output != current_output {
                    current_output = new_output.clone();
                    adapter.push(PipelinePacket::CapsChanged(new_output)).await?;
                }
            }
            packet => {
                mux.process(pad, packet, &mut adapter).await?;
            }
        }
    }
}

/// The PTS-ordered muxer arm (the opt-in alternative to [`muxer_arm`], selected
/// by [`DynMultiInputElement::input_pts_ordered`]): buffer each input's
/// `DataFrame`s in an [`InputAggregator`] and release the globally-earliest-PTS
/// one only once every still-open input has a head queued, so `process(pad, ..)`
/// sees frames in non-decreasing PTS across all pads. The runner does the
/// time-ordered interleave a multi-camera grid / PTS-synchronized compositor
/// would otherwise hand-roll. `Eos` (per-input flush + aggregation) and
/// `CapsChanged` (MX-1 / MX-2) are handled as in [`muxer_arm`]; only `DataFrame`s
/// are reordered.
#[allow(clippy::too_many_arguments)]
async fn muxer_arm_pts<'a>(
    mut mux: Box<dyn DynMultiInputElement + 'a>,
    pad_rxs: Vec<(usize, Receiver<PipelinePacket>)>,
    out_tx: LinkSender,
    input_count: usize,
    mut current_output: Caps,
    coord: GraphCoordHandle,
    node: NodeId,
    pad_upstream: Vec<Option<NodeId>>,
) -> Result<u64, G2gError> {
    let mut adapter = SenderSink::new(out_tx);
    let mut open = alloc::vec![true; input_count];
    let mut agg: InputAggregator<Frame> = InputAggregator::new(input_count);
    // Round-robin wake cursor, so a fast input does not bias the block path.
    let mut next = 0usize;
    loop {
        // Release every frame now safe to emit, in global PTS order: the
        // aggregator yields the earliest only once every still-contributing input
        // has a head, so no later input can still deliver something earlier.
        while let Some((slot, frame)) = agg.take_earliest_by(|f| f.timing.pts_ns) {
            let pad = pad_rxs[slot].0;
            mux.process(pad, PipelinePacket::DataFrame(frame), &mut adapter).await?;
        }
        // Once every input has ended, the loop above has drained the aggregator
        // (ended+empty inputs drop out of the round); emit the single merged Eos.
        if !open.iter().any(|&o| o) {
            adapter.push(PipelinePacket::Eos).await?;
            return Ok(0);
        }
        // Make progress: block for the next packet from any still-open input.
        let (slot, maybe) = muxer_recv_any(&pad_rxs, &open, next).await;
        next = (slot + 1) % input_count;
        let pad = pad_rxs[slot].0;
        // A closed channel with no `Eos` is an upstream end (as in `muxer_arm`).
        match maybe.unwrap_or(PipelinePacket::Eos) {
            PipelinePacket::DataFrame(frame) => agg.push(slot, frame),
            PipelinePacket::Eos => {
                mux.process(pad, PipelinePacket::Eos, &mut adapter).await?;
                open[slot] = false;
                agg.mark_ended(slot);
            }
            PipelinePacket::CapsChanged(new_caps) => {
                // MX-1 / MX-1β / MX-2, identical to `muxer_arm`: re-solve this
                // input's pad, re-cascade the per-pad β proposal up that branch,
                // and emit one downstream `CapsChanged` only when the merged
                // output actually shifts.
                let input_caps = solve_mux_input_dyn(&new_caps, &*mux, pad)?;
                mux.configure_pipeline(pad, &input_caps)?.reject_refixate()?;
                if let (Some(target), Some(p)) =
                    (pad_upstream[slot], mux.propose_allocation_for_input(pad, &input_caps))
                {
                    coord.report(Recascade { node, target: Some(target), proposal: Some(p) }).await;
                }
                let new_output = solve_mux_output_dyn(&*mux)?;
                if new_output != current_output {
                    current_output = new_output.clone();
                    adapter.push(PipelinePacket::CapsChanged(new_output)).await?;
                }
            }
            // Flush / Segment are not part of the muxer-input contract here.
            _ => {}
        }
    }
}

/// Clone a packet for a tee branch (M213, M250). Control packets clone trivially;
/// a data frame's memory is shared via [`MemoryDomain::share`]: a zero-copy
/// refcount bump for the GPU domains, the shared-CPU `SystemView`, and (once
/// `broadcast` has called `make_shareable`) owned-CPU `System` bytes too. So a
/// GPU-decoded or CPU frame fans out to several consumers (eg inference +
/// display) with no copy, where `System` previously deep-copied per branch and a
/// GPU frame failed loud.
pub(crate) fn try_clone_packet(packet: &PipelinePacket) -> Result<PipelinePacket, G2gError> {
    Ok(match packet {
        PipelinePacket::CapsChanged(caps) => PipelinePacket::CapsChanged(caps.clone()),
        PipelinePacket::Eos => PipelinePacket::Eos,
        PipelinePacket::Flush => PipelinePacket::Flush,
        PipelinePacket::Segment(seg) => PipelinePacket::Segment(*seg),
        // Tee clone: shares the buffer where the domain allows (GPU handles /
        // pre-shared System bytes refcount, owned CPU bytes deep-copy) and shares
        // per-frame metadata by Arc refcount with copy-on-write on mutation, so a
        // detector branch and a video branch carry the same AnalyticsMeta without
        // aliasing. The frame-level fan-out primitive.
        PipelinePacket::DataFrame(frame) => PipelinePacket::DataFrame(frame.share()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameTiming;
    use crate::memory::{MemoryDomain, SystemSlice};

    fn system_frame(bytes: &[u8], seq: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
            timing: FrameTiming { pts_ns: 7, ..FrameTiming::default() },
            sequence: seq,
            meta: Default::default(),
        })
    }

    #[test]
    fn buffering_bucket_bands_fill_into_quartiles() {
        // Empty, quarter steps, and full map to distinct bands; values within a
        // band collapse so the sink posts only on a level transition.
        assert_eq!(buffering_bucket(0), 0);
        assert_eq!(buffering_bucket(24), 0);
        assert_eq!(buffering_bucket(25), 1);
        assert_eq!(buffering_bucket(50), 2);
        assert_eq!(buffering_bucket(75), 3);
        assert_eq!(buffering_bucket(100), 4);
    }

    #[test]
    fn clones_system_frame_bytes_and_timing() {
        let original = system_frame(&[1, 2, 3, 4], 9);
        let cloned = try_clone_packet(&original).expect("system frame clones");
        let (PipelinePacket::DataFrame(a), PipelinePacket::DataFrame(b)) = (&original, &cloned)
        else {
            panic!("expected data frames");
        };
        let MemoryDomain::System(sa) = &a.domain else { panic!() };
        let MemoryDomain::System(sb) = &b.domain else { panic!() };
        assert_eq!(sa.as_slice(), sb.as_slice(), "bytes copied");
        assert_ne!(sb.as_slice().as_ptr(), sa.as_slice().as_ptr(), "distinct allocation");
        assert_eq!(b.timing, a.timing);
        assert_eq!(b.sequence, 9);
    }

    #[test]
    fn clones_control_packets() {
        assert!(matches!(
            try_clone_packet(&PipelinePacket::Eos),
            Ok(PipelinePacket::Eos)
        ));
        assert!(matches!(
            try_clone_packet(&PipelinePacket::Flush),
            Ok(PipelinePacket::Flush)
        ));
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn tee_clone_carries_analytics_meta() {
        use crate::meta::{AnalyticsMeta, BBox, ObjectDetection};
        // A detector attaches analytics; the tee clone must carry it onto the
        // sibling (video) branch so a downstream overlay can read it.
        let PipelinePacket::DataFrame(mut original) = system_frame(&[0, 0, 0, 0], 1) else {
            panic!("data frame");
        };
        let mut a = AnalyticsMeta::new();
        a.add_detection(ObjectDetection {
            bbox: BBox { x: 0.1, y: 0.1, w: 0.2, h: 0.2 },
            label: 5,
            confidence: 0.9,
        });
        original.meta.attach(a);

        let cloned = try_clone_packet(&PipelinePacket::DataFrame(original)).expect("clone");
        let PipelinePacket::DataFrame(b) = cloned else { panic!("data frame") };
        let meta = b.meta.get::<AnalyticsMeta>().expect("meta carried to tee branch");
        assert_eq!(meta.detections().count(), 1);
        assert_eq!(meta.detections().next().unwrap().label, 5);
    }
}
