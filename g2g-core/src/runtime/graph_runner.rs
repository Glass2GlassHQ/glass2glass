//! DAG pipeline runner (DESIGN_TODO "DAG runner" D3).
//!
//! [`run_graph`] drives an arbitrary multimedia DAG built with [`Graph`]:
//! whole-graph CSP negotiation via [`solve_graph`] (D2), then one spawned arm
//! per node over per-edge channels, joined with [`join_all`]. It collapses the
//! linear + fan-out runner shapes into one entry point.
//!
//! Scope: source / transform / sink / tee (fan-out) + muxer (fan-in). A tee
//! broadcasts each packet to all its branches; since [`PipelinePacket`] is not
//! `Clone` (a GPU-resident frame owns a non-copyable handle), the broadcast
//! deep-copies `System` frames and fails loud on a GPU domain (a refcounted
//! shareable frame for the zero-copy / GPU tee is a follow-up). A muxer node
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
//! structural tee nodes; a source or muxer terminates the walk. Tee branches
//! re-solve independently (each broadcast `CapsChanged` lands in its own arm);
//! muxer inputs re-configure per pad. A muxer is a β boundary: its inputs carry
//! no per-pad allocation channel, so the proposal stops there (a follow-up).

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::caps::{Caps, CapsSet};
use crate::clock::{elect_clock, ClockCandidate, ClockPriority, PipelineClock};
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, DynAsyncElement, OutputSink, Reconfigure,
};
use crate::error::G2gError;
use crate::fanout::MultiInputElement;
use crate::format_element::CapsConstraint;
use crate::frame::{Frame, PipelinePacket};
use crate::graph::{Graph, NodeId, NodeKind, ValidatedGraph};
use crate::memory::{MemoryDomain, SystemSlice};
use crate::query::{AllocationParams, LatencyReport};
use crate::runtime::channel::{bounded, link, LinkReceiver, LinkSender, Receiver, Sender, SenderSink};
use crate::runtime::coordinator::{realloc_local_dyn, ArmDirective};
use crate::runtime::fanin::{DynMultiInputElement, DynSourceLoop};
use crate::runtime::join::{join_all, select2, Either};
use crate::runtime::runner::{
    re_solve_downstream_dyn_sink, LinkCapacity, NullSink, RunStats, SourceLoop,
};
use crate::runtime::solver::{
    graph_downstream_feasibility, resolve_forward_output, solve_graph, solve_linear, ForwardResolve,
    NodeConstraint,
};

/// Element payload for a [`Graph`] driven by [`run_graph`]. Sources,
/// transforms/sinks, and muxers implement different traits (a source has no
/// input pad, a muxer has many), so the payload is an enum the runner matches
/// on per node kind. A tee carries no element (`Graph::add_tee` takes none).
pub enum GraphNode {
    Source(Box<dyn DynSourceLoop>),
    Element(Box<dyn DynAsyncElement>),
    Muxer(Box<dyn DynMultiInputElement>),
}

impl GraphNode {
    /// Box a source (`add_source`).
    pub fn source<S: SourceLoop + 'static>(source: S) -> Self {
        GraphNode::Source(Box::new(source))
    }

    /// Box a transform or sink (`add_transform` / `add_sink`).
    pub fn element<E: AsyncElement + 'static>(element: E) -> Self {
        GraphNode::Element(Box::new(element))
    }

    /// Box a fan-in muxer (`add_muxer`).
    pub fn muxer<M: MultiInputElement + 'static>(muxer: M) -> Self {
        GraphNode::Muxer(Box::new(muxer))
    }
}

impl core::fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GraphNode::Source(_) => f.write_str("GraphNode::Source(..)"),
            GraphNode::Element(_) => f.write_str("GraphNode::Element(..)"),
            GraphNode::Muxer(_) => f.write_str("GraphNode::Muxer(..)"),
        }
    }
}

/// A β allocation re-cascade report from an arm to the [`GraphCoordinator`].
/// A sink reports the proposal it re-derived on a mid-stream `CapsChanged`; an
/// interior transform reports the proposal it re-derived after applying an
/// upstream directive. `node` is the reporting node, so the coordinator walks
/// the proposal one hop further upstream through the graph topology.
#[derive(Debug, Clone)]
struct Recascade {
    node: NodeId,
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
/// nodes; a source or muxer terminates the walk). On each report it forwards an
/// `ArmDirective::Recascade` one hop upstream to those arms, which re-derive and
/// report again, so the cascade walks the DAG without a global lock. The walk is
/// reactive and non-blocking (`try_send`), so it never wedges the data plane.
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
                for &u in &self.upstream_arms[event.node.0 as usize] {
                    if let Some(ctrl) = &self.arm_ctrl[u.0 as usize] {
                        let _ = ctrl.try_send(ArmDirective::Recascade(p));
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

/// Drive an arbitrary DAG to EOS. Negotiates the whole graph at once, then runs
/// one arm per node over per-edge channels. `link_capacity` accepts a
/// [`LatencyProfile`](crate::runtime::LatencyProfile) or a `usize` depth.
///
/// Allocation, clock-election, and latency aggregation over a DAG are reported
/// as neutral values for now (like the fan-out / fan-in runners); the DAG-wide
/// folds are a follow-up. A non-`System` frame in a tee or a negotiation
/// conflict fails loud.
pub async fn run_graph<Clk: PipelineClock>(
    graph: Graph<GraphNode>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError> {
    let link_capacity: usize = link_capacity.into().get();
    let mut vg = graph.finish().map_err(|_| G2gError::CapsMismatch)?;
    let n = vg.node_count();
    if n < 2 {
        return Err(G2gError::CapsMismatch);
    }
    let topo = vg.topo().to_vec();

    // Phase 1: probe each source's caps (async) into an owned map, releasing
    // the mutable borrow before the constraint phase borrows every node.
    let mut source_caps: Vec<Option<Caps>> = (0..n).map(|_| None).collect();
    for &node in &topo {
        if matches!(vg.kind(node), NodeKind::Source) {
            let GraphNode::Source(src) = vg.element_mut(node).ok_or(G2gError::CapsMismatch)? else {
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
        let mut constraints: Vec<NodeConstraint<'_>> = Vec::with_capacity(n);
        for (i, src_caps) in source_caps.iter().enumerate() {
            let node = NodeId(i as u32);
            let nc = match vg.kind(node) {
                NodeKind::Source => {
                    let caps = src_caps.clone().ok_or(G2gError::CapsMismatch)?;
                    NodeConstraint::Element(CapsConstraint::Produces(CapsSet::one(caps)))
                }
                NodeKind::Transform => {
                    let elem = element_ref(&vg, node).ok_or(G2gError::CapsMismatch)?;
                    NodeConstraint::Element(elem.caps_constraint_as_transform())
                }
                NodeKind::Sink => {
                    let elem = element_ref(&vg, node).ok_or(G2gError::CapsMismatch)?;
                    NodeConstraint::Element(elem.caps_constraint_as_sink())
                }
                // A tee is structural; the solver ignores its slot.
                NodeKind::Tee(_) => NodeConstraint::Element(CapsConstraint::IdentityAny),
                NodeKind::Muxer(_) => {
                    let GraphNode::Muxer(elem) =
                        vg.element(node).ok_or(G2gError::CapsMismatch)?
                    else {
                        return Err(G2gError::CapsMismatch);
                    };
                    let inputs: Vec<CapsConstraint<'_>> = (0..elem.input_count())
                        .map(|pad| elem.caps_constraint_as_input(pad))
                        .collect();
                    let output = elem
                        .caps_constraint_for_output()
                        .map_err(|_| G2gError::CapsMismatch)?;
                    NodeConstraint::Muxer { inputs, output }
                }
            };
            constraints.push(nc);
        }
        let solution = solve_graph(&vg, &constraints).map_err(|_| G2gError::CapsMismatch)?;
        let feasibility = graph_downstream_feasibility(&vg, &constraints);
        (solution, feasibility)
    };

    // Phase 3: configure each element with its negotiated caps. Source nodes
    // take their single output edge's caps (no input); transforms and sinks
    // take their input edge's caps.
    for &node in &topo {
        match vg.kind(node) {
            NodeKind::Source => {
                let caps = solution[vg.out_edges(node)[0]].clone();
                let GraphNode::Source(src) =
                    vg.element_mut(node).ok_or(G2gError::CapsMismatch)?
                else {
                    return Err(G2gError::CapsMismatch);
                };
                if let ConfigureOutcome::ReFixate(_) = src.configure_pipeline(&caps)? {
                    return Err(G2gError::FixationFailed);
                }
            }
            NodeKind::Transform | NodeKind::Sink => {
                let caps = solution[vg.in_edges(node)[0]].clone();
                let GraphNode::Element(elem) =
                    vg.element_mut(node).ok_or(G2gError::CapsMismatch)?
                else {
                    return Err(G2gError::CapsMismatch);
                };
                if let ConfigureOutcome::ReFixate(_) = elem.configure_pipeline(&caps)? {
                    return Err(G2gError::FixationFailed);
                }
            }
            NodeKind::Tee(_) => {}
            NodeKind::Muxer(_) => {
                // Configure each input pad with its in-edge's negotiated caps.
                let in_edges: Vec<usize> = vg.in_edges(node).to_vec();
                for &eid in &in_edges {
                    let pad = vg.edge(eid).dst.index as usize;
                    let caps = solution[eid].clone();
                    let GraphNode::Muxer(elem) =
                        vg.element_mut(node).ok_or(G2gError::CapsMismatch)?
                    else {
                        return Err(G2gError::CapsMismatch);
                    };
                    if let ConfigureOutcome::ReFixate(_) = elem.configure_pipeline(pad, &caps)? {
                        return Err(G2gError::FixationFailed);
                    }
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
    // (most-demanding) onto its single input; a muxer is a boundary. The
    // source's absorbed proposal is the reported `allocation`. For a linear
    // chain this is byte-for-byte the linear runner's sink->source fold.
    let nee = vg.edge_count();
    let mut edge_proposal: Vec<Option<AllocationParams>> = (0..nee).map(|_| None).collect();
    let mut allocation: Option<AllocationParams> = None;
    for &node in topo.iter().rev() {
        match vg.kind(node) {
            NodeKind::Sink => {
                let in_e = vg.in_edges(node)[0];
                let caps = solution[in_e].clone();
                edge_proposal[in_e] = element_propose(&vg, node, &caps);
            }
            NodeKind::Transform => {
                let in_e = vg.in_edges(node)[0];
                let out_e = vg.out_edges(node)[0];
                if let Some(p) = edge_proposal[out_e] {
                    element_configure_alloc(&mut vg, node, &p);
                }
                let caps = solution[out_e].clone();
                edge_proposal[in_e] = element_propose(&vg, node, &caps);
            }
            NodeKind::Tee(_) => {
                let in_e = vg.in_edges(node)[0];
                let mut joined: Option<AllocationParams> = None;
                for &oe in vg.out_edges(node) {
                    joined = join_alloc(joined, edge_proposal[oe]);
                }
                edge_proposal[in_e] = joined;
            }
            NodeKind::Source => {
                let out_e = vg.out_edges(node)[0];
                if let Some(p) = edge_proposal[out_e] {
                    if let GraphNode::Source(src) = vg.element_mut(node).ok_or(G2gError::CapsMismatch)? {
                        src.configure_allocation(&p);
                    }
                    allocation = Some(p);
                }
            }
            NodeKind::Muxer(_) => {}
        }
    }

    // Latency fold + clock election over every element node (tee is structural;
    // a muxer contributes neither, like the fan-in runner).
    let mut latencies: Vec<LatencyReport> = Vec::with_capacity(n);
    let mut clocks: Vec<Option<ClockCandidate>> = Vec::with_capacity(n);
    for &node in &topo {
        if let Some(l) = element_latency(&vg, node) {
            latencies.push(l);
            clocks.push(element_clock(&vg, node));
        }
    }
    let latency = LatencyReport::aggregate(latencies);
    let elected = elect_clock(clocks);
    let (clock_priority, base_time_ns) = match &elected {
        Some(c) => (c.priority, c.clock.now_ns()),
        None => (ClockPriority::SystemFallback, clock.now_ns()),
    };

    // Phase 4: one bounded channel per edge, then one arm per node. Each arm
    // takes the senders of its outgoing edges and the receivers of its
    // incoming edges (a tee holds n senders, a sink one receiver, etc.).
    let ne = vg.edge_count();
    let mut txs: Vec<Option<LinkSender>> = Vec::with_capacity(ne);
    let mut rxs: Vec<Option<LinkReceiver>> = Vec::with_capacity(ne);
    for _ in 0..ne {
        let (tx, rx) = link(link_capacity);
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
    for &node in &topo {
        if matches!(vg.kind(node), NodeKind::Transform) {
            let (ctx, crx) = bounded::<ArmDirective>(link_capacity);
            arm_ctrl[node.0 as usize] = Some(ctx);
            arm_ctrl_rx[node.0 as usize] = Some(crx);
        }
    }
    // β reporters are transforms (after a directive) and sinks (on caps change);
    // each forwards to the nearest interior arm feeding its inputs.
    for &node in &topo {
        if matches!(vg.kind(node), NodeKind::Transform | NodeKind::Sink) {
            let mut ups: Vec<NodeId> = Vec::new();
            for &ie in vg.in_edges(node) {
                if let Some(u) = nearest_upstream_arm(&vg, ie) {
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

    let mut arms: Vec<BoxFuture<'static, Result<u64, G2gError>>> = Vec::with_capacity(n + 1);
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
            let Some(GraphNode::Muxer(mux)) = element else {
                return Err(G2gError::CapsMismatch);
            };
            let out_tx = out_txs.pop().expect("muxer output edge");
            let mux_out_caps = solution[out_e[0]].clone();
            let pads: Vec<usize> =
                in_e.iter().map(|&eid| vg.edge(eid).dst.index as usize).collect();
            let input_count = in_rxs.len();
            // Per-input forwarders tag each packet with its pad and feed one
            // tagged channel; only they keep it open (the runner drops its end).
            let (tagged_tx, tagged_rx) = bounded::<(usize, PipelinePacket)>(link_capacity);
            for (in_rx, pad) in in_rxs.into_iter().zip(pads) {
                let fwd: BoxFuture<'static, Result<u64, G2gError>> =
                    Box::pin(muxer_forwarder(in_rx, pad, tagged_tx.clone()));
                arms.push(fwd);
                arm_kinds.push(kind);
            }
            drop(tagged_tx);
            let arm: BoxFuture<'static, Result<u64, G2gError>> =
                Box::pin(muxer_arm(mux, tagged_rx, out_tx, input_count, mux_out_caps));
            arms.push(arm);
            arm_kinds.push(kind);
            continue;
        }

        let arm: BoxFuture<'static, Result<u64, G2gError>> = match kind {
            NodeKind::Source => {
                let Some(GraphNode::Source(src)) = element else {
                    return Err(G2gError::CapsMismatch);
                };
                let out_tx = out_txs.pop().expect("source output edge");
                Box::pin(source_arm(src, out_tx))
            }
            NodeKind::Transform => {
                let Some(GraphNode::Element(elem)) = element else {
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
                ))
            }
            NodeKind::Sink => {
                let Some(GraphNode::Element(elem)) = element else {
                    return Err(G2gError::CapsMismatch);
                };
                let in_rx = in_rxs.pop().expect("sink input edge");
                Box::pin(sink_arm(elem, in_rx, coord_handle.clone(), node))
            }
            NodeKind::Tee(_) => {
                let in_rx = in_rxs.pop().expect("tee input edge");
                Box::pin(tee_arm(in_rx, out_txs))
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

    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        latency,
        allocation,
        clock_priority,
        base_time_ns,
        coordinator_events,
    })
}

/// View a node's payload as a transform/sink element. `None` for a source or a
/// muxer (whose constraints the runner builds from their own trait methods).
fn element_ref(vg: &ValidatedGraph<GraphNode>, node: NodeId) -> Option<&dyn DynAsyncElement> {
    match vg.element(node)? {
        GraphNode::Element(elem) => Some(&**elem),
        GraphNode::Source(_) | GraphNode::Muxer(_) => None,
    }
}

/// A transform/sink node's allocation proposal from `caps` (its output-link caps
/// for a transform, its input-link caps for a sink). `None` for other kinds.
fn element_propose(
    vg: &ValidatedGraph<GraphNode>,
    node: NodeId,
    caps: &Caps,
) -> Option<AllocationParams> {
    match vg.element(node) {
        Some(GraphNode::Element(elem)) => elem.propose_allocation(caps),
        _ => None,
    }
}

/// Apply a downstream-derived allocation proposal to a transform's own pool.
fn element_configure_alloc(
    vg: &mut ValidatedGraph<GraphNode>,
    node: NodeId,
    params: &AllocationParams,
) {
    if let Some(GraphNode::Element(elem)) = vg.element_mut(node) {
        elem.configure_allocation(params);
    }
}

/// A node's latency contribution. `None` for structural (tee) and muxer nodes.
fn element_latency(vg: &ValidatedGraph<GraphNode>, node: NodeId) -> Option<LatencyReport> {
    match vg.element(node) {
        Some(GraphNode::Source(src)) => Some(src.latency()),
        Some(GraphNode::Element(elem)) => Some(elem.latency()),
        _ => None,
    }
}

/// A node's offered clock for the pipeline clock election.
fn element_clock(vg: &ValidatedGraph<GraphNode>, node: NodeId) -> Option<ClockCandidate> {
    match vg.element(node) {
        Some(GraphNode::Source(src)) => src.provide_clock(),
        Some(GraphNode::Element(elem)) => elem.provide_clock(),
        _ => None,
    }
}

/// Join two allocation proposals at a tee's input: keep the most-demanding
/// (largest `size_bytes`). No test exercises a divergent tee allocation yet; a
/// full per-param intersection is a follow-up (the DAG plan's open question).
fn join_alloc(
    a: Option<AllocationParams>,
    b: Option<AllocationParams>,
) -> Option<AllocationParams> {
    match (a, b) {
        (Some(x), Some(y)) => Some(if y.size_bytes > x.size_bytes { y } else { x }),
        (Some(x), None) => Some(x),
        (None, b) => b,
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

async fn source_arm(
    mut src: Box<dyn DynSourceLoop>,
    out_tx: LinkSender,
) -> Result<u64, G2gError> {
    let mut adapter = SenderSink::new(out_tx);
    src.run(&mut adapter).await
}

/// An interior transform arm. Besides forwarding data, it (D4) selects on a β
/// `ArmDirective` channel alongside its data link so an upstream re-cascade
/// reaches it while parked on data, and on a mid-stream `CapsChanged` it steers
/// its forwarded output toward a downstream-acceptable shape using its
/// `downstream_feasible` snapshot (Caps-α), failing loud via a reverse
/// reconfigure if downstream positively rejects every output it can produce.
#[allow(clippy::too_many_arguments)]
async fn transform_arm(
    mut elem: Box<dyn DynAsyncElement>,
    in_rx: LinkReceiver,
    out_tx: LinkSender,
    arm_rx: Receiver<ArmDirective>,
    coord: GraphCoordHandle,
    node: NodeId,
    mut out_caps: Caps,
    downstream_feasible: Option<CapsSet>,
) -> Result<u64, G2gError> {
    let mut adapter = SenderSink::new(out_tx);
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
                    coord.report(Recascade { node, proposal }).await;
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
                        // Strict default (matches `run_source_fanout`): a branch
                        // whose downstream positively rejects every output this
                        // element can produce fails the whole graph loud. A
                        // graceful per-branch drop is a future opt-in.
                        ForwardResolve::Infeasible(_failure) => {
                            return Err(G2gError::CapsMismatch);
                        }
                    }
                };
                match elem.configure_pipeline(&new_caps)? {
                    ConfigureOutcome::Accepted => {
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
                elem.process(packet, &mut adapter).await?;
            }
            None => return Ok(0),
        }
    }
}

/// A sink arm. On a mid-stream `CapsChanged` (D4) it re-solves its input against
/// its declared constraint; on accept it re-derives its own pool and reports the
/// proposal so the β cascade walks one hop upstream. A re-solve failure surfaces
/// loud as a reverse reconfigure into the boundary that emitted the change.
async fn sink_arm(
    mut elem: Box<dyn DynAsyncElement>,
    in_rx: LinkReceiver,
    coord: GraphCoordHandle,
    node: NodeId,
) -> Result<u64, G2gError> {
    let mut null = NullSink;
    let mut consumed = 0u64;
    loop {
        match in_rx.recv().await {
            Some(PipelinePacket::Eos) => {
                elem.process(PipelinePacket::Eos, &mut null).await?;
                return Ok(consumed);
            }
            Some(PipelinePacket::CapsChanged(new_caps)) => {
                // Strict default (matches `run_source_fanout`): a sink whose
                // declared constraint rejects the boundary's new output fails
                // the whole graph loud rather than reverse-reconfiguring a
                // shared upstream a tee can't satisfy per-branch.
                let sink_caps = re_solve_downstream_dyn_sink(&new_caps, &*elem)
                    .map_err(|_| G2gError::CapsMismatch)?;
                match elem.configure_pipeline(&sink_caps)? {
                    ConfigureOutcome::Accepted => {
                        let proposal = elem.propose_allocation(&sink_caps);
                        if let Some(p) = &proposal {
                            elem.configure_allocation(p);
                        }
                        coord.report(Recascade { node, proposal }).await;
                        elem.process(PipelinePacket::CapsChanged(sink_caps), &mut null)
                            .await?;
                    }
                    ConfigureOutcome::ReFixate(counter) => {
                        in_rx.request_reconfigure(Reconfigure::Propose(counter));
                    }
                }
            }
            Some(packet) => {
                if matches!(packet, PipelinePacket::DataFrame(_)) {
                    consumed += 1;
                }
                elem.process(packet, &mut null).await?;
            }
            None => return Ok(consumed),
        }
    }
}

async fn tee_arm(in_rx: LinkReceiver, out_txs: Vec<LinkSender>) -> Result<u64, G2gError> {
    let mut senders: Vec<SenderSink> = out_txs.into_iter().map(SenderSink::new).collect();
    loop {
        match in_rx.recv().await {
            Some(PipelinePacket::Eos) => {
                for s in senders.iter_mut() {
                    s.push(PipelinePacket::Eos).await?;
                }
                return Ok(0);
            }
            Some(packet) => {
                broadcast(&mut senders, packet).await?;
            }
            None => return Ok(0),
        }
    }
}

/// Send `packet` to every tee branch. Clones it to all but the last branch and
/// moves the original into the last, so a fan-out of `n` makes `n - 1` copies.
async fn broadcast(senders: &mut [SenderSink], packet: PipelinePacket) -> Result<(), G2gError> {
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
    pad: usize,
    tagged: Sender<(usize, PipelinePacket)>,
) -> Result<u64, G2gError> {
    loop {
        match in_rx.recv().await {
            Some(PipelinePacket::Eos) | None => {
                tagged
                    .send((pad, PipelinePacket::Eos))
                    .await
                    .map_err(|_| G2gError::Shutdown)?;
                return Ok(0);
            }
            Some(packet) => {
                tagged
                    .send((pad, packet))
                    .await
                    .map_err(|_| G2gError::Shutdown)?;
            }
        }
    }
}

/// The muxer arm: drain the tagged channel, combine each input's packets via
/// `process(pad, ..)`, and emit a single `Eos` once every input has ended. The
/// per-input `Eos` is delivered to the element first (so a stateful muxer can
/// flush) but the element must not forward it; the runner owns the merged one.
async fn muxer_arm(
    mut mux: Box<dyn DynMultiInputElement>,
    tagged_rx: Receiver<(usize, PipelinePacket)>,
    out_tx: LinkSender,
    input_count: usize,
    mut current_output: Caps,
) -> Result<u64, G2gError> {
    let mut adapter = SenderSink::new(out_tx);
    let mut ended = 0usize;
    loop {
        match tagged_rx.recv().await {
            Some((pad, PipelinePacket::Eos)) => {
                mux.process(pad, PipelinePacket::Eos, &mut adapter).await?;
                ended += 1;
                if ended == input_count {
                    adapter.push(PipelinePacket::Eos).await?;
                    return Ok(0);
                }
            }
            Some((pad, PipelinePacket::CapsChanged(new_caps))) => {
                // MX-1: re-solve this input against its pad constraint and
                // reconfigure the pad; the input-side `CapsChanged` is consumed,
                // not forwarded as if it were the merged output. A muxer is a β
                // allocation boundary (its inputs have no per-pad re-cascade
                // channel).
                let input_caps = solve_mux_input_dyn(&new_caps, &*mux, pad)?;
                if let ConfigureOutcome::ReFixate(_) = mux.configure_pipeline(pad, &input_caps)? {
                    return Err(G2gError::FixationFailed);
                }
                // MX-2: the per-input change may shift the merged output. Emit one
                // downstream `CapsChanged` only when it actually changed.
                let new_output = solve_mux_output_dyn(&*mux)?;
                if new_output != current_output {
                    current_output = new_output.clone();
                    adapter.push(PipelinePacket::CapsChanged(new_output)).await?;
                }
            }
            Some((pad, packet)) => {
                mux.process(pad, packet, &mut adapter).await?;
            }
            None => return Ok(0),
        }
    }
}

/// Clone a packet for a tee branch. Control packets clone trivially; a
/// `System` data frame deep-copies its bytes. A GPU-resident frame owns a
/// non-copyable handle, so it fails loud (a refcounted shareable frame is the
/// zero-copy follow-up).
fn try_clone_packet(packet: &PipelinePacket) -> Result<PipelinePacket, G2gError> {
    Ok(match packet {
        PipelinePacket::CapsChanged(caps) => PipelinePacket::CapsChanged(caps.clone()),
        PipelinePacket::Eos => PipelinePacket::Eos,
        PipelinePacket::Flush => PipelinePacket::Flush,
        PipelinePacket::DataFrame(frame) => {
            let MemoryDomain::System(slice) = &frame.domain else {
                return Err(G2gError::UnsupportedDomain);
            };
            let bytes = slice.as_slice().to_vec().into_boxed_slice();
            PipelinePacket::DataFrame(Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(bytes)),
                timing: frame.timing,
                sequence: frame.sequence,
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameTiming;

    fn system_frame(bytes: &[u8], seq: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
            timing: FrameTiming { pts_ns: 7, ..FrameTiming::default() },
            sequence: seq,
        })
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
}
