//! DAG pipeline graph + validation (DESIGN_TODO "DAG runner" D1).
//!
//! `Graph<E>` is the builder for an arbitrary multimedia DAG: linear, fan-out
//! (tee), fan-in (muxer), and nested branches in one topology. It carries an
//! opaque element payload `E` per source/transform/sink node so it stays
//! `no_std` and independent of the std-gated runner; the runner instantiates
//! `Graph<Box<dyn DynAsyncElement>>`, embedded/wasm callers use their own.
//!
//! `finish()` runs the validation the runner relies on: every pad linked
//! exactly once, no cycles (Kahn topological sort), and the pad counts match
//! each node kind. The solver (D2) and runner (D3) consume the resulting
//! `ValidatedGraph`'s topological order and adjacency. This module is data and
//! computation only, no I/O.

use alloc::vec;
use alloc::vec::Vec;

use crate::link::LinkPolicy;

/// Opaque index of a node within a [`Graph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// A pad on a node. In an edge's source position the index selects an output
/// pad; in the destination position, an input pad. Most 1-in-1-out elements
/// use index 0 via `NodeId: Into<PadId>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadId {
    pub node: NodeId,
    pub index: u8,
}

impl From<NodeId> for PadId {
    fn from(node: NodeId) -> Self {
        PadId { node, index: 0 }
    }
}

/// The topology role of a node, which fixes its pad counts. `Tee(n)` is
/// 1-in/n-out, `Muxer(n)` is n-in/1-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Source,
    Transform,
    Sink,
    Tee(u8),
    Muxer(u8),
}

impl NodeKind {
    /// Number of input pads this kind exposes.
    pub fn in_pads(self) -> u8 {
        match self {
            NodeKind::Source => 0,
            NodeKind::Transform | NodeKind::Sink | NodeKind::Tee(_) => 1,
            NodeKind::Muxer(n) => n,
        }
    }

    /// Number of output pads this kind exposes.
    pub fn out_pads(self) -> u8 {
        match self {
            NodeKind::Sink => 0,
            NodeKind::Source | NodeKind::Transform | NodeKind::Muxer(_) => 1,
            NodeKind::Tee(n) => n,
        }
    }
}

/// Input vs output side of a pad, for error reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadDir {
    In,
    Out,
}

/// The `NodeId` shift applied when one graph is merged into another
/// ([`Graph::merge`]). Translates a node id, and the pad ids on it, from the
/// merged-in graph's local id space into the host graph's. The shift is the
/// host's node count at merge time, because nodes are a flat `Vec` indexed by
/// `NodeId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeIdOffset(u32);

impl NodeIdOffset {
    /// Translate a node id from the merged-in graph into the host graph.
    pub fn apply(self, node: NodeId) -> NodeId {
        NodeId(node.0 + self.0)
    }

    /// Translate a pad (re-base its node id; the pad index is unchanged).
    pub fn apply_pad(self, pad: PadId) -> PadId {
        PadId { node: self.apply(pad.node), index: pad.index }
    }
}

/// A directed link from an output pad (`src`) to an input pad (`dst`), with
/// its backpressure policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub src: PadId,
    pub dst: PadId,
    pub policy: LinkPolicy,
}

/// A tee handle returned by [`Graph::add_tee`]: 1 input pad, `n` output pads.
#[derive(Debug, Clone, Copy)]
pub struct Tee(NodeId);

impl Tee {
    pub fn node(self) -> NodeId {
        self.0
    }
    pub fn input(self) -> PadId {
        PadId { node: self.0, index: 0 }
    }
    pub fn out(self, index: u8) -> PadId {
        PadId { node: self.0, index }
    }
}

/// A demux handle returned by [`Graph::add_demux`]: 1 input pad, `n` output
/// pads. Structurally a tee (its node kind is `Tee(n)`), but the node carries a
/// content-routing element rather than broadcasting; see
/// [`GraphNodeRef::Demux`](crate::runtime::GraphNodeRef::Demux).
#[derive(Debug, Clone, Copy)]
pub struct Demux(NodeId);

impl Demux {
    pub fn node(self) -> NodeId {
        self.0
    }
    pub fn input(self) -> PadId {
        PadId { node: self.0, index: 0 }
    }
    pub fn out(self, index: u8) -> PadId {
        PadId { node: self.0, index }
    }
}

/// A muxer handle returned by [`Graph::add_muxer`]: `n` input pads, 1 output.
#[derive(Debug, Clone, Copy)]
pub struct Muxer(NodeId);

impl Muxer {
    pub fn node(self) -> NodeId {
        self.0
    }
    pub fn input(self, index: u8) -> PadId {
        PadId { node: self.0, index }
    }
    pub fn output(self) -> PadId {
        PadId { node: self.0, index: 0 }
    }
}

/// Validation failures from [`Graph::link`] and [`Graph::finish`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// A linked pad referenced a node id that doesn't exist.
    UnknownNode(NodeId),
    /// A pad index is past the node kind's pad count for that direction.
    PadOutOfRange { node: NodeId, index: u8, direction: PadDir },
    /// A pad has no link where the kind requires one.
    UnlinkedPad { node: NodeId, index: u8, direction: PadDir },
    /// A pad has more than one link (a pad peers with exactly one other pad;
    /// fan-out/in is expressed with `Tee`/`Muxer`, not multi-linked pads).
    PadCountMismatch { node: NodeId, index: u8, direction: PadDir },
    /// A node participates in no link at all.
    OrphanNode(NodeId),
    /// The graph has a cycle; the listed nodes are the unresolved set.
    Cycle { nodes: Vec<NodeId> },
    /// The same interior pad was exposed as a ghost pad twice on a [`Bin`]. A
    /// ghost pad peers 1:1 with one internal pad (as in GStreamer), so a pad can
    /// back at most one ghost.
    DuplicateGhostPad { node: NodeId, index: u8, direction: PadDir },
}

struct Node<E> {
    kind: NodeKind,
    /// `Some` for source/transform/sink; `None` for tee/muxer (runner shapes).
    element: Option<E>,
}

/// Builder for a multimedia DAG. Add nodes, link their pads, then `finish()`
/// to validate and produce a [`ValidatedGraph`].
pub struct Graph<E> {
    nodes: Vec<Node<E>>,
    edges: Vec<Edge>,
}

impl<E> Default for Graph<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> Graph<E> {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new() }
    }

    pub fn add_source(&mut self, element: E) -> NodeId {
        self.push(NodeKind::Source, Some(element))
    }

    pub fn add_transform(&mut self, element: E) -> NodeId {
        self.push(NodeKind::Transform, Some(element))
    }

    pub fn add_sink(&mut self, element: E) -> NodeId {
        self.push(NodeKind::Sink, Some(element))
    }

    pub fn add_tee(&mut self, outputs: u8) -> Tee {
        Tee(self.push(NodeKind::Tee(outputs), None))
    }

    pub fn add_muxer(&mut self, element: E, inputs: u8) -> Muxer {
        Muxer(self.push(NodeKind::Muxer(inputs), Some(element)))
    }

    /// Add a content-routing demultiplexer: 1 input, `outputs` outputs. The node
    /// is `Tee(outputs)`-shaped (so it validates and negotiates exactly like a
    /// tee, all outputs initially carrying the input caps) but carries a
    /// routing `element`; the runner drives it via
    /// [`GraphNodeRef::Demux`](crate::runtime::GraphNodeRef::Demux) and each
    /// branch retypes from a per-output `CapsChanged` at runtime (M210). Unlike
    /// `add_tee`, a demux carries an element payload.
    pub fn add_demux(&mut self, element: E, outputs: u8) -> Demux {
        Demux(self.push(NodeKind::Tee(outputs), Some(element)))
    }

    fn push(&mut self, kind: NodeKind, element: Option<E>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node { kind, element });
        id
    }

    /// Link an output pad to an input pad with the default `Block` policy.
    pub fn link(
        &mut self,
        from: impl Into<PadId>,
        to: impl Into<PadId>,
    ) -> Result<(), GraphError> {
        self.link_with(from, to, LinkPolicy::Block)
    }

    /// Link an output pad to an input pad with an explicit backpressure policy.
    pub fn link_with(
        &mut self,
        from: impl Into<PadId>,
        to: impl Into<PadId>,
        policy: LinkPolicy,
    ) -> Result<(), GraphError> {
        let (src, dst) = (from.into(), to.into());
        self.check_pad(src, PadDir::Out)?;
        self.check_pad(dst, PadDir::In)?;
        self.edges.push(Edge { src, dst, policy });
        Ok(())
    }

    /// The edges in declaration order, including each one's backpressure
    /// [`LinkPolicy`]. Lets callers inspect the wiring before [`finish`](Self::finish)
    /// (e.g. the launch parser's `queue`-to-policy mapping).
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Number of nodes added so far. With [`edges`](Self::edges) and
    /// [`node_kind`](Self::node_kind) this is enough to render the wiring
    /// before validation (the DOT dump).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The [`NodeKind`] of a node, or `None` if the id is past the node count.
    pub fn node_kind(&self, node: NodeId) -> Option<NodeKind> {
        self.nodes.get(node.0 as usize).map(|n| n.kind)
    }

    /// Borrow a node's element payload (`None` for tee nodes or an unknown id),
    /// for labeling a pre-validation dump from the element itself.
    pub fn element(&self, node: NodeId) -> Option<&E> {
        self.nodes.get(node.0 as usize).and_then(|n| n.element.as_ref())
    }

    /// Append every node and edge of `inner` into this graph, returning the
    /// [`NodeIdOffset`] that maps `inner`'s ids into this graph's id space.
    /// Composition is a pure index shift: nodes are a flat `Vec` and edges carry
    /// only pad indices, so re-basing `inner`'s ids by the current node count is
    /// all it takes. The union is not re-validated here; the host's `finish()`
    /// validates the whole. This is the one primitive under bin flattening
    /// ([`add_bin`](Self::add_bin)) and the decodebin / uridecodebin / autoplug
    /// splices.
    pub fn merge(&mut self, inner: Graph<E>) -> NodeIdOffset {
        let offset = NodeIdOffset(self.nodes.len() as u32);
        self.nodes.extend(inner.nodes);
        for e in inner.edges {
            self.edges.push(Edge {
                src: offset.apply_pad(e.src),
                dst: offset.apply_pad(e.dst),
                policy: e.policy,
            });
        }
        offset
    }

    /// Flatten `bin` into this graph, returning a [`BinInstance`] whose ghost pads
    /// are this graph's pad ids: link them like any other pad
    /// (`graph.link(src, inst.input(0))`, `graph.link(inst.output(0), dst)`).
    /// Construction-time only, no new node kind, so the solver and runner see the
    /// flattened union with no awareness the bin ever existed.
    pub fn add_bin(&mut self, bin: Bin<E>) -> BinInstance {
        let Bin { graph, ghost_in, ghost_out } = bin;
        let offset = self.merge(graph);
        BinInstance {
            ghost_in: ghost_in.into_iter().map(|p| offset.apply_pad(p)).collect(),
            ghost_out: ghost_out.into_iter().map(|p| offset.apply_pad(p)).collect(),
        }
    }

    fn kind_of(&self, node: NodeId) -> Result<NodeKind, GraphError> {
        self.nodes
            .get(node.0 as usize)
            .map(|n| n.kind)
            .ok_or(GraphError::UnknownNode(node))
    }

    fn check_pad(&self, pad: PadId, direction: PadDir) -> Result<(), GraphError> {
        let kind = self.kind_of(pad.node)?;
        let count = match direction {
            PadDir::In => kind.in_pads(),
            PadDir::Out => kind.out_pads(),
        };
        if pad.index >= count {
            return Err(GraphError::PadOutOfRange { node: pad.node, index: pad.index, direction });
        }
        Ok(())
    }

    /// Validate the graph and compute its topological order + adjacency.
    pub fn finish(self) -> Result<ValidatedGraph<E>, GraphError> {
        let n = self.nodes.len();
        let mut in_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut out_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (eid, e) in self.edges.iter().enumerate() {
            out_edges[e.src.node.0 as usize].push(eid);
            in_edges[e.dst.node.0 as usize].push(eid);
        }

        for (i, node) in self.nodes.iter().enumerate() {
            let id = NodeId(i as u32);
            if in_edges[i].is_empty() && out_edges[i].is_empty() {
                return Err(GraphError::OrphanNode(id));
            }
            check_pads(
                node.kind.in_pads(),
                in_edges[i].iter().map(|&e| self.edges[e].dst.index),
                id,
                PadDir::In,
            )?;
            check_pads(
                node.kind.out_pads(),
                out_edges[i].iter().map(|&e| self.edges[e].src.index),
                id,
                PadDir::Out,
            )?;
        }

        let topo = topo_sort(n, &in_edges, &out_edges, &self.edges)?;
        Ok(ValidatedGraph { nodes: self.nodes, edges: self.edges, topo, in_edges, out_edges })
    }
}

impl<E> core::fmt::Debug for Graph<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kinds: Vec<NodeKind> = self.nodes.iter().map(|n| n.kind).collect();
        f.debug_struct("Graph")
            .field("nodes", &kinds)
            .field("edges", &self.edges)
            .finish()
    }
}

/// A reusable subgraph with designated ghost pads, flattened into a host graph
/// by [`Graph::add_bin`]. Build its interior with the same `add_*` / `link` calls
/// as a [`Graph`], then expose interior boundary pads as ghost pads (the bin's
/// external pads, in designation order).
///
/// A bin is never validated on its own: its ghost pads are intentionally
/// unlinked inside the bin and get their peer only when the host graph links the
/// returned [`BinInstance`], so the host's `finish()` is what validates. This is
/// pure construction-time encapsulation, no new [`NodeKind`]: the bin's nodes
/// become first-class host nodes on flattening (DESIGN.md the bins section).
pub struct Bin<E> {
    graph: Graph<E>,
    ghost_in: Vec<PadId>,
    ghost_out: Vec<PadId>,
}

impl<E> Default for Bin<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> Bin<E> {
    pub fn new() -> Self {
        Self { graph: Graph::new(), ghost_in: Vec::new(), ghost_out: Vec::new() }
    }

    pub fn add_source(&mut self, element: E) -> NodeId {
        self.graph.add_source(element)
    }

    pub fn add_transform(&mut self, element: E) -> NodeId {
        self.graph.add_transform(element)
    }

    pub fn add_sink(&mut self, element: E) -> NodeId {
        self.graph.add_sink(element)
    }

    pub fn add_tee(&mut self, outputs: u8) -> Tee {
        self.graph.add_tee(outputs)
    }

    pub fn add_muxer(&mut self, element: E, inputs: u8) -> Muxer {
        self.graph.add_muxer(element, inputs)
    }

    pub fn link(
        &mut self,
        from: impl Into<PadId>,
        to: impl Into<PadId>,
    ) -> Result<(), GraphError> {
        self.graph.link(from, to)
    }

    pub fn link_with(
        &mut self,
        from: impl Into<PadId>,
        to: impl Into<PadId>,
        policy: LinkPolicy,
    ) -> Result<(), GraphError> {
        self.graph.link_with(from, to, policy)
    }

    /// Expose an interior input pad as the bin's next ghost input pad. The pad
    /// must be a real input pad on a node in this bin, and not already a ghost.
    pub fn ghost_input(&mut self, interior: impl Into<PadId>) -> Result<(), GraphError> {
        let pad = interior.into();
        self.graph.check_pad(pad, PadDir::In)?;
        if self.ghost_in.contains(&pad) {
            return Err(GraphError::DuplicateGhostPad {
                node: pad.node,
                index: pad.index,
                direction: PadDir::In,
            });
        }
        self.ghost_in.push(pad);
        Ok(())
    }

    /// Expose an interior output pad as the bin's next ghost output pad. The pad
    /// must be a real output pad on a node in this bin, and not already a ghost.
    pub fn ghost_output(&mut self, interior: impl Into<PadId>) -> Result<(), GraphError> {
        let pad = interior.into();
        self.graph.check_pad(pad, PadDir::Out)?;
        if self.ghost_out.contains(&pad) {
            return Err(GraphError::DuplicateGhostPad {
                node: pad.node,
                index: pad.index,
                direction: PadDir::Out,
            });
        }
        self.ghost_out.push(pad);
        Ok(())
    }
}

impl<E> core::fmt::Debug for Bin<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Bin")
            .field("graph", &self.graph)
            .field("ghost_in", &self.ghost_in)
            .field("ghost_out", &self.ghost_out)
            .finish()
    }
}

/// A [`Bin`] flattened into a host graph: its ghost pads, as host-graph pad ids.
/// Link these like any pad to wire the bin into the surrounding graph.
#[derive(Debug, Clone)]
pub struct BinInstance {
    ghost_in: Vec<PadId>,
    ghost_out: Vec<PadId>,
}

impl BinInstance {
    /// The bin's `i`th ghost input pad, in host-graph space.
    pub fn input(&self, i: usize) -> PadId {
        self.ghost_in[i]
    }

    /// The bin's `i`th ghost output pad, in host-graph space.
    pub fn output(&self, i: usize) -> PadId {
        self.ghost_out[i]
    }

    /// Number of ghost input pads the bin exposes.
    pub fn input_count(&self) -> usize {
        self.ghost_in.len()
    }

    /// Number of ghost output pads the bin exposes.
    pub fn output_count(&self) -> usize {
        self.ghost_out.len()
    }
}

/// A validated DAG: every pad linked once, acyclic, pad counts consistent.
/// Carries the topological node order and per-node edge adjacency for the
/// solver and runner.
pub struct ValidatedGraph<E> {
    nodes: Vec<Node<E>>,
    edges: Vec<Edge>,
    topo: Vec<NodeId>,
    in_edges: Vec<Vec<usize>>,
    out_edges: Vec<Vec<usize>>,
}

impl<E> ValidatedGraph<E> {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Nodes in topological order (every node appears after all its inputs).
    pub fn topo(&self) -> &[NodeId] {
        &self.topo
    }

    pub fn kind(&self, node: NodeId) -> NodeKind {
        self.nodes[node.0 as usize].kind
    }

    pub fn edge(&self, id: usize) -> &Edge {
        &self.edges[id]
    }

    /// All edges, indexed by edge id (the same index the solver's `Vec<Caps>`
    /// solution and the DOT renderer's per-edge annotations use).
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// Edge ids feeding this node's input pads.
    pub fn in_edges(&self, node: NodeId) -> &[usize] {
        &self.in_edges[node.0 as usize]
    }

    /// Edge ids leaving this node's output pads.
    pub fn out_edges(&self, node: NodeId) -> &[usize] {
        &self.out_edges[node.0 as usize]
    }

    /// Take the element payload out of a node (the runner moves each element
    /// into its spawned arm). `None` for tee/muxer nodes or after a prior take.
    pub fn take_element(&mut self, node: NodeId) -> Option<E> {
        self.nodes[node.0 as usize].element.take()
    }

    /// Borrow a node's element payload, for building its negotiation
    /// constraint before the runner takes it. `None` for tee/muxer nodes.
    pub fn element(&self, node: NodeId) -> Option<&E> {
        self.nodes[node.0 as usize].element.as_ref()
    }

    /// Mutably borrow a node's element payload, for the async source caps
    /// probe and per-node `configure_pipeline`. `None` for tee/muxer nodes.
    pub fn element_mut(&mut self, node: NodeId) -> Option<&mut E> {
        self.nodes[node.0 as usize].element.as_mut()
    }
}

impl<E> core::fmt::Debug for ValidatedGraph<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kinds: Vec<NodeKind> = self.nodes.iter().map(|n| n.kind).collect();
        f.debug_struct("ValidatedGraph")
            .field("nodes", &kinds)
            .field("edges", &self.edges)
            .field("topo", &self.topo)
            .finish()
    }
}

/// Each pad index in `0..count` must be referenced by exactly one edge.
fn check_pads(
    count: u8,
    indices: impl Iterator<Item = u8>,
    node: NodeId,
    direction: PadDir,
) -> Result<(), GraphError> {
    let mut seen = vec![0u32; count as usize];
    for idx in indices {
        // link() range-checks pad indices, so idx is always in range here.
        seen[idx as usize] += 1;
    }
    for (idx, &c) in seen.iter().enumerate() {
        let index = idx as u8;
        if c == 0 {
            return Err(GraphError::UnlinkedPad { node, index, direction });
        }
        if c > 1 {
            return Err(GraphError::PadCountMismatch { node, index, direction });
        }
    }
    Ok(())
}

/// Kahn's algorithm: repeatedly remove zero-in-degree nodes. If fewer than `n`
/// come out, the remainder is a cycle.
fn topo_sort(
    n: usize,
    in_edges: &[Vec<usize>],
    out_edges: &[Vec<usize>],
    edges: &[Edge],
) -> Result<Vec<NodeId>, GraphError> {
    let mut indeg: Vec<usize> = in_edges.iter().map(|e| e.len()).collect();
    let mut queue: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut topo: Vec<NodeId> = Vec::with_capacity(n);
    let mut processed = vec![false; n];

    let mut head = 0;
    while head < queue.len() {
        let node = queue[head];
        head += 1;
        processed[node] = true;
        topo.push(NodeId(node as u32));
        for &eid in &out_edges[node] {
            let succ = edges[eid].dst.node.0 as usize;
            indeg[succ] -= 1;
            if indeg[succ] == 0 {
                queue.push(succ);
            }
        }
    }

    if topo.len() < n {
        let nodes = (0..n)
            .filter(|&i| !processed[i])
            .map(|i| NodeId(i as u32))
            .collect();
        return Err(GraphError::Cycle { nodes });
    }
    Ok(topo)
}

#[cfg(test)]
mod tests {
    use super::*;

    // element payload: a label, so tests read clearly.
    type G = Graph<&'static str>;

    #[test]
    fn linear_chain_validates_in_topo_order() {
        let mut g = G::new();
        let src = g.add_source("src");
        let tx = g.add_transform("tx");
        let sink = g.add_sink("sink");
        g.link(src, tx).unwrap();
        g.link(tx, sink).unwrap();
        let v = g.finish().expect("linear chain validates");
        assert_eq!(v.topo(), &[src, tx, sink]);
        assert_eq!(v.in_edges(src).len(), 0);
        assert_eq!(v.out_edges(sink).len(), 0);
    }

    #[test]
    fn fan_out_through_tee_validates() {
        let mut g = G::new();
        let src = g.add_source("src");
        let tee = g.add_tee(2);
        let a = g.add_sink("a");
        let b = g.add_sink("b");
        g.link(src, tee.input()).unwrap();
        g.link(tee.out(0), a).unwrap();
        g.link(tee.out(1), b).unwrap();
        let v = g.finish().expect("fan-out validates");
        assert_eq!(v.out_edges(tee.node()).len(), 2);
        // src precedes the tee precedes both sinks.
        let pos = |n: NodeId| v.topo().iter().position(|&x| x == n).unwrap();
        assert!(pos(src) < pos(tee.node()));
        assert!(pos(tee.node()) < pos(a) && pos(tee.node()) < pos(b));
    }

    #[test]
    fn fan_in_through_muxer_validates() {
        let mut g = G::new();
        let s0 = g.add_source("s0");
        let s1 = g.add_source("s1");
        let mux = g.add_muxer("mux", 2);
        let sink = g.add_sink("sink");
        g.link(s0, mux.input(0)).unwrap();
        g.link(s1, mux.input(1)).unwrap();
        g.link(mux.output(), sink).unwrap();
        let v = g.finish().expect("fan-in validates");
        assert_eq!(v.in_edges(mux.node()).len(), 2);
    }

    #[test]
    fn tee_to_muxer_diamond_validates() {
        let mut g = G::new();
        let src = g.add_source("src");
        let tee = g.add_tee(2);
        let a = g.add_transform("a");
        let b = g.add_transform("b");
        let mux = g.add_muxer("mux", 2);
        let sink = g.add_sink("sink");
        g.link(src, tee.input()).unwrap();
        g.link(tee.out(0), a).unwrap();
        g.link(tee.out(1), b).unwrap();
        g.link(a, mux.input(0)).unwrap();
        g.link(b, mux.input(1)).unwrap();
        g.link(mux.output(), sink).unwrap();
        let v = g.finish().expect("diamond validates");
        assert_eq!(v.node_count(), 6);
        let pos = |n: NodeId| v.topo().iter().position(|&x| x == n).unwrap();
        assert!(pos(a) < pos(mux.node()) && pos(b) < pos(mux.node()));
    }

    #[test]
    fn cycle_is_rejected() {
        // a -> b -> a: each pad linked once, but no zero-in-degree node.
        let mut g = G::new();
        let a = g.add_transform("a");
        let b = g.add_transform("b");
        g.link(a, b).unwrap();
        g.link(b, a).unwrap();
        match g.finish() {
            Err(GraphError::Cycle { nodes }) => {
                assert_eq!(nodes.len(), 2);
                assert!(nodes.contains(&a) && nodes.contains(&b));
            }
            other => panic!("expected Cycle, got {other:?}"),
        }
    }

    #[test]
    fn unlinked_pad_is_rejected() {
        // tee with one output left dangling.
        let mut g = G::new();
        let src = g.add_source("src");
        let tee = g.add_tee(2);
        let a = g.add_sink("a");
        g.link(src, tee.input()).unwrap();
        g.link(tee.out(0), a).unwrap();
        match g.finish() {
            Err(GraphError::UnlinkedPad { node, index, direction }) => {
                assert_eq!((node, index, direction), (tee.node(), 1, PadDir::Out));
            }
            other => panic!("expected UnlinkedPad, got {other:?}"),
        }
    }

    #[test]
    fn double_linked_pad_is_rejected() {
        // a sink's single input pad linked from two sources.
        let mut g = G::new();
        let s0 = g.add_source("s0");
        let s1 = g.add_source("s1");
        let sink = g.add_sink("sink");
        g.link(s0, sink).unwrap();
        g.link(s1, sink).unwrap();
        match g.finish() {
            Err(GraphError::PadCountMismatch { node, index, direction }) => {
                assert_eq!((node, index, direction), (sink, 0, PadDir::In));
            }
            other => panic!("expected PadCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn orphan_node_is_rejected() {
        let mut g = G::new();
        let src = g.add_source("src");
        let sink = g.add_sink("sink");
        let _orphan = g.add_transform("orphan");
        g.link(src, sink).unwrap();
        assert_eq!(g.finish().err(), Some(GraphError::OrphanNode(NodeId(2))));
    }

    #[test]
    fn pad_index_out_of_range_is_rejected_at_link() {
        let mut g = G::new();
        let src = g.add_source("src");
        let tee = g.add_tee(2);
        let s = g.add_sink("s");
        g.link(src, tee.input()).unwrap();
        // tee(2) has output pads 0 and 1; pad 2 is out of range.
        assert_eq!(
            g.link(tee.out(2), s).err(),
            Some(GraphError::PadOutOfRange { node: tee.node(), index: 2, direction: PadDir::Out })
        );
    }

    #[test]
    fn take_element_moves_payload_once() {
        let mut g = G::new();
        let src = g.add_source("src");
        let sink = g.add_sink("sink");
        g.link(src, sink).unwrap();
        let mut v = g.finish().unwrap();
        assert_eq!(v.take_element(src), Some("src"));
        assert_eq!(v.take_element(src), None, "payload taken only once");
    }

    #[test]
    fn merge_offsets_node_ids_and_edges() {
        // Host already has one node; merging a 2-node linked graph must re-base
        // the merged ids past it and carry the interior edge across.
        let mut host = G::new();
        let h0 = host.add_source("h0");
        assert_eq!(h0, NodeId(0));

        let mut inner = G::new();
        let i0 = inner.add_transform("i0");
        let i1 = inner.add_sink("i1");
        inner.link(i0, i1).unwrap();

        let off = host.merge(inner);
        // The first inner node lands right after the host's nodes.
        assert_eq!(off.apply(i0), NodeId(1));
        assert_eq!(off.apply(i1), NodeId(2));
        // Link host source into the merged transform; the interior edge survived.
        host.link(h0, off.apply(i0)).unwrap();
        let v = host.finish().expect("merged graph validates");
        assert_eq!(v.node_count(), 3);
        assert_eq!(v.topo(), &[NodeId(0), NodeId(1), NodeId(2)]);
        // The merged transform's element payload moved across intact.
        assert_eq!(v.element(NodeId(1)), Some(&"i0"));
    }

    #[test]
    fn add_bin_flattens_with_ghost_pads() {
        // A bin wrapping transform -> transform, exposing the first's input and
        // the second's output as ghost pads, flattens into source -> bin -> sink.
        let mut bin: Bin<&'static str> = Bin::new();
        let a = bin.add_transform("a");
        let b = bin.add_transform("b");
        bin.link(a, b).unwrap();
        bin.ghost_input(a).unwrap();
        bin.ghost_output(b).unwrap();

        let mut g = G::new();
        let src = g.add_source("src");
        let sink = g.add_sink("sink");
        let inst = g.add_bin(bin);
        assert_eq!(inst.input_count(), 1);
        assert_eq!(inst.output_count(), 1);
        g.link(src, inst.input(0)).unwrap();
        g.link(inst.output(0), sink).unwrap();

        let v = g.finish().expect("flattened bin validates");
        // The bin's two interior nodes are now first-class host nodes.
        assert_eq!(v.node_count(), 4);
        let pos = |n: NodeId| v.topo().iter().position(|&x| x == n).unwrap();
        assert!(pos(src) < pos(inst.input(0).node));
        assert!(pos(inst.output(0).node) < pos(sink));
    }

    #[test]
    fn bin_ghosts_an_interior_tee_output() {
        // Ghost pads can expose a specific pad index, e.g. one branch of a tee.
        let mut bin: Bin<&'static str> = Bin::new();
        let tx = bin.add_transform("tx");
        let tee = bin.add_tee(2);
        bin.link(tx, tee.input()).unwrap();
        bin.ghost_input(tx).unwrap();
        bin.ghost_output(tee.out(0)).unwrap();
        bin.ghost_output(tee.out(1)).unwrap();

        let mut g = G::new();
        let src = g.add_source("src");
        let a = g.add_sink("a");
        let b = g.add_sink("b");
        let inst = g.add_bin(bin);
        g.link(src, inst.input(0)).unwrap();
        g.link(inst.output(0), a).unwrap();
        g.link(inst.output(1), b).unwrap();
        let v = g.finish().expect("bin with a ghosted tee validates");
        // Both ghost outputs map onto the same interior tee node, distinct pads.
        assert_eq!(inst.output(0).node, inst.output(1).node);
        assert_ne!(inst.output(0).index, inst.output(1).index);
        assert_eq!(v.out_edges(inst.output(0).node).len(), 2);
    }

    #[test]
    fn duplicate_ghost_pad_is_rejected() {
        let mut bin: Bin<&'static str> = Bin::new();
        let a = bin.add_transform("a");
        bin.ghost_output(a).unwrap();
        assert_eq!(
            bin.ghost_output(a),
            Err(GraphError::DuplicateGhostPad { node: a, index: 0, direction: PadDir::Out }),
            "the same interior pad cannot back two ghosts",
        );
    }

    #[test]
    fn ghost_pad_out_of_range_is_rejected() {
        let mut bin: Bin<&'static str> = Bin::new();
        let a = bin.add_transform("a");
        // A transform has one output pad (index 0); index 1 is out of range.
        assert!(matches!(
            bin.ghost_output(PadId { node: a, index: 1 }),
            Err(GraphError::PadOutOfRange { .. }),
        ));
    }
}
