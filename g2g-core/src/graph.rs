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
}
