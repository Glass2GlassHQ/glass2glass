//! Live pipeline telemetry tap (dev tooling).
//!
//! The end-of-run [`RunStats`](crate::runtime::RunStats) report answers "how did
//! the run go" after it finishes. This tap answers "how is it going" while it
//! runs: an [`Observer`] handed to
//! [`run_graph_observed`](crate::runtime::run_graph_observed) captures the graph
//! topology and shares the per-element probes, so a concurrent task (a WebSocket
//! server, a TUI) can call [`Observer::snapshot`] at any time and read the live
//! per-element `process()` latency and input-link fill. The probes are the same
//! lock-free atomics the end-of-run report reads, so a snapshot mid-run costs a
//! handful of relaxed loads and never stalls an arm.
//!
//! std-only: it rides the graph runner, which is `std`-gated, and measured
//! timing needs the monotonic clock. Events (caps changes, errors, EOS, QoS,
//! buffering) already flow on the [`Bus`](crate::bus::Bus); the transport pairs a
//! bus with an observer rather than duplicating the event channel here.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use spin::Mutex;

use crate::caps::Caps;
use crate::graph::NodeKind;
use crate::runtime::channel::ProbeSlot;
use crate::runtime::instrument::Probe;
use crate::runtime::ElementLatency;

/// The topology role of a node: the serialization-friendly projection of
/// [`NodeKind`], dropping the tee / muxer pad counts the topology view carries
/// on the edges instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    Source,
    Transform,
    Sink,
    Tee,
    Muxer,
}

impl From<NodeKind> for NodeRole {
    fn from(k: NodeKind) -> Self {
        match k {
            NodeKind::Source => NodeRole::Source,
            NodeKind::Transform => NodeRole::Transform,
            NodeKind::Sink => NodeRole::Sink,
            NodeKind::Tee(_) => NodeRole::Tee,
            NodeKind::Muxer(_) | NodeKind::FaninSink(_) => NodeRole::Muxer,
        }
    }
}

/// A live handle onto a running graph's telemetry. Cloneable (clones share one
/// `Arc` of state): hand a clone to
/// [`run_graph_observed`](crate::runtime::run_graph_observed) and keep one to
/// poll [`snapshot`](Self::snapshot) from another task.
#[derive(Debug, Clone)]
pub struct Observer {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    start_ns: u64,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    /// Per node id, aligned with the graph's `NodeId` index space. Empty until
    /// the runner registers.
    names: Vec<String>,
    roles: Vec<NodeRole>,
    /// Per node id; `None` for a node without a `process()` probe (source / tee /
    /// muxer) or one the runner did not instrument.
    probes: Vec<Probe>,
    edges: Vec<EdgeInfo>,
    /// Per edge id (aligned with `edges`): the link's content-inspection slot and
    /// its negotiated caps, for the edge-content preview tap. Empty until the
    /// runner registers them (after channels are built).
    edge_probes: Vec<ProbeSlot>,
    edge_caps: Vec<Caps>,
}

impl Observer {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                start_ns: crate::metrics::monotonic_ns(),
                state: Mutex::new(State::default()),
            }),
        }
    }

    /// Install the graph's topology and probe set. Called once by the runner
    /// after negotiation, before any frame flows. `names`, `roles`, and `probes`
    /// are all indexed by `NodeId`; `probes` holds clones of the arms' `Arc`s, so
    /// reads see live counters.
    pub(crate) fn register(
        &self,
        names: Vec<String>,
        roles: Vec<NodeRole>,
        probes: Vec<Probe>,
        edges: Vec<EdgeInfo>,
    ) {
        let mut s = self.inner.state.lock();
        s.names = names;
        s.roles = roles;
        s.probes = probes;
        s.edges = edges;
    }

    /// Install the per-edge content-inspection slots + negotiated caps, aligned
    /// with the edges registered above. Called by the runner after the channels
    /// are built; separate from [`register`](Self::register) because the slots
    /// live on the links, which are created after negotiation.
    pub(crate) fn register_edges(&self, edge_probes: Vec<ProbeSlot>, edge_caps: Vec<Caps>) {
        let mut s = self.inner.state.lock();
        s.edge_probes = edge_probes;
        s.edge_caps = edge_caps;
    }

    /// The content-inspection slot for edge `idx`, for installing a
    /// [`LinkInterceptor`](crate::runtime::LinkInterceptor) that samples packets
    /// crossing that edge. `None` if the index is out of range.
    pub fn edge_probe(&self, idx: usize) -> Option<ProbeSlot> {
        self.inner.state.lock().edge_probes.get(idx).cloned()
    }

    /// The negotiated caps on edge `idx` (so a preview tap knows how to interpret
    /// the bytes). `None` if the index is out of range.
    pub fn edge_caps(&self, idx: usize) -> Option<Caps> {
        self.inner.state.lock().edge_caps.get(idx).cloned()
    }

    /// Number of edges registered (0 before the runner registers them).
    pub fn edge_count(&self) -> usize {
        self.inner.state.lock().edges.len()
    }

    /// A read of the current telemetry. Cheap: relaxed atomic loads off the
    /// shared probes plus a clone of the small topology vectors. An empty
    /// snapshot (no nodes) before the runner has registered.
    pub fn snapshot(&self) -> TelemetrySnapshot {
        let s = self.inner.state.lock();
        let nodes = s
            .names
            .iter()
            .zip(s.roles.iter())
            .zip(s.probes.iter())
            .enumerate()
            .map(|(id, ((name, role), probe))| NodeTelemetry {
                id,
                name: name.clone(),
                role: *role,
                latency: probe.as_ref().map(|p| p.snapshot()),
            })
            .collect();
        // Fill each edge's negotiated caps from the aligned `edge_caps` (present
        // once the runner has registered them, after negotiation).
        let edges = s
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| EdgeInfo {
                from: e.from,
                to: e.to,
                caps: s.edge_caps.get(i).map(|c| c.to_gst_string()),
            })
            .collect();
        TelemetrySnapshot {
            uptime_ns: crate::metrics::monotonic_ns().saturating_sub(self.inner.start_ns),
            nodes,
            edges,
        }
    }
}

impl Default for Observer {
    fn default() -> Self {
        Self::new()
    }
}

/// A point-in-time read of a running graph's telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TelemetrySnapshot {
    /// Nanoseconds since the observer was created.
    pub uptime_ns: u64,
    /// One entry per graph node, in `NodeId` order.
    pub nodes: Vec<NodeTelemetry>,
    /// The graph's directed links.
    pub edges: Vec<EdgeInfo>,
}

/// Per-node live telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeTelemetry {
    /// The node's `NodeId` index.
    pub id: usize,
    /// Instance name (`<category>N`), or empty for an unnamed structural node.
    pub name: String,
    pub role: NodeRole,
    /// Measured `process()` latency + input-link fill. `None` for a node without
    /// a probe (source / tee / muxer); the inner `proc.count` is `0` when no
    /// clock has yet timed a frame.
    pub latency: Option<ElementLatency>,
}

/// A directed link, by node index, with its negotiated caps (the `to_gst_string`
/// of the solved per-edge `Caps`). `caps` is `None` until the runner registers
/// the negotiated solution, and in a topology-only `EdgeInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdgeInfo {
    pub from: usize,
    pub to: usize,
    pub caps: Option<alloc::string::String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::instrument::ElementProbe;

    #[test]
    fn snapshot_before_register_is_empty() {
        let obs = Observer::new();
        let snap = obs.snapshot();
        assert!(snap.nodes.is_empty());
        assert!(snap.edges.is_empty());
    }

    #[test]
    fn snapshot_reflects_live_probe_writes() {
        let obs = Observer::new();
        let probe = ElementProbe::new(String::from("decode0"));
        obs.register(
            alloc::vec![String::from("src0"), String::from("decode0")],
            alloc::vec![NodeRole::Source, NodeRole::Transform],
            alloc::vec![None, Some(probe.clone())],
            alloc::vec![EdgeInfo {
                from: 0,
                to: 1,
                caps: None
            }],
        );

        // A read taken before any work: the transform's probe exists but is empty.
        let before = obs.snapshot();
        assert_eq!(before.nodes.len(), 2);
        assert_eq!(before.nodes[0].role, NodeRole::Source);
        assert!(before.nodes[0].latency.is_none(), "source has no probe");
        assert_eq!(before.nodes[1].latency.as_ref().unwrap().proc.count, 0);

        // Simulate the arm doing work, then read again through the same handle.
        probe.record_fill(80);
        probe.record_fill(100);
        let after = obs.snapshot();
        let lat = after.nodes[1].latency.as_ref().unwrap();
        assert_eq!(lat.fill_max_pct, 100);
        assert!(lat.fill_mean_pct > 0);
        assert_eq!(
            after.edges,
            alloc::vec![EdgeInfo {
                from: 0,
                to: 1,
                caps: None
            }]
        );
    }

    #[test]
    fn node_role_projects_kind() {
        assert_eq!(NodeRole::from(NodeKind::Tee(3)), NodeRole::Tee);
        assert_eq!(NodeRole::from(NodeKind::Muxer(2)), NodeRole::Muxer);
        assert_eq!(NodeRole::from(NodeKind::Sink), NodeRole::Sink);
    }
}
