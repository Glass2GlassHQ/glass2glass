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
use crate::runtime::instrument::{EdgeCounters, EdgeCounts, Probe, StageVisit};
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
            NodeKind::FanoutSrc(_) => NodeRole::Source,
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
    /// Per edge id: the link's live packet / byte / drop counters. `None` for an
    /// edge the runner did not instrument.
    edge_counters: Vec<Option<Arc<EdgeCounters>>>,
    /// The graph-wide default link depth, for the queueing floor a measured
    /// journey is compared against. `0` until the runner registers it.
    link_capacity: usize,
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

    /// Install the per-edge content-inspection slots, negotiated caps, and live
    /// traffic counters, aligned with the edges registered above. Called by the
    /// runner after the channels are built; separate from
    /// [`register`](Self::register) because all three live on the links, which
    /// are created after negotiation.
    pub(crate) fn register_edges(
        &self,
        edge_probes: Vec<ProbeSlot>,
        edge_caps: Vec<Caps>,
        edge_counters: Vec<Option<Arc<EdgeCounters>>>,
    ) {
        let mut s = self.inner.state.lock();
        s.edge_probes = edge_probes;
        s.edge_caps = edge_caps;
        s.edge_counters = edge_counters;
    }

    /// Append a node that appeared after [`register`](Self::register): a fan-out
    /// branch or a fan-in input attached while the run was going (M869). Returns
    /// its `NodeId`. The three per-node vectors are pushed under one lock hold,
    /// so a concurrent [`snapshot`](Self::snapshot) sees the node whole or not at
    /// all, never half of it.
    pub(crate) fn add_node(&self, name: String, role: NodeRole, probe: Probe) -> usize {
        let mut s = self.inner.state.lock();
        let id = s.names.len();
        s.names.push(name);
        s.roles.push(role);
        s.probes.push(probe);
        id
    }

    /// Append the link of a node registered by [`add_node`](Self::add_node),
    /// with its negotiated caps and taps. The incremental analog of
    /// [`register_edges`](Self::register_edges), under the same single lock hold.
    pub(crate) fn add_edge(&self, from: usize, to: usize, caps: Caps, tap: EdgeTap) {
        let mut s = self.inner.state.lock();
        s.edges.push(EdgeInfo {
            from,
            to,
            ..Default::default()
        });
        s.edge_probes.push(tap.probe);
        s.edge_caps.push(caps);
        s.edge_counters.push(tap.counters);
    }

    /// Record the graph-wide default link depth the run was built with, so the
    /// single-frame waterfall can state the `2 * capacity * frame_period`
    /// queueing floor its measured total is fighting.
    pub(crate) fn set_link_capacity(&self, capacity: usize) {
        self.inner.state.lock().link_capacity = capacity;
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
        // Fill each edge's negotiated caps and live counters from the aligned
        // `edge_caps` / `edge_counters` (present once the runner has registered
        // them, after negotiation).
        let edges = s
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| EdgeInfo {
                from: e.from,
                to: e.to,
                caps: s.edge_caps.get(i).map(|c| c.to_gst_string()),
                counts: s
                    .edge_counters
                    .get(i)
                    .and_then(|c| c.as_ref())
                    .map(|c| c.snapshot())
                    .unwrap_or_default(),
            })
            .collect();
        TelemetrySnapshot {
            uptime_ns: crate::metrics::monotonic_ns().saturating_sub(self.inner.start_ns),
            nodes,
            edges,
            journey: assemble_journey(&s),
        }
    }
}

/// The linear prefix of the graph starting at a source: nodes strung together by
/// single edges. The walk stops at the first fan node (a tee / demux / muxer, or
/// any node with more than one in- or out-edge) because one input frame becomes
/// N outputs there and the sequence id no longer identifies the same frame.
/// Returns the chain and whether it stopped short of a terminal node.
fn linear_chain(state: &State) -> Option<(Vec<usize>, bool)> {
    let n = state.roles.len();
    let mut in_deg = alloc::vec![0usize; n];
    let mut out_deg = alloc::vec![0usize; n];
    for e in &state.edges {
        if e.from < n && e.to < n {
            out_deg[e.from] += 1;
            in_deg[e.to] += 1;
        }
    }
    let start = (0..n).find(|&i| in_deg[i] == 0)?;
    let mut chain = alloc::vec![start];
    let mut cur = start;
    while out_deg[cur] == 1 {
        let Some(next) = state.edges.iter().find(|e| e.from == cur).map(|e| e.to) else {
            break;
        };
        if next >= n
            || in_deg[next] != 1
            || matches!(state.roles[next], NodeRole::Tee | NodeRole::Muxer)
        {
            break;
        }
        chain.push(next);
        cur = next;
    }
    Some((chain, out_deg[cur] != 0))
}

/// Join one frame's path across the graph's linear prefix. Each stage's probe
/// keeps a ring of recent [`StageVisit`]s keyed by sequence id; a journey is the
/// newest id every stage recorded whose stamps are consistent with one frame
/// flowing downstream. `None` when nothing is recorded (no observer, or too few
/// frames yet) or when no id survives the consistency check, which is the honest
/// answer for a graph whose elements restamp.
fn assemble_journey(state: &State) -> Option<FrameJourney> {
    let (chain, mut truncated) = linear_chain(state)?;
    // Leading nodes without records are the source (no `process()`); once stages
    // have started, a gap would make the next hop a fabricated join, so stop.
    let mut stages: Vec<(usize, Vec<StageVisit>)> = Vec::new();
    for &node in &chain {
        let visits = state
            .probes
            .get(node)
            .and_then(|p| p.as_ref())
            .map(|p| p.visits())
            .unwrap_or_default();
        if visits.is_empty() {
            if !stages.is_empty() {
                truncated = true;
                break;
            }
            continue;
        }
        stages.push((node, visits));
    }
    let (last_node, last_visits) = stages.last()?;
    truncated |= Some(*last_node) != chain.last().copied();
    let mut seqs: Vec<u64> = last_visits.iter().map(|v| v.sequence).collect();
    seqs.sort_unstable();
    seqs.dedup();

    for &sequence in seqs.iter().rev() {
        let path: Option<Vec<StageVisit>> = stages
            .iter()
            .map(|(_, v)| v.iter().rev().find(|x| x.sequence == sequence).copied())
            .collect();
        let Some(path) = path else { continue };
        if !one_frame_downstream(&path) {
            continue;
        }
        let first = path[0];
        let last = path[path.len() - 1];
        let frame_period_ns = mean_period_ns(&stages[0].1);
        let stage_rows = stages
            .iter()
            .zip(path.iter())
            .map(|((node, _), v)| JourneyStage {
                node: *node,
                name: state.names.get(*node).cloned().unwrap_or_default(),
                wait_ns: v.wait_ns,
                work_ns: v
                    .exit_ns
                    .saturating_sub(v.enter_ns)
                    .saturating_sub(v.push_wait_ns),
                blocked_ns: v.push_wait_ns,
            })
            .collect();
        return Some(FrameJourney {
            sequence,
            stages: stage_rows,
            // From the frame being queued for the first measured stage to the
            // last one finishing it: the span an outside observer would time.
            total_ns: last
                .exit_ns
                .saturating_sub(first.enter_ns.saturating_sub(first.wait_ns)),
            frame_period_ns,
            capacity: state.link_capacity,
            floor_ns: 2 * state.link_capacity as u64 * frame_period_ns,
            truncated,
        });
    }
    None
}

/// Whether `path` is consistent with one frame walking downstream: each stage
/// finishes after it starts, and a stage's frame was queued no earlier than the
/// upstream stage began producing it. Rejects a coincidental id collision from
/// an element that restamps its output.
fn one_frame_downstream(path: &[StageVisit]) -> bool {
    path.windows(2).all(|w| {
        w[1].exit_ns >= w[1].enter_ns && w[1].enter_ns.saturating_sub(w[1].wait_ns) >= w[0].enter_ns
    }) && path[0].exit_ns >= path[0].enter_ns
}

/// Mean spacing between consecutive frames entering a stage, the measured frame
/// period the queueing floor is expressed in. `0` with fewer than two records.
fn mean_period_ns(visits: &[StageVisit]) -> u64 {
    let (Some(first), Some(last)) = (visits.first(), visits.last()) else {
        return 0;
    };
    let spans = visits.len().saturating_sub(1) as u64;
    last.enter_ns
        .saturating_sub(first.enter_ns)
        .checked_div(spans)
        .unwrap_or(0)
}

impl Default for Observer {
    fn default() -> Self {
        Self::new()
    }
}

/// The observer-side handles of one link: its content-inspection slot and, when
/// the runner instrumented it, its live traffic counters.
#[derive(Debug, Default)]
pub(crate) struct EdgeTap {
    pub(crate) probe: ProbeSlot,
    pub(crate) counters: Option<Arc<EdgeCounters>>,
}

/// Build a link plus the observer-side taps a hand-built runner registers.
/// `tap` is false when no observer is attached, leaving the link exactly as
/// cheap as a bare [`link`](crate::runtime::link).
pub(crate) fn link_tapped(
    capacity: usize,
    tap: bool,
) -> (
    crate::runtime::LinkSender,
    crate::runtime::LinkReceiver,
    EdgeTap,
) {
    let (mut tx, rx) = crate::runtime::link(capacity);
    let counters = tap.then(|| {
        let c = Arc::new(EdgeCounters::default());
        tx.set_counters(c.clone());
        c
    });
    let edge = EdgeTap {
        probe: tx.probe.clone(),
        counters,
    };
    (tx, rx, edge)
}

/// One node of a hand-built runner's topology: instance name, role, and the
/// measured-latency probe of the element behind it (`None` for a source or a
/// structural node with no `process()`).
pub(crate) type TapNode = (String, NodeRole, Probe);

/// One link of a hand-built runner's topology: endpoints (indices into the node
/// list), negotiated caps, and the link's taps.
pub(crate) type TapEdge = (usize, usize, Caps, EdgeTap);

/// Install a hand-built runner's topology into `obs`. The fan-in / fan-out /
/// session runners have no `Graph` for the runner to walk, so they describe
/// their nodes and links directly; the resulting snapshot is the same shape
/// `run_graph_observed` produces.
pub(crate) fn register_runner_tap(obs: &Observer, nodes: Vec<TapNode>, edges: Vec<TapEdge>) {
    let mut names = Vec::with_capacity(nodes.len());
    let mut roles = Vec::with_capacity(nodes.len());
    let mut probes = Vec::with_capacity(nodes.len());
    for (name, role, probe) in nodes {
        names.push(name);
        roles.push(role);
        probes.push(probe);
    }
    let mut infos = Vec::with_capacity(edges.len());
    let mut caps = Vec::with_capacity(edges.len());
    let mut slots = Vec::with_capacity(edges.len());
    let mut counters = Vec::with_capacity(edges.len());
    for (from, to, edge_caps, tap) in edges {
        infos.push(EdgeInfo {
            from,
            to,
            ..Default::default()
        });
        caps.push(edge_caps);
        slots.push(tap.probe);
        counters.push(tap.counters);
    }
    obs.register(names, roles, probes, infos);
    obs.register_edges(slots, caps, counters);
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
    /// The newest single frame whose whole path could be joined across stages,
    /// or `None` when no observer-recorded journey assembles (see
    /// [`FrameJourney`]).
    pub journey: Option<FrameJourney>,
}

/// One frame's measured path through the graph's linear prefix (M851): the
/// per-stage wait + work + blocked of a *single* frame, as opposed to the
/// per-stage distributions the aggregate waterfall stacks.
///
/// Stages are joined on [`Frame::sequence`](crate::Frame), so the journey only
/// spans elements that carry the id through. It stops at a fan node (a tee,
/// demux, or muxer, where one input frame becomes N outputs) and at any element
/// that restamps, with `truncated` set; nothing past that point is guessed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameJourney {
    /// The frame's sequence id, as stamped by the source.
    pub sequence: u64,
    /// Per stage, upstream first. The source has no `process()` and so no row;
    /// its cost shows up as the first stage's `wait_ns`.
    pub stages: Vec<JourneyStage>,
    /// Measured end to end: from the frame being queued for the first stage to
    /// the last stage finishing it.
    pub total_ns: u64,
    /// Mean spacing between frames entering the first stage.
    pub frame_period_ns: u64,
    /// The graph-wide default link depth (`0` if the runner did not register it).
    pub capacity: usize,
    /// `2 * capacity * frame_period_ns`: the queueing floor a bounded link
    /// imposes regardless of how fast the elements are. A `total_ns` near this
    /// means the pipeline is capacity-bound, not compute-bound.
    pub floor_ns: u64,
    /// The journey covers only part of the graph: it ran into a fan node or a
    /// stage that did not record this id.
    pub truncated: bool,
}

/// One stage of a [`FrameJourney`]: what this one frame cost at one element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JourneyStage {
    /// The element's `NodeId` index.
    pub node: usize,
    /// Instance name (`<category>N`).
    pub name: String,
    /// How long this frame sat on the element's input link. `0` on an
    /// uninstrumented (leaky) edge.
    pub wait_ns: u64,
    /// How long the element computed on this frame: its `process()` span with
    /// `blocked_ns` taken out.
    pub work_ns: u64,
    /// How long that same `process()` call sat blocked pushing this frame into
    /// the output link, i.e. downstream backpressure rather than work. `0` for a
    /// sink, which pushes nowhere.
    pub blocked_ns: u64,
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
/// of the solved per-edge `Caps`) and its live traffic counters. `caps` is `None`
/// until the runner registers the negotiated solution, and in a topology-only
/// `EdgeInfo`; `counts` advances as packets cross and is all-zero on an
/// uninstrumented edge.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EdgeInfo {
    pub from: usize,
    pub to: usize,
    pub caps: Option<alloc::string::String>,
    pub counts: EdgeCounts,
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
                ..Default::default()
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
                ..Default::default()
            }]
        );
    }

    /// M869: a dynamic runner's arm attaches mid-run, so its node and link join
    /// an already-registered topology. The append lands both, keyed by the
    /// returned id, and the arm's live probe reads through the same snapshot.
    #[test]
    fn incremental_node_and_edge_join_a_registered_topology() {
        let obs = Observer::new();
        obs.register(
            alloc::vec![String::from("src0")],
            alloc::vec![NodeRole::Source],
            alloc::vec![None],
            Vec::new(),
        );
        assert_eq!(obs.snapshot().nodes.len(), 1);

        let probe = ElementProbe::new(String::from("fakesink0"));
        let counters = Arc::new(EdgeCounters::default());
        let id = obs.add_node(
            String::from("fakesink0"),
            NodeRole::Sink,
            Some(probe.clone()),
        );
        assert_eq!(id, 1, "appended after the registered source");
        obs.add_edge(
            0,
            id,
            Caps::Klv,
            EdgeTap {
                probe: ProbeSlot::default(),
                counters: Some(counters.clone()),
            },
        );

        probe.record_fill(60);
        counters.record_packet(128, 0);

        let snap = obs.snapshot();
        assert_eq!(snap.nodes.len(), 2);
        assert_eq!(snap.nodes[1].name, "fakesink0");
        assert_eq!(snap.nodes[1].role, NodeRole::Sink);
        assert_eq!(snap.nodes[1].latency.as_ref().unwrap().fill_max_pct, 60);
        assert_eq!(snap.edges.len(), 1);
        assert_eq!((snap.edges[0].from, snap.edges[0].to), (0, 1));
        assert!(snap.edges[0].caps.is_some(), "late edge carries its caps");
        assert_eq!(snap.edges[0].counts.packets, 1);
        assert_eq!(snap.edges[0].counts.bytes, 128);
        assert!(obs.edge_probe(0).is_some(), "late edge has an inspect slot");
    }

    /// A three-node chain (source -> transform -> sink) with hand-stamped
    /// visits: the join picks the newest sequence every stage saw and reports it
    /// upstream-first, with the floor computed off the measured frame period.
    #[test]
    fn journey_joins_one_frame_across_stages() {
        let obs = Observer::new();
        let xform = ElementProbe::with_journeys(String::from("scale0"));
        let sink = ElementProbe::with_journeys(String::from("fakesink0"));
        obs.register(
            alloc::vec![
                String::from("src0"),
                String::from("scale0"),
                String::from("fakesink0"),
            ],
            alloc::vec![NodeRole::Source, NodeRole::Transform, NodeRole::Sink],
            alloc::vec![None, Some(xform.clone()), Some(sink.clone())],
            alloc::vec![
                EdgeInfo {
                    from: 0,
                    to: 1,
                    ..Default::default()
                },
                EdgeInfo {
                    from: 1,
                    to: 2,
                    ..Default::default()
                },
            ],
        );
        obs.set_link_capacity(4);

        // Two frames, 1000 ns apart, each waiting 100 ns then working 200 ns at
        // the transform and waiting 50 ns then working 150 ns at the sink.
        for (seq, base) in [(0u64, 10_000u64), (1, 11_000)] {
            xform.push_visit(StageVisit {
                sequence: seq,
                wait_ns: 100,
                enter_ns: base,
                exit_ns: base + 200,
                push_wait_ns: 60,
            });
            sink.push_visit(StageVisit {
                sequence: seq,
                wait_ns: 50,
                enter_ns: base + 250,
                exit_ns: base + 400,
                push_wait_ns: 0,
            });
        }

        let j = obs.snapshot().journey.expect("journey assembles");
        assert_eq!(j.sequence, 1, "newest fully-crossed frame");
        assert!(!j.truncated, "chain reached the sink");
        // The transform's 200 ns span held 60 ns of downstream backpressure, so
        // its work segment is the remaining 140 ns.
        assert_eq!(
            j.stages
                .iter()
                .map(|s| (s.node, s.name.as_str(), s.wait_ns, s.work_ns, s.blocked_ns))
                .collect::<Vec<_>>(),
            alloc::vec![(1, "scale0", 100, 140, 60), (2, "fakesink0", 50, 150, 0)],
        );
        // Queued for the transform at 11_000-100, done at the sink at 11_400.
        assert_eq!(j.total_ns, 500);
        let stage_sum: u64 = j
            .stages
            .iter()
            .map(|s| s.wait_ns + s.work_ns + s.blocked_ns)
            .sum();
        assert!(j.total_ns >= stage_sum, "{} >= {}", j.total_ns, stage_sum);
        assert_eq!(j.frame_period_ns, 1_000, "measured inter-frame spacing");
        assert_eq!(j.capacity, 4);
        assert_eq!(j.floor_ns, 2 * 4 * 1_000);
    }

    /// A downstream stage that saw "sequence 0" before the upstream one ever
    /// started it is a restamp collision, not one frame's path. The join
    /// rejects it rather than inventing a hop.
    #[test]
    fn journey_rejects_an_inconsistent_join() {
        let obs = Observer::new();
        let xform = ElementProbe::with_journeys(String::from("dec0"));
        let sink = ElementProbe::with_journeys(String::from("fakesink0"));
        obs.register(
            alloc::vec![
                String::from("src0"),
                String::from("dec0"),
                String::from("fakesink0"),
            ],
            alloc::vec![NodeRole::Source, NodeRole::Transform, NodeRole::Sink],
            alloc::vec![None, Some(xform.clone()), Some(sink.clone())],
            alloc::vec![
                EdgeInfo {
                    from: 0,
                    to: 1,
                    ..Default::default()
                },
                EdgeInfo {
                    from: 1,
                    to: 2,
                    ..Default::default()
                },
            ],
        );
        xform.push_visit(StageVisit {
            sequence: 0,
            wait_ns: 0,
            enter_ns: 5_000,
            exit_ns: 5_100,
            push_wait_ns: 0,
        });
        sink.push_visit(StageVisit {
            sequence: 0,
            wait_ns: 0,
            enter_ns: 1_000,
            exit_ns: 1_100,
            push_wait_ns: 0,
        });
        assert!(obs.snapshot().journey.is_none());
    }

    /// A tee ends the linear chain: the frame's id space forks there, so the
    /// journey covers the prefix and says so.
    #[test]
    fn journey_stops_at_a_fan_node() {
        let obs = Observer::new();
        let xform = ElementProbe::with_journeys(String::from("scale0"));
        obs.register(
            alloc::vec![
                String::from("src0"),
                String::from("scale0"),
                String::new(),
                String::from("fakesink0"),
            ],
            alloc::vec![
                NodeRole::Source,
                NodeRole::Transform,
                NodeRole::Tee,
                NodeRole::Sink,
            ],
            alloc::vec![None, Some(xform.clone()), None, None],
            alloc::vec![
                EdgeInfo {
                    from: 0,
                    to: 1,
                    ..Default::default()
                },
                EdgeInfo {
                    from: 1,
                    to: 2,
                    ..Default::default()
                },
                EdgeInfo {
                    from: 2,
                    to: 3,
                    ..Default::default()
                },
            ],
        );
        xform.push_visit(StageVisit {
            sequence: 3,
            wait_ns: 10,
            enter_ns: 900,
            exit_ns: 1_000,
            push_wait_ns: 0,
        });
        let j = obs.snapshot().journey.expect("prefix assembles");
        assert_eq!(j.stages.len(), 1, "only the pre-tee stage");
        assert!(j.truncated, "the tee cut the walk short");
    }

    #[test]
    fn journey_absent_without_journey_probes() {
        let obs = Observer::new();
        obs.register(
            alloc::vec![String::from("src0"), String::from("fakesink0")],
            alloc::vec![NodeRole::Source, NodeRole::Sink],
            alloc::vec![None, Some(ElementProbe::new(String::from("fakesink0")))],
            alloc::vec![EdgeInfo {
                from: 0,
                to: 1,
                ..Default::default()
            }],
        );
        assert!(obs.snapshot().journey.is_none());
    }

    #[test]
    fn node_role_projects_kind() {
        assert_eq!(NodeRole::from(NodeKind::Tee(3)), NodeRole::Tee);
        assert_eq!(NodeRole::from(NodeKind::Muxer(2)), NodeRole::Muxer);
        assert_eq!(NodeRole::from(NodeKind::Sink), NodeRole::Sink);
    }
}
