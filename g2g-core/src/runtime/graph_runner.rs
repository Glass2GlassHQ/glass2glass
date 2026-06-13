//! DAG pipeline runner (DESIGN_TODO "DAG runner" D3).
//!
//! [`run_graph`] drives an arbitrary multimedia DAG built with [`Graph`]:
//! whole-graph CSP negotiation via [`solve_graph`] (D2), then one spawned arm
//! per node over per-edge channels, joined with [`join_all`]. It collapses the
//! linear + fan-out runner shapes into one entry point.
//!
//! Scope (D3): source / transform / sink / tee. A tee broadcasts each packet
//! to all its branches; since [`PipelinePacket`] is not `Clone` (a GPU-resident
//! frame owns a non-copyable handle), the broadcast deep-copies `System` frames
//! and fails loud on a GPU domain. A refcounted shareable frame (zero-copy tee)
//! and the muxer fan-in (which needs the per-input-pad constraint API) are the
//! follow-ups; a muxer node is rejected here. The mid-stream re-cascade
//! (coordinator) is D4, so an interior arm handles a `CapsChanged` locally
//! (configure + forward) without the β allocation walk.

use alloc::boxed::Box;
use alloc::vec::Vec;

use crate::caps::{Caps, CapsSet};
use crate::clock::{ClockPriority, PipelineClock};
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, DynAsyncElement, OutputSink, Reconfigure,
};
use crate::error::G2gError;
use crate::format_element::CapsConstraint;
use crate::frame::{Frame, PipelinePacket};
use crate::graph::{Graph, NodeId, NodeKind, ValidatedGraph};
use crate::memory::{MemoryDomain, SystemSlice};
use crate::query::LatencyReport;
use crate::runtime::channel::{link, LinkReceiver, LinkSender, SenderSink};
use crate::runtime::fanin::DynSourceLoop;
use crate::runtime::join::join_all;
use crate::runtime::runner::{LinkCapacity, NullSink, RunStats, SourceLoop};
use crate::runtime::solver::{solve_graph, NodeConstraint};

/// Element payload for a [`Graph`] driven by [`run_graph`]. Sources and
/// transforms/sinks implement different traits (a source has no input pad), so
/// the payload is an enum the runner matches on per node kind. Tee and muxer
/// nodes carry no element (`Graph::add_tee` / `add_muxer` take none).
pub enum GraphNode {
    Source(Box<dyn DynSourceLoop>),
    Element(Box<dyn DynAsyncElement>),
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
}

impl core::fmt::Debug for GraphNode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GraphNode::Source(_) => f.write_str("GraphNode::Source(..)"),
            GraphNode::Element(_) => f.write_str("GraphNode::Element(..)"),
        }
    }
}

/// Drive an arbitrary DAG to EOS. Negotiates the whole graph at once, then runs
/// one arm per node over per-edge channels. `link_capacity` accepts a
/// [`LatencyProfile`](crate::runtime::LatencyProfile) or a `usize` depth.
///
/// Negotiation, allocation, clock-election, and latency aggregation over a DAG
/// are reported as neutral values for now (like the fan-out / fan-in runners);
/// the DAG-wide folds are a follow-up. A muxer node, a non-`System` frame in a
/// tee, or a negotiation conflict fails loud.
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

    // Muxer fan-in needs the per-input-pad constraint API (DESIGN_TODO "Muxer
    // per-input-pad constraint API"); reject until that lands.
    if topo.iter().any(|&node| matches!(vg.kind(node), NodeKind::Muxer(_))) {
        return Err(G2gError::CapsMismatch);
    }

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

    // Phase 2: build a per-node constraint and solve the whole DAG. The
    // transform/sink constraints borrow their elements immutably (coexisting),
    // so the solution is computed and the borrows released before configure.
    let solution: Vec<Caps> = {
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
                NodeKind::Muxer(_) => return Err(G2gError::CapsMismatch),
            };
            constraints.push(nc);
        }
        solve_graph(&vg, &constraints).map_err(|_| G2gError::CapsMismatch)?
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
            NodeKind::Muxer(_) => unreachable!("muxer rejected above"),
        }
    }

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

    let mut arms: Vec<BoxFuture<'static, Result<u64, G2gError>>> = Vec::with_capacity(n);
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
                Box::pin(transform_arm(elem, in_rx, out_tx))
            }
            NodeKind::Sink => {
                let Some(GraphNode::Element(elem)) = element else {
                    return Err(G2gError::CapsMismatch);
                };
                let in_rx = in_rxs.pop().expect("sink input edge");
                Box::pin(sink_arm(elem, in_rx))
            }
            NodeKind::Tee(_) => {
                let in_rx = in_rxs.pop().expect("tee input edge");
                Box::pin(tee_arm(in_rx, out_txs))
            }
            NodeKind::Muxer(_) => unreachable!("muxer rejected above"),
        };
        arms.push(arm);
        arm_kinds.push(kind);
    }

    let results = join_all(arms).await;
    let mut emitted = 0u64;
    let mut consumed = 0u64;
    for (kind, result) in arm_kinds.into_iter().zip(results) {
        let count = result?;
        match kind {
            NodeKind::Source => emitted += count,
            NodeKind::Sink => consumed += count,
            _ => {}
        }
    }

    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        latency: LatencyReport::ZERO,
        allocation: None,
        clock_priority: ClockPriority::SystemFallback,
        base_time_ns: clock.now_ns(),
        coordinator_events: 0,
    })
}

/// View a node's payload as a transform/sink element. `None` for a source or a
/// tee/muxer (whose constraint the runner builds without the element).
fn element_ref(vg: &ValidatedGraph<GraphNode>, node: NodeId) -> Option<&dyn DynAsyncElement> {
    match vg.element(node)? {
        GraphNode::Element(elem) => Some(&**elem),
        GraphNode::Source(_) => None,
    }
}

async fn source_arm(
    mut src: Box<dyn DynSourceLoop>,
    out_tx: LinkSender,
) -> Result<u64, G2gError> {
    let mut adapter = SenderSink::new(out_tx);
    src.run(&mut adapter).await
}

async fn transform_arm(
    mut elem: Box<dyn DynAsyncElement>,
    in_rx: LinkReceiver,
    out_tx: LinkSender,
) -> Result<u64, G2gError> {
    let mut adapter = SenderSink::new(out_tx);
    loop {
        match in_rx.recv().await {
            Some(PipelinePacket::Eos) => {
                elem.process(PipelinePacket::Eos, &mut adapter).await?;
                adapter.push(PipelinePacket::Eos).await?;
                return Ok(0);
            }
            Some(PipelinePacket::CapsChanged(new_caps)) => {
                // D3 local handling: configure with the upstream caps and let
                // the element emit its own output `CapsChanged`. The
                // downstream-feasibility steering and β re-cascade are D4.
                match elem.configure_pipeline(&new_caps)? {
                    ConfigureOutcome::Accepted => {
                        elem.process(PipelinePacket::CapsChanged(new_caps), &mut adapter)
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

async fn sink_arm(
    mut elem: Box<dyn DynAsyncElement>,
    in_rx: LinkReceiver,
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
                match elem.configure_pipeline(&new_caps)? {
                    ConfigureOutcome::Accepted => {
                        elem.process(PipelinePacket::CapsChanged(new_caps), &mut null)
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
