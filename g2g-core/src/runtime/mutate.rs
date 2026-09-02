//! Structural mutation of a running graph (M1115): splicing a transform onto a
//! live edge and lifting one back off, and swapping the source or the sink at
//! either end of it (M1149), without stopping the pipeline.
//!
//! The data plane pays for this once per push: a relaxed load of the edge's
//! [`ProducerEndpoint`] gate. Everything else happens inside a mutation op,
//! which runs in the run future itself (`MutationService`), so it can touch the
//! arms, the links and the elements the way the runner does.
//!
//! Scope: a transform position on a 1:1 edge. The producing end is a source, a
//! transform or one output of a tee / demux; the consuming end is a transform, a
//! sink, one input pad of a muxer or terminal fan-in (M1133), or the single
//! input of a tee / demux (M1146). The structural nodes themselves are not
//! splice points ([`MutationError::NotMutable`]); the two ends are not splice
//! points either, but the elements sitting on them are replaceable in place
//! (M1149).

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use crate::bus::BusHandle;
use crate::caps::{Caps, CapsSet};
use crate::clock::ClockSync;
use crate::element::{BoxFuture, DynAsyncElement};
use crate::error::G2gError;
use crate::graph::NodeId;
use crate::link::LinkPolicy;
use crate::runtime::channel::{
    advertise_orientation, bounded, link, LinkSender, ProducerEndpoint, Receiver, Sender,
};
use crate::runtime::coordinator::ArmDirective;
use crate::runtime::fanin::DynSourceLoop;
use crate::runtime::graph_runner::{
    run_source, BranchMode, GraphCoordHandle, SinkArmIo, SourceArmIo, TransformArmIo,
};
use crate::runtime::progress::PipelineProgress;
use crate::runtime::runner::re_solve_downstream_dyn_sink;
use crate::segment::Segment;

/// Why a structural mutation was refused. Every one of these leaves the graph
/// running exactly as it was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MutationError {
    /// No element of the running graph carries this instance name.
    UnknownNode(String),
    /// The named position names no single mutable edge on the side asked for: a
    /// node with several edges there (a tee below, a muxer above), an end with
    /// none (a source above, a sink below), or a node that is not a transform
    /// when one is being lifted out.
    NotMutable(String),
    /// The element refused the caps flowing on the edge, or refused to fixate
    /// on them.
    Refused(G2gError),
    /// The caps this operation would change the edge to are not in the set the
    /// chain below it accepts, or there is no snapshot of that set to check
    /// them against. Consent has to be established before the caps move, since
    /// an element that turns one down mid-stream fails the whole run.
    DownstreamRefused,
    /// Nothing has crossed the edge yet, so there are no caps to configure the
    /// new element against.
    NoCaps,
    /// The run has ended (or its producer's arm has), so there is no graph left
    /// to mutate.
    GraphEnded,
}

/// A handle on a *running* graph's topology (M1115): splice a transform onto a
/// live edge, or lift one back off. Obtained beside the run future from
/// [`run_graph_mutable`](crate::runtime::run_graph_mutable) or
/// [`run_graph_threaded_mutable`](crate::runtime::run_graph_threaded_mutable),
/// like [`DynamicFanoutHandle`](crate::runtime::DynamicFanoutHandle): drive the
/// run future while using the handle from another task.
///
/// Positions are named by the instance name the runner assigned (a launch line's
/// `name=`, else `<category>N`), and an inserted element gets one of its own,
/// returned by [`insert_after`](Self::insert_after) /
/// [`insert_before`](Self::insert_before) so it can be removed later.
///
/// Each operation completes at the producer's next packet boundary, so a
/// producer that has gone quiet defers it until it pushes again.
pub struct GraphMutator<'a> {
    tx: Sender<MutationRequest<'a>>,
}

impl<'a> Clone for GraphMutator<'a> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl core::fmt::Debug for GraphMutator<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GraphMutator").finish_non_exhaustive()
    }
}

impl<'a> GraphMutator<'a> {
    /// Splice `element` onto the edge leaving the element named `after`, so the
    /// stream runs `after -> element -> (whatever `after` fed)`. Returns the
    /// instance name the new element was given.
    ///
    /// The element is negotiated against the caps flowing on that edge right now
    /// before anything is disturbed: it must accept them, and what it emits must
    /// be in the set the downstream chain accepts. A refusal leaves the graph
    /// running and drops the element.
    pub async fn insert_after(
        &self,
        after: &str,
        element: Box<dyn DynAsyncElement + 'a>,
    ) -> Result<String, MutationError> {
        self.insert(after, Side::Below, element).await
    }

    /// Splice `element` onto the edge entering the element named `before`, so
    /// the stream runs `(whatever fed `before`) -> element -> before`. Returns
    /// the instance name the new element was given.
    ///
    /// The counterpart of [`insert_after`](Self::insert_after) for the edges a
    /// producer cannot name on its own: a tee or demux has several edges below
    /// it, so its branch is addressed by the consumer on the far end. A node
    /// with several inbound edges (a muxer) is [`MutationError::NotMutable`]
    /// here, and is addressed with `insert_after` from the producer instead.
    ///
    /// Naming a tee or demux here addresses the one edge into it, so the splice
    /// feeds every branch at once; the caps it changes the edge to reach the
    /// branches as the tee broadcasts them.
    ///
    /// Negotiation and refusal work exactly as for `insert_after`.
    pub async fn insert_before(
        &self,
        before: &str,
        element: Box<dyn DynAsyncElement + 'a>,
    ) -> Result<String, MutationError> {
        self.insert(before, Side::Above, element).await
    }

    async fn insert(
        &self,
        node: &str,
        side: Side,
        element: Box<dyn DynAsyncElement + 'a>,
    ) -> Result<String, MutationError> {
        let (reply, answer) = bounded(1);
        self.tx
            .send(MutationRequest::Insert {
                node: node.to_string(),
                side,
                element,
                reply,
            })
            .await
            .map_err(|_| MutationError::GraphEnded)?;
        answer.recv().await.ok_or(MutationError::GraphEnded)?
    }

    /// Lift the transform named `node` out of the running graph and hand it
    /// back. Whatever is queued at its input drains through it first, and it is
    /// then flushed, so the frames it was holding internally reach the consumer
    /// too; only after that does its producer start feeding the consumer
    /// directly. No frame is lost or reordered.
    ///
    /// If the element was changing the caps, the consumer must accept the
    /// producer's caps, since that is what it will start receiving; a consumer
    /// that cannot is a [`MutationError::DownstreamRefused`] and the element
    /// stays in the graph.
    pub async fn remove(&self, node: &str) -> Result<Box<dyn DynAsyncElement + 'a>, MutationError> {
        let (reply, answer) = bounded(1);
        self.tx
            .send(MutationRequest::Remove {
                node: node.to_string(),
                reply,
            })
            .await
            .map_err(|_| MutationError::GraphEnded)?;
        answer.recv().await.ok_or(MutationError::GraphEnded)?
    }

    /// Take over the sink named `node` with `element` (M1149), and hand the old
    /// one back. Returns the instance name the replacement was given, which is a
    /// fresh one: the name the old sink had is never handed out again.
    ///
    /// `element` is negotiated against the caps flowing on the edge before
    /// anything is disturbed, exactly as a splice is, and a refusal leaves the
    /// graph running. The old sink then receives the frames still queued for it
    /// and an end of stream, so it finalizes (writes its trailer, closes its
    /// file) before it comes back; the replacement starts only once it has, so
    /// the two never render the same stretch of stream. The stall in between is
    /// bounded by what was queued, as it is for [`remove`](Self::remove).
    ///
    /// The replacement joins a stream already in flight: it starts in the
    /// playing state with no preroll, and it cannot change the caps (it
    /// accepted the ones flowing). It receives the clock the run elected at
    /// startup, latency fold included, so it paces frames the way the sink it
    /// took over from did; it only takes no part in the election itself
    /// (its `provide_clock` is ignored). A node that is not a sink is
    /// [`MutationError::NotMutable`]: a terminal fan-in taking several inputs,
    /// or one whose element the runner did not lend out.
    pub async fn replace_sink(
        &self,
        node: &str,
        element: Box<dyn DynAsyncElement + 'a>,
    ) -> Result<(String, Box<dyn DynAsyncElement + 'a>), MutationError> {
        let (reply, answer) = bounded(1);
        self.tx
            .send(MutationRequest::ReplaceSink {
                node: node.to_string(),
                element,
                reply,
            })
            .await
            .map_err(|_| MutationError::GraphEnded)?;
        answer.recv().await.ok_or(MutationError::GraphEnded)?
    }

    /// Take over the source named `node` with `source` (M1149), and hand the old
    /// one back. Returns the fresh instance name the replacement was given, like
    /// [`replace_sink`](Self::replace_sink).
    ///
    /// The replacement keeps the shape on the wire when it can produce it, and
    /// otherwise picks one the chain below is known to accept (a
    /// [`MutationError::DownstreamRefused`] when none is), announced with a
    /// `CapsChanged` ahead of its first frame. Everything the old source had
    /// already queued stays ahead of that, so nothing is lost or reordered; the
    /// one packet it was holding un-pushed at the swap is dropped.
    ///
    /// The two sources' timelines are stitched: the replacement opens with a
    /// segment whose base continues the running time its predecessor reached, so
    /// it can stamp from its own zero and the pipeline's running time still only
    /// moves forward. It takes no part in the clock election or the latency fold
    /// (`provide_clock` is ignored, the run keeps the clock it elected). A node
    /// that is not a source is [`MutationError::NotMutable`].
    pub async fn replace_source(
        &self,
        node: &str,
        source: Box<dyn DynSourceLoop + 'a>,
    ) -> Result<(String, Box<dyn DynSourceLoop + 'a>), MutationError> {
        let (reply, answer) = bounded(1);
        self.tx
            .send(MutationRequest::ReplaceSource {
                node: node.to_string(),
                source,
                reply,
            })
            .await
            .map_err(|_| MutationError::GraphEnded)?;
        answer.recv().await.ok_or(MutationError::GraphEnded)?
    }
}

/// Which edge of the named node a splice addresses: the one below it (the
/// element that produces into the edge) or the one above it (the element that
/// consumes from it). Whichever end is unique names the edge.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Side {
    Below,
    Above,
}

/// One operation, carrying its own reply channel. The work runs in the run
/// future, which is where the arms and links live.
#[allow(missing_debug_implementations)]
pub(crate) enum MutationRequest<'a> {
    Insert {
        node: String,
        side: Side,
        element: Box<dyn DynAsyncElement + 'a>,
        reply: Sender<Result<String, MutationError>>,
    },
    Remove {
        node: String,
        reply: Sender<Result<Box<dyn DynAsyncElement + 'a>, MutationError>>,
    },
    ReplaceSink {
        node: String,
        element: Box<dyn DynAsyncElement + 'a>,
        reply: Sender<ReplacedSink<'a>>,
    },
    ReplaceSource {
        node: String,
        source: Box<dyn DynSourceLoop + 'a>,
        reply: Sender<ReplacedSource<'a>>,
    },
}

/// What a sink replacement answers with: the fresh instance name, and the
/// element it took the place of.
pub(crate) type ReplacedSink<'a> = Result<(String, Box<dyn DynAsyncElement + 'a>), MutationError>;

/// [`ReplacedSink`] for a source replacement.
pub(crate) type ReplacedSource<'a> = Result<(String, Box<dyn DynSourceLoop + 'a>), MutationError>;

/// Build the mutator handle and the request channel the service reads. The
/// handle is returned to the caller before the run starts, the receiver goes
/// into the run future.
pub(crate) fn mutation_channel<'a>(
    capacity: usize,
) -> (GraphMutator<'a>, Receiver<MutationRequest<'a>>) {
    let (tx, rx) = bounded(capacity);
    (GraphMutator { tx }, rx)
}

/// One node of the runner's live topology, as the mutator sees it: which mutable
/// edges touch it, and how its element comes back.
#[allow(missing_debug_implementations)]
pub(crate) struct LiveNode<'a> {
    pub(crate) name: String,
    /// Mutable edges leaving this node, as indices into
    /// [`LiveTopology::edges`]. A tee has several, so it cannot be addressed
    /// from above; a sink has none.
    pub(crate) out_edges: Vec<usize>,
    /// Mutable edges entering this node. A muxer has several, so it cannot be
    /// addressed from below; a source has none.
    pub(crate) in_edges: Vec<usize>,
    /// How many edges leave this node in the underlying graph, mutable or not.
    /// Addressing needs exactly one: a fan node whose other branches merely are
    /// not mutation positions still names no single edge.
    pub(crate) total_out_edges: usize,
    /// How many edges enter this node in the underlying graph, mutable or not.
    pub(crate) total_in_edges: usize,
    /// Delivers this node's element once its arm ends, so a remove or a replace
    /// can hand it back. `None` on a node whose element the runner did not lend
    /// out.
    pub(crate) done: Option<Handback<'a>>,
    /// A transform: the only kind of node that can be lifted out.
    pub(crate) removable: bool,
}

/// How a lent-out element comes back when its arm ends. The two shapes are
/// different boxes, so a node says which one it holds rather than the mutator
/// guessing from the node's edges.
#[allow(missing_debug_implementations)]
pub(crate) enum Handback<'a> {
    Element(Receiver<Box<dyn DynAsyncElement + 'a>>),
    Source(Receiver<Box<dyn DynSourceLoop + 'a>>),
}

/// One mutable edge of the running graph: its retargetable producing end, the
/// node it feeds right now, and what the chain below it accepts. The model is
/// per edge rather than per node because neither end is unique in general: a tee
/// produces several edges, a muxer consumes several.
#[allow(missing_debug_implementations)]
pub(crate) struct LiveEdge {
    pub(crate) endpoint: Arc<ProducerEndpoint>,
    pub(crate) consumer: usize,
    /// What the chain below this edge can carry: the runner's startup snapshot,
    /// or, once something was spliced here, the one shape that element accepted.
    /// `None` where the runner could compute none, which refuses any caps change
    /// here rather than waving it through.
    pub(crate) feasible: Option<CapsSet>,
    pub(crate) policy: LinkPolicy,
    pub(crate) capacity: usize,
    /// How an element spliced here reacts to a caps change it cannot re-solve,
    /// taken from the consumer's position: a tee branch drops rather than
    /// failing the whole run.
    pub(crate) mode: BranchMode,
}

/// The runner's live topology as the mutator addresses it.
#[allow(missing_debug_implementations)]
pub(crate) struct LiveTopology<'a> {
    pub(crate) nodes: Vec<LiveNode<'a>>,
    pub(crate) edges: Vec<LiveEdge>,
}

/// Builds the arm future for an element the mutator spliced in. The cooperative
/// runner drives it on the caller's executor; the thread-per-arm runner hands it
/// to its spawner. `done` takes the element back when the arm ends.
pub(crate) type SpliceSpawn<'a, 's> = Box<
    dyn Fn(
            Box<dyn DynAsyncElement + 'a>,
            TransformArmIo,
            Sender<Box<dyn DynAsyncElement + 'a>>,
        ) -> BoxFuture<'a, Result<u64, G2gError>>
        + 's,
>;

/// [`SpliceSpawn`] for a replacement sink's arm (M1149).
pub(crate) type SinkSpawn<'a, 's> = Box<
    dyn Fn(
            Box<dyn DynAsyncElement + 'a>,
            SinkArmIo,
            Sender<Box<dyn DynAsyncElement + 'a>>,
        ) -> BoxFuture<'a, Result<u64, G2gError>>
        + 's,
>;

/// [`SpliceSpawn`] for a replacement source's arm (M1149).
pub(crate) type SourceSpawn<'a, 's> = Box<
    dyn Fn(
            Box<dyn DynSourceLoop + 'a>,
            SourceArmIo,
            Sender<Box<dyn DynSourceLoop + 'a>>,
        ) -> BoxFuture<'a, Result<u64, G2gError>>
        + 's,
>;

/// How the run driving this mutator starts an arm, one per position the mutator
/// can put an element in. The two runners differ only in where the arm ends up:
/// the caller's executor, or a worker thread of its own.
#[allow(missing_debug_implementations)]
pub(crate) struct MutationSpawn<'a, 's> {
    pub(crate) transform: SpliceSpawn<'a, 's>,
    pub(crate) sink: SinkSpawn<'a, 's>,
    pub(crate) source: SourceSpawn<'a, 's>,
}

/// Drive an element the mutator spliced in, handing it back through `done` when
/// its arm ends. The element stays owned here and is driven through the erased
/// handle, which is what makes giving it back possible.
pub(crate) async fn spliced_arm<'a>(
    mut element: Box<dyn DynAsyncElement + 'a>,
    io: TransformArmIo,
    done: Sender<Box<dyn DynAsyncElement + 'a>>,
) -> Result<u64, G2gError> {
    let result = {
        let borrowed: &mut (dyn DynAsyncElement + '_) = &mut *element;
        Box::new(borrowed).drive_transform_arm(io).await
    };
    let _ = done.try_send(element);
    result
}

/// [`spliced_arm`] for a sink position (M1149): a mutable run drives every sink
/// this way, so a replace can hand the old one back once it has finalized.
pub(crate) async fn lent_sink_arm<'a>(
    mut element: Box<dyn DynAsyncElement + 'a>,
    io: SinkArmIo,
    done: Sender<Box<dyn DynAsyncElement + 'a>>,
) -> Result<u64, G2gError> {
    let result = {
        let borrowed: &mut (dyn DynAsyncElement + '_) = &mut *element;
        Box::new(borrowed).drive_sink_arm(io).await
    };
    let _ = done.try_send(element);
    result
}

/// [`spliced_arm`] for a source position (M1149). A source that was retired ends
/// by failing its next push into the dead link the mutator left it, which is how
/// every source ends when its consumer goes away: that is the swap completing,
/// not the run failing, so its arm reports no frames rather than the error.
pub(crate) async fn lent_source_arm<'a>(
    mut source: Box<dyn DynSourceLoop + 'a>,
    io: SourceArmIo,
    done: Sender<Box<dyn DynSourceLoop + 'a>>,
) -> Result<u64, G2gError> {
    let endpoint = io.out_tx.mutation_endpoint();
    let result = run_source(&mut *source, io).await;
    // Read before the handback: the mutator clears the flag as soon as the
    // element reaches it, to give the endpoint to the replacement.
    let retired = endpoint.is_some_and(|e| e.retired());
    let _ = done.try_send(source);
    match retired {
        true => Ok(0),
        false => result,
    }
}

/// The mutator's other half, running inside the run future: it owns the live
/// topology and performs each operation against the arms and links themselves.
#[allow(missing_debug_implementations)]
pub(crate) struct MutationService<'a, 's> {
    rx: Receiver<MutationRequest<'a>>,
    nodes: Vec<LiveNode<'a>>,
    edges: Vec<LiveEdge>,
    /// Sends a spliced element's arm into the run's growable join set.
    arms: Sender<BoxFuture<'a, Result<u64, G2gError>>>,
    spawn: MutationSpawn<'a, 's>,
    bus: Option<BusHandle>,
    /// The run's position / duration handle, so a replacement source or sink
    /// keeps answering the queries the one it took over from did (M1149).
    progress: Option<PipelineProgress>,
    /// The run's leaky-drop tally, installed on a spliced hop so its drops
    /// reach `RunStats` like every other link's.
    dropped: Arc<spin::Mutex<u64>>,
    /// The liveness the runner folded at startup, handed to a spliced element
    /// the way it reached the negotiated ones (M1123).
    path_is_live: bool,
    /// The latency-folded [`ClockSync`] the negotiated sinks received, so a
    /// replacement sink paces its presentation the way the one it took over
    /// from did. `None` when the run elected no clock.
    sink_clock_sync: Option<ClockSync>,
}

impl<'a, 's> MutationService<'a, 's> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        rx: Receiver<MutationRequest<'a>>,
        topology: LiveTopology<'a>,
        arms: Sender<BoxFuture<'a, Result<u64, G2gError>>>,
        spawn: MutationSpawn<'a, 's>,
        bus: Option<BusHandle>,
        progress: Option<PipelineProgress>,
        dropped: Arc<spin::Mutex<u64>>,
        path_is_live: bool,
        sink_clock_sync: Option<ClockSync>,
    ) -> Self {
        Self {
            rx,
            nodes: topology.nodes,
            edges: topology.edges,
            arms,
            spawn,
            bus,
            progress,
            dropped,
            path_is_live,
            sink_clock_sync,
        }
    }

    /// Serve requests for as long as the run lasts. Never returns: the run ends
    /// when its arms do, and this is selected against them.
    pub(crate) async fn run(mut self) -> core::convert::Infallible {
        loop {
            match self.rx.recv().await {
                Some(MutationRequest::Insert {
                    node,
                    side,
                    element,
                    reply,
                }) => {
                    let result = self.insert(&node, side, element).await;
                    let _ = reply.try_send(result);
                }
                Some(MutationRequest::Remove { node, reply }) => {
                    let result = self.remove(&node).await;
                    let _ = reply.try_send(result);
                }
                Some(MutationRequest::ReplaceSink {
                    node,
                    element,
                    reply,
                }) => {
                    let result = self.replace_sink(&node, element).await;
                    let _ = reply.try_send(result);
                }
                Some(MutationRequest::ReplaceSource {
                    node,
                    source,
                    reply,
                }) => {
                    let result = self.replace_source(&node, source).await;
                    let _ = reply.try_send(result);
                }
                // Every handle is gone, so nothing more can arrive; the arms
                // still have a run to finish.
                None => core::future::pending::<()>().await,
            }
        }
    }

    fn find(&self, name: &str) -> Result<usize, MutationError> {
        self.nodes
            .iter()
            .position(|n| n.name == name)
            .ok_or_else(|| MutationError::UnknownNode(name.to_string()))
    }

    /// `<category>N`, counting past every name the graph already carries, so a
    /// spliced element is addressable the way a negotiated one is.
    fn unique_name(&self, category: &str) -> String {
        for n in 0.. {
            let candidate = alloc::format!("{category}{n}");
            if !self.nodes.iter().any(|node| node.name == candidate) {
                return candidate;
            }
        }
        unreachable!("an unbounded counter always reaches an unused name")
    }

    /// This node's element-return channel, taken out of the topology. A node
    /// holding a source's keeps it: the two are addressed by different
    /// operations, so reaching for the wrong one leaves the node as it was.
    fn take_element_handback(
        &mut self,
        node: usize,
    ) -> Option<Receiver<Box<dyn DynAsyncElement + 'a>>> {
        match self.nodes[node].done.take() {
            Some(Handback::Element(rx)) => Some(rx),
            other => {
                self.nodes[node].done = other;
                None
            }
        }
    }

    /// [`take_element_handback`](Self::take_element_handback) for a source.
    fn take_source_handback(
        &mut self,
        node: usize,
    ) -> Option<Receiver<Box<dyn DynSourceLoop + 'a>>> {
        match self.nodes[node].done.take() {
            Some(Handback::Source(rx)) => Some(rx),
            other => {
                self.nodes[node].done = other;
                None
            }
        }
    }

    /// The one mutable edge on `side` of the node named `name`. An end with
    /// several edges (a tee below, a muxer above) names no single edge, so it is
    /// addressed from the other end instead. The count is over every graph edge
    /// on that side, not just the mutable ones: with one mutable edge among
    /// several, the splice would land on a branch the caller never named.
    fn edge_at(&self, name: &str, side: Side) -> Result<usize, MutationError> {
        let node = self.find(name)?;
        let (edges, total) = match side {
            Side::Below => (
                &self.nodes[node].out_edges,
                self.nodes[node].total_out_edges,
            ),
            Side::Above => (&self.nodes[node].in_edges, self.nodes[node].total_in_edges),
        };
        if total != 1 {
            return Err(MutationError::NotMutable(name.to_string()));
        }
        match edges.as_slice() {
            [edge] => Ok(*edge),
            _ => Err(MutationError::NotMutable(name.to_string())),
        }
    }

    async fn insert(
        &mut self,
        name: &str,
        side: Side,
        mut element: Box<dyn DynAsyncElement + 'a>,
    ) -> Result<String, MutationError> {
        let edge = self.edge_at(name, side)?;
        let endpoint = self.edges[edge].endpoint.clone();
        let consumer = self.edges[edge].consumer;
        let edge_caps = endpoint.caps().ok_or(MutationError::NoCaps)?;

        // Negotiate before anything is disturbed: the element against the caps
        // on the wire, then its output against what the chain below accepts.
        let out_caps = element
            .intercept_caps(&edge_caps)
            .map_err(MutationError::Refused)?;
        if out_caps != edge_caps && !accepts_downstream(&self.edges[edge].feasible, &out_caps) {
            return Err(MutationError::DownstreamRefused);
        }
        // A counter-proposal has nowhere to go: the upstream element is already
        // running under the caps it fixated at startup, which is what
        // `reject_refixate` says at startup too.
        element
            .configure_pipeline(&edge_caps)
            .map_err(MutationError::Refused)?
            .reject_refixate()
            .map_err(MutationError::Refused)?;
        element
            .configure_output(&out_caps)
            .map_err(MutationError::Refused)?;
        element.configure_liveness(self.path_is_live);
        let name = self.unique_name(element.log_category());
        element.set_instance_name(name.clone());

        let policy = self.edges[edge].policy;
        let capacity = self.edges[edge].capacity;
        let (mut feed_tx, feed_rx) = link(capacity);
        feed_tx.set_policy(policy);
        // A leaky link's drops are a whole-run total, so the new hop counts into
        // the same tally. The edge's observer wiring (counters, transit ring,
        // probe) stays with the hop that kept the consumer instead, which is
        // what keeps the transit ring paired with the receiver popping it.
        if policy != LinkPolicy::Block {
            feed_tx.set_drop_counter(self.dropped.clone());
        }

        // Stop the producer and take the link it was pushing through: that link
        // becomes the spliced element's output, so everything already queued on
        // it stays ahead of anything the new element emits and no drain is
        // needed.
        endpoint.request_park();
        let mut original = Detach(&endpoint)
            .await
            .map_err(|_| MutationError::GraphEnded)?;
        let spliced_endpoint = ProducerEndpoint::new();
        spliced_endpoint.set_caps(&out_caps);
        original.set_mutation(Some(spliced_endpoint.clone()));

        // A caps-changing splice announces itself here, while the producer is
        // still stopped: behind the queued packets, ahead of its own output.
        if out_caps != edge_caps {
            if let Err(e) = original.send_caps(out_caps.clone()).await {
                endpoint.unpark(Some(original));
                return Err(MutationError::Refused(e));
            }
        }

        let spliced = self.nodes.len();
        let (done_tx, done_rx) = bounded(1);
        let io = TransformArmIo {
            in_rx: feed_rx,
            out_tx: original,
            arm_rx: dead_directives(),
            coord: GraphCoordHandle::detached(),
            node: NodeId(spliced as u32),
            out_caps: out_caps.clone(),
            downstream_feasible: self.edges[edge].feasible.clone(),
            mode: self.edges[edge].mode,
            bus: self.bus.clone(),
            probe: None,
            control: None,
        };
        let arm = (self.spawn.transform)(element, io, done_tx);
        if self.arms.send(arm).await.is_err() {
            endpoint.unpark(None);
            return Err(MutationError::GraphEnded);
        }
        endpoint.unpark(Some(feed_tx));

        // The edge the producer kept now ends at the spliced element, and a new
        // edge carries that element's output to the consumer.
        let below = self.edges.len();
        self.edges.push(LiveEdge {
            endpoint: spliced_endpoint,
            consumer,
            feasible: self.edges[edge].feasible.clone(),
            policy,
            capacity,
            mode: self.edges[edge].mode,
        });
        self.nodes.push(LiveNode {
            name: name.clone(),
            out_edges: alloc::vec![below],
            in_edges: alloc::vec![edge],
            total_out_edges: 1,
            total_in_edges: 1,
            done: Some(Handback::Element(done_rx)),
            removable: true,
        });
        self.edges[edge].consumer = spliced;
        // The one shape the edge above the spliced element is known to carry:
        // the caps that element accepted and was configured for. Nothing solved
        // what else it would take, or what the chain below would do with the
        // output that produced, so a further splice here has to keep the caps as
        // they are and is refused otherwise.
        self.edges[edge].feasible = Some(CapsSet::one(edge_caps));
        retarget(&mut self.nodes[consumer].in_edges, edge, below);
        Ok(name)
    }

    async fn remove(&mut self, node: &str) -> Result<Box<dyn DynAsyncElement + 'a>, MutationError> {
        let removed = self.find(node)?;
        let not_mutable = || MutationError::NotMutable(node.to_string());
        if !self.nodes[removed].removable {
            return Err(not_mutable());
        }
        let above = self.edge_at(node, Side::Above)?;
        let below = self.edge_at(node, Side::Below)?;
        let consumer = self.edges[below].consumer;
        let up_endpoint = self.edges[above].endpoint.clone();
        let own_endpoint = self.edges[below].endpoint.clone();
        let done = self
            .take_element_handback(removed)
            .ok_or_else(not_mutable)?;
        let upstream_caps = up_endpoint.caps().ok_or(MutationError::NoCaps)?;
        let own_caps = own_endpoint.caps();

        // The consumer is about to start receiving the producer's caps instead
        // of this element's. It has to accept them, or the bypass would break it.
        let caps_change = own_caps.as_ref() != Some(&upstream_caps);
        if caps_change && !accepts_downstream(&self.edges[below].feasible, &upstream_caps) {
            // Put the element channel back: the graph is unchanged.
            self.nodes[removed].done = Some(Handback::Element(done));
            return Err(MutationError::DownstreamRefused);
        }

        // Stopping the producer and dropping the link it gave up closes the
        // removed element's input, so its arm drains what is queued, forwards
        // the results, and ends.
        up_endpoint.request_park();
        let feed = match Detach(&up_endpoint).await {
            Ok(feed) => feed,
            // The producer's arm ended, nothing was disturbed: restore the
            // handback so a retry reports the run's end, not NotMutable.
            Err(_) => {
                self.nodes[removed].done = Some(Handback::Element(done));
                return Err(MutationError::GraphEnded);
            }
        };
        // M1146: the flags go up only once the producer's link is in hand.
        // Holding it keeps the element's input open, so the arm cannot have read
        // either of them yet: raised before a refusal, they leave an arm that
        // swallows its own end of stream and a claimed link holding its
        // consumer's channel open, and the consumer then waits for an end that
        // never comes.
        //
        // The output link is claimed because that is the link the producer takes
        // over: this one arm ending must not close the channel the way an arm
        // ending on its own does.
        own_endpoint.claim_on_end();
        // M1132: the arm flushes the element on that close, so frames it was
        // holding internally reach the consumer through the same link, ahead of
        // the first frame the producer sends once it bypasses the element.
        own_endpoint.begin_drain();
        drop(feed);
        // Past here the removed element's input is closed, so there is no
        // arrangement to restore: a failure resumes the producer onto the dead
        // link it parked with, which fails its arm loud rather than wedging it.
        let mut output = match Detach(&own_endpoint).await {
            Ok(output) => output,
            Err(_) => {
                up_endpoint.unpark(None);
                return Err(MutationError::GraphEnded);
            }
        };
        let Some(element) = done.recv().await else {
            up_endpoint.unpark(None);
            return Err(MutationError::GraphEnded);
        };
        output.set_mutation(None);

        if caps_change {
            if let Err(e) = output.send_caps(upstream_caps).await {
                up_endpoint.unpark(None);
                return Err(MutationError::Refused(e));
            }
        }
        up_endpoint.unpark(Some(output));

        // The producer's edge now ends at the consumer, and carries what the
        // element's own edge was known to carry. The element's edge is left
        // behind: nothing names it any more.
        self.edges[above].consumer = consumer;
        self.edges[above].feasible = self.edges[below].feasible.take();
        retarget(&mut self.nodes[consumer].in_edges, below, above);
        self.nodes[removed].in_edges.clear();
        self.nodes[removed].out_edges.clear();
        self.nodes[removed].removable = false;
        Ok(element)
    }

    /// M1149: swap the element at the bottom of an edge. See
    /// [`GraphMutator::replace_sink`].
    async fn replace_sink(
        &mut self,
        node: &str,
        mut element: Box<dyn DynAsyncElement + 'a>,
    ) -> ReplacedSink<'a> {
        let replaced = self.find(node)?;
        let not_mutable = || MutationError::NotMutable(node.to_string());
        // A sink is a node with nothing below it, holding an element the runner
        // lent out. A terminal fan-in fails one test or the other.
        if self.nodes[replaced].total_out_edges != 0
            || !matches!(self.nodes[replaced].done, Some(Handback::Element(_)))
        {
            return Err(not_mutable());
        }
        let edge = self.edge_at(node, Side::Above)?;
        let endpoint = self.edges[edge].endpoint.clone();
        let edge_caps = endpoint.caps().ok_or(MutationError::NoCaps)?;

        // Negotiate before anything is disturbed, the way a sink arm takes a
        // mid-stream shape: the element's own constraint says what it reads the
        // edge as, and it is configured on that. A sink has no output pad, so
        // the startup path's `configure_output` has no counterpart here.
        let sink_caps = re_solve_downstream_dyn_sink(&edge_caps, &*element)
            .map_err(|_| MutationError::Refused(G2gError::CapsMismatch))?;
        // What the edge will accept once this sink is on it: the set the
        // replacement declares, which is what the runner's startup sweep would
        // have read off it. A sink that declares no set leaves the edge carrying
        // the one shape it was configured for, and a later caps change here is
        // refused the way it is above a spliced element.
        let feasible = match element.caps_constraint_as_sink() {
            crate::format_element::CapsConstraint::Accepts(set) => set.clone(),
            _ => CapsSet::one(edge_caps),
        };
        element
            .configure_pipeline(&sink_caps)
            .map_err(MutationError::Refused)?
            .reject_refixate()
            .map_err(MutationError::Refused)?;
        element.configure_liveness(self.path_is_live);
        // The elected clock reaches the replacement the way the startup sweep
        // gave it to every negotiated sink, so it paces its presentation; only
        // the election itself is settled.
        if let Some(sync) = &self.sink_clock_sync {
            element.set_clock_sync(sync.clone());
        }
        let name = self.unique_name(element.log_category());
        element.set_instance_name(name.clone());

        // Past here the old sink is being ended, so every refusal is behind us,
        // and a failure leaves the producer resuming onto the dead link it
        // parked with rather than an arrangement to restore, as in `remove`.
        let done = self
            .take_element_handback(replaced)
            .expect("the handback was checked above");
        endpoint.request_park();
        let feed = match Detach(&endpoint).await {
            Ok(feed) => feed,
            // The producer's arm ended, so nothing was disturbed: put the
            // handback back so a retry reports the run's end, not NotMutable.
            Err(_) => {
                self.nodes[replaced].done = Some(Handback::Element(done));
                return Err(MutationError::GraphEnded);
            }
        };
        // The old sink ends on an end of stream rather than on a closed channel:
        // it renders what is still queued for it and then finalizes, which is
        // what a sink writing a container needs before it is handed back.
        // Dropping the link behind the marker closes the channel.
        if feed.send_eos().await.is_err() {
            endpoint.unpark(None);
            return Err(MutationError::GraphEnded);
        }
        drop(feed);
        let Some(old) = done.recv().await else {
            endpoint.unpark(None);
            return Err(MutationError::GraphEnded);
        };

        // Only now does the replacement start, so the two never render the same
        // stretch of stream.
        let policy = self.edges[edge].policy;
        let capacity = self.edges[edge].capacity;
        let (mut feed_tx, feed_rx) = link(capacity);
        feed_tx.set_policy(policy);
        if policy != LinkPolicy::Block {
            feed_tx.set_drop_counter(self.dropped.clone());
        }
        advertise_orientation(&feed_rx, element.absorbs_orientation());
        let replacement = self.nodes.len();
        let (done_tx, done_rx) = bounded(1);
        let io = SinkArmIo {
            in_rx: feed_rx,
            arm_rx: dead_directives(),
            coord: GraphCoordHandle::detached(),
            node: NodeId(replacement as u32),
            mode: self.edges[edge].mode,
            bus: self.bus.clone(),
            // A replacement joins a stream already in flight: there is no
            // preroll for it to take and no state transition to gate on.
            state: None,
            progress: self.progress.clone(),
            probe: None,
            control: None,
        };
        let arm = (self.spawn.sink)(element, io, done_tx);
        if self.arms.send(arm).await.is_err() {
            endpoint.unpark(None);
            return Err(MutationError::GraphEnded);
        }
        endpoint.unpark(Some(feed_tx));

        self.nodes.push(LiveNode {
            name: name.clone(),
            out_edges: Vec::new(),
            in_edges: alloc::vec![edge],
            total_out_edges: 0,
            total_in_edges: 1,
            done: Some(Handback::Element(done_rx)),
            removable: false,
        });
        self.edges[edge].consumer = replacement;
        self.edges[edge].feasible = Some(feasible);
        // The node the old sink sat on stays in the table, named and addressing
        // nothing, so its name is never handed out again.
        self.nodes[replaced].in_edges.clear();
        Ok((name, old))
    }

    /// M1149: swap the element at the top of an edge. See
    /// [`GraphMutator::replace_source`].
    async fn replace_source(
        &mut self,
        node: &str,
        mut source: Box<dyn DynSourceLoop + 'a>,
    ) -> ReplacedSource<'a> {
        let replaced = self.find(node)?;
        let not_mutable = || MutationError::NotMutable(node.to_string());
        // A source is a node with nothing above it, holding a source the runner
        // lent out. A terminal fan-out source fails the second test.
        if self.nodes[replaced].total_in_edges != 0
            || !matches!(self.nodes[replaced].done, Some(Handback::Source(_)))
        {
            return Err(not_mutable());
        }
        let edge = self.edge_at(node, Side::Below)?;
        let endpoint = self.edges[edge].endpoint.clone();
        let edge_caps = endpoint.caps().ok_or(MutationError::NoCaps)?;

        // Keeping the shape on the wire needs no consent from anyone, so it is
        // what the replacement is asked for first; only a source that cannot
        // produce it has to find one the chain below is known to accept.
        let produced = source
            .produced_caps()
            .await
            .map_err(MutationError::Refused)?;
        let caps = match produced.accepts(&edge_caps) {
            true => edge_caps.clone(),
            false => self.agreed_caps(edge, &produced)?,
        };
        source
            .configure_pipeline(&caps)
            .map_err(MutationError::Refused)?
            .reject_refixate()
            .map_err(MutationError::Refused)?;
        let name = self.unique_name(source.log_category());
        source.set_instance_name(name.clone());

        // Past here the old source is being retired, so every refusal is behind
        // us; a failure from now on leaves the edge without a producer, which
        // ends the stream below it rather than wedging it.
        let done = self
            .take_source_handback(replaced)
            .expect("the handback was checked above");
        endpoint.request_park();
        let out_tx = match Detach(&endpoint).await {
            Ok(out_tx) => out_tx,
            // The arm ended, nothing was disturbed: restore the handback so a
            // retry reports the run's end, not NotMutable.
            Err(_) => {
                self.nodes[replaced].done = Some(Handback::Source(done));
                return Err(MutationError::GraphEnded);
            }
        };
        // The link is in hand, so the old source resumes onto a dead one and
        // ends the way it does when its consumer goes away. What it had already
        // queued stays on the link the replacement inherits.
        endpoint.retire();
        let Some(old) = done.recv().await else {
            return Err(MutationError::GraphEnded);
        };
        // Read before the endpoint is handed on: `rehome` clears the timeline
        // for the producer taking over.
        let opening = continued_segment(&endpoint);
        endpoint.rehome();

        if caps != edge_caps {
            if let Err(e) = out_tx.send_caps(caps.clone()).await {
                return Err(MutationError::Refused(e));
            }
            // Queued from outside a push, so the sticky shape is set by hand.
            endpoint.set_caps(&caps);
        }
        let (done_tx, done_rx) = bounded(1);
        let io = SourceArmIo {
            out_tx,
            bus: self.bus.clone(),
            progress: self.progress.clone(),
            segment: opening,
        };
        let arm = (self.spawn.source)(source, io, done_tx);
        if self.arms.send(arm).await.is_err() {
            return Err(MutationError::GraphEnded);
        }

        self.nodes.push(LiveNode {
            name: name.clone(),
            out_edges: alloc::vec![edge],
            in_edges: Vec::new(),
            total_out_edges: 1,
            total_in_edges: 0,
            done: Some(Handback::Source(done_rx)),
            removable: false,
        });
        // As in `replace_sink`: the node the old source sat on keeps its name
        // and loses its edge.
        self.nodes[replaced].out_edges.clear();
        Ok((name, old))
    }

    /// One shape the replacement produces that the chain below `edge` accepts.
    /// No feasibility snapshot means no consent, exactly as for a splice.
    fn agreed_caps(&self, edge: usize, produced: &CapsSet) -> Result<Caps, MutationError> {
        let feasible = self.edges[edge]
            .feasible
            .as_ref()
            .ok_or(MutationError::DownstreamRefused)?;
        produced
            .intersect(feasible)
            .fixate()
            .ok_or(MutationError::DownstreamRefused)
    }
}

/// The segment a replacement source opens with: its own timeline, based at the
/// running time its predecessor reached, so the pipeline's running time keeps
/// moving forward while the new source stamps from its own zero. An edge nothing
/// has crossed yet continues from the base of the segment in force.
fn continued_segment(endpoint: &ProducerEndpoint) -> Segment {
    let (segment, last_frame) = endpoint.timeline();
    let segment = segment.unwrap_or_else(Segment::new);
    let base = match last_frame {
        Some((pts, duration)) => segment
            .to_running_time(pts)
            .unwrap_or(segment.base)
            .saturating_add(duration),
        None => segment.base,
    };
    Segment {
        base,
        ..Segment::new()
    }
}

/// Point a node's edge list at `to` wherever it named `from`, so a splice or a
/// removal moves the hop a consumer reads from.
fn retarget(edges: &mut [usize], from: usize, to: usize) {
    for e in edges.iter_mut() {
        if *e == from {
            *e = to;
        }
    }
}

/// Whether the chain below an edge consents to carrying `caps`. No snapshot of
/// what it accepts means no consent: the runner computes one per edge at
/// startup, and where it could not (a chain the backward sweep cannot express)
/// there is nothing to check a caps change against, and an element that refuses
/// one mid-stream fails the run rather than negotiating.
fn accepts_downstream(feasible: &Option<CapsSet>, caps: &crate::caps::Caps) -> bool {
    feasible.as_ref().is_some_and(|set| set.accepts(caps))
}

/// A β directive channel with no coordinator behind it: a spliced element takes
/// no part in the allocation cascade, so its arm sees the control side closed
/// and selects on data alone.
fn dead_directives() -> Receiver<ArmDirective> {
    let (tx, rx) = bounded(1);
    drop(tx);
    rx
}

/// Awaits the link a stopped producer (or an ended arm) left on its endpoint.
struct Detach<'e>(&'e ProducerEndpoint);

impl Future for Detach<'_> {
    type Output = Result<LinkSender, G2gError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.poll_detached(cx)
    }
}
