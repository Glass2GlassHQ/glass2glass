//! Structural mutation of a running graph: splicing a chain of
//! transforms onto a live edge and lifting one back off, swapping the source or
//! the sink at either end of it, attaching and dropping a tee branch and
//! attaching a muxer input, without stopping the pipeline.
//!
//! The data plane pays for this once per push: a relaxed load of the edge's
//! [`ProducerEndpoint`] gate. Everything else happens inside a mutation op,
//! which runs in the run future itself (`MutationService`), so it can touch the
//! arms, the links and the elements the way the runner does.
//!
//! Scope: a transform position on a 1:1 edge. The producing end is a source, a
//! transform or one output of a tee / demux; the consuming end is a transform, a
//! sink, one input pad of a muxer or terminal fan-in, or the single
//! input of a tee / demux. The structural nodes themselves are not
//! splice points ([`MutationError::NotMutable`]), but a tee takes a whole branch
//! and a muxer a whole input; the two ends are not splice points either, but the
//! elements sitting on them are replaceable in place.

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};
use core::time::Duration;

use crate::bus::{BusHandle, BusMessage};
use crate::caps::{Caps, CapsSet};
use crate::clock::{ClockSync, DynAsyncClock};
use crate::element::{BoxFuture, DynAsyncElement};
use crate::error::G2gError;
use crate::graph::NodeId;
use crate::link::LinkPolicy;
use crate::property::PropValue;
use crate::runtime::channel::{
    advertise_orientation, bounded, link, LinkReceiver, LinkSender, ProducerEndpoint, Receiver,
    Sender,
};
use crate::runtime::fanin::DynSourceLoop;
use crate::runtime::graph_runner::{
    dead_directives, run_source, BranchMode, GraphCoordHandle, SinkArmIo, SourceArmIo,
    TransformArmIo,
};
use crate::runtime::join::{select2, Either};
use crate::runtime::mutation_error::{MutationError, MutationKind};
use crate::runtime::progress::PipelineProgress;
use crate::runtime::property_mailbox::{PropertyMailbox, PropertyRequest};
use crate::runtime::runner::re_solve_downstream_dyn_sink;
use crate::runtime::solver::{resolve_forward_output, ForwardResolve};
use crate::segment::Segment;

/// Builds the converter chain a caps-changing operation needs when the shape it
/// would put on an edge is not in the set the chain below accepts: from the
/// shape produced to one in that set, in stream order. `None` when it has no
/// chain for the pair, which refuses the operation. Installed on a handle with
/// [`GraphMutator::with_converters`]; `g2g-plugins` builds one from its registry.
pub type ConverterFactory<'a> =
    Arc<dyn Fn(&Caps, &CapsSet) -> Option<Vec<Box<dyn DynAsyncElement + 'a>>> + Send + Sync + 'a>;

/// A handle on a *running* graph's topology: splice a transform onto a
/// live edge, or lift one back off. Obtained beside the run future from
/// [`RunBuilder::run_mutable`](crate::runtime::RunBuilder::run_mutable) or
/// [`run_graph_mutable`](crate::runtime::run_graph_mutable),
/// like [`DynamicFanoutHandle`](crate::runtime::DynamicFanoutHandle): drive the
/// run future while using the handle from another task.
///
/// Positions are named by the instance name the runner assigned (a launch line's
/// `name=`, else `<category>N`), and an inserted element gets one of its own,
/// returned by [`insert_after`](Self::insert_after) /
/// [`insert_before`](Self::insert_before) so it can be removed later.
///
/// Each operation completes at the producer's next packet boundary, so a
/// producer that has gone quiet defers it until it pushes again, or until the
/// handle's [`timeout`](Self::with_timeout) if it carries one.
pub struct GraphMutator<'a> {
    tx: Sender<MutationRequest<'a>>,
    timeout: Option<Duration>,
    converters: Option<ConverterFactory<'a>>,
}

impl<'a> Clone for GraphMutator<'a> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            timeout: self.timeout,
            converters: self.converters.clone(),
        }
    }
}

impl core::fmt::Debug for GraphMutator<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GraphMutator")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl<'a> GraphMutator<'a> {
    /// A handle whose operations give up after `timeout` of waiting for a
    /// producer to reach a packet boundary, answering
    /// [`MutationError::TimedOut`] with the graph untouched. The wait that is
    /// bounded is the one before anything moves: once an old element is being
    /// ended the operation completes regardless, since that cannot be undone.
    /// A run whose clock cannot sleep on a deadline answers
    /// [`MutationError::NoTimer`] instead.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// A handle that splices converters behind an element whose output the
    /// chain below would refuse, instead of refusing the operation. The factory
    /// is asked for the chain from what the element produces to what the edge
    /// accepts; the element and the converters go in as one splice, or not at
    /// all.
    pub fn with_converters(mut self, converters: ConverterFactory<'a>) -> Self {
        self.converters = Some(converters);
        self
    }

    /// Check whether an element can be inserted below `after`, using the same
    /// current caps and downstream checks as [`insert_after`](Self::insert_after).
    pub async fn validate_insert_after(
        &self,
        after: &str,
        element: Box<dyn DynAsyncElement + 'a>,
    ) -> Result<(), MutationError> {
        self.validate_insert(after, Side::Below, alloc::vec![element])
            .await
    }

    /// Check whether an element can be inserted above `before`, using the same
    /// current caps and downstream checks as [`insert_before`](Self::insert_before).
    pub async fn validate_insert_before(
        &self,
        before: &str,
        element: Box<dyn DynAsyncElement + 'a>,
    ) -> Result<(), MutationError> {
        self.validate_insert(before, Side::Above, alloc::vec![element])
            .await
    }

    async fn validate_insert(
        &self,
        node: &str,
        side: Side,
        elements: Vec<Box<dyn DynAsyncElement + 'a>>,
    ) -> Result<(), MutationError> {
        let (reply, answer) = bounded(1);
        self.send(
            MutationOp::ValidateInsert {
                node: node.to_string(),
                side,
                elements,
                reply,
            },
            answer,
        )
        .await
    }

    /// Splice `element` onto the edge leaving the element named `after`, so the
    /// stream runs `after -> element -> (whatever `after` fed)`. Returns the
    /// instance name the new element was given.
    ///
    /// The element is negotiated against the caps flowing on that edge right now
    /// before anything is disturbed: it must accept them, and what it emits must
    /// be in the set the downstream chain accepts (or a handle
    /// [with converters](Self::with_converters) must find a chain to it). A
    /// refusal leaves the graph running and drops the element.
    pub async fn insert_after(
        &self,
        after: &str,
        element: Box<dyn DynAsyncElement + 'a>,
    ) -> Result<String, MutationError> {
        self.insert(after, Side::Below, alloc::vec![element])
            .await
            .map(first_name)
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
        self.insert(before, Side::Above, alloc::vec![element])
            .await
            .map(first_name)
    }

    /// [`insert_after`](Self::insert_after) for a chain of elements, spliced in
    /// stream order as one operation: the producer stops once, and either every
    /// element goes in or none does. Each element is negotiated against what the
    /// one before it emits, and only the last one's output has to be accepted
    /// below. Returns the instance names in the same order. An empty chain
    /// changes nothing and returns no names.
    pub async fn insert_chain_after(
        &self,
        after: &str,
        elements: Vec<Box<dyn DynAsyncElement + 'a>>,
    ) -> Result<Vec<String>, MutationError> {
        self.insert(after, Side::Below, elements).await
    }

    /// [`insert_before`](Self::insert_before) for a chain of elements, as
    /// [`insert_chain_after`](Self::insert_chain_after).
    pub async fn insert_chain_before(
        &self,
        before: &str,
        elements: Vec<Box<dyn DynAsyncElement + 'a>>,
    ) -> Result<Vec<String>, MutationError> {
        self.insert(before, Side::Above, elements).await
    }

    async fn insert(
        &self,
        node: &str,
        side: Side,
        elements: Vec<Box<dyn DynAsyncElement + 'a>>,
    ) -> Result<Vec<String>, MutationError> {
        if elements.is_empty() {
            return Ok(Vec::new());
        }
        let (reply, answer) = bounded(1);
        self.send(
            MutationOp::Insert {
                node: node.to_string(),
                side,
                elements,
                reply,
            },
            answer,
        )
        .await
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
        self.send(
            MutationOp::Remove {
                node: node.to_string(),
                reply,
            },
            answer,
        )
        .await
    }

    /// Take over the sink named `node` with `element`, and hand the old
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
        self.send(
            MutationOp::ReplaceSink {
                node: node.to_string(),
                element,
                reply,
            },
            answer,
        )
        .await
    }

    /// Take over the source named `node` with `source`, and hand the old
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
        self.send(
            MutationOp::ReplaceSource {
                node: node.to_string(),
                source,
                reply,
            },
            answer,
        )
        .await
    }

    /// Set a property on the element named `node` while it runs, the
    /// `gst-launch` knob applied mid-stream. The element's own
    /// [`set_property`](crate::AsyncElement::set_property) runs inside the arm
    /// that owns it, at that arm's next packet boundary, so the element is never
    /// touched from another task and the value takes effect between packets
    /// rather than inside one.
    ///
    /// A value the element refuses comes back as
    /// [`MutationError::PropertyRejected`] carrying its own [`PropError`](crate::property::PropError), which
    /// is how an unknown property name is reported too. A node whose arm hands
    /// packets to no element (a source, a tee, a demux) is
    /// [`MutationError::NotMutable`]: only a transform, a sink or a fan-in
    /// position can act between packets, the same rule animated properties
    /// follow.
    pub async fn set_property(
        &self,
        node: &str,
        name: &str,
        value: PropValue,
    ) -> Result<(), MutationError> {
        let (reply, answer) = bounded(1);
        self.send(
            MutationOp::SetProperty {
                node: node.to_string(),
                name: name.to_string(),
                value,
                reply,
            },
            answer,
        )
        .await
    }

    /// Read a property off the element named `node` while it runs, at the same
    /// packet boundary [`set_property`](Self::set_property) writes at, so a read
    /// after a write sees that write. `Ok(None)` is the element's own answer for
    /// a name it does not carry.
    pub async fn get_property(
        &self,
        node: &str,
        name: &str,
    ) -> Result<Option<PropValue>, MutationError> {
        let (reply, answer) = bounded(1);
        self.send(
            MutationOp::GetProperty {
                node: node.to_string(),
                name: name.to_string(),
                reply,
            },
            answer,
        )
        .await
    }

    /// Attach `branch` under the tee named `tee`, the request-pad analog: the
    /// last element is the branch's sink, the ones before it transforms in stream
    /// order. The branch is negotiated against the caps the tee broadcasts, the
    /// sink taking the shape the last transform emits (or a chain of converters
    /// to one it takes, on a handle [with converters](Self::with_converters)),
    /// before the tee is touched; the tee then starts broadcasting to it between
    /// two packets, with nothing parked. Returns the instance names in order.
    ///
    /// The new sink starts playing, is handed the elected clock, and takes no
    /// part in the clock election, the latency fold or the allocation cascade,
    /// like a replacement sink. A node that is not a plain tee is
    /// [`MutationError::NotMutable`]; a tee nothing has reached yet is
    /// [`MutationError::NoCaps`].
    pub async fn add_branch(
        &self,
        tee: &str,
        branch: Vec<Box<dyn DynAsyncElement + 'a>>,
    ) -> Result<Vec<String>, MutationError> {
        let (reply, answer) = bounded(1);
        self.send(
            MutationOp::AddBranch {
                tee: tee.to_string(),
                branch,
                reply,
            },
            answer,
        )
        .await
    }

    /// Drop the branch ending at the sink named `sink` from under its tee, and
    /// hand its elements back in stream order, the sink last. The tee stops
    /// feeding the branch at its next packet boundary; what was already queued
    /// drains through every element and the sink takes an end of stream, so it
    /// finalizes before it comes back. The elements in between are the ones
    /// spliced onto the branch since, or built with it. A sink that is not under
    /// a tee is [`MutationError::NotMutable`].
    pub async fn remove_branch(
        &self,
        sink: &str,
    ) -> Result<Vec<Box<dyn DynAsyncElement + 'a>>, MutationError> {
        let (reply, answer) = bounded(1);
        self.send(
            MutationOp::RemoveBranch {
                sink: sink.to_string(),
                reply,
            },
            answer,
        )
        .await
    }

    /// Attach `source` to the next free declared pad of the muxer named `muxer`,
    /// the request sink-pad analog, and return the name the source was given.
    /// The muxer's arm negotiates the pad between two packets: the source's
    /// produce set against the pad's constraint, then the element's own veto
    /// ([`accepts_runtime_input`](crate::MultiInputElement::accepts_runtime_input)),
    /// which comes back as [`MutationError::Refused`]. A muxer with every
    /// declared pad in use is [`MutationError::NoFreePad`]. The source opens on
    /// a fresh segment and takes no part in the clock election or the
    /// allocation cascade.
    pub async fn add_input(
        &self,
        muxer: &str,
        source: Box<dyn DynSourceLoop + 'a>,
    ) -> Result<String, MutationError> {
        let (reply, answer) = bounded(1);
        self.send(
            MutationOp::AddInput {
                muxer: muxer.to_string(),
                source,
                reply,
            },
            answer,
        )
        .await
    }

    async fn send<T>(
        &self,
        op: MutationOp<'a>,
        answer: Receiver<Result<T, MutationError>>,
    ) -> Result<T, MutationError> {
        let request = MutationRequest {
            timeout_ns: self
                .timeout
                .map(|t| u64::try_from(t.as_nanos()).unwrap_or(u64::MAX)),
            converters: self.converters.clone(),
            op,
        };
        self.tx
            .send(request)
            .await
            .map_err(|_| MutationError::GraphEnded)?;
        answer.recv().await.ok_or(MutationError::GraphEnded)?
    }
}

/// The one name a single-element insert produced.
fn first_name(mut names: Vec<String>) -> String {
    names.swap_remove(0)
}

/// Which edge of the named node a splice addresses: the one below it (the
/// element that produces into the edge) or the one above it (the element that
/// consumes from it). Whichever end is unique names the edge.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Side {
    Below,
    Above,
}

/// One operation with the handle's policy for it. The work runs in the run
/// future, which is where the arms and links live.
#[allow(missing_debug_implementations)]
pub(crate) struct MutationRequest<'a> {
    timeout_ns: Option<u64>,
    converters: Option<ConverterFactory<'a>>,
    op: MutationOp<'a>,
}

/// One operation, carrying its own reply channel.
#[allow(missing_debug_implementations)]
pub(crate) enum MutationOp<'a> {
    ValidateInsert {
        node: String,
        side: Side,
        elements: Vec<Box<dyn DynAsyncElement + 'a>>,
        reply: Sender<Result<(), MutationError>>,
    },
    Insert {
        node: String,
        side: Side,
        elements: Vec<Box<dyn DynAsyncElement + 'a>>,
        reply: Sender<Result<Vec<String>, MutationError>>,
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
    SetProperty {
        node: String,
        name: String,
        value: PropValue,
        reply: Sender<Result<(), MutationError>>,
    },
    GetProperty {
        node: String,
        name: String,
        reply: Sender<Result<Option<PropValue>, MutationError>>,
    },
    AddBranch {
        tee: String,
        branch: Vec<Box<dyn DynAsyncElement + 'a>>,
        reply: Sender<Result<Vec<String>, MutationError>>,
    },
    RemoveBranch {
        sink: String,
        reply: Sender<Result<Vec<Box<dyn DynAsyncElement + 'a>>, MutationError>>,
    },
    AddInput {
        muxer: String,
        source: Box<dyn DynSourceLoop + 'a>,
        reply: Sender<Result<String, MutationError>>,
    },
}

/// What a sink replacement answers with: the fresh instance name, and the
/// element it took the place of.
pub(crate) type ReplacedSink<'a> = Result<(String, Box<dyn DynAsyncElement + 'a>), MutationError>;

/// [`ReplacedSink`] for a source replacement.
pub(crate) type ReplacedSource<'a> = Result<(String, Box<dyn DynSourceLoop + 'a>), MutationError>;

/// A runtime input on its way to a muxer arm: the pad reserved for it, the set
/// the source can produce, and the link the arm reads the pad from once it has
/// admitted it. The arm negotiates the pad against its own element, which only
/// it may touch, and answers with the fixated caps or the refusal.
#[allow(missing_debug_implementations)]
pub(crate) struct PadAdd {
    pub(crate) pad: usize,
    pub(crate) produced: CapsSet,
    pub(crate) pad_rx: LinkReceiver,
    pub(crate) reply: Sender<Result<Caps, G2gError>>,
}

/// Build the mutator handle and the request channel the service reads. The
/// handle is returned to the caller before the run starts, the receiver goes
/// into the run future.
pub(crate) fn mutation_channel<'a>(
    capacity: usize,
) -> (GraphMutator<'a>, Receiver<MutationRequest<'a>>) {
    let (tx, rx) = bounded(capacity);
    (
        GraphMutator {
            tx,
            timeout: None,
            converters: None,
        },
        rx,
    )
}

/// How a node takes new edges while it runs: the arms of a tee and a muxer
/// accept them between packets through a channel the service holds the other
/// end of.
#[allow(missing_debug_implementations)]
pub(crate) enum Growth {
    /// A transform, a source, a sink: edges come and go only by splicing.
    Fixed,
    /// A plain tee: a branch link sent here is broadcast to from the next
    /// packet on, under the tee's own policy for a branch that cannot re-solve.
    Branches {
        adds: Sender<LinkSender>,
        mode: BranchMode,
    },
    /// A muxer: an input sent here is negotiated and read from the next packet
    /// on. `pads_in_use` are the declared pad indices already taken, the
    /// graph's at startup and each admitted input since.
    Inputs {
        adds: Sender<PadAdd>,
        pads_in_use: Vec<usize>,
        declared: usize,
    },
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
    /// Where a live property operation is queued for this node's arm.
    /// `None` on a node whose arm hands packets to no element of its own.
    pub(crate) properties: Option<PropertyMailbox>,
    pub(crate) growth: Growth,
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

/// The runner's live topology as the mutator addresses it. The receiving ends of
/// the growth channels wait here, per node, for the arm that takes them.
#[allow(missing_debug_implementations)]
pub(crate) struct LiveTopology<'a> {
    pub(crate) nodes: Vec<LiveNode<'a>>,
    pub(crate) edges: Vec<LiveEdge>,
    pub(crate) branch_adds: Vec<Option<Receiver<LinkSender>>>,
    pub(crate) pad_adds: Vec<Option<Receiver<PadAdd>>>,
}

/// Builds the arm future for an element the mutator spliced in. The cooperative
/// runner drives it on the caller's executor; the thread-per-arm runner hands it
/// to its spawner. `done` takes the element back when the arm ends.
pub(crate) type SpliceSpawn<'a, 'f, 's> = Box<
    dyn Fn(
            Box<dyn DynAsyncElement + 'a>,
            TransformArmIo,
            Sender<Box<dyn DynAsyncElement + 'a>>,
        ) -> BoxFuture<'f, Result<u64, G2gError>>
        + 's,
>;

/// [`SpliceSpawn`] for a replacement sink's arm.
pub(crate) type SinkSpawn<'a, 'f, 's> = Box<
    dyn Fn(
            Box<dyn DynAsyncElement + 'a>,
            SinkArmIo,
            Sender<Box<dyn DynAsyncElement + 'a>>,
        ) -> BoxFuture<'f, Result<u64, G2gError>>
        + 's,
>;

/// [`SpliceSpawn`] for a replacement source's arm.
pub(crate) type SourceSpawn<'a, 'f, 's> = Box<
    dyn Fn(
            Box<dyn DynSourceLoop + 'a>,
            SourceArmIo,
            Sender<Box<dyn DynSourceLoop + 'a>>,
        ) -> BoxFuture<'f, Result<u64, G2gError>>
        + 's,
>;

/// How the run driving this mutator starts an arm, one per position the mutator
/// can put an element in. The two runners differ only in where the arm ends up:
/// the caller's executor, or a worker thread of its own. `'f` is the arm
/// future's lifetime, the run's rather than the elements', since the run also
/// borrows its clock.
#[allow(missing_debug_implementations)]
pub(crate) struct MutationSpawn<'a, 'f, 's> {
    pub(crate) transform: SpliceSpawn<'a, 'f, 's>,
    pub(crate) sink: SinkSpawn<'a, 'f, 's>,
    pub(crate) source: SourceSpawn<'a, 'f, 's>,
}

/// Drive an element the mutator spliced in, handing it back through `done` when
/// its arm ends. The element stays owned here and its arm runs on a borrow of
/// it, which is what makes giving it back possible.
pub(crate) async fn spliced_arm<'a>(
    mut element: Box<dyn DynAsyncElement + 'a>,
    io: TransformArmIo,
    done: Sender<Box<dyn DynAsyncElement + 'a>>,
) -> Result<u64, G2gError> {
    let result = element.drive_transform_arm_borrowed(io).await;
    let _ = done.try_send(element);
    result
}

/// [`spliced_arm`] for a sink position: a mutable run drives every sink
/// this way, so a replace can hand the old one back once it has finalized.
pub(crate) async fn lent_sink_arm<'a>(
    mut element: Box<dyn DynAsyncElement + 'a>,
    io: SinkArmIo,
    done: Sender<Box<dyn DynAsyncElement + 'a>>,
) -> Result<u64, G2gError> {
    let result = element.drive_sink_arm_borrowed(io).await;
    let _ = done.try_send(element);
    result
}

/// [`spliced_arm`] for a source position. A source that was retired ends
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

/// A spliced chain, negotiated and configured but not yet in the graph: the
/// edge it goes on, the shape that edge carries, and what each element emits.
struct ChainPlan {
    edge: usize,
    edge_caps: Caps,
    out_caps: Vec<Caps>,
}

/// The mutator's other half, running inside the run future: it owns the live
/// topology and performs each operation against the arms and links themselves.
#[allow(missing_debug_implementations)]
pub(crate) struct MutationService<'a, 'f, 's> {
    rx: Receiver<MutationRequest<'a>>,
    nodes: Vec<LiveNode<'a>>,
    edges: Vec<LiveEdge>,
    /// Sends a spliced element's arm into the run's growable join set.
    arms: Sender<BoxFuture<'f, Result<u64, G2gError>>>,
    spawn: MutationSpawn<'a, 'f, 's>,
    bus: Option<BusHandle>,
    /// The run's position / duration handle, so a replacement source or sink
    /// keeps answering the queries the one it took over from did.
    progress: Option<PipelineProgress>,
    /// The run's leaky-drop tally, installed on a spliced hop so its drops
    /// reach `RunStats` like every other link's.
    dropped: Arc<spin::Mutex<u64>>,
    /// The liveness the runner folded at startup, handed to a spliced element
    /// the way it reached the negotiated ones.
    path_is_live: bool,
    /// The latency-folded [`ClockSync`] the negotiated sinks received, so a
    /// replacement sink paces its presentation the way the one it took over
    /// from did. `None` when the run elected no clock.
    sink_clock_sync: Option<ClockSync>,
    /// The run's link depth, for an edge built from nothing (a new branch or
    /// input has no edge to copy one from).
    link_capacity: usize,
    /// The timer a handle's timeout is measured on: the run's tick clock, which
    /// is what sleeps on a deadline. `None` leaves every wait unbounded.
    ticker: Option<&'s dyn DynAsyncClock>,
}

impl<'a, 'f, 's> MutationService<'a, 'f, 's> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        rx: Receiver<MutationRequest<'a>>,
        topology: LiveTopology<'a>,
        arms: Sender<BoxFuture<'f, Result<u64, G2gError>>>,
        spawn: MutationSpawn<'a, 'f, 's>,
        bus: Option<BusHandle>,
        progress: Option<PipelineProgress>,
        dropped: Arc<spin::Mutex<u64>>,
        path_is_live: bool,
        sink_clock_sync: Option<ClockSync>,
        link_capacity: usize,
        ticker: Option<&'s dyn DynAsyncClock>,
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
            link_capacity,
            ticker,
        }
    }

    /// Serve requests for as long as the run lasts. Never returns: the run ends
    /// when its arms do, and this is selected against them.
    pub(crate) async fn run(mut self) -> core::convert::Infallible {
        loop {
            let Some(request) = self.rx.recv().await else {
                // Every handle is gone, so nothing more can arrive; the arms
                // still have a run to finish.
                core::future::pending::<()>().await;
                unreachable!("a pending future never resolves")
            };
            let MutationRequest {
                timeout_ns,
                converters,
                op,
            } = request;
            let deadline = match (timeout_ns, self.ticker) {
                (None, _) => Ok(None),
                (Some(timeout), Some(ticker)) => Ok(Some(ticker.now_ns().saturating_add(timeout))),
                (Some(_), None) => Err(MutationError::NoTimer),
            };
            match op {
                MutationOp::ValidateInsert {
                    node,
                    side,
                    mut elements,
                    reply,
                } => {
                    let result = self
                        .prepare_chain(&node, side, &mut elements, converters.as_ref())
                        .map(|_| ());
                    let _ = reply.try_send(result);
                }
                MutationOp::Insert {
                    node,
                    side,
                    elements,
                    reply,
                } => {
                    let result = match deadline {
                        Ok(deadline) => {
                            self.insert(&node, side, elements, converters.as_ref(), deadline)
                                .await
                        }
                        Err(e) => Err(e),
                    };
                    self.post(MutationKind::Insert, &node, result.clone());
                    let _ = reply.try_send(result);
                }
                MutationOp::Remove { node, reply } => {
                    let result = match deadline {
                        Ok(deadline) => self.remove(&node, deadline).await,
                        Err(e) => Err(e),
                    };
                    self.post(
                        MutationKind::Remove,
                        &node,
                        result.as_ref().map(|_| Vec::new()).map_err(Clone::clone),
                    );
                    let _ = reply.try_send(result);
                }
                MutationOp::ReplaceSink {
                    node,
                    element,
                    reply,
                } => {
                    let result = match deadline {
                        Ok(deadline) => self.replace_sink(&node, element, deadline).await,
                        Err(e) => Err(e),
                    };
                    self.post(
                        MutationKind::ReplaceSink,
                        &node,
                        result
                            .as_ref()
                            .map(|(name, _)| alloc::vec![name.clone()])
                            .map_err(Clone::clone),
                    );
                    let _ = reply.try_send(result);
                }
                MutationOp::ReplaceSource {
                    node,
                    source,
                    reply,
                } => {
                    let result = match deadline {
                        Ok(deadline) => self.replace_source(&node, source, deadline).await,
                        Err(e) => Err(e),
                    };
                    self.post(
                        MutationKind::ReplaceSource,
                        &node,
                        result
                            .as_ref()
                            .map(|(name, _)| alloc::vec![name.clone()])
                            .map_err(Clone::clone),
                    );
                    let _ = reply.try_send(result);
                }
                MutationOp::SetProperty {
                    node,
                    name,
                    value,
                    reply,
                } => {
                    let result = match deadline {
                        Ok(deadline) => self.set_property(&node, &name, value, deadline).await,
                        Err(e) => Err(e),
                    };
                    self.post(
                        MutationKind::SetProperty,
                        &node,
                        result.clone().map(|()| Vec::new()),
                    );
                    let _ = reply.try_send(result);
                }
                MutationOp::GetProperty { node, name, reply } => {
                    let result = match deadline {
                        Ok(deadline) => self.get_property(&node, &name, deadline).await,
                        Err(e) => Err(e),
                    };
                    let _ = reply.try_send(result);
                }
                MutationOp::AddBranch { tee, branch, reply } => {
                    let result = self.add_branch(&tee, branch, converters.as_ref()).await;
                    self.post(MutationKind::AddBranch, &tee, result.clone());
                    let _ = reply.try_send(result);
                }
                MutationOp::RemoveBranch { sink, reply } => {
                    let result = match deadline {
                        Ok(deadline) => self.remove_branch(&sink, deadline).await,
                        Err(e) => Err(e),
                    };
                    self.post(
                        MutationKind::RemoveBranch,
                        &sink,
                        result.as_ref().map(|_| Vec::new()).map_err(Clone::clone),
                    );
                    let _ = reply.try_send(result);
                }
                MutationOp::AddInput {
                    muxer,
                    source,
                    reply,
                } => {
                    let result = self.add_input(&muxer, source).await;
                    self.post(
                        MutationKind::AddInput,
                        &muxer,
                        result.clone().map(|name| alloc::vec![name]),
                    );
                    let _ = reply.try_send(result);
                }
            }
        }
    }

    /// Tell the bus what an operation did, for whoever watches the graph besides
    /// the caller: the names it created, or why it was refused.
    fn post(
        &self,
        operation: MutationKind,
        node: &str,
        outcome: Result<Vec<String>, MutationError>,
    ) {
        if let Some(bus) = &self.bus {
            bus.try_post(BusMessage::GraphMutated {
                operation,
                node: node.to_string(),
                outcome,
            });
        }
    }

    /// Wait for the producer behind `endpoint` to park and give up its link,
    /// within `deadline` if there is one. A deadline that passes first withdraws
    /// the park request, so a producer that then reaches its boundary runs on.
    async fn detach(
        &self,
        endpoint: &ProducerEndpoint,
        deadline: Option<u64>,
    ) -> Result<LinkSender, MutationError> {
        let Some((deadline, ticker)) = deadline.zip(self.ticker) else {
            return Detach(endpoint)
                .await
                .map_err(|_| MutationError::GraphEnded);
        };
        match select2(Detach(endpoint), ticker.sleep_until_ns(deadline)).await {
            Either::Left(Ok(link)) => Ok(link),
            Either::Left(Err(_)) => Err(MutationError::GraphEnded),
            Either::Right(()) => {
                endpoint.cancel_park();
                Err(MutationError::TimedOut)
            }
        }
    }

    /// Wait for an arm to answer a mailbox request, within `deadline` if there
    /// is one. A deadline that passes first withdraws the request if the arm has
    /// not taken it yet; one already taken is answered momentarily, so the wait
    /// continues and the answer is the real one.
    async fn await_answer<T>(
        &self,
        mailbox: &PropertyMailbox,
        ticket: u64,
        answer: Receiver<T>,
        deadline: Option<u64>,
    ) -> Result<T, MutationError> {
        let Some((deadline, ticker)) = deadline.zip(self.ticker) else {
            return answer.recv().await.ok_or(MutationError::GraphEnded);
        };
        match select2(answer.recv(), ticker.sleep_until_ns(deadline)).await {
            Either::Left(value) => value.ok_or(MutationError::GraphEnded),
            Either::Right(()) if mailbox.withdraw(ticket) => Err(MutationError::TimedOut),
            Either::Right(()) => answer.recv().await.ok_or(MutationError::GraphEnded),
        }
    }

    /// Queue the set for the arm that owns the element and wait for that
    /// arm's answer. A reply channel that closes without an answer means the arm
    /// ended (or the whole run did) before it reached a packet boundary.
    async fn set_property(
        &self,
        node: &str,
        name: &str,
        value: PropValue,
        deadline: Option<u64>,
    ) -> Result<(), MutationError> {
        let mailbox = self.mailbox(node)?;
        let (reply, answer) = bounded(1);
        let ticket = mailbox.push(PropertyRequest::Set {
            name: name.to_string(),
            value,
            reply,
        });
        self.await_answer(mailbox, ticket, answer, deadline)
            .await?
            .map_err(MutationError::PropertyRejected)
    }

    /// [`set_property`](Self::set_property) for a read: the element's own answer,
    /// `None` for a name it does not carry.
    async fn get_property(
        &self,
        node: &str,
        name: &str,
        deadline: Option<u64>,
    ) -> Result<Option<PropValue>, MutationError> {
        let mailbox = self.mailbox(node)?;
        let (reply, answer) = bounded(1);
        let ticket = mailbox.push(PropertyRequest::Get {
            name: name.to_string(),
            reply,
        });
        self.await_answer(mailbox, ticket, answer, deadline).await
    }

    fn mailbox(&self, node: &str) -> Result<&PropertyMailbox, MutationError> {
        let found = self.find(node)?;
        self.nodes[found]
            .properties
            .as_ref()
            .ok_or_else(|| MutationError::NotMutable(node.to_string()))
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
        self.unique_name_among(category, &[])
    }

    /// [`unique_name`](Self::unique_name) counting past `taken` too: the names
    /// one operation has given elements it has not put in the table yet.
    fn unique_name_among(&self, category: &str, taken: &[String]) -> String {
        for n in 0.. {
            let candidate = alloc::format!("{category}{n}");
            let in_graph = self.nodes.iter().any(|node| node.name == candidate);
            if !in_graph && !taken.contains(&candidate) {
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

    /// The node producing into `edge`: the one listing it as an out edge.
    fn producer_of(&self, edge: usize) -> Option<usize> {
        self.nodes
            .iter()
            .position(|node| node.out_edges.contains(&edge))
    }

    /// A link for a hop the mutator adds, carrying the policy and depth of the
    /// edge it stands in for, and counting its drops into the run's tally.
    fn new_link(&self, policy: LinkPolicy, capacity: usize) -> (LinkSender, LinkReceiver) {
        let (mut tx, rx) = link(capacity);
        tx.set_policy(policy);
        // A leaky link's drops are a whole-run total, so the new hop counts into
        // the same tally. The edge's observer wiring (counters, transit ring,
        // probe) stays with the hop that kept the consumer instead, which is
        // what keeps the transit ring paired with the receiver popping it.
        if policy != LinkPolicy::Block {
            tx.set_drop_counter(self.dropped.clone());
        }
        (tx, rx)
    }

    /// Negotiate `elements` as a chain fed by `input`, each against what the one
    /// before emits, and configure them. Returns what each emits. The last one's
    /// output is not checked against anything here: the caller knows what sits
    /// below.
    fn negotiate_chain(
        &self,
        input: &Caps,
        elements: &mut [Box<dyn DynAsyncElement + 'a>],
    ) -> Result<Vec<Caps>, MutationError> {
        let mut out_caps = Vec::with_capacity(elements.len());
        let mut current = input.clone();
        for element in elements.iter_mut() {
            let intercepted = element
                .intercept_caps(&current)
                .map_err(MutationError::Refused)?;
            // a converter's intercept_caps only checks its input, its constraint names the output
            let constraint = element.caps_constraint_as_transform();
            let out = match resolve_forward_output(&constraint, &current, None, None) {
                ForwardResolve::Fixed(caps) => caps,
                _ => intercepted,
            };
            out_caps.push(out.clone());
            current = out;
        }
        Ok(out_caps)
    }

    /// Configure a negotiated chain on its caps, in order.
    fn configure_chain(
        &self,
        input: &Caps,
        elements: &mut [Box<dyn DynAsyncElement + 'a>],
        out_caps: &[Caps],
    ) -> Result<(), MutationError> {
        let mut current = input.clone();
        for (element, out) in elements.iter_mut().zip(out_caps) {
            element
                .configure_pipeline(&current)
                .map_err(MutationError::Refused)?
                .reject_refixate()
                .map_err(MutationError::Refused)?;
            element
                .configure_output(out)
                .map_err(MutationError::Refused)?;
            current = out.clone();
        }
        Ok(())
    }

    /// Extend a chain whose output `accepted` refuses with the converters the
    /// handle's factory finds for the pair, negotiating them on. Without a
    /// factory, or without a chain from it, the refusal stands.
    fn convert_tail(
        &self,
        elements: &mut Vec<Box<dyn DynAsyncElement + 'a>>,
        out_caps: &mut Vec<Caps>,
        produced: &Caps,
        accepted: &CapsSet,
        converters: Option<&ConverterFactory<'a>>,
    ) -> Result<(), MutationError> {
        let factory = converters.ok_or(MutationError::DownstreamRefused)?;
        let mut tail = factory(produced, accepted).ok_or(MutationError::DownstreamRefused)?;
        let tail_caps = self.negotiate_chain(produced, &mut tail)?;
        let reaches = tail_caps.last().is_some_and(|caps| accepted.accepts(caps));
        if !reaches {
            return Err(MutationError::DownstreamRefused);
        }
        elements.append(&mut tail);
        out_caps.extend(tail_caps);
        Ok(())
    }

    /// Negotiate a chain for the edge on `side` of `name`: against the caps
    /// flowing there, with the last output accepted by the chain below (or
    /// converted to something that is), and configure it. Nothing in the graph
    /// is touched.
    fn prepare_chain(
        &self,
        name: &str,
        side: Side,
        elements: &mut Vec<Box<dyn DynAsyncElement + 'a>>,
        converters: Option<&ConverterFactory<'a>>,
    ) -> Result<ChainPlan, MutationError> {
        let edge = self.edge_at(name, side)?;
        let edge_caps = self.edges[edge]
            .endpoint
            .caps()
            .ok_or(MutationError::NoCaps)?;
        let mut out_caps = self.negotiate_chain(&edge_caps, elements)?;
        let produced = out_caps
            .last()
            .cloned()
            .unwrap_or_else(|| edge_caps.clone());
        if produced != edge_caps && !accepts_downstream(&self.edges[edge].feasible, &produced) {
            let accepted = self.edges[edge]
                .feasible
                .as_ref()
                .ok_or(MutationError::DownstreamRefused)?;
            self.convert_tail(elements, &mut out_caps, &produced, accepted, converters)?;
        }
        self.configure_chain(&edge_caps, elements, &out_caps)?;
        Ok(ChainPlan {
            edge,
            edge_caps,
            out_caps,
        })
    }

    async fn insert(
        &mut self,
        name: &str,
        side: Side,
        mut elements: Vec<Box<dyn DynAsyncElement + 'a>>,
        converters: Option<&ConverterFactory<'a>>,
        deadline: Option<u64>,
    ) -> Result<Vec<String>, MutationError> {
        let ChainPlan {
            edge,
            edge_caps,
            out_caps,
        } = self.prepare_chain(name, side, &mut elements, converters)?;
        let endpoint = self.edges[edge].endpoint.clone();
        let consumer = self.edges[edge].consumer;
        let policy = self.edges[edge].policy;
        let capacity = self.edges[edge].capacity;
        let mode = self.edges[edge].mode;
        let final_caps = out_caps
            .last()
            .cloned()
            .unwrap_or_else(|| edge_caps.clone());

        // Stop the producer and take the link it was pushing through: that link
        // becomes the last spliced element's output, so everything already
        // queued on it stays ahead of anything the chain emits and no drain is
        // needed.
        endpoint.request_park();
        let mut original = self.detach(&endpoint, deadline).await?;
        let last_endpoint = ProducerEndpoint::new();
        last_endpoint.set_caps(&final_caps);
        last_endpoint.continue_sequence_after(endpoint.last_sequence());
        original.set_mutation(Some(last_endpoint.clone()));

        // A caps-changing splice announces itself here, while the producer is
        // still stopped: behind the queued packets, ahead of its own output.
        if final_caps != edge_caps {
            if let Err(e) = original.send_caps(final_caps.clone()).await {
                endpoint.unpark(Some(original));
                return Err(MutationError::Refused(e));
            }
        }

        // One link per element's input, in stream order; each element's output
        // is the next one's input, the last one's the link the producer gave up.
        let count = elements.len();
        let mut feeds = Vec::with_capacity(count);
        let mut endpoints = Vec::with_capacity(count);
        for (i, caps) in out_caps.iter().enumerate() {
            feeds.push(self.new_link(policy, capacity));
            endpoints.push(match i + 1 == count {
                true => last_endpoint.clone(),
                false => {
                    let hop = ProducerEndpoint::new();
                    hop.set_caps(caps);
                    hop
                }
            });
        }
        let first_feed_tx = feeds[0].0.clone();
        let mut names = Vec::with_capacity(count);
        let mut spliced = Vec::with_capacity(count);
        let mut outputs: Vec<LinkSender> = feeds.iter().skip(1).map(|(tx, _)| tx.clone()).collect();
        outputs.push(original);
        let mut inputs: Vec<LinkReceiver> = feeds.into_iter().map(|(_, rx)| rx).collect();
        for (i, mut element) in elements.into_iter().enumerate() {
            element.configure_liveness(self.path_is_live);
            let name = self.unique_name_among(element.log_category(), &names);
            element.set_instance_name(name.clone());
            let mut out_tx = outputs.remove(0);
            if i + 1 < count {
                out_tx.set_mutation(Some(endpoints[i].clone()));
            }
            let node = self.nodes.len() + i;
            let (done_tx, done_rx) = bounded(1);
            // A spliced element takes live property operations the way a
            // negotiated transform does.
            let properties = PropertyMailbox::new();
            let io = TransformArmIo {
                in_rx: inputs.remove(0),
                out_tx,
                arm_rx: dead_directives(),
                coord: GraphCoordHandle::detached(),
                node: NodeId(node as u32),
                out_caps: out_caps[i].clone(),
                // Between two spliced elements the one known-good shape is what
                // the next accepted; the last hop keeps what the edge carried.
                downstream_feasible: match i + 1 == count {
                    true => self.edges[edge].feasible.clone(),
                    false => Some(CapsSet::one(out_caps[i].clone())),
                },
                mode,
                bus: self.bus.clone(),
                probe: None,
                control: None,
                properties: Some(properties.clone()),
            };
            let arm = (self.spawn.transform)(element, io, done_tx);
            if self.arms.send(arm).await.is_err() {
                endpoint.unpark(None);
                return Err(MutationError::GraphEnded);
            }
            names.push(name);
            spliced.push((done_rx, properties));
        }
        endpoint.unpark(Some(first_feed_tx));

        // The edge the producer kept now ends at the first spliced element, and
        // a new edge carries each element's output onward, the last one to the
        // consumer.
        let first_node = self.nodes.len();
        let mut above = edge;
        for (i, ((done_rx, properties), hop)) in spliced.into_iter().zip(endpoints).enumerate() {
            let node = first_node + i;
            let below = self.edges.len();
            self.edges.push(LiveEdge {
                endpoint: hop,
                consumer: match i + 1 == count {
                    true => consumer,
                    false => node + 1,
                },
                feasible: match i + 1 == count {
                    true => self.edges[edge].feasible.clone(),
                    false => Some(CapsSet::one(out_caps[i].clone())),
                },
                policy,
                capacity,
                mode,
            });
            self.nodes.push(LiveNode {
                name: names[i].clone(),
                out_edges: alloc::vec![below],
                in_edges: alloc::vec![above],
                total_out_edges: 1,
                total_in_edges: 1,
                done: Some(Handback::Element(done_rx)),
                removable: true,
                properties: Some(properties),
                growth: Growth::Fixed,
            });
            above = below;
        }
        self.edges[edge].consumer = first_node;
        // The one shape the edge above the spliced chain is known to carry: the
        // caps its first element accepted and was configured for. Nothing solved
        // what else it would take, or what the chain below would do with the
        // output that produced, so a further splice here has to keep the caps as
        // they are and is refused otherwise.
        self.edges[edge].feasible = Some(CapsSet::one(edge_caps));
        retarget(&mut self.nodes[consumer].in_edges, edge, above);
        Ok(names)
    }

    async fn remove(
        &mut self,
        node: &str,
        deadline: Option<u64>,
    ) -> Result<Box<dyn DynAsyncElement + 'a>, MutationError> {
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
        let feed = match self.detach(&up_endpoint, deadline).await {
            Ok(feed) => feed,
            // Nothing was disturbed: restore the handback so a retry reports
            // the refusal, not NotMutable.
            Err(e) => {
                self.nodes[removed].done = Some(Handback::Element(done));
                return Err(e);
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
        up_endpoint.continue_sequence_after(own_endpoint.last_sequence());
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

    /// Negotiate and configure a sink for `edge_caps`, the way a sink arm takes a
    /// mid-stream shape: the element's own constraint says what it reads the
    /// edge as, and it is configured on that. Returns the set the edge accepts
    /// once this sink is on it.
    fn prepare_sink(
        &self,
        edge_caps: &Caps,
        element: &mut Box<dyn DynAsyncElement + 'a>,
    ) -> Result<CapsSet, MutationError> {
        let sink_caps = re_solve_downstream_dyn_sink(edge_caps, &**element)
            .map_err(|_| MutationError::Refused(G2gError::CapsMismatch))?;
        // What the edge will accept once this sink is on it: the set the
        // replacement declares, which is what the runner's startup sweep would
        // have read off it. A sink that declares no set leaves the edge carrying
        // the one shape it was configured for, and a later caps change here is
        // refused the way it is above a spliced element.
        let feasible = match element.caps_constraint_as_sink() {
            crate::format_element::CapsConstraint::Accepts(set) => set.clone(),
            _ => CapsSet::one(edge_caps.clone()),
        };
        element
            .configure_pipeline(&sink_caps)
            .map_err(MutationError::Refused)?
            .reject_refixate()
            .map_err(MutationError::Refused)?;
        element.configure_liveness(self.path_is_live);
        // The elected clock reaches the new sink the way the startup sweep gave
        // it to every negotiated sink, so it paces its presentation; only the
        // election itself is settled.
        if let Some(sync) = &self.sink_clock_sync {
            element.set_clock_sync(sync.clone());
        }
        Ok(feasible)
    }

    /// Start a sink's arm on `feed_rx`, lending the element out, and return the
    /// node it occupies.
    async fn start_sink(
        &mut self,
        element: Box<dyn DynAsyncElement + 'a>,
        name: String,
        feed_rx: LinkReceiver,
        in_edge: usize,
        mode: BranchMode,
    ) -> Result<usize, MutationError> {
        advertise_orientation(&feed_rx, element.absorbed_orientations());
        let node = self.nodes.len();
        let (done_tx, done_rx) = bounded(1);
        let properties = PropertyMailbox::new();
        let io = SinkArmIo {
            in_rx: feed_rx,
            arm_rx: dead_directives(),
            coord: GraphCoordHandle::detached(),
            node: NodeId(node as u32),
            mode,
            bus: self.bus.clone(),
            // A sink added mid-run joins a stream already in flight: there is
            // no preroll for it to take and no state transition to gate on.
            state: None,
            progress: self.progress.clone(),
            probe: None,
            control: None,
            properties: Some(properties.clone()),
        };
        let arm = (self.spawn.sink)(element, io, done_tx);
        if self.arms.send(arm).await.is_err() {
            return Err(MutationError::GraphEnded);
        }
        self.nodes.push(LiveNode {
            name,
            out_edges: Vec::new(),
            in_edges: alloc::vec![in_edge],
            total_out_edges: 0,
            total_in_edges: 1,
            done: Some(Handback::Element(done_rx)),
            removable: false,
            properties: Some(properties),
            growth: Growth::Fixed,
        });
        Ok(node)
    }

    /// Swap the element at the bottom of an edge. See
    /// [`GraphMutator::replace_sink`].
    async fn replace_sink(
        &mut self,
        node: &str,
        mut element: Box<dyn DynAsyncElement + 'a>,
        deadline: Option<u64>,
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

        // Negotiate before anything is disturbed. A sink has no output pad, so
        // the startup path's `configure_output` has no counterpart here.
        let feasible = self.prepare_sink(&edge_caps, &mut element)?;
        let name = self.unique_name(element.log_category());
        element.set_instance_name(name.clone());

        let done = self
            .take_element_handback(replaced)
            .expect("the handback was checked above");
        endpoint.request_park();
        let feed = match self.detach(&endpoint, deadline).await {
            Ok(feed) => feed,
            // Nothing was disturbed: put the handback back so a retry reports
            // the refusal, not NotMutable.
            Err(e) => {
                self.nodes[replaced].done = Some(Handback::Element(done));
                return Err(e);
            }
        };
        // Past here the old sink is being ended, so every refusal is behind us,
        // and a failure leaves the producer resuming onto the dead link it
        // parked with rather than an arrangement to restore, as in `remove`.
        //
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
        let (feed_tx, feed_rx) = self.new_link(self.edges[edge].policy, self.edges[edge].capacity);
        let mode = self.edges[edge].mode;
        let replacement = match self
            .start_sink(element, name.clone(), feed_rx, edge, mode)
            .await
        {
            Ok(replacement) => replacement,
            Err(e) => {
                endpoint.unpark(None);
                return Err(e);
            }
        };
        endpoint.unpark(Some(feed_tx));

        self.edges[edge].consumer = replacement;
        self.edges[edge].feasible = Some(feasible);
        // The node the old sink sat on stays in the table, named and addressing
        // nothing, so its name is never handed out again.
        self.nodes[replaced].in_edges.clear();
        Ok((name, old))
    }

    /// Swap the element at the top of an edge. See
    /// [`GraphMutator::replace_source`].
    async fn replace_source(
        &mut self,
        node: &str,
        mut source: Box<dyn DynSourceLoop + 'a>,
        deadline: Option<u64>,
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

        let done = self
            .take_source_handback(replaced)
            .expect("the handback was checked above");
        endpoint.request_park();
        let out_tx = match self.detach(&endpoint, deadline).await {
            Ok(out_tx) => out_tx,
            // Nothing was disturbed: restore the handback so a retry reports
            // the refusal, not NotMutable.
            Err(e) => {
                self.nodes[replaced].done = Some(Handback::Source(done));
                return Err(e);
            }
        };
        // Past here the old source is being retired, so every refusal is behind
        // us; a failure from now on leaves the edge without a producer, which
        // ends the stream below it rather than wedging it.
        //
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
            // A source drives itself, so it has no packet boundary to act at.
            properties: None,
            growth: Growth::Fixed,
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

    /// The caps a tee broadcasts: what flows into it, or, when that edge is not
    /// a mutation position, what any branch below it carries.
    fn tee_caps(&self, tee: usize) -> Result<Caps, MutationError> {
        let node = &self.nodes[tee];
        node.in_edges
            .iter()
            .chain(node.out_edges.iter())
            .find_map(|&edge| self.edges[edge].endpoint.caps())
            .ok_or(MutationError::NoCaps)
    }

    /// See [`GraphMutator::add_branch`].
    async fn add_branch(
        &mut self,
        tee: &str,
        mut branch: Vec<Box<dyn DynAsyncElement + 'a>>,
        converters: Option<&ConverterFactory<'a>>,
    ) -> Result<Vec<String>, MutationError> {
        let tee_node = self.find(tee)?;
        let not_mutable = || MutationError::NotMutable(tee.to_string());
        let Growth::Branches { adds, mode } = &self.nodes[tee_node].growth else {
            return Err(not_mutable());
        };
        let (adds, mode) = (adds.clone(), *mode);
        let mut sink = branch.pop().ok_or_else(not_mutable)?;
        let sticky = self.tee_caps(tee_node)?;
        // The policy and depth of the branches the tee was built with, where it
        // has any, so a late branch queues like its siblings.
        let (policy, capacity) = self.nodes[tee_node]
            .out_edges
            .first()
            .map(|&edge| (self.edges[edge].policy, self.edges[edge].capacity))
            .unwrap_or((LinkPolicy::Block, self.link_capacity));

        // The transforms first, then the sink against what the last one emits;
        // a sink that will not take that shape gets converters ahead of it, as
        // a splice does.
        let mut out_caps = self.negotiate_chain(&sticky, &mut branch)?;
        let produced = out_caps.last().cloned().unwrap_or_else(|| sticky.clone());
        let feasible = match self.prepare_sink(&produced, &mut sink) {
            Ok(feasible) => feasible,
            Err(refusal) => {
                let accepted = match sink.caps_constraint_as_sink() {
                    crate::format_element::CapsConstraint::Accepts(set) => set.clone(),
                    _ => return Err(refusal),
                };
                self.convert_tail(&mut branch, &mut out_caps, &produced, &accepted, converters)
                    .map_err(|_| refusal)?;
                let converted = out_caps.last().cloned().unwrap_or_else(|| sticky.clone());
                self.prepare_sink(&converted, &mut sink)?
            }
        };
        self.configure_chain(&sticky, &mut branch, &out_caps)?;

        // Links in stream order: the tee's new port feeds the first element,
        // each element the next, the last the sink. Every hop is a mutation
        // position from the start, so the branch can be spliced on and lifted
        // off again.
        let count = branch.len() + 1;
        let mut links = Vec::with_capacity(count);
        for _ in 0..count {
            links.push(self.new_link(policy, capacity));
        }
        let mut endpoints = Vec::with_capacity(count);
        let hop_caps = core::iter::once(&sticky).chain(out_caps.iter());
        for ((tx, _), caps) in links.iter_mut().zip(hop_caps) {
            let hop = ProducerEndpoint::new();
            hop.set_caps(caps);
            tx.set_mutation(Some(hop.clone()));
            endpoints.push(hop);
        }
        let mut port = None;
        let mut inputs = Vec::with_capacity(count);
        let mut outputs = Vec::with_capacity(count);
        for (i, (tx, rx)) in links.into_iter().enumerate() {
            match i {
                0 => port = Some(tx),
                _ => outputs.push(tx),
            }
            inputs.push(rx);
        }
        let port = port.expect("one link per element, at least the sink's");

        let first_node = self.nodes.len();
        let first_edge = self.edges.len();
        let mut names = Vec::with_capacity(count);
        for (i, mut element) in branch.into_iter().enumerate() {
            element.configure_liveness(self.path_is_live);
            let name = self.unique_name(element.log_category());
            element.set_instance_name(name.clone());
            let node = first_node + i;
            let (done_tx, done_rx) = bounded(1);
            let properties = PropertyMailbox::new();
            let io = TransformArmIo {
                in_rx: inputs.remove(0),
                out_tx: outputs.remove(0),
                arm_rx: dead_directives(),
                coord: GraphCoordHandle::detached(),
                node: NodeId(node as u32),
                out_caps: out_caps[i].clone(),
                downstream_feasible: Some(CapsSet::one(out_caps[i].clone())),
                mode,
                bus: self.bus.clone(),
                probe: None,
                control: None,
                properties: Some(properties.clone()),
            };
            let arm = (self.spawn.transform)(element, io, done_tx);
            if self.arms.send(arm).await.is_err() {
                return Err(MutationError::GraphEnded);
            }
            self.edges.push(LiveEdge {
                endpoint: endpoints[i].clone(),
                consumer: node,
                feasible: Some(CapsSet::one(match i {
                    0 => sticky.clone(),
                    _ => out_caps[i - 1].clone(),
                })),
                policy,
                capacity,
                mode,
            });
            self.nodes.push(LiveNode {
                name: name.clone(),
                out_edges: alloc::vec![first_edge + i + 1],
                in_edges: alloc::vec![first_edge + i],
                total_out_edges: 1,
                total_in_edges: 1,
                done: Some(Handback::Element(done_rx)),
                removable: true,
                properties: Some(properties),
                growth: Growth::Fixed,
            });
            names.push(name);
        }
        let sink_name = self.unique_name(sink.log_category());
        sink.set_instance_name(sink_name.clone());
        let sink_edge = self.edges.len();
        self.edges.push(LiveEdge {
            endpoint: endpoints[count - 1].clone(),
            consumer: first_node + count - 1,
            feasible: Some(feasible),
            policy,
            capacity,
            mode,
        });
        let feed_rx = inputs.remove(0);
        self.start_sink(sink, sink_name.clone(), feed_rx, sink_edge, mode)
            .await?;
        names.push(sink_name);

        // The tee takes the port between two packets. Nothing was parked, so
        // the only way this fails is the tee's arm having ended.
        if adds.send(port).await.is_err() {
            return Err(MutationError::GraphEnded);
        }
        self.nodes[tee_node].out_edges.push(first_edge);
        self.nodes[tee_node].total_out_edges += 1;
        Ok(names)
    }

    /// See [`GraphMutator::remove_branch`].
    async fn remove_branch(
        &mut self,
        sink: &str,
        deadline: Option<u64>,
    ) -> Result<Vec<Box<dyn DynAsyncElement + 'a>>, MutationError> {
        let sink_node = self.find(sink)?;
        let not_mutable = || MutationError::NotMutable(sink.to_string());
        if self.nodes[sink_node].total_out_edges != 0
            || !matches!(self.nodes[sink_node].done, Some(Handback::Element(_)))
        {
            return Err(not_mutable());
        }
        // Walk up from the sink to the tee, collecting the chain between them:
        // every node on the way must be a transform the mutator can hand back.
        let mut chain = alloc::vec![sink_node];
        let mut edge = self.edge_at(sink, Side::Above)?;
        let (tee_node, top_edge) = loop {
            let producer = self.producer_of(edge).ok_or_else(not_mutable)?;
            if matches!(self.nodes[producer].growth, Growth::Branches { .. }) {
                break (producer, edge);
            }
            if !self.nodes[producer].removable
                || !matches!(self.nodes[producer].done, Some(Handback::Element(_)))
            {
                return Err(not_mutable());
            }
            chain.push(producer);
            edge = self.edge_at(&self.nodes[producer].name.clone(), Side::Above)?;
        };
        chain.reverse();

        let endpoint = self.edges[top_edge].endpoint.clone();
        endpoint.request_park();
        let feed = self.detach(&endpoint, deadline).await?;
        // The branch ends on an end of stream, so each element flushes and the
        // sink finalizes, then the channel behind the marker closes. The tee's
        // port is retired, so its next push there fails and the tee drops it.
        if feed.send_eos().await.is_err() {
            endpoint.unpark(None);
            return Err(MutationError::GraphEnded);
        }
        drop(feed);
        endpoint.retire();

        let mut elements = Vec::with_capacity(chain.len());
        for &node in &chain {
            let done = self
                .take_element_handback(node)
                .expect("every node on the chain was checked above");
            let Some(element) = done.recv().await else {
                return Err(MutationError::GraphEnded);
            };
            elements.push(element);
        }
        for &node in &chain {
            self.nodes[node].in_edges.clear();
            self.nodes[node].out_edges.clear();
            self.nodes[node].removable = false;
        }
        self.nodes[tee_node].out_edges.retain(|&e| e != top_edge);
        self.nodes[tee_node].total_out_edges -= 1;
        Ok(elements)
    }

    /// See [`GraphMutator::add_input`].
    async fn add_input(
        &mut self,
        muxer: &str,
        mut source: Box<dyn DynSourceLoop + 'a>,
    ) -> Result<String, MutationError> {
        let muxer_node = self.find(muxer)?;
        let Growth::Inputs {
            adds,
            pads_in_use,
            declared,
        } = &self.nodes[muxer_node].growth
        else {
            return Err(MutationError::NotMutable(muxer.to_string()));
        };
        let pad = (0..*declared)
            .find(|pad| !pads_in_use.contains(pad))
            .ok_or(MutationError::NoFreePad)?;
        let adds = adds.clone();
        let produced = source
            .produced_caps()
            .await
            .map_err(MutationError::Refused)?;

        // The muxer's arm owns the element, so it negotiates the pad: the
        // source's set against the pad's constraint, then the element's veto.
        // The arm reads the pad straight off the link the source will push into.
        let (mut out_tx, out_rx) = self.new_link(LinkPolicy::Block, self.link_capacity);
        let (reply, answer) = bounded(1);
        let add = PadAdd {
            pad,
            produced,
            pad_rx: out_rx,
            reply,
        };
        if adds.send(add).await.is_err() {
            return Err(MutationError::GraphEnded);
        }
        let fixated = match answer.recv().await {
            Some(Ok(caps)) => caps,
            Some(Err(e)) => return Err(MutationError::Refused(e)),
            None => return Err(MutationError::GraphEnded),
        };
        // The pad is admitted. A source that then refuses its side leaves the
        // pad with nothing feeding it: dropping `out_tx` closes the link, which
        // the arm takes as that input ending.
        source
            .configure_pipeline(&fixated)
            .map_err(MutationError::Refused)?
            .reject_refixate()
            .map_err(MutationError::Refused)?;
        let name = self.unique_name(source.log_category());
        source.set_instance_name(name.clone());

        let endpoint = ProducerEndpoint::new();
        endpoint.set_caps(&fixated);
        out_tx.set_mutation(Some(endpoint.clone()));
        let (done_tx, done_rx) = bounded(1);
        let io = SourceArmIo {
            out_tx,
            bus: self.bus.clone(),
            progress: self.progress.clone(),
            segment: Segment::new(),
        };
        let arm = (self.spawn.source)(source, io, done_tx);
        if self.arms.send(arm).await.is_err() {
            return Err(MutationError::GraphEnded);
        }

        let edge = self.edges.len();
        self.edges.push(LiveEdge {
            endpoint,
            consumer: muxer_node,
            feasible: Some(CapsSet::one(fixated)),
            policy: LinkPolicy::Block,
            capacity: self.link_capacity,
            mode: BranchMode::Reconfigure,
        });
        self.nodes.push(LiveNode {
            name: name.clone(),
            out_edges: alloc::vec![edge],
            in_edges: Vec::new(),
            total_out_edges: 1,
            total_in_edges: 0,
            done: Some(Handback::Source(done_rx)),
            removable: false,
            properties: None,
            growth: Growth::Fixed,
        });
        let muxer = &mut self.nodes[muxer_node];
        muxer.in_edges.push(edge);
        muxer.total_in_edges += 1;
        if let Growth::Inputs { pads_in_use, .. } = &mut muxer.growth {
            pads_in_use.push(pad);
        }
        Ok(name)
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

/// Awaits the link a stopped producer (or an ended arm) left on its endpoint.
struct Detach<'e>(&'e ProducerEndpoint);

impl Future for Detach<'_> {
    type Output = Result<LinkSender, G2gError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.poll_detached(cx)
    }
}
