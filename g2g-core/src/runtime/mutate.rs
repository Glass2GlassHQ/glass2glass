//! Structural mutation of a running graph (M1115): splicing a transform onto a
//! live edge and lifting one back off, without stopping the pipeline.
//!
//! The data plane pays for this once per push: a relaxed load of the edge's
//! [`ProducerEndpoint`] gate. Everything else happens inside a mutation op,
//! which runs in the run future itself (`MutationService`), so it can touch the
//! arms, the links and the elements the way the runner does.
//!
//! Scope: a transform position on a 1:1 edge, between a source or transform and
//! a transform or sink. Tee / demux / muxer positions and the source / sink ends
//! are refused ([`MutationError::NotMutable`]).

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use crate::bus::BusHandle;
use crate::caps::CapsSet;
use crate::element::{BoxFuture, ConfigureOutcome, DynAsyncElement};
use crate::error::G2gError;
use crate::graph::NodeId;
use crate::link::LinkPolicy;
use crate::runtime::channel::{bounded, link, LinkSender, ProducerEndpoint, Receiver, Sender};
use crate::runtime::coordinator::ArmDirective;
use crate::runtime::graph_runner::{BranchMode, GraphCoordHandle, TransformArmIo};

/// Why a structural mutation was refused. Every one of these leaves the graph
/// running exactly as it was.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MutationError {
    /// No element of the running graph carries this instance name.
    UnknownNode(String),
    /// The named position is not a transform position on a 1:1 edge: a tee /
    /// demux / muxer node, a sink, or the last node of a chain.
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
/// returned by [`insert_after`](Self::insert_after) so it can be removed later.
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
        let (reply, answer) = bounded(1);
        self.tx
            .send(MutationRequest::Insert {
                after: after.to_string(),
                element,
                reply,
            })
            .await
            .map_err(|_| MutationError::GraphEnded)?;
        answer.recv().await.ok_or(MutationError::GraphEnded)?
    }

    /// Lift the transform named `node` out of the running graph and hand it
    /// back. Whatever is queued at its input drains through it first, so no
    /// frame is lost or reordered; its producer then feeds its consumer
    /// directly.
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
}

/// One operation, carrying its own reply channel. The work runs in the run
/// future, which is where the arms and links live.
#[allow(missing_debug_implementations)]
pub(crate) enum MutationRequest<'a> {
    Insert {
        after: String,
        element: Box<dyn DynAsyncElement + 'a>,
        reply: Sender<Result<String, MutationError>>,
    },
    Remove {
        node: String,
        reply: Sender<Result<Box<dyn DynAsyncElement + 'a>, MutationError>>,
    },
}

/// Build the mutator handle and the request channel the service reads. The
/// handle is returned to the caller before the run starts, the receiver goes
/// into the run future.
pub(crate) fn mutation_channel<'a>(
    capacity: usize,
) -> (GraphMutator<'a>, Receiver<MutationRequest<'a>>) {
    let (tx, rx) = bounded(capacity);
    (GraphMutator { tx }, rx)
}

/// One node of the runner's live topology, as the mutator sees it: the
/// producing end of the edge below it, who is on either side of it right now,
/// and what the chain below that edge accepts.
#[allow(missing_debug_implementations)]
pub(crate) struct LiveNode<'a> {
    pub(crate) name: String,
    /// The retargetable producing end of this node's output edge. `None` where
    /// there is no mutable 1:1 edge below it (a sink, a fan node).
    pub(crate) endpoint: Option<Arc<ProducerEndpoint>>,
    pub(crate) next: Option<usize>,
    pub(crate) prev: Option<usize>,
    /// What the chain below this node's edge can carry: the runner's startup
    /// snapshot, or, once something was spliced there, the one shape that
    /// element accepted. `None` where the runner could compute none, which
    /// refuses any caps change here rather than waving it through.
    pub(crate) feasible: Option<CapsSet>,
    pub(crate) policy: LinkPolicy,
    pub(crate) capacity: usize,
    /// Delivers this node's element once its arm ends, so a remove can hand it
    /// back. `None` on a node whose element the runner did not lend out.
    pub(crate) done: Option<Receiver<Box<dyn DynAsyncElement + 'a>>>,
    /// A transform: the only kind of node that can be lifted out.
    pub(crate) removable: bool,
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

/// The mutator's other half, running inside the run future: it owns the live
/// topology and performs each operation against the arms and links themselves.
#[allow(missing_debug_implementations)]
pub(crate) struct MutationService<'a, 's> {
    rx: Receiver<MutationRequest<'a>>,
    nodes: Vec<LiveNode<'a>>,
    /// Sends a spliced element's arm into the run's growable join set.
    arms: Sender<BoxFuture<'a, Result<u64, G2gError>>>,
    spawn: SpliceSpawn<'a, 's>,
    bus: Option<BusHandle>,
    /// The run's leaky-drop tally, installed on a spliced hop so its drops
    /// reach `RunStats` like every other link's.
    dropped: Arc<spin::Mutex<u64>>,
    /// The liveness the runner folded at startup, handed to a spliced element
    /// the way it reached the negotiated ones (M1123).
    path_is_live: bool,
}

impl<'a, 's> MutationService<'a, 's> {
    pub(crate) fn new(
        rx: Receiver<MutationRequest<'a>>,
        nodes: Vec<LiveNode<'a>>,
        arms: Sender<BoxFuture<'a, Result<u64, G2gError>>>,
        spawn: SpliceSpawn<'a, 's>,
        bus: Option<BusHandle>,
        dropped: Arc<spin::Mutex<u64>>,
        path_is_live: bool,
    ) -> Self {
        Self {
            rx,
            nodes,
            arms,
            spawn,
            bus,
            dropped,
            path_is_live,
        }
    }

    /// Serve requests for as long as the run lasts. Never returns: the run ends
    /// when its arms do, and this is selected against them.
    pub(crate) async fn run(mut self) -> core::convert::Infallible {
        loop {
            match self.rx.recv().await {
                Some(MutationRequest::Insert {
                    after,
                    element,
                    reply,
                }) => {
                    let result = self.insert_after(&after, element).await;
                    let _ = reply.try_send(result);
                }
                Some(MutationRequest::Remove { node, reply }) => {
                    let result = self.remove(&node).await;
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

    async fn insert_after(
        &mut self,
        after: &str,
        mut element: Box<dyn DynAsyncElement + 'a>,
    ) -> Result<String, MutationError> {
        let producer = self.find(after)?;
        let not_mutable = || MutationError::NotMutable(after.to_string());
        let endpoint = self.nodes[producer]
            .endpoint
            .clone()
            .ok_or_else(not_mutable)?;
        let consumer = self.nodes[producer].next.ok_or_else(not_mutable)?;
        let edge_caps = endpoint.caps().ok_or(MutationError::NoCaps)?;

        // Negotiate before anything is disturbed: the element against the caps
        // on the wire, then its output against what the chain below accepts.
        let out_caps = element
            .intercept_caps(&edge_caps)
            .map_err(MutationError::Refused)?;
        if out_caps != edge_caps && !accepts_downstream(&self.nodes[producer].feasible, &out_caps) {
            return Err(MutationError::DownstreamRefused);
        }
        match element
            .configure_pipeline(&edge_caps)
            .map_err(MutationError::Refused)?
        {
            ConfigureOutcome::Accepted => {}
            // A counter-proposal has nowhere to go: the upstream element is
            // already running under the caps it fixated at startup.
            ConfigureOutcome::ReFixate(_) => {
                return Err(MutationError::Refused(G2gError::FixationFailed))
            }
        }
        element
            .configure_output(&out_caps)
            .map_err(MutationError::Refused)?;
        element.configure_liveness(self.path_is_live);
        let name = self.unique_name(element.log_category());
        element.set_instance_name(name.clone());

        let policy = self.nodes[producer].policy;
        let capacity = self.nodes[producer].capacity;
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

        let (done_tx, done_rx) = bounded(1);
        let io = TransformArmIo {
            in_rx: feed_rx,
            out_tx: original,
            arm_rx: dead_directives(),
            coord: GraphCoordHandle::detached(),
            node: NodeId(producer as u32),
            out_caps: out_caps.clone(),
            downstream_feasible: self.nodes[producer].feasible.clone(),
            mode: BranchMode::Reconfigure,
            bus: self.bus.clone(),
            probe: None,
            control: None,
        };
        let arm = (self.spawn)(element, io, done_tx);
        if self.arms.send(arm).await.is_err() {
            endpoint.unpark(None);
            return Err(MutationError::GraphEnded);
        }
        endpoint.unpark(Some(feed_tx));

        let spliced = self.nodes.len();
        self.nodes.push(LiveNode {
            name: name.clone(),
            endpoint: Some(spliced_endpoint),
            next: Some(consumer),
            prev: Some(producer),
            feasible: self.nodes[producer].feasible.clone(),
            policy,
            capacity,
            done: Some(done_rx),
            removable: true,
        });
        self.nodes[producer].next = Some(spliced);
        // The one shape the edge above the spliced element is known to carry:
        // the caps that element accepted and was configured for. Nothing solved
        // what else it would take, or what the chain below would do with the
        // output that produced, so a further splice here has to keep the caps as
        // they are and is refused otherwise.
        self.nodes[producer].feasible = Some(CapsSet::one(edge_caps));
        self.nodes[consumer].prev = Some(spliced);
        Ok(name)
    }

    async fn remove(&mut self, node: &str) -> Result<Box<dyn DynAsyncElement + 'a>, MutationError> {
        let removed = self.find(node)?;
        let not_mutable = || MutationError::NotMutable(node.to_string());
        if !self.nodes[removed].removable {
            return Err(not_mutable());
        }
        let producer = self.nodes[removed].prev.ok_or_else(not_mutable)?;
        let consumer = self.nodes[removed].next.ok_or_else(not_mutable)?;
        let up_endpoint = self.nodes[producer]
            .endpoint
            .clone()
            .ok_or_else(not_mutable)?;
        let own_endpoint = self.nodes[removed]
            .endpoint
            .clone()
            .ok_or_else(not_mutable)?;
        let done = self.nodes[removed].done.take().ok_or_else(not_mutable)?;
        let upstream_caps = up_endpoint.caps().ok_or(MutationError::NoCaps)?;
        let own_caps = own_endpoint.caps();

        // The consumer is about to start receiving the producer's caps instead
        // of this element's. It has to accept them, or the bypass would break it.
        let caps_change = own_caps.as_ref() != Some(&upstream_caps);
        if caps_change && !accepts_downstream(&self.nodes[removed].feasible, &upstream_caps) {
            // Put the element channel back: the graph is unchanged.
            self.nodes[removed].done = Some(done);
            return Err(MutationError::DownstreamRefused);
        }

        // Stopping the producer and dropping the link it gave up closes the
        // removed element's input, so its arm drains what is queued, forwards
        // the results, and ends. Its output link is claimed first: that is the
        // link the producer takes over, so this one arm ending must not close
        // the channel the way an arm ending on its own does.
        own_endpoint.claim_on_end();
        up_endpoint.request_park();
        let feed = Detach(&up_endpoint)
            .await
            .map_err(|_| MutationError::GraphEnded)?;
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

        self.nodes[producer].next = Some(consumer);
        self.nodes[producer].feasible = self.nodes[removed].feasible.take();
        self.nodes[consumer].prev = Some(producer);
        self.nodes[removed].next = None;
        self.nodes[removed].prev = None;
        self.nodes[removed].endpoint = None;
        self.nodes[removed].removable = false;
        Ok(element)
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
