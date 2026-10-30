//! One entry point for running a graph, whatever the run is asked to carry:
//! a bus, a state controller, progress, a copy budget, an observer, a flight
//! recorder, a visit recorder, a tick timer, and a [`GraphMutator`] over the running graph, in
//! any combination. The `run_graph_*` functions are shorthands for the common
//! single-feature cases.

use alloc::sync::Arc;
use core::future::Future;

use crate::bus::BusHandle;
use crate::clock::{DynAsyncClock, PipelineClock};
use crate::copyplan::CopyPolicy;
use crate::error::G2gError;
use crate::graph::Graph;
use crate::runtime::channel::Receiver;
use crate::runtime::graph_runner::{run_graph_inner, GraphNodeRef, MUTATION_QUEUE_DEPTH};
#[cfg(feature = "multi-thread")]
use crate::runtime::graph_runner::{run_graph_threaded_inner, GraphSpawner};
use crate::runtime::mutate::{mutation_channel, GraphMutator, MutationRequest};
use crate::runtime::progress::PipelineProgress;
use crate::runtime::runner::{LinkCapacity, RunStats};
use crate::runtime::state::StateController;
use crate::runtime::{FlightRecorder, Observer, SeekController, VisitRecorder};

/// A graph run being set up. `'a` is the elements' lifetime (and the
/// [`GraphMutator`]'s), `'c` the lifetime of the clock and the handles the run
/// reports into, which may be shorter: an application keeps the mutator past the
/// task whose locals it observes the run with.
///
/// ```ignore
/// let (mutator, run) = RunBuilder::new(graph, &clock, LatencyProfile::Live)
///     .with_bus(&bus)
///     .with_state(&state)
///     .run_mutable();
/// ```
#[allow(missing_debug_implementations)]
pub struct RunBuilder<'a, 'c, Clk: PipelineClock> {
    graph: Graph<GraphNodeRef<'a>>,
    clock: &'c Clk,
    link_capacity: LinkCapacity,
    bus: Option<&'c BusHandle>,
    state: Option<StateController>,
    progress: Option<&'c PipelineProgress>,
    copy_policy: Option<CopyPolicy>,
    observer: Option<&'c Observer>,
    recorder: Option<&'c FlightRecorder>,
    visit_recorder: Option<Arc<dyn VisitRecorder>>,
    ticker: Option<Arc<dyn DynAsyncClock + Send + Sync>>,
    seek: Option<SeekController>,
}

impl<'a: 'c, 'c, Clk: PipelineClock> RunBuilder<'a, 'c, Clk> {
    /// A run of `graph` on `clock`, with `link_capacity` packets queued per edge
    /// (a [`LatencyProfile`](crate::runtime::LatencyProfile) or a depth).
    pub fn new(
        graph: Graph<GraphNodeRef<'a>>,
        clock: &'c Clk,
        link_capacity: impl Into<LinkCapacity>,
    ) -> Self {
        Self {
            graph,
            clock,
            link_capacity: link_capacity.into(),
            bus: None,
            state: None,
            progress: None,
            copy_policy: None,
            observer: None,
            recorder: None,
            visit_recorder: None,
            ticker: None,
            seek: None,
        }
    }

    /// Post pipeline [`BusMessage`](crate::BusMessage)s to `bus`.
    pub fn with_bus(mut self, bus: &'c BusHandle) -> Self {
        self.bus = Some(bus);
        self
    }

    /// Drive the run through `state`'s `NULL -> READY -> PAUSED -> PLAYING`
    /// machine.
    pub fn with_state(mut self, state: &StateController) -> Self {
        self.state = Some(state.clone());
        self
    }

    /// Publish playback position and duration into `progress`.
    pub fn with_progress(mut self, progress: &'c PipelineProgress) -> Self {
        self.progress = Some(progress);
        self
    }

    /// Refuse to start if the negotiated copy plan breaks `policy`.
    pub fn with_copy_policy(mut self, policy: CopyPolicy) -> Self {
        self.copy_policy = Some(policy);
        self
    }

    /// Tap live telemetry into `observer`.
    pub fn with_observer(mut self, observer: &'c Observer) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Keep the last packets of every edge in `recorder` for a post-failure dump.
    pub fn with_recorder(mut self, recorder: &'c FlightRecorder) -> Self {
        self.recorder = Some(recorder);
        self
    }

    pub fn with_visit_recorder(mut self, visit_recorder: Arc<dyn VisitRecorder>) -> Self {
        self.visit_recorder = Some(visit_recorder);
        self
    }

    /// The timer fan-in arms sleep on for their deadline tick, overriding
    /// the one `clock` hands out. Shared, since the thread-per-arm run moves each
    /// arm onto its own thread.
    pub fn with_ticker(mut self, ticker: Arc<dyn DynAsyncClock + Send + Sync>) -> Self {
        self.ticker = Some(ticker);
        self
    }

    // the seekable source parks at the end of the stream and takes seeks there, the run ends once nothing more is sought
    pub fn with_seek_controller(mut self, controller: &SeekController) -> Self {
        self.seek = Some(controller.clone());
        self
    }

    /// Start the run on the caller's executor.
    pub fn run(self) -> impl Future<Output = Result<RunStats, G2gError>> + use<'a, 'c, Clk> {
        self.start(None)
    }

    /// Start the run on the caller's executor with a [`GraphMutator`] over it.
    /// Drive the future and use the handle from another task.
    pub fn run_mutable(
        self,
    ) -> (
        GraphMutator<'a>,
        impl Future<Output = Result<RunStats, G2gError>> + use<'a, 'c, Clk>,
    ) {
        let (mutator, requests) = mutation_channel(MUTATION_QUEUE_DEPTH);
        (mutator, self.start(Some(requests)))
    }

    async fn start(
        self,
        mutation: Option<Receiver<MutationRequest<'a>>>,
    ) -> Result<RunStats, G2gError> {
        run_graph_inner(
            self.graph,
            self.clock,
            self.link_capacity,
            self.bus,
            self.state,
            self.progress,
            self.copy_policy,
            self.observer,
            self.recorder,
            self.visit_recorder,
            self.ticker.as_deref().map(|t| t as &dyn DynAsyncClock),
            mutation,
            self.seek.as_ref(),
        )
        .await
    }
}

#[cfg(feature = "multi-thread")]
impl<'c, Clk: PipelineClock> RunBuilder<'static, 'c, Clk> {
    /// Start the run with one OS thread per arm from `spawner`. The
    /// graph owns its elements, since each moves onto its own thread.
    pub fn run_threaded<S: GraphSpawner>(
        self,
        spawner: &'c S,
    ) -> impl Future<Output = Result<RunStats, G2gError>> + use<'c, Clk, S> {
        self.start_threaded(spawner, None)
    }

    /// [`run_threaded`](Self::run_threaded) with a [`GraphMutator`] over the run.
    pub fn run_threaded_mutable<S: GraphSpawner>(
        self,
        spawner: &'c S,
    ) -> (
        GraphMutator<'static>,
        impl Future<Output = Result<RunStats, G2gError>> + use<'c, Clk, S>,
    ) {
        let (mutator, requests) = mutation_channel(MUTATION_QUEUE_DEPTH);
        (mutator, self.start_threaded(spawner, Some(requests)))
    }

    async fn start_threaded<S: GraphSpawner>(
        self,
        spawner: &'c S,
        mutation: Option<Receiver<MutationRequest<'static>>>,
    ) -> Result<RunStats, G2gError> {
        run_graph_threaded_inner(
            self.graph,
            self.clock,
            self.link_capacity,
            self.bus,
            self.state,
            self.progress,
            self.copy_policy,
            self.observer,
            self.recorder,
            self.visit_recorder,
            self.ticker,
            spawner,
            mutation,
            self.seek.as_ref(),
        )
        .await
    }
}
