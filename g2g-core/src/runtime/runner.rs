use core::future::Future;

use alloc::boxed::Box;

use crate::bus::BusHandle;
use crate::caps::Caps;
use crate::clock::{elect_clock, ClockCandidate, ClockPriority, ClockSync, PipelineClock};
use crate::element::{
    AsyncElement, BoxFuture, ConfigureOutcome, ElementBound, OutputSink, PushOutcome, Reconfigure,
};
use crate::error::G2gError;
use crate::format_element::CapsConstraint;
use crate::frame::PipelinePacket;
use crate::memory::{DomainSet, MemoryDomainKind};
use crate::property::{ElementMetadata, PropError, PropValue, PropertySpec};
use crate::query::{AllocationParams, LatencyReport};
use crate::runtime::channel::{link, SenderSink};
use crate::runtime::instrument::ElementProbe;
use crate::runtime::coordinator::{
    coordinator_with_recascade, negotiate_source_transform_sink, realloc_local,
    report_nego_failure, solve_last_link, ArmDirective, CoordinatorEvent, MAX_FIXATION_ATTEMPTS,
};
#[cfg(feature = "std")]
use crate::runtime::coordinator::realloc_local_dyn;
use crate::runtime::join::{select2, Either, Join2};
use crate::runtime::solver::{
    resolve_forward_output, solve_linear, ForwardResolve, NegotiationFailure,
};
use crate::runtime::state::{Flow, StateController};
use crate::segment::Segment;

/// Pick the most informative error from a pipeline's arm results. A closed-link
/// `Shutdown` is usually the *consequence* of another arm erroring first (it
/// dropped its channel end), so prefer any non-`Shutdown` error over it; fall
/// back to the first error otherwise (M81). `None` if every arm succeeded.
fn substantive_error<'a, I>(results: I) -> Option<G2gError>
where
    I: IntoIterator<Item = Option<&'a G2gError>>,
{
    let mut first: Option<G2gError> = None;
    for e in results.into_iter().flatten() {
        if *e != G2gError::Shutdown {
            return Some(e.clone());
        }
        first.get_or_insert_with(|| e.clone());
    }
    first
}

#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use crate::element::DynAsyncElement;
#[cfg(feature = "std")]
use crate::fanout::{MultiOutputElement, MultiOutputSink, MultiOutputSource, MultiSenderSink};
#[cfg(feature = "std")]
use crate::graph::Graph;
#[cfg(feature = "std")]
use crate::runtime::graph_runner::{broadcast, run_graph_inner, GraphNodeRef};
#[cfg(feature = "std")]
use crate::runtime::channel::{bounded, Sender};
#[cfg(feature = "std")]
use crate::runtime::join::{dynamic_join, join_all};

/// Source-side element trait. Sources have no input pad, so the packet-in /
/// packet-out shape of [`AsyncElement`] does not fit them. A `SourceLoop`
/// instead receives a single `run` call that iterates internally until EOS
/// and returns the count of `DataFrame` packets pushed.
pub trait SourceLoop: ElementBound {
    type RunFuture<'a>: Future<Output = Result<u64, G2gError>> + 'a
    where
        Self: 'a;

    /// Future returned by [`intercept_caps`]. Async so a source can perform
    /// I/O during negotiation (e.g. RTSP DESCRIBE + SDP parse, hardware
    /// capability probe). Sources that produce caps without I/O can return
    /// [`core::future::Ready`] and the runner pays no cost.
    type CapsFuture<'a>: Future<Output = Result<Caps, G2gError>> + 'a
    where
        Self: 'a;

    /// Negotiation-time caps query. Awaited by the runner during startup
    /// (and on re-fixate retries). `&mut self` because real implementations
    /// (e.g. `RtspSrc`) open a session here and stash the connected state
    /// for `run` to resume from. Synchronous sources just return
    /// `core::future::ready(Ok(caps))`.
    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a>;

    fn configure_pipeline(
        &mut self,
        absolute_caps: &Caps,
    ) -> Result<ConfigureOutcome, G2gError>;

    /// Runs the source until EOS or error. The implementation MUST emit a
    /// final `PipelinePacket::Eos` before returning `Ok`. Returns the number
    /// of `DataFrame` packets pushed (excluding `Eos`).
    fn run<'a>(
        &'a mut self,
        out: &'a mut dyn OutputSink,
    ) -> Self::RunFuture<'a>;

    /// Handle a downstream-originated `Reconfigure` request observed via
    /// `PushOutcome::Reconfigure` during `run`. Implementations that can
    /// retarget (eg picking a sub-stream over a main stream from an IP
    /// camera, or switching bitrate) return the new caps they will produce
    /// next; the source's `run` loop is then responsible for emitting a
    /// `CapsChanged` packet and resuming under those caps.
    ///
    /// Default: reject — most sources can't change their output shape and
    /// `FixationFailed` propagates as a fatal pipeline error.
    fn reconfigure(&mut self, _request: Reconfigure) -> Result<Caps, G2gError> {
        Err(G2gError::FixationFailed)
    }

    /// This source's latency contribution to the pipeline latency query (M12).
    /// Live capture sources (cameras, RTSP) override this to report `live`
    /// with their capture interval as `min_ns`; the default is zero, non-live
    /// (eg a file or test-pattern source that can produce data on demand).
    fn latency(&self) -> LatencyReport {
        LatencyReport::ZERO
    }

    /// The memory domain of the frames this source emits. Default
    /// [`System`](MemoryDomainKind::System); a GPU capture source (a hardware
    /// decoder source emitting VRAM frames) overrides it. Surfaced per edge by
    /// the negotiate-only path for the DOT dump (it is not part of `Caps`).
    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::System
    }

    /// The full set of memory domains this source can emit (M351). The
    /// producer-capability half of the two-sided allocation-domain negotiation;
    /// see [`AsyncElement::output_domains`](crate::element::AsyncElement::output_domains).
    /// Default: just [`output_memory`](Self::output_memory). A GPU capture source
    /// that can also deliver to System overrides this.
    fn output_domains(&self) -> DomainSet {
        DomainSet::only(self.output_memory())
    }

    /// The total stream duration in nanoseconds, the source's answer to the
    /// application `DURATION` query (M203). A source that knows the total length
    /// (a file / container source after reading its header, e.g. `Mp4Src`)
    /// overrides this; the runner publishes it on the
    /// [`PipelineProgress`](crate::runtime::PipelineProgress) handle and posts a
    /// [`DurationChanged`](crate::BusMessage::DurationChanged). The default
    /// `None` is "unknown" (a live / open-ended source, or one whose length is
    /// not yet parsed). Polled by the runner just before [`run`](Self::run).
    fn query_duration(&self) -> Option<u64> {
        None
    }

    /// Receive the downstream peer's allocation proposal (M12) so the source
    /// can allocate its output `BufferPool` from compatible parameters
    /// (size, count, alignment, domain). Default: ignore and allocate the
    /// source's own way. The proposal is advisory; a source that cannot honor
    /// it (eg cannot produce the requested domain) falls back silently.
    fn configure_allocation(&mut self, _params: &AllocationParams) {}

    /// Offer a clock to the pipeline's clock election (M12). Default: none.
    /// Live capture sources override this to provide their hardware capture
    /// clock at [`ClockPriority::LiveSource`](crate::ClockPriority::LiveSource)
    /// so the pipeline paces to capture cadence.
    fn provide_clock(&self) -> Option<ClockCandidate> {
        None
    }

    /// M16 step 5f: declare this source's negotiation-time constraint.
    /// Default: eagerly await `intercept_caps()` and wrap as a
    /// `LegacySource(Caps)` for the solver. Migrated sources override
    /// to return `Produces(CapsSet)` (or another native variant) and
    /// the chain takes the native arc-consistency path when every
    /// other element is also native.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        async move { Ok(CapsConstraint::LegacySource(self.intercept_caps().await?)) }
    }

    /// The fixed output caps this source already knows from its properties,
    /// readable synchronously without negotiation or I/O (M195). The auto-plug
    /// `decodebin` parser consults it to learn its upstream caps, so a property
    /// that re-types the output (a `filesrc`'s `bytestream-format`) is reflected
    /// into the chain search. The default `None` means "fall back to the
    /// registry's declared caps"; a source whose output media type is
    /// property-driven overrides it. Returns `None` when the caps are only known
    /// at run time (e.g. `bytestream-format=auto`, which sniffs the file header).
    fn configured_output_caps(&self) -> Option<Caps> {
        None
    }

    /// Like [`configured_output_caps`](Self::configured_output_caps) but permitted
    /// to do I/O to determine the type (M480): the auto-plug `decodebin` parser
    /// calls this once, at parse time, to pick the demuxer. A `bytestream-format=
    /// auto` source overrides it to sniff the file header now (so a mislabeled
    /// `.ts` that is really an MP4 still auto-plugs the right demuxer, the way
    /// GStreamer's runtime `typefind` would), where `configured_output_caps`
    /// returns `None` because it may not read. Default: the no-I/O caps.
    fn probe_output_caps(&mut self) -> Option<Caps> {
        self.configured_output_caps()
    }

    /// The runtime properties this source type exposes (M104), the GObject
    /// property-spec analog. Default: none. A source overrides this (and
    /// [`set_property`](Self::set_property) / [`get_property`](Self::get_property))
    /// to be settable by name from a `gst-launch` pipeline (eg `filesrc
    /// location=...`, `videotestsrc pattern=...`).
    fn properties(&self) -> &'static [PropertySpec] {
        &[]
    }

    /// Static introspection metadata for this source type (M178), the
    /// `gst-inspect` "Factory Details". Default: empty.
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::default()
    }

    /// Receive this source instance's log name (M179), assigned by the runner.
    /// Default: ignore.
    fn set_instance_name(&mut self, _name: alloc::string::String) {}

    /// Set a property by name (M104). Default: [`PropError::Unknown`].
    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    /// Read a property back by name (M104). Default: `None`.
    fn get_property(&self, _name: &str) -> Option<PropValue> {
        None
    }
}

/// Per-link queue depth handed to a runner. Each forward link between
/// elements holds this many in-flight packets before backpressure
/// kicks in; the steady-state glass-to-glass latency floor under
/// backpressure is roughly `2 * link_capacity * consumer_period`.
///
/// Construct via a [`LatencyProfile`] for intent-based selection
/// (`LatencyProfile::Live` for camera-to-display pipelines,
/// `LatencyProfile::Throughput` for batch jobs) or via `From<usize>`
/// for fine-grained tuning. Both forms compose through the runner's
/// `impl Into<LinkCapacity>` parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkCapacity(usize);

impl LinkCapacity {
    /// `n` clamped to at least 1: a zero-capacity link would deadlock
    /// the producer on its first push.
    pub fn new(n: usize) -> Self {
        Self(n.max(1))
    }

    /// Underlying queue depth. Used by the runner internals; callers
    /// typically pass the `LinkCapacity` (or a `LatencyProfile`)
    /// directly into the runner instead of unpacking.
    pub fn get(self) -> usize {
        self.0
    }
}

impl From<usize> for LinkCapacity {
    fn from(n: usize) -> Self {
        Self::new(n)
    }
}

/// Intent-based selector for [`LinkCapacity`]. Picks the link queue
/// depth from the workload's latency-vs-throughput tradeoff so callers
/// don't have to remember the steady-state floor formula
/// (`2 * cap * consumer_period`).
///
/// At 60 fps:
/// - `Live` (cap=2) -> ~67 ms floor. Right for RTSP -> decode -> display.
/// - `Throughput` (cap=8) -> ~267 ms floor. Right for file ingest /
///   batch where smoothing jitter matters more than time-to-glass.
/// - `Custom(n)` -> caller picks. Useful when a profile bisection or
///   live-edge tuning needs a specific value (a smoke test setting
///   `cap=1` to push the floor below one frame, for example).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatencyProfile {
    /// `link_capacity = 2`. Live camera -> display.
    Live,
    /// `link_capacity = 8`. Batch / throughput.
    Throughput,
    /// Caller-specified depth.
    Custom(usize),
}

impl LatencyProfile {
    /// The `LinkCapacity` this profile maps to.
    pub fn link_capacity(self) -> LinkCapacity {
        match self {
            Self::Live => LinkCapacity::new(2),
            Self::Throughput => LinkCapacity::new(8),
            Self::Custom(n) => LinkCapacity::new(n),
        }
    }
}

impl From<LatencyProfile> for LinkCapacity {
    fn from(p: LatencyProfile) -> Self {
        p.link_capacity()
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RunStats {
    pub frames_emitted: u64,
    pub frames_consumed: u64,
    /// Frames dropped by leaky links (`LinkPolicy::DropOldest`/`DropNewest`)
    /// under downstream stall. `0` for all-`Block` pipelines and for runners
    /// that don't expose per-edge policy (only `run_graph` does today).
    pub frames_dropped: u64,
    /// Aggregated source-to-sink latency (M12), computed once after
    /// negotiation. Linear runners fold every element's `latency()`; fan-in /
    /// fan-out runners leave this at `ZERO` (topology aggregation deferred).
    pub latency: LatencyReport,
    /// The allocation proposal (M12) handed to the head producer (the source),
    /// negotiated from downstream. `None` when no downstream element proposed
    /// one; always `None` for fan-in / fan-out runners (deferred).
    pub allocation: Option<AllocationParams>,
    /// Priority of the clock the pipeline elected (M12). `SystemFallback` when
    /// no element provided one (the supplied clock stands); always
    /// `SystemFallback` for fan-in / fan-out runners (deferred).
    pub clock_priority: ClockPriority,
    /// `now_ns()` of the elected clock, read once after election — the
    /// pipeline's base-time origin.
    pub base_time_ns: u64,
    /// M18 β scaffolding: number of `CoordinatorEvent`s the coordinator
    /// task observed over this run's control channel. Today the only
    /// event is a boundary forwarding a mid-stream `CapsChanged` the next
    /// element accepted; β will turn each into a `Recascade`. `0` for
    /// runners that don't yet spawn a coordinator (simple / fan-out /
    /// fan-in / muxer).
    pub coordinator_events: u64,
    /// Measured per-element telemetry (M399): each interior element's `process()`
    /// latency distribution (p50/p99) and input-link fill, in topological order.
    /// Populated by `run_graph` and the two linear runners
    /// (`run_simple_pipeline` / `run_source_transform_sink`); empty for the
    /// fan-in / fan-out / session / muxer runners and under `no_std` (no clock to
    /// measure with). Sources carry no `process()` so they do not appear; their
    /// cost surfaces as the downstream element's input fill.
    pub per_element: alloc::vec::Vec<crate::runtime::ElementLatency>,
}

impl RunStats {
    /// A human-readable end-of-run summary of the pipeline telemetry (M287):
    /// frame counts + drop rate, the aggregated declared latency window, the
    /// elected clock, and the negotiated head allocation. `g2g-launch` prints
    /// this at end (and a host can log it). The latency is the chain's
    /// *declared* min/max fold (each element's `latency()`), not a measured
    /// runtime histogram; per-element / per-link p50/p99 is a follow-up.
    pub fn report(&self) -> alloc::string::String {
        use alloc::format;
        let ms = |ns: u64| ns as f64 / 1.0e6;
        let seen = self.frames_consumed + self.frames_dropped;
        let drop_pct = if seen > 0 { self.frames_dropped as f64 * 100.0 / seen as f64 } else { 0.0 };

        let mut s = alloc::string::String::from("pipeline run summary:\n");
        s.push_str(&format!(
            "  frames:  emitted {}, consumed {}, dropped {} ({drop_pct:.1}% drop)\n",
            self.frames_emitted, self.frames_consumed, self.frames_dropped
        ));
        let max = match self.latency.max_ns {
            Some(m) => format!("{:.1} ms", ms(m)),
            None => alloc::string::String::from("unbounded"),
        };
        s.push_str(&format!(
            "  latency: {:.1} ms .. {max} ({}) [declared]\n",
            ms(self.latency.min_ns),
            if self.latency.live { "live" } else { "non-live" },
        ));
        s.push_str(&format!(
            "  clock:   {:?} (base {} ns)\n",
            self.clock_priority, self.base_time_ns
        ));
        if let Some(a) = &self.allocation {
            s.push_str(&format!(
                "  alloc:   {} B x {}, {:?}, align {}\n",
                a.size_bytes, a.min_buffers, a.domain, a.align
            ));
        }
        // Measured per-element process latency + input fill (M399). Only when the
        // runner collected it (graph / linear runners under `std`); declared-only
        // runs and `no_std` leave this empty.
        if !self.per_element.is_empty() {
            s.push_str("  per-element [measured]:\n");
            for e in &self.per_element {
                // proc.count == 0 means fill was sampled but no `process()` timing
                // was taken (no_std, no clock); show fill alone in that case.
                if e.proc.count > 0 {
                    s.push_str(&format!(
                        "    {:<16} proc p50 {:.2} ms / p99 {:.2} ms (n={}), in-fill {}%/{}% avg/max\n",
                        e.name,
                        ms(e.proc.p50_ns),
                        ms(e.proc.p99_ns),
                        e.proc.count,
                        e.fill_mean_pct,
                        e.fill_max_pct,
                    ));
                } else {
                    s.push_str(&format!(
                        "    {:<16} in-fill {}%/{}% avg/max\n",
                        e.name, e.fill_mean_pct, e.fill_max_pct,
                    ));
                }
            }
        }
        s
    }
}

/// M16 workaround #3 Phase B helper: re-solve the downstream subgraph
/// when a forward `CapsChanged` crosses a format boundary mid-stream.
///
/// Today's 3-element runner has a single downstream link (boundary →
/// sink), so the subgraph is one link and the solver's role is
/// structural: it queries the sink's declared `CapsConstraint` (which
/// may reject the boundary's output via `Accepts(set)` cleanly) before
/// the sink ever sees `configure_pipeline`. The returned `Caps` is what
/// the runner then hands to `configure_pipeline`.
///
/// Longer chains (4+ elements, future runner variants) will iterate
/// the solver result to reconfigure every changed downstream link, not
/// just the immediate next element — that's the structural unlock
/// DESIGN.md §4.13.4 calls out.
///
/// Forward × reverse race (§7): an `EmptyLink` here means the sink
/// can't take the boundary's output. The caller drops the forward
/// `CapsChanged` and signals a reverse `Reconfigure` *into the
/// boundary*, not past it to the source — that boundary owns the
/// derivation and is the right place to surface the structured
/// failure. An `Unfixable` (the boundary's caps left a ranged field
/// like `Rate::Any` — common for decoders that don't know framerate at
/// the pixel level) is *not* a failure: it means the sink accepted the
/// shape, the caps just aren't fully fixated. Pass `new_caps` through
/// unchanged.
fn re_solve_downstream_sink<S>(new_caps: &Caps, sink: &S) -> Result<Caps, NegotiationFailure>
where
    S: AsyncElement + ?Sized,
{
    re_solve_against_sink_constraint(new_caps, &sink.caps_constraint_as_sink())
}

/// Shared core of the downstream re-solve: solve `LegacySource(new_caps)`
/// against the sink's already-evaluated constraint. Factored out so the
/// generic ([`re_solve_downstream_sink`]) and `Box`-erased
/// ([`re_solve_downstream_dyn_sink`]) callers don't duplicate the
/// `Unfixable`-is-not-a-failure handling.
fn re_solve_against_sink_constraint(
    new_caps: &Caps,
    sink_c: &CapsConstraint<'_>,
) -> Result<Caps, NegotiationFailure> {
    let src_c = CapsConstraint::LegacySource(new_caps.clone());
    match solve_linear(&[&src_c, sink_c]) {
        Ok(links) => links.into_iter().last().ok_or(NegotiationFailure::Degenerate),
        Err(NegotiationFailure::Unfixable { .. }) => Ok(new_caps.clone()),
        Err(other) => Err(other),
    }
}

/// Fan-out Phase C FO-2: the [`DynAsyncElement`] counterpart of
/// [`re_solve_downstream_sink`], for `Box`-erased branch sinks.
#[cfg(feature = "std")]
pub(crate) fn re_solve_downstream_dyn_sink(
    new_caps: &Caps,
    sink: &dyn DynAsyncElement,
) -> Result<Caps, NegotiationFailure> {
    re_solve_against_sink_constraint(new_caps, &sink.caps_constraint_as_sink())
}

/// Drives a `source → sink` pipeline over a single bounded link.
/// Initial Phase 1+2 negotiation runs with bounded `ReFixate` backtrack
/// (M8 piece 5): if any element's `configure_pipeline()` returns a
/// counter-proposal, the runner restarts negotiation with that counter
/// as the new starting proposal, up to `MAX_FIXATION_ATTEMPTS` total.
pub async fn run_simple_pipeline<Src, Snk, Clk>(
    source: &mut Src,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_simple_pipeline_inner(source, sink, clock, link_capacity, None, None).await
}

/// As [`run_simple_pipeline`], but driven by a [`StateController`] (M76).
///
/// The sink arm gates on the controller: while the state is below
/// `Playing` the sink stops pulling, the bounded link fills, and backpressure
/// stalls the source. `set_state(Playing)` (from another task) opens the gate
/// and data flows; `set_state(Null)` stops the sink arm and ends the run.
/// Negotiation still runs eagerly at startup (it is resource acquisition, the
/// `READY` step); only data flow is gated. Pass the controller starting in
/// `Paused` for the common "build prerolled, then play" shape.
pub async fn run_simple_pipeline_stateful<Src, Snk, Clk>(
    source: &mut Src,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    state: &StateController,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_simple_pipeline_inner(source, sink, clock, link_capacity, None, Some(state.clone())).await
}

/// As [`run_simple_pipeline`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed) to
/// `bus` on a startup or mid-stream negotiation failure (M18 item 7).
pub async fn run_simple_pipeline_with_bus<Src, Snk, Clk>(
    source: &mut Src,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: &BusHandle,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_simple_pipeline_inner(source, sink, clock, link_capacity, Some(bus), None).await
}

async fn run_simple_pipeline_inner<Src, Snk, Clk>(
    source: &mut Src,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
    state: Option<StateController>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
    // M16 step 5f: startup negotiation honors `SourceLoop::caps_constraint`
    // so migrated native sources (e.g. `VideoTestSrc::Produces(...)`)
    // take the native solver path. `ReFixate` retry falls back to
    // `LegacySource(counter)` because counter-proposals are a legacy
    // model concept and native sources don't accept them.
    let mut refix_counter: Option<Caps> = None;
    let mut attempts = 0u32;
    let negotiated_caps = loop {
        attempts += 1;
        if attempts > MAX_FIXATION_ATTEMPTS {
            return Err(G2gError::FixationFailed);
        }
        // Resolve src_c in its own scope so its borrow of `source`
        // releases before `configure_pipeline(&fixated)` below.
        let fixated = {
            let src_c = match &refix_counter {
                Some(c) => CapsConstraint::LegacySource(c.clone()),
                None => source.caps_constraint().await?,
            };
            let sink_c = sink.caps_constraint_as_sink();
            solve_last_link(&[&src_c, &sink_c], bus)?
        };
        match source.configure_pipeline(&fixated)? {
            ConfigureOutcome::Accepted => {}
            ConfigureOutcome::ReFixate(counter) => {
                refix_counter = Some(counter);
                continue;
            }
        }
        match sink.configure_pipeline(&fixated)? {
            ConfigureOutcome::Accepted => break fixated,
            ConfigureOutcome::ReFixate(counter) => {
                refix_counter = Some(counter);
                continue;
            }
        }
    };

    // M12 latency query: fold the configured chain source → sink.
    let latency = LatencyReport::aggregate([source.latency(), AsyncElement::latency(sink)]);

    // M12 allocation query: the sink proposes buffers; the source allocates
    // its output pool to match (zero-copy handoff when it can honor them). M351:
    // the proposed domain is reconciled against what the source can actually emit
    // (a two-sided negotiation), so a multi-domain sink/source pair settles on a
    // shared domain (GPU-preferred) instead of the sink dictating unilaterally;
    // no shared domain is a loud conflict rather than a silent mismatch.
    let allocation = match sink.propose_allocation(&negotiated_caps) {
        Some(p) => Some(p.resolve_for_producer(SourceLoop::output_domains(source))?),
        None => None,
    };
    if let Some(p) = &allocation {
        source.configure_allocation(p);
    }

    // M12 clock distribution: elect the pipeline clock (source > sink > fallback).
    let elected = elect_clock([source.provide_clock(), AsyncElement::provide_clock(sink)]);
    let (clock_priority, base_time_ns) = match &elected {
        Some(c) => (c.priority, c.clock.now_ns()),
        None => (ClockPriority::SystemFallback, clock.now_ns()),
    };

    // Hand the elected clock + base time to the sink so it can present each
    // frame at its running-time deadline (PTS pacing). Only when a clock was
    // elected; without one the sink presents as fast as backpressure allows.
    // M176: under a state controller, arm a Playing-transition anchor so the
    // sink bases presentation on the play edge, not on startup / the preroll
    // frame; without one, the eager startup base time stands.
    if let Some(c) = &elected {
        let sync = match &state {
            Some(sc) => ClockSync::with_play_anchor(
                c.clock.clone(),
                base_time_ns,
                sc.arm_play_anchor(c.clock.clone()),
            ),
            None => ClockSync::new(c.clock.clone(), base_time_ns),
        };
        AsyncElement::set_clock_sync(sink, sync);
    }

    let (link_tx, link_rx) = link(link_capacity);

    let source_fut = async move {
        let mut adapter = SenderSink::new(link_tx);
        // M81: every stream opens with a SEGMENT, ahead of the source's data,
        // so a sink maps frame timestamps to running time from the first frame.
        let _ = adapter
            .push(PipelinePacket::Segment(Segment::new()))
            .await?;
        let emitted = source.run(&mut adapter).await?;
        Ok::<u64, G2gError>(emitted)
    };

    // M399: measured per-element telemetry for the sink (the linear runner's one
    // interior element with a `process()`); the source's cost surfaces as fill.
    let sink_probe = ElementProbe::new(alloc::string::String::from(
        crate::log::short_type_name::<Snk>(),
    ));
    let probe_for_sink = sink_probe.clone();

    let bus_for_sink = bus.cloned();
    let state_for_sink = state;
    let sink_fut = async move {
        let bus_for_sink = bus_for_sink;
        let state_for_sink = state_for_sink;
        let probe_for_sink = probe_for_sink;
        let mut null = NullSink;
        let mut consumed: u64 = 0;
        let mut prerolled_self = false;
        // M360 re-preroll: the generation this arm last prerolled at, and whether
        // it is currently draining stale pre-seek frames (after a paused flushing
        // seek) until the `Flush` arrives.
        let mut preroll_gen = state_for_sink.as_ref().map_or(0, |sc| sc.preroll_generation());
        let mut flushing = false;
        loop {
            // Flow gate (M76/M77): below `Playing` the sink parks here, so it
            // stops draining the link; the bounded channel fills and
            // backpressure stalls the source. `Playing` opens the gate; `Null`
            // ends the arm. In non-live `Paused` the gate admits exactly one
            // buffer (this sink's preroll frame) before it holds.
            if let Some(sc) = &state_for_sink {
                if sc.flow_gate(prerolled_self, preroll_gen).await == Flow::Stop {
                    return Ok::<u64, G2gError>(consumed);
                }
                // M360: a `request_repreroll` (paused flushing seek) bumped the
                // generation. Re-arm this arm's preroll and drain the stale
                // pre-seek frames until the `Flush`, so the post-flush target
                // becomes the new visible preroll rather than a stale buffer.
                let gen = sc.preroll_generation();
                if gen != preroll_gen {
                    preroll_gen = gen;
                    prerolled_self = false;
                    flushing = true;
                }
            }
            match link_rx.recv().await {
                // M360: discard stale pre-seek buffers while draining toward the
                // `Flush`; control packets fall through (the `Flush` ends drain).
                Some(PipelinePacket::DataFrame(_)) if flushing => continue,
                Some(PipelinePacket::Eos) => {
                    sink.process(PipelinePacket::Eos, &mut null).await?;
                    // M77: EOS during preroll still completes the async
                    // `Paused` transition (idempotent; no-op once playing).
                    if let Some(sc) = &state_for_sink {
                        sc.notify_prerolled();
                    }
                    return Ok::<u64, G2gError>(consumed);
                }
                Some(PipelinePacket::CapsChanged(new_caps)) => {
                    // M16 workaround #3 Phase B: re-solve the downstream
                    // subgraph before applying. For a 2-element chain
                    // the subgraph is one link, so the solver's role is
                    // structural — it checks the sink's declared
                    // `CapsConstraint::caps_constraint_as_sink()`
                    // (which a native sink may use to reject the new
                    // shape cleanly) before any `configure_pipeline`
                    // call. Failure becomes a structured upstream
                    // `Renegotiate` request instead of an opaque
                    // `CapsMismatch`.
                    let sink_caps = match re_solve_downstream_sink(&new_caps, &*sink) {
                        Ok(caps) => caps,
                        Err(failure) => {
                            report_nego_failure(bus_for_sink.as_ref(), failure);
                            link_rx
                                .request_reconfigure(Reconfigure::Renegotiate);
                            continue;
                        }
                    };
                    // M8 piece 1: runner cascades mid-stream caps changes
                    // through configure_pipeline before the element sees
                    // the notification packet. Guarantees DataFrames with
                    // the new caps never reach a stale element.
                    match sink.configure_pipeline(&sink_caps)? {
                        ConfigureOutcome::Accepted => {
                            // M18 α: element-local re-allocation under the
                            // new caps before the sink sees the packet.
                            realloc_local(sink, &sink_caps);
                            sink.process(
                                PipelinePacket::CapsChanged(sink_caps),
                                &mut null,
                            )
                            .await?;
                        }
                        // M8 piece 5: a sink that rejects new caps fires
                        // its counter-proposal upstream as a Reconfigure
                        // signal. The source observes it on its next push
                        // (piece 4 wires source-side handling). The
                        // CapsChanged packet is dropped — caps were not
                        // accepted — and we keep draining old-caps frames
                        // until the source emits a fresh CapsChanged.
                        ConfigureOutcome::ReFixate(counter) => {
                            link_rx.request_reconfigure(Reconfigure::Propose(counter));
                        }
                    }
                }
                // M360: the `Flush` ends the re-preroll drain; the next
                // (post-flush) DataFrame becomes the new visible preroll.
                Some(PipelinePacket::Flush) => {
                    flushing = false;
                    sink.process(PipelinePacket::Flush, &mut null).await?;
                }
                Some(packet) => {
                    let is_buffer = matches!(packet, PipelinePacket::DataFrame(_));
                    if is_buffer {
                        consumed += 1;
                    }
                    // M399: time the data-frame `process()` and sample input fill.
                    let timed = is_buffer.then(|| &*probe_for_sink);
                    if let Some(p) = timed {
                        p.record_fill(link_rx.fill_percent());
                    }
                    let t0 = ElementProbe::mark();
                    sink.process(packet, &mut null).await?;
                    if let Some(p) = timed {
                        p.record_proc_since(t0);
                    }
                    // M174 upstream QoS: a sink that dropped a late frame asks to
                    // shed load; forward its report onto the incoming link, where
                    // the source observes it as `PushOutcome::Qos` and skips ahead.
                    if let Some(qos) = sink.take_qos() {
                        link_rx.request_qos(qos);
                    }
                    // Keyframe-request / renegotiation a sink originates (e.g. a
                    // WebRTC sink on a remote PLI): forward it up the same reverse
                    // channel, where the encoder sees it as `PushOutcome::Reconfigure`.
                    if let Some(reconf) = sink.take_reconfigure() {
                        link_rx.request_reconfigure(reconf);
                    }
                    // Target bitrate (WebRTC BWE) up the same reverse channel; the
                    // upstream encoder observes it as `PushOutcome::Bitrate`.
                    if let Some(bps) = sink.take_bitrate() {
                        link_rx.request_bitrate(bps);
                    }
                    // M77: the first buffer in non-live `Paused` is the preroll
                    // frame; mark this arm prerolled so the gate flips from
                    // preroll-grant to hold, and report it for aggregation.
                    // Idempotent and a no-op while `Playing`.
                    if is_buffer && !prerolled_self {
                        prerolled_self = true;
                        if let Some(sc) = &state_for_sink {
                            sc.notify_prerolled();
                        }
                    }
                }
                None => return Ok(consumed),
            }
        }
    };

    let (src_res, snk_res) = Join2::new(source_fut, sink_fut).await;
    // M81: a closed-link `Shutdown` on the source arm can be the consequence of
    // the sink arm's real error (it dropped the link), so surface the
    // substantive one rather than whichever arm we check first.
    if let Some(e) = substantive_error([src_res.as_ref().err(), snk_res.as_ref().err()]) {
        return Err(e);
    }
    let emitted = src_res?;
    let consumed = snk_res?;

    // M399: the sink arm has joined, so its probe is settled; snapshot it.
    let per_element = alloc::vec![sink_probe.snapshot()];
    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        frames_dropped: 0,
        latency,
        allocation,
        clock_priority,
        base_time_ns,
        coordinator_events: 0,
        per_element,
    })
}

/// Drives a `source → fan-out element → N sinks` pipeline (M9 fan-out core).
/// The fan-out element (a [`MultiOutputElement`], e.g. `Router`) sends each
/// `DataFrame` to one branch and broadcasts `CapsChanged` to all; the runner
/// broadcasts `Eos` to every branch on shutdown.
///
/// Heterogeneous branches arrive as `Box`-erased `&mut dyn DynAsyncElement`
/// (std only). Negotiation fixates the source proposal once and configures
/// every element with it (DESIGN.md §4.2); per-branch caps negotiation is
/// M10, so a sink returning `ReFixate` here fails with `FixationFailed`.
#[cfg(feature = "std")]
pub async fn run_source_fanout<Src, Tx, Clk>(
    source: &mut Src,
    fanout: &mut Tx,
    sinks: Vec<&mut dyn DynAsyncElement>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: MultiOutputElement,
    Clk: PipelineClock,
{
    run_source_fanout_inner(source, fanout, sinks, clock, link_capacity, None).await
}

/// As [`run_source_fanout`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed) to
/// `bus` on a startup or per-branch mid-stream negotiation failure (item 7).
#[cfg(feature = "std")]
pub async fn run_source_fanout_with_bus<Src, Tx, Clk>(
    source: &mut Src,
    fanout: &mut Tx,
    sinks: Vec<&mut dyn DynAsyncElement>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: &BusHandle,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: MultiOutputElement,
    Clk: PipelineClock,
{
    run_source_fanout_inner(source, fanout, sinks, clock, link_capacity, Some(bus)).await
}

#[cfg(feature = "std")]
async fn run_source_fanout_inner<Src, Tx, Clk>(
    source: &mut Src,
    fanout: &mut Tx,
    sinks: Vec<&mut dyn DynAsyncElement>,
    _clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: MultiOutputElement,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
    let branch_count = sinks.len();
    assert!(branch_count > 0, "fan-out needs at least one sink");

    // M18 step 1: solve source → fanout via the solver using the new
    // `MultiOutputElement::caps_constraint_as_input()` trait method
    // (M16 step 4c had this constructing `LegacySink` inline). The
    // fan-out acts as the linear "sink" of the negotiation chain;
    // the real sinks downstream of it broadcast-receive the same
    // fixated caps and don't participate in narrowing. Phase C FO-2
    // (per-branch downstream re-solve once a mid-stream `CapsChanged`
    // crosses the fan-out boundary) lands once β (the coordinator
    // restructure) does — it slots in here via per-branch calls to
    // `re_solve_downstream_sink`.
    let fixated = {
        let src_c = source.caps_constraint().await?;
        let fanout_c = fanout.caps_constraint_as_input();
        solve_last_link(&[&src_c, &fanout_c], bus)?
    };

    source.configure_pipeline(&fixated)?.reject_refixate()?;
    MultiOutputElement::configure_pipeline(fanout, &fixated)?.reject_refixate()?;
    let mut sinks = sinks;
    for sink in sinks.iter_mut() {
        sink.configure_pipeline(&fixated)?.reject_refixate()?;
    }

    let (src_tx, src_rx) = link(link_capacity);
    let mut branch_senders = Vec::with_capacity(branch_count);
    let mut branch_receivers = Vec::with_capacity(branch_count);
    for _ in 0..branch_count {
        let (tx, rx) = link(link_capacity);
        branch_senders.push(SenderSink::new(tx));
        branch_receivers.push(rx);
    }

    let source_fut: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
        let mut adapter = SenderSink::new(src_tx);
        source.run(&mut adapter).await
    });

    let router_fut: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
        let mut multi = MultiSenderSink::new(branch_senders);
        loop {
            match src_rx.recv().await {
                Some(PipelinePacket::Eos) => {
                    MultiOutputElement::process(fanout, PipelinePacket::Eos, &mut multi).await?;
                    for port in 0..branch_count {
                        multi.push_to(port, PipelinePacket::Eos).await?;
                    }
                    return Ok::<u64, G2gError>(0);
                }
                Some(packet) => {
                    MultiOutputElement::process(fanout, packet, &mut multi).await?;
                }
                None => return Ok(0),
            }
        }
    });

    let mut arms: Vec<BoxFuture<'_, Result<u64, G2gError>>> =
        Vec::with_capacity(branch_count + 2);
    arms.push(source_fut);
    arms.push(router_fut);

    for (sink, rx) in sinks.into_iter().zip(branch_receivers) {
        let bus_for_branch = bus.cloned();
        let sink_fut: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
            let bus_for_branch = bus_for_branch;
            let mut null = NullSink;
            let mut consumed: u64 = 0;
            loop {
                match rx.recv().await {
                    Some(PipelinePacket::Eos) => {
                        sink.process(PipelinePacket::Eos, &mut null).await?;
                        return Ok::<u64, G2gError>(consumed);
                    }
                    Some(PipelinePacket::CapsChanged(new_caps)) => {
                        // M18 Phase C FO-2: per-branch downstream re-solve
                        // (Phase B applied per branch). Each branch runs in
                        // its own arm, so the broadcast `CapsChanged` is
                        // re-solved on every branch concurrently. FO-1
                        // strict default: a branch whose declared
                        // `caps_constraint_as_sink()` rejects the new caps
                        // fails the fan-out loud (matches GStreamer's
                        // `tee`-with-rejecting-downstream). `AllowBranchDrop`
                        // graceful degradation is a future opt-in.
                        let branch_caps = re_solve_downstream_dyn_sink(&new_caps, &*sink)
                            .map_err(|f| {
                                report_nego_failure(bus_for_branch.as_ref(), f);
                                G2gError::CapsMismatch
                            })?;
                        match sink.configure_pipeline(&branch_caps)? {
                            ConfigureOutcome::Accepted => {
                                // M18 α: element-local re-allocation of this
                                // branch under its re-solved caps.
                                realloc_local_dyn(sink, &branch_caps);
                                sink.process(
                                    PipelinePacket::CapsChanged(branch_caps),
                                    &mut null,
                                )
                                .await?;
                            }
                            ConfigureOutcome::ReFixate(counter) => {
                                rx.request_reconfigure(Reconfigure::Propose(counter));
                            }
                        }
                    }
                    Some(packet) => {
                        if matches!(packet, PipelinePacket::DataFrame(_)) {
                            consumed += 1;
                        }
                        sink.process(packet, &mut null).await?;
                    }
                    None => return Ok(consumed),
                }
            }
        });
        arms.push(sink_fut);
    }

    let results = join_all(arms).await;
    // M81: a real branch error closes the shared links, surfacing as Shutdown on
    // the sibling arms; surface the substantive error rather than whichever arm
    // the count loop unwraps first (consistent with the linear path).
    if let Some(e) = substantive_error(results.iter().map(|r| r.as_ref().err())) {
        return Err(e);
    }
    let mut counts = Vec::with_capacity(results.len());
    for r in results {
        counts.push(r?);
    }
    // Arm order: [source, router, sink0, sink1, ...].
    let emitted = counts[0];
    let consumed: u64 = counts[2..].iter().copied().sum();
    // Fan-out latency / allocation / clock election across N branches is
    // deferred (M12 covers the linear path); report neutral values rather than
    // a misleading partial one.
    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        frames_dropped: 0,
        latency: LatencyReport::ZERO,
        allocation: None,
        clock_priority: ClockPriority::SystemFallback,
        base_time_ns: 0,
        coordinator_events: 0,
        per_element: alloc::vec::Vec::new(),
    })
}

// ===========================================================================
// M310: runtime request pads (dynamic fan-out branches).
// ===========================================================================

/// Self-identifying outcome of a dynamic-fan-out arm. Arm indices are not stable
/// once branches are added at runtime, so each arm reports what it was.
#[cfg(feature = "std")]
enum DynArmOut {
    /// The source produced this many `DataFrame`s.
    Source(u64),
    /// The router (no count of its own).
    Router,
    /// A branch consumed this many `DataFrame`s.
    Branch(u64),
}

/// A handle to add output branches to a *running* dynamic fan-out (M310): the
/// runtime equivalent of GStreamer's tee request pads. Each
/// [`add_branch`](Self::add_branch) attaches a new sink the router routes frames
/// to; the new branch configures from the fan-out's sticky caps on attach, then
/// receives its share of subsequent frames. Cheap to clone (channel senders), so
/// several controllers can request pads.
///
/// `'a` is the run's lifetime: the handle is used concurrently with the run
/// future and must be dropped no later than it. Branches added after the source
/// has ended are rejected ([`G2gError::Shutdown`]).
#[derive(Clone)]
#[allow(missing_debug_implementations)]
#[cfg(feature = "std")]
pub struct DynamicFanoutHandle<'a> {
    new_branch_tx: Sender<alloc::boxed::Box<dyn DynAsyncElement + 'a>>,
}

#[cfg(feature = "std")]
impl<'a> DynamicFanoutHandle<'a> {
    /// Request a new output pad: attach `sink` as a branch of the running
    /// fan-out. Returns [`G2gError::Shutdown`] if the fan-out has already
    /// finished (the source ended), so no branch can be added.
    pub fn add_branch(
        &self,
        sink: alloc::boxed::Box<dyn DynAsyncElement + 'a>,
    ) -> Result<(), G2gError> {
        self.new_branch_tx.try_send(sink).map_err(|_| G2gError::Shutdown)
    }
}

/// How a dynamic fan-out distributes each `DataFrame` across its branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "std")]
enum FanOutMode {
    /// Each `DataFrame` goes to exactly one branch, round-robin (the `Router`
    /// model). Used when a frame must not be duplicated (load spreading).
    Route,
    /// Each `DataFrame` is shared to *every* branch (the `tee` model). The
    /// frame's memory is made shareable once (M250), so the per-branch copies
    /// are zero-copy refcount bumps, not byte copies.
    Broadcast,
}

/// Drives `source -> dynamic router -> N branches`, where branches can be added
/// at runtime through the returned [`DynamicFanoutHandle`] (M310 request pads).
///
/// The router distributes `DataFrame`s round-robin across the currently-attached
/// branches (frames are not `Clone`, so this routes rather than broadcasts, the
/// `Router` model) and broadcasts `CapsChanged` / `Eos` to every branch. See
/// [`run_source_tee_dynamic`] for the broadcast (tee) variant. The fan-out's caps
/// are "sticky": the source's fixated output caps are replayed to each branch the
/// moment it attaches, so a late branch configures correctly without having seen
/// the original negotiation.
///
/// Returns the handle plus the run future; drive them concurrently (await the
/// future while using the handle from another task). The run completes once the
/// source ends and every attached branch has drained.
#[cfg(feature = "std")]
pub fn run_source_router_dynamic<'a, Src>(
    source: &'a mut Src,
    link_capacity: impl Into<LinkCapacity>,
) -> (DynamicFanoutHandle<'a>, impl Future<Output = Result<RunStats, G2gError>> + 'a)
where
    Src: SourceLoop + 'a,
{
    run_source_fanout_dynamic(source, link_capacity, FanOutMode::Route)
}

/// Drives `source -> dynamic tee -> N branches` (M319): the broadcast counterpart
/// of [`run_source_router_dynamic`]. Each `DataFrame` is shared to *every*
/// currently-attached branch via the M250 zero-copy frame-sharing path
/// ([`MemoryDomain::make_shareable`](crate::MemoryDomain::make_shareable) once,
/// then a refcount handle per branch), so a CPU or GPU frame fans out to N
/// consumers with no byte copies. This is the runtime equivalent of GStreamer's
/// `tee` request pads: an inference branch and a display branch both see every
/// frame, and either can be attached while the pipeline runs.
///
/// Branch attach, sticky caps, and shutdown are identical to
/// [`run_source_router_dynamic`]; only `DataFrame` distribution differs (broadcast
/// vs round-robin). `CapsChanged` / `Segment` / `Flush` / `Eos` are broadcast in
/// both modes.
#[cfg(feature = "std")]
pub fn run_source_tee_dynamic<'a, Src>(
    source: &'a mut Src,
    link_capacity: impl Into<LinkCapacity>,
) -> (DynamicFanoutHandle<'a>, impl Future<Output = Result<RunStats, G2gError>> + 'a)
where
    Src: SourceLoop + 'a,
{
    run_source_fanout_dynamic(source, link_capacity, FanOutMode::Broadcast)
}

/// Shared driver behind [`run_source_router_dynamic`] (route) and
/// [`run_source_tee_dynamic`] (broadcast); `mode` selects how each `DataFrame` is
/// distributed across the attached branches.
#[cfg(feature = "std")]
fn run_source_fanout_dynamic<'a, Src>(
    source: &'a mut Src,
    link_capacity: impl Into<LinkCapacity>,
    mode: FanOutMode,
) -> (DynamicFanoutHandle<'a>, impl Future<Output = Result<RunStats, G2gError>> + 'a)
where
    Src: SourceLoop + 'a,
{
    let link_capacity: usize = link_capacity.into().get();
    // Control channel: handle -> router (new branch elements).
    let (new_branch_tx, new_branch_rx) =
        bounded::<alloc::boxed::Box<dyn DynAsyncElement + 'a>>(link_capacity);
    // Arm channel: router -> join (the spawned branch futures).
    let (new_arm_tx, new_arm_rx) =
        bounded::<BoxFuture<'a, Result<DynArmOut, G2gError>>>(link_capacity);

    let handle = DynamicFanoutHandle { new_branch_tx };

    let run = async move {
        // The source self-fixates its output caps; that becomes the sticky caps
        // replayed to every branch on attach.
        let sticky = source.intercept_caps().await?;
        source.configure_pipeline(&sticky)?.reject_refixate()?;

        let (src_tx, src_rx) = link(link_capacity);
        let source_fut: BoxFuture<'a, Result<DynArmOut, G2gError>> = Box::pin(async move {
            let mut adapter = SenderSink::new(src_tx);
            source.run(&mut adapter).await.map(DynArmOut::Source)
        });

        let router_fut: BoxFuture<'a, Result<DynArmOut, G2gError>> = Box::pin(async move {
            let mut ports: Vec<SenderSink> = Vec::new();
            let mut rr = 0usize; // round-robin cursor
            let mut accepting = true; // poll the control channel until the handle drops
            loop {
                // Attach every branch queued so far BEFORE routing the next
                // packet, so a branch that arrived before a frame never misses
                // it (select2 below is left-biased toward the source, so without
                // this drain a backlog of frames would be routed to an empty
                // port set and dropped).
                while let Some(sink) = new_branch_rx.try_recv() {
                    attach_branch(sink, link_capacity, &sticky, &mut ports, &new_arm_tx).await?;
                }

                if accepting {
                    match select2(src_rx.recv(), new_branch_rx.recv()).await {
                        Either::Left(pkt) => {
                            if route_packet(pkt, &mut ports, &mut rr, mode).await? {
                                return Ok(DynArmOut::Router); // source ended
                            }
                        }
                        Either::Right(Some(sink)) => {
                            attach_branch(sink, link_capacity, &sticky, &mut ports, &new_arm_tx)
                                .await?;
                        }
                        // Handle dropped: stop watching for new branches.
                        Either::Right(None) => accepting = false,
                    }
                } else if route_packet(src_rx.recv().await, &mut ports, &mut rr, mode).await? {
                    return Ok(DynArmOut::Router);
                }
            }
        });

        // Drop our keep-alive arm sender into the router so that when the router
        // returns (source EOS), the arm channel closes and the join can finish.
        // (new_arm_tx is moved into router_fut above.)
        let arms: Vec<BoxFuture<'a, Result<DynArmOut, G2gError>>> = alloc::vec![source_fut, router_fut];
        let results = dynamic_join(arms, new_arm_rx).await;

        let mut emitted = 0u64;
        let mut consumed = 0u64;
        for r in results {
            match r? {
                DynArmOut::Source(n) => emitted = n,
                DynArmOut::Branch(n) => consumed += n,
                DynArmOut::Router => {}
            }
        }
        Ok(RunStats {
            frames_emitted: emitted,
            frames_consumed: consumed,
            frames_dropped: 0,
            latency: LatencyReport::ZERO,
            allocation: None,
            clock_priority: ClockPriority::SystemFallback,
            base_time_ns: 0,
            coordinator_events: 0,
            per_element: alloc::vec::Vec::new(),
        })
    };

    (handle, run)
}

/// Attach a runtime-requested branch: give it its own link, replay the sticky
/// caps into it so it configures before any frame, add its sender to the port
/// set, and hand its loop future to the dynamic join.
#[cfg(feature = "std")]
async fn attach_branch<'a>(
    sink: alloc::boxed::Box<dyn DynAsyncElement + 'a>,
    link_capacity: usize,
    sticky: &Caps,
    ports: &mut Vec<SenderSink>,
    new_arm_tx: &Sender<BoxFuture<'a, Result<DynArmOut, G2gError>>>,
) -> Result<(), G2gError> {
    let (btx, brx) = link(link_capacity);
    let mut port = SenderSink::new(btx);
    port.push(PipelinePacket::CapsChanged(sticky.clone())).await?;
    ports.push(port);
    let arm: BoxFuture<'a, Result<DynArmOut, G2gError>> = Box::pin(async move {
        let mut sink = sink;
        dyn_branch_loop(sink.as_mut(), brx).await.map(DynArmOut::Branch)
    });
    new_arm_tx.try_send(arm).map_err(|_| G2gError::Shutdown)
}

/// Route one received source packet to the branch ports. A `DataFrame` goes
/// either to the next branch round-robin ([`FanOutMode::Route`]) or, shared
/// zero-copy, to every branch ([`FanOutMode::Broadcast`], the tee). `CapsChanged`
/// / `Segment` / `Flush` broadcast to every branch in both modes (those are
/// cloneable). `Eos` (or a closed source channel) is broadcast to all branches
/// and returns `Ok(true)` to tell the router the source has ended.
#[cfg(feature = "std")]
async fn route_packet(
    pkt: Option<PipelinePacket>,
    ports: &mut [SenderSink],
    rr: &mut usize,
    mode: FanOutMode,
) -> Result<bool, G2gError> {
    match pkt {
        Some(PipelinePacket::DataFrame(frame)) => {
            if !ports.is_empty() {
                match mode {
                    FanOutMode::Route => {
                        let idx = *rr % ports.len();
                        *rr = rr.wrapping_add(1);
                        ports[idx].push(PipelinePacket::DataFrame(frame)).await?;
                    }
                    // Tee: share the frame's memory once, then each branch gets a
                    // refcount handle (M250 zero-copy fan-out), no byte copy.
                    FanOutMode::Broadcast => {
                        broadcast(ports, PipelinePacket::DataFrame(frame)).await?;
                    }
                }
            }
            // No branches attached: drop the frame (a tee with no src pads).
            Ok(false)
        }
        Some(PipelinePacket::CapsChanged(caps)) => {
            for p in ports.iter_mut() {
                p.push(PipelinePacket::CapsChanged(caps.clone())).await?;
            }
            Ok(false)
        }
        Some(PipelinePacket::Segment(seg)) => {
            for p in ports.iter_mut() {
                p.push(PipelinePacket::Segment(seg)).await?;
            }
            Ok(false)
        }
        Some(PipelinePacket::Flush) => {
            for p in ports.iter_mut() {
                p.push(PipelinePacket::Flush).await?;
            }
            Ok(false)
        }
        Some(PipelinePacket::Eos) | None => {
            for p in ports.iter_mut() {
                p.push(PipelinePacket::Eos).await?;
            }
            Ok(true)
        }
    }
}

/// A dynamic branch's loop: configure from the first (sticky) `CapsChanged`, then
/// process frames until `Eos` / channel close. The configure path mirrors the
/// static fan-out branch arm so a runtime branch negotiates exactly like a built
/// one.
#[cfg(feature = "std")]
async fn dyn_branch_loop(
    sink: &mut dyn DynAsyncElement,
    rx: crate::runtime::channel::LinkReceiver,
) -> Result<u64, G2gError> {
    let mut null = NullSink;
    let mut consumed = 0u64;
    loop {
        match rx.recv().await {
            Some(PipelinePacket::Eos) => {
                sink.process(PipelinePacket::Eos, &mut null).await?;
                return Ok(consumed);
            }
            Some(PipelinePacket::CapsChanged(caps)) => {
                let branch_caps =
                    re_solve_downstream_dyn_sink(&caps, &*sink).map_err(|_| G2gError::CapsMismatch)?;
                match sink.configure_pipeline(&branch_caps)? {
                    ConfigureOutcome::Accepted => {
                        sink.process(PipelinePacket::CapsChanged(branch_caps), &mut null).await?;
                    }
                    ConfigureOutcome::ReFixate(counter) => {
                        rx.request_reconfigure(Reconfigure::Propose(counter));
                    }
                }
            }
            Some(packet) => {
                if matches!(packet, PipelinePacket::DataFrame(_)) {
                    consumed += 1;
                }
                sink.process(packet, &mut null).await?;
            }
            None => return Ok(consumed),
        }
    }
}

/// Drives a terminal multi-output *source* ([`MultiOutputSource`]: 0 inputs to N
/// outputs) into N sinks, with no upstream, the fan-out mirror of
/// [`run_fanin_session`](crate::runtime::run_fanin_session). A WHEP session
/// receiving A/V over one PeerConnection emits each track on its own pad. Each
/// output's caps configure the matching sink; the session pushes via a
/// [`MultiSenderSink`] and emits a per-output `Eos` when it ends, which the sink
/// arms observe to finish. Per-branch mid-stream re-solve is not wired here yet
/// (a follow-up, as for the egress session runner).
#[cfg(feature = "std")]
pub async fn run_fanout_session<Sess, Clk>(
    session: &mut Sess,
    sinks: Vec<&mut dyn DynAsyncElement>,
    _clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Sess: MultiOutputSource,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
    let branch_count = sinks.len();
    assert!(branch_count > 0, "fan-out session needs at least one sink");
    assert!(
        session.output_count() == branch_count,
        "session output count must match the number of sinks"
    );

    // Negotiate per output: the session self-fixates each output's caps and
    // configures the matching sink (no peer narrowing, like run_fanin_session).
    let mut sinks = sinks;
    for (i, sink) in sinks.iter_mut().enumerate() {
        let fixated = session.output_caps(i)?.fixate()?;
        sink.configure_pipeline(&fixated)?.reject_refixate()?;
    }

    let mut branch_senders = Vec::with_capacity(branch_count);
    let mut branch_receivers = Vec::with_capacity(branch_count);
    for _ in 0..branch_count {
        let (tx, rx) = link(link_capacity);
        branch_senders.push(SenderSink::new(tx));
        branch_receivers.push(rx);
    }

    let session_fut: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
        let mut multi = MultiSenderSink::new(branch_senders);
        session.run(&mut multi).await
    });

    let mut arms: Vec<BoxFuture<'_, Result<u64, G2gError>>> = Vec::with_capacity(branch_count + 1);
    arms.push(session_fut);
    for (sink, rx) in sinks.into_iter().zip(branch_receivers) {
        let sink_fut: BoxFuture<'_, Result<u64, G2gError>> = Box::pin(async move {
            let mut null = NullSink;
            let mut consumed: u64 = 0;
            loop {
                match rx.recv().await {
                    Some(PipelinePacket::Eos) => {
                        sink.process(PipelinePacket::Eos, &mut null).await?;
                        return Ok::<u64, G2gError>(consumed);
                    }
                    Some(PipelinePacket::CapsChanged(new_caps)) => {
                        match sink.configure_pipeline(&new_caps)? {
                            ConfigureOutcome::Accepted => {
                                sink.process(PipelinePacket::CapsChanged(new_caps), &mut null)
                                    .await?;
                            }
                            ConfigureOutcome::ReFixate(counter) => {
                                rx.request_reconfigure(Reconfigure::Propose(counter));
                            }
                        }
                    }
                    Some(packet) => {
                        if matches!(packet, PipelinePacket::DataFrame(_)) {
                            consumed += 1;
                        }
                        sink.process(packet, &mut null).await?;
                    }
                    None => return Ok(consumed),
                }
            }
        });
        arms.push(sink_fut);
    }

    let results = join_all(arms).await;
    let mut counts = Vec::with_capacity(results.len());
    for r in results {
        counts.push(r?);
    }
    // Arm order: [session, sink0, sink1, ...].
    let emitted = counts[0];
    let consumed: u64 = counts[1..].iter().copied().sum();
    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        frames_dropped: 0,
        latency: LatencyReport::ZERO,
        allocation: None,
        clock_priority: ClockPriority::SystemFallback,
        base_time_ns: 0,
        coordinator_events: 0,
        per_element: alloc::vec::Vec::new(),
    })
}

/// Drives an arbitrary-length linear pipeline:
/// `source -> transforms[0] -> ... -> transforms[N-1] -> sink`.
///
/// M18 item 4. Generalizes [`run_source_transform_sink`] (one transform) and
/// [`run_simple_pipeline`] (zero) past their fixed arity, lifting the
/// "runner caps at 3 elements" limit so chains like
/// `decoder -> capsfilter -> converter -> sink` are expressible. Interior
/// elements are `&mut dyn DynAsyncElement` (heterogeneous, std-only, the same
/// erasure the fan-out runner uses); source and sink stay statically typed.
///
/// Negotiation runs the solver over all `N + 2` constraints at once and
/// configures each element with its input-side caps (the source with link 0).
/// Data flows over `N + 1` bounded links across `N + 2` concurrently-joined
/// arms. On a mid-stream `CapsChanged` each interior element re-fixates its
/// output against a downstream feasibility snapshot (Caps-α), re-allocates its
/// own pool (α), and the β allocation re-cascade walks the demand back through
/// every interior hop. Clock election and latency aggregation fold the source,
/// every interior element, and the sink (via the dyn-safe `DynAsyncElement`
/// mirrors).
///
/// Owed: Caps-β, a forward coordinator re-solve walk for a downstream
/// `DerivedOutput` element that must re-derive mid-stream (driver-gated,
/// DESIGN.md §4.13.4). ReFixate at startup fails loud
/// (`FixationFailed`), as in `run_source_fanout`.
#[cfg(feature = "std")]
pub async fn run_linear_chain<Src, Snk, Clk>(
    source: &mut Src,
    transforms: Vec<&mut dyn DynAsyncElement>,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_linear_chain_inner(source, transforms, sink, clock, link_capacity, None).await
}

/// As [`run_linear_chain`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed) to
/// `bus` on a startup or mid-stream negotiation failure (M18 item 7).
#[cfg(feature = "std")]
pub async fn run_linear_chain_with_bus<Src, Snk, Clk>(
    source: &mut Src,
    transforms: Vec<&mut dyn DynAsyncElement>,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: &BusHandle,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_linear_chain_inner(source, transforms, sink, clock, link_capacity, Some(bus)).await
}

#[cfg(feature = "std")]
async fn run_linear_chain_inner<Src, Snk, Clk>(
    source: &mut Src,
    transforms: Vec<&mut dyn DynAsyncElement>,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    // D5: thin builder over the DAG runner. A linear chain maps onto a
    // source -> transform* -> sink path; `run_graph` owns negotiation, the M12
    // stat folds, the β allocation re-cascade, and the Caps-α mid-stream
    // re-solve (graceful on this single-producer chain: no tee upstream).
    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let mut prev = g.add_source(GraphNodeRef::source_ref(source));
    for t in transforms {
        let node = g.add_transform(GraphNodeRef::element_ref(t));
        g.link(prev, node).map_err(|_| G2gError::CapsMismatch)?;
        prev = node;
    }
    let snk = g.add_sink(GraphNodeRef::element_ref(sink));
    g.link(prev, snk).map_err(|_| G2gError::CapsMismatch)?;

    run_graph_inner(g, clock, link_capacity, bus, None, None, None, None).await
}

/// Sentinel sink for terminal elements (sinks proper): swallows pushes.
/// Process implementations of true sinks should not emit, but the type
/// system still requires an `&mut dyn OutputSink` parameter.
#[derive(Debug)]
pub(crate) struct NullSink;

impl OutputSink for NullSink {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

/// Drives a `source → transform → sink` pipeline over two bounded links.
///
/// Transform contract: `process(Eos)` may flush buffered state as
/// `DataFrame` packets but MUST NOT emit `Eos` itself — the runner forwards
/// the EOS sentinel downstream after `process(Eos)` returns.
///
/// `link_capacity` is the primary glass-to-glass latency knob. Under
/// steady-state backpressure each link sits full, so the latency floor is
/// roughly `2 * link_capacity * consumer_period`. For live video pipelines
/// (RTSP -> decode -> display) prefer **2**; for batch / throughput-oriented
/// workloads larger values are fine.
pub async fn run_source_transform_sink<Src, Tx, Snk, Clk>(
    source: &mut Src,
    transform: &mut Tx,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: AsyncElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_source_transform_sink_inner(source, transform, sink, clock, link_capacity, None).await
}

/// As [`run_source_transform_sink`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed)
/// to `bus` so the application learns *which* link conflicted (the returned
/// error stays the opaque `CapsMismatch`). M18 item 7. Covers both startup
/// negotiation and the mid-stream re-solve sites (the sink's Phase-B re-solve
/// and the transform's Caps-α `Infeasible`). The bus is opt-in to keep the
/// common call site unchanged.
pub async fn run_source_transform_sink_with_bus<Src, Tx, Snk, Clk>(
    source: &mut Src,
    transform: &mut Tx,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: &BusHandle,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: AsyncElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    run_source_transform_sink_inner(source, transform, sink, clock, link_capacity, Some(bus)).await
}

async fn run_source_transform_sink_inner<Src, Tx, Snk, Clk>(
    source: &mut Src,
    transform: &mut Tx,
    sink: &mut Snk,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: AsyncElement,
    Snk: AsyncElement,
    Clk: PipelineClock,
{
    let link_capacity: usize = link_capacity.into().get();
    // M18 Session C: the startup negotiation loop (solver + per-link
    // configure cascade with bounded `ReFixate` retry) is owned by the
    // coordinator module now, since β reuses the same machinery for the
    // mid-stream re-cascade. `sink_link` is the downstream-facing caps
    // (transform output = sink input) that M12 allocation flows along,
    // so it stands in for the loop's former `negotiated_caps`.
    // M12 allocation query now runs *inside* negotiation, before the
    // `configure_pipeline` cascade, so a transform (e.g. a hardware decoder)
    // sizes its buffer pool from the downstream `min_buffers` at open time.
    // The folded source-facing proposal comes back on `RunStats`.
    let negotiation = negotiate_source_transform_sink(source, transform, sink, bus).await?;
    let allocation = negotiation.allocation;

    // Caps-α: the transform's downstream subgraph is the single sink link, so
    // its feasibility snapshot is just the sink's accept set (the N-hop sweep
    // in `run_linear_chain` reduces to this for one transform). A wildcard or
    // legacy sink leaves it unconstrained, so the transform keeps forwarding
    // greedily (Defer).
    let downstream_feasible = match sink.caps_constraint_as_sink() {
        CapsConstraint::Accepts(s) => Some(s.clone()),
        _ => None,
    };

    // M12 latency query: fold the configured chain source → transform → sink.
    let latency = LatencyReport::aggregate([
        source.latency(),
        AsyncElement::latency(transform),
        AsyncElement::latency(sink),
    ]);

    // M12 clock distribution: elect the pipeline clock from any element that
    // offers one (live source > provider > system fallback) and read its epoch.
    let elected = elect_clock([
        source.provide_clock(),
        AsyncElement::provide_clock(transform),
        AsyncElement::provide_clock(sink),
    ]);
    let (clock_priority, base_time_ns) = match &elected {
        Some(c) => (c.priority, c.clock.now_ns()),
        None => (ClockPriority::SystemFallback, clock.now_ns()),
    };

    // Hand the elected clock + base time to the sink so it can present each
    // frame at its running-time deadline (PTS pacing). Only when a clock was
    // elected; without one the sink presents as fast as backpressure allows.
    if let Some(c) = &elected {
        AsyncElement::set_clock_sync(sink, ClockSync::new(c.clock.clone(), base_time_ns));
    }

    let (link1_tx, link1_rx) = link(link_capacity);
    let (link2_tx, link2_rx) = link(link_capacity);

    // M18 β: a single coordinator task owns the cross-element re-cascade.
    // The sink arm reports an applied mid-stream `CapsChanged` (with its
    // re-derived allocation proposal) out-of-band
    // (DESIGN.md §4.13.5); the coordinator
    // forwards the proposal one hop upstream over `transform_ctrl_rx` to the
    // transform's `configure_allocation`. The transform arm selects on that
    // control receiver alongside its data link, so the directive reaches it
    // even while it is parked on `recv().await`. When the sink arm finishes,
    // the handle drops, the coordinator drains and closes `transform_ctrl_rx`,
    // and the transform arm's EOS-drain unblocks.
    let (coord, coord_handle, transform_ctrl_rx) = coordinator_with_recascade(link_capacity);

    // M399: measured per-element telemetry for the two interior elements; each
    // arm writes its own probe, the runner snapshots them once both have joined.
    let transform_probe =
        ElementProbe::new(alloc::string::String::from(crate::log::short_type_name::<Tx>()));
    let sink_probe =
        ElementProbe::new(alloc::string::String::from(crate::log::short_type_name::<Snk>()));
    let probe_for_transform = transform_probe.clone();
    let probe_for_sink = sink_probe.clone();

    let source_fut = async move {
        let mut adapter = SenderSink::new(link1_tx);
        // M81: unlike `run_simple_pipeline` and `run_graph`, this bespoke
        // 3-element runner does NOT emit an opening SEGMENT. Prepending a packet
        // here can exactly fill a link feeding a buffering transform and trip a
        // shutdown race in this hand-rolled data plane (a latent exact-capacity
        // fragility, tracked separately). The opening SEGMENT lands once this
        // runner is re-expressed as a thin builder over `run_graph` (as
        // `run_linear_chain` already is). Use `run_graph` / `run_linear_chain`
        // for the SEGMENT-emitting path.
        source.run(&mut adapter).await
    };

    let bus_for_transform = bus.cloned();
    let transform_fut = async move {
        let ctrl_rx = transform_ctrl_rx;
        let probe_for_transform = probe_for_transform;
        let mut adapter = SenderSink::new(link2_tx);
        // M175: relay a QoS report from the sink (seen on the transform's output
        // link) onto the transform's input link, so the source observes it as
        // `PushOutcome::Qos` and sheds load. Without this the report dies at the
        // transform (its `process` push outcome is discarded).
        adapter.relay_qos_to(link1_rx.qos_slot());
        // β: while the coordinator is alive, race the data link against the
        // re-cascade control channel so a directive is applied promptly. Once
        // control closes (coordinator gone) we degrade to data-only so the
        // closed arm can't spin.
        let mut control_open = true;
        loop {
            let packet = if control_open {
                match select2(ctrl_rx.recv(), link1_rx.recv()).await {
                    Either::Left(Some(ArmDirective::Recascade(params))) => {
                        // β: apply the sink's downstream-derived proposal to
                        // our own output pool, then keep waiting for data.
                        transform.configure_allocation(&params);
                        continue;
                    }
                    Either::Left(None) => {
                        control_open = false;
                        continue;
                    }
                    Either::Right(packet) => packet,
                }
            } else {
                link1_rx.recv().await
            };
            match packet {
                Some(PipelinePacket::Eos) => {
                    transform.process(PipelinePacket::Eos, &mut adapter).await?;
                    adapter.push(PipelinePacket::Eos).await?;
                    // β: the EOS we just forwarded will, once the sink applies
                    // its final `CapsChanged` and the coordinator forwards the
                    // matching re-cascade, close this control channel. Drain it
                    // first so a tail-end proposal is applied before we exit
                    // (in a live stream these apply inline above; this only
                    // covers the directive still in flight at shutdown).
                    while control_open {
                        match ctrl_rx.recv().await {
                            Some(ArmDirective::Recascade(params)) => {
                                transform.configure_allocation(&params);
                            }
                            None => control_open = false,
                        }
                    }
                    return Ok::<(), G2gError>(());
                }
                Some(PipelinePacket::CapsChanged(new_caps)) => {
                    // Caps-α (D3): derive the forwarded output from the
                    // transform's constraint, steered by the sink's accept set,
                    // instead of forwarding greedily. Mirrors `run_linear_chain`;
                    // here the downstream subgraph is the single sink link.
                    // `Infeasible` means the sink positively rejects every
                    // output the transform can produce: surface it loud as a
                    // reverse reconfigure into this boundary.
                    let forward_caps = {
                        let constraint = transform.caps_constraint_as_transform();
                        match resolve_forward_output(
                            &constraint,
                            &new_caps,
                            downstream_feasible.as_ref(),
                        ) {
                            ForwardResolve::Fixed(caps) => caps,
                            ForwardResolve::Defer => new_caps.clone(),
                            ForwardResolve::Infeasible(failure) => {
                                report_nego_failure(bus_for_transform.as_ref(), failure);
                                link1_rx.request_reconfigure(Reconfigure::Renegotiate);
                                continue;
                            }
                        }
                    };
                    match transform.configure_pipeline(&new_caps)? {
                        ConfigureOutcome::Accepted => {
                            // M188: a caps-driven transform re-resolves its output
                            // target on the mid-stream change too, not just at
                            // startup, so a videoscale/videoconvert fed by a
                            // downstream capsfilter retargets when caps shift.
                            // No-op for property-driven / passthrough elements.
                            AsyncElement::configure_output(transform, &forward_caps)?;
                            // M18 α: element-local re-allocation under the
                            // re-fixated output caps before forwarding.
                            realloc_local(transform, &forward_caps);
                            transform
                                .process(
                                    PipelinePacket::CapsChanged(forward_caps),
                                    &mut adapter,
                                )
                                .await?;
                        }
                        // Mid-stream ReFixate: fire upstream via this
                        // element's input link, drop the rejected
                        // CapsChanged. Piece 4 will source-side react.
                        ConfigureOutcome::ReFixate(counter) => {
                            link1_rx.request_reconfigure(Reconfigure::Propose(counter));
                        }
                    }
                }
                Some(packet) => {
                    // M399: time the data-frame `process()` and sample input fill.
                    let timed = matches!(&packet, PipelinePacket::DataFrame(_))
                        .then(|| &*probe_for_transform);
                    if let Some(p) = timed {
                        p.record_fill(link1_rx.fill_percent());
                    }
                    let t0 = ElementProbe::mark();
                    transform.process(packet, &mut adapter).await?;
                    if let Some(p) = timed {
                        p.record_proc_since(t0);
                    }
                }
                None => return Ok(()),
            }
        }
    };

    let bus_for_sink = bus.cloned();
    let sink_fut = async move {
        let coord_handle = coord_handle;
        let bus_for_sink = bus_for_sink;
        let probe_for_sink = probe_for_sink;
        let mut null = NullSink;
        let mut consumed: u64 = 0;
        loop {
            match link2_rx.recv().await {
                Some(PipelinePacket::Eos) => {
                    sink.process(PipelinePacket::Eos, &mut null).await?;
                    return Ok::<u64, G2gError>(consumed);
                }
                Some(PipelinePacket::CapsChanged(new_caps)) => {
                    // M16 workaround #3 Phase B: re-solve the downstream
                    // subgraph (boundary → sink) before applying. The
                    // boundary that emitted this `CapsChanged` is the
                    // transform; the subgraph is the one link feeding
                    // the sink. A `NegotiationFailure` here means the
                    // sink's declared `CapsConstraint` rejects the
                    // boundary's output — surface it as a reverse
                    // Reconfigure into the transform (§7 forward ×
                    // reverse race: terminates at the boundary, does
                    // not propagate past the source).
                    let sink_caps = match re_solve_downstream_sink(&new_caps, &*sink) {
                        Ok(caps) => caps,
                        Err(failure) => {
                            // M18 item 7: the mid-stream re-solve is the case
                            // the bus matters most for, there is no synchronous
                            // return to carry the detail. Post the structured
                            // failure, then drive the reverse Reconfigure.
                            report_nego_failure(bus_for_sink.as_ref(), failure);
                            link2_rx
                                .request_reconfigure(Reconfigure::Renegotiate);
                            continue;
                        }
                    };
                    match sink.configure_pipeline(&sink_caps)? {
                        ConfigureOutcome::Accepted => {
                            // M18 α: element-local re-allocation under the
                            // new caps before the sink sees the packet. The
                            // returned proposal is what the sink now wants
                            // its upstream to allocate.
                            let proposal = realloc_local(sink, &sink_caps);
                            // M18 β: report the applied caps change plus the
                            // sink's re-derived proposal so the coordinator
                            // forwards it one hop upstream to the transform's
                            // `configure_allocation` (the single-hop cascade).
                            coord_handle
                                .report(CoordinatorEvent::CapsChanged {
                                    caps: sink_caps.clone(),
                                    proposal,
                                })
                                .await;
                            sink.process(
                                PipelinePacket::CapsChanged(sink_caps),
                                &mut null,
                            )
                            .await?;
                        }
                        ConfigureOutcome::ReFixate(counter) => {
                            link2_rx.request_reconfigure(Reconfigure::Propose(counter));
                        }
                    }
                }
                Some(packet) => {
                    let is_buffer = matches!(packet, PipelinePacket::DataFrame(_));
                    if is_buffer {
                        consumed += 1;
                    }
                    // M399: time the data-frame `process()` and sample input fill.
                    let timed = is_buffer.then(|| &*probe_for_sink);
                    if let Some(p) = timed {
                        p.record_fill(link2_rx.fill_percent());
                    }
                    let t0 = ElementProbe::mark();
                    sink.process(packet, &mut null).await?;
                    if let Some(p) = timed {
                        p.record_proc_since(t0);
                    }
                    // M175 upstream QoS: a late sink stores its report on the
                    // link feeding it; the transform's output adapter relays it
                    // one hop further upstream (see `relay_qos_to` above).
                    if let Some(qos) = sink.take_qos() {
                        link2_rx.request_qos(qos);
                    }
                    // Keyframe-request / renegotiation up the reverse channel; the
                    // transform's output adapter relays it one hop toward the encoder.
                    if let Some(reconf) = sink.take_reconfigure() {
                        link2_rx.request_reconfigure(reconf);
                    }
                    if let Some(bps) = sink.take_bitrate() {
                        link2_rx.request_bitrate(bps);
                    }
                }
                None => return Ok(consumed),
            }
        }
    };

    // The coordinator task drains the control channel until the sink arm
    // drops its handle. Joined as a fourth arm so it runs concurrently.
    let coordinator_fut = coord.run();

    let (src_res, (tx_res, (snk_res, coordinator_events))) = Join2::new(
        source_fut,
        Join2::new(transform_fut, Join2::new(sink_fut, coordinator_fut)),
    )
    .await;
    // M81: prefer a substantive error over a secondary `Shutdown`. A real error
    // in the transform or sink closes a link, which can surface as `Shutdown` on
    // the source arm (checked first); without this, that masks the real cause.
    if let Some(e) =
        substantive_error([src_res.as_ref().err(), tx_res.as_ref().err(), snk_res.as_ref().err()])
    {
        return Err(e);
    }
    let emitted = src_res?;
    tx_res?;
    let consumed = snk_res?;

    // M399: both arms have joined; snapshot the transform and sink probes in
    // topological order (source carries no `process()` and is omitted).
    let per_element = alloc::vec![transform_probe.snapshot(), sink_probe.snapshot()];
    Ok(RunStats {
        frames_emitted: emitted,
        frames_consumed: consumed,
        frames_dropped: 0,
        latency,
        allocation,
        clock_priority,
        base_time_ns,
        coordinator_events,
        per_element,
    })
}

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn live_profile_maps_to_capacity_2() {
        assert_eq!(LatencyProfile::Live.link_capacity().get(), 2);
    }

    #[test]
    fn run_stats_report_formats_drops_and_latency() {
        let stats = RunStats {
            frames_emitted: 100,
            frames_consumed: 90,
            frames_dropped: 10,
            latency: LatencyReport { live: true, min_ns: 5_000_000, max_ns: Some(20_000_000) },
            ..RunStats::default()
        };
        let r = stats.report();
        // Frame line with the computed drop rate (10 / (90 + 10) = 10%).
        assert!(r.contains("emitted 100, consumed 90, dropped 10 (10.0% drop)"), "{r}");
        // Declared latency window + live flag.
        assert!(r.contains("5.0 ms .. 20.0 ms (live) [declared]"), "{r}");
        assert!(r.contains("clock:"), "{r}");

        // An unbounded-latency, lossless pipeline reads cleanly too.
        let clean = RunStats {
            frames_emitted: 5,
            frames_consumed: 5,
            latency: LatencyReport { live: false, min_ns: 0, max_ns: None },
            ..RunStats::default()
        };
        let r = clean.report();
        assert!(r.contains("(0.0% drop)"), "{r}");
        assert!(r.contains("0.0 ms .. unbounded (non-live)"), "{r}");
    }

    #[test]
    fn throughput_profile_maps_to_capacity_8() {
        assert_eq!(LatencyProfile::Throughput.link_capacity().get(), 8);
    }

    #[test]
    fn custom_profile_passes_through() {
        assert_eq!(LatencyProfile::Custom(16).link_capacity().get(), 16);
    }

    #[test]
    fn link_capacity_clamps_zero_to_one() {
        // A zero-depth link would deadlock the producer on its first push;
        // the constructor clamps so callers passing `0` (or a misconfigured
        // env var) get a runnable pipeline rather than a hang.
        assert_eq!(LinkCapacity::new(0).get(), 1);
        assert_eq!(LinkCapacity::from(0usize).get(), 1);
        assert_eq!(LatencyProfile::Custom(0).link_capacity().get(), 1);
    }

    #[test]
    fn from_usize_and_from_profile_compose_through_into() {
        // The runner takes `impl Into<LinkCapacity>`; both an integer and
        // a profile must reach the same internal usize without ceremony.
        fn take<C: Into<LinkCapacity>>(c: C) -> usize {
            c.into().get()
        }
        assert_eq!(take(4usize), 4);
        assert_eq!(take(LatencyProfile::Live), 2);
        assert_eq!(take(LatencyProfile::Throughput), 8);
    }
}
