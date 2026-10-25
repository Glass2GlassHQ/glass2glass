#[cfg(feature = "std")]
use alloc::boxed::Box;
use core::future::Future;

use crate::bus::BusHandle;
use crate::caps::{Caps, CapsSet};
#[cfg(feature = "std")]
use crate::clock::{elect_clock, PipelineClock};
use crate::clock::{ClockCandidate, ClockPriority, ClockSync};
#[cfg(feature = "std")]
use crate::element::{AsyncElement, BoxFuture, OutputSinkExt, PushOutcome};
use crate::element::{ConfigureOutcome, ElementBound, OutputSink, Reconfigure};
use crate::error::G2gError;
use crate::format_element::{CapsConstraint, CapsPreferences};
#[cfg(feature = "std")]
use crate::frame::PipelinePacket;
#[cfg(feature = "std")]
use crate::log::ElementLogName;
use crate::memory::{DomainSet, MemoryDomainKind};
use crate::property::{ElementMetadata, PropError, PropValue, PropertySpec};
use crate::query::{AllocationParams, LatencyReport};
#[cfg(feature = "std")]
use crate::runtime::channel::{advertise_orientation, link, SenderSink};
#[cfg(feature = "std")]
use crate::runtime::coordinator::solve_last_link;
#[cfg(feature = "std")]
use crate::runtime::instrument::ElementProbe;
#[cfg(feature = "std")]
use crate::runtime::join::{select2, Either, Join2};
use crate::runtime::seek::{SeekController, SeekFormat};
#[cfg(feature = "std")]
use crate::runtime::solver::{solve_linear, NegotiationFailure};
#[cfg(feature = "std")]
use crate::runtime::state::StateController;
#[cfg(feature = "std")]
use crate::segment::Segment;

/// Startup negotiation passes before a runner gives up with `FixationFailed`:
/// enough for a source, sink, source counter-proposal chain.
#[cfg(feature = "std")]
pub(crate) const MAX_FIXATION_ATTEMPTS: u32 = 3;

/// Pick the most informative error from a pipeline's arm results. A closed-link
/// `Shutdown` is usually the *consequence* of another arm erroring first (it
/// dropped its channel end), so prefer any non-`Shutdown` error over it; fall
/// back to the first error otherwise. `None` if every arm succeeded.
///
/// Each result is paired with its arm's instance name, and the one that wins is
/// logged naming that element, since `G2gError` carries no element identity.
#[cfg(feature = "std")]
fn substantive_error<'a, I>(results: I) -> Option<G2gError>
where
    I: IntoIterator<Item = (&'a str, Option<&'a G2gError>)>,
{
    let mut first: Option<(&str, &G2gError)> = None;
    for (name, e) in results {
        let Some(e) = e else { continue };
        if *e != G2gError::Shutdown {
            crate::log::report_element_failure(Some(name), e);
            return Some(e.clone());
        }
        first.get_or_insert((name, e));
    }
    let (name, e) = first?;
    crate::log::report_element_failure(Some(name), e);
    Some(e.clone())
}

#[cfg(feature = "std")]
use crate::element::DynAsyncElement;
#[cfg(feature = "std")]
use crate::fanout::{MultiOutputElement, MultiOutputSource};
#[cfg(feature = "std")]
use crate::graph::Graph;
#[cfg(feature = "std")]
use crate::runtime::channel::{bounded, Sender};
#[cfg(feature = "std")]
use crate::runtime::graph_runner::{
    broadcast, run_graph_inner, run_source, sink_arm_on, GraphNodeRef, SinkArmIo, SourceArmIo,
};
#[cfg(feature = "std")]
use crate::runtime::join::dynamic_join;
#[cfg(feature = "std")]
use crate::runtime::observe::{link_tapped, register_runner_tap};
#[cfg(feature = "std")]
use crate::runtime::{NodeRole, Observer, Probe};
#[cfg(feature = "std")]
use alloc::sync::Arc;
#[cfg(feature = "std")]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use spin::Mutex;

/// Source-side element trait. Sources have no input pad, so the packet-in /
/// packet-out shape of [`AsyncElement`] does not fit them. A `SourceLoop`
/// instead receives a single `run` call that iterates internally until EOS
/// and returns the count of `DataFrame` packets pushed.
pub trait SourceLoop: ElementBound {
    type RunFuture<'a>: Future<Output = Result<u64, G2gError>> + 'a
    where
        Self: 'a;

    /// Future returned by [`SourceLoop::intercept_caps`]. Async so a source can perform
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

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError>;

    /// Runs the source until EOS or error. The implementation MUST emit a
    /// final `PipelinePacket::Eos` before returning `Ok`. Returns the number
    /// of `DataFrame` packets pushed (excluding `Eos`).
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a>;

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

    /// This source's latency contribution to the pipeline latency query.
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

    /// The full set of memory domains this source can emit. The
    /// producer-capability half of the two-sided allocation-domain negotiation;
    /// see [`AsyncElement::output_domains`](crate::element::AsyncElement::output_domains).
    /// Default: just [`output_memory`](Self::output_memory). A GPU capture source
    /// that can also deliver to System overrides this.
    fn output_domains(&self) -> DomainSet {
        DomainSet::only(self.output_memory())
    }

    /// The total stream duration in nanoseconds, the source's answer to the
    /// application `DURATION` query. A source that knows the total length
    /// (a file / container source after reading its header, e.g. `Mp4Src`)
    /// overrides this; the runner publishes it on the
    /// [`PipelineProgress`](crate::runtime::PipelineProgress) handle and posts a
    /// [`DurationChanged`](crate::BusMessage::DurationChanged). The default
    /// `None` is "unknown" (a live / open-ended source, or one whose length is
    /// not yet parsed). Polled by the runner just before [`run`](Self::run).
    fn query_duration(&self) -> Option<u64> {
        None
    }

    /// Receive the downstream peer's allocation proposal so the source
    /// can allocate its output `BufferPool` from compatible parameters
    /// (size, count, alignment, domain). Default: ignore and allocate the
    /// source's own way. The proposal is advisory; a source that cannot honor
    /// it (eg cannot produce the requested domain) falls back silently.
    fn configure_allocation(&mut self, _params: &AllocationParams) {}

    /// Offer a clock to the pipeline's clock election. Default: none.
    /// Live capture sources override this to provide their hardware capture
    /// clock at [`ClockPriority::LiveSource`](crate::ClockPriority::LiveSource)
    /// so the pipeline paces to capture cadence.
    fn provide_clock(&self) -> Option<ClockCandidate> {
        None
    }

    /// Receive the pipeline's elected clock + base time, so a live capture
    /// source can stamp running time rather than its own zero-based timeline
    /// (see [`CaptureAnchor`](crate::clock::CaptureAnchor)). Called once before
    /// [`run`](Self::run), only when a clock was elected. Default: ignore, which
    /// stamps from the source's own zero.
    fn set_clock_sync(&mut self, _sync: ClockSync) {}

    /// Declare this source's negotiation-time constraint.
    /// Default: eagerly await `intercept_caps()` and produce that one caps.
    /// A source offering alternatives (a camera's pixel formats) overrides
    /// this to return the whole `Produces(CapsSet)`.
    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        async move {
            Ok(CapsConstraint::Produces(CapsSet::one(
                self.intercept_caps().await?,
            )))
        }
    }

    /// What this source is willing to pay for each alternative of the produce
    /// set its [`caps_constraint`](Self::caps_constraint) advertises. Default
    /// `None`: the alternatives are already in preference order and cost their
    /// index. A source overrides this to declare *equal* cost between formats it
    /// does not care about (so a downstream element's preference decides) or a
    /// gap wide enough that a downstream preference cannot pull the chain onto
    /// its fallback.
    fn caps_preferences(&self) -> Option<CapsPreferences> {
        None
    }

    /// The fixed output caps this source already knows from its properties,
    /// readable synchronously without negotiation or I/O. The auto-plug
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
    /// to do I/O to determine the type: the auto-plug `decodebin` parser
    /// calls this once, at parse time, to pick the demuxer. A `bytestream-format=
    /// auto` source overrides it to sniff the file header now (so a mislabeled
    /// `.ts` that is really an MP4 still auto-plugs the right demuxer, the way
    /// GStreamer's runtime `typefind` would), where `configured_output_caps`
    /// returns `None` because it may not read. `Err` when the source read its
    /// content and recognized no media type. Default: the no-I/O caps.
    fn probe_output_caps(&mut self) -> Result<Option<Caps>, G2gError> {
        Ok(self.configured_output_caps())
    }

    /// The runtime properties this source type exposes, the GObject
    /// property-spec analog. Default: none. A source overrides this (and
    /// [`set_property`](Self::set_property) / [`get_property`](Self::get_property))
    /// to be settable by name from a `gst-launch` pipeline (eg `filesrc
    /// location=...`, `videotestsrc pattern=...`).
    fn properties(&self) -> &'static [PropertySpec] {
        &[]
    }

    /// Static introspection metadata for this source type, the
    /// `gst-inspect` "Factory Details". Default: empty.
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::default()
    }

    /// Receive this source instance's log name, assigned by the runner.
    /// Default: ignore.
    fn set_instance_name(&mut self, _name: alloc::string::String) {}

    /// Receive the run's message bus, so a source can report out of band
    /// what a property read cannot answer for it (its restart status, say): a
    /// source arm owns its element for the whole run. Every runner built on
    /// the graph runner hands it out when given a bus. Default: ignore.
    fn set_bus(&mut self, _bus: BusHandle) {}

    // the unit of the positions this source repositions to, `None` when it cannot seek
    fn seek_format(&self) -> Option<SeekFormat> {
        None
    }

    // called only on a source whose `seek_format` is `Some`
    fn set_seek(&mut self, _controller: SeekController) {}

    /// Override this instance's log category, mirroring
    /// [`AsyncElement::set_log_category`](crate::AsyncElement::set_log_category).
    /// Default: ignore. A source that logs about itself stores it in a
    /// [`LogName`](crate::log::LogName) and returns it from
    /// `LogSource::log_category_override`.
    fn set_log_category(&mut self, _category: alloc::string::String) {}

    /// Set a property by name. Default: [`PropError::Unknown`].
    fn set_property(&mut self, _name: &str, _value: PropValue) -> Result<(), PropError> {
        Err(PropError::Unknown)
    }

    /// Read a property back by name. Default: `None`.
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
    /// Aggregated source-to-sink latency, computed once after
    /// negotiation from every element's `latency()`. The hand-built fan-in and
    /// dynamic runners leave this at `ZERO`.
    pub latency: LatencyReport,
    /// The allocation proposal handed to the head producer (the source),
    /// negotiated from downstream. `None` when no downstream element proposed
    /// one; always `None` for the hand-built fan-in and dynamic runners.
    pub allocation: Option<AllocationParams>,
    /// Priority of the clock the pipeline elected. `SystemFallback` when
    /// no element provided one (the supplied clock stands); always
    /// `SystemFallback` for the hand-built fan-in and dynamic runners.
    pub clock_priority: ClockPriority,
    /// `now_ns()` of the elected clock, read once after election — the
    /// pipeline's base-time origin.
    pub base_time_ns: u64,
    /// Scaffolding: number of `CoordinatorEvent`s the coordinator
    /// task observed over this run's control channel. Today the only
    /// event is a boundary forwarding a mid-stream `CapsChanged` the next
    /// element accepted; β will turn each into a `Recascade`. `0` for
    /// runners that don't spawn a coordinator (simple / fan-in / dynamic).
    pub coordinator_events: u64,
    /// Measured per-element telemetry: each interior element's `process()`
    /// latency distribution (p50/p99) and input-link fill, in topological order.
    /// Populated by `run_graph` and the two linear runners
    /// (`run_simple_pipeline` / `run_source_transform_sink`); empty for the
    /// fan-in / fan-out / session / muxer runners and under `no_std` (no clock to
    /// measure with). Sources carry no `process()` so they do not appear; their
    /// cost surfaces as the downstream element's input fill.
    pub per_element: alloc::vec::Vec<crate::runtime::ElementLatency>,
}

impl RunStats {
    /// A human-readable end-of-run summary of the pipeline telemetry:
    /// frame counts + drop rate, the aggregated declared latency window, the
    /// elected clock, and the negotiated head allocation. `g2g-launch` prints
    /// this at end (and a host can log it). The latency is the chain's
    /// *declared* min/max fold (each element's `latency()`), not a measured
    /// runtime histogram; per-element / per-link p50/p99 is a follow-up.
    pub fn report(&self) -> alloc::string::String {
        use alloc::format;
        let ms = |ns: u64| ns as f64 / 1.0e6;
        let seen = self.frames_consumed + self.frames_dropped;
        let drop_pct = if seen > 0 {
            self.frames_dropped as f64 * 100.0 / seen as f64
        } else {
            0.0
        };

        let mut s = alloc::string::String::from("pipeline run summary:\n");
        s.push_str(&format!(
            "  frames:  emitted {}, consumed {}, dropped on links {} ({drop_pct:.1}% drop)\n",
            self.frames_emitted, self.frames_consumed, self.frames_dropped
        ));
        let max = match self.latency.max_ns {
            Some(m) => format!("{:.1} ms", ms(m)),
            None => alloc::string::String::from("unbounded"),
        };
        s.push_str(&format!(
            "  latency: {:.1} ms .. {max} ({}) [declared]\n",
            ms(self.latency.min_ns),
            if self.latency.live {
                "live"
            } else {
                "non-live"
            },
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
                    // Include measured input-link queue-residency (transit) when
                    // the edge was instrumented; it is the "wait" half of the
                    // per-stage latency, complementing the `process()` "work".
                    let transit = if e.transit.count > 0 {
                        format!(
                            ", wait p50 {:.2} ms / p99 {:.2} ms",
                            ms(e.transit.p50_ns),
                            ms(e.transit.p99_ns)
                        )
                    } else {
                        alloc::string::String::new()
                    };
                    // M947: what the element spent parked on a full output link.
                    // `proc` excludes it, so a back-pressured element reads as
                    // blocked rather than as expensive. Absent for an element the
                    // runner does not attribute output pushes to (every sink).
                    let blocked = if e.push_wait.max_ns > 0 {
                        format!(
                            ", blocked p50 {:.2} ms / p99 {:.2} ms",
                            ms(e.push_wait.p50_ns),
                            ms(e.push_wait.p99_ns)
                        )
                    } else {
                        alloc::string::String::new()
                    };
                    // Frame age at this element's output push: the number that
                    // exposes internal buffering (a decoder's reorder queue),
                    // which cheap `proc` percentiles hide.
                    let age = if e.age_at_emit.count > 0 {
                        format!(
                            ", age-out p50 {:.2} ms / p99 {:.2} ms",
                            ms(e.age_at_emit.p50_ns),
                            ms(e.age_at_emit.p99_ns)
                        )
                    } else {
                        alloc::string::String::new()
                    };
                    s.push_str(&format!(
                        "    {:<16} proc p50 {:.2} ms / p99 {:.2} ms (n={}){blocked}{transit}{age}, in-fill {}%/{}% avg/max\n",
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
        // What a paced sink actually put on the display, vs what it shed: a
        // stalling presentation shows up here even when `consumed` looks healthy.
        for e in &self.per_element {
            if let Some(p) = e.presentation {
                s.push_str(&format!(
                    "  present: {:<16} {} presented, {} dropped, {} late-dropped\n",
                    e.name, p.presented, p.dropped, p.late_dropped
                ));
            }
        }
        s
    }
}

/// Re-solve a mid-stream `CapsChanged` against a sink's declared constraint
/// before the sink is configured with it (per branch for fan-out
/// FO-2). An `Unfixable` result (a ranged field like `Rate::Any` left open) is
/// not a failure: the sink accepted the shape, so `new_caps` passes through.
#[cfg(feature = "std")]
pub(crate) fn re_solve_downstream_dyn_sink(
    new_caps: &Caps,
    sink: &dyn DynAsyncElement,
) -> Result<Caps, NegotiationFailure> {
    let src_c = CapsConstraint::Produces(CapsSet::one(new_caps.clone()));
    match solve_linear(&[&src_c, &sink.caps_constraint_as_sink()]) {
        Ok(links) => links
            .into_iter()
            .last()
            .ok_or(NegotiationFailure::Degenerate),
        Err(NegotiationFailure::Unfixable { .. }) => Ok(new_caps.clone()),
        Err(other) => Err(other),
    }
}

/// Drives a `source → sink` pipeline over a single bounded link.
/// Initial Phase 1+2 negotiation runs with bounded `ReFixate` backtrack
/// (M8 piece 5): if any element's `configure_pipeline()` returns a
/// counter-proposal, the runner restarts negotiation with that counter
/// as the new starting proposal, up to `MAX_FIXATION_ATTEMPTS` total.
#[cfg(feature = "std")]
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

/// As [`run_simple_pipeline`], but driven by a [`StateController`].
///
/// The sink arm gates on the controller: while the state is below
/// `Playing` the sink stops pulling, the bounded link fills, and backpressure
/// stalls the source. `set_state(Playing)` (from another task) opens the gate
/// and data flows; `set_state(Null)` stops the sink arm and ends the run.
/// Negotiation still runs eagerly at startup (it is resource acquisition, the
/// `READY` step); only data flow is gated. Pass the controller starting in
/// `Paused` for the common "build prerolled, then play" shape.
#[cfg(feature = "std")]
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
    run_simple_pipeline_inner(
        source,
        sink,
        clock,
        link_capacity,
        None,
        Some(state.clone()),
    )
    .await
}

/// As [`run_simple_pipeline`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed) to
/// `bus` on a startup or mid-stream negotiation failure.
#[cfg(feature = "std")]
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

#[cfg(feature = "std")]
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
    // M842: name and log the instances the way `run_graph` does, before
    // negotiation, so an element driven by this runner logs under `<category>N`
    // too. The sink's name also keys its measured-latency probe below.
    let mut namer = crate::log::InstanceNamer::new();
    let source_category = crate::log::short_type_name::<Src>();
    let source_name = namer.add(source_category, None);
    SourceLoop::set_instance_name(source, source_name.clone());
    let sink_category = crate::log::short_type_name::<Snk>();
    let sink_name = namer.add(sink_category, None);
    AsyncElement::set_instance_name(sink, sink_name.clone());
    // M16 step 5f: startup negotiation honors `SourceLoop::caps_constraint`.
    // A `ReFixate` retry negotiates the one counter-proposed caps instead.
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
                Some(c) => CapsConstraint::Produces(CapsSet::one(c.clone())),
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
    // M1123: hand the fold's liveness back down, so an element that tunes for
    // throughput only when nothing is pacing the stream knows which it is.
    AsyncElement::configure_liveness(sink, latency.live);

    // M12 allocation query: the sink proposes buffers; the source allocates
    // its output pool to match (zero-copy handoff when it can honor them). M351:
    // the proposed domain is reconciled against what the source can actually emit
    // (a two-sided negotiation), so a multi-domain sink/source pair settles on a
    // shared domain (GPU-preferred) instead of the sink dictating unilaterally;
    // no shared domain is a loud conflict rather than a silent mismatch.
    // M976: the sink's meta requests ride the same proposal, so the source can
    // ask what anyone downstream wants attached before it produces a frame.
    let proposed = crate::query::with_meta_demand(
        sink.propose_allocation(&negotiated_caps),
        AsyncElement::meta_requests(sink),
    );
    // A demand-only proposal accepts every domain, so reconciling it would let a
    // metadata request pick the source's memory domain: pass it through as is.
    let allocation = match proposed {
        Some(p) if p.constrains_pool() => {
            Some(p.resolve_for_producer(SourceLoop::output_domains(source))?)
        }
        other => other,
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
        let sync = sync.with_path_latency(latency);
        // The source gets it too, so a live capture stamps running time rather
        // than its own zero.
        SourceLoop::set_clock_sync(source, sync.clone());
        AsyncElement::set_clock_sync(sink, sync);
    }

    let (link_tx, link_rx) = link(link_capacity);
    advertise_orientation(&link_rx, AsyncElement::absorbed_orientations(sink));

    // M399: measured per-element telemetry for the sink (the linear runner's one
    // interior element with a `process()`); the source's cost surfaces as fill.
    let sink_probe = ElementProbe::new(ElementLogName::new(sink_category, sink_name.clone()));
    let source_io = SourceArmIo {
        out_tx: link_tx,
        bus: None,
        progress: None,
        segment: Segment::new(),
        log_name: ElementLogName::new(source_category, source_name.clone()),
        probe: None,
    };
    let sink_io = SinkArmIo {
        bus: bus.cloned(),
        state,
        ..SinkArmIo::standalone(link_rx, Some(sink_probe.clone()))
    };
    let (src_res, snk_res) =
        Join2::new(run_source(source, source_io), sink_arm_on(sink, sink_io)).await;
    // M81: a closed-link `Shutdown` on the source arm can be the consequence of
    // the sink arm's real error (it dropped the link), so surface the
    // substantive one rather than whichever arm we check first.
    if let Some(e) = substantive_error([
        (source_name.as_str(), src_res.as_ref().err()),
        (sink_name.as_str(), snk_res.as_ref().err()),
    ]) {
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
/// every element with it (design/README.md). This runner has no per-branch
/// caps negotiation, so a sink returning `ReFixate` here fails with `FixationFailed`.
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
    run_source_fanout_inner(source, fanout, sinks, clock, link_capacity, None, None).await
}

/// As [`run_source_fanout`], but taps live telemetry into `observer`, the
/// hand-built analog of
/// [`run_graph_observed`](crate::runtime::run_graph_observed): the topology is
/// the source, the fan-out element, and the N branch sinks, each with its
/// measured `process()` latency and per-link packet / byte / drop counters.
#[cfg(feature = "std")]
pub async fn run_source_fanout_observed<Src, Tx, Clk>(
    source: &mut Src,
    fanout: &mut Tx,
    sinks: Vec<&mut dyn DynAsyncElement>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    observer: &Observer,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: MultiOutputElement,
    Clk: PipelineClock,
{
    run_source_fanout_inner(
        source,
        fanout,
        sinks,
        clock,
        link_capacity,
        None,
        Some(observer),
    )
    .await
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
    run_source_fanout_inner(source, fanout, sinks, clock, link_capacity, Some(bus), None).await
}

#[cfg(feature = "std")]
#[allow(clippy::too_many_arguments)]
async fn run_source_fanout_inner<Src, Tx, Clk>(
    source: &mut Src,
    fanout: &mut Tx,
    sinks: Vec<&mut dyn DynAsyncElement>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    bus: Option<&BusHandle>,
    observer: Option<&Observer>,
) -> Result<RunStats, G2gError>
where
    Src: SourceLoop,
    Tx: MultiOutputElement,
    Clk: PipelineClock,
{
    assert!(!sinks.is_empty(), "fan-out needs at least one sink");
    // A thin builder over the DAG runner: the fan-out element is the graph's
    // demux node, each sink hangs off one of its ports.
    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source_ref(source));
    let port_count = u8::try_from(sinks.len()).map_err(|_| G2gError::CapsMismatch)?;
    let demux = g.add_demux(GraphNodeRef::demux_ref(fanout), port_count);
    // Named after the element type, as a fan-out has no category of its own.
    let fanout_name =
        crate::log::InstanceNamer::new().add(crate::log::short_type_name::<Tx>(), None);
    g.set_node_name(demux.node(), fanout_name);
    g.link(src, demux.input())
        .map_err(|_| G2gError::CapsMismatch)?;
    for (port, sink) in (0..port_count).zip(sinks) {
        let node = g.add_sink(GraphNodeRef::element_ref(sink));
        g.link(demux.out(port), node)
            .map_err(|_| G2gError::CapsMismatch)?;
    }
    run_graph_inner(
        g,
        clock,
        link_capacity,
        bus,
        None,
        None,
        None,
        observer,
        None,
        None,
        None,
        None,
    )
    .await
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

/// A handle to add output branches to a *running* dynamic fan-out: the
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
        self.new_branch_tx
            .try_send(sink)
            .map_err(|_| G2gError::Shutdown)
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
    /// frame's memory is made shareable once, so the per-branch copies
    /// are zero-copy refcount bumps, not byte copies.
    Broadcast,
}

#[cfg(feature = "std")]
impl FanOutMode {
    /// Naming category of the runner's own routing stage, so its telemetry node
    /// reads `router0` / `tee0`.
    fn category(self) -> &'static str {
        match self {
            FanOutMode::Route => "router",
            FanOutMode::Broadcast => "tee",
        }
    }
}

/// Node id of the runner's routing stage in a dynamic fan-out's telemetry
/// topology: the source is 0, and branches append from 2 as they attach.
#[cfg(feature = "std")]
const DYN_FANOUT_NODE: usize = 1;

/// Telemetry bookkeeping the router arm of a dynamic fan-out carries, so a branch
/// attached mid-run gets named, probed, and put in the observer's topology before
/// its first packet. `probes` is shared with the run future, which
/// snapshots it into [`RunStats::per_element`] once the arms have joined.
#[cfg(feature = "std")]
#[derive(Debug)]
struct FanoutTap {
    obs: Option<Observer>,
    namer: crate::log::InstanceNamer,
    probes: Arc<Mutex<Vec<Probe>>>,
}

/// Drives `source -> dynamic router -> N branches`, where branches can be added
/// at runtime through the returned [`DynamicFanoutHandle`] (request pads).
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
) -> (
    DynamicFanoutHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
where
    Src: SourceLoop + 'a,
{
    run_source_fanout_dynamic(source, link_capacity, FanOutMode::Route, None)
}

/// As [`run_source_router_dynamic`], but taps live telemetry into `observer`.
/// The topology starts as the source plus the runner's routing stage;
/// each branch attached through the handle appends its own node and link before
/// it sees a packet, so a dashboard polling [`Observer::snapshot`] sees runtime
/// branches appear with their measured `process()` latency and link counters.
#[cfg(feature = "std")]
pub fn run_source_router_dynamic_observed<'a, Src>(
    source: &'a mut Src,
    link_capacity: impl Into<LinkCapacity>,
    observer: &Observer,
) -> (
    DynamicFanoutHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
where
    Src: SourceLoop + 'a,
{
    run_source_fanout_dynamic(
        source,
        link_capacity,
        FanOutMode::Route,
        Some(observer.clone()),
    )
}

/// Drives `source -> dynamic tee -> N branches`: the broadcast counterpart
/// of [`run_source_router_dynamic`]. Each `DataFrame` is shared to *every*
/// currently-attached branch via the zero-copy frame-sharing path
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
) -> (
    DynamicFanoutHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
where
    Src: SourceLoop + 'a,
{
    run_source_fanout_dynamic(source, link_capacity, FanOutMode::Broadcast, None)
}

/// As [`run_source_tee_dynamic`], but taps live telemetry into `observer`, the
/// broadcast counterpart of [`run_source_router_dynamic_observed`].
#[cfg(feature = "std")]
pub fn run_source_tee_dynamic_observed<'a, Src>(
    source: &'a mut Src,
    link_capacity: impl Into<LinkCapacity>,
    observer: &Observer,
) -> (
    DynamicFanoutHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
where
    Src: SourceLoop + 'a,
{
    run_source_fanout_dynamic(
        source,
        link_capacity,
        FanOutMode::Broadcast,
        Some(observer.clone()),
    )
}

/// Shared driver behind [`run_source_router_dynamic`] (route) and
/// [`run_source_tee_dynamic`] (broadcast); `mode` selects how each `DataFrame` is
/// distributed across the attached branches.
#[cfg(feature = "std")]
fn run_source_fanout_dynamic<'a, Src>(
    source: &'a mut Src,
    link_capacity: impl Into<LinkCapacity>,
    mode: FanOutMode,
    observer: Option<Observer>,
) -> (
    DynamicFanoutHandle<'a>,
    impl Future<Output = Result<RunStats, G2gError>> + 'a,
)
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
        // M869: instance naming as in the static fan-out. The routing stage is the
        // runner itself (there is no fan-out element), so it is named after the
        // distribution mode; each branch is named and probed when it attaches.
        let mut namer = crate::log::InstanceNamer::new();
        let source_category = crate::log::short_type_name::<Src>();
        let source_name = namer.add(source_category, None);
        SourceLoop::set_instance_name(source, source_name.clone());
        let fanout_name = namer.add(mode.category(), None);

        // The source self-fixates its output caps; that becomes the sticky caps
        // replayed to every branch on attach.
        let sticky = source.intercept_caps().await?;
        source.configure_pipeline(&sticky)?.reject_refixate()?;

        let (src_tx, src_rx, src_tap) = link_tapped(link_capacity, observer.is_some());
        let source_log_name = ElementLogName::new(source_category, source_name.clone());

        // Dev-tooling tap: source 0, the routing stage 1. The routing stage has no
        // `process()` of its own (like a tee in `run_graph`), so it has no probe.
        if let Some(obs) = &observer {
            register_runner_tap(
                obs,
                alloc::vec![
                    (source_name, NodeRole::Source, None),
                    (fanout_name, NodeRole::Tee, None),
                ],
                alloc::vec![(0, DYN_FANOUT_NODE, sticky.clone(), src_tap)],
            );
        }

        // Branch probes are minted in the router arm as branches attach, and read
        // back here for the report once the arms have joined.
        let branch_probes: Arc<Mutex<Vec<Probe>>> = Arc::new(Mutex::new(Vec::new()));
        let mut tap = FanoutTap {
            obs: observer,
            namer,
            probes: branch_probes.clone(),
        };

        let source_fut: BoxFuture<'a, Result<DynArmOut, G2gError>> = Box::pin(async move {
            let mut adapter = SenderSink::new(src_tx);
            adapter.log_pushes_as(source_log_name);
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
                    attach_branch(
                        sink,
                        link_capacity,
                        &sticky,
                        &mut ports,
                        &new_arm_tx,
                        &mut tap,
                    )
                    .await?;
                }

                if accepting {
                    match select2(src_rx.recv(), new_branch_rx.recv()).await {
                        Either::Left(pkt) => {
                            if route_packet(pkt, &mut ports, &mut rr, mode).await? {
                                return Ok(DynArmOut::Router); // source ended
                            }
                        }
                        Either::Right(Some(sink)) => {
                            attach_branch(
                                sink,
                                link_capacity,
                                &sticky,
                                &mut ports,
                                &new_arm_tx,
                                &mut tap,
                            )
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
        let arms: Vec<BoxFuture<'a, Result<DynArmOut, G2gError>>> =
            alloc::vec![source_fut, router_fut];
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
        let per_element = crate::runtime::snapshot_all(&branch_probes.lock());
        Ok(RunStats {
            frames_emitted: emitted,
            frames_consumed: consumed,
            frames_dropped: 0,
            latency: LatencyReport::ZERO,
            allocation: None,
            clock_priority: ClockPriority::SystemFallback,
            base_time_ns: 0,
            coordinator_events: 0,
            per_element,
        })
    };

    (handle, run)
}

/// Attach a runtime-requested branch: name and probe it, register it with the
/// observer, give it its own link, replay the sticky caps into it so it
/// configures before any frame, add its sender to the port set, and hand its loop
/// future to the dynamic join.
#[cfg(feature = "std")]
async fn attach_branch<'a>(
    sink: alloc::boxed::Box<dyn DynAsyncElement + 'a>,
    link_capacity: usize,
    sticky: &Caps,
    ports: &mut Vec<SenderSink>,
    new_arm_tx: &Sender<BoxFuture<'a, Result<DynArmOut, G2gError>>>,
    tap: &mut FanoutTap,
) -> Result<(), G2gError> {
    let mut sink = sink;
    // Named and probed like a branch the static fan-out was built with, so its
    // telemetry row is indistinguishable from one declared up front.
    let category = sink.log_category();
    let name = tap.namer.add(category, None);
    sink.set_instance_name(name.clone());
    let probe = ElementProbe::new(ElementLogName::new(category, name.clone()));
    tap.probes.lock().push(Some(probe.clone()));

    let (btx, brx, edge) = link_tapped(link_capacity, tap.obs.is_some());
    // Register before the sticky caps are queued, so the node is in the topology
    // before the arm sees its first packet.
    if let Some(obs) = &tap.obs {
        let id = obs.add_node(name, NodeRole::Sink, Some(probe.clone()));
        obs.add_edge(DYN_FANOUT_NODE, id, sticky.clone(), edge);
    }

    let mut port = SenderSink::new(btx);
    port.push(PipelinePacket::CapsChanged(sticky.clone()))
        .await?;
    ports.push(port);
    let arm: BoxFuture<'a, Result<DynArmOut, G2gError>> = Box::pin(async move {
        // The sticky caps arrive as the arm's first packet, so it configures
        // the way any sink arm does.
        sink.drive_sink_arm_borrowed(SinkArmIo::standalone(brx, Some(probe)))
            .await
            .map(DynArmOut::Branch)
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
        // A fan-in arm's deadline tick is runner-internal and never crosses a
        // link, so nothing to route.
        Some(PipelinePacket::Tick) => Ok(false),
    }
}

/// Drives a terminal multi-output *source* ([`MultiOutputSource`]: 0 inputs to N
/// outputs) into N sinks, with no upstream, the fan-out mirror of
/// [`run_fanin_session`](crate::runtime::run_fanin_session). A WHEP session
/// receiving A/V over one PeerConnection emits each track on its own pad. Each
/// output's caps configure the matching sink; the session pushes via a
/// [`MultiSenderSink`](crate::MultiSenderSink) and emits a per-output `Eos`
/// when it ends, which the sink arms observe to finish. Per-branch mid-stream re-solve is not wired here yet
/// (a follow-up, as for the egress session runner).
#[cfg(feature = "std")]
pub async fn run_fanout_session<Sess, Clk>(
    session: &mut Sess,
    sinks: Vec<&mut dyn DynAsyncElement>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
) -> Result<RunStats, G2gError>
where
    Sess: MultiOutputSource,
    Clk: PipelineClock,
{
    run_fanout_session_inner(session, sinks, clock, link_capacity, None).await
}

/// As [`run_fanout_session`], but taps live telemetry into `observer`:
/// the session plus its N recv sinks, each sink's measured `process()` latency
/// and every branch link's packet / byte / drop counters readable mid-run via
/// [`Observer::snapshot`].
#[cfg(feature = "std")]
pub async fn run_fanout_session_observed<Sess, Clk>(
    session: &mut Sess,
    sinks: Vec<&mut dyn DynAsyncElement>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    observer: &Observer,
) -> Result<RunStats, G2gError>
where
    Sess: MultiOutputSource,
    Clk: PipelineClock,
{
    run_fanout_session_inner(session, sinks, clock, link_capacity, Some(observer)).await
}

#[cfg(feature = "std")]
async fn run_fanout_session_inner<Sess, Clk>(
    session: &mut Sess,
    sinks: Vec<&mut dyn DynAsyncElement>,
    clock: &Clk,
    link_capacity: impl Into<LinkCapacity>,
    observer: Option<&Observer>,
) -> Result<RunStats, G2gError>
where
    Sess: MultiOutputSource,
    Clk: PipelineClock,
{
    assert!(!sinks.is_empty(), "fan-out session needs at least one sink");
    assert!(
        session.output_count() == sinks.len(),
        "session output count must match the number of sinks"
    );
    // A thin builder over the DAG runner: the session is the graph's fan-out
    // source node, each sink hangs off one of its outputs.
    let mut g: Graph<GraphNodeRef<'_>> = Graph::new();
    let port_count = u8::try_from(sinks.len()).map_err(|_| G2gError::CapsMismatch)?;
    let fanout = g.add_fanout_src(GraphNodeRef::fanout_source_ref(session), port_count);
    // Named after the session type, as a session has no category of its own.
    let session_name =
        crate::log::InstanceNamer::new().add(crate::log::short_type_name::<Sess>(), None);
    g.set_node_name(fanout.node(), session_name);
    for (port, sink) in (0..port_count).zip(sinks) {
        let node = g.add_sink(GraphNodeRef::element_ref(sink));
        g.link(fanout.output(port), node)
            .map_err(|_| G2gError::CapsMismatch)?;
    }
    run_graph_inner(
        g,
        clock,
        link_capacity,
        None,
        None,
        None,
        None,
        observer,
        None,
        None,
        None,
        None,
    )
    .await
}

/// Drives an arbitrary-length linear pipeline:
/// `source -> transforms[0] -> ... -> transforms[N-1] -> sink`.
///
/// Generalizes [`run_source_transform_sink`] (one transform) and
/// [`run_simple_pipeline`] (zero) past their fixed arity, lifting the
/// "runner caps at 3 elements" limit so chains like
/// `decoder -> capsfilter -> converter -> sink` are expressible. Interior
/// elements are `&mut dyn DynAsyncElement` (heterogeneous, std-only, the same
/// erasure the fan-out runner uses); source and sink stay statically typed.
///
/// Negotiation runs the solver over all `N + 2` constraints at once and
/// configures each element with its input-side caps (the source with link 0).
/// A startup `ReFixate` restarts negotiation with the counter as the source's
/// caps, up to `MAX_FIXATION_ATTEMPTS` passes.
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
/// design/caps.md).
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
    run_linear_chain_inner(source, transforms, sink, clock, link_capacity, None, None).await
}

/// As [`run_linear_chain`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed) to
/// `bus` on a startup or mid-stream negotiation failure.
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
    run_linear_chain_inner(
        source,
        transforms,
        sink,
        clock,
        link_capacity,
        Some(bus),
        None,
    )
    .await
}

#[cfg(feature = "std")]
async fn run_linear_chain_inner<Src, Snk, Clk>(
    source: &mut Src,
    transforms: Vec<&mut dyn DynAsyncElement>,
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

    run_graph_inner(
        g,
        clock,
        link_capacity,
        bus,
        state,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .await
}

/// Sentinel sink for terminal elements (sinks proper): swallows pushes.
/// Process implementations of true sinks should not emit, but the type
/// system still requires an `&mut dyn OutputSink` parameter.
#[cfg(feature = "std")]
#[derive(Debug)]
pub(crate) struct NullSink;

#[cfg(feature = "std")]
impl OutputSink for NullSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        packet.take();
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

/// Drives a `source → transform → sink` pipeline over two bounded links.
///
/// Transform contract: `process(Eos)` may flush buffered state as
/// `DataFrame` packets; the runner forwards the EOS sentinel downstream after
/// `process(Eos)` returns, and skips its own push when the element already
/// forwarded one, so exactly one `Eos` reaches the sink either way.
///
/// `link_capacity` is the primary glass-to-glass latency knob. Under
/// steady-state backpressure each link sits full, so the latency floor is
/// roughly `2 * link_capacity * consumer_period`. For live video pipelines
/// (RTSP -> decode -> display) prefer **2**; for batch / throughput-oriented
/// workloads larger values are fine.
#[cfg(feature = "std")]
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
    run_linear_chain_inner(
        source,
        alloc::vec![transform as &mut dyn DynAsyncElement],
        sink,
        clock,
        link_capacity,
        None,
        None,
    )
    .await
}

/// As [`run_source_transform_sink`], but posts a structured
/// [`BusMessage::NegotiationFailed`](crate::BusMessage::NegotiationFailed)
/// to `bus` so the application learns *which* link conflicted (the returned
/// error stays the opaque `CapsMismatch`). Covers both startup
/// negotiation and the mid-stream re-solve sites (the sink's Phase-B re-solve
/// and the transform's Caps-α `Infeasible`). The bus is opt-in to keep the
/// common call site unchanged.
#[cfg(feature = "std")]
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
    run_linear_chain_inner(
        source,
        alloc::vec![transform as &mut dyn DynAsyncElement],
        sink,
        clock,
        link_capacity,
        Some(bus),
        None,
    )
    .await
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
            latency: LatencyReport {
                live: true,
                min_ns: 5_000_000,
                max_ns: Some(20_000_000),
            },
            ..RunStats::default()
        };
        let r = stats.report();
        // Frame line with the computed drop rate (10 / (90 + 10) = 10%).
        assert!(
            r.contains("emitted 100, consumed 90, dropped on links 10 (10.0% drop)"),
            "{r}"
        );
        // Declared latency window + live flag.
        assert!(r.contains("5.0 ms .. 20.0 ms (live) [declared]"), "{r}");
        assert!(r.contains("clock:"), "{r}");

        // An unbounded-latency, lossless pipeline reads cleanly too.
        let clean = RunStats {
            frames_emitted: 5,
            frames_consumed: 5,
            latency: LatencyReport {
                live: false,
                min_ns: 0,
                max_ns: None,
            },
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

#[cfg(all(test, feature = "std"))]
mod run_tests {
    use super::*;
    use crate::fanout::{MultiOutputSink, Router};
    use crate::frame::{Frame, FrameTiming};
    use crate::memory::{MemoryDomain, SystemSlice};
    use crate::runtime::block_on;
    use crate::{Bus, BusMessage, Dim, HardwareError, PipelineState, Rate, RawVideoFormat};
    use core::pin::Pin;

    const FRAMES: u64 = 6;
    const LINK_CAPACITY: usize = 2;

    struct ZeroClock;

    impl PipelineClock for ZeroClock {
        fn now_ns(&self) -> u64 {
            0
        }
    }

    fn caps() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(2),
            height: Dim::Fixed(2),
            framerate: Rate::Fixed(30 << 16),
            interlace: crate::Interlace::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        }
    }

    fn frame(sequence: u64) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 16]))),
            FrameTiming::default(),
            sequence,
        ))
    }

    fn every_frame() -> Vec<u64> {
        (0..FRAMES).collect()
    }

    // answers its first `refixates` configures with a counter-proposal
    #[derive(Default)]
    struct TestSource {
        refixates: u32,
    }

    impl SourceLoop for TestSource {
        type RunFuture<'a>
            = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
        where
            Self: 'a;
        type CapsFuture<'a>
            = core::future::Ready<Result<Caps, G2gError>>
        where
            Self: 'a;

        fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
            core::future::ready(Ok(caps()))
        }

        fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
            if self.refixates == 0 {
                return Ok(ConfigureOutcome::Accepted);
            }
            self.refixates -= 1;
            Ok(ConfigureOutcome::ReFixate(caps.clone()))
        }

        fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
            Box::pin(async move {
                for sequence in 0..FRAMES {
                    out.push(frame(sequence)).await?;
                }
                out.push(PipelinePacket::Eos).await?;
                Ok(FRAMES)
            })
        }
    }

    type Seen = Arc<Mutex<Vec<u64>>>;

    #[derive(Default)]
    struct TestSink {
        seen: Seen,
        fail_from: Option<u64>,
        proposal: Option<AllocationParams>,
        requests: crate::meta::MetaRequests,
    }

    fn sink_failure() -> G2gError {
        G2gError::Hardware(HardwareError::Io(5))
    }

    impl AsyncElement for TestSink {
        type ProcessFuture<'a>
            = core::future::Ready<Result<(), G2gError>>
        where
            Self: 'a;

        fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
            Ok(upstream.clone())
        }

        fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }

        fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
            self.proposal
        }

        fn meta_requests(&self) -> crate::meta::MetaRequests {
            self.requests
        }

        fn process<'a>(
            &'a mut self,
            packet: PipelinePacket,
            _out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            if let PipelinePacket::DataFrame(f) = &packet {
                if self.fail_from.is_some_and(|from| f.sequence >= from) {
                    return core::future::ready(Err(sink_failure()));
                }
                self.seen.lock().push(f.sequence);
            }
            core::future::ready(Ok(()))
        }
    }

    struct Pass;

    impl AsyncElement for Pass {
        type ProcessFuture<'a>
            = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
        where
            Self: 'a;

        fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
            Ok(upstream.clone())
        }

        fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }

        fn process<'a>(
            &'a mut self,
            packet: PipelinePacket,
            out: &'a mut dyn OutputSink,
        ) -> Self::ProcessFuture<'a> {
            Box::pin(async move { out.push(packet).await.map(|_| ()) })
        }
    }

    struct TwoOutputSession;

    impl MultiOutputSource for TwoOutputSession {
        type RunFuture<'a>
            = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
        where
            Self: 'a;

        fn output_count(&self) -> usize {
            2
        }

        fn output_caps(&self, _output: usize) -> Result<Caps, G2gError> {
            Ok(caps())
        }

        fn run<'a>(&'a mut self, out: &'a mut dyn MultiOutputSink) -> Self::RunFuture<'a> {
            Box::pin(async move {
                let ports = out.port_count();
                for sequence in 0..FRAMES {
                    for port in 0..ports {
                        out.push_to(port, frame(sequence)).await?;
                    }
                }
                for port in 0..ports {
                    out.push_to(port, PipelinePacket::Eos).await?;
                }
                Ok(FRAMES * ports as u64)
            })
        }
    }

    fn assert_ran_every_frame(stats: Result<RunStats, G2gError>, seen: &Seen, entry: &str) {
        let stats = stats.unwrap_or_else(|e| panic!("{entry}: {e:?}"));
        assert_eq!(stats.frames_emitted, FRAMES, "{entry}");
        assert_eq!(stats.frames_consumed, FRAMES, "{entry}");
        assert_eq!(*seen.lock(), every_frame(), "{entry}");
    }

    #[test]
    fn simple_pipeline_reports_the_sink_failure_not_the_shutdown_it_causes() {
        let mut sink = TestSink {
            fail_from: Some(1),
            ..TestSink::default()
        };
        let result = block_on(run_simple_pipeline(
            &mut TestSource::default(),
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY,
        ));
        assert_eq!(result, Err(sink_failure()));
    }

    #[test]
    fn startup_negotiation_takes_every_allowed_attempt() {
        let mut source = TestSource {
            refixates: MAX_FIXATION_ATTEMPTS - 1,
        };
        let mut sink = TestSink::default();
        let stats = block_on(run_simple_pipeline(
            &mut source,
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY,
        ));
        assert_ran_every_frame(stats, &sink.seen, "last attempt");

        let mut source = TestSource {
            refixates: MAX_FIXATION_ATTEMPTS,
        };
        let result = block_on(run_simple_pipeline(
            &mut source,
            &mut TestSink::default(),
            &ZeroClock,
            LINK_CAPACITY,
        ));
        assert_eq!(result, Err(G2gError::FixationFailed));
    }

    #[test]
    fn simple_pipeline_settles_the_allocation_on_a_domain_the_source_emits() {
        let cuda_or_system = AllocationParams {
            accepts: DomainSet::only(MemoryDomainKind::Cuda).with(MemoryDomainKind::System),
            ..AllocationParams::cuda(64, 2, 1)
        };
        let mut source = TestSource::default();
        let emits = SourceLoop::output_domains(&source);
        let mut sink = TestSink {
            proposal: Some(cuda_or_system),
            ..TestSink::default()
        };
        let stats = block_on(run_simple_pipeline(
            &mut source,
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY,
        ))
        .expect("runs");
        let settled = stats.allocation.expect("the sink proposed one");
        assert!(emits.contains(settled.domain), "{settled:?}");
        assert_eq!(settled.size_bytes, cuda_or_system.size_bytes);
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn simple_pipeline_passes_a_metadata_only_proposal_through() {
        let requests = crate::meta::MetaRequests::new().request::<crate::meta::AnalyticsMeta>();
        let mut sink = TestSink {
            requests,
            ..TestSink::default()
        };
        let stats = block_on(run_simple_pipeline(
            &mut TestSource::default(),
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY,
        ))
        .expect("runs");
        assert_eq!(
            stats.allocation,
            Some(AllocationParams::meta_demand(requests))
        );
    }

    #[test]
    fn stateful_pipeline_flows_while_playing_and_stops_at_null() {
        let mut sink = TestSink::default();
        let playing = StateController::new(PipelineState::Playing);
        let stats = block_on(run_simple_pipeline_stateful(
            &mut TestSource::default(),
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY,
            &playing,
        ));
        assert_ran_every_frame(stats, &sink.seen, "playing");

        let mut sink = TestSink::default();
        let null = StateController::new(PipelineState::Null);
        let _ = block_on(run_simple_pipeline_stateful(
            &mut TestSource::default(),
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY,
            &null,
        ));
        assert!(sink.seen.lock().is_empty());
    }

    #[test]
    fn simple_pipeline_with_a_bus_reports_the_sink_link_fill() {
        let (bus, handle) = Bus::new(64);
        let mut sink = TestSink::default();
        let stats = block_on(run_simple_pipeline_with_bus(
            &mut TestSource::default(),
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY,
            &handle,
        ));
        assert_ran_every_frame(stats, &sink.seen, "with bus");
        let mut buffering = 0;
        while let Some(message) = bus.try_recv() {
            if matches!(message, BusMessage::Buffering { .. }) {
                buffering += 1;
            }
        }
        assert!(buffering > 0);
    }

    #[test]
    fn linear_entry_points_run_the_stream_through_a_transform() {
        let (_bus, handle) = Bus::new(64);

        let mut sink = TestSink::default();
        let stats = block_on(run_linear_chain(
            &mut TestSource::default(),
            alloc::vec![&mut Pass as &mut dyn DynAsyncElement],
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY,
        ));
        assert_ran_every_frame(stats, &sink.seen, "run_linear_chain");

        let mut sink = TestSink::default();
        let stats = block_on(run_linear_chain_with_bus(
            &mut TestSource::default(),
            alloc::vec![&mut Pass as &mut dyn DynAsyncElement],
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY,
            &handle,
        ));
        assert_ran_every_frame(stats, &sink.seen, "run_linear_chain_with_bus");

        let mut sink = TestSink::default();
        let stats = block_on(run_source_transform_sink_with_bus(
            &mut TestSource::default(),
            &mut Pass,
            &mut sink,
            &ZeroClock,
            LINK_CAPACITY,
            &handle,
        ));
        assert_ran_every_frame(stats, &sink.seen, "run_source_transform_sink_with_bus");
    }

    #[test]
    fn fan_out_entry_points_deliver_to_the_selected_branch() {
        let (_bus, handle) = Bus::new(64);
        let observer = Observer::new();
        let selected = 1;
        for entry in ["run_source_fanout", "observed", "with_bus"] {
            let mut router = Router::new(2);
            router.handle().select(selected);
            let (mut idle, mut chosen) = (TestSink::default(), TestSink::default());
            let sinks: Vec<&mut dyn DynAsyncElement> = alloc::vec![&mut idle, &mut chosen];
            let mut source = TestSource::default();
            let run = match entry {
                "run_source_fanout" => block_on(run_source_fanout(
                    &mut source,
                    &mut router,
                    sinks,
                    &ZeroClock,
                    LINK_CAPACITY,
                )),
                "observed" => block_on(run_source_fanout_observed(
                    &mut source,
                    &mut router,
                    sinks,
                    &ZeroClock,
                    LINK_CAPACITY,
                    &observer,
                )),
                _ => block_on(run_source_fanout_with_bus(
                    &mut source,
                    &mut router,
                    sinks,
                    &ZeroClock,
                    LINK_CAPACITY,
                    &handle,
                )),
            };
            assert_ran_every_frame(run, &chosen.seen, entry);
            assert!(idle.seen.lock().is_empty(), "{entry}");
        }
    }

    #[test]
    fn fan_out_session_entry_points_feed_every_output() {
        let observer = Observer::new();
        for observed in [false, true] {
            let (mut first, mut second) = (TestSink::default(), TestSink::default());
            let sinks: Vec<&mut dyn DynAsyncElement> = alloc::vec![&mut first, &mut second];
            let mut session = TwoOutputSession;
            let ports = session.output_count() as u64;
            let stats = if observed {
                block_on(run_fanout_session_observed(
                    &mut session,
                    sinks,
                    &ZeroClock,
                    LINK_CAPACITY,
                    &observer,
                ))
            } else {
                block_on(run_fanout_session(
                    &mut session,
                    sinks,
                    &ZeroClock,
                    LINK_CAPACITY,
                ))
            }
            .expect("session runs");
            assert_eq!(stats.frames_consumed, FRAMES * ports, "observed {observed}");
            for sink in [&first, &second] {
                assert_eq!(*sink.seen.lock(), every_frame(), "observed {observed}");
            }
        }
    }

    #[test]
    fn dynamic_router_deals_frames_round_robin() {
        let branches: Vec<TestSink> = (0..2).map(|_| TestSink::default()).collect();
        let seen: Vec<Seen> = branches.iter().map(|b| b.seen.clone()).collect();
        let mut source = TestSource::default();
        let (handle, run) = run_source_router_dynamic(&mut source, LINK_CAPACITY);
        for branch in branches {
            handle
                .add_branch(Box::new(branch))
                .expect("router accepts the branch");
        }
        let stats = block_on(run).expect("router runs");
        assert_eq!(stats.frames_consumed, FRAMES);
        for (index, branch) in seen.iter().enumerate() {
            let dealt: Vec<u64> = every_frame()
                .into_iter()
                .filter(|sequence| *sequence as usize % seen.len() == index)
                .collect();
            assert_eq!(*branch.lock(), dealt, "branch {index}");
        }
    }

    struct PropertyTypedSource;

    impl SourceLoop for PropertyTypedSource {
        type RunFuture<'a>
            = core::future::Ready<Result<u64, G2gError>>
        where
            Self: 'a;
        type CapsFuture<'a>
            = core::future::Ready<Result<Caps, G2gError>>
        where
            Self: 'a;

        fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
            core::future::ready(Ok(caps()))
        }

        fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
            Ok(ConfigureOutcome::Accepted)
        }

        fn run<'a>(&'a mut self, _out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
            core::future::ready(Ok(0))
        }

        fn configured_output_caps(&self) -> Option<Caps> {
            Some(caps())
        }
    }

    #[test]
    fn source_defaults_claim_no_duration_and_no_properties() {
        let mut source = PropertyTypedSource;
        assert_eq!(source.query_duration(), None);
        assert_eq!(
            source.set_property("location", PropValue::Str("clip.mp4".into())),
            Err(PropError::Unknown)
        );
        assert_eq!(
            source.probe_output_caps(),
            Ok(source.configured_output_caps())
        );
        assert!(source.configured_output_caps().is_some());
    }
}
