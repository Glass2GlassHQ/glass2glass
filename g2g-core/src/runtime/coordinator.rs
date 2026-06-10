//! M18 β scaffolding — the pipeline coordinator (control channel only).
//!
//! Today's linear runners spawn source / transform / sink as
//! independent, spawn-and-forget futures. Mid-stream cross-element
//! coordination (allocation re-cascade β, Phase C per-branch /
//! per-input re-solve, a future mid-stream clock change) has nowhere to
//! live in that topology: each arm only sees its own two links. The
//! coordinator is the single task that will own that coordination
//! (`DESIGN-M16-workaround3-reconfigure.md` §9.4 β; R2: single-task
//! coordinator, not shared `Arc<Mutex>` on every element; R3:
//! out-of-band coordinator-channel, not in-band `PipelinePacket`s).
//!
//! This is Session B: the channel topology plus an observe-only
//! coordinator. No reconfiguration logic moves here yet, so the data
//! plane behaves exactly as before. Runner arms *report* the events the
//! coordinator will later act on; the stub only counts them. Session C
//! moves startup negotiation in; Session D adds α hooks; Session E turns
//! `CoordinatorEvent` into a real `Recascade` cascade.

use crate::caps::Caps;
use crate::element::{AsyncElement, ConfigureOutcome};
use crate::error::G2gError;
use crate::format_element::CapsConstraint;
use crate::runtime::channel::{bounded, Receiver, Sender};
use crate::runtime::runner::SourceLoop;
use crate::runtime::solver::solve_linear;

/// Maximum number of Phase 1 + Phase 2 negotiation passes before a setup
/// gives up with `FixationFailed`. Three is enough for any reasonable
/// `ReFixate` chain (source → sink → source counter) while still being
/// a hard backstop against pathologically-counter-proposing elements.
pub(crate) const MAX_FIXATION_ATTEMPTS: u32 = 3;

/// An event a runner arm reports to the coordinator. Distinct from
/// [`PipelinePacket`](crate::frame::PipelinePacket): these travel
/// out-of-band on the coordinator-channel (R3) and never enter the data
/// plane, so they cannot collide with the reverse `Reconfigure` slot the
/// data links already carry.
#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    /// A boundary element forwarded a mid-stream `CapsChanged`
    /// downstream and the next element accepted it. β turns this into a
    /// `Recascade { caps }` that re-runs the allocation cascade over the
    /// affected subgraph; the Session B stub only records it.
    CapsChanged(Caps),
}

/// Producer end of the control channel, handed (by clone) to a runner
/// arm so it can report [`CoordinatorEvent`]s. Wraps the in-house mpsc
/// [`Sender`], so multiple arms can report to one coordinator and the
/// channel closes only when the last handle drops.
#[derive(Debug, Clone)]
pub struct CoordinatorHandle {
    tx: Sender<CoordinatorEvent>,
}

impl CoordinatorHandle {
    /// Report an event to the coordinator. Best-effort: a closed channel
    /// means the coordinator already terminated (shutdown), so the event
    /// is dropped. Session B never depends on delivery beyond the
    /// observe-only count.
    pub async fn report(&self, event: CoordinatorEvent) {
        let _ = self.tx.send(event).await;
    }
}

/// The coordinator task. Session B behavior: drain the control channel,
/// counting observed events, until every [`CoordinatorHandle`] has
/// dropped (channel close). Returns the count so the runner can surface
/// it in [`RunStats`](crate::runtime::RunStats) for topology validation.
#[derive(Debug)]
pub struct Coordinator {
    rx: Receiver<CoordinatorEvent>,
}

impl Coordinator {
    /// Run to completion. Resolves once all handles drop.
    pub async fn run(self) -> u64 {
        let mut observed = 0u64;
        while self.rx.recv().await.is_some() {
            observed += 1;
        }
        observed
    }
}

/// Build the control channel: a [`Coordinator`] task and one
/// [`CoordinatorHandle`] for the runner to clone to its arms. `capacity`
/// bounds in-flight events; mid-stream caps changes are rare relative to
/// data frames, so a small bound is ample.
pub fn coordinator(capacity: usize) -> (Coordinator, CoordinatorHandle) {
    let (tx, rx) = bounded(capacity);
    (Coordinator { rx }, CoordinatorHandle { tx })
}

/// Per-link fixated caps from the linear startup negotiation the
/// coordinator owns. For a `source → transform → sink` chain the solver
/// returns two links; this names them so the per-link structure the β
/// re-cascade will reconfigure is explicit at the call site.
#[derive(Debug, Clone)]
pub(crate) struct LinearNegotiation {
    /// Caps on the source-output / transform-input link. Read by β's
    /// re-cascade (Session E); retained now to name the per-link
    /// structure the negotiation produces.
    #[allow(dead_code)]
    pub(crate) source_link: Caps,
    /// Caps on the transform-output / sink-input link.
    pub(crate) sink_link: Caps,
}

/// M18 Session C: startup negotiation for a `source → transform → sink`
/// chain, relocated verbatim from `run_source_transform_sink` to the
/// coordinator module (its conceptual home, since β turns the same
/// solver-plus-configure cascade into the mid-stream re-cascade). No
/// behavior change: the runner calls this and uses the returned
/// `sink_link` exactly where it used the loop's `negotiated_caps`.
///
/// M16 step 4b: negotiation routes through `solve_linear` via the legacy
/// bridge. The bridge wraps today's `intercept_caps` / `propose_output_caps`
/// callbacks as `LegacyTransform` / `LegacySink` constraints; the solver's
/// legacy cascade runs the forward chain identically to the pre-M16 inline
/// cascade and fixates the final caps. `ReFixate` retry stays here (the
/// solver doesn't model counter-proposals): on each retry the source's
/// `LegacySource` seed is replaced by the counter and the solver is re-run.
pub(crate) fn negotiate_source_transform_sink<Src, Tx, Snk>(
    source: &mut Src,
    transform: &mut Tx,
    sink: &mut Snk,
) -> Result<LinearNegotiation, G2gError>
where
    Src: SourceLoop,
    Tx: AsyncElement,
    Snk: AsyncElement,
{
    let mut refix_counter: Option<Caps> = None;
    let mut attempts = 0u32;
    let (source_link, sink_link) = loop {
        attempts += 1;
        if attempts > MAX_FIXATION_ATTEMPTS {
            return Err(G2gError::FixationFailed);
        }
        // Build the constraint chain in a scope so the immutable
        // borrows of `transform` / `sink` are released before the
        // `configure_pipeline` calls below take mutable access.
        let (src_caps, sink_caps) = {
            // M16 step 5f: honor `SourceLoop::caps_constraint` on the
            // first attempt; refixate retries fall back to
            // `LegacySource(counter)` since counter-proposals are a
            // legacy concept.
            let src_c = match &refix_counter {
                Some(c) => CapsConstraint::LegacySource(c.clone()),
                None => source.caps_constraint()?,
            };
            let tx_c = transform.caps_constraint_as_transform();
            let sink_c = sink.caps_constraint_as_sink();
            let links = solve_linear(&[&src_c, &tx_c, &sink_c])
                .map_err(|_| G2gError::CapsMismatch)?;
            // M16 step 5d: per-link configure. For a 3-element chain
            // links has length 2: [source-output / transform-input,
            // transform-output / sink-input]. The transform's
            // `configure_pipeline` historically receives one caps; we
            // pass its *input* side, which is what existing decoders
            // (e.g. `FfmpegH264Dec`) expect.
            if links.len() != 2 {
                return Err(G2gError::CapsMismatch);
            }
            (links[0].clone(), links[1].clone())
        };

        let mut refixate: Option<Caps> = None;
        for outcome in [
            source.configure_pipeline(&src_caps)?,
            transform.configure_pipeline(&src_caps)?,
            sink.configure_pipeline(&sink_caps)?,
        ] {
            match outcome {
                ConfigureOutcome::Accepted => {}
                ConfigureOutcome::ReFixate(counter) => {
                    refixate = Some(counter);
                    break;
                }
            }
        }
        match refixate {
            Some(counter) => refix_counter = Some(counter),
            None => break (src_caps, sink_caps),
        }
    };
    Ok(LinearNegotiation { source_link, sink_link })
}

/// M18 α — element-local re-allocation on a mid-stream caps change.
///
/// When a `CapsChanged` is applied to an element mid-stream, the element
/// re-derives its own allocation params from the new caps
/// (`propose_allocation`) and stores them for its next-frame allocation
/// (`configure_allocation`). This is the cheap, element-local phase of
/// the re-cascade (DESIGN-M16-workaround3-reconfigure.md §9.4 α): a sink
/// resizes its own pool, a decoder re-derives its scratch buffer. There
/// is deliberately **no** cross-element propagation here, that is β; the
/// element both proposes and configures itself rather than answering a
/// peer.
///
/// Safe against double allocation by the per-`Frame.caps` invariant
/// (§5): in-flight old-caps frames already hold old-pool buffers; only
/// frames after this point key the new params.
pub(crate) fn realloc_local<E>(element: &mut E, caps: &Caps)
where
    E: AsyncElement,
{
    if let Some(params) = element.propose_allocation(caps) {
        element.configure_allocation(&params);
    }
}
