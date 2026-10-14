//! M18 β scaffolding — the pipeline coordinator (control channel only).
//!
//! Today's linear runners spawn source / transform / sink as
//! independent, spawn-and-forget futures. Mid-stream cross-element
//! coordination (allocation re-cascade β, Phase C per-branch /
//! per-input re-solve, a future mid-stream clock change) has nowhere to
//! live in that topology: each arm only sees its own two links. The
//! coordinator is the single task that will own that coordination
//! (design/caps.md; R2: single-task
//! coordinator, not shared `Arc<Mutex>` on every element; R3:
//! out-of-band coordinator-channel, not in-band `PipelinePacket`s).
//!
//! This is Session B: the channel topology plus an observe-only
//! coordinator. No reconfiguration logic moves here yet, so the data
//! plane behaves exactly as before. Runner arms *report* the events the
//! coordinator will later act on; the stub only counts them. Session C
//! moves startup negotiation in; Session D adds α hooks; Session E turns
//! `CoordinatorEvent` into a real `Recascade` cascade.

#[cfg(feature = "std")]
use crate::bus::{BusHandle, BusMessage};
use crate::caps::Caps;
#[cfg(feature = "std")]
use crate::error::G2gError;
#[cfg(feature = "std")]
use crate::format_element::CapsConstraint;
use crate::query::AllocationParams;
use crate::runtime::channel::{bounded, Receiver, Sender};
#[cfg(feature = "std")]
use crate::runtime::solver::{solve_linear, NegotiationFailure};

/// M18 item 7: post a structured negotiation failure to the bus when one is
/// wired, so the application learns which link conflicted on what. The runner
/// still surfaces the opaque `G2gError::CapsMismatch` to its caller (startup)
/// or drives a reverse `Reconfigure` (mid-stream); this only adds the detail
/// the error type can't carry. Non-blocking: a control message must never
/// stall a runner on a full bus. Shared by every solve site across the
/// runners (`runner.rs`, `fanin.rs`, and here).
#[cfg(feature = "std")]
pub(crate) fn report_nego_failure(bus: Option<&BusHandle>, failure: NegotiationFailure) {
    if let Some(b) = bus {
        b.try_post(BusMessage::NegotiationFailed(failure));
    }
}

/// Wrap an arm's mid-stream `configure_pipeline` / `configure_output` result:
/// a rejection is logged under the `caps` category (ERROR, so it prints by
/// default) naming the element instance and the caps it rejected, before the
/// error unwinds the arm as an otherwise-bare `CapsMismatch`.
#[cfg(feature = "std")]
pub(crate) fn log_caps_rejected<T>(
    instance: Option<&str>,
    caps: &Caps,
    result: Result<T, G2gError>,
) -> Result<T, G2gError> {
    if let Err(err) = &result {
        crate::g2g_error!(
            crate::log::Target {
                category: crate::log::CAPS_CATEGORY,
                instance,
            },
            "rejected mid-stream caps {}: {err:?}",
            caps.to_gst_string()
        );
    }
    result
}

/// Narrate an arm's mid-stream forward resolution under the `caps` category
/// (DEBUG, visible via `G2G_CAPS_TRACE=1`): the incoming caps and what the arm
/// forwards downstream, marking whether the output was steered (re-fixated
/// against the downstream feasibility snapshot) or deferred (input forwarded
/// unchanged, the element's own `process` re-emits its real output).
#[cfg(feature = "std")]
pub(crate) fn log_caps_forward(instance: Option<&str>, input: &Caps, forward: &Caps, fixed: bool) {
    crate::g2g_debug!(
        crate::log::Target {
            category: crate::log::CAPS_CATEGORY,
            instance,
        },
        "mid-stream caps {} -> forwarding {} ({})",
        input.to_gst_string(),
        forward.to_gst_string(),
        if fixed { "steered" } else { "deferred" }
    );
}

/// Solve a linear constraint chain and return its final (sink-side) link caps,
/// posting a structured failure to the bus on conflict. Collapses the
/// `solve_linear(..).map_err(report_nego_failure)?; links.last().cloned()` block
/// repeated across the runner startup paths.
#[cfg(feature = "std")]
pub(crate) fn solve_last_link(
    constraints: &[&CapsConstraint<'_>],
    bus: Option<&BusHandle>,
) -> Result<Caps, G2gError> {
    let links = solve_linear(constraints).map_err(|f| {
        report_nego_failure(bus, f);
        G2gError::CapsMismatch
    })?;
    links.last().cloned().ok_or(G2gError::CapsMismatch)
}

/// An event a runner arm reports to the coordinator. Distinct from
/// [`PipelinePacket`](crate::frame::PipelinePacket): these travel
/// out-of-band on the coordinator-channel (R3) and never enter the data
/// plane, so they cannot collide with the reverse `Reconfigure` slot the
/// data links already carry.
#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    /// A boundary element forwarded a mid-stream `CapsChanged` downstream
    /// and the sink accepted it, with the sink's re-derived allocation
    /// `proposal` (`None` when it declares no allocation needs).
    CapsChanged {
        caps: Caps,
        proposal: Option<AllocationParams>,
    },

    /// An interior arm (index `index` among the interior elements) re-derived
    /// its own allocation proposal.
    ArmProposal {
        index: usize,
        proposal: Option<AllocationParams>,
    },
}

/// A directive the coordinator sends to an interruptible upstream arm to
/// apply a mid-stream allocation re-cascade (β). The arm calls
/// `configure_allocation` on its element at its next [`select2`] await
/// point, so a `recv().await` parked on data does not block the directive.
///
/// [`select2`]: crate::runtime::select2
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub(crate) enum ArmDirective {
    /// Apply this downstream-derived proposal to the element's own pool:
    /// the upstream half of the cascade (`transform.configure_allocation`).
    Recascade(AllocationParams),

    /// The pool a *producer* settled on for this arm's input link, pushed
    /// downstream when a muxer re-derives its merged-output allocation (M839).
    /// The arm absorbs it (`configure_allocation`) and does **not** re-propose:
    /// it is being told what it will receive, not asked what it wants, and a
    /// reply would bounce straight back into the muxer it came from. Only the
    /// DAG runner's coordinator sends it.
    ProducerAllocation(AllocationParams),
}

#[cfg(feature = "std")]
impl ArmDirective {
    /// The allocation this directive carries, for the arms that absorb it the
    /// same way whichever direction it came from.
    pub(crate) fn params(&self) -> &AllocationParams {
        match self {
            ArmDirective::Recascade(p) | ArmDirective::ProducerAllocation(p) => p,
        }
    }
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
    /// is dropped.
    pub async fn report(&self, event: CoordinatorEvent) {
        let _ = self.tx.send(event).await;
    }
}

/// The coordinator task. Drains the control channel until every
/// [`CoordinatorHandle`] has dropped (channel close), counting observed
/// events for [`RunStats`](crate::runtime::RunStats). Observe-only: the graph
/// runner's own coordinator walks the β re-cascade.
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

/// Build an observe-only control channel: a [`Coordinator`] and one
/// [`CoordinatorHandle`] to clone to the arms. `capacity` bounds in-flight
/// events; mid-stream caps changes are rare relative to data frames, so a
/// small bound is ample.
pub fn coordinator(capacity: usize) -> (Coordinator, CoordinatorHandle) {
    let (tx, rx) = bounded(capacity);
    (Coordinator { rx }, CoordinatorHandle { tx })
}

/// M18 α: element-local re-allocation on a mid-stream caps change. The element
/// re-derives its allocation from the new caps and configures itself with it;
/// cross-element propagation is β, not this.
#[cfg(feature = "std")]
pub(crate) fn realloc_local_dyn(element: &mut dyn crate::element::DynAsyncElement, caps: &Caps) {
    if let Some(params) = element.propose_allocation(caps) {
        element.configure_allocation(&params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{Caps, Dim, Rate, RawVideoFormat};
    use core::future::Future;
    use core::task::{Context, Poll, Waker};

    /// Busy-poll a future to completion. Every channel op in these tests
    /// resolves without truly parking (capacity is always available and the
    /// senders close deterministically), so this never spins.
    fn block_on<F: Future>(fut: F) -> F::Output {
        let mut cx = Context::from_waker(Waker::noop());
        let mut pinned = core::pin::pin!(fut);
        loop {
            if let Poll::Ready(v) = pinned.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
            interlace: crate::Interlace::Any,
            colorimetry: crate::Colorimetry::UNKNOWN,
        }
    }

    #[test]
    fn observe_only_coordinator_never_forwards() {
        // The 2-element pipeline's coordinator has no transform leg.
        let (coord, handle) = coordinator(2);
        block_on(handle.report(CoordinatorEvent::CapsChanged {
            caps: nv12(1280, 720),
            proposal: Some(AllocationParams::system(8192, 1)),
        }));
        drop(handle);
        assert_eq!(block_on(coord.run()), 1);
    }
}
