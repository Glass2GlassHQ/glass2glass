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

use alloc::vec::Vec;

use crate::bus::{BusHandle, BusMessage};
use crate::caps::Caps;
use crate::element::{AsyncElement, ConfigureOutcome};
use crate::error::G2gError;
use crate::format_element::CapsConstraint;
use crate::query::AllocationParams;
use crate::runtime::channel::{bounded, Receiver, Sender};
use crate::runtime::runner::SourceLoop;
use crate::runtime::solver::{solve_linear, NegotiationFailure};

/// M18 item 7: post a structured negotiation failure to the bus when one is
/// wired, so the application learns which link conflicted on what. The runner
/// still surfaces the opaque `G2gError::CapsMismatch` to its caller (startup)
/// or drives a reverse `Reconfigure` (mid-stream); this only adds the detail
/// the error type can't carry. Non-blocking: a control message must never
/// stall a runner on a full bus. Shared by every solve site across the
/// runners (`runner.rs`, `fanin.rs`, and here).
pub(crate) fn report_nego_failure(bus: Option<&BusHandle>, failure: NegotiationFailure) {
    if let Some(b) = bus {
        b.try_post(BusMessage::NegotiationFailed(failure));
    }
}

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
    /// A boundary element forwarded a mid-stream `CapsChanged` downstream
    /// and the sink accepted it. β: the coordinator forwards the sink's
    /// re-derived allocation `proposal` one hop upstream to the *last*
    /// interior element's `configure_allocation`, kicking off the upstream
    /// re-cascade. `proposal` is `None` when the sink declares no allocation
    /// needs, in which case there is nothing to cascade and the event is
    /// observe-only.
    CapsChanged {
        caps: Caps,
        proposal: Option<AllocationParams>,
    },

    /// β N-hop: an interior arm (index `index` among the interior elements)
    /// applied a `Recascade` directive, re-derived its own proposal, and
    /// reports it so the coordinator forwards it one hop further upstream to
    /// element `index - 1`. Index 0 is the first interior element; its reply
    /// terminates the cascade (the source is not an interruptible arm). The
    /// single-transform runner never emits this (its lone transform doesn't
    /// reply), so its cascade stays one hop.
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
#[derive(Debug, Clone)]
pub(crate) enum ArmDirective {
    /// Apply this downstream-derived proposal to the element's own pool:
    /// the upstream half of the cascade (`transform.configure_allocation`).
    Recascade(AllocationParams),
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
/// events for [`RunStats`](crate::runtime::RunStats).
///
/// β: it owns one [`ArmDirective`] sender per interior arm (`arm_ctrl`,
/// ordered source-to-sink). The cascade is purely reactive, so it never
/// blocks on a walk: a [`CoordinatorEvent::CapsChanged`] kicks it off by
/// forwarding the sink's proposal to the *last* interior arm; each interior
/// arm then replies with [`CoordinatorEvent::ArmProposal`], which the
/// coordinator forwards one hop further upstream (to `index - 1`) until index
/// 0 terminates it. A single-transform chain passes one sender and its arm
/// never replies, so the cascade is exactly one hop (the prior single-hop β).
/// `arm_ctrl` is empty for an observe-only coordinator.
#[derive(Debug)]
pub struct Coordinator {
    rx: Receiver<CoordinatorEvent>,
    arm_ctrl: Vec<Sender<ArmDirective>>,
}

impl Coordinator {
    /// Run to completion. Resolves once all handles drop.
    pub async fn run(self) -> u64 {
        let mut observed = 0u64;
        while let Some(event) = self.rx.recv().await {
            observed += 1;
            match event {
                CoordinatorEvent::CapsChanged { proposal, .. } => {
                    // β: start the cascade at the last interior arm. Serial,
                    // so the bounded control channel never blocks; a closed
                    // channel means that arm already drained and exited.
                    if let (Some(ctrl), Some(p)) = (self.arm_ctrl.last(), proposal) {
                        let _ = ctrl.send(ArmDirective::Recascade(p)).await;
                    }
                }
                CoordinatorEvent::ArmProposal { index, proposal } => {
                    // β N-hop: forward the arm's re-derived proposal one hop
                    // further upstream. Index 0's reply terminates the cascade.
                    if index > 0 {
                        if let (Some(ctrl), Some(p)) =
                            (self.arm_ctrl.get(index - 1), proposal)
                        {
                            let _ = ctrl.send(ArmDirective::Recascade(p)).await;
                        }
                    }
                }
            }
        }
        observed
    }
}

/// Build an observe-only control channel: a [`Coordinator`] with no
/// upstream re-cascade and one [`CoordinatorHandle`] to clone to the arms.
/// `capacity` bounds in-flight events; mid-stream caps changes are rare
/// relative to data frames, so a small bound is ample.
pub fn coordinator(capacity: usize) -> (Coordinator, CoordinatorHandle) {
    let (tx, rx) = bounded(capacity);
    (
        Coordinator { rx, arm_ctrl: Vec::new() },
        CoordinatorHandle { tx },
    )
}

/// β single-hop: build the control channel with one upstream re-cascade leg
/// (the `source -> transform -> sink` runner). Returns the [`ArmDirective`]
/// receiver the transform arm selects on alongside its data link.
pub(crate) fn coordinator_with_recascade(
    capacity: usize,
) -> (Coordinator, CoordinatorHandle, Receiver<ArmDirective>) {
    let (tx, rx) = bounded(capacity);
    let (ctrl_tx, ctrl_rx) = bounded(capacity);
    (
        Coordinator { rx, arm_ctrl: alloc::vec![ctrl_tx] },
        CoordinatorHandle { tx },
        ctrl_rx,
    )
}

/// β N-hop: build the control channel with one re-cascade leg per interior
/// element of an `N`-element linear chain (`run_linear_chain`). Returns the
/// per-arm [`ArmDirective`] receivers, ordered source-to-sink, that each
/// interior arm selects on. `n == 0` yields an observe-only coordinator.
pub(crate) fn coordinator_with_recascade_n(
    capacity: usize,
    n: usize,
) -> (Coordinator, CoordinatorHandle, Vec<Receiver<ArmDirective>>) {
    let (tx, rx) = bounded(capacity);
    let mut arm_ctrl = Vec::with_capacity(n);
    let mut arm_rx = Vec::with_capacity(n);
    for _ in 0..n {
        let (ctrl_tx, ctrl_rx) = bounded(capacity);
        arm_ctrl.push(ctrl_tx);
        arm_rx.push(ctrl_rx);
    }
    (Coordinator { rx, arm_ctrl }, CoordinatorHandle { tx }, arm_rx)
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
    /// Caps on the transform-output / sink-input link. Retained for β's
    /// re-cascade (Session E), the same as `source_link`; the M12 allocation
    /// query that used to read it now runs inside negotiation against the
    /// per-link caps directly.
    #[allow(dead_code)]
    pub(crate) sink_link: Caps,
    /// The folded M12 allocation proposal handed to the source (the
    /// most-demanding of the sink's and transform's requirements), or `None`
    /// if no element proposed. The runner records it on `RunStats`. Resolved
    /// before `configure_pipeline` so a transform (e.g. a hardware decoder)
    /// can size its buffer pool from `min_buffers` at open time.
    pub(crate) allocation: Option<AllocationParams>,
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
pub(crate) async fn negotiate_source_transform_sink<Src, Tx, Snk>(
    source: &mut Src,
    transform: &mut Tx,
    sink: &mut Snk,
    bus: Option<&BusHandle>,
) -> Result<LinearNegotiation, G2gError>
where
    Src: SourceLoop,
    Tx: AsyncElement,
    Snk: AsyncElement,
{
    let mut refix_counter: Option<Caps> = None;
    let mut attempts = 0u32;
    let (source_link, sink_link, allocation) = loop {
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
                None => source.caps_constraint().await?,
            };
            let tx_c = transform.caps_constraint_as_transform();
            let sink_c = sink.caps_constraint_as_sink();
            let links = solve_linear(&[&src_c, &tx_c, &sink_c]).map_err(|f| {
                // M18 item 7: surface the structured failure to the bus before
                // collapsing it to the opaque `CapsMismatch` the caller gets.
                report_nego_failure(bus, f);
                G2gError::CapsMismatch
            })?;
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

        // M12 allocation query, resolved *before* the `configure_pipeline`
        // cascade so a transform (e.g. a hardware decoder) can size its buffer
        // pool from the downstream consumer's `min_buffers` when it opens.
        // Resolve sink → transform first so the transform folds the sink's
        // requirement into the proposal it answers to the source. (Previously
        // this ran in the runner *after* negotiation, i.e. after the decoder
        // had already opened with a fixed default.)
        if let Some(p) = sink.propose_allocation(&sink_caps) {
            transform.configure_allocation(&p);
        }
        let allocation = transform.propose_allocation(&sink_caps);
        if let Some(p) = &allocation {
            source.configure_allocation(p);
        }

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
            None => break (src_caps, sink_caps, allocation),
        }
    };
    Ok(LinearNegotiation { source_link, sink_link, allocation })
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
///
/// Returns the element's re-derived proposal so the caller can also feed
/// it to β's cross-element cascade (the sink's proposal flows one hop
/// upstream to the transform). `None` when the element declares no
/// allocation needs.
pub(crate) fn realloc_local<E>(element: &mut E, caps: &Caps) -> Option<AllocationParams>
where
    E: AsyncElement,
{
    let params = element.propose_allocation(caps);
    if let Some(p) = &params {
        element.configure_allocation(p);
    }
    params
}

/// [`DynAsyncElement`] counterpart of [`realloc_local`], for `Box`-erased
/// fan-out branch sinks.
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
    use core::pin::Pin;
    use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    static NOOP_VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(core::ptr::null(), &NOOP_VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );

    fn noop_waker() -> Waker {
        // SAFETY: every NOOP_VTABLE fn is a no-op that never dereferences the
        // data pointer, so a null data pointer is sound.
        unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &NOOP_VTABLE)) }
    }

    /// Busy-poll a future to completion. Every channel op in these tests
    /// resolves without truly parking (capacity is always available and the
    /// senders close deterministically), so this never spins.
    fn block_on<F: Future>(fut: F) -> F::Output {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        let mut fut = fut;
        // SAFETY: `fut` lives on this stack frame and is not moved after
        // pinning.
        let mut pinned = unsafe { Pin::new_unchecked(&mut fut) };
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
        }
    }

    #[test]
    fn forwards_proposal_as_recascade_directive() {
        let (coord, handle, ctrl_rx) = coordinator_with_recascade(2);
        block_on(handle.report(CoordinatorEvent::CapsChanged {
            caps: nv12(1920, 1080),
            proposal: Some(AllocationParams::system(4096, 2)),
        }));
        // Drop the handle so the coordinator terminates after draining.
        drop(handle);

        let observed = block_on(coord.run());
        assert_eq!(observed, 1, "the one reported event is counted");
        match ctrl_rx.try_recv() {
            Some(ArmDirective::Recascade(p)) => assert_eq!(p.size_bytes, 4096),
            other => panic!("expected Recascade(4096), got {other:?}"),
        }
    }

    #[test]
    fn caps_change_without_proposal_forwards_nothing() {
        let (coord, handle, ctrl_rx) = coordinator_with_recascade(2);
        block_on(handle.report(CoordinatorEvent::CapsChanged {
            caps: nv12(640, 480),
            proposal: None,
        }));
        drop(handle);

        let observed = block_on(coord.run());
        assert_eq!(observed, 1, "the event is still counted");
        assert!(
            ctrl_rx.try_recv().is_none(),
            "no proposal means no upstream re-cascade directive"
        );
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

    fn taken_size(rx: &Receiver<ArmDirective>) -> Option<usize> {
        match rx.try_recv() {
            Some(ArmDirective::Recascade(p)) => Some(p.size_bytes),
            None => None,
        }
    }

    #[test]
    fn n_hop_cascade_walks_upstream_one_hop_per_reply() {
        // Three interior arms. The sink's CapsChanged starts the cascade at
        // the last arm; each arm's reply forwards one hop further upstream;
        // the first arm's reply (index 0) terminates it. Capacity 8 holds the
        // four pre-queued events (the coordinator isn't draining yet).
        let (coord, handle, arm_rx) = coordinator_with_recascade_n(8, 3);
        block_on(handle.report(CoordinatorEvent::CapsChanged {
            caps: nv12(1920, 1080),
            proposal: Some(AllocationParams::system(10, 1)),
        }));
        block_on(handle.report(CoordinatorEvent::ArmProposal {
            index: 2,
            proposal: Some(AllocationParams::system(20, 1)),
        }));
        block_on(handle.report(CoordinatorEvent::ArmProposal {
            index: 1,
            proposal: Some(AllocationParams::system(30, 1)),
        }));
        block_on(handle.report(CoordinatorEvent::ArmProposal {
            index: 0,
            proposal: Some(AllocationParams::system(40, 1)),
        }));
        drop(handle);

        assert_eq!(block_on(coord.run()), 4);
        // Last arm got the sink's proposal; each upstream arm got its
        // downstream neighbour's re-derived proposal; index 0's reply is a
        // no-op (the source is not an interruptible arm).
        assert_eq!(taken_size(&arm_rx[2]), Some(10));
        assert_eq!(taken_size(&arm_rx[1]), Some(20));
        assert_eq!(taken_size(&arm_rx[0]), Some(30));
    }

    #[test]
    fn n_hop_cascade_stops_when_a_reply_has_no_proposal() {
        // A middle arm with no allocation needs (proposal None) ends the
        // cascade: nothing reaches the arms above it.
        let (coord, handle, arm_rx) = coordinator_with_recascade_n(8, 3);
        block_on(handle.report(CoordinatorEvent::CapsChanged {
            caps: nv12(640, 480),
            proposal: Some(AllocationParams::system(10, 1)),
        }));
        block_on(handle.report(CoordinatorEvent::ArmProposal { index: 2, proposal: None }));
        drop(handle);

        assert_eq!(block_on(coord.run()), 2);
        assert_eq!(taken_size(&arm_rx[2]), Some(10));
        assert_eq!(taken_size(&arm_rx[1]), None, "None proposal stops the cascade");
        assert_eq!(taken_size(&arm_rx[0]), None);
    }
}
