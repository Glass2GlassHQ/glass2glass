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
use crate::runtime::channel::{bounded, Receiver, Sender};

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
