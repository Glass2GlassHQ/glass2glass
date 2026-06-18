//! Pipeline message bus (M11).
//!
//! A pipeline-level channel for asynchronous out-of-band messages: elements
//! notify the application of lifecycle events (EOS, errors, warnings) or
//! custom signals without holding a back-reference to it (DESIGN.md §4.9.1).
//! Many elements produce, one application consumes — a thin **mp-sc** wrapper
//! over the runtime channel ([`crate::runtime::bounded`]).
//!
//! Posting is non-blocking by default (`try_post`): a control message must
//! never stall an element on the data path. The application drains with
//! `try_recv` (non-blocking) or `recv` (awaiting).

use crate::error::G2gError;
use crate::runtime::solver::NegotiationFailure;
use crate::runtime::{bounded, Receiver, Sender};
use crate::state::PipelineState;

/// An out-of-band message from an element to the application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BusMessage {
    /// End-of-stream observed by the posting element.
    Eos,
    /// Fatal error; the application should tear the pipeline down.
    Error(G2gError),
    /// Non-fatal condition worth surfacing.
    Warning(G2gError),
    /// A caps negotiation failed, carrying the structured
    /// [`NegotiationFailure`] that names *which* link conflicted on *what*.
    /// The runner still returns the (opaque) `G2gError::CapsMismatch` to its
    /// caller; this preserves the detail the error type can't, so the
    /// application can report the offending element pair (M18 item 7).
    NegotiationFailed(NegotiationFailure),
    /// The pipeline's lifecycle state changed (M76). Posted by
    /// [`StateController::set_state`](crate::runtime::StateController) on every
    /// effective transition along the `NULL → READY → PAUSED → PLAYING` ladder.
    StateChanged {
        /// State before the change.
        old: PipelineState,
        /// State after the change.
        new: PipelineState,
    },
    /// An async state change completed (M77): a non-live `Paused` transition
    /// finished once the sink took its preroll buffer. The GStreamer
    /// `ASYNC_DONE` analog; posted once per preroll by
    /// [`StateController::notify_prerolled`](crate::runtime::StateController).
    AsyncDone,
    /// Application-defined signal carrying an opaque code.
    Custom(u64),
}

/// Producer end of the [`Bus`], held by elements. Cloneable so every element
/// gets its own producer; the bus closes once all handles drop.
#[derive(Debug, Clone)]
pub struct BusHandle {
    tx: Sender<BusMessage>,
}

impl BusHandle {
    /// Post without blocking. Returns `false` if the bus is full or closed —
    /// control messages must never stall an element, so a full bus drops the
    /// message rather than applying backpressure.
    pub fn try_post(&self, message: BusMessage) -> bool {
        self.tx.try_send(message).is_ok()
    }

    /// Post, awaiting capacity. `Shutdown` if the bus (its [`Bus`]) is gone.
    pub async fn post(&self, message: BusMessage) -> Result<(), G2gError> {
        self.tx.send(message).await.map_err(|_| G2gError::Shutdown)
    }
}

/// Consumer end of the pipeline bus, held by the application.
#[derive(Debug)]
pub struct Bus {
    rx: Receiver<BusMessage>,
}

impl Bus {
    /// Build a bus with a bounded backlog and its first producer handle.
    /// Clone the handle to hand producers to more elements.
    pub fn new(capacity: usize) -> (Bus, BusHandle) {
        let (tx, rx) = bounded::<BusMessage>(capacity);
        (Bus { rx }, BusHandle { tx })
    }

    /// Non-blocking drain of one message; `None` when empty.
    pub fn try_recv(&self) -> Option<BusMessage> {
        self.rx.try_recv()
    }

    /// Await the next message; `None` once every handle has dropped and the
    /// backlog is drained.
    pub async fn recv(&self) -> Option<BusMessage> {
        self.rx.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::future::Future;
    use core::pin::Pin;

    /// Single-poll block_on; the bus futures here resolve immediately.
    fn block_on<F: Future>(mut fut: F) -> F::Output {
        use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
        static VT: RawWakerVTable = RawWakerVTable::new(
            |_| RawWaker::new(core::ptr::null(), &VT),
            |_| {},
            |_| {},
            |_| {},
        );
        // SAFETY: VT's hooks never dereference the data pointer.
        let waker = unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VT)) };
        let mut cx = Context::from_waker(&waker);
        // SAFETY: `fut` is pinned to the stack for the duration of this call.
        let mut pinned = unsafe { Pin::new_unchecked(&mut fut) };
        match pinned.as_mut().poll(&mut cx) {
            Poll::Ready(v) => v,
            Poll::Pending => panic!("bus::tests::block_on saw Pending"),
        }
    }

    #[test]
    fn try_post_then_try_recv_is_fifo() {
        let (bus, handle) = Bus::new(4);
        assert!(handle.try_post(BusMessage::Custom(1)));
        assert!(handle.try_post(BusMessage::Eos));
        assert_eq!(bus.try_recv(), Some(BusMessage::Custom(1)));
        assert_eq!(bus.try_recv(), Some(BusMessage::Eos));
        assert_eq!(bus.try_recv(), None);
    }

    #[test]
    fn try_post_drops_on_full() {
        let (_bus, handle) = Bus::new(1);
        assert!(handle.try_post(BusMessage::Custom(1)));
        assert!(!handle.try_post(BusMessage::Custom(2)), "full bus drops the message");
    }

    #[test]
    fn try_recv_none_when_empty() {
        let (bus, _handle) = Bus::new(2);
        assert_eq!(bus.try_recv(), None);
    }

    #[test]
    fn recv_none_after_all_handles_drop_and_drained() {
        let (bus, handle) = Bus::new(2);
        handle.try_post(BusMessage::Error(G2gError::Shutdown));
        drop(handle);
        assert_eq!(block_on(bus.recv()), Some(BusMessage::Error(G2gError::Shutdown)));
        assert_eq!(block_on(bus.recv()), None, "closed and drained");
    }

    #[test]
    fn async_post_and_recv_round_trip() {
        let (bus, handle) = Bus::new(2);
        block_on(handle.post(BusMessage::Custom(9))).unwrap();
        assert_eq!(block_on(bus.recv()), Some(BusMessage::Custom(9)));
    }
}
