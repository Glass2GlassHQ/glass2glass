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
use crate::tag::TagList;

/// An out-of-band message from an element to the application.
// Not `Eq`: `StreamCollection` carries `Caps`, which is only `PartialEq` (its
// geometry / rate placeholders are not totally ordered). `PartialEq` is enough
// for the bus (assert_eq! in tests, direct comparison); nothing keys on it.
#[derive(Debug, Clone, PartialEq)]
pub enum BusMessage {
    /// A new stream has started: posted by the runner's source arm before the
    /// source produces any data, one per source (the GStreamer
    /// `GST_MESSAGE_STREAM_START` analog, M206). Brackets a stream with the
    /// matching [`Eos`](BusMessage::Eos) so an application can track stream
    /// lifetime (e.g. reset per-stream UI on each start).
    StreamStart,
    /// End-of-stream observed by the posting element.
    Eos,
    /// An informational, non-error notification (the GStreamer
    /// `GST_MESSAGE_INFO` analog, M206), the third severity below
    /// [`Warning`](BusMessage::Warning) and [`Error`](BusMessage::Error). Carries
    /// a human-readable message. Posted by elements or the application for
    /// progress / status that is not a problem (a reconnect, a fallback taken),
    /// so it never tears the pipeline down.
    Info(alloc::string::String),
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
    /// Quality-of-service report (M85): a sink is running behind the pipeline
    /// clock and dropped a frame that arrived too late to present. The
    /// GStreamer `GST_MESSAGE_QOS` analog. Posted by a synchronizing sink
    /// (e.g. [`SyncSink`](../../g2g_plugins/syncsink/struct.SyncSink.html)) when
    /// it drops a late frame, so the application can react (lower the source
    /// rate, simplify the pipeline) instead of silently falling behind.
    Qos {
        /// Running time (PTS) of the frame this report concerns.
        running_time_ns: u64,
        /// How far past its deadline the frame was, in ns. Signed: positive is
        /// late (behind the clock), negative early.
        jitter_ns: i64,
        /// Frames the sink has presented so far (cumulative).
        processed: u64,
        /// Frames the sink has dropped so far (cumulative, this drop included).
        dropped: u64,
    },
    /// Buffering level report (M87): the fill percent (0-100) of a monitored
    /// link feeding a sink. The GStreamer `GST_MESSAGE_BUFFERING` analog. g2g
    /// has no `queue` element (per-edge `LinkPolicy` is the leaky-queue analog),
    /// so this reports the bounded link channel's own occupancy, posted by the
    /// runner's sink arm when the level crosses a quartile band. An application
    /// can pause until it sees `100`, or surface a "buffering..." indicator
    /// while it is low.
    Buffering {
        /// Fill of the sink's input link, 0 (empty / underrun) to 100 (full).
        percent: u8,
    },
    /// Stream metadata a demuxer recovered from the container (the GStreamer
    /// `GST_MESSAGE_TAG` analog). Posted out of band so the application can read
    /// title / artist / encoder / etc. without intercepting the data path. A
    /// demuxer with a tag source (e.g. `oggdemux` parsing VorbisComment) posts
    /// it once the metadata header is parsed.
    Tag(TagList),
    /// The elementary streams a demuxer found in the container (the GStreamer
    /// `GST_MESSAGE_STREAM_COLLECTION` analog, M376, the data model playbin is
    /// built on). Posted out of band once the demuxer has parsed its track list,
    /// listing *every* available audio / video / text stream (its type and
    /// [`Caps`](crate::caps::Caps)) regardless of which one(s) the demuxer
    /// forwards, so the application can discover what is in the container. App
    /// driven selection among them is a follow-up.
    StreamCollection(crate::stream::StreamCollection),
    /// The set of streams a demuxer now forwards changed in response to an
    /// application selection (the GStreamer `GST_MESSAGE_STREAMS_SELECTED`
    /// analog, M377). Carries the active stream ids (the ids from the
    /// [`StreamCollection`](BusMessage::StreamCollection)), so the app confirms
    /// which streams took effect after a
    /// [`StreamSelectController::select`](crate::runtime::StreamSelectController::select).
    StreamsSelected {
        /// The stream ids the demuxer is now forwarding.
        ids: alloc::vec::Vec<alloc::string::String>,
    },
    /// The total stream duration became known or changed (the GStreamer
    /// `GST_MESSAGE_DURATION_CHANGED` analog, M203). Posted by the runner's
    /// source arm when a source first reports a duration
    /// ([`SourceLoop::query_duration`](crate::runtime::SourceLoop::query_duration)),
    /// so an application can refresh a seek bar's length. The value is also
    /// readable any time from the [`PipelineProgress`](crate::runtime::PipelineProgress)
    /// handle; this message is the push notification of the change.
    DurationChanged {
        /// The new total duration in nanoseconds.
        duration_ns: u64,
    },
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
        assert!(
            !handle.try_post(BusMessage::Custom(2)),
            "full bus drops the message"
        );
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
        assert_eq!(
            block_on(bus.recv()),
            Some(BusMessage::Error(G2gError::Shutdown))
        );
        assert_eq!(block_on(bus.recv()), None, "closed and drained");
    }

    #[test]
    fn async_post_and_recv_round_trip() {
        let (bus, handle) = Bus::new(2);
        block_on(handle.post(BusMessage::Custom(9))).unwrap();
        assert_eq!(block_on(bus.recv()), Some(BusMessage::Custom(9)));
    }

    #[test]
    fn info_and_stream_start_round_trip() {
        let (bus, handle) = Bus::new(4);
        handle.try_post(BusMessage::StreamStart);
        handle.try_post(BusMessage::Info(alloc::string::String::from(
            "reconnecting",
        )));
        assert_eq!(bus.try_recv(), Some(BusMessage::StreamStart));
        match bus.try_recv() {
            Some(BusMessage::Info(s)) => assert_eq!(s, "reconnecting"),
            other => panic!("expected Info, got {other:?}"),
        }
    }
}
