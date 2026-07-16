//! Stream-selection request channel (M377), the GStreamer `GST_EVENT_SELECT_STREAMS`
//! analog and the sibling of [`SeekController`](crate::runtime::SeekController).
//!
//! A cloneable handle carrying a pending stream selection from the application to
//! a selection-aware demuxer. After the demuxer announces what streams exist (a
//! [`StreamCollection`](crate::stream::StreamCollection) on the bus, M376), the app
//! names the stream id(s) it wants via [`StreamSelectController::select`]; the
//! demuxer's `process` calls [`take_pending`](StreamSelectController::take_pending)
//! and switches which stream it forwards (re-negotiating caps if the kind
//! changes), then confirms the active set on the bus
//! ([`BusMessage::StreamsSelected`](crate::bus::BusMessage::StreamsSelected)).
//!
//! Selection travels upstream to the demuxer in GStreamer; here the app holds the
//! controller and a clone lives in the demuxer, so a request reaches it without a
//! back-reference, exactly as [`SeekController`](crate::runtime::SeekController)
//! carries a seek. The latest request wins (an app re-selecting only needs the
//! final set). Polling (not waking) is deliberate: a demuxer checks between
//! pushes, and it can only act on a selection while it is processing data.
//!
//! The ids are the opaque stream ids the demuxer published in its collection
//! (e.g. `"matroska-track-2"`), so the app round-trips exactly what it was told.

use alloc::string::String;
use alloc::vec::Vec;

use spin::Mutex;

use alloc::sync::Arc;

#[derive(Debug, Default)]
struct SelectInner {
    /// The latest unhandled selection (a list of stream ids), or `None`.
    pending: Mutex<Option<Vec<String>>>,
}

/// Cloneable stream-selection channel. Every clone shares one pending-selection
/// slot, so an app-held controller and a demuxer-held clone see the same request.
#[derive(Debug, Clone, Default)]
pub struct StreamSelectController {
    inner: Arc<SelectInner>,
}

impl StreamSelectController {
    /// A controller with no pending selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Application side: request that the demuxer forward exactly the streams
    /// named by `ids` (the ids it published in its collection). Replaces any
    /// prior unhandled request (latest-wins).
    pub fn select(&self, ids: Vec<String>) {
        *self.inner.pending.lock() = Some(ids);
    }

    /// Demuxer side: take and clear the pending selection, or `None` if none is
    /// set since the last take.
    pub fn take_pending(&self) -> Option<Vec<String>> {
        self.inner.pending.lock().take()
    }

    /// Whether a selection is currently pending (not yet taken).
    pub fn has_pending(&self) -> bool {
        self.inner.pending.lock().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_pending_clears_and_returns_latest() {
        let c = StreamSelectController::new();
        assert!(c.take_pending().is_none());
        assert!(!c.has_pending());

        c.select(alloc::vec![String::from("matroska-track-1")]);
        // Latest-wins: a second select replaces the first.
        c.select(alloc::vec![String::from("matroska-track-2")]);
        assert!(c.has_pending());

        let got = c.take_pending().expect("a selection was pending");
        assert_eq!(got, alloc::vec![String::from("matroska-track-2")]);
        assert!(c.take_pending().is_none(), "taking clears the slot");
    }

    #[test]
    fn a_clone_shares_the_slot() {
        let app = StreamSelectController::new();
        let demux = app.clone();
        app.select(alloc::vec![String::from("matroska-track-2")]);
        assert_eq!(
            demux.take_pending(),
            Some(alloc::vec![String::from("matroska-track-2")]),
            "the demuxer clone sees the app's request",
        );
    }
}
