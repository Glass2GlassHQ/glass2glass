//! Seek request channel (M82).
//!
//! A cloneable handle carrying a pending [`Seek`] from the application to a
//! seek-aware source. The app calls [`SeekController::seek`]; the source's run
//! loop calls [`SeekController::take_pending`] between frames and, on a flushing
//! seek, emits `Flush`, repositions, emits the post-flush
//! [`Segment`](crate::segment::Segment), and resumes from the new position.
//!
//! Seeks travel upstream to the source in GStreamer; here the app holds the
//! controller and a clone lives in the source, so a seek reaches the producer
//! without a back-reference. The latest request wins: an app scrubbing fast
//! only needs the final target, so a new `seek` replaces any prior unhandled
//! one. Polling (rather than waking) is deliberate — a producing source checks
//! between frames; a parked source isn't producing, so there is nothing to
//! reposition until it resumes.

use alloc::sync::Arc;

use spin::Mutex;

use crate::segment::Seek;

#[derive(Debug, Default)]
struct SeekInner {
    pending: Mutex<Option<Seek>>,
}

/// Cloneable seek channel. Every clone shares one pending-seek slot.
#[derive(Debug, Clone, Default)]
pub struct SeekController {
    inner: Arc<SeekInner>,
}

impl SeekController {
    /// A controller with no pending seek.
    pub fn new() -> Self {
        Self::default()
    }

    /// Application side: request a seek. Replaces any prior unhandled request
    /// (latest-wins).
    pub fn seek(&self, seek: Seek) {
        *self.inner.pending.lock() = Some(seek);
    }

    /// Source side: take and clear the pending seek, or `None` if none is set.
    pub fn take_pending(&self) -> Option<Seek> {
        self.inner.pending.lock().take()
    }

    /// Whether a seek is currently pending (not yet taken).
    pub fn has_pending(&self) -> bool {
        self.inner.pending.lock().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment::{Seek, SeekFlags, SeekType};

    #[test]
    fn take_pending_clears_and_returns() {
        let c = SeekController::new();
        assert!(!c.has_pending());
        assert_eq!(c.take_pending(), None);

        c.seek(Seek::flush_to(5_000));
        assert!(c.has_pending());
        let s = c.take_pending().expect("a seek was pending");
        assert_eq!(s.start, 5_000);
        assert!(s.is_flush());
        // Taken: now empty.
        assert!(!c.has_pending());
        assert_eq!(c.take_pending(), None);
    }

    #[test]
    fn latest_seek_wins() {
        let c = SeekController::new();
        c.seek(Seek::flush_to(1_000));
        c.seek(Seek {
            rate: 1.0,
            flags: SeekFlags::FLUSH,
            start_type: SeekType::Set,
            start: 9_000,
            stop_type: SeekType::None,
            stop: 0,
        });
        // Only the final request survives.
        assert_eq!(c.take_pending().map(|s| s.start), Some(9_000));
    }

    #[test]
    fn clones_share_one_slot() {
        let app = SeekController::new();
        let src = app.clone();
        app.seek(Seek::flush_to(42));
        // The source-side clone observes the app-side request.
        assert_eq!(src.take_pending().map(|s| s.start), Some(42));
        assert!(!app.has_pending());
    }
}
