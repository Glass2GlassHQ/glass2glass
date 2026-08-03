//! Duration-keyed prebuffer window for the adaptive segment loops
//! ([`HlsSrc`](crate::hlssrc) / [`DashSrc`](crate::dashsrc)): the
//! [`HttpSrc`](crate::httpsrc) byte-window analog, keyed by segment duration
//! because an adaptive source knows each segment's play time (`#EXTINF` / MPD
//! timing) but not its byte rate. The loop fetches segments into the window
//! while it is below its duration target and emits from the front otherwise,
//! posting [`BusMessage::Buffering`] percent on the attached bus during the
//! startup / post-seek fill (quartile bands, like `HttpSrc`), silent in steady
//! state. Init segments ride the window with duration 0 so re-emission on an
//! ABR switch stays ordered behind already-queued media.
//!
//! The loops are sequential (one fetch or one push at a time), so the window
//! is a startup / seek buffer plus a bounded lookahead; it cannot keep emitting
//! while a fetch stalls.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use g2g_core::{BusHandle, BusMessage};

#[derive(Debug)]
pub(crate) struct SegmentPrebuffer {
    target_ns: u64,
    /// Payloads in emit order, each with the play duration it contributes.
    queue: VecDeque<(Vec<u8>, u64)>,
    buffered_ns: u64,
    /// Startup / post-seek fill in progress: admissions post Buffering percent.
    filling: bool,
    bus: Option<BusHandle>,
    last_bucket: Option<u8>,
}

impl SegmentPrebuffer {
    /// `target_ms` of media to hold before emitting; 0 disables prebuffering
    /// (the window passes each segment straight through).
    pub(crate) fn new(target_ms: u64, bus: Option<BusHandle>) -> Self {
        Self {
            target_ns: target_ms.saturating_mul(1_000_000),
            queue: VecDeque::new(),
            buffered_ns: 0,
            filling: target_ms > 0,
            bus,
            last_bucket: None,
        }
    }

    /// Whether the loop should fetch another segment before emitting: the
    /// window is empty, or still below its duration target.
    pub(crate) fn wants_fetch(&self) -> bool {
        self.queue.is_empty() || self.buffered_ns < self.target_ns
    }

    /// Queue a fetched payload. `duration_ns` is its play time (0 for an init
    /// segment).
    pub(crate) fn admit(&mut self, bytes: Vec<u8>, duration_ns: u64) {
        self.queue.push_back((bytes, duration_ns));
        self.buffered_ns = self.buffered_ns.saturating_add(duration_ns);
        if self.filling {
            self.post_level();
            if self.buffered_ns >= self.target_ns {
                self.filling = false;
            }
        }
    }

    /// Dequeue the next payload to emit. Popping while still filling means
    /// nothing more is fetchable right now (live edge / end of playlist), which
    /// completes buffering early, like the stream ending in `HttpSrc`.
    pub(crate) fn pop(&mut self) -> Option<Vec<u8>> {
        let (bytes, duration_ns) = self.queue.pop_front()?;
        if self.filling {
            self.filling = false;
            self.post(100);
        }
        self.buffered_ns = self.buffered_ns.saturating_sub(duration_ns);
        Some(bytes)
    }

    /// Drop the window (a flushing seek discards queued media) and re-arm the
    /// fill, so the refill posts Buffering again.
    pub(crate) fn clear(&mut self) {
        self.queue.clear();
        self.buffered_ns = 0;
        if self.target_ns > 0 {
            self.filling = true;
            self.last_bucket = None;
        }
    }

    fn post_level(&mut self) {
        let pct = ((self.buffered_ns.saturating_mul(100) / self.target_ns.max(1)).min(100)) as u8;
        self.post(pct);
    }

    /// Post on quartile-band transitions only, so a fill is a handful of
    /// messages, not one per segment.
    fn post(&mut self, pct: u8) {
        if let Some(b) = &self.bus {
            let bucket = (pct / 25).min(4);
            if self.last_bucket != Some(bucket) {
                self.last_bucket = Some(bucket);
                b.try_post(BusMessage::Buffering {
                    percent: pct,
                    element: None,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_window_passes_straight_through() {
        let mut w = SegmentPrebuffer::new(0, None);
        assert!(w.wants_fetch());
        w.admit(alloc::vec![1u8], 2_000_000_000);
        assert!(!w.wants_fetch());
        assert_eq!(w.pop(), Some(alloc::vec![1u8]));
        assert!(w.wants_fetch());
    }

    #[test]
    fn fills_to_target_then_emits() {
        let mut w = SegmentPrebuffer::new(4_000, None);
        w.admit(alloc::vec![0u8], 0); // init rides with duration 0
        w.admit(alloc::vec![1u8], 2_000_000_000);
        assert!(w.wants_fetch(), "2 s of a 4 s target still fills");
        w.admit(alloc::vec![2u8], 2_000_000_000);
        assert!(!w.wants_fetch(), "target reached");
        assert_eq!(w.pop(), Some(alloc::vec![0u8]));
        assert_eq!(w.pop(), Some(alloc::vec![1u8]));
        assert!(w.wants_fetch(), "below target again after draining");
    }

    #[test]
    fn clear_rearms_the_fill() {
        let mut w = SegmentPrebuffer::new(2_000, None);
        w.admit(alloc::vec![1u8], 2_000_000_000);
        assert!(!w.wants_fetch());
        w.clear();
        assert_eq!(w.pop(), None);
        assert!(w.wants_fetch());
    }
}
