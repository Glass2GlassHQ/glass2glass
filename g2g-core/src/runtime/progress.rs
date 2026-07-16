//! Pipeline progress handle (M203): the application's window into a running
//! pipeline's position and duration, the GStreamer `POSITION` / `DURATION`
//! query analog.
//!
//! GStreamer answers `POSITION` / `DURATION` by sending a query upstream along
//! the pads: a sink answers position from its segment plus last buffer, a source
//! or demuxer answers duration. g2g composes paths statically and pushes
//! forward, so rather than a query object travelling upstream, the runner
//! *publishes* into a shared handle the application holds. The sink arm publishes
//! the stream-time position of each buffer it consumes; the source arm publishes
//! the duration its source reports ([`SourceLoop::query_duration`]). The app
//! polls [`position`](PipelineProgress::position) /
//! [`duration`](PipelineProgress::duration) between draws (a seek bar, a progress
//! readout). This mirrors the [`SeekController`](crate::runtime::SeekController)
//! handle (the app holds it, the source reads it), inverted: here the runner
//! writes and the app reads.

use alloc::sync::Arc;
use core::sync::atomic::Ordering;

use portable_atomic::AtomicU64;

/// Sentinel for an unknown position: `0` is a valid position (stream start), so
/// position uses the maximum value to mean "not known yet".
const POSITION_UNKNOWN: u64 = u64::MAX;
/// Sentinel for an unknown duration: a real duration is always `> 0`, so `0`
/// cleanly means "unknown" (not reported, or a live / open-ended stream). Using
/// `0` lets [`PipelineProgress::publish_duration`] fold sources with `fetch_max`.
const DURATION_UNKNOWN: u64 = 0;

#[derive(Debug)]
struct ProgressInner {
    position_ns: AtomicU64,
    duration_ns: AtomicU64,
}

/// A cloneable handle to a running pipeline's progress. The application
/// constructs one, passes it to
/// [`run_graph_with_progress`](crate::runtime::run_graph_with_progress), and
/// polls [`position`](Self::position) / [`duration`](Self::duration) while the
/// pipeline runs. Every clone shares one set of values.
#[derive(Debug, Clone)]
pub struct PipelineProgress {
    inner: Arc<ProgressInner>,
}

impl Default for PipelineProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineProgress {
    /// A handle with no known position or duration yet.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ProgressInner {
                position_ns: AtomicU64::new(POSITION_UNKNOWN),
                duration_ns: AtomicU64::new(DURATION_UNKNOWN),
            }),
        }
    }

    /// Application side: the current playback position in nanoseconds (stream
    /// time), or `None` before the first buffer reaches a sink.
    pub fn position(&self) -> Option<u64> {
        match self.inner.position_ns.load(Ordering::Relaxed) {
            POSITION_UNKNOWN => None,
            ns => Some(ns),
        }
    }

    /// Application side: the total stream duration in nanoseconds, or `None` if
    /// unknown (not yet reported, or a live / open-ended stream).
    pub fn duration(&self) -> Option<u64> {
        match self.inner.duration_ns.load(Ordering::Relaxed) {
            DURATION_UNKNOWN => None,
            ns => Some(ns),
        }
    }

    /// Runner side: publish the latest position (the stream time of the buffer
    /// just consumed at a sink). Latest-writer-wins, matching GStreamer
    /// answering position from the sink's most recent buffer; a seek that
    /// rewinds simply publishes a smaller value next.
    pub fn set_position(&self, ns: u64) {
        // A real position never collides with the "unknown" sentinel; clamp the
        // (absurd, ~584-year) maximum so the sentinel stays meaningful.
        let ns = if ns == POSITION_UNKNOWN { POSITION_UNKNOWN - 1 } else { ns };
        self.inner.position_ns.store(ns, Ordering::Relaxed);
    }

    /// Runner side: report a source's duration. Takes the maximum across
    /// sources (the pipeline runs until its longest stream ends). Returns `true`
    /// when this changed the stored duration, so the caller posts a single
    /// [`DurationChanged`](crate::BusMessage::DurationChanged) on a real change.
    /// A `0` (unknown) report is ignored.
    pub fn publish_duration(&self, ns: u64) -> bool {
        if ns == DURATION_UNKNOWN {
            return false;
        }
        let prev = self.inner.duration_ns.fetch_max(ns, Ordering::Relaxed);
        prev < ns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_until_published() {
        let p = PipelineProgress::new();
        assert_eq!(p.position(), None);
        assert_eq!(p.duration(), None);
    }

    #[test]
    fn position_is_latest_writer() {
        let p = PipelineProgress::new();
        p.set_position(0);
        assert_eq!(p.position(), Some(0), "zero is a valid position, not unknown");
        p.set_position(5_000);
        assert_eq!(p.position(), Some(5_000));
        // A rewind (seek) just publishes a smaller value.
        p.set_position(1_000);
        assert_eq!(p.position(), Some(1_000));
    }

    #[test]
    fn duration_takes_max_and_reports_change() {
        let p = PipelineProgress::new();
        assert!(p.publish_duration(10_000), "first known duration is a change");
        assert_eq!(p.duration(), Some(10_000));
        // A shorter source does not lower the pipeline duration, nor report a change.
        assert!(!p.publish_duration(4_000));
        assert_eq!(p.duration(), Some(10_000));
        // A longer source raises it and reports the change.
        assert!(p.publish_duration(20_000));
        assert_eq!(p.duration(), Some(20_000));
        // An "unknown" (0) report is ignored.
        assert!(!p.publish_duration(0));
        assert_eq!(p.duration(), Some(20_000));
    }

    #[test]
    fn clones_share_state() {
        let app = PipelineProgress::new();
        let runner = app.clone();
        runner.set_position(7_777);
        runner.publish_duration(99_999);
        assert_eq!(app.position(), Some(7_777));
        assert_eq!(app.duration(), Some(99_999));
    }
}
