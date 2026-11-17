// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Pipeline progress handle: the application's window into a running
//! pipeline's position and duration, the GStreamer `POSITION` / `DURATION`
//! query analog.
//!
//! GStreamer answers `POSITION` / `DURATION` by sending a query upstream along
//! the pads: a sink answers position from its segment plus last buffer, a source
//! or demuxer answers duration. g2g composes paths statically and pushes
//! forward, so rather than a query object travelling upstream, the runner
//! *publishes* into a shared handle the application holds. The sink arm publishes
//! the stream-time position of each buffer it consumes, unless the sink adopts
//! its [`SinkPosition`] to publish what it has played; the source arm publishes
//! the duration its source reports
//! ([`SourceLoop::query_duration`](crate::runtime::SourceLoop::query_duration)).
//! The app polls [`position`](PipelineProgress::position) /
//! [`duration`](PipelineProgress::duration) between draws (a seek bar, a progress
//! readout). This mirrors the [`SeekController`](crate::runtime::SeekController)
//! handle (the app holds it, the source reads it), inverted: here the runner
//! writes and the app reads.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::Ordering;

use portable_atomic::AtomicU64;

use crate::segment::Segment;
use spin::Mutex;

/// Sentinel for an unknown position: `0` is a valid position (stream start), so
/// position uses the maximum value to mean "not known yet".
const POSITION_UNKNOWN: u64 = u64::MAX;
/// Sentinel for an unknown duration: a real duration is always `> 0`, so `0`
/// cleanly means "unknown" (not reported, or a live / open-ended stream). Using
/// `0` lets [`PipelineProgress::publish_duration`] fold sources with `fetch_max`.
const DURATION_UNKNOWN: u64 = 0;

#[derive(Debug)]
struct ProgressInner {
    sink_positions_ns: Mutex<Vec<Arc<AtomicU64>>>,
    // what any sink published last, the answer while every sink is between a flush and its next buffer
    latest_position_ns: AtomicU64,
    duration_ns: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct SinkPosition {
    position_ns: Arc<AtomicU64>,
    progress: Arc<ProgressInner>,
}

impl SinkPosition {
    pub fn set(&self, ns: u64) {
        let ns = known_position(ns);
        self.position_ns.store(ns, Ordering::Relaxed);
        self.progress
            .latest_position_ns
            .store(ns, Ordering::Relaxed);
    }

    pub fn set_at_pts(&self, segment: Option<&Segment>, pts_ns: u64) {
        // a pts outside its segment, or before any segment, stands for its own stream time
        let stream_ns = segment
            .and_then(|segment| segment.to_stream_time(pts_ns))
            .unwrap_or(pts_ns);
        self.set(stream_ns);
    }

    pub fn forget(&self) {
        self.position_ns.store(POSITION_UNKNOWN, Ordering::Relaxed);
    }
}

fn known_position(ns: u64) -> u64 {
    // the absurd ~584-year maximum is the "unknown" sentinel, so a real position stops just below it
    ns.min(POSITION_UNKNOWN - 1)
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
                sink_positions_ns: Mutex::new(Vec::new()),
                latest_position_ns: AtomicU64::new(POSITION_UNKNOWN),
                duration_ns: AtomicU64::new(DURATION_UNKNOWN),
            }),
        }
    }

    /// Application side: the current playback position in nanoseconds (stream
    /// time), or `None` before the first buffer reaches a sink.
    pub fn position(&self) -> Option<u64> {
        // gstreamer's bin answers a position query with the furthest of its sinks
        let sinks = self.inner.sink_positions_ns.lock();
        let furthest = sinks
            .iter()
            .map(|slot| slot.load(Ordering::Relaxed))
            .filter(|&ns| ns != POSITION_UNKNOWN)
            .max();
        let latest = self.inner.latest_position_ns.load(Ordering::Relaxed);
        furthest.or((latest != POSITION_UNKNOWN).then_some(latest))
    }

    pub fn sink_position(&self) -> SinkPosition {
        let position_ns = Arc::new(AtomicU64::new(POSITION_UNKNOWN));
        self.inner
            .sink_positions_ns
            .lock()
            .push(Arc::clone(&position_ns));
        SinkPosition {
            position_ns,
            progress: Arc::clone(&self.inner),
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
    fn a_sink_rewinds_by_publishing_a_smaller_value() {
        let p = PipelineProgress::new();
        let sink = p.sink_position();
        assert_eq!(
            p.position(),
            None,
            "a sink with no buffer yet knows no position"
        );
        sink.set(0);
        assert_eq!(
            p.position(),
            Some(0),
            "zero is a valid position, not unknown"
        );
        sink.set(5_000);
        assert_eq!(p.position(), Some(5_000));
        sink.set(1_000);
        assert_eq!(p.position(), Some(1_000));
    }

    #[test]
    fn position_is_the_furthest_sink_whichever_wrote_last() {
        let p = PipelineProgress::new();
        let audio = p.sink_position();
        let video = p.sink_position();
        audio.set(5_000);
        video.set(4_000);
        assert_eq!(p.position(), Some(5_000));
        video.set(6_000);
        audio.set(5_500);
        assert_eq!(p.position(), Some(6_000));
    }

    #[test]
    fn a_seek_back_lands_once_the_flushed_sinks_publish_again() {
        let p = PipelineProgress::new();
        let audio = p.sink_position();
        let video = p.sink_position();
        audio.set(9_000);
        video.set(8_000);
        audio.forget();
        video.forget();
        assert_eq!(p.position(), Some(8_000), "the last published value holds");
        video.set(1_000);
        assert_eq!(p.position(), Some(1_000));
    }

    #[test]
    fn duration_takes_max_and_reports_change() {
        let p = PipelineProgress::new();
        assert!(
            p.publish_duration(10_000),
            "first known duration is a change"
        );
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
        runner.sink_position().set(7_777);
        runner.publish_duration(99_999);
        assert_eq!(app.position(), Some(7_777));
        assert_eq!(app.duration(), Some(99_999));
    }
}
