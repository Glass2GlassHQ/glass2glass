//! Shared seek helper for byte-stream demuxers (M362).
//!
//! A demuxer (`tsdemux`, `fmp4demux`, `mkvdemux`, `flvdemux`, `oggdemux`) is a
//! transform fed a byte stream by an upstream source; unlike [`Mp4Src`] it has no
//! random access of its own. It becomes seek-aware by driving its upstream byte
//! source: on an app time seek (its own [`SeekController`]) it asks the source
//! (typically [`FileSrc`], holding a clone of the upstream controller) to
//! reposition by byte offset, then re-syncs from the returned `Flush`.
//!
//! Completeness over speed: the byte-seek target is the file start (offset `0`),
//! so any container seeks correctly without an index, by re-scanning decoded
//! units up to the target time. (An index-derived byte offset is a later
//! performance refinement, not a correctness gap.)
//!
//! Lifecycle, per demuxer `process`:
//! - [`poll_request`](DemuxSeek::poll_request) at the top: a pending flushing
//!   time seek triggers an upstream byte-seek to `0` and enters `AwaitingFlush`.
//! - [`dropping_input`](DemuxSeek::dropping_input): while awaiting the flush, the
//!   demuxer ignores input (the in-flight pre-seek bytes) rather than emit stale
//!   units.
//! - [`on_flush`](DemuxSeek::on_flush) when the source's `Flush` arrives: the
//!   demuxer resets its parser and the helper enters `Discarding`.
//! - [`admit`](DemuxSeek::admit) per decoded unit: keeps the units from the last
//!   keyframe at or before the target and, once a unit at or after the target
//!   arrives, hands them back with a segment start at the target, as `Mp4Src`
//!   does, so the decoder drops the frames before it. Then it goes idle.
//!
//! [`Mp4Src`]: crate::mp4src
//! [`FileSrc`]: crate::filesrc
//! [`SeekController`]: g2g_core::runtime::SeekController

use alloc::vec::Vec;
use core::iter::Chain;

use g2g_core::runtime::SeekController;
use g2g_core::Seek;

// a stream without a keyframe for this many units releases them early, the segment still clips them
const KEPT_UNITS_MAX: usize = 1024;

#[derive(Debug)]
pub(crate) struct Admitted<U> {
    // set when a seek completes: the caller sends a segment starting here before the units
    pub(crate) segment_start_ns: Option<u64>,
    units: Chain<alloc::vec::IntoIter<U>, core::option::IntoIter<U>>,
}

impl<U> Admitted<U> {
    pub(crate) fn emit(unit: U) -> Self {
        Self::release(None, Vec::new(), Some(unit))
    }

    fn empty() -> Self {
        Self::release(None, Vec::new(), None)
    }

    fn release(segment_start_ns: Option<u64>, kept: Vec<U>, unit: Option<U>) -> Self {
        Self {
            segment_start_ns,
            units: kept.into_iter().chain(unit),
        }
    }
}

impl<U> Iterator for Admitted<U> {
    type Item = U;

    fn next(&mut self) -> Option<U> {
        self.units.next()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Phase {
    #[default]
    Idle,
    /// Upstream byte-seek requested; drop input until the source's `Flush`.
    AwaitingFlush { target: SeekTarget },
    /// Re-syncing after the flush; keep the units from the last keyframe at or
    /// before the target until a unit at or after it arrives.
    Discarding { target: SeekTarget },
}

// a seek target counts from the stream's first pts, as ffmpeg -ss counts from the start time
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeekTarget {
    Pts(u64),
    // taken before the stream's first unit, so the pts is known only once that unit arrives
    FromStreamStart(u64),
}

/// Seek state a demuxer embeds, generic over the demuxer's decoded unit. Inert
/// (`Idle`, zero controllers) unless [`with`](DemuxSeek::with) wired it, so a
/// demuxer with no seek controller pays nothing.
#[derive(Debug)]
pub(crate) struct DemuxSeek<U> {
    /// App -> demuxer time seeks.
    app: Option<SeekController>,
    /// Demuxer -> upstream byte source (e.g. `FileSrc`) byte-offset seeks.
    upstream: Option<SeekController>,
    phase: Phase,
    /// Whether the in-flight seek resolved to a real byte offset (an index hit,
    /// `poll_request_indexed`) rather than a re-scan from `0`. The caller reads it
    /// in its `Flush` handler to decide whether to keep mid-stream parser state
    /// (a mid-segment landing) or fully reset (a from-start re-scan).
    keep_state: bool,
    // empty until the first keyframe after the flush, since nothing before it decodes
    kept: Vec<U>,
    stream_start_ns: Option<u64>,
}

impl<U> Default for DemuxSeek<U> {
    fn default() -> Self {
        Self {
            app: None,
            upstream: None,
            phase: Phase::Idle,
            keep_state: false,
            kept: Vec::new(),
            stream_start_ns: None,
        }
    }
}

impl<U> DemuxSeek<U> {
    /// Wire the app-facing time-seek controller and the upstream byte-seek
    /// controller (the source's). Both are needed to seek.
    pub(crate) fn with(&mut self, app: SeekController, upstream: SeekController) {
        self.app = Some(app);
        self.upstream = Some(upstream);
        // a seek queued before the run reaches the source before its first chunk
        self.poll_request();
    }

    /// Poll for a pending flushing time seek, re-scanning from the file start
    /// (offset `0`): correct for any container without an index. A no-op without
    /// both controllers. (The default; [`poll_request_indexed`](Self::poll_request_indexed)
    /// adds an index fast path.)
    pub(crate) fn poll_request(&mut self) -> bool {
        self.poll_request_indexed(|_| None)
    }

    /// Like [`poll_request`](Self::poll_request), but `resolve(target_pts_ns)` may
    /// return an upstream byte offset from an index (e.g. Matroska `Cues`) to seek
    /// directly to the target region. `Some(offset)` seeks there and marks the
    /// seek state-preserving ([`keeps_state`](Self::keeps_state)) since the landing
    /// is mid-stream; `None` falls back to a re-scan from `0` (full reset).
    pub(crate) fn poll_request_indexed(
        &mut self,
        resolve: impl FnOnce(u64) -> Option<u64>,
    ) -> bool {
        if self.phase != Phase::Idle || self.upstream.is_none() {
            return false;
        }
        let Some(app) = &self.app else {
            return false;
        };
        let Some(seek) = app.take_pending().filter(|seek| seek.is_flush()) else {
            return false;
        };
        // an index lookup needs the target as a pts, which the stream's first unit fixes
        let Some(stream_start_ns) = self.stream_start_ns else {
            self.await_flush(SeekTarget::FromStreamStart(seek.start), 0, false);
            return true;
        };
        let target_pts_ns = seek.start.saturating_add(stream_start_ns);
        match resolve(target_pts_ns) {
            Some(offset) => self.begin_seek_at(target_pts_ns, offset, true),
            None => self.begin_seek_at(target_pts_ns, 0, false),
        }
        true
    }

    // takes the pts of a unit of any track, the first one is where seek targets count from
    pub(crate) fn note_stream_start(&mut self, pts_ns: u64) {
        self.stream_start_ns.get_or_insert(pts_ns);
    }

    pub(crate) fn stream_start_ns(&self) -> Option<u64> {
        self.stream_start_ns
    }

    /// The upstream stream length in bytes, when the source published one, so a
    /// caller guessing a byte offset can keep it below the end.
    pub(crate) fn upstream_len(&self) -> Option<u64> {
        self.upstream.as_ref().and_then(|c| c.stream_len())
    }

    /// Whether the in-flight seek landed mid-stream via an index hit (so the
    /// caller keeps its parser's stream state) rather than re-scanning from `0`.
    pub(crate) fn keeps_state(&self) -> bool {
        self.keep_state
    }

    /// Whether a seek is in flight (awaiting its flush or discarding to the
    /// target). The caller skips starting a new one until this clears.
    pub(crate) fn is_seeking(&self) -> bool {
        self.phase != Phase::Idle
    }

    /// Begin an indexed seek to an already-resolved byte `offset` for the pts
    /// `target_pts_ns`, when the caller consumed the app seek itself (the multi-hop
    /// Matroska `Cues` prefetch: the `Cues` are fetched first, then this seeks to
    /// the target Cluster). Issues the upstream byte-seek, marks the seek
    /// state-preserving, and awaits its flush (so `admit` then re-syncs to the target).
    pub(crate) fn begin_indexed_seek(&mut self, target_pts_ns: u64, offset: u64) {
        self.begin_seek_at(target_pts_ns, offset, true);
    }

    /// Begin (or restart) a byte-seek to `offset` for the pts `target_pts_ns` the
    /// caller resolved itself. `keep_state` marks the landing mid-stream
    /// ([`keeps_state`](Self::keeps_state)); `false` is a re-scan from the file
    /// start, the fallback when a resolved offset turns out not to serve the
    /// target (the Ogg proportional guess, M862).
    pub(crate) fn begin_seek_at(&mut self, target_pts_ns: u64, offset: u64, keep_state: bool) {
        self.await_flush(SeekTarget::Pts(target_pts_ns), offset, keep_state);
    }

    fn await_flush(&mut self, target: SeekTarget, offset: u64, keep_state: bool) {
        if let Some(upstream) = &self.upstream {
            upstream.seek(Seek::flush_to(offset));
        }
        self.keep_state = keep_state;
        self.kept.clear();
        self.phase = Phase::AwaitingFlush { target };
    }

    /// Whether the demuxer should ignore input now (awaiting the upstream flush,
    /// so the in-flight pre-seek bytes must not produce output).
    pub(crate) fn dropping_input(&self) -> bool {
        matches!(self.phase, Phase::AwaitingFlush { .. })
    }

    /// Handle a `Flush` packet. If it is the upstream flush we asked for, advance
    /// to discarding and return `true` (the caller resets its parser). A `Flush`
    /// while idle also returns `true` (a discontinuity always resets the parser),
    /// without changing phase.
    pub(crate) fn on_flush(&mut self) -> bool {
        if let Phase::AwaitingFlush { target } = self.phase {
            self.phase = Phase::Discarding { target };
        }
        self.kept.clear();
        true
    }

    /// Classify a decoded `unit` by its `pts_ns` and whether it is a keyframe.
    /// Idle emits it. While discarding, a keyframe at or before the target
    /// restarts the kept units, and the first unit at or after the target
    /// releases them with the segment start, after which the helper is idle.
    pub(crate) fn admit(&mut self, pts_ns: u64, keyframe: bool, unit: U) -> Admitted<U> {
        let stream_start_ns = *self.stream_start_ns.get_or_insert(pts_ns);
        let target_ns = match self.phase {
            Phase::Idle => return Admitted::emit(unit),
            Phase::AwaitingFlush { .. } => return Admitted::empty(),
            Phase::Discarding {
                target: SeekTarget::Pts(target_ns),
            } => target_ns,
            Phase::Discarding {
                target: SeekTarget::FromStreamStart(target_ns),
            } => target_ns.saturating_add(stream_start_ns),
        };
        if keyframe && pts_ns <= target_ns {
            self.kept.clear();
        } else if self.kept.is_empty() && !keyframe {
            return Admitted::empty();
        }
        if pts_ns < target_ns && self.kept.len() < KEPT_UNITS_MAX {
            self.kept.push(unit);
            return Admitted::empty();
        }
        self.phase = Phase::Idle;
        Admitted::release(Some(target_ns), core::mem::take(&mut self.kept), Some(unit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use g2g_core::runtime::SeekController;

    const TARGET_NS: u64 = 5_000;

    fn units(admitted: Admitted<u64>) -> (Option<u64>, Vec<u64>) {
        (admitted.segment_start_ns, admitted.collect())
    }

    fn discarding_to_target(stream_start_ns: u64) -> DemuxSeek<u64> {
        let app = SeekController::new();
        let mut s = DemuxSeek::default();
        s.with(app.clone(), SeekController::new());
        s.note_stream_start(stream_start_ns);
        app.seek(Seek::flush_to(TARGET_NS));
        assert!(s.poll_request(), "a pending flushing seek starts the cycle");
        assert!(s.on_flush());
        s
    }

    #[test]
    fn idle_admits_everything() {
        let mut s = DemuxSeek::default();
        assert_eq!(units(s.admit(0, false, 1)), (None, vec![1]));
        assert_eq!(units(s.admit(1000, true, 2)), (None, vec![2]));
        assert!(!s.dropping_input());
    }

    #[test]
    fn full_seek_cycle_releases_from_the_keyframe_before_the_target() {
        let app = SeekController::new();
        let upstream = SeekController::new();
        let upstream_src = upstream.clone();
        let mut s = DemuxSeek::default();
        s.with(app.clone(), upstream);

        app.seek(Seek::flush_to(TARGET_NS));
        assert!(s.poll_request(), "a pending flushing seek starts the cycle");
        // The upstream source sees a byte-seek to offset 0.
        assert_eq!(upstream_src.take_pending().map(|k| k.start), Some(0));
        // Awaiting the flush: input is dropped and units are not emitted.
        assert!(s.dropping_input());
        assert_eq!(units(s.admit(0, true, 0)), (None, vec![]));

        assert!(s.on_flush());
        assert!(!s.dropping_input());
        assert_eq!(units(s.admit(0, true, 0)), (None, vec![]));
        assert_eq!(units(s.admit(1_000, false, 1_000)), (None, vec![]));
        // a later keyframe before the target restarts the kept units
        assert_eq!(units(s.admit(4_000, true, 4_000)), (None, vec![]));
        assert_eq!(units(s.admit(4_500, false, 4_500)), (None, vec![]));
        assert_eq!(
            units(s.admit(6_000, false, 6_000)),
            (Some(TARGET_NS), vec![4_000, 4_500, 6_000]),
            "the first unit at or after the target releases the kept ones"
        );
        // reordered B frames below the target follow in decode order, the segment clips them
        assert_eq!(units(s.admit(4_800, false, 4_800)), (None, vec![4_800]));
    }

    #[test]
    fn units_before_the_first_keyframe_never_go_out() {
        let mut s = discarding_to_target(0);
        assert_eq!(units(s.admit(1_000, false, 1_000)), (None, vec![]));
        assert_eq!(units(s.admit(6_000, false, 6_000)), (None, vec![]));
        assert_eq!(
            units(s.admit(7_000, true, 7_000)),
            (Some(TARGET_NS), vec![7_000]),
            "a landing past the target resumes on its first keyframe"
        );
    }

    #[test]
    fn a_keyframe_on_the_target_starts_there() {
        let mut s = discarding_to_target(0);
        assert_eq!(units(s.admit(0, true, 0)), (None, vec![]));
        assert_eq!(
            units(s.admit(TARGET_NS, true, TARGET_NS)),
            (Some(TARGET_NS), vec![TARGET_NS])
        );
    }

    #[test]
    fn a_long_run_without_keyframes_releases_early() {
        let mut s = discarding_to_target(0);
        assert_eq!(units(s.admit(0, true, 0)), (None, vec![]));
        for unit in 1..KEPT_UNITS_MAX as u64 {
            assert_eq!(units(s.admit(1, false, unit)), (None, vec![]));
        }
        let (segment_start, released) = units(s.admit(1, false, KEPT_UNITS_MAX as u64));
        assert_eq!(segment_start, Some(TARGET_NS));
        assert_eq!(released, (0..=KEPT_UNITS_MAX as u64).collect::<Vec<_>>());
    }

    #[test]
    fn the_target_counts_from_the_first_unit() {
        const STREAM_START_NS: u64 = 1_000;
        let mut s = discarding_to_target(STREAM_START_NS);
        // a later unit does not move the start
        s.note_stream_start(0);
        assert_eq!(units(s.admit(STREAM_START_NS, true, 0)), (None, vec![]));
        assert_eq!(
            units(s.admit(TARGET_NS, false, 1)),
            (None, vec![]),
            "the target pts is the start plus the seek time"
        );
        assert_eq!(
            units(s.admit(TARGET_NS + STREAM_START_NS, false, 2)),
            (Some(TARGET_NS + STREAM_START_NS), vec![0, 1, 2])
        );
    }

    #[test]
    fn an_index_lookup_waits_for_the_stream_start() {
        const STREAM_START_NS: u64 = 1_000;
        const INDEXED_OFFSET: u64 = 4_096;
        let app = SeekController::new();
        let upstream = SeekController::new();
        let mut s: DemuxSeek<u64> = DemuxSeek::default();
        s.with(app.clone(), upstream.clone());

        app.seek(Seek::flush_to(TARGET_NS));
        assert!(s.poll_request_indexed(|_| panic!("no pts to look up yet")));
        assert_eq!(upstream.take_pending().map(|seek| seek.start), Some(0));
        assert!(!s.keeps_state(), "a re-scan from the file start");
        s.on_flush();
        s.admit(STREAM_START_NS, true, 0);
        s.admit(TARGET_NS + STREAM_START_NS, true, 1);

        let mut looked_up = None;
        app.seek(Seek::flush_to(TARGET_NS));
        assert!(s.poll_request_indexed(|target_pts_ns| {
            looked_up = Some(target_pts_ns);
            Some(INDEXED_OFFSET)
        }));
        assert_eq!(looked_up, Some(TARGET_NS + STREAM_START_NS));
        assert_eq!(
            upstream.take_pending().map(|seek| seek.start),
            Some(INDEXED_OFFSET)
        );
        assert!(s.keeps_state());
    }

    #[test]
    fn idle_flush_resets_without_seeking() {
        let mut s = DemuxSeek::default();
        // A plain flush (no seek pending) still asks the caller to reset, but
        // stays idle so normal emission continues.
        assert!(s.on_flush());
        assert_eq!(units(s.admit(0, false, 0)), (None, vec![0]));
    }
}
