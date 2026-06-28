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
//! so any container seeks correctly without an index, by re-scanning and
//! discarding decoded units until the first keyframe at or after the target time.
//! (An index-derived byte offset is a later performance refinement, not a
//! correctness gap.)
//!
//! Lifecycle, per demuxer `process`:
//! - [`poll_request`](DemuxSeek::poll_request) at the top: a pending flushing
//!   time seek triggers an upstream byte-seek to `0` and enters `AwaitingFlush`.
//! - [`dropping_input`](DemuxSeek::dropping_input): while awaiting the flush, the
//!   demuxer ignores input (the in-flight pre-seek bytes) rather than emit stale
//!   units.
//! - [`on_flush`](DemuxSeek::on_flush) when the source's `Flush` arrives: the
//!   demuxer resets its parser and the helper enters `Discarding`.
//! - [`admit`](DemuxSeek::admit) per decoded unit: drops units until a keyframe
//!   at/after the target, then returns [`Admit::Resume`] (emit a fresh segment
//!   and the unit) and goes idle.
//!
//! [`Mp4Src`]: crate::mp4src
//! [`FileSrc`]: crate::filesrc
//! [`SeekController`]: g2g_core::runtime::SeekController

use g2g_core::runtime::SeekController;
use g2g_core::Seek;

/// What a demuxer should do with a decoded unit (sample / access unit / packet)
/// during seeking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Admit {
    /// Not seeking: emit the unit normally.
    Emit,
    /// Seeking: discard this unit (pre-flush, or before the target keyframe).
    Drop,
    /// Seek complete on this keyframe: emit a fresh segment starting at the
    /// carried stream-time (ns), then the unit. The helper returns to idle.
    Resume(u64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Phase {
    #[default]
    Idle,
    /// Upstream byte-seek requested; drop input until the source's `Flush`.
    AwaitingFlush { target_ns: u64 },
    /// Re-syncing after the flush; drop decoded units until a keyframe >= target.
    Discarding { target_ns: u64 },
}

/// Seek state a demuxer embeds. Inert (`Idle`, zero controllers) unless
/// [`with`](DemuxSeek::with) wired it, so a demuxer with no seek controller pays
/// nothing.
#[derive(Debug, Default)]
pub(crate) struct DemuxSeek {
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
}

impl DemuxSeek {
    /// Wire the app-facing time-seek controller and the upstream byte-seek
    /// controller (the source's). Both are needed to seek.
    pub(crate) fn with(&mut self, app: SeekController, upstream: SeekController) {
        self.app = Some(app);
        self.upstream = Some(upstream);
    }

    /// Poll for a pending flushing time seek, re-scanning from the file start
    /// (offset `0`): correct for any container without an index. A no-op without
    /// both controllers. (The default; [`poll_request_indexed`](Self::poll_request_indexed)
    /// adds an index fast path.)
    pub(crate) fn poll_request(&mut self) -> bool {
        self.poll_request_indexed(|_| None)
    }

    /// Like [`poll_request`](Self::poll_request), but `resolve(target_ns)` may
    /// return an upstream byte offset from an index (e.g. Matroska `Cues`) to seek
    /// directly to the target region. `Some(offset)` seeks there and marks the
    /// seek state-preserving ([`keeps_state`](Self::keeps_state)) since the landing
    /// is mid-stream; `None` falls back to a re-scan from `0` (full reset).
    pub(crate) fn poll_request_indexed(
        &mut self,
        resolve: impl FnOnce(u64) -> Option<u64>,
    ) -> bool {
        if self.phase != Phase::Idle {
            return false;
        }
        let (Some(app), Some(upstream)) = (&self.app, &self.upstream) else {
            return false;
        };
        match app.take_pending() {
            Some(seek) if seek.is_flush() => {
                let (offset, keep) = match resolve(seek.start) {
                    Some(o) => (o, true),
                    None => (0, false),
                };
                upstream.seek(Seek::flush_to(offset));
                self.keep_state = keep;
                self.phase = Phase::AwaitingFlush { target_ns: seek.start };
                true
            }
            _ => false,
        }
    }

    /// Whether the in-flight seek landed mid-stream via an index hit (so the
    /// caller keeps its parser's stream state) rather than re-scanning from `0`.
    pub(crate) fn keeps_state(&self) -> bool {
        self.keep_state
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
        if let Phase::AwaitingFlush { target_ns } = self.phase {
            self.phase = Phase::Discarding { target_ns };
        }
        true
    }

    /// Classify a decoded unit by its stream-time `pts_ns` and whether it is a
    /// keyframe. Idle emits; while discarding, drops until a keyframe at/after the
    /// target, which resumes (and returns to idle).
    pub(crate) fn admit(&mut self, pts_ns: u64, keyframe: bool) -> Admit {
        match self.phase {
            Phase::Idle => Admit::Emit,
            Phase::AwaitingFlush { .. } => Admit::Drop,
            Phase::Discarding { target_ns } => {
                if keyframe && pts_ns >= target_ns {
                    self.phase = Phase::Idle;
                    Admit::Resume(pts_ns)
                } else {
                    Admit::Drop
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::runtime::SeekController;

    #[test]
    fn idle_admits_everything() {
        let mut s = DemuxSeek::default();
        assert_eq!(s.admit(0, false), Admit::Emit);
        assert_eq!(s.admit(1000, true), Admit::Emit);
        assert!(!s.dropping_input());
    }

    #[test]
    fn full_seek_cycle_drops_until_target_keyframe() {
        let app = SeekController::new();
        let upstream = SeekController::new();
        let upstream_src = upstream.clone();
        let mut s = DemuxSeek::default();
        s.with(app.clone(), upstream);

        // App requests a time seek to 5000 ns.
        app.seek(Seek::flush_to(5_000));
        assert!(s.poll_request(), "a pending flushing seek starts the cycle");
        // The upstream source sees a byte-seek to offset 0.
        assert_eq!(upstream_src.take_pending().map(|k| k.start), Some(0));
        // Awaiting the flush: input is dropped and units are not emitted.
        assert!(s.dropping_input());
        assert_eq!(s.admit(0, true), Admit::Drop);

        // The source's flush arrives: reset, then discard until a keyframe >= 5000.
        assert!(s.on_flush());
        assert!(!s.dropping_input());
        assert_eq!(s.admit(0, true), Admit::Drop, "keyframe before target: drop");
        assert_eq!(s.admit(6_000, false), Admit::Drop, "after target but not a keyframe: drop");
        assert_eq!(s.admit(8_000, true), Admit::Resume(8_000), "first keyframe >= target resumes");
        // Back to idle.
        assert_eq!(s.admit(9_000, false), Admit::Emit);
    }

    #[test]
    fn idle_flush_resets_without_seeking() {
        let mut s = DemuxSeek::default();
        // A plain flush (no seek pending) still asks the caller to reset, but
        // stays idle so normal emission continues.
        assert!(s.on_flush());
        assert_eq!(s.admit(0, false), Admit::Emit);
    }
}
