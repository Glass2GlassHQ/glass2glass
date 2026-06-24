use core::future::Future;
use core::sync::atomic::Ordering;

use alloc::sync::Arc;

// `portable_atomic` (not `core`) so the 64-bit clock counter compiles on targets
// without native 64-bit atomics (Cortex-M, RISC-V32), same as the metrics
// histogram. Native where available; the `critical-section` feature makes the
// lock-based fallback interrupt-safe on real hardware.
use portable_atomic::AtomicU64;

/// Single source of truth for timestamps within a pipeline.
///
/// All `FrameTiming::pts_ns` / `dts_ns` / `duration_ns` values are expressed
/// relative to the implementation's `now_ns()` epoch. Source elements map
/// their hardware capture clock onto this domain at `configure_pipeline` time.
pub trait PipelineClock {
    fn now_ns(&self) -> u64;
}

/// Pipeline clock with async sleep capability. Used by elements that
/// schedule work against the clock — sync sinks waiting for PTS, paced
/// sources pacing themselves to a target framerate, jitter buffers, etc.
///
/// `sleep_until_ns(deadline)` resolves immediately if `deadline <= now_ns()`.
pub trait AsyncClock: PipelineClock {
    type SleepFuture<'a>: Future<Output = ()> + 'a
    where
        Self: 'a;

    fn sleep_until_ns<'a>(&'a self, deadline_ns: u64) -> Self::SleepFuture<'a>;
}

/// Election priority of a clock candidate (M12 live clock distribution).
///
/// A pipeline runs against a single clock. When a live element provides one
/// (a camera or RTSP source pacing to a hardware capture clock, an audio sink
/// pacing to its DAC), the pipeline should adopt it over the default system
/// clock so synchronisation follows real capture/playout cadence rather than
/// wall time — GStreamer's clock selection. Higher variants win.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub enum ClockPriority {
    /// The default system / wall clock: the fallback when nothing else
    /// provides a clock.
    #[default]
    SystemFallback,
    /// A non-live element that can drive timing (eg an audio sink clock).
    Provider,
    /// A live capture source whose hardware clock should pace the pipeline.
    LiveSource,
}

/// A clock an element offers to the pipeline's clock election, tagged with its
/// [`ClockPriority`]. The `clock` is shared (`Arc`) because the elected clock
/// is distributed to every element that synchronises.
#[derive(Clone)]
pub struct ClockCandidate {
    pub priority: ClockPriority,
    pub clock: Arc<dyn PipelineClock + Send + Sync>,
}

impl ClockCandidate {
    pub fn new(priority: ClockPriority, clock: Arc<dyn PipelineClock + Send + Sync>) -> Self {
        Self { priority, clock }
    }
}

impl core::fmt::Debug for ClockCandidate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClockCandidate")
            .field("priority", &self.priority)
            .field("now_ns", &self.clock.now_ns())
            .finish()
    }
}

/// A presentation base time resolved lazily at the `Playing` transition (M176).
///
/// The eager `ClockSync::base_time_ns` is sampled at runner startup, before the
/// data plane and the `Playing` transition. For a non-live, prerolled pipeline
/// that sits in `Paused` for a while before the application presses play, that
/// is the wrong epoch: the preroll frame is consumed during `Paused`, and a sink
/// that anchored on it then rushes/drops once `Playing` finally arrives. A
/// `PlayAnchor` is a shared cell the [`StateController`](crate::runtime) stamps
/// with `clock.now_ns()` at the exact `Playing` edge, so a sink can anchor
/// presentation to when streaming actually began.
///
/// `u64::MAX` is the unset sentinel (a base time that large is never a real
/// clock reading in this epoch).
#[derive(Clone, Debug, Default)]
pub struct PlayAnchor {
    inner: Arc<AtomicU64>,
}

impl PlayAnchor {
    const UNSET: u64 = u64::MAX;

    /// A fresh, unstamped anchor.
    pub fn new() -> Self {
        Self { inner: Arc::new(AtomicU64::new(Self::UNSET)) }
    }

    /// Stamp the base time (the elected clock's `now_ns()` at the `Playing`
    /// edge). Latest-wins so a re-`Playing` after a stop re-anchors.
    pub fn stamp(&self, base_time_ns: u64) {
        self.inner.store(base_time_ns, Ordering::Release);
    }

    /// Clear the anchor (a transition down to `Ready`/`Null`), so the next
    /// `Playing` re-stamps rather than reusing a stale epoch.
    pub fn clear(&self) {
        self.inner.store(Self::UNSET, Ordering::Release);
    }

    /// The stamped base time, or `None` until `Playing` stamps it.
    pub fn get(&self) -> Option<u64> {
        match self.inner.load(Ordering::Acquire) {
            Self::UNSET => None,
            v => Some(v),
        }
    }
}

/// The pipeline's elected clock plus its base time, handed to a sink so it can
/// present each frame at the right wall-clock moment (the "use PTS to decide
/// when to display" path).
///
/// A frame's presentation deadline on `clock` is `base_time + running_time`,
/// where running time is the frame's `pts_ns` mapped through the active
/// [`Segment`](crate::segment::Segment) (or the PTS directly when no segment is
/// set). `clock` is the [`elected`](elect_clock) pipeline clock. The base time
/// comes from [`base_time`](ClockSync::base_time): the `Playing`-stamped
/// [`PlayAnchor`] once armed and stamped, else the eager `base_time_ns` sampled
/// when streaming began (running-time zero).
///
/// The runner calls [`set_clock_sync`](crate::AsyncElement::set_clock_sync) on
/// each element once, after clock election. A sink that wants to synchronise
/// reads `clock.now_ns()` and waits until it reaches the deadline; a sink that
/// ignores it presents as fast as backpressure allows (the pre-sync behaviour).
#[derive(Clone)]
pub struct ClockSync {
    /// The elected pipeline clock; shared because every synchronising element
    /// reads the same timeline.
    pub clock: Arc<dyn PipelineClock + Send + Sync>,
    /// `clock.now_ns()` at running-time zero, sampled at runner startup. The
    /// eager fallback used when no `Playing` anchor is armed (the non-stateful
    /// runners) or before it is stamped.
    pub base_time_ns: u64,
    /// Optional `Playing`-transition anchor (M176). When armed and stamped it
    /// supersedes `base_time_ns`; `None` on the eager path.
    play_anchor: Option<PlayAnchor>,
}

impl ClockSync {
    /// Eager base time, no `Playing` anchor (the non-stateful runners).
    pub fn new(clock: Arc<dyn PipelineClock + Send + Sync>, base_time_ns: u64) -> Self {
        Self { clock, base_time_ns, play_anchor: None }
    }

    /// As [`new`](ClockSync::new), but carries a [`PlayAnchor`] the
    /// `StateController` stamps at `Playing`, so the sink anchors to when
    /// streaming actually began rather than to startup or the preroll frame.
    pub fn with_play_anchor(
        clock: Arc<dyn PipelineClock + Send + Sync>,
        base_time_ns: u64,
        play_anchor: PlayAnchor,
    ) -> Self {
        Self { clock, base_time_ns, play_anchor: Some(play_anchor) }
    }

    /// Current time on the elected clock.
    pub fn now_ns(&self) -> u64 {
        self.clock.now_ns()
    }

    /// The presentation base time: the `Playing`-stamped anchor when armed and
    /// stamped, otherwise the eager `base_time_ns`. A sink reads this each frame
    /// so that, once `Playing` stamps the anchor, deadlines re-base onto the
    /// play epoch.
    pub fn base_time(&self) -> u64 {
        match &self.play_anchor {
            Some(a) => a.get().unwrap_or(self.base_time_ns),
            None => self.base_time_ns,
        }
    }

    /// Whether a `Playing` anchor is armed and has been stamped. A sink uses
    /// this to decide whether to trust [`base_time`](ClockSync::base_time) as a
    /// real anchor or fall back to first-frame anchoring until `Playing`.
    pub fn play_anchored(&self) -> bool {
        self.play_anchor.as_ref().is_some_and(|a| a.get().is_some())
    }
}

impl core::fmt::Debug for ClockSync {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClockSync")
            .field("base_time_ns", &self.base_time_ns)
            .field("play_anchored", &self.play_anchored())
            .field("now_ns", &self.clock.now_ns())
            .finish()
    }
}

/// Elect the pipeline clock from a set of candidates (most upstream first):
/// the highest-priority candidate wins, ties resolve to the earliest (most
/// upstream) one. `None` means no element offered a clock, so the caller's
/// fallback system clock stands.
pub fn elect_clock<I>(candidates: I) -> Option<ClockCandidate>
where
    I: IntoIterator<Item = Option<ClockCandidate>>,
{
    candidates.into_iter().flatten().fold(None, |best, c| match best {
        // `>=` keeps the earlier candidate on a priority tie.
        Some(b) if b.priority >= c.priority => Some(b),
        _ => Some(c),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fixed(u64);
    impl PipelineClock for Fixed {
        fn now_ns(&self) -> u64 {
            self.0
        }
    }

    fn cand(priority: ClockPriority, now: u64) -> Option<ClockCandidate> {
        Some(ClockCandidate::new(priority, Arc::new(Fixed(now))))
    }

    #[test]
    fn elects_highest_priority() {
        let elected = elect_clock([
            cand(ClockPriority::SystemFallback, 1),
            cand(ClockPriority::LiveSource, 5),
            cand(ClockPriority::Provider, 3),
        ])
        .expect("a candidate must win");
        assert_eq!(elected.priority, ClockPriority::LiveSource);
        assert_eq!(elected.clock.now_ns(), 5);
    }

    #[test]
    fn no_candidates_elects_nothing() {
        assert!(elect_clock([None, None]).is_none());
        assert!(elect_clock(core::iter::empty()).is_none());
    }

    #[test]
    fn ties_resolve_to_earliest() {
        let elected = elect_clock([
            cand(ClockPriority::Provider, 10),
            cand(ClockPriority::Provider, 20),
        ])
        .unwrap();
        assert_eq!(elected.clock.now_ns(), 10, "first (most upstream) wins a tie");
    }

    #[test]
    fn priority_is_ordered() {
        assert!(ClockPriority::LiveSource > ClockPriority::Provider);
        assert!(ClockPriority::Provider > ClockPriority::SystemFallback);
        assert_eq!(ClockPriority::default(), ClockPriority::SystemFallback);
    }

    #[test]
    fn play_anchor_resolves_base_time() {
        // Eager `ClockSync` (no anchor) always reports its startup base time.
        let eager = ClockSync::new(Arc::new(Fixed(0)), 100);
        assert_eq!(eager.base_time(), 100);
        assert!(!eager.play_anchored());

        // Armed but unstamped: falls back to the eager base time, not yet
        // play-anchored (so a sink first-frame-anchors until `Playing`).
        let anchor = PlayAnchor::new();
        let sync = ClockSync::with_play_anchor(Arc::new(Fixed(7_000)), 100, anchor.clone());
        assert_eq!(sync.base_time(), 100, "unstamped anchor uses eager fallback");
        assert!(!sync.play_anchored());

        // Stamped at the play edge: supersedes the eager base time.
        anchor.stamp(7_000);
        assert_eq!(sync.base_time(), 7_000, "stamped anchor supersedes eager base");
        assert!(sync.play_anchored());

        // Cleared (a stop): back to the eager fallback until the next stamp.
        anchor.clear();
        assert_eq!(sync.base_time(), 100);
        assert!(!sync.play_anchored());
    }
}
