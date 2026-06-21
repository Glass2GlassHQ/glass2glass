use core::future::Future;

use alloc::sync::Arc;

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

/// The pipeline's elected clock plus its base time, handed to a sink so it can
/// present each frame at the right wall-clock moment (the "use PTS to decide
/// when to display" path).
///
/// A frame's presentation deadline on `clock` is `base_time_ns + running_time`,
/// where running time is the frame's `pts_ns` mapped through the active
/// [`Segment`](crate::segment::Segment) (or the PTS directly when no segment is
/// set). `clock` is the [`elected`](elect_clock) pipeline clock; `base_time_ns`
/// is its `now_ns()` sampled when streaming began (running-time zero).
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
    /// `clock.now_ns()` at running-time zero (streaming start / `Playing`).
    pub base_time_ns: u64,
}

impl ClockSync {
    pub fn new(clock: Arc<dyn PipelineClock + Send + Sync>, base_time_ns: u64) -> Self {
        Self { clock, base_time_ns }
    }

    /// Current time on the elected clock.
    pub fn now_ns(&self) -> u64 {
        self.clock.now_ns()
    }
}

impl core::fmt::Debug for ClockSync {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ClockSync")
            .field("base_time_ns", &self.base_time_ns)
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
}
