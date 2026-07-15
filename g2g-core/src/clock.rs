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

/// The process-wide monotonic wall clock ([`monotonic_ns`](crate::metrics::monotonic_ns)),
/// as a shareable [`PipelineClock`]. This is the natural reference clock a
/// [`DriftClock`] projects and the fallback timeline a display sink paces to;
/// several sinks previously each defined their own copy of it. `std`-only
/// (the monotonic source is).
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, Default)]
pub struct MonotonicClock;

#[cfg(feature = "std")]
impl PipelineClock for MonotonicClock {
    fn now_ns(&self) -> u64 {
        crate::metrics::monotonic_ns()
    }
}

/// Election priority of a clock candidate (M12 live clock distribution).
///
/// A pipeline runs against a single clock. When a live element provides one
/// (a camera or RTSP source pacing to a hardware capture clock, an audio sink
/// pacing to its DAC), the pipeline should adopt it over the default system
/// clock so synchronisation follows real capture/playout cadence rather than
/// wall time — GStreamer's clock selection. Higher variants win.
// Closed set: intentionally exhaustive (not #[non_exhaustive]); see STABILITY.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub enum ClockPriority {
    /// The default system / wall clock: the fallback when nothing else
    /// provides a clock.
    #[default]
    SystemFallback,
    /// A non-live element that can drive timing from a monotonic clock (eg a
    /// video display sink pacing to its presentation timeline).
    Provider,
    /// An audio sink pacing to its DAC's real playout rate (M590). Preferred
    /// over a plain [`Provider`](Self::Provider) so audio becomes the master
    /// and video slaves to it (GStreamer's model), but still below a live
    /// capture source, whose hardware clock leads a live pipeline.
    AudioProvider,
    /// A live capture source whose hardware clock should pace the pipeline.
    LiveSource,
    /// A PTP grandmaster-disciplined clock (M593). The shared network reference
    /// every device in a synchronised system (Pro AV / SMPTE ST 2110) slaves to,
    /// so it outranks even a local live-capture clock: when a grandmaster is
    /// present the whole facility, capture included, follows it.
    PtpGrandmaster,
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

/// A disciplined clock that slaves a smooth pipeline timeline to a real
/// hardware playout rate (M590 A/V sync, phase 2).
///
/// The problem it solves: every clock the pipeline elects today is just
/// `monotonic_ns()`, so there is a single wall-clock timeline and true A/V
/// synchronisation cannot exist. An audio sink actually plays samples at its
/// DAC's rate, which drifts from wall time by tens to hundreds of ppm. To make
/// audio the master (GStreamer's model), something has to turn the sink's
/// coarse, jittery "how many samples have really played" readings into a
/// continuous clock the video sink can pace to. That is this type.
///
/// It is fed observations `(local_ns, master_ns)`: `local_ns` sampled from a
/// reference monotonic clock (via [`reference_now`](DriftClock::reference_now))
/// and `master_ns` the true playout position (for audio,
/// `(frames_written - snd_pcm_delay()) * 1e9 / rate`). Over a sliding window it
/// fits `master ≈ slope * local + offset` by least squares, and
/// [`now_ns`](PipelineClock::now_ns) projects the current reference time through
/// that fit. The regression both estimates the playout rate (`slope`, ~1.0
/// plus the drift) and smooths the per-observation jitter, so the timeline the
/// video sink reads is continuous even though the underlying `snd_pcm_delay`
/// readings step.
///
/// Single-writer by contract: one worker (the audio sink) calls
/// [`observe`](DriftClock::observe); any number of sinks call `now_ns`. Both are
/// serialised by an internal spin lock, so it is `Send + Sync` and shares as an
/// `Arc<dyn PipelineClock>` through clock election. Before the first observation
/// it passes the reference clock through unchanged, so it is usable immediately.
#[cfg(feature = "runtime")]
pub struct DriftClock {
    /// The reference monotonic clock the fit is expressed against. `now_ns`
    /// projects `reference.now_ns()`; the writer must sample its `local_ns`
    /// from the same source (see [`reference_now`](DriftClock::reference_now)).
    reference: Arc<dyn PipelineClock + Send + Sync>,
    inner: spin::Mutex<DriftState>,
}

#[cfg(feature = "runtime")]
#[derive(Debug)]
struct DriftState {
    /// Sliding window of `(local_ns, master_ns)` observations, oldest first.
    samples: alloc::collections::VecDeque<(u64, u64)>,
    /// Maximum observations kept for the fit; older ones are evicted.
    capacity: usize,
    /// Published fit, `None` until the first observation. Anchored on the most
    /// recent sample's `local_ns` (exact `u64`) so the large subtraction in the
    /// projection stays precise; `master` and `slope` are `f64`.
    fit: Option<DriftFit>,
}

/// `master_est(local) = master + slope * (local - anchor_local)`.
#[cfg(feature = "runtime")]
#[derive(Clone, Copy, Debug)]
struct DriftFit {
    anchor_local: u64,
    master: f64,
    slope: f64,
}

#[cfg(feature = "runtime")]
impl DriftClock {
    /// Default observation window. At a ~10 Hz discipline cadence this is a few
    /// seconds of history, long enough to average out `snd_pcm_delay` jitter
    /// without lagging a real rate change.
    pub const DEFAULT_WINDOW: usize = 64;

    /// A drift clock over `reference` with the [`DEFAULT_WINDOW`](Self::DEFAULT_WINDOW).
    pub fn new(reference: Arc<dyn PipelineClock + Send + Sync>) -> Self {
        Self::with_window(reference, Self::DEFAULT_WINDOW)
    }

    /// A drift clock keeping the last `window` observations (clamped to at
    /// least 2, since a slope needs two points).
    pub fn with_window(reference: Arc<dyn PipelineClock + Send + Sync>, window: usize) -> Self {
        let capacity = window.max(2);
        Self {
            reference,
            inner: spin::Mutex::new(DriftState {
                samples: alloc::collections::VecDeque::with_capacity(capacity),
                capacity,
                fit: None,
            }),
        }
    }

    /// Sample the reference clock. The disciplining worker must read its
    /// `local_ns` from here so the fit's domain matches what `now_ns` projects.
    pub fn reference_now(&self) -> u64 {
        self.reference.now_ns()
    }

    /// Record one `(local_ns, master_ns)` observation and refit. `local_ns`
    /// must come from [`reference_now`](Self::reference_now); `master_ns` is the
    /// true playout position. Call this from a single worker.
    pub fn observe(&self, local_ns: u64, master_ns: u64) {
        let mut st = self.inner.lock();
        if st.samples.len() == st.capacity {
            st.samples.pop_front();
        }
        st.samples.push_back((local_ns, master_ns));
        st.fit = Some(Self::compute_fit(&st.samples));
    }

    /// The current playout-rate estimate: `d(master)/d(local)`. `1.0` means no
    /// drift; `1.001` means the master runs 0.1% fast relative to the
    /// reference. `1.0` before enough samples exist to estimate it.
    pub fn slope(&self) -> f64 {
        self.inner.lock().fit.map_or(1.0, |f| f.slope)
    }

    /// Number of observations currently in the window. `>= 2` means a real
    /// two-point (or better) rate estimate is in effect rather than the
    /// pass-through / single-point fallback; useful to confirm a live device
    /// has actually disciplined the clock.
    pub fn observations(&self) -> usize {
        self.inner.lock().samples.len()
    }

    /// Project an arbitrary reference time through the current fit, giving the
    /// estimated master time at that reference instant. [`now_ns`](PipelineClock::now_ns)
    /// is this applied to `reference.now_ns()`. Used by a servo (eg PTP) to score
    /// a fresh observation against the fit before folding it in. Identity before
    /// the first observation; a negative projection saturates to `0`.
    pub fn project_ns(&self, local_ns: u64) -> u64 {
        match self.inner.lock().fit {
            None => local_ns,
            Some(f) => {
                let est = f.master + f.slope * (local_ns as i128 - f.anchor_local as i128) as f64;
                if est <= 0.0 {
                    0
                } else {
                    est as u64
                }
            }
        }
    }

    /// Least-squares fit of the window. Centres on the integer means so the
    /// `f64` sums stay well-conditioned, anchors the published fit on the most
    /// recent (exact `u64`) local time.
    fn compute_fit(samples: &alloc::collections::VecDeque<(u64, u64)>) -> DriftFit {
        let n = samples.len();
        let &(anchor_local, anchor_master) = samples.back().expect("observe pushed a sample");

        if n == 1 {
            // One point fixes the offset only; assume no drift until a second
            // observation gives a rate.
            return DriftFit { anchor_local, master: anchor_master as f64, slope: 1.0 };
        }

        // Integer means keep the centred deltas small and exact before the
        // f64 accumulation, avoiding catastrophic cancellation at ~1e18 ns.
        let sum_x: i128 = samples.iter().map(|&(x, _)| x as i128).sum();
        let sum_y: i128 = samples.iter().map(|&(_, y)| y as i128).sum();
        let mean_x = sum_x / n as i128;
        let mean_y = sum_y / n as i128;

        let mut sxx = 0.0f64;
        let mut sxy = 0.0f64;
        for &(x, y) in samples {
            let dx = (x as i128 - mean_x) as f64;
            let dy = (y as i128 - mean_y) as f64;
            sxx += dx * dx;
            sxy += dx * dy;
        }
        // Degenerate spread (all local times equal): fall back to no drift.
        let slope = if sxx > 0.0 { sxy / sxx } else { 1.0 };

        // Evaluate the fitted line at the anchor: master = ȳ + slope*(anchor - x̄).
        let master = mean_y as f64 + slope * (anchor_local as i128 - mean_x) as f64;
        DriftFit { anchor_local, master, slope }
    }
}

#[cfg(feature = "runtime")]
impl PipelineClock for DriftClock {
    fn now_ns(&self) -> u64 {
        // Project the current reference time through the fit (identity until the
        // first observation). See `project_ns`.
        self.project_ns(self.reference.now_ns())
    }
}

#[cfg(feature = "runtime")]
impl core::fmt::Debug for DriftClock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DriftClock")
            .field("slope", &self.slope())
            .field("now_ns", &self.now_ns())
            .finish()
    }
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
        assert!(ClockPriority::PtpGrandmaster > ClockPriority::LiveSource);
        assert!(ClockPriority::LiveSource > ClockPriority::AudioProvider);
        assert!(ClockPriority::AudioProvider > ClockPriority::Provider);
        assert!(ClockPriority::Provider > ClockPriority::SystemFallback);
        assert_eq!(ClockPriority::default(), ClockPriority::SystemFallback);
    }

    #[test]
    fn audio_master_beats_video_but_yields_to_live_capture() {
        // Playback: an audio sink (AudioProvider) outranks a video sink
        // (Provider), so audio becomes the master and video slaves to it.
        let playback = elect_clock([
            cand(ClockPriority::Provider, 1),    // video display sink
            cand(ClockPriority::AudioProvider, 2), // audio sink
        ])
        .unwrap();
        assert_eq!(playback.priority, ClockPriority::AudioProvider);
        assert_eq!(playback.clock.now_ns(), 2);

        // Live capture: the source's hardware clock still leads.
        let live = elect_clock([
            cand(ClockPriority::AudioProvider, 2),
            cand(ClockPriority::LiveSource, 9),
        ])
        .unwrap();
        assert_eq!(live.priority, ClockPriority::LiveSource);
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

#[cfg(all(test, feature = "runtime"))]
mod drift_tests {
    use super::*;

    /// A reference clock we can advance by hand, standing in for the system
    /// monotonic clock the drift clock projects.
    #[derive(Debug, Default)]
    struct Tick(portable_atomic::AtomicU64);
    impl Tick {
        fn set(&self, v: u64) {
            self.0.store(v, Ordering::Release);
        }
    }
    impl PipelineClock for Tick {
        fn now_ns(&self) -> u64 {
            self.0.load(Ordering::Acquire)
        }
    }

    #[test]
    fn passes_reference_through_before_any_observation() {
        let tick = Arc::new(Tick::default());
        let drift = DriftClock::new(tick.clone());
        tick.set(42_000);
        assert_eq!(drift.now_ns(), 42_000, "undisciplined clock is the reference");
        assert_eq!(drift.slope(), 1.0);
    }

    #[test]
    fn converges_to_the_master_playout_rate() {
        // Master runs 0.1% fast (1.001x) relative to the reference, plus a
        // fixed offset, exactly the shape of a DAC drifting from wall time.
        let tick = Arc::new(Tick::default());
        let drift = DriftClock::new(tick.clone());

        const RATE: f64 = 1.001;
        const OFFSET: i64 = 5_000_000;
        // Base the reference well above zero so the f64 conditioning is realistic.
        const BASE: u64 = 1_000_000_000_000_000;
        let master_at = |local: u64| -> u64 {
            (BASE as f64 * RATE + OFFSET as f64
                + RATE * (local - BASE) as f64) as u64
        };

        // Discipline once every 100 ms for a few seconds.
        for i in 0..40u64 {
            let local = BASE + i * 100_000_000;
            tick.set(local);
            drift.observe(drift.reference_now(), master_at(local));
        }

        // Slope should track the 1.001x playout rate closely.
        assert!(
            (drift.slope() - RATE).abs() < 1e-4,
            "slope {} did not converge to {RATE}",
            drift.slope()
        );

        // And the projected timeline should track the true master within a
        // millisecond, including a step *beyond* the last observation (the
        // extrapolation a video sink relies on between discipline ticks).
        let probe = BASE + 40 * 100_000_000 + 33_000_000;
        tick.set(probe);
        let est = drift.now_ns() as i64;
        let truth = master_at(probe) as i64;
        assert!(
            (est - truth).abs() < 1_000_000,
            "projected {est} vs true master {truth} differ by more than 1ms",
        );
    }

    #[test]
    fn a_slaved_reader_tracks_the_disciplined_timeline() {
        // One shared clock: the audio worker disciplines it through the typed
        // handle, a video sink reads it through the Arc<dyn> reader, and both
        // see the same timeline because they are the same object.
        let tick = Arc::new(Tick::default());
        let master = Arc::new(DriftClock::new(tick.clone()));
        let reader: Arc<dyn PipelineClock + Send + Sync> = master.clone();

        const RATE: f64 = 0.9995; // master running slightly slow
        const BASE: u64 = 2_000_000_000_000_000;
        let master_at = |local: u64| -> u64 { (RATE * (local - BASE) as f64) as u64 + BASE };

        for i in 0..30u64 {
            let local = BASE + i * 50_000_000;
            tick.set(local);
            master.observe(master.reference_now(), master_at(local));
        }

        // Advance past the last observation and confirm the slaved reader
        // tracks the master, and that a slow master makes its timeline advance
        // *slower* than the reference.
        let now = BASE + 30 * 50_000_000;
        tick.set(now);
        let reader_view = reader.now_ns() as i64;
        let truth = master_at(now) as i64;
        assert!((reader_view - truth).abs() < 1_000_000, "reader {reader_view} vs {truth}");
        assert!(reader_view < now as i64, "slow master must lag the reference");
    }
}
