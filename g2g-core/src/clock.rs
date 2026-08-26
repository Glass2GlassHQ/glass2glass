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

    /// This clock as a deadline sleeper, when it is one. The cooperative runner
    /// reads it to derive a fan-in element's tick timer
    /// ([`PipelinePacket::Tick`](crate::PipelinePacket::Tick), M880) from the
    /// pipeline clock itself, so a `parse_launch` line running against any
    /// sleepable clock ticks without a separate entry point.
    ///
    /// Every [`AsyncClock`] implementor should override this with `Some(self)`;
    /// a blanket impl cannot, since `AsyncClock: PipelineClock` means it would
    /// collide with this default. `None` (the default) means the clock only
    /// tells time, so fan-in arms run untimed.
    fn as_ticker(&self) -> Option<&dyn DynAsyncClock> {
        None
    }

    /// The same timer as [`as_ticker`](Self::as_ticker), as an owned shared
    /// handle. The thread-per-arm runner builds each arm's future on its own OS
    /// thread, which a borrow cannot cross, so that is where it reads the tick
    /// from instead.
    ///
    /// Override it on any sleepable clock that is `Send + Sync` and cheap to
    /// clone (`Some(Arc::new(self.clone()))`); a clock with state to share puts
    /// that state behind its own handle so the copy reads the same timeline.
    /// `None` (the default) leaves the threaded arms untimed unless the caller
    /// passes a ticker itself.
    fn shared_ticker(&self) -> Option<Arc<dyn DynAsyncClock + Send + Sync>> {
        None
    }

    /// Whether this clock still holds the reference it disciplines to. A clock
    /// that tells time from a source it cannot lose (a monotonic counter, a
    /// DAC) is always healthy, which is the default; a disciplined clock
    /// ([`PtpClock`](crate::ptp::PtpClock)) reports `false` once its master is
    /// gone. The runner polls the elected clock and, on a loss, posts
    /// [`BusMessage::ClockLost`](crate::BusMessage::ClockLost) and re-elects
    /// over the candidates that are still healthy.
    fn healthy(&self) -> bool {
        true
    }
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

/// Object-safe companion to [`AsyncClock`]: the same absolute-deadline sleep, but
/// returning a [`BoxFuture`](crate::element::BoxFuture) instead of a GAT, so it
/// can be held as `&dyn DynAsyncClock` (the same reason
/// [`DynAsyncElement`](crate::element::DynAsyncElement) mirrors `AsyncElement`).
/// The runner's fan-in arm needs an erased clock to sleep on its tick deadline.
///
/// Blanket-implemented for every `AsyncClock`, so no clock implements it directly.
pub trait DynAsyncClock: PipelineClock {
    fn sleep_until_ns<'a>(&'a self, deadline_ns: u64) -> crate::element::BoxFuture<'a, ()>;
}

impl<T: AsyncClock> DynAsyncClock for T {
    fn sleep_until_ns<'a>(&'a self, deadline_ns: u64) -> crate::element::BoxFuture<'a, ()> {
        alloc::boxed::Box::pin(AsyncClock::sleep_until_ns(self, deadline_ns))
    }
}

/// A shared clock is a clock, so an `Arc<dyn DynAsyncClock + Send + Sync>` can also
/// be handed to an API taking `&dyn PipelineClock` (the threaded runner's ticked
/// entry does exactly that). Upcasting one trait object to its supertrait needs a
/// newer compiler than the MSRV, so the shared handle carries the supertrait itself.
impl<T: PipelineClock + ?Sized> PipelineClock for Arc<T> {
    fn now_ns(&self) -> u64 {
        (**self).now_ns()
    }

    fn as_ticker(&self) -> Option<&dyn DynAsyncClock> {
        (**self).as_ticker()
    }

    fn shared_ticker(&self) -> Option<Arc<dyn DynAsyncClock + Send + Sync>> {
        (**self).shared_ticker()
    }

    fn healthy(&self) -> bool {
        (**self).healthy()
    }
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
        Self {
            inner: Arc::new(AtomicU64::new(Self::UNSET)),
        }
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
        Self {
            clock,
            base_time_ns,
            play_anchor: None,
        }
    }

    /// As [`new`](ClockSync::new), but carries a [`PlayAnchor`] the
    /// `StateController` stamps at `Playing`, so the sink anchors to when
    /// streaming actually began rather than to startup or the preroll frame.
    pub fn with_play_anchor(
        clock: Arc<dyn PipelineClock + Send + Sync>,
        base_time_ns: u64,
        play_anchor: PlayAnchor,
    ) -> Self {
        Self {
            clock,
            base_time_ns,
            play_anchor: Some(play_anchor),
        }
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
    candidates
        .into_iter()
        .flatten()
        .fold(None, |best, c| match best {
            // `>=` keeps the earlier candidate on a priority tie.
            Some(b) if b.priority >= c.priority => Some(b),
            _ => Some(c),
        })
}

/// The elected pipeline clock as the sinks hold it: a stable shared handle whose
/// target can be replaced (M1004).
///
/// [`ClockSync`] is handed to each sink once, before any frame flows, and the
/// runner cannot reach the sinks again afterwards (their elements have moved
/// into the arms, on other threads under the thread-per-arm runner). So when the
/// elected clock degrades and the runner re-elects, the way the new choice
/// reaches every sink is this handle: the `ClockSync` they hold points at the
/// `ElectedClock`, and [`swap`](Self::swap) retargets it in place.
///
/// [`as_ticker`](PipelineClock::as_ticker) stays `None` here: it lends out a
/// borrow of the clock, which a target that can be replaced cannot give.
/// [`shared_ticker`](PipelineClock::shared_ticker) is owned, so it forwards and
/// follows the swap. The runner only installs this indirection when it is going
/// to monitor clock health; otherwise sinks get the elected clock directly.
pub struct ElectedClock {
    target: spin::Mutex<Arc<dyn PipelineClock + Send + Sync>>,
}

impl ElectedClock {
    /// A handle pointing at the freshly elected clock.
    pub fn new(target: Arc<dyn PipelineClock + Send + Sync>) -> Self {
        Self {
            target: spin::Mutex::new(target),
        }
    }

    /// Retarget every holder at `target` (a re-election). Reads after this see
    /// the new clock's timeline; a sink re-anchors on its next frame the same
    /// way it does for any other epoch change.
    pub fn swap(&self, target: Arc<dyn PipelineClock + Send + Sync>) {
        *self.target.lock() = target;
    }

    /// The clock currently pointed at.
    pub fn target(&self) -> Arc<dyn PipelineClock + Send + Sync> {
        self.target.lock().clone()
    }
}

impl PipelineClock for ElectedClock {
    fn now_ns(&self) -> u64 {
        self.target().now_ns()
    }

    fn shared_ticker(&self) -> Option<Arc<dyn DynAsyncClock + Send + Sync>> {
        self.target().shared_ticker()
    }

    fn healthy(&self) -> bool {
        self.target().healthy()
    }
}

impl core::fmt::Debug for ElectedClock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ElectedClock")
            .field("now_ns", &self.now_ns())
            .field("healthy", &self.healthy())
            .finish()
    }
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
/// The fit is exponentially weighted: the newest sample carries weight 1 and
/// each older one [`RECENCY_DECAY`](DriftClock::RECENCY_DECAY) times its
/// successor, so half of a rate step is taken up 24 samples after it rather
/// than 32. The window is however many observations have arrived, up to its
/// capacity, so the slope is usable from the second one.
///
/// A sample landing further than the outlier gate from the current fit is
/// dropped rather than folded in: an underrun recovery or a stale
/// `snd_pcm_delay()` reading would otherwise bend the fit for a whole window.
/// [`MAX_CONSECUTIVE_REJECTS`](DriftClock::MAX_CONSECUTIVE_REJECTS) rejections
/// in a row mean the timeline genuinely moved (a device re-open), so the window
/// is cleared and the fit restarts from the new samples.
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
    /// How far from the current fit a sample may land before it is rejected.
    outlier_gate_ns: u64,
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
    /// Samples rejected by the outlier gate since the last accepted one.
    consecutive_rejects: u32,
}

/// What [`DriftClock::observe`] did with a sample.
#[cfg(feature = "runtime")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriftObservation {
    /// Folded into the fit.
    Folded,
    /// Further from the current fit than the outlier gate: dropped, the fit is
    /// unchanged.
    Rejected,
    /// [`MAX_CONSECUTIVE_REJECTS`](DriftClock::MAX_CONSECUTIVE_REJECTS)
    /// rejections in a row, so the timeline really moved: the window was
    /// cleared and this sample is the first of the new one.
    Restarted,
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

    /// Default outlier gate. An audio sink's `snd_pcm_delay()` jitters by a few
    /// milliseconds in normal running, so 10 ms only catches a real glitch (an
    /// underrun recovery, a stale delay reading).
    pub const DEFAULT_OUTLIER_GATE_NS: u64 = 10_000_000;

    /// Weight of each sample relative to the one after it, newest first. Chosen
    /// against the alternative of no weighting at all: over a full 64-sample
    /// window a rate step is taken up twice as fast (0.29 of it after 16 more
    /// samples, against 0.15 unweighted) for 37% more slope noise. 0.9 doubles
    /// the speed again but costs 2.6x the noise.
    pub const RECENCY_DECAY: f64 = 0.95;

    /// Samples the window must hold before the outlier gate applies. Below this
    /// the slope is still noisy enough that a good sample could be scored as an
    /// outlier against it.
    pub const MIN_SAMPLES_TO_GATE: usize = 8;

    /// Rejections in a row that mean the timeline moved rather than one sample
    /// glitching, so the window restarts.
    pub const MAX_CONSECUTIVE_REJECTS: u32 = 8;

    /// A drift clock over `reference` with the [`DEFAULT_WINDOW`](Self::DEFAULT_WINDOW)
    /// and [`DEFAULT_OUTLIER_GATE_NS`](Self::DEFAULT_OUTLIER_GATE_NS).
    pub fn new(reference: Arc<dyn PipelineClock + Send + Sync>) -> Self {
        Self::with_window(reference, Self::DEFAULT_WINDOW)
    }

    /// A drift clock keeping the last `window` observations (clamped to at
    /// least 2, since a slope needs two points).
    pub fn with_window(reference: Arc<dyn PipelineClock + Send + Sync>, window: usize) -> Self {
        Self::with_window_and_gate(reference, window, Self::DEFAULT_OUTLIER_GATE_NS)
    }

    /// A drift clock whose outlier gate is `outlier_gate_ns` rather than the
    /// default. A servo over a noisier medium sets its own width: the PTP servo
    /// runs at 20 ms, since a queued packet is a much bigger excursion than a
    /// DAC's delay jitter.
    pub fn with_gate(
        reference: Arc<dyn PipelineClock + Send + Sync>,
        outlier_gate_ns: u64,
    ) -> Self {
        Self::with_window_and_gate(reference, Self::DEFAULT_WINDOW, outlier_gate_ns)
    }

    fn with_window_and_gate(
        reference: Arc<dyn PipelineClock + Send + Sync>,
        window: usize,
        outlier_gate_ns: u64,
    ) -> Self {
        let capacity = window.max(2);
        Self {
            reference,
            outlier_gate_ns,
            inner: spin::Mutex::new(DriftState {
                samples: alloc::collections::VecDeque::with_capacity(capacity),
                capacity,
                fit: None,
                consecutive_rejects: 0,
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
    ///
    /// A sample past the outlier gate is dropped and the fit left alone. A run
    /// of them restarts the window. See [`DriftObservation`].
    pub fn observe(&self, local_ns: u64, master_ns: u64) -> DriftObservation {
        let mut st = self.inner.lock();

        let gated = st.samples.len() >= Self::MIN_SAMPLES_TO_GATE
            && st.fit.is_some_and(|fit| {
                (master_ns as f64 - Self::project_through(&fit, local_ns)).abs()
                    > self.outlier_gate_ns as f64
            });
        let mut outcome = DriftObservation::Folded;
        if gated {
            st.consecutive_rejects += 1;
            if st.consecutive_rejects < Self::MAX_CONSECUTIVE_REJECTS {
                return DriftObservation::Rejected;
            }
            st.samples.clear();
            outcome = DriftObservation::Restarted;
        }

        st.consecutive_rejects = 0;
        if st.samples.len() == st.capacity {
            st.samples.pop_front();
        }
        st.samples.push_back((local_ns, master_ns));
        st.fit = Some(Self::compute_fit(&st.samples));
        outcome
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
                let est = Self::project_through(&f, local_ns);
                if est <= 0.0 {
                    0
                } else {
                    est as u64
                }
            }
        }
    }

    fn project_through(fit: &DriftFit, local_ns: u64) -> f64 {
        fit.master + fit.slope * (local_ns as i128 - fit.anchor_local as i128) as f64
    }

    /// Exponentially weighted least-squares fit of the window: sample weights
    /// run `1, RECENCY_DECAY, RECENCY_DECAY², ...` newest to oldest, so recent
    /// observations set the rate while the older tail still lengthens the
    /// baseline. Centres on the integer means so the `f64` sums stay
    /// well-conditioned, anchors the published fit on the most recent (exact
    /// `u64`) local time.
    fn compute_fit(samples: &alloc::collections::VecDeque<(u64, u64)>) -> DriftFit {
        let n = samples.len();
        let &(anchor_local, anchor_master) = samples.back().expect("observe pushed a sample");

        if n == 1 {
            // One point fixes the offset only; assume no drift until a second
            // observation gives a rate.
            return DriftFit {
                anchor_local,
                master: anchor_master as f64,
                slope: 1.0,
            };
        }

        // Integer means keep the centred deltas small and exact before the
        // f64 accumulation, avoiding catastrophic cancellation at ~1e18 ns.
        let sum_x: i128 = samples.iter().map(|&(x, _)| x as i128).sum();
        let sum_y: i128 = samples.iter().map(|&(_, y)| y as i128).sum();
        let mean_x = sum_x / n as i128;
        let mean_y = sum_y / n as i128;
        let centred = |x: u64, y: u64| ((x as i128 - mean_x) as f64, (y as i128 - mean_y) as f64);

        // Weighted centroid first, so the second pass subtracts it exactly
        // rather than recovering it from a difference of large sums.
        let mut weight_total = 0.0f64;
        let mut weighted_x = 0.0f64;
        let mut weighted_y = 0.0f64;
        let mut weight = 1.0f64;
        for &(x, y) in samples.iter().rev() {
            let (dx, dy) = centred(x, y);
            weight_total += weight;
            weighted_x += weight * dx;
            weighted_y += weight * dy;
            weight *= Self::RECENCY_DECAY;
        }
        let centroid_x = weighted_x / weight_total;
        let centroid_y = weighted_y / weight_total;

        let mut sxx = 0.0f64;
        let mut sxy = 0.0f64;
        let mut weight = 1.0f64;
        for &(x, y) in samples.iter().rev() {
            let (dx, dy) = centred(x, y);
            let ex = dx - centroid_x;
            let ey = dy - centroid_y;
            sxx += weight * ex * ex;
            sxy += weight * ex * ey;
            weight *= Self::RECENCY_DECAY;
        }
        // Degenerate spread (all local times equal): fall back to no drift.
        let slope = if sxx > 0.0 { sxy / sxx } else { 1.0 };

        // Evaluate the fitted line at the anchor, in centred coordinates.
        let anchor_dx = (anchor_local as i128 - mean_x) as f64;
        let master = mean_y as f64 + centroid_y + slope * (anchor_dx - centroid_x);
        DriftFit {
            anchor_local,
            master,
            slope,
        }
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
        assert_eq!(
            elected.clock.now_ns(),
            10,
            "first (most upstream) wins a tie"
        );
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
            cand(ClockPriority::Provider, 1),      // video display sink
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
        assert_eq!(
            sync.base_time(),
            100,
            "unstamped anchor uses eager fallback"
        );
        assert!(!sync.play_anchored());

        // Stamped at the play edge: supersedes the eager base time.
        anchor.stamp(7_000);
        assert_eq!(
            sync.base_time(),
            7_000,
            "stamped anchor supersedes eager base"
        );
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
        assert_eq!(
            drift.now_ns(),
            42_000,
            "undisciplined clock is the reference"
        );
        assert_eq!(drift.slope(), 1.0);
    }

    /// Master running 0.1% fast (1.001x) relative to the reference, plus a
    /// fixed offset, exactly the shape of a DAC drifting from wall time. The
    /// reference base is well above zero so the f64 conditioning is realistic.
    const BASE: u64 = 1_000_000_000_000_000;
    const PERIOD: u64 = 100_000_000;
    const RATE: f64 = 1.001;
    const OFFSET: f64 = 5_000_000.0;
    fn master_at(local: u64) -> u64 {
        (BASE as f64 * RATE + OFFSET + RATE * (local - BASE) as f64) as u64
    }

    /// Deterministic pseudo-random jitter in `[-range, range]` (splitmix64),
    /// standing in for the spread of a real `snd_pcm_delay()` reading.
    fn jitter(step: u64, range: i64) -> i64 {
        let mut z = step.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
        z ^= z >> 30;
        z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z % (2 * range as u64 + 1)) as i64 - range
    }

    #[test]
    fn converges_to_the_master_playout_rate() {
        let tick = Arc::new(Tick::default());
        let drift = DriftClock::new(tick.clone());

        // Discipline once every 100 ms for a few seconds.
        for i in 0..40u64 {
            let local = BASE + i * PERIOD;
            tick.set(local);
            drift.observe(drift.reference_now(), master_at(local));
            // The fit must already be usable one second in, not only once the
            // window has filled.
            if i == 9 {
                assert!(
                    (drift.slope() - RATE).abs() < 1e-3,
                    "slope {} after 10 samples is not within 1e-3 of {RATE}",
                    drift.slope()
                );
            }
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
    fn converges_under_delay_jitter() {
        // The bars of `converges_to_the_master_playout_rate`, but on readings
        // that jitter +-200 us the way a real playout position does. Weighting
        // costs slope noise, so the bars have to survive that cost.
        const JITTER_NS: i64 = 200_000;
        let tick = Arc::new(Tick::default());
        let drift = DriftClock::new(tick.clone());

        for i in 0..40u64 {
            let local = BASE + i * PERIOD;
            tick.set(local);
            let master = (master_at(local) as i64 + jitter(i, JITTER_NS)) as u64;
            drift.observe(drift.reference_now(), master);
            if i == 9 {
                assert!(
                    (drift.slope() - RATE).abs() < 1e-3,
                    "jittered slope {} after 10 samples is not within 1e-3 of {RATE}",
                    drift.slope()
                );
            }
        }
        assert!(
            (drift.slope() - RATE).abs() < 1e-4,
            "jittered slope {} after 40 samples is not within 1e-4 of {RATE}",
            drift.slope()
        );
    }

    #[test]
    fn tracks_a_rate_change_before_the_window_turns_over() {
        // What the recency weighting is for. Fill the window at one rate, then
        // move the master to another (a device re-clocking, a resampler
        // changing its ratio) and check how much of the step the slope has
        // taken up a quarter of a window later. An unweighted fit over the same
        // 64 samples reaches 0.15 of the step after 16, the weighted one
        // reaches 0.29, so this bar fails if the weighting goes.
        const FROM: f64 = 1.000;
        const TO: f64 = 1.002;
        const AFTER: u64 = 16;
        const MIN_UPTAKE: f64 = 0.25;

        let tick = Arc::new(Tick::default());
        let drift = DriftClock::new(tick.clone());

        let mut local = BASE;
        let mut master = BASE as f64;
        for _ in 0..DriftClock::DEFAULT_WINDOW {
            tick.set(local);
            drift.observe(local, master as u64);
            local += PERIOD;
            master += FROM * PERIOD as f64;
        }
        for _ in 0..AFTER {
            tick.set(local);
            assert_eq!(
                drift.observe(local, master as u64),
                DriftObservation::Folded,
                "a rate change is not an outlier: its residual stays inside the gate"
            );
            local += PERIOD;
            master += TO * PERIOD as f64;
        }

        let uptake = (drift.slope() - FROM) / (TO - FROM);
        assert!(
            uptake > MIN_UPTAKE,
            "slope {} took up only {uptake} of the {FROM} -> {TO} step after {AFTER} samples",
            drift.slope()
        );
    }

    #[test]
    fn rejects_a_glitched_delay_sample() {
        let tick = Arc::new(Tick::default());
        let drift = DriftClock::new(tick.clone());

        let mut local = BASE;
        for _ in 0..16 {
            tick.set(local);
            assert_eq!(
                drift.observe(local, master_at(local)),
                DriftObservation::Folded
            );
            local += PERIOD;
        }
        let slope_before = drift.slope();
        let observations_before = drift.observations();

        // One glitch: an underrun recovery leaves `delay()` stale, so the
        // playout position reads 50 ms ahead of the line.
        tick.set(local);
        assert_eq!(
            drift.observe(local, master_at(local) + 50_000_000),
            DriftObservation::Rejected
        );
        assert_eq!(
            drift.observations(),
            observations_before,
            "a rejected sample must not enter the window"
        );
        assert!(
            (drift.slope() - slope_before).abs() < 1e-6,
            "slope moved from {slope_before} to {} on a rejected sample",
            drift.slope()
        );

        // The next good reading is folded as normal: one glitch is not a state.
        local += PERIOD;
        tick.set(local);
        assert_eq!(
            drift.observe(local, master_at(local)),
            DriftObservation::Folded
        );
    }

    #[test]
    fn resets_after_a_sustained_jump() {
        // The playout timeline really moves (a device re-open restarts the
        // sample counter), so every reading is off by the same step.
        const JUMP: u64 = 200_000_000;
        let tick = Arc::new(Tick::default());
        let drift = DriftClock::new(tick.clone());

        let mut local = BASE;
        for _ in 0..16 {
            tick.set(local);
            drift.observe(local, master_at(local));
            local += PERIOD;
        }

        for i in 1..DriftClock::MAX_CONSECUTIVE_REJECTS {
            tick.set(local);
            assert_eq!(
                drift.observe(local, master_at(local) + JUMP),
                DriftObservation::Rejected,
                "sample {i} of the jump is still a suspected glitch"
            );
            local += PERIOD;
        }
        tick.set(local);
        assert_eq!(
            drift.observe(local, master_at(local) + JUMP),
            DriftObservation::Restarted,
            "past the reject cap the window has to follow the new timeline"
        );
        assert_eq!(drift.observations(), 1, "the window restarted");

        // Re-disciplined on the new timeline, the clock reads it, not the old
        // one it would still be projecting had the window never reset.
        for _ in 0..16 {
            local += PERIOD;
            tick.set(local);
            assert_eq!(
                drift.observe(local, master_at(local) + JUMP),
                DriftObservation::Folded
            );
        }
        let truth = (master_at(local) + JUMP) as i64;
        let est = drift.now_ns() as i64;
        assert!(
            (est - truth).abs() < 1_000_000,
            "projected {est} vs jumped master {truth}, off by {}",
            est - truth
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
        assert!(
            (reader_view - truth).abs() < 1_000_000,
            "reader {reader_view} vs {truth}"
        );
        assert!(
            reader_view < now as i64,
            "slow master must lag the reference"
        );
    }
}
