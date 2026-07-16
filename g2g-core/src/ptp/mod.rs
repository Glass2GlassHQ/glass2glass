//! PTP (IEEE 1588 / SMPTE ST 2059-2) clock servo, phase A of the `PtpClock`
//! milestone.
//!
//! This is the servo *brain*: it turns PTP message-timestamp exchanges into a
//! disciplined estimate of the grandmaster's timeline, so multiple g2g processes
//! (and eventually machines) can share one clock and present frame-accurately,
//! the sync backbone Pro AV / ST 2110 is built on. It is pure `no_std` math with
//! no network or OS coupling: a caller (a software PTP client, or a thin reader
//! of an OS PHC) feeds it the four timestamps of each delay request-response and
//! reads back a lock state and a master-time estimate. The actual PTP wire
//! protocol is later phases.
//!
//! ## What it computes
//!
//! Each exchange carries four timestamps:
//! - `t1`: Sync sent by the master (master clock),
//! - `t2`: Sync received by us (our reference clock),
//! - `t3`: Delay_Req sent by us (our reference clock),
//! - `t4`: Delay_Req received by the master (master clock).
//!
//! From these, PTP's standard estimators are
//! `offset = ((t2 - t1) - (t4 - t3)) / 2` and
//! `mean_path_delay = ((t2 - t1) + (t4 - t3)) / 2`. The master time at the
//! instant of `t2` is then `t2 - offset`, so we feed `(t2, t2 - offset)` into a
//! [`DriftClock`], which fits our reference clock to the master timeline (both
//! smoothing the per-exchange jitter and estimating our oscillator's rate error
//! against the grandmaster). `now_ns()` then reads the master estimate at the
//! current reference time.
//!
//! Because our reference clock is a monotonic clock with an arbitrary epoch
//! (`metrics::monotonic_ns`) while the master is absolute TAI, the raw `offset`
//! is dominated by that constant epoch gap and is not a useful "sync quality"
//! number; [`error_ns`](PtpServo::error_ns) (how far a fresh sample lands from
//! the fit) is the servo error that hovers near zero once locked, and drives
//! lock detection and outlier rejection.
//!
//! ## Precision note
//!
//! Master timestamps are TAI (~1.7e18 ns), near the edge of `f64`'s exact
//! integer range, so the [`DriftClock`] fit quantises the master anchor to a few
//! hundred ns. That is well under software-timestamp PTP's jitter (10s-100s of
//! us), so it does not bound accuracy here; hardware-timestamped, uncompressed
//! ST 2110-20 timing would want a higher-resolution fit.

use alloc::collections::VecDeque;
use alloc::sync::Arc;

use spin::Mutex;

use crate::clock::{ClockCandidate, ClockPriority, DriftClock, PipelineClock};
use crate::time::{RefNs, TaiNs};

// PTP-over-UDP wire format (M594): parse the messages a SLAVE ordinary clock
// consumes and build the Delay_Req it sends. `no_std`, bounds-checked.
pub mod wire;
// SLAVE-mode delay request-response state machine (M594): assembles a
// (t1,t2,t3,t4) exchange from the message stream, transport-agnostic.
pub mod slave;

pub use slave::{PtpSlave, SlaveAction};
pub use wire::{PtpHeader, PtpMessageType};

/// Lock state of the servo, the signal an election / sink uses to decide whether
/// to trust the master estimate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtpState {
    /// Not yet synchronised: too few samples, or lock was lost. The master
    /// estimate should not be trusted (the servo should not be elected master).
    FreeRunning,
    /// Synchronised: recent samples agree with the fit within the lock
    /// threshold, so `now_ns()` is a good master-time estimate.
    Locked,
    /// Was locked, but the grandmaster has gone silent past the holdover
    /// timeout. `now_ns()` still projects on the last fit (coasting on the
    /// estimated rate), degrading until a fresh sample re-locks.
    Holdover,
}

/// Outcome of feeding one exchange, so a caller / test can see whether the
/// sample was folded in or rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExchangeResult {
    /// Folded into the fit.
    Accepted,
    /// Impossible timestamps (negative path delay, non-positive master time):
    /// dropped without touching the fit.
    RejectedInvalid,
    /// A gross outlier against the current fit while locked (a queued / delayed
    /// packet): dropped so one bad exchange cannot yank the estimate.
    RejectedOutlier,
}

/// A PTP clock servo disciplining a monotonic reference to a grandmaster.
///
/// Feed it exchanges with [`sync_exchange`](Self::sync_exchange); read the master
/// estimate with [`now_ns`](Self::now_ns) and the lock state with
/// [`state`](Self::state). Single-writer (`&mut self` to update); the underlying
/// [`DriftClock`] is itself `Send + Sync`, so a phase-B `PtpClock` can share the
/// estimate as an `Arc<dyn PipelineClock>` while one worker drives the servo.
pub struct PtpServo {
    /// The monotonic reference the master estimate is projected from; must be
    /// the same clock the caller reads `t2` / `t3` from.
    reference: Arc<dyn PipelineClock + Send + Sync>,
    /// The servo core: fits reference time to master time.
    drift: DriftClock,
    /// Recent absolute servo errors (|measured master - fit prediction|), for
    /// lock detection; capped at [`ERROR_WINDOW`](Self::ERROR_WINDOW).
    errors: VecDeque<u64>,
    /// Accepted samples folded into the fit.
    samples: u64,
    /// Reference time of the most recent accepted sample (for staleness).
    last_update_ns: u64,
    /// Signed servo error of the last accepted sample (measured - predicted).
    last_error_ns: i64,
    /// Mean path delay of the last accepted sample.
    last_delay_ns: i64,
    /// Whether the recent window currently satisfies the lock criterion.
    locked: bool,
    /// Whether lock was ever achieved (distinguishes Holdover from FreeRunning).
    ever_locked: bool,
    /// Consecutive outlier rejections while locked; enough of them drops lock.
    consecutive_rejects: u32,
}

impl core::fmt::Debug for PtpServo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PtpServo")
            .field("state", &self.state())
            .field("samples", &self.samples)
            .field("error_ns", &self.last_error_ns)
            .field("mean_path_delay_ns", &self.last_delay_ns)
            .field("slope", &self.drift.slope())
            .finish()
    }
}

impl PtpServo {
    /// Samples required before lock can be declared.
    pub const MIN_LOCK_SAMPLES: u64 = 8;
    /// Recent servo errors kept for the lock decision.
    pub const ERROR_WINDOW: usize = 16;
    /// Max servo error across the window to be considered locked (software
    /// timestamping is good to well under this).
    pub const LOCK_THRESHOLD_NS: u64 = 1_000_000;
    /// A single sample landing this far from the fit while locked is treated as
    /// a delayed / queued packet and dropped, not folded in.
    pub const OUTLIER_GATE_NS: u64 = 20_000_000;
    /// Consecutive outlier rejections that drop lock (the fit is genuinely stale).
    pub const MAX_CONSECUTIVE_REJECTS: u32 = 8;
    /// Reference time without an accepted sample before a locked servo enters
    /// holdover.
    pub const HOLDOVER_TIMEOUT_NS: u64 = 5_000_000_000;

    /// A servo over `reference` (the same monotonic clock the caller timestamps
    /// `t2` / `t3` against).
    pub fn new(reference: Arc<dyn PipelineClock + Send + Sync>) -> Self {
        Self {
            drift: DriftClock::new(reference.clone()),
            reference,
            errors: VecDeque::with_capacity(Self::ERROR_WINDOW),
            samples: 0,
            last_update_ns: 0,
            last_error_ns: 0,
            last_delay_ns: 0,
            locked: false,
            ever_locked: false,
            consecutive_rejects: 0,
        }
    }

    /// Feed one PTP delay request-response exchange. The timestamps are typed by
    /// clock so master and reference can never be swapped: `t1` / `t4` are on the
    /// master clock ([`TaiNs`]), `t2` / `t3` on this servo's monotonic reference
    /// ([`RefNs`]). Returns whether the sample was folded in or why it was dropped.
    pub fn sync_exchange(&mut self, t1: TaiNs, t2: RefNs, t3: RefNs, t4: TaiNs) -> ExchangeResult {
        // Widen to i128: the master-minus-reference differences span the whole
        // TAI-vs-monotonic epoch gap (~1.7e18) and must not overflow.
        let (t1, t2, t3, t4) =
            (t1.get() as i128, t2.get() as i128, t3.get() as i128, t4.get() as i128);
        let master_to_us = t2 - t1;
        let us_to_master = t4 - t3;
        let offset = (master_to_us - us_to_master) / 2;
        let delay = (master_to_us + us_to_master) / 2;

        // A negative path delay or a non-positive master time is physically
        // impossible: reject without disturbing the fit.
        if delay < 0 {
            return ExchangeResult::RejectedInvalid;
        }
        let master = t2 - offset;
        if master <= 0 {
            return ExchangeResult::RejectedInvalid;
        }
        let master = master as u64;
        let local = t2 as u64;

        let result = self.fold(local, master);
        if result == ExchangeResult::Accepted {
            self.last_delay_ns = delay as i64;
        }
        result
    }

    /// Fold a direct `(reference, master)` observation, bypassing the PTP message
    /// math. This is the entry point for a *delegate* backend that already has an
    /// absolute master time, e.g. an OS PTP stack (`linuxptp`) disciplining
    /// `CLOCK_TAI` / a PHC, which a host worker samples against the reference
    /// clock. Same lock / holdover / outlier logic as [`sync_exchange`](Self::sync_exchange);
    /// [`mean_path_delay_ns`](Self::mean_path_delay_ns) is left untouched since a
    /// delegate has no path measurement.
    pub fn observe_master(&mut self, local: RefNs, master: TaiNs) -> ExchangeResult {
        if master.get() == 0 {
            return ExchangeResult::RejectedInvalid;
        }
        self.fold(local.get(), master.get())
    }

    /// Score a `(reference, master)` sample against the current fit and, unless
    /// it is a gross outlier while locked, fold it in and update lock state.
    /// Shared by [`sync_exchange`](Self::sync_exchange) and
    /// [`observe_master`](Self::observe_master).
    fn fold(&mut self, local: u64, master: u64) -> ExchangeResult {
        // Score against the current fit (only once one exists). A gross outlier
        // while locked is a delayed packet / glitch, not a real step: drop it,
        // and only let a run of them break lock.
        let error = if self.samples >= 1 {
            master as i64 - self.drift.project_ns(local) as i64
        } else {
            0
        };
        if self.locked && error.unsigned_abs() > Self::OUTLIER_GATE_NS {
            self.consecutive_rejects += 1;
            if self.consecutive_rejects >= Self::MAX_CONSECUTIVE_REJECTS {
                self.locked = false;
            }
            return ExchangeResult::RejectedOutlier;
        }

        // Fold the sample in.
        self.consecutive_rejects = 0;
        self.drift.observe(local, master);
        self.samples += 1;
        self.last_update_ns = local;
        self.last_error_ns = error;
        if self.samples >= 2 {
            if self.errors.len() == Self::ERROR_WINDOW {
                self.errors.pop_front();
            }
            self.errors.push_back(error.unsigned_abs());
        }

        // Locked once enough samples agree with the fit within the threshold.
        let worst = self.errors.iter().copied().max().unwrap_or(u64::MAX);
        self.locked = self.samples >= Self::MIN_LOCK_SAMPLES
            && !self.errors.is_empty()
            && worst <= Self::LOCK_THRESHOLD_NS;
        if self.locked {
            self.ever_locked = true;
        }
        ExchangeResult::Accepted
    }

    /// The current lock state, accounting for staleness (a locked servo whose
    /// grandmaster has gone silent past the holdover timeout reports
    /// [`Holdover`](PtpState::Holdover)).
    pub fn state(&self) -> PtpState {
        if self.samples < Self::MIN_LOCK_SAMPLES {
            return PtpState::FreeRunning;
        }
        let stale = self.reference.now_ns().saturating_sub(self.last_update_ns)
            > Self::HOLDOVER_TIMEOUT_NS;
        if stale {
            return if self.ever_locked { PtpState::Holdover } else { PtpState::FreeRunning };
        }
        if self.locked {
            PtpState::Locked
        } else {
            PtpState::FreeRunning
        }
    }

    /// Whether the servo is currently [`Locked`](PtpState::Locked).
    pub fn is_locked(&self) -> bool {
        self.state() == PtpState::Locked
    }

    /// The estimated master (grandmaster / TAI) time now, projected from the
    /// reference clock. Before any sample it is the reference passed through.
    pub fn now_ns(&self) -> u64 {
        self.drift.now_ns()
    }

    /// Signed servo error of the last accepted sample: measured master time
    /// minus the fit's prediction. Hovers near zero once locked; the sync-quality
    /// metric (unlike the raw PTP offset, which carries the reference epoch gap).
    pub fn error_ns(&self) -> i64 {
        self.last_error_ns
    }

    /// Estimated one-way path delay from the last accepted sample.
    pub fn mean_path_delay_ns(&self) -> i64 {
        self.last_delay_ns
    }

    /// Estimated rate of the reference clock against the master
    /// (`d(master)/d(reference)`): `1.0` means our oscillator matches the
    /// grandmaster, `1.0000001` that it runs 0.1 ppm slow.
    pub fn slope(&self) -> f64 {
        self.drift.slope()
    }

    /// Accepted samples folded into the fit.
    pub fn samples(&self) -> u64 {
        self.samples
    }
}

/// A shareable, disciplined PTP clock: a [`PtpServo`] behind interior mutability
/// so one worker can drive it (`sync_exchange`) while the pipeline reads the
/// master estimate through an `Arc<dyn PipelineClock>` (M593 phase B).
///
/// A PTP-driven element offers it to clock election with
/// [`candidate`](Self::candidate), which yields a
/// [`PtpGrandmaster`](ClockPriority::PtpGrandmaster) candidate only while the
/// servo is synchronised (`Locked` or `Holdover`), so an unsynchronised clock is
/// never made the pipeline master. Once elected it is the shared timeline every
/// sink slaves to, and, being grandmaster-derived, matches the timeline other
/// machines locked to the same grandmaster read: that is what makes A/V sync
/// hold across devices, not just within one process.
pub struct PtpClock {
    servo: Mutex<PtpServo>,
}

impl core::fmt::Debug for PtpClock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PtpClock").field("servo", &*self.servo.lock()).finish()
    }
}

impl PtpClock {
    /// A PTP clock over `reference` (the monotonic clock the caller timestamps
    /// `t2` / `t3` against), wrapping a fresh [`PtpServo`].
    pub fn new(reference: Arc<dyn PipelineClock + Send + Sync>) -> Self {
        Self { servo: Mutex::new(PtpServo::new(reference)) }
    }

    /// Wrap an existing servo (eg one already driven to lock).
    pub fn from_servo(servo: PtpServo) -> Self {
        Self { servo: Mutex::new(servo) }
    }

    /// Feed one PTP exchange to the servo. Takes `&self` (interior mutability) so
    /// the driving worker can hold the shared `Arc<PtpClock>`. `t1` / `t4` are
    /// master ([`TaiNs`]) times, `t2` / `t3` reference ([`RefNs`]) times.
    pub fn sync_exchange(&self, t1: TaiNs, t2: RefNs, t3: RefNs, t4: TaiNs) -> ExchangeResult {
        self.servo.lock().sync_exchange(t1, t2, t3, t4)
    }

    /// Fold a direct `(reference, master)` observation (delegate backend path);
    /// see [`PtpServo::observe_master`]. Takes `&self` for the shared handle.
    pub fn observe_master(&self, local: RefNs, master: TaiNs) -> ExchangeResult {
        self.servo.lock().observe_master(local, master)
    }

    /// Current servo lock state.
    pub fn state(&self) -> PtpState {
        self.servo.lock().state()
    }

    /// Whether the servo is currently locked.
    pub fn is_locked(&self) -> bool {
        self.servo.lock().is_locked()
    }

    /// Last servo error (fit residual); the sync-quality metric.
    pub fn error_ns(&self) -> i64 {
        self.servo.lock().error_ns()
    }

    /// Estimated reference-vs-master rate.
    pub fn slope(&self) -> f64 {
        self.servo.lock().slope()
    }

    /// Election candidate at the [`PtpGrandmaster`](ClockPriority::PtpGrandmaster)
    /// tier, offered only when the servo is synchronised (`Locked` or
    /// `Holdover`); a `FreeRunning` servo returns `None` so an untrusted clock is
    /// never elected master and the pipeline falls back to a local clock.
    pub fn candidate(self: &Arc<Self>) -> Option<ClockCandidate> {
        if self.state() == PtpState::FreeRunning {
            return None;
        }
        let clock: Arc<dyn PipelineClock + Send + Sync> = self.clone();
        Some(ClockCandidate::new(ClockPriority::PtpGrandmaster, clock))
    }
}

impl PipelineClock for PtpClock {
    fn now_ns(&self) -> u64 {
        self.servo.lock().now_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicU64, Ordering};

    /// A monotonic reference we advance by hand (the servo's local clock).
    #[derive(Debug, Default)]
    struct ManualClock(AtomicU64);
    impl ManualClock {
        fn set(&self, v: u64) {
            self.0.store(v, Ordering::Release);
        }
    }
    impl PipelineClock for ManualClock {
        fn now_ns(&self) -> u64 {
            self.0.load(Ordering::Acquire)
        }
    }

    /// Realistic TAI-scale grandmaster epoch (~2024 in ns since the PTP epoch).
    const EPOCH: i128 = 1_700_000_000_000_000_000;
    /// One-way link delay (100 us), symmetric.
    const DELAY: i128 = 100_000;
    /// Slave gap between receiving Sync and sending Delay_Req (1 ms).
    const GAP: u64 = 1_000_000;

    /// Build the four timestamps of an exchange at reference time `local`, where
    /// the master time is `master(x) = slope*x + EPOCH`. `EPOCH` is kept exact in
    /// i128; only the small `slope*x` term goes through f64.
    fn exchange(local: u64, slope: f64) -> (u64, u64, u64, u64) {
        let master = |x: u64| -> i128 { EPOCH + (slope * x as f64) as i128 };
        let t1 = (master(local) - DELAY) as u64; // Sync left master DELAY ago
        let t2 = local; // we received Sync now
        let t3 = local + GAP; // we send Delay_Req a bit later
        let t4 = (master(local + GAP) + DELAY) as u64; // master received it DELAY later
        (t1, t2, t3, t4)
    }

    /// Drive `count` exchanges at `period` spacing, advancing the reference each
    /// time so staleness stays zero, starting at reference time `start`.
    fn drive(servo: &mut PtpServo, clk: &ManualClock, start: u64, period: u64, count: u64, slope: f64) -> u64 {
        let mut local = start;
        for _ in 0..count {
            clk.set(local);
            let (t1, t2, t3, t4) = exchange(local, slope);
            servo.sync_exchange(TaiNs(t1), RefNs(t2), RefNs(t3), TaiNs(t4));
            local += period;
        }
        local
    }

    #[test]
    fn free_running_before_enough_samples() {
        let clk = Arc::new(ManualClock::default());
        let mut servo = PtpServo::new(clk.clone());
        // A couple of exchanges: not enough to lock.
        drive(&mut servo, &clk, 0, 125_000_000, 3, 1.0);
        assert_eq!(servo.state(), PtpState::FreeRunning);
        assert!(!servo.is_locked());
    }

    #[test]
    fn locks_onto_a_stable_grandmaster() {
        let clk = Arc::new(ManualClock::default());
        let mut servo = PtpServo::new(clk.clone());
        // ~8/s for 3 s.
        let last = drive(&mut servo, &clk, 1_000_000_000, 125_000_000, 24, 1.0);

        assert_eq!(servo.state(), PtpState::Locked, "servo locks onto a steady GM");
        assert!(servo.error_ns().unsigned_abs() < 1_000, "servo error is sub-us: {}", servo.error_ns());
        assert!((servo.slope() - 1.0).abs() < 1e-6, "rate ~1.0: {}", servo.slope());
        // Path delay recovered (~100 us; the tiny drift-over-gap term is < 1 us).
        assert!(
            (servo.mean_path_delay_ns() - DELAY as i64).abs() < 1_000,
            "path delay ~{DELAY}: {}",
            servo.mean_path_delay_ns()
        );

        // now_ns() estimates real TAI: at reference `last`, master ~= EPOCH + last.
        clk.set(last);
        let est = servo.now_ns() as i128;
        let truth = EPOCH + last as i128;
        assert!((est - truth).abs() < 100_000, "master estimate within 100 us: off by {}", est - truth);
    }

    #[test]
    fn tracks_a_fast_local_oscillator() {
        // Our reference runs 100 ppm fast vs the GM: master advances 0.9999 ns
        // per reference ns, so the fitted slope should recover that.
        let clk = Arc::new(ManualClock::default());
        let mut servo = PtpServo::new(clk.clone());
        let slope_true = 0.9999;
        let last = drive(&mut servo, &clk, 1_000_000_000, 125_000_000, 40, slope_true);

        assert_eq!(servo.state(), PtpState::Locked);
        assert!(
            (servo.slope() - slope_true).abs() < 1e-5,
            "recovered oscillator rate {} vs {slope_true}",
            servo.slope()
        );
        // Estimate holds even extrapolating a frame past the last exchange.
        let probe = last + 40_000_000;
        clk.set(probe);
        let est = servo.now_ns() as i128;
        let truth = EPOCH + (slope_true * probe as f64) as i128;
        assert!((est - truth).abs() < 100_000, "drifted-clock estimate off by {}", est - truth);
    }

    #[test]
    fn rejects_an_outlier_and_keeps_lock() {
        let clk = Arc::new(ManualClock::default());
        let mut servo = PtpServo::new(clk.clone());
        let last = drive(&mut servo, &clk, 1_000_000_000, 125_000_000, 24, 1.0);
        assert!(servo.is_locked());

        // A delayed packet: inject 50 ms of extra one-way delay on the Sync leg,
        // which throws the offset far off. It must be rejected, lock preserved.
        // Sample the estimate at the same instant before and after so any change
        // is the fit moving, not the clock advancing.
        clk.set(last);
        let before = servo.now_ns();
        let (t1, t2, t3, t4) = exchange(last, 1.0);
        let bad_t1 = t1 - 50_000_000; // Sync appears to have left 50 ms earlier
        assert_eq!(
            servo.sync_exchange(TaiNs(bad_t1), RefNs(t2), RefNs(t3), TaiNs(t4)),
            ExchangeResult::RejectedOutlier
        );
        assert_eq!(servo.state(), PtpState::Locked, "one outlier does not break lock");
        // The estimate is essentially unmoved (fit untouched).
        assert!((servo.now_ns() as i64 - before as i64).abs() < 1_000_000);
    }

    #[test]
    fn invalid_timestamps_are_rejected() {
        let clk = Arc::new(ManualClock::default());
        let mut servo = PtpServo::new(clk.clone());
        // t4 < t3 with t1 > t2 gives a negative mean path delay: impossible.
        assert_eq!(
            servo.sync_exchange(
                TaiNs(EPOCH as u64 + 1_000_000),
                RefNs(10),
                RefNs(20),
                TaiNs(EPOCH as u64 - 1_000_000)
            ),
            ExchangeResult::RejectedInvalid
        );
        assert_eq!(servo.samples(), 0);
    }

    #[test]
    fn observe_master_locks_from_direct_pairs() {
        // The delegate path: feed absolute (reference, master) pairs directly, as
        // a worker reading an OS PTP-disciplined CLOCK_TAI would, no PTP message
        // math. Master runs 50 ppm slow vs our reference.
        let clk = Arc::new(ManualClock::default());
        let mut servo = PtpServo::new(clk.clone());
        let slope = 0.99995;
        let mut local = 1_000_000_000u64;
        for _ in 0..24 {
            clk.set(local);
            let master = EPOCH + (slope * local as f64) as i128;
            servo.observe_master(RefNs(local), TaiNs(master as u64));
            local += 62_500_000; // ~16 Hz
        }
        assert_eq!(servo.state(), PtpState::Locked, "delegate observations lock the servo");
        assert!((servo.slope() - slope).abs() < 1e-5, "recovered rate {}", servo.slope());
        // No PTP exchange, so no path-delay measurement.
        assert_eq!(servo.mean_path_delay_ns(), 0);

        clk.set(local);
        let est = servo.now_ns() as i128;
        let truth = EPOCH + (slope * local as f64) as i128;
        assert!((est - truth).abs() < 100_000, "master estimate off by {}", est - truth);

        // Zero master time is rejected as invalid.
        assert_eq!(servo.observe_master(RefNs(local), TaiNs(0)), ExchangeResult::RejectedInvalid);
    }

    #[test]
    fn ptp_clock_is_offered_to_election_only_once_locked() {
        let clk = Arc::new(ManualClock::default());
        let ptp = Arc::new(PtpClock::new(clk.clone()));

        // FreeRunning: not offered, so an unsynchronised clock never becomes master.
        assert_eq!(ptp.state(), PtpState::FreeRunning);
        assert!(ptp.candidate().is_none());

        // Drive it to lock through the shared handle (interior mutability).
        let mut local = 1_000_000_000u64;
        for _ in 0..24 {
            clk.set(local);
            let (t1, t2, t3, t4) = exchange(local, 1.0);
            ptp.sync_exchange(TaiNs(t1), RefNs(t2), RefNs(t3), TaiNs(t4));
            local += 125_000_000;
        }
        assert!(ptp.is_locked());
        let cand = ptp.candidate().expect("a locked PTP clock is offered");
        assert_eq!(cand.priority, ClockPriority::PtpGrandmaster);

        // The candidate's clock is the PTP estimate (TAI), not the raw reference.
        clk.set(local);
        assert_eq!(cand.clock.now_ns(), ptp.now_ns());
        assert!(cand.clock.now_ns() > EPOCH as u64, "reads TAI, not monotonic");
    }

    #[test]
    fn enters_holdover_when_the_grandmaster_goes_silent() {
        let clk = Arc::new(ManualClock::default());
        let mut servo = PtpServo::new(clk.clone());
        let last = drive(&mut servo, &clk, 1_000_000_000, 125_000_000, 24, 1.0);
        assert_eq!(servo.state(), PtpState::Locked);

        // No new exchange, but wall time marches past the holdover timeout.
        clk.set(last + PtpServo::HOLDOVER_TIMEOUT_NS + 1);
        assert_eq!(servo.state(), PtpState::Holdover, "silent GM past timeout -> holdover");
        // Still projects a plausible master time (coasting on the last fit).
        let est = servo.now_ns() as i128;
        let truth = EPOCH + (last + PtpServo::HOLDOVER_TIMEOUT_NS) as i128;
        assert!((est - truth).abs() < 1_000_000, "holdover still coasts: off by {}", est - truth);
    }
}
