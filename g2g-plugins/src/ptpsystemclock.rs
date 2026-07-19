//! PTP system clock (M593 phase C): a [`PtpClock`] driven from the OS's
//! PTP-disciplined system time, the "works with `linuxptp` today" delegate
//! backend.
//!
//! On a host running the standard PTP stack (`ptp4l` disciplining the NIC's PHC,
//! `phc2sys` copying the PHC onto the system clock), `CLOCK_TAI` tracks the
//! grandmaster. This element samples `(CLOCK_MONOTONIC, CLOCK_TAI)` on a worker
//! thread and feeds the pairs to a [`PtpClock`] via
//! [`observe_master`](g2g_core::PtpClock::observe_master), which fits the
//! monotonic reference to the grandmaster's TAI timeline and reports lock. The
//! `PtpClock` is then offered to clock election at the
//! [`PtpGrandmaster`](g2g_core::ClockPriority::PtpGrandmaster) tier through
//! [`candidate`](PtpSystemClock::candidate), so a whole facility of g2g processes
//! locked to the same grandmaster shares one timeline (see the M593 design).
//!
//! ## Honesty about "lock"
//!
//! This delegates to the OS clock and cannot itself confirm the grandmaster is
//! actually synced: `CLOCK_TAI` is always readable and advances smoothly whether
//! or not `ptp4l` is running (absent it, it is `CLOCK_REALTIME` plus the kernel
//! TAI offset). So "locked" here means the servo is tracking the OS clock
//! consistently, which under a real `ptp4l` / `phc2sys` deployment *is*
//! grandmaster time. Confirming true grandmaster lock independently needs either
//! the in-process software PTP client (M593 phase D) or querying `ptp4l`'s state,
//! a later refinement. Linux-only (`CLOCK_TAI`).

use core::sync::atomic::{AtomicBool, Ordering};

use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::string::String;

use g2g_core::metrics::monotonic_ns;
use g2g_core::{ClockCandidate, MonotonicClock, PipelineClock, PtpClock, PtpState, RefNs, TaiNs};

/// A [`PtpClock`] disciplined from the OS PTP-synced `CLOCK_TAI` by a background
/// worker. Drop stops the worker.
pub struct PtpSystemClock {
    clock: Arc<PtpClock>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl core::fmt::Debug for PtpSystemClock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PtpSystemClock")
            .field("state", &self.state())
            .field("now_ns", &self.now_ns())
            .finish()
    }
}

impl PtpSystemClock {
    /// Default sampling interval (~16 Hz), so a lock forms within ~1 s.
    pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(62);

    /// Start disciplining from `CLOCK_TAI` at [`DEFAULT_INTERVAL`](Self::DEFAULT_INTERVAL).
    pub fn new() -> Self {
        Self::with_interval(Self::DEFAULT_INTERVAL)
    }

    /// As [`new`](Self::new) with an explicit sampling interval.
    pub fn with_interval(interval: Duration) -> Self {
        // The servo's reference and the worker's `local` samples must be the same
        // monotonic source, so the fit's domain matches what `now_ns` projects.
        let reference: Arc<dyn PipelineClock + Send + Sync> = Arc::new(MonotonicClock);
        let clock = Arc::new(PtpClock::new(reference));
        let stop = Arc::new(AtomicBool::new(false));

        let worker_clock = clock.clone();
        let worker_stop = stop.clone();
        let worker = thread::Builder::new()
            .name(String::from("g2g-ptpsysclock"))
            .spawn(move || {
                while !worker_stop.load(Ordering::Relaxed) {
                    // Sample the reference next to CLOCK_TAI so the pair lines up.
                    if let Some(tai) = read_clock_tai() {
                        worker_clock.observe_master(RefNs(monotonic_ns()), TaiNs(tai));
                    }
                    thread::sleep(interval);
                }
            })
            .ok(); // spawn failure leaves the clock free-running (never elected).

        Self {
            clock,
            stop,
            worker,
        }
    }

    /// The disciplined clock, to share via an element's `provide_clock` or read.
    pub fn clock(&self) -> Arc<PtpClock> {
        self.clock.clone()
    }

    /// Election candidate at the `PtpGrandmaster` tier, offered only once the
    /// servo has locked onto the OS clock; `None` while still free-running.
    pub fn candidate(&self) -> Option<ClockCandidate> {
        self.clock.candidate()
    }

    /// Whether the servo has locked onto the OS clock.
    pub fn is_locked(&self) -> bool {
        self.clock.is_locked()
    }

    /// Current servo state.
    pub fn state(&self) -> PtpState {
        self.clock.state()
    }

    /// The grandmaster (TAI) time estimate now.
    pub fn now_ns(&self) -> u64 {
        self.clock.now_ns()
    }

    /// Last servo error (fit residual); the sync-quality metric.
    pub fn error_ns(&self) -> i64 {
        self.clock.error_ns()
    }
}

impl Default for PtpSystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PtpSystemClock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(join) = self.worker.take() {
            let _ = join.join();
        }
    }
}

/// Read `CLOCK_TAI` (the OS PTP-disciplined absolute clock) as nanoseconds, or
/// `None` if the call fails or the value is out of range.
fn read_clock_tai() -> Option<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec`; `clock_gettime` only writes
    // into it and returns 0 on success.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_TAI, &mut ts) };
    if rc != 0 {
        return None;
    }
    let secs = u64::try_from(ts.tv_sec).ok()?;
    let nsec = u64::try_from(ts.tv_nsec).ok()?;
    secs.checked_mul(1_000_000_000)?.checked_add(nsec)
}
