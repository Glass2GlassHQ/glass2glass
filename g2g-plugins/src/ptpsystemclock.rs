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
//! [`is_locked`](PtpSystemClock::is_locked) is about the servo only: `CLOCK_TAI`
//! is always readable and advances smoothly whether or not `ptp4l` is running
//! (absent it, it is `CLOCK_REALTIME` plus the kernel TAI offset), so the servo
//! locks either way. Whether that timeline really comes from a grandmaster is a
//! separate question, answered by
//! [`grandmaster_locked`](PtpSystemClock::grandmaster_locked): a second worker
//! asks the local `ptp4l` over its management socket (see [`crate::ptp4l`]) and
//! reports the port state behind the clock, or `None` on a host with no daemon to
//! ask. Election is left on the servo's lock, so a host whose `CLOCK_TAI` is
//! disciplined by something other than `ptp4l` still offers its clock; a caller
//! that needs proof reads `grandmaster_locked` itself. Linux-only (`CLOCK_TAI`).

use core::sync::atomic::{AtomicBool, Ordering};

use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::metrics::monotonic_ns;
use g2g_core::{ClockCandidate, MonotonicClock, PipelineClock, PtpClock, PtpState, RefNs, TaiNs};

use crate::ptp4l::{self, Ptp4lStatus};

/// A [`PtpClock`] disciplined from the OS PTP-synced `CLOCK_TAI` by a background
/// worker, plus a second worker polling the local `ptp4l` for the sync state
/// behind that clock. Drop stops both.
pub struct PtpSystemClock {
    clock: Arc<PtpClock>,
    stop: Arc<AtomicBool>,
    ptp4l_status: Arc<Mutex<Option<Ptp4lStatus>>>,
    workers: Vec<JoinHandle<()>>,
}

impl core::fmt::Debug for PtpSystemClock {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PtpSystemClock")
            .field("state", &self.state())
            .field("now_ns", &self.now_ns())
            .field("grandmaster_locked", &self.grandmaster_locked())
            .finish()
    }
}

impl PtpSystemClock {
    /// Default sampling interval (~16 Hz), so a lock forms within ~1 s.
    pub const DEFAULT_INTERVAL: Duration = Duration::from_millis(62);

    /// How often the `ptp4l` state is re-read. Port states change on the scale of
    /// announce timeouts (seconds), so this is deliberately slow.
    pub const PTP4L_POLL_INTERVAL: Duration = Duration::from_secs(2);

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
        // A spawn failure leaves the clock free-running (never elected), and the
        // ptp4l state unknown.
        let sampler = thread::Builder::new()
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
            .ok();

        let ptp4l_status = Arc::new(Mutex::new(None));
        let poll_status = ptp4l_status.clone();
        let poll_stop = stop.clone();
        let poller = thread::Builder::new()
            .name(String::from("g2g-ptp4lstate"))
            .spawn(move || {
                while !poll_stop.load(Ordering::Relaxed) {
                    let status = ptp4l::query_local_ptp4l();
                    *poll_status.lock().unwrap() = status;
                    sleep_watching_stop(&poll_stop, Self::PTP4L_POLL_INTERVAL);
                }
            })
            .ok();

        Self {
            clock,
            stop,
            ptp4l_status,
            workers: [sampler, poller].into_iter().flatten().collect(),
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

    /// Whether the local `ptp4l` reports a port following a grandmaster, so the
    /// `CLOCK_TAI` this clock reads really is grandmaster time. `None` while no
    /// `ptp4l` answered (none running, or the first poll has not finished).
    pub fn grandmaster_locked(&self) -> Option<bool> {
        Some(self.ptp4l_status()?.locked_to_grandmaster())
    }

    /// The last state the local `ptp4l` reported (its port states and offset from
    /// master), `None` if it has not answered.
    pub fn ptp4l_status(&self) -> Option<Ptp4lStatus> {
        self.ptp4l_status.lock().unwrap().clone()
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
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Sleep for `total`, waking often enough that a stopped worker exits promptly.
fn sleep_watching_stop(stop: &AtomicBool, total: Duration) {
    const TICK: Duration = Duration::from_millis(100);
    let mut slept = Duration::ZERO;
    while slept < total && !stop.load(Ordering::Relaxed) {
        thread::sleep(TICK);
        slept += TICK;
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
