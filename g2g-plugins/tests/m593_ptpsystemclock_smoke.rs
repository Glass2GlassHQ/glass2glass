//! M593 phase C: `PtpSystemClock` host smoke test.
//!
//! Drives a `PtpSystemClock` against the host's `CLOCK_TAI` and asserts the
//! delegate backend plumbing works: the worker samples the OS clock, the servo
//! locks, and `now_ns()` reads TAI-scale absolute time that advances with wall
//! time. On a host running `ptp4l` / `phc2sys` this is real grandmaster time; on
//! a plain host `CLOCK_TAI` is still a smooth absolute clock, so the plumbing and
//! lock are exercised either way (see the module docs on what "lock" means here).
//!
//! Run: `cargo test -p g2g-plugins --features ptp --test m593_ptpsystemclock_smoke`.
//! Linux-only (`CLOCK_TAI`); if the clock cannot be read the test skips.
#![cfg(all(target_os = "linux", feature = "ptp"))]

use std::thread::sleep;
use std::time::Duration;

use g2g_plugins::ptpsystemclock::PtpSystemClock;

/// TAI is ns since ~1970; anything past this (~2020-09) confirms we read a real
/// absolute clock rather than the monotonic pass-through of an unstarted servo.
const TAI_FLOOR_NS: u64 = 1_600_000_000_000_000_000;

#[test]
fn ptp_system_clock_locks_onto_the_os_tai_clock() {
    let ptp = PtpSystemClock::new();

    // Let the ~16 Hz worker gather more than the servo's lock window.
    sleep(Duration::from_millis(1200));

    let now = ptp.now_ns();
    if now < TAI_FLOOR_NS {
        // CLOCK_TAI unreadable (servo still free-running on monotonic): skip.
        eprintln!("skip m593 ptpsystemclock: CLOCK_TAI unavailable (now_ns={now})");
        return;
    }

    assert!(ptp.is_locked(), "servo should lock onto the smooth OS clock");
    assert!(
        ptp.candidate().is_some(),
        "a locked PTP system clock is offered to election"
    );
    // The fit tracks the OS clock tightly (sub-ms residual).
    assert!(
        ptp.error_ns().unsigned_abs() < 1_000_000,
        "servo error should be sub-ms: {}",
        ptp.error_ns()
    );

    // It is a live absolute clock: advances by roughly the elapsed real time.
    let t0 = ptp.now_ns();
    sleep(Duration::from_millis(100));
    let t1 = ptp.now_ns();
    let advanced = t1.saturating_sub(t0);
    assert!(
        (50_000_000..250_000_000).contains(&advanced),
        "clock advanced {advanced} ns over a 100 ms sleep (expected ~100 ms)",
    );

    eprintln!(
        "m593 ptpsystemclock: locked, now_ns={} (TAI), error {} ns, advanced {} ms/100 ms",
        t1,
        ptp.error_ns(),
        advanced / 1_000_000
    );
}
