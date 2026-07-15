//! M594: in-process software PTP client host smoke test.
//!
//! The CI-provable core (wire parser + slave state machine + servo) is unit
//! tested in `g2g-core` (`ptp::wire`, `ptp::slave`, including a full
//! parse -> slave -> servo lock with no sockets). This exercises the real UDP
//! transport, which cannot run in CI: PTP uses privileged ports (319 / 320, need
//! root) and a real lock needs a grandmaster on the network (`ptp4l -m` or
//! hardware). So this test:
//!   - attempts to start a `PtpClient`; if the ports cannot be bound (no
//!     privilege, the normal case in CI / a plain shell) it skips;
//!   - if it starts, runs briefly and, only if it actually locked (a grandmaster
//!     is present), asserts the estimate is TAI-scale and the servo error is sane.
//!
//! Run on a host with a grandmaster: `sudo -E cargo test -p g2g-plugins
//! --features ptp --test m594_ptpclient_smoke -- --nocapture`.
#![cfg(feature = "ptp")]

use std::thread::sleep;
use std::time::Duration;

use g2g_plugins::ptpclient::PtpClient;

/// TAI ns floor (~2020-09): a genuine grandmaster estimate is past this.
const TAI_FLOOR_NS: u64 = 1_600_000_000_000_000_000;

#[test]
fn ptp_client_binds_and_locks_when_a_grandmaster_is_present() {
    let client = match PtpClient::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("skip m594 ptpclient: cannot bind PTP ports (need root): {e}");
            return;
        }
    };

    // Give a grandmaster's Sync/Follow_Up/Delay_Resp stream time to lock us.
    sleep(Duration::from_secs(3));

    if !client.is_locked() {
        eprintln!(
            "skip m594 ptpclient: bound OK but no grandmaster locked (state {:?}); \
             run with ptp4l -m as GM on the LAN",
            client.state()
        );
        return;
    }

    // Locked: it followed a real grandmaster.
    assert!(client.candidate().is_some(), "a locked PTP client is offered to election");
    let now = client.now_ns();
    assert!(now > TAI_FLOOR_NS, "grandmaster estimate should be TAI-scale, got {now}");
    assert!(
        client.error_ns().unsigned_abs() < 1_000_000,
        "servo error should be sub-ms once locked: {}",
        client.error_ns()
    );

    eprintln!("m594 ptpclient: LOCKED to grandmaster, now_ns={now} (TAI), error {} ns", client.error_ns());
}
