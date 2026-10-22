#![cfg(feature = "std")]

use std::process::Command;
use std::time::{Duration, Instant};

const BUFFERS: usize = 6_400;
// a launcher draining its 64-message bus once per 100 ms tick needs this long for BUFFERS lines
const TIMER_DRAIN_FLOOR: Duration = Duration::from_secs(10);

#[test]
fn checksum_lines_are_drained_as_they_are_posted() {
    let started = Instant::now();
    let output = Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .args(["-q", "audiotestsrc", "samplerate=1000", "channels=1"])
        .arg(format!("num-buffers={BUFFERS}"))
        .args(["!", "checksumsink"])
        .output()
        .expect("g2g-launch runs");
    let elapsed = started.elapsed();
    assert!(
        output.status.success(),
        "status {:?}, stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).lines().count(),
        BUFFERS,
        "one checksum line per buffer"
    );
    assert!(
        elapsed < TIMER_DRAIN_FLOOR / 2,
        "the run took {elapsed:?}, the bus is drained on a timer"
    );
}
