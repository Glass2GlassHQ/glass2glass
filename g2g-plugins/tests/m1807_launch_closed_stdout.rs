#![cfg(all(feature = "std", unix))]

use std::io::{BufRead, BufReader};
use std::os::unix::process::ExitStatusExt;
use std::process::{Command, Stdio};

// many more checksum lines than the pipe buffer holds, so the writer meets the closed pipe
const BUFFERS: usize = 100_000;

#[test]
fn closed_stdout_ends_the_run_by_sigpipe_like_gst_launch() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .args(["-q", "videotestsrc", &format!("num-buffers={BUFFERS}")])
        .args(["!", "checksumsink"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("g2g-launch starts");
    let mut stdout = BufReader::new(child.stdout.take().expect("stdout is piped"));
    let mut first_line = String::new();
    stdout.read_line(&mut first_line).expect("a checksum line");
    assert!(!first_line.is_empty(), "the run printed a checksum line");
    drop(stdout);

    let output = child.wait_with_output().expect("g2g-launch ends");
    assert_eq!(
        output.status.signal(),
        Some(libc::SIGPIPE),
        "status {:?}, stderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "nothing printed on stderr");
}

// a write to a socket whose peer reset it raises SIGPIPE, as retina's RTSP keepalive writes do
#[cfg(target_os = "linux")]
#[test]
fn a_running_pipeline_ignores_sigpipe() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .args(["-q", "videotestsrc", "!", "checksumsink"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("g2g-launch starts");
    let mut stdout = BufReader::new(child.stdout.take().expect("stdout is piped"));
    let mut first_line = String::new();
    stdout.read_line(&mut first_line).expect("a checksum line");
    let status = std::fs::read_to_string(format!("/proc/{}/status", child.id()))
        .expect("read the process status");
    child.kill().expect("stop g2g-launch");
    child.wait().expect("g2g-launch ends");

    let ignored_mask = status
        .lines()
        .find_map(|line| line.strip_prefix("SigIgn:"))
        .map(|mask| u64::from_str_radix(mask.trim(), 16).expect("a hex signal mask"))
        .expect("the status lists ignored signals");
    let sigpipe_bit = 1u64 << (libc::SIGPIPE - 1);
    assert_ne!(ignored_mask & sigpipe_bit, 0, "SigIgn {ignored_mask:x}");
}
