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
