//! M1015: a file element that cannot open its path logs the path and the OS
//! message. `Hardware(Io(errno))` alone says neither, so the error a run reports
//! is only actionable together with this line.
#![cfg(feature = "std")]

use core::task::Poll;
use std::sync::{Mutex, MutexGuard};

use g2g_core::element::PushOutcome;
use g2g_core::log::{self, LogLevel, RingSink};
use g2g_core::runtime::{block_on, SourceLoop};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, G2gError, HardwareError, OutputSink, PipelinePacket,
};
use g2g_plugins::filesink::FileSink;
use g2g_plugins::filesrc::FileSrc;

/// A path no build can open: the parent exists but is not a directory.
const MISSING_FILE: &str = "/dev/null/g2g-m1015-missing.ts";
/// A path no build can create: `/proc` takes no new files.
const UNWRITABLE_FILE: &str = "/proc/g2g-m1015-unwritable.ts";

/// The log sink is process-global, so a test that installs one must not run
/// alongside another that does.
static SINK_IN_USE: Mutex<()> = Mutex::new(());

/// Discards whatever a directly-driven element pushes.
struct Discard;

impl OutputSink for Discard {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> Poll<Result<PushOutcome, G2gError>> {
        packet_slot.take().expect("poll_push without a packet");
        Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    }
}

/// Run `failing` with a recorder installed and return the messages it logged,
/// asserting the failure surfaced as an I/O error.
fn logs_of(failing: impl FnOnce() -> G2gError) -> (Vec<String>, MutexGuard<'static, ()>) {
    let guard = SINK_IN_USE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ring = RingSink::new(64);
    log::set_sink(Box::new(ring.clone()));
    log::set_default_level(LogLevel::Error);
    let err = failing();
    assert!(
        matches!(err, G2gError::Hardware(HardwareError::Io(_))),
        "an I/O failure: {err:?}"
    );
    (ring.drain().into_iter().map(|r| r.message).collect(), guard)
}

#[test]
fn a_source_that_cannot_open_its_file_logs_the_path() {
    let mut src = FileSrc::new(MISSING_FILE, caps());
    src.configure_pipeline(&caps())
        .expect("a byte source configures without touching the file");
    let (lines, _guard) =
        logs_of(|| block_on(src.run(&mut Discard)).expect_err("the file cannot be opened"));
    assert!(
        lines
            .iter()
            .any(|line| line.contains(MISSING_FILE) && line.contains("cannot open")),
        "the path and what went wrong: {lines:?}"
    );
}

#[test]
fn a_sink_that_cannot_create_its_file_logs_the_path() {
    let mut sink = FileSink::new(UNWRITABLE_FILE);
    let (lines, _guard) = logs_of(|| {
        sink.configure_pipeline(&caps())
            .expect_err("the file cannot be created")
    });
    assert!(
        lines
            .iter()
            .any(|line| line.contains(UNWRITABLE_FILE) && line.contains("cannot create")),
        "the path and what went wrong: {lines:?}"
    );
}
