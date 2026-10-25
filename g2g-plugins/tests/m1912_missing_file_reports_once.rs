// a missing input file fails as a missing file, logs its open error once, and logs nothing else
#![cfg(feature = "std")]

use std::sync::{Mutex, MutexGuard};

use g2g_core::log::{self, LogLevel, OwnedLogRecord, RingSink};
use g2g_core::runtime::{block_on, fallback_factory, parse_launch, run_graph};
use g2g_core::G2gError;
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

const LINK_CAPACITY: usize = 4;

// the log sink is process-global
static SINK_IN_USE: Mutex<()> = Mutex::new(());

fn capture_logs() -> (RingSink, MutexGuard<'static, ()>) {
    let guard = SINK_IN_USE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ring = RingSink::new(64);
    log::set_sink(Box::new(ring.clone()));
    log::set_default_level(LogLevel::Warn);
    (ring, guard)
}

fn finish_capture(ring: &RingSink) -> Vec<OwnedLogRecord> {
    let records = ring.drain();
    log::reset();
    records
}

#[test]
fn a_missing_file_with_a_container_extension_fails_once_as_not_found() {
    let missing =
        std::env::temp_dir().join(format!("g2g_m1912_{}_missing.mp4", std::process::id()));
    let line = format!("filesrc location={} ! fakesink", missing.display());
    let graph = parse_launch(&default_registry(), &line).expect("the line parses");

    let (ring, _guard) = capture_logs();
    let result = block_on(run_graph(graph, &WallClock::new(), LINK_CAPACITY));
    let records = finish_capture(&ring);

    let open_errors = records
        .iter()
        .filter(|record| record.message.starts_with("cannot open"))
        .count();
    assert_eq!(open_errors, 1, "one open error: {records:?}");
    assert_eq!(result.err(), Some(G2gError::FileNotFound));
}

#[test]
fn mapping_a_failed_source_back_to_a_factory_logs_nothing() {
    let registry = default_registry();
    let (ring, _guard) = capture_logs();
    let factory = fallback_factory(
        &registry,
        "filesrc location=x.txt ! fakesink",
        "FileSrc0",
        &[],
    );
    let records = finish_capture(&ring);

    assert_eq!(factory, None, "a source is never re-plugged");
    assert!(records.is_empty(), "nothing is logged: {records:?}");
}
