//! M680 record / replay: `recordsink` dumps the packet stream to a file and
//! `replaysrc` plays it back as a source. The round-trip must preserve frame
//! data exactly, so a bug that needed a live source can be reproduced from a
//! recording.
//!
//! std-gated (registry + file I/O): `cargo test -p g2g-plugins --features std
//! --test m680_record_replay`.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn run(line: &str) -> g2g_core::runtime::RunStats {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).expect("pipeline parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("pipeline runs")
}

#[tokio::test]
async fn replay_reproduces_recorded_frames_byte_for_byte() {
    let dir = std::env::temp_dir();
    let direct = dir.join("g2g_m680_direct.bin");
    let recording = dir.join("g2g_m680_rec.g2g");
    let replayed = dir.join("g2g_m680_replayed.bin");
    for p in [&direct, &recording, &replayed] {
        let _ = std::fs::remove_file(p);
    }

    // A direct run's raw frame bytes, the reference.
    run(&format!(
        "videotestsrc num-buffers=6 ! videoscale width=64 height=48 ! filesink location={}",
        direct.display()
    ))
    .await;

    // The same source recorded, then replayed out to raw bytes.
    let rec_stats = run(&format!(
        "videotestsrc num-buffers=6 ! videoscale width=64 height=48 ! recordsink location={}",
        recording.display()
    ))
    .await;
    assert_eq!(rec_stats.frames_consumed, 6, "recorder saw every frame");
    assert!(
        std::fs::metadata(&recording).unwrap().len() > 0,
        "recording is non-empty"
    );

    let replay_stats = run(&format!(
        "replaysrc location={} ! filesink location={}",
        recording.display(),
        replayed.display()
    ))
    .await;
    assert_eq!(
        replay_stats.frames_consumed, 6,
        "replay emitted every recorded frame"
    );

    let a = std::fs::read(&direct).unwrap();
    let b = std::fs::read(&replayed).unwrap();
    assert!(!a.is_empty());
    assert_eq!(a, b, "replayed frame bytes are identical to the direct run");

    for p in [&direct, &recording, &replayed] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn replay_missing_file_fails_cleanly() {
    let reg = default_registry();
    let graph =
        parse_launch(&reg, "replaysrc location=/no/such/g2g/recording ! fakesink").expect("parses");
    // A missing recording fails negotiation (no leading caps), not a panic.
    assert!(run_graph(graph, &ZeroClock, 4).await.is_err());
}
