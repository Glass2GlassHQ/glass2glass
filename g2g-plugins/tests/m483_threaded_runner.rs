//! M483 opt-in thread-per-arm graph runner: the same parsed pipeline run through
//! the cooperative `run_graph` and the multicore `run_graph_threaded` (one OS
//! thread per arm via `TokioThreadSpawner`) must move the same frames to the
//! sink. This proves the threaded driver negotiates and drives a DAG identically
//! to the cooperative one, just spread across threads.
#![cfg(all(feature = "std", feature = "multi-thread"))]

use std::collections::BTreeSet;

use g2g_core::meta::Orientation;
use g2g_core::runtime::{
    parse_launch, run_graph, run_graph_threaded, run_graph_threaded_with_bus, run_graph_with_bus,
    LaunchFactory, Registry, SourceFactory, ThreadSpawner,
};
use g2g_core::{Bus, BusMessage, Caps, Dim, PipelineClock, Rate, RawVideoFormat};
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::videoflip::VideoFlip;
use g2g_plugins::videorate::VideoRate;
use g2g_plugins::videotestsrc::VideoTestSrc;
use g2g_plugins::TokioThreadSpawner;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
        interlace: g2g_core::Interlace::Any,
    }
}

fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_source(SourceFactory::new("videotestsrc", rgba_any(), || {
        Box::new(VideoTestSrc::new(64, 48, 30, 0))
    }));
    reg.register_launch(LaunchFactory::of::<VideoFlip>("videoflip", || {
        Box::new(VideoFlip::new(Orientation::HorizontalMirror))
    }));
    reg.register_launch(LaunchFactory::new("videorate", Vec::new(), || {
        Box::new(VideoRate::new(30.0))
    }));
    reg.register_launch(LaunchFactory::new("fakesink", Vec::new(), || {
        Box::new(FakeSink::new())
    }));
    reg
}

const PIPELINE: &str =
    "videotestsrc num-buffers=8 ! videoflip method=rotate-180 ! videorate ! fakesink";

/// The threaded runner (one thread per arm, tokio reactor each) delivers the
/// same frame counts as the cooperative runner on the same pipeline.
#[tokio::test]
async fn threaded_matches_cooperative() {
    let reg = registry();

    let coop = run_graph(parse_launch(&reg, PIPELINE).expect("parses"), &ZeroClock, 4)
        .await
        .expect("cooperative run");
    assert_eq!(
        coop.frames_emitted, 8,
        "cooperative: source emitted num-buffers"
    );
    assert_eq!(
        coop.frames_consumed, 8,
        "cooperative: all frames reached the sink"
    );

    let threaded = run_graph_threaded(
        parse_launch(&reg, PIPELINE).expect("parses"),
        &ZeroClock,
        4,
        &TokioThreadSpawner,
    )
    .await
    .expect("threaded run");

    assert_eq!(
        threaded.frames_emitted, coop.frames_emitted,
        "threaded: same emitted count"
    );
    assert_eq!(
        threaded.frames_consumed, coop.frames_consumed,
        "threaded: same consumed count"
    );
    assert_eq!(
        threaded.frames_dropped, coop.frames_dropped,
        "threaded: same drop count"
    );
}

/// The dependency-free core [`ThreadSpawner`] (std threads + park-based block_on,
/// no reactor) also drives a pure-core pipeline to completion.
#[tokio::test]
async fn core_thread_spawner_runs() {
    let reg = registry();
    let stats = run_graph_threaded(
        parse_launch(&reg, PIPELINE).expect("parses"),
        &ZeroClock,
        4,
        &ThreadSpawner,
    )
    .await
    .expect("core ThreadSpawner run");
    assert_eq!(
        stats.frames_consumed, 8,
        "all frames reached the sink under ThreadSpawner"
    );
}

/// Each arm thread brackets itself with a `StreamStatus` enter / leave pair, so
/// an application can see the graph's real thread fan-out. The cooperative
/// runner, which multiplexes every arm on one executor, posts none.
#[tokio::test]
async fn stream_status_brackets_every_arm_thread() {
    let reg = registry();
    let (bus, handle) = Bus::new(64);

    let stats = run_graph_threaded_with_bus(
        parse_launch(&reg, PIPELINE).expect("parses"),
        &ZeroClock,
        4,
        &handle,
        &TokioThreadSpawner,
    )
    .await
    .expect("threaded run");
    assert_eq!(
        stats.frames_consumed, 8,
        "the run still delivers its frames"
    );

    let (mut entered, mut left) = (Vec::new(), Vec::new());
    while let Some(m) = bus.try_recv() {
        match m {
            BusMessage::StreamStatus {
                entered: true,
                thread_id,
            } => entered.push(thread_id),
            BusMessage::StreamStatus {
                entered: false,
                thread_id,
            } => left.push(thread_id),
            _ => {}
        }
    }
    // One thread per graph arm (source, videoflip, videorate, fakesink) plus the
    // re-cascade coordinator's own.
    assert_eq!(entered.len(), 5, "one enter per spawned arm thread");
    let distinct: BTreeSet<u64> = entered.iter().copied().collect();
    assert_eq!(distinct.len(), entered.len(), "each arm on its own thread");
    entered.sort_unstable();
    left.sort_unstable();
    assert_eq!(entered, left, "every thread that entered also left");

    // Negative control: the cooperative runner has no streaming threads.
    let (coop_bus, coop_handle) = Bus::new(64);
    run_graph_with_bus(
        parse_launch(&reg, PIPELINE).expect("parses"),
        &ZeroClock,
        4,
        &coop_handle,
    )
    .await
    .expect("cooperative run");
    let mut posted = 0;
    while let Some(m) = coop_bus.try_recv() {
        if matches!(m, BusMessage::StreamStatus { .. }) {
            posted += 1;
        }
    }
    assert_eq!(posted, 0, "the cooperative runner posts no stream-status");
}
