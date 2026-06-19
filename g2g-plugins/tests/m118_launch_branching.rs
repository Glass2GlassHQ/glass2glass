//! M118 `gst-launch` branching, end to end: a `tee` written in text fans one
//! source to several sinks, parsed into a `Graph` and *run* through `run_graph`.
//! The payoff is the last gst-launch grammar gap closed: a branching pipeline
//! expressed as text actually broadcasts frames to every branch.
//!
//! `default_registry` is `std`-gated, so this file is too: run with
//! `cargo test -p g2g-plugins --features std`.
#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph, ParseError};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

#[tokio::test]
async fn tee_fans_out_to_two_sinks() {
    let reg = default_registry();
    // One source, a tee, two sinks: the inline branch and the `t.` branch.
    let graph = parse_launch(&reg, "videotestsrc num-buffers=3 ! tee name=t ! fakesink t. ! fakesink")
        .expect("branching pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("branching pipeline runs");

    assert_eq!(stats.frames_emitted, 3, "the single source emitted num-buffers frames");
    // The tee broadcasts every frame to both branches, so the two sinks consume
    // twice the emitted count between them.
    assert_eq!(stats.frames_consumed, 6, "both branches consumed every frame");
}

#[tokio::test]
async fn tee_branch_carries_a_transform() {
    let reg = default_registry();
    // One branch runs a transform before its sink; the other is a bare sink.
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=2 ! tee name=t ! identity ! fakesink t. ! fakesink",
    )
    .expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("pipeline runs");

    assert_eq!(stats.frames_emitted, 2);
    // Two sinks, each fed every frame (the transform passes them through).
    assert_eq!(stats.frames_consumed, 4, "frames flowed through both branches");
}

#[tokio::test]
async fn tee_fans_out_three_ways() {
    let reg = default_registry();
    // Output width is derived from the branch count: one inline + two `t.` refs.
    let graph = parse_launch(
        &reg,
        "videotestsrc num-buffers=2 ! tee name=t ! fakesink t. ! fakesink t. ! fakesink",
    )
    .expect("pipeline parses");
    let stats = run_graph(graph, &ZeroClock, 4).await.expect("pipeline runs");

    assert_eq!(stats.frames_emitted, 2);
    assert_eq!(stats.frames_consumed, 6, "three sinks each consumed every frame");
}

#[tokio::test]
async fn fan_out_without_tee_is_rejected() {
    let reg = default_registry();
    // A bare source feeding two consumers without a tee is a clear error, named
    // against the offending element rather than a raw graph failure.
    let err = parse_launch(&reg, "videotestsrc name=s ! fakesink s. ! fakesink").unwrap_err();
    assert_eq!(err, ParseError::FanOutWithoutTee("videotestsrc".to_string()));
}
