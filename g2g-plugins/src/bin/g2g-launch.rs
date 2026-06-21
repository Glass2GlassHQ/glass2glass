//! `g2g-launch`: the `gst-launch` analog. Parses a text pipeline against the
//! standard element registry and runs it to completion.
//!
//! Usage:
//!   g2g-launch videotestsrc num-buffers=30 ! videoconvert format=nv12 ! fakesink
//!
//! The arguments are joined with spaces into one pipeline string, parsed by
//! [`g2g_core::runtime::parse_launch`] (M106) against
//! [`g2g_plugins::registry::default_registry`] (M107), and driven on a
//! single-thread tokio runtime against the [`WallClock`]. Requires the `std`
//! feature (registry + wall clock are std-only).

use std::process;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

// Steady-state link depth. Matches the integration-test default; small enough to
// keep latency low without starving the source (see DESIGN notes on
// link_capacity dominating glass-to-glass latency).
const LINK_CAPACITY: usize = 4;

fn main() {
    let pipeline = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if pipeline.trim().is_empty() {
        eprintln!("usage: g2g-launch <element> [key=value ...] ! <element> ! ...");
        process::exit(2);
    }

    let reg = default_registry();
    let graph = match parse_launch(&reg, &pipeline) {
        Ok(graph) => graph,
        Err(err) => {
            eprintln!("parse error: {err:?}");
            process::exit(1);
        }
    };

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("build tokio runtime");

    println!("Setting pipeline to PLAYING ...");
    let clock = WallClock::new();
    match rt.block_on(run_graph(graph, &clock, LINK_CAPACITY)) {
        Ok(stats) => {
            println!(
                "Done. frames emitted: {}, consumed: {}, dropped: {}",
                stats.frames_emitted, stats.frames_consumed, stats.frames_dropped
            );
        }
        Err(err) => {
            eprintln!("pipeline error: {err:?}");
            process::exit(1);
        }
    }
}
