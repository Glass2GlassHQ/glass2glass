// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::process;

use g2g_core::runtime::{parse_launch, run_graph_threaded};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::TokioThreadSpawner;

// each slot adds about two frame periods of latency on a full live link
const LINK_CAPACITY: usize = 2;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    g2g_core::log::init_from_env();
    let line = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if line.is_empty() {
        eprintln!("usage: ml_launch <launch line>");
        process::exit(2);
    }
    let mut registry = default_registry();
    g2g_ml::register(&mut registry);
    let graph = parse_launch(&registry, &line).unwrap_or_else(|error| {
        eprintln!("ml_launch: {error:?}");
        process::exit(1);
    });
    match run_graph_threaded(graph, &WallClock::new(), LINK_CAPACITY, &TokioThreadSpawner).await {
        Ok(stats) => print!("{}", stats.report()),
        Err(error) => {
            eprintln!("ml_launch: {error:?}");
            process::exit(1);
        }
    }
}
