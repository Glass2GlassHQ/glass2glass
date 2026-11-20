// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::process;

use g2g_core::runtime::{parse_launch, run_graph_threaded, Registry};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::TokioThreadSpawner;

pub(crate) fn registry_with_ml_elements() -> Registry {
    let mut registry = default_registry();
    g2g_ml::register(&mut registry);
    registry
}

pub(crate) async fn run_from_command_line(
    program: &str,
    registry: &Registry,
    link_capacity: usize,
) {
    g2g_core::log::init_from_env();
    let line = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if line.is_empty() {
        eprintln!("usage: {program} <launch line>");
        process::exit(2);
    }
    let graph = parse_launch(registry, &line).unwrap_or_else(|error| {
        eprintln!("{program}: {error:?}");
        process::exit(1);
    });
    match run_graph_threaded(graph, &WallClock::new(), link_capacity, &TokioThreadSpawner).await {
        Ok(stats) => print!("{}", stats.report()),
        Err(error) => {
            eprintln!("{program}: {error:?}");
            process::exit(1);
        }
    }
}
