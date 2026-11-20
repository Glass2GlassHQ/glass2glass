// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

mod launch_line;

// each slot adds about two frame periods of latency on a full live link
const LINK_CAPACITY: usize = 2;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let registry = launch_line::registry_with_ml_elements();
    launch_line::run_from_command_line("ml_launch", &registry, LINK_CAPACITY).await;
}
