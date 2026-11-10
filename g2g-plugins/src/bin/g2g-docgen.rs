// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

fn main() {
    g2g_plugins::docgen::run(
        &g2g_plugins::registry::default_registry(),
        std::env::args().nth(1),
    );
}
