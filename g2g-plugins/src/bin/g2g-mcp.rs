// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

fn main() {
    g2g_plugins::mcp::McpServer::new().serve_stdio();
}
