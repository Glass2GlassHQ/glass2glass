// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::path::PathBuf;

const DRI_DIRECTORY: &str = "/dev/dri";
const RENDER_NODE_PREFIX: &str = "renderD";

pub(crate) fn render_nodes() -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(DRI_DIRECTORY) else {
        return Vec::new();
    };
    let mut nodes: Vec<PathBuf> = entries
        .map(|entry| entry.expect("lists /dev/dri").path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(RENDER_NODE_PREFIX))
        })
        .collect();
    nodes.sort();
    nodes
}
