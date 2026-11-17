// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::process::{Command, Output};

pub(crate) fn launch(options: &[&str], pipeline: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_g2g-launch"))
        .arg("-q")
        .args(options)
        .args(pipeline.split_whitespace())
        .output()
        .expect("g2g-launch starts")
}

pub(crate) fn checksum_lines(output: &Output) -> Vec<String> {
    assert!(
        output.status.success(),
        "g2g-launch fails: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_string)
        .collect()
}
