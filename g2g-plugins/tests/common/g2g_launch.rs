// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![allow(dead_code)] // not every test file uses every helper here

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

pub(crate) fn written_bytes(pipeline: &str, file_name: &str) -> Vec<u8> {
    let path = format!("{}/{file_name}", env!("CARGO_TARGET_TMPDIR"));
    let output = launch(&[], &format!("{pipeline} ! filesink location={path}"));
    assert!(
        output.status.success(),
        "{pipeline} fails: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read(&path).unwrap_or_else(|error| panic!("cannot read {path}: {error}"))
}
