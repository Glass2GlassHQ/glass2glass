// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "ffmpeg")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use g2g_core::runtime::{is_raw_video, parse_launch, run_graph, GraphNode};
use g2g_core::Graph;
use g2g_plugins::clock::WallClock;
use g2g_plugins::filesink::FileSink;
use g2g_plugins::registry::default_registry;

const MKV_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_64x48_25fps_gop10.mkv"
);
const LINK_CAPACITY: usize = 4;
const MAX_DEPTH: usize = 6;

// cargo runs these tests in parallel, and each writes the decodebin reference
static OUTPUT_SERIAL: AtomicUsize = AtomicUsize::new(0);

fn output_path(tag: &str) -> PathBuf {
    let serial = OUTPUT_SERIAL.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "g2g-m2362-{tag}-{}-{serial}.yuv",
        std::process::id()
    ))
}

async fn written_bytes(tag: &str, graph: impl FnOnce(&str) -> Graph<GraphNode>) -> Vec<u8> {
    let out = output_path(tag);
    let _ = std::fs::remove_file(&out);
    run_graph(
        graph(&out.display().to_string()),
        &WallClock::new(),
        LINK_CAPACITY,
    )
    .await
    .unwrap_or_else(|error| panic!("runs the {tag} graph: {error:?}"));
    let written = std::fs::read(&out).expect("the graph writes its output");
    let _ = std::fs::remove_file(&out);
    written
}

async fn launched_bytes(tag: &str, line: impl Fn(&str) -> String) -> Vec<u8> {
    written_bytes(tag, |out| {
        let line = line(out);
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"))
    })
    .await
}

async fn decodebin_bytes() -> Vec<u8> {
    launched_bytes("decodebin", |out| {
        format!("filesrc location={MKV_FIXTURE} ! decodebin ! filesink location={out}")
    })
    .await
}

#[tokio::test]
async fn launch_uridecodebin_decodes_a_matroska_file_as_decodebin_does() {
    let uridecodebin = launched_bytes("uridecodebin", |out| {
        format!("uridecodebin uri=file://{MKV_FIXTURE} ! filesink location={out}")
    })
    .await;
    assert_eq!(uridecodebin, decodebin_bytes().await);
}

#[tokio::test]
async fn build_uridecodebin_decodes_a_matroska_file_as_decodebin_does() {
    let built = written_bytes("build-uridecodebin", |out| {
        default_registry()
            .build_uridecodebin(
                &format!("file://{MKV_FIXTURE}"),
                FileSink::new(out),
                &is_raw_video,
                MAX_DEPTH,
            )
            .expect("builds the uridecodebin graph")
    })
    .await;
    assert_eq!(built, decodebin_bytes().await);
}
