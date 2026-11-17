// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

const WAV: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sine_8k_stereo_s16le.wav"
);
const LINK_CAPACITY: usize = 4;

#[tokio::test]
async fn playbin_plays_a_wav_file_on_its_audio_sink() {
    let registry = default_registry();
    let line = format!("playbin uri=file://{WAV} audio-sink=fakesink");
    let graph = parse_launch(&registry, &line).unwrap();
    let valid = graph.finish().unwrap();
    let names: Vec<String> = valid
        .topo()
        .iter()
        .filter_map(|&node| valid.element(node).map(|e| e.log_category().to_string()))
        .collect();
    assert_eq!(
        names[names.len() - 3..],
        ["AudioConvert", "AudioResample", "FakeSink"],
        "{names:?}"
    );

    let graph = parse_launch(&registry, &line).unwrap();
    let stats = run_graph(graph, &WallClock::default(), LINK_CAPACITY)
        .await
        .unwrap();
    assert!(stats.frames_consumed > 0);
}
