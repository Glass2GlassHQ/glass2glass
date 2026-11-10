// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(
    any(target_os = "linux", target_os = "macos", windows),
    feature = "ffmpeg"
))]

use std::path::{Path, PathBuf};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

// tools/mkv-codec-delay-fixtures.sh writes each beside ffmpeg's s16le decode of it
const MP3: &str = "mkv_mp3_codec_delay.mka";
const AAC: &str = "mkv_aac_codec_delay.mka";
const MP2: &str = "mkv_mp2_codec_delay.mka";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

async fn decode(name: &str) -> Vec<u8> {
    let out = std::env::temp_dir().join(format!("g2g-m2200-{name}-{}.raw", std::process::id()));
    let line = format!(
        "filesrc location={} ! decodebin ! audioconvert ! audio/x-raw,format=S16LE ! filesink location={}",
        fixture(name).display(),
        out.display()
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    let bytes = std::fs::read(&out).expect("the sink wrote its file");
    std::fs::remove_file(&out).ok();
    bytes
}

fn expected(name: &str) -> Vec<u8> {
    std::fs::read(fixture(&format!("{name}.s16le"))).expect("the expected decode is committed")
}

#[tokio::test]
async fn mp3_in_matroska_decodes_as_ffmpeg_does() {
    assert_eq!(decode(MP3).await, expected(MP3));
}

#[tokio::test]
async fn aac_in_matroska_decodes_as_ffmpeg_does() {
    assert_eq!(decode(AAC).await, expected(AAC));
}

#[tokio::test]
async fn mp2_in_matroska_decodes_as_ffmpeg_does() {
    assert_eq!(decode(MP2).await, expected(MP2));
}
