// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

// Each codec a fan-in muxer takes remuxes out of `parsebin` and decodes to the same
// samples or pictures as its source.
#![cfg(all(
    feature = "std",
    feature = "ffmpeg",
    any(target_os = "linux", target_os = "macos", windows)
))]

use std::path::{Path, PathBuf};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

struct Remux {
    fixture: &'static str,
    muxer: &'static str,
    extension: &'static str,
}

const fn mkv_audio(fixture: &'static str) -> Remux {
    Remux {
        fixture,
        muxer: "matroskamux",
        extension: "mkv",
    }
}

const REMUXES: &[Remux] = &[
    mkv_audio("vorbis_stereo_48k.ogg"),
    mkv_audio("flac_stereo_48k.mka"),
    mkv_audio("mp3_stereo_44100_id3v2.mp3"),
    mkv_audio("mkv_mp3_codec_delay.mka"),
    mkv_audio("av_mpeg2_mp2.mpg"),
    mkv_audio("ac3_stereo_48000.ac3"),
    mkv_audio("sine_8k_stereo_s16le.wav"),
    mkv_audio("sine_8k_mono_u8.mka"),
    mkv_audio("mov_lpcm_s24be.mov"),
    mkv_audio("mov_fl32_le.mov"),
    mkv_audio("sine_8k_mono_alaw.wav"),
    mkv_audio("sine_8k_mono_mulaw.wav"),
    mkv_audio("sine_8k_stereo_imaadpcm.wav"),
];

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

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m2287-{}-{name}", std::process::id()))
}

async fn run(line: &str) {
    let graph =
        parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
}

async fn decoded(path: &Path) -> Vec<u8> {
    let out = temp_path("decoded.raw");
    run(&format!(
        "filesrc location={} ! decodebin current-audio=0 ! filesink location={}",
        path.display(),
        out.display()
    ))
    .await;
    let bytes = std::fs::read(&out).expect("the decode wrote its file");
    std::fs::remove_file(&out).ok();
    assert!(!bytes.is_empty(), "{} decoded nothing", path.display());
    bytes
}

#[tokio::test]
async fn each_codec_remuxes_and_decodes_like_its_source() {
    for remux in REMUXES {
        let source = fixture(remux.fixture);
        let remuxed = temp_path(&format!("remux.{}", remux.extension));
        run(&format!(
            "filesrc location={} ! parsebin name=p  p.audio_0 ! queue ! m.  {} name=m ! filesink location={}",
            source.display(),
            remux.muxer,
            remuxed.display()
        ))
        .await;
        let original = decoded(&source).await;
        let copy = decoded(&remuxed).await;
        std::fs::remove_file(&remuxed).ok();
        assert!(
            copy == original,
            "{} through {} decodes to {} bytes, the source to {}",
            remux.fixture,
            remux.muxer,
            copy.len(),
            original.len()
        );
    }
}
