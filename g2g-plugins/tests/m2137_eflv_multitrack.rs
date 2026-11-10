// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

mod ffprobe;
use ffprobe::{fixture_path, Probe};

use std::path::{Path, PathBuf};

use g2g_core::runtime::{parse_launch, run_graph, RunStats};
use g2g_core::PipelineClock;
use g2g_plugins::registry::default_registry;

const TWO_TRACK_FIXTURE: &str = "aac_two_tracks_ffmpeg.flv";
const AAC_FIXTURE: &str = "aac_stereo_44100_adts.aac";
const MP3_FIXTURE: &str = "mp3_stereo_44100_id3v2.mp3";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn work_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g2g-m2137-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create work dir");
    dir
}

fn at(dir: &Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

async fn run(line: &str) -> RunStats {
    let graph = parse_launch(&default_registry(), line)
        .unwrap_or_else(|error| panic!("{line} parses: {error:?}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|error| panic!("{line} runs: {error:?}"))
}

async fn aac_frames(path: &str) -> u64 {
    run(&format!(
        "filesrc location={path} ! typefind ! aacparse ! fakesink"
    ))
    .await
    .frames_consumed
}

#[tokio::test]
async fn ffmpeg_two_track_flv_demuxes_each_track() {
    let probe = Probe::read(TWO_TRACK_FIXTURE);
    let dir = work_dir("ffmpeg");
    let (first, second) = (at(&dir, "track0.aac"), at(&dir, "track1.aac"));
    run(&format!(
        "filesrc location={} ! flvdemux name=d \
         d.audio ! filesink location={first} d.audio_1 ! filesink location={second}",
        fixture_path(TWO_TRACK_FIXTURE).display()
    ))
    .await;
    for track in [&first, &second] {
        assert_eq!(aac_frames(track).await, probe.frames, "{track}");
    }
    let (first, second) = (std::fs::read(&first), std::fs::read(&second));
    assert_ne!(
        first.expect("track 0"),
        second.expect("track 1"),
        "the two sine tones stay on their own tracks"
    );
}

#[tokio::test]
async fn eflvmux_round_trips_a_legacy_and_a_multitrack_track() {
    let dir = work_dir("roundtrip");
    let aac = fixture_path(AAC_FIXTURE).display().to_string();
    let mp3 = fixture_path(MP3_FIXTURE).display().to_string();
    let (aac_parsed, mp3_parsed) = (at(&dir, "parsed.aac"), at(&dir, "parsed.mp3"));
    run(&format!(
        "filesrc location={aac} ! typefind ! aacparse ! filesink location={aac_parsed}"
    ))
    .await;
    run(&format!(
        "filesrc location={mp3} ! typefind ! mpegaudioparse ! filesink location={mp3_parsed}"
    ))
    .await;

    let muxed = at(&dir, "muxed.flv");
    run(&format!(
        "eflvmux name=m ! filesink location={muxed} \
         filesrc location={aac} ! typefind ! aacparse ! m.audio \
         filesrc location={mp3} ! typefind ! mpegaudioparse ! m.audio_1"
    ))
    .await;
    let (aac_out, mp3_out) = (at(&dir, "out.aac"), at(&dir, "out.mp3"));
    run(&format!(
        "filesrc location={muxed} ! flvdemux name=d \
         d.audio ! filesink location={aac_out} d.audio_1 ! filesink location={mp3_out}"
    ))
    .await;

    let read = |path: &str| std::fs::read(path).unwrap_or_else(|_| panic!("{path}"));
    assert_eq!(
        read(&mp3_out),
        read(&mp3_parsed),
        "track 1 carries the MP3 frames"
    );
    assert_eq!(
        read(&aac_out),
        read(&aac_parsed),
        "the legacy track carries the AAC frames"
    );
}
