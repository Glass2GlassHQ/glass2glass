// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "ffmpeg")]

use std::process::Command;
use std::time::{Duration, Instant};

use g2g_player::{
    LoadOptions, Media, Player, Property, PropertyChange, SoundOutput, StreamType, TrackChoice,
};

const TARGET: Duration = Duration::from_millis(600);
const TIMEOUT: Duration = Duration::from_secs(10);

fn wait_until(mut accept: impl FnMut() -> bool) {
    let deadline = Instant::now() + TIMEOUT;
    while !accept() {
        assert!(
            Instant::now() < deadline,
            "player did not reach the expected state"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn position(player: &Player) -> Option<Duration> {
    match player.property(Property::TimePos) {
        PropertyChange::TimePos(position) => position,
        _ => unreachable!("queried time-pos"),
    }
}

fn sample_rate(player: &Player) -> Option<u32> {
    match player.property(Property::AudioParams) {
        PropertyChange::AudioParams(params) => params.map(|params| params.sample_rate),
        _ => unreachable!("queried audio-params"),
    }
}

#[test]
fn switches_audio_codecs_in_a_local_file_at_the_paused_position() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping without ffmpeg on PATH");
        return;
    }
    let path = std::env::temp_dir().join(format!("g2g-player-tracks-{}.mkv", std::process::id()));
    let result = Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=25:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=44100:duration=3",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:a",
            "-c:v",
            "mpeg2video",
            "-g",
            "10",
            "-bf",
            "0",
            "-c:a:0",
            "pcm_s16le",
            "-c:a:1",
            "aac",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:1",
            "language=fra",
        ])
        .arg(&path)
        .output()
        .expect("run ffmpeg");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let player = Player::new();
    player.set_sound_output(SoundOutput::Discard);
    let uri = format!("file://{}", path.display());
    player.observe(Property::TrackList);
    player
        .load(
            Media::Uri(&uri),
            LoadOptions {
                start: Some(TARGET),
                pause: Some(true),
            },
        )
        .unwrap();
    wait_until(|| sample_rate(&player) == Some(48_000) && position(&player) == Some(TARGET));

    let tracks = player.track_list();
    let audio: Vec<_> = tracks
        .iter()
        .filter(|track| track.kind == StreamType::Audio)
        .collect();
    assert_eq!(audio.len(), 2);
    assert!(audio[0].selected && !audio[1].selected);
    assert!(audio[1]
        .tags
        .tags()
        .iter()
        .any(|tag| tag.key() == "language" && tag.value_string() == "fra"));
    assert!(player
        .select_track(StreamType::Audio, TrackChoice::Track(2))
        .is_err());
    assert_eq!(
        player.property(Property::TrackList),
        PropertyChange::TrackList(tracks)
    );

    player
        .select_track(StreamType::Audio, TrackChoice::Track(1))
        .unwrap();
    wait_until(|| sample_rate(&player) == Some(44_100));
    let selected = player.track_list();
    assert!(selected
        .iter()
        .any(|track| track.kind == StreamType::Audio && track.index == 1 && track.selected));
    assert_eq!(
        selected
            .iter()
            .filter(|track| track.kind == StreamType::Audio && track.selected)
            .count(),
        1
    );
    assert_eq!(position(&player), Some(TARGET));
    assert_eq!(
        player.property(Property::Pause),
        PropertyChange::Pause(true)
    );
    assert!(
        std::iter::from_fn(|| player.wait_event(Duration::ZERO)).any(|event| event
            == g2g_player::Event::PropertyChange(PropertyChange::TrackList(selected.clone())))
    );

    player.play();
    wait_until(|| position(&player).is_some_and(|position| position > TARGET));
    let was_playing = position(&player).unwrap();
    player
        .select_track(StreamType::Audio, TrackChoice::Auto)
        .unwrap();
    wait_until(|| sample_rate(&player) == Some(48_000));
    assert_eq!(
        player.property(Property::Pause),
        PropertyChange::Pause(false)
    );
    assert!(position(&player).unwrap() >= was_playing);
    player.stop();
    assert!(player.track_list().is_empty());
    std::fs::remove_file(path).expect("remove test file");
}

#[test]
fn switches_tracks_in_an_audio_only_mp4() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping without ffmpeg on PATH");
        return;
    }
    let path = std::env::temp_dir().join(format!("g2g-player-tracks-{}.m4a", std::process::id()));
    let result = Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:sample_rate=44100:duration=3",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-c:a",
            "aac",
        ])
        .arg(&path)
        .output()
        .expect("run ffmpeg");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let player = Player::new();
    player.set_sound_output(SoundOutput::Discard);
    let uri = format!("file://{}", path.display());
    player
        .load(
            Media::Uri(&uri),
            LoadOptions {
                start: Some(TARGET),
                pause: Some(true),
            },
        )
        .unwrap();
    wait_until(|| sample_rate(&player) == Some(48_000) && position(&player).is_some());
    let before = position(&player).unwrap();
    assert!(
        before.abs_diff(TARGET) <= Duration::from_millis(25),
        "{before:?}"
    );
    assert_eq!(player.track_list().len(), 2);
    assert!(player
        .select_track(StreamType::Audio, TrackChoice::Disabled)
        .is_err());
    player
        .select_track(StreamType::Audio, TrackChoice::Track(1))
        .unwrap();
    wait_until(|| sample_rate(&player) == Some(44_100));
    assert_eq!(
        player.property(Property::Pause),
        PropertyChange::Pause(true)
    );
    let after = position(&player).expect("the replacement retains a position");
    assert!(
        after.abs_diff(before) <= Duration::from_millis(25),
        "{before:?} to {after:?}"
    );
    assert!(player.track_list()[1].selected);
    player.stop();
    std::fs::remove_file(path).expect("remove test file");
}
