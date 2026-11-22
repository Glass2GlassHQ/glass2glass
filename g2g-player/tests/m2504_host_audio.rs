// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use g2g_player::{AudioFormat, AudioOutputEvent, AudioParams, Event, LoadOptions, Media, Player};

const SOUND: &str = concat!(
    "file://",
    env!("CARGO_MANIFEST_DIR"),
    "/../g2g-plugins/tests/fixtures/sine_8k_stereo_s16le.wav"
);
const TIMEOUT: Duration = Duration::from_secs(5);
const PAUSED_WINDOW: Duration = Duration::from_millis(150);

#[derive(Debug, PartialEq, Eq)]
enum Observed {
    Samples {
        format: AudioFormat,
        params: AudioParams,
        pts: u64,
        bytes: usize,
        nonzero: bool,
    },
    Flush,
    EndOfStream,
}

fn host_audio(player: &Player) -> mpsc::Receiver<Observed> {
    let (send, receive) = mpsc::channel();
    player.set_audio_callback(move |event| {
        let observed = match event {
            AudioOutputEvent::Samples(buffer) => Observed::Samples {
                format: buffer.format,
                params: buffer.params,
                pts: buffer.timing.pts_ns,
                bytes: buffer.bytes.len(),
                nonzero: buffer.bytes.iter().any(|&byte| byte != 0),
            },
            AudioOutputEvent::Flush => Observed::Flush,
            AudioOutputEvent::EndOfStream => Observed::EndOfStream,
        };
        let _ = send.send(observed);
    });
    receive
}

#[test]
fn a_paused_uri_delivers_pcm_after_play_with_the_negotiated_format() {
    let player = Player::new();
    let audio = host_audio(&player);
    player
        .load(
            Media::Uri(SOUND),
            LoadOptions {
                start: None,
                pause: Some(true),
            },
        )
        .unwrap();
    assert_eq!(
        audio.recv_timeout(PAUSED_WINDOW),
        Err(RecvTimeoutError::Timeout)
    );
    let started = Instant::now();
    player.play();

    let first = audio.recv_timeout(TIMEOUT).unwrap();
    let Observed::Samples {
        format,
        params,
        pts,
        bytes,
        nonzero,
    } = first
    else {
        panic!("expected PCM, got {first:?}");
    };
    assert_eq!(format, AudioFormat::PcmS16Le);
    assert_eq!(params.channel_count, 2);
    assert_eq!(params.sample_rate, 48_000);
    assert_eq!(pts, 0);
    assert!(bytes > 0 && bytes.is_multiple_of(4) && nonzero);

    let deadline = Instant::now() + TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if audio.recv_timeout(remaining).expect("PCM end of stream") == Observed::EndOfStream {
            break;
        }
    }
    assert!(started.elapsed() >= Duration::from_millis(500));
}

#[test]
fn an_audio_only_launch_line_uses_the_player_audio_sink() {
    let player = Player::new();
    let line = "audiotestsrc samplerate=16000 channels=1 num-buffers=1 ! playeraudiosink";
    assert!(matches!(
        player.load(Media::LaunchLine(line), LoadOptions::default()),
        Err(g2g_player::LoadError::NoAudioCallback)
    ));
    let audio = host_audio(&player);
    player
        .load(Media::LaunchLine(line), LoadOptions::default())
        .unwrap();
    let first = audio.recv_timeout(TIMEOUT).unwrap();
    assert!(
        matches!(first, Observed::Samples { params, .. }
        if params.sample_rate == 16_000 && params.channel_count == 1),
        "{first:?}"
    );
    assert_eq!(audio.recv_timeout(TIMEOUT), Ok(Observed::EndOfStream));
}

#[test]
fn a_paused_seek_flushes_pcm_before_the_new_position_plays() {
    let player = Player::new();
    let audio = host_audio(&player);
    player
        .load(
            Media::Uri(SOUND),
            LoadOptions {
                start: None,
                pause: Some(true),
            },
        )
        .unwrap();
    let target = Duration::from_millis(500);
    player.seek(target).unwrap();
    assert_eq!(audio.recv_timeout(TIMEOUT), Ok(Observed::Flush));
    let deadline = Instant::now() + TIMEOUT;
    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .expect("seek completes");
        if player.wait_event(remaining) == Some(Event::PlaybackRestart) {
            break;
        }
    }
    assert_eq!(
        audio.recv_timeout(PAUSED_WINDOW),
        Err(RecvTimeoutError::Timeout)
    );
    player.play();
    let first = audio.recv_timeout(TIMEOUT).unwrap();
    assert!(
        matches!(first, Observed::Samples { pts, .. } if pts >= 500_000_000),
        "{first:?}"
    );
}
