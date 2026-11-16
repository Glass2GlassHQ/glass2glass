// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "openh264", feature = "wgpu"))]

use std::sync::mpsc;
use std::time::{Duration, Instant};

use g2g_player::{EndFile, Event, Media, Player, Property, PropertyChange, WgpuRenderContext};
use g2g_plugins::gpu::{read_texture_plane, whole_texture, GpuContext};

const CLIP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../g2g-plugins/tests/fixtures/h264_64x48_25fps_gop10.mp4"
);
const CLIP_WIDTH: u32 = 64;
const CLIP_HEIGHT: u32 = 48;
const CLIP_FRAMES_PER_SECOND: u32 = 25;
// time-pos at the end is the last frame's start, one frame before the duration
const END_POSITION_TOLERANCE: Duration = Duration::from_secs(2)
    .checked_div(CLIP_FRAMES_PER_SECOND)
    .unwrap();
const PLAYED_BEFORE_PAUSE: Duration = Duration::from_millis(400);
const RGBA_BYTES_PER_PIXEL: u32 = 4;
const EVENT_TIMEOUT: Duration = Duration::from_secs(10);
const PAUSE_LENGTH: Duration = Duration::from_millis(1500);
const SETTLE: Duration = Duration::from_millis(300);

struct EventLog<'a> {
    player: &'a Player,
    seen: Vec<Event>,
}

impl EventLog<'_> {
    fn wait_for(&mut self, wanted: &Event) {
        let deadline = Instant::now() + EVENT_TIMEOUT;
        while Instant::now() < deadline {
            let Some(event) = self.player.wait_event(deadline - Instant::now()) else {
                continue;
            };
            self.seen.push(event.clone());
            if &event == wanted {
                return;
            }
        }
        panic!(
            "no {wanted:?} within {EVENT_TIMEOUT:?}, saw {:?}",
            self.seen
        );
    }
}

fn time_pos(player: &Player) -> Duration {
    match player.property(Property::TimePos) {
        PropertyChange::TimePos(Some(position)) => position,
        other => panic!("no playback position yet: {other:?}"),
    }
}

fn wait_until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + EVENT_TIMEOUT;
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "condition not met within {EVENT_TIMEOUT:?}"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn a_clip_renders_pauses_seeks_and_stays_open_at_the_end() {
    let Ok(context) = pollster::block_on(GpuContext::headless()) else {
        return;
    };
    let player = Player::new();
    player.observe(Property::Duration);
    player.observe(Property::EofReached);
    let mut render = WgpuRenderContext::new(&player, context.clone());
    let (frame_arrived, frames) = mpsc::channel();
    player.set_update_callback(move || {
        let _ = frame_arrived.send(());
    });
    let mut events = EventLog {
        player: &player,
        seen: Vec::new(),
    };

    player.load(Media::Uri(&format!("file://{CLIP}"))).unwrap();
    events.wait_for(&Event::FileLoaded);
    frames.recv_timeout(EVENT_TIMEOUT).expect("a decoded frame");

    let target = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("m2346-target"),
        size: wgpu::Extent3d {
            width: CLIP_WIDTH,
            height: CLIP_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    render.render(&target).unwrap();
    let pixels = read_texture_plane(
        &context.device,
        &context.queue,
        &target,
        wgpu::TextureAspect::All,
        whole_texture(CLIP_WIDTH, CLIP_HEIGHT),
        RGBA_BYTES_PER_PIXEL,
    )
    .unwrap();
    assert!(
        pixels.chunks(4).any(|pixel| pixel[..3] != [0, 0, 0]),
        "the rendered target is still black"
    );

    wait_until(|| time_pos(&player) >= PLAYED_BEFORE_PAUSE);
    player.pause();
    std::thread::sleep(SETTLE);
    let paused_at = time_pos(&player);
    std::thread::sleep(PAUSE_LENGTH);
    assert_eq!(time_pos(&player), paused_at, "time-pos moved while paused");
    player.play();
    std::thread::sleep(SETTLE);
    let resumed = time_pos(&player);
    assert!(
        resumed > paused_at && resumed < paused_at + PAUSE_LENGTH / 2,
        "resumed at {resumed:?} after pausing at {paused_at:?}"
    );

    player.seek(PLAYED_BEFORE_PAUSE).unwrap();
    events.wait_for(&Event::Seek);
    events.wait_for(&Event::PlaybackRestart);
    wait_until(|| time_pos(&player) < resumed);

    events.wait_for(&Event::PropertyChange(PropertyChange::EofReached(true)));
    assert!(events.seen.iter().any(|event| matches!(
        event,
        Event::PropertyChange(PropertyChange::Duration(Some(_)))
    )));
    assert_eq!(
        player.property(Property::Pause),
        PropertyChange::Pause(true)
    );
    let PropertyChange::Duration(Some(duration)) = player.property(Property::Duration) else {
        panic!("no duration at the end");
    };
    let end_position = time_pos(&player);
    assert!(
        duration.abs_diff(end_position) <= END_POSITION_TOLERANCE,
        "stopped at {end_position:?} of a {duration:?} clip"
    );

    player.seek(PLAYED_BEFORE_PAUSE).unwrap();
    events.wait_for(&Event::PlaybackRestart);
    assert_eq!(
        player.property(Property::EofReached),
        PropertyChange::EofReached(false)
    );
    wait_until(|| time_pos(&player) < end_position);

    player.stop();
    events.wait_for(&Event::EndFile(EndFile::Stop));
}
