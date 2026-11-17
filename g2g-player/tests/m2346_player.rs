// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "openh264", feature = "wgpu"))]

use std::sync::mpsc;
use std::time::{Duration, Instant};

use g2g_player::{
    EndFile, Event, LoadOptions, Media, OsdDimensions, Player, Property, PropertyChange, VideoSize,
    WgpuRenderContext,
};
use g2g_plugins::gpu::{read_texture_plane, whole_texture, GpuContext};

const CLIP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../g2g-plugins/tests/fixtures/h264_64x48_25fps_gop10.mp4"
);
const CLIP_WIDTH: u32 = 64;
const CLIP_HEIGHT: u32 = 48;
const CLIP_FRAMES_PER_SECOND: u32 = 25;
const CLIP_FRAME_PERIOD: Duration = Duration::from_secs(1)
    .checked_div(CLIP_FRAMES_PER_SECOND)
    .unwrap();
// time-pos at the end is the last frame's start, one frame before the duration
const END_POSITION_TOLERANCE: Duration = CLIP_FRAME_PERIOD.saturating_mul(2);
// between two keyframes of the clip's ten-frame GOP
const START: Duration = Duration::from_secs(1);
const RELATIVE_SEEK: Duration = Duration::from_millis(400);
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

fn rgba_target(context: &GpuContext) -> wgpu::Texture {
    context.device.create_texture(&wgpu::TextureDescriptor {
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
    })
}

fn rendered_pixels(
    render: &mut WgpuRenderContext,
    context: &GpuContext,
    target: &wgpu::Texture,
) -> Vec<u8> {
    render.render(target).unwrap();
    read_texture_plane(
        &context.device,
        &context.queue,
        target,
        wgpu::TextureAspect::All,
        whole_texture(CLIP_WIDTH, CLIP_HEIGHT),
        RGBA_BYTES_PER_PIXEL,
    )
    .unwrap()
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

    player
        .load(
            Media::Uri(&format!("file://{CLIP}")),
            LoadOptions::default(),
        )
        .unwrap();
    events.wait_for(&Event::FileLoaded);
    frames.recv_timeout(EVENT_TIMEOUT).expect("a decoded frame");

    let target = rgba_target(&context);
    let pixels = rendered_pixels(&mut render, &context, &target);
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

#[test]
fn a_clip_loads_paused_at_its_start_position() {
    let player = Player::new();
    let prerolled_at = load_paused_at_start(&player);
    assert!(
        prerolled_at >= START && prerolled_at < START + CLIP_FRAME_PERIOD,
        "the first frame shown is at {prerolled_at:?}, the load started at {START:?}"
    );
    std::thread::sleep(PAUSE_LENGTH);
    assert_eq!(
        time_pos(&player),
        prerolled_at,
        "time-pos moved after a paused load"
    );
    assert_eq!(
        player.property(Property::Pause),
        PropertyChange::Pause(true)
    );
}

fn load_paused_at_start(player: &Player) -> Duration {
    player
        .load(
            Media::Uri(&format!("file://{CLIP}")),
            LoadOptions {
                start: Some(START),
                pause: Some(true),
            },
        )
        .unwrap();
    wait_until(|| {
        matches!(
            player.property(Property::TimePos),
            PropertyChange::TimePos(Some(_))
        )
    });
    time_pos(player)
}

fn moved_from(player: &Player, before: Duration) -> Duration {
    wait_until(|| time_pos(player) != before);
    time_pos(player)
}

#[test]
fn relative_seeks_and_frame_steps_land_on_their_frames_paused() {
    let player = Player::new();
    let start = load_paused_at_start(&player);

    player.frame_step().unwrap();
    let stepped = moved_from(&player, start);
    assert_eq!(stepped, start + CLIP_FRAME_PERIOD, "frame step");
    player.frame_back_step().unwrap();
    assert_eq!(moved_from(&player, stepped), start, "frame back step");

    player.seek_forward(RELATIVE_SEEK).unwrap();
    let forward = moved_from(&player, start);
    assert_eq!(forward, start + RELATIVE_SEEK, "relative seek forward");
    player.seek_backward(START + RELATIVE_SEEK + START).unwrap();
    assert_eq!(
        moved_from(&player, forward),
        Duration::ZERO,
        "relative seek back past the start"
    );
    assert_eq!(
        player.property(Property::Pause),
        PropertyChange::Pause(true)
    );
}

// ffprobe: pcm_s16le, 8000 Hz, 2 channels, 1 s
const SOUND: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../g2g-plugins/tests/fixtures/sine_8k_stereo_s16le.wav"
);
const FIRST_CUT_LENGTH: Duration = Duration::from_millis(400);
const SECOND_CUT_LENGTH: Duration = Duration::from_millis(200);

#[test]
fn an_edl_timeline_plays_its_cuts_and_its_sound_stream() {
    let timeline_length = FIRST_CUT_LENGTH + SECOND_CUT_LENGTH;
    let uri = format!(
        "edl://{CLIP},{},{};{CLIP},0,{};!new_stream;!no_clip;{SOUND},0,{}",
        START.as_secs_f64(),
        FIRST_CUT_LENGTH.as_secs_f64(),
        SECOND_CUT_LENGTH.as_secs_f64(),
        timeline_length.as_secs_f64()
    );
    let player = Player::new();
    player.observe(Property::EofReached);
    let (frame_arrived, frames) = mpsc::channel();
    player.set_update_callback(move || {
        let _ = frame_arrived.send(());
    });
    let mut events = EventLog {
        player: &player,
        seen: Vec::new(),
    };

    player
        .load(Media::Uri(&uri), LoadOptions::default())
        .unwrap();
    events.wait_for(&Event::FileLoaded);
    frames.recv_timeout(EVENT_TIMEOUT).expect("a decoded frame");
    events.wait_for(&Event::PropertyChange(PropertyChange::EofReached(true)));
    assert_eq!(
        player.property(Property::Duration),
        PropertyChange::Duration(Some(timeline_length))
    );

    player.seek(FIRST_CUT_LENGTH).unwrap();
    events.wait_for(&Event::PlaybackRestart);
    wait_until(|| time_pos(&player) >= FIRST_CUT_LENGTH);
    player.stop();
    events.wait_for(&Event::EndFile(EndFile::Stop));
}

// the clip denoised, shrunk to half size and centred on black: dcpwizard's filters
const HALF_SIZE_ON_BLACK: [&str; 3] = [
    "hqdn3d",
    "scale=w=32:h=24:flags=lanczos",
    "pad=w=64:h=48:x=16:y=12:color=black",
];

const PICTURE_LEFT: u32 = 16;
const PICTURE_TOP: u32 = 12;
const PICTURE_WIDTH: u32 = 32;
const PICTURE_HEIGHT: u32 = 24;
// the render samples linearly, so the pixels next to the picture's edge blend it with the pad
const EDGE_BLEND: u32 = 1;
const DARK_LEVEL: u8 = 8;

fn is_dark(pixels: &[u8], x: u32, y: u32) -> bool {
    let at = ((y * CLIP_WIDTH + x) * RGBA_BYTES_PER_PIXEL) as usize;
    pixels[at..at + 3].iter().all(|&level| level < DARK_LEVEL)
}

fn in_pad(x: u32, y: u32) -> bool {
    let outside = |position: u32, start: u32, length: u32| {
        position + EDGE_BLEND < start || position >= start + length + EDGE_BLEND
    };
    outside(x, PICTURE_LEFT, PICTURE_WIDTH) || outside(y, PICTURE_TOP, PICTURE_HEIGHT)
}

fn shows_the_picture_on_a_black_pad(pixels: &[u8]) -> bool {
    let pad_is_black = (0..CLIP_HEIGHT)
        .flat_map(|y| (0..CLIP_WIDTH).map(move |x| (x, y)))
        .filter(|&(x, y)| in_pad(x, y))
        .all(|(x, y)| is_dark(pixels, x, y));
    pad_is_black && !is_dark(pixels, CLIP_WIDTH / 2, CLIP_HEIGHT / 2)
}

#[test]
fn video_filters_shrink_the_picture_onto_a_black_pad() {
    let Ok(context) = pollster::block_on(GpuContext::headless()) else {
        return;
    };
    let player = Player::new();
    let mut render = WgpuRenderContext::new(&player, context.clone());
    let target = rgba_target(&context);
    player.set_video_filters(&HALF_SIZE_ON_BLACK).unwrap();
    player
        .load(
            Media::Uri(&format!("file://{CLIP}")),
            LoadOptions::default(),
        )
        .unwrap();
    let mut wait_for_frame = |padded: bool| {
        let deadline = Instant::now() + EVENT_TIMEOUT;
        loop {
            assert!(
                Instant::now() < deadline,
                "no frame with padded={padded} within {EVENT_TIMEOUT:?}"
            );
            if !player.has_new_frame() {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
            let pixels = rendered_pixels(&mut render, &context, &target);
            if shows_the_picture_on_a_black_pad(&pixels) == padded {
                return;
            }
        }
    };
    wait_for_frame(true);
    player.set_video_filters(&[]).unwrap();
    wait_for_frame(false);
    player.stop();
}

// the clip padded to 2:1, which a 4:3 target letterboxes
const PADDED_WIDTH: u32 = 96;
const WIDENING_PAD: &str = "pad=w=96:h=48:x=16:y=0:color=black";

#[test]
fn video_params_osd_dimensions_and_the_filtered_size_are_reported() {
    let Ok(context) = pollster::block_on(GpuContext::headless()) else {
        return;
    };
    let player = Player::new();
    let mut render = WgpuRenderContext::new(&player, context.clone());
    let target = rgba_target(&context);
    player.set_video_filters(&[WIDENING_PAD]).unwrap();
    player
        .load(
            Media::Uri(&format!("file://{CLIP}")),
            LoadOptions::default(),
        )
        .unwrap();
    let decoded = VideoSize {
        width: CLIP_WIDTH,
        height: CLIP_HEIGHT,
    };
    let padded = VideoSize {
        width: PADDED_WIDTH,
        height: CLIP_HEIGHT,
    };
    wait_until(|| {
        player.property(Property::VideoOutParams) == PropertyChange::VideoOutParams(Some(padded))
    });
    assert_eq!(
        player.property(Property::VideoParams),
        PropertyChange::VideoParams(Some(decoded))
    );
    wait_until(|| {
        if player.has_new_frame() {
            rendered_pixels(&mut render, &context, &target);
        }
        let PropertyChange::OsdDimensions(Some(osd)) = player.property(Property::OsdDimensions)
        else {
            return false;
        };
        osd.margin_top > 0
    });
    let picture_height = CLIP_WIDTH * CLIP_HEIGHT / PADDED_WIDTH;
    let bar = (CLIP_HEIGHT - picture_height) / 2;
    assert_eq!(
        player.property(Property::OsdDimensions),
        PropertyChange::OsdDimensions(Some(OsdDimensions {
            width: CLIP_WIDTH,
            height: CLIP_HEIGHT,
            margin_left: 0,
            margin_top: bar,
            margin_right: 0,
            margin_bottom: bar,
        }))
    );
    player.stop();
    assert_eq!(
        player.property(Property::VideoParams),
        PropertyChange::VideoParams(None)
    );
}

const METER_WINDOW: Duration = Duration::from_millis(300);
const DB_TOLERANCE: f64 = 0.01;

#[test]
fn the_level_meter_reads_the_decoded_audio_in_db() {
    // audiotestsrc's square wave swings between plus and minus half of i16 full scale
    let half_scale = f64::from(i16::MAX / 2) / f64::from(i16::MAX as i32 + 1);
    let expected_db = 20.0 * half_scale.log10();
    let player = Player::new();
    player.set_level_meter(Some(METER_WINDOW));
    player
        .load(
            Media::LaunchLine(
                "videotestsrc ! playervideosink \
                 audiotestsrc wave=square channels=2 ! audioconvert ! audioresample ! autoaudiosink",
            ),
            LoadOptions::default(),
        )
        .unwrap();
    wait_until(|| player.audio_levels().is_some());
    let levels = player.audio_levels().expect("a reading");
    assert_eq!(levels.peak_db.len(), 2);
    for level_db in levels.peak_db.iter().chain(&levels.rms_db) {
        assert!(
            (level_db - expected_db).abs() < DB_TOLERANCE,
            "{level_db} dB, a half-scale square wave reads {expected_db} dB"
        );
    }
    player.set_level_meter(None);
    wait_until(|| player.audio_levels().is_none());
    player.stop();
}
