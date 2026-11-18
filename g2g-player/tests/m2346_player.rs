// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(feature = "openh264", feature = "wgpu"))]

use std::sync::mpsc;
use std::time::{Duration, Instant};

use g2g_player::{
    AudioParams, ChannelLayout, EndFile, Event, LoadOptions, Media, OsdDimensions,
    OverlayRectangle, Player, Property, PropertyChange, SubtitleSlot, VideoSize, WgpuRenderContext,
};
use g2g_plugins::gpu::{read_texture_plane, whole_texture, GpuContext};
use g2g_plugins::launched::LaunchedPipeline;

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

fn rgba_target(context: &GpuContext, (width, height): (u32, u32)) -> wgpu::Texture {
    context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("m2346-target"),
        size: wgpu::Extent3d {
            width,
            height,
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
        whole_texture(target.width(), target.height()),
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

    let target = rgba_target(&context, (CLIP_WIDTH, CLIP_HEIGHT));
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

// time-pos leaves an off-grid target once the seek's frame is shown
const PAUSED_SEEKS: [Duration; 2] = [Duration::from_millis(1_610), Duration::from_millis(1_210)];
const PLAYING_SEEKS: [Duration; 2] = [Duration::from_millis(410), Duration::from_millis(210)];
const PLAY_WINDOW: Duration = Duration::from_millis(600);

fn seek_lands(player: &Player, target: Duration, frames_past_target: u32) -> Duration {
    player.seek(target).unwrap();
    let landed = moved_from(player, target);
    // time-pos is the frame the sink last took, which while playing can be the one after the frame shown
    assert!(
        landed >= target && landed < target + CLIP_FRAME_PERIOD * frames_past_target,
        "a seek to {target:?} showed {landed:?}"
    );
    landed
}

fn plays_on(player: &Player) {
    let before = time_pos(player);
    std::thread::sleep(PLAY_WINDOW);
    let played = time_pos(player).saturating_sub(before);
    assert!(
        played >= PLAY_WINDOW / 2,
        "time-pos moved {played:?} in {PLAY_WINDOW:?} of playing"
    );
}

#[test]
fn seeks_between_pauses_and_plays_land_and_play_on() {
    let player = Player::new();
    player
        .load(
            Media::Uri(&format!("file://{CLIP}")),
            LoadOptions::default(),
        )
        .unwrap();
    wait_until(|| {
        matches!(
            player.property(Property::TimePos),
            PropertyChange::TimePos(Some(position)) if position >= PLAYED_BEFORE_PAUSE
        )
    });
    for (paused_target, playing_target) in PAUSED_SEEKS.into_iter().zip(PLAYING_SEEKS) {
        player.pause();
        let landed = seek_lands(&player, paused_target, 1);
        std::thread::sleep(SETTLE);
        assert_eq!(
            time_pos(&player),
            landed,
            "time-pos moved after a paused seek"
        );
        player.play();
        plays_on(&player);
        seek_lands(&player, playing_target, 2);
        plays_on(&player);
    }
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

#[test]
fn time_pos_shows_a_seek_target_before_its_frame_lands() {
    let player = Player::new();
    let start = load_paused_at_start(&player);
    let target = start + RELATIVE_SEEK;
    player.seek(target).unwrap();
    assert_eq!(
        player.property(Property::TimePos),
        PropertyChange::TimePos(Some(target))
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

const SOUND_LENGTH: Duration = Duration::from_secs(1);
// audiotestsrc emits ten millisecond buffers
const TEST_TONE_BUFFER: Duration = Duration::from_millis(10);
const POSITION_WATCH: Duration = Duration::from_millis(600);
// the video sink sets its position as a frame arrives, before holding it until its time
const POSITION_LEAD: Duration = CLIP_FRAME_PERIOD.saturating_mul(2);

fn write_48k_sound(file_name: &str) -> std::path::PathBuf {
    // wavparse reads a 48 kHz file in buffers of about a third of a second, which a sink queues ahead of playout
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(file_name);
    let buffers = SOUND_LENGTH.as_millis() / TEST_TONE_BUFFER.as_millis();
    let line = format!(
        "audiotestsrc wave=silence samplerate=48000 channels=2 num-buffers={buffers} ! wavenc ! filesink location={}",
        path.display()
    );
    LaunchedPipeline::launch(&line, "m2346-sound")
        .unwrap()
        .wait()
        .unwrap();
    path
}

fn plays_no_further_than_the_time_since_loading(uri: &str) {
    let player = Player::new();
    let loaded = Instant::now();
    player
        .load(Media::Uri(uri), LoadOptions::default())
        .unwrap();
    while loaded.elapsed() < POSITION_WATCH {
        if let PropertyChange::TimePos(Some(position)) = player.property(Property::TimePos) {
            assert!(
                position <= loaded.elapsed() + POSITION_LEAD,
                "time-pos {position:?} {:?} after the load",
                loaded.elapsed()
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(time_pos(&player) > Duration::ZERO);
    player.stop();
}

#[test]
fn time_pos_follows_the_sound_played_not_the_sound_queued() {
    let sound = write_48k_sound("m2346_48k_beside_the_clip.wav");
    let uri = format!(
        "edl://{CLIP},0,{length};!new_stream;!no_clip;{},0,{length}",
        sound.display(),
        length = SOUND_LENGTH.as_secs_f64()
    );
    plays_no_further_than_the_time_since_loading(&uri);
}

#[test]
fn time_pos_follows_a_sound_file_played_alone() {
    let sound = write_48k_sound("m2346_48k_alone.wav");
    plays_no_further_than_the_time_since_loading(&format!("file://{}", sound.display()));
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
    let target = rgba_target(&context, (CLIP_WIDTH, CLIP_HEIGHT));
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
            if !player.wants_redraw() {
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
    let target = rgba_target(&context, (CLIP_WIDTH, CLIP_HEIGHT));
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
        if player.wants_redraw() {
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

// twice the clip's width, so the clip is drawn at its own size between two bars
const WIDE_TARGET: (u32, u32) = (2 * CLIP_WIDTH, CLIP_HEIGHT);
const PILLAR_WIDTH: u32 = (WIDE_TARGET.0 - CLIP_WIDTH) / 2;
const OVERLAY_RED: [u8; 3] = [255, 0, 0];
const MARKED_AREA: OverlayRectangle = OverlayRectangle {
    x: 8,
    y: 12,
    width: 16,
    height: 12,
    colour: OVERLAY_RED,
    alpha: u8::MAX,
};

fn rgb_at(pixels: &[u8], target_width: u32, x: u32, y: u32) -> [u8; 3] {
    let at = ((y * target_width + x) * RGBA_BYTES_PER_PIXEL) as usize;
    [pixels[at], pixels[at + 1], pixels[at + 2]]
}

#[test]
fn an_overlay_lands_on_its_picture_pixels_and_redraws_a_paused_frame() {
    let Ok(context) = pollster::block_on(GpuContext::headless()) else {
        return;
    };
    let player = Player::new();
    let mut render = WgpuRenderContext::new(&player, context.clone());
    let target = rgba_target(&context, WIDE_TARGET);
    load_paused_at_start(&player);
    wait_until(|| player.wants_redraw());
    let plain = rendered_pixels(&mut render, &context, &target);
    assert!(!player.wants_redraw());
    let inside = (
        PILLAR_WIDTH + (MARKED_AREA.x + MARKED_AREA.width / 2) as u32,
        (MARKED_AREA.y + MARKED_AREA.height / 2) as u32,
    );
    let left_of_it = (PILLAR_WIDTH + MARKED_AREA.x as u32 - 1, inside.1);
    let above_it = (inside.0, MARKED_AREA.y as u32 - 1);
    let colour = |pixels: &[u8], (x, y): (u32, u32)| rgb_at(pixels, WIDE_TARGET.0, x, y);
    assert_ne!(colour(&plain, inside), OVERLAY_RED);

    player.set_overlay(vec![MARKED_AREA]);
    assert!(player.wants_redraw());
    let marked = rendered_pixels(&mut render, &context, &target);
    assert_eq!(colour(&marked, inside), OVERLAY_RED);
    assert_eq!(colour(&marked, left_of_it), colour(&plain, left_of_it));
    assert_eq!(colour(&marked, above_it), colour(&plain, above_it));

    player.set_overlay(Vec::new());
    assert!(player.wants_redraw());
    let cleared = rendered_pixels(&mut render, &context, &target);
    assert_eq!(colour(&cleared, inside), colour(&plain, inside));
    player.stop();
}

// SRT times of a cue over the frame at START and of one that ends a frame period before it
const CUE_OVER_START: &str = "1\n00:00:00,800 --> 00:00:01,200\nSUBTITLE\n";
const CUE_BEFORE_START: &str = "1\n00:00:00,000 --> 00:00:00,960\nSUBTITLE\n";

fn subtitle_file(name: &str, cues: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::write(&path, cues).unwrap();
    path
}

// a 48 pixel high picture has no room for a line of text in one half
const SUBTITLE_TARGET: (u32, u32) = (4 * CLIP_WIDTH, 4 * CLIP_HEIGHT);

fn rows(pixels: &[u8], rows: std::ops::Range<u32>) -> &[u8] {
    let row_bytes = (SUBTITLE_TARGET.0 * RGBA_BYTES_PER_PIXEL) as usize;
    &pixels[rows.start as usize * row_bytes..rows.end as usize * row_bytes]
}

#[test]
fn subtitles_draw_in_their_slot_only_while_a_cue_covers_the_frame() {
    let Ok(context) = pollster::block_on(GpuContext::headless()) else {
        return;
    };
    let over_start = subtitle_file("m2385_over_start.srt", CUE_OVER_START);
    let before_start = subtitle_file("m2385_before_start.srt", CUE_BEFORE_START);
    let player = Player::new();
    let mut render = WgpuRenderContext::new(&player, context.clone());
    let target = rgba_target(&context, SUBTITLE_TARGET);
    load_paused_at_start(&player);
    wait_until(|| player.wants_redraw());
    let plain = rendered_pixels(&mut render, &context, &target);
    let top = 0..SUBTITLE_TARGET.1 / 2;
    let bottom = SUBTITLE_TARGET.1 / 2..SUBTITLE_TARGET.1;
    let mut redrawn = |player: &Player| {
        assert!(player.wants_redraw());
        rendered_pixels(&mut render, &context, &target)
    };

    player
        .set_subtitle_file(SubtitleSlot::Subtitle, Some(&over_start))
        .unwrap();
    let subtitled = redrawn(&player);
    assert_ne!(
        rows(&subtitled, bottom.clone()),
        rows(&plain, bottom.clone())
    );
    assert_eq!(rows(&subtitled, top.clone()), rows(&plain, top.clone()));

    player.set_subtitle_visibility(SubtitleSlot::Subtitle, false);
    assert_eq!(redrawn(&player), plain);
    player.set_subtitle_visibility(SubtitleSlot::Subtitle, true);
    player
        .set_subtitle_file(SubtitleSlot::Subtitle, None)
        .unwrap();
    player
        .set_subtitle_file(SubtitleSlot::Caption, Some(&over_start))
        .unwrap();
    let captioned = redrawn(&player);
    assert_ne!(rows(&captioned, top.clone()), rows(&plain, top));
    assert_eq!(rows(&captioned, bottom.clone()), rows(&plain, bottom));

    player
        .set_subtitle_file(SubtitleSlot::Caption, Some(&before_start))
        .unwrap();
    assert_eq!(redrawn(&player), plain);

    player
        .set_subtitle_file(SubtitleSlot::Caption, Some(&over_start))
        .unwrap();
    player.stop();
    load_paused_at_start(&player);
    wait_until(|| player.wants_redraw());
    assert_eq!(rendered_pixels(&mut render, &context, &target), plain);
    player.stop();
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

#[test]
fn audio_params_name_the_decoded_channels() {
    const SAMPLE_RATE: u32 = 44_100;
    const CHANNELS: u8 = 6;
    let player = Player::new();
    player.observe(Property::AudioParams);
    player
        .load(
            Media::LaunchLine(&format!(
                "videotestsrc ! playervideosink \
                 audiotestsrc samplerate={SAMPLE_RATE} channels={CHANNELS} ! audioconvert ! audioresample ! autoaudiosink"
            )),
            LoadOptions::default(),
        )
        .unwrap();
    wait_until(|| {
        matches!(
            player.property(Property::AudioParams),
            PropertyChange::AudioParams(Some(_))
        )
    });
    assert_eq!(
        player.property(Property::AudioParams),
        PropertyChange::AudioParams(Some(AudioParams {
            sample_rate: SAMPLE_RATE,
            channel_count: CHANNELS,
            channel_layout: ChannelLayout::SURROUND_5_1,
        }))
    );
    player.stop();
    assert_eq!(
        player.property(Property::AudioParams),
        PropertyChange::AudioParams(None)
    );
}

#[test]
fn container_fps_is_the_video_frame_rate() {
    let player = Player::new();
    player.observe(Property::ContainerFps);
    player
        .load(
            Media::Uri(&format!("file://{CLIP}")),
            LoadOptions {
                start: None,
                pause: Some(true),
            },
        )
        .unwrap();
    let deadline = Instant::now() + EVENT_TIMEOUT;
    let first_reported = loop {
        assert!(
            Instant::now() < deadline,
            "no frame rate within {EVENT_TIMEOUT:?}"
        );
        if let Some(Event::PropertyChange(PropertyChange::ContainerFps(Some(fps)))) =
            player.wait_event(deadline - Instant::now())
        {
            break fps;
        }
    };
    assert_eq!(first_reported, f64::from(CLIP_FRAMES_PER_SECOND));
    player.stop();
    assert_eq!(
        player.property(Property::ContainerFps),
        PropertyChange::ContainerFps(None)
    );
}

#[test]
fn video_params_first_reports_the_decoded_size() {
    // the matroska demuxer opens its port at a placeholder size, the mp4 one at the stored size
    const MKV_CLIP: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../g2g-plugins/tests/fixtures/h264_64x48_25fps_gop10.mkv"
    );
    for clip in [CLIP, MKV_CLIP] {
        let player = Player::new();
        player.observe(Property::VideoParams);
        player
            .load(
                Media::Uri(&format!("file://{clip}")),
                LoadOptions {
                    start: None,
                    pause: Some(true),
                },
            )
            .unwrap();
        let deadline = Instant::now() + EVENT_TIMEOUT;
        let first_reported = loop {
            assert!(
                Instant::now() < deadline,
                "{clip}: no video size within {EVENT_TIMEOUT:?}"
            );
            if let Some(Event::PropertyChange(PropertyChange::VideoParams(Some(size)))) =
                player.wait_event(deadline - Instant::now())
            {
                break size;
            }
        };
        assert_eq!(
            first_reported,
            VideoSize {
                width: CLIP_WIDTH,
                height: CLIP_HEIGHT,
            },
            "{clip}"
        );
    }
}

#[test]
fn an_flv_file_plays() {
    const FLV_CLIP: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../g2g-plugins/tests/fixtures/h264_64x48_25fps_gop10_bframes_offset.flv"
    );
    let player = Player::new();
    player
        .load(
            Media::Uri(&format!("file://{FLV_CLIP}")),
            LoadOptions {
                start: None,
                pause: Some(true),
            },
        )
        .unwrap();
    wait_until(|| {
        player.property(Property::VideoParams)
            == PropertyChange::VideoParams(Some(VideoSize {
                width: CLIP_WIDTH,
                height: CLIP_HEIGHT,
            }))
    });
}

#[cfg(feature = "ffmpeg")]
#[test]
fn a_jpeg_2000_picture_with_no_matching_raw_format_plays() {
    // 48-bit rgb, which g2g has no raw format for, see tools/mxf-fixtures.sh
    const MXF_CLIP: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../g2g-plugins/tests/fixtures/mxf_jpeg2000_rgb48.mxf"
    );
    let player = Player::new();
    player.observe(Property::EofReached);
    let mut events = EventLog {
        player: &player,
        seen: Vec::new(),
    };
    player
        .load(
            Media::Uri(&format!("file://{MXF_CLIP}")),
            LoadOptions {
                start: None,
                pause: Some(false),
            },
        )
        .unwrap();
    events.wait_for(&Event::PropertyChange(PropertyChange::EofReached(true)));
    assert_eq!(
        player.property(Property::VideoParams),
        PropertyChange::VideoParams(Some(VideoSize {
            width: CLIP_WIDTH,
            height: CLIP_HEIGHT,
        }))
    );
}

#[test]
fn an_mxf_file_reports_its_size_rate_and_duration() {
    const MXF_CLIP: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../g2g-plugins/tests/fixtures/mxf_h264.mxf"
    );
    // ten frames, see tools/mxf-fixtures.sh
    const MXF_FRAMES: u32 = 10;
    let player = Player::new();
    player
        .load(
            Media::Uri(&format!("file://{MXF_CLIP}")),
            LoadOptions {
                start: None,
                pause: Some(true),
            },
        )
        .unwrap();
    wait_until(|| {
        matches!(
            player.property(Property::ContainerFps),
            PropertyChange::ContainerFps(Some(_))
        )
    });
    assert_eq!(
        player.property(Property::ContainerFps),
        PropertyChange::ContainerFps(Some(f64::from(CLIP_FRAMES_PER_SECOND)))
    );
    assert_eq!(
        player.property(Property::VideoParams),
        PropertyChange::VideoParams(Some(VideoSize {
            width: CLIP_WIDTH,
            height: CLIP_HEIGHT,
        }))
    );
    assert_eq!(
        player.property(Property::Duration),
        PropertyChange::Duration(Some(CLIP_FRAME_PERIOD * MXF_FRAMES))
    );
    player.stop();
}
