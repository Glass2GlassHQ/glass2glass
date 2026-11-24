// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "display"))]

use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::runtime::{parse_launch, run_graph, SourceLoop};
use g2g_core::{
    Caps, Colorimetry, Dim, G2gError, Interlace, PipelineClock, PixelAspect, Rate, RawVideoFormat,
};
use g2g_plugins::appsink::register_appsink_pull;
use g2g_plugins::registry::default_registry;
use g2g_wayland_display::waylanddisplaysrc::{WaylandDisplaySrc, SOFTWARE_RENDERING};

const RGBA_BYTES_PER_PIXEL: usize = 4;
const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
const TIMEOUT: Duration = Duration::from_secs(10);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn first_frame(channel: &str, source_and_filter: &str) -> Frame {
    let pull = register_appsink_pull(channel);
    let mut registry = default_registry();
    g2g_wayland_display::register(&mut registry);
    let launch = format!("{source_and_filter} ! appsink channel={channel}");
    let graph = parse_launch(&registry, &launch).expect("parses");
    tokio::time::timeout(TIMEOUT, run_graph(graph, &ZeroClock, 2))
        .await
        .expect("one frame ends the run")
        .expect("runs");
    pull.pull().await.expect("the frame reaches the sink")
}

fn picture_bytes(frame: &Frame) -> usize {
    frame.domain.as_system_slice().expect("system memory").len()
}

#[tokio::test]
async fn frames_take_the_size_and_rate_a_caps_filter_asks_for() {
    const WIDTH: usize = 48;
    const HEIGHT: usize = 32;
    const FPS: u64 = 15;
    let frame = first_frame(
        "m2541_filter_size_and_rate",
        &format!(
            "waylanddisplaysrc render-node={SOFTWARE_RENDERING} num-buffers=1 \
             ! video/x-raw,width={WIDTH},height={HEIGHT},framerate={FPS}/1"
        ),
    )
    .await;
    assert_eq!(picture_bytes(&frame), WIDTH * HEIGHT * RGBA_BYTES_PER_PIXEL);
    assert_eq!(frame.timing.duration_ns, NANOSECONDS_PER_SECOND / FPS);
}

#[tokio::test]
async fn fields_the_filter_leaves_unset_keep_the_source_properties() {
    const SOURCE_HEIGHT: usize = 24;
    const SOURCE_FPS: u64 = 20;
    const FILTER_WIDTH: usize = 40;
    let frame = first_frame(
        "m2541_filter_partial",
        &format!(
            "waylanddisplaysrc render-node={SOFTWARE_RENDERING} num-buffers=1 \
             height={SOURCE_HEIGHT} framerate={SOURCE_FPS} ! video/x-raw,width={FILTER_WIDTH}"
        ),
    )
    .await;
    assert_eq!(
        picture_bytes(&frame),
        FILTER_WIDTH * SOURCE_HEIGHT * RGBA_BYTES_PER_PIXEL
    );
    assert_eq!(
        frame.timing.duration_ns,
        NANOSECONDS_PER_SECOND / SOURCE_FPS
    );
}

const NV12_FPS: u32 = 30;

fn nv12_caps(width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        framerate: Rate::Fixed(NV12_FPS << 16),
        interlace: Interlace::Progressive,
        colorimetry: Colorimetry::BT709,
        pixel_aspect: PixelAspect::SQUARE,
    }
}

#[test]
fn nv12_takes_only_even_sizes() {
    const EVEN: u32 = 64;
    const ODD: u32 = 65;
    let mut source = WaylandDisplaySrc::new();
    assert!(source.configure_pipeline(&nv12_caps(EVEN, EVEN)).is_ok());
    for (width, height) in [(ODD, EVEN), (EVEN, ODD)] {
        assert!(matches!(
            source.configure_pipeline(&nv12_caps(width, height)),
            Err(G2gError::CapsMismatch)
        ));
    }
}
