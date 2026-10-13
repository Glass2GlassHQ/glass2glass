#![cfg(feature = "std")]

use g2g_core::frame::Frame;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{Colorimetry, PipelineClock, RawVideoFormat};
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::registry::default_registry;
use g2g_plugins::videoconvert::convert;

const RGBA_BYTES_PER_PIXEL: usize = 4;
const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
const FRAMES: u64 = 2;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn run_into_appsink(channel: &str, source_and_filter: &str) -> Vec<Frame> {
    let pull = register_appsink_pull(channel);
    let launch = format!("{source_and_filter} ! appsink channel={channel}");
    let graph = parse_launch(&default_registry(), &launch).expect("parses");
    run_graph(graph, &ZeroClock, 4).await.expect("runs");
    let mut frames = Vec::new();
    while let Pull::Frame(frame) = pull.try_pull() {
        frames.push(frame);
    }
    assert_eq!(frames.len() as u64, FRAMES, "every frame reaches the sink");
    frames
}

fn frame_bytes(frame: &Frame) -> usize {
    frame.domain.as_system_slice().expect("system memory").len()
}

#[tokio::test]
async fn frames_take_the_size_a_caps_filter_asks_for() {
    const WIDTH: usize = 1280;
    const HEIGHT: usize = 720;
    let frames = run_into_appsink(
        "videotestsrc_caps_filter_size",
        &format!("videotestsrc num-buffers={FRAMES} ! video/x-raw,width={WIDTH},height={HEIGHT}"),
    )
    .await;
    for frame in &frames {
        assert_eq!(frame_bytes(frame), WIDTH * HEIGHT * RGBA_BYTES_PER_PIXEL);
    }
}

#[tokio::test]
async fn fields_the_filter_leaves_unset_keep_the_source_properties() {
    const SOURCE_HEIGHT: usize = 180;
    const FILTER_WIDTH: usize = 640;
    const FILTER_FPS: u64 = 15;
    let frames = run_into_appsink(
        "videotestsrc_caps_filter_partial",
        &format!(
            "videotestsrc num-buffers={FRAMES} height={SOURCE_HEIGHT} \
             ! video/x-raw,width={FILTER_WIDTH},framerate={FILTER_FPS}/1"
        ),
    )
    .await;
    for frame in &frames {
        assert_eq!(
            frame_bytes(frame),
            FILTER_WIDTH * SOURCE_HEIGHT * RGBA_BYTES_PER_PIXEL
        );
        assert_eq!(
            frame.timing.duration_ns,
            NANOSECONDS_PER_SECOND / FILTER_FPS
        );
    }
}

#[tokio::test]
async fn frames_take_the_format_a_caps_filter_asks_for() {
    const CASES: [(&str, RawVideoFormat, u32, u32); 6] = [
        ("I420", RawVideoFormat::I420, 64, 48),
        ("NV12", RawVideoFormat::Nv12, 64, 48),
        ("BGRA", RawVideoFormat::Bgra8, 64, 48),
        ("I420_10LE", RawVideoFormat::I420p10, 64, 48),
        ("YUY2", RawVideoFormat::Yuyv, 64, 48),
        ("NV12", RawVideoFormat::Nv12, 33, 17),
    ];
    for (name, format, width, height) in CASES {
        let frames = run_into_appsink(
            &format!("videotestsrc_caps_filter_format_{name}_{width}x{height}"),
            &format!(
                "videotestsrc num-buffers={FRAMES} \
                 ! video/x-raw,format={name},width={width},height={height}"
            ),
        )
        .await;
        let expected = format.unpadded_frame_bytes(width, height).expect("fits") as usize;
        for frame in &frames {
            assert_eq!(frame_bytes(frame), expected, "{name} {width}x{height}");
        }
    }
}

#[tokio::test]
async fn converted_frames_carry_the_drawn_pattern() {
    const WIDTH: usize = 70;
    const HEIGHT: usize = 40;
    const SMPTE_BARS: usize = 7;
    const ROUND_TRIP_TOLERANCE: u8 = 4;
    let source = format!("videotestsrc num-buffers={FRAMES} pattern=smpte");
    let geometry = format!("width={WIDTH},height={HEIGHT}");
    let drawn = run_into_appsink(
        "videotestsrc_caps_filter_pattern_rgba",
        &format!("{source} ! video/x-raw,format=RGBA,{geometry}"),
    )
    .await;
    let converted = run_into_appsink(
        "videotestsrc_caps_filter_pattern_i420",
        &format!("{source} ! video/x-raw,format=I420,{geometry}"),
    )
    .await;
    let drawn = drawn[0].domain.as_system_slice().expect("system memory");
    let converted = converted[0]
        .domain
        .as_system_slice()
        .expect("system memory");
    let back = convert(
        converted,
        RawVideoFormat::I420,
        RawVideoFormat::Rgba8,
        WIDTH,
        HEIGHT,
        Colorimetry::UNKNOWN,
    );
    let row = HEIGHT / 2;
    for bar in 0..SMPTE_BARS {
        let column = (2 * bar + 1) * WIDTH / (2 * SMPTE_BARS);
        let pixel = (row * WIDTH + column) * RGBA_BYTES_PER_PIXEL;
        for channel in pixel..pixel + RGBA_BYTES_PER_PIXEL {
            assert!(
                drawn[channel].abs_diff(back[channel]) <= ROUND_TRIP_TOLERANCE,
                "bar {bar} byte {channel}: drawn {} came back {}",
                drawn[channel],
                back[channel]
            );
        }
    }
}
