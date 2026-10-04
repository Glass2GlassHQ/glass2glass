#![cfg(feature = "std")]

use std::io::Write;
use std::process::{Command, Stdio};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{Colorimetry, PipelineClock, RawVideoFormat};
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::registry::default_registry;
use g2g_plugins::videoconvert::convert;

// odd on both axes, then odd width alone
const SIZES: [(u32, u32); 2] = [(37, 5), (37, 4)];
const FORMATS: [(RawVideoFormat, &str); 2] = [
    (RawVideoFormat::Nv12, "nv12"),
    (RawVideoFormat::P010, "p010le"),
];

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}

// one test-pattern frame as ffmpeg lays it out
fn ffmpeg_frame(width: u32, height: u32, pixel_format: &str) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-f", "lavfi", "-i"])
        .arg(format!("testsrc=size={width}x{height}"))
        .args([
            "-frames:v",
            "1",
            "-pix_fmt",
            pixel_format,
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("ffmpeg runs");
    assert!(out.status.success(), "ffmpeg failed: {out:?}");
    out.stdout
}

// ffmpeg's conversion of one raw frame from `from` to `to`
fn ffmpeg_convert(frame: &[u8], width: u32, height: u32, from: &str, to: &str) -> Vec<u8> {
    ffmpeg_filter(frame, width, height, from, "null", to)
}

// one raw frame through an ffmpeg video filter, converted from `from` to `to`
fn ffmpeg_filter(
    frame: &[u8],
    width: u32,
    height: u32,
    from: &str,
    filter: &str,
    to: &str,
) -> Vec<u8> {
    let mut child = Command::new("ffmpeg")
        .args(["-v", "error", "-f", "rawvideo", "-pix_fmt", from, "-s"])
        .arg(format!("{width}x{height}"))
        .args([
            "-i", "-", "-vf", filter, "-pix_fmt", to, "-f", "rawvideo", "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("ffmpeg runs");
    let mut stdin = child.stdin.take().expect("piped stdin");
    let input = frame.to_vec();
    let feeder = std::thread::spawn(move || stdin.write_all(&input).expect("feed ffmpeg"));
    let out = child.wait_with_output().expect("ffmpeg finishes");
    feeder.join().expect("feeder thread");
    assert!(out.status.success(), "ffmpeg failed: {out:?}");
    out.stdout
}

// odd and even on each axis, and the smallest frame
const CONVERT_SIZES: [(u32, u32); 4] = [(37, 5), (37, 4), (38, 5), (1, 1)];
const CONVERT_FORMATS: [(RawVideoFormat, &str); 3] = [
    (RawVideoFormat::Nv12, "nv12"),
    (RawVideoFormat::I420, "yuv420p"),
    (RawVideoFormat::Yuyv, "yuyv422"),
];

// the bound a lossy webp decode is held to against libwebp, whose chroma filter differs too
const MEAN_ABS_DIFF_BOUND: f64 = 3.0;

const GRADIENT_BASE: [usize; 3] = [40, 60, 200];
const GRADIENT_STEP: usize = 4;

// red rises along x, green along y, blue falls along both
fn gradient_rgba(width: u32, height: u32) -> Vec<u8> {
    let [red, green, blue] = GRADIENT_BASE;
    (0..height as usize)
        .flat_map(|y| {
            (0..width as usize).flat_map(move |x| {
                [
                    (red + GRADIENT_STEP * x) as u8,
                    (green + GRADIENT_STEP * y) as u8,
                    (blue - GRADIENT_STEP * (x + y)) as u8,
                    u8::MAX,
                ]
            })
        })
        .collect()
}

fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "same sample count");
    let total: u64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| u64::from(x.abs_diff(*y)))
        .sum();
    total as f64 / a.len() as f64
}

#[test]
fn rgba_converts_at_odd_sizes_like_ffmpeg() {
    let ffmpeg = have_ffmpeg();
    if !ffmpeg {
        eprintln!("ffmpeg not present: checking sizes only");
    }
    for (width, height) in CONVERT_SIZES {
        let (w, h) = (width as usize, height as usize);
        let rgba = gradient_rgba(width, height);
        for (format, pixel_format) in CONVERT_FORMATS {
            let label = format!("{pixel_format} {width}x{height}");
            let ours = convert(
                &rgba,
                RawVideoFormat::Rgba8,
                format,
                w,
                h,
                Colorimetry::UNKNOWN,
            );
            assert_eq!(
                Some(ours.len() as u64),
                format.unpadded_frame_bytes(width, height),
                "{label}"
            );
            if !ffmpeg {
                continue;
            }
            let reference = ffmpeg_convert(&rgba, width, height, "rgba", pixel_format);
            assert_eq!(ours.len(), reference.len(), "{label}");
            let drift = mean_abs_diff(&ours, &reference);
            assert!(drift < MEAN_ABS_DIFF_BOUND, "rgba to {label}: {drift}");

            let back = convert(
                &reference,
                format,
                RawVideoFormat::Rgba8,
                w,
                h,
                Colorimetry::UNKNOWN,
            );
            let back_reference = ffmpeg_convert(&reference, width, height, pixel_format, "rgba");
            let drift = mean_abs_diff(&back, &back_reference);
            assert!(drift < MEAN_ABS_DIFF_BOUND, "{label} to rgba: {drift}");
        }
    }
}

#[test]
fn nv12_repacks_to_i420_at_odd_sizes_like_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    for (width, height) in CONVERT_SIZES {
        let (w, h) = (width as usize, height as usize);
        let nv12 = ffmpeg_frame(width, height, "nv12");
        let i420 = ffmpeg_convert(&nv12, width, height, "nv12", "yuv420p");
        let repacked = convert(
            &nv12,
            RawVideoFormat::Nv12,
            RawVideoFormat::I420,
            w,
            h,
            Colorimetry::UNKNOWN,
        );
        assert_eq!(*repacked, *i420, "{width}x{height}");
        let back = convert(
            &i420,
            RawVideoFormat::I420,
            RawVideoFormat::Nv12,
            w,
            h,
            Colorimetry::UNKNOWN,
        );
        assert_eq!(*back, *nv12, "{width}x{height}");
    }
}

// the subsampled formats videoconvert produces, by their caps name
const LAUNCH_TARGETS: [(RawVideoFormat, &str); 3] = [
    (RawVideoFormat::Nv12, "NV12"),
    (RawVideoFormat::I420, "I420"),
    (RawVideoFormat::Yuyv, "YUY2"),
];

// even width, then odd on both axes
const YUYV_SIZES: [(u32, u32); 2] = [(38, 4), (37, 5)];
const YUYV_CAPS_NAME: &str = "YUY2";
const YUYV_PIXEL_FORMAT: &str = "yuyv422";
// the formats videoconvert packs to YUYV, by caps name and ffmpeg pixel format
const YUYV_SOURCES: [(&str, &str); 3] = [("RGBA", "rgba"), ("NV12", "nv12"), ("I420", "yuv420p")];

// one frame through `appsrc ! {elements} ! appsink`, as the appsink received it
async fn launch_one_frame(
    frame: &[u8],
    format: &str,
    (width, height): (u32, u32),
    elements: &str,
    channel: &str,
) -> Vec<u8> {
    let in_channel = format!("odd_{channel}_in");
    let out_channel = format!("odd_{channel}_out");
    let feed = register_appsrc(&in_channel);
    assert!(feed.push(frame, 0));
    feed.end_of_stream();
    let pull = register_appsink_pull(&out_channel);
    let line = format!(
        "appsrc channel={in_channel} caps=video/x-raw,format={format},width={width},height={height},framerate=30/1 \
         ! {elements} ! appsink channel={out_channel}"
    );
    let graph = parse_launch(&default_registry(), &line).expect("parses");
    run_graph(graph, &ZeroClock, 4).await.expect("runs");
    let Pull::Frame(out) = pull.try_pull() else {
        panic!("no frame reached the appsink of `{line}`");
    };
    out.domain
        .as_system_slice()
        .expect("system memory")
        .to_vec()
}

#[tokio::test]
async fn videoconvert_launches_at_an_odd_size() {
    let size = CONVERT_SIZES[0];
    let (width, height) = size;
    let rgba = gradient_rgba(width, height);
    for (format, name) in LAUNCH_TARGETS {
        let out = launch_one_frame(
            &rgba,
            "RGBA",
            size,
            &format!("videoconvert ! video/x-raw,format={name}"),
            &format!("convert_{name}"),
        )
        .await;
        let expected = convert(
            &rgba,
            RawVideoFormat::Rgba8,
            format,
            width as usize,
            height as usize,
            Colorimetry::UNKNOWN,
        );
        assert_eq!(out, *expected, "{name}");
    }
}

#[tokio::test]
async fn videoconvert_format_property_produces_yuyv() {
    for size in YUYV_SIZES {
        let (width, height) = size;
        let rgba = gradient_rgba(width, height);
        let out = launch_one_frame(
            &rgba,
            "RGBA",
            size,
            &format!("videoconvert format={YUYV_CAPS_NAME}"),
            &format!("yuyv_property_{width}x{height}"),
        )
        .await;
        assert_eq!(
            Some(out.len() as u64),
            RawVideoFormat::Yuyv.unpadded_frame_bytes(width, height),
            "{width}x{height}"
        );
        let expected = convert(
            &rgba,
            RawVideoFormat::Rgba8,
            RawVideoFormat::Yuyv,
            width as usize,
            height as usize,
            Colorimetry::UNKNOWN,
        );
        assert_eq!(out, *expected, "{width}x{height}");
    }
}

#[tokio::test]
async fn videoconvert_packs_yuyv_like_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    for size in YUYV_SIZES {
        let (width, height) = size;
        let rgba = gradient_rgba(width, height);
        for (name, pixel_format) in YUYV_SOURCES {
            let label = format!("{name} {width}x{height}");
            let frame = ffmpeg_convert(&rgba, width, height, "rgba", pixel_format);
            let out = launch_one_frame(
                &frame,
                name,
                size,
                &format!("videoconvert format={YUYV_CAPS_NAME}"),
                &format!("yuyv_from_{name}_{width}x{height}"),
            )
            .await;
            let reference = ffmpeg_convert(&frame, width, height, pixel_format, YUYV_PIXEL_FORMAT);
            assert_eq!(out.len(), reference.len(), "{label}");
            let drift = mean_abs_diff(&out, &reference);
            assert!(drift < MEAN_ABS_DIFF_BOUND, "{label} to yuyv: {drift}");
        }
    }
}

#[tokio::test]
async fn yuyv_round_trips_through_rgba() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    for size in YUYV_SIZES {
        let (width, height) = size;
        let rgba = gradient_rgba(width, height);
        let yuyv = ffmpeg_convert(&rgba, width, height, "rgba", YUYV_PIXEL_FORMAT);
        let back = launch_one_frame(
            &yuyv,
            YUYV_CAPS_NAME,
            size,
            &format!("videoconvert format=RGBA ! videoconvert format={YUYV_CAPS_NAME}"),
            &format!("yuyv_round_trip_{width}x{height}"),
        )
        .await;
        let drift = mean_abs_diff(&back, &yuyv);
        assert!(drift < MEAN_ABS_DIFF_BOUND, "{width}x{height}: {drift}");
    }
}

#[tokio::test]
async fn videocrop_crops_an_odd_frame_like_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    // even insets on an odd frame leave an odd crop
    const TOP: u32 = 2;
    const LEFT: u32 = 2;
    const RIGHT: u32 = 2;
    let size = CONVERT_SIZES[0];
    let (width, height) = size;
    let (out_w, out_h) = (width - LEFT - RIGHT, height - TOP);
    for (name, pixel_format) in [("NV12", "nv12"), ("I420", "yuv420p")] {
        let frame = ffmpeg_frame(width, height, pixel_format);
        let out = launch_one_frame(
            &frame,
            name,
            size,
            &format!("videocrop top={TOP} left={LEFT} right={RIGHT}"),
            &format!("crop_{name}"),
        )
        .await;
        // without exact=1 ffmpeg rounds a 4:2:0 crop size down to even
        let reference = ffmpeg_filter(
            &frame,
            width,
            height,
            pixel_format,
            &format!("crop={out_w}:{out_h}:{LEFT}:{TOP}:exact=1"),
            pixel_format,
        );
        assert_eq!(out, reference, "{name}");
    }
}

#[test]
fn the_tight_size_matches_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    for (format, pixel_format) in FORMATS {
        for (width, height) in SIZES {
            assert_eq!(
                Some(ffmpeg_frame(width, height, pixel_format).len() as u64),
                format.unpadded_frame_bytes(width, height),
                "{pixel_format} {width}x{height}"
            );
        }
    }
}

#[tokio::test]
async fn an_ffmpeg_nv12_frame_composites_unchanged() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    for (width, height) in SIZES {
        let frame = ffmpeg_frame(width, height, "nv12");
        let out = launch_one_frame(
            &frame,
            "NV12",
            (width, height),
            &format!("compositor width={width} height={height} format=nv12"),
            &format!("compositor_{width}x{height}"),
        )
        .await;
        assert_eq!(out, frame, "{width}x{height}");
    }
}
