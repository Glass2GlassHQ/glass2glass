#![cfg(feature = "std")]

use std::process::Command;

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{PipelineClock, RawVideoFormat};
use g2g_plugins::appsink::{register_appsink_pull, Pull};
use g2g_plugins::appsrc::register_appsrc;
use g2g_plugins::registry::default_registry;

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
        let in_channel = format!("odd_nv12_in_{width}x{height}");
        let out_channel = format!("odd_nv12_out_{width}x{height}");
        let feed = register_appsrc(&in_channel);
        assert!(feed.push(&frame, 0));
        feed.end_of_stream();
        let pull = register_appsink_pull(&out_channel);
        let line = format!(
            "appsrc channel={in_channel} caps=video/x-raw,format=NV12,width={width},height={height},framerate=30/1 \
             ! compositor width={width} height={height} format=nv12 ! appsink channel={out_channel}"
        );
        let graph = parse_launch(&default_registry(), &line).expect("parses");
        run_graph(graph, &ZeroClock, 4).await.expect("runs");
        let Pull::Frame(out) = pull.try_pull() else {
            panic!("no frame reached the appsink");
        };
        assert_eq!(
            out.domain.as_system_slice().expect("system memory"),
            frame.as_slice(),
            "{width}x{height}"
        );
    }
}
