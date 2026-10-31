//! M2039: `pipewirevideosink` publishes raw video as a PipeWire camera node.
//! GStreamer's `pipewiresrc` captures from it, and the frames it gets match the
//! ones g2g renders straight to a file. Skips without a PipeWire daemon or
//! `gst-launch-1.0`.
#![cfg(all(target_os = "linux", feature = "std", feature = "pipewire"))]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{MonotonicClock, PipelineClock};
use g2g_plugins::registry::default_registry;

const CAPS: &str = "video/x-raw,format=I420,width=320,height=240,framerate=30/1";
const FRAME_BYTES: usize = 320 * 240 * 3 / 2;
const CAPTURED_FRAMES: usize = 3;
// long enough for the consumer to link and take its frames
const PUBLISHED_FRAMES: u32 = 150;
const NODE_DEADLINE: Duration = Duration::from_secs(5);
// a consumer that links but gets no frames would otherwise wait forever
const CONSUMER_LIMIT: Duration = Duration::from_secs(10);

struct ZeroClock;

impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g-m2039-{}-{name}", std::process::id()))
}

fn succeeds(program: &str, args: &[&str]) -> bool {
    Command::new(program)
        .args(args)
        .output()
        .is_ok_and(|output| output.status.success())
}

fn node_listed(name: &str) -> bool {
    Command::new("pw-cli")
        .args(["ls", "Node"])
        .output()
        .is_ok_and(|output| {
            String::from_utf8_lossy(&output.stdout).contains(&format!("\"{name}\""))
        })
}

// the consumer sets media.type, which WirePlumber's linking policy needs and gst leaves out for ANY caps
fn capture_with_gst(node: &str, out: &Path) -> bool {
    let deadline = Instant::now() + NODE_DEADLINE;
    while !node_listed(node) {
        if Instant::now() > deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let frames = CAPTURED_FRAMES.to_string();
    let location = format!("location={}", out.display());
    let limit = CONSUMER_LIMIT.as_secs().to_string();
    succeeds(
        "timeout",
        &[
            &limit,
            "gst-launch-1.0",
            "-q",
            "pipewiresrc",
            &format!("target-object={node}"),
            "stream-properties=props,media.type=Video",
            &format!("num-buffers={frames}"),
            "!",
            CAPS,
            "!",
            "filesink",
            &location,
        ],
    )
}

async fn run(line: &str, clock: &impl PipelineClock) {
    let graph = parse_launch(&default_registry(), line).unwrap_or_else(|e| panic!("`{line}`: {e}"));
    run_graph(graph, clock, 4)
        .await
        .unwrap_or_else(|e| panic!("`{line}` runs: {e:?}"));
}

#[tokio::test]
async fn gst_captures_the_frames_g2g_publishes() {
    if !succeeds("pw-cli", &["info", "0"]) || !succeeds("gst-launch-1.0", &["--version"]) {
        eprintln!("skipping m2039: no PipeWire daemon or gst-launch-1.0");
        return;
    }
    let reference = temp_path("reference.yuv");
    let captured = temp_path("captured.yuv");
    run(
        &format!(
            "videotestsrc pattern=smpte num-buffers=1 ! {CAPS} ! filesink location={}",
            reference.display()
        ),
        &ZeroClock,
    )
    .await;

    let node = format!("g2g-m2039-{}", std::process::id());
    let consumer = {
        let (node, captured) = (node.clone(), captured.clone());
        std::thread::spawn(move || capture_with_gst(&node, &captured))
    };
    // clocksync paces the source at its framerate, as a camera delivers
    run(
        &format!(
            "videotestsrc pattern=smpte num-buffers={PUBLISHED_FRAMES} ! {CAPS} ! clocksync \
             ! pipewirevideosink client-name={node}"
        ),
        &MonotonicClock,
    )
    .await;
    let consumed = consumer.join().expect("the consumer thread");

    let expected = std::fs::read(&reference).expect("read the reference frame");
    let got = std::fs::read(&captured).unwrap_or_default();
    let _ = std::fs::remove_file(&reference);
    let _ = std::fs::remove_file(&captured);
    assert!(consumed, "gst-launch-1.0 captured from the g2g camera");
    assert_eq!(expected.len(), FRAME_BYTES);
    assert_eq!(
        got.len(),
        CAPTURED_FRAMES * FRAME_BYTES,
        "captured frame count"
    );
    for (i, frame) in got.chunks(FRAME_BYTES).enumerate() {
        assert!(
            frame == expected.as_slice(),
            "captured frame {i} matches the rendered one"
        );
    }
}
