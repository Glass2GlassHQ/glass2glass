//! Live picture-in-picture demo for the compositor (M93). Builds a DAG that
//! overlays the webcam, scaled down, onto a synthetic full-screen background and
//! displays the result in a Wayland window:
//!
//! ```text
//! VideoTestSrc(RGBA 1280x720) ───────────────────────────────────┐
//!                                                                 ▼
//!                                              Compositor(1280x720 RGBA) ─► VideoConvert(NV12) ─► WaylandSink
//!                                                                 ▲
//! V4l2Src(YUYV 640x480) ─► VideoConvert(RGBA) ─► VideoScale(320x240) ┘  (PiP inset, z=1)
//! ```
//!
//! The background (input 0) drives the output cadence; the webcam inset overlays
//! at the bottom-right. Ignored by default. Needs a Wayland session and a
//! `/dev/videoN` device (override with `G2G_V4L2_DEVICE`):
//!
//! ```sh
//! cargo test -p g2g-plugins --features "v4l2 wayland-sink" \
//!     --test pip_smoke -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "linux", feature = "v4l2", feature = "wayland-sink"))]

use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{Graph, PipelineClock, RawVideoFormat};
use g2g_plugins::compositor::{Compositor, CompositorPad};
use g2g_plugins::v4l2src::V4l2Src;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};
use g2g_plugins::waylandsink::WaylandSink;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

const CANVAS_W: u32 = 1280;
const CANVAS_H: u32 = 720;
const PIP_W: u32 = 320;
const PIP_H: u32 = 240;

#[tokio::test]
#[ignore = "needs a Wayland session + a /dev/videoN device (set G2G_V4L2_DEVICE)"]
async fn webcam_picture_in_picture_over_a_test_pattern() {
    if std::env::var_os("WAYLAND_DISPLAY").is_none() {
        eprintln!("skipping: no WAYLAND_DISPLAY (run under a Wayland session)");
        return;
    }
    let device = std::env::var("G2G_V4L2_DEVICE").unwrap_or_else(|_| "/dev/video0".to_string());
    let frames: u64 = std::env::var("G2G_PIP_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    eprintln!("PiP: background test pattern + webcam {device} inset, {frames} frames");

    let mut g: Graph<GraphNode> = Graph::new();

    // Background (input 0, the timing driver): a full-canvas RGBA test pattern.
    // A sweeping bar makes the background's own motion obvious, so a frozen
    // frame is easy to spot independently of the webcam inset.
    let bg = g.add_source(GraphNode::source(
        VideoTestSrc::new(CANVAS_W, CANVAS_H, 30, frames).with_pattern(Pattern::MovingBar),
    ));

    // Webcam capture, teed two ways: one branch feeds the compositor inset, the
    // other goes straight to its own Wayland window so the raw camera motion is
    // visible side by side (a reference for the inset).
    let cam = g.add_source(GraphNode::source(
        V4l2Src::new(device)
            .with_size(640, 480)
            .with_fps(30)
            .with_frame_limit(frames),
    ));
    // The inset branch only converts to RGBA; the compositor pad scales it to
    // the inset size (no upstream VideoScale needed, M97 per-pad scaling).
    let cam_tee = g.add_tee(2);
    let cam_rgba = g.add_transform(GraphNode::element(VideoConvert::new(RawVideoFormat::Rgba8)));

    // Reference branch: raw webcam -> NV12 -> its own Wayland sink.
    let cam_nv12 = g.add_transform(GraphNode::element(VideoConvert::new(RawVideoFormat::Nv12)));
    let cam_sink = g.add_sink(GraphNode::element(
        WaylandSink::new().with_title("glass2glass webcam (raw)"),
    ));

    // Compositor: background full-frame at (0,0), webcam inset bottom-right (z=1).
    let inset_x = (CANVAS_W - PIP_W - 20) as i32;
    let inset_y = (CANVAS_H - PIP_H - 20) as i32;
    let comp = g.add_muxer(
        GraphNode::muxer(Compositor::new(
            CANVAS_W,
            CANVAS_H,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(inset_x, inset_y)
                    .with_zorder(1)
                    .with_size(PIP_W, PIP_H),
            ]),
        )),
        2,
    );

    // Display wants NV12.
    let to_nv12 = g.add_transform(GraphNode::element(VideoConvert::new(RawVideoFormat::Nv12)));
    let sink = g.add_sink(GraphNode::element(
        WaylandSink::new().with_title("glass2glass picture-in-picture"),
    ));

    g.link(bg, comp.input(0)).unwrap();
    g.link(cam, cam_tee.input()).unwrap();
    // Tee branch 0 -> RGBA -> compositor inset (scaled by the pad).
    g.link(cam_tee.out(0), cam_rgba).unwrap();
    g.link(cam_rgba, comp.input(1)).unwrap();
    // Tee branch 1 -> standalone webcam window.
    g.link(cam_tee.out(1), cam_nv12).unwrap();
    g.link(cam_nv12, cam_sink).unwrap();
    g.link(comp.output(), to_nv12).unwrap();
    g.link(to_nv12, sink).unwrap();

    let timeout_s: u64 = std::env::var("G2G_PIP_TIMEOUT_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_s),
        run_graph(g, &ZeroClock, 4),
    )
    .await
    .expect("PiP pipeline should finish within the timeout")
    .expect("PiP DAG should run");

    eprintln!(
        "stats: emitted={} consumed={}",
        stats.frames_emitted, stats.frames_consumed
    );
    assert!(stats.frames_emitted > 0, "sources produced no frames");
}
