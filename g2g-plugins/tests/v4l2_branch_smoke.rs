//! Isolates the PiP webcam branch from the compositor and the Wayland sink, to
//! pin down "frozen whitish inset" reports: does V4l2Src -> VideoConvert(RGBA)
//! -> VideoScale actually deliver many *distinct* frames, or stall after one?
//!
//! ```text
//! V4l2Src(YUYV 640x480) -> VideoConvert(RGBA) -> VideoScale(320x240) -> Recorder
//! ```
//!
//! The recorder logs per-frame mean luma; the test asserts frames flow and the
//! content changes over time (motion). Ignored by default; needs `/dev/videoN`
//! (override with `G2G_V4L2_DEVICE`). No Wayland required.
//!
//! ```sh
//! cargo test -p g2g-plugins --features v4l2 \
//!     --test v4l2_branch_smoke -- --ignored --nocapture
//! ```

#![cfg(all(target_os = "linux", feature = "v4l2"))]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::memory::MemoryDomain;
use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, G2gError, Graph, OutputSink,
    PipelineClock, PipelinePacket, RawVideoFormat,
};
use g2g_plugins::compositor::{Compositor, CompositorPad};
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::videoscale::VideoScale;
use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};
use g2g_plugins::v4l2src::V4l2Src;

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

/// Records each frame's mean byte value (a cheap brightness proxy). Shares the
/// log out via an `Arc<Mutex<..>>` so the test can inspect it after the run.
#[derive(Clone)]
struct Recorder {
    means: Arc<Mutex<Vec<(u32, u64)>>>,
}

impl AsyncElement for Recorder {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                // Optional backpressure: throttle the sink to mimic an
                // output-paced consumer (the compositor + Wayland case).
                if let Ok(ms) = std::env::var("G2G_SINK_DELAY_MS") {
                    if let Ok(ms) = ms.parse::<u64>() {
                        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    }
                }
                if let MemoryDomain::System(slice) = &frame.domain {
                    let s = slice.as_slice();
                    // FNV-1a over the whole frame: distinguishes a static-but-live
                    // scene (noise -> differing hashes) from a repeated buffer
                    // (identical hashes), which the mean alone cannot.
                    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
                    for &b in s {
                        hash ^= b as u64;
                        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                    }
                    let sum: u64 = s.iter().map(|&b| b as u64).sum();
                    let mean = (sum / (s.len() as u64).max(1)) as u32;
                    self.means.lock().unwrap().push((mean, hash));
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
#[ignore = "needs a /dev/videoN device (set G2G_V4L2_DEVICE)"]
async fn webcam_branch_delivers_distinct_frames() {
    let device = std::env::var("G2G_V4L2_DEVICE").unwrap_or_else(|_| "/dev/video0".to_string());
    let frames: u64 = std::env::var("G2G_PIP_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    eprintln!("webcam branch: {device}, {frames} frames");

    let means = Arc::new(Mutex::new(Vec::new()));
    let recorder = Recorder { means: means.clone() };

    // G2G_STAGES: "raw" (v4l2 only), "convert" (+RGBA), or "full" (+scale).
    let stages = std::env::var("G2G_STAGES").unwrap_or_else(|_| "full".to_string());
    eprintln!("stages: {stages}");

    let mut g: Graph<GraphNode> = Graph::new();
    let cam = g.add_source(GraphNode::source(
        V4l2Src::new(device).with_size(640, 480).with_fps(30).with_frame_limit(frames),
    ));
    let sink = g.add_sink(GraphNode::element(recorder));
    match stages.as_str() {
        "raw" => {
            g.link(cam, sink).unwrap();
        }
        "convert" => {
            let rgba = g.add_transform(GraphNode::element(VideoConvert::new(RawVideoFormat::Rgba8)));
            g.link(cam, rgba).unwrap();
            g.link(rgba, sink).unwrap();
        }
        _ => {
            let rgba = g.add_transform(GraphNode::element(VideoConvert::new(RawVideoFormat::Rgba8)));
            let small = g.add_transform(GraphNode::element(VideoScale::new(320, 240)));
            g.link(cam, rgba).unwrap();
            g.link(rgba, small).unwrap();
            g.link(small, sink).unwrap();
        }
    }

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        run_graph(g, &ZeroClock, 4),
    )
    .await
    .expect("webcam branch should finish within 30s")
    .expect("webcam branch DAG should run");

    let means = means.lock().unwrap();
    let only_means: Vec<u32> = means.iter().map(|&(m, _)| m).take(20).collect();
    let distinct_hashes = means.iter().map(|&(_, h)| h).collect::<std::collections::HashSet<_>>().len();
    eprintln!("delivered {} frames; means: {only_means:?}", means.len());
    eprintln!("distinct frame contents: {distinct_hashes} of {}", means.len());
    eprintln!("stats: emitted={} consumed={}", stats.frames_emitted, stats.frames_consumed);

    assert!(means.len() as u64 >= frames / 2, "branch stalled: only {} of {frames} frames", means.len());
    assert!(distinct_hashes > 1, "every frame byte-identical across {} frames (repeated buffer)", means.len());
}

/// Hashes a rectangular sub-region (the PiP inset) of each RGBA output frame, so
/// the test can tell whether the overlay actually animates or is latched on one
/// webcam frame.
#[derive(Clone)]
struct InsetRecorder {
    canvas_w: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    hashes: Arc<Mutex<Vec<u64>>>,
}

impl AsyncElement for InsetRecorder {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let Ok(ms) = std::env::var("G2G_SINK_DELAY_MS") {
                    if let Ok(ms) = ms.parse::<u64>() {
                        tokio::time::sleep(std::time::Duration::from_millis(ms)).await;
                    }
                }
                if let MemoryDomain::System(slice) = &frame.domain {
                    let s = slice.as_slice();
                    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
                    for row in 0..self.h {
                        let start = ((self.y + row) * self.canvas_w + self.x) * 4;
                        let end = start + self.w * 4;
                        if end <= s.len() {
                            for &b in &s[start..end] {
                                hash ^= b as u64;
                                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                            }
                        }
                    }
                    self.hashes.lock().unwrap().push(hash);
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
#[ignore = "needs a /dev/videoN device (set G2G_V4L2_DEVICE)"]
async fn compositor_inset_animates_with_live_webcam() {
    // Faithful headless mirror of the live PiP topology: a free-running test
    // background (input 0) + the real webcam branch (input 1) into the
    // compositor, with an output-paced sink. Inspects the inset region for
    // motion, so a frozen overlay fails objectively (no Wayland needed).
    let device = std::env::var("G2G_V4L2_DEVICE").unwrap_or_else(|_| "/dev/video0".to_string());
    let frames: u64 = std::env::var("G2G_PIP_FRAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    const CW: u32 = 640;
    const CH: u32 = 360;
    const IW: u32 = 160;
    const IH: u32 = 120;
    let ix = (CW - IW - 10) as i32;
    let iy = (CH - IH - 10) as i32;

    let hashes = Arc::new(Mutex::new(Vec::new()));
    let recorder = InsetRecorder {
        canvas_w: CW as usize,
        x: ix as usize,
        y: iy as usize,
        w: IW as usize,
        h: IH as usize,
        hashes: hashes.clone(),
    };

    let mut g: Graph<GraphNode> = Graph::new();
    let bg = g.add_source(GraphNode::source(
        VideoTestSrc::new(CW, CH, 30, frames).with_pattern(Pattern::MovingBar),
    ));
    let cam = g.add_source(GraphNode::source(
        V4l2Src::new(device).with_size(640, 480).with_fps(30).with_frame_limit(frames),
    ));
    let rgba = g.add_transform(GraphNode::element(VideoConvert::new(RawVideoFormat::Rgba8)));
    let small = g.add_transform(GraphNode::element(VideoScale::new(IW, IH)));
    let comp = g.add_muxer(
        GraphNode::muxer(Compositor::new(
            CW,
            CH,
            vec![CompositorPad::at(0, 0), CompositorPad::at(ix, iy).with_zorder(1)],
        )),
        2,
    );
    let sink = g.add_sink(GraphNode::element(recorder));
    g.link(bg, comp.input(0)).unwrap();
    g.link(cam, rgba).unwrap();
    g.link(rgba, small).unwrap();
    g.link(small, comp.input(1)).unwrap();
    g.link(comp.output(), sink).unwrap();

    let stats = tokio::time::timeout(
        std::time::Duration::from_secs(40),
        run_graph(g, &ZeroClock, 4),
    )
    .await
    .expect("compositor PiP should finish within 40s")
    .expect("compositor PiP DAG should run");

    let hashes = hashes.lock().unwrap();
    let distinct = hashes.iter().collect::<std::collections::HashSet<_>>().len();
    eprintln!(
        "composited {} frames; distinct inset contents: {distinct}; emitted={} consumed={}",
        hashes.len(),
        stats.frames_emitted,
        stats.frames_consumed,
    );
    assert!(distinct > 1, "inset frozen: {distinct} distinct over {} frames", hashes.len());
}
