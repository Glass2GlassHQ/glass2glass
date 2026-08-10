//! M1005: `SubPictureOverlay` puts decoded bitmap-subtitle cues on the picture.
//!
//! The subpicture decoders render each cue onto a full-frame transparent RGBA
//! canvas; this is the element that blends those canvases onto the video. The
//! synthetic cases drive the real fan-in graph (`video -> o.video`,
//! `canvases -> o.text`) through `run_graph` and assert pixels: a cue paints only
//! where its canvas is opaque and only from its own PTS, the clearing canvas takes
//! it down, and a later cue replaces an earlier one. The end-to-end case runs the
//! same graph with a real decoder on the hand-authored VobSub pair, so the cue
//! rectangles and palette colours on the video are the fixture's own.
#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::sync::{Arc, Mutex};

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, Graph, MemoryDomain,
    OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::subpictureoverlay::SubPictureOverlay;

mod vobsub_fixture;
use vobsub_fixture::{author_vobsub, cues, H, PALETTE, W};

/// Synthetic-case video geometry, small enough to assert whole frames on.
const VW: u32 = 16;
const VH: u32 = 8;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(25 << 16),
        interlace: g2g_core::Interlace::Any,
    }
}

fn frame(pixels: Vec<u8>, pts_ns: u64) -> Frame {
    Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(pixels.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    )
}

/// Opaque black RGBA8 frames at the given PTS values, then Eos.
struct BlackVideoSrc {
    width: u32,
    height: u32,
    pts: Vec<u64>,
}

impl SourceLoop for BlackVideoSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(rgba(self.width, self.height)))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let pixels = (self.width * self.height) as usize;
            for &pts in &self.pts {
                let buf = [0u8, 0, 0, 255].repeat(pixels);
                out.push(PipelinePacket::DataFrame(frame(buf, pts))).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.pts.len() as u64)
        })
    }
}

/// Ready-made RGBA canvases at the given PTS values, standing in for a decoder.
struct CanvasSrc {
    width: u32,
    height: u32,
    canvases: Vec<(u64, Vec<u8>)>,
}

impl SourceLoop for CanvasSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(rgba(self.width, self.height)))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let n = self.canvases.len() as u64;
            for (pts, pixels) in core::mem::take(&mut self.canvases) {
                out.push(PipelinePacket::DataFrame(frame(pixels, pts)))
                    .await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(n)
        })
    }
}

/// The overlaid frames a run collected, `(pts, pixels)` in output order.
type FrameLog = Arc<Mutex<Vec<(u64, Vec<u8>)>>>;

/// Records every frame reaching the end of the graph.
struct RecSink {
    log: FrameLog,
}

impl AsyncElement for RecSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let Some(buf) = f.domain.as_system_slice() {
                    self.log
                        .lock()
                        .unwrap()
                        .push((f.timing.pts_ns, buf.to_vec()));
                }
            }
            Ok(())
        })
    }
}

/// Run `video -> overlay.video` + `canvases -> overlay.text` and collect the
/// overlaid frames, in the order they left the graph.
async fn overlay_run(video: BlackVideoSrc, canvases: CanvasSrc) -> Vec<(u64, Vec<u8>)> {
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let v = g.add_source(GraphNode::source(video));
    let c = g.add_source(GraphNode::source(canvases));
    let overlay = g.add_muxer(GraphNode::muxer(SubPictureOverlay::new()), 2);
    let sink = g.add_sink(GraphNode::element(RecSink { log: log.clone() }));
    g.link(v, overlay.input(0)).unwrap();
    g.link(c, overlay.input(1)).unwrap();
    g.link(overlay.output(), sink).unwrap();

    run_graph(g, &NullClock, 8)
        .await
        .expect("overlay graph runs");
    let out = log.lock().unwrap().clone();
    out
}

/// A transparent canvas with an opaque block of `color` over
/// `[x0, x1) x [y0, y1)`.
fn canvas_block(w: u32, h: u32, rect: (u32, u32, u32, u32), color: [u8; 4]) -> Vec<u8> {
    let (x0, y0, x1, y1) = rect;
    let mut px = vec![0u8; (w * h * 4) as usize];
    for y in y0..y1 {
        for x in x0..x1 {
            let d = ((y * w + x) * 4) as usize;
            px[d..d + 4].copy_from_slice(&color);
        }
    }
    px
}

fn pixel(buf: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let d = ((y * w + x) * 4) as usize;
    buf[d..d + 4].try_into().expect("four channels")
}

/// The RGBA a `.idx` palette entry paints opaque.
fn opaque(entry: u32) -> [u8; 4] {
    [(entry >> 16) as u8, (entry >> 8) as u8, entry as u8, 255]
}

const RED: [u8; 4] = [255, 0, 0, 255];
const BLACK: [u8; 4] = [0, 0, 0, 255];

#[tokio::test]
async fn a_cue_paints_the_canvas_over_the_video_only_in_its_window() {
    // A cue shown at 1s over [4, 8) x [2, 4), cleared at 3s.
    let shown = canvas_block(VW, VH, (4, 2, 8, 4), RED);
    let cleared = vec![0u8; (VW * VH * 4) as usize];
    let frames = overlay_run(
        BlackVideoSrc {
            width: VW,
            height: VH,
            // Straddling the window: 0 before, 1.5s and 2.5s inside, 4s after.
            pts: vec![0, 1_500_000_000, 2_500_000_000, 4_000_000_000],
        },
        CanvasSrc {
            width: VW,
            height: VH,
            canvases: vec![(1_000_000_000, shown), (3_000_000_000, cleared)],
        },
    )
    .await;

    assert_eq!(frames.len(), 4, "every video frame reaches the sink");
    for (pts, buf) in &frames {
        let in_window = (1_000_000_000..3_000_000_000).contains(pts);
        let expected = if in_window { RED } else { BLACK };
        assert_eq!(pixel(buf, VW, 5, 3), expected, "inside the cue at {pts} ns");
        assert_eq!(
            pixel(buf, VW, 5, 1),
            BLACK,
            "above the cue's block at {pts} ns"
        );
        assert_eq!(
            pixel(buf, VW, 1, 3),
            BLACK,
            "left of the cue's block at {pts} ns"
        );
    }
}

#[tokio::test]
async fn a_second_cue_replaces_the_first() {
    let first = canvas_block(VW, VH, (0, 0, 4, 4), RED);
    let second = canvas_block(VW, VH, (8, 4, 12, 8), [0, 0, 255, 255]);
    let frames = overlay_run(
        BlackVideoSrc {
            width: VW,
            height: VH,
            pts: vec![2_000_000_000, 4_000_000_000],
        },
        CanvasSrc {
            width: VW,
            height: VH,
            canvases: vec![(1_000_000_000, first), (3_000_000_000, second)],
        },
    )
    .await;

    assert_eq!(pixel(&frames[0].1, VW, 1, 1), RED, "the first cue shows");
    assert_eq!(pixel(&frames[0].1, VW, 9, 5), BLACK, "and only the first");
    assert_eq!(
        pixel(&frames[1].1, VW, 9, 5),
        [0, 0, 255, 255],
        "the second cue replaces it"
    );
    assert_eq!(
        pixel(&frames[1].1, VW, 1, 1),
        BLACK,
        "the first cue is gone with it"
    );
}

/// The end-to-end case the element exists for: `vobsubsrc ! vobsubdec` decodes
/// the authored sidecar pair and the overlay paints those cues onto the video,
/// each in its own window and in the `.idx` palette's colours.
#[tokio::test]
async fn a_decoded_vobsub_track_paints_the_video() {
    use g2g_plugins::vobsubdec::VobSubDec;
    use g2g_plugins::vobsubsrc::VobSubSrc;

    let dir = std::env::temp_dir();
    let idx = dir.join(format!("g2g-m1005-{}-overlay.idx", std::process::id()));
    let sub = dir.join(format!("g2g-m1005-{}-overlay.sub", std::process::id()));
    author_vobsub(&idx, &sub);

    let fixture = cues();
    let (first, second) = (&fixture[0], &fixture[1]);
    // 0.5s: before the first cue. 2s: inside it. 4s: after it cleared (it hides
    // at 3.548s). 6s: inside the second cue (5.5s -> 7.548s).
    let probes = [500_000_000u64, 2_000_000_000, 4_000_000_000, 6_000_000_000];

    let log = Arc::new(Mutex::new(Vec::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let video = g.add_source(GraphNode::source(BlackVideoSrc {
        width: W,
        height: H,
        pts: probes.to_vec(),
    }));
    let src = g.add_source(GraphNode::source(VobSubSrc::new(&idx)));
    let dec = g.add_transform(GraphNode::element(VobSubDec::new()));
    let overlay = g.add_muxer(GraphNode::muxer(SubPictureOverlay::new()), 2);
    let sink = g.add_sink(GraphNode::element(RecSink { log: log.clone() }));
    g.link(video, overlay.input(0)).unwrap();
    g.link(src, dec).unwrap();
    g.link(dec, overlay.input(1)).unwrap();
    g.link(overlay.output(), sink).unwrap();
    run_graph(g, &NullClock, 8)
        .await
        .expect("the sidecar overlay graph runs");

    let frames = log.lock().unwrap().clone();
    assert_eq!(frames.len(), probes.len(), "every video frame is overlaid");

    // The fixture's cues are a one-pixel border of sample 3 around sample 1, and
    // each cue's colormap sends those to different palette entries.
    let inside = |cue: &vobsub_fixture::Cue| (cue.x + cue.w / 2, cue.y + cue.h / 2);
    let (fx, fy) = inside(first);
    let (sx, sy) = inside(second);
    let first_fill = opaque(PALETTE[first.colormap[1] as usize]);
    let second_fill = opaque(PALETTE[second.colormap[1] as usize]);

    let at = |i: usize, x: u32, y: u32| pixel(&frames[i].1, W, x, y);
    assert_eq!(
        at(0, fx, fy),
        BLACK,
        "nothing is showing before the first cue"
    );
    assert_eq!(at(1, fx, fy), first_fill, "the first cue is on the video");
    assert_eq!(
        at(1, first.x - 1, fy),
        BLACK,
        "and only inside its display rectangle"
    );
    assert_eq!(
        at(2, fx, fy),
        BLACK,
        "the first cue clears at its hide time"
    );
    assert_eq!(at(3, sx, sy), second_fill, "the second cue is on the video");
    assert_eq!(at(3, fx, fy), BLACK, "where the first one no longer is");
    // The two cues index different palette entries, so a colormap that collapsed
    // them would fail here.
    assert_ne!(first_fill, second_fill);

    for path in [idx, sub] {
        let _ = std::fs::remove_file(path);
    }
}

/// The container half: a Matroska file whose subtitle track is bitmap cues
/// auto-plugs the subpicture decoder and this overlay, where a text track plugs
/// `subparse` and the text overlay. ffmpeg both muxes the fixture and provides
/// the H.264 decoder the video branch needs.
#[cfg(feature = "ffmpeg")]
#[test]
fn mkv_playbin_auto_plugs_a_bitmap_subtitle_track() {
    use vobsub_fixture::have_ffmpeg;

    if !have_ffmpeg() {
        eprintln!("ffmpeg not present: skipping");
        return;
    }
    let dir = std::env::temp_dir();
    let idx = dir.join(format!("g2g-m1005-{}-mkv.idx", std::process::id()));
    let sub = dir.join(format!("g2g-m1005-{}-mkv.sub", std::process::id()));
    let mkv = dir.join(format!("g2g-m1005-{}-mkv.mkv", std::process::id()));
    author_vobsub(&idx, &sub);
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "lavfi", "-i", "color=c=black:s=320x240:r=10:d=9"])
        .arg("-i")
        .arg(&idx)
        .args(["-map", "0:v", "-map", "1:s"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:s", "copy"])
        .arg(&mkv)
        .status()
        .expect("run ffmpeg");
    assert!(status.success(), "ffmpeg muxed the S_VOBSUB fixture");

    let line = format!("playbin uri=file://{}", mkv.display());
    let graph = g2g_core::runtime::parse_launch(&g2g_plugins::registry::default_registry(), &line)
        .unwrap_or_else(|e| panic!("playbin builds `{line}`: {e}"));
    let dot = graph.to_dot(
        "pipeline",
        |n| graph.element(n).map(|e| e.log_category().to_string()),
        &g2g_core::DotAnnotations::default(),
    );
    assert!(
        dot.contains("label=\"VobSubDec\""),
        "the subpicture track decodes to cue canvases: {dot}"
    );
    assert!(
        !dot.contains("label=\"SubParse\""),
        "a bitmap cue has no text for the subtitle parser: {dot}"
    );

    for path in [idx, sub, mkv] {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn a_launch_line_wires_the_overlay_by_named_pads() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;

    let reg = default_registry();
    let graph = parse_launch(
        &reg,
        "videotestsrc ! videoconvert ! o.video \
         vobsubsrc location=movie.idx ! vobsubdec ! o.text \
         subpictureoverlay name=o ! videoconvert ! fakesink",
    )
    .expect("the subpicture-overlay launch line parses");
    let vg = graph.finish().expect("valid graph");
    let muxers: Vec<g2g_core::graph::NodeKind> = vg
        .topo()
        .iter()
        .map(|&n| vg.kind(n))
        .filter(|k| matches!(k, g2g_core::graph::NodeKind::Muxer(_)))
        .collect();
    assert_eq!(
        muxers,
        [g2g_core::graph::NodeKind::Muxer(2)],
        "the video and subpicture branches fan into one overlay"
    );
}
