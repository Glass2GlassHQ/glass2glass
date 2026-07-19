//! Fan-in integration for the compositor (M93): two solid-colour sources are
//! mixed through `run_muxer_sink` into a capturing sink. Asserts the output
//! cadence (one frame per input-0 frame), the output geometry, and that the
//! base layer covers the canvas. Pixel-blend correctness is unit-tested in the
//! module; this exercises the real fan-in negotiation + runner wiring.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, run_muxer_sink, DynSourceLoop, GraphNode, SourceLoop};
use g2g_core::Graph;
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::compositor::{Compositor, CompositorPad};
use g2g_plugins::videoscale::VideoScale;
use g2g_plugins::videotestsrc::{Pattern, VideoTestSrc};

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn rgba(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Emits `count` solid-`color` RGBA frames of `w` x `h`, then EOS.
struct ColorSrc {
    w: u32,
    h: u32,
    color: [u8; 4],
    count: u64,
    configured: bool,
}

impl SourceLoop for ColorSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(rgba(self.w, self.h)))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let bytes = (self.w * self.h) as usize * 4;
            for seq in 0..self.count {
                let mut buf = alloc_vec(bytes);
                for px in buf.chunks_exact_mut(4) {
                    px.copy_from_slice(&self.color);
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                    timing: Default::default(),
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.count)
        })
    }
}

fn alloc_vec(n: usize) -> Vec<u8> {
    vec![0u8; n]
}

/// Records every received DataFrame's byte length and the top-left pixel, plus
/// the most recent full frame for pixel sampling.
#[derive(Default)]
struct CapturingSink {
    lens: Vec<usize>,
    top_left: Vec<[u8; 4]>,
    last: Option<Box<[u8]>>,
}

impl AsyncElement for CapturingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
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
            if let PipelinePacket::DataFrame(frame) = packet {
                if let MemoryDomain::System(slice) = &frame.domain {
                    let s = slice.as_slice();
                    self.lens.push(s.len());
                    self.top_left.push([s[0], s[1], s[2], s[3]]);
                    self.last = Some(s.into());
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn compositor_emits_one_frame_per_base_frame() {
    // input 0: 4x4 red background (the timing driver, 5 frames).
    // input 1: 2x2 blue overlay at (1,1) (3 frames).
    let mut base = ColorSrc {
        w: 4,
        h: 4,
        color: [255, 0, 0, 255],
        count: 5,
        configured: false,
    };
    let mut overlay = ColorSrc {
        w: 2,
        h: 2,
        color: [0, 0, 255, 255],
        count: 3,
        configured: false,
    };
    let mut comp = Compositor::new(
        4,
        4,
        Vec::from([
            CompositorPad::at(0, 0),
            CompositorPad::at(1, 1).with_zorder(1),
        ]),
    );
    let mut sink = CapturingSink::default();
    let clock = ZeroClock;

    let sources: Vec<&mut dyn DynSourceLoop> = vec![&mut base, &mut overlay];
    run_muxer_sink(sources, &mut comp, &mut sink, &clock, 4usize)
        .await
        .expect("compositor fan-in runs");

    // One composited output per input-0 (base) frame.
    assert_eq!(sink.lens.len(), 5, "one output frame per base frame");
    assert_eq!(comp.emitted(), 5);
    // Output is the 4x4 RGBA canvas.
    assert!(
        sink.lens.iter().all(|&l| l == 4 * 4 * 4),
        "every output is the canvas size"
    );
    // The base layer covers the whole canvas, so (0,0) is always red regardless
    // of how the two inputs interleaved (the blue overlay only covers (1,1)).
    assert!(
        sink.top_left.iter().all(|&p| p == [255, 0, 0, 255]),
        "base red covers the top-left in every frame"
    );
}

use std::sync::{Arc, Mutex};

/// Sink that writes each frame into a shared cell so the test can sample pixels
/// after `run_graph` (which owns the sink).
struct ShareSink(Arc<Mutex<Vec<Box<[u8]>>>>);
impl AsyncElement for ShareSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                if let MemoryDomain::System(slice) = &frame.domain {
                    self.0.lock().unwrap().push(slice.as_slice().into());
                }
            }
            Ok(())
        })
    }
}

// PiP-scale, deterministic (solid-colour "camera", no real device): a green
// overlay scaled down and placed at an offset must be visible in the inset
// region and absent outside it. Mirrors the live PiP graph
// (source -> VideoScale -> compositor.input(1)) that showed no inset.
#[tokio::test]
async fn pip_scale_overlay_is_visible_in_the_inset() {
    const CW: usize = 320;
    const CH: usize = 240;
    const IW: u32 = 80;
    const IH: u32 = 60;
    const IX: i32 = 220;
    const IY: i32 = 170;

    let frames = Arc::new(Mutex::new(Vec::<Box<[u8]>>::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let bg = g.add_source(GraphNode::source(ColorSrc {
        w: CW as u32,
        h: CH as u32,
        color: [255, 0, 0, 255],
        count: 8,
        configured: false,
    }));
    let cam = g.add_source(GraphNode::source(ColorSrc {
        w: 160,
        h: 120,
        color: [0, 255, 0, 255],
        count: 8,
        configured: false,
    }));
    let scale = g.add_transform(GraphNode::element(VideoScale::new(IW, IH)));
    let comp = g.add_muxer(
        GraphNode::muxer(Compositor::new(
            CW as u32,
            CH as u32,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(IX, IY).with_zorder(1),
            ]),
        )),
        2,
    );
    let snk = g.add_sink(GraphNode::element(ShareSink(frames.clone())));
    g.link(bg, comp.input(0)).unwrap();
    g.link(cam, scale).unwrap();
    g.link(scale, comp.input(1)).unwrap();
    g.link(comp.output(), snk).unwrap();

    run_graph(g, &ZeroClock, 4)
        .await
        .expect("PiP-scale DAG runs");

    let frames = frames.lock().unwrap();
    let pixel = |buf: &[u8], x: usize, y: usize| {
        let i = (y * CW + x) * 4;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    };
    // Some output frame must carry the inset (the last one, after the overlay
    // has been cached). Sample the inset centre and an outside point.
    let last = frames.last().expect("at least one composited frame");
    let inset = pixel(last, (IX as usize) + 40, (IY as usize) + 30);
    let outside = pixel(last, 10, 10);
    eprintln!(
        "inset={inset:?} outside={outside:?} frames={}",
        frames.len()
    );
    assert_eq!(
        outside,
        [255, 0, 0, 255],
        "background red outside the inset"
    );
    assert_eq!(inset, [0, 255, 0, 255], "green camera visible in the inset");
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

/// A solid NV12 frame: Y plane of `y`, then an interleaved half-res UV plane.
fn solid_nv12(w: usize, h: usize, y: u8, u: u8, v: u8) -> Vec<u8> {
    let mut buf = vec![y; w * h];
    for _ in 0..(w / 2) * (h / 2) {
        buf.push(u);
        buf.push(v);
    }
    buf
}

/// Emits `count` solid NV12 frames of `w` x `h`, then EOS.
struct Nv12ColorSrc {
    w: u32,
    h: u32,
    yuv: [u8; 3],
    count: u64,
    configured: bool,
}

impl SourceLoop for Nv12ColorSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(nv12_caps(self.w, self.h)))
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            for seq in 0..self.count {
                let buf = solid_nv12(
                    self.w as usize,
                    self.h as usize,
                    self.yuv[0],
                    self.yuv[1],
                    self.yuv[2],
                );
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                    timing: Default::default(),
                    sequence: seq,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(self.count)
        })
    }
}

// End-to-end NV12 fan-in through the real runner: a base and an overlay in NV12
// mix planar (no RGBA round-trip) and the composited Y/chroma land correctly.
#[tokio::test]
async fn nv12_fan_in_mixes_planar_end_to_end() {
    const W: usize = 8;
    const H: usize = 8;

    let frames = Arc::new(Mutex::new(Vec::<Box<[u8]>>::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let base = g.add_source(GraphNode::source(Nv12ColorSrc {
        w: W as u32,
        h: H as u32,
        yuv: [50, 60, 70],
        count: 4,
        configured: false,
    }));
    let overlay = g.add_source(GraphNode::source(Nv12ColorSrc {
        w: 4,
        h: 4,
        yuv: [200, 100, 150],
        count: 4,
        configured: false,
    }));
    let comp = g.add_muxer(
        GraphNode::muxer(
            Compositor::new(
                W as u32,
                H as u32,
                Vec::from([
                    CompositorPad::at(0, 0),
                    CompositorPad::at(2, 2).with_zorder(1),
                ]),
            )
            .with_format(RawVideoFormat::Nv12),
        ),
        2,
    );
    let snk = g.add_sink(GraphNode::element(ShareSink(frames.clone())));
    g.link(base, comp.input(0)).unwrap();
    g.link(overlay, comp.input(1)).unwrap();
    g.link(comp.output(), snk).unwrap();

    run_graph(g, &ZeroClock, 4)
        .await
        .expect("NV12 compositor DAG runs");

    let frames = frames.lock().unwrap();
    let last = frames.last().expect("a composited frame");
    assert_eq!(last.len(), W * H * 3 / 2, "output is a full NV12 frame");
    // luma: overlay inside (2,2), base outside.
    let y_at = |b: &[u8], x: usize, yy: usize| b[yy * W + x];
    assert_eq!(y_at(last, 2, 2), 200, "overlay luma inside");
    assert_eq!(y_at(last, 0, 0), 50, "base luma outside");
    // chroma (interleaved UV at half res) under luma (2,2): cx=cy=1.
    let uv_base = W * H;
    let u_at = |b: &[u8], cx: usize, cy: usize| b[uv_base + (cy * (W / 2) + cx) * 2];
    let v_at = |b: &[u8], cx: usize, cy: usize| b[uv_base + (cy * (W / 2) + cx) * 2 + 1];
    assert_eq!(u_at(last, 1, 1), 100, "overlay U inside");
    assert_eq!(v_at(last, 1, 1), 150, "overlay V inside");
    assert_eq!(u_at(last, 0, 0), 60, "base U outside");
}

/// Records the inset-centre pixel of each output frame (and counts frames),
/// throttling slightly so the background does not race arbitrarily ahead of the
/// overlay (mirrors an output-paced consumer).
struct InsetProbe {
    cw: usize,
    x: usize,
    y: usize,
    pixels: Arc<Mutex<Vec<[u8; 4]>>>,
}
impl AsyncElement for InsetProbe {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                if let MemoryDomain::System(slice) = &frame.domain {
                    let s = slice.as_slice();
                    let i = (self.y * self.cw + self.x) * 4;
                    self.pixels
                        .lock()
                        .unwrap()
                        .push([s[i], s[i + 1], s[i + 2], s[i + 3]]);
                }
            }
            Ok(())
        })
    }
}

// A *changing* overlay (per-frame snow) must animate in the inset across the
// run, and the background must not collapse: every input-0 frame produces an
// output. This is the regression guard for the priming freeze, where a
// free-running background outran a slower overlay and the inset latched on one
// frame (only PENDING_CAP frames ever emitted).
#[tokio::test]
async fn live_overlay_animates_and_background_never_collapses() {
    const CW: u32 = 160;
    const CH: u32 = 120;
    const IW: u32 = 48;
    const IH: u32 = 36;
    const IX: i32 = 100;
    const IY: i32 = 80;
    const N: u64 = 60;

    let pixels = Arc::new(Mutex::new(Vec::<[u8; 4]>::new()));
    let mut g: Graph<GraphNode> = Graph::new();
    let bg = g.add_source(GraphNode::source(
        VideoTestSrc::new(CW, CH, 30, N).with_pattern(Pattern::MovingBar),
    ));
    // Overlay produces distinct per-frame content (snow) at the inset size.
    let overlay = g.add_source(GraphNode::source(
        VideoTestSrc::new(IW, IH, 30, N).with_pattern(Pattern::Snow),
    ));
    let comp = g.add_muxer(
        GraphNode::muxer(Compositor::new(
            CW,
            CH,
            Vec::from([
                CompositorPad::at(0, 0),
                CompositorPad::at(IX, IY).with_zorder(1),
            ]),
        )),
        2,
    );
    let snk = g.add_sink(GraphNode::element(InsetProbe {
        cw: CW as usize,
        x: (IX as usize) + (IW as usize) / 2,
        y: (IY as usize) + (IH as usize) / 2,
        pixels: pixels.clone(),
    }));
    g.link(bg, comp.input(0)).unwrap();
    g.link(overlay, comp.input(1)).unwrap();
    g.link(comp.output(), snk).unwrap();

    let stats = run_graph(g, &ZeroClock, 4)
        .await
        .expect("compositor DAG runs");

    let pixels = pixels.lock().unwrap();
    // No collapse: one output per background frame.
    assert_eq!(
        pixels.len() as u64,
        N,
        "every background frame produced an output"
    );
    assert_eq!(stats.frames_consumed, N);
    // The overlay animated: the inset took many distinct values, not one frozen.
    let distinct = pixels
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct > (N as usize) / 2,
        "inset froze: only {distinct} distinct over {N} frames"
    );
}
