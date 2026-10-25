#![cfg(all(
    target_os = "linux",
    feature = "nvdec",
    feature = "cuda-wgpu",
    feature = "wgpu-sink",
    feature = "vello-overlay"
))]
//! M1900: an NVDEC frame in CUDA memory gets its detection boxes drawn on the
//! GPU and reaches a wgpu sink without entering system memory:
//! `nvdec -> detections -> cudatowgpu -> wgpucolorconvert -> vellooverlay -> sink`.
//!
//! The detection step stands in for a hosted inference element in CUDA mode: it
//! takes CUDA frames, attaches `AnalyticsMeta` and passes them on. The sink is an
//! offscreen `WgpuSink` on the CUDA bridge's device, since no compositor runs
//! here.
//!
//! Hardware test: needs an NVIDIA GPU with NVCUVID plus a wgpu adapter. Skips
//! when the decoder or the interop device cannot start.

use core::future::Future;
use core::pin::Pin;
use std::boxed::Box;
use std::sync::{Arc, Mutex};

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::runtime::{run_graph, GraphNode};
use g2g_core::{
    AllocationParams, AnalyticsMeta, AsyncElement, BBox, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, Dim, G2gError, Graph, Interlace, ObjectDetection, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::cudatowgpu::CudaToWgpu;
use g2g_plugins::cudawgpu::shared_interop_device;
use g2g_plugins::filesrc::FileSrc;
use g2g_plugins::gpu::GpuContext;
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::nvdec::NvDec;
use g2g_plugins::vellooverlay::VelloAnalyticsOverlay;
use g2g_plugins::wgpucolorconvert::WgpuColorConvert;
use g2g_plugins::wgpusink::WgpuSink;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h264_640x480.h264"
);
const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FRAMERATE: u32 = 30 << 16;

/// The detection every frame carries: its outline runs along pixels 160 and 480
/// across, 120 and 360 down.
const DETECTION: BBox = BBox {
    x: 0.25,
    y: 0.25,
    w: 0.5,
    h: 0.5,
};

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: g2g_core::VideoCodec::H264,
        width: Dim::Fixed(WIDTH),
        height: Dim::Fixed(HEIGHT),
        framerate: Rate::Fixed(FRAMERATE),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn skip_without_hardware(err: &G2gError) -> bool {
    let skip = matches!(err, G2gError::Hardware(_));
    if skip {
        eprintln!("skipping: no usable NVDEC / wgpu device here ({err:?})");
    }
    skip
}

/// Takes CUDA frames and attaches `AnalyticsMeta`, as a hosted inference element
/// in CUDA mode does, with `DETECTION` in it when `draw_box` is set.
struct AttachDetections {
    draw_box: bool,
    domains: Arc<Mutex<Vec<MemoryDomainKind>>>,
}

impl AsyncElement for AttachDetections {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::Cuda)
    }
    fn output_memory(&self) -> MemoryDomainKind {
        MemoryDomainKind::Cuda
    }
    fn propose_allocation(&self, _caps: &Caps) -> Option<AllocationParams> {
        Some(AllocationParams::cuda(
            (WIDTH * HEIGHT * 3 / 2) as usize,
            1,
            1,
        ))
    }
    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    self.domains.lock().unwrap().push(frame.domain.kind());
                    let mut analytics = AnalyticsMeta::new();
                    if self.draw_box {
                        analytics.add_detection(ObjectDetection {
                            bbox: DETECTION,
                            label: 0,
                            confidence: 0.9,
                        });
                    }
                    frame.meta.attach(analytics);
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Presents into an offscreen `WgpuSink` and keeps the last frame's pixels and
/// every frame's memory domain. Each frame is presented and released inside
/// `process`, because a bridged frame must not outlive the decoder whose CUDA
/// context its image was imported into.
struct PresentOffscreen {
    sink: WgpuSink,
    domains: Arc<Mutex<Vec<MemoryDomainKind>>>,
    pixels: Arc<Mutex<Vec<u8>>>,
}

impl AsyncElement for PresentOffscreen {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }))
    }
    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::WgpuTexture)
    }
    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.sink.configure_pipeline(absolute_caps)
    }
    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(frame) = packet {
                self.domains.lock().unwrap().push(frame.domain.kind());
                self.sink.present_frame(&frame.domain)?;
                *self.pixels.lock().unwrap() = self.sink.read_target()?;
            }
            Ok(())
        })
    }
}

/// The domains frames had entering the detection step and the sink, and the
/// last frame the sink presented.
struct ChainRun {
    detection_domains: Vec<MemoryDomainKind>,
    sink_domains: Vec<MemoryDomainKind>,
    pixels: Vec<u8>,
}

async fn run_chain(ctx: &GpuContext, draw_box: bool) -> Result<ChainRun, G2gError> {
    let detection_domains = Arc::new(Mutex::new(Vec::new()));
    let sink_domains = Arc::new(Mutex::new(Vec::new()));
    let pixels = Arc::new(Mutex::new(Vec::new()));

    let mut graph: Graph<GraphNode> = Graph::new();
    let src = graph.add_source(GraphNode::source(FileSrc::new(FIXTURE, h264_caps())));
    let parse = graph.add_transform(GraphNode::element(H264Parse::new()));
    let decode = graph.add_transform(GraphNode::element(NvDec::new()));
    let detect = graph.add_transform(GraphNode::element(AttachDetections {
        draw_box,
        domains: Arc::clone(&detection_domains),
    }));
    let bridge = graph.add_transform(GraphNode::element(CudaToWgpu::new()));
    let convert = graph.add_transform(GraphNode::element(WgpuColorConvert::new()));
    let overlay = graph.add_transform(GraphNode::element(
        VelloAnalyticsOverlay::new().with_thickness(4.0),
    ));
    let present = graph.add_sink(GraphNode::element(PresentOffscreen {
        sink: WgpuSink::offscreen(ctx.clone(), WIDTH, HEIGHT),
        domains: Arc::clone(&sink_domains),
        pixels: Arc::clone(&pixels),
    }));
    for (from, to) in [
        (src, parse),
        (parse, decode),
        (decode, detect),
        (detect, bridge),
        (bridge, convert),
        (convert, overlay),
        (overlay, present),
    ] {
        graph.link(from, to).unwrap();
    }
    run_graph(graph, &NullClock, 2).await?;

    let detection_domains = detection_domains.lock().unwrap().clone();
    let sink_domains = sink_domains.lock().unwrap().clone();
    let pixels = pixels.lock().unwrap().clone();
    Ok(ChainRun {
        detection_domains,
        sink_domains,
        pixels,
    })
}

#[tokio::test]
async fn nvdec_frames_get_boxes_drawn_on_the_gpu() {
    let interop = match shared_interop_device().await {
        Ok(interop) => interop,
        Err(e) => {
            assert!(skip_without_hardware(&e));
            return;
        }
    };
    let ctx = GpuContext::from_wgpu(
        interop.instance.clone(),
        interop.adapter.clone(),
        interop.device.clone(),
        interop.queue.clone(),
    );
    let plain = match run_chain(&ctx, false).await {
        Ok(run) => run,
        Err(e) if skip_without_hardware(&e) => return,
        Err(e) => panic!("unexpected error: {e:?}"),
    };
    let boxed = run_chain(&ctx, true).await.expect("the boxed run");

    for run in [&plain, &boxed] {
        assert!(!run.sink_domains.is_empty(), "frames reached the sink");
        assert_eq!(run.detection_domains.len(), run.sink_domains.len());
        assert!(
            run.detection_domains
                .iter()
                .all(|k| *k == MemoryDomainKind::Cuda),
            "the detection step saw CUDA frames: {:?}",
            run.detection_domains
        );
        assert!(
            run.sink_domains
                .iter()
                .all(|k| *k == MemoryDomainKind::WgpuTexture),
            "the sink saw wgpu textures: {:?}",
            run.sink_domains
        );
        assert_eq!(run.pixels.len(), (WIDTH * HEIGHT * 4) as usize);
    }

    let px = |pixels: &[u8], x: u32, y: u32| -> [u8; 4] {
        let i = ((y * WIDTH + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
    };
    let lit = plain
        .pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| p[0] > 16 || p[1] > 16 || p[2] > 16)
        .count();
    assert!(
        lit > (WIDTH * HEIGHT / 10) as usize,
        "the presented frame carries picture ({lit} lit pixels)"
    );

    // class 0 strokes in the palette's red, 0xFF3B30
    for (x, y) in [(320, 119), (320, 360), (159, 240), (480, 240)] {
        let p = px(&boxed.pixels, x, y);
        assert!(
            p[0] > 240 && p[1] < 80 && p[2] < 70,
            "stroke pixel ({x}, {y}) is red: {p:?}"
        );
    }
    // the 4px stroke covers 158..162 and 478..482 across, antialiasing one more
    let near_outline = |x: u32, y: u32| {
        let within = |v: u32, lo: u32, hi: u32| (lo..hi).contains(&v);
        within(x, 157, 483) && within(y, 117, 363) && !(within(x, 163, 477) && within(y, 123, 357))
    };
    let mut changed = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if !near_outline(x, y) && px(&plain.pixels, x, y) != px(&boxed.pixels, x, y) {
                changed.push((x, y));
            }
        }
    }
    assert!(
        changed.is_empty(),
        "{} pixels away from the box differ from the frame without one, first {:?}",
        changed.len(),
        changed.first()
    );
}
