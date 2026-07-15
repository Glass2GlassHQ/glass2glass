//! The visual end of the one-graph detector (M452): a real YOLO's structured
//! detections drawn back onto the picture, the screenshot-able payoff of the
//! `inference -> AnalyticsMeta -> overlay -> display` story.
//!
//! The detector runs once on the fixture's preprocessed tensor (`OrtInference ->
//! DetectionPostprocess`, the exact M452 chain), producing an [`AnalyticsMeta`]
//! of normalized boxes. The frame the model saw is reconstructed straight from
//! that tensor (CHW f32 -> HWC RGBA8) so the boxes land pixel-exact, and a single
//! graph paints them on and presents:
//!
//!   ImageSource(RGBA8 + AnalyticsMeta) -> AnalyticsOverlay -> <display>
//!
//! Two display backends, like `textoverlay_demo`. This is the M214 fan-out
//! pattern (meta carried on the frame, the overlay draws it) but with a real
//! detector and a desktop sink instead of a hard-coded box and a fake sink.
//!
//!   # Still: write the annotated frame as a PPM (no display, headless-safe)
//!   tools/detect-fixture.sh   # fetch the gitignored YOLO model + sample tensor
//!   cargo run -p g2g-ml --features "ort analytics" --example detect_overlay -- /tmp/detect.ppm
//!
//!   # Live: loop the annotated frame into a Wayland window (Fedora Wayland session)
//!   cargo run -p g2g-ml --features detect-overlay-live --example detect_overlay -- 600
//!
//! Skips with a message when the fixtures are absent (the "validated locally, not
//! CI" pattern of the GPU / Android probes).

use core::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use g2g_core::element::{AsyncElement, BoxFuture, DynAsyncElement, OutputSink};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{run_linear_chain, SourceLoop};
use g2g_core::{
    AnalyticsMeta, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, PipelineClock,
    Rate, RawVideoFormat, TensorDType, TensorLayout, TensorShape,
};
use g2g_ml::detect::DetectionPostprocess;
use g2g_ml::ortinfer::OrtInference;
use g2g_plugins::analyticsoverlay::AnalyticsOverlay;

const SIZE: u32 = 640;

fn detect_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/detect")
}

/// COCO-80 class names, indexed by the label the YOLO model emits, so the demo
/// prints "dog 0.90" rather than "class 16 0.90".
const COCO: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
];

fn label_name(label: u32) -> &'static str {
    COCO.get(label as usize).copied().unwrap_or("?")
}

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

fn tensor_caps(dtype: TensorDType, shape: &[u32]) -> Caps {
    Caps::Tensor { dtype, shape: TensorShape::from_slice(shape).unwrap(), layout: TensorLayout::Nchw }
}

fn rgba_caps(w: u32, h: u32, fps: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(fps << 16),
    }
}

// --- detection: the M452 chain, run once to obtain the AnalyticsMeta ----------

/// Emits one f32 NCHW tensor frame then EOS (the model input), the head of the
/// detection chain. Mirrors `runner_driven_detect.rs`.
struct TensorSource {
    caps: Caps,
    data: Option<Vec<u8>>,
}

impl SourceLoop for TensorSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(self.caps.clone()))
    }

    fn caps_constraint(&mut self) -> impl Future<Output = Result<CapsConstraint<'_>, G2gError>> {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps.clone()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let data = self.data.take().ok_or(G2gError::NotConfigured)?;
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
                timing: FrameTiming::default(),
                sequence: 0,
                meta: Default::default(),
            };
            out.push(PipelinePacket::DataFrame(frame)).await?;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

/// Captures the `AnalyticsMeta` off the terminal frame of the detection chain.
#[derive(Default)]
struct MetaSink {
    meta: Option<AnalyticsMeta>,
}

impl AsyncElement for MetaSink {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
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
            if let PipelinePacket::DataFrame(f) = &packet {
                if let Some(a) = f.meta.get::<AnalyticsMeta>() {
                    self.meta = Some(a.clone());
                }
            }
            Ok(())
        })
    }
}

/// Run the real `TensorSource -> OrtInference -> DetectionPostprocess` chain on
/// the preprocessed model input and return the structured detections it produces.
async fn run_detection(model: &[u8], input: Vec<u8>) -> AnalyticsMeta {
    let mut src =
        TensorSource { caps: tensor_caps(TensorDType::F32, &[1, 3, SIZE, SIZE]), data: Some(input) };
    let mut infer = OrtInference::from_memory(model).expect("model loads").with_tensor_input();
    let mut decode = DetectionPostprocess::new(0.25, 0.45).with_input_size(SIZE, SIZE);
    let mut sink = MetaSink::default();

    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut infer, &mut decode];
    run_linear_chain(&mut src, transforms, &mut sink, &NullClock, 4)
        .await
        .expect("detector chain runs");
    sink.meta.unwrap_or_default()
}

/// Reconstruct the RGBA8 frame the model saw from its preprocessed input tensor:
/// `[1, 3, H, W]` f32 in `[0, 1]`, channel-major RGB, back to interleaved HWC
/// RGBA8. The detector's normalized boxes therefore land pixel-exact on it.
fn reconstruct_rgba(input_f32: &[u8], size: u32) -> Vec<u8> {
    let hw = (size as usize) * (size as usize);
    let f32_at = |i: usize| -> f32 {
        let b = &input_f32[i * 4..i * 4 + 4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };
    let to_u8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut rgba = vec![0u8; hw * 4];
    for i in 0..hw {
        rgba[i * 4] = to_u8(f32_at(i)); // R plane
        rgba[i * 4 + 1] = to_u8(f32_at(hw + i)); // G plane
        rgba[i * 4 + 2] = to_u8(f32_at(2 * hw + i)); // B plane
        rgba[i * 4 + 3] = 255;
    }
    rgba
}

// --- the display graph's source: RGBA8 frames carrying the detections ---------

/// Emits `frames` copies of the annotated RGBA8 image, each carrying the
/// detector's `AnalyticsMeta`, PTS-paced at `fps`, then EOS. The overlay draws
/// the boxes; downstream the sink presents (or a still capture writes a PPM).
struct ImageSource {
    rgba: Vec<u8>,
    meta: AnalyticsMeta,
    width: u32,
    height: u32,
    fps: u32,
    frames: u64,
}

impl SourceLoop for ImageSource {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&mut self) -> Self::CapsFuture<'_> {
        core::future::ready(Ok(rgba_caps(self.width, self.height, self.fps)))
    }

    fn caps_constraint(&mut self) -> impl Future<Output = Result<CapsConstraint<'_>, G2gError>> {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(rgba_caps(
            self.width,
            self.height,
            self.fps,
        )))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            let pts_step_ns: u64 = (1_000_000_000u64 << 16) / u64::from(self.fps << 16);
            let mut pushed = 0u64;
            for seq in 0..self.frames {
                let mut frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(
                        self.rgba.clone().into_boxed_slice(),
                    )),
                    timing: FrameTiming {
                        pts_ns: seq * pts_step_ns,
                        dts_ns: seq * pts_step_ns,
                        duration_ns: pts_step_ns,
                        keyframe: true,
                        ..Default::default()
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                // Carry the detections on the frame; the overlay reads and draws
                // them (the M214 meta-on-frame fan-out contract).
                frame.meta.attach(self.meta.clone());
                out.push(PipelinePacket::DataFrame(frame)).await?;
                pushed += 1;
            }
            out.push(PipelinePacket::Eos).await?;
            Ok(pushed)
        })
    }
}

// --- still backend: capture the annotated RGBA8 frame and write a PPM ---------

/// Keeps the last RGBA8 frame so the still path can write it to disk.
#[derive(Default)]
struct CaptureSink {
    width: u32,
    height: u32,
    last: Option<Vec<u8>>,
}

impl AsyncElement for CaptureSink {
    type ProcessFuture<'a>
        = BoxFuture<'a, Result<(), G2gError>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if let Caps::RawVideo { width: Dim::Fixed(w), height: Dim::Fixed(h), .. } = caps {
            self.width = *w;
            self.height = *h;
        }
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
                    self.last = Some(slice.as_slice().to_vec());
                }
            }
            Ok(())
        })
    }
}

fn write_ppm(path: &str, rgba: &[u8], w: u32, h: u32) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = Vec::with_capacity((w * h * 3) as usize + 32);
    write!(out, "P6\n{w} {h}\n255\n")?;
    for px in rgba.chunks_exact(4) {
        out.extend_from_slice(&px[..3]);
    }
    std::fs::write(path, out)
}

/// Still path: `ImageSource -> AnalyticsOverlay -> CaptureSink`, one frame, write PPM.
fn render_still(rgba: Vec<u8>, meta: AnalyticsMeta, path: &str) {
    let mut src = ImageSource { rgba, meta, width: SIZE, height: SIZE, fps: 30, frames: 1 };
    let mut overlay = AnalyticsOverlay::new().with_thickness(3);
    let mut sink = CaptureSink::default();

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("rt");
    let transforms: Vec<&mut dyn DynAsyncElement> = vec![&mut overlay];
    rt.block_on(run_linear_chain(&mut src, transforms, &mut sink, &NullClock, 4))
        .expect("overlay pipeline runs");

    let frame = sink.last.expect("a frame reached the sink");
    write_ppm(path, &frame, sink.width, sink.height).expect("write ppm");
    println!("wrote annotated frame -> {path} ({}x{})", sink.width, sink.height);
}

/// Live path: `ImageSource -> AnalyticsOverlay -> VideoConvert(NV12) -> WaylandSink`.
#[cfg(feature = "detect-overlay-live")]
fn render_live(rgba: Vec<u8>, meta: AnalyticsMeta, frames: u64) {
    use g2g_core::graph::Graph;
    use g2g_core::runtime::{run_graph, GraphNodeRef};
    use g2g_plugins::clock::WallClock;
    use g2g_plugins::videoconvert::VideoConvert;
    use g2g_plugins::waylandsink::WaylandSink;

    let mut g: Graph<GraphNodeRef<'static>> = Graph::new();
    let src = g.add_source(GraphNodeRef::source(ImageSource {
        rgba,
        meta,
        width: SIZE,
        height: SIZE,
        fps: 30,
        frames,
    }));
    let overlay = g.add_transform(GraphNodeRef::element(AnalyticsOverlay::new().with_thickness(3)));
    // WaylandSink consumes NV12, so convert sits between the RGBA8 overlay and it.
    let convert = g.add_transform(GraphNodeRef::element(VideoConvert::new(RawVideoFormat::Nv12)));
    let sink = g.add_sink(GraphNodeRef::element(
        WaylandSink::new().with_title("g2g detect overlay demo"),
    ));
    g.link(src, overlay).expect("link src->overlay");
    g.link(overlay, convert).expect("link overlay->convert");
    g.link(convert, sink).expect("link convert->sink");

    let clock = WallClock::new();
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("rt");
    println!("presenting {frames} frames of {SIZE}x{SIZE} with detections...");
    match rt.block_on(run_graph(g, &clock, 4)) {
        Ok(stats) => println!("done: {} frames presented", stats.frames_consumed),
        Err(e) => eprintln!("pipeline error: {e:?}"),
    }
}

fn main() {
    let dir = detect_dir();
    let model_path = dir.join("model.onnx");
    let input_path = dir.join("input_f32.bin");
    if !model_path.exists() || !input_path.exists() {
        eprintln!(
            "detect fixtures absent ({}); run tools/detect-fixture.sh. skipping.",
            dir.display()
        );
        return;
    }
    let model = std::fs::read(&model_path).expect("read model");
    let input = std::fs::read(&input_path).expect("read input");

    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().expect("rt");
    let meta = rt.block_on(run_detection(&model, input.clone()));

    println!("detector found {} object(s):", meta.detections().count());
    for d in meta.detections() {
        println!("  {:<14} {:.3}", label_name(d.label), d.confidence);
    }
    if meta.detections().count() == 0 {
        eprintln!("no detections; overlay would be a no-op. check the model / fixture.");
    }

    let rgba = reconstruct_rgba(&input, SIZE);
    let arg = std::env::args().nth(1);

    #[cfg(feature = "detect-overlay-live")]
    {
        // A numeric arg means a frame count (live window); a path means still PPM.
        if let Some(frames) = arg.as_deref().and_then(|a| a.parse::<u64>().ok()) {
            render_live(rgba, meta, frames);
            return;
        }
        render_still(rgba, meta, arg.as_deref().unwrap_or("/tmp/detect_overlay.ppm"));
    }
    #[cfg(not(feature = "detect-overlay-live"))]
    {
        render_still(rgba, meta, arg.as_deref().unwrap_or("/tmp/detect_overlay.ppm"));
    }
}
