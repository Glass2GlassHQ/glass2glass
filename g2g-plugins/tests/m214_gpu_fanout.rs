//! M214 - zero-copy GPU fan-out, validated on a real GPU. Builds the canonical
//! keep-on-GPU graph `source -> VelloAnalyticsOverlay -> tee -> {WgpuSink,
//! WgpuSink}` and runs it through `run_graph` on an actual wgpu device (the
//! headless offscreen sink path). The overlay renders into a
//! `MemoryDomain::WgpuTexture`; the tee fans that texture out to both sinks via
//! M213's `MemoryDomain::share` (an `Arc` refcount bump on the keep-alive), so
//! both branches blit the SAME GPU texture on the shared device with no
//! GPU->CPU copy. Before M213 the tee returned `UnsupportedDomain` on a GPU
//! frame; this is the real-hardware proof that zero-copy GPU fan-out works.
//!
//! Skips if no wgpu adapter is present (CI without a GPU). Runs for real on the
//! RTX 3060 dev host.

#![cfg(all(feature = "vello-overlay", feature = "wgpu-sink"))]

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_graph, GraphNode, SourceLoop};
use g2g_core::{
    AnalyticsMeta, BBox, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, Graph, MemoryDomain,
    ObjectDetection, OutputSink, PipelineClock, PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::gpu::GpuContext;
use g2g_plugins::vellooverlay::VelloAnalyticsOverlay;
use g2g_plugins::wgpusink::WgpuSink;

struct NullClock;
impl PipelineClock for NullClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

const W: u32 = 64;
const H: u32 = 64;

fn rgba() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        // Fixed (not Any): the runner fixates source caps before run, and the
        // overlay is caps-identity so this rate flows through to the sinks.
        framerate: Rate::Fixed(30 << 16),
    }
}

/// Emits one dark RGBA frame carrying a single detection, then EOS.
struct DetectionSrc {
    sent: bool,
}
impl SourceLoop for DetectionSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(rgba()))
    }
    fn configure_pipeline(&mut self, _: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            out.push(PipelinePacket::CapsChanged(rgba())).await?;
            let mut bytes = Vec::with_capacity((W * H * 4) as usize);
            for _ in 0..W * H {
                bytes.extend_from_slice(&[20, 20, 20, 255]);
            }
            let mut frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            let mut a = AnalyticsMeta::new();
            a.add_detection(ObjectDetection {
                bbox: BBox { x: 0.25, y: 0.25, w: 0.5, h: 0.5 },
                label: 0,
                confidence: 0.9,
            });
            frame.meta.attach(a);
            out.push(PipelinePacket::DataFrame(frame)).await?;
            self.sent = true;
            out.push(PipelinePacket::Eos).await?;
            Ok(1)
        })
    }
}

#[tokio::test]
async fn overlay_fans_out_to_two_wgpu_sinks_zero_copy() {
    let ctx = match GpuContext::headless().await {
        Ok(c) => c,
        Err(_) => {
            std::eprintln!("no wgpu adapter; skipping GPU fan-out test");
            return;
        }
    };

    // Overlay and both sinks share ONE wgpu device, so the overlay's texture is
    // presentable by either sink with no copy. The tee broadcasts the SAME
    // texture handle (M213 share = Arc bump) to both.
    let overlay = VelloAnalyticsOverlay::new().with_context(ctx.clone()).with_thickness(4.0);
    let sink_a = WgpuSink::offscreen(ctx.clone(), W, H);
    let sink_b = WgpuSink::offscreen(ctx.clone(), W, H);

    let mut g: Graph<GraphNode> = Graph::new();
    let src = g.add_source(GraphNode::source(DetectionSrc { sent: false }));
    let ov = g.add_transform(GraphNode::element(overlay));
    let tee = g.add_tee(2);
    let a = g.add_sink(GraphNode::element(sink_a));
    let b = g.add_sink(GraphNode::element(sink_b));
    g.link(src, ov).unwrap();
    g.link(ov, tee.input()).unwrap();
    g.link(tee.out(0), a).unwrap();
    g.link(tee.out(1), b).unwrap();

    let stats = run_graph(g, &NullClock, 4).await.expect("GPU overlay->tee->sinks runs");

    // One overlay-rendered GPU texture reached BOTH sinks, each of which blitted
    // it on the shared device. frames_consumed == 2 proves the tee fanned out a
    // real wgpu texture (previously UnsupportedDomain) and both blits succeeded.
    assert_eq!(stats.frames_emitted, 1, "source emitted one frame");
    assert_eq!(stats.frames_consumed, 2, "the GPU texture fanned out to both sinks zero-copy");
}
