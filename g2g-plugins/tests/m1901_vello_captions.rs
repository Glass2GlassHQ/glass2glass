#![cfg(feature = "vello-overlay")]

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AnalyticsMeta, AsyncElement, BBox, Caps, Dim, FrameTiming, G2gError, MemoryDomain,
    ObjectDetection, OutputSink, PipelinePacket, PropValue, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::gpu::{read_rgba_texture, texture_of, GpuContext};
use g2g_plugins::textshape::TextShaper;
use g2g_plugins::vellooverlay::VelloAnalyticsOverlay;

const W: u32 = 256;
const H: u32 = 256;
const BACKDROP: [u8; 4] = [20, 20, 20, 255];
const CLASS_0_COLOR: [u8; 4] = [0xFF, 0x3B, 0x30, 0xFF];
const BOX_X: u32 = 64;
const BOX_Y: u32 = 64;

// Parallel per-test device creation intermittently segfaults in the NVIDIA
// driver, so the GPU tests take one lock for their whole body.
static GPU_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn caps() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn frame(class_names: Option<&[&str]>) -> Frame {
    let mut bytes = Vec::with_capacity((W * H * 4) as usize);
    for _ in 0..W * H {
        bytes.extend_from_slice(&BACKDROP);
    }
    let mut frame = Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        FrameTiming::default(),
        0,
    );
    let mut analytics = AnalyticsMeta::new();
    if let Some(names) = class_names {
        analytics.set_class_names(names.iter().copied());
    }
    analytics.add_detection(ObjectDetection {
        bbox: BBox {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        },
        label: 0,
        confidence: 0.9,
    });
    frame.meta.attach(analytics);
    frame
}

#[derive(Default)]
struct FrameSink {
    last: Option<Frame>,
}
impl OutputSink for FrameSink {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        core::task::Poll::Ready({
            if let PipelinePacket::DataFrame(frame) = packet {
                self.last = Some(frame);
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

async fn gpu_with_fonts() -> Option<GpuContext> {
    if TextShaper::new(&[]).default_face().is_none() {
        eprintln!("no system font; skipping");
        return None;
    }
    let ctx = GpuContext::headless().await.ok();
    if ctx.is_none() {
        eprintln!("no wgpu adapter; skipping");
    }
    ctx
}

async fn render(ctx: &GpuContext, mut overlay: VelloAnalyticsOverlay, frame: Frame) -> Vec<u8> {
    overlay.configure_pipeline(&caps()).unwrap();
    let mut sink = FrameSink::default();
    overlay
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    let out = sink.last.expect("frame forwarded");
    let MemoryDomain::WgpuTexture(owned) = &out.domain else {
        panic!("output is a GPU texture domain");
    };
    read_rgba_texture(ctx, texture_of(owned).expect("texture keep-alive"))
}

fn px(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

// stops two rows short of the box's 3px outline
fn above_box(pixels: &[u8]) -> Vec<[u8; 4]> {
    let mut band = Vec::new();
    for y in BOX_Y - 30..BOX_Y - 2 {
        for x in BOX_X..BOX_X + 64 {
            band.push(px(pixels, x, y));
        }
    }
    band
}

#[tokio::test]
async fn show_label_draws_the_class_name_in_a_bar_above_the_box() {
    let _gpu = GPU_LOCK.lock().await;
    let Some(ctx) = gpu_with_fonts().await else {
        return;
    };
    let mut overlay = VelloAnalyticsOverlay::new().with_context(ctx.clone());
    overlay
        .set_property("show-label", PropValue::Bool(true))
        .unwrap();
    let pixels = render(&ctx, overlay, frame(Some(&["person"]))).await;

    let band = above_box(&pixels);
    let bar = band.iter().filter(|p| **p == CLASS_0_COLOR).count();
    // white ink, since black does not read on the red bar
    let ink = band
        .iter()
        .filter(|p| p[..3].iter().all(|c| *c > 200))
        .count();
    assert!(
        bar > 200,
        "a red bar sits on the box's top edge: {bar} pixels"
    );
    assert!(ink > 20, "the class name is drawn in white: {ink} pixels");
    assert_eq!(
        px(&pixels, BOX_X - 4, BOX_Y - 10),
        BACKDROP,
        "bar starts at the box's left edge"
    );
}

#[tokio::test]
async fn show_label_without_a_name_table_draws_no_caption() {
    let _gpu = GPU_LOCK.lock().await;
    let Some(ctx) = gpu_with_fonts().await else {
        return;
    };
    let overlay = VelloAnalyticsOverlay::new()
        .with_context(ctx.clone())
        .with_label(true);
    let pixels = render(&ctx, overlay, frame(None)).await;

    assert!(above_box(&pixels).iter().all(|p| *p == BACKDROP));
}
