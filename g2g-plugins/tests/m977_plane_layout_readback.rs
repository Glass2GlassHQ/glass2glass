//! M977: the `PlaneLayout` meta removing the GPU compositor's readback repack.
//!
//! The compositor's canvas comes back from the GPU at a 256-byte row pitch, and
//! every frame is repacked into tight rows before it is pushed. When a consumer
//! downstream has asked for a `PlaneLayout` (the M976 demand signal), the padded
//! buffer goes out as it is with the pitch declared, and the repack disappears.
//!
//! The proof is byte equality: `VideoConvert`'s output off the padded frame must
//! match its output off the repacked one exactly, so the removed copy changed
//! nothing but the work done.
//!
//! Needs a wgpu adapter; skips (passes) without one.
#![cfg(all(feature = "wgpu-sink", feature = "metadata"))]

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::SystemSlice;
use g2g_core::meta::{MetaRequests, PlaneLayout, RequestPolicy};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, Dim, G2gError, MemoryDomain, MultiInputElement,
    OutputSink, PipelinePacket, PushOutcome, Rate, RawVideoFormat,
};
use g2g_plugins::compositor::CompositorPad;
use g2g_plugins::gpu::GpuContext;
use g2g_plugins::videoconvert::VideoConvert;
use g2g_plugins::wgpucompositor::WgpuCompositor;

/// 100 px of RGBA8 is 400 bytes a row, which the GPU pads to 512: the readback
/// has real padding to either repack or declare.
const WIDTH: u32 = 100;
const HEIGHT: u32 = 4;
const TIGHT_ROW: usize = WIDTH as usize * 4;
const PADDED_ROW: usize = 512;

#[derive(Default)]
struct FrameSink {
    frames: std::vec::Vec<Frame>,
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
                self.frames.push(frame);
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn rgba_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
        interlace: g2g_core::Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// A recognizable RGBA pattern: every pixel differs from its neighbours, so a
/// row read at the wrong pitch cannot pass by luck.
fn pattern() -> std::vec::Vec<u8> {
    let mut px = std::vec::Vec::with_capacity(TIGHT_ROW * HEIGHT as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            px.extend_from_slice(&[(x % 251) as u8, (y * 37 + 11) as u8, (x % 97) as u8, 255]);
        }
    }
    px
}

fn frame_of(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

/// Composite one full-canvas input and return the emitted frame. `demand` is the
/// allocation the runner would hand the fan-in from downstream.
async fn composite(ctx: &GpuContext, demand: Option<AllocationParams>) -> Frame {
    let mut comp = WgpuCompositor::new(WIDTH, HEIGHT, std::vec![CompositorPad::at(0, 0)])
        .with_context(ctx.clone());
    comp.configure_pipeline(0, &rgba_caps(WIDTH, HEIGHT))
        .unwrap();
    if let Some(params) = demand {
        comp.configure_allocation_for_output(&params);
    }
    let mut sink = FrameSink::default();
    comp.process(0, frame_of(&pattern()), &mut sink)
        .await
        .unwrap();
    sink.frames.pop().expect("one composited frame")
}

/// Convert `frame` to I420 through the real element, returning its output bytes.
async fn convert_to_i420(frame: Frame) -> std::vec::Vec<u8> {
    let mut convert = VideoConvert::new(RawVideoFormat::I420);
    convert
        .configure_pipeline(&rgba_caps(WIDTH, HEIGHT))
        .unwrap();
    let mut sink = FrameSink::default();
    convert
        .process(PipelinePacket::DataFrame(frame), &mut sink)
        .await
        .unwrap();
    let out = sink.frames.pop().expect("converted frame");
    out.domain
        .as_system_slice()
        .expect("system memory out")
        .to_vec()
}

#[tokio::test]
async fn padded_readback_under_demand_converts_identically() {
    let Some(ctx) = GpuContext::headless().await.ok() else {
        std::eprintln!("no wgpu adapter; skipping the M977 readback test");
        return;
    };

    // No demand: the readback repacks, exactly as before this milestone.
    let repacked = composite(&ctx, None).await;
    assert!(
        repacked.meta.get::<PlaneLayout>().is_none(),
        "nothing asked for a layout, so none is declared"
    );
    assert_eq!(
        repacked.domain.as_system_slice().unwrap().len(),
        TIGHT_ROW * HEIGHT as usize,
        "tight rows"
    );

    // Demand: the GPU's own padded rows go downstream with their pitch declared.
    let demand = AllocationParams::meta_demand(
        MetaRequests::new().request_from_every_consumer::<PlaneLayout>(),
    );
    let padded = composite(&ctx, Some(demand)).await;
    let layout = padded
        .meta
        .get::<PlaneLayout>()
        .copied()
        .expect("the requested layout is attached");
    assert_eq!(
        layout.plane(0).map(|p| p.stride),
        Some(PADDED_ROW),
        "the GPU's row pitch, declared"
    );
    assert_eq!(
        padded.domain.as_system_slice().unwrap().len(),
        PADDED_ROW * HEIGHT as usize,
        "the padded buffer went out as it is"
    );

    // The zero-copy proof: the consumer's output is the same either way.
    let from_repacked = convert_to_i420(repacked).await;
    let from_padded = convert_to_i420(padded).await;
    assert_eq!(
        from_padded, from_repacked,
        "reading the padded rows where they lie converts to the same pixels"
    );
    assert!(!from_repacked.is_empty());
}

#[tokio::test]
async fn a_consumer_that_asks_for_nothing_still_gets_tight_rows() {
    // The demand signal is what flips the producer: an allocation proposal with
    // no request must leave the readback repacking.
    let Some(ctx) = GpuContext::headless().await.ok() else {
        std::eprintln!("no wgpu adapter; skipping the M977 no-demand test");
        return;
    };
    let frame = composite(&ctx, Some(AllocationParams::system(4096, 2))).await;
    assert!(frame.meta.get::<PlaneLayout>().is_none());
    assert_eq!(
        frame.domain.as_system_slice().unwrap().len(),
        TIGHT_ROW * HEIGHT as usize
    );
}

#[tokio::test]
async fn videoconvert_asks_for_the_layout_from_every_consumer() {
    // The demand the runner carries upstream comes from the consumer itself, and
    // under the policy that lets any consumer sharing the producer veto it: a
    // padded buffer read as tightly packed is corruption, not a missed win.
    let convert = VideoConvert::new(RawVideoFormat::I420);
    assert_eq!(
        AsyncElement::meta_requests(&convert).policy::<PlaneLayout>(),
        Some(RequestPolicy::EveryConsumer)
    );
}
