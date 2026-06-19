//! Fan-in integration for the compositor (M93): two solid-colour sources are
//! mixed through `run_muxer_sink` into a capturing sink. Asserts the output
//! cadence (one frame per input-0 frame), the output geometry, and that the
//! base layer covers the canvas. Pixel-blend correctness is unit-tested in the
//! module; this exercises the real fan-in negotiation + runner wiring.

use core::future::Future;
use core::pin::Pin;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::{run_muxer_sink, DynSourceLoop, SourceLoop};
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, G2gError, MemoryDomain, OutputSink, PipelineClock,
    PipelinePacket, Rate, RawVideoFormat,
};
use g2g_plugins::compositor::{Compositor, CompositorPad};

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
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>> where Self: 'a;
    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>> where Self: 'a;

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

/// Records every received DataFrame's byte length and the top-left pixel.
#[derive(Default)]
struct CapturingSink {
    lens: Vec<usize>,
    top_left: Vec<[u8; 4]>,
}

impl AsyncElement for CapturingSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> where Self: 'a;

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
    let mut base = ColorSrc { w: 4, h: 4, color: [255, 0, 0, 255], count: 5, configured: false };
    let mut overlay = ColorSrc { w: 2, h: 2, color: [0, 0, 255, 255], count: 3, configured: false };
    let mut comp = Compositor::new(
        4,
        4,
        Vec::from([CompositorPad::at(0, 0), CompositorPad::at(1, 1).with_zorder(1)]),
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
    assert!(sink.lens.iter().all(|&l| l == 4 * 4 * 4), "every output is the canvas size");
    // The base layer covers the whole canvas, so (0,0) is always red regardless
    // of how the two inputs interleaved (the blue overlay only covers (1,1)).
    assert!(
        sink.top_left.iter().all(|&p| p == [255, 0, 0, 255]),
        "base red covers the top-left in every frame"
    );
}
