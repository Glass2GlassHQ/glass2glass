//! Synthetic animated RGBA source (browser/wasm). Generates a moving colour
//! gradient as `RawVideo` `Rgba8` `System` frames on a timer, the "capture" side of
//! the browser send demo (`PatternSrc -> WebCodecsEncode -> WebSocketSink`) when no
//! camera is available. A real capture source (getUserMedia +
//! `MediaStreamTrackProcessor` -> `VideoFrame`) is a follow-up; this keeps the encode
//! + egress path self-contained and testable.
//!
//! Paces itself with `setTimeout`, which also yields to the event loop each frame so
//! the downstream encoder's async output callbacks can fire (a tight synchronous
//! loop would starve them).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, MemoryDomain,
    OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[derive(Debug)]
pub struct PatternSrc {
    width: u32,
    height: u32,
    fps: u32,
    /// Number of frames to emit before EOS (0 = run until the socket closes; the
    /// demo uses a finite count so the receiver gets a clean end).
    frames: u64,
    configured: bool,
}

impl PatternSrc {
    /// A `width` x `height` animated gradient at `fps`, emitting `frames` frames
    /// then EOS.
    pub fn new(width: u32, height: u32, fps: u32, frames: u64) -> Self {
        Self {
            width,
            height,
            fps,
            frames,
            configured: false,
        }
    }

    fn caps(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Fixed(self.fps << 16),
        }
    }
}

impl SourceLoop for PatternSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a>
        = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let (w, h) = (self.width as usize, self.height as usize);
            let frame_ns = 1_000_000_000u64 / (self.fps.max(1) as u64);
            let delay_ms = (1000.0 / self.fps.max(1) as f64) as i32;

            out.push(PipelinePacket::CapsChanged(self.caps())).await?;

            for i in 0..self.frames {
                let mut buf = vec![0u8; w * h * 4];
                fill_pattern(&mut buf, w, h, i);
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(buf.into_boxed_slice())),
                    timing: {
                        let pts = i * frame_ns;
                        FrameTiming {
                            pts_ns: pts,
                            dts_ns: pts,
                            capture_ns: pts,
                            ..FrameTiming::default()
                        }
                    },
                    sequence: i,
                    meta: Default::default(),
                };
                out.push(PipelinePacket::DataFrame(frame)).await?;
                // Pace + yield so the encoder's output callbacks can run.
                sleep_ms(delay_ms).await;
            }

            out.push(PipelinePacket::Eos).await?;
            Ok(self.frames)
        })
    }
}

/// Paint a cheap moving gradient into a packed RGBA buffer; `i` (frame index)
/// scrolls it, so the encoded stream shows real motion.
fn fill_pattern(buf: &mut [u8], w: usize, h: usize, i: u64) {
    let t = i as usize;
    for y in 0..h {
        for x in 0..w {
            let o = (y * w + x) * 4;
            buf[o] = ((x + t * 4) & 0xff) as u8;
            buf[o + 1] = ((y + t * 2) & 0xff) as u8;
            buf[o + 2] = ((x + y + t * 6) & 0xff) as u8;
            buf[o + 3] = 255;
        }
    }
}

/// Await `setTimeout(ms)` as a future (paces + yields to the event loop). Resolves
/// immediately if there is no `window`.
async fn sleep_ms(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| match web_sys::window() {
        Some(win) => {
            let _ = win.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms);
        }
        None => {
            let _ = resolve.call0(&JsValue::NULL);
        }
    });
    let _ = JsFuture::from(promise).await;
}
