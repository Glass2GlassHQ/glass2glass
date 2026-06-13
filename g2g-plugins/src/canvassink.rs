//! HTML canvas presentation sink (browser/wasm). Consumes decoded RGBA
//! `System` frames and draws them to a `<canvas>` via the 2D context
//! (`ImageData` + `putImageData`), completing the in-browser glass-to-glass
//! path `WebSocketSrc -> WebCodecsDecode -> CanvasSink` (M41).
//!
//! 2D presentation is the robust, dependency-free path. A WebGPU zero-copy sink
//! (decoded `MemoryDomain::WebGPUBuffer` straight into a `GPUTexture`) is a
//! follow-up: it needs an async adapter/device handshake and a core keep-alive
//! for the WebGPU domain (the §5.1 wgpu compute pillar builds on the same).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, HardwareError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate, RawVideoFormat,
};

use wasm_bindgen::{Clamped, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

#[derive(Debug)]
pub struct CanvasSink {
    canvas_id: String,
    ctx: Option<CanvasRenderingContext2d>,
    width: u32,
    height: u32,
    configured: bool,
    presented: u64,
}

impl CanvasSink {
    /// `canvas_id` is the `id` of an existing `<canvas>` element in the DOM;
    /// the context is acquired in `configure_pipeline`.
    pub fn new(canvas_id: impl Into<String>) -> Self {
        Self {
            canvas_id: canvas_id.into(),
            ctx: None,
            width: 0,
            height: 0,
            configured: false,
            presented: 0,
        }
    }

    /// Count of frames drawn to the canvas. Useful in tests.
    pub fn presented(&self) -> u64 {
        self.presented
    }

    fn present(&mut self, frame: &Frame) -> Result<(), G2gError> {
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 {
            return Ok(()); // no caps yet: nothing to size an ImageData with
        }
        let MemoryDomain::System(slice) = &frame.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let bytes = slice.as_slice();
        if bytes.len() != (w as usize) * (h as usize) * 4 {
            return Err(G2gError::CapsMismatch);
        }
        let image = ImageData::new_with_u8_clamped_array_and_sh(Clamped(bytes), w, h)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let ctx = self.ctx.as_ref().ok_or(G2gError::NotConfigured)?;
        put_image_data(ctx, &image)?;
        self.presented += 1;
        Ok(())
    }
}

impl AsyncElement for CanvasSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&rgba_any())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(rgba_any()))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let err = || G2gError::Hardware(HardwareError::Other);
        let window = web_sys::window().ok_or_else(err)?;
        let document = window.document().ok_or_else(err)?;
        let element = document.get_element_by_id(&self.canvas_id).ok_or_else(err)?;
        let canvas: HtmlCanvasElement = element.dyn_into().map_err(|_| err())?;
        let ctx = canvas
            .get_context("2d")
            .map_err(|_| err())?
            .ok_or_else(err)?
            .dyn_into::<CanvasRenderingContext2d>()
            .map_err(|_| err())?;
        self.ctx = Some(ctx);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::CapsChanged(Caps::RawVideo {
                    format: RawVideoFormat::Rgba8,
                    width,
                    height,
                    ..
                }) => {
                    self.width = fixed_or_zero(&width);
                    self.height = fixed_or_zero(&height);
                }
                // A non-RGBA caps change is a negotiation error for this sink.
                PipelinePacket::CapsChanged(_) => return Err(G2gError::CapsMismatch),
                PipelinePacket::DataFrame(frame) => self.present(&frame)?,
                PipelinePacket::Flush | PipelinePacket::Eos => {}
            }
            Ok(())
        })
    }
}

impl PadTemplates for CanvasSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(rgba_any()))])
    }
}

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

/// `putImageData(image, 0, 0)`. The dx/dy argument type differs by web-sys cfg
/// (`f64` on the stable bindings, `i32` under `web_sys_unstable_apis`, which the
/// `web-codecs` build sets globally), so the overload is selected at compile
/// time. The `allow` keeps the custom cfg quiet across the 1.75 MSRV (where the
/// lint name itself is unknown) and newer toolchains alike.
#[allow(unknown_lints, unexpected_cfgs)]
fn put_image_data(ctx: &CanvasRenderingContext2d, image: &ImageData) -> Result<(), G2gError> {
    #[cfg(web_sys_unstable_apis)]
    let r = ctx.put_image_data(image, 0, 0);
    #[cfg(not(web_sys_unstable_apis))]
    let r = ctx.put_image_data(image, 0.0, 0.0);
    r.map_err(|_| G2gError::Hardware(HardwareError::Other))
}

fn fixed_or_zero(d: &Dim) -> u32 {
    match d {
        Dim::Fixed(v) => *v,
        _ => 0,
    }
}
