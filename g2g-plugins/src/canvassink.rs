//! HTML canvas presentation sink (browser/wasm). Consumes decoded RGBA
//! `System` frames and draws them to a `<canvas>` via the 2D context
//! (`ImageData` + `putImageData`), completing the in-browser glass-to-glass
//! path `WebSocketSrc -> WebCodecsDecode -> CanvasSink` (M41).
//!
//! 2D presentation is the robust, dependency-free path;
//! [`WebGpuCanvasSink`](crate::webgpucanvassink::WebGpuCanvasSink) is the
//! zero-copy one, sampling the decoded `VideoFrame` as a `GPUExternalTexture`
//! with no readback into wasm memory.
//!
//! [`CanvasSink::from_offscreen_canvas`] takes an `OffscreenCanvas` instead of
//! a canvas id, which is how the sink presents from inside a worker (M1054):
//! a worker has no `document` to look an id up in.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::element::QosMessage;
use g2g_core::frame::Frame;
use g2g_core::{
    AsyncElement, BusHandle, Caps, CapsConstraint, CapsSet, ClockSync, ConfigureOutcome, Dim,
    G2gError, HardwareError, Interlace, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PresentationPacer, PropError, PropValue, PropertySpec, Rate, RawVideoFormat, PACING_PROPERTIES,
};

use crate::wasmclock::wait_to_present;
use crate::webutil::{Canvas, CanvasTarget};

use wasm_bindgen::{Clamped, JsCast};
use web_sys::{
    CanvasRenderingContext2d, ImageData, OffscreenCanvas, OffscreenCanvasRenderingContext2d,
};

/// # Example
///
/// ```ignore
/// use g2g_plugins::canvassink::CanvasSink;
///
/// let sink = CanvasSink::new("video-canvas").with_max_lateness_ns(20_000_000);
/// ```
#[derive(Debug)]
pub struct CanvasSink {
    target: CanvasTarget,
    ctx: Option<Context2d>,
    width: u32,
    height: u32,
    configured: bool,
    presented: u64,
    /// PTS pacing + QoS late-drop: idle until the runner hands over a clock (a
    /// `WasmClock` in the browser), and the default lateness bound never drops.
    pacer: PresentationPacer,
}

impl CanvasSink {
    /// `canvas_id` is the `id` of an existing `<canvas>` element in the DOM;
    /// the context is acquired in `configure_pipeline`. Main thread only, since
    /// the id is looked up in `document`.
    pub fn new(canvas_id: impl Into<String>) -> Self {
        Self::with_target(CanvasTarget::ElementId(canvas_id.into()))
    }

    /// Present to an `OffscreenCanvas` the page transferred in
    /// (`canvas.transferControlToOffscreen()`), which is how a graph running
    /// inside a worker draws to the page.
    pub fn from_offscreen_canvas(canvas: OffscreenCanvas) -> Self {
        Self::with_target(CanvasTarget::Offscreen(canvas))
    }

    fn with_target(target: CanvasTarget) -> Self {
        Self {
            target,
            ctx: None,
            width: 0,
            height: 0,
            configured: false,
            presented: 0,
            pacer: PresentationPacer::new(),
        }
    }

    /// Count of frames drawn to the canvas. Useful in tests.
    pub fn presented(&self) -> u64 {
        self.presented
    }

    /// QoS late-drop bound: once PTS pacing is engaged, a frame past its
    /// deadline by more than `ns` is dropped instead of drawn late, so the sink
    /// catches up. The default (`u64::MAX`) never drops.
    pub fn with_max_lateness_ns(mut self, ns: u64) -> Self {
        self.pacer.set_max_lateness_ns(ns);
        self
    }

    /// Post a running-stats `Qos` report every `ns` of clock time while frames
    /// draw, on top of the per-drop reports. `0` (the default) reports only
    /// drops.
    pub fn with_qos_interval_ns(mut self, ns: u64) -> Self {
        self.pacer.set_report_interval_ns(ns);
        self
    }

    /// Attach the pipeline bus so QoS reports reach the application.
    pub fn with_bus(mut self, bus: BusHandle) -> Self {
        self.pacer.set_bus(bus);
        self
    }

    /// Frames dropped by QoS late-drop (past their deadline beyond the bound).
    pub fn late_dropped(&self) -> u64 {
        self.pacer.late_dropped()
    }

    fn present(&mut self, frame: &Frame) -> Result<(), G2gError> {
        let (w, h) = (self.width, self.height);
        if w == 0 || h == 0 {
            return Ok(()); // no caps yet: nothing to size an ImageData with
        }
        let slice = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        let bytes = slice;
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
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&rgba_any())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(rgba_any()))
    }

    /// Adopt the elected clock + base time so frames draw at their PTS deadline
    /// rather than as fast as the decoder delivers them.
    fn set_clock_sync(&mut self, sync: ClockSync) {
        self.pacer.set_clock_sync(sync);
    }

    /// Relay a late drop upstream (M174): the runner forwards it onto the
    /// incoming link, where the producer can shed load.
    fn take_qos(&mut self) -> Option<QosMessage> {
        self.pacer.take_qos()
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PACING_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        self.pacer
            .set_property(name, &value)
            .unwrap_or(Err(PropError::Unknown))
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        self.pacer.get_property(name)
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let err = || G2gError::Hardware(HardwareError::Other);
        let canvas = self.target.resolve()?;
        let object = canvas.context(CONTEXT_2D)?;
        self.ctx = Some(match canvas {
            Canvas::Element(_) => Context2d::Element(object.dyn_into().map_err(|_| err())?),
            Canvas::Offscreen(_) => Context2d::Offscreen(object.dyn_into().map_err(|_| err())?),
        });
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
                PipelinePacket::DataFrame(frame) => {
                    // PTS pacing on `setTimeout`: hold the frame until its deadline
                    // on the elected clock, or drop it when it is already too late
                    // (the QoS bound) or outside the segment. Unpaced without a
                    // clock: draw as fast as frames arrive.
                    let paced = self.pacer.judge(frame.timing.pts_ns, self.presented);
                    if !wait_to_present(paced).await {
                        return Ok(());
                    }
                    self.present(&frame)?
                }
                // Track the playback segment so PTS maps to running time (correct
                // across a seek), and re-anchor after a seek flush.
                PipelinePacket::Segment(seg) => self.pacer.set_segment(seg),
                PipelinePacket::Flush => self.pacer.flush(),
                PipelinePacket::Eos => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
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
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

/// The 2D context id, and the two context types it yields: an `OffscreenCanvas`
/// hands back its own context type, not a `CanvasRenderingContext2d`.
const CONTEXT_2D: &str = "2d";

#[derive(Debug)]
enum Context2d {
    Element(CanvasRenderingContext2d),
    Offscreen(OffscreenCanvasRenderingContext2d),
}

/// `putImageData(image, 0, 0)`. The dx/dy argument type differs by web-sys cfg
/// (`f64` on the stable bindings, `i32` under `web_sys_unstable_apis`, which the
/// `web-codecs` build sets globally), so the overload is selected at compile
/// time. The `allow` keeps the custom cfg quiet across the 1.75 MSRV (where the
/// lint name itself is unknown) and newer toolchains alike.
#[allow(unknown_lints, unexpected_cfgs)]
fn put_image_data(ctx: &Context2d, image: &ImageData) -> Result<(), G2gError> {
    #[cfg(web_sys_unstable_apis)]
    let (x, y) = (0, 0);
    #[cfg(not(web_sys_unstable_apis))]
    let (x, y) = (0.0, 0.0);
    match ctx {
        Context2d::Element(ctx) => ctx.put_image_data(image, x, y),
        Context2d::Offscreen(ctx) => ctx.put_image_data(image, x, y),
    }
    .map_err(|_| G2gError::Hardware(HardwareError::Other))
}

fn fixed_or_zero(d: &Dim) -> u32 {
    match d {
        Dim::Fixed(v) => *v,
        _ => 0,
    }
}
