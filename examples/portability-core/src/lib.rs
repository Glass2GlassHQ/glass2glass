//! The shared processing core of the portability showcase: proof that the same
//! g2g graph runs across targets.
//!
//! [`overlay_transforms`] builds the detection-overlay processing stages once,
//! from portable g2g elements. The native runner (`portability-native`) and the
//! browser build (`g2g-web`) both construct their middle-of-the-graph from this
//! exact function; only the source (a test pattern / a WebCodecs decode) and the
//! sink (a file / a `<canvas>`) differ per target. Same `AsyncElement`s, same
//! `Caps` negotiation, same `run_graph` runner underneath, from a Cortex-M-capable
//! `no_std` core up through a CPU server and the browser.
//!
//! The detection here is synthetic (a planted box decoded through the *real*
//! [`DetectionPostprocess`]), so the core needs no model and runs identically
//! everywhere; swap `SyntheticDetect` for a real inference element (native
//! `OrtInference`, browser `ort-web`, or a remote server) without touching the
//! rest of the graph, exactly the Architecture A / B story.

use core::future::Future;
use core::pin::Pin;

use g2g_core::{
    AnalyticsMeta, AsyncElement, BBox, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError,
    ObjectDetection, OutputSink, PipelinePacket, RawVideoFormat, TensorDType, TensorLayout,
    TensorShape,
};
use g2g_ml::detect::DetectionPostprocess;
use g2g_plugins::analyticsoverlay::AnalyticsOverlay;

/// A synthetic YOLOv8 output tensor with one planted class-0 detection covering
/// the centre half of the frame (normalized box `[0.25, 0.25, 0.5, 0.5]`).
///
/// Channel-major `[1, 4 + C, A]` with `C = 1`, `A = 1`, so the flat f32s are
/// `[cx, cy, w, h, score]`. With a 640x640 model input, `cx = cy = w = h = 320`
/// decodes (via `(cx - w/2)/640` etc.) to the `[0.25, 0.25, 0.5, 0.5]` box;
/// `score = 0.9` clears the confidence threshold.
const SYNTH_CHANNELS: u32 = 5;
const SYNTH_ANCHORS: u32 = 1;
const SYNTH_TENSOR: [f32; 5] = [320.0, 320.0, 320.0, 320.0, 0.9];

/// Reference to the planted detection for assertions: the class-0 box covering
/// the centre half of the frame, normalized `[0, 1]`.
pub const SYNTH_BOX: BBox = BBox { x: 0.25, y: 0.25, w: 0.5, h: 0.5 };

/// RGBA8 identity transform that attaches synthetic detection metadata, decoded
/// through the real [`DetectionPostprocess`]. Portable (pure Rust, no OS / web
/// deps): the same element compiles for native and wasm. The stand-in for a real
/// inference element in the portability showcase.
#[derive(Debug)]
pub struct SyntheticDetect {
    postprocess: DetectionPostprocess,
    configured: bool,
    emitted: u64,
}

impl Default for SyntheticDetect {
    fn default() -> Self {
        Self::new()
    }
}

impl SyntheticDetect {
    /// A detector with a 0.5 confidence / 0.5 IoU threshold at 640x640 input.
    pub fn new() -> Self {
        Self {
            postprocess: DetectionPostprocess::new(0.5, 0.5).with_input_size(640, 640),
            configured: false,
            emitted: 0,
        }
    }

    /// Count of frames a detection set was attached to. Useful in tests.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn accepts(caps: &Caps) -> bool {
        matches!(caps, Caps::RawVideo { format: RawVideoFormat::Rgba8, .. })
    }

    /// Run the real decode over the synthetic model output.
    fn detections(&self) -> Vec<ObjectDetection> {
        self.postprocess.detect(&SYNTH_TENSOR)
    }
}

impl AsyncElement for SyntheticDetect {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::accepts(upstream_caps) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Identity: pixels and geometry pass through; only metadata is added.
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if Self::accepts(input) {
                CapsSet::one(input.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !Self::accepts(absolute_caps) {
            return Err(G2gError::CapsMismatch);
        }
        let tensor_caps = Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, SYNTH_CHANNELS, SYNTH_ANCHORS]),
            layout: TensorLayout::Nchw,
        };
        self.postprocess.configure_pipeline(&tensor_caps)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    let mut analytics = AnalyticsMeta::new();
                    for d in self.detections() {
                        analytics.add_detection(d);
                    }
                    frame.meta.attach(analytics);
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                // The runner's transform arm forwards EOS; don't double it.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// The portable processing stages of the showcase pipeline: `SyntheticDetect`
/// (attach detections) then `AnalyticsOverlay` (draw the boxes). This single
/// definition is the middle of the graph on every target; the caller wires it
/// between a target-specific source and sink with
/// `vec![&mut stages.detect, &mut stages.overlay]` (concrete `&mut` so the
/// unsizing to `&mut dyn DynAsyncElement` picks the borrow's lifetime freely,
/// which a `Box<dyn ..>` can't under `&mut` invariance). The browser prepends its
/// `WebCodecsDecode`; the native runner prepends a test-pattern source.
#[derive(Debug)]
pub struct OverlayStages {
    /// Attaches the (synthetic) detections.
    pub detect: SyntheticDetect,
    /// Draws the detection boxes onto the RGBA frame.
    pub overlay: AnalyticsOverlay,
}

/// Build the shared processing stages. `overlay_thickness` is the box outline
/// width in pixels.
pub fn overlay_stages(overlay_thickness: u32) -> OverlayStages {
    OverlayStages {
        detect: SyntheticDetect::new(),
        overlay: AnalyticsOverlay::new().with_thickness(overlay_thickness),
    }
}
