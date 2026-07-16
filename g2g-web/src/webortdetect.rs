//! `WebOrtDetect`: Stage 2 of the in-browser YOLO chain (Architecture A) - a
//! real ONNX YOLOv8 detector running in the browser via ONNX Runtime Web.
//!
//! An RGBA8 identity transform (pixels pass through unchanged) that runs a real
//! model over each frame and attaches the decoded detections as [`AnalyticsMeta`],
//! the drop-in replacement for Stage 1's synthetic [`WebDetect`]. It owns the
//! g2g half of the work in Rust and delegates only the ONNX `session.run` to a
//! small JS shim (`ort-shim.js`), so the pipeline stays one typed graph:
//!
//! 1. preprocess: decoded RGBA -> a `[1, 3, 640, 640]` NCHW f32 tensor (RGB,
//!    normalized to `[0, 1]`), the standard YOLOv8 input;
//! 2. inference: hand the flat f32s to `ort_run`, which feeds ONNX Runtime Web
//!    and returns the output `Float32Array` + its dims (`[1, 84, 8400]` for a
//!    COCO YOLOv8n: 4 box + 80 class channels over 8400 anchors);
//! 3. postprocess: decode + NMS through the SAME `g2g-ml` [`DetectionPostprocess`]
//!    the native chain uses (channel-major decode, normalized `[0,1]` boxes);
//! 4. attach [`AnalyticsMeta`] and forward the frame, for `AnalyticsOverlay` to
//!    draw and `CanvasSink` to present.
//!
//! Boxes are normalized to the model input, and the frame is resized whole-to-whole
//! (no letterbox), so the `[0,1]` coordinates map straight onto the display frame
//! the overlay denormalizes against.
//!
//! ONNX Runtime Web is loaded single-threaded from the CDN by the shim (no
//! SharedArrayBuffer / cross-origin isolation), and the session is created lazily
//! on the first frame (`configure_pipeline` cannot be async). Lives in the wasm
//! leaf crate for the same reason as [`WebDetect`]: it reuses `g2g-ml`.
//!
//! [`WebDetect`]: crate::webdetect::WebDetect

use core::future::Future;
use core::pin::Pin;

use g2g_core::{
    AnalyticsMeta, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError,
    HardwareError, MemoryDomain, OutputSink, PipelinePacket, RawVideoFormat, TensorDType,
    TensorLayout, TensorShape,
};
use g2g_ml::detect::DetectionPostprocess;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

// The ONNX session lives in JS (onnxruntime-web); g2g owns pre/postprocess. See
// ort-shim.js. Both calls are async (Promise-returning); `catch` maps a rejection
// to `Err(JsValue)`.
#[wasm_bindgen(module = "/ort-shim.js")]
extern "C" {
    #[wasm_bindgen(js_name = ort_init, catch)]
    async fn ort_init(model_url: &str) -> Result<(), JsValue>;

    #[wasm_bindgen(js_name = ort_run, catch)]
    async fn ort_run(data: Vec<f32>, n: u32, c: u32, h: u32, w: u32) -> Result<JsValue, JsValue>;
}

/// YOLOv8 model input side length (square). The frame is resized to this.
const MODEL_SIZE: u32 = 640;

/// Real in-browser YOLOv8 detector. See the module docs.
#[derive(Debug)]
pub struct WebOrtDetect {
    /// URL the browser fetches the `.onnx` model from (same origin recommended).
    model_url: String,
    /// YOLOv8 decode + NMS, configured from the model's real output dims on the
    /// first inference.
    postprocess: DetectionPostprocess,
    postprocess_configured: bool,
    /// Set once `ort_init` has created the session (lazy, on the first frame).
    session_ready: bool,
    /// Display geometry of the incoming RGBA frames, for the resize.
    width: u32,
    height: u32,
    configured: bool,
    emitted: u64,
}

impl WebOrtDetect {
    /// A detector fetching the model from `model_url`, with YOLOv8's usual 0.25
    /// confidence / 0.45 IoU thresholds (the class scores are post-sigmoid `[0,1]`
    /// in a standard export). Box coordinates normalize against the 640 input.
    pub fn new(model_url: impl Into<String>) -> Self {
        Self {
            model_url: model_url.into(),
            postprocess: DetectionPostprocess::new(0.25, 0.45)
                .with_input_size(MODEL_SIZE, MODEL_SIZE),
            postprocess_configured: false,
            session_ready: false,
            width: 0,
            height: 0,
            configured: false,
            emitted: 0,
        }
    }

    /// Count of frames a detection set was attached to. Useful in tests.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// Whether `caps` is RGBA8 (geometry may still be unfixed at negotiation).
    fn accepts(caps: &Caps) -> bool {
        matches!(caps, Caps::RawVideo { format: RawVideoFormat::Rgba8, .. })
    }

    /// Resize the RGBA8 frame (whole-to-whole, nearest-neighbour) into a
    /// `[1, 3, 640, 640]` NCHW f32 tensor: channel-planar R, then G, then B, each
    /// normalized to `[0, 1]`. Whole-to-whole (not letterboxed) so the decoded
    /// `[0,1]` boxes map directly onto the display frame.
    fn preprocess(&self, rgba: &[u8]) -> Vec<f32> {
        let (sw, sh) = (MODEL_SIZE as usize, MODEL_SIZE as usize);
        let (w, h) = (self.width as usize, self.height as usize);
        let plane = sw * sh;
        let mut out = vec![0f32; 3 * plane];
        // sw == sh == MODEL_SIZE (a nonzero const), so the nearest-neighbour source
        // index never divides by zero.
        for y in 0..sh {
            let sy = y * h / sh;
            for x in 0..sw {
                let sx = x * w / sw;
                let si = (sy * w + sx) * 4;
                let di = y * sw + x;
                out[di] = rgba[si] as f32 / 255.0; // R
                out[plane + di] = rgba[si + 1] as f32 / 255.0; // G
                out[2 * plane + di] = rgba[si + 2] as f32 / 255.0; // B
            }
        }
        out
    }
}

/// Map any JS-side failure to a hardware error (the browser boundary).
fn js_err(_: JsValue) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

impl AsyncElement for WebOrtDetect {
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
        let Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width,
            height,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        // Geometry may still be a placeholder at negotiation; the real dims land
        // via CapsChanged before the first frame (the browser-decode contract).
        self.width = fixed_or_zero(width);
        self.height = fixed_or_zero(height);
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
                    // Lazy session init (configure_pipeline is sync; this is not).
                    if !self.session_ready {
                        ort_init(&self.model_url).await.map_err(js_err)?;
                        self.session_ready = true;
                    }
                    if self.width == 0 || self.height == 0 {
                        // No fixed geometry yet: forward untouched.
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    }
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let need = self.width as usize * self.height as usize * 4;
                    let bytes = slice.as_slice();
                    if bytes.len() < need {
                        return Err(G2gError::CapsMismatch);
                    }
                    let input = self.preprocess(&bytes[..need]);

                    let result = ort_run(input, 1, 3, MODEL_SIZE, MODEL_SIZE)
                        .await
                        .map_err(js_err)?;
                    let data_val = js_sys::Reflect::get(&result, &JsValue::from_str("data"))
                        .map_err(js_err)?;
                    let dims_val = js_sys::Reflect::get(&result, &JsValue::from_str("dims"))
                        .map_err(js_err)?;
                    let data = data_val
                        .dyn_into::<js_sys::Float32Array>()
                        .map_err(js_err)?
                        .to_vec();

                    // Configure the decoder from the model's real output shape once
                    // (e.g. [1, 84, 8400] for COCO YOLOv8n).
                    if !self.postprocess_configured {
                        let dims: Vec<u32> = js_sys::Array::from(&dims_val)
                            .iter()
                            .map(|v| v.as_f64().unwrap_or(0.0) as u32)
                            .collect();
                        let caps = Caps::Tensor {
                            dtype: TensorDType::F32,
                            shape: TensorShape(dims),
                            layout: TensorLayout::Nchw,
                        };
                        self.postprocess.configure_pipeline(&caps)?;
                        self.postprocess_configured = true;
                    }

                    let detections = self.postprocess.detect(&data);
                    // Log the per-frame detection count + top labels: the demo's
                    // ground-truth signal that a real model ran (the overlay only
                    // shows boxes, not counts).
                    let mut labels: Vec<u32> = detections.iter().map(|d| d.label).collect();
                    labels.sort_unstable();
                    labels.dedup();
                    web_sys::console::log_1(&JsValue::from_str(&format!(
                        "g2g[ort]: frame {} -> {} detections, classes {:?}",
                        self.emitted,
                        detections.len(),
                        labels
                    )));
                    let mut analytics = AnalyticsMeta::new();
                    for d in detections {
                        analytics.add_detection(d);
                    }
                    frame.meta.attach(analytics);
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    if let Caps::RawVideo { width, height, .. } = &caps {
                        self.width = fixed_or_zero(width);
                        self.height = fixed_or_zero(height);
                    }
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

fn fixed_or_zero(d: &Dim) -> u32 {
    match d {
        Dim::Fixed(v) => *v,
        _ => 0,
    }
}
