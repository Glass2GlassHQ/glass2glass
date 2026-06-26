//! Detection post-processing (DESIGN.md §5.3): the element that turns a
//! YOLO-style model output tensor into structured bounding-box detections, the
//! first producer of the per-frame analytics metadata graph (`g2g-core::meta`).
//! Composes after inference: `... -> OrtInference -> DetectionPostprocess`.
//!
//! Decodes the common YOLOv8 output layout `[1, 4 + C, A]` (channel-major: 4
//! box channels then `C` class scores, across `A` anchors), applies a
//! confidence threshold and per-class non-maximum suppression, and attaches an
//! [`AnalyticsMeta`] of [`ObjectDetection`]s to the frame, which it forwards
//! unchanged (an identity pass-through carrying metadata). Box coordinates are
//! normalized to `[0, 1]` by the model input size, so they survive a downstream
//! scale / crop without a rewrite.
//!
//! Pure Rust, no inference engine; gated behind the `analytics` feature (which
//! pulls `g2g-core/metadata`).

use core::future::Future;
use core::pin::Pin;

use g2g_core::{
    AnalyticsMeta, AsyncElement, BBox, Caps, CapsConstraint, CapsSet, ConfigureOutcome, G2gError,
    MemoryDomain, ObjectDetection, OutputSink, PipelinePacket, TensorDType,
};

/// Default model input resolution used to normalize box coordinates.
const DEFAULT_INPUT_SIZE: f32 = 640.0;

#[derive(Debug)]
pub struct DetectionPostprocess {
    /// Minimum class score to emit a detection.
    conf_threshold: f32,
    /// IoU above which a lower-scoring same-class box is suppressed.
    iou_threshold: f32,
    /// Model input size used to normalize box coordinates to `[0, 1]`.
    input_w: f32,
    input_h: f32,
    /// Channels (`4 + C`) and anchors (`A`), parsed from the configured tensor
    /// shape `[1, 4 + C, A]`.
    channels: usize,
    anchors: usize,
    input_caps: Option<Caps>,
    configured: bool,
    emitted: u64,
}

impl DetectionPostprocess {
    /// A decoder with the given confidence and IoU thresholds, default
    /// 640x640 input normalization.
    pub fn new(conf_threshold: f32, iou_threshold: f32) -> Self {
        Self {
            conf_threshold,
            iou_threshold,
            input_w: DEFAULT_INPUT_SIZE,
            input_h: DEFAULT_INPUT_SIZE,
            channels: 0,
            anchors: 0,
            input_caps: None,
            configured: false,
            emitted: 0,
        }
    }

    /// Set the model input resolution box coordinates are normalized against.
    pub fn with_input_size(mut self, width: u32, height: u32) -> Self {
        self.input_w = width as f32;
        self.input_h = height as f32;
        self
    }

    /// Number of frames processed (one detection set per input frame).
    pub fn processed_count(&self) -> u64 {
        self.emitted
    }

    /// Accept an F32 tensor of rank-3 shape `[1, 4 + C, A]` with `C >= 1`.
    fn parse_shape(caps: &Caps) -> Option<(usize, usize)> {
        let Caps::Tensor { dtype: TensorDType::F32, shape, .. } = caps else {
            return None;
        };
        match shape.0.as_slice() {
            [1, ch, a] if *ch >= 5 && *a >= 1 => Some((*ch as usize, *a as usize)),
            _ => None,
        }
    }

    /// Decode the flat channel-major tensor into normalized detections above the
    /// confidence threshold (pre-NMS).
    fn decode(&self, values: &[f32]) -> Vec<ObjectDetection> {
        let a = self.anchors;
        let classes = self.channels - 4;
        let mut out = Vec::new();
        for ai in 0..a {
            let cx = values[ai];
            let cy = values[a + ai];
            let w = values[2 * a + ai];
            let h = values[3 * a + ai];
            // Best class score for this anchor.
            let mut best_label = 0u32;
            let mut best_score = f32::MIN;
            for cls in 0..classes {
                let s = values[(4 + cls) * a + ai];
                if s > best_score {
                    best_score = s;
                    best_label = cls as u32;
                }
            }
            if best_score >= self.conf_threshold {
                out.push(ObjectDetection {
                    bbox: BBox {
                        x: (cx - w / 2.0) / self.input_w,
                        y: (cy - h / 2.0) / self.input_h,
                        w: w / self.input_w,
                        h: h / self.input_h,
                    },
                    label: best_label,
                    confidence: best_score,
                });
            }
        }
        out
    }

    /// Greedy per-class non-maximum suppression: highest confidence first,
    /// dropping later same-class boxes that overlap a kept one beyond the IoU
    /// threshold.
    fn nms(&self, mut dets: Vec<ObjectDetection>) -> Vec<ObjectDetection> {
        dets.sort_by(|x, y| {
            y.confidence.partial_cmp(&x.confidence).unwrap_or(core::cmp::Ordering::Equal)
        });
        let mut kept: Vec<ObjectDetection> = Vec::new();
        for cand in dets {
            let suppressed = kept.iter().any(|k| {
                k.label == cand.label && k.bbox.iou(&cand.bbox) > self.iou_threshold
            });
            if !suppressed {
                kept.push(cand);
            }
        }
        kept
    }
}

impl AsyncElement for DetectionPostprocess {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::parse_shape(upstream_caps).is_some() {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Identity: the tensor passes through unchanged; only metadata is added.
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if Self::parse_shape(input).is_some() {
                CapsSet::one(input.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (channels, anchors) = Self::parse_shape(absolute_caps).ok_or(G2gError::CapsMismatch)?;
        self.channels = channels;
        self.anchors = anchors;
        self.input_caps = Some(absolute_caps.clone());
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let bytes = slice.as_slice();
                    let expected = self.channels * self.anchors * 4;
                    if bytes.len() != expected {
                        return Err(G2gError::CapsMismatch);
                    }
                    let values: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    if values.iter().any(|v| !v.is_finite()) {
                        return Err(G2gError::CapsMismatch);
                    }

                    let detections = self.nms(self.decode(&values));
                    let mut analytics = AnalyticsMeta::new();
                    for d in detections {
                        analytics.add_detection(d);
                    }
                    // Attach the graph and forward the (unchanged) tensor frame.
                    frame.meta.attach(analytics);
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // Control packets pass through; a per-input caps change re-parses
                // the shape so the decoder tracks `C` / `A`.
                PipelinePacket::CapsChanged(caps) => {
                    if let Some((ch, a)) = Self::parse_shape(&caps) {
                        self.channels = ch;
                        self.anchors = a;
                        self.input_caps = Some(caps.clone());
                    }
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                // Drop EOS: the runner's transform arm forwards it; re-pushing it
                // here double-pushes onto a full link (the project EOS contract).
                // Forward the remaining control packets unchanged.
                PipelinePacket::Eos => {}
                PipelinePacket::Flush => {
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, PushOutcome, TensorLayout, TensorShape};

    /// Capturing sink that records the AnalyticsMeta of the last frame pushed.
    #[derive(Default)]
    struct MetaSink {
        last: Option<AnalyticsMeta>,
    }
    impl OutputSink for MetaSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                if let PipelinePacket::DataFrame(frame) = packet {
                    if let Some(a) = frame.meta.get::<AnalyticsMeta>() {
                        self.last = Some(a.clone());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn tensor_caps(ch: u32, a: u32) -> Caps {
        Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape(vec![1, ch, a]),
            layout: TensorLayout::Nchw,
        }
    }

    /// Channel-major [1, ch, a] tensor from per-channel rows, as LE f32 bytes.
    fn tensor_frame(channels: &[[f32; 3]]) -> Frame {
        let mut bytes = Vec::new();
        for ch in channels {
            for v in ch {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        }
    }

    #[tokio::test]
    async fn decodes_and_nms_suppresses_overlapping_same_class() {
        // [1, 6, 3]: 4 box channels + 2 classes, 3 anchors. Anchor 0 and 1 are
        // the same class and overlap (NMS keeps the higher); anchor 2 is a
        // distinct class far away.
        let mut det = DetectionPostprocess::new(0.5, 0.5).with_input_size(640, 640);
        det.configure_pipeline(&tensor_caps(6, 3)).unwrap();

        let frame = tensor_frame(&[
            [100.0, 105.0, 400.0], // cx
            [100.0, 102.0, 400.0], // cy
            [40.0, 40.0, 40.0],    // w
            [40.0, 40.0, 40.0],    // h
            [0.90, 0.80, 0.05],    // class 0 scores
            [0.10, 0.05, 0.95],    // class 1 scores
        ]);

        let mut sink = MetaSink::default();
        det.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();

        let meta = sink.last.expect("frame carries AnalyticsMeta");
        let mut got: Vec<(u32, f32)> =
            meta.detections().map(|d| (d.label, d.confidence)).collect();
        got.sort_by_key(|(l, _)| *l);
        assert_eq!(got.len(), 2, "the overlapping same-class box was suppressed");
        assert_eq!(got[0].0, 0, "kept the class-0 detection");
        assert!((got[0].1 - 0.90).abs() < 1e-6, "kept the higher-confidence class-0 box");
        assert_eq!(got[1].0, 1, "kept the distinct class-1 detection");

        // Box 0 is normalized: x = (100 - 20)/640.
        let d0 = meta.detections().find(|d| d.label == 0).unwrap();
        assert!((d0.bbox.x - (80.0 / 640.0)).abs() < 1e-6);
        assert!((d0.bbox.w - (40.0 / 640.0)).abs() < 1e-6);
    }

    #[tokio::test]
    async fn below_threshold_yields_no_detections() {
        let mut det = DetectionPostprocess::new(0.9, 0.5);
        det.configure_pipeline(&tensor_caps(6, 3)).unwrap();
        let frame = tensor_frame(&[
            [100.0, 105.0, 400.0],
            [100.0, 102.0, 400.0],
            [40.0, 40.0, 40.0],
            [40.0, 40.0, 40.0],
            [0.5, 0.4, 0.05], // all class scores below 0.9
            [0.1, 0.05, 0.6],
        ]);
        let mut sink = MetaSink::default();
        det.process(PipelinePacket::DataFrame(frame), &mut sink).await.unwrap();
        let meta = sink.last.expect("empty AnalyticsMeta still attached");
        assert_eq!(meta.detections().count(), 0);
    }

    #[test]
    fn rejects_non_tensor_and_wrong_rank() {
        let det = DetectionPostprocess::new(0.5, 0.5);
        assert!(det.intercept_caps(&tensor_caps(6, 3)).is_ok());
        // Rank-2 tensor (a classifier head) is not a detection output.
        let flat = Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape(vec![1, 10]),
            layout: TensorLayout::Nchw,
        };
        assert!(det.intercept_caps(&flat).is_err());
    }
}
