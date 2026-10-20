//! Detection post-processing (design/ml.md): the element that turns a
//! YOLO-style model output tensor into structured bounding-box detections, the
//! first producer of the per-frame analytics metadata graph (`g2g-core::meta`).
//! Composes after inference: `... -> OrtInference -> DetectionPostprocess`.
//!
//! Decodes the common YOLOv8 output layout `[1, 4 + C, A]` (channel-major: 4
//! box channels then `C` class scores, across `A` anchors), applies a
//! confidence threshold and per-class non-maximum suppression, and attaches an
//! [`AnalyticsMeta`] of [`ObjectDetection`]s to the frame, which it forwards
//! unchanged (an identity pass-through carrying metadata). The `head` property
//! switches to a `-pose` export (`[1, 5 + 3K, A]`, emitting [`Pose`]s) or an
//! `-obb` export (`[1, 4 + C + 1, A]`, emitting [`OrientedDetection`]s). Box
//! coordinates are normalized to `[0, 1]` by the model input size, or by the
//! picture when `picture-width` / `picture-height` say the model input was a
//! letterbox of it, so they survive a downstream scale / crop without a rewrite.
//!
//! Pure Rust, no inference engine; gated behind the `analytics` feature (which
//! pulls `g2g-core/metadata`).

use core::future::Future;
use core::pin::Pin;

use crate::tensor_bytes::finite_f32_from_le_bytes;

use g2g_core::{
    AnalyticsMeta, AnalyticsNode, AsyncElement, BBox, Caps, CapsConstraint, CapsSet,
    ConfigureOutcome, G2gError, Keypoint, ObjectDetection, OrientedDetection, OutputSink,
    PipelinePacket, Pose, PropError, PropKind, PropValue, PropertySpec, TensorDType,
};

/// Model input resolution assumed when neither the properties nor a video link
/// give one.
const DEFAULT_INPUT_SIZE: f32 = 640.0;

/// Box channels at the front of every YOLO head: centre x, centre y, width,
/// height, in model input pixels.
const BOX_CHANNELS: usize = 4;
/// A pose keypoint is three channels: x, y, visibility.
const KEYPOINT_CHANNELS: usize = 3;

/// Which YOLO export layout the model output follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Head {
    Detect,
    Pose,
    Obb,
}

/// The `head` property's values, each one `Head::from_name` accepts.
const HEAD_NICKS: &str = "detect | pose | obb";

impl Head {
    fn from_name(name: &str) -> Option<Head> {
        match name {
            "detect" => Some(Head::Detect),
            "pose" => Some(Head::Pose),
            "obb" => Some(Head::Obb),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Head::Detect => "detect",
            Head::Pose => "pose",
            Head::Obb => "obb",
        }
    }

    /// The class and trailing-channel counts of a `channels`-wide output, `None`
    /// when this head cannot read that width. Pose has one person score then
    /// `K` keypoints, OBB has its classes then one angle channel.
    fn split(self, channels: usize) -> Option<(usize, usize)> {
        let after_box = channels.checked_sub(BOX_CHANNELS)?;
        match self {
            Head::Detect if after_box >= 1 => Some((after_box, 0)),
            Head::Pose => {
                let keypoint_channels = after_box.checked_sub(1)?;
                (keypoint_channels > 0 && keypoint_channels % KEYPOINT_CHANNELS == 0)
                    .then_some((1, keypoint_channels))
            }
            Head::Obb if after_box >= 2 => Some((after_box - 1, 1)),
            _ => None,
        }
    }
}

/// # Example
///
/// ```no_run
/// use g2g_ml::detect::DetectionPostprocess;
///
/// let postprocess = DetectionPostprocess::new(0.25, 0.45).with_input_size(640, 640);
/// ```
#[derive(Debug)]
pub struct DetectionPostprocess {
    /// Minimum class score to emit a detection.
    conf_threshold: f32,
    /// IoU above which a lower-scoring same-class box is suppressed.
    iou_threshold: f32,
    /// Model input size used to normalize box coordinates to `[0, 1]`. Unset
    /// takes it from the video link, which carries the model input size.
    input_w: Option<f32>,
    input_h: Option<f32>,
    /// Channels (`4 + C`) and anchors (`A`), parsed from the configured tensor
    /// shape `[1, 4 + C, A]`.
    channels: usize,
    anchors: usize,
    /// Which attached tensor to decode when the frame carries several. Empty
    /// takes the only one.
    tensor_name: String,
    head: Head,
    /// Size of the picture the model input was letterboxed from. Zero for both
    /// means the input was a plain stretch of the picture.
    picture_w: u32,
    picture_h: u32,
    input_caps: Option<Caps>,
    configured: bool,
    emitted: u64,
}

impl DetectionPostprocess {
    /// A decoder with the given confidence and IoU thresholds, normalizing
    /// against the video link's size, or 640x640 when there is none.
    pub fn new(conf_threshold: f32, iou_threshold: f32) -> Self {
        Self {
            conf_threshold,
            iou_threshold,
            input_w: None,
            input_h: None,
            channels: 0,
            anchors: 0,
            tensor_name: String::new(),
            head: Head::Detect,
            picture_w: 0,
            picture_h: 0,
            input_caps: None,
            configured: false,
            emitted: 0,
        }
    }

    /// Set the model input resolution box coordinates are normalized against.
    pub fn with_input_size(mut self, width: u32, height: u32) -> Self {
        self.input_w = Some(width as f32);
        self.input_h = Some(height as f32);
        self
    }

    /// The size the box coordinates divide by. `ortinfer` accepts video only at
    /// the model's own input size, so on the `attach-tensor` link the frame
    /// carries that size and an unset property should follow it rather than
    /// assume a square model.
    fn normalize_size(&self) -> (f32, f32) {
        let from_link = match &self.input_caps {
            Some(Caps::RawVideo {
                width: g2g_core::Dim::Fixed(w),
                height: g2g_core::Dim::Fixed(h),
                ..
            }) => Some((*w as f32, *h as f32)),
            _ => None,
        };
        (
            self.input_w
                .or(from_link.map(|(w, _)| w))
                .unwrap_or(DEFAULT_INPUT_SIZE),
            self.input_h
                .or(from_link.map(|(_, h)| h))
                .unwrap_or(DEFAULT_INPUT_SIZE),
        )
    }

    /// Decode the attached tensor of this name, for a frame carrying more than
    /// one model's output. Unset takes the only one.
    pub fn with_tensor_name(mut self, name: &str) -> Self {
        self.tensor_name = String::from(name);
        self
    }

    /// Number of frames processed (one detection set per input frame).
    pub fn processed_count(&self) -> u64 {
        self.emitted
    }

    /// Accept an F32 tensor of rank-3 shape `[1, channels, A]` whose channel
    /// count `head` can read.
    fn parse_shape(head: Head, caps: &Caps) -> Option<(usize, usize)> {
        let Caps::Tensor {
            dtype: TensorDType::F32,
            shape,
            ..
        } = caps
        else {
            return None;
        };
        Self::parse_dims(head, shape.dims())
    }

    /// The `(channels, A)` of a YOLO output shape `[1, channels, A]`.
    fn parse_dims(head: Head, dims: &[u32]) -> Option<(usize, usize)> {
        match dims {
            [1, ch, a] if *a >= 1 && head.split(*ch as usize).is_some() => {
                Some((*ch as usize, *a as usize))
            }
            _ => None,
        }
    }

    /// How model input pixels map onto the picture. Letterboxed only when both
    /// `picture-*` sizes are set: `configure_pipeline` and `process` refuse one
    /// without the other.
    fn picture_mapping(&self) -> PictureMapping {
        let (input_w, input_h) = self.normalize_size();
        if self.picture_w == 0 || self.picture_h == 0 {
            return PictureMapping::stretch(input_w, input_h);
        }
        PictureMapping::letterbox(
            input_w,
            input_h,
            self.picture_w as f32,
            self.picture_h as f32,
        )
    }

    fn check_picture_size(&self) -> Result<(), G2gError> {
        if (self.picture_w == 0) != (self.picture_h == 0) {
            return Err(G2gError::NotConfigured);
        }
        Ok(())
    }

    /// The two link shapes this element decodes on: a tensor frame (the model
    /// output IS the frame), or a video frame carrying the model output as a
    /// `TensorMeta` (`ortinfer attach-tensor=true`). Either way the packet is
    /// forwarded unchanged with an `AnalyticsMeta` added.
    fn accepts(head: Head, caps: &Caps) -> bool {
        Self::parse_shape(head, caps).is_some() || matches!(caps, Caps::RawVideo { .. })
    }

    /// The `channels * anchors` little-endian f32 values of a model output.
    /// Sizes with `checked_mul`, so an overflowing product fails the length
    /// check instead of panicking (debug) or wrapping to a small value that
    /// would admit a bogus payload.
    fn read_f32(bytes: &[u8], channels: usize, anchors: usize) -> Result<Vec<f32>, G2gError> {
        let expected = channels
            .checked_mul(anchors)
            .and_then(|n| n.checked_mul(TensorDType::F32.size()));
        if expected != Some(bytes.len()) {
            return Err(G2gError::CapsMismatch);
        }
        finite_f32_from_le_bytes(bytes)
    }

    /// Decode a flat channel-major YOLOv8 output tensor `[1, 4 + C, A]` (as f32
    /// values, sized by the configured `channels` / `anchors`) into normalized,
    /// NMS-filtered detections. This is the reusable core of [`process`]; a caller
    /// that already holds the model output as f32 (e.g. a browser `ort-web`
    /// session's `Float32Array`, decoded off the tensor-frame path) can decode
    /// directly without routing a `Caps::Tensor` frame through the element.
    ///
    /// [`process`]: AsyncElement::process
    pub fn detect(&self, values: &[f32]) -> Vec<ObjectDetection> {
        let layout = AnchorLayout {
            anchors: self.anchors,
            classes: self.channels.saturating_sub(BOX_CHANNELS),
            extra: 0,
        };
        let picture = self.picture_mapping();
        suppress_overlaps(
            decode_anchors(values, layout, self.conf_threshold),
            self.iou_threshold,
        )
        .into_iter()
        .map(|b| ObjectDetection {
            bbox: picture.bbox(b.detection.bbox),
            ..b.detection
        })
        .collect()
    }

    /// Decode a channel-major `-pose` output `[1, 5 + 3K, A]` into people with
    /// `K` keypoints each. One class, so suppression is class-agnostic.
    fn poses(&self, values: &[f32], layout: AnchorLayout) -> Vec<Pose> {
        let picture = self.picture_mapping();
        suppress_overlaps(
            decode_anchors(values, layout, self.conf_threshold),
            self.iou_threshold,
        )
        .into_iter()
        .map(|b| Pose {
            bbox: picture.bbox(b.detection.bbox),
            label: b.detection.label,
            confidence: b.detection.confidence,
            keypoints: b
                .extra
                .as_chunks::<KEYPOINT_CHANNELS>()
                .0
                .iter()
                .map(|&[x, y, confidence]| picture.keypoint(x, y, confidence))
                .collect(),
        })
        .collect()
    }

    /// Decode a channel-major `-obb` output `[1, 4 + C + 1, A]`, whose last
    /// channel is each box's angle in radians.
    fn oriented_detections(&self, values: &[f32], layout: AnchorLayout) -> Vec<OrientedDetection> {
        let candidates = decode_anchors(values, layout, self.conf_threshold)
            .into_iter()
            .map(|b| {
                let bbox = b.detection.bbox;
                OrientedDetection {
                    cx: bbox.x + bbox.w / 2.0,
                    cy: bbox.y + bbox.h / 2.0,
                    w: bbox.w,
                    h: bbox.h,
                    angle_radians: b.extra[0],
                    label: b.detection.label,
                    confidence: b.detection.confidence,
                }
            })
            .collect();
        let picture = self.picture_mapping();
        suppress_rotated_overlaps(candidates, self.iou_threshold)
            .into_iter()
            .map(|o| picture.oriented(o))
            .collect()
    }

    /// Decode `values` as the configured head into the frame's analytics graph.
    fn analyze(&self, values: &[f32]) -> Result<AnalyticsMeta, G2gError> {
        self.check_picture_size()?;
        let (classes, extra) = self
            .head
            .split(self.channels)
            .ok_or(G2gError::CapsMismatch)?;
        let layout = AnchorLayout {
            anchors: self.anchors,
            classes,
            extra,
        };
        let mut analytics = AnalyticsMeta::new();
        match self.head {
            Head::Detect => {
                for d in self.detect(values) {
                    analytics.add_detection(d);
                }
            }
            Head::Pose => {
                for p in self.poses(values, layout) {
                    analytics.push(AnalyticsNode::Pose(p));
                }
            }
            Head::Obb => {
                for o in self.oriented_detections(values, layout) {
                    analytics.push(AnalyticsNode::OrientedDetection(o));
                }
            }
        }
        Ok(analytics)
    }
}

/// Maps model input pixels to normalized `[0, 1]` picture coordinates as
/// `(v - pad) / span`. A stretched picture fills the input (no pad, the span is
/// the input size). A letterboxed one sits centred at the largest scale that
/// fits, as Ultralytics `LetterBox(center=True, auto=False)` places it
/// (ultralytics/data/augment.py), and its boxes and keypoints are clipped to the
/// picture as `clip_boxes` / `clip_coords` do (ultralytics/utils/ops.py).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PictureMapping {
    pad_x: f32,
    pad_y: f32,
    span_x: f32,
    span_y: f32,
    clip_to_picture: bool,
}

impl PictureMapping {
    pub(crate) fn stretch(input_w: f32, input_h: f32) -> Self {
        Self {
            pad_x: 0.0,
            pad_y: 0.0,
            span_x: input_w,
            span_y: input_h,
            clip_to_picture: false,
        }
    }

    /// Ultralytics rounds the resized size and the pad to whole pixels, which
    /// moves a point by under one model input pixel from this exact placement.
    fn letterbox(input_w: f32, input_h: f32, picture_w: f32, picture_h: f32) -> Self {
        let scale = (input_w / picture_w).min(input_h / picture_h);
        let span_x = picture_w * scale;
        let span_y = picture_h * scale;
        Self {
            pad_x: (input_w - span_x) / 2.0,
            pad_y: (input_h - span_y) / 2.0,
            span_x,
            span_y,
            clip_to_picture: true,
        }
    }

    fn x(&self, model_x: f32) -> f32 {
        (model_x - self.pad_x) / self.span_x
    }

    fn y(&self, model_y: f32) -> f32 {
        (model_y - self.pad_y) / self.span_y
    }

    fn clip(&self, v: f32) -> f32 {
        if self.clip_to_picture {
            v.clamp(0.0, 1.0)
        } else {
            v
        }
    }

    /// A box in model input pixels (top-left corner and size) as a normalized
    /// picture box.
    pub(crate) fn bbox(&self, model: BBox) -> BBox {
        if !self.clip_to_picture {
            return BBox {
                x: self.x(model.x),
                y: self.y(model.y),
                w: model.w / self.span_x,
                h: model.h / self.span_y,
            };
        }
        let x0 = self.clip(self.x(model.x));
        let y0 = self.clip(self.y(model.y));
        let x1 = self.clip(self.x(model.x + model.w));
        let y1 = self.clip(self.y(model.y + model.h));
        BBox {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }

    fn keypoint(&self, model_x: f32, model_y: f32, confidence: f32) -> Keypoint {
        Keypoint {
            x: self.clip(self.x(model_x)),
            y: self.clip(self.y(model_y)),
            confidence,
        }
    }

    /// Ultralytics leaves rotated boxes unclipped (`scale_boxes` with
    /// `xywh=True` returns before `clip_boxes`), so this does too.
    fn oriented(&self, model: OrientedDetection) -> OrientedDetection {
        OrientedDetection {
            cx: self.x(model.cx),
            cy: self.y(model.cy),
            w: model.w / self.span_x,
            h: model.h / self.span_y,
            ..model
        }
    }
}

/// How to read one YOLO channel-major output tensor `[1, 4 + classes + extra,
/// anchors]`: the 4 box channels, the class scores, then `extra` trailing
/// channels a head other than plain detection carries (the mask coefficients of
/// a `-seg` export, the keypoints of a `-pose` one, the angle of an `-obb` one).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AnchorLayout {
    pub anchors: usize,
    pub classes: usize,
    pub extra: usize,
}

/// One anchor's decoded box, in model input pixels, plus its trailing
/// non-class channel values (empty for a plain detector).
#[derive(Debug, Clone)]
pub(crate) struct AnchorBox {
    pub detection: ObjectDetection,
    pub extra: Vec<f32>,
}

/// Decode a flat channel-major output tensor into model-pixel boxes above
/// `conf_threshold` (pre-NMS). An input shorter than the layout describes yields
/// nothing rather than indexing past its end.
pub(crate) fn decode_anchors(
    values: &[f32],
    layout: AnchorLayout,
    conf_threshold: f32,
) -> Vec<AnchorBox> {
    let a = layout.anchors;
    let channels = BOX_CHANNELS + layout.classes + layout.extra;
    let needed = channels.checked_mul(a);
    if layout.classes == 0 || needed.is_none_or(|n| n > values.len()) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for ai in 0..a {
        let cx = values[ai];
        let cy = values[a + ai];
        let w = values[2 * a + ai];
        let h = values[3 * a + ai];
        // Best class score for this anchor.
        let mut best_label = 0u32;
        let mut best_score = f32::MIN;
        for cls in 0..layout.classes {
            let s = values[(BOX_CHANNELS + cls) * a + ai];
            if s > best_score {
                best_score = s;
                best_label = cls as u32;
            }
        }
        if best_score >= conf_threshold {
            let first_extra = BOX_CHANNELS + layout.classes;
            out.push(AnchorBox {
                detection: ObjectDetection {
                    bbox: BBox {
                        x: cx - w / 2.0,
                        y: cy - h / 2.0,
                        w,
                        h,
                    },
                    label: best_label,
                    confidence: best_score,
                },
                extra: (0..layout.extra)
                    .map(|e| values[(first_extra + e) * a + ai])
                    .collect(),
            });
        }
    }
    out
}

/// Greedy per-class non-maximum suppression: highest confidence first, dropping
/// later same-class boxes that overlap a kept one beyond the IoU threshold.
pub(crate) fn suppress_overlaps(mut boxes: Vec<AnchorBox>, iou_threshold: f32) -> Vec<AnchorBox> {
    boxes.sort_by(|x, y| {
        y.detection
            .confidence
            .partial_cmp(&x.detection.confidence)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut kept: Vec<AnchorBox> = Vec::new();
    for cand in boxes {
        let suppressed = kept.iter().any(|k| {
            k.detection.label == cand.detection.label
                && k.detection.bbox.iou(&cand.detection.bbox) > iou_threshold
        });
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

/// Ultralytics' rotated-box suppression (`non_max_suppression` with
/// `rotated=True` calls `TorchNMS.fast_nms` with `batch_probiou`,
/// ultralytics/utils/nms.py): Fast-NMS, which drops a box that overlaps any
/// higher-scoring same-class box, even one that was itself dropped.
fn suppress_rotated_overlaps(
    mut boxes: Vec<OrientedDetection>,
    iou_threshold: f32,
) -> Vec<OrientedDetection> {
    boxes.sort_by(|x, y| {
        y.confidence
            .partial_cmp(&x.confidence)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    (0..boxes.len())
        .filter(|&i| {
            !boxes[..i]
                .iter()
                .any(|k| k.label == boxes[i].label && probiou(k, &boxes[i]) > iou_threshold)
        })
        .map(|i| boxes[i])
        .collect()
}

/// `batch_probiou` in ultralytics/utils/metrics.py adds this to each
/// denominator and logarithm argument.
const PROBIOU_EPS: f32 = 1e-7;
/// `batch_probiou` clamps the Bhattacharyya distance to this.
const PROBIOU_MAX_DISTANCE: f32 = 100.0;
/// A uniform spread over a length `l` has variance `l^2 / 12`.
const UNIFORM_VARIANCE_DIVISOR: f32 = 12.0;

/// The covariance `[[a, c], [c, b]]` of the Gaussian a rotated box stands for
/// (`_get_covariance_matrix` in ultralytics/utils/metrics.py).
fn box_covariance(o: &OrientedDetection) -> (f32, f32, f32) {
    let along_w = o.w * o.w / UNIFORM_VARIANCE_DIVISOR;
    let along_h = o.h * o.h / UNIFORM_VARIANCE_DIVISOR;
    let (sin, cos) = o.angle_radians.sin_cos();
    (
        along_w * cos * cos + along_h * sin * sin,
        along_w * sin * sin + along_h * cos * cos,
        (along_w - along_h) * cos * sin,
    )
}

/// Ultralytics' probabilistic IoU of two rotated boxes (`batch_probiou`,
/// ultralytics/utils/metrics.py): each box is a Gaussian and the overlap is
/// `1 - sqrt(1 - exp(-bd))`, `bd` their Bhattacharyya distance
/// `t1 + t2 + t3`, with `t1 = ((a1+a2)(y1-y2)^2 + (b1+b2)(x1-x2)^2) / (4 D)`,
/// `t2 = (c1+c2)(x2-x1)(y1-y2) / (2 D)`, `D = (a1+a2)(b1+b2) - (c1+c2)^2`, and
/// `t3 = ln(D / (4 sqrt(det1 det2))) / 2`.
fn probiou(p: &OrientedDetection, q: &OrientedDetection) -> f32 {
    let (a1, b1, c1) = box_covariance(p);
    let (a2, b2, c2) = box_covariance(q);
    let (a, b, c) = (a1 + a2, b1 + b2, c1 + c2);
    let joint = a * b - c * c;
    let t1 = (a * (p.cy - q.cy).powi(2) + b * (p.cx - q.cx).powi(2)) / (joint + PROBIOU_EPS) * 0.25;
    let t2 = (c * (q.cx - p.cx) * (p.cy - q.cy)) / (joint + PROBIOU_EPS) * 0.5;
    let determinants = (a1 * b1 - c1 * c1).max(0.0) * (a2 * b2 - c2 * c2).max(0.0);
    let t3 = (joint / (4.0 * determinants.sqrt() + PROBIOU_EPS) + PROBIOU_EPS).ln() * 0.5;
    let distance = (t1 + t2 + t3).clamp(PROBIOU_EPS, PROBIOU_MAX_DISTANCE);
    1.0 - (1.0 - (-distance).exp() + PROBIOU_EPS).sqrt()
}

impl AsyncElement for DetectionPostprocess {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        if Self::accepts(self.head, upstream_caps) {
            Ok(upstream_caps.clone())
        } else {
            Err(G2gError::CapsMismatch)
        }
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        // Identity: the frame passes through unchanged; only metadata is added.
        let head = self.head;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            if Self::accepts(head, input) {
                CapsSet::one(input.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        if !Self::accepts(self.head, absolute_caps) {
            return Err(G2gError::CapsMismatch);
        }
        self.check_picture_size()?;
        // On a video link the shape is not in the caps; it comes off each
        // frame's TensorMeta instead.
        if let Some((channels, anchors)) = Self::parse_shape(self.head, absolute_caps) {
            self.channels = channels;
            self.anchors = anchors;
        }
        self.input_caps = Some(absolute_caps.clone());
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        DETECT_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "conf-threshold" | "iou-threshold" => {
                let v = value.as_double().ok_or(PropError::Type)?;
                // Both are scores in [0, 1]; anything else would silently suppress
                // every detection or none.
                if !(0.0..=1.0).contains(&v) {
                    return Err(PropError::Value);
                }
                if name == "conf-threshold" {
                    self.conf_threshold = v as f32;
                } else {
                    self.iou_threshold = v as f32;
                }
                Ok(())
            }
            "tensor-name" => {
                self.tensor_name = String::from(value.as_str().ok_or(PropError::Type)?);
                Ok(())
            }
            "head" => {
                let name = value.as_str().ok_or(PropError::Type)?;
                self.head = Head::from_name(name).ok_or(PropError::Value)?;
                Ok(())
            }
            "picture-width" | "picture-height" => {
                let v = value.as_uint().ok_or(PropError::Type)?;
                let v = u32::try_from(v).map_err(|_| PropError::Value)?;
                if name == "picture-width" {
                    self.picture_w = v;
                } else {
                    self.picture_h = v;
                }
                Ok(())
            }
            "input-width" | "input-height" => {
                let v = value.as_uint().ok_or(PropError::Type)?;
                // The size divides the box coordinates, so zero has no meaning.
                if v == 0 || v > u32::MAX as u64 {
                    return Err(PropError::Value);
                }
                if name == "input-width" {
                    self.input_w = Some(v as f32);
                } else {
                    self.input_h = Some(v as f32);
                }
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "conf-threshold" => Some(PropValue::Double(self.conf_threshold as f64)),
            "iou-threshold" => Some(PropValue::Double(self.iou_threshold as f64)),
            "input-width" => Some(PropValue::Uint(self.normalize_size().0 as u64)),
            "input-height" => Some(PropValue::Uint(self.normalize_size().1 as u64)),
            "tensor-name" => Some(PropValue::Str(self.tensor_name.clone())),
            "head" => Some(PropValue::Str(String::from(self.head.name()))),
            "picture-width" => Some(PropValue::Uint(self.picture_w as u64)),
            "picture-height" => Some(PropValue::Uint(self.picture_h as u64)),
            _ => None,
        }
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
                    // The model output is either the frame itself (tensor link)
                    // or riding on it as meta (video link, `attach-tensor`).
                    if let Some(tensors) = frame.meta.get::<g2g_core::TensorMeta>() {
                        // Unnamed takes the only tensor: with several on the
                        // frame the pipeline has to say which, rather than get
                        // whichever model happened to run first.
                        let tensor = if self.tensor_name.is_empty() {
                            tensors.only()
                        } else {
                            tensors.get(&self.tensor_name)
                        }
                        .ok_or(G2gError::CapsMismatch)?;
                        if tensor.dtype != TensorDType::F32 {
                            return Err(G2gError::CapsMismatch);
                        }
                        let (channels, anchors) = Self::parse_dims(self.head, tensor.shape.dims())
                            .ok_or(G2gError::CapsMismatch)?;
                        let values = Self::read_f32(&tensor.data, channels, anchors)?;
                        self.channels = channels;
                        self.anchors = anchors;
                        let analytics = self.analyze(&values)?;
                        frame.meta.attach(analytics);
                        self.emitted += 1;
                        out.push(PipelinePacket::DataFrame(frame)).await?;
                        return Ok(());
                    }
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let values = Self::read_f32(slice, self.channels, self.anchors)?;
                    let analytics = self.analyze(&values)?;
                    // Attach the graph and forward the (unchanged) tensor frame.
                    frame.meta.attach(analytics);
                    self.emitted += 1;
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // Control packets pass through; a per-input caps change re-parses
                // the shape so the decoder tracks `C` / `A`.
                PipelinePacket::CapsChanged(caps) => {
                    if let Some((ch, a)) = Self::parse_shape(self.head, &caps) {
                        self.channels = ch;
                        self.anchors = a;
                    }
                    self.input_caps = Some(caps.clone());
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
                // PipelinePacket is non_exhaustive: forward variants added
                // since unchanged.
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

/// Settable properties: the two thresholds, the model input size box
/// coordinates are normalized against, the head layout and the letterboxed
/// picture size, so a `gst-launch` line can tune the decoder without the
/// builder.
static DETECT_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "conf-threshold",
        PropKind::Double,
        "minimum class score to emit a detection, 0..1",
    ),
    PropertySpec::new(
        "iou-threshold",
        PropKind::Double,
        "IoU above which a lower-scoring same-class box is suppressed, 0..1",
    ),
    PropertySpec::new(
        "input-width",
        PropKind::Uint,
        "model input width used to normalize box coordinates, default the video link's",
    ),
    PropertySpec::new(
        "input-height",
        PropKind::Uint,
        "model input height used to normalize box coordinates, default the video link's",
    ),
    PropertySpec::new(
        "tensor-name",
        PropKind::Str,
        "which attached tensor to decode when the frame carries several",
    ),
    PropertySpec::new(
        "head",
        PropKind::Str,
        "YOLO export the tensor comes from: boxes, pose keypoints or oriented boxes",
    )
    .with_enum_values(HEAD_NICKS)
    .with_default("detect"),
    PropertySpec::new(
        "picture-width",
        PropKind::Uint,
        "width of the picture the model input was letterboxed from, 0 for a plain stretch",
    )
    .with_default("0"),
    PropertySpec::new(
        "picture-height",
        PropKind::Uint,
        "height of the picture the model input was letterboxed from, 0 for a plain stretch",
    )
    .with_default("0"),
];

/// A tensor caps carries a concrete shape (the model's `[1, 4 + C, A]`), so
/// there is no static superset to advertise; the element negotiates through
/// `caps_constraint` instead, like the other tensor elements.
#[cfg(feature = "launch")]
impl g2g_core::PadTemplates for DetectionPostprocess {
    fn pad_templates() -> Vec<g2g_core::PadTemplate> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_bytes::f32_le_bytes;
    use g2g_core::frame::Frame;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{FrameTiming, PushOutcome, TensorLayout, TensorShape};

    /// Capturing sink that records the AnalyticsMeta of the last frame pushed.
    #[derive(Default)]
    struct MetaSink {
        last: Option<AnalyticsMeta>,
    }
    impl OutputSink for MetaSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
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
            shape: TensorShape::new([1, ch, a]),
            layout: TensorLayout::Nchw,
        }
    }

    /// Channel-major [1, ch, a] tensor from per-channel rows, as LE f32 bytes.
    fn tensor_frame(channels: &[[f32; 3]]) -> Frame {
        let bytes = f32_le_bytes(channels.as_flattened());
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        }
    }

    /// The same `[1, ch, 3]` payload as `tensor_frame`, but riding on an RGBA
    /// video frame as a `TensorMeta` the way `ortinfer attach-tensor=true`
    /// delivers it.
    fn video_frame_with_tensor(channels: &[[f32; 3]], pixels: usize) -> Frame {
        let bytes = f32_le_bytes(channels.as_flattened());
        let mut frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(
                vec![7u8; pixels * 4].into_boxed_slice(),
            )),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        };
        let mut tensors = g2g_core::TensorMeta::new();
        tensors.push(named_tensor("", channels.len() as u32, bytes));
        frame.meta.attach(tensors);
        frame
    }

    fn named_tensor(name: &str, channels: u32, data: Vec<u8>) -> g2g_core::NamedTensor {
        g2g_core::NamedTensor {
            name: String::from(name),
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, channels, 3]),
            layout: TensorLayout::Nchw,
            data,
        }
    }

    /// Channel-major rows as little-endian f32 bytes.
    fn tensor_bytes(channels: &[[f32; 3]]) -> Vec<u8> {
        f32_le_bytes(channels.as_flattened())
    }

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: g2g_core::RawVideoFormat::Rgba8,
            width: g2g_core::Dim::Fixed(w),
            height: g2g_core::Dim::Fixed(h),
            framerate: g2g_core::Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    /// The attach-tensor link: the decoder reads the model output off the
    /// frame's meta, and the video frame reaches the sink with its pixels intact
    /// so an overlay downstream still has a picture to draw on.
    #[tokio::test]
    async fn decodes_a_tensor_carried_as_meta_on_a_video_frame() {
        let mut det = DetectionPostprocess::new(0.5, 0.5).with_input_size(640, 640);
        det.configure_pipeline(&rgba_caps(4, 4)).unwrap();

        let frame = video_frame_with_tensor(
            &[
                [100.0, 105.0, 400.0],
                [100.0, 102.0, 400.0],
                [40.0, 40.0, 40.0],
                [40.0, 40.0, 40.0],
                [0.90, 0.80, 0.05],
                [0.10, 0.05, 0.95],
            ],
            16,
        );

        let mut sink = PixelAndMetaSink::default();
        det.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let meta = sink.meta.expect("frame carries AnalyticsMeta");
        assert_eq!(meta.detections().count(), 2, "decoded from the meta tensor");
        let d0 = meta.detections().find(|d| d.label == 0).unwrap();
        assert!((d0.bbox.x - (80.0 / 640.0)).abs() < 1e-6);
        assert_eq!(
            sink.pixels.expect("frame forwarded"),
            vec![7u8; 64],
            "the picture passed through untouched"
        );
    }

    /// With no size set, boxes normalize against the video link rather than a
    /// square default. A 640x384 model drawn against 640x640 put every box at
    /// 0.6 of its true height, so nothing lined up with what it detected.
    #[tokio::test]
    async fn an_unset_input_size_follows_the_video_link() {
        let mut det = DetectionPostprocess::new(0.5, 0.5);
        det.configure_pipeline(&rgba_caps(640, 384)).unwrap();

        let frame = video_frame_with_tensor(
            &[
                [100.0, 105.0, 400.0],
                [100.0, 102.0, 400.0],
                [40.0, 40.0, 40.0],
                [40.0, 40.0, 40.0],
                [0.90, 0.80, 0.05],
                [0.10, 0.05, 0.95],
            ],
            16,
        );

        let mut sink = PixelAndMetaSink::default();
        det.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let meta = sink.meta.expect("frame carries AnalyticsMeta");
        let d0 = meta.detections().find(|d| d.label == 0).unwrap();
        assert!(
            (d0.bbox.x - (80.0 / 640.0)).abs() < 1e-6,
            "width from the link"
        );
        assert!(
            (d0.bbox.y - (80.0 / 384.0)).abs() < 1e-6,
            "height from the link, not the 640 default"
        );

        // An explicit property still wins over the link.
        let mut pinned = DetectionPostprocess::new(0.5, 0.5).with_input_size(640, 640);
        pinned.configure_pipeline(&rgba_caps(640, 384)).unwrap();
        assert_eq!(
            pinned.get_property("input-height"),
            Some(PropValue::Uint(640))
        );
    }

    /// With two models' outputs on one frame, the decoder takes the one it was
    /// named for; unnamed it refuses rather than picking whichever ran first.
    #[tokio::test]
    async fn several_tensors_on_a_frame_are_selected_by_name() {
        let strong = tensor_bytes(&[
            [100.0, 105.0, 400.0],
            [100.0, 102.0, 400.0],
            [40.0, 40.0, 40.0],
            [40.0, 40.0, 40.0],
            [0.90, 0.80, 0.05],
            [0.10, 0.05, 0.95],
        ]);
        // Same boxes, every score below any sane threshold.
        let weak = tensor_bytes(&[
            [100.0, 105.0, 400.0],
            [100.0, 102.0, 400.0],
            [40.0, 40.0, 40.0],
            [40.0, 40.0, 40.0],
            [0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01],
        ]);
        let build = || {
            let mut frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(
                    vec![7u8; 64].into_boxed_slice(),
                )),
                timing: FrameTiming::default(),
                sequence: 0,
                meta: Default::default(),
            };
            let mut tensors = g2g_core::TensorMeta::new();
            tensors.push(named_tensor("people", 6, strong.clone()));
            tensors.push(named_tensor("ball", 6, weak.clone()));
            frame.meta.attach(tensors);
            frame
        };

        let mut det = DetectionPostprocess::new(0.5, 0.5).with_tensor_name("people");
        det.configure_pipeline(&rgba_caps(4, 4)).unwrap();
        let mut sink = MetaSink::default();
        det.process(PipelinePacket::DataFrame(build()), &mut sink)
            .await
            .unwrap();
        assert_eq!(
            sink.last.expect("analytics").detections().count(),
            2,
            "decoded the tensor it was named for"
        );

        let mut other = DetectionPostprocess::new(0.5, 0.5).with_tensor_name("ball");
        other.configure_pipeline(&rgba_caps(4, 4)).unwrap();
        let mut sink = MetaSink::default();
        other
            .process(PipelinePacket::DataFrame(build()), &mut sink)
            .await
            .unwrap();
        assert_eq!(
            sink.last.expect("analytics").detections().count(),
            0,
            "decoded the other tensor, whose scores are all below threshold"
        );

        let mut unnamed = DetectionPostprocess::new(0.5, 0.5);
        unnamed.configure_pipeline(&rgba_caps(4, 4)).unwrap();
        let mut sink = MetaSink::default();
        assert!(
            matches!(
                unnamed
                    .process(PipelinePacket::DataFrame(build()), &mut sink)
                    .await,
                Err(G2gError::CapsMismatch)
            ),
            "ambiguous without a name"
        );
    }

    /// A video frame with no tensor on it is a mismatch, not a silent pass: the
    /// decoder would otherwise read the pixels as a model output.
    #[tokio::test]
    async fn a_video_frame_without_a_tensor_is_rejected() {
        let mut det = DetectionPostprocess::new(0.5, 0.5);
        det.configure_pipeline(&rgba_caps(4, 4)).unwrap();
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 64].into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        };
        let mut sink = MetaSink::default();
        assert!(matches!(
            det.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await,
            Err(G2gError::CapsMismatch)
        ));
    }

    /// Capturing sink that keeps both the forwarded pixels and the analytics.
    #[derive(Default)]
    struct PixelAndMetaSink {
        pixels: Option<Vec<u8>>,
        meta: Option<AnalyticsMeta>,
    }
    impl OutputSink for PixelAndMetaSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(frame) = packet {
                    if let Some(a) = frame.meta.get::<AnalyticsMeta>() {
                        self.meta = Some(a.clone());
                    }
                    if let Some(s) = frame.domain.as_system_slice() {
                        self.pixels = Some(s.to_vec());
                    }
                }
                Ok(PushOutcome::Accepted)
            })
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
        det.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let meta = sink.last.expect("frame carries AnalyticsMeta");
        let mut got: Vec<(u32, f32)> = meta.detections().map(|d| (d.label, d.confidence)).collect();
        got.sort_by_key(|(l, _)| *l);
        assert_eq!(
            got.len(),
            2,
            "the overlapping same-class box was suppressed"
        );
        assert_eq!(got[0].0, 0, "kept the class-0 detection");
        assert!(
            (got[0].1 - 0.90).abs() < 1e-6,
            "kept the higher-confidence class-0 box"
        );
        assert_eq!(got[1].0, 1, "kept the distinct class-1 detection");

        // Box 0 is normalized: x = (100 - 20)/640, with no picture size set.
        assert_eq!(det.get_property("picture-width"), Some(PropValue::Uint(0)));
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
        det.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        let meta = sink.last.expect("empty AnalyticsMeta still attached");
        assert_eq!(meta.detections().count(), 0);
    }

    #[tokio::test]
    async fn overflowing_shape_rejected_not_panicked() {
        // A model output shape whose channels*anchors*4 overflows usize must fail
        // the length check (CapsMismatch), not panic (debug) or wrap to a small
        // value that admits a bogus frame.
        let mut det = DetectionPostprocess::new(0.5, 0.5);
        det.configure_pipeline(&tensor_caps(u32::MAX, u32::MAX))
            .unwrap();
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 16].into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
            meta: Default::default(),
        };
        let mut sink = MetaSink::default();
        let err = det
            .process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap_err();
        assert!(matches!(err, G2gError::CapsMismatch));
    }

    #[test]
    fn rejects_non_tensor_and_wrong_rank() {
        let det = DetectionPostprocess::new(0.5, 0.5);
        assert!(det.intercept_caps(&tensor_caps(6, 3)).is_ok());
        // Rank-2 tensor (a classifier head) is not a detection output.
        let flat = Caps::Tensor {
            dtype: TensorDType::F32,
            shape: TensorShape::new([1, 10]),
            layout: TensorLayout::Nchw,
        };
        assert!(det.intercept_caps(&flat).is_err());
    }

    /// Tolerance for coordinates that pass through one f32 division.
    const COORDINATE_TOLERANCE: f32 = 1e-6;

    fn assert_close(got: f32, want: f32, what: &str) {
        assert!(
            (got - want).abs() < COORDINATE_TOLERANCE,
            "{what}: got {got}, want {want}"
        );
    }

    /// A 1280x640 picture letterboxed into a 640x640 input: scale 0.5, so the
    /// picture spans 640x320 model pixels with 160 rows of pad above and below.
    fn wide_picture_letterbox() -> PictureMapping {
        PictureMapping::letterbox(640.0, 640.0, 1280.0, 640.0)
    }

    #[test]
    fn a_box_touching_the_letterboxed_picture_edge_maps_to_zero_and_one() {
        let picture = wide_picture_letterbox();
        let whole = picture.bbox(BBox {
            x: 0.0,
            y: 160.0,
            w: 640.0,
            h: 320.0,
        });
        assert_eq!(
            whole,
            BBox {
                x: 0.0,
                y: 0.0,
                w: 1.0,
                h: 1.0
            }
        );
        // A box reaching into the top pad is clipped at the picture's top edge.
        let overhanging = picture.bbox(BBox {
            x: 320.0,
            y: 100.0,
            w: 64.0,
            h: 100.0,
        });
        assert_eq!(overhanging.y, 0.0);
        assert_close(
            overhanging.h,
            40.0 / 320.0,
            "height left inside the picture",
        );
        assert_close(overhanging.x, 0.5, "x untouched by the vertical pad");
    }

    #[tokio::test]
    async fn pose_keypoints_land_where_the_tensor_says_through_the_letterbox() {
        let mut det = DetectionPostprocess::new(0.5, 0.5).with_input_size(640, 640);
        det.set_property("head", PropValue::Str("pose".into()))
            .unwrap();
        det.set_property("picture-width", PropValue::Uint(1280))
            .unwrap();
        det.set_property("picture-height", PropValue::Uint(640))
            .unwrap();
        // [1, 5 + 3 * 2, 3]: box, person score, then two keypoints as x, y,
        // visibility rows. Only anchor 0 scores above threshold.
        det.configure_pipeline(&tensor_caps(11, 3)).unwrap();
        let frame = tensor_frame(&[
            [320.0, 0.0, 0.0],
            [320.0, 0.0, 0.0],
            [64.0, 0.0, 0.0],
            [64.0, 0.0, 0.0],
            [0.9, 0.1, 0.1],
            [160.0, 0.0, 0.0],
            [240.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            [480.0, 0.0, 0.0],
            [400.0, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ]);
        let mut sink = MetaSink::default();
        det.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let meta = sink.last.expect("frame carries AnalyticsMeta");
        assert_eq!(meta.detections().count(), 0, "a pose head emits poses only");
        let poses: Vec<&Pose> = meta.poses().collect();
        assert_eq!(poses.len(), 1);
        let pose = poses[0];
        assert_close(pose.confidence, 0.9, "person score");
        assert_close(pose.bbox.x, 288.0 / 640.0, "box left");
        assert_close(pose.bbox.y, (288.0 - 160.0) / 320.0, "box top");
        assert_close(pose.bbox.w, 64.0 / 640.0, "box width");
        assert_close(pose.bbox.h, 64.0 / 320.0, "box height");
        assert_eq!(pose.keypoints.len(), 2);
        let want = [(0.25, 0.25, 0.8), (0.75, 0.75, 0.3)];
        for (keypoint, (x, y, confidence)) in pose.keypoints.iter().zip(want) {
            assert_close(keypoint.x, x, "keypoint x");
            assert_close(keypoint.y, y, "keypoint y");
            assert_close(keypoint.confidence, confidence, "keypoint visibility");
        }
    }

    #[tokio::test]
    async fn obb_reads_the_angle_and_probiou_keeps_the_higher_of_two_near_twins() {
        let mut det = DetectionPostprocess::new(0.5, 0.5).with_input_size(640, 640);
        det.set_property("head", PropValue::Str("obb".into()))
            .unwrap();
        // [1, 4 + 1 + 1, 3]: anchors 0 and 1 are near-identical rotated boxes,
        // anchor 2 sits far away.
        det.configure_pipeline(&tensor_caps(6, 3)).unwrap();
        let frame = tensor_frame(&[
            [100.0, 101.0, 400.0],
            [100.0, 100.0, 400.0],
            [80.0, 80.0, 40.0],
            [20.0, 20.0, 40.0],
            [0.8, 0.9, 0.7],
            [0.5, 0.52, 1.0],
        ]);
        let mut sink = MetaSink::default();
        det.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let meta = sink.last.expect("frame carries AnalyticsMeta");
        let mut boxes: Vec<OrientedDetection> = meta.oriented_detections().copied().collect();
        boxes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        assert_eq!(boxes.len(), 2, "the lower-scoring twin was suppressed");
        assert_close(boxes[0].confidence, 0.9, "kept the higher twin");
        assert_close(boxes[0].angle_radians, 0.52, "angle from the last channel");
        assert_close(boxes[0].cx, 101.0 / 640.0, "centre x");
        assert_close(boxes[0].w, 80.0 / 640.0, "width");
        assert_close(boxes[1].angle_radians, 1.0, "the far box keeps its angle");
    }

    /// Ultralytics' Fast-NMS drops a box overlapping any higher-scoring one,
    /// even one that was itself dropped: in a chain of three where only
    /// neighbours overlap, only the first survives.
    #[test]
    fn rotated_suppression_drops_a_box_whose_suppressor_was_dropped() {
        let at = |cx: f32, confidence: f32| OrientedDetection {
            cx,
            cy: 100.0,
            w: 80.0,
            h: 20.0,
            angle_radians: 0.0,
            label: 0,
            confidence,
        };
        let (first, second, third) = (at(100.0, 0.9), at(130.0, 0.8), at(160.0, 0.7));
        let iou_threshold = 0.45;
        assert!(probiou(&first, &second) > iou_threshold);
        assert!(probiou(&second, &third) > iou_threshold);
        assert!(probiou(&first, &third) < iou_threshold);
        let kept = suppress_rotated_overlaps(vec![third, first, second], iou_threshold);
        assert_eq!(kept, vec![first]);
    }

    #[test]
    fn head_layouts_reject_widths_the_head_cannot_read() {
        let mut det = DetectionPostprocess::new(0.5, 0.5);
        det.set_property("head", PropValue::Str("pose".into()))
            .unwrap();
        // 9 - 5 = 4 keypoint channels is not a whole number of keypoints.
        assert!(det.intercept_caps(&tensor_caps(9, 3)).is_err());
        assert!(det.configure_pipeline(&tensor_caps(9, 3)).is_err());
        assert!(det.intercept_caps(&tensor_caps(8, 3)).is_ok());

        det.set_property("head", PropValue::Str("obb".into()))
            .unwrap();
        // Box plus angle with no class channel between them.
        assert!(det.intercept_caps(&tensor_caps(5, 3)).is_err());
        assert!(det.intercept_caps(&tensor_caps(6, 3)).is_ok());
    }

    #[tokio::test]
    async fn a_pose_head_rejects_a_meta_tensor_of_the_wrong_width() {
        let mut det = DetectionPostprocess::new(0.5, 0.5);
        det.set_property("head", PropValue::Str("pose".into()))
            .unwrap();
        det.configure_pipeline(&rgba_caps(4, 4)).unwrap();
        // Six channels is a detect layout, one keypoint channel short of a pose.
        let frame = video_frame_with_tensor(&[[0.0; 3]; 6], 16);
        let mut sink = MetaSink::default();
        assert!(matches!(
            det.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await,
            Err(G2gError::CapsMismatch)
        ));
    }

    #[test]
    fn half_a_picture_size_is_refused() {
        let mut det = DetectionPostprocess::new(0.5, 0.5);
        det.set_property("picture-width", PropValue::Uint(1280))
            .unwrap();
        assert!(matches!(
            det.configure_pipeline(&tensor_caps(6, 3)),
            Err(G2gError::NotConfigured)
        ));
    }

    #[test]
    fn every_declared_head_nick_names_a_head() {
        let spec = DETECT_PROPS
            .iter()
            .find(|p| p.name == "head")
            .expect("head is declared");
        let nicks: Vec<&str> = spec.enum_nicks().collect();
        assert_eq!(nicks.len(), 3);
        for nick in nicks {
            assert_eq!(Head::from_name(nick).map(Head::name), Some(nick));
        }
    }
}
