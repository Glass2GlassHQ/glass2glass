//! Instance-segmentation decode (DESIGN.md §5.3): turns the two output tensors of
//! a YOLO `-seg` model into [`Segmentation`] and [`Roi`] analytics nodes, the
//! mask-producing sibling of the box decoder in [`crate::detect`].
//!
//! A `-seg` export (Ultralytics YOLOv8-seg / YOLO11-seg) emits
//! - `[1, 4 + C + M, A]`: the detector's box + class channels, plus `M` mask
//!   coefficients per anchor, and
//! - `[1, M, mh, mw]`: `M` mask prototype planes at a quarter of the input size.
//!
//! An instance's mask is the coefficient-weighted sum of the prototypes through a
//! sigmoid, read over the instance's box, so the two outputs together carry every
//! instance's pixels. This module is the pure-Rust decode of that pair (no
//! inference engine, `analytics` feature only), so an `ort-web` caller in the
//! browser that already holds both outputs as `Float32Array`s can decode without
//! routing tensor frames through an element. [`crate::ortsegment::OrtSegmentation`]
//! is the in-graph producer built on it.
//!
//! Emitted per surviving instance: a `Segmentation` (box, class, confidence, mask)
//! and a `Roi` tightened to the mask's covered samples, related `Segmentation ->
//! Roi` by [`RelationKind::Contains`]. The ROI is the box an encoder or tracker
//! should treat specially: the object's actual pixels, which for anything
//! non-rectangular is smaller than the detector's box.
//!
//! The mask spans exactly the instance's box at prototype resolution, so a consumer
//! places sample `i` of `mask.width()` at `bbox.x + (i + 0.5) / mask.width() *
//! bbox.w` and needs nothing else to draw it.

use g2g_core::{AnalyticsMeta, AnalyticsNode, BBox, Mask, RelationKind, Roi, Segmentation};

use crate::detect::{decode_anchors, suppress_overlaps, AnchorLayout};

/// The geometry a segmentation model's two outputs define: the box output's
/// anchors and class count, and the mask output's prototype grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentationGeometry {
    /// Model input size the box coordinates are normalized against.
    pub input_width: u32,
    pub input_height: u32,
    /// Anchors (`A`) and classes (`C`) of the `[1, 4 + C + M, A]` box output.
    pub anchors: usize,
    pub classes: usize,
    /// Prototype count (`M`) and grid of the `[1, M, mh, mw]` mask output.
    pub prototypes: usize,
    pub proto_width: usize,
    pub proto_height: usize,
}

/// The three cutoffs the decode applies: which anchors become instances, which
/// overlapping same-class boxes survive, and which mask samples count as covered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SegmentationThresholds {
    /// Minimum class score for an anchor to become an instance.
    pub confidence: f32,
    /// IoU above which a lower-scoring same-class box is suppressed.
    pub iou: f32,
    /// Minimum mask probability `[0, 1]` for a sample to count as covered.
    pub coverage: f32,
}

impl Default for SegmentationThresholds {
    /// The Ultralytics defaults for confidence / IoU, and the 0.5 mask cut a
    /// sigmoid output implies.
    fn default() -> Self {
        Self {
            confidence: 0.25,
            iou: 0.45,
            coverage: 0.5,
        }
    }
}

/// Decode both model outputs into the per-frame analytics graph: one
/// `Segmentation` plus its mask-tight `Roi` per surviving instance.
///
/// `box_values` is the flat channel-major `[1, 4 + C + M, A]` output and
/// `proto_values` the flat `[1, M, mh, mw]` prototypes. Values shorter than the
/// geometry describes, or an instance whose thresholded mask covers nothing, yield
/// no nodes rather than an indexing panic or an empty mask downstream.
pub fn decode_instances(
    geometry: &SegmentationGeometry,
    thresholds: &SegmentationThresholds,
    box_values: &[f32],
    proto_values: &[f32],
) -> AnalyticsMeta {
    let mut meta = AnalyticsMeta::new();
    let (proto_w, proto_h) = (geometry.proto_width, geometry.proto_height);
    let plane = match proto_w.checked_mul(proto_h) {
        Some(p) => p,
        None => return meta,
    };
    let protos_len = match plane.checked_mul(geometry.prototypes) {
        Some(n) if n <= proto_values.len() && geometry.prototypes > 0 => n,
        _ => return meta,
    };
    let protos = &proto_values[..protos_len];

    let instances = suppress_overlaps(
        decode_anchors(
            box_values,
            AnchorLayout {
                anchors: geometry.anchors,
                classes: geometry.classes,
                extra: geometry.prototypes,
                input_w: geometry.input_width as f32,
                input_h: geometry.input_height as f32,
            },
            thresholds.confidence,
        ),
        thresholds.iou,
    );

    for (index, instance) in instances.iter().enumerate() {
        let bbox = instance.detection.bbox;
        let mask_w = mask_samples(bbox.w, proto_w);
        let mask_h = mask_samples(bbox.h, proto_h);
        if mask_w == 0 || mask_h == 0 {
            continue;
        }
        let mut data = Vec::with_capacity(mask_w * mask_h);
        let mut covered = CoveredBounds::default();
        for j in 0..mask_h {
            let y = sample_position(bbox.y, bbox.h, j, mask_h, proto_h);
            for i in 0..mask_w {
                let x = sample_position(bbox.x, bbox.w, i, mask_w, proto_w);
                let mut sum = 0.0f32;
                for (p, coefficient) in instance.extra.iter().enumerate() {
                    sum += coefficient * protos[p * plane + y * proto_w + x];
                }
                if sigmoid(sum) >= thresholds.coverage {
                    covered.include(i, j);
                    data.push(u8::MAX);
                } else {
                    data.push(0);
                }
            }
        }
        let Some(mask) = Mask::new(mask_w as u32, mask_h as u32, mask_w as u32, data) else {
            continue;
        };
        // A box whose mask covers nothing is not an instance the model located in
        // the picture, so it contributes no nodes.
        let Some(roi_bbox) = covered.within(&bbox, mask_w, mask_h) else {
            continue;
        };
        let label = instance.detection.label;
        let segmentation = meta.push(AnalyticsNode::Segmentation(Segmentation {
            bbox,
            label,
            confidence: instance.detection.confidence,
            mask,
        }));
        let roi = meta.push(AnalyticsNode::Roi(Roi {
            bbox: roi_bbox,
            // Per-frame instance index, highest confidence first: a tracker is
            // what turns this into an identity across frames.
            id: index as u32,
            label,
        }));
        meta.relate(segmentation, roi, RelationKind::Contains);
    }
    meta
}

/// How many samples a mask spends on one axis: the box's share of the prototype
/// grid, rounded up so a box narrower than a prototype sample still gets one.
/// The mask spans exactly the box, so a consumer places sample `i` of `n` at
/// `box_start + (i + 0.5) / n * box_length` without knowing anything else.
fn mask_samples(box_length: f32, proto_samples: usize) -> usize {
    if box_length.is_nan() || box_length <= 0.0 {
        return 0;
    }
    let wanted = (box_length * proto_samples as f32).ceil();
    (wanted as usize).clamp(1, proto_samples)
}

/// Prototype-grid index the mask's sample `index` of `samples` reads, i.e. that
/// sample's centre mapped into the grid and clamped to it (a YOLO box can hang off
/// the picture).
fn sample_position(
    box_start: f32,
    box_length: f32,
    index: usize,
    samples: usize,
    proto_samples: usize,
) -> usize {
    let center = box_start + (index as f32 + 0.5) / samples as f32 * box_length;
    let scaled = center * proto_samples as f32;
    (scaled.max(0.0) as usize).min(proto_samples - 1)
}

/// Bounding box of the covered samples, in mask-sample coordinates.
#[derive(Debug, Default)]
struct CoveredBounds {
    span: Option<(usize, usize, usize, usize)>,
}

impl CoveredBounds {
    fn include(&mut self, x: usize, y: usize) {
        self.span = Some(match self.span {
            None => (x, y, x, y),
            Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
        });
    }

    /// The covered span as a normalized frame rectangle inside `bbox`, `None` when
    /// nothing was covered. A covered sample spans a full cell of the mask, so the
    /// far edge is inclusive.
    fn within(&self, bbox: &BBox, mask_w: usize, mask_h: usize) -> Option<BBox> {
        let (x0, y0, x1, y1) = self.span?;
        let (w, h) = (mask_w as f32, mask_h as f32);
        Some(BBox {
            x: bbox.x + x0 as f32 / w * bbox.w,
            y: bbox.y + y0 as f32 / h * bbox.h,
            w: (x1 + 1 - x0) as f32 / w * bbox.w,
            h: (y1 + 1 - y0) as f32 / h * bbox.h,
        })
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ANCHORS: usize = 2;
    const CLASSES: usize = 3;
    const PROTOS: usize = 2;
    const GRID: usize = 8;

    fn geometry() -> SegmentationGeometry {
        SegmentationGeometry {
            input_width: 32,
            input_height: 32,
            anchors: ANCHORS,
            classes: CLASSES,
            prototypes: PROTOS,
            proto_width: GRID,
            proto_height: GRID,
        }
    }

    /// Channel-major `[1, 4 + C + M, A]`: one anchor covering the whole picture
    /// with class 1, one below the confidence threshold. `coefficients` are that
    /// first anchor's mask coefficients.
    fn box_output(coefficients: [f32; PROTOS]) -> Vec<f32> {
        let mut channels: Vec<[f32; ANCHORS]> = Vec::from([
            [16.0, 16.0], // cx
            [16.0, 16.0], // cy
            [32.0, 4.0],  // w
            [32.0, 4.0],  // h
            [0.01, 0.01], // class 0
            [0.90, 0.05], // class 1: only the first anchor passes 0.25
            [0.02, 0.02], // class 2
        ]);
        for c in coefficients {
            channels.push([c, 0.0]);
        }
        channels.into_iter().flatten().collect()
    }

    /// Two prototype planes: the first is strongly positive inside a 4x4 square in
    /// the top-left quadrant and strongly negative elsewhere, the second is flat
    /// negative, so a `[1, 0]` coefficient vector selects exactly that square.
    fn proto_output() -> Vec<f32> {
        let mut values = vec![-8.0f32; PROTOS * GRID * GRID];
        for y in 0..4 {
            for x in 0..4 {
                values[y * GRID + x] = 8.0;
            }
        }
        values
    }

    #[test]
    fn mask_follows_the_prototypes_and_roi_tightens_to_it() {
        let meta = decode_instances(
            &geometry(),
            &SegmentationThresholds::default(),
            &box_output([1.0, 0.0]),
            &proto_output(),
        );
        let segmentation = meta.segmentations().next().expect("one instance");
        assert_eq!(segmentation.label, 1);
        let mask = &segmentation.mask;
        assert_eq!(
            (mask.width(), mask.height()),
            (GRID as u32, GRID as u32),
            "the box spans the picture, so the mask spans the prototype grid"
        );
        for y in 0..GRID as u32 {
            for x in 0..GRID as u32 {
                let inside = x < 4 && y < 4;
                assert_eq!(
                    mask.sample(x, y) == Some(u8::MAX),
                    inside,
                    "sample ({x}, {y}) covered = {inside}"
                );
            }
        }

        // The ROI is the mask's covered span, i.e. the top-left half of the box on
        // each axis, not the box itself.
        let roi = meta.rois().next().expect("one ROI");
        assert_eq!(roi.label, 1);
        assert_eq!(roi.id, 0);
        assert!((roi.bbox.w - 0.5).abs() < 1e-6 && (roi.bbox.h - 0.5).abs() < 1e-6);
        assert!(roi.bbox.x.abs() < 1e-6 && roi.bbox.y.abs() < 1e-6);
        assert!(
            roi.bbox.w * roi.bbox.h < segmentation.bbox.w * segmentation.bbox.h,
            "mask-tight ROI is smaller than the detection box"
        );
        assert_eq!(
            meta.relations
                .iter()
                .map(|r| (r.from, r.to, r.kind))
                .collect::<Vec<_>>(),
            Vec::from([(0, 1, RelationKind::Contains)]),
            "the segmentation contains its ROI"
        );
    }

    #[test]
    fn negated_coefficients_invert_the_mask() {
        // Same box and prototypes, opposite coefficient sign: the covered samples
        // are exactly the complement, so the mask is the prototypes' doing and not
        // a function of the box alone.
        let meta = decode_instances(
            &geometry(),
            &SegmentationThresholds::default(),
            &box_output([-1.0, 0.0]),
            &proto_output(),
        );
        let mask = &meta.segmentations().next().expect("one instance").mask;
        for y in 0..GRID as u32 {
            for x in 0..GRID as u32 {
                let inside_square = x < 4 && y < 4;
                assert_eq!(mask.sample(x, y) == Some(u8::MAX), !inside_square);
            }
        }
        let roi = meta.rois().next().expect("one ROI");
        assert!(
            (roi.bbox.w - 1.0).abs() < 1e-6 && (roi.bbox.h - 1.0).abs() < 1e-6,
            "the complement reaches every edge, so its ROI is the whole grid"
        );
    }

    #[test]
    fn an_all_negative_mask_emits_no_nodes() {
        // Coefficients selecting only the flat-negative prototype: every sample
        // falls below the coverage threshold, so the instance is dropped rather
        // than emitted with an empty mask.
        let meta = decode_instances(
            &geometry(),
            &SegmentationThresholds::default(),
            &box_output([0.0, 1.0]),
            &proto_output(),
        );
        assert_eq!(meta.nodes.len(), 0);
    }

    #[test]
    fn short_outputs_yield_no_nodes() {
        let geometry = geometry();
        let thresholds = SegmentationThresholds::default();
        let truncated = box_output([1.0, 0.0]);
        assert_eq!(
            decode_instances(
                &geometry,
                &thresholds,
                &truncated[..truncated.len() - 1],
                &proto_output()
            )
            .nodes
            .len(),
            0
        );
        let protos = proto_output();
        assert_eq!(
            decode_instances(
                &geometry,
                &thresholds,
                &box_output([1.0, 0.0]),
                &protos[..protos.len() - 1]
            )
            .nodes
            .len(),
            0
        );
    }
}
