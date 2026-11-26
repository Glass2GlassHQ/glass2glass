// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Analytics overlay: draws the [`AnalyticsMeta`] carried on a
//! frame onto its raw RGBA8 pixels, the visible end of a detector -> overlay
//! pipeline. The `cairooverlay` / `ovrenderhud` analog for ML analytics.
//!
//! Four shapes: a detection box as a solid outline in its class colour, an
//! instance segmentation as a translucent fill of its mask, a region of
//! interest as a dashed outline in the colour of the mask that contains it, and
//! a pose as its box plus a dot on each keypoint, joined into limbs for the 17
//! COCO body keypoints. `keypoint-threshold` hides keypoints the model scores as
//! not visible.
//!
//! `show-label` / `show-track` / `show-score` add a caption bar above each
//! detection box with its class name, tracking id and confidence. A node stores
//! a `u32` label id, so the name comes from the meta's shared `class_names`
//! table; a producer that publishes none leaves `show-label` with nothing to
//! draw. A classification with no `Classifies` relation labels the whole frame,
//! and the same properties caption it at the frame's top left.
//!
//! `show-trail` draws where each tracked object has been, as a polyline through
//! the bottom edge of its recent boxes that fades out towards the oldest point.
//! It is the one thing here that remembers anything between frames, so a trail
//! outlives a few missed detections and is dropped once its track stops coming.
//! A detection result held on several frames (`analyticsjoin hold-last`) adds
//! one point, not one per frame.
//!
//! Pairs with the metadata-through-fan-out path: a `decode -> tee ->
//! {detect, video} -> overlay -> display` diamond runs the detector on one branch
//! and carries its `AnalyticsMeta` (shared by Arc) onto the video branch, where
//! this element renders them onto the picture that actually reaches the sink.
//!
//! CPU, `no_std` baseline like the other raw-video transforms. Input and output
//! are both RGBA8 at the negotiated geometry (put a `VideoConvert` upstream of a
//! non-RGBA source); the element is an identity transform on the pixels apart from
//! the shapes it paints. Boxes are normalized `[0,1]` in the metadata and a mask
//! spans exactly its instance's box, so this works at any frame size without an
//! upstream coordinate rewrite. A frame with no `AnalyticsMeta` passes through
//! untouched. The Vello GPU backend is the separate `vello-overlay` feature, and
//! draws the same captions in a shaped system font.
//!
//! When a downstream sink asks for `OverlayCompositionMeta` (`waylandsink` or
//! `wgpusink` with `render-overlays=true`), the shapes go on the frame as one
//! picture-sized bitmap and the video pixels stay untouched.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use g2g_core::memory::{DomainSet, MemoryDomainKind};
use g2g_core::meta::PlaneLayout;
use g2g_core::{
    AnalyticsMeta, AnalyticsNode, AsyncElement, BBox, Caps, CapsConstraint, CapsSet,
    Classification, Colorimetry, ConfigureOutcome, Dim, ElementMetadata, G2gError, Keypoint,
    MemoryDomain, ObjectDetection, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, RawVideoFormat, RelationKind, Roi, Segmentation,
};

use crate::bitmapfont::{glyph, GLYPH_ADVANCE, GLYPH_HEIGHT};
#[cfg(test)]
use crate::paint::blend_u8;
use crate::paint::{blend_layer_into_nv12, blend_px, Canvas};
use crate::yuvmatrix::YuvRgbMatrix;

const DRAWN_FORMATS: [RawVideoFormat; 2] = [RawVideoFormat::Rgba8, RawVideoFormat::Nv12];

/// Default mask fill alpha: faint enough that the picture reads through the fill.
pub(crate) const MASK_ALPHA_DEFAULT: u8 = 96;

/// Dash period of an ROI outline, in canvas pixels: `ROI_DASH_PX` on, then the
/// same off. Measured in absolute canvas coordinates, so the four sides of one
/// rectangle keep a common phase.
pub(crate) const ROI_DASH_PX: i32 = 6;

// coco's limbs as zero-based index pairs into its 17 body keypoints, the order yolo -pose models emit
pub(crate) const COCO_SKELETON: [(usize, usize); 19] = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
];

const COCO_KEYPOINTS: usize = 17;

pub(crate) const KEYPOINT_THRESHOLD_DEFAULT: f32 = 0.25;

pub(crate) const KEYPOINT_THRESHOLD_PROPERTY: PropertySpec = PropertySpec::new(
    "keypoint-threshold",
    PropKind::Double,
    "lowest keypoint visibility score a pose keypoint is drawn at (0..1)",
)
.with_range("0", "1")
.with_default("0.25");

/// How many past positions a trail keeps by default, i.e. about a second of
/// movement at 30 fps.
const TRAIL_LENGTH_DEFAULT: usize = 30;

/// A trail is dropped once its track has gone this many frames without a
/// detection, so an object that leaves the scene stops being drawn.
const TRAIL_TTL_FRAMES: u64 = 30;

/// The faintest a trail segment is blended at, so its oldest end stays visible
/// rather than fading to nothing.
const TRAIL_MIN_ALPHA: u32 = 40;

/// Draws the detection boxes, poses, segmentation masks and regions of interest of an
/// attached [`AnalyticsMeta`] onto an RGBA8 frame. Outline thickness and mask fill
/// alpha are configurable; colours come from a fixed palette, indexed by class
/// label for a detection box and by instance for a mask and its ROI.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::analyticsoverlay::AnalyticsOverlay;
///
/// let overlay = AnalyticsOverlay::new()
///     .with_thickness(3)
///     .with_mask_alpha(120)
///     .with_score(true);
/// ```
#[derive(Debug)]
pub struct AnalyticsOverlay {
    width: u32,
    height: u32,
    format: RawVideoFormat,
    matrix: YuvRgbMatrix,
    scratch_rgba: Vec<u8>,
    /// Outline thickness in pixels (>= 1).
    thickness: u32,
    /// Alpha the mask fill is blended at (0 = invisible, 255 = opaque).
    mask_alpha: u8,
    keypoint_threshold: f32,
    caption: CaptionParts,
    /// Draw the path each tracked object took.
    show_trail: bool,
    /// How many past positions a trail keeps.
    trail_length: usize,
    /// The path of each tracked object, keyed by tracking id.
    trails: BTreeMap<u64, Trail>,
    configured: bool,
    drawn: u64,
    previous_analytics: Option<AnalyticsMeta>,
    #[cfg(feature = "metadata")]
    downstream_renders: bool,
    #[cfg(feature = "metadata")]
    cached_composition: Option<g2g_core::OverlayCompositionMeta>,
}

/// The recent path of one tracked object: normalized bottom-centre points of its
/// boxes, oldest first, and the frame the newest one arrived on.
#[derive(Debug, Default)]
struct Trail {
    points: VecDeque<(f32, f32)>,
    last_seen: u64,
}

/// A detection box to outline, with the id of the tracking node related to it
/// when the producer wired one (`Detection -Tracks-> Tracking`).
#[derive(Debug)]
pub(crate) struct PaintedDetection {
    pub detection: ObjectDetection,
    pub track_id: Option<u64>,
}

#[derive(Debug)]
pub(crate) struct PaintedPose {
    pub keypoints: Vec<Keypoint>,
    pub label: u32,
}

impl PaintedPose {
    pub(crate) fn visible_keypoints(
        &self,
        threshold: f32,
    ) -> impl Iterator<Item = (f32, f32)> + '_ {
        self.keypoints
            .iter()
            .filter(move |k| k.confidence >= threshold)
            .map(|k| (k.x, k.y))
    }

    pub(crate) fn visible_limbs(
        &self,
        threshold: f32,
    ) -> impl Iterator<Item = ((f32, f32), (f32, f32))> + '_ {
        let limbs: &[(usize, usize)] = if self.keypoints.len() == COCO_KEYPOINTS {
            &COCO_SKELETON
        } else {
            &[]
        };
        limbs.iter().filter_map(move |&(from, to)| {
            let (from, to) = (self.keypoints[from], self.keypoints[to]);
            (from.confidence >= threshold && to.confidence >= threshold)
                .then_some(((from.x, from.y), (to.x, to.y)))
        })
    }
}

/// A segmentation mask to fill, with the palette slot of the instance it belongs
/// to.
#[derive(Debug)]
pub(crate) struct PaintedMask {
    pub segmentation: Segmentation,
    pub palette_index: u32,
}

/// A region of interest to outline, in the palette slot of the mask that contains
/// it (or of its own `id` when it stands alone), so an instance's mask and ROI are
/// visually paired.
#[derive(Debug)]
pub(crate) struct PaintedRoi {
    pub roi: Roi,
    pub palette_index: u32,
}

/// Everything an overlay draws for one frame, copied out of the frame's
/// [`AnalyticsMeta`] so the immutable meta borrow ends before the pixels (or the
/// scene) are written. Shared by the CPU and Vello backends, so both resolve the
/// same palette slots and the same containment pairing.
#[derive(Debug, Default)]
pub(crate) struct AnalyticsShapes {
    pub detections: Vec<PaintedDetection>,
    pub poses: Vec<PaintedPose>,
    pub masks: Vec<PaintedMask>,
    pub rois: Vec<PaintedRoi>,
    pub frame_classifications: Vec<Classification>,
    /// The meta's class-name table, carried so a caption can name a label
    /// without the shapes borrowing the meta.
    pub class_names: Option<alloc::sync::Arc<[alloc::boxed::Box<str>]>>,
}

impl AnalyticsShapes {
    /// Read the drawable nodes out of `meta`, numbering the segmentations for the
    /// palette and giving every ROI the slot of the segmentation that contains it.
    pub(crate) fn collect(meta: &AnalyticsMeta) -> Self {
        let mut shapes = Self {
            class_names: meta.class_names.clone(),
            ..Default::default()
        };
        let mut slot_of_node: Vec<Option<u32>> = alloc::vec![None; meta.nodes.len()];
        for (index, node) in meta.nodes.iter().enumerate() {
            match node {
                AnalyticsNode::Detection(detection) => shapes.detections.push(PaintedDetection {
                    detection: *detection,
                    track_id: meta.tracking_id(index),
                }),
                AnalyticsNode::Pose(pose) => {
                    shapes.detections.push(PaintedDetection {
                        detection: ObjectDetection {
                            bbox: pose.bbox,
                            label: pose.label,
                            confidence: pose.confidence,
                        },
                        track_id: meta.tracking_id(index),
                    });
                    shapes.poses.push(PaintedPose {
                        keypoints: pose.keypoints.clone(),
                        label: pose.label,
                    });
                }
                AnalyticsNode::Classification(classification)
                    if !meta
                        .relations
                        .iter()
                        .any(|r| r.to == index && r.kind == RelationKind::Classifies) =>
                {
                    shapes.frame_classifications.push(*classification)
                }
                AnalyticsNode::Segmentation(segmentation) => {
                    let palette_index = shapes.masks.len() as u32;
                    slot_of_node[index] = Some(palette_index);
                    shapes.masks.push(PaintedMask {
                        segmentation: segmentation.clone(),
                        palette_index,
                    });
                }
                _ => {}
            }
        }
        // A second pass, because a containment relation may name a segmentation
        // that appears after its ROI in the node list.
        for (index, node) in meta.nodes.iter().enumerate() {
            let AnalyticsNode::Roi(roi) = node else {
                continue;
            };
            let container = meta
                .relations
                .iter()
                .find(|r| r.to == index && r.kind == RelationKind::Contains)
                .and_then(|r| slot_of_node.get(r.from).copied().flatten());
            shapes.rois.push(PaintedRoi {
                roi: *roi,
                palette_index: container.unwrap_or(roi.id),
            });
        }
        shapes
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.detections.is_empty()
            && self.masks.is_empty()
            && self.rois.is_empty()
            && self.frame_classifications.is_empty()
    }

    pub(crate) fn frame_caption(&self, parts: &CaptionParts) -> Option<(String, u32)> {
        let first = self.frame_classifications.first()?;
        let captions: Vec<String> = self
            .frame_classifications
            .iter()
            .filter_map(|c| parts.text(self.class_name(c.label), None, c.confidence))
            .collect();
        (!captions.is_empty()).then(|| (captions.join(" "), first.label))
    }

    pub(crate) fn class_name(&self, label: u32) -> Option<&str> {
        let names = self.class_names.as_ref()?;
        names.get(label as usize).map(|name| &**name)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CaptionParts {
    pub label: bool,
    pub track: bool,
    pub score: bool,
}

impl CaptionParts {
    /// The caption for one detection or classification, or `None` when no part
    /// is enabled (or a tracking id was asked for and the producer wired none).
    pub(crate) fn text(
        &self,
        name: Option<&str>,
        track_id: Option<u64>,
        confidence: f32,
    ) -> Option<String> {
        let mut caption = String::new();
        let mut add = |part: &str| {
            if !caption.is_empty() {
                caption.push(' ');
            }
            caption.push_str(part);
        };
        if let Some(name) = name.filter(|_| self.label) {
            add(name);
        }
        if let Some(id) = track_id.filter(|_| self.track) {
            add(&format!("ID:{id}"));
        }
        if self.score {
            add(&score_text(confidence));
        }
        (!caption.is_empty()).then_some(caption)
    }
}

pub(crate) const SHOW_LABEL_PROPERTY: PropertySpec = PropertySpec::new(
    "show-label",
    PropKind::Bool,
    "draw each detection's class name above its box, and a frame classification's at the top left",
)
.with_default("false");

pub(crate) const SHOW_SCORE_PROPERTY: PropertySpec = PropertySpec::new(
    "show-score",
    PropKind::Bool,
    "draw each detection's confidence above its box, and a frame classification's at the top left",
)
.with_default("false");

pub(crate) const SHOW_TRACK_PROPERTY: PropertySpec = PropertySpec::new(
    "show-track",
    PropKind::Bool,
    "draw each detection's tracking id above its box",
)
.with_default("false");

pub(crate) const WHOLE_FRAME: BBox = BBox {
    x: 0.0,
    y: 0.0,
    w: 1.0,
    h: 1.0,
};

pub(crate) fn caption_bar_origin(
    (x0, y0): (i32, i32),
    (bar_w, bar_h): (i32, i32),
    frame_w: i32,
) -> (i32, i32) {
    let bar_x = x0.min((frame_w - bar_w).max(0)).max(0);
    let bar_y = if y0 - bar_h >= 0 { y0 - bar_h } else { y0 };
    (bar_x, bar_y)
}

impl Default for AnalyticsOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalyticsOverlay {
    /// A new overlay with a 2px outline. Geometry is set at negotiation.
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            format: RawVideoFormat::Rgba8,
            matrix: YuvRgbMatrix::new(Colorimetry::UNKNOWN),
            scratch_rgba: Vec::new(),
            thickness: 2,
            mask_alpha: MASK_ALPHA_DEFAULT,
            keypoint_threshold: KEYPOINT_THRESHOLD_DEFAULT,
            caption: CaptionParts::default(),
            show_trail: false,
            trail_length: TRAIL_LENGTH_DEFAULT,
            trails: BTreeMap::new(),
            configured: false,
            drawn: 0,
            previous_analytics: None,
            #[cfg(feature = "metadata")]
            downstream_renders: false,
            #[cfg(feature = "metadata")]
            cached_composition: None,
        }
    }

    /// Draw each detection's class name above its box.
    pub fn with_label(mut self, show: bool) -> Self {
        self.caption.label = show;
        self
    }

    /// Draw each detection's confidence above its box.
    pub fn with_score(mut self, show: bool) -> Self {
        self.caption.score = show;
        self
    }

    /// Draw each detection's tracking id above its box.
    pub fn with_track(mut self, show: bool) -> Self {
        self.caption.track = show;
        self
    }

    /// Draw the path each tracked object took.
    pub fn with_trail(mut self, show: bool) -> Self {
        self.show_trail = show;
        self
    }

    /// Set how many past positions a trail keeps (clamped to at least 2, since a
    /// segment needs both ends).
    pub fn with_trail_length(mut self, points: usize) -> Self {
        self.trail_length = points.max(2);
        self
    }

    /// Set the box outline thickness in pixels (clamped to at least 1).
    pub fn with_thickness(mut self, px: u32) -> Self {
        self.thickness = px.max(1);
        self
    }

    /// Set the alpha the segmentation mask fill is blended at (0..=255).
    pub fn with_mask_alpha(mut self, alpha: u8) -> Self {
        self.mask_alpha = alpha;
        self
    }

    pub fn with_keypoint_threshold(mut self, threshold: f32) -> Self {
        self.keypoint_threshold = threshold;
        self
    }

    /// Count of frames processed (whether or not they carried detections).
    pub fn drawn_count(&self) -> u64 {
        self.drawn
    }

    fn configure_picture(&mut self, caps: &Caps) -> Result<(), G2gError> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            colorimetry,
            ..
        } = caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !DRAWN_FORMATS.contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        self.width = *width;
        self.height = *height;
        self.format = *format;
        self.matrix = YuvRgbMatrix::new(*colorimetry);
        #[cfg(feature = "metadata")]
        {
            self.cached_composition = None;
        }
        Ok(())
    }

    fn accepts(caps: &Caps) -> bool {
        matches!(caps, Caps::RawVideo { format, .. } if DRAWN_FORMATS.contains(format))
    }

    /// Paint every shape onto the frame whose planes `layout` places in `buf`.
    fn paint(
        &mut self,
        buf: &mut [u8],
        layout: &PlaneLayout,
        shapes: &AnalyticsShapes,
    ) -> Result<(), G2gError> {
        let (width, height) = (self.width as usize, self.height as usize);
        let mut scratch = core::mem::take(&mut self.scratch_rgba);
        scratch.clear();
        let painted = match self.format {
            RawVideoFormat::Rgba8 => {
                self.paint_rgba(buf, layout, shapes, (width, height), &mut scratch)
            }
            _ => {
                scratch.resize(width * height * 4, 0);
                self.render(&mut scratch, shapes);
                blend_layer_into_nv12(&scratch, buf, layout, (width, height), &self.matrix)
            }
        };
        self.scratch_rgba = scratch;
        painted
    }

    // `render` draws onto tight rows, so padded ones are drawn in `scratch` and copied back
    fn paint_rgba(
        &self,
        buf: &mut [u8],
        layout: &PlaneLayout,
        shapes: &AnalyticsShapes,
        (width, height): (usize, usize),
        scratch: &mut Vec<u8>,
    ) -> Result<(), G2gError> {
        let row_bytes = width * 4;
        let rows = |row| {
            layout
                .row_range(0, row, row_bytes)
                .ok_or(G2gError::CapsMismatch)
        };
        let last_row = rows(height.checked_sub(1).ok_or(G2gError::CapsMismatch)?)?;
        let plane = layout.plane(0).ok_or(G2gError::CapsMismatch)?;
        if plane.stride == row_bytes {
            let frame = buf
                .get_mut(plane.offset..last_row.end)
                .ok_or(G2gError::CapsMismatch)?;
            self.render(frame, shapes);
            return Ok(());
        }
        for row in 0..height {
            scratch.extend_from_slice(buf.get(rows(row)?).ok_or(G2gError::CapsMismatch)?);
        }
        self.render(scratch, shapes);
        for (row, pixels) in scratch.chunks_exact(row_bytes).enumerate() {
            buf[rows(row)?].copy_from_slice(pixels);
        }
        Ok(())
    }

    #[cfg(feature = "metadata")]
    fn attach_composition(
        &mut self,
        frame: &mut g2g_core::Frame,
        shapes: &AnalyticsShapes,
        unchanged: bool,
    ) -> Result<(), G2gError> {
        let composition = match &self.cached_composition {
            Some(composition) if unchanged => composition.clone(),
            _ => {
                let composition = self.render_composition(shapes)?;
                self.cached_composition = Some(composition.clone());
                composition
            }
        };
        frame.meta.attach(composition);
        Ok(())
    }

    #[cfg(feature = "metadata")]
    fn render_composition(
        &self,
        shapes: &AnalyticsShapes,
    ) -> Result<g2g_core::OverlayCompositionMeta, G2gError> {
        let mut layer = alloc::vec![0; self.width as usize * self.height as usize * 4];
        self.render(&mut layer, shapes);
        let whole_picture = g2g_core::OverlayPlacement {
            x: 0,
            y: 0,
            width: self.width,
            height: self.height,
        };
        let rectangle =
            g2g_core::OverlayRectangle::new(whole_picture, self.width, self.height, layer.into())
                .ok_or(G2gError::CapsMismatch)?;
        Ok(g2g_core::OverlayCompositionMeta::new(alloc::vec![
            rectangle
        ]))
    }

    /// Paint every shape onto the RGBA8 `buf` of `self.width` x `self.height`.
    /// Mask fills go down first, so a box or ROI edge stays readable over one.
    fn render(&self, buf: &mut [u8], shapes: &AnalyticsShapes) {
        if self.show_trail {
            self.draw_trails(buf);
        }
        for mask in &shapes.masks {
            self.fill_mask(buf, mask);
        }
        for pose in &shapes.poses {
            self.draw_pose(buf, pose);
        }
        for painted in &shapes.detections {
            let color = palette_color(painted.detection.label);
            self.outline(buf, painted.detection.bbox, color, false);
            let name = shapes.class_name(painted.detection.label);
            let caption = self
                .caption
                .text(name, painted.track_id, painted.detection.confidence);
            if let Some(text) = caption {
                self.draw_caption(buf, painted.detection.bbox, &text, color);
            }
        }
        if let Some((text, label)) = shapes.frame_caption(&self.caption) {
            self.draw_caption(buf, WHOLE_FRAME, &text, palette_color(label));
        }
        for roi in &shapes.rois {
            self.outline(buf, roi.roi.bbox, palette_color(roi.palette_index), true);
        }
    }

    /// Extend each tracked object's path with this frame's position, and forget
    /// the tracks that have stopped arriving. `drawn` counts every frame, so it
    /// is what a trail's age is measured against. A `repeated` detection result
    /// keeps its tracks alive without adding a point. Returns whether any trail
    /// changed.
    fn record_trails(&mut self, shapes: &AnalyticsShapes, repeated: bool) -> bool {
        let mut changed = false;
        for painted in &shapes.detections {
            let Some(id) = painted.track_id else {
                continue;
            };
            if repeated {
                if let Some(trail) = self.trails.get_mut(&id) {
                    trail.last_seen = self.drawn;
                }
                continue;
            }
            changed = true;
            let bbox = painted.detection.bbox;
            let trail = self.trails.entry(id).or_default();
            trail
                .points
                .push_back((bbox.x + bbox.w / 2.0, bbox.y + bbox.h));
            while trail.points.len() > self.trail_length {
                trail.points.pop_front();
            }
            trail.last_seen = self.drawn;
        }
        let cutoff = self.drawn.saturating_sub(TRAIL_TTL_FRAMES);
        let tracked = self.trails.len();
        self.trails.retain(|_, trail| trail.last_seen >= cutoff);
        changed || self.trails.len() != tracked
    }

    fn draw_pose(&self, buf: &mut [u8], pose: &PaintedPose) {
        let color = palette_color(pose.label);
        let half = self.thickness as i32 / 2;
        for (from, to) in pose.visible_limbs(self.keypoint_threshold) {
            self.segment(
                buf,
                self.canvas_point(from),
                self.canvas_point(to),
                half,
                color,
                255,
            );
        }
        let ink = contrast_ink(color);
        for point in pose.visible_keypoints(self.keypoint_threshold) {
            let (x, y) = self.canvas_point(point);
            self.dot(buf, x, y, self.thickness as i32, ink, 255);
        }
    }

    fn canvas_point(&self, (x, y): (f32, f32)) -> (i32, i32) {
        (
            (x * self.width as f32 + 0.5) as i32,
            (y * self.height as f32 + 0.5) as i32,
        )
    }

    /// Stroke every trail, each in the palette slot of its tracking id so two
    /// objects of the same class still read apart, fading towards its old end.
    fn draw_trails(&self, buf: &mut [u8]) {
        let half = self.thickness as i32 / 2;
        for (id, trail) in &self.trails {
            let color = palette_color(*id as u32);
            let count = trail.points.len() as u32;
            let mut previous = None;
            for (index, point) in trail.points.iter().enumerate() {
                let to = self.canvas_point(*point);
                if let Some(from) = previous {
                    let ramp = (255 - TRAIL_MIN_ALPHA) * (index as u32 + 1) / count;
                    let alpha = (TRAIL_MIN_ALPHA + ramp) as u8;
                    self.segment(buf, from, to, half, color, alpha);
                }
                previous = Some(to);
            }
        }
    }

    /// Blend a straight run of `half`-radius squares between two canvas points
    /// (Bresenham, integer only for the `no_std` baseline).
    fn segment(
        &self,
        buf: &mut [u8],
        from: (i32, i32),
        to: (i32, i32),
        half: i32,
        color: [u8; 4],
        alpha: u8,
    ) {
        let (mut x, mut y) = from;
        let dx = (to.0 - x).abs();
        let dy = -(to.1 - y).abs();
        let sx = if x < to.0 { 1 } else { -1 };
        let sy = if y < to.1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.dot(buf, x, y, half, color, alpha);
            if x == to.0 && y == to.1 {
                return;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Blend a square `2 * half + 1` pixels wide centred on a canvas point,
    /// clipped to the canvas.
    fn dot(&self, buf: &mut [u8], x: i32, y: i32, half: i32, color: [u8; 4], alpha: u8) {
        let side = 2 * half + 1;
        let mut canvas = Canvas {
            pixels: buf,
            width: self.width as i32,
            height: self.height as i32,
        };
        canvas.fill_rect_alpha(x - half, y - half, side, side, color, alpha);
    }

    /// One source font pixel per this many output pixels, from the frame height
    /// so a caption stays readable across resolutions.
    fn text_scale(&self) -> i32 {
        (self.height / 240).max(1) as i32
    }

    /// Draw `text` in a filled bar the colour of its box, sitting on the box's
    /// top edge, or just inside it when the box starts at the top of the frame.
    fn draw_caption(&self, buf: &mut [u8], bbox: BBox, text: &str, color: [u8; 4]) {
        let w = self.width as i32;
        let h = self.height as i32;
        let Some((x0, y0, _, _)) = pixel_rect(bbox, w, h) else {
            return;
        };
        let scale = self.text_scale();
        let pad = scale;
        let cell_w = GLYPH_ADVANCE as i32 * scale;
        let bar_w = text.chars().count() as i32 * cell_w + 2 * pad;
        let bar_h = GLYPH_HEIGHT as i32 * scale + 2 * pad;
        let (bar_x, bar_y) = caption_bar_origin((x0, y0), (bar_w, bar_h), w);
        let mut canvas = Canvas {
            pixels: buf,
            width: w,
            height: h,
        };
        canvas.fill_rect(bar_x, bar_y, bar_w, bar_h, color);
        let ink = contrast_ink(color);
        let mut gx = bar_x + pad;
        for c in text.chars() {
            canvas.blit_glyph(gx, bar_y + pad, scale, glyph(c), ink);
            gx += cell_w;
        }
    }

    /// Stroke the outline of a normalized box, `dashed` for an ROI so it reads
    /// apart from a detection box.
    fn outline(&self, buf: &mut [u8], bbox: BBox, color: [u8; 4], dashed: bool) {
        let w = self.width as i32;
        let h = self.height as i32;
        let t = self.thickness as i32;
        let Some((x0, y0, x1, y1)) = pixel_rect(bbox, w, h) else {
            return;
        };
        let paint = SpanPaint { color, dashed };
        // Four outline bands, each `t` pixels thick, clipped to the canvas.
        for dy in 0..t {
            hspan(buf, w, h, x0, x1, y0 + dy, paint);
            hspan(buf, w, h, x0, x1, y1 - dy, paint);
        }
        for dx in 0..t {
            vspan(buf, w, h, y0, y1, x0 + dx, paint);
            vspan(buf, w, h, y0, y1, x1 - dx, paint);
        }
    }

    /// Blend an instance's mask over its box as a translucent fill. The mask spans
    /// exactly the box (its grid is the model's, not the frame's), so a pixel reads
    /// the sample covering its share of the box.
    fn fill_mask(&self, buf: &mut [u8], painted: &PaintedMask) {
        let w = self.width as i32;
        let h = self.height as i32;
        let mask = &painted.segmentation.mask;
        let Some((x0, y0, x1, y1)) = pixel_rect(painted.segmentation.bbox, w, h) else {
            return;
        };
        let (span_w, span_h) = ((x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32);
        let color = palette_color(painted.palette_index);
        for y in y0.max(0)..=y1.min(h - 1) {
            let j = (y - y0) as u32 * mask.height() / span_h;
            for x in x0.max(0)..=x1.min(w - 1) {
                let i = (x - x0) as u32 * mask.width() / span_w;
                let coverage = mask.sample(i, j).unwrap_or(0) as u32;
                if coverage == 0 {
                    continue;
                }
                let alpha = (coverage * self.mask_alpha as u32 / 255) as u8;
                blend_px(buf, ((y * w + x) * 4) as usize, color, alpha);
            }
        }
    }
}

/// The pixel rectangle (inclusive) a normalized box covers on a `w` x `h` canvas,
/// or `None` when it collapses to nothing. `+ 0.5` rounds without the std-only
/// `f32::round` (the `no_std` baseline has no float intrinsics).
fn pixel_rect(bbox: BBox, w: i32, h: i32) -> Option<(i32, i32, i32, i32)> {
    let x0 = (bbox.x * w as f32 + 0.5) as i32;
    let y0 = (bbox.y * h as f32 + 0.5) as i32;
    let x1 = ((bbox.x + bbox.w) * w as f32 + 0.5) as i32 - 1;
    let y1 = ((bbox.y + bbox.h) * h as f32 + 0.5) as i32 - 1;
    if x1 < x0 || y1 < y0 {
        return None;
    }
    Some((x0, y0, x1, y1))
}

/// How one outline band paints: its colour, and whether it dashes.
#[derive(Debug, Clone, Copy)]
struct SpanPaint {
    color: [u8; 4],
    dashed: bool,
}

impl SpanPaint {
    /// Whether the band paints at this canvas coordinate.
    fn paints_at(&self, coordinate: i32) -> bool {
        !self.dashed || (coordinate / ROI_DASH_PX) % 2 == 0
    }
}

/// Blend a horizontal run `x0..=x1` at row `y`, clipped to the canvas.
fn hspan(buf: &mut [u8], w: i32, h: i32, x0: i32, x1: i32, y: i32, paint: SpanPaint) {
    if y < 0 || y >= h {
        return;
    }
    let xs = x0.max(0);
    let xe = x1.min(w - 1);
    for x in xs..=xe {
        if paint.paints_at(x) {
            blend_px(buf, ((y * w + x) * 4) as usize, paint.color, 255);
        }
    }
}

/// Blend a vertical run `y0..=y1` at column `x`, clipped to the canvas.
fn vspan(buf: &mut [u8], w: i32, h: i32, y0: i32, y1: i32, x: i32, paint: SpanPaint) {
    if x < 0 || x >= w {
        return;
    }
    let ys = y0.max(0);
    let ye = y1.min(h - 1);
    for y in ys..=ye {
        if paint.paints_at(y) {
            blend_px(buf, ((y * w + x) * 4) as usize, paint.color, 255);
        }
    }
}

/// A detection's confidence as a fixed two-decimal string (`0.92`). Built from
/// integer parts: the `no_std` baseline has no float formatting to lean on, and
/// `+ 0.5` rounds without `f32::round`.
fn score_text(confidence: f32) -> String {
    let hundredths = (confidence.clamp(0.0, 1.0) * 100.0 + 0.5) as u32;
    format!("{}.{:02}", hundredths / 100, hundredths % 100)
}

/// Black or white, whichever reads on `background`. Rec. 601 luma, since the
/// palette is sRGB and the threshold only has to separate light from dark.
pub(crate) fn contrast_ink(background: [u8; 4]) -> [u8; 4] {
    let luma =
        (background[0] as u32 * 299 + background[1] as u32 * 587 + background[2] as u32 * 114)
            / 1000;
    if luma > 140 {
        [0x00, 0x00, 0x00, 0xFF]
    } else {
        [0xFF, 0xFF, 0xFF, 0xFF]
    }
}

/// A fixed, opaque RGB palette so adjacent slots are visually distinct: a
/// detection box indexes it by class label, a mask and its ROI by instance.
/// Cycles for indices beyond the palette length. Shared with the Vello overlay
/// backend (`vello-overlay`) so both draw a slot the same colour.
pub(crate) fn palette_rgb(index: u32) -> [u8; 3] {
    const PALETTE: [[u8; 3]; 8] = [
        [0xFF, 0x3B, 0x30], // red
        [0x34, 0xC7, 0x59], // green
        [0x00, 0x7A, 0xFF], // blue
        [0xFF, 0xCC, 0x00], // yellow
        [0xAF, 0x52, 0xDE], // purple
        [0xFF, 0x95, 0x00], // orange
        [0x5A, 0xC8, 0xFA], // cyan
        [0xFF, 0x2D, 0x95], // magenta
    ];
    PALETTE[(index as usize) % PALETTE.len()]
}

/// The opaque RGBA outline colour of a palette slot.
pub(crate) fn palette_color(index: u32) -> [u8; 4] {
    let c = palette_rgb(index);
    [c[0], c[1], c[2], 0xFF]
}

impl AsyncElement for AnalyticsOverlay {
    fn input_domains(&self) -> DomainSet {
        let system = DomainSet::only(MemoryDomainKind::System);
        // linux draws into a dma-buf through a CPU mapping
        #[cfg(all(target_os = "linux", feature = "std"))]
        let system = system.with(MemoryDomainKind::DmaBuf);
        system
    }

    fn output_domain_follows_input(&self) -> bool {
        true
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Analytics overlay",
            "Filter/Editing/Video",
            "Draws detection boxes, poses, segmentation masks and regions of interest onto raw video",
            crate::AARON_BOXER,
        )
    }
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
        // Identity: pixels and geometry pass through; only boxes are painted.
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| {
            if Self::accepts(input) {
                CapsSet::one(input.clone())
            } else {
                CapsSet::from_alternatives(Vec::new())
            }
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configure_picture(absolute_caps)?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    #[cfg(feature = "metadata")]
    fn configure_allocation(&mut self, params: &g2g_core::AllocationParams) {
        self.downstream_renders = params
            .meta_requests
            .wants::<g2g_core::OverlayCompositionMeta>();
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new(
                "thickness",
                PropKind::Uint,
                "box outline thickness in pixels (>= 1)",
            )
            .with_range("1", "65535")
            .with_default("2"),
            PropertySpec::new(
                "mask-alpha",
                PropKind::Uint,
                "alpha the segmentation mask fill is blended at (0..255)",
            )
            .with_range("0", "255")
            .with_default("96"),
            SHOW_LABEL_PROPERTY,
            SHOW_SCORE_PROPERTY,
            SHOW_TRACK_PROPERTY,
            PropertySpec::new(
                "show-trail",
                PropKind::Bool,
                "draw the path each tracked object took",
            )
            .with_default("false"),
            PropertySpec::new(
                "trail-length",
                PropKind::Uint,
                "how many past positions a trail keeps (>= 2)",
            )
            .with_range("2", "65535")
            .with_default("30"),
            KEYPOINT_THRESHOLD_PROPERTY,
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "thickness" => {
                self.thickness = (value.as_uint().ok_or(PropError::Type)? as u32).max(1);
                Ok(())
            }
            "mask-alpha" => {
                self.mask_alpha = value.as_uint().ok_or(PropError::Type)?.min(255) as u8;
                Ok(())
            }
            "show-label" => {
                self.caption.label = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "show-score" => {
                self.caption.score = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "show-track" => {
                self.caption.track = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "show-trail" => {
                self.show_trail = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            "trail-length" => {
                self.trail_length = (value.as_uint().ok_or(PropError::Type)? as usize).max(2);
                Ok(())
            }
            "keypoint-threshold" => {
                self.keypoint_threshold = value.as_double().ok_or(PropError::Type)? as f32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "thickness" => Some(PropValue::Uint(self.thickness as u64)),
            "mask-alpha" => Some(PropValue::Uint(self.mask_alpha as u64)),
            "show-label" => Some(PropValue::Bool(self.caption.label)),
            "show-score" => Some(PropValue::Bool(self.caption.score)),
            "show-track" => Some(PropValue::Bool(self.caption.track)),
            "show-trail" => Some(PropValue::Bool(self.show_trail)),
            "trail-length" => Some(PropValue::Uint(self.trail_length as u64)),
            "keypoint-threshold" => Some(PropValue::Double(self.keypoint_threshold as f64)),
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
                    // Copy out the shapes so the immutable meta borrow ends before
                    // the mutable pixel borrow below.
                    let analytics = frame.meta.get::<AnalyticsMeta>();
                    // analyticsjoin's hold-last puts one detection result on several frames
                    let repeated = analytics == self.previous_analytics.as_ref();
                    let shapes = analytics.map(AnalyticsShapes::collect).unwrap_or_default();
                    if !repeated {
                        self.previous_analytics = analytics.cloned();
                    }
                    self.drawn += 1;
                    let trails_changed = self.show_trail && self.record_trails(&shapes, repeated);
                    // A live trail outlives the detections that fed it, so this
                    // frame may have something to paint with no shapes of its own.
                    let has_trail = self.show_trail && !self.trails.is_empty();
                    #[cfg(feature = "metadata")]
                    let downstream_renders = self.downstream_renders;
                    #[cfg(not(feature = "metadata"))]
                    let downstream_renders = false;
                    let has_drawing = !shapes.is_empty() || has_trail;
                    #[cfg(feature = "metadata")]
                    let composition_unchanged = repeated && !trails_changed;
                    if has_drawing && downstream_renders {
                        #[cfg(feature = "metadata")]
                        self.attach_composition(&mut frame, &shapes, composition_unchanged)?;
                    } else if has_drawing {
                        match &mut frame.domain {
                            MemoryDomain::System(slice) => {
                                let layout =
                                    PlaneLayout::tight(self.format, self.width, self.height)
                                        .ok_or(G2gError::CapsMismatch)?;
                                self.paint(slice.as_mut_slice(), &layout, &shapes)?;
                            }
                            #[cfg(all(target_os = "linux", feature = "std"))]
                            MemoryDomain::DmaBuf(dmabuf) => {
                                let (layout, _) = crate::paddedrows::dmabuf_plane_layout(
                                    dmabuf,
                                    self.format,
                                    self.width,
                                    self.height,
                                )
                                .ok_or(G2gError::CapsMismatch)?;
                                let mut map = crate::dmabufmap::DmaBufWriteMap::write(dmabuf)?;
                                self.paint(map.from_fd_start_mut(), &layout, &shapes)?;
                            }
                            _ => return Err(G2gError::UnsupportedDomain),
                        }
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                PipelinePacket::CapsChanged(caps) => {
                    if Self::accepts(&caps) {
                        self.configure_picture(&caps)?;
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

// shoulders are coco keypoints 5 and 6, the only ones visible here
#[cfg(test)]
pub(crate) fn shoulders_only_pose() -> AnalyticsMeta {
    let hidden = Keypoint {
        x: 0.5,
        y: 0.25,
        confidence: 0.1,
    };
    let mut keypoints = alloc::vec![hidden; COCO_KEYPOINTS];
    keypoints[5] = Keypoint {
        x: 0.25,
        y: 0.5,
        confidence: 0.9,
    };
    keypoints[6] = Keypoint {
        x: 0.75,
        y: 0.5,
        confidence: 0.9,
    };
    let mut meta = AnalyticsMeta::new();
    meta.push(AnalyticsNode::Pose(g2g_core::Pose {
        bbox: BBox {
            x: 0.125,
            y: 0.125,
            w: 0.75,
            h: 0.75,
        },
        label: 0,
        confidence: 0.9,
        keypoints,
    }));
    meta
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::frame_byte_size;
    use g2g_core::frame::Frame;

    use g2g_core::{FrameTiming, Mask, PushOutcome, Rate, Tracking};

    fn solid(w: usize, h: usize, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 4);
        for _ in 0..w * h {
            v.extend_from_slice(&rgba);
        }
        v
    }

    fn px(buf: &[u8], cw: usize, x: usize, y: usize) -> [u8; 4] {
        let i = (y * cw + x) * 4;
        [buf[i], buf[i + 1], buf[i + 2], buf[i + 3]]
    }

    fn rgba_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        }
    }

    fn det(x: f32, y: f32, w: f32, h: f32, label: u32) -> ObjectDetection {
        ObjectDetection {
            bbox: BBox { x, y, w, h },
            label,
            confidence: 0.9,
        }
    }

    fn overlay(width: u32, height: u32, thickness: u32) -> AnalyticsOverlay {
        AnalyticsOverlay {
            width,
            height,
            thickness,
            configured: true,
            ..AnalyticsOverlay::new()
        }
    }

    /// A meta holding one detection wired to the tracking id `track`.
    fn tracked_meta(detection: ObjectDetection, track: u64) -> AnalyticsMeta {
        let mut meta = AnalyticsMeta::new();
        let node = meta.add_detection(detection);
        let tracking = meta.push(AnalyticsNode::Tracking(Tracking { object_id: track }));
        meta.relate(node, tracking, RelationKind::Tracks);
        meta
    }

    /// The shapes of a meta holding just these detections.
    fn detection_shapes(detections: &[ObjectDetection]) -> AnalyticsShapes {
        let mut meta = AnalyticsMeta::new();
        for d in detections {
            meta.add_detection(*d);
        }
        AnalyticsShapes::collect(&meta)
    }

    /// A segmentation over `bbox` whose `mask_w` x `mask_h` mask is covered where
    /// `covered(i, j)` holds.
    fn seg(
        bbox: BBox,
        label: u32,
        mask_w: u32,
        mask_h: u32,
        covered: impl Fn(u32, u32) -> bool,
    ) -> Segmentation {
        let mut data = Vec::with_capacity((mask_w * mask_h) as usize);
        for j in 0..mask_h {
            for i in 0..mask_w {
                data.push(if covered(i, j) { u8::MAX } else { 0 });
            }
        }
        Segmentation {
            bbox,
            label,
            confidence: 0.9,
            mask: Mask::new(mask_w, mask_h, mask_w, data).expect("mask geometry"),
        }
    }

    #[test]
    fn score_text_is_two_decimals() {
        assert_eq!(score_text(0.923), "0.92");
        assert_eq!(score_text(0.5), "0.50");
        assert_eq!(score_text(1.0), "1.00");
        assert_eq!(score_text(0.0), "0.00");
    }

    #[test]
    fn collect_pairs_a_detection_with_the_tracking_it_tracks_as() {
        let mut meta = AnalyticsMeta::new();
        let tracked = meta.add_detection(det(0.1, 0.1, 0.2, 0.2, 0));
        let untracked = meta.add_detection(det(0.5, 0.5, 0.2, 0.2, 1));
        let tracking = meta.push(AnalyticsNode::Tracking(Tracking { object_id: 77 }));
        meta.relate(tracked, tracking, RelationKind::Tracks);

        let shapes = AnalyticsShapes::collect(&meta);
        assert_eq!(shapes.detections[0].track_id, Some(77));
        assert_eq!(
            shapes.detections[1].track_id, None,
            "detection {untracked} has no Tracks relation"
        );
    }

    #[test]
    fn captions_are_off_until_asked_for() {
        let shapes = detection_shapes(&[det(0.25, 0.25, 0.5, 0.5, 0)]);
        let painted = &shapes.detections[0];
        assert_eq!(
            overlay(64, 64, 1)
                .caption
                .text(None, painted.track_id, painted.detection.confidence),
            None
        );
        assert_eq!(
            overlay(64, 64, 1).with_score(true).caption.text(
                None,
                painted.track_id,
                painted.detection.confidence
            ),
            Some(String::from("0.90"))
        );
    }

    #[test]
    fn caption_shows_the_tracking_id_when_one_is_wired() {
        let mut meta = AnalyticsMeta::new();
        let detection = meta.add_detection(det(0.25, 0.25, 0.5, 0.5, 0));
        let tracking = meta.push(AnalyticsNode::Tracking(Tracking { object_id: 4 }));
        meta.relate(detection, tracking, RelationKind::Tracks);
        let shapes = AnalyticsShapes::collect(&meta);
        let painted = &shapes.detections[0];

        assert_eq!(
            overlay(64, 64, 1).with_track(true).caption.text(
                None,
                painted.track_id,
                painted.detection.confidence
            ),
            Some(String::from("ID:4"))
        );
        assert_eq!(
            overlay(64, 64, 1)
                .with_track(true)
                .with_score(true)
                .caption
                .text(None, painted.track_id, painted.detection.confidence),
            Some(String::from("ID:4 0.90"))
        );
    }

    #[test]
    fn caption_paints_a_bar_above_the_box_and_leaves_the_box_alone() {
        // A 64x64 canvas so the glyph scale is 1 and the box has room above it.
        let ov = overlay(64, 64, 1).with_score(true);
        let mut buf = solid(64, 64, [0, 0, 0, 255]);
        // Box pixels (16,16)..(47,47); the caption bar sits directly above row 16.
        ov.render(&mut buf, &detection_shapes(&[det(0.25, 0.25, 0.5, 0.5, 0)]));
        let bar_row = 16 - (GLYPH_HEIGHT as usize + 2);
        assert_ne!(
            px(&buf, 64, 16, bar_row),
            [0, 0, 0, 255],
            "caption bar painted above the box"
        );
        assert_eq!(
            px(&buf, 64, 20, 20),
            [0, 0, 0, 255],
            "box interior still untouched"
        );
        assert_eq!(
            px(&buf, 64, 16, bar_row - 1),
            [0, 0, 0, 255],
            "nothing painted above the bar"
        );
    }

    #[test]
    fn caption_on_a_box_at_the_top_edge_stays_on_canvas() {
        // No room above, so the bar drops inside the box rather than off-canvas.
        let ov = overlay(32, 32, 1).with_score(true);
        let mut buf = solid(32, 32, [0, 0, 0, 255]);
        ov.render(&mut buf, &detection_shapes(&[det(0.0, 0.0, 1.0, 1.0, 0)]));
        assert_ne!(
            px(&buf, 32, 1, 1),
            [0, 0, 0, 255],
            "caption drawn inside the top edge"
        );
    }

    #[test]
    fn frame_classification_draws_a_caption_in_the_top_left_corner() {
        let mut meta = AnalyticsMeta::new();
        meta.set_class_names(["person"]);
        meta.push(AnalyticsNode::Classification(Classification {
            label: 0,
            confidence: 0.9,
        }));
        let shapes = AnalyticsShapes::collect(&meta);
        let black = [0, 0, 0, 255];

        let mut plain = solid(64, 64, black);
        overlay(64, 64, 1).render(&mut plain, &shapes);
        assert_eq!(px(&plain, 64, 0, 0), black, "captions are off by default");

        let captioned = overlay(64, 64, 1).with_label(true).with_score(true);
        assert_eq!(
            shapes.frame_caption(&captioned.caption),
            Some((String::from("person 0.90"), 0))
        );
        let mut buf = solid(64, 64, black);
        captioned.render(&mut buf, &shapes);
        assert_eq!(px(&buf, 64, 0, 0), palette_color(0));
    }

    #[test]
    fn a_classification_of_a_detection_gets_no_frame_caption() {
        let mut meta = AnalyticsMeta::new();
        let detection = meta.add_detection(det(0.25, 0.25, 0.5, 0.5, 0));
        let classification = meta.push(AnalyticsNode::Classification(Classification {
            label: 0,
            confidence: 0.9,
        }));
        meta.relate(detection, classification, RelationKind::Classifies);
        let shapes = AnalyticsShapes::collect(&meta);
        assert!(shapes.frame_classifications.is_empty());
    }

    #[test]
    fn render_paints_box_border_and_leaves_interior() {
        // 8x8 black canvas; a normalized box covering [0.25,0.75] -> pixels (2,2)
        // to (5,5). A 1px red (class 0) outline; the interior stays black.
        let ov = overlay(8, 8, 1);
        let mut buf = solid(8, 8, [0, 0, 0, 255]);
        ov.render(&mut buf, &detection_shapes(&[det(0.25, 0.25, 0.5, 0.5, 0)]));
        let red = palette_color(0);
        assert_eq!(px(&buf, 8, 2, 2), red, "top-left corner on the border");
        assert_eq!(px(&buf, 8, 5, 5), red, "bottom-right corner on the border");
        assert_eq!(px(&buf, 8, 5, 2), red, "top-right corner on the border");
        assert_eq!(px(&buf, 8, 3, 3), [0, 0, 0, 255], "interior untouched");
        assert_eq!(
            px(&buf, 8, 0, 0),
            [0, 0, 0, 255],
            "outside the box untouched"
        );
    }

    #[test]
    fn render_clips_box_to_canvas_bounds() {
        // A box running off the right/bottom edge must not panic or write OOB.
        let ov = overlay(4, 4, 2);
        let mut buf = solid(4, 4, [0, 0, 0, 255]);
        ov.render(&mut buf, &detection_shapes(&[det(0.5, 0.5, 1.0, 1.0, 1)]));
        // The far corner is on the clipped border, painted the class-1 colour.
        assert_eq!(
            px(&buf, 4, 3, 3),
            palette_color(1),
            "clipped corner painted"
        );
    }

    #[test]
    fn render_blends_a_mask_over_the_covered_half_of_its_box() {
        // 8x8 black canvas; a box over pixels (2,2)..(5,5) with a 2x2 mask whose
        // left column is covered, i.e. columns 2 and 3 of the box.
        let ov = overlay(8, 8, 1);
        let mut buf = solid(8, 8, [0, 0, 0, 255]);
        let mut meta = AnalyticsMeta::new();
        meta.push(AnalyticsNode::Segmentation(seg(
            BBox {
                x: 0.25,
                y: 0.25,
                w: 0.5,
                h: 0.5,
            },
            0,
            2,
            2,
            |i, _| i == 0,
        )));
        ov.render(&mut buf, &AnalyticsShapes::collect(&meta));

        // Palette slot 0 (0xFF3B30) at alpha 96 over opaque black: each channel is
        // (c * 96 + 127) / 255, the alpha stays opaque.
        let fill = [96, 22, 18, 255];
        assert_eq!(px(&buf, 8, 2, 2), fill, "covered sample blended");
        assert_eq!(px(&buf, 8, 3, 5), fill, "covered sample blended");
        assert_eq!(
            px(&buf, 8, 4, 3),
            [0, 0, 0, 255],
            "uncovered half of the box untouched"
        );
        assert_eq!(
            px(&buf, 8, 1, 2),
            [0, 0, 0, 255],
            "outside the box untouched"
        );
    }

    #[test]
    fn render_draws_a_pose_box_with_limbs_between_visible_keypoints_only() {
        let black = [0, 0, 0, 255];
        let red = palette_color(0);
        let shapes = AnalyticsShapes::collect(&shoulders_only_pose());
        let mut buf = solid(64, 64, black);
        overlay(64, 64, 1).render(&mut buf, &shapes);
        assert_eq!(px(&buf, 64, 8, 8), red, "pose box corner");
        assert_eq!(px(&buf, 64, 32, 32), red, "limb between the shoulders");
        assert_eq!(px(&buf, 64, 16, 32), contrast_ink(red), "left shoulder dot");
        assert_eq!(px(&buf, 64, 32, 16), black, "hidden nose");

        let mut buf = solid(64, 64, black);
        overlay(64, 64, 1)
            .with_keypoint_threshold(0.95)
            .render(&mut buf, &shapes);
        assert_eq!(px(&buf, 64, 32, 32), black, "limb hidden above its score");
        assert_eq!(px(&buf, 64, 8, 8), red, "box still drawn");
    }

    #[test]
    fn render_dashes_an_roi_outline() {
        // 16x16; an ROI over pixels (4,4)..(11,11) with a 1px outline. The dash
        // phase is absolute, so along the top edge x=4,5 paint and x=6..11 do not.
        let ov = overlay(16, 16, 1);
        let mut buf = solid(16, 16, [0, 0, 0, 255]);
        let mut meta = AnalyticsMeta::new();
        meta.push(AnalyticsNode::Roi(Roi {
            bbox: BBox {
                x: 0.25,
                y: 0.25,
                w: 0.5,
                h: 0.5,
            },
            id: 1,
            label: 3,
        }));
        ov.render(&mut buf, &AnalyticsShapes::collect(&meta));

        // A lone ROI takes its palette slot from its own id, not its label.
        let green = palette_color(1);
        assert_eq!(px(&buf, 16, 4, 4), green, "dash on at x=4");
        assert_eq!(px(&buf, 16, 5, 4), green, "dash on at x=5");
        assert_eq!(px(&buf, 16, 8, 4), [0, 0, 0, 255], "dash off at x=8");
        assert_eq!(
            px(&buf, 16, 4, 8),
            [0, 0, 0, 255],
            "dash off down the left edge at y=8"
        );
    }

    #[test]
    fn collect_gives_an_roi_the_palette_slot_of_the_mask_containing_it() {
        // Two instances, each with a contained ROI whose id disagrees with the
        // instance order, so the pairing has to come from the relation.
        let mut meta = AnalyticsMeta::new();
        let bbox = BBox {
            x: 0.0,
            y: 0.0,
            w: 0.5,
            h: 0.5,
        };
        let first = meta.push(AnalyticsNode::Segmentation(seg(bbox, 7, 1, 1, |_, _| true)));
        let second = meta.push(AnalyticsNode::Segmentation(seg(bbox, 7, 1, 1, |_, _| true)));
        let first_roi = meta.push(AnalyticsNode::Roi(Roi {
            bbox,
            id: 40,
            label: 7,
        }));
        let second_roi = meta.push(AnalyticsNode::Roi(Roi {
            bbox,
            id: 41,
            label: 7,
        }));
        meta.relate(second, second_roi, RelationKind::Contains);
        meta.relate(first, first_roi, RelationKind::Contains);

        let shapes = AnalyticsShapes::collect(&meta);
        assert_eq!(shapes.masks.len(), 2);
        assert_eq!(
            (shapes.masks[0].palette_index, shapes.masks[1].palette_index),
            (0, 1),
            "masks are numbered in instance order"
        );
        assert_eq!(
            (shapes.rois[0].palette_index, shapes.rois[1].palette_index),
            (0, 1),
            "each ROI paints in its container's slot, not its own id"
        );
    }

    /// Capturing sink that keeps the last forwarded frame's pixels.
    #[derive(Default)]
    struct PixelSink {
        last: Option<Vec<u8>>,
        #[cfg(feature = "metadata")]
        last_composition: Option<g2g_core::OverlayCompositionMeta>,
    }
    impl OutputSink for PixelSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(frame) = packet {
                    if let Some(slice) = frame.domain.as_system_slice() {
                        self.last = Some(slice.to_vec());
                    }
                    #[cfg(feature = "metadata")]
                    {
                        self.last_composition = frame
                            .meta
                            .get::<g2g_core::OverlayCompositionMeta>()
                            .cloned();
                    }
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    fn rgba_frame_with_meta(w: u32, h: u32, meta: AnalyticsMeta) -> Frame {
        let bytes = solid(w as usize, h as usize, [0, 0, 0, 255]);
        let mut frame = Frame::from_system_bytes(bytes, FrameTiming::default(), 0);
        frame.meta.attach(meta);
        frame
    }

    #[tokio::test]
    async fn process_draws_attached_detections_onto_the_frame() {
        let mut ov = AnalyticsOverlay::new().with_thickness(1);
        ov.configure_pipeline(&rgba_caps(8, 8)).unwrap();
        let mut meta = AnalyticsMeta::new();
        meta.add_detection(det(0.25, 0.25, 0.5, 0.5, 0));
        let frame = rgba_frame_with_meta(8, 8, meta);

        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let out = sink.last.expect("frame forwarded");
        assert_eq!(px(&out, 8, 2, 2), palette_color(0), "box border drawn");
        assert_eq!(px(&out, 8, 3, 3), [0, 0, 0, 255], "interior untouched");
        assert_eq!(ov.drawn_count(), 1);
    }

    #[tokio::test]
    async fn process_draws_an_attached_segmentation_and_its_roi() {
        // The M992 shape: a Segmentation plus the mask-tight Roi it contains, drawn
        // through the real process path at an explicit mask alpha.
        let mut ov = AnalyticsOverlay::new()
            .with_thickness(1)
            .with_mask_alpha(255);
        ov.configure_pipeline(&rgba_caps(16, 16)).unwrap();
        let bbox = BBox {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        };
        let mut meta = AnalyticsMeta::new();
        // Box pixels 4..11 with a 2x2 mask covered only at (0,0), so the fill is
        // its top-left quarter, pixels 4..7. The ROI is that quarter, mask-tight.
        let mask = meta.push(AnalyticsNode::Segmentation(seg(bbox, 0, 2, 2, |i, j| {
            i == 0 && j == 0
        })));
        let roi = meta.push(AnalyticsNode::Roi(Roi {
            bbox: BBox {
                x: 0.25,
                y: 0.25,
                w: 0.25,
                h: 0.25,
            },
            id: 9,
            label: 0,
        }));
        meta.relate(mask, roi, RelationKind::Contains);
        let frame = rgba_frame_with_meta(16, 16, meta);

        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let out = sink.last.expect("frame forwarded");
        // At alpha 255 the fill is the palette colour itself; the ROI outline is the
        // same colour, so the fill is read inside it and the dashes on its edge.
        let fill = palette_color(0);
        assert_eq!(px(&out, 16, 5, 5), fill, "mask fill inside the ROI");
        assert_eq!(px(&out, 16, 6, 6), fill, "mask fill inside the ROI");
        assert_eq!(px(&out, 16, 4, 4), fill, "ROI dash on its top-left corner");
        assert_eq!(
            px(&out, 16, 9, 9),
            [0, 0, 0, 255],
            "uncovered mask samples left alone"
        );
        assert_eq!(ov.drawn_count(), 1);
    }

    #[tokio::test]
    async fn process_passes_through_a_frame_without_meta() {
        let mut ov = AnalyticsOverlay::new();
        ov.configure_pipeline(&rgba_caps(4, 4)).unwrap();
        let bytes = solid(4, 4, [10, 20, 30, 255]);
        let frame = Frame::from_system_bytes(bytes.clone(), FrameTiming::default(), 0);

        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        assert_eq!(
            sink.last.expect("forwarded"),
            bytes,
            "pixels unchanged without meta"
        );
    }

    #[tokio::test]
    async fn trail_joins_a_track_across_frames_and_leaves_it_alone_untracked() {
        // Two frames of one tracked box moving down the right half of a 32x32
        // canvas. The trail runs through the bottom edge of each box, so the
        // midpoint between the two bottom edges must be painted on frame two.
        let mut ov = AnalyticsOverlay::new().with_thickness(1).with_trail(true);
        ov.configure_pipeline(&rgba_caps(32, 32)).unwrap();
        let mut sink = PixelSink::default();
        for y in [0.25_f32, 0.5] {
            let meta = tracked_meta(det(0.5, y, 0.25, 0.25, 0), 3);
            let frame = rgba_frame_with_meta(32, 32, meta);
            ov.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        let out = sink.last.expect("frame forwarded");
        // Bottom-centre of the boxes: (20, 16) then (20, 24); the segment
        // between them covers (20, 20).
        assert_ne!(
            px(&out, 32, 20, 20),
            [0, 0, 0, 255],
            "trail drawn between the two positions"
        );

        // The same movement without a Tracks relation records nothing to draw.
        let mut untracked = AnalyticsOverlay::new().with_thickness(1).with_trail(true);
        untracked.configure_pipeline(&rgba_caps(32, 32)).unwrap();
        let mut sink = PixelSink::default();
        for y in [0.25_f32, 0.5] {
            let mut meta = AnalyticsMeta::new();
            meta.add_detection(det(0.5, y, 0.25, 0.25, 0));
            let frame = rgba_frame_with_meta(32, 32, meta);
            untracked
                .process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        assert!(untracked.trails.is_empty(), "no track, no trail");
        assert_eq!(
            px(&sink.last.expect("forwarded"), 32, 20, 20),
            [0, 0, 0, 255],
            "nothing painted between the boxes"
        );
    }

    #[tokio::test]
    async fn trail_is_capped_at_its_length_and_expires_with_its_track() {
        let mut ov = AnalyticsOverlay::new()
            .with_trail(true)
            .with_trail_length(4);
        ov.configure_pipeline(&rgba_caps(32, 32)).unwrap();
        let mut sink = PixelSink::default();
        let moving = |step: u8| tracked_meta(det(0.4, 0.04 * f32::from(step), 0.2, 0.2, 0), 8);
        for step in 0..10 {
            let frame = rgba_frame_with_meta(32, 32, moving(step));
            ov.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        assert_eq!(
            ov.trails[&8].points.len(),
            4,
            "only the last trail-length points are kept"
        );
        let oldest = ov.trails[&8].points.front().copied();
        let frame = rgba_frame_with_meta(32, 32, moving(9));
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        assert_eq!(
            ov.trails[&8].points.front().copied(),
            oldest,
            "a repeated detection result adds no point to push the oldest out"
        );

        // Frames with no detections age the trail out; it survives the first few.
        for _ in 0..TRAIL_TTL_FRAMES {
            let frame = rgba_frame_with_meta(32, 32, AnalyticsMeta::new());
            ov.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .unwrap();
        }
        assert!(
            ov.trails.contains_key(&8),
            "trail outlives a gap in the track"
        );
        let frame = rgba_frame_with_meta(32, 32, AnalyticsMeta::new());
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();
        assert!(
            ov.trails.is_empty(),
            "trail dropped once its track is stale"
        );
    }

    fn video_caps(format: RawVideoFormat, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        }
    }

    #[test]
    fn intercept_takes_rgba_and_nv12_only() {
        let ov = AnalyticsOverlay::new();
        assert!(ov
            .intercept_caps(&video_caps(RawVideoFormat::I420, 8, 8))
            .is_err());
        assert!(ov
            .intercept_caps(&video_caps(RawVideoFormat::Nv12, 8, 8))
            .is_ok());
        assert!(ov.intercept_caps(&rgba_caps(8, 8)).is_ok());
    }

    #[tokio::test]
    async fn process_draws_detections_onto_an_nv12_frame() {
        const SIDE: u32 = 8;
        const GREY_LUMA: u8 = 100;
        const NEUTRAL_CHROMA: u8 = 128;
        let mut ov = AnalyticsOverlay::new().with_thickness(1);
        ov.configure_pipeline(&video_caps(RawVideoFormat::Nv12, SIDE, SIDE))
            .unwrap();
        let mut meta = AnalyticsMeta::new();
        meta.add_detection(det(0.25, 0.25, 0.5, 0.5, 0));
        let luma = (SIDE * SIDE) as usize;
        let mut pixels = alloc::vec![GREY_LUMA; luma];
        pixels.resize(
            frame_byte_size(RawVideoFormat::Nv12, SIDE, SIDE),
            NEUTRAL_CHROMA,
        );
        let mut frame = Frame::from_system_bytes(pixels, FrameTiming::default(), 0);
        frame.meta.attach(meta);

        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame), &mut sink)
            .await
            .unwrap();

        let out = sink.last.expect("frame forwarded");
        let matrix = YuvRgbMatrix::new(g2g_core::Colorimetry::UNKNOWN);
        let [r, g, b, _] = palette_color(0).map(i32::from);
        let (y, u, v) = matrix.rgb_to_yuv(r, g, b);
        let at = |x: usize, y: usize| y * SIDE as usize + x;
        assert_eq!(out[at(2, 2)], y as u8, "box border drawn in luma");
        assert_eq!(out[at(3, 3)], GREY_LUMA, "interior untouched");
        // the 2x2 chroma block at (2, 2) is the box's corner: three of its four pixels are border
        let block = luma + 2 * (SIDE as usize / 2 + 1);
        let three_quarters =
            |neutral: u8, drawn: i32| blend_u8(neutral, drawn as u8, (3 * 255 / 4) as u8);
        assert_eq!(out[block], three_quarters(NEUTRAL_CHROMA, u));
        assert_eq!(out[block + 1], three_quarters(NEUTRAL_CHROMA, v));
    }

    #[cfg(feature = "metadata")]
    #[tokio::test]
    async fn a_rendering_sink_gets_the_shapes_as_a_composition_and_the_pixels_stay() {
        use g2g_core::meta::MetaRequests;
        use g2g_core::{AllocationParams, OverlayCompositionMeta};
        const SIDE: u32 = 8;
        let mut ov = AnalyticsOverlay::new().with_thickness(1).with_trail(true);
        ov.configure_pipeline(&video_caps(RawVideoFormat::Nv12, SIDE, SIDE))
            .unwrap();
        ov.configure_allocation(&AllocationParams::meta_demand(
            MetaRequests::new().request_from_every_consumer::<OverlayCompositionMeta>(),
        ));
        let meta = tracked_meta(det(0.25, 0.25, 0.5, 0.5, 0), 0);
        let pixels = alloc::vec![100; frame_byte_size(RawVideoFormat::Nv12, SIDE, SIDE)];
        let frame = |meta: &AnalyticsMeta| {
            let mut frame = Frame::from_system_bytes(pixels.clone(), FrameTiming::default(), 0);
            frame.meta.attach(meta.clone());
            frame
        };

        let mut sink = PixelSink::default();
        ov.process(PipelinePacket::DataFrame(frame(&meta)), &mut sink)
            .await
            .unwrap();
        assert_eq!(sink.last.as_deref(), Some(&pixels[..]), "video untouched");
        let composition = sink.last_composition.take().expect("composition attached");
        let [rectangle] = composition.rectangles() else {
            panic!("one rectangle, got {}", composition.rectangles().len());
        };
        assert_eq!(rectangle.pixel_size(), (SIDE, SIDE));
        let bitmap = rectangle.pixels();
        assert_eq!(
            px(bitmap, SIDE as usize, 2, 2),
            palette_color(0),
            "box border"
        );
        assert_eq!(
            px(bitmap, SIDE as usize, 3, 3),
            [0; 4],
            "transparent interior"
        );

        ov.process(PipelinePacket::DataFrame(frame(&meta)), &mut sink)
            .await
            .unwrap();
        let repeated = sink.last_composition.expect("composition attached");
        assert!(
            alloc::sync::Arc::ptr_eq(repeated.rectangles()[0].pixels(), bitmap),
            "a held detection with an unchanged trail reuses the bitmap"
        );
    }
}
