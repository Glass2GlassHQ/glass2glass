//! Per-frame metadata system: typed blobs that travel with a [`Frame`] (the
//! GstMeta / GstAnalyticsRelationMeta analog), and the `AnalyticsMeta` relation
//! graph for ML detection / classification / tracking results.
//!
//! Gated behind the `metadata` cargo feature. When **off** (the default, and the
//! only configuration the `no_std` / Cortex-M baseline uses) [`FrameMetaSet`] is
//! a zero-sized unit: the `Frame::meta` field exists for API stability but costs
//! nothing per frame. When **on** it is a list of typed [`FrameMeta`] trait
//! objects with attach / typed-get / iterate / propagate, and the standard
//! [`AnalyticsMeta`] is available for detection pipelines.
//!
//! **Why now:** the field was reserved at M88; the trait body and the relation
//! graph land with the first metadata-producing element (a YOLO-style detection
//! postprocess), so a real client shapes the API rather than speculation.
//!
//! [`Frame`]: crate::frame::Frame

// ---- feature off: the zero-sized placeholder ----

/// Per-frame attachable metadata set (feature `metadata` **off**): a zero-sized
/// unit, so the baseline pays nothing. See the module docs.
#[cfg(not(feature = "metadata"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameMetaSet;

#[cfg(not(feature = "metadata"))]
impl FrameMetaSet {
    /// An empty metadata set. `const` so frame construction stays trivial.
    #[inline]
    pub const fn new() -> Self {
        FrameMetaSet
    }
}

// ---- feature on: the real typed container + analytics graph ----

#[cfg(feature = "metadata")]
pub use on::*;

#[cfg(feature = "metadata")]
mod on {
    use alloc::boxed::Box;
    use alloc::vec::Vec;
    use core::any::Any;

    /// How a piece of metadata survives a transform, the GstMeta
    /// `transform_func` analog. Reported by [`FrameMeta::propagate`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Transform {
        /// A deep copy (e.g. a tee branch clone): meta is duplicated.
        Copy,
        /// A geometry resample (videoscale / compositor pad scale).
        Scale,
        /// A spatial crop (videocrop).
        Crop,
        /// A re-encode to a compressed codec: pixel-derived meta is lost.
        Encode,
    }

    /// Whether a meta is kept through a [`Transform`] or dropped.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Propagation {
        Keep,
        Drop,
    }

    /// A typed, per-frame, attachable piece of metadata (the `GstMeta` analog).
    ///
    /// `as_any` enables typed retrieval via downcast (trait upcasting is not on
    /// the MSRV); `propagate` is the per-transform survival policy. Meta is
    /// `Send + Sync` so a frame crosses a multi-thread runtime.
    pub trait FrameMeta: core::fmt::Debug + Send + Sync {
        fn as_any(&self) -> &dyn Any;
        fn as_any_mut(&mut self) -> &mut dyn Any;
        /// How this meta survives `transform`. Default keeps it through
        /// everything; override to drop on transforms that invalidate it.
        fn propagate(&self, _transform: Transform) -> Propagation {
            Propagation::Keep
        }
    }

    /// A list of typed [`FrameMeta`] attached to a frame. Empty (no allocation)
    /// on a freshly constructed frame.
    #[derive(Debug, Default)]
    pub struct FrameMetaSet(Vec<Box<dyn FrameMeta>>);

    impl FrameMetaSet {
        /// An empty metadata set with no backing allocation.
        #[inline]
        pub fn new() -> Self {
            FrameMetaSet(Vec::new())
        }

        /// Attach one piece of metadata.
        pub fn attach<T: FrameMeta + 'static>(&mut self, meta: T) {
            self.0.push(Box::new(meta));
        }

        /// The first attached meta of type `T`, if any.
        pub fn get<T: FrameMeta + 'static>(&self) -> Option<&T> {
            self.0.iter().find_map(|m| m.as_any().downcast_ref::<T>())
        }

        /// Mutable access to the first attached meta of type `T`, if any.
        pub fn get_mut<T: FrameMeta + 'static>(&mut self) -> Option<&mut T> {
            self.0.iter_mut().find_map(|m| m.as_any_mut().downcast_mut::<T>())
        }

        /// Iterate every attached meta as a trait object.
        pub fn iter(&self) -> impl Iterator<Item = &dyn FrameMeta> {
            self.0.iter().map(|b| b.as_ref())
        }

        pub fn len(&self) -> usize {
            self.0.len()
        }

        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }

        /// Apply a [`Transform`]: retain only metas whose `propagate` returns
        /// [`Propagation::Keep`]. An element that resamples / re-encodes calls
        /// this so stale meta never rides a frame it no longer describes.
        pub fn propagate(&mut self, transform: Transform) {
            self.0.retain(|m| m.propagate(transform) == Propagation::Keep);
        }
    }

    /// A normalized bounding box: all fields in `[0, 1]` relative to the frame,
    /// `(x, y)` the top-left corner and `(w, h)` the size. Normalized so a box
    /// survives a downstream scale / crop without a coordinate rewrite.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct BBox {
        pub x: f32,
        pub y: f32,
        pub w: f32,
        pub h: f32,
    }

    impl BBox {
        /// Intersection-over-union with `other`, the NMS overlap metric.
        pub fn iou(&self, other: &BBox) -> f32 {
            let ix0 = self.x.max(other.x);
            let iy0 = self.y.max(other.y);
            let ix1 = (self.x + self.w).min(other.x + other.w);
            let iy1 = (self.y + self.h).min(other.y + other.h);
            let iw = (ix1 - ix0).max(0.0);
            let ih = (iy1 - iy0).max(0.0);
            let inter = iw * ih;
            let union = self.w * self.h + other.w * other.h - inter;
            if union <= 0.0 {
                0.0
            } else {
                inter / union
            }
        }
    }

    /// A detected object: its box, class label index, and confidence `[0, 1]`.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct ObjectDetection {
        pub bbox: BBox,
        pub label: u32,
        pub confidence: f32,
    }

    /// A whole-region or per-detection classification result.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Classification {
        pub label: u32,
        pub confidence: f32,
    }

    /// A persistent tracking identity across frames.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Tracking {
        pub object_id: u64,
    }

    /// A node in the [`AnalyticsMeta`] relation graph.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum AnalyticsNode {
        Detection(ObjectDetection),
        Classification(Classification),
        Tracking(Tracking),
    }

    /// The kind of a directed edge between two analytics nodes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum RelationKind {
        /// A detection has-a classification (detection -> classification).
        Classifies,
        /// A detection has-a tracking identity (detection -> tracking).
        Tracks,
        /// A generic containment / part-of relation.
        Contains,
    }

    /// A directed edge between two nodes by index into [`AnalyticsMeta::nodes`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Relation {
        pub from: usize,
        pub to: usize,
        pub kind: RelationKind,
    }

    /// The per-frame analytics relation graph (the `GstAnalyticsRelationMeta`
    /// analog): typed detection / classification / tracking nodes plus directed
    /// relations between them, so downstream elements (overlay, recorder, alarm)
    /// read results by node kind and traversal instead of decoding raw tensors.
    #[derive(Debug, Default, Clone, PartialEq)]
    pub struct AnalyticsMeta {
        pub nodes: Vec<AnalyticsNode>,
        pub relations: Vec<Relation>,
    }

    impl AnalyticsMeta {
        pub fn new() -> Self {
            Self::default()
        }

        /// Append a node, returning its index (used to wire relations).
        pub fn push(&mut self, node: AnalyticsNode) -> usize {
            self.nodes.push(node);
            self.nodes.len() - 1
        }

        /// Append a detection node, returning its index.
        pub fn add_detection(&mut self, detection: ObjectDetection) -> usize {
            self.push(AnalyticsNode::Detection(detection))
        }

        /// Wire a directed relation between two node indices.
        pub fn relate(&mut self, from: usize, to: usize, kind: RelationKind) {
            self.relations.push(Relation { from, to, kind });
        }

        /// Iterate the detection nodes.
        pub fn detections(&self) -> impl Iterator<Item = &ObjectDetection> {
            self.nodes.iter().filter_map(|n| match n {
                AnalyticsNode::Detection(d) => Some(d),
                _ => None,
            })
        }
    }

    impl FrameMeta for AnalyticsMeta {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        /// Normalized coordinates survive a scale / crop / copy unchanged; a
        /// re-encode to a compressed codec discards pixel-derived analytics.
        fn propagate(&self, transform: Transform) -> Propagation {
            match transform {
                Transform::Encode => Propagation::Drop,
                _ => Propagation::Keep,
            }
        }
    }
}

#[cfg(all(test, feature = "metadata"))]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, label: u32, conf: f32) -> ObjectDetection {
        ObjectDetection { bbox: BBox { x, y, w, h }, label, confidence: conf }
    }

    #[test]
    fn attach_and_typed_get_round_trip() {
        let mut set = FrameMetaSet::new();
        assert!(set.is_empty());
        let mut a = AnalyticsMeta::new();
        a.add_detection(det(0.1, 0.1, 0.2, 0.2, 7, 0.9));
        set.attach(a);
        assert_eq!(set.len(), 1);
        let got = set.get::<AnalyticsMeta>().expect("AnalyticsMeta attached");
        assert_eq!(got.detections().count(), 1);
        assert_eq!(got.detections().next().unwrap().label, 7);
    }

    #[test]
    fn get_mut_allows_in_place_update() {
        let mut set = FrameMetaSet::new();
        set.attach(AnalyticsMeta::new());
        set.get_mut::<AnalyticsMeta>()
            .unwrap()
            .add_detection(det(0.0, 0.0, 0.5, 0.5, 1, 0.5));
        assert_eq!(set.get::<AnalyticsMeta>().unwrap().nodes.len(), 1);
    }

    #[test]
    fn propagate_keeps_through_scale_drops_on_encode() {
        let mut set = FrameMetaSet::new();
        set.attach(AnalyticsMeta::new());
        set.propagate(Transform::Scale);
        assert_eq!(set.len(), 1, "normalized analytics survive a scale");
        set.propagate(Transform::Encode);
        assert!(set.is_empty(), "a re-encode drops pixel-derived analytics");
    }

    #[test]
    fn relation_graph_links_detection_to_classification() {
        let mut a = AnalyticsMeta::new();
        let d = a.add_detection(det(0.2, 0.2, 0.3, 0.3, 2, 0.8));
        let c = a.push(AnalyticsNode::Classification(Classification { label: 42, confidence: 0.7 }));
        a.relate(d, c, RelationKind::Classifies);
        assert_eq!(a.relations.len(), 1);
        assert_eq!(a.relations[0], Relation { from: d, to: c, kind: RelationKind::Classifies });
    }

    #[test]
    fn iou_is_zero_for_disjoint_and_one_for_identical() {
        let a = BBox { x: 0.0, y: 0.0, w: 0.2, h: 0.2 };
        let b = BBox { x: 0.5, y: 0.5, w: 0.2, h: 0.2 };
        assert_eq!(a.iou(&b), 0.0, "disjoint boxes do not overlap");
        assert!((a.iou(&a) - 1.0).abs() < 1e-6, "identical boxes fully overlap");
        // Half-overlap: a and c share half their area horizontally.
        let c = BBox { x: 0.1, y: 0.0, w: 0.2, h: 0.2 };
        let iou = a.iou(&c);
        assert!(iou > 0.3 && iou < 0.34, "half-shifted overlap ~1/3 IoU: {iou}");
    }
}
