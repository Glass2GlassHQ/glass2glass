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
    use alloc::string::String;
    use alloc::sync::Arc;
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
        /// A boxed deep copy of this meta, the GstMeta `copy_func` analog. Backs
        /// the copy-on-write of a shared meta when a tee branch mutates it (see
        /// [`FrameMetaSet::get_mut`]); each `FrameMeta` impl is its own concrete
        /// type, so the duplication can only be expressed on the trait.
        fn clone_box(&self) -> Box<dyn FrameMeta>;
        /// How this meta survives `transform`. Default keeps it through
        /// everything; override to drop on transforms that invalidate it.
        fn propagate(&self, _transform: Transform) -> Propagation {
            Propagation::Keep
        }
    }

    /// A list of typed [`FrameMeta`] attached to a frame. Empty (no allocation)
    /// on a freshly constructed frame.
    ///
    /// Each entry is an [`Arc`] so a fan-out (tee) clone shares the metadata by
    /// refcount instead of deep-copying it (cheap, and the analytics graph is
    /// identical on both branches). A branch that mutates one entry pays a
    /// copy-on-write deep copy only then (see [`get_mut`](Self::get_mut)), so the
    /// other branch never observes the change.
    #[derive(Debug, Default, Clone)]
    pub struct FrameMetaSet(Vec<Arc<dyn FrameMeta>>);

    impl FrameMetaSet {
        /// An empty metadata set with no backing allocation.
        #[inline]
        pub fn new() -> Self {
            FrameMetaSet(Vec::new())
        }

        /// Attach one piece of metadata.
        pub fn attach<T: FrameMeta + 'static>(&mut self, meta: T) {
            self.0.push(Arc::new(meta));
        }

        /// The first attached meta of type `T`, if any.
        pub fn get<T: FrameMeta + 'static>(&self) -> Option<&T> {
            self.0.iter().find_map(|m| m.as_any().downcast_ref::<T>())
        }

        /// Mutable access to the first attached meta of type `T`, if any.
        ///
        /// Copy-on-write: if the entry is shared with another frame (a tee
        /// branch holds the same [`Arc`]), it is first deep-copied via
        /// [`FrameMeta::clone_box`] so this mutation stays private to this frame.
        /// When the entry is uniquely owned no copy is made.
        pub fn get_mut<T: FrameMeta + 'static>(&mut self) -> Option<&mut T> {
            let idx = self.0.iter().position(|m| m.as_any().is::<T>())?;
            // Ensure unique ownership before handing out a mutable reference.
            if Arc::get_mut(&mut self.0[idx]).is_none() {
                self.0[idx] = Arc::from(self.0[idx].clone_box());
            }
            Arc::get_mut(&mut self.0[idx])
                .expect("entry is unique after the COW above")
                .as_any_mut()
                .downcast_mut::<T>()
        }

        /// Iterate every attached meta as a trait object.
        pub fn iter(&self) -> impl Iterator<Item = &dyn FrameMeta> {
            self.0.iter().map(|m| m.as_ref())
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
            self.0
                .retain(|m| m.propagate(transform) == Propagation::Keep);
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

    /// A per-pixel coverage mask: `width` x `height` 8-bit samples with `stride`
    /// bytes per row (0 = not covered, 255 = fully covered). Its own grid, not
    /// the frame's, so it stays valid when the frame is scaled.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Mask {
        width: u32,
        height: u32,
        stride: u32,
        data: Vec<u8>,
    }

    impl Mask {
        /// Build a mask over `data`, or `None` if the geometry does not fit it.
        /// Dimensions reach this from a model output or a wire peer, so they are
        /// checked here once and the accessors can then index without guessing.
        pub fn new(width: u32, height: u32, stride: u32, data: Vec<u8>) -> Option<Self> {
            if stride < width {
                return None;
            }
            let needed = (stride as u64).checked_mul(height as u64)?;
            if needed > data.len() as u64 {
                return None;
            }
            Some(Mask {
                width,
                height,
                stride,
                data,
            })
        }

        pub fn width(&self) -> u32 {
            self.width
        }
        pub fn height(&self) -> u32 {
            self.height
        }
        pub fn stride(&self) -> u32 {
            self.stride
        }
        pub fn data(&self) -> &[u8] {
            &self.data
        }

        /// Coverage at `(x, y)`, `None` outside the mask.
        pub fn sample(&self, x: u32, y: u32) -> Option<u8> {
            if x >= self.width || y >= self.height {
                return None;
            }
            let idx = y as usize * self.stride as usize + x as usize;
            self.data.get(idx).copied()
        }
    }

    /// An instance segmentation: the object's normalized box, its class, and the
    /// coverage mask over that box (the mask grid is the model's own resolution,
    /// not the frame's).
    #[derive(Debug, Clone, PartialEq)]
    pub struct Segmentation {
        pub bbox: BBox,
        pub label: u32,
        pub confidence: f32,
        pub mask: Mask,
    }

    /// A region of interest: a normalized rectangle an encoder, a tracker, or a
    /// downstream analytic should treat specially (the
    /// `GstVideoRegionOfInterestMeta` analog). `id` names this region across
    /// frames; `label` is its class index, as on a detection.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Roi {
        pub bbox: BBox,
        pub id: u32,
        pub label: u32,
    }

    /// A node in the [`AnalyticsMeta`] relation graph.
    #[derive(Debug, Clone, PartialEq)]
    pub enum AnalyticsNode {
        Detection(ObjectDetection),
        Classification(Classification),
        Tracking(Tracking),
        Segmentation(Segmentation),
        Roi(Roi),
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

        /// Iterate the instance-segmentation nodes.
        pub fn segmentations(&self) -> impl Iterator<Item = &Segmentation> {
            self.nodes.iter().filter_map(|n| match n {
                AnalyticsNode::Segmentation(s) => Some(s),
                _ => None,
            })
        }

        /// Iterate the region-of-interest nodes.
        pub fn rois(&self) -> impl Iterator<Item = &Roi> {
            self.nodes.iter().filter_map(|n| match n {
                AnalyticsNode::Roi(r) => Some(r),
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
        fn clone_box(&self) -> Box<dyn FrameMeta> {
            Box::new(self.clone())
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

    /// One opaque tagged blob: a `header` tag plus a serialized `payload`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Blob {
        pub header: String,
        pub payload: Vec<u8>,
    }

    /// Opaque tagged side-data carried with a frame (the GstMeta custom-blob
    /// analog): serialized results a producer attaches and a specific consumer
    /// decodes by `header`, e.g. an ML embedding's little-endian f32 bytes or a
    /// JSON record. A single `BlobMeta` holds every blob on a frame, since a
    /// [`FrameMetaSet`] keys by concrete type.
    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub struct BlobMeta {
        pub blobs: Vec<Blob>,
    }

    impl BlobMeta {
        pub fn new() -> Self {
            Self::default()
        }

        /// Append a tagged blob.
        pub fn push(&mut self, header: impl Into<String>, payload: Vec<u8>) {
            self.blobs.push(Blob {
                header: header.into(),
                payload,
            });
        }

        /// Iterate the carried blobs in attach order.
        pub fn iter(&self) -> impl Iterator<Item = &Blob> {
            self.blobs.iter()
        }

        pub fn is_empty(&self) -> bool {
            self.blobs.is_empty()
        }

        pub fn len(&self) -> usize {
            self.blobs.len()
        }
    }

    impl FrameMeta for BlobMeta {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn clone_box(&self) -> Box<dyn FrameMeta> {
            Box::new(self.clone())
        }
        // Opaque serialized results are not pixel-coordinate bound, so they
        // survive every transform (including a re-encode); the default `Keep`
        // is correct, stated here for intent.
        fn propagate(&self, _transform: Transform) -> Propagation {
            Propagation::Keep
        }
    }

    /// One closed-caption byte triple: a two-bit `cc_type` and the two caption
    /// data bytes, the ATSC A/53 `cc_data` element. `cc_type` 0/1 are the two
    /// CEA-608 line-21 fields, 2/3 CEA-708 DTVCC packet bytes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CaptionTriple {
        pub cc_type: u8,
        pub b0: u8,
        pub b1: u8,
    }

    /// The closed-caption bytes the coded picture this frame came from carried
    /// (its A/53 `GA94` caption SEI, or a container caption track). Lets a
    /// decode -> re-encode chain re-author captions the decoder would otherwise
    /// have dropped with the bitstream.
    ///
    /// Only the triples are stored: the rest of the A/53 `cc_data` header is
    /// either constant (`process_cc_data_flag`, `em_data`) or derived
    /// (`cc_count` = the triple count), so a rebuilt SEI is byte-identical.
    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    pub struct CaptionMeta {
        pub triples: Vec<CaptionTriple>,
    }

    impl CaptionMeta {
        pub fn new() -> Self {
            Self::default()
        }

        /// Append one caption triple, in transmission order.
        pub fn push(&mut self, triple: CaptionTriple) {
            self.triples.push(triple);
        }

        /// Iterate the carried triples in transmission order.
        pub fn iter(&self) -> impl Iterator<Item = &CaptionTriple> {
            self.triples.iter()
        }

        pub fn is_empty(&self) -> bool {
            self.triples.is_empty()
        }

        pub fn len(&self) -> usize {
            self.triples.len()
        }
    }

    impl FrameMeta for CaptionMeta {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn clone_box(&self) -> Box<dyn FrameMeta> {
            Box::new(self.clone())
        }
        /// Captions are text on a timeline, not pixel geometry, so they survive
        /// a scale / crop / copy *and* a re-encode: the whole point is that a
        /// caption inserter downstream of an encoder can re-author them into the
        /// new bitstream.
        fn propagate(&self, _transform: Transform) -> Propagation {
            Propagation::Keep
        }
    }

    /// A CIE 1931 xy chromaticity.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Chromaticity {
        pub x: f32,
        pub y: f32,
    }

    /// The SMPTE ST 2086 mastering display colour volume: the primaries and white
    /// point of the display the content was graded on, and its luminance range in
    /// cd/m^2.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct MasteringDisplay {
        /// Display primaries in **R, G, B** order (the SEI codes them G, B, R).
        pub display_primaries: [Chromaticity; 3],
        pub white_point: Chromaticity,
        pub max_luminance: f32,
        pub min_luminance: f32,
    }

    /// HDR10 static metadata as carried by the H.264 / H.265
    /// `mastering_display_colour_volume` and `content_light_level_info` SEI
    /// messages: how the content was graded, which a display sink hands to the
    /// driver so the panel maps the highlights the way the colourist saw them.
    ///
    /// Each half is independent: a stream may carry either, both, or (then no meta
    /// is attached at all) neither. The colour primaries / transfer function /
    /// matrix themselves are *not* here: those are CICP codepoints in the SPS VUI
    /// that the decode path already resolves for itself.
    #[derive(Debug, Default, Clone, Copy, PartialEq)]
    pub struct HdrStaticMeta {
        pub mastering: Option<MasteringDisplay>,
        /// MaxCLL: the brightest single pixel in the stream, cd/m^2.
        pub max_content_light_level: Option<u16>,
        /// MaxFALL: the brightest frame average in the stream, cd/m^2.
        pub max_frame_average_light_level: Option<u16>,
    }

    impl HdrStaticMeta {
        /// Whether anything was actually recovered (an all-empty meta is not
        /// worth attaching).
        pub fn is_empty(&self) -> bool {
            self.mastering.is_none()
                && self.max_content_light_level.is_none()
                && self.max_frame_average_light_level.is_none()
        }
    }

    impl FrameMeta for HdrStaticMeta {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn clone_box(&self) -> Box<dyn FrameMeta> {
            Box::new(*self)
        }
        /// A grading description of the content, not of the sample grid: it
        /// survives every transform, including a re-encode (the new bitstream
        /// describes the same graded picture).
        fn propagate(&self, _transform: Transform) -> Propagation {
            Propagation::Keep
        }
    }

    /// The SMPTE ST 12M timecode a coded picture carries (H.264 `pic_timing` /
    /// H.265 `time_code` SEI, or a container timecode track): where this frame
    /// sits on the source's own clock, which is what an edit list, a broadcast
    /// log, or a burnt-in overlay refers to.
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub struct TimecodeMeta {
        pub hours: u8,
        pub minutes: u8,
        pub seconds: u8,
        pub frames: u8,
        /// NTSC drop-frame counting (the 29.97 / 59.94 fps count that skips two
        /// frame numbers a minute). Rendered with a `;` before the frame count.
        pub drop_frame: bool,
        /// Frames per second the count runs at, Q16 fixed point like
        /// [`Rate::Fixed`](crate::Rate::Fixed). `None` when the source declared
        /// none, so a consumer cannot convert the count to a duration.
        pub framerate_q16: Option<u32>,
    }

    impl FrameMeta for TimecodeMeta {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn clone_box(&self) -> Box<dyn FrameMeta> {
            Box::new(*self)
        }
        /// A position on the source's clock: unchanged by any pixel work, and it
        /// is exactly what a re-encode should carry into the new bitstream.
        fn propagate(&self, _transform: Transform) -> Propagation {
            Propagation::Keep
        }
    }
}

#[cfg(all(test, feature = "metadata"))]
mod tests {
    use super::*;

    fn det(x: f32, y: f32, w: f32, h: f32, label: u32, conf: f32) -> ObjectDetection {
        ObjectDetection {
            bbox: BBox { x, y, w, h },
            label,
            confidence: conf,
        }
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
        let c = a.push(AnalyticsNode::Classification(Classification {
            label: 42,
            confidence: 0.7,
        }));
        a.relate(d, c, RelationKind::Classifies);
        assert_eq!(a.relations.len(), 1);
        assert_eq!(
            a.relations[0],
            Relation {
                from: d,
                to: c,
                kind: RelationKind::Classifies
            }
        );
    }

    #[test]
    fn clone_shares_then_get_mut_copies_on_write() {
        // A tee clone shares the analytics graph by Arc; mutating one side must
        // not leak into the other (copy-on-write deep copy on get_mut).
        let mut a = FrameMetaSet::new();
        a.attach({
            let mut m = AnalyticsMeta::new();
            m.add_detection(det(0.1, 0.1, 0.2, 0.2, 7, 0.9));
            m
        });
        let mut b = a.clone();
        assert_eq!(a.get::<AnalyticsMeta>().unwrap().nodes.len(), 1);
        assert_eq!(b.get::<AnalyticsMeta>().unwrap().nodes.len(), 1);

        // Mutate the clone: COW splits the shared entry.
        b.get_mut::<AnalyticsMeta>()
            .unwrap()
            .add_detection(det(0.5, 0.5, 0.1, 0.1, 3, 0.8));
        assert_eq!(
            b.get::<AnalyticsMeta>().unwrap().nodes.len(),
            2,
            "clone mutated"
        );
        assert_eq!(
            a.get::<AnalyticsMeta>().unwrap().nodes.len(),
            1,
            "original untouched after copy-on-write"
        );
    }

    #[test]
    fn iou_is_zero_for_disjoint_and_one_for_identical() {
        let a = BBox {
            x: 0.0,
            y: 0.0,
            w: 0.2,
            h: 0.2,
        };
        let b = BBox {
            x: 0.5,
            y: 0.5,
            w: 0.2,
            h: 0.2,
        };
        assert_eq!(a.iou(&b), 0.0, "disjoint boxes do not overlap");
        assert!(
            (a.iou(&a) - 1.0).abs() < 1e-6,
            "identical boxes fully overlap"
        );
        // Half-overlap: a and c share half their area horizontally.
        let c = BBox {
            x: 0.1,
            y: 0.0,
            w: 0.2,
            h: 0.2,
        };
        let iou = a.iou(&c);
        assert!(
            iou > 0.3 && iou < 0.34,
            "half-shifted overlap ~1/3 IoU: {iou}"
        );
    }
}
