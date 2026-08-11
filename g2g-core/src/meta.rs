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

/// The metadata types a downstream element asks its producers to attach
/// (feature `metadata` **off**): a zero-sized always-empty set. The plumbing
/// that carries it ([`AllocationParams`](crate::AllocationParams)) compiles
/// either way; `request` / `wants` exist only with the feature on.
#[cfg(not(feature = "metadata"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MetaRequests;

#[cfg(not(feature = "metadata"))]
impl MetaRequests {
    /// An empty request set.
    #[inline]
    pub const fn new() -> Self {
        MetaRequests
    }

    /// Always true: without the feature nothing can be requested.
    #[inline]
    pub fn is_empty(&self) -> bool {
        true
    }

    /// The demand two sibling consumers put on their shared producer, i.e. still
    /// nothing.
    #[inline]
    pub fn join_branches(self, _other: Self) -> Self {
        self
    }

    /// The demand this element passes to its producer, i.e. still nothing.
    #[inline]
    pub fn carry_upstream(self, _downstream: Self) -> Self {
        self
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
    use core::any::{Any, TypeId};

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

        /// Attach one piece of metadata, replacing any entry of the same type
        /// (in place, so the order of the other entries is unchanged).
        ///
        /// A set holds at most one meta per type: [`get`](Self::get) /
        /// [`get_mut`](Self::get_mut) key by type, so a second entry of a type
        /// would be unreachable, and an empty one already on the frame would
        /// hide what an element attaches later.
        pub fn attach<T: FrameMeta + 'static>(&mut self, meta: T) {
            match self.0.iter().position(|m| m.as_any().is::<T>()) {
                Some(idx) => self.0[idx] = Arc::new(meta),
                None => self.0.push(Arc::new(meta)),
            }
        }

        /// The attached meta of type `T`, if any. At most one is ever attached
        /// (see [`attach`](Self::attach)).
        pub fn get<T: FrameMeta + 'static>(&self) -> Option<&T> {
            self.0.iter().find_map(|m| m.as_any().downcast_ref::<T>())
        }

        /// Mutable access to the attached meta of type `T`, if any (at most one
        /// is ever attached, see [`attach`](Self::attach)).
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

    /// How many distinct meta types one [`MetaRequests`] carries. Requests past
    /// this are dropped, which costs an optimization, never correctness: a
    /// producer that sees no request just produces what it always did.
    pub const MAX_META_REQUESTS: usize = 4;

    /// What one request needs of the *other* consumers reading the same frames,
    /// which decides how it survives a fan-out or an intermediate hop.
    ///
    /// `Ord` ranks [`EveryConsumer`](Self::EveryConsumer) above
    /// [`AnyConsumer`](Self::AnyConsumer): when two elements request one meta
    /// under different policies the stricter one stands, since it is the one
    /// that can be misread.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub enum RequestPolicy {
        /// One asking consumer is enough. Attaching the meta costs a consumer
        /// that did not ask nothing: it reads the same frame it always did and
        /// ignores the extra ([`AnalyticsMeta`], [`CaptionMeta`],
        /// [`TimecodeMeta`]).
        AnyConsumer,
        /// Every consumer must ask. Honouring the request changes the *buffer*,
        /// so a consumer that did not ask would misread it: a frame whose rows
        /// were left padded, read as tightly packed, is corruption rather than a
        /// missed optimization.
        EveryConsumer,
    }

    /// The metadata types a downstream element wants attached to the frames it
    /// receives, each keyed by [`TypeId`] and carrying its [`RequestPolicy`].
    /// The pull half of the metadata system (the GStreamer allocation-query
    /// `add_meta` analog): a consumer declares its requests from
    /// [`AsyncElement::meta_requests`](crate::AsyncElement::meta_requests), the
    /// runner carries them up the allocation cascade on
    /// [`AllocationParams`](crate::AllocationParams), and a producer asks
    /// [`wants`](Self::wants) when it configures, so optional metadata is
    /// produced only where somebody reads it.
    ///
    /// A small fixed-capacity set, so it rides the `Copy` allocation params
    /// without an allocation. Entries are kept sorted, so two sets built in
    /// different orders compare equal (the cascade suppresses a re-propose when
    /// the params are unchanged).
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct MetaRequests {
        entries: [Option<(TypeId, RequestPolicy)>; MAX_META_REQUESTS],
    }

    impl MetaRequests {
        /// An empty request set: the element wants no optional metadata.
        pub const fn new() -> Self {
            MetaRequests {
                entries: [None; MAX_META_REQUESTS],
            }
        }

        /// This set plus a request for meta `T` that one consumer asking is
        /// enough for ([`RequestPolicy::AnyConsumer`]). Builder form, since a
        /// request set is usually written inline:
        /// `MetaRequests::new().request::<AnalyticsMeta>()`.
        pub fn request<T: FrameMeta + 'static>(self) -> Self {
            self.with(TypeId::of::<T>(), RequestPolicy::AnyConsumer)
        }

        /// This set plus a request for meta `T` that is only honoured when every
        /// consumer sharing the producer asks for it too
        /// ([`RequestPolicy::EveryConsumer`]). For a meta whose presence changes
        /// the buffer, which a consumer that did not ask would misread.
        pub fn request_from_every_consumer<T: FrameMeta + 'static>(self) -> Self {
            self.with(TypeId::of::<T>(), RequestPolicy::EveryConsumer)
        }

        /// Whether meta `T` was requested, under either policy.
        pub fn wants<T: FrameMeta + 'static>(&self) -> bool {
            self.policy_of(TypeId::of::<T>()).is_some()
        }

        /// The policy meta `T` was requested under, `None` if it was not.
        pub fn policy<T: FrameMeta + 'static>(&self) -> Option<RequestPolicy> {
            self.policy_of(TypeId::of::<T>())
        }

        /// The demand two *sibling* consumers put on the one producer they share
        /// (the branches of a tee). An [`AnyConsumer`](RequestPolicy::AnyConsumer)
        /// request survives from either side; an
        /// [`EveryConsumer`](RequestPolicy::EveryConsumer) one only when the
        /// other branch asks for that meta too, so a branch that would misread
        /// the changed buffer vetoes it.
        pub fn join_branches(self, other: Self) -> Self {
            let mut out = Self::new();
            for (id, policy) in self.iter().chain(other.iter()) {
                if policy == RequestPolicy::AnyConsumer
                    || (self.policy_of(id).is_some() && other.policy_of(id).is_some())
                {
                    out = out.with(id, policy);
                }
            }
            out
        }

        /// The demand this element (`self`, its own requests) passes on to its
        /// producer, given what arrived from `downstream`. Its own requests
        /// always travel: it reads the producer's frames itself. A downstream
        /// [`EveryConsumer`](RequestPolicy::EveryConsumer) request travels only
        /// when this element asks for that meta too, since the producer's frames
        /// pass through here first and a hop that cannot read the changed buffer
        /// vetoes it just as a sibling branch does.
        pub fn carry_upstream(self, downstream: Self) -> Self {
            let mut out = self;
            for (id, policy) in downstream.iter() {
                if policy == RequestPolicy::AnyConsumer || self.policy_of(id).is_some() {
                    out = out.with(id, policy);
                }
            }
            out
        }

        pub fn is_empty(&self) -> bool {
            self.entries[0].is_none()
        }

        pub fn len(&self) -> usize {
            self.entries.iter().flatten().count()
        }

        fn iter(&self) -> impl Iterator<Item = (TypeId, RequestPolicy)> + '_ {
            self.entries.iter().flatten().copied()
        }

        fn policy_of(&self, id: TypeId) -> Option<RequestPolicy> {
            self.iter().find(|(i, _)| *i == id).map(|(_, p)| p)
        }

        fn with(mut self, id: TypeId, policy: RequestPolicy) -> Self {
            let mut free = MAX_META_REQUESTS;
            for (i, slot) in self.entries.iter_mut().enumerate() {
                match slot {
                    Some((present, held)) if *present == id => {
                        // Two elements asking for one meta under different
                        // policies: the stricter one is the one that can be
                        // misread, so it stands.
                        *held = (*held).max(policy);
                        return self;
                    }
                    Some(_) => {}
                    None => {
                        free = i;
                        break;
                    }
                }
            }
            if free == MAX_META_REQUESTS {
                return self;
            }
            self.entries[free] = Some((id, policy));
            // The occupied prefix is packed at the front, so sorting it keeps it
            // packed and makes the set order-independent under `PartialEq`.
            self.entries[..=free].sort_unstable();
            self
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

    /// One inference output riding along with the frame it was computed from.
    ///
    /// Carries what a `Caps::Tensor` link would have carried, since it stands in
    /// for exactly that: the descriptor plus the raw little-endian bytes. `name`
    /// says which stage produced it, so a frame can hold the outputs of several
    /// models at once; it comes from the producing element's configuration, not
    /// from anything read out of a model file, so every inference backend tags
    /// its output the same way. Empty is the ordinary single-model case.
    #[derive(Debug, Clone, PartialEq)]
    pub struct NamedTensor {
        pub name: String,
        pub dtype: crate::caps::TensorDType,
        pub shape: crate::caps::TensorShape,
        pub layout: crate::caps::TensorLayout,
        pub data: Vec<u8>,
    }

    /// The inference outputs attached to a frame, for the elements that keep the
    /// picture on the wire instead of replacing it with the tensor. That is what
    /// lets `inference -> post-process -> overlay` stay one straight chain: the
    /// frame reaching the overlay is still the video the model saw.
    ///
    /// Holds every tensor on the frame, since a [`FrameMetaSet`] keys by concrete
    /// type and would otherwise let a second model's output replace the first.
    /// These are not detections; those are the post-processor's [`AnalyticsMeta`].
    #[derive(Debug, Default, Clone, PartialEq)]
    pub struct TensorMeta(Vec<NamedTensor>);

    impl TensorMeta {
        /// An empty set, the starting point a producer pushes onto.
        pub fn new() -> Self {
            TensorMeta(Vec::new())
        }

        /// Add one output, replacing any earlier tensor of the same name so a
        /// re-run of the same stage updates rather than accumulates.
        pub fn push(&mut self, tensor: NamedTensor) {
            match self.0.iter().position(|t| t.name == tensor.name) {
                Some(idx) => self.0[idx] = tensor,
                None => self.0.push(tensor),
            }
        }

        /// Every attached tensor, in the order the stages produced them.
        pub fn iter(&self) -> impl Iterator<Item = &NamedTensor> {
            self.0.iter()
        }

        /// The tensor named `name`.
        pub fn get(&self, name: &str) -> Option<&NamedTensor> {
            self.0.iter().find(|t| t.name == name)
        }

        /// The tensor when the frame carries exactly one, so the ordinary
        /// single-model pipeline needs no names. `None` when there are several:
        /// the consumer must then say which it wants rather than be handed an
        /// arbitrary one.
        pub fn only(&self) -> Option<&NamedTensor> {
            match self.0.as_slice() {
                [one] => Some(one),
                _ => None,
            }
        }

        pub fn is_empty(&self) -> bool {
            self.0.is_empty()
        }
    }

    impl FrameMeta for TensorMeta {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn clone_box(&self) -> Box<dyn FrameMeta> {
            Box::new(self.clone())
        }
        /// The tensors describe the picture the models saw, so a re-encode ends
        /// their usefulness the way it ends an analytics graph's. A scale or crop
        /// leaves them readable: what they mean is fixed by the model input size
        /// the post-processor normalizes against, not the frame's current size.
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

        /// The first blob tagged `header`, if any.
        pub fn get(&self, header: &str) -> Option<&Blob> {
            self.blobs.iter().find(|b| b.header == header)
        }
    }

    /// A [`Blob`] payload decoded by the [`BLOB_DECODERS`] registry.
    #[derive(Debug, Clone, PartialEq)]
    pub enum DecodedBlob {
        /// A little-endian `f32` vector (an ML embedding / feature vector).
        Embedding(Vec<f32>),
        /// UTF-8 text.
        Text(String),
    }

    /// Turns one known header's payload into a [`DecodedBlob`], or `None` when
    /// the bytes do not match the shape that header promises.
    pub type BlobDecoder = fn(&[u8]) -> Option<DecodedBlob>;

    fn decode_embedding(payload: &[u8]) -> Option<DecodedBlob> {
        if payload.is_empty() || payload.len() % 4 != 0 {
            return None;
        }
        Some(DecodedBlob::Embedding(
            payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        ))
    }

    fn decode_text(payload: &[u8]) -> Option<DecodedBlob> {
        core::str::from_utf8(payload)
            .ok()
            .map(|s| DecodedBlob::Text(String::from(s)))
    }

    /// The blob headers this workspace knows how to decode, and their decoders.
    ///
    /// [`BlobMeta`] is deliberately opaque: a producer and its own consumer agree
    /// on a header and nobody else has to care. This table is the escape hatch
    /// for the headers that *are* a shared vocabulary, so a generic consumer (an
    /// inspector, a bridge, a recorder) can render them without knowing which
    /// element produced them. These three come from the Python element host,
    /// where a `gst-python-ml` element tags its results with them.
    ///
    /// A plain `const` table, not a registry a plugin mutates: the decoders are
    /// pure functions, and a global mutable map would need a lock the `no_std`
    /// baseline does not have.
    pub const BLOB_DECODERS: &[(&str, BlobDecoder)] = &[
        ("embedding", decode_embedding as BlobDecoder),
        ("model_name", decode_text as BlobDecoder),
        ("device", decode_text as BlobDecoder),
    ];

    /// The decoder registered for `header`, if it is a known one.
    pub fn blob_decoder(header: &str) -> Option<BlobDecoder> {
        BLOB_DECODERS
            .iter()
            .find(|(h, _)| *h == header)
            .map(|(_, d)| *d)
    }

    /// Decode `blob` if its header is known and its payload matches the shape
    /// that header promises. `None` for an unknown header (the normal case for
    /// an application's private side-data) and for a malformed payload.
    pub fn decode_blob(blob: &Blob) -> Option<DecodedBlob> {
        blob_decoder(&blob.header).and_then(|d| d(&blob.payload))
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

    /// How many planes a [`PlaneLayout`] describes. Four covers every format in
    /// the workspace (planar YUV with alpha is the widest).
    pub const MAX_PLANES: usize = 4;

    /// Where one plane's rows sit in the frame's buffer.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Plane {
        /// Byte offset of the plane's first row from the start of the buffer.
        pub offset: usize,
        /// Bytes from the start of one row to the start of the next. At least
        /// the row's own byte width; more when the rows are padded.
        pub stride: usize,
    }

    /// Where each plane's rows really sit in a raw video frame's buffer (the
    /// `GstVideoMeta` analog). Without it a raw frame is assumed tightly packed:
    /// every row exactly `width * bytes_per_pixel` and every plane immediately
    /// after the last. A producer whose rows are padded (a GPU readback at the
    /// API's 256-byte row alignment, a capture driver's `bytesperline`) has to
    /// repack them into that shape, row by row, before pushing the frame.
    ///
    /// A consumer that asks for this meta
    /// ([`MetaRequests`](crate::meta::MetaRequests)) says it will read rows where
    /// they lie, so the producer can hand over the padded buffer as it is and the
    /// repack disappears.
    ///
    /// Only the geometry of the *buffer* is described here, never the picture:
    /// width, height and format stay in the caps.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PlaneLayout {
        planes: [Plane; MAX_PLANES],
        count: usize,
    }

    impl PlaneLayout {
        /// Describe `planes` (1 to [`MAX_PLANES`] of them), or `None` for an
        /// empty / oversized list.
        pub fn new(planes: &[Plane]) -> Option<Self> {
            if planes.is_empty() || planes.len() > MAX_PLANES {
                return None;
            }
            let mut slots = [Plane {
                offset: 0,
                stride: 0,
            }; MAX_PLANES];
            slots[..planes.len()].copy_from_slice(planes);
            Some(PlaneLayout {
                planes: slots,
                count: planes.len(),
            })
        }

        /// One plane at `offset` 0 with row pitch `stride`: the packed-format
        /// case (RGBA, YUYV), which is most of what pads rows in practice.
        pub fn single(stride: usize) -> Self {
            PlaneLayout {
                planes: [Plane { offset: 0, stride }; MAX_PLANES],
                count: 1,
            }
        }

        pub fn count(&self) -> usize {
            self.count
        }

        /// Plane `index`, or `None` past the described ones.
        pub fn plane(&self, index: usize) -> Option<Plane> {
            (index < self.count).then(|| self.planes[index])
        }

        /// Byte range of row `row` of plane `index`, `row_bytes` wide. `None`
        /// when the plane does not exist, the stride cannot hold the row, or the
        /// arithmetic overflows: a layout can come off a wire or a driver, so
        /// every offset derived from it is checked here once and a caller can
        /// then slice with what it gets back.
        pub fn row_range(
            &self,
            index: usize,
            row: usize,
            row_bytes: usize,
        ) -> Option<core::ops::Range<usize>> {
            let plane = self.plane(index)?;
            if plane.stride < row_bytes {
                return None;
            }
            let start = plane.offset.checked_add(row.checked_mul(plane.stride)?)?;
            let end = start.checked_add(row_bytes)?;
            Some(start..end)
        }
    }

    impl FrameMeta for PlaneLayout {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn clone_box(&self) -> Box<dyn FrameMeta> {
            Box::new(*self)
        }
        /// Dropped by every transform: it describes one specific buffer, and an
        /// element only declares a [`Transform`] when it writes a *new* one (a
        /// videoconvert says `Copy` and still emits its own tightly-packed
        /// frame). A tee branch, which shares the very buffer this describes,
        /// clones the meta set without applying a transform, so the layout
        /// survives a fan-out.
        fn propagate(&self, _transform: Transform) -> Propagation {
            Propagation::Drop
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
    fn attach_replaces_the_same_type_and_keeps_other_types() {
        let mut set = FrameMetaSet::new();
        set.attach(AnalyticsMeta::new());
        set.attach(BlobMeta::new());

        let mut second = AnalyticsMeta::new();
        second.add_detection(det(0.1, 0.1, 0.2, 0.2, 7, 0.9));
        set.attach(second);

        assert_eq!(set.len(), 2, "the replacement is not a second entry");
        assert_eq!(
            set.get::<AnalyticsMeta>().unwrap().detections().count(),
            1,
            "the meta attached last is the one that can be read back"
        );
        assert!(
            set.get::<BlobMeta>().is_some(),
            "another type is untouched by the replacement"
        );
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
    fn known_blob_headers_decode_to_typed_values() {
        let mut m = BlobMeta::new();
        m.push("embedding", alloc::vec![0, 0, 0x80, 0x3F, 0, 0, 0, 0x40]); // 1.0, 2.0
        m.push("device", alloc::vec![b'c', b'u', b'd', b'a', b':', b'0']);
        m.push("private/thing", alloc::vec![0xDE, 0xAD]);

        assert_eq!(
            decode_blob(m.get("embedding").unwrap()),
            Some(DecodedBlob::Embedding(alloc::vec![1.0, 2.0]))
        );
        assert_eq!(
            decode_blob(m.get("device").unwrap()),
            Some(DecodedBlob::Text(alloc::string::String::from("cuda:0")))
        );
        // An unregistered header stays opaque, which is the point of BlobMeta.
        assert!(blob_decoder("private/thing").is_none());
        assert_eq!(decode_blob(m.get("private/thing").unwrap()), None);
    }

    #[test]
    fn a_payload_that_does_not_match_its_header_does_not_decode() {
        // A registered header is a promise about the bytes, not a guarantee: a
        // producer that breaks it must fail the decode, not yield garbage.
        let ragged = Blob {
            header: alloc::string::String::from("embedding"),
            payload: alloc::vec![1, 2, 3],
        };
        assert_eq!(decode_blob(&ragged), None, "not a whole number of f32s");
        let empty = Blob {
            header: alloc::string::String::from("embedding"),
            payload: alloc::vec::Vec::new(),
        };
        assert_eq!(decode_blob(&empty), None, "an empty vector is not a vector");
        let not_utf8 = Blob {
            header: alloc::string::String::from("model_name"),
            payload: alloc::vec![0xFF, 0xFE],
        };
        assert_eq!(decode_blob(&not_utf8), None);
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
