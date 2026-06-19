use crate::caps::Caps;
use crate::memory::MemoryDomain;
use crate::segment::Segment;

#[derive(Debug)]
pub enum PipelinePacket {
    CapsChanged(Caps),
    DataFrame(Frame),
    Eos,
    /// Seek flush: discard in-flight and buffered data and reset position
    /// state. Unlike `Eos`, the stream resumes after a flush, so elements
    /// reset rather than terminate.
    Flush,
    /// The playback [`Segment`] in force for subsequent `DataFrame`s (M80, the
    /// GStreamer SEGMENT-event analog). Like `CapsChanged` it is **ordered** in
    /// the stream: it sits before the first `DataFrame` it governs, so a sink
    /// maps each frame's timestamp to running time via the most recent
    /// `Segment`. Every stream opens with one; a flushing seek emits a fresh
    /// one after the `Flush`. Elements forward it downstream unchanged unless
    /// they remap time.
    Segment(Segment),
}

#[derive(Debug)]
pub struct Frame {
    pub domain: MemoryDomain,
    pub timing: FrameTiming,
    pub sequence: u64,
    /// Reserved per-frame metadata side-channel: typed blobs that travel with
    /// the buffer (the GstMeta / GstAnalyticsRelationMeta analog). Empty on
    /// construction. A zero-sized unit when the `metadata` feature is off, so
    /// the no_std / RTOS baseline pays nothing. The trait body and the
    /// relation-graph layer are deferred until a real detection element needs
    /// them (see [`FrameMetaSet`] and DESIGN_TODO "Per-frame metadata system").
    pub meta: FrameMetaSet,
}

impl Frame {
    /// Construct a frame with an empty metadata set. Prefer this over the bare
    /// struct literal at new construction sites so a future `Frame` field
    /// addition does not break them.
    #[inline]
    pub fn new(domain: MemoryDomain, timing: FrameTiming, sequence: u64) -> Self {
        Frame { domain, timing, sequence, meta: FrameMetaSet::new() }
    }
}

/// Per-frame attachable metadata set, the reserved extension point on
/// [`Frame`]. Gated behind the `metadata` cargo feature.
///
/// When the feature is **off** (the default, and the only configuration the
/// `no_std` / Cortex-M path uses) this is a zero-sized unit: the field exists
/// for API stability but costs nothing per frame. When **on** it carries a
/// list of typed [`FrameMeta`] trait objects. Either way it is empty on a
/// freshly constructed frame; populating it (and the GstMeta-style
/// transform / copy / free propagation contract) lands with the first
/// metadata-producing element.
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

/// A typed, per-frame, attachable piece of metadata (the `GstMeta` analog).
///
/// Minimal shell for now: the propagation contract (GstMeta's
/// `transform_func` / `copy_func` / `free_func`, expressed in Rust as a
/// `propagate(kind) -> Propagation` method) and the `AnalyticsMeta`
/// relation-graph layer land with the first real meta type. The trait exists
/// so the [`FrameMetaSet`] field can hold something concrete when the
/// `metadata` feature is enabled.
#[cfg(feature = "metadata")]
pub trait FrameMeta: core::fmt::Debug + Send + Sync {}

#[cfg(feature = "metadata")]
#[derive(Debug, Default)]
// The backing list is intentionally write-only until the metadata API (attach /
// iterate / propagate) lands with the first meta-producing element; reserved
// extension point, see the type doc above.
pub struct FrameMetaSet(#[allow(dead_code)] alloc::vec::Vec<alloc::boxed::Box<dyn FrameMeta>>);

#[cfg(feature = "metadata")]
impl FrameMetaSet {
    /// An empty metadata set with no backing allocation.
    #[inline]
    pub fn new() -> Self {
        FrameMetaSet(alloc::vec::Vec::new())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameTiming {
    pub pts_ns: u64,
    pub dts_ns: u64,
    pub duration_ns: u64,
    /// Media-clock capture time (e.g. RTP-derived). Stream-relative.
    pub capture_ns: u64,
    /// Wall-clock monotonic nanoseconds stamped at source ingestion,
    /// using the process-wide epoch from `metrics::monotonic_ns`. The
    /// glass-to-glass latency is `sink_now - arrival_ns`. Zero on
    /// frames synthesized by transforms or unit tests.
    pub arrival_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{MemoryDomain, SystemSlice};
    use alloc::boxed::Box;

    fn frame() -> Frame {
        Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(Box::new([0u8; 4]))),
            FrameTiming::default(),
            0,
        )
    }

    #[test]
    fn new_constructs_a_frame_with_an_empty_meta_set() {
        // The constructor is the future-proof path: it fills `meta` so new
        // call sites do not break when more fields land.
        let f = frame();
        assert_eq!(f.sequence, 0);
        // `meta` is empty either way; this also pins that `Frame::new` and the
        // struct literal stay in sync (a compile check more than a value check).
        let _ = f.meta;
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn metadata_on_set_is_an_empty_container() {
        // With the feature on, `FrameMetaSet` is the Vec-backed container, not
        // the ZST. A fresh frame still carries nothing.
        let f = frame();
        assert!(f.meta.0.is_empty(), "a fresh frame's metadata set is empty");
    }
}
