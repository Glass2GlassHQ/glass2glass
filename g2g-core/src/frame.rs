use crate::caps::Caps;
use crate::memory::MemoryDomain;
use crate::meta::FrameMetaSet;
use crate::segment::Segment;

#[derive(Debug)]
#[non_exhaustive]
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
    /// Per-frame metadata side-channel: typed blobs that travel with the buffer
    /// (the GstMeta / GstAnalyticsRelationMeta analog). Empty on construction. A
    /// zero-sized unit when the `metadata` feature is off, so the no_std / RTOS
    /// baseline pays nothing; with the feature on it is a typed
    /// [`FrameMetaSet`](crate::meta::FrameMetaSet) carrying e.g.
    /// [`AnalyticsMeta`](crate::meta) detections. See `crate::meta`.
    pub meta: FrameMetaSet,
}

impl Frame {
    /// Construct a frame with an empty metadata set. Prefer this over the bare
    /// struct literal at new construction sites so a future `Frame` field
    /// addition does not break them.
    #[inline]
    pub fn new(domain: MemoryDomain, timing: FrameTiming, sequence: u64) -> Self {
        Frame {
            domain,
            timing,
            sequence,
            meta: FrameMetaSet::new(),
        }
    }

    /// Duplicate this frame for fan-out (a tee branch, or a multicast route).
    /// The buffer is shared where the memory domain allows it: GPU handles and
    /// pre-shared `System` bytes are refcounted (cheap), owned `System` bytes are
    /// deep-copied (the honest cost of handing CPU bytes to a second consumer).
    /// Per-frame metadata is shared by `Arc` refcount, with copy-on-write on the
    /// branch that mutates it, so the copies never alias. `Frame` is deliberately
    /// not `Clone` (owned CPU bytes make a silent clone a surprise cost); this is
    /// the explicit, named fan-out primitive instead. See
    /// [`MemoryDomain::share`](crate::MemoryDomain::share).
    ///
    /// Fan-out is a heap operation (the domain `share` refcounts or deep-copies),
    /// so it is gated behind `alloc`; the heap-free MCU path is a single linear
    /// chain with no tee.
    #[cfg(feature = "alloc")]
    pub fn share(&self) -> Frame {
        // When `metadata` is off, FrameMetaSet is a Copy ZST and this is a no-op
        // copy (clippy's clone_on_copy fires only in that config).
        #[allow(clippy::clone_on_copy)]
        let meta = self.meta.clone();
        Frame {
            domain: self.domain.share(),
            timing: self.timing,
            sequence: self.sequence,
            meta,
        }
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
    /// Whether this frame begins an independently-decodable unit (a keyframe /
    /// IDR). Set by the parsers that detect it; consumed by trick-mode KEY_UNIT
    /// playback (the sink drops non-keyframes under a `TRICKMODE` segment) and by
    /// keyframe-aware seeking. `false` when unknown (the safe default: a frame is
    /// treated as a dependent frame unless a producer marks it a keyframe).
    pub keyframe: bool,
}

#[cfg(all(test, feature = "alloc"))]
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

    #[test]
    fn share_duplicates_bytes_and_metadata_for_fanout() {
        // The fan-out primitive: a shared frame carries the same bytes, timing and
        // sequence, and (for owned System memory) an independent copy, so a branch
        // that mutates one does not disturb the other.
        let mut orig = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(Box::new([1u8, 2, 3, 4]))),
            FrameTiming {
                pts_ns: 42,
                ..FrameTiming::default()
            },
            7,
        );
        let dup = orig.share();
        assert_eq!(dup.sequence, 7);
        assert_eq!(dup.timing.pts_ns, 42);
        match (&mut orig.domain, &dup.domain) {
            (MemoryDomain::System(a), MemoryDomain::System(b)) => {
                assert_eq!(a.as_slice(), b.as_slice(), "the copy sees the same bytes");
                a.as_mut_slice()[0] = 99;
                assert_eq!(
                    b.as_slice()[0],
                    1,
                    "owned CPU bytes are deep-copied, not aliased"
                );
            }
            _ => panic!("expected System memory on both"),
        }
    }

    #[cfg(feature = "metadata")]
    #[test]
    fn metadata_on_set_is_an_empty_container() {
        // With the feature on, `FrameMetaSet` is the Vec-backed container, not
        // the ZST. A fresh frame still carries nothing.
        let f = frame();
        assert!(f.meta.is_empty(), "a fresh frame's metadata set is empty");
    }
}
