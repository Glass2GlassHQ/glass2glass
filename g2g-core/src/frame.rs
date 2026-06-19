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
