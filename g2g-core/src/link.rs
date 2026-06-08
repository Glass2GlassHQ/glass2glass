/// Per-link backpressure policy. Chosen at graph construction time because a
/// single pipeline may have lossy preview branches and lossless recording
/// branches sharing an upstream source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkPolicy {
    /// Block the upstream future until the channel has capacity. Lossless;
    /// raises latency under load.
    Block,
    /// Drop the oldest queued frame on downstream stall. Default for live
    /// camera sources.
    DropOldest,
    /// Drop the newest (incoming) frame on downstream stall. Use when temporal
    /// coherence matters more than freshness.
    DropNewest,
}
