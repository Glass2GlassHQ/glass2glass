//! Pipeline lifecycle states (M76).
//!
//! Mirrors GStreamer's `NULL → READY → PAUSED → PLAYING` ladder. The state
//! enum and the change-return code are pure data (no allocation, no OS), so
//! they live in the `no_std` baseline and the bus can carry a
//! [`BusMessage::StateChanged`](crate::BusMessage::StateChanged). The
//! controller that elements / runners actually gate on lives in
//! [`crate::runtime::StateController`] (feature `runtime`, needs a `Waker`
//! registry).
//!
//! Semantics, matching GStreamer where it costs nothing to:
//! - **`Null`** — no resources, no data flow. The torn-down / initial state.
//! - **`Ready`** — resources acquired, no data flow, no clock.
//! - **`Paused`** — data flows up to the sink, which holds (preroll). The
//!   pipeline is ready to play instantly. A non-live pipeline takes exactly one
//!   preroll buffer here (M77); a live pipeline takes none.
//! - **`Playing`** — the clock runs and data flows to presentation.
//!
//! The variants are ordered `Null < Ready < Paused < Playing` so a gate can
//! ask "are we at least `Playing`?" with a comparison.

/// Lifecycle state of a pipeline. Ordered low-to-high along the
/// `NULL → READY → PAUSED → PLAYING` ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PipelineState {
    /// No resources, no data flow. Initial and torn-down state.
    Null,
    /// Resources acquired; no data flow, no running clock.
    Ready,
    /// Data flows to the sink, which holds it (preroll-ready).
    Paused,
    /// Clock running, data flowing to presentation.
    Playing,
}

impl PipelineState {
    /// Stable `u8` encoding for storage in an `AtomicU8`. Matches the ladder
    /// order so the atomic value is directly comparable.
    pub fn as_u8(self) -> u8 {
        match self {
            PipelineState::Null => 0,
            PipelineState::Ready => 1,
            PipelineState::Paused => 2,
            PipelineState::Playing => 3,
        }
    }

    /// Inverse of [`PipelineState::as_u8`]. Any out-of-range byte saturates to
    /// `Playing`; the value only ever comes from `as_u8`, so this is a total
    /// function for paranoia, not a real branch.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => PipelineState::Null,
            1 => PipelineState::Ready,
            2 => PipelineState::Paused,
            _ => PipelineState::Playing,
        }
    }
}

/// Outcome of a requested state change, mirroring GStreamer's
/// `GstStateChangeReturn`.
///
/// A non-live `Paused` returns `Async` (the change completes when the sink
/// prerolls); a live `Paused` returns `NoPreroll` (no preroll buffer is
/// coming); every other target returns `Success` (M77).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateChangeReturn {
    /// The change took effect immediately.
    Success,
    /// The change is in progress: a non-live sink is prerolling. Completes on
    /// [`BusMessage::AsyncDone`](crate::BusMessage::AsyncDone) /
    /// [`StateController::await_prerolled`](crate::runtime::StateController::await_prerolled).
    Async,
    /// The change succeeded but no preroll buffer is expected (live pipeline).
    NoPreroll,
    /// The change was refused.
    Failure,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_is_ordered() {
        assert!(PipelineState::Null < PipelineState::Ready);
        assert!(PipelineState::Ready < PipelineState::Paused);
        assert!(PipelineState::Paused < PipelineState::Playing);
    }

    #[test]
    fn u8_round_trips_every_variant() {
        for s in [
            PipelineState::Null,
            PipelineState::Ready,
            PipelineState::Paused,
            PipelineState::Playing,
        ] {
            assert_eq!(PipelineState::from_u8(s.as_u8()), s);
        }
    }
}
