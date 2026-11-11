// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;

use g2g_core::{Frame, PropError, PropKind, PropValue, PropertySpec};

// gst's aggregator advances 40 ms at a time while it knows no frame rate
pub(crate) const UNKNOWN_FRAME_PERIOD_TICK_NS: u64 = 40_000_000;

pub(crate) const LIVE_MUX_LATENCY_PROPERTY: PropertySpec = PropertySpec::new(
    "latency",
    PropKind::Uint,
    "nanoseconds a live mux waits past a frame's end for late inputs",
)
.with_default("0");

pub(crate) static LIVE_MUX_PROPS: &[PropertySpec] = &[LIVE_MUX_LATENCY_PROPERTY];

#[cfg(feature = "std")]
pub(crate) fn wall_clock_ns() -> Option<u64> {
    Some(g2g_core::metrics::monotonic_ns())
}

#[cfg(not(feature = "std"))]
pub(crate) fn wall_clock_ns() -> Option<u64> {
    None
}

#[derive(Debug, Default)]
pub(crate) struct LiveMuxDeadline {
    live: bool,
    latency_ns: u64,
    // the most any of each input's frames arrived behind its pts, as wall time minus pts
    arrival_offsets_ns: Vec<Option<i128>>,
}

impl LiveMuxDeadline {
    pub(crate) fn configure_liveness(&mut self, live: bool) {
        self.live = live;
    }

    pub(crate) fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "latency" => {
                self.latency_ns = value.as_uint().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    pub(crate) fn get_property(&self, name: &str) -> Option<PropValue> {
        (name == "latency").then_some(PropValue::Uint(self.latency_ns))
    }

    pub(crate) fn tick_interval_ns(&self) -> Option<u64> {
        (self.live && wall_clock_ns().is_some()).then_some(UNKNOWN_FRAME_PERIOD_TICK_NS)
    }

    pub(crate) fn note_arrival(&mut self, input: usize, frame: &Frame, now_ns: Option<u64>) {
        let (Some(pts_ns), Some(now_ns)) = (frame.timing.pts(), now_ns) else {
            return;
        };
        if input >= self.arrival_offsets_ns.len() {
            self.arrival_offsets_ns.resize(input + 1, None);
        }
        let offset_ns = i128::from(now_ns) - i128::from(pts_ns);
        let largest_ns = self.arrival_offsets_ns[input].get_or_insert(offset_ns);
        *largest_ns = (*largest_ns).max(offset_ns);
    }

    pub(crate) fn is_due(&self, frame: &Frame, now_ns: Option<u64>) -> bool {
        let (Some(pts_ns), Some(now_ns)) = (frame.timing.pts(), now_ns) else {
            return false;
        };
        // the input whose frames arrive furthest behind their pts sets every deadline
        let Some(slowest_offset_ns) = self.arrival_offsets_ns.iter().flatten().max() else {
            return false;
        };
        let end_ns = i128::from(pts_ns) + i128::from(frame.timing.duration_ns);
        self.live && i128::from(now_ns) >= end_ns + slowest_offset_ns + i128::from(self.latency_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::FrameTiming;

    const FRAME_NS: u64 = 20_000_000;

    fn frame_at(pts_ns: u64) -> Frame {
        let mut frame = crate::testutil::frame(Vec::new());
        frame.timing = FrameTiming {
            pts_ns,
            duration_ns: FRAME_NS,
            ..frame.timing
        };
        frame
    }

    #[test]
    fn the_slowest_input_and_the_latency_set_the_deadline() {
        let mut deadline = LiveMuxDeadline::default();
        deadline.configure_liveness(true);
        deadline
            .set_property("latency", PropValue::Uint(5_000_000))
            .unwrap();
        // input 0 arrives 1 ms behind its pts, input 1 arrives 30 ms behind
        deadline.note_arrival(0, &frame_at(0), Some(1_000_000));
        deadline.note_arrival(1, &frame_at(0), Some(30_000_000));
        let frame = frame_at(100_000_000);
        let due_ns = 100_000_000 + FRAME_NS + 30_000_000 + 5_000_000;
        assert!(!deadline.is_due(&frame, Some(due_ns - 1)));
        assert!(deadline.is_due(&frame, Some(due_ns)));

        // a later frame that arrives 40 ms behind raises the bar, an earlier-arriving one does not lower it
        deadline.note_arrival(0, &frame_at(50_000_000), Some(90_000_000));
        deadline.note_arrival(1, &frame_at(50_000_000), Some(60_000_000));
        assert!(!deadline.is_due(&frame, Some(due_ns)));
        assert!(deadline.is_due(&frame, Some(due_ns + 10_000_000)));

        deadline.configure_liveness(false);
        assert!(
            !deadline.is_due(&frame, Some(due_ns)),
            "only a live mux times out"
        );
    }
}
