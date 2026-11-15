// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#[cfg(feature = "webrtc")]
use alloc::vec::Vec;

use g2g_core::clock::{CaptureAnchor, ClockSync};
#[cfg(feature = "webrtc")]
use g2g_core::frame::Frame;
#[cfg(feature = "webrtc")]
use g2g_core::memory::SystemSlice;
#[cfg(any(
    feature = "webrtc",
    feature = "remote",
    feature = "remote-ws",
    feature = "webtransport",
    all(
        target_os = "linux",
        any(feature = "local-ipc", feature = "local-dmabuf")
    ),
    all(target_arch = "wasm32", feature = "web")
))]
use g2g_core::FrameTiming;
#[cfg(feature = "webrtc")]
use g2g_core::{FrameMetaSet, MemoryDomain};

#[derive(Debug)]
pub(crate) struct ArrivalStamp {
    clock_sync: Option<ClockSync>,
    anchor: CaptureAnchor,
    first_arrival_ns: Option<u64>,
}

impl ArrivalStamp {
    pub(crate) fn new(clock_sync: Option<ClockSync>) -> Self {
        Self {
            clock_sync,
            anchor: CaptureAnchor::new(),
            first_arrival_ns: None,
        }
    }

    fn running_time_ns(&mut self, arrival_ns: u64) -> u64 {
        let elapsed_ns = arrival_ns - *self.first_arrival_ns.get_or_insert(arrival_ns);
        match &self.clock_sync {
            Some(sync) => self.anchor.stamp(sync, elapsed_ns, 0),
            None => elapsed_ns,
        }
    }

    #[cfg(feature = "webrtc")]
    pub(crate) fn frame(
        &mut self,
        data: Vec<u8>,
        keyframe: bool,
        sequence: u64,
        meta: FrameMetaSet,
    ) -> Frame {
        let arrival_ns = g2g_core::metrics::monotonic_ns();
        let pts_ns = self.running_time_ns(arrival_ns);
        Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: 0,
                capture_ns: pts_ns,
                arrival_ns,
                keyframe,
                ..FrameTiming::default()
            },
            sequence,
            meta,
        }
    }
}

#[cfg(any(
    feature = "remote",
    feature = "remote-ws",
    feature = "webtransport",
    all(
        target_os = "linux",
        any(feature = "local-ipc", feature = "local-dmabuf")
    ),
    feature = "st2110",
    all(target_arch = "wasm32", feature = "web")
))]
#[derive(Debug)]
pub(crate) struct SenderTimelineShift {
    arrival: ArrivalStamp,
    offset_ns: Option<i128>,
}

#[cfg(any(
    feature = "remote",
    feature = "remote-ws",
    feature = "webtransport",
    all(
        target_os = "linux",
        any(feature = "local-ipc", feature = "local-dmabuf")
    ),
    feature = "st2110",
    all(target_arch = "wasm32", feature = "web")
))]
impl SenderTimelineShift {
    pub(crate) fn new(clock_sync: Option<ClockSync>) -> Self {
        Self {
            arrival: ArrivalStamp::new(clock_sync),
            offset_ns: None,
        }
    }

    pub(crate) fn shift_ns(&mut self, sender_ns: u64) -> u64 {
        let offset_ns = match self.offset_ns {
            Some(offset_ns) => offset_ns,
            None => {
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                let local_ns = self.arrival.running_time_ns(arrival_ns);
                *self
                    .offset_ns
                    .insert(i128::from(local_ns) - i128::from(sender_ns))
            }
        };
        (i128::from(sender_ns) + offset_ns).clamp(0, i128::from(u64::MAX)) as u64
    }

    #[cfg(any(
        feature = "remote",
        feature = "remote-ws",
        feature = "webtransport",
        all(
            target_os = "linux",
            any(feature = "local-ipc", feature = "local-dmabuf")
        ),
        all(target_arch = "wasm32", feature = "web")
    ))]
    pub(crate) fn apply(&mut self, mut timing: FrameTiming) -> FrameTiming {
        timing.dts_ns = self.shift_ns(timing.dts_ns);
        timing.pts_ns = self.shift_ns(timing.pts_ns);
        timing
    }
}

#[cfg(feature = "st2110")]
#[derive(Debug)]
pub(crate) struct RtpArrivalTimeline {
    rebase: crate::rtpjitter::RtpPtsRebase,
    shift: SenderTimelineShift,
}

#[cfg(feature = "st2110")]
impl RtpArrivalTimeline {
    pub(crate) fn new(clock_sync: Option<ClockSync>) -> Self {
        Self {
            rebase: crate::rtpjitter::RtpPtsRebase::new(),
            shift: SenderTimelineShift::new(clock_sync),
        }
    }

    pub(crate) fn pts_ns(&mut self, rtp_timestamp: u32, clock_rate: u32) -> u64 {
        let sender_ns = self.rebase.pts_ns(rtp_timestamp, clock_rate);
        self.shift.shift_ns(sender_ns)
    }
}

#[cfg(all(
    test,
    any(
        feature = "remote",
        feature = "remote-ws",
        feature = "webtransport",
        all(
            target_os = "linux",
            any(feature = "local-ipc", feature = "local-dmabuf")
        ),
        all(target_arch = "wasm32", feature = "web")
    )
))]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    const SENDER_START_NS: u64 = 5_000_000_000;
    const FRAME_PERIOD_NS: u64 = 40_000_000;
    const REORDER_GAP_NS: u64 = 80_000_000;

    fn sender_timing(index: u64) -> FrameTiming {
        let dts_ns = SENDER_START_NS + index * FRAME_PERIOD_NS;
        FrameTiming {
            pts_ns: dts_ns + REORDER_GAP_NS,
            dts_ns,
            ..FrameTiming::default()
        }
    }

    #[test]
    fn shift_moves_the_first_decode_time_to_local_arrival_and_keeps_spacing() {
        let mut timeline = SenderTimelineShift::new(None);
        let shifted: Vec<FrameTiming> = (0..3)
            .map(|index| timeline.apply(sender_timing(index)))
            .collect();

        assert_eq!(
            shifted[0].dts_ns, 0,
            "with no clock the first arrival is running time zero"
        );
        for (index, timing) in (0u64..).zip(&shifted) {
            assert_eq!(
                timing.dts_ns,
                index * FRAME_PERIOD_NS,
                "sender spacing kept"
            );
            assert_eq!(
                timing.pts_ns - timing.dts_ns,
                REORDER_GAP_NS,
                "decode-to-presentation gap kept"
            );
        }
    }
}
