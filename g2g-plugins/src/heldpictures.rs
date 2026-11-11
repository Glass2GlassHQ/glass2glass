// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::collections::BTreeSet;

use g2g_core::query::LatencyReport;
use g2g_core::Rate;

#[derive(Debug, Default)]
pub(crate) struct HeldPictures {
    // pts of the units a decoder has taken and not yet output
    pending_pts: BTreeSet<u64>,
    most_held: usize,
}

impl HeldPictures {
    pub(crate) fn fed(&mut self, pts_ns: u64) {
        // the set before the insert is what the decoder kept after the previous unit
        self.most_held = self.most_held.max(self.pending_pts.len());
        self.pending_pts.insert(pts_ns);
    }

    #[cfg(all(target_os = "linux", feature = "openh264"))]
    pub(crate) fn remove(&mut self, pts_ns: u64) {
        self.pending_pts.remove(&pts_ns);
    }

    #[cfg(all(target_os = "linux", feature = "openh264"))]
    pub(crate) fn earliest(&self) -> Option<u64> {
        self.pending_pts.first().copied()
    }

    // decoders output in presentation order, so any earlier pts still held was dropped
    pub(crate) fn output(&mut self, pts_ns: u64) {
        self.pending_pts = self.pending_pts.split_off(&pts_ns.saturating_add(1));
    }

    pub(crate) fn latency(&self, framerate: Rate) -> LatencyReport {
        let held = self.most_held as u32;
        LatencyReport::held_pictures(held, held, framerate)
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropped_units_do_not_count_as_held() {
        let mut held = HeldPictures::default();
        // every other unit is dropped by the decoder and never output
        for (dropped_pts, output_pts) in [(10, 20), (30, 40), (50, 60)] {
            held.fed(dropped_pts);
            held.fed(output_pts);
            held.output(output_pts);
        }
        held.fed(70);
        assert_eq!(
            held.latency(Rate::Any),
            LatencyReport::held_pictures(1, 1, Rate::Any)
        );
    }
}
