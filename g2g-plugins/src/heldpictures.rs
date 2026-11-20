// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::collections::btree_map::OccupiedEntry;
use alloc::collections::BTreeMap;

#[derive(Debug, Default)]
pub(crate) struct HeldPictures {
    // how many units with each pts a decoder has taken and not yet output
    pending_pts: BTreeMap<u64, usize>,
    most_held: usize,
}

impl HeldPictures {
    fn held(&self) -> usize {
        self.pending_pts.values().sum()
    }

    pub(crate) fn fed(&mut self, pts_ns: u64) {
        // the count before the insert is what the decoder kept after the previous unit
        self.most_held = self.most_held.max(self.held());
        *self.pending_pts.entry(pts_ns).or_default() += 1;
    }

    #[cfg(all(target_os = "linux", feature = "openh264"))]
    pub(crate) fn remove(&mut self, pts_ns: u64) {
        if let alloc::collections::btree_map::Entry::Occupied(entry) =
            self.pending_pts.entry(pts_ns)
        {
            take_one(entry);
        }
    }

    // mp4s with no composition offsets stamp B-frame streams in decode order, and pictures leave in display order
    #[cfg(any(
        all(target_os = "linux", feature = "openh264"),
        all(target_os = "android", feature = "mediacodec"),
        all(target_os = "windows", feature = "d3d12-decode"),
        all(target_os = "linux", feature = "v4l2")
    ))]
    pub(crate) fn take_earliest(&mut self) -> Option<u64> {
        let entry = self.pending_pts.first_entry()?;
        let pts_ns = *entry.key();
        take_one(entry);
        Some(pts_ns)
    }

    // decoders output in presentation order, so any earlier pts still held was dropped
    #[cfg(any(all(target_os = "windows", feature = "mf-decode"), test))]
    pub(crate) fn output(&mut self, pts_ns: u64) {
        self.pending_pts = self.pending_pts.split_off(&pts_ns.saturating_add(1));
    }

    #[cfg(any(
        all(target_os = "android", feature = "mediacodec"),
        all(target_os = "windows", feature = "mf-decode"),
        all(target_os = "linux", feature = "openh264"),
        test
    ))]
    pub(crate) fn latency(&self, framerate: g2g_core::Rate) -> g2g_core::query::LatencyReport {
        let held = self.most_held as u32;
        g2g_core::query::LatencyReport::held_pictures(held, held, framerate)
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(any(
    all(target_os = "linux", feature = "openh264"),
    all(target_os = "android", feature = "mediacodec"),
    all(target_os = "windows", feature = "d3d12-decode"),
    all(target_os = "linux", feature = "v4l2")
))]
fn take_one(mut entry: OccupiedEntry<'_, u64, usize>) {
    *entry.get_mut() -= 1;
    if *entry.get() == 0 {
        entry.remove();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::query::LatencyReport;
    use g2g_core::Rate;

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
