// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::collections::VecDeque;

use g2g_core::time::ticks_to_ns;

// matroska and flv stamp blocks in whole milliseconds, so a stamp this close to the count is the count rounded
const STAMP_ROUNDING_NS: u64 = 1_000_000;

// presentation times for a parser that counts samples between the upstream timestamps it receives
#[derive(Debug, Default)]
pub(crate) struct FrameClock {
    base_ns: u64,
    samples_since_base: u64,
    // (stream offset, pts) of each timestamped input chunk, oldest first
    pending: VecDeque<(u64, u64)>,
    // (stream offset of the chunk's end, end discard) of each input chunk that trims its last frame
    end_discards: VecDeque<(u64, u64)>,
}

impl FrameClock {
    // a frame only goes out once the next chunk confirms its end, so several chunks can be pending
    pub(crate) fn note_chunk(&mut self, consumed_offset: u64, chunk_offset: u64, pts_ns: u64) {
        let reached = self
            .pending
            .iter()
            .take_while(|(at, _)| *at <= consumed_offset)
            .count();
        self.pending.drain(..reached.saturating_sub(1));
        self.pending.push_back((chunk_offset, pts_ns));
    }

    pub(crate) fn frame_pts(
        &mut self,
        start_offset: u64,
        frame_samples: u64,
        sample_rate: u32,
    ) -> u64 {
        while let Some(&(at, pts_ns)) = self.pending.front() {
            if at > start_offset {
                break;
            }
            self.pending.pop_front();
            let counting = self.samples_since_base > 0;
            if counting && pts_ns.abs_diff(self.counted_ns(sample_rate)) < STAMP_ROUNDING_NS {
                continue;
            }
            self.base_ns = pts_ns;
            self.samples_since_base = 0;
        }
        let pts_ns = self.counted_ns(sample_rate);
        self.samples_since_base += frame_samples;
        pts_ns
    }

    fn counted_ns(&self, sample_rate: u32) -> u64 {
        self.base_ns
            .saturating_add(ticks_to_ns(self.samples_since_base, sample_rate))
    }

    pub(crate) fn note_end_discard(&mut self, chunk_end_offset: u64, end_discard_ns: u64) {
        self.end_discards
            .push_back((chunk_end_offset, end_discard_ns));
    }

    // the end discard of the chunk that ends within this frame, 0 when none does
    pub(crate) fn end_discard_ns(&mut self, frame_end_offset: u64) -> u64 {
        let mut end_discard_ns = 0;
        while let Some(&(chunk_end_offset, discard_ns)) = self.end_discards.front() {
            if chunk_end_offset > frame_end_offset {
                break;
            }
            end_discard_ns = discard_ns;
            self.end_discards.pop_front();
        }
        end_discard_ns
    }

    // a flush restarts the count, which a seek back to the start never re-bases with a nonzero pts
    pub(crate) fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;
    const FRAME_SAMPLES: u64 = 1_536;
    const CHUNK_BYTES: u64 = 768;
    const FIRST_PTS_NS: u64 = 375_833_333;
    const CHUNK_PTS_STEP_NS: u64 = 32_000_000;

    #[test]
    fn a_chunk_timestamp_applies_after_later_chunks_arrive() {
        let mut clock = FrameClock::default();
        let mut stamps = alloc::vec::Vec::new();
        for chunk in 0..3 {
            clock.note_chunk(
                0,
                chunk * CHUNK_BYTES,
                FIRST_PTS_NS + chunk * CHUNK_PTS_STEP_NS,
            );
        }
        for frame in 0..2 {
            stamps.push(clock.frame_pts(frame * CHUNK_BYTES, FRAME_SAMPLES, RATE));
        }
        assert_eq!(stamps, [FIRST_PTS_NS, FIRST_PTS_NS + CHUNK_PTS_STEP_NS]);
    }

    #[test]
    fn millisecond_stamps_keep_the_sample_count_and_a_jump_rebases() {
        const AAC_RATE: u32 = 44_100;
        const AAC_FRAME_SAMPLES: u64 = 1_024;
        const MILLISECOND_NS: u64 = 1_000_000;
        const JUMP_NS: u64 = 5 * MILLISECOND_NS;
        let mut clock = FrameClock::default();
        for frame in 0..3 {
            let exact_ns = ticks_to_ns(frame * AAC_FRAME_SAMPLES, AAC_RATE);
            let rounded_ns = exact_ns / MILLISECOND_NS * MILLISECOND_NS;
            clock.note_chunk(frame * CHUNK_BYTES, frame * CHUNK_BYTES, rounded_ns);
            let pts_ns = clock.frame_pts(frame * CHUNK_BYTES, AAC_FRAME_SAMPLES, AAC_RATE);
            assert_eq!(pts_ns, exact_ns, "frame {frame}");
        }
        let jumped_ns = ticks_to_ns(3 * AAC_FRAME_SAMPLES, AAC_RATE) + JUMP_NS;
        clock.note_chunk(3 * CHUNK_BYTES, 3 * CHUNK_BYTES, jumped_ns);
        let pts_ns = clock.frame_pts(3 * CHUNK_BYTES, AAC_FRAME_SAMPLES, AAC_RATE);
        assert_eq!(pts_ns, jumped_ns);
    }

    #[test]
    fn chunks_behind_the_consumed_offset_are_dropped() {
        let mut clock = FrameClock::default();
        for chunk in 0..1_000 {
            clock.note_chunk(chunk * CHUNK_BYTES, chunk * CHUNK_BYTES, FIRST_PTS_NS);
        }
        assert_eq!(clock.pending.len(), 2);
    }
}
