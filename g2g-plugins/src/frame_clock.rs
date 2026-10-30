use alloc::collections::VecDeque;

use g2g_core::time::ticks_to_ns;

// presentation times for a parser that counts samples between the upstream timestamps it receives
#[derive(Debug, Default)]
pub(crate) struct FrameClock {
    base_ns: u64,
    samples_since_base: u64,
    // (stream offset, pts) of each timestamped input chunk, oldest first
    pending: VecDeque<(u64, u64)>,
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
            self.base_ns = pts_ns;
            self.samples_since_base = 0;
            self.pending.pop_front();
        }
        let pts_ns = self
            .base_ns
            .saturating_add(ticks_to_ns(self.samples_since_base, sample_rate));
        self.samples_since_base += frame_samples;
        pts_ns
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
    fn chunks_behind_the_consumed_offset_are_dropped() {
        let mut clock = FrameClock::default();
        for chunk in 0..1_000 {
            clock.note_chunk(chunk * CHUNK_BYTES, chunk * CHUNK_BYTES, FIRST_PTS_NS);
        }
        assert_eq!(clock.pending.len(), 2);
    }
}
