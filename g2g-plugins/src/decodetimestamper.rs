use alloc::collections::{BinaryHeap, VecDeque};
use alloc::vec::Vec;
use core::cmp::Reverse;

// a frame arrives at most `reorder_depth` frames after its presentation rank, so then the smallest pending pts is the next dts
#[derive(Debug)]
pub(crate) struct DecodeTimestamper<T> {
    reorder_depth: usize,
    held: VecDeque<T>,
    pending_pts_ns: BinaryHeap<Reverse<u64>>,
}

impl<T> DecodeTimestamper<T> {
    pub(crate) fn new(reorder_depth: usize) -> Self {
        Self {
            reorder_depth,
            held: VecDeque::new(),
            pending_pts_ns: BinaryHeap::new(),
        }
    }

    // takes frames in decode order, returns the one now stamped with its decode time
    pub(crate) fn push(&mut self, pts_ns: u64, frame: T) -> Option<(u64, T)> {
        self.held.push_back(frame);
        self.pending_pts_ns.push(Reverse(pts_ns));
        if self.held.len() <= self.reorder_depth {
            return None;
        }
        self.release()
    }

    // at the end of the stream nothing later can present earlier
    pub(crate) fn drain(&mut self) -> Vec<(u64, T)> {
        core::iter::from_fn(|| self.release()).collect()
    }

    pub(crate) fn clear(&mut self) {
        self.held.clear();
        self.pending_pts_ns.clear();
    }

    fn release(&mut self) -> Option<(u64, T)> {
        let frame = self.held.pop_front()?;
        let Reverse(dts_ns) = self
            .pending_pts_ns
            .pop()
            .expect("every held frame has a pending presentation time");
        Some((dts_ns, frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_NS: u64 = 40_000_000;

    #[test]
    fn decode_times_count_up_through_the_sorted_presentation_times() {
        // I P B B P B B in decode order, presenting I B B P B B P
        let decode_order_ranks = [0u64, 3, 1, 2, 6, 4, 5];
        let mut timestamper = DecodeTimestamper::new(2);
        let mut stamped = Vec::new();
        for rank in decode_order_ranks {
            stamped.extend(timestamper.push(rank * FRAME_NS, rank));
        }
        stamped.extend(timestamper.drain());

        let decode_times: Vec<u64> = stamped.iter().map(|(dts_ns, _)| *dts_ns).collect();
        let expected: Vec<u64> = (0..7).map(|rank| rank * FRAME_NS).collect();
        assert_eq!(decode_times, expected);
        let frames: Vec<u64> = stamped.iter().map(|(_, rank)| *rank).collect();
        assert_eq!(frames, decode_order_ranks, "frames leave in decode order");
    }

    #[test]
    fn a_stream_shifted_by_the_reorder_depth_never_decodes_after_it_presents() {
        let reorder_depth = 2;
        let shift_ns = reorder_depth * FRAME_NS;
        let decode_order_ranks = [0u64, 3, 1, 2, 6, 4, 5, 9, 7, 8];
        let mut timestamper = DecodeTimestamper::new(reorder_depth as usize);
        let mut stamped = Vec::new();
        for rank in decode_order_ranks {
            let pts_ns = rank * FRAME_NS;
            stamped.extend(timestamper.push(pts_ns, pts_ns + shift_ns));
        }
        stamped.extend(timestamper.drain());
        assert!(stamped
            .iter()
            .all(|(dts_ns, shifted_pts_ns)| dts_ns <= shifted_pts_ns));
    }
}
