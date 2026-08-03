//! Per-input collector for `MultiInputElement` aggregators (M199).
//!
//! Every N-in-1-out element (muxer, compositor, tensor batcher, the Python
//! aggregator host) needs the same bookkeeping: buffer items arriving per input
//! pad, track which inputs have ended, and release them under a sync policy.
//! The trait ([`MultiInputElement`](crate::MultiInputElement)) and the fan-in
//! runner (per-input negotiation + EOS aggregation) already exist; this is the
//! middle layer between them, owned by the muxers, the compositor and the
//! tensor batcher.
//!
//! It is the composable, typed analog of GStreamer's `GstAggregator` pad
//! collection: a helper an element *owns*, not a base class it inherits, so it
//! stays generic over the queued item `T` and free of the trait's caps / async
//! surface.
//!
//! Two release policies:
//!
//! - Synchronized rounds ([`take_round`](InputAggregator::take_round), and the
//!   time-ordered [`take_earliest_by`](InputAggregator::take_earliest_by) a
//!   muxer merges with): an input keeps contributing while its queue drains,
//!   then drops out of future rounds once it has ended and emptied, so the
//!   round shrinks as sources end.
//! - Latest-wins ([`take_round_latest`](InputAggregator::take_round_latest)):
//!   one input drives the cadence and the others contribute the newest item
//!   they delivered, read in place rather than consumed, so a faster or slower
//!   contributor neither gates nor multiplies output.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

/// Buffers items per input pad and releases synchronized rounds. Generic over
/// the queued item `T` (a `Frame`, a decoded plane, raw samples, ...).
#[derive(Debug)]
pub struct InputAggregator<T> {
    queues: Vec<VecDeque<T>>,
    ended: Vec<bool>,
    max_depth: usize,
    dropped: u64,
}

impl<T> InputAggregator<T> {
    /// A collector for `inputs` pads, unbounded per-input depth.
    pub fn new(inputs: usize) -> Self {
        Self {
            queues: (0..inputs).map(|_| VecDeque::new()).collect(),
            ended: (0..inputs).map(|_| false).collect(),
            max_depth: usize::MAX,
            dropped: 0,
        }
    }

    /// Cap each input's queue depth; pushing beyond it drops the oldest item (a
    /// leaky bound on inter-input skew, like a `DropOldest` link). Default:
    /// unbounded. A depth of 0 is treated as 1.
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth.max(1);
        self
    }

    /// Number of input pads.
    pub fn input_count(&self) -> usize {
        self.queues.len()
    }

    /// Queue an item for `input`, dropping (and counting) the oldest if the
    /// per-input cap is exceeded.
    pub fn push(&mut self, input: usize, item: T) {
        let q = &mut self.queues[input];
        q.push_back(item);
        while q.len() > self.max_depth {
            q.pop_front();
            self.dropped += 1;
        }
    }

    /// Queue `item` for `input` as its *latest*, discarding whatever it held.
    /// The latest-wins counterpart to [`push`](Self::push), for an input whose
    /// items are read in place ([`latest`](Self::latest)) rather than consumed.
    /// A superseded item is not counted in [`dropped`](Self::dropped): the
    /// policy replaced it, no depth cap evicted it.
    pub fn push_latest(&mut self, input: usize, item: T) {
        let q = &mut self.queues[input];
        q.clear();
        q.push_back(item);
    }

    /// Newest item held for `input`, left in place: later rounds see it again
    /// until a newer one replaces it.
    pub fn latest(&self, input: usize) -> Option<&T> {
        self.queues[input].back()
    }

    /// Items currently queued for `input`.
    pub fn queued(&self, input: usize) -> usize {
        self.queues[input].len()
    }

    /// Drop everything queued for `input`, keeping its ended flag: a flush, or
    /// a caps change that invalidates what it holds.
    pub fn clear(&mut self, input: usize) {
        self.queues[input].clear();
    }

    /// Mark `input` as ended (its source-pad EOS). It keeps contributing while
    /// its queue drains, then drops out of future rounds.
    pub fn mark_ended(&mut self, input: usize) {
        self.ended[input] = true;
    }

    /// Whether `input` has been marked ended.
    pub fn is_ended(&self, input: usize) -> bool {
        self.ended[input]
    }

    /// Count of items dropped to the per-input depth cap.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Inputs that can still contribute to a round: every pad except those both
    /// ended and drained.
    fn contributors(&self) -> Vec<usize> {
        (0..self.queues.len())
            .filter(|&i| !(self.ended[i] && self.queues[i].is_empty()))
            .collect()
    }

    /// Pop one item from every still-contributing input, returned as
    /// `(input, item)` pairs in input order, iff every contributor has one
    /// queued (a complete synchronized round). Returns `None` while any
    /// contributor is still waiting, or once fully drained. Call in a loop to
    /// flush every round currently complete.
    pub fn take_round(&mut self) -> Option<Vec<(usize, T)>> {
        let contributors = self.contributors();
        if contributors.is_empty() || contributors.iter().any(|&i| self.queues[i].is_empty()) {
            return None;
        }
        Some(
            contributors
                .iter()
                .map(|&i| (i, self.queues[i].pop_front().expect("checked non-empty")))
                .collect(),
        )
    }

    /// Pop the single earliest item across inputs, keyed by `key` (e.g. a
    /// frame's PTS), iff every still-contributing input has one queued. This is
    /// the time-ordered interleave a muxer needs: with each input's items
    /// monotonic in `key`, holding output until every contributor has a head
    /// guarantees the popped item is globally earliest, so no later input can
    /// still deliver something that should have preceded it. Ties go to the
    /// lowest input index (a stable, deterministic merge). Returns `None` while
    /// any contributor is still waiting (back-pressure: wait for it), or once
    /// fully drained. Call in a loop to flush every item currently safe to emit.
    ///
    /// Contrast [`take_round`](Self::take_round), which releases one item from
    /// *every* input per call (synchronized rounds, for a compositor / mixer);
    /// this releases *one* item per call (a merge, for a muxer).
    pub fn take_earliest_by<K, F>(&mut self, key: F) -> Option<(usize, T)>
    where
        K: Ord,
        F: Fn(&T) -> K,
    {
        let contributors = self.contributors();
        if contributors.is_empty() || contributors.iter().any(|&i| self.queues[i].is_empty()) {
            return None;
        }
        // Every contributor has a head; pick the one with the smallest key.
        // `min_by_key` keeps the first on ties, so the lowest input index wins.
        let winner = contributors
            .iter()
            .copied()
            .min_by_key(|&i| key(self.queues[i].front().expect("checked non-empty")))?;
        Some((
            winner,
            self.queues[winner].pop_front().expect("checked non-empty"),
        ))
    }

    /// True once every input other than `timing` holds an item: the startup
    /// gate for a latest-wins element that wants each contributor's first item
    /// before it starts releasing (a compositor waiting on its overlays).
    pub fn latest_ready(&self, timing: usize) -> bool {
        (0..self.queues.len()).all(|i| i == timing || !self.queues[i].is_empty())
    }

    /// Latest-wins release: pop the oldest item queued for the `timing` input,
    /// the one driving the output cadence (one release per item it delivers).
    /// The other inputs contribute through [`latest`](Self::latest) instead,
    /// which leaves their item queued across releases; one that has yet to
    /// deliver contributes nothing, so gate startup on
    /// [`latest_ready`](Self::latest_ready) to wait for it.
    ///
    /// Contrast [`take_round`](Self::take_round), which consumes one item from
    /// *every* input per call (synchronized rounds, for a batcher).
    pub fn take_round_latest(&mut self, timing: usize) -> Option<T> {
        self.queues[timing].pop_front()
    }

    /// True once every input has ended and all queues have drained: no further
    /// rounds will ever be produced.
    pub fn is_drained(&self) -> bool {
        self.ended.iter().all(|&e| e) && self.queues.iter().all(|q| q.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn waits_for_every_input_then_zips_in_order() {
        let mut agg = InputAggregator::new(3);
        agg.push(0, "a0");
        agg.push(2, "c0");
        // input 1 still missing -> no round.
        assert!(agg.take_round().is_none());

        agg.push(1, "b0");
        let round = agg.take_round().expect("all three present");
        assert_eq!(round, vec![(0, "a0"), (1, "b0"), (2, "c0")]);
        // consumed, so nothing more.
        assert!(agg.take_round().is_none());
    }

    #[test]
    fn drains_multiple_complete_rounds() {
        let mut agg = InputAggregator::new(2);
        agg.push(0, 1);
        agg.push(0, 2);
        agg.push(1, 10);
        agg.push(1, 20);
        assert_eq!(agg.take_round(), Some(vec![(0, 1), (1, 10)]));
        assert_eq!(agg.take_round(), Some(vec![(0, 2), (1, 20)]));
        assert_eq!(agg.take_round(), None);
    }

    #[test]
    fn ended_input_drains_then_round_shrinks() {
        let mut agg = InputAggregator::new(2);
        agg.push(0, 1);
        agg.push(1, 10);
        // input 1 ends but still has a queued item: it contributes this round.
        agg.mark_ended(1);
        assert_eq!(agg.take_round(), Some(vec![(0, 1), (1, 10)]));

        // input 1 now ended and empty: it drops out, so input 0 alone forms a round.
        agg.push(0, 2);
        assert_eq!(agg.take_round(), Some(vec![(0, 2)]));
        assert!(!agg.is_drained(), "input 0 has not ended");

        agg.mark_ended(0);
        assert!(agg.is_drained());
    }

    #[test]
    fn max_depth_drops_oldest_and_counts() {
        let mut agg = InputAggregator::new(1).with_max_depth(2);
        agg.push(0, 1);
        agg.push(0, 2);
        agg.push(0, 3); // evicts 1
        assert_eq!(agg.dropped(), 1);
        assert_eq!(agg.take_round(), Some(vec![(0, 2)]));
        assert_eq!(agg.take_round(), Some(vec![(0, 3)]));
        assert_eq!(agg.take_round(), None);
    }

    #[test]
    fn no_contributors_yields_no_round() {
        let mut agg: InputAggregator<i32> = InputAggregator::new(0);
        assert!(agg.take_round().is_none());
        assert!(agg.is_drained());
    }

    #[test]
    fn earliest_merges_by_key_across_inputs() {
        // Items are (pts, tag). Two interleaved streams emit in global PTS order.
        let mut agg = InputAggregator::new(2);
        agg.push(0, (10u64, "a10"));
        agg.push(0, (30, "a30"));
        agg.push(1, (20, "b20"));
        // Both inputs have a head: 10 < 20 wins.
        assert_eq!(agg.take_earliest_by(|&(p, _)| p), Some((0, (10, "a10"))));
        // Now input 0 head is 30, input 1 head is 20: 20 wins.
        assert_eq!(agg.take_earliest_by(|&(p, _)| p), Some((1, (20, "b20"))));
        // Input 1 empty (not ended): must wait, even though input 0 has 30.
        assert_eq!(agg.take_earliest_by(|&(p, _)| p), None);
        // Input 1 delivers a later one; 30 < 40, so input 0's 30 goes next.
        agg.push(1, (40, "b40"));
        assert_eq!(agg.take_earliest_by(|&(p, _)| p), Some((0, (30, "a30"))));
    }

    #[test]
    fn earliest_flushes_remaining_once_inputs_end() {
        let mut agg = InputAggregator::new(2);
        agg.push(0, (10u64, "a10"));
        agg.push(1, (20, "b20"));
        agg.push(1, (30, "b30"));
        agg.mark_ended(0);
        agg.mark_ended(1);
        // Both ended: drain in global order, input 0 dropping out once empty.
        assert_eq!(agg.take_earliest_by(|&(p, _)| p), Some((0, (10, "a10"))));
        assert_eq!(agg.take_earliest_by(|&(p, _)| p), Some((1, (20, "b20"))));
        assert_eq!(agg.take_earliest_by(|&(p, _)| p), Some((1, (30, "b30"))));
        assert_eq!(agg.take_earliest_by(|&(p, _)| p), None);
        assert!(agg.is_drained());
    }

    #[test]
    fn latest_wins_releases_once_per_timing_item_and_keeps_the_rest() {
        let mut agg = InputAggregator::new(2);
        // input 1 has not delivered: not ready, but a queued timing item still
        // releases (the caller decides whether to gate on readiness).
        agg.push(0, "base0");
        assert!(!agg.latest_ready(0));
        assert_eq!(agg.latest(1), None);

        agg.push_latest(1, "overlay0");
        agg.push_latest(1, "overlay1"); // supersedes, not a depth-cap drop
        assert!(agg.latest_ready(0));
        assert_eq!(agg.latest(1), Some(&"overlay1"));
        assert_eq!(agg.queued(1), 1);
        assert_eq!(agg.dropped(), 0);

        // Two timing items, two releases; the overlay stays for both.
        agg.push(0, "base1");
        assert_eq!(agg.take_round_latest(0), Some("base0"));
        assert_eq!(agg.latest(1), Some(&"overlay1"));
        assert_eq!(agg.take_round_latest(0), Some("base1"));
        assert_eq!(agg.take_round_latest(0), None);
        assert_eq!(agg.latest(1), Some(&"overlay1"));
    }

    #[test]
    fn clear_drops_one_inputs_queue_only() {
        let mut agg = InputAggregator::new(2);
        agg.push(0, 1);
        agg.push(0, 2);
        agg.push_latest(1, 10);
        agg.mark_ended(0);
        agg.clear(0);
        assert_eq!(agg.queued(0), 0);
        assert!(agg.is_ended(0), "clearing keeps the ended flag");
        assert_eq!(agg.latest(1), Some(&10));
    }

    #[test]
    fn earliest_breaks_ties_by_lowest_input() {
        let mut agg = InputAggregator::new(2);
        agg.push(0, (5u64, "a"));
        agg.push(1, (5, "b"));
        // Equal keys: the lower input index wins (stable, deterministic).
        assert_eq!(agg.take_earliest_by(|&(p, _)| p), Some((0, (5, "a"))));
    }
}
