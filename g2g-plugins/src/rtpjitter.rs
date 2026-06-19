//! Sans-IO RTP jitter buffer (RFC 3550 receive-side), the reordering stage that
//! sits between a socket and the [`RtpH264Depayloader`](crate::rtpdepay). A UDP
//! network delivers RTP packets out of order, duplicated, or not at all; feeding
//! that raw stream straight to the depayloader corrupts reassembly (a reorder
//! looks like a loss and resets the in-flight access unit). This buffer absorbs
//! that: it orders packets by sequence number and releases them in order, holds
//! a packet only until its missing predecessors are either filled or declared
//! lost (a bounded latency), and drops duplicates and packets that arrive too
//! late to matter.
//!
//! No I/O and no clock of its own: the caller supplies a monotonic `now_ns` to
//! [`push`](RtpJitterBuffer::push) / [`pop`](RtpJitterBuffer::pop) and uses
//! [`next_deadline_ns`](RtpJitterBuffer::next_deadline_ns) to schedule a flush
//! when the network goes quiet. `UdpSrc` wires this in; the unit tests drive it
//! directly with hand-built packets.
//!
//! Scope: reordering, loss/duplicate/late detection, and bounded-latency
//! release. RTCP receiver reports, NACK/RTX retransmission, and FEC are the
//! larger receive-side follow-ups (DESIGN_TODO).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

/// Minimum RTP header (V/P/X/CC, M/PT, sequence, timestamp, ssrc); the sequence
/// number lives at bytes 2..4.
const RTP_HEADER_LEN: usize = 12;

/// Release policy. A packet missing a predecessor is held at most `max_hold_ns`
/// (then the gap is declared lost), and the buffer never grows past `max_depth`
/// packets (a flood forces release). `max_depth == 0` disables buffering:
/// packets pass straight through in arrival order (the pre-M94 behaviour).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitterConfig {
    pub max_hold_ns: u64,
    pub max_depth: usize,
}

impl JitterConfig {
    /// A modest live default: hold a gap up to `ms` milliseconds, cap at `depth`
    /// buffered packets.
    pub fn new(ms: u64, depth: usize) -> Self {
        Self { max_hold_ns: ms * 1_000_000, max_depth: depth }
    }
}

impl Default for JitterConfig {
    fn default() -> Self {
        // 50 ms / 64 packets: tolerates typical LAN reorder without adding much
        // latency, and bounds memory under a burst.
        Self::new(50, 64)
    }
}

/// Running counters, for observability and tests.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct JitterStats {
    /// Packets accepted into the buffer (excludes malformed / late / duplicate).
    pub received: u64,
    /// Packets that arrived out of order (an earlier sequence after a later one).
    pub reordered: u64,
    /// Packets declared lost (skipped gaps).
    pub lost: u64,
    /// Duplicate sequence numbers dropped.
    pub duplicates: u64,
    /// Packets dropped because their sequence was already released (too late).
    pub late: u64,
}

struct Buffered {
    data: Vec<u8>,
    arrival_ns: u64,
}

/// A jitter buffer keyed by *extended* sequence number, so the 16-bit RTP
/// sequence's wraparound is handled by unrolling it into a monotonic counter.
pub struct RtpJitterBuffer {
    config: JitterConfig,
    /// Buffered packets awaiting in-order release, keyed by extended sequence.
    packets: BTreeMap<u64, Buffered>,
    /// Next extended sequence to release; `None` until the first packet sets the
    /// baseline.
    next: Option<u64>,
    /// Highest extended sequence seen, for wraparound unrolling.
    last_ext: Option<u64>,
    stats: JitterStats,
}

// Extended-sequence base: the first packet maps to `BASE + seq`, well clear of
// zero, so a reordered earlier packet's extended sequence stays positive.
const SEQ_BASE: u64 = 1 << 32;

impl core::fmt::Debug for RtpJitterBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RtpJitterBuffer")
            .field("config", &self.config)
            .field("buffered", &self.packets.len())
            .field("next", &self.next)
            .field("stats", &self.stats)
            .finish()
    }
}

impl RtpJitterBuffer {
    pub fn new(config: JitterConfig) -> Self {
        Self {
            config,
            packets: BTreeMap::new(),
            next: None,
            last_ext: None,
            stats: JitterStats::default(),
        }
    }

    pub fn stats(&self) -> JitterStats {
        self.stats
    }

    /// Number of packets currently buffered.
    pub fn buffered(&self) -> usize {
        self.packets.len()
    }

    /// Unroll a 16-bit RTP sequence into a monotonic extended sequence, relative
    /// to the highest one seen (RFC 3550 serial-number arithmetic). A forward
    /// step advances the high bits; a small negative step (a reorder) maps to a
    /// slightly lower extended value, so ordering is preserved across wrap.
    fn extend(&mut self, seq: u16) -> u64 {
        match self.last_ext {
            None => {
                let ext = SEQ_BASE + seq as u64;
                self.last_ext = Some(ext);
                ext
            }
            Some(last) => {
                let last16 = (last & 0xFFFF) as u16;
                // Signed 16-bit distance: forward (0..=32767) or backward
                // (-32768..=-1). Adding it to the last extended value lands on
                // the right side of any wrap.
                let delta = seq.wrapping_sub(last16) as i16 as i64;
                let ext = (last as i64 + delta) as u64;
                if ext > last {
                    self.last_ext = Some(ext);
                }
                ext
            }
        }
    }

    /// Insert one RTP packet, stamped with arrival time `now_ns`. Malformed,
    /// duplicate, and too-late packets are counted and dropped; everything else
    /// is buffered for in-order release. The first packet sets the release
    /// baseline.
    pub fn push(&mut self, packet: &[u8], now_ns: u64) {
        if packet.len() < RTP_HEADER_LEN {
            return;
        }
        let seq = u16::from_be_bytes([packet[2], packet[3]]);
        let prev_high = self.last_ext;
        let ext = self.extend(seq);

        match self.next {
            None => self.next = Some(ext),
            Some(next) => {
                if ext < next {
                    // Already released past this sequence: too late to use.
                    self.stats.late += 1;
                    return;
                }
            }
        }
        if self.packets.contains_key(&ext) {
            self.stats.duplicates += 1;
            return;
        }
        // Out of order if it sits behind a packet we have already seen.
        if matches!(prev_high, Some(high) if ext < high) {
            self.stats.reordered += 1;
        }
        self.stats.received += 1;
        self.packets.insert(ext, Buffered { data: packet.to_vec(), arrival_ns: now_ns });
    }

    /// Release the next in-order packet if one is ready: either the next
    /// expected sequence is present, or its predecessors are overdue
    /// (`max_hold_ns` elapsed for the held head) or the buffer is at
    /// `max_depth`, in which case the gap is declared lost and skipped. Returns
    /// `None` while still waiting. Call repeatedly to drain.
    pub fn pop(&mut self, now_ns: u64) -> Option<Vec<u8>> {
        let next = self.next?;
        let (&head, head_buf) = self.packets.iter().next()?;
        if head == next {
            let buf = self.packets.remove(&head).expect("head present");
            self.next = Some(next + 1);
            return Some(buf.data);
        }
        // A gap precedes the head. Hold for late predecessors up to the bound;
        // past it (by time or depth), declare [next, head) lost and skip ahead.
        debug_assert!(head > next, "released sequences are never re-buffered");
        let overdue = now_ns.saturating_sub(head_buf.arrival_ns) >= self.config.max_hold_ns;
        if overdue || self.packets.len() >= self.config.max_depth.max(1) {
            self.stats.lost += head - next;
            let buf = self.packets.remove(&head).expect("head present");
            self.next = Some(head + 1);
            return Some(buf.data);
        }
        None
    }

    /// The 16-bit sequence numbers currently missing: holes between the next
    /// expected sequence and the highest one buffered. Used to build NACK
    /// feedback; the span is bounded by the buffer depth.
    pub fn missing_seqs(&self) -> Vec<u16> {
        let mut out = Vec::new();
        let Some(next) = self.next else { return out };
        let Some((&last, _)) = self.packets.iter().next_back() else { return out };
        let mut ext = next;
        while ext < last {
            if !self.packets.contains_key(&ext) {
                out.push((ext & 0xFFFF) as u16);
            }
            ext += 1;
        }
        out
    }

    /// Nanoseconds until [`pop`](Self::pop) would next release a packet without
    /// any new arrival: `Some(0)` if one is ready now, `Some(delay)` if the head
    /// is waiting on a deadline, `None` if the buffer is empty (block on the
    /// socket). Drives the receive loop's timeout so a held packet still flushes
    /// when the network goes quiet.
    pub fn next_deadline_ns(&self, now_ns: u64) -> Option<u64> {
        let next = self.next?;
        let (&head, head_buf) = self.packets.iter().next()?;
        if head == next || self.packets.len() >= self.config.max_depth.max(1) {
            return Some(0);
        }
        let due = head_buf.arrival_ns + self.config.max_hold_ns;
        Some(due.saturating_sub(now_ns))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal RTP packet: header with `seq`, `marker`, and a one-byte
    /// payload tag so reordering is observable in the output.
    fn pkt(seq: u16, tag: u8) -> Vec<u8> {
        let s = seq.to_be_bytes();
        alloc::vec![0x80, 96, s[0], s[1], 0, 0, 0, 0, 0, 0, 0, 0, tag]
    }

    fn tag(p: &[u8]) -> u8 {
        p[RTP_HEADER_LEN]
    }

    #[test]
    fn in_order_passes_straight_through() {
        let mut jb = RtpJitterBuffer::new(JitterConfig::new(50, 64));
        let mut out = Vec::new();
        for s in 0..5u16 {
            jb.push(&pkt(s, s as u8), 0);
            while let Some(p) = jb.pop(0) {
                out.push(tag(&p));
            }
        }
        assert_eq!(out, alloc::vec![0, 1, 2, 3, 4]);
        assert_eq!(jb.stats().reordered, 0);
        assert_eq!(jb.stats().lost, 0);
    }

    #[test]
    fn reordered_packets_are_released_in_sequence() {
        let mut jb = RtpJitterBuffer::new(JitterConfig::new(50, 64));
        // Arrive 0, 2, 1, 3: the buffer must emit 0,1,2,3 in order.
        for (s, t) in [(0u16, 0u8), (2, 2), (1, 1), (3, 3)] {
            jb.push(&pkt(s, t), 0);
        }
        let mut out = Vec::new();
        while let Some(p) = jb.pop(0) {
            out.push(tag(&p));
        }
        assert_eq!(out, alloc::vec![0, 1, 2, 3], "reordered into sequence");
        assert_eq!(jb.stats().reordered, 1, "packet 1 arrived after 2");
        assert_eq!(jb.stats().lost, 0, "nothing lost, just reordered");
    }

    #[test]
    fn missing_packet_is_skipped_after_the_hold_deadline() {
        let mut jb = RtpJitterBuffer::new(JitterConfig::new(50, 64));
        jb.push(&pkt(0, 0), 0);
        assert_eq!(jb.pop(0).map(|p| tag(&p)), Some(0));
        // Sequence 1 never arrives; 2 shows up at t=0. Before the deadline pop
        // waits; after 50 ms it gives up on 1 and releases 2.
        jb.push(&pkt(2, 2), 0);
        assert!(jb.pop(0).is_none(), "holds for the missing predecessor");
        assert_eq!(jb.pop(60_000_000).map(|p| tag(&p)), Some(2), "skips the gap once overdue");
        assert_eq!(jb.stats().lost, 1, "sequence 1 declared lost");
    }

    #[test]
    fn full_buffer_forces_release_without_waiting() {
        // depth 2: a gap at seq 1 with two later packets buffered must release
        // immediately (no time elapsed), declaring the gap lost.
        let mut jb = RtpJitterBuffer::new(JitterConfig::new(10_000, 2));
        jb.push(&pkt(0, 0), 0);
        assert_eq!(jb.pop(0).map(|p| tag(&p)), Some(0));
        jb.push(&pkt(2, 2), 0);
        jb.push(&pkt(3, 3), 0);
        // Buffer is at depth; release proceeds even though max_hold is huge.
        assert_eq!(jb.pop(0).map(|p| tag(&p)), Some(2), "depth cap forces release");
        assert_eq!(jb.stats().lost, 1);
    }

    #[test]
    fn duplicate_and_late_packets_are_dropped() {
        let mut jb = RtpJitterBuffer::new(JitterConfig::new(50, 64));
        jb.push(&pkt(5, 5), 0);
        jb.push(&pkt(5, 5), 0); // duplicate before release
        assert_eq!(jb.stats().duplicates, 1);
        assert_eq!(jb.pop(0).map(|p| tag(&p)), Some(5));
        // Sequence 5 already released: a re-arrival is too late.
        jb.push(&pkt(5, 5), 0);
        assert_eq!(jb.stats().late, 1);
        assert!(jb.pop(0).is_none());
    }

    #[test]
    fn handles_16bit_sequence_wraparound() {
        let mut jb = RtpJitterBuffer::new(JitterConfig::new(50, 64));
        // Around the wrap: 65534, 65535, 0, 1 must stay in order.
        for (s, t) in [(65534u16, 10u8), (0, 12), (65535, 11), (1, 13)] {
            jb.push(&pkt(s, t), 0);
        }
        let mut out = Vec::new();
        while let Some(p) = jb.pop(0) {
            out.push(tag(&p));
        }
        assert_eq!(out, alloc::vec![10, 11, 12, 13], "monotonic across the u16 wrap");
        assert_eq!(jb.stats().lost, 0);
    }

    #[test]
    fn missing_seqs_reports_holes_up_to_the_highest_buffered() {
        let mut jb = RtpJitterBuffer::new(JitterConfig::new(50, 64));
        // Buffer 0, then 2 and 4 with 1 and 3 missing.
        for s in [0u16, 2, 4] {
            jb.push(&pkt(s, s as u8), 0);
        }
        assert_eq!(jb.pop(0).map(|p| tag(&p)), Some(0), "release the contiguous head");
        // Now next=1; holes before the highest buffered (4) are 1 and 3.
        assert_eq!(jb.missing_seqs(), alloc::vec![1, 3]);
    }

    #[test]
    fn passthrough_mode_when_depth_zero() {
        let mut jb = RtpJitterBuffer::new(JitterConfig::new(0, 0));
        // depth 0 -> .max(1) means a single gap releases immediately: behaves
        // like the old in-order forwarder (no reorder tolerance).
        jb.push(&pkt(0, 0), 0);
        assert_eq!(jb.pop(0).map(|p| tag(&p)), Some(0));
        jb.push(&pkt(2, 2), 0);
        assert_eq!(jb.pop(0).map(|p| tag(&p)), Some(2), "no holding without depth");
    }
}
