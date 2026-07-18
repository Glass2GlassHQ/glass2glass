//! ST 2110-7 seamless protection (M608): the receive-side reconstruction that
//! merges two (or more) identical redundant RTP streams into one lossless stream.
//!
//! A -7 sender emits the *same* RTP packets (same SSRC, sequence numbers, and
//! timestamps) down two disjoint network paths ("red" / "blue", usually separate
//! NICs and multicast groups). A receiver joins both and, for each RTP sequence
//! number, forwards the first copy to arrive and discards the later duplicate; if a
//! packet is lost on one path, the other path's copy fills the gap, so a single
//! path failure or drop is hitless.
//!
//! [`SeamlessDedup`] is the essence-agnostic core of that (it reads only the 16-bit
//! RTP sequence number, so it works for -20 / -22 / -30 / -40 alike): feed it every
//! packet from every path and it answers whether the packet is novel (forward it
//! downstream) or a duplicate already seen (drop it). It keeps a sliding window of
//! seen sequence numbers, extending the 16-bit number against the highest seen to
//! survive the `2^16` wrap (the extended-sequence technique the -20 / media-clock
//! code uses), so it never falsely drops a packet a whole wrap later.
//!
//! **Never trust the stream:** a packet too short to hold an RTP header is rejected;
//! a sequence number older than the window is treated as already delivered (dropped)
//! rather than reopening a closed gap.

use g2g_core::rtp::RTP_HEADER_LEN;

/// Sliding-window width (sequence numbers) the dedup remembers. 8192 covers a large
/// inter-path skew / burst while the bitset stays tiny ([u64; 128] = 1 KiB).
const WINDOW: u64 = 8192;
const WORDS: usize = (WINDOW / 64) as usize;

/// Merges redundant ST 2110-7 RTP streams by RTP sequence number: the first arrival
/// of a sequence number is forwarded, later duplicates dropped, so two lossy paths
/// reconstruct one complete stream.
#[derive(Debug)]
pub struct SeamlessDedup {
    /// Highest extended (wrap-resolved) sequence number seen, or `None` until the
    /// first packet anchors the window.
    high: Option<u64>,
    /// Bitset of seen extended sequence numbers over `(high - WINDOW, high]`, indexed
    /// by `ext % WINDOW`.
    seen: [u64; WORDS],
}

impl Default for SeamlessDedup {
    fn default() -> Self {
        Self::new()
    }
}

impl SeamlessDedup {
    /// A fresh de-duplicator (no packets seen yet).
    pub fn new() -> Self {
        Self {
            high: None,
            seen: [0; WORDS],
        }
    }

    /// Extend a 16-bit RTP sequence number against the highest seen, choosing the
    /// wrap that lands nearest (within half the sequence space).
    fn extend(&self, seq: u16) -> u64 {
        match self.high {
            None => u64::from(seq),
            Some(high) => {
                let base = high & !0xFFFF;
                let mut ext = base | u64::from(seq);
                // Pick the candidate (previous / this / next 16-bit epoch) closest to
                // `high`, so a wrap resolves to the intended absolute sequence.
                let candidates = [ext.wrapping_sub(0x1_0000), ext, ext.wrapping_add(0x1_0000)];
                ext = *candidates
                    .iter()
                    .min_by_key(|&&c| (c as i64).wrapping_sub(high as i64).unsigned_abs())
                    .unwrap();
                ext
            }
        }
    }

    fn bit(&self, ext: u64) -> (usize, u64) {
        let idx = (ext % WINDOW) as usize;
        (idx / 64, 1u64 << (idx % 64))
    }

    /// Offer one RTP packet (from any path). Returns `true` if it is the first time
    /// this sequence number has been seen (forward it downstream), `false` if it is a
    /// duplicate or too old to matter (drop it). A packet too short for an RTP header
    /// is dropped.
    pub fn accept(&mut self, packet: &[u8]) -> bool {
        if packet.len() < RTP_HEADER_LEN {
            return false;
        }
        let seq = u16::from_be_bytes([packet[2], packet[3]]);
        let ext = self.extend(seq);

        let Some(high) = self.high else {
            // First packet: anchor the window and mark it seen.
            self.high = Some(ext);
            let (w, b) = self.bit(ext);
            self.seen[w] |= b;
            return true;
        };

        if ext > high {
            // Advance the window, clearing the sequence slots it now uncovers so a
            // number from the previous wrap cannot masquerade as already-seen.
            for s in (high + 1)..=ext {
                let (w, b) = self.bit(s);
                self.seen[w] &= !b;
            }
            self.high = Some(ext);
        } else if high.saturating_sub(ext) >= WINDOW {
            // Older than the window: assume already delivered, drop.
            return false;
        }

        let (w, b) = self.bit(ext);
        if self.seen[w] & b != 0 {
            return false; // duplicate
        }
        self.seen[w] |= b;
        true
    }
}

// The socket-bound receiver that joins redundant paths on the wire. Needs std
// (`UdpSocket`), so it lives behind the `st2110` feature; the sans-IO `SeamlessDedup`
// above stays in the no_std baseline.
#[cfg(feature = "st2110")]
mod net {
    use super::SeamlessDedup;
    use std::io;
    use std::net::UdpSocket;
    use std::time::Duration;

    /// Joins redundant ST 2110-7 RTP paths into one deduplicated packet stream.
    ///
    /// Construct it over the already-bound receive sockets (one per path: "red",
    /// "blue", ...); [`Self::recv_novel`] polls them all, feeds every datagram through
    /// the shared [`SeamlessDedup`], and returns the next packet not seen before, so a
    /// caller depacketizes each RTP sequence number exactly once even when both paths
    /// deliver it and reconstructs the stream when a packet is lost on one path.
    /// Essence-agnostic (video / JPEG XS / audio / ancillary alike): it reads only the
    /// RTP sequence number, so any `St2110*Src` can bind two sockets and drain here.
    #[derive(Debug)]
    pub struct RedundantRtpReceiver<'s> {
        sockets: &'s [&'s UdpSocket],
        dedup: SeamlessDedup,
        poll: Duration,
        idle_limit: Duration,
        /// Round-robin cursor: the path to poll first next. Advancing it after each
        /// delivered packet fairly interleaves the paths, so two in-order streams merge
        /// back into order (draining one path fully first would reorder packets past a
        /// frame's marker and lose the other path's tail).
        next: usize,
    }

    impl<'s> RedundantRtpReceiver<'s> {
        /// A receiver over `sockets` that polls each with `poll` granularity and
        /// declares the stream ended once every path has been idle for `idle_limit`.
        /// Sets each socket's read timeout to `poll` so polling an idle path returns
        /// promptly instead of blocking.
        pub fn new(
            sockets: &'s [&'s UdpSocket],
            poll: Duration,
            idle_limit: Duration,
        ) -> io::Result<Self> {
            for s in sockets {
                s.set_read_timeout(Some(poll))?;
            }
            Ok(Self {
                sockets,
                dedup: SeamlessDedup::new(),
                poll,
                idle_limit,
                next: 0,
            })
        }

        /// Receive the next novel RTP packet into `buf`, returning its length, or
        /// `None` once all paths have been idle for `idle_limit` (the stream ended). A
        /// duplicate already delivered on another path is dropped silently and polling
        /// continues. Paths are polled round-robin from a rolling cursor so the merge
        /// preserves sequence order.
        pub fn recv_novel(&mut self, buf: &mut [u8]) -> io::Result<Option<usize>> {
            let n_socks = self.sockets.len();
            if n_socks == 0 {
                return Ok(None);
            }
            let mut idle = Duration::ZERO;
            loop {
                let mut got_any = false;
                for k in 0..n_socks {
                    let idx = (self.next + k) % n_socks;
                    match self.sockets[idx].recv_from(buf) {
                        Ok((n, _)) => {
                            got_any = true;
                            if self.dedup.accept(&buf[..n]) {
                                self.next = (idx + 1) % n_socks;
                                return Ok(Some(n));
                            }
                        }
                        // An idle poll of this path; try the next one.
                        Err(e)
                            if matches!(
                                e.kind(),
                                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                            ) => {}
                        Err(e) => return Err(e),
                    }
                }
                if got_any {
                    idle = Duration::ZERO;
                } else {
                    idle = idle.saturating_add(self.poll);
                    if idle >= self.idle_limit {
                        return Ok(None);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "st2110")]
pub use net::RedundantRtpReceiver;

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    /// A minimal RTP packet carrying just a sequence number (+ a payload byte).
    fn pkt(seq: u16, payload: u8) -> Vec<u8> {
        let mut p = vec![0x80, 96, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        p[2..4].copy_from_slice(&seq.to_be_bytes());
        p.push(payload);
        p
    }

    #[test]
    fn forwards_first_copy_drops_duplicate() {
        let mut d = SeamlessDedup::new();
        assert!(d.accept(&pkt(100, 1)), "first copy forwarded");
        assert!(!d.accept(&pkt(100, 1)), "second copy dropped");
        assert!(d.accept(&pkt(101, 2)), "next sequence forwarded");
        assert!(!d.accept(&pkt(101, 2)), "its duplicate dropped");
    }

    #[test]
    fn two_lossy_paths_reconstruct_the_complete_stream() {
        // 200 sequence numbers. Path A drops seq%3==0, path B drops seq%3==1, so
        // seq%3==2 arrives on both (a duplicate to drop) and no sequence is lost on
        // both. Interleave arrivals (A then B per seq) and assert the dedup forwards
        // each sequence exactly once, in order.
        let mut d = SeamlessDedup::new();
        let mut forwarded = Vec::new();
        for seq in 0u16..200 {
            let a_has = seq % 3 != 0;
            let b_has = seq % 3 != 1;
            assert!(a_has || b_has, "no sequence lost on both paths");
            if a_has && d.accept(&pkt(seq, 0xAA)) {
                forwarded.push(seq);
            }
            if b_has && d.accept(&pkt(seq, 0xBB)) {
                forwarded.push(seq);
            }
        }
        let expected: Vec<u16> = (0..200).collect();
        assert_eq!(
            forwarded, expected,
            "every sequence delivered exactly once, in order"
        );
    }

    #[test]
    fn survives_the_16bit_sequence_wrap() {
        let mut d = SeamlessDedup::new();
        // Start just below the wrap and cross it; each sequence, sent on two paths,
        // is delivered exactly once.
        let seqs: Vec<u16> = (0..10).map(|i| 65_530u16.wrapping_add(i)).collect();
        let mut count = 0;
        for &s in &seqs {
            let first = d.accept(&pkt(s, 1));
            let second = d.accept(&pkt(s, 1));
            assert!(
                first && !second,
                "seq {s}: first forwarded, dup dropped across the wrap"
            );
            count += 1;
        }
        assert_eq!(count, 10);
    }

    #[test]
    fn rejects_short_packets() {
        let mut d = SeamlessDedup::new();
        assert!(
            !d.accept(&[0u8; 8]),
            "shorter than an RTP header is dropped"
        );
    }

    #[test]
    fn reconstructs_a_frame_through_the_real_20_depacketizer() {
        use crate::st2110video::{Sampling, St2110VideoDepacketizer, St2110VideoPacketizer};
        use g2g_core::RawVideoFormat;

        // Packetize an 8x4 RGBA frame into many small packets, then split them across
        // two lossy paths (each dropping a different subset), merge with the dedup, and
        // feed the survivors to one -20 depacketizer. The frame must still complete.
        let (w, h) = (8usize, 4usize);
        let frame: Vec<u8> = (0..w * 4 * h).map(|i| (i * 13 + 1) as u8).collect();
        let mut tx = St2110VideoPacketizer::new(96, 0xABCD, Sampling::Rgba8, 12 + 2 + 6 + 8);
        let packets = tx
            .packetize(&frame, w, h, 1_000_000_000)
            .expect("packetizes");
        assert!(packets.len() > 4, "frame split into several packets");

        let mut dedup = SeamlessDedup::new();
        let mut rx = St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, w, h).unwrap();
        let mut done = None;
        for (i, p) in packets.iter().enumerate() {
            // Path A drops i%3==0, path B drops i%3==1: i%3==2 arrives on BOTH (a real
            // duplicate the dedup must drop), the rest fill from one path. No packet is
            // lost on both, so -7 recovers the whole frame.
            let a_has = i % 3 != 0;
            let b_has = i % 3 != 1;
            for present in [a_has, b_has] {
                if present && dedup.accept(p) {
                    if let Some(f) = rx.depacketize(p) {
                        done = Some(f);
                    }
                }
            }
        }
        let f = done.expect("frame completes despite each path dropping half the packets");
        assert_eq!(
            f.bytes, frame,
            "the RGBA frame is reconstructed byte-exact via -7 merge"
        );
    }
}
