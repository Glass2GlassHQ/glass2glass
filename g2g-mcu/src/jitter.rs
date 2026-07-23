//! Heap-free RTP jitter buffer: the receive-side element that absorbs network
//! arrival jitter and reordering, so a real-time decode/playout path downstream
//! sees packets in sequence. It is the RX counterpart of the TX chain's egress
//! stages, and the piece a `no_std` receive path was missing.
//!
//! [`JitterBuffer`] is a [`StaticTransform`]: each received [`Frame`] (whose
//! `sequence` is the RTP sequence number, stamped by [`RtpSrc`](crate::RtpSrc))
//! goes into a fixed reorder window of `N` slots, and the buffer emits the
//! next-in-sequence packet once it has built up a target depth of buffering.
//! Everything is fixed-capacity and heap-free: `N` inline slots of `BYTES`
//! bytes, no allocation, bounded latency (a packet is delayed at most the window
//! before it is played or declared lost).
//!
//! Reception hazards are handled explicitly and counted, so the receive path's
//! health is observable (the RX analog of the supervisor's fault accounting):
//! - **Reorder:** a packet arriving after a higher sequence is buffered and
//!   emitted in order.
//! - **Duplicate:** a sequence already buffered is dropped.
//! - **Late:** a sequence already played out (below the play cursor), or so far
//!   ahead it falls outside the window, is dropped.
//! - **Loss:** when the next expected sequence is absent but a later one has
//!   arrived, the missing packet is declared lost and the cursor advances (a
//!   gap this tick), so one lost packet never stalls the stream. (Packet-loss
//!   concealment, synthesizing a replacement, is a documented future refinement;
//!   this buffer produces the gap and counts it.)
//!
//! The output frame borrows the buffer's own slot zero-copy, sound under the
//! single-frame-in-flight discipline the static runners follow (each emitted
//! frame is consumed and dropped before the next `process` call, so the slot is
//! never overwritten while lent), the same argument as
//! [`SpscFrameRing::borrow`](g2g_core::SpscFrameRing).

use g2g_core::error::G2gError;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::staticelem::StaticTransform;

/// A fixed reorder window of `N` slots, each holding a `BYTES`-byte payload. The
/// window (`N`) must exceed the largest reorder distance plus the prime depth;
/// `BYTES` must fit the largest payload. Usable buffering is `N` packets.
pub struct JitterBuffer<const N: usize, const BYTES: usize> {
    slots: [[u8; BYTES]; N],
    len: [u16; N],
    seq_of: [u64; N],
    pts_of: [u64; N],
    present: [bool; N],
    /// Next sequence number to play out.
    cursor: u64,
    /// Packets to buffer before playout starts (jitter absorption). Clamped to
    /// `N - 1` so priming is always reachable.
    depth: u32,
    primed: bool,
    started: bool,
    present_count: u32,
    /// Highest sequence ever inserted (for reorder detection).
    max_seq: u64,
    // Reception accounting.
    reordered: u32,
    duplicates: u32,
    late: u32,
    lost: u32,
}

impl<const N: usize, const BYTES: usize> JitterBuffer<N, BYTES> {
    /// A jitter buffer that builds up `depth` packets of buffering before it
    /// starts playing out. `depth` is clamped to `N - 1` (priming must be
    /// reachable); a larger `depth` trades latency for jitter tolerance.
    pub const fn new(depth: u32) -> Self {
        // Clamp depth < N so present_count can reach it.
        let cap = if N > 1 { (N - 1) as u32 } else { 0 };
        let depth = if depth > cap { cap } else { depth };
        Self {
            slots: [[0u8; BYTES]; N],
            len: [0u16; N],
            seq_of: [0u64; N],
            pts_of: [0u64; N],
            present: [false; N],
            cursor: 0,
            depth,
            primed: false,
            started: false,
            present_count: 0,
            max_seq: 0,
            reordered: 0,
            duplicates: 0,
            late: 0,
            lost: 0,
        }
    }

    /// Packets buffered and re-ordered ahead of a higher sequence.
    pub fn reordered(&self) -> u32 {
        self.reordered
    }
    /// Duplicate packets dropped.
    pub fn duplicates(&self) -> u32 {
        self.duplicates
    }
    /// Packets dropped for arriving too late (already played) or out of window.
    pub fn late(&self) -> u32 {
        self.late
    }
    /// Missing packets declared lost (a later packet arrived past the gap).
    pub fn lost(&self) -> u32 {
        self.lost
    }
    /// Packets currently buffered (not yet played out).
    pub fn buffered(&self) -> u32 {
        self.present_count
    }

    /// Insert `payload` at sequence `seq` into its window slot, returning true
    /// if it was newly buffered (false = duplicate / late / out of window).
    fn insert(&mut self, seq: u64, pts_ns: u64, payload: &[u8]) -> bool {
        if seq < self.cursor {
            self.late = self.late.wrapping_add(1);
            return false;
        }
        if seq >= self.cursor.wrapping_add(N as u64) {
            // Beyond the reorder window: the buffer is behind (a stuck cursor
            // clears via the loss path in `try_emit`); drop and count as late.
            self.late = self.late.wrapping_add(1);
            return false;
        }
        let slot = (seq % N as u64) as usize;
        let Some(present) = self.present.get(slot) else {
            return false; // slot index always < N; never panic
        };
        if *present && self.seq_of.get(slot) == Some(&seq) {
            self.duplicates = self.duplicates.wrapping_add(1);
            return false;
        }
        if self.started && seq < self.max_seq {
            self.reordered = self.reordered.wrapping_add(1);
        }
        // Copy the payload into the slot (a bounded copy: the buffer must own
        // the bytes, since it holds many packets while lending only the head).
        if let (Some(dst), Some(len_cell), Some(seq_cell), Some(present_cell)) = (
            self.slots.get_mut(slot),
            self.len.get_mut(slot),
            self.seq_of.get_mut(slot),
            self.present.get_mut(slot),
        ) {
            let n = payload.len().min(BYTES);
            if let (Some(d), Some(src)) = (dst.get_mut(..n), payload.get(..n)) {
                d.copy_from_slice(src);
            }
            *len_cell = n as u16;
            *seq_cell = seq;
            if let Some(pts_cell) = self.pts_of.get_mut(slot) {
                *pts_cell = pts_ns;
            }
            if !*present_cell {
                self.present_count = self.present_count.saturating_add(1);
            }
            *present_cell = true;
        }
        if !self.started || seq > self.max_seq {
            self.max_seq = seq;
        }
        true
    }

    /// Emit the next in-sequence packet, skipping losses within the window.
    /// Returns the lent frame, or `None` if the head is not yet available.
    fn try_emit(&mut self) -> Option<Frame> {
        loop {
            let c = (self.cursor % N as u64) as usize;
            let is_head = matches!(self.present.get(c), Some(true))
                && self.seq_of.get(c) == Some(&self.cursor);
            if is_head {
                let len = self.len.get(c).copied().unwrap_or(0) as usize;
                let pts_ns = self.pts_of.get(c).copied().unwrap_or(0);
                let seq = self.cursor;
                if let Some(present) = self.present.get_mut(c) {
                    *present = false;
                }
                self.present_count = self.present_count.saturating_sub(1);
                self.cursor = self.cursor.wrapping_add(1);
                let slot = self.slots.get(c)?;
                let bytes = slot.get(..len)?;
                // SAFETY: the slot bytes are valid and stable, and are not
                // overwritten until a future `process` inserts into slot `c`,
                // which the single-frame-in-flight discipline guarantees happens
                // only after this frame is dropped (each emitted frame is
                // consumed before the next `process`). `free` is None: the buffer
                // reclaims the slot itself by clearing `present`.
                let slice = unsafe {
                    SystemSlice::from_foreign(bytes.as_ptr(), len, None, core::ptr::null_mut())
                };
                return Some(Frame::new(
                    MemoryDomain::System(slice),
                    FrameTiming {
                        pts_ns,
                        ..FrameTiming::default()
                    },
                    seq,
                ));
            }
            // Head missing. A jitter buffer must WAIT for a late/reordered
            // packet up to its latency budget, not declare loss the instant a
            // later packet arrives (that would defeat reordering). Declare the
            // head lost only once a packet more than `depth` sequences ahead has
            // arrived (so waiting longer would exceed the latency budget), or the
            // window is full (nothing more can be buffered). Otherwise wait.
            let max_buffered = self
                .present
                .iter()
                .zip(self.seq_of.iter())
                .filter_map(|(&p, &s)| if p { Some(s) } else { None })
                .max();
            let window_full = self.present_count as usize >= N.saturating_sub(1);
            match max_buffered {
                Some(m) if m > self.cursor.wrapping_add(self.depth as u64) || window_full => {
                    self.lost = self.lost.wrapping_add(1);
                    self.cursor = self.cursor.wrapping_add(1);
                    continue;
                }
                _ => return None,
            }
        }
    }
}

impl<const N: usize, const BYTES: usize> StaticTransform for JitterBuffer<N, BYTES> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let Some(payload) = input.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        if payload.len() > BYTES {
            return Err(G2gError::CapsMismatch);
        }
        let seq = input.sequence;
        if !self.started {
            self.cursor = seq;
            self.started = true;
        }
        self.insert(seq, input.timing.pts_ns, payload);

        if !self.primed {
            if self.present_count >= self.depth {
                self.primed = true;
            } else {
                return Ok(None); // still building the jitter cushion
            }
        }
        Ok(self.try_emit())
    }
}

impl<const N: usize, const BYTES: usize> g2g_core::supervise::Recover for JitterBuffer<N, BYTES> {
    /// Recover by flushing the buffer to its initial (un-primed) state, so
    /// playout re-syncs to the next received sequence after a fault. Counters
    /// are preserved (they are the run's health record).
    async fn recover(&mut self) -> Result<(), G2gError> {
        self.present = [false; N];
        self.present_count = 0;
        self.primed = false;
        self.started = false;
        Ok(())
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for JitterBuffer<N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("JitterBuffer")
            .field("slots", &N)
            .field("slot_bytes", &BYTES)
            .field("depth", &self.depth)
            .field("cursor", &self.cursor)
            .field("buffered", &self.present_count)
            .field("reordered", &self.reordered)
            .field("duplicates", &self.duplicates)
            .field("late", &self.late)
            .field("lost", &self.lost)
            .finish()
    }
}
