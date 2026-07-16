//! Single-producer / single-consumer capture ring: the heap-free hand-off across
//! the interrupt boundary a real MCU capture needs. A DMA-completion (or timer)
//! ISR produces frames into the ring in *interrupt context*; the pipeline, in
//! the main/task context, consumes them, concurrently and lock-free. This is the
//! piece the synchronous [`StaticLendRing`](crate::staticpool::StaticLendRing)
//! lend model (single cooperative context) does not cover: there the same task
//! fills and drains; here the producer and consumer run in genuinely different
//! execution contexts.
//!
//! It is a FIFO ring (unlike the lend pool's any-order `acquire`), so frames are
//! consumed in capture order, and a fixed `N` slots make it heap-free. Only the
//! producer writes `tail` and any slot it fills; only the consumer writes `head`
//! and reads the published slots; the `head`/`tail` Acquire/Release stores order
//! the slot bytes across the boundary, so an ISR producer and a main-context
//! consumer never race a slot. It uses only atomic load/store (no compare-and-
//! swap), so it builds on Cortex-M targets without atomic CAS (e.g. `thumbv6m`),
//! matching the rest of the no-alloc path.
//!
//! Back-pressure is explicit and non-blocking, because a producer in an ISR must
//! not block: if the consumer falls behind and the ring fills, [`produce`] drops
//! the frame and bumps an overrun counter the consumer can read
//! ([`overruns`](SpscFrameRing::overruns)), rather than stalling the interrupt.
//!
//! [`produce`]: SpscFrameRing::produce

use core::cell::UnsafeCell;
use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use crate::error::G2gError;
use crate::frame::{Frame, FrameTiming};
use crate::memory::{MemoryDomain, SystemSlice};
use crate::staticelem::StaticSource;
use crate::supervise::Recover;

/// [`SpscFrameRing::produce`] found the ring full (the consumer is behind), so
/// this call could not enqueue. For the canonical ISR producer, which attempts
/// each frame once and does not retry, that means the frame was dropped; it is
/// counted in [`SpscFrameRing::overruns`]. (A producer that instead retries the
/// same frame will see this per full-ring event, not per lost frame.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Overrun;

/// A fixed-capacity SPSC FIFO of `BYTES`-sized frames for the ISR-to-pipeline
/// capture hand-off. `N` slots live inline (no `alloc`); usable capacity is
/// `N - 1` (one slot separates a full ring from an empty one), so `N >= 2`, and
/// `N >= 3` gives a double-buffer plus one frame in flight.
pub struct SpscFrameRing<const N: usize, const BYTES: usize> {
    slots: [UnsafeCell<[u8; BYTES]>; N],
    /// Consumer cursor: index of the next slot to consume. Only the consumer
    /// writes it; the producer reads it (Acquire) for the full check.
    head: AtomicUsize,
    /// Producer cursor: index of the next slot to fill. Only the producer writes
    /// it (Release, to publish); the consumer reads it (Acquire) for emptiness.
    tail: AtomicUsize,
    /// Frames the producer dropped on a full ring. Only the producer writes it
    /// (it is the sole producer, so a load+store increment needs no CAS).
    overruns: AtomicU32,
}

// SAFETY: strict single-producer / single-consumer. The producer is the sole
// writer of `tail` and of any slot it fills; the consumer is the sole writer of
// `head` and the sole reader of the published slots [head, tail). A slot is
// filled only when free (the full check keeps `tail` from catching `head`, so
// the producer's slot and the consumer's slot are always distinct) and read only
// once published. The producer's slot write is ordered before its `tail` Release
// store, which the consumer's `tail` Acquire load synchronizes-with before it
// reads the slot; symmetrically the consumer's `head` Release store frees a slot
// the producer's `head` Acquire load observes before reuse. So an ISR producer
// and a main-context consumer never form a data race. Only atomic load/store is
// used (no CAS), so it builds on targets without atomic CAS (e.g. `thumbv6m`).
unsafe impl<const N: usize, const BYTES: usize> Sync for SpscFrameRing<N, BYTES> {}

impl<const N: usize, const BYTES: usize> SpscFrameRing<N, BYTES> {
    // Associated const as the array-repeat operand: the MSRV-1.75 way to build
    // the slot array in a `const fn` (inline-const repeat needs 1.79). Copying a
    // fresh zeroed slot into each array element is exactly the intent.
    #[allow(clippy::declare_interior_mutable_const)]
    const EMPTY_SLOT: UnsafeCell<[u8; BYTES]> = UnsafeCell::new([0u8; BYTES]);

    /// Build an empty ring. `const`, so it lives in a `static` (the DMA-ring
    /// idiom) shared between the producer ISR and the consumer. `N >= 2`.
    pub const fn new() -> Self {
        Self {
            slots: [Self::EMPTY_SLOT; N],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            overruns: AtomicU32::new(0),
        }
    }

    /// Slot count (the const `N`); usable capacity is `N - 1`.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Full-ring events: [`produce`](Self::produce) calls that found no free
    /// slot. For the canonical ISR producer (one attempt per frame, no retry)
    /// this is the count of frames dropped to back-pressure.
    pub fn overruns(&self) -> u32 {
        self.overruns.load(Ordering::Relaxed)
    }

    /// True if the ring currently holds no published frames.
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    /// The next index after `i`, wrapping at `N`. `i` is always `< N` (the
    /// cursors only advance through here), so this both wraps and keeps them in
    /// range without a bounds panic.
    fn wrap(i: usize) -> usize {
        if i + 1 >= N {
            0
        } else {
            i + 1
        }
    }

    /// PRODUCER side (call from exactly one context, e.g. a DMA-completion ISR):
    /// fill the next free slot via `fill` and publish it to the consumer.
    ///
    /// Returns `Err(Overrun)` if the ring is full (the consumer is behind): the
    /// frame is dropped and counted, never blocked on, because a producer in an
    /// interrupt cannot wait.
    pub fn produce(&self, fill: impl FnOnce(&mut [u8; BYTES])) -> Result<(), Overrun> {
        let tail = self.tail.load(Ordering::Relaxed); // sole producer owns tail
        let next = Self::wrap(tail);
        if next == self.head.load(Ordering::Acquire) {
            // Full: drop and count (single producer, so a load+store bump on the
            // counter needs no CAS and cannot lose an update).
            self.overruns.store(self.overruns.load(Ordering::Relaxed).wrapping_add(1), Ordering::Relaxed);
            return Err(Overrun);
        }
        let Some(cell) = self.slots.get(tail) else {
            // `tail` is always `< N`; unreachable, but never panic on a bad index.
            return Err(Overrun);
        };
        // SAFETY: slot[tail] is free (outside the published range [head, tail),
        // guaranteed by the full check above), and the producer is its sole
        // writer until the Release store below publishes it.
        fill(unsafe { &mut *cell.get() });
        self.tail.store(next, Ordering::Release); // publish
        Ok(())
    }

    /// CONSUMER side: borrow the oldest published frame zero-copy as a
    /// [`SystemSlice`], or `None` if the ring is empty. The borrow aliases the
    /// ring slot (no copy); it stays valid until [`release`](Self::release)
    /// advances past it.
    ///
    /// Contract: borrow at most one frame at a time and [`release`] it before the
    /// next borrow, after the frame (and its slice) is dropped, the single-frame-
    /// in-flight discipline the static runners already follow (each frame is
    /// dropped before the next `next()`). Releasing a still-referenced slice
    /// would let the producer reuse the slot under the reader.
    pub fn borrow(&self) -> Option<SystemSlice> {
        let head = self.head.load(Ordering::Relaxed); // sole consumer owns head
        if head == self.tail.load(Ordering::Acquire) {
            return None; // empty
        }
        let ptr = self.slots.get(head)?.get() as *const u8;
        // SAFETY: slot[head] was fully written by the producer before the
        // `tail` Release store this Acquire load observed, so the bytes are
        // valid and stable; the producer will not reuse this slot until
        // `release` advances `head` past it (the full check), so the read-only
        // lend stays valid. `free` is None: the consumer reclaims the slot
        // explicitly via `release`, not on the slice's drop.
        Some(unsafe { SystemSlice::from_foreign(ptr, BYTES, None, core::ptr::null_mut()) })
    }

    /// CONSUMER side: reclaim the slot last returned by [`borrow`], freeing it for
    /// the producer to refill. Call once per consumed frame, after the frame is
    /// dropped. A `release` with nothing borrowed is a no-op.
    pub fn release(&self) {
        let head = self.head.load(Ordering::Relaxed);
        if head != self.tail.load(Ordering::Acquire) {
            self.head.store(Self::wrap(head), Ordering::Release);
        }
    }
}

impl<const N: usize, const BYTES: usize> Default for SpscFrameRing<N, BYTES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for SpscFrameRing<N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpscFrameRing")
            .field("capacity", &N)
            .field("slot_bytes", &BYTES)
            .field("empty", &self.is_empty())
            .field("overruns", &self.overruns())
            .finish()
    }
}

/// The consumer side of an [`SpscFrameRing`] as a [`StaticSource`]: it drains the
/// ring the producer ISR fills, yielding each captured frame zero-copy (the frame
/// borrows the ring slot; the slot is reclaimed when the runner drops the frame
/// and the next `next()` releases it). While the ring is empty it calls the
/// caller-supplied `idle` hook and retries, so the consumer sleeps instead of
/// spinning: on hardware `idle` is `cortex_m::asm::wfi` (wait for the capture
/// interrupt), in a host test a yield/`spin_loop`. This is the ISR-driven capture
/// source, the concurrent twin of the synchronous `GrabberSrc`.
///
/// Single frame in flight (the static runners drop each frame before the next
/// `next()`), which is what makes the zero-copy borrow sound: the borrowed slot
/// is released only after its frame is gone.
pub struct SpscCaptureSrc<'r, I, const N: usize, const BYTES: usize> {
    ring: &'r SpscFrameRing<N, BYTES>,
    idle: I,
    frame_interval_ns: u64,
    remaining: Option<u32>,
    seq: u64,
    holding: bool,
}

impl<'r, I: FnMut(), const N: usize, const BYTES: usize> SpscCaptureSrc<'r, I, N, BYTES> {
    /// A capture source draining `ring` (filled by a producer in another context,
    /// e.g. a DMA/timer ISR). `idle` runs while waiting for the producer to
    /// publish a frame, `cortex_m::asm::wfi` on hardware (sleep until the next
    /// interrupt), a yield or `core::hint::spin_loop` in a host test.
    /// `frame_interval_ns` sets the derived PTS cadence.
    pub fn new(ring: &'r SpscFrameRing<N, BYTES>, idle: I, frame_interval_ns: u64) -> Self {
        Self { ring, idle, frame_interval_ns, remaining: None, seq: 0, holding: false }
    }

    /// End the stream after `frames` captures (a capture is endless by default).
    pub fn with_frame_limit(mut self, frames: u32) -> Self {
        self.remaining = Some(frames);
        self
    }
}

impl<I: FnMut(), const N: usize, const BYTES: usize> StaticSource
    for SpscCaptureSrc<'_, I, N, BYTES>
{
    async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
        // The frame from the previous `next()` has been consumed and dropped by
        // the runner; reclaim its ring slot for the producer.
        if self.holding {
            self.ring.release();
            self.holding = false;
        }
        if let Some(remaining) = &mut self.remaining {
            if *remaining == 0 {
                return Ok(None);
            }
            *remaining -= 1;
        }
        // Wait for the producer (ISR) to publish a frame, idling (WFI) meanwhile.
        loop {
            if let Some(slice) = self.ring.borrow() {
                self.holding = true;
                let pts_ns = self.seq.saturating_mul(self.frame_interval_ns);
                let frame = Frame::new(
                    MemoryDomain::System(slice),
                    FrameTiming { pts_ns, ..FrameTiming::default() },
                    self.seq,
                );
                self.seq = self.seq.wrapping_add(1);
                return Ok(Some(frame));
            }
            (self.idle)();
        }
    }
}

impl<I: FnMut(), const N: usize, const BYTES: usize> Recover for SpscCaptureSrc<'_, I, N, BYTES> {
    /// Recover a capture source after a fault by dropping any stale buffered
    /// frames, so the pipeline resumes from live data instead of replaying a
    /// backlog that accumulated while the fault was handled (the real-time
    /// choice for a display / egress path). Bounded by the ring capacity: the
    /// producer ISR can refill during the drain, but at most `N` slots exist,
    /// so this cannot spin.
    async fn recover(&mut self) -> Result<(), G2gError> {
        if self.holding {
            self.ring.release();
            self.holding = false;
        }
        for _ in 0..N {
            if self.ring.is_empty() {
                break;
            }
            self.ring.release();
        }
        Ok(())
    }
}

impl<I, const N: usize, const BYTES: usize> core::fmt::Debug for SpscCaptureSrc<'_, I, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SpscCaptureSrc")
            .field("slots", &N)
            .field("slot_bytes", &BYTES)
            .field("frame_interval_ns", &self.frame_interval_ns)
            .field("remaining", &self.remaining)
            .field("seq", &self.seq)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first payload byte of the currently-borrowed frame, or `None` if empty.
    fn peek<const N: usize, const B: usize>(ring: &SpscFrameRing<N, B>) -> Option<u8> {
        ring.borrow().map(|s| s.as_slice()[0])
    }

    #[test]
    fn fifo_order_and_capacity() {
        let ring: SpscFrameRing<4, 2> = SpscFrameRing::new();
        assert_eq!(ring.capacity(), 4);
        assert!(ring.is_empty());
        // Fill to usable capacity (N-1 = 3).
        for k in 0..3u8 {
            ring.produce(|b| b[0] = k + 1).expect("space");
        }
        assert!(!ring.is_empty());
        // Consume in capture order: 1, 2, 3.
        for k in 0..3u8 {
            assert_eq!(peek(&ring), Some(k + 1), "FIFO order");
            ring.release();
        }
        assert!(ring.is_empty());
        assert_eq!(ring.overruns(), 0);
    }

    #[test]
    fn full_ring_drops_and_counts_overruns() {
        let ring: SpscFrameRing<3, 1> = SpscFrameRing::new(); // usable capacity 2
        assert!(ring.produce(|b| b[0] = 10).is_ok());
        assert!(ring.produce(|b| b[0] = 20).is_ok());
        // Third produce with no consume: full, dropped, counted.
        assert_eq!(ring.produce(|b| b[0] = 30), Err(Overrun));
        assert_eq!(ring.overruns(), 1);
        // The dropped frame never entered the FIFO: consumer still sees 10, 20.
        assert_eq!(peek(&ring), Some(10));
        ring.release();
        // A slot freed; the producer can enqueue again (the newest, 40).
        assert!(ring.produce(|b| b[0] = 40).is_ok());
        assert_eq!(peek(&ring), Some(20));
        ring.release();
        assert_eq!(peek(&ring), Some(40));
        ring.release();
        assert!(ring.is_empty());
    }

    #[test]
    fn interleaved_produce_consume_wraps_around() {
        // Cycle many more frames than N through the ring, one in flight at a
        // time (the pipeline's single-frame discipline), forcing several wraps.
        let ring: SpscFrameRing<3, 1> = SpscFrameRing::new();
        for k in 0..20u8 {
            ring.produce(|b| b[0] = k).expect("space (one in flight)");
            assert_eq!(peek(&ring), Some(k), "each frame consumed in order across wraps");
            ring.release();
        }
        assert!(ring.is_empty());
        assert_eq!(ring.overruns(), 0);
    }

    #[test]
    fn release_without_borrow_is_a_noop() {
        let ring: SpscFrameRing<2, 1> = SpscFrameRing::new();
        ring.release(); // empty: must not corrupt the cursor
        assert!(ring.is_empty());
        ring.produce(|b| b[0] = 7).expect("space");
        assert_eq!(peek(&ring), Some(7));
    }
}
