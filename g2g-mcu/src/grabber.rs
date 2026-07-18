//! Camera capture seam: the DCMI / CSI-shaped [`FrameGrabber`] contract plus
//! [`GrabberSrc`], the heap-free [`StaticSource`] that lends each captured
//! frame downstream zero-copy from a [`StaticLendRing`]. The element owns the
//! ring/lease/timing logic; a board supplies only the grabber, an adapter
//! whose `capture` typically arms the peripheral's DMA into the lent slot and
//! awaits the completion interrupt (an Embassy DCMI transfer, a HAL callback).
//!
//! Like the rest of `g2g-mcu`, this is host-testable: a mock grabber that
//! fills the buffer synchronously exercises the real element end to end (see
//! `m630_grabber_src.rs`, which runs a whole camera -> display pipeline on
//! mock peripherals).

use g2g_core::error::G2gError;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticSource;

/// One camera capture into a caller-provided buffer (a lent ring slot).
///
/// `capture` fills `buf` with one frame and returns the byte count written
/// (a fixed-format camera returns `buf.len()`; a compressed-output one may
/// return less). A DCMI/CSI adapter arms DMA at `buf` and awaits completion;
/// returning more than `buf.len()` is a contract violation [`GrabberSrc`]
/// rejects as [`G2gError::CapsMismatch`].
#[allow(async_fn_in_trait)]
pub trait FrameGrabber {
    /// Capture one frame into `buf`, returning the bytes written.
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError>;

    /// Re-initialize the capture peripheral after a fault (re-arm the DMA,
    /// re-enable the DCMI/CSI, clear an error flag), so the next `capture`
    /// starts from a known-good state. The default is a no-op for a grabber
    /// with no peripheral state to reset; the supervisor's
    /// [`Recovery::Reset`](g2g_core::supervise::Recovery::Reset) invokes it via
    /// [`GrabberSrc`]'s [`Recover`](g2g_core::supervise::Recover) impl.
    async fn reset(&mut self) -> Result<(), G2gError> {
        Ok(())
    }
}

/// A heap-free capture [`StaticSource`]: acquires a [`StaticLendRing`] slot,
/// has the [`FrameGrabber`] fill it, and publishes it downstream zero-copy
/// with sequence numbering and interval-derived PTS.
///
/// The ring must have more slots than the pipeline holds in flight at once
/// (for the static runners, which drop each frame before the next `next()`
/// call, two slots already suffice); an exhausted ring is a sizing bug
/// surfaced as [`G2gError::PoolExhausted`], not silently waited out (with a
/// single cooperative executor there is nothing that could free a slot while
/// the source spins).
pub struct GrabberSrc<'r, G, const N: usize, const BYTES: usize> {
    grabber: G,
    ring: &'r StaticLendRing<N, BYTES>,
    frame_interval_ns: u64,
    remaining: Option<u32>,
    seq: u64,
}

impl<G: FrameGrabber, const N: usize, const BYTES: usize> GrabberSrc<'static, G, N, BYTES> {
    /// A capture source over a `'static` ring (the MCU idiom: the DMA ring
    /// lives in a `static` / `StaticCell`), which makes the zero-copy lend
    /// safe by construction: the ring outlives every published frame.
    /// `frame_interval_ns` is the nominal frame period used to derive PTS.
    pub fn new(
        grabber: G,
        ring: &'static StaticLendRing<N, BYTES>,
        frame_interval_ns: u64,
    ) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(grabber, ring, frame_interval_ns) }
    }
}

impl<'r, G: FrameGrabber, const N: usize, const BYTES: usize> GrabberSrc<'r, G, N, BYTES> {
    /// A capture source over a borrowed (e.g. stack-local) ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this source publishes (the
    /// [`RingSlot::publish`](g2g_core::staticpool::RingSlot::publish)
    /// contract): the pipeline must drain before the ring is dropped.
    pub unsafe fn with_ring(
        grabber: G,
        ring: &'r StaticLendRing<N, BYTES>,
        frame_interval_ns: u64,
    ) -> Self {
        Self {
            grabber,
            ring,
            frame_interval_ns,
            remaining: None,
            seq: 0,
        }
    }

    /// End the stream after `frames` captures (a camera is endless by
    /// default; proofs and tests bound it).
    pub fn with_frame_limit(mut self, frames: u32) -> Self {
        self.remaining = Some(frames);
        self
    }

    /// Release the grabber (e.g. to reconfigure the peripheral).
    pub fn free(self) -> G {
        self.grabber
    }
}

impl<G: FrameGrabber, const N: usize, const BYTES: usize> StaticSource
    for GrabberSrc<'_, G, N, BYTES>
{
    async fn next(&mut self) -> Result<Option<Frame>, G2gError> {
        if self.remaining == Some(0) {
            return Ok(None);
        }
        let Some(mut slot) = self.ring.acquire() else {
            return Err(G2gError::PoolExhausted);
        };
        let len = self.grabber.capture(slot.buf_mut()).await?;
        if len > BYTES {
            return Err(G2gError::CapsMismatch);
        }
        // Count the frame only once it is actually produced, so a fault (an
        // early return above) does not consume the limit; under the supervisor a
        // retried capture must not eat into the frame budget.
        if let Some(remaining) = &mut self.remaining {
            *remaining -= 1;
        }
        // SAFETY: the constructor established that the ring outlives every
        // published frame (`new`: 'static; `with_ring`: caller's contract).
        let slice = unsafe { slot.publish(len) };
        let pts_ns = self.seq.saturating_mul(self.frame_interval_ns);
        let frame = Frame::new(
            MemoryDomain::System(slice),
            FrameTiming {
                pts_ns,
                ..FrameTiming::default()
            },
            self.seq,
        );
        self.seq += 1;
        Ok(Some(frame))
    }
}

impl<G: FrameGrabber, const N: usize, const BYTES: usize> g2g_core::supervise::Recover
    for GrabberSrc<'_, G, N, BYTES>
{
    /// Recover the capture source by re-initializing its [`FrameGrabber`]
    /// peripheral (`reset`), so a supervised pipeline re-arms the camera after a
    /// bus fault. Sequence numbering and PTS cadence continue unchanged (a reset
    /// re-arms the peripheral, it does not restart the stream).
    async fn recover(&mut self) -> Result<(), G2gError> {
        self.grabber.reset().await
    }
}

impl<G, const N: usize, const BYTES: usize> core::fmt::Debug for GrabberSrc<'_, G, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GrabberSrc")
            .field("slots", &N)
            .field("slot_bytes", &BYTES)
            .field("frame_interval_ns", &self.frame_interval_ns)
            .field("remaining", &self.remaining)
            .field("seq", &self.seq)
            .finish()
    }
}
