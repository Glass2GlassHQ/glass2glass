//! Shared plumbing for payload-producing codec transforms: validate the
//! input, convert into an acquired [`StaticLendRing`] slot, publish it
//! zero-copy (the capture source's lend model applied to a transform's
//! output).

use g2g_core::error::G2gError;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;

/// Lend a ring slot for a produced payload: `out_len` is the exact output
/// byte count, `fill` writes it. An output that cannot fit a slot is a
/// sizing bug surfaced as [`G2gError::CapsMismatch`]; an exhausted ring is
/// [`G2gError::PoolExhausted`] (see [`GrabberSrc`](crate::GrabberSrc) for
/// why it is not waited out). The output frame carries the given timing and
/// sequence (the producer decides which input is the timing master).
///
/// # Safety
/// The ring must outlive the published frame (the
/// [`RingSlot::publish`](g2g_core::staticpool::RingSlot::publish) contract);
/// the element constructors establish it (`new`: `'static`; `with_ring`:
/// the caller's contract).
pub(crate) unsafe fn lend_slot<const N: usize, const BYTES: usize>(
    ring: &StaticLendRing<N, BYTES>,
    timing: FrameTiming,
    sequence: u64,
    out_len: usize,
    fill: impl FnOnce(&mut [u8]),
) -> Result<Frame, G2gError> {
    if out_len > BYTES {
        return Err(G2gError::CapsMismatch);
    }
    let Some(mut slot) = ring.acquire() else {
        return Err(G2gError::PoolExhausted);
    };
    if let Some(dst) = slot.buf_mut().get_mut(..out_len) {
        fill(dst);
    }
    // SAFETY: the caller (the element constructor contract) established that
    // the ring outlives every published frame.
    let out = unsafe { slot.publish(out_len) };
    Ok(Frame::new(MemoryDomain::System(out), timing, sequence))
}

/// [`lend_slot`] for the 1:1 transform shape: `fill` converts the input
/// payload, and the output frame inherits the input's timing and sequence.
///
/// # Safety
/// Same contract as [`lend_slot`].
pub(crate) unsafe fn lend_converted<const N: usize, const BYTES: usize>(
    ring: &StaticLendRing<N, BYTES>,
    input: &Frame,
    out_len: usize,
    fill: impl FnOnce(&[u8], &mut [u8]),
) -> Result<Frame, G2gError> {
    let Some(slice) = input.domain.as_system_slice() else {
        return Err(G2gError::UnsupportedDomain);
    };
    // SAFETY: forwarded; the caller upholds the ring-outlives-frames contract.
    unsafe {
        lend_slot(ring, input.timing, input.sequence, out_len, |dst| {
            fill(slice, dst)
        })
    }
}
