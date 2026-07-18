//! Helpers shared by the codec test binaries (`m638_g711`, `m639_adpcm`):
//! ring-lent frame construction and payload access, plus the single-poll
//! driver. One definition, included per test binary via `mod util;`.

use std::future::Future;

use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;

pub(crate) fn block_on<F: Future>(fut: F) -> F::Output {
    g2g_core::drive_ready(fut).expect("the static chain never suspends")
}

/// Lend `payload` out of a ring as a frame, as a capture path would.
#[allow(dead_code)] // not every test binary that includes this module uses it
pub(crate) fn frame_of<const N: usize, const B: usize>(
    ring: &StaticLendRing<N, B>,
    payload: &[u8],
    pts_ns: u64,
    seq: u64,
) -> Frame {
    let mut slot = ring.acquire().expect("free slot");
    slot.buf_mut()[..payload.len()].copy_from_slice(payload);
    // SAFETY: every test keeps the ring alive past the frame.
    let slice = unsafe { slot.publish(payload.len()) };
    Frame::new(
        MemoryDomain::System(slice),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        seq,
    )
}

#[allow(dead_code)] // not every test binary that includes this module uses it
pub(crate) fn payload(frame: &Frame) -> &[u8] {
    let MemoryDomain::System(s) = &frame.domain else {
        panic!("system frame")
    };
    s.as_slice()
}

// Not every test binary that includes this module uses every helper.
#[allow(dead_code)]
pub(crate) fn le_bytes(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}
