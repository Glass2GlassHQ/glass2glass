//! Audio render seam: the I2S / SAI-shaped [`PcmWriter`] contract plus
//! [`PcmSink`], the heap-free [`StaticSink`] that streams a frame's
//! interleaved S16LE payload to the audio peripheral. The element owns the
//! byte -> sample decoding and framing validation; a board supplies only the
//! writer, an adapter whose `write` typically feeds the peripheral's DMA ring
//! and awaits space (an Embassy SAI/I2S transfer).
//!
//! Like the rest of `g2g-mcu`, this is host-testable: a mock writer records
//! the decoded samples and the tests assert them against the datasheet-level
//! contract (see `m631_pcm_sink.rs`).

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{StaticSink, StaticTransform};

use crate::lend::lend_converted;

/// Samples decoded per write burst; the element needs only this fixed stack
/// buffer regardless of frame size.
const CHUNK_SAMPLES: usize = 64;

/// One block of interleaved signed 16-bit PCM handed to the audio peripheral.
///
/// An I2S/SAI adapter typically copies (or DMA-feeds) the block and awaits
/// ring space; blocking here is the pipeline's natural audio back-pressure.
#[allow(async_fn_in_trait)]
pub trait PcmWriter {
    /// Render one block of interleaved samples.
    async fn write(&mut self, samples: &[i16]) -> Result<(), G2gError>;
}

/// A heap-free [`StaticSink`] rendering S16LE interleaved audio frames
/// through a [`PcmWriter`], decoding via a fixed [`CHUNK_SAMPLES`] stack
/// buffer. A frame whose payload is not a whole number of interleaved sample
/// frames (`2 bytes x channels`) is rejected as [`G2gError::CapsMismatch`]
/// before anything reaches the peripheral.
pub struct PcmSink<W> {
    writer: W,
    channels: u8,
}

impl<W: PcmWriter> PcmSink<W> {
    /// A sink rendering `channels`-channel interleaved S16LE frames.
    pub fn new(writer: W, channels: u8) -> Self {
        Self { writer, channels }
    }

    /// Release the writer (e.g. to reconfigure the peripheral).
    pub fn free(self) -> W {
        self.writer
    }
}

impl<W: PcmWriter> StaticSink for PcmSink<W> {
    async fn consume(&mut self, frame: Frame) -> Result<(), G2gError> {
        let MemoryDomain::System(slice) = &frame.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let bytes = slice.as_slice();
        // Whole interleaved sample frames only: 2 bytes per sample x channels.
        let frame_bytes = 2usize.saturating_mul(self.channels.max(1) as usize);
        if bytes.len() % frame_bytes != 0 {
            return Err(G2gError::CapsMismatch);
        }
        let mut buf = [0i16; CHUNK_SAMPLES];
        for block in bytes.chunks(CHUNK_SAMPLES * 2) {
            let mut used = 0usize;
            for (dst, src) in buf.iter_mut().zip(block.chunks_exact(2)) {
                // Slice patterns, not indexing: no bounds-check panic path may
                // enter the no-alloc subset (chunk sizes make this infallible).
                let &[lo, hi] = src else { continue };
                *dst = i16::from_le_bytes([lo, hi]);
                used = used.saturating_add(1);
            }
            let Some(out) = buf.get(..used) else { continue };
            self.writer.write(out).await?;
        }
        Ok(())
    }
}

impl<W> core::fmt::Debug for PcmSink<W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PcmSink").field("channels", &self.channels).finish()
    }
}

/// Narrow one 32-bit capture slot to S16: keep the top 16 bits (arithmetic
/// shift). This is the left-justified I2S/SAI convention, where a 24-bit (or
/// 18/20-bit) converter's sample sits at the top of the 32-bit slot; the low
/// bits are the extra precision S16 drops.
pub const fn s32_slot_to_s16(slot: i32) -> i16 {
    (slot >> 16) as i16
}

/// The `convert` stage of the reference audio chain (M644): interleaved
/// S32LE capture slots in (the shape an I2S/SAI DMA delivers when the bus
/// runs 32-bit slots, left-justified), interleaved S16LE out (2:1), lent
/// from a [`StaticLendRing`] like the codec transforms. A payload that is
/// not whole 32-bit slots is rejected as [`G2gError::CapsMismatch`]. The
/// per-slot narrowing is the free [`s32_slot_to_s16`]; channel count is
/// irrelevant (per-sample, interleaving preserved).
pub struct PcmConvert<'r, const N: usize, const BYTES: usize> {
    ring: &'r StaticLendRing<N, BYTES>,
}

impl<const N: usize, const BYTES: usize> PcmConvert<'static, N, BYTES> {
    /// A converter over a `'static` ring (the MCU idiom), which makes the
    /// zero-copy lend safe by construction.
    pub fn new(ring: &'static StaticLendRing<N, BYTES>) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(ring) }
    }
}

impl<'r, const N: usize, const BYTES: usize> PcmConvert<'r, N, BYTES> {
    /// A converter over a borrowed (e.g. stack-local) ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this element publishes: the
    /// pipeline must drain before the ring is dropped.
    pub unsafe fn with_ring(ring: &'r StaticLendRing<N, BYTES>) -> Self {
        Self { ring }
    }
}

impl<const N: usize, const BYTES: usize> StaticTransform for PcmConvert<'_, N, BYTES> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let MemoryDomain::System(slice) = &input.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let len = slice.as_slice().len();
        // Whole 32-bit slots only.
        if len % 4 != 0 {
            return Err(G2gError::CapsMismatch);
        }
        // SAFETY: the constructor established the ring-outlives-frames
        // contract (`new`: 'static; `with_ring`: caller's contract).
        let out = unsafe {
            lend_converted(self.ring, &input, len / 2, |src, dst| {
                for (slot, pair) in src.chunks_exact(4).zip(dst.chunks_exact_mut(2)) {
                    // Slice patterns, not indexing: no bounds-check panic
                    // path may enter the no-alloc subset.
                    let &[b0, b1, b2, b3] = slot else { continue };
                    let [lo, hi] = pair else { continue };
                    let narrow = s32_slot_to_s16(i32::from_le_bytes([b0, b1, b2, b3]));
                    [*lo, *hi] = narrow.to_le_bytes();
                }
            })?
        };
        Ok(Some(out))
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for PcmConvert<'_, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PcmConvert").finish_non_exhaustive()
    }
}
