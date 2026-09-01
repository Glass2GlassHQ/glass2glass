//! Timestamp burn-in (`timestampburn`, `latency-bench` feature): overwrites the
//! top-left of an I420 frame's luma plane with a 64-block strip carrying the
//! `CLOCK_MONOTONIC` nanoseconds at which the frame passed through. A consumer
//! that decodes the strip and subtracts it from its own `CLOCK_MONOTONIC` gets
//! the elapsed time across everything in between, measured by one clock.
//!
//! This is the instrument behind `tools/latency-bench-e2e.sh`, which puts the
//! same burned stream through a g2g consumer and a GStreamer one and reads both
//! with the same binary, so the two numbers cover the same span. The encode and
//! decode halves live here together, and the `g2g-latency-reader` binary calls
//! the decode half, so the layout cannot drift between writer and reader.
//!
//! The strip is one bit per block, most significant first, blocks laid left to
//! right along the top edge. A block is a solid square of video-range white for
//! a 1 and video-range black for a 0, sized so a lossy encoder at a normal
//! bitrate cannot blur one into its neighbour. Chroma is untouched, so the strip
//! renders grey on a colour frame. The layout is fixed: no properties.
//!
//! Sub-nanosecond framing detail: the timestamp is read inside `apply`, after
//! the frame arrives, so the span a reader computes starts at burn time and not
//! at capture.
//!
//! Unix only: it reads `clock_gettime(CLOCK_MONOTONIC)` through `libc`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, ElementMetadata, G2gError, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, RawVideoFormat,
};

use crate::pixel::frame_byte_size;
use crate::videofx::{self, FilterState, PixelFilter};

const FORMATS: [RawVideoFormat; 1] = [RawVideoFormat::I420];

/// Bits in the burned timestamp, one per block.
pub const BLOCK_COUNT: usize = 64;
/// Side of one block, in luma samples. Large enough that an ultrafast H.264
/// encode keeps each block's interior flat.
pub const BLOCK_PX: usize = 16;
/// Luma of a block standing for a 1 bit (video-range white).
pub const LUMA_ONE: u8 = 235;
/// Luma of a block standing for a 0 bit (video-range black).
pub const LUMA_ZERO: u8 = 16;
/// Luma samples the strip spans horizontally. A frame narrower than this cannot
/// carry a timestamp.
pub const STRIP_WIDTH_PX: usize = BLOCK_COUNT * BLOCK_PX;

/// Side of the square sampled at each block's centre when decoding. Compression
/// smears the block edges, so the reader keeps clear of them.
const SAMPLE_PX: usize = 8;
/// How far a sampled block mean may sit from [`LUMA_ZERO`] or [`LUMA_ONE`] and
/// still count as legible.
const SAMPLE_TOLERANCE: u8 = 48;

/// Write `ns` across the top-left of a luma plane as [`BLOCK_COUNT`] blocks,
/// most significant bit first. `luma` must hold at least [`BLOCK_PX`] rows of
/// `stride` bytes, and `stride` at least [`STRIP_WIDTH_PX`].
pub fn burn(luma: &mut [u8], stride: usize, ns: u64) {
    debug_assert!(stride >= STRIP_WIDTH_PX && luma.len() >= stride * BLOCK_PX);
    for row in 0..BLOCK_PX {
        let line = &mut luma[row * stride..row * stride + STRIP_WIDTH_PX];
        for (block, chunk) in line.as_chunks_mut::<BLOCK_PX>().0.iter_mut().enumerate() {
            let bit = (ns >> (BLOCK_COUNT - 1 - block)) & 1;
            chunk.fill(if bit == 1 { LUMA_ONE } else { LUMA_ZERO });
        }
    }
}

/// Read back a timestamp [`burn`] wrote, or `None` when the strip is too
/// degraded to trust: at least half the blocks must sample close to one of the
/// two levels.
pub fn decode(luma: &[u8], stride: usize) -> Option<u64> {
    if stride < STRIP_WIDTH_PX || luma.len() < stride * BLOCK_PX {
        return None;
    }
    const OFFSET: usize = (BLOCK_PX - SAMPLE_PX) / 2;
    const MIDPOINT: u8 = LUMA_ZERO / 2 + LUMA_ONE / 2;

    let mut ns = 0u64;
    let mut legible = 0usize;
    for block in 0..BLOCK_COUNT {
        let left = block * BLOCK_PX + OFFSET;
        let mut sum = 0u32;
        for row in OFFSET..OFFSET + SAMPLE_PX {
            let start = row * stride + left;
            sum += luma[start..start + SAMPLE_PX]
                .iter()
                .map(|&s| u32::from(s))
                .sum::<u32>();
        }
        let mean = (sum / (SAMPLE_PX * SAMPLE_PX) as u32) as u8;
        if mean.abs_diff(LUMA_ZERO) <= SAMPLE_TOLERANCE
            || mean.abs_diff(LUMA_ONE) <= SAMPLE_TOLERANCE
        {
            legible += 1;
        }
        ns = (ns << 1) | u64::from(mean >= MIDPOINT);
    }
    (legible * 2 >= BLOCK_COUNT).then_some(ns)
}

/// `CLOCK_MONOTONIC` as nanoseconds. Raw rather than [`std::time::Instant`] so
/// two processes on one machine can subtract each other's readings.
pub fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, writable `timespec`; `clock_gettime` only writes
    // into it and returns 0 on success.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime(CLOCK_MONOTONIC) failed");
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}

/// # Example
///
/// ```no_run
/// use g2g_plugins::timestampburn::TimestampBurn;
///
/// // videotestsrc ! timestampburn ! x264enc
/// let burn = TimestampBurn::new();
/// ```
#[derive(Debug, Default)]
pub struct TimestampBurn {
    state: FilterState,
}

impl TimestampBurn {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PixelFilter for TimestampBurn {
    const FORMATS: &'static [RawVideoFormat] = &FORMATS;

    fn state(&self) -> &FilterState {
        &self.state
    }

    fn state_mut(&mut self) -> &mut FilterState {
        &mut self.state
    }

    fn apply(&mut self, format: RawVideoFormat, w: u32, h: u32, src: &[u8]) -> Box<[u8]> {
        let bytes = frame_byte_size(format, w, h);
        let mut dst = vec![0u8; bytes].into_boxed_slice();
        dst.copy_from_slice(&src[..bytes]);
        let stride = w as usize;
        burn(&mut dst[..stride * h as usize], stride, monotonic_ns());
        dst
    }
}

impl AsyncElement for TimestampBurn {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Monotonic timestamp burn-in",
            "Filter/Effect/Video",
            "Burns the current CLOCK_MONOTONIC time into the luma plane as a block strip",
            "g2g",
        )
    }

    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        videofx::intercept_caps::<Self>(upstream_caps)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        videofx::same_caps_constraint::<Self>()
    }

    /// Rejects a frame too small to hold the strip, so a mis-sized bench
    /// pipeline fails at negotiation instead of burning a truncated timestamp.
    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (_, w, h, _) = videofx::accept_input::<Self>(absolute_caps)?;
        if (w as usize) < STRIP_WIDTH_PX || (h as usize) < BLOCK_PX {
            return Err(G2gError::CapsMismatch);
        }
        videofx::configure(self, absolute_caps)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        videofx::drive(self, packet, out)
    }
}

impl PadTemplates for TimestampBurn {
    fn pad_templates() -> alloc::vec::Vec<PadTemplate> {
        videofx::pad_templates::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRIDE: usize = STRIP_WIDTH_PX;

    #[test]
    fn burned_timestamps_round_trip_exactly() {
        let mut luma = vec![0u8; STRIDE * BLOCK_PX];
        for ns in [
            0,
            1,
            u64::MAX,
            0xA5A5_5A5A_A5A5_5A5A,
            monotonic_ns(),
            1_234_567_890_123_456_789,
        ] {
            burn(&mut luma, STRIDE, ns);
            assert_eq!(decode(&luma, STRIDE), Some(ns), "round trip of {ns}");
        }
    }

    #[test]
    fn a_wider_frame_keeps_its_pixels_outside_the_strip() {
        const WIDE: usize = STRIP_WIDTH_PX + 64;
        let mut luma = vec![7u8; WIDE * (BLOCK_PX + 4)];
        burn(&mut luma, WIDE, 42);
        assert_eq!(decode(&luma, WIDE), Some(42));
        assert!(luma[STRIP_WIDTH_PX..WIDE].iter().all(|&s| s == 7));
        assert!(luma[WIDE * BLOCK_PX..].iter().all(|&s| s == 7));
    }

    /// Blur of the size an ultrafast encode leaves still decodes: the reader
    /// samples the block centre, and the tolerance covers a shifted level.
    #[test]
    fn a_degraded_strip_still_decodes() {
        let mut luma = vec![0u8; STRIDE * BLOCK_PX];
        burn(&mut luma, STRIDE, 0xDEAD_BEEF_0BAD_F00D);
        for sample in luma.iter_mut() {
            *sample = sample.saturating_add(30).min(LUMA_ONE);
        }
        assert_eq!(decode(&luma, STRIDE), Some(0xDEAD_BEEF_0BAD_F00D));
    }

    #[test]
    fn a_frame_with_no_strip_is_rejected() {
        // mid-grey everywhere: no block sits near either level.
        let luma = vec![128u8; STRIDE * BLOCK_PX];
        assert_eq!(decode(&luma, STRIDE), None);
        // and a frame too narrow to hold the strip.
        assert_eq!(decode(&luma, STRIP_WIDTH_PX - 1), None);
    }
}
