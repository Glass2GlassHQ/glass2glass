//! Saturating Q15-gain audio mixer (M642): the first fan-in element on the
//! heap-free subset, and the `mix` stage of the reference audio chain
//! (capture -> convert -> resample -> mix -> encode -> RTP). Two interleaved
//! S16LE streams in, one out: `out = sat(a * gain_a + b * gain_b)` per
//! sample, gains in Q15.
//!
//! The per-sample math is a free `const fn` ([`mix_q15`]), implemented once
//! here for any consumer, like the G.711 / ADPCM conversions. The
//! accumulator is widened to i64 before the product sum: with both samples
//! and both gains at `i16::MIN` the two products alone are exactly `2^31`,
//! one past `i32::MAX`, so an i32 accumulator would overflow on real input
//! (full-scale negative against inverted-phase gain), not just in theory.
//! The Q15 result is rounded (round-half-up) then saturated to i16.
//!
//! [`Mixer`] wraps the math as a heap-free [`StaticFanIn2`], the two-input
//! stage driven by `run_sources_fanin_sink` (M642's const-arity fan-in
//! runner). Both inputs must carry the same byte count (the lockstep runner
//! plus a deterministic upstream makes equal chunking the invariant, and a
//! silent shorter-input policy would hide a real rate bug); input `a` is the
//! timing master, so the output inherits `a`'s timing and sequence. Q15
//! cannot express exactly 1.0: `32767` is ~0.99997 (attenuation-only mixing
//! is the normal use; equal two-way mixing is `16384` = 0.5 each, exact).

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticFanIn2;

use crate::lend::lend_slot;

/// Mix one sample pair with Q15 gains: `sat_i16(round((a * gain_a + b *
/// gain_b) / 2^15))`. Pure integer, no panic path (the i64 accumulator
/// cannot overflow: each product is at most 2^30 in magnitude).
pub const fn mix_q15(a: i16, b: i16, gain_a: i16, gain_b: i16) -> i16 {
    let acc = (a as i64) * (gain_a as i64) + (b as i64) * (gain_b as i64) + (1 << 14);
    let v = acc >> 15;
    if v > i16::MAX as i64 {
        i16::MAX
    } else if v < i16::MIN as i64 {
        i16::MIN
    } else {
        v as i16
    }
}

/// A heap-free two-input audio mixer [`StaticFanIn2`]: interleaved S16LE PCM
/// on both inputs (same sample rate, channel layout, and per-frame byte
/// count; a rate mismatch belongs to an upstream [`Resampler`]
/// (crate::Resampler)), interleaved S16LE out, lent from a
/// [`StaticLendRing`]. Payloads that are not whole 16-bit samples or whose
/// lengths differ are rejected as [`G2gError::CapsMismatch`]. Negative gains
/// invert phase, so `(32767, -32768)` is a difference tap.
pub struct Mixer<'r, const N: usize, const BYTES: usize> {
    gain_a: i16,
    gain_b: i16,
    ring: &'r StaticLendRing<N, BYTES>,
}

impl<const N: usize, const BYTES: usize> Mixer<'static, N, BYTES> {
    /// A mixer over a `'static` ring (the MCU idiom), which makes the
    /// zero-copy lend safe by construction. Gains are Q15 (`16384` = 0.5).
    pub fn new(gain_a: i16, gain_b: i16, ring: &'static StaticLendRing<N, BYTES>) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(gain_a, gain_b, ring) }
    }
}

impl<'r, const N: usize, const BYTES: usize> Mixer<'r, N, BYTES> {
    /// A mixer over a borrowed (e.g. stack-local) ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this element publishes: the
    /// pipeline must drain before the ring is dropped.
    pub unsafe fn with_ring(gain_a: i16, gain_b: i16, ring: &'r StaticLendRing<N, BYTES>) -> Self {
        Self { gain_a, gain_b, ring }
    }
}

impl<const N: usize, const BYTES: usize> StaticFanIn2 for Mixer<'_, N, BYTES> {
    async fn process2(&mut self, a: Frame, b: Frame) -> Result<Option<Frame>, G2gError> {
        let (MemoryDomain::System(pa), MemoryDomain::System(pb)) = (&a.domain, &b.domain) else {
            return Err(G2gError::UnsupportedDomain);
        };
        let (pa, pb) = (pa.as_slice(), pb.as_slice());
        // Whole 16-bit samples, equal counts on both inputs.
        if pa.len() % 2 != 0 || pa.len() != pb.len() {
            return Err(G2gError::CapsMismatch);
        }
        let (gain_a, gain_b) = (self.gain_a, self.gain_b);
        // SAFETY: the constructor established the ring-outlives-frames
        // contract (`new`: 'static; `with_ring`: caller's contract).
        let out = unsafe {
            lend_slot(self.ring, a.timing, a.sequence, pa.len(), |dst| {
                let pairs = pa.chunks_exact(2).zip(pb.chunks_exact(2));
                for ((ca, cb), d) in pairs.zip(dst.chunks_exact_mut(2)) {
                    // Slice patterns, not indexing: no bounds-check panic
                    // path may enter the no-alloc subset.
                    let (&[al, ah], &[bl, bh]) = (ca, cb) else { continue };
                    let [dl, dh] = d else { continue };
                    let mixed = mix_q15(
                        i16::from_le_bytes([al, ah]),
                        i16::from_le_bytes([bl, bh]),
                        gain_a,
                        gain_b,
                    );
                    [*dl, *dh] = mixed.to_le_bytes();
                }
            })?
        };
        Ok(Some(out))
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for Mixer<'_, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Mixer")
            .field("gain_a", &self.gain_a)
            .field("gain_b", &self.gain_b)
            .finish_non_exhaustive()
    }
}
