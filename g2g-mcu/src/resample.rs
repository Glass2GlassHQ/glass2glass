//! Fixed-point polyphase audio resampler (M641): the first stage of the
//! deterministic reference audio chain (capture -> convert -> **resample**
//! -> mix -> encode -> RTP). Mono S16LE over the telephony rate set
//! {8, 16, 48} kHz, pure integer Q14 MACs over generated coefficient tables
//! ([`crate::resample_tables`], Blackman-windowed sinc, every phase branch
//! summing to exactly unity so DC gain is exact and ripple-free), streaming
//! state carried across frames (a fixed history window, no block artifacts
//! at frame boundaries).
//!
//! Unlike the codecs there is no bit-exact external reference (every
//! resampler chooses its own filter), so the oracle is analytic:
//! `m641_resample.rs` asserts the table invariants, exact DC gain, in-band
//! tone SNR, alias rejection when decimating, and that chunked streaming is
//! byte-identical to one-shot processing.
//!
//! [`Resampler`] rides the ring-lend transform model like the codec
//! elements; a same-rate configuration passes frames through untouched.

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;

use crate::lend::lend_converted;
use crate::resample_tables::{self, COEFF_SHIFT};

/// Largest per-phase tap count across the tables (the divide-by-6
/// decimator's anti-alias filter); sizes the history window.
pub const MAX_TAPS: usize = 192;

/// One (L, M) ratio's polyphase coefficients, phase-major
/// (`coeffs[phase * taps + k]` = prototype `h[phase + k * L]`), Q14, each
/// phase branch summing to exactly `1 << COEFF_SHIFT`.
#[derive(Debug)]
pub struct RateTable {
    /// Interpolation factor (output phases).
    pub l: u32,
    /// Decimation factor.
    pub m: u32,
    /// Taps per phase.
    pub taps: usize,
    /// `l * taps` coefficients, phase-major.
    pub coeffs: &'static [i16],
}

/// The supported sample rates (the telephony set; the table pairs cover
/// every ordered pair).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleRate {
    Hz8000,
    Hz16000,
    Hz48000,
}

/// The coefficient table converting `from` -> `to`; `None` when `from ==
/// to` (identity: no filtering wanted).
pub const fn table_for(from: SampleRate, to: SampleRate) -> Option<&'static RateTable> {
    use SampleRate::*;
    match (from, to) {
        (Hz8000, Hz16000) => Some(&resample_tables::UP_2),
        (Hz16000, Hz48000) => Some(&resample_tables::UP_3),
        (Hz8000, Hz48000) => Some(&resample_tables::UP_6),
        (Hz16000, Hz8000) => Some(&resample_tables::DOWN_2),
        (Hz48000, Hz16000) => Some(&resample_tables::DOWN_3),
        (Hz48000, Hz8000) => Some(&resample_tables::DOWN_6),
        (Hz8000, Hz8000) | (Hz16000, Hz16000) | (Hz48000, Hz48000) => None,
    }
}

/// S16LE sample `i` of a payload; out of range reads as silence (the
/// element validates framing up front, so this is the panic-free accessor,
/// not error handling).
fn sample_at(bytes: &[u8], i: usize) -> i16 {
    match (bytes.get(2 * i), bytes.get(2 * i + 1)) {
        (Some(&lo), Some(&hi)) => i16::from_le_bytes([lo, hi]),
        _ => 0,
    }
}

/// A heap-free mono S16LE resampler [`StaticTransform`] over a
/// [`StaticLendRing`]. Output frames inherit the input's timing and
/// sequence; per-frame output length follows the rational ratio exactly
/// (state carries the fractional position, so long streams neither drift
/// nor accumulate error).
pub struct Resampler<'r, const N: usize, const BYTES: usize> {
    /// `None` = same-rate pass-through.
    table: Option<&'static RateTable>,
    /// The last `taps` input samples (the filter's look-back window).
    hist: [i16; MAX_TAPS],
    /// Total input samples consumed.
    consumed: u64,
    /// Upsampled-domain index (rate `fs * l`) of the next output sample.
    next_u: u64,
    ring: &'r StaticLendRing<N, BYTES>,
}

impl<const N: usize, const BYTES: usize> Resampler<'static, N, BYTES> {
    /// A resampler over a `'static` ring (the MCU idiom), which makes the
    /// zero-copy lend safe by construction.
    pub fn new(from: SampleRate, to: SampleRate, ring: &'static StaticLendRing<N, BYTES>) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(from, to, ring) }
    }
}

impl<'r, const N: usize, const BYTES: usize> Resampler<'r, N, BYTES> {
    /// A resampler over a borrowed (e.g. stack-local) ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this element publishes: the
    /// pipeline must drain before the ring is dropped.
    pub unsafe fn with_ring(
        from: SampleRate,
        to: SampleRate,
        ring: &'r StaticLendRing<N, BYTES>,
    ) -> Self {
        Self { table: table_for(from, to), hist: [0; MAX_TAPS], consumed: 0, next_u: 0, ring }
    }

    /// Output samples the next `n_in`-sample frame will produce (exact,
    /// from the carried fractional position).
    fn out_count(&self, n_in: usize, t: &RateTable) -> usize {
        let limit = (self.consumed + n_in as u64) * t.l as u64;
        if self.next_u >= limit {
            return 0;
        }
        // `.max(1)`: the table constants are never zero, but the loaded value
        // is opaque to the optimizer, and a division-by-zero panic path may
        // not enter the no-alloc archive (the M644 flagship proof).
        (limit - self.next_u).div_ceil((t.m as u64).max(1)) as usize
    }

    /// One polyphase MAC: output at upsampled index `u`, reading history
    /// for pre-frame samples. Q14 round-to-nearest, saturated.
    fn mac(&self, src: &[u8], u: u64, t: &RateTable) -> i16 {
        // `.max(1)`: discharge the division-by-zero check (see `out_count`).
        let l = (t.l as u64).max(1);
        let base = u / l;
        let phase = (u % l) as usize;
        let branch = t.coeffs.get(phase * t.taps..(phase + 1) * t.taps).unwrap_or(&[]);
        let mut acc: i32 = 1 << (COEFF_SHIFT - 1);
        for (k, &c) in branch.iter().enumerate() {
            let Some(idx) = base.checked_sub(k as u64) else { break };
            let s = if idx >= self.consumed {
                sample_at(src, (idx - self.consumed) as usize)
            } else {
                // Window position: hist[i] = x[consumed - taps + i].
                let back = (self.consumed - idx) as usize;
                match t.taps.checked_sub(back).and_then(|i| self.hist.get(i)) {
                    Some(&h) => h,
                    None => 0,
                }
            };
            acc += c as i32 * s as i32;
        }
        (acc >> COEFF_SHIFT).clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }

    /// Slide the history window past this frame's `n` samples.
    fn push_history(&mut self, src: &[u8], n: usize, taps: usize) {
        let taps = taps.min(MAX_TAPS);
        if n >= taps {
            for (i, h) in self.hist.iter_mut().take(taps).enumerate() {
                *h = sample_at(src, n - taps + i);
            }
        } else {
            self.hist.copy_within(n..taps, 0);
            for (i, h) in self.hist.iter_mut().take(taps).enumerate().skip(taps - n) {
                *h = sample_at(src, i - (taps - n));
            }
        }
        self.consumed += n as u64;
    }
}

impl<const N: usize, const BYTES: usize> StaticTransform for Resampler<'_, N, BYTES> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let Some(t) = self.table else {
            // Same-rate: identity, zero-copy.
            return Ok(Some(input));
        };
        let MemoryDomain::System(slice) = &input.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let len = slice.as_slice().len();
        // Whole 16-bit samples only.
        if len % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }
        let n_in = len / 2;
        let count = self.out_count(n_in, t);
        let Some(out_len) = count.checked_mul(2) else {
            return Err(G2gError::CapsMismatch);
        };
        let ring = self.ring;
        let mut next_u = self.next_u;
        // A two-phase borrow dance is avoided by finishing all `&self` reads
        // (mac) before the `&mut self` state update below.
        let out = {
            let this = &*self;
            // SAFETY: the constructor established the ring-outlives-frames
            // contract (`new`: 'static; `with_ring`: caller's contract).
            unsafe {
                lend_converted(ring, &input, out_len, |src, dst| {
                    for pair in dst.chunks_exact_mut(2) {
                        let [lo, hi] = pair else { continue };
                        [*lo, *hi] = this.mac(src, next_u, t).to_le_bytes();
                        next_u += t.m as u64;
                    }
                })?
            }
        };
        self.next_u = next_u;
        if let MemoryDomain::System(s) = &input.domain {
            self.push_history(s.as_slice(), n_in, t.taps);
        }
        Ok(Some(out))
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for Resampler<'_, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Resampler")
            .field("table", &self.table.map(|t| (t.l, t.m)))
            .field("consumed", &self.consumed)
            .finish_non_exhaustive()
    }
}
