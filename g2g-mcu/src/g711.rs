//! G.711 (mu-law / A-law) fixed-point audio codec: the first MCU-fit codec
//! (M638). Pure integer companding, one byte per sample, no tables beyond
//! what the expressions fold to, so it runs on the heap-free subset like the
//! peripheral elements around it.
//!
//! The sample conversions are free `const fn`s ([`mulaw_encode`],
//! [`mulaw_decode`], [`alaw_encode`], [`alaw_decode`]), implemented once here
//! for any consumer (a host wrapper reuses these, it does not reimplement
//! them). Decode reconstructs the ITU-defined linear levels (identical to the
//! classic Sun `g711.c` and ffmpeg). Encode quantizes to the *nearest*
//! reconstruction level in the 14-bit domain, which is bit-exact with
//! ffmpeg's `pcm_mulaw` / `pcm_alaw` encoders (validated by a full 65536-
//! sample sweep in `m638_g711.rs`) and strictly lower error than the
//! truncating segment search some implementations use. Mu-law keeps its
//! inherent two zero codes (`0x7F` = -0, `0xFF` = +0); both decode to 0,
//! which re-encodes to `0xFF`, exactly as the reference peers behave.
//!
//! [`G711Enc`] / [`G711Dec`] wrap the conversions as heap-free
//! [`StaticTransform`]s: the first payload-*producing* static transforms
//! (companding changes the byte count, so unlike a pass-through stage they
//! need somewhere to put the output). They reuse the [`StaticLendRing`] lend
//! model the capture source established: convert into an acquired slot,
//! publish it zero-copy, and inherit the input frame's timing (companding is
//! per-sample, no algorithmic delay).

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{AudioFormat, StaticTransform};

/// Which G.711 companding law a codec element applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Law {
    /// Mu-law (North America / Japan telephony, RTP payload type 0 / PCMU).
    Mulaw,
    /// A-law (Europe / international telephony, RTP payload type 8 / PCMA).
    Alaw,
}

impl Law {
    /// The compressed side's caps format for this law.
    pub const fn format(self) -> AudioFormat {
        match self {
            Law::Mulaw => AudioFormat::Mulaw,
            Law::Alaw => AudioFormat::Alaw,
        }
    }

    /// Compand one 16-bit sample to this law's 8-bit code.
    pub const fn encode(self, sample: i16) -> u8 {
        match self {
            Law::Mulaw => mulaw_encode(sample),
            Law::Alaw => alaw_encode(sample),
        }
    }

    /// Expand one 8-bit code to its 16-bit reconstruction level.
    pub const fn decode(self, code: u8) -> i16 {
        match self {
            Law::Mulaw => mulaw_decode(code),
            Law::Alaw => alaw_decode(code),
        }
    }
}

/// Ascending reconstruction magnitude of mu-law magnitude code `i`
/// (0..=127): `((mantissa * 8 + 0x84) << exponent) - 0x84`, max 32124.
const fn mu_mag(i: i32) -> i32 {
    ((((i & 15) << 3) + 0x84) << (i >> 4)) - 0x84
}

/// Ascending reconstruction magnitude of A-law magnitude code `i`
/// (0..=127): segment 0 is linear, higher segments double, max 32256.
const fn a_mag(i: i32) -> i32 {
    if i < 16 {
        (i << 4) + 8
    } else {
        (((i & 15) << 4) + 0x108) << ((i >> 4) - 1)
    }
}

/// Nearest-reconstruction-level quantization in the 14-bit (`>> 2`) domain,
/// shared by both laws; `wire_pos_xor` maps the magnitude code to the law's
/// positive wire byte (mu: `^0xFF` complement, A: `^0xD5` = sign bit + even-
/// bit inversion). Bounded loop, shifts bounded by the segment layout: no
/// panic path.
const fn encode_nearest(sample: i16, law: Law, wire_pos_xor: u8) -> u8 {
    // Floor to the 14-bit companding domain; +-0 both land on the positive
    // zero code, matching the reference peers.
    let j = (((sample as i32) + 32768) >> 2) - 8192;
    let (m, neg) = if j >= 0 { (j, false) } else { (-j, true) };
    let mut i = 0i32;
    while i < 127 {
        let (lo, hi) = match law {
            Law::Mulaw => (mu_mag(i), mu_mag(i + 1)),
            Law::Alaw => (a_mag(i), a_mag(i + 1)),
        };
        // Midpoint of adjacent reconstruction levels, rounded, in the same
        // >>2 domain as `m`: below it, code `i` is the nearest level.
        if m < (lo + hi + 4) >> 3 {
            break;
        }
        i += 1;
    }
    let code = (i as u8) ^ wire_pos_xor;
    if neg {
        code ^ 0x80
    } else {
        code
    }
}

/// Compand one 16-bit sample to a mu-law byte (nearest reconstruction level;
/// bit-exact with ffmpeg's `pcm_mulaw` encoder).
pub const fn mulaw_encode(sample: i16) -> u8 {
    encode_nearest(sample, Law::Mulaw, 0xFF)
}

/// Expand one mu-law byte to its 16-bit reconstruction level (the ITU table
/// values, +-32124 max).
pub const fn mulaw_decode(code: u8) -> i16 {
    let u = !code;
    let exponent = ((u >> 4) & 7) as i32;
    let mantissa = (u & 0x0F) as i32;
    let x = (((mantissa << 3) + 0x84) << exponent) - 0x84;
    if u & 0x80 != 0 {
        -x as i16
    } else {
        x as i16
    }
}

/// Compand one 16-bit sample to an A-law byte (nearest reconstruction level;
/// bit-exact with ffmpeg's `pcm_alaw` encoder).
pub const fn alaw_encode(sample: i16) -> u8 {
    encode_nearest(sample, Law::Alaw, 0xD5)
}

/// Expand one A-law byte to its 16-bit reconstruction level (the ITU table
/// values, +-32256 max). On the A-law wire the sign bit set means positive.
pub const fn alaw_decode(code: u8) -> i16 {
    let u = code ^ 0x55;
    let exponent = ((u >> 4) & 7) as i32;
    let mantissa = (u & 0x0F) as i32;
    let x = if exponent == 0 {
        (mantissa << 4) + 8
    } else {
        ((mantissa << 4) + 0x108) << (exponent - 1)
    };
    if u & 0x80 != 0 {
        x as i16
    } else {
        -x as i16
    }
}

use crate::lend::lend_converted;

/// A heap-free G.711 encoder [`StaticTransform`]: interleaved S16LE PCM in,
/// one companded byte per sample out (2:1), lent from a [`StaticLendRing`].
/// A payload that is not whole 16-bit samples is rejected as
/// [`G2gError::CapsMismatch`]. Channel count is irrelevant to the math
/// (companding is per-sample), so any interleaving passes through unchanged.
pub struct G711Enc<'r, const N: usize, const BYTES: usize> {
    law: Law,
    ring: &'r StaticLendRing<N, BYTES>,
}

impl<const N: usize, const BYTES: usize> G711Enc<'static, N, BYTES> {
    /// An encoder over a `'static` ring (the MCU idiom), which makes the
    /// zero-copy lend safe by construction.
    pub fn new(law: Law, ring: &'static StaticLendRing<N, BYTES>) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(law, ring) }
    }
}

impl<'r, const N: usize, const BYTES: usize> G711Enc<'r, N, BYTES> {
    /// An encoder over a borrowed (e.g. stack-local) ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this element publishes: the
    /// pipeline must drain before the ring is dropped.
    pub unsafe fn with_ring(law: Law, ring: &'r StaticLendRing<N, BYTES>) -> Self {
        Self { law, ring }
    }
}

impl<const N: usize, const BYTES: usize> StaticTransform for G711Enc<'_, N, BYTES> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let MemoryDomain::System(slice) = &input.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let len = slice.as_slice().len();
        // Whole 16-bit samples only.
        if len % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }
        let law = self.law;
        // SAFETY: the constructor established the ring-outlives-frames
        // contract (`new`: 'static; `with_ring`: caller's contract).
        let out = unsafe {
            lend_converted(self.ring, &input, len / 2, |src, dst| {
                for (pair, byte) in src.chunks_exact(2).zip(dst.iter_mut()) {
                    // Slice patterns, not indexing: no bounds-check panic
                    // path may enter the no-alloc subset.
                    let &[lo, hi] = pair else { continue };
                    *byte = law.encode(i16::from_le_bytes([lo, hi]));
                }
            })?
        };
        Ok(Some(out))
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for G711Enc<'_, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("G711Enc")
            .field("law", &self.law)
            .finish_non_exhaustive()
    }
}

// A pure fixed-point codec has no peripheral state, so the default no-op
// recover applies; the impl lets it sit in a supervised sink chain.
impl<const N: usize, const BYTES: usize> g2g_core::supervise::Recover for G711Enc<'_, N, BYTES> {}
impl<const N: usize, const BYTES: usize> g2g_core::supervise::Recover for G711Dec<'_, N, BYTES> {}

/// A heap-free G.711 decoder [`StaticTransform`]: one companded byte per
/// sample in, interleaved S16LE PCM out (1:2), lent from a
/// [`StaticLendRing`]. The inverse of [`G711Enc`].
pub struct G711Dec<'r, const N: usize, const BYTES: usize> {
    law: Law,
    ring: &'r StaticLendRing<N, BYTES>,
}

impl<const N: usize, const BYTES: usize> G711Dec<'static, N, BYTES> {
    /// A decoder over a `'static` ring (the MCU idiom), which makes the
    /// zero-copy lend safe by construction.
    pub fn new(law: Law, ring: &'static StaticLendRing<N, BYTES>) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(law, ring) }
    }
}

impl<'r, const N: usize, const BYTES: usize> G711Dec<'r, N, BYTES> {
    /// A decoder over a borrowed (e.g. stack-local) ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this element publishes: the
    /// pipeline must drain before the ring is dropped.
    pub unsafe fn with_ring(law: Law, ring: &'r StaticLendRing<N, BYTES>) -> Self {
        Self { law, ring }
    }
}

impl<const N: usize, const BYTES: usize> StaticTransform for G711Dec<'_, N, BYTES> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let MemoryDomain::System(slice) = &input.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let len = slice.as_slice().len();
        let Some(out_len) = len.checked_mul(2) else {
            return Err(G2gError::CapsMismatch);
        };
        let law = self.law;
        // SAFETY: the constructor established the ring-outlives-frames
        // contract (`new`: 'static; `with_ring`: caller's contract).
        let out = unsafe {
            lend_converted(self.ring, &input, out_len, |src, dst| {
                for (&code, pair) in src.iter().zip(dst.chunks_exact_mut(2)) {
                    let [lo, hi] = pair else { continue };
                    [*lo, *hi] = law.decode(code).to_le_bytes();
                }
            })?
        };
        Ok(Some(out))
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for G711Dec<'_, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("G711Dec")
            .field("law", &self.law)
            .finish_non_exhaustive()
    }
}
