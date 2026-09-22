//! IMA ADPCM fixed-point audio codec (M639): 4 bits per sample (4:1 vs
//! S16LE), the WAV / DVI block layout, mono. Pure integer step-adaptive
//! delta modulation: an 89-entry step table and shift/multiply arithmetic,
//! MCU-fit like [`g711`](crate::g711).
//!
//! The block layout is the `WAVE_FORMAT_IMA_ADPCM` (a.k.a. DVI4 / IMA-WAV)
//! one: each block opens with a 4-byte header (initial predictor as S16LE,
//! step index, reserved 0) followed by data bytes carrying two samples each,
//! low nibble first, so a block of `B` bytes holds `(B - 4) * 2 + 1` samples
//! (the header predictor is sample 0, emitted verbatim). The encoder carries
//! the step index across blocks and restarts the predictor from each block's
//! first sample, exactly as ffmpeg's `adpcm_ima_wav` does.
//!
//! Validated bit-exact against ffmpeg in *all three* directions
//! (`m639_adpcm.rs`): our encode of a full-range signal matches ffmpeg's
//! encoder byte-for-byte, our decode of ffmpeg's stream matches ffmpeg's own
//! decode, and ffmpeg decodes our stream to exactly what we decode. The two
//! directions deliberately use *different* reconstruction arithmetic,
//! mirroring ffmpeg's own pair: the encoder tracks its predictor with the
//! multiplicative form (`diff = (2 * delta + 1) * step >> 3`, ffmpeg's
//! `adpcm_ima_compress_sample`), while the decoder reconstructs with the IMA
//! spec's bit-serial form (`step>>3` plus per-bit terms, ffmpeg >= 8.1's
//! `ff_adpcm_ima_qt_expand_nibble`; older ffmpeg decoded 4-bit IMA-WAV with
//! the multiplicative form). The two differ by a few LSBs of truncation, so
//! matching each direction's reference exactly is what keeps the byte-level
//! interop asserts green.
//!
//! [`AdpcmEnc`] / [`AdpcmDec`] wrap the block conversions as heap-free
//! [`StaticTransform`]s over the shared [`StaticLendRing`] lend model. They
//! stream nibbles directly between the frame payloads, so no block-sized
//! sample buffer ever lands on the MCU stack.

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;

use crate::lend::lend_converted;

/// The IMA ADPCM step table (89 quantizer step sizes).
const STEP_TABLE: [u16; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
];

/// Step-index adaptation per 3-bit magnitude (small deltas cool down, large
/// ones heat up).
const INDEX_TABLE: [i8; 8] = [-1, -1, -1, -1, 2, 4, 6, 8];

/// One channel's ADPCM predictor state. The same state type drives both
/// directions; each direction tracks its own reference implementation's
/// arithmetic (see the module note on the encode/decode split).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImaState {
    /// Last reconstructed sample.
    pub predictor: i16,
    /// Index into the step table, always kept in 0..=88.
    pub step_index: u8,
}

impl ImaState {
    /// The current quantizer step. The index is <= 88 by construction; the
    /// clamp lets the optimizer discharge the table bounds check (no panic
    /// path in the no-alloc subset).
    fn step(self) -> i32 {
        STEP_TABLE[(self.step_index as usize).min(88)] as i32
    }

    fn adapt(&mut self, nibble: u8) {
        let idx = self.step_index as i32 + INDEX_TABLE[(nibble & 7) as usize] as i32;
        self.step_index = idx.clamp(0, 88) as u8;
    }

    /// Quantize one sample to a 4-bit code, updating the state to the
    /// decoder's reconstruction (ffmpeg's `adpcm_ima_compress_sample`).
    pub fn encode_sample(&mut self, sample: i16) -> u8 {
        let step = self.step();
        let delta = sample as i32 - self.predictor as i32;
        let nibble = ((delta.unsigned_abs() as i32 * 4 / step.max(1)).min(7)
            + 8 * ((delta < 0) as i32)) as u8;
        let mag = step * (2 * (nibble & 7) as i32 + 1) / 8;
        let p = self.predictor as i32 + if nibble & 8 != 0 { -mag } else { mag };
        self.predictor = p.clamp(-32768, 32767) as i16;
        self.adapt(nibble);
        nibble
    }

    /// Reconstruct one sample from a 4-bit code, using the IMA spec's
    /// bit-serial expansion (ffmpeg >= 8.1's `ff_adpcm_ima_qt_expand_nibble`;
    /// summing the truncated per-bit terms differs from the encoder's single
    /// truncated multiply by a few LSBs).
    pub fn decode_sample(&mut self, nibble: u8) -> i16 {
        let step = self.step();
        let mut diff = step >> 3;
        if nibble & 4 != 0 {
            diff += step;
        }
        if nibble & 2 != 0 {
            diff += step >> 1;
        }
        if nibble & 1 != 0 {
            diff += step >> 2;
        }
        let p = self.predictor as i32 + if nibble & 8 != 0 { -diff } else { diff };
        self.predictor = p.clamp(-32768, 32767) as i16;
        self.adapt(nibble);
        self.predictor
    }
}

/// Block header bytes (predictor S16LE + step index + reserved).
pub const BLOCK_HEADER: usize = 4;

/// Samples one `block_bytes`-sized mono block carries: the header predictor
/// plus two per data byte. Saturating so a bogus size cannot wrap.
pub const fn samples_per_block(block_bytes: usize) -> usize {
    block_bytes
        .saturating_sub(BLOCK_HEADER)
        .saturating_mul(2)
        .saturating_add(1)
}

/// Bytes one channel's interleave group carries, and the samples in it: a
/// multi-channel block alternates 4-byte groups, 8 samples each, per channel.
const GROUP_BYTES: usize = 4;
const GROUP_SAMPLES: usize = 8;

/// Samples one channel carries in a `block_bytes` block holding `channels`
/// interleaved channels. Mono packs every data byte; a multi-channel block
/// alternates whole 4-byte groups, so it carries whole groups only.
pub const fn samples_per_block_channels(block_bytes: usize, channels: usize) -> usize {
    if channels <= 1 {
        return samples_per_block(block_bytes);
    }
    let data = block_bytes.saturating_sub(BLOCK_HEADER.saturating_mul(channels));
    (data / (GROUP_BYTES.saturating_mul(channels)))
        .saturating_mul(GROUP_SAMPLES)
        .saturating_add(1)
}

/// Which channel a data byte belongs to, and its index within that channel's
/// own bytes. Mono is the identity.
const fn byte_owner(index: usize, channels: usize) -> (usize, usize) {
    let group = index / GROUP_BYTES;
    (
        group % channels,
        (group / channels) * GROUP_BYTES + index % GROUP_BYTES,
    )
}

/// Data bytes a block of `channels` channels actually fills: mono packs the
/// block out, a multi-channel one stops at the last whole interleave group.
const fn data_bytes_used(data_len: usize, channels: usize) -> usize {
    if channels <= 1 {
        return data_len;
    }
    (data_len / (GROUP_BYTES * channels)) * (GROUP_BYTES * channels)
}

/// Encode one block of `states.len()` interleaved channels: `src` is exactly
/// `samples_per_block_channels * channels * 2` bytes of interleaved S16LE,
/// `dst` exactly `block_bytes`. Each channel's step index carries across
/// blocks (the caller keeps `states`); its predictor restarts from that
/// channel's first sample, as the reference encoder does. Wrong slice sizes
/// yield `None`.
pub fn encode_block_channels(states: &mut [ImaState], src: &[u8], dst: &mut [u8]) -> Option<()> {
    let channels = states.len();
    if channels == 0 || dst.len() < BLOCK_HEADER * channels {
        return None;
    }
    let per_channel = samples_per_block_channels(dst.len(), channels);
    if src.len() != per_channel * channels * SAMPLE_BYTES {
        return None;
    }
    let (headers, data) = dst.split_at_mut(BLOCK_HEADER * channels);
    for (channel, state) in states.iter_mut().enumerate() {
        state.predictor = sample_at(src, channels, channel, 0)?;
        state.step_index = state.step_index.min(88);
        let header = headers
            .get_mut(channel * BLOCK_HEADER..)?
            .get_mut(..BLOCK_HEADER)?;
        let [h0, h1, h2, h3] = header else {
            return None;
        };
        [*h0, *h1] = state.predictor.to_le_bytes();
        *h2 = state.step_index;
        *h3 = 0;
    }
    let used = data_bytes_used(data.len(), channels);
    for (index, byte) in data.iter_mut().enumerate().take(used) {
        let (channel, within) = byte_owner(index, channels);
        let state = states.get_mut(channel)?;
        let first = 1 + within * 2;
        let low = state.encode_sample(sample_at(src, channels, channel, first)?);
        let high = state.encode_sample(sample_at(src, channels, channel, first + 1)?);
        *byte = low | (high << 4);
    }
    Some(())
}

/// Decode one block of `channels` interleaved channels: `src` is exactly
/// `block_bytes`, `dst` exactly `samples_per_block_channels * channels * 2`
/// bytes of interleaved S16LE. Wrong slice sizes yield `None`.
pub fn decode_block_channels(channels: usize, src: &[u8], dst: &mut [u8]) -> Option<()> {
    if channels == 0 || channels > MAX_BLOCK_CHANNELS || src.len() < BLOCK_HEADER * channels {
        return None;
    }
    let per_channel = samples_per_block_channels(src.len(), channels);
    if dst.len() != per_channel * channels * SAMPLE_BYTES {
        return None;
    }
    let (headers, data) = src.split_at(BLOCK_HEADER * channels);
    let mut states = [ImaState::default(); MAX_BLOCK_CHANNELS];
    for (channel, state) in states.iter_mut().enumerate().take(channels) {
        let header = headers.get(channel * BLOCK_HEADER..)?.get(..BLOCK_HEADER)?;
        let &[p0, p1, index, _reserved] = header else {
            return None;
        };
        state.predictor = i16::from_le_bytes([p0, p1]);
        state.step_index = index.min(88);
        put_sample(dst, channels, channel, 0, state.predictor)?;
    }
    let used = data_bytes_used(data.len(), channels);
    for (index, &byte) in data.iter().enumerate().take(used) {
        let (channel, within) = byte_owner(index, channels);
        let state = states.get_mut(channel)?;
        let first = 1 + within * 2;
        let low = state.decode_sample(byte & 0x0F);
        put_sample(dst, channels, channel, first, low)?;
        let high = state.decode_sample(byte >> 4);
        put_sample(dst, channels, channel, first + 1, high)?;
    }
    Some(())
}

/// Bytes per S16LE sample.
const SAMPLE_BYTES: usize = 2;

/// Channels one block can interleave: the WAV layout is mono or stereo.
pub const MAX_BLOCK_CHANNELS: usize = 2;

/// Read interleaved sample `index` of `channel` out of S16LE `src`.
fn sample_at(src: &[u8], channels: usize, channel: usize, index: usize) -> Option<i16> {
    let at = (index * channels + channel) * SAMPLE_BYTES;
    let &[lo, hi] = src.get(at..)?.get(..SAMPLE_BYTES)? else {
        return None;
    };
    Some(i16::from_le_bytes([lo, hi]))
}

/// Write interleaved sample `index` of `channel` into S16LE `dst`.
fn put_sample(
    dst: &mut [u8],
    channels: usize,
    channel: usize,
    index: usize,
    sample: i16,
) -> Option<()> {
    let at = (index * channels + channel) * SAMPLE_BYTES;
    let slot = dst.get_mut(at..)?.get_mut(..SAMPLE_BYTES)?;
    let [lo, hi] = slot else { return None };
    [*lo, *hi] = sample.to_le_bytes();
    Some(())
}

/// Encode one mono block: `src` is exactly `samples_per_block * 2` bytes of
/// S16LE, `dst` exactly `block_bytes`; `step_index` carries across blocks
/// (pass the previous block's return). Wrong slice sizes yield `None` (the
/// element maps it to [`G2gError::CapsMismatch`]); the conversion itself
/// cannot fail.
pub fn encode_block(step_index: u8, src: &[u8], dst: &mut [u8]) -> Option<u8> {
    let mut states = [ImaState {
        predictor: 0,
        step_index,
    }];
    encode_block_channels(&mut states, src, dst)?;
    Some(states[0].step_index)
}

/// Decode one mono block: `src` is exactly `block_bytes`, `dst` exactly
/// `samples_per_block * 2` bytes of S16LE. Wrong slice sizes yield `None`.
pub fn decode_block(src: &[u8], dst: &mut [u8]) -> Option<()> {
    decode_block_channels(1, src, dst)
}

/// A heap-free IMA ADPCM encoder [`StaticTransform`]: mono S16LE in, IMA-WAV
/// blocks out (about 4:1), lent from a [`StaticLendRing`]. Each input frame
/// must be a whole number of blocks' worth of samples
/// ([`samples_per_block`]); anything else is [`G2gError::CapsMismatch`]. The
/// step index carries across blocks and frames, like the reference encoder.
pub struct AdpcmEnc<'r, const N: usize, const BYTES: usize> {
    ring: &'r StaticLendRing<N, BYTES>,
    block_bytes: usize,
    step_index: u8,
}

impl<const N: usize, const BYTES: usize> AdpcmEnc<'static, N, BYTES> {
    /// An encoder over a `'static` ring (the MCU idiom), emitting
    /// `block_bytes`-sized blocks (>= 5; WAV uses 1024 by default).
    pub fn new(ring: &'static StaticLendRing<N, BYTES>, block_bytes: usize) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(ring, block_bytes) }
    }
}

impl<'r, const N: usize, const BYTES: usize> AdpcmEnc<'r, N, BYTES> {
    /// An encoder over a borrowed (e.g. stack-local) ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this element publishes: the
    /// pipeline must drain before the ring is dropped.
    pub unsafe fn with_ring(ring: &'r StaticLendRing<N, BYTES>, block_bytes: usize) -> Self {
        Self {
            ring,
            block_bytes: block_bytes.max(BLOCK_HEADER + 1),
            step_index: 0,
        }
    }
}

impl<const N: usize, const BYTES: usize> StaticTransform for AdpcmEnc<'_, N, BYTES> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let Some(slice) = input.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        let src_block = samples_per_block(self.block_bytes) * 2;
        let len = slice.len();
        // Whole blocks only.
        if src_block == 0 || len % src_block != 0 {
            return Err(G2gError::CapsMismatch);
        }
        let blocks = len / src_block;
        let Some(out_len) = blocks.checked_mul(self.block_bytes) else {
            return Err(G2gError::CapsMismatch);
        };
        let block_bytes = self.block_bytes;
        let mut index = self.step_index;
        let out = unsafe {
            // SAFETY: the constructor established the ring-outlives-frames
            // contract (`new`: 'static; `with_ring`: caller's contract).
            lend_converted(self.ring, &input, out_len, |src, dst| {
                for (s, d) in src
                    .chunks_exact(src_block)
                    .zip(dst.chunks_exact_mut(block_bytes))
                {
                    // Sizes are exact by construction; `encode_block` cannot
                    // return None here.
                    if let Some(next) = encode_block(index, s, d) {
                        index = next;
                    }
                }
            })?
        };
        self.step_index = index;
        Ok(Some(out))
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for AdpcmEnc<'_, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AdpcmEnc")
            .field("block_bytes", &self.block_bytes)
            .field("step_index", &self.step_index)
            .finish_non_exhaustive()
    }
}

/// A heap-free IMA ADPCM decoder [`StaticTransform`]: whole IMA-WAV blocks
/// in, mono S16LE out, lent from a [`StaticLendRing`]. The inverse of
/// [`AdpcmEnc`]; each block is self-contained (its header carries the
/// predictor state), so any conformant stream decodes regardless of where
/// its encoder's step index wandered.
pub struct AdpcmDec<'r, const N: usize, const BYTES: usize> {
    ring: &'r StaticLendRing<N, BYTES>,
    block_bytes: usize,
}

impl<const N: usize, const BYTES: usize> AdpcmDec<'static, N, BYTES> {
    /// A decoder over a `'static` ring (the MCU idiom), consuming
    /// `block_bytes`-sized blocks.
    pub fn new(ring: &'static StaticLendRing<N, BYTES>, block_bytes: usize) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(ring, block_bytes) }
    }
}

impl<'r, const N: usize, const BYTES: usize> AdpcmDec<'r, N, BYTES> {
    /// A decoder over a borrowed (e.g. stack-local) ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this element publishes: the
    /// pipeline must drain before the ring is dropped.
    pub unsafe fn with_ring(ring: &'r StaticLendRing<N, BYTES>, block_bytes: usize) -> Self {
        Self {
            ring,
            block_bytes: block_bytes.max(BLOCK_HEADER + 1),
        }
    }
}

impl<const N: usize, const BYTES: usize> StaticTransform for AdpcmDec<'_, N, BYTES> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let Some(slice) = input.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        let len = slice.len();
        // Whole blocks only.
        if len % self.block_bytes != 0 {
            return Err(G2gError::CapsMismatch);
        }
        let dst_block = samples_per_block(self.block_bytes) * 2;
        let Some(out_len) = (len / self.block_bytes).checked_mul(dst_block) else {
            return Err(G2gError::CapsMismatch);
        };
        let block_bytes = self.block_bytes;
        let out = unsafe {
            // SAFETY: the constructor established the ring-outlives-frames
            // contract (`new`: 'static; `with_ring`: caller's contract).
            lend_converted(self.ring, &input, out_len, |src, dst| {
                for (s, d) in src
                    .chunks_exact(block_bytes)
                    .zip(dst.chunks_exact_mut(dst_block))
                {
                    let _ = decode_block(s, d);
                }
            })?
        };
        Ok(Some(out))
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for AdpcmDec<'_, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("AdpcmDec")
            .field("block_bytes", &self.block_bytes)
            .finish_non_exhaustive()
    }
}
