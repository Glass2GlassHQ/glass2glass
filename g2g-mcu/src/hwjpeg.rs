//! Hardware JPEG codec seam (M640): the [`JpegDecoder`] contract, shaped
//! like the STM32H7 JPEG codec peripheral (feed one complete JFIF bitstream,
//! the silicon parses the header itself and reports the image parameters,
//! then emits decoded YCbCr in its native MCU-block order), plus
//! [`HwJpegDec`], the heap-free [`StaticTransform`] that owns everything
//! around the peripheral: bitstream framing validation, output sizing from
//! the reported parameters (checked math over attacker-controlled
//! dimensions), the ring lend, and fault surfacing.
//!
//! A board adapter typically DMA-feeds the input FIFO, waits for the
//! header-processed interrupt to latch the info registers, and DMA-drains
//! the output FIFO into the lent slot; the mock in `m640_hwjpeg.rs` replays
//! that contract, so the element's real logic is host-tested. The output
//! payload is the peripheral's native MCU-block-ordered YCbCr (not raster):
//! raster conversion is a separate stage (ST's examples do it with a second
//! DMA pass), and pretending otherwise here would misdescribe the silicon.
//! On-device validation (a `Hardware` conformance row on a real H7) needs a
//! board and stays open, like `VtDecode`'s runtime tier.

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;

/// Chroma subsampling of the decoded image, as the peripheral reports it
/// (the STM32H7 `JPEG_CONFR1.NF`/`COLORSPACE` shape reduced to the baseline
/// cases hardware codecs accept).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JpegSubsampling {
    /// Single-component grayscale (8x8 MCU, one block).
    Gray,
    /// YCbCr 4:4:4 (8x8 MCU, three blocks).
    Ycbcr444,
    /// YCbCr 4:2:2 (16x8 MCU, 2 Y + Cb + Cr blocks).
    Ycbcr422,
    /// YCbCr 4:2:0 (16x16 MCU, 4 Y + Cb + Cr blocks).
    Ycbcr420,
}

impl JpegSubsampling {
    /// (MCU width, MCU height, bytes per MCU): the tiling the peripheral's
    /// block-ordered output follows.
    const fn mcu(self) -> (usize, usize, usize) {
        match self {
            JpegSubsampling::Gray => (8, 8, 64),
            JpegSubsampling::Ycbcr444 => (8, 8, 192),
            JpegSubsampling::Ycbcr422 => (16, 8, 256),
            JpegSubsampling::Ycbcr420 => (16, 16, 384),
        }
    }
}

/// Decoded-image parameters the peripheral latches once it has parsed the
/// bitstream's header. The dimensions come from the (attacker-controlled)
/// bitstream, so every consumer sizes with checked math ([`decoded_len`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JpegImageInfo {
    /// Image width in pixels.
    pub width: u16,
    /// Image height in pixels.
    pub height: u16,
    /// Chroma subsampling of the emitted block stream.
    pub subsampling: JpegSubsampling,
}

/// Byte count of the peripheral's MCU-block-ordered output for `info`:
/// the image dimensions rounded up to whole MCUs times the MCU payload.
/// `None` for zero dimensions or arithmetic overflow, so a malformed header
/// fails the decode instead of sizing a bogus buffer.
pub const fn decoded_len(info: JpegImageInfo) -> Option<usize> {
    if info.width == 0 || info.height == 0 {
        return None;
    }
    let (mw, mh, bytes) = info.subsampling.mcu();
    let mcus_x = (info.width as usize).div_ceil(mw);
    let mcus_y = (info.height as usize).div_ceil(mh);
    match mcus_x.checked_mul(mcus_y) {
        Some(mcus) => mcus.checked_mul(bytes),
        None => None,
    }
}

/// One whole-bitstream decode on the codec peripheral (the STM32H7 flow: DMA
/// the JFIF stream into the input FIFO, await the header-processed flag for
/// the image parameters, drain the output FIFO into `out`). Returns the
/// parsed parameters and the byte count written; the element cross-checks
/// the two ([`decoded_len`]), so a peripheral that disagrees with its own
/// header report is surfaced as a fault, not trusted. An `out` too small for
/// the image must fail (the adapter knows `out.len()` up front).
#[allow(async_fn_in_trait)]
pub trait JpegDecoder {
    /// Decode one complete JPEG bitstream into `out`.
    async fn decode(
        &mut self,
        jpeg: &[u8],
        out: &mut [u8],
    ) -> Result<(JpegImageInfo, usize), G2gError>;
}

/// A heap-free hardware-JPEG decode [`StaticTransform`]: one complete JPEG
/// bitstream per input frame (`CompressedVideo{Mjpeg}` framing), the
/// peripheral's MCU-block-ordered YCbCr out, lent from a [`StaticLendRing`].
/// A payload without the SOI/EOI markers is rejected as
/// [`G2gError::CapsMismatch`] before anything reaches the peripheral; an
/// output byte count that contradicts the reported image parameters is a
/// peripheral fault ([`HardwareError::Peripheral`](g2g_core::error::HardwareError)).
pub struct HwJpegDec<'r, D, const N: usize, const BYTES: usize> {
    decoder: D,
    ring: &'r StaticLendRing<N, BYTES>,
    info: Option<JpegImageInfo>,
}

impl<D: JpegDecoder, const N: usize, const BYTES: usize> HwJpegDec<'static, D, N, BYTES> {
    /// A decode element over a `'static` ring (the MCU idiom), which makes
    /// the zero-copy lend safe by construction.
    pub fn new(decoder: D, ring: &'static StaticLendRing<N, BYTES>) -> Self {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(decoder, ring) }
    }
}

impl<'r, D: JpegDecoder, const N: usize, const BYTES: usize> HwJpegDec<'r, D, N, BYTES> {
    /// A decode element over a borrowed (e.g. stack-local) ring.
    ///
    /// # Safety
    /// The ring must outlive every frame this element publishes: the
    /// pipeline must drain before the ring is dropped.
    pub unsafe fn with_ring(decoder: D, ring: &'r StaticLendRing<N, BYTES>) -> Self {
        Self {
            decoder,
            ring,
            info: None,
        }
    }

    /// The image parameters of the most recent decode (geometry +
    /// subsampling for the app's downstream wiring).
    pub fn info(&self) -> Option<JpegImageInfo> {
        self.info
    }

    /// Release the peripheral (e.g. to reconfigure it).
    pub fn free(self) -> D {
        self.decoder
    }
}

impl<D: JpegDecoder, const N: usize, const BYTES: usize> StaticTransform
    for HwJpegDec<'_, D, N, BYTES>
{
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let MemoryDomain::System(slice) = &input.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        let jpeg = slice.as_slice();
        // One complete JFIF stream per frame: SOI first, EOI last. Checked
        // before any peripheral traffic (the silicon would stall on a
        // truncated stream, not error).
        let framed = matches!(jpeg, [0xFF, 0xD8, .., 0xFF, 0xD9]);
        if !framed {
            return Err(G2gError::CapsMismatch);
        }
        let Some(mut slot) = self.ring.acquire() else {
            return Err(G2gError::PoolExhausted);
        };
        let (info, written) = self.decoder.decode(jpeg, slot.buf_mut()).await?;
        // The peripheral's write count must equal what its own header report
        // implies; anything else is a fault, not something to propagate.
        if decoded_len(info) != Some(written) || written > BYTES {
            return Err(G2gError::Hardware(
                g2g_core::error::HardwareError::Peripheral,
            ));
        }
        self.info = Some(info);
        // The shared `lend_converted` helper takes a sync fill closure; here
        // the peripheral itself writes the slot during the async decode, so
        // this element publishes directly.
        // SAFETY: the constructor established the ring-outlives-frames
        // contract (`new`: 'static; `with_ring`: caller's contract).
        let out = unsafe { slot.publish(written) };
        Ok(Some(Frame::new(
            MemoryDomain::System(out),
            input.timing,
            input.sequence,
        )))
    }
}

impl<D, const N: usize, const BYTES: usize> core::fmt::Debug for HwJpegDec<'_, D, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HwJpegDec")
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}
