//! Hardware H.264 encoder seam (M660): the [`H264Encoder`] contract, shaped
//! like the ESP32-P4 (and STM32MP / i.MX) hardware H.264 encoder block (feed
//! one raw I420 frame, the silicon emits one Annex-B access unit and reports
//! its byte count + whether it is an IDR keyframe), plus [`HwH264Enc`], the
//! heap-free [`StaticTransform`] that owns everything around the peripheral:
//! input geometry validation (checked 4:2:0 sizing), output sizing against the
//! reported byte count, the ring lend, and fault surfacing.
//!
//! This is the encode twin of [`HwJpegDec`](crate::HwJpegDec): where that
//! decodes one whole JFIF stream per frame, this encodes one raw frame into
//! one compressed access unit per frame. The board adapter typically DMA-feeds
//! the input planes and DMA-drains the bitstream FIFO; the mock in
//! `m660_hwh264.rs` replays that contract, so the element's real logic is
//! host-tested. The output is Annex-B (start-code delimited NAL units), matching
//! the framing the rest of the pipeline assumes (`h264parse`, `rtppay`). On the
//! ESP32-P4 the encoder is reached over the M650 C-seam (`CH264Encoder`): the
//! ESP-IDF encoder driver *is* the peripheral. On-device validation (a
//! `Hardware` conformance row on real silicon) needs a board and stays open,
//! like [`HwJpegDec`]'s and `VtDecode`'s runtime tiers.

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;

/// What the peripheral reports after encoding one frame: the access unit's
/// byte count and whether it is an IDR (keyframe). Downstream (an RTP
/// payloader) needs the keyframe flag to set the stream's random-access
/// points; the element records the most recent one ([`HwH264Enc::info`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H264EncodeInfo {
    /// Bytes written to the output slot for this access unit.
    pub len: usize,
    /// True for an IDR access unit (a decodable-from-here keyframe).
    pub keyframe: bool,
}

/// Byte count of one raw I420 (YUV 4:2:0 planar) frame: a full-resolution Y
/// plane plus two quarter-resolution chroma planes. `None` for zero or odd
/// dimensions (4:2:0 needs even width/height) or arithmetic overflow, so a
/// mis-sized capture fails the encode rather than feeding the peripheral a
/// partial frame.
pub const fn i420_len(width: u16, height: u16) -> Option<usize> {
    if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
        return None;
    }
    let (w, h) = (width as usize, height as usize);
    // luma = w*h; each chroma plane = (w/2)*(h/2); total = w*h*3/2.
    match w.checked_mul(h) {
        Some(luma) => match (w / 2).checked_mul(h / 2) {
            Some(chroma) => match chroma.checked_mul(2) {
                Some(chroma2) => luma.checked_add(chroma2),
                None => None,
            },
            None => None,
        },
        None => None,
    }
}

/// One raw-frame encode on the codec peripheral (the ESP32-P4 flow: DMA the
/// I420 planes into the encoder, await done, drain the bitstream FIFO into
/// `out`). Returns the access unit's byte count and keyframe flag; the element
/// cross-checks the count against the output slot, so a peripheral that claims
/// more than it wrote is surfaced as a fault, not trusted. An `out` too small
/// for the access unit must fail (the adapter knows `out.len()` up front).
#[allow(async_fn_in_trait)]
pub trait H264Encoder {
    /// Encode one raw I420 frame into `out`, returning the access unit info.
    async fn encode(&mut self, raw: &[u8], out: &mut [u8]) -> Result<H264EncodeInfo, G2gError>;
}

/// A heap-free hardware-H.264 encode [`StaticTransform`]: one raw I420 frame in
/// (`width x height`, validated as 4:2:0), one Annex-B access unit out, lent
/// from a [`StaticLendRing`]. A payload whose size is not the frame's I420
/// length is rejected as [`G2gError::CapsMismatch`] before anything reaches the
/// peripheral; an encoded byte count of zero or beyond the slot is a peripheral
/// fault ([`HardwareError::Peripheral`](g2g_core::error::HardwareError)).
pub struct HwH264Enc<'r, E, const N: usize, const BYTES: usize> {
    encoder: E,
    ring: &'r StaticLendRing<N, BYTES>,
    raw_len: usize,
    info: Option<H264EncodeInfo>,
}

impl<E: H264Encoder, const N: usize, const BYTES: usize> HwH264Enc<'static, E, N, BYTES> {
    /// An encode element for `width x height` I420 frames over a `'static` ring
    /// (the MCU idiom), which makes the zero-copy lend safe by construction.
    /// Returns `None` for invalid 4:2:0 geometry.
    pub fn new(
        encoder: E,
        width: u16,
        height: u16,
        ring: &'static StaticLendRing<N, BYTES>,
    ) -> Option<Self> {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(encoder, width, height, ring) }
    }
}

impl<'r, E: H264Encoder, const N: usize, const BYTES: usize> HwH264Enc<'r, E, N, BYTES> {
    /// An encode element over a borrowed (e.g. stack-local) ring. `None` for
    /// invalid 4:2:0 geometry.
    ///
    /// # Safety
    /// The ring must outlive every frame this element publishes: the pipeline
    /// must drain before the ring is dropped.
    pub unsafe fn with_ring(
        encoder: E,
        width: u16,
        height: u16,
        ring: &'r StaticLendRing<N, BYTES>,
    ) -> Option<Self> {
        Some(Self {
            encoder,
            ring,
            raw_len: i420_len(width, height)?,
            info: None,
        })
    }

    /// The access unit info (byte count + keyframe flag) of the most recent
    /// encode, for the app's downstream wiring (e.g. RTP marker / random access).
    pub fn info(&self) -> Option<H264EncodeInfo> {
        self.info
    }

    /// Release the peripheral (e.g. to reconfigure the bitrate).
    pub fn free(self) -> E {
        self.encoder
    }
}

impl<E: H264Encoder, const N: usize, const BYTES: usize> StaticTransform
    for HwH264Enc<'_, E, N, BYTES>
{
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let Some(raw) = input.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        // Exactly one full I420 frame per input; a partial/mis-sized capture is
        // rejected before any peripheral traffic (the silicon would encode
        // garbage or stall, not error).
        if raw.len() != self.raw_len {
            return Err(G2gError::CapsMismatch);
        }
        let Some(mut slot) = self.ring.acquire() else {
            return Err(G2gError::PoolExhausted);
        };
        // `raw` borrows the input frame, the slot borrows the ring: distinct
        // buffers, so the encoder reads one and writes the other with no alias.
        let encoded = self.encoder.encode(raw, slot.buf_mut()).await?;
        // A zero-length access unit or one larger than the slot is a fault, not
        // something to propagate as a frame.
        if encoded.len == 0 || encoded.len > BYTES {
            return Err(G2gError::Hardware(
                g2g_core::error::HardwareError::Peripheral,
            ));
        }
        self.info = Some(encoded);
        // SAFETY: the constructor established the ring-outlives-frames contract
        // (`new`: 'static; `with_ring`: caller's contract).
        let out = unsafe { slot.publish(encoded.len) };
        Ok(Some(Frame::new(
            MemoryDomain::System(out),
            input.timing,
            input.sequence,
        )))
    }
}

impl<E, const N: usize, const BYTES: usize> core::fmt::Debug for HwH264Enc<'_, E, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HwH264Enc")
            .field("raw_len", &self.raw_len)
            .field("info", &self.info)
            .finish_non_exhaustive()
    }
}
