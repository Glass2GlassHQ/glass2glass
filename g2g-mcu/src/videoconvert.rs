//! Heap-free raw-video color convert (M661): packed YUYV 4:2:2 (the order a
//! DCMI/DVP camera like the OV2640 puts on the bus) to planar I420 4:2:0 (what
//! the hardware H.264 encoder wants), as a `no_std` [`StaticTransform`] that
//! lends its output from a [`StaticLendRing`].
//!
//! This is the MCU twin of the host `g2g-plugins::VideoConvert` (M23): that
//! element does the full format matrix but is `alloc`-based (`Vec`/`Box`, a
//! boxed future per frame), so it cannot enter the heap-free MCU pipeline. This
//! one does the single conversion the camera -> encode path needs, in place,
//! through the ring slot, no allocation. It closes the gap between `GrabberSrc`
//! (a YUYV camera) and [`HwH264Enc`](crate::HwH264Enc) (an I420 encoder).
//!
//! Chroma downsample is vertical 2:1 averaging: packed YUYV is already 4:2:0's
//! horizontal chroma resolution (one U/V per 2 pixels), so only the two rows of
//! a 2x2 block are averaged into one I420 chroma sample.

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::memory::MemoryDomain;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::StaticTransform;

use crate::hwh264::i420_len;
use crate::lend::lend_converted;

/// Byte count of one packed YUYV (YUY2) 4:2:2 frame: 2 bytes per pixel. `None`
/// for zero or odd width (a YUYV pair covers two horizontal pixels) or
/// overflow, so a mis-sized frame fails rather than sizing a bogus buffer.
pub const fn yuyv_len(width: u16, height: u16) -> Option<usize> {
    if width == 0 || height == 0 || width % 2 != 0 {
        return None;
    }
    match (width as usize).checked_mul(height as usize) {
        Some(px) => px.checked_mul(2),
        None => None,
    }
}

/// Convert one packed YUYV frame (`src`, `width*height*2` bytes) into planar
/// I420 (`dst`, `width*height*3/2` bytes). Both lengths are the caller's
/// contract; all access is bounds-guarded (slice patterns + `get`), so the
/// conversion adds no panic path to the no-alloc subset.
fn yuyv_to_i420(width: usize, height: usize, src: &[u8], dst: &mut [u8]) {
    let luma = width.saturating_mul(height);
    let cw = width / 2; // chroma samples per row (one U/V per pixel pair)
    let (y_plane, chroma) = dst.split_at_mut(luma.min(dst.len()));
    let csize = cw.saturating_mul(height / 2);
    let (u_plane, v_plane) = chroma.split_at_mut(csize.min(chroma.len()));

    // Y plane: the first byte of every 2-byte YUYV group, in pixel order.
    for (yo, px) in y_plane.iter_mut().zip(src.chunks_exact(2)) {
        if let [y, _] = px {
            *yo = *y;
        }
    }

    // Chroma: for each 2x2 block, average the U (and V) of the two source rows.
    // YUYV per pair is [Y0, U, Y1, V], so U is byte 1 and V byte 3 of a 4-chunk.
    let row_bytes = width.saturating_mul(2);
    for cby in 0..(height / 2) {
        let top = src
            .get(cby.saturating_mul(2).saturating_mul(row_bytes)..)
            .map(|s| s.get(..row_bytes).unwrap_or(s));
        let bot = src
            .get((cby.saturating_mul(2).saturating_add(1)).saturating_mul(row_bytes)..)
            .map(|s| s.get(..row_bytes).unwrap_or(s));
        let (Some(top), Some(bot)) = (top, bot) else {
            continue;
        };
        let base = cby.saturating_mul(cw);
        let (Some(u_row), Some(v_row)) = (
            u_plane.get_mut(base..base + cw),
            v_plane.get_mut(base..base + cw),
        ) else {
            continue;
        };
        for (i, (tq, bq)) in top.chunks_exact(4).zip(bot.chunks_exact(4)).enumerate() {
            if let (Some(uo), [_, tu, _, tv], [_, bu, _, bv]) = (u_row.get_mut(i), tq, bq) {
                *uo = ((*tu as u16 + *bu as u16) / 2) as u8;
                if let Some(vo) = v_row.get_mut(i) {
                    *vo = ((*tv as u16 + *bv as u16) / 2) as u8;
                }
            }
        }
    }
}

/// A heap-free packed-YUYV 4:2:2 -> planar I420 4:2:0 [`StaticTransform`],
/// lending its output from a [`StaticLendRing`]. Constructed for a fixed
/// `width x height`; a frame whose payload is not `width*height*2` bytes is
/// rejected as [`G2gError::CapsMismatch`].
pub struct YuyvToI420<'r, const N: usize, const BYTES: usize> {
    ring: &'r StaticLendRing<N, BYTES>,
    width: usize,
    height: usize,
    in_len: usize,
    out_len: usize,
}

impl<const N: usize, const BYTES: usize> YuyvToI420<'static, N, BYTES> {
    /// A converter for `width x height` frames over a `'static` ring (the MCU
    /// idiom). `None` for invalid 4:2:0/4:2:2 geometry (zero or odd dimensions).
    pub fn new(width: u16, height: u16, ring: &'static StaticLendRing<N, BYTES>) -> Option<Self> {
        // SAFETY: `'static` trivially satisfies with_ring's outlives contract.
        unsafe { Self::with_ring(width, height, ring) }
    }
}

impl<'r, const N: usize, const BYTES: usize> YuyvToI420<'r, N, BYTES> {
    /// A converter over a borrowed (e.g. stack-local) ring. `None` for invalid
    /// geometry.
    ///
    /// # Safety
    /// The ring must outlive every frame this element publishes.
    pub unsafe fn with_ring(
        width: u16,
        height: u16,
        ring: &'r StaticLendRing<N, BYTES>,
    ) -> Option<Self> {
        Some(Self {
            ring,
            width: width as usize,
            height: height as usize,
            in_len: yuyv_len(width, height)?,
            out_len: i420_len(width, height)?,
        })
    }
}

impl<const N: usize, const BYTES: usize> StaticTransform for YuyvToI420<'_, N, BYTES> {
    async fn process(&mut self, input: Frame) -> Result<Option<Frame>, G2gError> {
        let MemoryDomain::System(slice) = &input.domain else {
            return Err(G2gError::UnsupportedDomain);
        };
        if slice.as_slice().len() != self.in_len {
            return Err(G2gError::CapsMismatch);
        }
        let (w, h) = (self.width, self.height);
        // SAFETY: the constructor established the ring-outlives-frames contract
        // (`new`: 'static; `with_ring`: caller's contract).
        let out = unsafe {
            lend_converted(self.ring, &input, self.out_len, |src, dst| {
                yuyv_to_i420(w, h, src, dst)
            })
        }?;
        Ok(Some(out))
    }
}

impl<const N: usize, const BYTES: usize> core::fmt::Debug for YuyvToI420<'_, N, BYTES> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("YuyvToI420")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}
