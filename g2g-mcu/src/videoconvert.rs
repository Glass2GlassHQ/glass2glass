//! Heap-free raw-video color convert: packed YUYV 4:2:2 (the order a
//! DCMI/DVP camera like the OV2640 puts on the bus) to planar I420 4:2:0 (what
//! the hardware H.264 encoder wants), as a `no_std` [`StaticTransform`] that
//! lends its output from a [`StaticLendRing`].
//!
//! This is the MCU twin of the host `g2g-plugins::VideoConvert`: that
//! element does the full format matrix but is `alloc`-based (`Vec`/`Box`, a
//! boxed future per frame), so it cannot enter the heap-free MCU pipeline. This
//! one does the single conversion the camera -> encode path needs, in place,
//! through the ring slot, no allocation. It closes the gap between `GrabberSrc`
//! (a YUYV camera) and [`HwH264Enc`](crate::HwH264Enc) (an I420 encoder).
//!
//! Chroma downsample is vertical 2:1 averaging: packed YUYV is already 4:2:0's
//! horizontal chroma resolution (one U/V per 2 pixels), so only the two rows of
//! a 2x2 block are averaged into one I420 chroma sample. The host converter
//! runs the same [`yuyv_to_yuv420`].

use g2g_core::error::G2gError;
use g2g_core::frame::Frame;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{RawVideoFormat, StaticTransform};

use crate::hwh264::i420_len;
use crate::lend::lend_converted;

/// Byte count of one packed YUYV (YUY2) 4:2:2 frame: 2 bytes per pixel. `None`
/// for zero or odd width (a YUYV pair covers two horizontal pixels) or
/// overflow, so a mis-sized frame fails rather than sizing a bogus buffer.
pub fn yuyv_len(width: u16, height: u16) -> Option<usize> {
    if width == 0 || height == 0 || width % 2 != 0 {
        return None;
    }
    RawVideoFormat::Yuyv
        .unpadded_frame_bytes(u32::from(width), u32::from(height))
        .and_then(|bytes| usize::try_from(bytes).ok())
}

/// Bytes in one YUYV `Y0 U Y1 V` group, two pixels.
const YUYV_GROUP_BYTES: usize = 4;

/// Convert one packed YUYV 4:2:2 frame (`Y0 U Y1 V`, rows of `width` rounded up
/// to whole groups) into 8-bit 4:2:0, NV12 chroma when `interleaved`, else
/// I420. The luma plane is a direct deinterleave; each chroma sample is the
/// rounded average of the two source rows that share it (an odd height's last
/// row stands alone). `None` when `src` or `dst` is short of a frame. Heap-free
/// and panic-free: every access is bounds-checked.
pub fn yuyv_to_yuv420(
    src: &[u8],
    width: usize,
    height: usize,
    interleaved: bool,
    dst: &mut [u8],
) -> Option<()> {
    if width == 0 || height == 0 {
        return Some(());
    }
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let stride = chroma_width.checked_mul(YUYV_GROUP_BYTES)?;
    let luma = width.checked_mul(height)?;
    let chroma = chroma_width.checked_mul(chroma_height)?;
    let src = src.get(..stride.checked_mul(height)?)?;
    let dst = dst.get_mut(..luma.checked_add(chroma.checked_mul(2)?)?)?;
    let (y_plane, chroma_planes) = dst.split_at_mut(luma);

    for (y_row, src_row) in y_plane
        .chunks_exact_mut(width)
        .zip(src.chunks_exact(stride))
    {
        for (pair, group) in y_row
            .chunks_mut(2)
            .zip(src_row.chunks_exact(YUYV_GROUP_BYTES))
        {
            if let [y0, _, y1, _] = group {
                for (out, sample) in pair.iter_mut().zip([*y0, *y1]) {
                    *out = sample;
                }
            }
        }
    }

    for row in 0..chroma_height {
        let top_row = row.checked_mul(2)?;
        let bottom_row = top_row.saturating_add(1).min(height - 1);
        let top = src.get(top_row.checked_mul(stride)?..)?;
        let bottom = src.get(bottom_row.checked_mul(stride)?..)?;
        let groups = top
            .chunks_exact(YUYV_GROUP_BYTES)
            .zip(bottom.chunks_exact(YUYV_GROUP_BYTES))
            .take(chroma_width);
        for (column, (top_group, bottom_group)) in groups.enumerate() {
            let ([_, tu, _, tv], [_, bu, _, bv]) = (top_group, bottom_group) else {
                return None;
            };
            let average = |a: u8, b: u8| (u16::from(a) + u16::from(b)).div_ceil(2) as u8;
            let at = row.checked_mul(chroma_width)?.checked_add(column)?;
            let (u_at, v_at) = match interleaved {
                true => (at.checked_mul(2)?, at.checked_mul(2)?.checked_add(1)?),
                false => (at, chroma.checked_add(at)?),
            };
            *chroma_planes.get_mut(u_at)? = average(*tu, *bu);
            *chroma_planes.get_mut(v_at)? = average(*tv, *bv);
        }
    }
    Some(())
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
        let Some(slice) = input.domain.as_system_slice() else {
            return Err(G2gError::UnsupportedDomain);
        };
        if slice.len() != self.in_len {
            return Err(G2gError::CapsMismatch);
        }
        let (w, h) = (self.width, self.height);
        let mut converted = false;
        // SAFETY: the constructor established the ring-outlives-frames contract
        // (`new`: 'static; `with_ring`: caller's contract).
        let out = unsafe {
            lend_converted(self.ring, &input, self.out_len, |src, dst| {
                converted = yuyv_to_yuv420(src, w, h, false, dst).is_some();
            })
        }?;
        if !converted {
            return Err(G2gError::CapsMismatch);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_odd_frame_keeps_its_edge_chroma_and_rounds_the_average() {
        // 3x3: two groups a row, the second group's Y1 past the edge
        const WIDTH: usize = 3;
        const HEIGHT: usize = 3;
        let src = [
            1, 10, 2, 20, 3, 30, 0, 40, //
            4, 11, 5, 21, 6, 31, 0, 41, //
            7, 12, 8, 22, 9, 32, 0, 42,
        ];
        let mut i420 = [0u8; 9 + 2 * 4];
        yuyv_to_yuv420(&src, WIDTH, HEIGHT, false, &mut i420).unwrap();
        assert_eq!(
            i420,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 31, 12, 32, 21, 41, 22, 42]
        );
        let mut nv12 = [0u8; 9 + 2 * 4];
        yuyv_to_yuv420(&src, WIDTH, HEIGHT, true, &mut nv12).unwrap();
        assert_eq!(
            nv12,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 21, 31, 41, 12, 22, 32, 42]
        );
        let short = &src[..src.len() - 1];
        assert!(yuyv_to_yuv420(short, WIDTH, HEIGHT, false, &mut i420).is_none());
    }
}
