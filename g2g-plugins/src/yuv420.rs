//! Shared `YUV_420_888` -> NV12 packer for the Android `ndk`-image elements
//! (camera2src capture, mediacodecdec decode), whose packing was byte-identical.

use alloc::vec::Vec;

use ndk::media::image_reader::Image;

/// Pack a decoded `YUV_420_888` image to tight NV12 (Y plane then interleaved
/// UV), returning the packed bytes plus the width / height it used. Each plane's
/// row and pixel strides describe whatever layout the producer chose, so this
/// one path handles planar I420, semi-planar, and vendor formats alike (a chroma
/// pixel stride of 2 is an already-interleaved semi-planar source; 1 is planar).
/// `None` if the image reports a zero dimension or any plane access is out of
/// bounds.
pub(crate) fn pack_yuv420_to_nv12(img: &Image) -> Option<(Vec<u8>, u32, u32)> {
    let w = img.width().ok()?.max(0) as usize;
    let h = img.height().ok()?.max(0) as usize;
    if w == 0 || h == 0 {
        return None;
    }
    let y = img.plane_data(0).ok()?;
    let y_rs = img.plane_row_stride(0).ok()? as usize;
    let u = img.plane_data(1).ok()?;
    let u_rs = img.plane_row_stride(1).ok()? as usize;
    let u_ps = img.plane_pixel_stride(1).ok()? as usize;
    let v = img.plane_data(2).ok()?;
    let v_rs = img.plane_row_stride(2).ok()? as usize;
    let v_ps = img.plane_pixel_stride(2).ok()? as usize;

    let (cw, ch) = (w / 2, h / 2);
    let mut nv12 = Vec::with_capacity(w * h + 2 * cw * ch);
    // Luma: w bytes per row, row-stride apart.
    for row in 0..h {
        let off = row * y_rs;
        nv12.extend_from_slice(y.get(off..off + w)?);
    }
    // Chroma: interleave Cb,Cr honoring each plane's row + pixel stride.
    for row in 0..ch {
        for col in 0..cw {
            nv12.push(*u.get(row * u_rs + col * u_ps)?);
            nv12.push(*v.get(row * v_rs + col * v_ps)?);
        }
    }
    Some((nv12, w as u32, h as u32))
}
