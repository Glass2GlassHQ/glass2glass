//! Shared `CVPixelBuffer` helpers for the macOS elements (`VtDecode`,
//! `AvfVideoSrc`, `ScreenCaptureSrc`): tight-NV12 packing out of a bi-planar
//! buffer, the retained-buffer keep-alive behind the zero-copy `CvPixelBuffer`
//! domain, and the delegate-to-run-loop handoff for the capture sources.

use objc2_core_foundation::CFRetained;
use objc2_core_media::CMSampleBuffer;
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferGetBaseAddressOfPlane, CVPixelBufferGetBytesPerRowOfPlane,
    CVPixelBufferGetHeight, CVPixelBufferGetHeightOfPlane, CVPixelBufferGetIOSurface,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferGetWidthOfPlane,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
};

use g2g_core::CvPixelBufferKeepAlive;

use alloc::boxed::Box;
use alloc::vec::Vec;
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

/// One captured video frame, retained straight from a capture delegate.
pub(crate) struct CapturedVideo {
    pub(crate) buf: CFRetained<CVPixelBuffer>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) pixel_format: u32,
    pub(crate) io_surface_backed: bool,
}

// SAFETY: the retained pixel buffer crosses from the delegate queue to the run
// loop through a mutex; CoreFoundation retain/release is thread-safe and the
// pixels are immutable after capture.
unsafe impl Send for CapturedVideo {}

/// Retain a delegate-delivered sample buffer's pixel buffer and read its
/// geometry / format / backing, ready to queue for the element's run loop.
pub(crate) fn captured_video_from_sample(sample_buffer: &CMSampleBuffer) -> Option<CapturedVideo> {
    // SAFETY: the sample buffer is valid for the delegate callback; the image
    // buffer is retained (+1) by the accessor, keeping it alive past the
    // delegate's scope as Apple's docs require.
    let image = unsafe { sample_buffer.image_buffer() }?;
    // A CVImageBufferRef IS a CVPixelBufferRef for video output.
    // SAFETY: same-representation CF types; the retain moves over.
    let buf: CFRetained<CVPixelBuffer> = unsafe { CFRetained::cast_unchecked(image) };
    let width = CVPixelBufferGetWidth(&buf) as u32;
    let height = CVPixelBufferGetHeight(&buf) as u32;
    let pixel_format = CVPixelBufferGetPixelFormatType(&buf);
    let io_surface_backed = CVPixelBufferGetIOSurface(Some(&buf)).is_some();
    Some(CapturedVideo {
        buf,
        width,
        height,
        pixel_format,
        io_surface_backed,
    })
}

/// Shared between a capture delegate (on its dispatch queue) and the element's
/// run loop.
pub(crate) struct Shared<T> {
    pub(crate) filled: Mutex<VecDeque<T>>,
    pub(crate) cv: Condvar,
}

// Manual impls: derives would bound `T: Default` / `T: Debug`, which the
// captured payloads don't carry.
impl<T> Default for Shared<T> {
    fn default() -> Self {
        Self {
            filled: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
        }
    }
}

impl<T> core::fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Shared")
            .field("queued", &self.filled.lock().map(|q| q.len()).unwrap_or(0))
            .finish_non_exhaustive()
    }
}

/// The two NV12 (4:2:0 bi-planar) pixel formats CoreVideo produces:
/// video-range `'420v'` and full-range `'420f'`. We accept either and pack to
/// our NV12 byte layout; the BT.601 / range semantics ride in caps, not here.
pub(crate) const K_CV_PIXEL_FORMAT_420V: u32 = 0x3432_3076; // '420v'
pub(crate) const K_CV_PIXEL_FORMAT_420F: u32 = 0x3432_3066; // '420f'

/// Pins a `CVPixelBuffer` for a downstream frame's lifetime (the keep-alive
/// inside `OwnedCvPixelBuffer`); the last drop releases the producer's buffer.
pub(crate) struct CvBufferOwner(pub(crate) CFRetained<CVPixelBuffer>);

impl core::fmt::Debug for CvBufferOwner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CvBufferOwner({:p})", CFRetained::as_ptr(&self.0))
    }
}

impl CvPixelBufferKeepAlive for CvBufferOwner {}

// SAFETY: CoreFoundation retain/release is thread-safe, and the pixels are
// immutable once the producer (a decoder's output callback, a capture
// delegate) hands the buffer over, so sharing read-only across threads is
// sound. Same contract as `MfDecode`'s sample owner.
unsafe impl Send for CvBufferOwner {}
// SAFETY: see the `Send` justification; the owner exposes no mutation.
unsafe impl Sync for CvBufferOwner {}

/// Copy the locked bi-planar pixel buffer into a tight NV12 byte buffer
/// (`w*h` luma + `w*(h/2)` interleaved chroma), stripping per-row padding.
///
/// SAFETY: `pb` is locked for read; plane base addresses / strides are valid for
/// the plane dimensions CoreVideo reports.
pub(crate) unsafe fn pack_nv12(
    pb: &CVPixelBuffer,
    width: usize,
    height: usize,
) -> Option<Box<[u8]>> {
    let mut out = Vec::with_capacity(width * height * 3 / 2);
    // Plane 0: luma (w x h). Plane 1: interleaved CbCr (w x h/2 bytes/row).
    for plane in 0..2usize {
        let base = CVPixelBufferGetBaseAddressOfPlane(pb, plane) as *const u8;
        if base.is_null() {
            return None;
        }
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, plane);
        let pw = CVPixelBufferGetWidthOfPlane(pb, plane); // luma: w, chroma: w/2 (CbCr pairs)
        let ph = CVPixelBufferGetHeightOfPlane(pb, plane); // luma: h, chroma: h/2
                                                           // Bytes per row of valid data: luma = pw, chroma = pw * 2 (CbCr pair).
        let row_bytes = if plane == 0 { pw } else { pw * 2 };
        for row in 0..ph {
            // SAFETY: row < plane height, row_bytes <= stride, base valid for the
            // plane; the source slice stays within the locked plane.
            let src = unsafe { core::slice::from_raw_parts(base.add(row * stride), row_bytes) };
            out.extend_from_slice(src);
        }
    }
    Some(out.into_boxed_slice())
}

/// Lock `pb` for read, pack it to tight NV12, and unlock. `None` when the lock
/// or a plane pointer fails.
pub(crate) fn pack_nv12_locked(
    pb: &CVPixelBuffer,
    width: usize,
    height: usize,
) -> Option<Box<[u8]>> {
    // SAFETY: lock for read while the planes are copied out, unlock after.
    let lock = unsafe { CVPixelBufferLockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly) };
    if lock != 0 {
        return None;
    }
    // SAFETY: locked above.
    let packed = unsafe { pack_nv12(pb, width, height) };
    // SAFETY: paired with the lock above.
    unsafe {
        CVPixelBufferUnlockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly);
    }
    packed
}
