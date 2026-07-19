//! C FFI seam adapters (M650): let existing C code *be* a `g2g-mcu` peripheral
//! seam, so a shop with a large C driver investment integrates without writing
//! Rust. The board keeps its C DMA/capture routine and its C network stack; it
//! registers them as function pointers, and g2g calls them each frame across
//! [`CFrameGrabber`] (the [`FrameGrabber`] capture seam) and [`CPacketSender`]
//! (the [`PacketSender`] egress seam).
//!
//! This is the inverse direction of the existing C ABI: `examples/g2g-freertos`
//! links the whole pipeline as a Rust static library the C app *calls into*;
//! here C code is called *back* as the peripheral, so the drivers a C/RTOS shop
//! already owns drive a g2g graph with no Rust adapter to hand-write. The
//! `examples/g2g-cffi` staticlib composes these into a C-driven, frame-stepped
//! pipeline (`g2g_audio_egress_init` + `g2g_audio_egress_step`) proven end to
//! end from a C harness.
//!
//! Everything here is `no_std` with no `alloc` and adds no panic path (a
//! negative callback return is mapped to an error, never an `unwrap`), so the
//! C-seam path keeps the same heap-free / panic-free guarantees as the rest of
//! `g2g-mcu` (asserted on the `g2g-cffi` archive by `tools/cffi-check.sh`).

use core::ffi::c_void;
use core::fmt;

use g2g_core::error::{G2gError, HardwareError};
use g2g_core::rtp::RTP_HEADER_LEN;

use crate::grabber::FrameGrabber;
use crate::hwh264::{H264EncodeInfo, H264Encoder};
use crate::rtp::PacketSender;

/// A C capture callback: fill up to `len` bytes at `buf` with one frame and
/// return the byte count written (`>= 0`, typically `len` for a fixed-format
/// camera / mic), or a negative value to report a capture fault. The board's
/// DMA/driver code lives behind this pointer; `ctx` is an opaque handle passed
/// back verbatim (the driver's own state).
pub type CaptureFn = unsafe extern "C" fn(ctx: *mut c_void, buf: *mut u8, len: usize) -> isize;

/// A C send callback: transmit one datagram, `header` (the RTP fixed header,
/// [`RTP_HEADER_LEN`] bytes) immediately followed by `payload`. Return `0` on
/// success or a negative value on a transport fault. The board's network stack
/// lives behind this pointer (lwIP, Zephyr sockets, smoltcp); `ctx` is its
/// opaque handle. Header and payload are separate pointers so a scatter-gather
/// stack stays zero-copy; a flat-buffer stack concatenates them itself.
pub type SendFn = unsafe extern "C" fn(
    ctx: *mut c_void,
    header: *const u8,
    header_len: usize,
    payload: *const u8,
    payload_len: usize,
) -> i32;

/// A [`FrameGrabber`] backed by a C capture callback: the zero-Rust capture
/// seam. The board registers its C DMA/driver function once; [`GrabberSrc`]
/// then calls it to fill each lent ring slot.
///
/// [`GrabberSrc`]: crate::grabber::GrabberSrc
pub struct CFrameGrabber {
    capture: CaptureFn,
    ctx: *mut c_void,
}

impl CFrameGrabber {
    /// Wrap a C capture callback and its opaque context.
    ///
    /// # Safety
    /// `capture` must stay a valid function for the life of this grabber and
    /// `ctx` a valid handle to pass it; each call must write no more than the
    /// requested `len` bytes at `buf` and return the count (or a negative fault
    /// code). These are the same invariants the C side upholds for its own DMA.
    pub unsafe fn new(capture: CaptureFn, ctx: *mut c_void) -> Self {
        Self { capture, ctx }
    }
}

impl FrameGrabber for CFrameGrabber {
    async fn capture(&mut self, buf: &mut [u8]) -> Result<usize, G2gError> {
        // SAFETY: the constructor's contract: `capture` is a valid function over
        // `ctx`, and it writes at most `buf.len()` bytes at `buf`.
        let n = unsafe { (self.capture)(self.ctx, buf.as_mut_ptr(), buf.len()) };
        if n < 0 {
            // The driver reported a fault (DMA error, peripheral not ready).
            return Err(G2gError::Hardware(HardwareError::Peripheral));
        }
        let n = n as usize;
        if n > buf.len() {
            // A driver claiming more than the slot holds is a contract violation
            // (the same one `GrabberSrc` rejects); fail loud, never truncate.
            return Err(G2gError::CapsMismatch);
        }
        Ok(n)
    }
}

impl fmt::Debug for CFrameGrabber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CFrameGrabber").finish_non_exhaustive()
    }
}

/// A C H.264-encode callback: encode one raw I420 frame (`raw`, `raw_len`
/// bytes) into `out` (capacity `out_cap`), writing whether it is an IDR
/// keyframe to `*keyframe` (non-zero = keyframe). Return the access unit's byte
/// count (`>= 0`) or a negative value on an encoder fault. The board's hardware
/// H.264 driver (ESP-IDF `esp_h264`, an i.MX/STM32 VPU) lives behind this
/// pointer; `ctx` is its opaque handle.
pub type EncodeFn = unsafe extern "C" fn(
    ctx: *mut c_void,
    raw: *const u8,
    raw_len: usize,
    out: *mut u8,
    out_cap: usize,
    keyframe: *mut i32,
) -> isize;

/// An [`H264Encoder`] backed by a C encode callback: the zero-Rust hardware
/// H.264 seam. [`HwH264Enc`] hands each raw frame to the board's C encoder
/// driver through this pointer and publishes the returned access unit.
///
/// [`HwH264Enc`]: crate::hwh264::HwH264Enc
pub struct CH264Encoder {
    encode: EncodeFn,
    ctx: *mut c_void,
}

impl CH264Encoder {
    /// Wrap a C encode callback and its opaque context.
    ///
    /// # Safety
    /// `encode` must stay a valid function for the life of this encoder and
    /// `ctx` a valid handle to pass it; each call must write no more than
    /// `out_cap` bytes at `out`, set `*keyframe`, and return the byte count (or
    /// a negative fault code).
    pub unsafe fn new(encode: EncodeFn, ctx: *mut c_void) -> Self {
        Self { encode, ctx }
    }
}

impl H264Encoder for CH264Encoder {
    async fn encode(&mut self, raw: &[u8], out: &mut [u8]) -> Result<H264EncodeInfo, G2gError> {
        let mut keyframe: i32 = 0;
        // SAFETY: the constructor's contract: `encode` is a valid function over
        // `ctx`; it reads `raw_len` bytes at `raw`, writes at most `out_cap`
        // bytes at `out`, and sets `*keyframe`.
        let n = unsafe {
            (self.encode)(
                self.ctx,
                raw.as_ptr(),
                raw.len(),
                out.as_mut_ptr(),
                out.len(),
                &mut keyframe,
            )
        };
        if n < 0 {
            return Err(G2gError::Hardware(HardwareError::Peripheral));
        }
        let len = n as usize;
        if len > out.len() {
            // An encoder claiming more than the slot holds is a contract
            // violation (the same one HwH264Enc rejects); fail loud.
            return Err(G2gError::CapsMismatch);
        }
        Ok(H264EncodeInfo {
            len,
            keyframe: keyframe != 0,
        })
    }
}

impl fmt::Debug for CH264Encoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CH264Encoder").finish_non_exhaustive()
    }
}

/// A [`PacketSender`] backed by a C send callback: the zero-Rust egress seam.
/// [`RtpSink`] hands each packet's header and payload to the board's C network
/// stack through this pointer.
///
/// [`RtpSink`]: crate::rtp::RtpSink
pub struct CPacketSender {
    send: SendFn,
    ctx: *mut c_void,
}

impl CPacketSender {
    /// Wrap a C send callback and its opaque context.
    ///
    /// # Safety
    /// `send` must stay a valid function for the life of this sender and `ctx`
    /// a valid handle to pass it.
    pub unsafe fn new(send: SendFn, ctx: *mut c_void) -> Self {
        Self { send, ctx }
    }
}

impl PacketSender for CPacketSender {
    async fn send(
        &mut self,
        header: &[u8; RTP_HEADER_LEN],
        payload: &[u8],
    ) -> Result<(), G2gError> {
        // SAFETY: the constructor's contract: `send` is a valid function over
        // `ctx`; both slices are valid for their lengths for the call's duration.
        let rc = unsafe {
            (self.send)(
                self.ctx,
                header.as_ptr(),
                header.len(),
                payload.as_ptr(),
                payload.len(),
            )
        };
        if rc < 0 {
            return Err(G2gError::Hardware(HardwareError::Peripheral));
        }
        Ok(())
    }
}

impl fmt::Debug for CPacketSender {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CPacketSender").finish_non_exhaustive()
    }
}
