//! Linux H.264 decode element using ffmpeg / libavcodec.
//!
//! M13 (Linux production path): consumes Annex-B H.264 `DataFrame`s (the
//! bitstream `RtspSrc` / `H264Parse` already emit, `MemoryDomain::System`)
//! and produces decoded I420 frames, also `MemoryDomain::System` (CPU copy
//! out of libavcodec's frame buffer). A `CapsChanged(I420, w, h)` is emitted
//! before the first decoded frame and again whenever the decoder signals a
//! resolution change.
//!
//! Why this element exists alongside `VaapiH264Dec`: cros-codecs 0.0.6
//! cannot allocate decoder output surfaces on AMD desktop GPUs (see
//! `vaapidec.rs` module header for the full diagnosis). libavcodec's
//! software H.264 decoder works on every Linux system out of the box; this
//! is the production-ready baseline. VAAPI hwaccel through ffmpeg
//! (`h264_vaapi` codec + AV_HWDEVICE_TYPE_VAAPI) is a follow-up that
//! preserves the same `AsyncElement` shape.
//!
//! Pipeline:
//!
//! ```text
//! RtspSrc ─► H264Parse ─► FfmpegH264Dec ─► [downstream sink / ML]
//!  (System/H264 Annex-B)        (System/I420)
//! ```
//!
//! Threading: `ffmpeg::decoder::Video` wraps a raw `AVCodecContext*`, which
//! is `!Send` and `!Sync` by default. The element is moved between worker
//! threads but never shared (the runner holds at most one `&mut self`
//! reference at a time), so an `unsafe impl Send` is sound on the same
//! grounds as `MfDecode` and `VaapiH264Dec`: ownership transfer, never
//! aliasing.
//!
//! Deferred:
//! - NV12 output (currently I420). H.264 decodes natively to YUV420P/I420;
//!   downstream consumers that prefer NV12 can use a `ColorConvert` element
//!   (planned) or swscale within this module behind a `nv12-output` knob.
//! - VAAPI hwaccel: open `h264_vaapi` codec with an attached
//!   `AVHWDeviceContext(VAAPI)`, register `get_format` to claim
//!   `AV_PIX_FMT_VAAPI`, and `av_hwframe_transfer_data` the decoded surface
//!   into System memory. This stays inside this module — public shape
//!   (`AsyncElement`, input/output caps) doesn't change.
//! - YUV444P / 10-bit pixel formats. Mainline H.264 cameras emit YUV420P;
//!   other formats are rejected with `CapsMismatch` so the failure is loud.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use ffmpeg_next as ffmpeg;
use ffmpeg::codec::{self, Id};
use ffmpeg::format::Pixel;
use ffmpeg::frame::Video as FfVideo;
use ffmpeg::packet::Packet;
use ffmpeg::Error as FfError;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, ConfigureOutcome, Dim, FrameTiming, G2gError, HardwareError, MemoryDomain,
    OutputSink, PipelinePacket, Rate, VideoFormat,
};

/// One decoded picture, pixels already copied out of the libavcodec frame.
struct DecodedI420 {
    bytes: Box<[u8]>,
    width: u32,
    height: u32,
    pts_ns: u64,
}

pub struct FfmpegH264Dec {
    decoder: Option<ffmpeg::decoder::Video>,
    last_caps: Option<Caps>,
    configured: bool,
    emitted: u64,
}

// SAFETY: `ffmpeg::decoder::Video` wraps a raw `*mut AVCodecContext` and is
// `!Send` by default. The multi-thread runner requires `Send` so it can move
// the element between worker tasks. We uphold that by construction: the
// runner drives the element through `&mut self` (never concurrently), so the
// context is owned and moved, never aliased.
unsafe impl Send for FfmpegH264Dec {}

impl core::fmt::Debug for FfmpegH264Dec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FfmpegH264Dec")
            .field("configured", &self.configured)
            .field("emitted", &self.emitted)
            .finish()
    }
}

impl Default for FfmpegH264Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl FfmpegH264Dec {
    pub fn new() -> Self {
        Self {
            decoder: None,
            last_caps: None,
            configured: false,
            emitted: 0,
        }
    }

    /// Count of decoded `DataFrame`s pushed downstream. Useful in tests.
    pub fn decoded_count(&self) -> u64 {
        self.emitted
    }

    /// Send one access unit to the decoder and drain whatever it is ready
    /// to release. libavcodec buffers for B-frame reordering, so early
    /// inputs commonly yield zero outputs.
    fn feed_access_unit(
        &mut self,
        bitstream: &[u8],
        pts_ns: u64,
        decoded: &mut Vec<DecodedI420>,
    ) -> Result<(), G2gError> {
        let mut packet = Packet::copy(bitstream);
        // libavcodec uses the packet's PTS verbatim; the unit is opaque to
        // the codec layer and is echoed back on the decoded frame. We feed
        // nanoseconds straight through.
        packet.set_pts(Some(pts_ns as i64));
        packet.set_dts(Some(pts_ns as i64));

        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
        decoder
            .send_packet(&packet)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.drain_frames(decoded)
    }

    fn drain_frames(&mut self, decoded: &mut Vec<DecodedI420>) -> Result<(), G2gError> {
        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
        let mut frame = FfVideo::empty();
        loop {
            match decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    let bytes = copy_i420(&frame)?;
                    // libavcodec returns the PTS we fed in (or AV_NOPTS_VALUE
                    // = INT64_MIN if it could not propagate one); treat the
                    // sentinel as zero so we don't return a wild timestamp.
                    let pts_ns = match frame.pts() {
                        Some(p) if p >= 0 => p as u64,
                        _ => 0,
                    };
                    decoded.push(DecodedI420 {
                        bytes,
                        width: frame.width(),
                        height: frame.height(),
                        pts_ns,
                    });
                }
                Err(FfError::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                    // Need more input.
                    return Ok(());
                }
                Err(FfError::Eof) => return Ok(()),
                Err(_) => return Err(G2gError::Hardware(HardwareError::Other)),
            }
        }
    }

    fn drain_eos(&mut self, decoded: &mut Vec<DecodedI420>) -> Result<(), G2gError> {
        if let Some(d) = self.decoder.as_mut() {
            d.send_eof()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        }
        self.drain_frames(decoded)
    }
}

impl AsyncElement for FfmpegH264Dec {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::Video {
                format: VideoFormat::H264,
                ..
            } => {}
            _ => return Err(G2gError::CapsMismatch),
        }

        // ffmpeg::init() registers codecs once per process; calling it
        // repeatedly is safe and cheap.
        ffmpeg::init().map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        let codec = codec::decoder::find(Id::H264).ok_or(G2gError::Hardware(HardwareError::Other))?;
        let decoder = codec::decoder::new()
            .open_as(codec)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            .video()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.decoder = Some(decoder);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut decoded = Vec::new();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed_access_unit(slice.as_slice(), frame.timing.pts_ns, &mut decoded)?;
                }
                PipelinePacket::CapsChanged(_) => {
                    // Upstream H.264 caps are swallowed; we emit I420
                    // CapsChanged from the decoder's first decoded frame and
                    // again on geometry changes.
                }
                PipelinePacket::Flush => {
                    if let Some(d) = self.decoder.as_mut() {
                        d.flush();
                    }
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    self.drain_eos(&mut decoded)?;
                }
            }

            for d in decoded {
                let new_caps = i420_caps(d.width, d.height);
                if self.last_caps.as_ref() != Some(&new_caps) {
                    out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                    self.last_caps = Some(new_caps.clone());
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(d.bytes)),
                    caps: new_caps,
                    timing: FrameTiming {
                        pts_ns: d.pts_ns,
                        dts_ns: d.pts_ns,
                        duration_ns: 0,
                        capture_ns: d.pts_ns,
                    },
                    sequence: self.emitted,
                };
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
        })
    }
}

fn i420_caps(w: u32, h: u32) -> Caps {
    Caps::Video {
        format: VideoFormat::I420,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

/// Copy YUV420P / I420 pixels out of a libavcodec frame into a packed
/// `width * height * 3 / 2` buffer: Y plane (w*h), then U plane (w*h/4),
/// then V plane (w*h/4). Each source plane's pitch may exceed its visible
/// width due to alignment, so rows are copied individually.
///
/// Rejects any pixel format the H.264 decoder may emit that isn't an
/// 8-bit 4:2:0 YUV layout — those streams need a `ColorConvert` element
/// upstream of any I420 consumer.
fn copy_i420(frame: &FfVideo) -> Result<Box<[u8]>, G2gError> {
    match frame.format() {
        // YUVJ420P is YUV420P with JPEG (full) range. Same plane layout, so
        // accept it; range fidelity is preserved in the pixel values and can
        // be advertised by a future colour-metadata field on `Caps::Video`.
        Pixel::YUV420P | Pixel::YUVJ420P => {}
        _ => return Err(G2gError::CapsMismatch),
    }
    if frame.planes() < 3 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let y_size = w * h;
    let c_size = cw * ch;
    let total = y_size + 2 * c_size;

    let mut out = alloc::vec![0u8; total];

    // Y plane (full resolution).
    let y_src = frame.data(0);
    let y_pitch = frame.stride(0);
    for row in 0..h {
        let src_off = row * y_pitch;
        let dst_off = row * w;
        out[dst_off..dst_off + w].copy_from_slice(&y_src[src_off..src_off + w]);
    }
    // U plane (half resolution).
    let u_src = frame.data(1);
    let u_pitch = frame.stride(1);
    for row in 0..ch {
        let src_off = row * u_pitch;
        let dst_off = y_size + row * cw;
        out[dst_off..dst_off + cw].copy_from_slice(&u_src[src_off..src_off + cw]);
    }
    // V plane (half resolution).
    let v_src = frame.data(2);
    let v_pitch = frame.stride(2);
    for row in 0..ch {
        let src_off = row * v_pitch;
        let dst_off = y_size + c_size + row * cw;
        out[dst_off..dst_off + cw].copy_from_slice(&v_src[src_off..src_off + cw]);
    }

    Ok(out.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i420_caps_are_fixed() {
        assert_eq!(
            i420_caps(640, 480),
            Caps::Video {
                format: VideoFormat::I420,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Any,
            }
        );
    }

    #[test]
    fn intercept_rejects_non_h264() {
        let dec = FfmpegH264Dec::new();
        let vp9 = Caps::Video {
            format: VideoFormat::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(dec.intercept_caps(&vp9), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn intercept_narrows_h264_geometry() {
        let dec = FfmpegH264Dec::new();
        let proposal = Caps::Video {
            format: VideoFormat::H264,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
        };
        assert_eq!(dec.intercept_caps(&proposal), Ok(proposal));
    }

    #[test]
    fn unconfigured_decoder_reports_zero_decoded() {
        let dec = FfmpegH264Dec::new();
        assert_eq!(dec.decoded_count(), 0);
    }
}
