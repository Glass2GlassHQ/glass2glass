//! Linux H.264 video *encode* element using ffmpeg / libavcodec: the
//! encode-side mirror of [`crate::ffmpegdec::FfmpegVideoDec`]. `RawVideo` (see
//! the input formats below) in, `CompressedVideo{H264}` Annex-B out, so the
//! Linux production path finally
//! has a hardware H.264 encoder, the codec `WebRtcSink` / `RtpH264Packetizer` /
//! the RTSP server need (the existing software encoders are AV1 / VP8/9 / MJPEG).
//!
//! Two backends, selected at construction, differing only in the libavcodec
//! encoder opened (the `AsyncElement` / caps shape is identical):
//!
//! - [`Backend::Nvenc`] (default): NVIDIA NVENC via the `h264_nvenc` encoder,
//!   hardware-fast and realtime on any NVENC-capable GPU. The server-side
//!   render-and-stream path (Bevy -> g2g) wants this. Requires the libavcodec
//!   build to include `h264_nvenc` (check `ffmpeg -encoders | grep nvenc`) and a
//!   working NVIDIA driver at runtime; `configure_pipeline` fails loud
//!   (`HardwareError::Other`) otherwise so the caller can fall back to software.
//! - [`Backend::Software`]: libx264 (`libx264`), the portable CPU encoder for
//!   hosts without an NVIDIA GPU (CI, laptops). Present only if libavcodec was
//!   built `--enable-libx264`.
//!
//! Tuned for low latency: no B-frames (`max_b_frames = 0`, so output is in
//! presentation order, no reorder delay), in-band SPS/PPS (the `GLOBAL_HEADER`
//! flag is *not* set, so parameter sets ride on each IDR, the Annex-B elementary
//! stream a network sink expects), and a per-backend low-latency preset/tune. A
//! downstream keyframe request (`Reconfigure::ForceKeyframe`, a WebRTC PLI)
//! forces an IDR on the next frame via the picture type.
//!
//! Threading: `ffmpeg::encoder::Encoder` wraps a raw `*mut AVCodecContext`, which
//! is `!Send`. The runner moves the element between worker threads but never
//! shares it (`&mut self` only, never concurrently), so `unsafe impl Send` is
//! sound on the ownership-transfer grounds documented on `FfmpegVideoDec` /
//! `MfDecode`.
//!
//! Runtime bitrate retarget: a libavcodec encoder fixes its rate at
//! open, so a downstream BWE target reopens the encoder (hysteresis-gated 20%,
//! old encoder flushed first so nothing in flight is lost, fresh encoder starts
//! on an IDR). Setting the `bitrate` property mid-stream reopens the same way,
//! ungated: an explicit set is intent, not an estimate. A target of 0 is the
//! shed-layer idle hint: frames are skipped unencoded except a sparse 1-in-32
//! keep-alive that keeps the reverse-signal path alive for the resume.
//!
//! Input formats: `I420`, `NV12` (so an NV12-emitting decoder or capture
//! source feeds the encoder without a `videoconvert`), and 10-bit `I420_10LE`
//! (High 10 profile). The pixel format is checked against the encoder's
//! advertised list at configure, so a 10-bit input on an 8-bit-only libx264 or
//! on NVENC (whose H.264 encoder is 8-bit only, whatever the shared pixel-format
//! list claims) fails negotiation loud rather than encoding garbage.
//!
//! Colorimetry: the negotiated input's colour description goes onto the
//! encoder context before open, so libx264 and NVENC write it into the SPS VUI,
//! and onto the H.264 output caps. An untagged input stays untagged.
//!
//! Known driver issue: two NVENC instances in one process crash intermittently
//! inside a libnvcuvid worker thread under concurrent load (observed on driver
//! 580.173.02 / ffmpeg 7.1.5, same faulting instruction each time; single
//! instances are stable, and the same double-instance sequence is stable when
//! run serially). Multi-encoder graphs (simulcast fan) should prefer
//! [`Backend::Software`] until the driver is fixed.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use ffmpeg::codec::encoder::video::Encoder as VideoEncoder;
use ffmpeg::format::Pixel;
use ffmpeg::frame::Video as FfVideo;
use ffmpeg::packet::Packet;
use ffmpeg::Dictionary;
use ffmpeg::Error as FfError;
use ffmpeg::Rational;
use ffmpeg_next as ffmpeg;

use crate::ffmpegdec::{open_hw_device, AvBufferRef};
use g2g_core::caps::framerate_q16_to_fraction;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ColorRange, Colorimetry, ConfigureOutcome, Dim,
    ElementMetadata, G2gError, HardwareError, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

/// Default constant target bitrate (bits/second) when the caller sets none. 4
/// Mbps is a reasonable 1080p30 streaming default.
const DEFAULT_BITRATE_BPS: usize = 4_000_000;

/// Default GOP length (frames between IDRs) when framerate is unknown. One IDR
/// per ~2 seconds at 30 fps; a network sink also forces IDRs on demand (PLI).
const DEFAULT_GOP: u32 = 60;

const DEFAULT_FRAMERATE: (u32, u32) = (30, 1);

// gst x264enc's `pass` nicks, for the two modes this element applies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pass {
    Cbr,
    // quantizer 0 on libx264 is lossless
    Quant,
}

/// libavcodec H.264 encoder backend. The element shape (raw video in, H.264
/// Annex-B out) is identical; only the encoder opened changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA NVENC (`h264_nvenc`). Hardware, realtime. Default.
    Nvenc,
    /// libx264 software encoder (`libx264`). Portable CPU fallback.
    Software,
    /// VA-API on a DRM render node (`h264_vaapi` / `hevc_vaapi` / `av1_vaapi`).
    /// Frames are uploaded from system memory into VA surfaces.
    Vaapi(VaapiCodec),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaapiCodec {
    H264,
    H265,
    Av1,
}

impl VaapiCodec {
    // gst names the constant quantizer `qpp` on vah264enc / vah265enc and `qp` on vaav1enc
    fn quantizer_property(self) -> &'static str {
        match self {
            VaapiCodec::H264 | VaapiCodec::H265 => "qpp",
            VaapiCodec::Av1 => "qp",
        }
    }
}

impl Backend {
    /// The libavcodec encoder name to look up.
    fn encoder_name(self) -> &'static str {
        match self {
            Backend::Nvenc => "h264_nvenc",
            Backend::Software => "libx264",
            Backend::Vaapi(VaapiCodec::H264) => "h264_vaapi",
            Backend::Vaapi(VaapiCodec::H265) => "hevc_vaapi",
            Backend::Vaapi(VaapiCodec::Av1) => "av1_vaapi",
        }
    }

    fn codec(self) -> VideoCodec {
        match self {
            Backend::Nvenc | Backend::Software | Backend::Vaapi(VaapiCodec::H264) => {
                VideoCodec::H264
            }
            Backend::Vaapi(VaapiCodec::H265) => VideoCodec::H265,
            Backend::Vaapi(VaapiCodec::Av1) => VideoCodec::Av1,
        }
    }

    // the VA-API formats match gst's vah264enc / vah265enc / vaav1enc sink pads
    fn input_formats(self) -> &'static [RawVideoFormat] {
        match self {
            Backend::Nvenc | Backend::Software => &[
                RawVideoFormat::I420,
                RawVideoFormat::Nv12,
                RawVideoFormat::I420p10,
            ],
            Backend::Vaapi(VaapiCodec::H264 | VaapiCodec::Av1) => &[RawVideoFormat::Nv12],
            Backend::Vaapi(VaapiCodec::H265) => &[RawVideoFormat::Nv12, RawVideoFormat::P010],
        }
    }

    // gst x264enc `quantizer`, vah26xenc `qpp` and vaav1enc `qp` defaults and upper bounds
    fn default_quantizer(self) -> u32 {
        match self {
            Backend::Nvenc | Backend::Software => 21,
            Backend::Vaapi(VaapiCodec::H264 | VaapiCodec::H265) => 26,
            Backend::Vaapi(VaapiCodec::Av1) => 128,
        }
    }

    fn max_quantizer(self) -> u32 {
        match self {
            Backend::Nvenc | Backend::Software => 50,
            Backend::Vaapi(VaapiCodec::H264 | VaapiCodec::H265) => 51,
            Backend::Vaapi(VaapiCodec::Av1) => 255,
        }
    }
}

/// The libavcodec pixel format for a supported input format, or `None` if the
/// element does not take that layout. Also fixes the per-plane copy geometry
/// (see [`plane_layout`]).
fn pixel_for(format: RawVideoFormat) -> Option<Pixel> {
    Some(match format {
        RawVideoFormat::I420 => Pixel::YUV420P,
        RawVideoFormat::Nv12 => Pixel::NV12,
        RawVideoFormat::I420p10 => Pixel::YUV420P10LE,
        RawVideoFormat::P010 => Pixel::P010LE,
        _ => return None,
    })
}

/// Plane layout of an input frame as (bytes per row, rows) per libavcodec plane,
/// in the order the planes are packed in the incoming buffer. NV12 is
/// semi-planar (one interleaved chroma plane of `2 * cw` bytes per row); the
/// 10-bit format stores each sample in a little-endian 2-byte word.
fn plane_layout(format: RawVideoFormat, w: usize, h: usize) -> Option<Vec<(usize, usize)>> {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    Some(match format {
        RawVideoFormat::I420 => Vec::from([(w, h), (cw, ch), (cw, ch)]),
        RawVideoFormat::Nv12 => Vec::from([(w, h), (cw * 2, ch)]),
        RawVideoFormat::I420p10 => Vec::from([(w * 2, h), (cw * 2, ch), (cw * 2, ch)]),
        RawVideoFormat::P010 => Vec::from([(w * 2, h), (cw * 4, ch)]),
        _ => return None,
    })
}

/// Write the negotiated colorimetry onto the encoder context, which libx264 and
/// NVENC copy into the SPS VUI colour description. libavcodec's colour enums are
/// the CICP codepoints (H.273), so each field goes in as the codepoint the caps
/// value names; an `Unknown` field passes CICP 2 (unspecified), leaving the
/// context at its default so nothing is written.
fn set_color_description(
    video: &mut ffmpeg::encoder::video::Video,
    colorimetry: Colorimetry,
) -> Result<(), G2gError> {
    video.set_color_range(match colorimetry.range {
        ColorRange::Full => ffmpeg::color::Range::JPEG,
        ColorRange::Limited => ffmpeg::color::Range::MPEG,
        _ => ffmpeg::color::Range::Unspecified,
    });
    for (option, codepoint) in [
        (c"color_primaries", colorimetry.primaries.to_cicp()),
        (c"color_trc", colorimetry.transfer.to_cicp()),
        (c"colorspace", colorimetry.matrix.to_cicp()),
    ] {
        // SAFETY: `video` owns a live AVCodecContext that is not yet open, and
        // each name is one of its int options.
        let set = unsafe {
            ffmpeg::ffi::av_opt_set_int(
                video.as_mut_ptr().cast(),
                option.as_ptr(),
                codepoint as i64,
                0,
            )
        };
        if set < 0 {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
    }
    Ok(())
}

/// Encodes raw I420 / NV12 / I420_10LE video into an H.264 Annex-B elementary
/// stream.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::ffmpegenc::{Backend, FfmpegVideoEnc};
///
/// let encoder = FfmpegVideoEnc::new()
///     .with_backend(Backend::Software)
///     .with_bitrate(4_000_000);
/// ```
pub struct FfmpegVideoEnc {
    backend: Backend,
    width: u32,
    height: u32,
    /// The negotiated input format; fixes the libavcodec pixel format the
    /// encoder opens with and the per-frame plane copy.
    format: RawVideoFormat,
    framerate: Rate,
    /// The negotiated input colour description, written into the VUI and onto
    /// the output caps. `UNKNOWN` for an untagged input: nothing is tagged.
    colorimetry: Colorimetry,
    /// Target constant bitrate (bits/second).
    bitrate_bps: usize,
    pass: Pass,
    quantizer: u32,
    gop: u32,
    // VA-API render node; `None` lets libva pick
    device_path: Option<String>,
    // the VA surface pool frames are uploaded into, VA-API only
    vaapi_frames: Option<AvBufferRef>,
    /// The opened video encoder. Derefs to the base `Encoder` for
    /// `send_frame` / `receive_packet`.
    encoder: Option<VideoEncoder>,
    /// The AVFrame each input is copied into, reused across frames of one
    /// encoder session.
    input_frame: Option<FfVideo>,
    /// Source PTS per input frame number, indexed by the frame counter we stamp
    /// as the encoder PTS. With `max_b_frames = 0` output is in order, but the
    /// map survives any reorder and recovers the original nanosecond PTS. Keyed
    /// by frame number and drained on output, so it stays bounded by the
    /// encoder's lookahead instead of growing for the stream lifetime.
    pts_by_frameno: alloc::collections::BTreeMap<u64, u64>,
    /// Monotonic input frame counter, stamped as each frame's encoder PTS (in
    /// `time_base` units) and used as the key into `pts_by_frameno`.
    frame_no: i64,
    emitted: u64,
    caps_sent: bool,
    /// A downstream PLI latched a keyframe request; the next encode forces an IDR.
    force_keyframe: bool,
    /// Shed-layer idle (`Bitrate(0)`): most frames are skipped unencoded;
    /// a sparse 1-in-32 keep-alive encode keeps the push cadence (and thus the
    /// reverse-signal path that will deliver the resume target) alive.
    idle: bool,
    /// Frame counter driving the sparse keep-alive cadence while idle.
    idle_skip: u32,
    /// Packets flushed out of the previous encoder by a bitrate retarget,
    /// waiting to lead the next emitted batch.
    pending: Vec<(Vec<u8>, u64)>,
    configured: bool,
}

// SAFETY: `ffmpeg::encoder::Encoder` wraps a raw `*mut AVCodecContext` and is
// `!Send`. The multi-thread runner moves the element between worker tasks but
// drives it through `&mut self` only (never concurrently), so the context is
// owned and moved, never aliased, the same contract upheld by `FfmpegVideoDec`.
unsafe impl Send for FfmpegVideoEnc {}

impl core::fmt::Debug for FfmpegVideoEnc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FfmpegVideoEnc")
            .field("backend", &self.backend)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("emitted", &self.emitted)
            .field("configured", &self.configured)
            .finish()
    }
}

impl Default for FfmpegVideoEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl FfmpegVideoEnc {
    pub fn new() -> Self {
        Self {
            backend: Backend::Nvenc,
            width: 0,
            height: 0,
            format: RawVideoFormat::I420,
            framerate: Rate::Any,
            colorimetry: Colorimetry::UNKNOWN,
            bitrate_bps: DEFAULT_BITRATE_BPS,
            pass: Pass::Cbr,
            quantizer: Backend::Nvenc.default_quantizer(),
            gop: DEFAULT_GOP,
            device_path: None,
            vaapi_frames: None,
            encoder: None,
            input_frame: None,
            pts_by_frameno: alloc::collections::BTreeMap::new(),
            frame_no: 0,
            emitted: 0,
            caps_sent: false,
            force_keyframe: false,
            idle: false,
            idle_skip: 0,
            pending: Vec::new(),
            configured: false,
        }
    }

    /// Select the encoder backend (default [`Backend::Nvenc`]). Resets the
    /// constant quantizer to the backend's default.
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self.quantizer = backend.default_quantizer();
        self
    }

    /// Pin the VA-API render node, e.g. `/dev/dri/renderD128`.
    pub fn with_device_path(mut self, path: &str) -> Self {
        self.device_path = Some(path.into());
        self
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Set the constant target bitrate (bits/second). Default 4 Mbps.
    pub fn with_bitrate(mut self, bps: usize) -> Self {
        self.bitrate_bps = bps.max(1);
        self
    }

    /// Count of H.264 access units emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// Any-geometry sink caps for one accepted input format.
    fn input_template(format: RawVideoFormat) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: self.backend.codec(),
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
            colorimetry: self.colorimetry,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        }
    }

    /// Frames per second as `(num, den)` from the negotiated framerate,
    /// defaulting to 30/1 when unspecified.
    fn framerate_fraction(&self) -> (u32, u32) {
        match self.framerate {
            Rate::Fixed(q16) if q16 > 0 => framerate_q16_to_fraction(q16),
            _ => DEFAULT_FRAMERATE,
        }
    }

    /// Low-latency encoder options for the active backend, applied at open via an
    /// `AVDictionary` (the `gst-launch`-equivalent of `option=value` on the
    /// element). NVENC: low-latency tuning, CBR, zero reorder delay. libx264:
    /// the `zerolatency` tune (no lookahead / no B-frames / sliced threads).
    fn open_options(&self) -> Dictionary<'static> {
        let mut opts = Dictionary::new();
        match self.backend {
            Backend::Nvenc => {
                // p1..p7 = fastest..slowest; "ll" tune = low latency. `delay=0`
                // releases each frame as soon as it is encoded (no reorder hold).
                opts.set("preset", "p4");
                opts.set("tune", "ll");
                opts.set("delay", "0");
                opts.set("zerolatency", "1");
                match self.pass {
                    Pass::Cbr => opts.set("rc", "cbr"),
                    Pass::Quant => opts.set("rc", "constqp"),
                }
            }
            Backend::Software => {
                opts.set("preset", "veryfast");
                opts.set("tune", "zerolatency");
            }
            Backend::Vaapi(_) => {
                // one frame in flight instead of two
                opts.set("async_depth", "1");
                match self.pass {
                    Pass::Cbr => opts.set("rc_mode", "CBR"),
                    Pass::Quant => opts.set("rc_mode", "CQP"),
                }
            }
        }
        if self.pass == Pass::Quant {
            // av1_vaapi has no `qp` option and reads the base q index from global_quality
            let option = match self.backend {
                Backend::Vaapi(VaapiCodec::Av1) => "global_quality",
                Backend::Nvenc | Backend::Software | Backend::Vaapi(_) => "qp",
            };
            opts.set(option, &self.quantizer.to_string());
        }
        opts
    }

    /// Build and open the libavcodec encoder on the negotiated geometry. Fails
    /// loud if the encoder is absent (libavcodec built without it, or no NVIDIA
    /// driver for NVENC) so the caller can pick another backend.
    fn open_encoder(&mut self) -> Result<(), G2gError> {
        let codec = ffmpeg::encoder::find_by_name(self.backend.encoder_name())
            .ok_or(G2gError::Hardware(HardwareError::Other))?;
        let pixel = pixel_for(self.format).ok_or(G2gError::CapsMismatch)?;
        let vaapi = matches!(self.backend, Backend::Vaapi(_));
        let encoder_pixel = if vaapi { Pixel::VAAPI } else { pixel };
        // Reject a pixel format this libavcodec build's encoder does not list
        // (an 8-bit-only libx264 against 10-bit input) here rather than letting
        // avcodec_open2 fail deeper: the caller sees the negotiation fail.
        if let Ok(video) = codec.video() {
            if let Some(mut advertised) = video.formats() {
                if !advertised.any(|f| f == encoder_pixel) {
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
            }
        }
        self.vaapi_frames = if vaapi {
            Some(self.open_vaapi_frames(pixel)?)
        } else {
            None
        };

        let (fps_num, fps_den) = self.framerate_fraction();
        let fps_num = i32::try_from(fps_num).map_err(|_| G2gError::CapsMismatch)?;
        let fps_den = i32::try_from(fps_den).map_err(|_| G2gError::CapsMismatch)?;
        // Allocate the context *with* the codec so its AVClass defaults apply.
        // A codec-less `encoder::new()` leaves the generic legacy AVCodecContext
        // defaults (`qmin=2`, `qmax=31`, `max_qdiff=3`, `qcompress=0.5`,
        // `me_range=0`), which is exactly libx264's "broken ffmpeg default
        // settings" fingerprint: it scores those fields and aborts the open at
        // score >= 5 even though we pass a `preset`. Allocating with the codec
        // gives the encoder-appropriate defaults the `ffmpeg` CLI gets.
        let mut video = ffmpeg::codec::context::Context::new_with_codec(codec)
            .encoder()
            .video()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        video.set_width(self.width);
        video.set_height(self.height);
        // The bit depth alone picks the profile: libx264 encodes 10-bit input as
        // High 10, so no explicit profile is set here.
        video.set_format(encoder_pixel);
        if let Some(frames) = &self.vaapi_frames {
            // SAFETY: `video` owns a live AVCodecContext that is not yet open,
            // and takes ownership of the new frames reference.
            unsafe { (*video.as_mut_ptr()).hw_frames_ctx = frames.new_reference()? };
        }
        // time_base = 1/fps, so a frame's PTS is just its index; frame_rate lets
        // the encoder pace its rate control.
        video.set_time_base(Rational::new(fps_den, fps_num));
        video.set_frame_rate(Some(Rational::new(fps_num, fps_den)));
        if self.pass == Pass::Cbr {
            video.set_bit_rate(self.bitrate_bps);
            video.set_max_bit_rate(self.bitrate_bps);
        }
        video.set_gop(self.gop);
        // No B-frames: output stays in presentation order (no reorder latency),
        // which the low-latency streaming path wants.
        video.set_max_b_frames(0);
        set_color_description(&mut video, self.colorimetry)?;

        let opened = video
            .open_as_with(codec, self.open_options())
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.encoder = Some(opened);
        self.input_frame = None;
        self.pts_by_frameno.clear();
        self.frame_no = 0;
        Ok(())
    }

    fn open_vaapi_frames(&self, software_pixel: Pixel) -> Result<AvBufferRef, G2gError> {
        let device = open_hw_device(
            ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
            self.device_path.as_deref(),
        )?;
        // SAFETY: `device` is a live hardware device reference.
        let frames =
            AvBufferRef::from_raw(unsafe { ffmpeg::ffi::av_hwframe_ctx_alloc(device.as_ptr()) })?;
        // SAFETY: the data of a fresh frames reference is its AVHWFramesContext,
        // filled in here before init and not shared yet.
        let initialised = unsafe {
            let context = (*frames.as_ptr()).data as *mut ffmpeg::ffi::AVHWFramesContext;
            (*context).format = ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_VAAPI;
            (*context).sw_format = software_pixel.into();
            (*context).width = i32::try_from(self.width).map_err(|_| G2gError::CapsMismatch)?;
            (*context).height = i32::try_from(self.height).map_err(|_| G2gError::CapsMismatch)?;
            (*context).initial_pool_size = VAAPI_SURFACE_POOL;
            ffmpeg::ffi::av_hwframe_ctx_init(frames.as_ptr())
        };
        if initialised < 0 {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        Ok(frames)
    }

    /// Copy a raw access unit into the session's AVFrame of the negotiated pixel
    /// format (honouring the frame's plane strides), forcing an IDR if a keyframe
    /// was requested, and drain whatever packets the encoder releases.
    fn encode(&mut self, raw: &[u8], pts_ns: u64) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let (w, h) = (self.width as usize, self.height as usize);
        let planes = plane_layout(self.format, w, h).ok_or(G2gError::CapsMismatch)?;
        let needed: usize = planes.iter().map(|(row, rows)| row * rows).sum();
        if raw.len() < needed {
            return Err(G2gError::CapsMismatch);
        }

        let pixel = pixel_for(self.format).ok_or(G2gError::CapsMismatch)?;
        let frame = self
            .input_frame
            .get_or_insert_with(|| FfVideo::new(pixel, self.width, self.height));
        // SAFETY: `frame` is an allocated AVFrame we own; buffers the encoder still
        // references from the previous frame are swapped for fresh ones first.
        if unsafe { ffmpeg::ffi::av_frame_make_writable(frame.as_mut_ptr()) } < 0 {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let mut off = 0;
        for (i, (row_bytes, rows)) in planes.iter().enumerate() {
            // Read the stride before borrowing the plane data mutably (the borrow
            // checker won't allow `data_mut` and `stride` in one call).
            let stride = frame.stride(i);
            let end = off + row_bytes * rows;
            copy_plane(frame.data_mut(i), stride, &raw[off..end], *row_bytes, *rows);
            off = end;
        }

        let mut surface = match &self.vaapi_frames {
            Some(frames) => Some(upload_to_surface(frames, frame)?),
            None => None,
        };
        let frame = surface.as_mut().unwrap_or(frame);
        let frameno = self.frame_no;
        frame.set_pts(Some(frameno));
        // Picture type I requests an IDR on this frame. NVENC, libx264 and the
        // VA-API encoders all honour `pict_type` for forced key frames.
        let pict_type = if core::mem::take(&mut self.force_keyframe) {
            ffmpeg::ffi::AVPictureType::AV_PICTURE_TYPE_I
        } else {
            ffmpeg::ffi::AVPictureType::AV_PICTURE_TYPE_NONE
        };
        // SAFETY: `frame` is a writable AVFrame we own.
        unsafe { (*frame.as_mut_ptr()).pict_type = pict_type };
        // pts_by_frameno is keyed by the frame counter we stamped as the PTS.
        self.pts_by_frameno.insert(frameno as u64, pts_ns);
        self.frame_no += 1;

        let encoder = self.encoder.as_mut().ok_or(G2gError::NotConfigured)?;
        encoder
            .send_frame(frame)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.drain()
    }

    /// Flush the encoder at EOS and return the remaining packets.
    fn flush(&mut self) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        if let Some(enc) = self.encoder.as_mut() {
            enc.send_eof()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        }
        self.drain()
    }

    /// Drain ready packets as `(annex_b_bytes, pts_ns)`, mapping the encoder PTS
    /// (the frame index we stamped) back to the source nanosecond timestamp.
    fn drain(&mut self) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let mut out = Vec::new();
        let encoder = self.encoder.as_mut().ok_or(G2gError::NotConfigured)?;
        loop {
            let mut packet = Packet::empty();
            match encoder.receive_packet(&mut packet) {
                Ok(()) => {
                    let pts_ns = match packet.pts() {
                        Some(idx) if idx >= 0 => {
                            self.pts_by_frameno.remove(&(idx as u64)).unwrap_or(0)
                        }
                        _ => 0,
                    };
                    if let Some(data) = packet.data() {
                        let data = match self.backend {
                            Backend::Vaapi(VaapiCodec::H265) => {
                                crop_padded_h265_sps(data, self.width, self.height)?
                            }
                            _ => data.to_vec(),
                        };
                        out.push((data, pts_ns));
                    }
                }
                Err(FfError::Other { errno }) if errno == ffmpeg::error::EAGAIN => break,
                Err(FfError::Eof) => break,
                Err(_) => return Err(G2gError::Hardware(HardwareError::Other)),
            }
        }
        Ok(out)
    }

    async fn emit(
        &mut self,
        packets: Vec<(Vec<u8>, u64)>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = self.output_caps();
        // Anything a retarget flushed out of the previous encoder leads the
        // batch: it is older than every packet the new encoder has produced.
        let mut batch = core::mem::take(&mut self.pending);
        batch.extend(packets);
        let feedback = crate::encoder_base::emit_packets(
            &mut self.caps_sent,
            &mut self.emitted,
            batch,
            &caps,
            out,
        )
        .await?;
        // A downstream PLI latches a forced IDR on the next encode.
        if feedback.force_keyframe {
            self.force_keyframe = true;
        }
        // Runtime bitrate retarget (M722): a libavcodec encoder fixes its rate
        // at open, so a BWE change reopens the encoder. Hysteresis-gated like
        // `Av1Enc`, since each retarget costs a keyframe. A target of 0 is the
        // shed-layer idle hint (see `PushOutcome::Bitrate`).
        // a constant quantizer encode has no rate to retarget
        let bitrate_target = feedback.bitrate_bps.filter(|_| self.pass == Pass::Cbr);
        if let Some(bps) = bitrate_target {
            if bps == 0 {
                self.idle = true;
            } else {
                if self.idle {
                    self.idle = false;
                    self.force_keyframe = true;
                }
                if crate::encoder_base::bitrate_change_is_significant(
                    self.bitrate_bps as u64,
                    bps as u64,
                ) {
                    self.retarget(bps as usize)?;
                }
            }
        }
        Ok(())
    }

    fn set_quantizer(&mut self, quantizer: u64) -> Result<(), PropError> {
        self.quantizer = u32::try_from(quantizer)
            .ok()
            .filter(|&quantizer| quantizer <= self.backend.max_quantizer())
            .ok_or(PropError::Value)?;
        self.reopen().map_err(|_| PropError::Value)
    }

    // gst vah264enc / vah265enc / vaav1enc names and units: bitrate in kbit/s
    fn set_vaapi_property(
        &mut self,
        codec: VaapiCodec,
        name: &str,
        value: PropValue,
    ) -> Result<(), PropError> {
        match name {
            "bitrate" => {
                let kbps = value.as_uint().ok_or(PropError::Type)?;
                if !(1..=MAX_VAAPI_BITRATE_KBPS).contains(&kbps) {
                    return Err(PropError::Value);
                }
                self.retarget(kbps as usize * 1000)
                    .map_err(|_| PropError::Value)
            }
            "rate-control" => {
                self.pass = match value.as_str().ok_or(PropError::Type)? {
                    "cbr" => Pass::Cbr,
                    "cqp" => Pass::Quant,
                    _ => return Err(PropError::Value),
                };
                self.reopen().map_err(|_| PropError::Value)
            }
            "key-int-max" => {
                let frames = value.as_uint().ok_or(PropError::Type)?;
                if frames > MAX_VAAPI_KEY_INT {
                    return Err(PropError::Value);
                }
                // gst reads 0 as "pick the GOP length"
                self.gop = if frames == 0 {
                    DEFAULT_GOP
                } else {
                    frames as u32
                };
                self.reopen().map_err(|_| PropError::Value)
            }
            "device-path" => {
                let path = value.as_str().ok_or(PropError::Type)?;
                self.device_path = (!path.is_empty()).then(|| path.into());
                self.reopen().map_err(|_| PropError::Value)
            }
            name if name == codec.quantizer_property() => {
                let quantizer = value.as_uint().ok_or(PropError::Type)?;
                self.set_quantizer(quantizer)
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn vaapi_property(&self, codec: VaapiCodec, name: &str) -> Option<PropValue> {
        match name {
            "bitrate" => Some(PropValue::Uint(self.bitrate_bps as u64 / 1000)),
            "rate-control" => Some(PropValue::Str(
                match self.pass {
                    Pass::Cbr => "cbr",
                    Pass::Quant => "cqp",
                }
                .into(),
            )),
            "key-int-max" => Some(PropValue::Uint(u64::from(self.gop))),
            "device-path" => Some(PropValue::Str(self.device_path.clone().unwrap_or_default())),
            name if name == codec.quantizer_property() => {
                Some(PropValue::Uint(u64::from(self.quantizer)))
            }
            _ => None,
        }
    }

    /// Apply a new target bitrate. libavcodec fixes the rate at open, so a
    /// running encoder is flushed (its remaining packets are held in `pending`
    /// for the next emit rather than dropped) and reopened at the new rate; the
    /// fresh encoder starts on an IDR. Before configure this only records the
    /// rate, which the first open then picks up.
    fn retarget(&mut self, bps: usize) -> Result<(), G2gError> {
        self.bitrate_bps = bps.max(1);
        self.reopen()
    }

    fn reopen(&mut self) -> Result<(), G2gError> {
        if self.encoder.is_none() {
            return Ok(());
        }
        let drained = self.flush()?;
        self.pending.extend(drained);
        self.open_encoder()
    }
}

const H265_SPS_NAL_TYPE: u8 = 33;

// libavcodec 7.1's hevc_vaapi codes radeonsi's 16-aligned picture size with no conformance window
fn crop_padded_h265_sps(au: &[u8], width: u32, height: u32) -> Result<Vec<u8>, G2gError> {
    let has_sps = crate::annexb::nal_units(au)
        .any(|nal| crate::annexb::nalu_type(VideoCodec::H265, nal) == H265_SPS_NAL_TYPE);
    if !has_sps {
        return Ok(au.to_vec());
    }
    let mut rebuilt = Vec::with_capacity(au.len() + 8);
    for nal in crate::annexb::nal_units(au) {
        rebuilt.extend_from_slice(&[0, 0, 0, 1]);
        if crate::annexb::nalu_type(VideoCodec::H265, nal) == H265_SPS_NAL_TYPE {
            rebuilt.extend_from_slice(&crop_h265_sps(nal, width, height)?);
        } else {
            rebuilt.extend_from_slice(nal);
        }
    }
    Ok(rebuilt)
}

const H265_NAL_HEADER_BYTES: usize = 2;

// adds a conformance window cropping the coded picture to width x height
fn crop_h265_sps(nal: &[u8], width: u32, height: u32) -> Result<Vec<u8>, G2gError> {
    let malformed = || G2gError::Hardware(HardwareError::Other);
    let header = nal.get(..H265_NAL_HEADER_BYTES).ok_or_else(malformed)?;
    let rbsp = crate::annexb::strip_emulation_prevention(&nal[H265_NAL_HEADER_BYTES..]);
    let (head, reader) = crate::h265parse::parse_h265_sps_head(&rbsp).ok_or_else(malformed)?;
    let (coded_width, coded_height) = (
        head.pic_width_in_luma_samples,
        head.pic_height_in_luma_samples,
    );
    if head.conformance_window_flag || (coded_width, coded_height) == (width, height) {
        return Ok(nal.to_vec());
    }
    let (chroma_step_x, chroma_step_y) = match head.chroma_format_idc {
        1 => (2, 2),
        2 => (2, 1),
        _ => (1, 1),
    };
    let right = coded_width.checked_sub(width).ok_or_else(malformed)?;
    let bottom = coded_height.checked_sub(height).ok_or_else(malformed)?;
    if right % chroma_step_x != 0 || bottom % chroma_step_y != 0 {
        return Err(malformed());
    }
    // the reader stopped just past the zero conformance_window_flag
    let flag_bit = reader.bit_pos() - 1;
    let stop_bit = rbsp_stop_bit(&rbsp).ok_or_else(malformed)?;

    let mut source = crate::annexb::BitReader::new(&rbsp);
    let mut writer = crate::annexb::BitWriter::default();
    for _ in 0..flag_bit {
        writer.write_bit(source.read_bit().ok_or_else(malformed)?);
    }
    source.read_bit();
    writer.write_bit(1);
    for offset in [0, right / chroma_step_x, 0, bottom / chroma_step_y] {
        writer.write_ue(offset);
    }
    for _ in flag_bit + 1..stop_bit {
        writer.write_bit(source.read_bit().ok_or_else(malformed)?);
    }
    writer.write_bit(1);
    writer.align_to_byte();

    let mut cropped = header.to_vec();
    cropped.extend(crate::annexb::add_emulation_prevention(
        &writer.into_bytes(),
    ));
    Ok(cropped)
}

// position of rbsp_stop_one_bit, the last set bit of the RBSP
fn rbsp_stop_bit(rbsp: &[u8]) -> Option<usize> {
    let last = rbsp.iter().rposition(|&byte| byte != 0)?;
    Some(last * 8 + 7 - rbsp[last].trailing_zeros() as usize)
}

// surfaces for in-flight and reference frames, the size ffmpeg's vaapi_encode example uses
const VAAPI_SURFACE_POOL: i32 = 20;

fn upload_to_surface(frames: &AvBufferRef, software: &FfVideo) -> Result<FfVideo, G2gError> {
    let mut surface = FfVideo::empty();
    // SAFETY: `frames` is an initialised VA-API frames context and `surface` an
    // empty AVFrame we own.
    if unsafe { ffmpeg::ffi::av_hwframe_get_buffer(frames.as_ptr(), surface.as_mut_ptr(), 0) } < 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    // SAFETY: both frames are allocated, the surface with the software frame's
    // geometry and pixel format as its frames context's sw_format.
    if unsafe { ffmpeg::ffi::av_hwframe_transfer_data(surface.as_mut_ptr(), software.as_ptr(), 0) }
        < 0
    {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    Ok(surface)
}

/// Copy `src` (tightly packed `w` bytes per row, `h` rows) into a libavcodec
/// plane whose row pitch is `stride` (>= `w`, alignment padding at the end).
fn copy_plane(dst: &mut [u8], stride: usize, src: &[u8], w: usize, h: usize) {
    for row in 0..h {
        let s = &src[row * w..row * w + w];
        let d = &mut dst[row * stride..row * stride + w];
        d.copy_from_slice(s);
    }
}

impl AsyncElement for FfmpegVideoEnc {
    // M759: a re-encode to a compressed codec drops pixel-derived meta
    // (AnalyticsMeta); opaque side-data (BlobMeta) rides through.
    #[cfg(feature = "metadata")]
    fn meta_transform(&self) -> Option<g2g_core::meta::Transform> {
        Some(g2g_core::meta::Transform::Encode)
    }

    fn handles_keyframe_requests(&self) -> bool {
        true
    }

    fn handles_bitrate_requests(&self) -> bool {
        true
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Accept any supported raw format, narrowing only geometry; the encoder
        // opens with the matching libavcodec pixel format.
        if let Caps::RawVideo { format, .. } = upstream_caps {
            if self.backend.input_formats().contains(format) {
                return upstream_caps.intersect(&Self::input_template(*format));
            }
        }
        Err(G2gError::CapsMismatch)
    }

    /// Native `DerivedOutput`: a supported raw format (any geometry) in, H.264 at
    /// the same dims and framerate out. Anything else yields an empty set,
    /// rejected at solve.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let formats = self.backend.input_formats();
        let codec = self.backend.codec();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| match input {
            Caps::RawVideo {
                format,
                width,
                height,
                framerate,
                colorimetry,
                ..
            } if formats.contains(format) => CapsSet::one(Caps::CompressedVideo {
                codec,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
                colorimetry: *colorimetry,
                alignment: g2g_core::BitstreamAlignment::Unspecified,
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format,
            width,
            height,
            framerate,
            colorimetry,
            ..
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        if !self.backend.input_formats().contains(format) {
            return Err(G2gError::CapsMismatch);
        }
        let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) else {
            return Err(G2gError::CapsMismatch);
        };
        // ffmpeg::init() registers codecs once per process; safe to repeat.
        ffmpeg::init().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.width = *w;
        self.height = *h;
        self.format = *format;
        self.framerate = framerate.clone();
        self.colorimetry = *colorimetry;
        self.open_encoder()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        let (name, description) = match self.backend {
            Backend::Nvenc | Backend::Software => (
                "FFmpeg H.264 encoder",
                "Encodes raw I420 / NV12 / I420_10LE video to H.264 Annex-B via libavcodec (NVENC / libx264)",
            ),
            Backend::Vaapi(VaapiCodec::H264) => (
                "VA-API H.264 encoder",
                "Encodes raw NV12 video to H.264 Annex-B on a VA-API GPU via libavcodec",
            ),
            Backend::Vaapi(VaapiCodec::H265) => (
                "VA-API H.265 encoder",
                "Encodes raw NV12 / P010 video to H.265 Annex-B on a VA-API GPU via libavcodec",
            ),
            Backend::Vaapi(VaapiCodec::Av1) => (
                "VA-API AV1 encoder",
                "Encodes raw NV12 video to an AV1 OBU stream on a VA-API GPU via libavcodec",
            ),
        };
        ElementMetadata::new(name, "Codec/Encoder/Video", description, "g2g")
    }

    fn properties(&self) -> &'static [PropertySpec] {
        match self.backend {
            Backend::Nvenc | Backend::Software => FFMPEGENC_PROPS,
            Backend::Vaapi(VaapiCodec::H264 | VaapiCodec::H265) => VAAPI_H26X_PROPS,
            Backend::Vaapi(VaapiCodec::Av1) => VAAPI_AV1_PROPS,
        }
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        if let Backend::Vaapi(codec) = self.backend {
            return self.set_vaapi_property(codec, name, value);
        }
        match name {
            "backend" => {
                self.backend = match value.as_str().ok_or(PropError::Type)? {
                    "nvenc" | "nvenc-h264" | "h264_nvenc" => Backend::Nvenc,
                    "software" | "libx264" | "x264" => Backend::Software,
                    _ => return Err(PropError::Value),
                };
                Ok(())
            }
            "bitrate" => {
                // bits per second. An explicit set is intent rather than an
                // estimate, so it retargets a running encoder without the
                // hysteresis gate the BWE path uses.
                let bps = value.as_uint().ok_or(PropError::Type)?;
                self.retarget(bps as usize).map_err(|_| PropError::Value)?;
                Ok(())
            }
            "pass" => {
                self.pass = match value.as_str().ok_or(PropError::Type)? {
                    "cbr" => Pass::Cbr,
                    "quant" => Pass::Quant,
                    _ => return Err(PropError::Value),
                };
                self.reopen().map_err(|_| PropError::Value)
            }
            "quantizer" => {
                let quantizer = value.as_uint().ok_or(PropError::Type)?;
                self.set_quantizer(quantizer)
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        if let Backend::Vaapi(codec) = self.backend {
            return self.vaapi_property(codec, name);
        }
        match name {
            "backend" => Some(PropValue::Str(
                match self.backend {
                    Backend::Nvenc => "nvenc",
                    Backend::Software => "software",
                    Backend::Vaapi(_) => return None,
                }
                .into(),
            )),
            "bitrate" => Some(PropValue::Uint(self.bitrate_bps as u64)),
            "pass" => Some(PropValue::Str(
                match self.pass {
                    Pass::Cbr => "cbr",
                    Pass::Quant => "quant",
                }
                .into(),
            )),
            "quantizer" => Some(PropValue::Uint(u64::from(self.quantizer))),
            _ => None,
        }
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    // Shed-layer idle: skip the encode for most frames, keeping
                    // a sparse keep-alive cadence so the resume signal (which
                    // rides push outcomes) can still reach this element.
                    if self.idle {
                        self.idle_skip = self.idle_skip.wrapping_add(1);
                        if self.idle_skip % 32 != 1 {
                            return Ok(());
                        }
                    }
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let packets = self.encode(slice, frame.timing.pts_ns)?;
                    self.emit(packets, out).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the encoder; the runner's transform arm forwards EOS.
                    let packets = self.flush()?;
                    self.emit(packets, out).await?;
                }
                // an encoder drained at Eos refuses frames, and the stream after a flush starts over
                PipelinePacket::Flush => {
                    if self.encoder.is_some() {
                        self.open_encoder()?;
                    }
                    self.pending.clear();
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl FfmpegVideoEnc {
    pub fn backend_pad_templates(backend: Backend) -> Vec<PadTemplate> {
        let out = Caps::CompressedVideo {
            codec: backend.codec(),
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        let sink = CapsSet::from_alternatives(
            backend
                .input_formats()
                .iter()
                .map(|&format| Self::input_template(format))
                .collect(),
        );
        Vec::from([
            PadTemplate::sink(sink),
            PadTemplate::source(CapsSet::one(out)),
        ])
    }
}

impl PadTemplates for FfmpegVideoEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        Self::backend_pad_templates(Backend::Nvenc)
    }
}

/// Settable properties: backend (nvenc | software), rate control and its target, so a
/// `gst-launch` line can pick the encoder and rate without the builder.
static FFMPEGENC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "backend",
        PropKind::Str,
        "h264 encoder: nvenc | software (libx264)",
    )
    // Every spelling `set_property` accepts, so the launch parser can reject an
    // unknown one by name and list these.
    .with_enum_values("nvenc | nvenc-h264 | h264_nvenc | software | libx264 | x264")
    .with_default("nvenc"),
    PropertySpec::new(
        "bitrate",
        PropKind::Uint,
        "constant target bitrate, bits/second, under pass=cbr",
    ),
    PropertySpec::new("pass", PropKind::Str, "rate control: cbr | quant")
        .with_enum_values("cbr | quant")
        .with_default("cbr"),
    PropertySpec::new(
        "quantizer",
        PropKind::Uint,
        "constant quantizer under pass=quant, 0 is lossless on libx264",
    )
    .with_default("21")
    .with_range("0", "50"),
];

// gst vah264enc's bitrate and key-int-max upper bounds
const MAX_VAAPI_BITRATE_KBPS: u64 = 2_048_000;
const MAX_VAAPI_KEY_INT: u64 = 1024;

const VAAPI_BITRATE_PROP: PropertySpec = PropertySpec::new(
    "bitrate",
    PropKind::Uint,
    "target bitrate in kbit/s under rate-control=cbr",
)
.with_default("4000")
.with_range("1", "2048000");
const VAAPI_RATE_CONTROL_PROP: PropertySpec =
    PropertySpec::new("rate-control", PropKind::Str, "rate control: cbr | cqp")
        .with_enum_values("cbr | cqp")
        .with_default("cbr");
const VAAPI_KEY_INT_MAX_PROP: PropertySpec = PropertySpec::new(
    "key-int-max",
    PropKind::Uint,
    "frames between keyframes, 0 picks the default of 60",
)
.with_default("60")
.with_range("0", "1024");
const VAAPI_DEVICE_PATH_PROP: PropertySpec = PropertySpec::new(
    "device-path",
    PropKind::Str,
    "DRM render node, e.g. /dev/dri/renderD128 (empty = libva default)",
);

static VAAPI_H26X_PROPS: &[PropertySpec] = &[
    VAAPI_BITRATE_PROP,
    VAAPI_RATE_CONTROL_PROP,
    PropertySpec::new(
        "qpp",
        PropKind::Uint,
        "constant quantizer under rate-control=cqp",
    )
    .with_default("26")
    .with_range("0", "51"),
    VAAPI_KEY_INT_MAX_PROP,
    VAAPI_DEVICE_PATH_PROP,
];

static VAAPI_AV1_PROPS: &[PropertySpec] = &[
    VAAPI_BITRATE_PROP,
    VAAPI_RATE_CONTROL_PROP,
    PropertySpec::new(
        "qp",
        PropKind::Uint,
        "constant base q index under rate-control=cqp",
    )
    .with_default("128")
    .with_range("0", "255"),
    VAAPI_KEY_INT_MAX_PROP,
    VAAPI_DEVICE_PATH_PROP,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffmpegdec::FfmpegVideoDec;
    use g2g_core::frame::Frame;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::{FrameTiming, PushOutcome};

    const W: u32 = 320;
    const H: u32 = 240;

    /// Flat chroma in every test frame, so a decoded round trip can check the
    /// chroma planes landed where the input layout said they were.
    const U_VAL: u8 = 110;
    const V_VAL: u8 = 150;

    /// A moving test pattern so successive frames differ (a flat image would let
    /// the encoder emit near-empty inter frames and weaken the round-trip check).
    fn luma(seq: u64) -> Vec<u8> {
        let (w, h) = (W as usize, H as usize);
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                v.push(((x + y + seq as usize * 7) & 0xff) as u8);
            }
        }
        v
    }

    fn chroma_dims() -> (usize, usize) {
        ((W as usize).div_ceil(2), (H as usize).div_ceil(2))
    }

    fn i420_frame(seq: u64) -> Vec<u8> {
        let (cw, ch) = chroma_dims();
        let mut v = luma(seq);
        v.extend(core::iter::repeat_n(U_VAL, cw * ch));
        v.extend(core::iter::repeat_n(V_VAL, cw * ch));
        v
    }

    /// Semi-planar: one luma plane, then interleaved U/V samples.
    fn nv12_frame(seq: u64) -> Vec<u8> {
        let (cw, ch) = chroma_dims();
        let mut v = luma(seq);
        for _ in 0..cw * ch {
            v.push(U_VAL);
            v.push(V_VAL);
        }
        v
    }

    /// Planar 10-bit: each 8-bit test sample scaled into a little-endian 2-byte
    /// word (`<< 2`, the 8-to-10-bit shift).
    fn i420p10_frame(seq: u64) -> Vec<u8> {
        let (cw, ch) = chroma_dims();
        let mut v = Vec::new();
        let widen = |v: &mut Vec<u8>, s: u8| v.extend_from_slice(&((s as u16) << 2).to_le_bytes());
        for s in luma(seq) {
            widen(&mut v, s);
        }
        for _ in 0..cw * ch {
            widen(&mut v, U_VAL);
        }
        for _ in 0..cw * ch {
            widen(&mut v, V_VAL);
        }
        v
    }

    fn raw_caps(format: RawVideoFormat, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    fn i420_caps(w: u32, h: u32) -> Caps {
        raw_caps(RawVideoFormat::I420, w, h)
    }

    #[derive(Default)]
    struct CaptureSink {
        caps: Vec<Caps>,
        frames: Vec<Vec<u8>>,
    }
    impl OutputSink for CaptureSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                match packet {
                    PipelinePacket::CapsChanged(c) => self.caps.push(c),
                    PipelinePacket::DataFrame(f) => {
                        if let Some(s) = f.domain.as_system_slice() {
                            self.frames.push(s.to_vec());
                        }
                    }
                    _ => {}
                }
                Ok(PushOutcome::Accepted)
            })
        }
    }

    /// Encode 10 frames + EOS through `backend`. Returns `None` if the encoder is
    /// not available on this host (no NVIDIA driver / libavcodec built without it),
    /// so the test skips rather than failing on a machine that can't run it.
    async fn encode_with(backend: Backend) -> Option<CaptureSink> {
        let mut enc = FfmpegVideoEnc::new().with_backend(backend);
        if enc.configure_pipeline(&i420_caps(W, H)).is_err() {
            return None; // encoder absent on this host
        }
        let mut sink = CaptureSink::default();
        for i in 0..10u64 {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(i420_frame(i).into_boxed_slice())),
                FrameTiming {
                    pts_ns: i * 33_000_000,
                    ..FrameTiming::default()
                },
                i,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .ok()?;
        }
        enc.process(PipelinePacket::Eos, &mut sink).await.ok()?;
        Some(sink)
    }

    #[tokio::test]
    async fn encodes_again_after_eos_and_a_flush() {
        const FRAMES: u64 = 10;
        let mut enc = FfmpegVideoEnc::new().with_backend(Backend::Software);
        if enc.configure_pipeline(&i420_caps(W, H)).is_err() {
            std::eprintln!("skipping: libx264 not available");
            return;
        }
        let mut sink = CaptureSink::default();
        for pass in 0..2 {
            if pass > 0 {
                enc.process(PipelinePacket::Flush, &mut sink).await.unwrap();
            }
            for i in 0..FRAMES {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(i420_frame(i).into_boxed_slice())),
                    FrameTiming {
                        pts_ns: i * 33_000_000,
                        ..FrameTiming::default()
                    },
                    i,
                );
                enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                    .await
                    .unwrap();
            }
            enc.process(PipelinePacket::Eos, &mut sink).await.unwrap();
        }
        assert_eq!(
            sink.frames.len() as u64,
            2 * FRAMES,
            "the stream after the flush encodes as the first did"
        );
    }

    /// Sink that reports a fixed downstream bitrate target on every push (the
    /// WebRTC BWE shape), plus captured frames.
    struct BitrateSink {
        bps: u32,
        frames: Vec<Vec<u8>>,
    }
    impl OutputSink for BitrateSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                if let PipelinePacket::DataFrame(f) = packet {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.frames.push(s.to_vec());
                    }
                }
                Ok(PushOutcome::Bitrate(self.bps))
            })
        }
    }

    /// Pseudorandom (incompressible) I420 frame, so the encoder's rate target
    /// actually bites: the scrolling pattern hits x264's entropy floor at any
    /// bitrate and would mask a retarget.
    fn noise_frame(seq: u64) -> Vec<u8> {
        let (w, h) = (W as usize, H as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut state = seq.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut v = Vec::with_capacity(w * h + 2 * cw * ch);
        for _ in 0..w * h {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v.push((state >> 33) as u8);
        }
        v.extend(core::iter::repeat_n(110u8, cw * ch));
        v.extend(core::iter::repeat_n(150u8, cw * ch));
        v
    }

    /// Runtime retarget: a downstream BWE target far below the configured
    /// rate reopens the encoder, and the later access units come out smaller.
    #[tokio::test]
    async fn bitrate_feedback_retargets_via_reopen() {
        async fn bytes_after_feedback(bps: u32) -> Option<f64> {
            let mut enc = FfmpegVideoEnc::new()
                .with_backend(Backend::Software)
                .with_bitrate(2_000_000);
            if enc.configure_pipeline(&i420_caps(W, H)).is_err() {
                return None; // libx264 absent on this host
            }
            let mut sink = BitrateSink {
                bps,
                frames: Vec::new(),
            };
            for i in 0..60u64 {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(
                        noise_frame(i).into_boxed_slice(),
                    )),
                    FrameTiming {
                        pts_ns: i * 33_000_000,
                        ..FrameTiming::default()
                    },
                    i,
                );
                enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                    .await
                    .ok()?;
            }
            // Compare the tail (well after the reopen), skipping the fresh IDR.
            let tail = &sink.frames[sink.frames.len().saturating_sub(20)..];
            let total: usize = tail.iter().skip(1).map(|f| f.len()).sum();
            Some(total as f64 / tail.len().saturating_sub(1).max(1) as f64)
        }

        let (Some(low), Some(high)) = (
            bytes_after_feedback(100_000).await,
            bytes_after_feedback(2_000_000).await,
        ) else {
            std::eprintln!("skipping: libx264 not available");
            return;
        };
        assert!(
            low * 1.5 < high,
            "100 kb/s tail ({low:.0} B/AU) must be far smaller than 2 Mb/s ({high:.0} B/AU)"
        );
    }

    /// Shed-layer idle: `Bitrate(0)` mostly stops the encoder (sparse
    /// 1-in-32 keep-alives only), and a non-zero target resumes it on an IDR.
    #[tokio::test]
    async fn bitrate_zero_idles_and_resume_restarts_on_idr() {
        let mut enc = FfmpegVideoEnc::new().with_backend(Backend::Software);
        if enc.configure_pipeline(&i420_caps(W, H)).is_err() {
            std::eprintln!("skipping: libx264 not available");
            return;
        }
        async fn push_n(enc: &mut FfmpegVideoEnc, n: u64, base: u64, bps: u32) -> Vec<Vec<u8>> {
            let mut sink = BitrateSink {
                bps,
                frames: Vec::new(),
            };
            for i in 0..n {
                let seq = base + i;
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(
                        i420_frame(seq).into_boxed_slice(),
                    )),
                    FrameTiming {
                        pts_ns: seq * 33_000_000,
                        ..FrameTiming::default()
                    },
                    seq,
                );
                enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                    .await
                    .unwrap();
            }
            sink.frames
        }
        // Prime (delivers the Bitrate(0) idle hint on these pushes)...
        let primed = push_n(&mut enc, 5, 0, 0).await;
        assert!(!primed.is_empty(), "primed frames encoded");
        // ...then 60 idle frames encode at most the sparse keep-alives.
        let idle = push_n(&mut enc, 60, 5, 0).await;
        assert!(
            idle.len() <= 3,
            "idle encodes only sparse keep-alives, got {}",
            idle.len()
        );
        // A non-zero target resumes, and the first resumed AU is an IDR.
        let resumed = push_n(&mut enc, 5, 65, 1_000_000).await;
        assert!(resumed.len() >= 4, "resume encodes again");
        assert!(
            crate::annexb::au_is_keyframe(
                g2g_core::VideoCodec::H264,
                &resumed[1.min(resumed.len() - 1)]
            ) || crate::annexb::au_is_keyframe(g2g_core::VideoCodec::H264, &resumed[0]),
            "resume starts on a keyframe"
        );
    }

    /// The encoded stream must be a valid H.264 Annex-B elementary stream: the
    /// first access unit begins with a start code (the in-band SPS/PPS + IDR),
    /// and `FfmpegVideoDec` decodes it back to I420 at the original geometry. Runs
    /// for whichever backend this host has; both skip cleanly if absent.
    async fn round_trip(backend: Backend) {
        let Some(sink) = encode_with(backend).await else {
            std::eprintln!(
                "skipping: {:?} H.264 encoder not available on this host",
                backend
            );
            return;
        };
        assert!(
            !sink.frames.is_empty(),
            "{backend:?} produced H.264 access units"
        );
        assert_eq!(
            sink.caps,
            std::vec![Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Fixed(W),
                height: Dim::Fixed(H),
                framerate: Rate::Fixed(30 << 16),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                alignment: g2g_core::BitstreamAlignment::Unspecified,
            }],
            "output caps announced once"
        );
        // Annex-B: the first unit starts with a 3- or 4-byte start code.
        let first = &sink.frames[0];
        let annex_b = first.starts_with(&[0, 0, 0, 1]) || first.starts_with(&[0, 0, 1]);
        assert!(
            annex_b,
            "{backend:?} output is Annex-B framed, got {:?}",
            &first[..4.min(first.len())]
        );

        // Decode the stream back and confirm it yields I420 at the right geometry,
        // proving the encoder produced a real, decodable H.264 bitstream.
        let mut dec = FfmpegVideoDec::new();
        dec.configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        })
        .expect("open H.264 decoder");
        let mut dsink = CaptureSink::default();
        for au in &sink.frames {
            let f = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            dec.process(PipelinePacket::DataFrame(f), &mut dsink)
                .await
                .expect("decode AU");
        }
        dec.process(PipelinePacket::Eos, &mut dsink)
            .await
            .expect("drain decoder");

        let geometry = dsink.caps.iter().find_map(|c| match c {
            Caps::RawVideo {
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                ..
            } => Some((*w, *h)),
            _ => None,
        });
        assert_eq!(
            geometry,
            Some((W, H)),
            "{backend:?} stream decodes back to {W}x{H}"
        );
        assert!(
            !dsink.frames.is_empty(),
            "{backend:?} stream decoded to raw frames"
        );
        let expected = (W * H + 2 * W.div_ceil(2) * H.div_ceil(2)) as usize;
        assert!(
            dsink.frames.iter().all(|f| f.len() == expected),
            "decoded frames are full I420 ({expected} bytes)"
        );
    }

    /// Push `frames` (already in `format`'s layout) plus EOS through `backend`.
    /// `None` when that encoder cannot take that input on this host.
    async fn encode_frames(
        backend: Backend,
        format: RawVideoFormat,
        frames: Vec<Vec<u8>>,
    ) -> Option<CaptureSink> {
        let mut enc = FfmpegVideoEnc::new().with_backend(backend);
        enc.configure_pipeline(&raw_caps(format, W, H)).ok()?;
        let mut sink = CaptureSink::default();
        for (i, raw) in frames.into_iter().enumerate() {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(raw.into_boxed_slice())),
                FrameTiming {
                    pts_ns: i as u64 * 33_000_000,
                    ..FrameTiming::default()
                },
                i as u64,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .expect("encode frame");
        }
        enc.process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("flush encoder");
        Some(sink)
    }

    /// Decode an encoded stream back with the in-repo ffmpeg decoder, returning
    /// the decoded raw frames and the format the decoder announced. `Auto` output
    /// so the decode preserves whatever chroma and depth the stream carries.
    async fn decode_back(codec: VideoCodec, aus: &[Vec<u8>]) -> (RawVideoFormat, Vec<Vec<u8>>) {
        let mut dec =
            FfmpegVideoDec::new().with_output_format(crate::ffmpegdec::OutputFormat::Auto);
        dec.configure_pipeline(&Caps::CompressedVideo {
            codec,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        })
        .expect("open decoder");
        let mut sink = CaptureSink::default();
        for au in aus {
            let f = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            dec.process(PipelinePacket::DataFrame(f), &mut sink)
                .await
                .expect("decode AU");
        }
        dec.process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("drain decoder");
        let format = sink
            .caps
            .iter()
            .find_map(|c| match c {
                Caps::RawVideo { format, .. } => Some(*format),
                _ => None,
            })
            .expect("decoder announced raw caps");
        (format, sink.frames)
    }

    fn mean(samples: &[u8]) -> f64 {
        samples.iter().map(|&s| s as f64).sum::<f64>() / samples.len().max(1) as f64
    }

    /// NV12 input encodes without a `videoconvert` in front, and the
    /// decoded round trip puts luma and both chroma components back where they
    /// belong (a mis-read interleave would grey the chroma out).
    #[tokio::test]
    async fn nv12_input_round_trips_through_the_decoder() {
        nv12_round_trip(Backend::Software).await;
    }

    #[tokio::test]
    async fn vaapi_h264_nv12_round_trips_through_the_decoder() {
        nv12_round_trip(Backend::Vaapi(VaapiCodec::H264)).await;
    }

    #[tokio::test]
    async fn vaapi_h265_nv12_round_trips_through_the_decoder() {
        nv12_round_trip(Backend::Vaapi(VaapiCodec::H265)).await;
    }

    #[tokio::test]
    async fn vaapi_av1_nv12_round_trips_through_the_decoder() {
        nv12_round_trip(Backend::Vaapi(VaapiCodec::Av1)).await;
    }

    async fn nv12_round_trip(backend: Backend) {
        let frames = (0..10u64).map(nv12_frame).collect();
        let Some(sink) = encode_frames(backend, RawVideoFormat::Nv12, frames).await else {
            std::eprintln!("skipping: {backend:?} encoder not available");
            return;
        };
        assert!(!sink.frames.is_empty(), "NV12 input produced access units");
        assert!(
            matches!(sink.caps.as_slice(), [Caps::CompressedVideo { codec, .. }] if *codec == backend.codec()),
            "{backend:?} announces {:?} output, got {:?}",
            backend.codec(),
            sink.caps
        );

        let (format, decoded) = decode_back(backend.codec(), &sink.frames).await;
        assert_eq!(format, RawVideoFormat::I420, "decoder emits 8-bit I420");
        assert!(!decoded.is_empty(), "NV12-sourced stream decodes back");
        let (w, h) = (W as usize, H as usize);
        let (cw, ch) = chroma_dims();
        assert!(
            decoded.iter().all(|f| f.len() == w * h + 2 * cw * ch),
            "decoded frames are full I420"
        );
        let f = &decoded[0];
        let (u, v) = (
            mean(&f[w * h..w * h + cw * ch]),
            mean(&f[w * h + cw * ch..]),
        );
        assert!(
            (u - U_VAL as f64).abs() < 8.0 && (v - V_VAL as f64).abs() < 8.0,
            "NV12 chroma survives the round trip, got U={u:.0} V={v:.0}"
        );
        let y = mean(&f[..w * h]);
        let src = mean(&luma(0));
        assert!(
            (y - src).abs() < 8.0,
            "NV12 luma survives the round trip, got {y:.0} for a {src:.0} source"
        );
    }

    /// 10-bit input encodes as High 10 and decodes back to 10-bit planar.
    /// Skips (with the reason) on a libx264 built 8-bit only.
    #[tokio::test]
    async fn ten_bit_input_encodes_high10_and_round_trips() {
        let frames = (0..10u64).map(i420p10_frame).collect();
        let Some(sink) = encode_frames(Backend::Software, RawVideoFormat::I420p10, frames).await
        else {
            std::eprintln!(
                "skipping: this host's libx264 has no 10-bit support \
                 (yuv420p10le is not in the encoder's pixel formats)"
            );
            return;
        };
        assert!(
            !sink.frames.is_empty(),
            "10-bit input produced access units"
        );
        // profile_idc 110 (0x6E) is High 10; an 8-bit encode would say 0x42/0x64.
        let codec = crate::h264util::h264_codec_string(&sink.frames[0])
            .expect("first access unit carries the SPS");
        assert!(
            codec.starts_with("avc1.6E"),
            "10-bit input encodes as High 10, got {codec}"
        );

        let (format, decoded) = decode_back(VideoCodec::H264, &sink.frames).await;
        assert_eq!(
            format,
            RawVideoFormat::I420p10,
            "the stream decodes back as 10-bit planar"
        );
        let (w, h) = (W as usize, H as usize);
        let (cw, ch) = chroma_dims();
        assert!(
            decoded.iter().all(|f| f.len() == 2 * (w * h + 2 * cw * ch)),
            "decoded frames are full 10-bit I420 (2 bytes per sample)"
        );
    }

    /// Semi-planar 10-bit: each 8-bit test sample as a little-endian word with
    /// the 10 significant bits at the top (`<< 8`).
    fn p010_frame(seq: u64) -> Vec<u8> {
        let (cw, ch) = chroma_dims();
        let mut v = Vec::new();
        let widen = |v: &mut Vec<u8>, s: u8| v.extend_from_slice(&((s as u16) << 8).to_le_bytes());
        for s in luma(seq) {
            widen(&mut v, s);
        }
        for _ in 0..cw * ch {
            widen(&mut v, U_VAL);
            widen(&mut v, V_VAL);
        }
        v
    }

    #[tokio::test]
    async fn vaapi_h265_p010_round_trips_as_ten_bit() {
        let backend = Backend::Vaapi(VaapiCodec::H265);
        let frames = (0..10u64).map(p010_frame).collect();
        let Some(sink) = encode_frames(backend, RawVideoFormat::P010, frames).await else {
            std::eprintln!("skipping: no VA-API H.265 Main10 encoder");
            return;
        };
        let (format, decoded) = decode_back(VideoCodec::H265, &sink.frames).await;
        // the decoder's auto output keeps the depth, as planar or semi-planar
        let eight_bit_shift = match format {
            RawVideoFormat::I420p10 => 2,
            RawVideoFormat::P010 => 8,
            other => panic!("Main10 stream decodes to a 10-bit format, got {other:?}"),
        };
        let (w, h) = (W as usize, H as usize);
        let luma_words = &decoded.first().expect("stream decodes back")[..w * h * 2];
        let decoded_luma: Vec<u8> = luma_words
            .as_chunks::<2>()
            .0
            .iter()
            .map(|word| (u16::from_le_bytes(*word) >> eight_bit_shift) as u8)
            .collect();
        let (y, src) = (mean(&decoded_luma), mean(&luma(0)));
        assert!(
            (y - src).abs() < 8.0,
            "P010 luma survives the round trip, got {y:.0} for a {src:.0} source"
        );
    }

    #[test]
    fn vaapi_encoders_take_gst_property_names_and_units() {
        let mut h264 = FfmpegVideoEnc::new().with_backend(Backend::Vaapi(VaapiCodec::H264));
        h264.set_property("bitrate", PropValue::Uint(2500))
            .expect("bitrate in kbit/s");
        assert_eq!(h264.bitrate_bps, 2_500_000);
        assert_eq!(h264.get_property("bitrate"), Some(PropValue::Uint(2500)));
        assert_eq!(h264.get_property("qpp"), Some(PropValue::Uint(26)));
        h264.set_property("rate-control", PropValue::Str("cqp".into()))
            .expect("cqp rate control");
        assert_eq!(h264.pass, Pass::Quant);
        assert_eq!(
            h264.set_property("qpp", PropValue::Uint(52)),
            Err(PropError::Value)
        );
        assert_eq!(
            h264.set_property("quantizer", PropValue::Uint(20)),
            Err(PropError::Unknown),
            "the x264enc name is not a vah264enc property"
        );

        let mut av1 = FfmpegVideoEnc::new().with_backend(Backend::Vaapi(VaapiCodec::Av1));
        av1.set_property("qp", PropValue::Uint(200))
            .expect("AV1 q index up to 255");
        assert_eq!(av1.get_property("qpp"), None);
        assert!(av1.properties().iter().any(|spec| spec.name == "qp"));
    }

    // the SPS libavcodec 7.1 hevc_vaapi writes on radeonsi for 640x360 input
    const RADEONSI_PADDED_SPS: &[u8] = &[
        0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03, 0x00, 0xb0, 0x00,
        0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x3f, 0xa0, 0x05, 0x02, 0x01, 0x71, 0x62, 0x4b, 0x92,
        0x42, 0x95, 0xa2, 0x01, 0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x00, 0x03, 0x00, 0x1e, 0xc4,
    ];

    fn bits_after_window(nal: &[u8]) -> (crate::h265parse::H265SpsHead, Vec<u32>) {
        let rbsp = crate::annexb::strip_emulation_prevention(&nal[H265_NAL_HEADER_BYTES..]);
        let (head, mut reader) = crate::h265parse::parse_h265_sps_head(&rbsp).expect("SPS parses");
        let stop = rbsp_stop_bit(&rbsp).expect("SPS has a stop bit");
        let bits = (reader.bit_pos()..stop)
            .map(|_| reader.read_bit().expect("bit before the stop bit"))
            .collect();
        (head, bits)
    }

    #[test]
    fn padded_h265_sps_gets_a_conformance_window() {
        let au = RADEONSI_PADDED_SPS;
        let (padded, padded_rest) = bits_after_window(&au[4..]);
        assert!(!padded.conformance_window_flag);
        let coded = (
            padded.pic_width_in_luma_samples,
            padded.pic_height_in_luma_samples,
        );
        let (width, height) = (640, 360);
        assert_ne!(coded, (width, height), "the fixture is the padded case");

        let rebuilt = crop_padded_h265_sps(au, width, height).expect("SPS rewrite");
        let cropped_nal = &rebuilt[4..];
        let (cropped, cropped_rest) = bits_after_window(cropped_nal);
        assert!(cropped.conformance_window_flag);
        assert_eq!(
            (
                cropped.pic_width_in_luma_samples,
                cropped.pic_height_in_luma_samples
            ),
            coded
        );
        // 4:2:0 offsets count chroma samples, two luma rows each
        assert_eq!(
            cropped.conformance_window,
            [0, (coded.0 - width) / 2, 0, (coded.1 - height) / 2]
        );
        assert_eq!(
            cropped_rest, padded_rest,
            "fields after the window are unchanged"
        );
        assert_eq!(
            crop_padded_h265_sps(&rebuilt, width, height).expect("second pass"),
            rebuilt,
            "an SPS with a window is left alone"
        );
    }

    /// Fail loud: NVENC's H.264 encoder is 8-bit only, so 10-bit input must
    /// fail negotiation rather than open and emit garbage.
    #[tokio::test]
    async fn ten_bit_on_nvenc_fails_negotiation() {
        let mut enc = FfmpegVideoEnc::new().with_backend(Backend::Nvenc);
        assert!(
            enc.configure_pipeline(&raw_caps(RawVideoFormat::I420p10, W, H))
                .is_err(),
            "h264_nvenc has no 10-bit path: negotiation must fail"
        );
    }

    /// The element advertises exactly the raw formats it can encode.
    #[test]
    fn accepts_i420_nv12_and_10_bit_input_caps() {
        let enc = FfmpegVideoEnc::new();
        for format in [
            RawVideoFormat::I420,
            RawVideoFormat::Nv12,
            RawVideoFormat::I420p10,
        ] {
            assert_eq!(
                enc.intercept_caps(&raw_caps(format, W, H)).unwrap(),
                raw_caps(format, W, H),
                "{format:?} accepted unchanged"
            );
        }
        for format in [RawVideoFormat::Rgba8, RawVideoFormat::I444] {
            assert!(
                enc.intercept_caps(&raw_caps(format, W, H)).is_err(),
                "{format:?} rejected"
            );
        }
    }

    /// Setting the `bitrate` property mid-stream retargets a running
    /// encoder (libavcodec fixes its rate at open, so this reopens it): output
    /// continues, restarts on an IDR, and the access units get smaller.
    #[tokio::test]
    async fn bitrate_property_retargets_a_running_encoder() {
        let mut enc = FfmpegVideoEnc::new()
            .with_backend(Backend::Software)
            .with_bitrate(2_000_000);
        if enc.configure_pipeline(&i420_caps(W, H)).is_err() {
            std::eprintln!("skipping: libx264 not available");
            return;
        }
        let mut sink = CaptureSink::default();
        async fn push_n(
            enc: &mut FfmpegVideoEnc,
            sink: &mut CaptureSink,
            range: core::ops::Range<u64>,
        ) {
            for i in range {
                let frame = Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(
                        noise_frame(i).into_boxed_slice(),
                    )),
                    FrameTiming {
                        pts_ns: i * 33_000_000,
                        ..FrameTiming::default()
                    },
                    i,
                );
                enc.process(PipelinePacket::DataFrame(frame), sink)
                    .await
                    .expect("encode frame");
            }
        }
        push_n(&mut enc, &mut sink, 0..40).await;
        let before = sink.frames.len();
        assert!(before >= 30, "the 2 Mb/s run emitted access units");

        enc.set_property("bitrate", PropValue::Uint(100_000))
            .expect("set bitrate");
        assert_eq!(enc.get_property("bitrate"), Some(PropValue::Uint(100_000)));

        push_n(&mut enc, &mut sink, 40..80).await;
        enc.process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("flush encoder");
        assert!(
            sink.frames.len() >= before + 30,
            "the stream continues across the retarget ({} -> {})",
            before,
            sink.frames.len()
        );
        assert!(
            crate::annexb::au_is_keyframe(g2g_core::VideoCodec::H264, &sink.frames[before]),
            "the reopened encoder starts on an IDR"
        );
        // Skip the IDR at each window's head; compare steady-state AU sizes.
        let high = mean_len(&sink.frames[10..before]);
        let low = mean_len(&sink.frames[before + 5..]);
        assert!(
            low * 1.5 < high,
            "100 kb/s tail ({low:.0} B/AU) must be far smaller than 2 Mb/s ({high:.0} B/AU)"
        );
    }

    #[tokio::test]
    async fn quantizer_zero_round_trips_bit_exact() {
        let mut enc = FfmpegVideoEnc::new().with_backend(Backend::Software);
        enc.set_property("pass", PropValue::Str("quant".into()))
            .expect("set pass");
        enc.set_property("quantizer", PropValue::Uint(0))
            .expect("set quantizer");
        assert_eq!(
            enc.set_property(
                "quantizer",
                PropValue::Uint(u64::from(Backend::Software.max_quantizer()) + 1)
            ),
            Err(PropError::Value)
        );
        if enc.configure_pipeline(&i420_caps(W, H)).is_err() {
            std::eprintln!("skipping: libx264 not available");
            return;
        }
        let luma_bytes = (W * H) as usize;
        let frames: Vec<Vec<u8>> = (0..10u64)
            .map(|seq| {
                let mut frame = noise_frame(seq);
                for (i, chroma) in frame[luma_bytes..].iter_mut().enumerate() {
                    *chroma = (i as u64 * 7 + seq) as u8;
                }
                frame
            })
            .collect();
        let mut sink = CaptureSink::default();
        for (seq, raw) in frames.iter().enumerate() {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(raw.clone().into_boxed_slice())),
                FrameTiming {
                    pts_ns: seq as u64 * 33_000_000,
                    ..FrameTiming::default()
                },
                seq as u64,
            );
            enc.process(PipelinePacket::DataFrame(frame), &mut sink)
                .await
                .expect("encode frame");
        }
        enc.process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("flush encoder");
        // profile_idc 244 (0xF4) is High 4:4:4 Predictive, the only profile with lossless coding
        let codec = crate::h264util::h264_codec_string(&sink.frames[0])
            .expect("first access unit carries the SPS");
        assert!(
            codec.starts_with("avc1.F4"),
            "lossless profile, got {codec}"
        );
        let (format, decoded) = decode_back(VideoCodec::H264, &sink.frames).await;
        assert_eq!(format, RawVideoFormat::I420);
        assert!(decoded == frames, "decoded frames equal the input frames");
    }

    fn mean_len(frames: &[Vec<u8>]) -> f64 {
        frames.iter().map(|f| f.len() as f64).sum::<f64>() / frames.len().max(1) as f64
    }

    #[tokio::test]
    async fn nvenc_h264_round_trips_through_the_decoder() {
        round_trip(Backend::Nvenc).await;
    }

    #[tokio::test]
    async fn software_h264_round_trips_through_the_decoder() {
        round_trip(Backend::Software).await;
    }
}
