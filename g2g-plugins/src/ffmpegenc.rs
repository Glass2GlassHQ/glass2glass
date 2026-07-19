//! Linux H.264 video *encode* element using ffmpeg / libavcodec (M266): the
//! encode-side mirror of [`crate::ffmpegdec::FfmpegVideoDec`]. `RawVideo{I420}`
//! in, `CompressedVideo{H264}` Annex-B out, so the Linux production path finally
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
//! Deferred (v1): runtime bitrate retarget (a libavcodec encoder fixes the rate
//! at open; a BWE change would need a reopen, like `Av1Enc`'s context rebuild),
//! NV12 input (I420 only for now), and 10-bit. The downstream bitrate feedback is
//! recorded but not yet acted on.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use ffmpeg::codec::encoder::video::Encoder as VideoEncoder;
use ffmpeg::format::Pixel;
use ffmpeg::frame::Video as FfVideo;
use ffmpeg::packet::Packet;
use ffmpeg::Dictionary;
use ffmpeg::Error as FfError;
use ffmpeg::Rational;
use ffmpeg_next as ffmpeg;

use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

/// Default constant target bitrate (bits/second) when the caller sets none. 4
/// Mbps is a reasonable 1080p30 streaming default.
const DEFAULT_BITRATE_BPS: usize = 4_000_000;

/// Default GOP length (frames between IDRs) when framerate is unknown. One IDR
/// per ~2 seconds at 30 fps; a network sink also forces IDRs on demand (PLI).
const DEFAULT_GOP: u32 = 60;

/// libavcodec H.264 encoder backend. The element shape (I420 in, H.264 Annex-B
/// out) is identical; only the encoder opened changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA NVENC (`h264_nvenc`). Hardware, realtime. Default.
    Nvenc,
    /// libx264 software encoder (`libx264`). Portable CPU fallback.
    Software,
}

impl Backend {
    /// The libavcodec encoder name to look up.
    fn encoder_name(self) -> &'static str {
        match self {
            Backend::Nvenc => "h264_nvenc",
            Backend::Software => "libx264",
        }
    }
}

/// Encodes raw I420 video into an H.264 Annex-B elementary stream.
pub struct FfmpegH264Enc {
    backend: Backend,
    width: u32,
    height: u32,
    framerate: Rate,
    /// Target constant bitrate (bits/second).
    bitrate_bps: usize,
    /// The opened video encoder. Derefs to the base `Encoder` for
    /// `send_frame` / `receive_packet`.
    encoder: Option<VideoEncoder>,
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
    configured: bool,
}

// SAFETY: `ffmpeg::encoder::Encoder` wraps a raw `*mut AVCodecContext` and is
// `!Send`. The multi-thread runner moves the element between worker tasks but
// drives it through `&mut self` only (never concurrently), so the context is
// owned and moved, never aliased, the same contract upheld by `FfmpegVideoDec`.
unsafe impl Send for FfmpegH264Enc {}

impl core::fmt::Debug for FfmpegH264Enc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FfmpegH264Enc")
            .field("backend", &self.backend)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("emitted", &self.emitted)
            .field("configured", &self.configured)
            .finish()
    }
}

impl Default for FfmpegH264Enc {
    fn default() -> Self {
        Self::new()
    }
}

impl FfmpegH264Enc {
    pub fn new() -> Self {
        Self {
            backend: Backend::Nvenc,
            width: 0,
            height: 0,
            framerate: Rate::Any,
            bitrate_bps: DEFAULT_BITRATE_BPS,
            encoder: None,
            pts_by_frameno: alloc::collections::BTreeMap::new(),
            frame_no: 0,
            emitted: 0,
            caps_sent: false,
            force_keyframe: false,
            configured: false,
        }
    }

    /// Select the encoder backend (default [`Backend::Nvenc`]).
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
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

    fn input_template() -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
        }
    }

    /// Frames per second from the negotiated framerate, defaulting to 30 when
    /// unspecified. The framerate is a Q16.16 fixed-point value.
    fn fps(&self) -> u32 {
        match self.framerate {
            Rate::Fixed(q16) => (q16 >> 16).max(1),
            _ => 30,
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
                opts.set("rc", "cbr");
                opts.set("delay", "0");
                opts.set("zerolatency", "1");
            }
            Backend::Software => {
                opts.set("preset", "veryfast");
                opts.set("tune", "zerolatency");
            }
        }
        opts
    }

    /// Build and open the libavcodec encoder on the negotiated geometry. Fails
    /// loud if the encoder is absent (libavcodec built without it, or no NVIDIA
    /// driver for NVENC) so the caller can pick another backend.
    fn open_encoder(&mut self) -> Result<(), G2gError> {
        let codec = ffmpeg::encoder::find_by_name(self.backend.encoder_name())
            .ok_or(G2gError::Hardware(HardwareError::Other))?;

        let fps = self.fps();
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
        video.set_format(Pixel::YUV420P);
        // time_base = 1/fps, so a frame's PTS is just its index; frame_rate lets
        // the encoder pace its rate control.
        video.set_time_base(Rational::new(1, fps as i32));
        video.set_frame_rate(Some(Rational::new(fps as i32, 1)));
        video.set_bit_rate(self.bitrate_bps);
        video.set_max_bit_rate(self.bitrate_bps);
        video.set_gop(DEFAULT_GOP);
        // No B-frames: output stays in presentation order (no reorder latency),
        // which the low-latency streaming path wants.
        video.set_max_b_frames(0);

        let opened = video
            .open_as_with(codec, self.open_options())
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.encoder = Some(opened);
        self.pts_by_frameno.clear();
        self.frame_no = 0;
        Ok(())
    }

    /// Copy an I420 access unit into a fresh `YUV420P` AVFrame (honouring the
    /// frame's plane strides), forcing an IDR if a keyframe was requested, and
    /// drain whatever packets the encoder releases.
    fn encode(&mut self, i420: &[u8], pts_ns: u64) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let (y_size, c_size) = (w * h, cw * ch);
        if i420.len() < y_size + 2 * c_size {
            return Err(G2gError::CapsMismatch);
        }

        let mut frame = FfVideo::new(Pixel::YUV420P, self.width, self.height);
        // Read each plane's stride before borrowing the plane data mutably (the
        // borrow checker won't allow `data_mut` and `stride` in one call).
        let (s0, s1, s2) = (frame.stride(0), frame.stride(1), frame.stride(2));
        copy_plane(frame.data_mut(0), s0, &i420[..y_size], w, h);
        copy_plane(
            frame.data_mut(1),
            s1,
            &i420[y_size..y_size + c_size],
            cw,
            ch,
        );
        copy_plane(
            frame.data_mut(2),
            s2,
            &i420[y_size + c_size..y_size + 2 * c_size],
            cw,
            ch,
        );

        let frameno = self.frame_no;
        frame.set_pts(Some(frameno));
        if core::mem::take(&mut self.force_keyframe) {
            // SAFETY: `frame` is a freshly allocated, writable AVFrame we own;
            // setting the picture type to I requests an IDR on this frame. NVENC
            // and libx264 both honour `pict_type` for forced key frames.
            unsafe {
                (*frame.as_mut_ptr()).pict_type = ffmpeg::ffi::AVPictureType::AV_PICTURE_TYPE_I;
            }
        }
        // pts_by_frameno is keyed by the frame counter we stamped as the PTS.
        self.pts_by_frameno.insert(frameno as u64, pts_ns);
        self.frame_no += 1;

        let encoder = self.encoder.as_mut().ok_or(G2gError::NotConfigured)?;
        encoder
            .send_frame(&frame)
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
                        out.push((data.to_vec(), pts_ns));
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
        let feedback = crate::encoder_base::emit_packets(
            &mut self.caps_sent,
            &mut self.emitted,
            packets,
            &caps,
            out,
        )
        .await?;
        // A downstream PLI latches a forced IDR on the next encode. Runtime
        // bitrate retarget (feedback.bitrate_bps) is recorded by the encoder API
        // but not yet acted on (a libavcodec encoder fixes its rate at open).
        if feedback.force_keyframe {
            self.force_keyframe = true;
        }
        Ok(())
    }
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

impl AsyncElement for FfmpegH264Enc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&Self::input_template())
    }

    /// Native `DerivedOutput`: I420 (any geometry) in, H.264 at the same dims and
    /// framerate out. Non-I420 input yields an empty set, rejected at solve.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format: RawVideoFormat::I420,
                width,
                height,
                framerate,
            } => CapsSet::one(Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo {
            format: RawVideoFormat::I420,
            width,
            height,
            framerate,
        } = absolute_caps
        else {
            return Err(G2gError::CapsMismatch);
        };
        let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) else {
            return Err(G2gError::CapsMismatch);
        };
        // ffmpeg::init() registers codecs once per process; safe to repeat.
        ffmpeg::init().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.width = *w;
        self.height = *h;
        self.framerate = framerate.clone();
        self.open_encoder()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "FFmpeg H.264 encoder",
            "Codec/Encoder/Video",
            "Encodes raw I420 video to H.264 Annex-B via libavcodec (NVENC / libx264)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FFMPEGENC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
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
                // bits per second.
                let bps = value.as_uint().ok_or(PropError::Type)?;
                self.bitrate_bps = (bps as usize).max(1);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "backend" => Some(PropValue::Str(
                match self.backend {
                    Backend::Nvenc => "nvenc",
                    Backend::Software => "software",
                }
                .into(),
            )),
            "bitrate" => Some(PropValue::Uint(self.bitrate_bps as u64)),
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let packets = self.encode(slice.as_slice(), frame.timing.pts_ns)?;
                    self.emit(packets, out).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the encoder; the runner's transform arm forwards EOS.
                    let packets = self.flush()?;
                    self.emit(packets, out).await?;
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

impl PadTemplates for FfmpegH264Enc {
    fn pad_templates() -> Vec<PadTemplate> {
        let out = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(Self::input_template())),
            PadTemplate::source(CapsSet::one(out)),
        ])
    }
}

/// Settable properties: backend (nvenc | software) and the target bitrate, so a
/// `gst-launch` line can pick the encoder and rate without the builder.
static FFMPEGENC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "backend",
        PropKind::Str,
        "h264 encoder: nvenc | software (libx264)",
    ),
    PropertySpec::new(
        "bitrate",
        PropKind::Uint,
        "constant target bitrate, bits/second",
    ),
];

/// Preferred alias once this encodes more than H.264 (HEVC via `hevc_nvenc` is
/// the natural next backend); the struct keeps its current name for now.
pub type FfmpegVideoEnc = FfmpegH264Enc;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffmpegdec::FfmpegVideoDec;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{FrameTiming, PushOutcome};

    const W: u32 = 320;
    const H: u32 = 240;

    /// A moving test pattern so successive frames differ (a flat image would let
    /// the encoder emit near-empty inter frames and weaken the round-trip check).
    fn i420_frame(seq: u64) -> Vec<u8> {
        let (w, h) = (W as usize, H as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut v = Vec::with_capacity(w * h + 2 * cw * ch);
        for y in 0..h {
            for x in 0..w {
                v.push(((x + y + seq as usize * 7) & 0xff) as u8);
            }
        }
        v.extend(core::iter::repeat_n(110u8, cw * ch)); // U
        v.extend(core::iter::repeat_n(150u8, cw * ch)); // V
        v
    }

    fn i420_caps(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    #[derive(Default)]
    struct CaptureSink {
        caps: Vec<Caps>,
        frames: Vec<Vec<u8>>,
    }
    impl OutputSink for CaptureSink {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
            Box::pin(async move {
                match packet {
                    PipelinePacket::CapsChanged(c) => self.caps.push(c),
                    PipelinePacket::DataFrame(f) => {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.frames.push(s.as_slice().to_vec());
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
        let mut enc = FfmpegH264Enc::new().with_backend(backend);
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

    #[tokio::test]
    async fn nvenc_h264_round_trips_through_the_decoder() {
        round_trip(Backend::Nvenc).await;
    }

    #[tokio::test]
    async fn software_h264_round_trips_through_the_decoder() {
        round_trip(Backend::Software).await;
    }
}
