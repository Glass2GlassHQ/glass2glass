//! M306: Android hardware H.264 / H.265 encode via the NDK MediaCodec
//! (`AMediaCodec`).
//!
//! `MediaCodecEnc` is the encode mirror of [`MediaCodecDec`](crate::mediacodecdec::MediaCodecDec)
//! and the Android counterpart of `MfEncode` (Windows) / `VtEncode` (macOS): it
//! consumes raw NV12 frames (`MemoryDomain::System`, `Caps::RawVideo`) and
//! produces Annex-B H.264 / H.265 access units (`MemoryDomain::System`,
//! `Caps::CompressedVideo`). Unlike the decoder (whose output had to go through an
//! `ImageReader` Surface because vendor decoders only emit graphic buffers), the
//! encoder uses the ordinary ByteBuffer path on both sides: NV12 in via input
//! buffers, the encoded bitstream out via output buffers.
//!
//! **Colour format.** The element configures the encoder with
//! `COLOR_FormatYUV420SemiPlanar` (21 = NV12), matching the decoder's output
//! layout, so a decode -> encode transcode hands frames across unchanged. Some
//! vendor encoders prefer `COLOR_FormatYUV420Flexible`; if a device rejects
//! SemiPlanar this is the first thing to revisit (the same colour-format caveat
//! the decoder hit on its first on-device run).
//!
//! **Codec-specific data.** The MediaCodec H.264 / H.265 encoders emit the
//! parameter sets as Annex-B (start-code framed): once up front in a buffer
//! flagged `BUFFER_FLAG_CODEC_CONFIG`, and the per-frame NALs already carry start
//! codes too. We capture that config buffer (SPS/PPS, or VPS+SPS+PPS for HEVC)
//! and prepend it to every key frame, so the emitted elementary stream is a
//! self-contained Annex-B stream every downstream H.26x element can parse
//! (`project_h264_framing`: the pipeline is Annex-B and parameter sets ride the
//! key frames).
//!
//! Drives the codec synchronously (queue one input buffer, drain ready output),
//! wrapping the safe `ndk` crate, the same single-thread contract as the decoder
//! (`Send` asserted under that contract so the multi-thread runner accepts it).
//! Built against `ndk` 0.9 MediaCodec (api-level-24); validated on a device via
//! `tests/android_mediacodec_enc_probe.rs` / `tools/android-mediacodec-enc-smoke.sh`.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use ndk::media::media_codec::{DequeuedOutputBufferInfoResult, MediaCodec, MediaCodecDirection};
use ndk::media::media_format::MediaFormat;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

use crate::mediacodec_common::{
    queue_input, BUFFER_FLAG_CODEC_CONFIG, BUFFER_FLAG_END_OF_STREAM, BUFFER_FLAG_KEY_FRAME,
    MAX_OUTPUT_POLLS,
};

use alloc::boxed::Box;
use alloc::vec::Vec;

/// `COLOR_FormatYUV420SemiPlanar` (NV12): the encoder input colour format, matching
/// `MediaCodecDec`'s NV12 output so a transcode is copy-compatible.
const COLOR_FORMAT_NV12: i32 = 21;

/// Default target bitrate (4 Mbps), the `MfEncode` / `VtEncode` default.
const DEFAULT_BITRATE: u32 = 4_000_000;
/// Default frame rate hint when the input caps leave it unspecified.
const DEFAULT_FRAMERATE: i32 = 30;
/// Default key-frame interval in seconds.
const DEFAULT_KEYFRAME_INTERVAL_S: i32 = 1;

/// One encoded access unit ready to emit.
#[derive(Debug)]
struct EncodedChunk {
    data: Box<[u8]>,
    pts_ns: u64,
    keyframe: bool,
}

/// Live encoder plus the geometry it was configured with.
struct CodecState {
    codec: MediaCodec,
    width: u32,
    height: u32,
}

impl core::fmt::Debug for CodecState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CodecState")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

/// Android hardware video encoder: NV12 in, Annex-B H.264 / H.265 out.
#[derive(Debug)]
pub struct MediaCodecEnc {
    codec: VideoCodec,
    bitrate: u32,
    framerate: i32,
    keyframe_interval_s: i32,
    width: u32,
    height: u32,
    framerate_caps: Rate,
    configured: bool,
    state: Option<CodecState>,
    /// The captured codec-specific data (Annex-B SPS/PPS, or VPS+SPS+PPS for
    /// HEVC), prepended to every key frame.
    csd: Vec<u8>,
    last_caps: Option<Caps>,
    emitted: u64,
}

// SAFETY: like `MediaCodecDec`, the codec pointer is only ever touched from the
// element's owning task on a single-thread executor; we assert `Send` under that
// documented contract so the multi-thread runner accepts the element.
unsafe impl Send for MediaCodecEnc {}

impl Default for MediaCodecEnc {
    fn default() -> Self {
        Self::h264()
    }
}

impl MediaCodecEnc {
    /// An H.264 MediaCodec encoder.
    pub fn h264() -> Self {
        Self::new(VideoCodec::H264)
    }

    /// An H.265 / HEVC MediaCodec encoder.
    pub fn h265() -> Self {
        Self::new(VideoCodec::H265)
    }

    fn new(codec: VideoCodec) -> Self {
        Self {
            codec,
            bitrate: DEFAULT_BITRATE,
            framerate: DEFAULT_FRAMERATE,
            keyframe_interval_s: DEFAULT_KEYFRAME_INTERVAL_S,
            width: 0,
            height: 0,
            framerate_caps: Rate::Any,
            configured: false,
            state: None,
            csd: Vec::new(),
            last_caps: None,
            emitted: 0,
        }
    }

    /// Set the target bitrate in bits per second (default 4 Mbps).
    pub fn with_bitrate(mut self, bits_per_sec: u32) -> Self {
        self.bitrate = bits_per_sec;
        self
    }

    /// Set the frame-rate hint passed to the encoder (default 30).
    pub fn with_framerate(mut self, fps: i32) -> Self {
        self.framerate = fps.max(1);
        self
    }

    /// Set the key-frame interval in seconds (default 1; 0 = every frame is a key
    /// frame).
    pub fn with_keyframe_interval(mut self, seconds: i32) -> Self {
        self.keyframe_interval_s = seconds.max(0);
        self
    }

    /// The codec this encoder produces.
    pub fn codec(&self) -> VideoCodec {
        self.codec
    }

    /// Count of encoded access units pushed downstream. Useful in tests.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The MediaCodec MIME type for this element's codec.
    fn mime(&self) -> &'static str {
        match self.codec {
            VideoCodec::H265 => "video/hevc",
            _ => "video/avc",
        }
    }

    /// Create and start the encoder for the configured geometry.
    fn build_codec(&mut self) -> Result<(), G2gError> {
        if self.width == 0 || self.height == 0 {
            return Err(G2gError::NotConfigured);
        }
        let codec = MediaCodec::from_encoder_type(self.mime()).ok_or(G2gError::NotConfigured)?;

        // The frame-rate the caps pin (if any) overrides the hint, so the
        // encoder's rate control matches the real cadence.
        let fps = match self.framerate_caps {
            Rate::Fixed(q16) if q16 > 0 => ((q16 as u64 + (1 << 15)) >> 16) as i32,
            _ => self.framerate,
        }
        .max(1);

        let mut format = MediaFormat::new();
        format.set_str("mime", self.mime());
        format.set_i32("width", self.width as i32);
        format.set_i32("height", self.height as i32);
        format.set_i32("color-format", COLOR_FORMAT_NV12);
        format.set_i32("bitrate", self.bitrate as i32);
        format.set_i32("frame-rate", fps);
        format.set_i32("i-frame-interval", self.keyframe_interval_s);

        codec
            .configure(&format, None, MediaCodecDirection::Encoder)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        codec
            .start()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        self.state = Some(CodecState {
            codec,
            width: self.width,
            height: self.height,
        });
        Ok(())
    }

    /// Submit one NV12 frame, then drain whatever output is ready.
    fn feed(
        &mut self,
        nv12: &[u8],
        pts_ns: u64,
        out: &mut Vec<EncodedChunk>,
    ) -> Result<(), G2gError> {
        // A tight NV12 frame is width*height (Y) + width*height/2 (interleaved UV).
        let expected = (self.width as usize * self.height as usize * 3) / 2;
        if nv12.len() < expected {
            return Err(G2gError::CapsMismatch);
        }
        self.queue_input(&nv12[..expected], pts_ns / 1000, 0)?;
        self.pump_output(out, false)
    }

    /// Hand `data` to a free input buffer with the given microsecond pts + flags
    /// (see `mediacodec_common::queue_input`).
    fn queue_input(&self, data: &[u8], pts_us: u64, flags: u32) -> Result<(), G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        queue_input(&st.codec, data, pts_us, flags)
    }

    /// Drain ready output buffers. In steady state (`until_eos == false`) makes
    /// one non-blocking pass; at EOS polls (bounded) until the codec raises the
    /// end-of-stream flag. The codec-config buffer is captured as the parameter
    /// sets; each key frame gets them prepended so the stream is self-contained.
    fn pump_output(
        &mut self,
        out: &mut Vec<EncodedChunk>,
        until_eos: bool,
    ) -> Result<(), G2gError> {
        let timeout = if until_eos {
            Duration::from_millis(20)
        } else {
            Duration::ZERO
        };
        for _ in 0..MAX_OUTPUT_POLLS {
            let mut got = false;
            let mut eos = false;
            {
                let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
                match st
                    .codec
                    .dequeue_output_buffer(timeout)
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?
                {
                    DequeuedOutputBufferInfoResult::Buffer(buffer) => {
                        got = true;
                        let info = *buffer.info();
                        let flags = info.flags();
                        eos = flags & BUFFER_FLAG_END_OF_STREAM != 0;
                        let off = info.offset().max(0) as usize;
                        let size = info.size().max(0) as usize;
                        let bytes = buffer.buffer();
                        let au = bytes.get(off..off + size).unwrap_or(&[]);
                        if flags & BUFFER_FLAG_CODEC_CONFIG != 0 {
                            // Parameter sets (Annex-B): keep, do not emit.
                            self.csd = au.to_vec();
                        } else if size > 0 {
                            let keyframe = flags & BUFFER_FLAG_KEY_FRAME != 0;
                            // Prepend the captured parameter sets to a key frame,
                            // but only if the encoder did not already inline them
                            // (some repeat SPS/PPS on each IDR, some emit them only
                            // once via the codec-config buffer; on-device the Pixel
                            // encoder inlines them, so avoid the duplicate).
                            let prepend = keyframe
                                && !self.csd.is_empty()
                                && !starts_with_parameter_set(self.codec, au);
                            let mut data = Vec::with_capacity(self.csd.len() + size);
                            if prepend {
                                data.extend_from_slice(&self.csd);
                            }
                            data.extend_from_slice(au);
                            let pts_ns = (info.presentation_time_us().max(0) as u64) * 1000;
                            out.push(EncodedChunk {
                                data: data.into_boxed_slice(),
                                pts_ns,
                                keyframe,
                            });
                        }
                        st.codec
                            .release_output_buffer(buffer, false)
                            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    }
                    DequeuedOutputBufferInfoResult::OutputFormatChanged
                    | DequeuedOutputBufferInfoResult::OutputBuffersChanged => got = true,
                    DequeuedOutputBufferInfoResult::TryAgainLater => {}
                }
            }
            if eos {
                break;
            }
            if !until_eos && !got {
                break;
            }
        }
        Ok(())
    }
}

impl AsyncElement for MediaCodecEnc {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let codec = self.codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            derive_output_caps(codec, input)
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h, framerate) = match absolute_caps {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                framerate,
            } => (*w, *h, framerate.clone()),
            _ => return Err(G2gError::CapsMismatch),
        };
        if w == 0 || h == 0 || w % 2 != 0 || h % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }
        self.width = w;
        self.height = h;
        self.framerate_caps = framerate;
        self.build_codec()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("bitrate", PropKind::Uint, "target bitrate, bits/second")
                .with_default("4000000"),
            PropertySpec::new("framerate", PropKind::Uint, "encode frame rate, fps")
                .with_default("30"),
            PropertySpec::new(
                "i-frame-interval",
                PropKind::Uint,
                "keyframe interval, seconds",
            )
            .with_default("1"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "bitrate" => {
                self.bitrate = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "framerate" => {
                self.framerate = (value.as_uint().ok_or(PropError::Type)? as i32).max(1);
                Ok(())
            }
            "i-frame-interval" => {
                self.keyframe_interval_s = value.as_uint().ok_or(PropError::Type)? as i32;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "bitrate" => Some(PropValue::Uint(self.bitrate as u64)),
            "framerate" => Some(PropValue::Uint(self.framerate.max(0) as u64)),
            "i-frame-interval" => Some(PropValue::Uint(self.keyframe_interval_s.max(0) as u64)),
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
            let mut encoded = Vec::new();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed(slice.as_slice(), frame.timing.pts_ns, &mut encoded)?;
                }
                PipelinePacket::CapsChanged(c) => {
                    match &c {
                        Caps::RawVideo {
                            format: RawVideoFormat::Nv12,
                            ..
                        } => {}
                        // The runner's pre-fixed output caps (our compressed
                        // codec): forward so the sink sees them before the first
                        // access unit (M733/M734, see `ffmpegdec.rs`).
                        Caps::CompressedVideo { codec, .. } if *codec == self.codec => {
                            out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                            self.last_caps = Some(c);
                        }
                        _ => return Err(G2gError::CapsMismatch),
                    }
                    // Geometry / rate changes would need a codec rebuild; v1 keeps
                    // the configured geometry (the runner pins it at startup).
                }
                PipelinePacket::Flush => {
                    if let Some(st) = self.state.as_ref() {
                        let _ = st.codec.flush();
                    }
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    if self.state.is_some() {
                        let _ = self.queue_input(&[], 0, BUFFER_FLAG_END_OF_STREAM);
                        self.pump_output(&mut encoded, true)?;
                    }
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                    return Ok(());
                }
                other => {
                    out.push(other).await?;
                    return Ok(());
                }
            }

            // Never emit an `Any` rate (a downstream transform cannot fixate()
            // it); default to 30/1 when the input caps did not declare one.
            let framerate = match &self.framerate_caps {
                Rate::Fixed(q) => Rate::Fixed(*q),
                _ => Rate::Fixed(30 << 16),
            };
            let new_caps = compressed_caps(self.codec, self.width, self.height, &framerate);
            if self.last_caps.as_ref() != Some(&new_caps) {
                out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                    .await?;
                self.last_caps = Some(new_caps);
            }
            for chunk in encoded {
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(chunk.data)),
                    timing: FrameTiming {
                        pts_ns: chunk.pts_ns,
                        dts_ns: chunk.pts_ns,
                        duration_ns: 0,
                        capture_ns: chunk.pts_ns,
                        keyframe: chunk.keyframe,
                        ..FrameTiming::default()
                    },
                    sequence: self.emitted,
                    meta: Default::default(),
                };
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
        })
    }
}

impl PadTemplates for MediaCodecEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let compressed = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(nv12)),
            PadTemplate::source(CapsSet::from_alternatives(Vec::from([
                compressed(VideoCodec::H264),
                compressed(VideoCodec::H265),
            ]))),
        ])
    }
}

fn compressed_caps(codec: VideoCodec, w: u32, h: u32, framerate: &Rate) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: framerate.clone(),
    }
}

/// Whether `au` (an Annex-B access unit) already begins with a parameter-set NAL,
/// so the encoder inlined the codec config on this key frame and the captured
/// `csd` must not be prepended again (avoids duplicate SPS/PPS).
fn starts_with_parameter_set(codec: VideoCodec, au: &[u8]) -> bool {
    let nal = if au.starts_with(&[0, 0, 0, 1]) {
        &au[4..]
    } else if au.starts_with(&[0, 0, 1]) {
        &au[3..]
    } else {
        au
    };
    let Some(&b) = nal.first() else { return false };
    match codec {
        // H.265 NAL type is bits 6..1; VPS=32, SPS=33, PPS=34.
        VideoCodec::H265 => matches!((b >> 1) & 0x3f, 32..=34),
        // H.264 NAL type is the low 5 bits; SPS=7, PPS=8.
        _ => matches!(b & 0x1f, 7 | 8),
    }
}

fn derive_output_caps(codec: VideoCodec, input: &Caps) -> CapsSet {
    match input {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width,
            height,
            framerate,
        } => CapsSet::one(Caps::CompressedVideo {
            codec,
            width: width.clone(),
            height: height.clone(),
            framerate: framerate.clone(),
        }),
        _ => CapsSet::from_alternatives(Vec::new()),
    }
}
