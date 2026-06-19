//! WebCodecs hardware video decode element (browser/wasm). Wraps the browser
//! `VideoDecoder`, consuming Annex-B H.264 access units and producing decoded
//! RGBA frames in the system-memory domain: the browser analog of `MfDecode`
//! (Windows) and `FfmpegH264Dec` (Linux). M40.
//!
//! Output is RGBA, not the decoder's native YUV: `VideoFrame.copyTo` is asked
//! to convert via `VideoFrameCopyToOptions::format`, so negotiation fixates one
//! deterministic output format that pairs with the RGBA-consuming elements
//! (e.g. `OrtInference`). A tightly-packed RGBA copy-out is assumed; row-stride
//! de-padding and visible-rect cropping are documented follow-ups.
//!
//! `with_gpu_output()` instead hands the decoded `VideoFrame` forward in
//! `MemoryDomain::WebGPUExternalTexture` for a zero-copy WebGPU import; the
//! frame stays open until the keep-alive owner drops downstream (a
//! VideoFrame-sourced external texture is valid until the frame is closed).
//! Default stays system RGBA so the CPU-consumer sinks are unaffected. (P2.2)
//!
//! Async shape (unlike the synchronous MFT / libav decoders): `decode()` queues
//! work and the browser delivers `VideoFrame`s later through the decoder's
//! output callback, bridged to the `run` loop by a [`crate::webutil::Inbox`].
//! Each `process(DataFrame)` feeds one chunk and drains whatever frames are
//! ready; `process(Eos)` awaits `flush()` then drains the reorder tail.
//!
//! Build requires `--cfg=web_sys_unstable_apis` (the WebCodecs web-sys bindings
//! are unstable). H.264 only for M40; the HEVC `codec` string is a follow-up.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, OwnedWebGPUExternalTexture, PadTemplate, PadTemplates,
    PipelinePacket, Rate, RawVideoFormat, VideoCodec, WebGPUKeepAlive,
};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    EncodedVideoChunk, EncodedVideoChunkInit, EncodedVideoChunkType, VideoDecoder,
    VideoDecoderConfig, VideoDecoderInit, VideoFrame, VideoFrameCopyToOptions, VideoPixelFormat,
};

use crate::h264util::{h264_au_is_keyframe, h264_codec_string};
use crate::webutil::Inbox;

pub struct WebCodecsDecode {
    codec: VideoCodec,
    /// When set, hand the `VideoFrame` forward as a GPU-resident external
    /// texture instead of copying it out to system RGBA.
    gpu_output: bool,
    configured: bool,
    width: u32,
    height: u32,
    decoder: Option<VideoDecoder>,
    inbox: Option<Inbox<VideoFrame>>,
    // Kept alive for the decoder's lifetime; JS holds raw references to them.
    _on_output: Option<Closure<dyn FnMut(VideoFrame)>>,
    _on_error: Option<Closure<dyn FnMut(JsValue)>>,
    /// Whether `decoder.configure()` has run. Deferred to the first access unit
    /// because the WebCodecs `codec` string is derived from the in-band SPS.
    decoder_configured: bool,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for WebCodecsDecode {
    fn default() -> Self {
        Self::new()
    }
}

impl WebCodecsDecode {
    pub fn new() -> Self {
        Self {
            codec: VideoCodec::H264,
            gpu_output: false,
            configured: false,
            width: 0,
            height: 0,
            decoder: None,
            inbox: None,
            _on_output: None,
            _on_error: None,
            decoder_configured: false,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Emit decoded frames as GPU-resident `WebGPUExternalTexture` for a
    /// downstream WebGPU import, instead of the default system-RGBA copy-out.
    pub fn with_gpu_output(mut self) -> Self {
        self.gpu_output = true;
        self
    }

    /// Count of decoded `DataFrame`s pushed downstream. Useful in tests.
    pub fn decoded_count(&self) -> u64 {
        self.emitted
    }

    /// Queue one access unit for decode. Configuration is lazy: the first AU
    /// carrying an SPS supplies the `codec` string; AUs before that are
    /// undecodable and skipped.
    fn feed(&mut self, au: &[u8], pts_ns: u64) -> Result<(), G2gError> {
        let decoder = self.decoder.as_ref().ok_or(G2gError::NotConfigured)?;
        if !self.decoder_configured {
            let Some(codec_str) = h264_codec_string(au) else {
                return Ok(()); // no SPS yet: cannot configure, skip until a keyframe
            };
            let config = VideoDecoderConfig::new(&codec_str);
            if self.width != 0 && self.height != 0 {
                config.set_coded_width(self.width);
                config.set_coded_height(self.height);
            }
            decoder
                .configure(&config)
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
            self.decoder_configured = true;
        }

        let data = js_sys::Uint8Array::new_with_length(au.len() as u32);
        data.copy_from(au);
        let chunk_type = if h264_au_is_keyframe(au) {
            EncodedVideoChunkType::Key
        } else {
            EncodedVideoChunkType::Delta
        };
        // WebCodecs timestamps are microseconds (i32); clamp so a long stream
        // can't wrap into a negative timestamp the decoder would reject.
        let timestamp_us = (pts_ns / 1000).min(i32::MAX as u64) as i32;
        let init = EncodedVideoChunkInit::new_with_u8_array(&data, timestamp_us, chunk_type);
        let chunk =
            EncodedVideoChunk::new(&init).map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        decoder
            .decode(&chunk)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        Ok(())
    }

    /// Drain every decoded frame currently in the inbox, pushing each
    /// downstream (with a `CapsChanged` on the first frame and on any geometry
    /// change). Each frame goes out as GPU-resident external texture or
    /// system RGBA per the configured output mode.
    async fn drain_ready(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        while let Some(frame) = self.inbox.as_ref().and_then(|i| i.try_pop()) {
            if self.gpu_output {
                self.emit_external_texture(frame, out).await?;
            } else {
                self.emit_system_rgba(frame, out).await?;
            }
        }
        Ok(())
    }

    /// Announce the output caps if they changed since the last frame.
    async fn announce_caps(&mut self, w: u32, h: u32, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let new_caps = rgba_caps(w, h);
        if self.last_caps.as_ref() != Some(&new_caps) {
            out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
            self.last_caps = Some(new_caps);
        }
        Ok(())
    }

    /// Copy the decoded frame out to system RGBA (the CPU-consumer path).
    async fn emit_system_rgba(&mut self, frame: VideoFrame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let (bytes, w, h, pts_ns) = copy_out_rgba(&frame).await?;
        frame.close();
        self.announce_caps(w, h, out).await?;
        let out_frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
            timing: FrameTiming { pts_ns, dts_ns: pts_ns, capture_ns: pts_ns, ..FrameTiming::default() },
            sequence: self.emitted,
            meta: Default::default(),
        };
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(out_frame)).await?;
        Ok(())
    }

    /// Hand the decoded frame forward as a GPU-resident external texture. The
    /// frame is not closed here: a VideoFrame-sourced external texture is valid
    /// until the frame closes, so the keep-alive owner closes it on drop once
    /// downstream has imported and used it.
    async fn emit_external_texture(&mut self, frame: VideoFrame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let (w, h) = (frame.coded_width(), frame.coded_height());
        let pts_ns = (frame.timestamp().max(0.0) as u64).saturating_mul(1000);
        self.announce_caps(w, h, out).await?;
        let domain = MemoryDomain::WebGPUExternalTexture(OwnedWebGPUExternalTexture::new(
            w,
            h,
            Box::new(VideoFrameOwner::new(frame)),
        ));
        let out_frame = Frame {
            domain,
            timing: FrameTiming { pts_ns, dts_ns: pts_ns, capture_ns: pts_ns, ..FrameTiming::default() },
            sequence: self.emitted,
            meta: Default::default(),
        };
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(out_frame)).await?;
        Ok(())
    }
}

impl AsyncElement for WebCodecsDecode {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = Caps::CompressedVideo {
            codec: self.codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    /// Native `DerivedOutput`: accepts H.264 at any geometry and produces RGBA
    /// at the same dims/framerate, mirroring `MfDecode` (which emits NV12). The
    /// closure rejects a non-matching codec with an empty set so the solver
    /// fails non-H.264 upstream at negotiation time.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let codec = self.codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| derive_output_caps(codec, input)))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = match absolute_caps {
            Caps::CompressedVideo { codec, width, height, .. } if *codec == self.codec => {
                (fixed_or_zero(width), fixed_or_zero(height))
            }
            _ => return Err(G2gError::CapsMismatch),
        };
        // Only H.264 has a `codec`-string builder wired for M40.
        if self.codec != VideoCodec::H264 {
            return Err(G2gError::CapsMismatch);
        }
        self.width = w;
        self.height = h;

        let inbox: Inbox<VideoFrame> = Inbox::new();
        let on_output = {
            let tx = inbox.sender();
            Closure::<dyn FnMut(VideoFrame)>::new(move |frame: VideoFrame| tx.push(frame))
        };
        let on_error = {
            let tx = inbox.sender();
            Closure::<dyn FnMut(JsValue)>::new(move |_e: JsValue| tx.close())
        };
        let init = VideoDecoderInit::new(
            on_error.as_ref().unchecked_ref(),
            on_output.as_ref().unchecked_ref(),
        );
        let decoder =
            VideoDecoder::new(&init).map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        self.inbox = Some(inbox);
        self._on_output = Some(on_output);
        self._on_error = Some(on_error);
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed(slice.as_slice(), frame.timing.pts_ns)?;
                    self.drain_ready(out).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // Reject an incompatible mid-stream codec swap loud; a
                    // geometry-change reconfigure is a follow-up.
                    match &c {
                        Caps::CompressedVideo { codec, .. } if *codec == self.codec => {}
                        _ => return Err(G2gError::CapsMismatch),
                    }
                }
                PipelinePacket::Flush => {
                    if let Some(d) = self.decoder.as_ref() {
                        d.reset()
                            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    }
                    // After a flush the next AU re-supplies the SPS/config.
                    self.decoder_configured = false;
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    if self.decoder_configured {
                        if let Some(d) = self.decoder.as_ref() {
                            // flush() resolves once every queued decode has been
                            // delivered to the output callback (reorder tail).
                            let p: js_sys::Promise = d.flush().unchecked_into();
                            let _ = JsFuture::from(p).await;
                        }
                    }
                    self.drain_ready(out).await?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for WebCodecsDecode {
    /// Consumes H.264 and produces RGBA, both at any geometry (the decoder
    /// derives the output dims from the stream). The memory domain (System) is
    /// not encoded in caps.
    fn pad_templates() -> Vec<PadTemplate> {
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let rgba = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264)),
            PadTemplate::source(CapsSet::one(rgba)),
        ])
    }
}

impl core::fmt::Debug for WebCodecsDecode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebCodecsDecode")
            .field("codec", &self.codec)
            .field("configured", &self.configured)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("decoder_configured", &self.decoder_configured)
            .field("emitted", &self.emitted)
            .finish_non_exhaustive()
    }
}

/// Copy one decoded `VideoFrame` out as packed RGBA: `(bytes, width, height,
/// pts_ns)`. `copyTo` is asked to convert to RGBA; the copy targets a JS
/// `Uint8Array` (not a wasm-memory slice) because the await can outlive a
/// linear-memory grow that would dangle a raw slice view.
async fn copy_out_rgba(frame: &VideoFrame) -> Result<(Vec<u8>, u32, u32, u64), G2gError> {
    let w = frame.coded_width();
    let h = frame.coded_height();
    // VideoFrame timestamps are microseconds; map back to ns.
    let pts_ns = (frame.timestamp().max(0.0) as u64).saturating_mul(1000);

    let opts = VideoFrameCopyToOptions::new();
    opts.set_format(VideoPixelFormat::Rgba);
    let size = frame
        .allocation_size_with_options(&opts)
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

    let buf = js_sys::Uint8Array::new_with_length(size);
    let promise: js_sys::Promise = frame
        .copy_to_with_u8_array_and_options(&buf, &opts)
        .unchecked_into();
    JsFuture::from(promise)
        .await
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

    Ok((buf.to_vec(), w, h, pts_ns))
}

fn rgba_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

/// Output-side caps derivation: H.264 at any geometry maps to RGBA at the same
/// dims/framerate; a non-matching codec yields an empty set so the solver
/// rejects it. Shared by the `DerivedOutput` constraint closure.
fn derive_output_caps(codec: VideoCodec, input: &Caps) -> CapsSet {
    match input {
        Caps::CompressedVideo { codec: c, width, height, framerate } if *c == codec => {
            CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            })
        }
        _ => CapsSet::from_alternatives(Vec::new()),
    }
}

fn fixed_or_zero(d: &Dim) -> u32 {
    match d {
        Dim::Fixed(v) => *v,
        _ => 0,
    }
}

/// Owns a decoded `VideoFrame` handed forward as a
/// [`MemoryDomain::WebGPUExternalTexture`]. A downstream element downcasts the
/// [`WebGPUKeepAlive`] (via `as_any`) to recover the frame for
/// `importExternalTexture`; dropping this owner closes the frame and frees the
/// decoder's output slot.
pub struct VideoFrameOwner {
    frame: VideoFrame,
}

impl VideoFrameOwner {
    fn new(frame: VideoFrame) -> Self {
        Self { frame }
    }

    /// The backing `VideoFrame`, for a consumer to import into WebGPU.
    pub fn frame(&self) -> &VideoFrame {
        &self.frame
    }
}

impl Drop for VideoFrameOwner {
    fn drop(&mut self) {
        self.frame.close();
    }
}

impl core::fmt::Debug for VideoFrameOwner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VideoFrameOwner").finish_non_exhaustive()
    }
}

impl WebGPUKeepAlive for VideoFrameOwner {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

// SAFETY: wasm32-unknown-unknown is single-threaded (built without atomics), so
// the VideoFrame never crosses a thread. Asserting Send satisfies the
// WebGPUKeepAlive contract, matching the D3D11KeepAlive / MfDecode precedent.
unsafe impl Send for VideoFrameOwner {}
