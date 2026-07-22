//! WebCodecs hardware video encode element (browser/wasm). Wraps the browser
//! `VideoEncoder`, consuming raw RGBA `System` frames and producing H.264 Annex-B
//! access units (`CompressedVideo`, `System`): the browser analog of the native
//! encoders, and the send side of the browser pipeline
//! (`PatternSrc -> WebCodecsEncode -> WebSocketSink`).
//!
//! Output is Annex-B (SPS/PPS in-band on keyframes), forced via the encoder's
//! `avc: { format: "annexb" }` config, so the byte stream is self-contained (a
//! plain `.h264` a `WebCodecsDecode` / ffmpeg can consume without a separate
//! description). Async shape mirrors `WebCodecsDecode`: `encode()` queues work and
//! the browser delivers `EncodedVideoChunk`s later through the output callback,
//! bridged to `process` by an [`crate::webutil::Inbox`].
//!
//! Build requires `--cfg=web_sys_unstable_apis`. H.264 only for now.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate,
    RawVideoFormat, VideoCodec,
};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    EncodedVideoChunk, VideoEncoder, VideoEncoderConfig, VideoEncoderInit, VideoFrame,
    VideoFrameBufferInit, VideoPixelFormat,
};

use crate::webutil::Inbox;

pub struct WebCodecsEncode {
    width: u32,
    height: u32,
    fps: u32,
    configured: bool,
    encoder: Option<VideoEncoder>,
    inbox: Option<Inbox<Vec<u8>>>,
    _on_output: Option<Closure<dyn FnMut(EncodedVideoChunk, JsValue)>>,
    _on_error: Option<Closure<dyn FnMut(JsValue)>>,
    last_caps: Option<Caps>,
    emitted: u64,
}

impl Default for WebCodecsEncode {
    fn default() -> Self {
        Self::new()
    }
}

impl WebCodecsEncode {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            fps: 30,
            configured: false,
            encoder: None,
            inbox: None,
            _on_output: None,
            _on_error: None,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Count of encoded access units pushed downstream. Useful in tests.
    pub fn encoded_count(&self) -> u64 {
        self.emitted
    }

    /// Wrap one RGBA `System` frame as a browser `VideoFrame` and queue it for
    /// encode. The pixel buffer is copied into a JS `Uint8Array` (a wasm-memory
    /// slice could dangle across the encode's internal async work).
    fn feed(&self, bytes: &[u8], pts_ns: u64) -> Result<(), G2gError> {
        let encoder = self.encoder.as_ref().ok_or(G2gError::NotConfigured)?;
        let data = js_sys::Uint8Array::new_with_length(bytes.len() as u32);
        data.copy_from(bytes);
        // WebCodecs timestamps are microseconds.
        let init = VideoFrameBufferInit::new(
            self.height,
            self.width,
            VideoPixelFormat::Rgba,
            (pts_ns / 1000) as i32,
        );
        let frame = VideoFrame::new_with_u8_array_and_video_frame_buffer_init(&data, &init)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let r = encoder.encode(&frame);
        frame.close();
        r.map_err(|_| G2gError::Hardware(HardwareError::Other))
    }

    /// Drain every encoded access unit currently in the inbox, pushing each
    /// downstream (with a leading `CapsChanged`).
    async fn drain_ready(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        while let Some(bytes) = self.inbox.as_ref().and_then(|i| i.try_pop()) {
            self.announce_caps(out).await?;
            let frame = Frame {
                domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                timing: FrameTiming::default(),
                sequence: self.emitted,
                meta: Default::default(),
            };
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }

    async fn announce_caps(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let caps = h264_caps(self.width, self.height, self.fps);
        if self.last_caps.as_ref() != Some(&caps) {
            out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
            self.last_caps = Some(caps);
        }
        Ok(())
    }
}

impl AsyncElement for WebCodecsEncode {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&rgba_any())
    }

    /// Native `DerivedOutput`: RGBA at any geometry -> H.264 at the same
    /// dims/framerate (mirror of `WebCodecsDecode` in reverse).
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(derive_output_caps))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h, fps) = match absolute_caps {
            Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width,
                height,
                framerate,
            } => (
                fixed_or_zero(width),
                fixed_or_zero(height),
                rate_fps(framerate),
            ),
            _ => return Err(G2gError::CapsMismatch),
        };
        if w == 0 || h == 0 {
            return Err(G2gError::CapsMismatch);
        }
        self.width = w;
        self.height = h;
        self.fps = fps;

        let inbox: Inbox<Vec<u8>> = Inbox::new();
        let on_output = {
            let tx = inbox.sender();
            Closure::<dyn FnMut(EncodedVideoChunk, JsValue)>::new(
                move |chunk: EncodedVideoChunk, _meta: JsValue| {
                    let len = chunk.byte_length();
                    let buf = js_sys::Uint8Array::new_with_length(len);
                    if chunk.copy_to_with_u8_array(&buf).is_ok() {
                        tx.push(buf.to_vec());
                    }
                },
            )
        };
        let on_error = {
            let tx = inbox.sender();
            Closure::<dyn FnMut(JsValue)>::new(move |_e: JsValue| tx.close())
        };
        let init = VideoEncoderInit::new(
            on_error.as_ref().unchecked_ref(),
            on_output.as_ref().unchecked_ref(),
        );
        let encoder =
            VideoEncoder::new(&init).map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // avc1.42001f = constrained baseline, widely supported. Force Annex-B output
        // (SPS/PPS in-band) via the `avc` member, which web-sys has no typed setter
        // for, so set it by reflection.
        let config = VideoEncoderConfig::new("avc1.42001f", h, w);
        config.set_bitrate(2_000_000);
        config.set_framerate(fps as f64);
        let avc = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &avc,
            &JsValue::from_str("format"),
            &JsValue::from_str("annexb"),
        );
        let _ = js_sys::Reflect::set(config.as_ref(), &JsValue::from_str("avc"), &avc);
        encoder
            .configure(&config)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        self.inbox = Some(inbox);
        self._on_output = Some(on_output);
        self._on_error = Some(on_error);
        self.encoder = Some(encoder);
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed(slice, frame.timing.pts_ns)?;
                    self.drain_ready(out).await?;
                }
                // The runner's transform arm hands us our *derived output* caps
                // (H.264), not the RGBA input, and expects the element to re-announce
                // its own output caps from real encoded data (which `drain_ready` does
                // via `announce_caps`). Input validation lives in `configure_pipeline`,
                // so swallow this like `FfmpegEnc` rather than rejecting it.
                PipelinePacket::CapsChanged(_) => {}
                PipelinePacket::Eos => {
                    if let Some(enc) = self.encoder.as_ref() {
                        // flush() resolves once every queued frame has been emitted.
                        let p: js_sys::Promise = enc.flush().unchecked_into();
                        let _ = JsFuture::from(p).await;
                    }
                    self.drain_ready(out).await?;
                    out.push(PipelinePacket::Eos).await?;
                    return Ok(());
                }
                PipelinePacket::Flush => {
                    if let Some(enc) = self.encoder.as_ref() {
                        let _ = enc.reset();
                    }
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                // future PipelinePacket variants: forward unchanged.
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for WebCodecsEncode {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::one(rgba_any())),
            PadTemplate::source(CapsSet::one(h264_any())),
        ])
    }
}

impl core::fmt::Debug for WebCodecsEncode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WebCodecsEncode")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("fps", &self.fps)
            .field("configured", &self.configured)
            .field("emitted", &self.emitted)
            .finish_non_exhaustive()
    }
}

fn derive_output_caps(input: &Caps) -> CapsSet {
    match input {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
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
    }
}

fn rgba_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn h264_any() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

fn h264_caps(w: u32, h: u32, fps: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(fps << 16),
    }
}

fn fixed_or_zero(d: &Dim) -> u32 {
    match d {
        Dim::Fixed(v) => *v,
        _ => 0,
    }
}

/// Integer fps from a Q16.16 `Rate::Fixed`, defaulting to 30 for a non-fixed rate.
fn rate_fps(r: &Rate) -> u32 {
    match r {
        Rate::Fixed(q16) => (q16 >> 16).max(1),
        _ => 30,
    }
}
