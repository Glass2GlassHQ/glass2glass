//! Camera capture source (browser/wasm). Opens a webcam via `getUserMedia`,
//! reads the track's frames through a `MediaStreamTrackProcessor`, and emits each
//! as an RGBA `System` `DataFrame`: the real "capture" side of the browser egress
//! pipeline (`WebCameraSrc -> WebCodecsEncode -> WebSocketSink`), replacing the
//! synthetic `PatternSrc`.
//!
//! `MediaStreamTrackProcessor.readable` is a `ReadableStream<VideoFrame>`; the
//! `run` loop reads one `VideoFrame` at a time (each `read()` is a promise, awaited
//! like the WebCodecs decode path) and copies it out to packed RGBA with the shared
//! `copy_out_rgba` helper, so the frames flow straight into `WebCodecsEncode`.
//!
//! Requires a secure context + camera permission (`getUserMedia` is gated), so it
//! is not headless-testable without a fake device (`--use-fake-device-for-media-stream`).
//! The requested resolution is advertised as the source caps; a browser that hands
//! back a different size still works (real dims land mid-stream via `CapsChanged`),
//! though the paired encoder is sized from the request. Build requires
//! `--cfg=web_sys_unstable_apis` (WebCodecs `VideoFrame` + mediacapture-transform).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, HardwareError,
    MemoryDomain, OutputSink, PipelinePacket, Rate, RawVideoFormat,
};

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    MediaStream, MediaStreamConstraints, MediaStreamTrack, MediaStreamTrackProcessor,
    MediaStreamTrackProcessorInit, ReadableStreamDefaultReader, VideoFrame,
};

use crate::webcodecsdecode::copy_out_rgba;

#[derive(Debug)]
pub struct WebCameraSrc {
    width: u32,
    height: u32,
    /// Frames to emit before EOS (0 = run until the track ends / is stopped).
    frames: u64,
    configured: bool,
    last_dims: Option<(u32, u32)>,
}

impl WebCameraSrc {
    /// Request a `width` x `height` camera stream, emitting `frames` frames then
    /// EOS (0 = until the track ends).
    pub fn new(width: u32, height: u32, frames: u64) -> Self {
        Self { width, height, frames, configured: false, last_dims: None }
    }

    fn caps(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            // A fixable placeholder, not `Rate::Any` (which `fixate()` rejects at
            // negotiation): the true capture rate is not known until frames arrive,
            // and the paired encoder just needs a sane rate to configure with.
            framerate: Rate::Fixed(30 << 16),
        }
    }

    /// Announce the current output caps if the frame geometry changed. Mirrors
    /// `WebCodecsDecode`: the real dims come from the delivered `VideoFrame`, which
    /// may differ from the requested resolution.
    async fn announce(&mut self, w: u32, h: u32, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        if self.last_dims != Some((w, h)) {
            let caps = Caps::RawVideo {
                format: RawVideoFormat::Rgba8,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                framerate: Rate::Fixed(30 << 16),
            };
            out.push(PipelinePacket::CapsChanged(caps)).await?;
            self.last_dims = Some((w, h));
        }
        Ok(())
    }
}

impl SourceLoop for WebCameraSrc {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let hw = || G2gError::Hardware(HardwareError::Other);

            // getUserMedia({ video: { width, height } }). The constraints are a plain
            // JS object (web-sys's typed MediaTrackConstraints would need extra
            // feature flags for no gain here).
            let devices = web_sys::window()
                .ok_or_else(hw)?
                .navigator()
                .media_devices()
                .map_err(|_| hw())?;
            let video = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&video, &JsValue::from_str("width"), &JsValue::from_f64(self.width as f64));
            let _ = js_sys::Reflect::set(&video, &JsValue::from_str("height"), &JsValue::from_f64(self.height as f64));
            let constraints = MediaStreamConstraints::new();
            constraints.set_video(&video);
            let stream: MediaStream =
                JsFuture::from(devices.get_user_media_with_constraints(&constraints).map_err(|_| hw())?)
                    .await
                    .map_err(|_| hw())?
                    .dyn_into()
                    .map_err(|_| hw())?;

            let track: MediaStreamTrack =
                stream.get_video_tracks().get(0).dyn_into().map_err(|_| hw())?;

            // MediaStreamTrackProcessor exposes the track as a ReadableStream<VideoFrame>.
            let init = MediaStreamTrackProcessorInit::new(&track);
            let processor = MediaStreamTrackProcessor::new(&init).map_err(|_| hw())?;
            let reader: ReadableStreamDefaultReader =
                processor.readable().get_reader().dyn_into().map_err(|_| hw())?;

            let mut sequence = 0u64;
            let result = async {
                loop {
                    // { value: VideoFrame, done: bool }
                    let res = JsFuture::from(reader.read()).await.map_err(|_| hw())?;
                    let done = js_sys::Reflect::get(&res, &JsValue::from_str("done"))
                        .map(|v| v.as_bool().unwrap_or(false))
                        .unwrap_or(true);
                    if done {
                        break;
                    }
                    let value = js_sys::Reflect::get(&res, &JsValue::from_str("value")).map_err(|_| hw())?;
                    let frame: VideoFrame = value.dyn_into().map_err(|_| hw())?;

                    let (bytes, w, h, pts_ns) = copy_out_rgba(&frame).await?;
                    frame.close();

                    self.announce(w, h, out).await?;
                    let out_frame = Frame {
                        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                        timing: FrameTiming { pts_ns, dts_ns: pts_ns, capture_ns: pts_ns, ..FrameTiming::default() },
                        sequence,
                        meta: Default::default(),
                    };
                    sequence += 1;
                    out.push(PipelinePacket::DataFrame(out_frame)).await?;

                    if self.frames != 0 && sequence >= self.frames {
                        break;
                    }
                }
                Ok::<u64, G2gError>(sequence)
            }
            .await;

            // Always release the camera, even on error, then propagate EOS.
            track.stop();
            let count = result?;
            out.push(PipelinePacket::Eos).await?;
            Ok(count)
        })
    }
}
