//! M219: Android hardware H.264 decode via the NDK MediaCodec (`AMediaCodec`).
//!
//! `MediaCodecDec` is the Android counterpart of `VtDecode` (macOS VideoToolbox)
//! and `MfDecode` (Windows Media Foundation): it consumes Annex-B H.264
//! `DataFrame`s (`MemoryDomain::System`, what `RtspSrc` / `H264Parse` emit) and
//! produces decoded NV12 frames, also `MemoryDomain::System`. It is the first
//! element of the Android platform track (DESIGN_TODO.md "Platform: Android"); a
//! zero-copy `AHardwareBuffer` / `SurfaceTexture` path is the follow-up.
//!
//! Unlike VideoToolbox (which wants AVCC + out-of-band parameter sets),
//! MediaCodec takes the access units as Annex-B directly and the SPS/PPS as
//! `csd-0` / `csd-1` buffers in the `MediaFormat`. So the element reuses
//! [`crate::annexb::h264_parameter_sets`] for the codec-specific data but feeds
//! each frame's bytes unchanged (no AVCC conversion). It drives the codec
//! synchronously (queue one input buffer, render+drain ready output), wrapping
//! the safe `ndk` crate rather than raw FFI.
//!
//! **Output via an `ImageReader` Surface.** Modern vendor decoders only deliver
//! decoded frames as graphic (gralloc) buffers; configuring MediaCodec with no
//! Surface and reading ByteBuffers stalls them (validated on a Pixel: the codec's
//! graphic-block handoff corrupts and a binder call hangs ~23 s). So the codec
//! is configured to render into an `ImageReader`'s Surface; each output buffer is
//! released with `render=true`, then the decoded `Image` is acquired and its
//! `YUV_420_888` planes (whose row/pixel strides describe any vendor layout) are
//! packed to NV12. This needs the frame geometry up front (the `ImageReader` and
//! MediaCodec both require it): it arrives via `configure_pipeline`, as
//! `h264parse` supplies it from the SPS.
//!
//! **Binder threadpool (headless callers).** Codec2 allocates the decoder's
//! output graphic buffers by calling back into this process over binder. An
//! Android app gets a binder threadpool from the framework, but a bare native
//! process (a test binary, a CLI) has none, so the allocation transaction stalls.
//! Such a process must start one (`ABinderProcess_startThreadPool`) before
//! decoding; see `tests/android_mediacodec_smoke.rs` for the dlsym helper.
//!
//! Built against the `ndk` 0.9 MediaCodec + ImageReader (api-level-24) API.
//! Cross-compiled to `aarch64-linux-android` by CI; decode is validated on a
//! device (the `android_mediacodec_smoke` test, run via
//! `tools/android-mediacodec-smoke.sh`).

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use ndk::media::image_reader::{AcquireResult, Image, ImageFormat, ImageReader};
use ndk::media::media_codec::{
    DequeuedInputBufferResult, DequeuedOutputBufferInfoResult, MediaCodec, MediaCodecDirection,
};
use ndk::media::media_format::MediaFormat;
use ndk::native_window::NativeWindow;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate,
    RawVideoFormat, VideoCodec,
};

use crate::annexb::{h264_parameter_sets, h265_parameter_sets};

use alloc::boxed::Box;
use alloc::vec::Vec;

/// `AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM`: mark the final (empty) input buffer.
const BUFFER_FLAG_END_OF_STREAM: u32 = 4;

/// Images the `ImageReader` may hold un-acquired. Must exceed the decoder's
/// output reorder depth so rendering never stalls waiting for us to acquire.
const MAX_IMAGES: i32 = 8;

/// Bounded output polls so the EOS drain waits for the codec to flush without
/// spinning forever if it never raises the end-of-stream flag.
const MAX_OUTPUT_POLLS: u32 = 256;

/// Bounded retries when the codec has no free input buffer yet, so a stuck
/// codec surfaces as an error rather than spinning forever.
const MAX_INPUT_RETRIES: u32 = 100;

/// One decoded frame ready to emit: NV12 bytes in the CPU path, or (in the M304
/// zero-copy GPU path) the decoded `AHardwareBuffer` to convert to a wgpu texture.
#[derive(Debug)]
enum DecodedFrame {
    Nv12 { nv12: Box<[u8]>, width: u32, height: u32, pts_ns: u64 },
    /// An acquired reference to a decoded frame's `AHardwareBuffer`, held so it
    /// outlives the transient `Image`; converted to RGBA on the GPU in `process`.
    #[cfg(feature = "mediacodec-wgpu")]
    Ahb { ahb: ndk::hardware_buffer::HardwareBufferRef, width: u32, height: u32, pts_ns: u64 },
}

/// Live codec plus the parameter sets it was configured with (so a mid-stream
/// SPS/PPS change rebuilds it) and the current output geometry / layout read from
/// the codec's output format.
struct CodecState {
    // `codec` is declared first so it drops (and stops) before the `reader` /
    // `window` backing its output Surface.
    codec: MediaCodec,
    /// The codec renders decoded frames into this reader's Surface; we acquire
    /// `Image`s and read their planes (handles any vendor colour layout).
    reader: ImageReader,
    /// The reader's Surface, handed to `configure`; kept resident for the codec.
    _window: NativeWindow,
    /// The codec-specific data the codec was configured with (Annex-B), kept so a
    /// mid-stream parameter-set change rebuilds it. `csd-0` and optional `csd-1`.
    csd0: Vec<u8>,
    csd1: Option<Vec<u8>>,
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

#[derive(Debug)]
pub struct MediaCodecDec {
    codec: VideoCodec,
    width: u32,
    height: u32,
    configured: bool,
    state: Option<CodecState>,
    last_caps: Option<Caps>,
    input_caps: Option<Caps>,
    emitted: u64,
    /// M304 probe hook: an owned reference to the first decoded frame's backing
    /// `AHardwareBuffer`, captured so the GPU-interop bridge can query its Vulkan
    /// format / import it without re-driving the decode. Acquired (refcount bump)
    /// so it outlives the transient `Image` it came from.
    #[cfg(feature = "mediacodec-wgpu")]
    captured_ahb: Option<ndk::hardware_buffer::HardwareBufferRef>,
    /// M304 zero-copy output: when set, each decoded frame is converted to an
    /// RGBA `wgpu::Texture` on the GPU (no CPU NV12 pack) and emitted as
    /// `MemoryDomain::WgpuTexture` instead of system memory.
    #[cfg(feature = "mediacodec-wgpu")]
    gpu_output: bool,
    /// The wgpu/Vulkan interop device, created lazily on the first decoded frame
    /// in GPU-output mode (its creation is async).
    #[cfg(feature = "mediacodec-wgpu")]
    interop: Option<crate::mediacodec_wgpu::InteropDevice>,
    /// The reusable YCbCr -> RGBA converter, built lazily once the first decoded
    /// buffer's external format + geometry are known.
    #[cfg(feature = "mediacodec-wgpu")]
    converter: Option<crate::mediacodec_wgpu::YcbcrToRgba>,
}

// SAFETY: `ndk::media::MediaCodec` wraps a raw `AMediaCodec` pointer and is not
// `Send` by default. Like `MfDecode` / `VtDecode`, `MediaCodecDec` is built for a
// single-thread executor: every codec call lands on the element's owning task, so
// the pointer is never touched from two threads. We assert `Send` under that
// documented contract so the multi-thread runner accepts the element.
unsafe impl Send for MediaCodecDec {}

impl Default for MediaCodecDec {
    fn default() -> Self {
        Self::h264()
    }
}

impl MediaCodecDec {
    /// An H.264 MediaCodec decoder.
    pub fn h264() -> Self {
        Self::new(VideoCodec::H264)
    }

    /// An H.265 / HEVC MediaCodec decoder. Same shape as H.264; differs only in
    /// the MIME type and that the VPS+SPS+PPS pack into a single `csd-0`.
    pub fn h265() -> Self {
        Self::new(VideoCodec::H265)
    }

    fn new(codec: VideoCodec) -> Self {
        Self {
            codec,
            width: 0,
            height: 0,
            configured: false,
            state: None,
            last_caps: None,
            input_caps: None,
            emitted: 0,
            #[cfg(feature = "mediacodec-wgpu")]
            captured_ahb: None,
            #[cfg(feature = "mediacodec-wgpu")]
            gpu_output: false,
            #[cfg(feature = "mediacodec-wgpu")]
            interop: None,
            #[cfg(feature = "mediacodec-wgpu")]
            converter: None,
        }
    }

    /// Enable the M304 zero-copy GPU output path: decoded frames are converted to
    /// RGBA `wgpu::Texture`s on the GPU (importing each `AHardwareBuffer` through
    /// an immutable YCbCr-conversion sampler) and emitted as
    /// `MemoryDomain::WgpuTexture`, instead of the default CPU NV12 pack. The
    /// element must run on a single-thread executor in this mode (it submits to
    /// the wgpu device's queue directly), the same contract the codec pointer
    /// already requires.
    #[cfg(feature = "mediacodec-wgpu")]
    pub fn with_gpu_output(mut self) -> Self {
        self.gpu_output = true;
        self
    }

    /// Like [`with_gpu_output`](Self::with_gpu_output), but decode onto a
    /// caller-supplied interop device instead of one created lazily (M305). The
    /// on-screen present path needs the decoder and the [`WgpuSink`] to share a
    /// single wgpu device (a texture binds only to its own device): the app builds
    /// one [`InteropDevice`](crate::mediacodec_wgpu::InteropDevice), derives a
    /// surface + sink from it via
    /// [`gpu_context`](crate::mediacodec_wgpu::InteropDevice::gpu_context) /
    /// [`create_android_surface`](crate::mediacodec_wgpu::create_android_surface),
    /// then hands the device here so the decoder converts frames on that very
    /// device. The same single-thread executor contract as `with_gpu_output`.
    #[cfg(feature = "mediacodec-wgpu")]
    pub fn with_gpu_device(mut self, dev: crate::mediacodec_wgpu::InteropDevice) -> Self {
        self.gpu_output = true;
        self.interop = Some(dev);
        self
    }

    /// Convert a decoded `AHardwareBuffer` to an RGBA `wgpu::Texture` on the GPU
    /// and wrap it as a `MemoryDomain::WgpuTexture`. Creates the interop device
    /// (async) and the reusable converter lazily on the first frame. The buffer
    /// is released back to the decoder as soon as the conversion completes.
    #[cfg(feature = "mediacodec-wgpu")]
    async fn convert_ahb_to_domain(
        &mut self,
        ahb: ndk::hardware_buffer::HardwareBufferRef,
        width: u32,
        height: u32,
    ) -> Result<MemoryDomain, G2gError> {
        use crate::mediacodec_wgpu::{
            ahb_format_info, create_android_interop_device, WgpuRgbaTexture, YcbcrToRgba,
        };

        if self.interop.is_none() {
            self.interop = Some(create_android_interop_device().await?);
        }
        let ahb_ptr = ahb.as_ptr() as *const ash::vk::AHardwareBuffer;
        if self.converter.is_none() {
            let dev = self.interop.as_ref().unwrap();
            // SAFETY: `dev` is the interop device; `ahb` is a live decoded buffer.
            let info = unsafe { ahb_format_info(dev, ahb_ptr)? };
            // SAFETY: same device; `info` is this buffer's format.
            self.converter = Some(unsafe { YcbcrToRgba::new(dev, &info, width, height)? });
        }
        // Clone the device/queue handles before the &mut converter borrow.
        let (wdev, wqueue) = {
            let dev = self.interop.as_ref().unwrap();
            (dev.device.clone(), dev.queue.clone())
        };
        // SAFETY: `ahb` is live (held here); the converter was built on this
        // device for this format / geometry. The conversion is pipelined (submit
        // without wait); importing `ahb` gave Vulkan its own reference, so the
        // buffer may be released as soon as `convert` returns.
        let texture = unsafe { self.converter.as_mut().unwrap().convert(ahb_ptr)? };
        drop(ahb);
        let owner = WgpuRgbaTexture::new(wdev, wqueue, texture);
        Ok(MemoryDomain::WgpuTexture(g2g_core::OwnedWgpuTexture::new(
            width,
            height,
            alloc::sync::Arc::new(owner),
        )))
    }

    /// M304 probe accessor: the first decoded frame's `AHardwareBuffer` (owned
    /// reference), once at least one frame has been drained. `None` before then.
    /// Used by the `mediacodec_wgpu` bridge to inspect the vendor buffer's Vulkan
    /// format on-device.
    #[cfg(feature = "mediacodec-wgpu")]
    pub fn captured_hardware_buffer(&self) -> Option<&ndk::hardware_buffer::HardwareBufferRef> {
        self.captured_ahb.as_ref()
    }

    /// The MediaCodec MIME type for this element's codec.
    fn mime(&self) -> &'static str {
        match self.codec {
            VideoCodec::H265 => "video/hevc",
            _ => "video/avc",
        }
    }

    /// Extract the codec-specific data (`csd-0`, optional `csd-1`) for the current
    /// codec from an access unit, or `None` until every parameter set has been
    /// seen. H.264 splits SPS (`csd-0`) and PPS (`csd-1`); H.265 concatenates
    /// VPS+SPS+PPS into `csd-0`. Each NAL is re-prefixed with an Annex-B start code.
    fn codec_specific_data(&self, au: &[u8]) -> Option<(Vec<u8>, Option<Vec<u8>>)> {
        match self.codec {
            VideoCodec::H265 => {
                let (vps, sps, pps) = h265_parameter_sets(au);
                if vps.is_empty() || sps.is_empty() || pps.is_empty() {
                    return None;
                }
                let mut csd0 = annexb_join(&vps);
                csd0.extend_from_slice(&annexb_join(&sps));
                csd0.extend_from_slice(&annexb_join(&pps));
                Some((csd0, None))
            }
            _ => {
                let (sps, pps) = h264_parameter_sets(au);
                if sps.is_empty() || pps.is_empty() {
                    return None;
                }
                Some((annexb_join(&sps), Some(annexb_join(&pps))))
            }
        }
    }

    /// Count of decoded NV12 `DataFrame`s pushed downstream. Useful in tests.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// Whether the zero-copy GPU output path is active (always false without the
    /// `mediacodec-wgpu` feature). Drives the negotiated output format.
    fn gpu_output_enabled(&self) -> bool {
        #[cfg(feature = "mediacodec-wgpu")]
        {
            self.gpu_output
        }
        #[cfg(not(feature = "mediacodec-wgpu"))]
        {
            false
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: self.codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    /// (Re)create the codec when the parameter sets first appear or change.
    fn ensure_codec(&mut self, au: &[u8]) -> Result<(), G2gError> {
        let Some((csd0, csd1)) = self.codec_specific_data(au) else {
            return Ok(()); // wait for a keyframe's parameter sets
        };
        if let Some(st) = self.state.as_ref() {
            if st.csd0 == csd0 && st.csd1 == csd1 {
                return Ok(());
            }
        }

        // MediaCodec.configure and the ImageReader both need the frame geometry;
        // it arrives via configure_pipeline (h264parse derives it from the SPS).
        if self.width == 0 || self.height == 0 {
            return Err(G2gError::NotConfigured);
        }

        let codec = MediaCodec::from_decoder_type(self.mime()).ok_or(G2gError::NotConfigured)?;

        // Decode into an ImageReader's Surface, not ByteBuffers: modern vendor
        // decoders only deliver graphic buffers (a no-Surface configure stalls on
        // them), and the YUV_420_888 Image planes expose a uniform, stride-
        // described layout we pack to NV12 regardless of the codec's colour format.
        let reader = ImageReader::new(
            self.width as i32,
            self.height as i32,
            ImageFormat::YUV_420_888,
            MAX_IMAGES,
        )
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let window = reader.window().map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        let mut format = MediaFormat::new();
        format.set_str("mime", self.mime());
        format.set_i32("width", self.width as i32);
        format.set_i32("height", self.height as i32);
        format.set_buffer("csd-0", &csd0);
        if let Some(csd1) = &csd1 {
            format.set_buffer("csd-1", csd1);
        }

        codec
            .configure(&format, Some(&window), MediaCodecDirection::Decoder)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        codec.start().map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        self.state = Some(CodecState {
            codec,
            reader,
            _window: window,
            csd0,
            csd1,
            width: self.width,
            height: self.height,
        });
        Ok(())
    }

    /// Submit one Annex-B access unit, then drain whatever output is ready.
    fn feed(&mut self, au: &[u8], pts_ns: u64, out: &mut Vec<DecodedFrame>) -> Result<(), G2gError> {
        self.ensure_codec(au)?;
        if self.state.is_none() {
            return Ok(()); // pre-keyframe: nothing to decode yet
        }
        self.queue_input(au, pts_ns / 1000, 0)?;
        self.pump_output(out, false)
    }

    /// Hand `data` to a free input buffer with the given microsecond pts + flags.
    /// Retries a bounded number of times while the codec reports no free buffer
    /// (it frees them as it drains), then errors rather than spinning forever.
    fn queue_input(&self, data: &[u8], pts_us: u64, flags: u32) -> Result<(), G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        for _ in 0..MAX_INPUT_RETRIES {
            match st
                .codec
                .dequeue_input_buffer(Duration::from_millis(10))
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            {
                DequeuedInputBufferResult::Buffer(mut input) => {
                    let dst = input.buffer_mut();
                    if dst.len() < data.len() {
                        // A single access unit larger than an input buffer needs
                        // splitting across buffers; not handled in v1.
                        return Err(G2gError::Hardware(HardwareError::Other));
                    }
                    for (d, &s) in dst.iter_mut().zip(data) {
                        d.write(s);
                    }
                    st.codec
                        .queue_input_buffer(input, 0, data.len(), pts_us, flags)
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    return Ok(());
                }
                DequeuedInputBufferResult::TryAgainLater => continue,
            }
        }
        Err(G2gError::Hardware(HardwareError::Other))
    }

    /// Render every ready output buffer into the codec's ImageReader Surface,
    /// then pack the resulting images to NV12. In steady state (`until_eos ==
    /// false`) this makes one non-blocking pass; at EOS it polls (bounded) until
    /// the codec raises the end-of-stream flag, so the reorder queue fully drains.
    fn pump_output(&mut self, out: &mut Vec<DecodedFrame>, until_eos: bool) -> Result<(), G2gError> {
        let timeout = if until_eos { Duration::from_millis(20) } else { Duration::ZERO };
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
                        let info = buffer.info();
                        // A zero-size buffer is codec config: release without rendering.
                        let render = info.size() > 0;
                        eos = info.flags() & BUFFER_FLAG_END_OF_STREAM != 0;
                        st.codec
                            .release_output_buffer(buffer, render)
                            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                    }
                    DequeuedOutputBufferInfoResult::OutputFormatChanged
                    | DequeuedOutputBufferInfoResult::OutputBuffersChanged => got = true,
                    DequeuedOutputBufferInfoResult::TryAgainLater => {}
                }
            }
            // Acquire whatever images the renders have produced so far.
            self.drain_images(out)?;
            if eos {
                break;
            }
            // Steady state: one pass over the currently-ready buffers is enough.
            if !until_eos && !got {
                break;
            }
        }
        Ok(())
    }

    /// Acquire and pack every image the rendered output buffers have produced.
    /// The `YUV_420_888` plane strides describe whatever layout the decoder chose.
    fn drain_images(&mut self, out: &mut Vec<DecodedFrame>) -> Result<(), G2gError> {
        // M304: stash the first decoded frame's AHardwareBuffer for GPU interop.
        // Captured into a local while `st` borrows `self`, then moved onto the
        // element after the loop (can't touch `self.captured_ahb` while borrowed).
        #[cfg(feature = "mediacodec-wgpu")]
        let mut first_ahb: Option<ndk::hardware_buffer::HardwareBufferRef> = None;
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        // NoBufferAvailable / MaxImagesAcquired ends the loop: nothing more now.
        while let AcquireResult::Image(img) = st
            .reader
            .acquire_next_image()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
        {
            // Acquire an owned reference (refcount bump) before `img` is dropped,
            // so the buffer outlives the transient Image.
            #[cfg(feature = "mediacodec-wgpu")]
            if first_ahb.is_none() && self.captured_ahb.is_none() {
                if let Ok(hb) = img.hardware_buffer() {
                    first_ahb = Some(hb.acquire());
                }
            }
            let pts_ns = img.timestamp().unwrap_or(0).max(0) as u64;
            // Zero-copy GPU path: hand the decoded AHardwareBuffer downstream as
            // an acquired reference (converted to a wgpu texture in `process`),
            // skipping the CPU NV12 pack.
            #[cfg(feature = "mediacodec-wgpu")]
            if self.gpu_output {
                if let Ok(hb) = img.hardware_buffer() {
                    let w = img.width().unwrap_or(0).max(0) as u32;
                    let h = img.height().unwrap_or(0).max(0) as u32;
                    out.push(DecodedFrame::Ahb { ahb: hb.acquire(), width: w, height: h, pts_ns });
                }
                continue;
            }
            if let Some(frame) = image_to_nv12(&img, pts_ns) {
                out.push(frame);
            }
        }
        #[cfg(feature = "mediacodec-wgpu")]
        if self.captured_ahb.is_none() {
            self.captured_ahb = first_ahb;
        }
        Ok(())
    }
}

impl AsyncElement for MediaCodecDec {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.output_caps())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let codec = self.codec;
        // GPU-output mode derives RGBA (a WgpuTexture frame), else NV12.
        let rgba = self.gpu_output_enabled();
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            derive_output_caps(codec, rgba, input)
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo { codec, width, height, .. } if *codec == self.codec => {
                // Geometry is a hint for the initial MediaFormat; the codec's
                // output format is authoritative for packing.
                self.width = fixed_or_zero(width);
                self.height = fixed_or_zero(height);
                self.configured = true;
                Ok(ConfigureOutcome::Accepted)
            }
            _ => Err(G2gError::CapsMismatch),
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
            let mut decoded = Vec::new();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed(slice.as_slice(), frame.timing.pts_ns, &mut decoded)?;
                }
                PipelinePacket::CapsChanged(c) => {
                    match &c {
                        Caps::CompressedVideo { codec, .. } if *codec == self.codec => {}
                        _ => return Err(G2gError::CapsMismatch),
                    }
                    self.input_caps = Some(c);
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
                    // Signal end of stream with an empty input buffer, then drain.
                    if self.state.is_some() {
                        let _ = self.queue_input(&[], 0, BUFFER_FLAG_END_OF_STREAM);
                        self.pump_output(&mut decoded, true)?;
                    }
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                    return Ok(());
                }
            }

            for d in decoded {
                let (new_caps, domain, pts_ns) = match d {
                    DecodedFrame::Nv12 { nv12, width, height, pts_ns } => (
                        nv12_caps(width, height),
                        MemoryDomain::System(SystemSlice::from_boxed(nv12)),
                        pts_ns,
                    ),
                    // Zero-copy GPU path: convert the decoded AHardwareBuffer to an
                    // RGBA wgpu texture and emit it as MemoryDomain::WgpuTexture.
                    #[cfg(feature = "mediacodec-wgpu")]
                    DecodedFrame::Ahb { ahb, width, height, pts_ns } => {
                        let domain = self.convert_ahb_to_domain(ahb, width, height).await?;
                        (rgba_caps(width, height), domain, pts_ns)
                    }
                };
                if self.last_caps.as_ref() != Some(&new_caps) {
                    out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                    self.last_caps = Some(new_caps);
                }
                let frame = Frame {
                    domain,
                    timing: FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        duration_ns: 0,
                        capture_ns: pts_ns,
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

impl PadTemplates for MediaCodecDec {
    fn pad_templates() -> Vec<PadTemplate> {
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        // The source advertises both the default NV12 (CPU) output and the M304
        // RGBA WgpuTexture output (GPU mode); `caps_constraint_as_transform`
        // picks the active one per instance.
        let rgba8 = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::one(h264)),
            PadTemplate::source(CapsSet::from_alternatives(Vec::from([nv12, rgba8]))),
        ])
    }
}

/// Concatenate NAL units as Annex-B (each prefixed with a 4-byte start code), for
/// the MediaFormat `csd-*` buffers.
fn annexb_join(nals: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for nal in nals {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    out
}

/// Pack a decoded `YUV_420_888` image into a `DecodedFrame::Nv12`; the layout
/// handling lives in `yuv420::pack_yuv420_to_nv12` (shared with camera2src).
fn image_to_nv12(img: &Image, pts_ns: u64) -> Option<DecodedFrame> {
    let (nv12, width, height) = crate::yuv420::pack_yuv420_to_nv12(img)?;
    Some(DecodedFrame::Nv12 {
        nv12: nv12.into_boxed_slice(),
        width,
        height,
        pts_ns,
    })
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

/// Caps for the M304 zero-copy GPU path: RGBA (the converter's output format),
/// carried on a `MemoryDomain::WgpuTexture` frame.
#[cfg(feature = "mediacodec-wgpu")]
fn rgba_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Rgba8,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

fn derive_output_caps(codec: VideoCodec, rgba: bool, input: &Caps) -> CapsSet {
    let format = if rgba { RawVideoFormat::Rgba8 } else { RawVideoFormat::Nv12 };
    match input {
        Caps::CompressedVideo { codec: c, width, height, framerate } if *c == codec => {
            CapsSet::one(Caps::RawVideo {
                format,
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
