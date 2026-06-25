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

#[derive(Debug)]
struct DecodedFrame {
    nv12: Box<[u8]>,
    width: u32,
    height: u32,
    pts_ns: u64,
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
        }
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
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        loop {
            match st
                .reader
                .acquire_next_image()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            {
                AcquireResult::Image(img) => {
                    let pts_ns = img.timestamp().unwrap_or(0).max(0) as u64;
                    if let Some(frame) = image_to_nv12(&img, pts_ns) {
                        out.push(frame);
                    }
                }
                // NoBufferAvailable / MaxImagesAcquired: nothing more right now.
                _ => return Ok(()),
            }
        }
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
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| derive_output_caps(codec, input)))
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
                let new_caps = nv12_caps(d.width, d.height);
                if self.last_caps.as_ref() != Some(&new_caps) {
                    out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                    self.last_caps = Some(new_caps);
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(d.nv12)),
                    timing: FrameTiming {
                        pts_ns: d.pts_ns,
                        dts_ns: d.pts_ns,
                        duration_ns: 0,
                        capture_ns: d.pts_ns,
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
        Vec::from([PadTemplate::sink(CapsSet::one(h264)), PadTemplate::source(CapsSet::one(nv12))])
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

/// Pack a decoded `YUV_420_888` image to tight NV12 (Y plane then interleaved
/// UV). Each plane's row and pixel strides describe whatever layout the decoder
/// used (planar I420, semi-planar, or a vendor format), so this single path
/// handles them all, unlike the old byte-buffer packing.
fn image_to_nv12(img: &Image, pts_ns: u64) -> Option<DecodedFrame> {
    let w = img.width().ok()?.max(0) as usize;
    let h = img.height().ok()?.max(0) as usize;
    if w == 0 || h == 0 {
        return None;
    }
    let y = img.plane_data(0).ok()?;
    let y_rs = img.plane_row_stride(0).ok()? as usize;
    let u = img.plane_data(1).ok()?;
    let u_rs = img.plane_row_stride(1).ok()? as usize;
    let u_ps = img.plane_pixel_stride(1).ok()? as usize;
    let v = img.plane_data(2).ok()?;
    let v_rs = img.plane_row_stride(2).ok()? as usize;
    let v_ps = img.plane_pixel_stride(2).ok()? as usize;

    let (cw, ch) = (w / 2, h / 2);
    let mut nv12 = Vec::with_capacity(w * h + 2 * cw * ch);
    // Luma: w bytes per row, row-stride apart.
    for row in 0..h {
        let off = row * y_rs;
        nv12.extend_from_slice(y.get(off..off + w)?);
    }
    // Chroma: interleave Cb,Cr honoring each plane's row + pixel stride (a pixel
    // stride of 2 is an already-interleaved semi-planar source; 1 is planar).
    for row in 0..ch {
        for col in 0..cw {
            nv12.push(*u.get(row * u_rs + col * u_ps)?);
            nv12.push(*v.get(row * v_rs + col * v_ps)?);
        }
    }
    Some(DecodedFrame { nv12: nv12.into_boxed_slice(), width: w as u32, height: h as u32, pts_ns })
}

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

fn derive_output_caps(codec: VideoCodec, input: &Caps) -> CapsSet {
    match input {
        Caps::CompressedVideo { codec: c, width, height, framerate } if *c == codec => {
            CapsSet::one(Caps::RawVideo {
                format: RawVideoFormat::Nv12,
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
