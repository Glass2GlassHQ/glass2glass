//! M219: Android hardware H.264 decode via the NDK MediaCodec (`AMediaCodec`).
//!
//! `MediaCodecDec` is the Android counterpart of `VtDecode` (macOS VideoToolbox)
//! and `MfDecode` (Windows Media Foundation): it consumes Annex-B H.264
//! `DataFrame`s (`MemoryDomain::System`, what `RtspSrc` / `H264Parse` emit) and
//! produces decoded NV12 frames, also `MemoryDomain::System` (a CPU copy out of
//! the codec's output buffer). It is the first element of the Android platform
//! track (DESIGN_TODO.md "Platform: Android"); a zero-copy `AHardwareBuffer` /
//! `SurfaceTexture` path and a `Surface` present sink are the follow-ups.
//!
//! Unlike VideoToolbox (which wants AVCC + out-of-band parameter sets),
//! MediaCodec takes the access units as Annex-B directly and the SPS/PPS as
//! `csd-0` / `csd-1` buffers in the `MediaFormat`. So the element reuses
//! [`crate::annexb::h264_parameter_sets`] for the codec-specific data but feeds
//! each frame's bytes unchanged (no AVCC conversion). It drives the codec
//! synchronously (queue one input buffer, drain ready output buffers), wrapping
//! the safe `ndk` crate rather than raw FFI.
//!
//! Built against the `ndk` 0.9 MediaCodec API. The dev host is Linux, so this is
//! compiled (cross-compiled to `aarch64-linux-android`) by CI, not here; actual
//! decode is validated on a device. The output color-format handling (semi-planar
//! vs planar -> NV12) is the device-dependent part: vendor / flexible formats are
//! a follow-up (the `AImageReader` plane path), marked `// NOTE`.

use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use ndk::media::media_codec::{
    DequeuedInputBufferResult, DequeuedOutputBufferInfoResult, MediaCodec, MediaCodecDirection,
};
use ndk::media::media_format::MediaFormat;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate,
    RawVideoFormat, VideoCodec,
};

use crate::annexb::{h264_nal_type, h264_parameter_sets};

use alloc::boxed::Box;
use alloc::vec::Vec;

/// `AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM`: mark the final (empty) input buffer.
const BUFFER_FLAG_END_OF_STREAM: u32 = 4;

/// Android `MediaCodecInfo.CodecCapabilities` color formats we can pack to NV12.
const COLOR_FORMAT_YUV420_PLANAR: i32 = 19; // I420 (repack to NV12)
const COLOR_FORMAT_YUV420_SEMIPLANAR: i32 = 21; // Y + interleaved UV (NV12 layout)

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
    codec: MediaCodec,
    sps: Vec<Vec<u8>>,
    pps: Vec<Vec<u8>>,
    color_format: i32,
    width: u32,
    height: u32,
    stride: u32,
    slice_height: u32,
}

impl core::fmt::Debug for CodecState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CodecState")
            .field("color_format", &self.color_format)
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
    /// An H.264 MediaCodec decoder. (HEVC is a follow-up: `video/hevc` + the
    /// VPS/SPS/PPS csd; the element shape is identical.)
    pub fn h264() -> Self {
        Self {
            codec: VideoCodec::H264,
            width: 0,
            height: 0,
            configured: false,
            state: None,
            last_caps: None,
            input_caps: None,
            emitted: 0,
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
    fn ensure_codec(&mut self, sps: &[Vec<u8>], pps: &[Vec<u8>]) -> Result<(), G2gError> {
        if sps.is_empty() || pps.is_empty() {
            return Ok(()); // wait for a keyframe's parameter sets
        }
        if let Some(st) = self.state.as_ref() {
            if st.sps == sps && st.pps == pps {
                return Ok(());
            }
        }

        let codec = MediaCodec::from_decoder_type("video/avc").ok_or(G2gError::NotConfigured)?;
        let mut format = MediaFormat::new();
        format.set_str("mime", "video/avc");
        if self.width > 0 && self.height > 0 {
            format.set_i32("width", self.width as i32);
            format.set_i32("height", self.height as i32);
        }
        // csd-0 = SPS, csd-1 = PPS, each as Annex-B (start-code prefixed), the
        // MediaCodec convention. h264_parameter_sets strips the start codes, so
        // re-add them.
        format.set_buffer("csd-0", &annexb_join(sps));
        format.set_buffer("csd-1", &annexb_join(pps));

        codec
            .configure(&format, None, MediaCodecDirection::Decoder)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        codec.start().map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        self.state = Some(CodecState {
            codec,
            sps: sps.to_vec(),
            pps: pps.to_vec(),
            // Filled in on the first OUTPUT_FORMAT_CHANGED; sane defaults until then.
            color_format: COLOR_FORMAT_YUV420_SEMIPLANAR,
            width: self.width,
            height: self.height,
            stride: self.width,
            slice_height: self.height,
        });
        Ok(())
    }

    /// Submit one Annex-B access unit, then drain whatever output is ready.
    fn feed(&mut self, au: &[u8], pts_ns: u64, out: &mut Vec<DecodedFrame>) -> Result<(), G2gError> {
        let (sps, pps) = h264_parameter_sets(au);
        self.ensure_codec(&sps, &pps)?;
        if self.state.is_none() {
            return Ok(()); // pre-keyframe: nothing to decode yet
        }
        self.queue_input(au, pts_ns / 1000, 0)?;
        self.drain_output(out)
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

    /// Pull every currently-ready output buffer, packing each to NV12. Reads the
    /// dequeue result in an inner scope so its borrow of `self.state` is released
    /// before an `OutputFormatChanged` re-borrows it mutably.
    fn drain_output(&mut self, out: &mut Vec<DecodedFrame>) -> Result<(), G2gError> {
        loop {
            // Layout read from OUTPUT_FORMAT_CHANGED, applied after the borrow ends.
            let reformat: Option<(i32, u32, u32, u32, u32)> = {
                let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
                match st
                    .codec
                    .dequeue_output_buffer(Duration::ZERO)
                    .map_err(|_| G2gError::Hardware(HardwareError::Other))?
                {
                    DequeuedOutputBufferInfoResult::Buffer(buffer) => {
                        let info = buffer.info();
                        let (offset, size) = (info.offset() as usize, info.size() as usize);
                        let pts_ns = (info.presentation_time_us().max(0) as u64) * 1000;
                        if size > 0 {
                            let bytes = buffer.buffer();
                            let frame = pack_nv12(st, &bytes[offset..offset + size], pts_ns)
                                .ok_or(G2gError::Hardware(HardwareError::Other))?;
                            out.push(frame);
                        }
                        st.codec
                            .release_output_buffer(buffer, false)
                            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
                        None
                    }
                    DequeuedOutputBufferInfoResult::OutputFormatChanged => {
                        let fmt = st.codec.output_format();
                        let cf = fmt.i32("color-format").unwrap_or(COLOR_FORMAT_YUV420_SEMIPLANAR);
                        let w = fmt.i32("width").unwrap_or(self.width as i32).max(0) as u32;
                        let h = fmt.i32("height").unwrap_or(self.height as i32).max(0) as u32;
                        let stride = fmt.i32("stride").unwrap_or(w as i32).max(0) as u32;
                        let slice = fmt.i32("slice-height").unwrap_or(h as i32).max(0) as u32;
                        Some((cf, w, h, stride, slice))
                    }
                    // The NDK manages the buffer set; nothing to do.
                    DequeuedOutputBufferInfoResult::OutputBuffersChanged => None,
                    DequeuedOutputBufferInfoResult::TryAgainLater => return Ok(()),
                }
            };
            if let Some((cf, w, h, stride, slice)) = reformat {
                let st = self.state.as_mut().ok_or(G2gError::NotConfigured)?;
                st.color_format = cf;
                st.width = w;
                st.height = h;
                st.stride = if stride == 0 { w } else { stride };
                st.slice_height = if slice == 0 { h } else { slice };
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
                        self.drain_output(&mut decoded)?;
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

/// Pack a MediaCodec byte-buffer output frame to tight NV12, honoring the codec's
/// stride / slice-height. Handles semi-planar (NV12 layout) and planar (I420,
/// repacked). Returns `None` for a color format we do not pack (vendor / flexible
/// formats need the `AImageReader` plane path, a follow-up).
fn pack_nv12(st: &CodecState, data: &[u8], pts_ns: u64) -> Option<DecodedFrame> {
    let (w, h) = (st.width as usize, st.height as usize);
    let stride = st.stride as usize;
    let slice = st.slice_height as usize;
    if w == 0 || h == 0 || stride < w {
        return None;
    }
    let mut nv12 = Vec::with_capacity(w * h * 3 / 2);

    // Luma: h rows of w bytes, stride apart, from the start of the frame.
    for row in 0..h {
        let off = row * stride;
        let line = data.get(off..off + w)?;
        nv12.extend_from_slice(line);
    }
    // Chroma starts after the luma plane (stride * slice_height bytes).
    let chroma_base = stride * slice;
    match st.color_format {
        COLOR_FORMAT_YUV420_SEMIPLANAR => {
            // Interleaved CbCr, w bytes per row, h/2 rows, stride apart.
            for row in 0..h / 2 {
                let off = chroma_base + row * stride;
                let line = data.get(off..off + w)?;
                nv12.extend_from_slice(line);
            }
        }
        COLOR_FORMAT_YUV420_PLANAR => {
            // Separate U then V planes, w/2 wide, h/2 tall, (stride/2) apart.
            let cstride = stride / 2;
            let v_base = chroma_base + cstride * (slice / 2);
            for row in 0..h / 2 {
                let u_off = chroma_base + row * cstride;
                let v_off = v_base + row * cstride;
                let u = data.get(u_off..u_off + w / 2)?;
                let v = data.get(v_off..v_off + w / 2)?;
                for i in 0..w / 2 {
                    nv12.push(u[i]);
                    nv12.push(v[i]);
                }
            }
        }
        // NOTE (device validation): vendor / COLOR_FormatYUV420Flexible formats
        // are not byte-packable from a plain buffer; they need the AImageReader
        // plane API. Surface rather than mis-pack.
        _ => return None,
    }
    Some(DecodedFrame { nv12: nv12.into_boxed_slice(), width: st.width, height: st.height, pts_ns })
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
