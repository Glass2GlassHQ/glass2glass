//! M218: macOS hardware H.264 / H.265 decode via VideoToolbox
//! (`VTDecompressionSession`).
//!
//! `VtDecode` is the macOS counterpart of `MfDecode` (Windows Media Foundation):
//! it consumes Annex-B H.264 *or* H.265 (HEVC) `DataFrame`s (`MemoryDomain::
//! System`, what `RtspSrc` / `H264Parse` / `H265Parse` emit) and produces decoded
//! NV12 frames, also `MemoryDomain::System` (a CPU copy out of the decoder's
//! `CVPixelBuffer`). In `cv-output` mode (M735) the copy is skipped: frames are
//! emitted as retained IOSurface-backed `CVPixelBuffer`s
//! (`MemoryDomain::CvPixelBuffer`), so a VideoToolbox encoder or Metal consumer
//! reads the decoder's buffer directly. A Metal present sink is the follow-up.
//!
//! The two codecs differ only in the format-description constructor
//! (`...FromH264ParameterSets` vs `...FromHEVCParameterSets`, the latter taking an
//! extra `extensions` dictionary) and the parameter-set NAL types (H.264 SPS 7 /
//! PPS 8; H.265 VPS 32 / SPS 33 / PPS 34); the session, decode loop, and NV12
//! packing are shared.
//!
//! Unlike Media Foundation, VideoToolbox wants AVCC framing and the SPS/PPS
//! parameter sets supplied out of band (in the `CMVideoFormatDescription`), not
//! Annex-B with in-band parameter sets. The element therefore pulls the SPS/PPS
//! out of each access unit ([`crate::annexb::h264_parameter_sets`]) to build the
//! format description, and converts the remaining VCL/SEI NALs to AVCC
//! ([`crate::annexb::to_avcc`]) for the decode sample. Those helpers are pure and
//! unit-tested on the host; the VideoToolbox session itself is macOS-only.
//!
//! This module is `#[cfg(all(target_os = "macos", feature = "vtdecode"))]`, so
//! it never builds on the Linux dev host; the macOS CI job compiles it and runs
//! the `m731_videotoolbox` tests (reference-fixture decode to NV12, H.264 +
//! HEVC), so the decode path is runtime-validated on a real Mac.

use core::ffi::c_void;
use core::ptr::{self, NonNull};

use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString, CFType};

// OSStatus is `pub(crate)` in the objc2 framework crates (not importable). It is
// a transparent `i32` alias, so a local alias matches the FFI signatures exactly.
#[allow(non_camel_case_types)]
type OSStatus = i32;
use objc2_core_media::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMSampleTimingInfo, CMTime, CMTimeFlags,
    CMVideoFormatDescriptionCreateFromH264ParameterSets,
    CMVideoFormatDescriptionCreateFromHEVCParameterSets,
};
use objc2_core_video::{
    kCVPixelBufferIOSurfacePropertiesKey, kCVPixelBufferPixelFormatTypeKey, CVImageBuffer,
    CVPixelBuffer, CVPixelBufferGetHeight, CVPixelBufferGetIOSurface,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
};
use objc2_video_toolbox::{
    VTDecodeFrameFlags, VTDecodeInfoFlags, VTDecompressionOutputCallbackRecord,
    VTDecompressionSession,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, MemoryDomain, MemoryDomainKind, OutputSink, OwnedCvPixelBuffer, PadTemplate,
    PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    RawVideoFormat, VideoCodec,
};

use crate::annexb::{
    h264_nal_type, h264_parameter_sets, h265_nal_type, h265_parameter_sets, to_avcc,
};

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;

use crate::cvnv12::{
    pack_nv12_locked, CvBufferOwner, K_CV_PIXEL_FORMAT_420F, K_CV_PIXEL_FORMAT_420V,
};

/// One decoded frame the output callback captured: packed to tight NV12 bytes
/// (the default), or the decoder's retained `CVPixelBuffer` unread (`cv-output`
/// mode, the M735 zero-copy domain).
enum DecodedFrame {
    Nv12 {
        nv12: Box<[u8]>,
        width: u32,
        height: u32,
        pts_ns: u64,
    },
    Cv {
        buf: CFRetained<CVPixelBuffer>,
        width: u32,
        height: u32,
        pixel_format: u32,
        io_surface_backed: bool,
        pts_ns: u64,
    },
}

// Manual impl: `CFRetained` has no `Debug`.
impl core::fmt::Debug for DecodedFrame {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DecodedFrame::Nv12 { width, height, .. } => f
                .debug_struct("Nv12")
                .field("width", width)
                .field("height", height)
                .finish_non_exhaustive(),
            DecodedFrame::Cv {
                width,
                height,
                pixel_format,
                io_surface_backed,
                ..
            } => f
                .debug_struct("Cv")
                .field("width", width)
                .field("height", height)
                .field("pixel_format", pixel_format)
                .field("io_surface_backed", io_surface_backed)
                .finish_non_exhaustive(),
        }
    }
}

/// Shared sink the VideoToolbox output callback writes into. The session's
/// callback record holds a raw `*mut Collector` (the refcon); the box keeps a
/// stable address for the session's whole life. Accessed by the callback only
/// during `DecodeFrame` / `WaitForAsynchronousFrames` (synchronous, on the
/// element thread), and drained by `process` only after the wait returns, so the
/// two never touch it at the same time.
#[derive(Debug, Default)]
struct Collector {
    frames: Vec<DecodedFrame>,
    error: Option<G2gError>,
    /// `cv-output` mode: retain the decoder's pixel buffer instead of packing
    /// NV12 bytes (set once at session build).
    cv_output: bool,
}

/// Live VideoToolbox session plus the parameter sets it was built from (so a
/// mid-stream SPS/PPS change triggers a rebuild) and the boxed collector the
/// callback writes into.
struct DecoderState {
    session: CFRetained<VTDecompressionSession>,
    // Kept alive because the session's CMVideoFormatDescription references it.
    _format: CFRetained<CMFormatDescription>,
    collector: Box<Collector>,
    /// The parameter-set NALs the session was built from (H.264 SPS+PPS, or H.265
    /// VPS+SPS+PPS), in the order handed to VideoToolbox; a mid-stream change
    /// rebuilds the session.
    params: Vec<Vec<u8>>,
}

impl core::fmt::Debug for DecoderState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecoderState")
            .field("collector", &self.collector)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct VtDecode {
    codec: VideoCodec,
    configured: bool,
    state: Option<DecoderState>,
    last_caps: Option<Caps>,
    /// The upstream framerate (from configure / input CapsChanged), carried on
    /// our emitted output caps: a downstream transform cannot `fixate()` an
    /// `Any` rate, so we never emit one (mirrors `FfmpegVideoDec`).
    input_framerate: Rate,
    /// M735 zero-copy output: emit decoded frames as retained (IOSurface-backed)
    /// `CVPixelBuffer`s (`MemoryDomain::CvPixelBuffer`) instead of packing NV12
    /// bytes to system memory.
    cv_output: bool,
    emitted: u64,
}

// SAFETY: a `VTDecompressionSession` and the CoreMedia objects are CoreFoundation
// types, thread-safe to retain/release but used here single-threaded. Like
// `MfDecode`, `VtDecode` is built for a single-thread executor: every decode call
// lands on the element's owning task, and the boxed `Collector` the callback
// writes is only read on that same task after `WaitForAsynchronousFrames`. The
// raw `*mut Collector` in the callback record never escapes the session, which
// `VtDecode` owns. We assert `Send` under that contract so the multi-thread
// runner accepts the element, mirroring `MfDecode`.
unsafe impl Send for VtDecode {}

impl Default for VtDecode {
    fn default() -> Self {
        Self::h264()
    }
}

impl VtDecode {
    /// An H.264 VideoToolbox decoder.
    pub fn h264() -> Self {
        Self::for_codec(VideoCodec::H264)
    }

    /// An H.265 (HEVC) VideoToolbox decoder. Differs from [`h264`](Self::h264)
    /// only in the format-description constructor and the parameter-set NAL types;
    /// the session + decode loop are shared.
    pub fn h265() -> Self {
        Self::for_codec(VideoCodec::H265)
    }

    fn for_codec(codec: VideoCodec) -> Self {
        Self {
            codec,
            configured: false,
            state: None,
            last_caps: None,
            input_framerate: Rate::Any,
            cv_output: false,
            emitted: 0,
        }
    }

    /// Emit decoded frames as retained `CVPixelBuffer`s
    /// (`MemoryDomain::CvPixelBuffer`, IOSurface-backed) instead of packed NV12
    /// system bytes, so a VideoToolbox encoder or Metal consumer reads them
    /// with no CPU copy. Also settable as the `cv-output` property.
    pub fn with_cv_output(mut self) -> Self {
        self.cv_output = true;
        self
    }

    /// Count of decoded NV12 `DataFrame`s pushed downstream. Useful in tests.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// (Re)build the VideoToolbox session when the parameter sets first appear or
    /// change. No-op when the current session already matches `params`. `params`
    /// is the ordered parameter-set NAL list (H.264 SPS+PPS, H.265 VPS+SPS+PPS).
    fn ensure_session(&mut self, params: &[Vec<u8>]) -> Result<(), G2gError> {
        if params.is_empty() {
            return Ok(()); // can't build without parameter sets; wait for a keyframe
        }
        if let Some(st) = self.state.as_ref() {
            if st.params == params {
                return Ok(());
            }
        }
        // SAFETY: all pointers below are into live local buffers (the parameter
        // set NAL bytes and the pointer/size arrays) that outlive the create
        // call, and the out-params are valid stack slots. The created session and
        // format description are +1 retained (the CoreFoundation Create rule) and
        // adopted into `CFRetained` so they release on drop.
        let state = unsafe { build_session(self.codec, params, self.cv_output)? };
        self.state = Some(state);
        Ok(())
    }

    /// Decode one Annex-B access unit, packing any produced NV12 frames into the
    /// state collector for `process` to drain.
    fn feed(&mut self, au: &[u8], pts_ns: u64) -> Result<(), G2gError> {
        // VideoToolbox builds the format description from the parameter sets, so
        // pull them out of the access unit and (re)build the session on a
        // keyframe; until the first parameter sets arrive there is nothing to
        // decode against (mirrors RtspSrc's first-keyframe skip). H.264 orders
        // SPS then PPS; H.265 orders VPS, SPS, PPS.
        let params = match self.codec {
            VideoCodec::H265 => {
                let (vps, sps, pps) = h265_parameter_sets(au);
                [vps, sps, pps].concat()
            }
            _ => {
                let (sps, pps) = h264_parameter_sets(au);
                [sps, pps].concat()
            }
        };
        self.ensure_session(&params)?;
        let Some(state) = self.state.as_mut() else {
            return Ok(()); // pre-keyframe: skip
        };

        // The decode sample carries only the VCL (+ SEI) NALs in AVCC framing;
        // the parameter sets + AUD live in the format description. H.264 excludes
        // SPS(7)/PPS(8)/AUD(9); H.265 excludes VPS(32)/SPS(33)/PPS(34)/AUD(35).
        let codec = self.codec;
        let avcc = to_avcc(au, |nal| match codec {
            VideoCodec::H265 => !matches!(h265_nal_type(nal), Some(32 | 33 | 34 | 35)),
            _ => !matches!(h264_nal_type(nal), Some(7 | 8 | 9)),
        });
        if avcc.is_empty() {
            return Ok(()); // parameter-set-only access unit, nothing to decode
        }

        // SAFETY: `state.session` is live; `avcc` outlives the synchronous
        // decode + wait; the refcon points at the boxed collector that lives as
        // long as the session. All FFI args are validated below.
        unsafe { decode_into(state, &avcc, pts_ns)? };

        if let Some(err) = state.collector.error.take() {
            return Err(err);
        }
        Ok(())
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: self.codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }
}

impl AsyncElement for VtDecode {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Consumes the configured codec at any geometry; intersecting narrows the
        // proposal and rejects a mismatched codec. Mirrors `MfDecode`.
        upstream_caps.intersect(&self.output_caps())
    }

    /// Native `DerivedOutput`: H.264 at any geometry in, NV12 at the same dims /
    /// framerate out (VideoToolbox derives the dims from the stream). Mirrors
    /// `MfDecode` / `FfmpegH264Dec`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let codec = self.codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            derive_output_caps(codec, input)
        }))
    }

    /// The emitted domain follows the mode: retained `CVPixelBuffer`s in
    /// `cv-output`, packed System bytes otherwise.
    fn output_memory(&self) -> MemoryDomainKind {
        if self.cv_output {
            MemoryDomainKind::CvPixelBuffer
        } else {
            MemoryDomainKind::System
        }
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[PropertySpec::new(
            "cv-output",
            PropKind::Bool,
            "emit retained IOSurface-backed CVPixelBuffers (zero-copy) instead of packed NV12",
        )
        .with_default("false")];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "cv-output" => {
                self.cv_output = value.as_bool().ok_or(PropError::Type)?;
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "cv-output" => Some(PropValue::Bool(self.cv_output)),
            _ => None,
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec, framerate, ..
            } if *codec == self.codec => {
                // The session is built lazily on the first access unit carrying
                // parameter sets, since VideoToolbox needs the actual SPS/PPS
                // bytes (in-band), not just the geometry.
                self.input_framerate = framerate.clone();
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
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed(slice, frame.timing.pts_ns)?;
                    // Drain whatever the callback packed.
                    let decoded = match self.state.as_mut() {
                        Some(st) => core::mem::take(&mut st.collector.frames),
                        None => Vec::new(),
                    };
                    for d in decoded {
                        self.emit(&d, out).await?;
                    }
                }
                PipelinePacket::CapsChanged(c) => {
                    // Two callers, like FfmpegVideoDec: an input CapsChanged from
                    // upstream (record the framerate for our emitted output caps;
                    // a resolution change is handled by the parameter-set rebuild
                    // in `feed`), or the runner's pre-fixed output caps for the
                    // downstream sink, forwarded so the sink sees them before the
                    // first decoded frame. Anything else is rejected loud, like
                    // MfDecode.
                    match &c {
                        Caps::CompressedVideo {
                            codec, framerate, ..
                        } if *codec == self.codec => {
                            self.input_framerate = framerate.clone();
                        }
                        Caps::RawVideo {
                            format: RawVideoFormat::Nv12,
                            ..
                        } => {
                            out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                            self.last_caps = Some(c);
                        }
                        _ => return Err(G2gError::CapsMismatch),
                    }
                }
                PipelinePacket::Flush => {
                    if let Some(st) = self.state.as_ref() {
                        // SAFETY: live session; synchronous wait on the owning
                        // thread. Discards in-flight frames on a flush.
                        unsafe {
                            st.session.wait_for_asynchronous_frames();
                        }
                    }
                    if let Some(st) = self.state.as_mut() {
                        st.collector.frames.clear();
                        st.collector.error = None;
                    }
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                }
                PipelinePacket::Eos => {
                    if let Some(st) = self.state.as_ref() {
                        // SAFETY: live session; drain delayed frames at EOS.
                        unsafe {
                            st.session.wait_for_asynchronous_frames();
                        }
                    }
                    let decoded = match self.state.as_mut() {
                        Some(st) => core::mem::take(&mut st.collector.frames),
                        None => Vec::new(),
                    };
                    for d in decoded {
                        self.emit(&d, out).await?;
                    }
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl VtDecode {
    async fn emit(&mut self, d: &DecodedFrame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let (width, height, pts_ns) = match d {
            DecodedFrame::Nv12 {
                width,
                height,
                pts_ns,
                ..
            }
            | DecodedFrame::Cv {
                width,
                height,
                pts_ns,
                ..
            } => (*width, *height, *pts_ns),
        };
        // A compressed stream's rate is advisory (per-frame PTS carries the
        // real timing); default to 30/1 when upstream did not declare one.
        let framerate = match &self.input_framerate {
            Rate::Fixed(q) => Rate::Fixed(*q),
            _ => Rate::Fixed(30 << 16),
        };
        // Both '420v' and '420f' are the NV12 byte layout, so the caps are the
        // same in either mode; only the memory domain differs.
        let new_caps = nv12_caps(width, height, framerate);
        if self.last_caps.as_ref() != Some(&new_caps) {
            out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                .await?;
            self.last_caps = Some(new_caps);
        }
        let domain = match d {
            DecodedFrame::Nv12 { nv12, .. } => {
                MemoryDomain::System(SystemSlice::from_boxed(nv12.clone()))
            }
            DecodedFrame::Cv {
                buf,
                pixel_format,
                io_surface_backed,
                ..
            } => {
                let ptr = CFRetained::as_ptr(buf).as_ptr() as u64;
                MemoryDomain::CvPixelBuffer(OwnedCvPixelBuffer::new(
                    ptr,
                    width,
                    height,
                    *pixel_format,
                    *io_surface_backed,
                    Arc::new(CvBufferOwner(buf.clone())),
                ))
            }
        };
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
        Ok(())
    }
}

impl PadTemplates for VtDecode {
    /// Consumes H.264 or H.265 and produces NV12, all at any geometry. Memory
    /// domain (System, or `CvPixelBuffer` in `cv-output` mode) is not in caps.
    fn pad_templates() -> Vec<PadTemplate> {
        let compressed = |codec| Caps::CompressedVideo {
            codec,
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
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(Vec::from([
                compressed(VideoCodec::H264),
                compressed(VideoCodec::H265),
            ]))),
            PadTemplate::source(CapsSet::one(nv12)),
        ])
    }
}

fn nv12_caps(w: u32, h: u32, framerate: Rate) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate,
    }
}

fn derive_output_caps(codec: VideoCodec, input: &Caps) -> CapsSet {
    match input {
        Caps::CompressedVideo {
            codec: c,
            width,
            height,
            framerate,
        } if *c == codec => CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: width.clone(),
            height: height.clone(),
            framerate: framerate.clone(),
        }),
        _ => CapsSet::from_alternatives(Vec::new()),
    }
}

/// VideoToolbox output callback: VT hands us a decoded `CVImageBuffer` (a
/// `CVPixelBuffer`); we lock it, pack the bi-planar NV12 to a tight buffer, and
/// push it into the `Collector` the refcon points at.
///
/// SAFETY: invoked by VideoToolbox during `DecodeFrame` / `WaitForAsynchronous-
/// Frames` with a valid (or null, on error) image buffer; `refcon` is the
/// `*mut Collector` we passed at session create, which outlives the session.
unsafe extern "C-unwind" fn output_callback(
    refcon: *mut c_void,
    _source_frame_refcon: *mut c_void,
    status: OSStatus,
    _info_flags: VTDecodeInfoFlags,
    image_buffer: *mut CVImageBuffer,
    presentation_ts: CMTime,
    _duration: CMTime,
) {
    // SAFETY: refcon is the boxed Collector's stable address (see Collector).
    let collector = unsafe { &mut *(refcon as *mut Collector) };
    if status != 0 {
        collector.error = Some(G2gError::Hardware(HardwareError::Other));
        return;
    }
    if image_buffer.is_null() {
        return; // dropped frame, not an error
    }
    // A CVPixelBufferRef IS a CVImageBufferRef in CoreVideo (typedef), so this
    // reinterpret is sound.
    let pb = unsafe { &*(image_buffer as *const CVPixelBuffer) };

    let fmt = CVPixelBufferGetPixelFormatType(pb);
    if fmt != K_CV_PIXEL_FORMAT_420V && fmt != K_CV_PIXEL_FORMAT_420F {
        // In the default (None attributes) mode VT picked the format; if it is
        // not NV12, surface it rather than mis-packing. cv-output mode pins
        // '420v' via the destination attributes, so this cannot trip there.
        collector.error = Some(G2gError::Hardware(HardwareError::Other));
        return;
    }

    if collector.cv_output {
        // Zero-copy: keep the decoder's buffer and hand it downstream unread.
        // SAFETY: `pb` is valid for the callback; retain takes our own +1 so the
        // buffer outlives VideoToolbox's interest in it.
        let buf = unsafe { CFRetained::retain(NonNull::from(pb)) };
        let io_surface_backed = CVPixelBufferGetIOSurface(Some(pb)).is_some();
        collector.frames.push(DecodedFrame::Cv {
            buf,
            width: CVPixelBufferGetWidth(pb) as u32,
            height: CVPixelBufferGetHeight(pb) as u32,
            pixel_format: fmt,
            io_surface_backed,
            pts_ns: cmtime_to_ns(presentation_ts),
        });
        return;
    }

    let width = CVPixelBufferGetWidth(pb) as u32;
    let height = CVPixelBufferGetHeight(pb) as u32;
    let packed = pack_nv12_locked(pb, width as usize, height as usize);

    match packed {
        Some(nv12) => collector.frames.push(DecodedFrame::Nv12 {
            nv12,
            width,
            height,
            pts_ns: cmtime_to_ns(presentation_ts),
        }),
        None => collector.error = Some(G2gError::Hardware(HardwareError::Other)),
    }
}

/// Build a VideoToolbox session from the parameter sets (H.264 SPS+PPS or H.265
/// VPS+SPS+PPS, in `params` order), with the NV12 output callback wired to a
/// fresh boxed collector. Dispatches to the H.264 or HEVC format-description
/// constructor by `codec`.
///
/// SAFETY: parameter-set byte buffers and the pointer/size arrays are live for
/// the create call; out-params are valid slots; the boxed collector outlives the
/// session (stored in the returned state).
unsafe fn build_session(
    codec: VideoCodec,
    params: &[Vec<u8>],
    cv_output: bool,
) -> Result<DecoderState, G2gError> {
    // Parameter-set pointer + size arrays, as VideoToolbox expects for the
    // ...FromH264/HEVCParameterSets constructor.
    let ptrs: Vec<NonNull<u8>> = params
        .iter()
        .map(|p| NonNull::new(p.as_ptr() as *mut u8).ok_or(G2gError::CapsMismatch))
        .collect::<Result<_, _>>()?;
    let sizes: Vec<usize> = params.iter().map(|p| p.len()).collect();
    let ptrs_arg = NonNull::new(ptrs.as_ptr() as *mut NonNull<u8>).ok_or(G2gError::CapsMismatch)?;
    let sizes_arg = NonNull::new(sizes.as_ptr() as *mut usize).ok_or(G2gError::CapsMismatch)?;

    let mut fmt: *const CMFormatDescription = ptr::null();
    // SAFETY: ptrs/sizes outlive the call; out slot is valid. HEVC takes an extra
    // `extensions` dictionary (None here); otherwise the two constructors match.
    let st = unsafe {
        match codec {
            VideoCodec::H265 => CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                None,
                params.len(),
                ptrs_arg,
                sizes_arg,
                4,    // 4-byte AVCC length prefix (matches `to_avcc`)
                None, // no extensions dictionary
                NonNull::from(&mut fmt),
            ),
            _ => CMVideoFormatDescriptionCreateFromH264ParameterSets(
                None,
                params.len(),
                ptrs_arg,
                sizes_arg,
                4,
                NonNull::from(&mut fmt),
            ),
        }
    };
    if st != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    // Adopt the +1 Create result (the CoreFoundation Create rule);
    // CMVideoFormatDescription is an alias of CMFormatDescription.
    let format = unsafe {
        CFRetained::from_raw(
            NonNull::new(fmt as *mut CMFormatDescription)
                .ok_or(G2gError::Hardware(HardwareError::Other))?,
        )
    };

    let mut collector = Box::new(Collector {
        cv_output,
        ..Collector::default()
    });
    let callback = VTDecompressionOutputCallbackRecord {
        decompressionOutputCallback: Some(output_callback),
        decompressionOutputRefCon: collector.as_mut() as *mut Collector as *mut c_void,
    };

    // cv-output mode pins the destination: '420v' NV12 in IOSurface-backed
    // buffers, so the handed-out frames are Metal-importable. The default
    // (None) lets VT pick, which the callback validates as NV12 either way.
    let dest_attrs = cv_output.then(|| {
        let fourcc = CFNumber::new_i32(K_CV_PIXEL_FORMAT_420V as i32);
        let io_props = CFDictionary::<CFString, CFType>::from_slices(&[], &[]);
        // SAFETY: the kCV keys are static CFStrings.
        let keys: [&CFString; 2] = unsafe {
            [
                kCVPixelBufferPixelFormatTypeKey,
                kCVPixelBufferIOSurfacePropertiesKey,
            ]
        };
        let values: [&CFType; 2] = [fourcc.as_ref(), io_props.as_ref()];
        CFDictionary::<CFString, CFType>::from_slices(&keys, &values)
    });
    let dest_attrs_ref: Option<&CFDictionary> = dest_attrs.as_deref().map(|d| d.as_ref());

    let mut session: *mut VTDecompressionSession = ptr::null_mut();
    // SAFETY: format description is live; callback record + out slot are valid;
    // None destination attributes lets VT pick a (bi-planar NV12) output, which
    // the callback validates.
    let st = unsafe {
        VTDecompressionSession::create(
            None,
            &format,
            None,
            dest_attrs_ref,
            &callback,
            NonNull::from(&mut session),
        )
    };
    if st != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let session = unsafe {
        CFRetained::from_raw(NonNull::new(session).ok_or(G2gError::Hardware(HardwareError::Other))?)
    };

    Ok(DecoderState {
        session,
        _format: format,
        collector,
        params: params.to_vec(),
    })
}

/// Decode one AVCC access unit synchronously: wrap it in a `CMBlockBuffer` +
/// `CMSampleBuffer` with the frame timing, submit, and wait for the callback.
///
/// SAFETY: `state.session` is live; `avcc` outlives the synchronous decode; the
/// created CoreMedia objects are adopted into `CFRetained` and released on drop.
unsafe fn decode_into(state: &DecoderState, avcc: &[u8], pts_ns: u64) -> Result<(), G2gError> {
    // Block buffer that owns a copy of the AVCC bytes (null memory_block + a
    // length lets VT allocate; ReplaceDataBytes fills it).
    let mut block: *mut CMBlockBuffer = ptr::null_mut();
    let st = unsafe {
        CMBlockBuffer::create_with_memory_block(
            None,
            ptr::null_mut(),
            avcc.len(),
            None, // default allocator -> VT allocates the backing block
            ptr::null(),
            0,
            avcc.len(),
            0, // CMBlockBufferFlags is a u32 alias, not bitflags: no empty()
            NonNull::from(&mut block),
        )
    };
    if st != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let block = unsafe {
        CFRetained::from_raw(NonNull::new(block).ok_or(G2gError::Hardware(HardwareError::Other))?)
    };
    // SAFETY: copy our AVCC bytes into the freshly allocated block.
    let st = unsafe {
        CMBlockBuffer::replace_data_bytes(
            NonNull::new(avcc.as_ptr() as *mut c_void)
                .ok_or(G2gError::Hardware(HardwareError::Other))?,
            &block,
            0,
            avcc.len(),
        )
    };
    if st != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }

    let timing = CMSampleTimingInfo {
        duration: cm_time_invalid(),
        presentationTimeStamp: cm_time(pts_ns),
        decodeTimeStamp: cm_time_invalid(),
    };
    let sizes = [avcc.len()];
    let mut sample: *mut CMSampleBuffer = ptr::null_mut();
    // SAFETY: block + format are live; arrays outlive the call; out slot valid.
    let st = unsafe {
        CMSampleBuffer::create_ready(
            None,
            // CFRetained<T> -> &T explicitly: Option does not deref-coerce through
            // the &, unlike a bare &T argument.
            Some(&*block),
            Some(&*state._format),
            1,
            1,
            &timing,
            1,
            sizes.as_ptr(),
            NonNull::from(&mut sample),
        )
    };
    if st != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let sample = unsafe {
        CFRetained::from_raw(NonNull::new(sample).ok_or(G2gError::Hardware(HardwareError::Other))?)
    };

    let mut info = VTDecodeInfoFlags::empty();
    // SAFETY: session + sample live; synchronous decode (no async flag), so the
    // callback runs before the wait returns.
    let st = unsafe {
        state.session.decode_frame(
            &sample,
            VTDecodeFrameFlags::empty(),
            ptr::null_mut(),
            &mut info,
        )
    };
    if st != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    // SAFETY: drain so the callback has fired for this frame before we return.
    unsafe {
        state.session.wait_for_asynchronous_frames();
    }
    Ok(())
}

/// A `CMTime` for `pts_ns` at nanosecond timescale, valid.
fn cm_time(pts_ns: u64) -> CMTime {
    CMTime {
        value: pts_ns as i64,
        timescale: 1_000_000_000,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}

/// An invalid `CMTime` (unknown duration / DTS).
fn cm_time_invalid() -> CMTime {
    CMTime {
        value: 0,
        timescale: 0,
        flags: CMTimeFlags::empty(),
        epoch: 0,
    }
}

/// Convert a valid nanosecond-or-other-timescale `CMTime` back to nanoseconds.
fn cmtime_to_ns(t: CMTime) -> u64 {
    if t.timescale <= 0 || !t.flags.contains(CMTimeFlags::Valid) {
        return 0;
    }
    ((t.value as i128 * 1_000_000_000) / t.timescale as i128) as u64
}
