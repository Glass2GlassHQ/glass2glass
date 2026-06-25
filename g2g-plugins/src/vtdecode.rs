//! M218: macOS hardware H.264 decode via VideoToolbox (`VTDecompressionSession`).
//!
//! `VtDecode` is the macOS counterpart of `MfDecode` (Windows Media Foundation):
//! it consumes Annex-B H.264 `DataFrame`s (`MemoryDomain::System`, what
//! `RtspSrc` / `H264Parse` emit) and produces decoded NV12 frames, also
//! `MemoryDomain::System` (a CPU copy out of the decoder's `CVPixelBuffer`). It
//! is the first element of the macOS platform track (DESIGN_TODO.md "Platform:
//! macOS"); a zero-copy `CVPixelBuffer` / `IOSurface` memory domain and a Metal
//! present sink are the follow-ups.
//!
//! Unlike Media Foundation, VideoToolbox wants AVCC framing and the SPS/PPS
//! parameter sets supplied out of band (in the `CMVideoFormatDescription`), not
//! Annex-B with in-band parameter sets. The element therefore pulls the SPS/PPS
//! out of each access unit ([`crate::annexb::h264_parameter_sets`]) to build the
//! format description, and converts the remaining VCL/SEI NALs to AVCC
//! ([`crate::annexb::to_avcc`]) for the decode sample. Those helpers are pure and
//! unit-tested on the host; the VideoToolbox session itself is macOS-only.
//!
//! COMPILE-PENDING: this module is `#[cfg(all(target_os = "macos", feature =
//! "vtdecode"))]`, so it never builds in this repo's Linux CI. It is written
//! against the real objc2 0.3.2 binding signatures (verified against the fetched
//! crate source, per the `mf-decode` rule in AGENTS.md), but it has NOT been
//! compiled. Expect to adjust on the first `cargo build` on a Mac: a few objc2
//! import paths (`OSStatus`, the `CFRetained` adopt helper), the
//! `CVImageBuffer` -> `CVPixelBuffer` cast, and the `CMVideoFormatDescription` /
//! `CMFormatDescription` type relationship. Each such spot is marked `// NOTE`.
//! The g2g element contract (caps, pad templates, `process` loop, `Send`) mirrors
//! `MfDecode` and is the stable part.

// objc2 renamed these free CoreMedia / VideoToolbox functions to associated
// functions (e.g. `CMBlockBuffer::create_with_memory_block`). VtDecode is
// compile-pending (never run on a Mac), so the migration is deferred to when it
// is validated on real hardware; the deprecated calls behave identically. TODO:
// migrate to the associated-function forms.
#![allow(deprecated)]

use core::ffi::c_void;
use core::ptr::{self, NonNull};

use objc2_core_foundation::CFRetained;

// OSStatus is `pub(crate)` in the objc2 framework crates (not importable). It is
// a transparent `i32` alias, so a local alias matches the FFI signatures exactly.
#[allow(non_camel_case_types)]
type OSStatus = i32;
use objc2_core_media::{
    CMBlockBuffer, CMBlockBufferCreateWithMemoryBlock,
    CMBlockBufferReplaceDataBytes, CMFormatDescription, CMSampleBuffer, CMSampleBufferCreateReady,
    CMSampleTimingInfo, CMTime, CMTimeFlags, CMVideoFormatDescriptionCreateFromH264ParameterSets,
};
use objc2_core_video::{
    CVImageBuffer, CVPixelBuffer, CVPixelBufferGetBaseAddressOfPlane,
    CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeight, CVPixelBufferGetHeightOfPlane,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferGetWidthOfPlane,
    CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
};
use objc2_video_toolbox::{
    VTDecodeFrameFlags, VTDecodeInfoFlags, VTDecompressionOutputCallbackRecord,
    VTDecompressionSession, VTDecompressionSessionCreate, VTDecompressionSessionDecodeFrame,
    VTDecompressionSessionWaitForAsynchronousFrames,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, Rate,
    RawVideoFormat, VideoCodec,
};

use crate::annexb::{h264_nal_type, h264_parameter_sets, to_avcc};

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;

/// The two NV12 (4:2:0 bi-planar) pixel formats VideoToolbox emits for H.264:
/// video-range `'420v'` and full-range `'420f'`. We accept either and pack to
/// our NV12 byte layout; the BT.601 / range semantics ride in caps, not here.
const K_CV_PIXEL_FORMAT_420V: u32 = 0x3432_3076; // '420v'
const K_CV_PIXEL_FORMAT_420F: u32 = 0x3432_3066; // '420f'

/// One decoded frame the output callback has packed to tight NV12.
#[derive(Debug)]
struct DecodedFrame {
    nv12: Box<[u8]>,
    width: u32,
    height: u32,
    pts_ns: u64,
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
}

/// Live VideoToolbox session plus the parameter sets it was built from (so a
/// mid-stream SPS/PPS change triggers a rebuild) and the boxed collector the
/// callback writes into.
struct DecoderState {
    session: CFRetained<VTDecompressionSession>,
    // Kept alive because the session's CMVideoFormatDescription references it.
    _format: CFRetained<CMFormatDescription>,
    collector: Box<Collector>,
    sps: Vec<Vec<u8>>,
    pps: Vec<Vec<u8>>,
}

impl core::fmt::Debug for DecoderState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DecoderState").field("collector", &self.collector).finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct VtDecode {
    codec: VideoCodec,
    configured: bool,
    state: Option<DecoderState>,
    last_caps: Option<Caps>,
    input_caps: Option<Caps>,
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
    /// An H.264 VideoToolbox decoder. (HEVC is a follow-up: the only difference
    /// is `CMVideoFormatDescriptionCreateFromHEVCParameterSets` and the VPS/SPS/
    /// PPS NAL types; the element shape is identical.)
    pub fn h264() -> Self {
        Self {
            codec: VideoCodec::H264,
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

    /// (Re)build the VideoToolbox session when the parameter sets first appear or
    /// change. No-op when the current session already matches `sps`/`pps`.
    fn ensure_session(&mut self, sps: &[Vec<u8>], pps: &[Vec<u8>]) -> Result<(), G2gError> {
        if sps.is_empty() || pps.is_empty() {
            return Ok(()); // can't build without parameter sets; wait for a keyframe
        }
        if let Some(st) = self.state.as_ref() {
            if st.sps == sps && st.pps == pps {
                return Ok(());
            }
        }
        // SAFETY: all pointers below are into live local buffers (the parameter
        // set NAL bytes and the pointer/size arrays) that outlive the create
        // call, and the out-params are valid stack slots. The created session and
        // format description are +1 retained (the CoreFoundation Create rule) and
        // adopted into `CFRetained` so they release on drop.
        let state = unsafe { build_session(sps, pps)? };
        self.state = Some(state);
        Ok(())
    }

    /// Decode one Annex-B access unit, packing any produced NV12 frames into the
    /// state collector for `process` to drain.
    fn feed(&mut self, au: &[u8], pts_ns: u64) -> Result<(), G2gError> {
        // VideoToolbox builds the format description from the parameter sets, so
        // pull SPS/PPS out of the access unit and (re)build the session on a
        // keyframe; until the first parameter sets arrive there is nothing to
        // decode against (mirrors RtspSrc's first-keyframe skip).
        let (sps, pps) = h264_parameter_sets(au);
        self.ensure_session(&sps, &pps)?;
        let Some(state) = self.state.as_mut() else {
            return Ok(()); // pre-keyframe: skip
        };

        // The decode sample carries only the VCL (+ SEI) NALs in AVCC framing;
        // SPS / PPS / AUD live in the format description.
        let avcc = to_avcc(au, |nal| !matches!(h264_nal_type(nal), Some(7 | 8 | 9)));
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
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
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
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| derive_output_caps(codec, input)))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo { codec, .. } if *codec == self.codec => {
                // The session is built lazily on the first access unit carrying
                // parameter sets, since VideoToolbox needs the actual SPS/PPS
                // bytes (in-band), not just the geometry.
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed(slice.as_slice(), frame.timing.pts_ns)?;
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
                    // Reject an incompatible mid-stream codec change loud, like
                    // MfDecode; a resolution change is handled by the parameter-
                    // set rebuild in `feed`.
                    match &c {
                        Caps::CompressedVideo { codec, .. } if *codec == self.codec => {}
                        _ => return Err(G2gError::CapsMismatch),
                    }
                    self.input_caps = Some(c);
                }
                PipelinePacket::Flush => {
                    if let Some(st) = self.state.as_ref() {
                        // SAFETY: live session; synchronous wait on the owning
                        // thread. Discards in-flight frames on a flush.
                        unsafe {
                            VTDecompressionSessionWaitForAsynchronousFrames(&st.session);
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
                            VTDecompressionSessionWaitForAsynchronousFrames(&st.session);
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
            }
            Ok(())
        })
    }
}

impl VtDecode {
    async fn emit(&mut self, d: &DecodedFrame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let new_caps = nv12_caps(d.width, d.height);
        if self.last_caps.as_ref() != Some(&new_caps) {
            out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
            self.last_caps = Some(new_caps);
        }
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(d.nv12.clone())),
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
        Ok(())
    }
}

impl PadTemplates for VtDecode {
    /// Consumes H.264 and produces NV12, both at any geometry. Memory domain
    /// (System today, `CVPixelBuffer`/`IOSurface` later) is not encoded in caps.
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
    // NOTE (verify on-device): a CVPixelBufferRef IS a CVImageBufferRef in
    // CoreVideo (typedef), so this reinterpret is sound; objc2 may instead expose
    // a checked downcast (`CVImageBuffer` -> `CVPixelBuffer`) to prefer.
    let pb = unsafe { &*(image_buffer as *const CVPixelBuffer) };

    let fmt = CVPixelBufferGetPixelFormatType(pb);
    if fmt != K_CV_PIXEL_FORMAT_420V && fmt != K_CV_PIXEL_FORMAT_420F {
        // We did not request a destination format (None attributes), so VT picked
        // one; if it is not NV12, surface it rather than mis-packing. To force
        // NV12, build destination_image_buffer_attributes with
        // kCVPixelBufferPixelFormatTypeKey = '420v' (a CFDictionary).
        collector.error = Some(G2gError::Hardware(HardwareError::Other));
        return;
    }

    // SAFETY: lock for read while we copy the planes out, unlock after.
    let lock = unsafe { CVPixelBufferLockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly) };
    if lock != 0 {
        collector.error = Some(G2gError::Hardware(HardwareError::Other));
        return;
    }
    let width = CVPixelBufferGetWidth(pb) as u32;
    let height = CVPixelBufferGetHeight(pb) as u32;
    let packed = unsafe { pack_nv12(pb, width as usize, height as usize) };
    // SAFETY: paired with the lock above.
    unsafe {
        CVPixelBufferUnlockBaseAddress(pb, CVPixelBufferLockFlags::ReadOnly);
    }

    match packed {
        Some(nv12) => collector.frames.push(DecodedFrame {
            nv12,
            width,
            height,
            pts_ns: cmtime_to_ns(presentation_ts),
        }),
        None => collector.error = Some(G2gError::Hardware(HardwareError::Other)),
    }
}

/// Copy the locked bi-planar pixel buffer into a tight NV12 byte buffer
/// (`w*h` luma + `w*(h/2)` interleaved chroma), stripping per-row padding.
///
/// SAFETY: `pb` is locked for read; plane base addresses / strides are valid for
/// the plane dimensions VideoToolbox reports.
unsafe fn pack_nv12(pb: &CVPixelBuffer, width: usize, height: usize) -> Option<Box<[u8]>> {
    let mut out = Vec::with_capacity(width * height * 3 / 2);
    // Plane 0: luma (w x h). Plane 1: interleaved CbCr (w x h/2 bytes/row).
    for plane in 0..2usize {
        let base = CVPixelBufferGetBaseAddressOfPlane(pb, plane) as *const u8;
        if base.is_null() {
            return None;
        }
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pb, plane);
        let pw = CVPixelBufferGetWidthOfPlane(pb, plane); // luma: w, chroma: w/2 (CbCr pairs)
        let ph = CVPixelBufferGetHeightOfPlane(pb, plane); // luma: h, chroma: h/2
        // Bytes per row of valid data: luma = pw, chroma = pw * 2 (CbCr pair).
        let row_bytes = if plane == 0 { pw } else { pw * 2 };
        for row in 0..ph {
            // SAFETY: row < plane height, row_bytes <= stride, base valid for the
            // plane; the source slice stays within the locked plane.
            let src = unsafe { core::slice::from_raw_parts(base.add(row * stride), row_bytes) };
            out.extend_from_slice(src);
        }
    }
    Some(out.into_boxed_slice())
}

/// Build a VideoToolbox session from the H.264 parameter sets, with the NV12
/// output callback wired to a fresh boxed collector.
///
/// SAFETY: parameter-set byte buffers and the pointer/size arrays are live for
/// the create call; out-params are valid slots; the boxed collector outlives the
/// session (stored in the returned state).
unsafe fn build_session(sps: &[Vec<u8>], pps: &[Vec<u8>]) -> Result<DecoderState, G2gError> {
    // Parameter-set pointer + size arrays (SPS first, then PPS), as VideoToolbox
    // expects for CMVideoFormatDescriptionCreateFromH264ParameterSets.
    let params: Vec<&[u8]> = sps.iter().chain(pps.iter()).map(|p| p.as_slice()).collect();
    let ptrs: Vec<NonNull<u8>> = params
        .iter()
        .map(|p| NonNull::new(p.as_ptr() as *mut u8).ok_or(G2gError::CapsMismatch))
        .collect::<Result<_, _>>()?;
    let sizes: Vec<usize> = params.iter().map(|p| p.len()).collect();

    let mut fmt: *const CMFormatDescription = ptr::null();
    // SAFETY: ptrs/sizes outlive the call; out slot is valid.
    let st = unsafe {
        CMVideoFormatDescriptionCreateFromH264ParameterSets(
            None,
            params.len(),
            NonNull::new(ptrs.as_ptr() as *mut NonNull<u8>).ok_or(G2gError::CapsMismatch)?,
            NonNull::new(sizes.as_ptr() as *mut usize).ok_or(G2gError::CapsMismatch)?,
            4, // 4-byte AVCC length prefix (matches `to_avcc`)
            NonNull::from(&mut fmt),
        )
    };
    if st != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    // NOTE (verify on-device): adopt the +1 Create result. The exact adopt helper
    // (`CFRetained::from_raw`) and whether CMVideoFormatDescription is the same
    // type as CMFormatDescription may need a tweak.
    let format = unsafe {
        CFRetained::from_raw(NonNull::new(fmt as *mut CMFormatDescription).ok_or(
            G2gError::Hardware(HardwareError::Other),
        )?)
    };

    let mut collector = Box::new(Collector::default());
    let callback = VTDecompressionOutputCallbackRecord {
        decompressionOutputCallback: Some(output_callback),
        decompressionOutputRefCon: collector.as_mut() as *mut Collector as *mut c_void,
    };

    let mut session: *mut VTDecompressionSession = ptr::null_mut();
    // SAFETY: format description is live; callback record + out slot are valid;
    // None destination attributes lets VT pick a (bi-planar NV12) output, which
    // the callback validates.
    let st = unsafe {
        VTDecompressionSessionCreate(
            None,
            &format,
            None,
            None,
            &callback,
            NonNull::from(&mut session),
        )
    };
    if st != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let session = unsafe {
        CFRetained::from_raw(
            NonNull::new(session).ok_or(G2gError::Hardware(HardwareError::Other))?,
        )
    };

    Ok(DecoderState {
        session,
        _format: format,
        collector,
        sps: sps.to_vec(),
        pps: pps.to_vec(),
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
        CMBlockBufferCreateWithMemoryBlock(
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
        CMBlockBufferReplaceDataBytes(
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
        CMSampleBufferCreateReady(
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
        VTDecompressionSessionDecodeFrame(
            &state.session,
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
        VTDecompressionSessionWaitForAsynchronousFrames(&state.session);
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
    CMTime { value: 0, timescale: 0, flags: CMTimeFlags::empty(), epoch: 0 }
}

/// Convert a valid nanosecond-or-other-timescale `CMTime` back to nanoseconds.
fn cmtime_to_ns(t: CMTime) -> u64 {
    if t.timescale <= 0 || !t.flags.contains(CMTimeFlags::Valid) {
        return 0;
    }
    ((t.value as i128 * 1_000_000_000) / t.timescale as i128) as u64
}
