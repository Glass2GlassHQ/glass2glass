//! M231: macOS hardware H.264 / H.265 encode via VideoToolbox
//! (`VTCompressionSession`).
//!
//! `VtEncode` is the encode counterpart of [`VtDecode`](crate::vtdecode) and the
//! macOS analog of `MfEncode` (Windows Media Foundation): it consumes raw NV12
//! `DataFrame`s (`MemoryDomain::System`) and produces H.264 *or* H.265 (HEVC)
//! access units, also `MemoryDomain::System`, in Annex-B framing (what the rest
//! of the pipeline expects). The two codecs differ only in the session codec
//! FourCC (`'avc1'` vs `'hvc1'`), the keyframe-prefix parameter-set accessor
//! (`...GetH264/HEVCParameterSetAtIndex`), and IRAP detection; the session,
//! encode loop, and pixel-buffer fill are shared.
//!
//! VideoToolbox emits length-prefixed (AVCC) NALs and keeps the SPS/PPS in the
//! output `CMVideoFormatDescription`, not in the stream. So the element converts
//! each encoded sample to Annex-B ([`crate::annexb::avcc_to_annexb`]) and, on a
//! keyframe, prepends the parameter sets (pulled from the sample's format
//! description) so the elementary stream is self-contained for a downstream
//! decoder / muxer. Those framing helpers are pure and host-tested; the
//! VideoToolbox session is macOS-only.
//!
//! Gated `#[cfg(all(target_os = "macos", feature = "vtencode"))]`, so it never
//! builds on the Linux dev host; the macOS CI job compiles it and runs the
//! `m731_videotoolbox` tests (encode to Annex-B + decode round trip, H.264 +
//! HEVC), so the encode path is runtime-validated on a real Mac.

use core::ffi::{c_char, c_void};
use core::ptr::{self, NonNull};

use objc2_core_foundation::CFRetained;

// OSStatus is `pub(crate)` in the objc2 framework crates; a local i32 alias
// matches the FFI signatures exactly (same as VtDecode).
#[allow(non_camel_case_types)]
type OSStatus = i32;
use objc2_core_media::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMTime, CMTimeFlags,
    CMVideoFormatDescriptionGetH264ParameterSetAtIndex,
    CMVideoFormatDescriptionGetHEVCParameterSetAtIndex,
};
use objc2_core_video::{
    CVImageBuffer, CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
    CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferGetHeightOfPlane,
    CVPixelBufferGetWidthOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress,
};
use objc2_video_toolbox::{
    kVTCompressionPropertyKey_AllowFrameReordering, kVTCompressionPropertyKey_AverageBitRate,
    kVTCompressionPropertyKey_MaxKeyFrameInterval, kVTCompressionPropertyKey_RealTime,
    VTCompressionSession, VTEncodeInfoFlags, VTSessionSetProperty,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat, VideoCodec,
};

use crate::annexb::{au_is_keyframe, avcc_to_annexb};

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::future::Future;
use core::pin::Pin;

/// H.264 codec type FourCC `'avc1'` for `VTCompressionSessionCreate`.
const CODEC_H264: u32 = 0x6176_6331;
/// H.265 (HEVC) codec type FourCC `'hvc1'` for `VTCompressionSessionCreate`.
const CODEC_H265: u32 = 0x6876_6331;
/// NV12 4:2:0 bi-planar video-range `'420v'`, the pixel format we feed in.
const K_CV_PIXEL_FORMAT_420V: u32 = 0x3432_3076;

/// One encoded access unit the output callback has framed to Annex-B.
#[derive(Debug)]
struct EncodedFrame {
    annexb: Box<[u8]>,
    pts_ns: u64,
    keyframe: bool,
}

/// Shared sink the VideoToolbox output callback writes into. The session's
/// callback refcon is a raw `*mut Collector`; the box keeps a stable address for
/// the session's life. Written only during `EncodeFrame` / `CompleteFrames`
/// (synchronous, on the element task) and drained by `process` after, so the two
/// never touch it at once. Mirrors `VtDecode::Collector`.
#[derive(Debug)]
struct Collector {
    frames: Vec<EncodedFrame>,
    error: Option<G2gError>,
    /// The session codec, so the output callback picks the H.264 vs HEVC
    /// parameter-set accessor and IRAP detection.
    codec: VideoCodec,
}

/// Live VideoToolbox compression session + the boxed collector its callback
/// writes into.
struct EncoderState {
    session: CFRetained<VTCompressionSession>,
    collector: Box<Collector>,
    width: u32,
    height: u32,
}

impl core::fmt::Debug for EncoderState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EncoderState")
            .field("collector", &self.collector)
            .field("width", &self.width)
            .field("height", &self.height)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub struct VtEncode {
    codec: VideoCodec,
    bitrate: u32,
    keyframe_interval: u32,
    framerate: Rate,
    configured: bool,
    state: Option<EncoderState>,
    last_caps: Option<Caps>,
    emitted: u64,
}

// SAFETY: a `VTCompressionSession` and the CoreMedia objects are CoreFoundation
// types, thread-safe to retain/release but used here single-threaded. Like
// `VtDecode`, `VtEncode` is built for a single-thread executor: every encode call
// lands on the element's owning task, and the boxed `Collector` the callback
// writes is only read on that same task after `CompleteFrames`. The raw
// `*mut Collector` refcon never escapes the session, which `VtEncode` owns. We
// assert `Send` under that contract so the multi-thread runner accepts it.
unsafe impl Send for VtEncode {}

impl Default for VtEncode {
    fn default() -> Self {
        Self::h264()
    }
}

impl VtEncode {
    /// An H.264 VideoToolbox encoder. Defaults: 4 Mbps, a keyframe every 60
    /// frames, realtime + no frame reordering (DTS == PTS, low latency).
    pub fn h264() -> Self {
        Self::for_codec(VideoCodec::H264)
    }

    /// An H.265 (HEVC) VideoToolbox encoder, same defaults as [`h264`](Self::h264).
    /// Differs only in the session codec FourCC and the keyframe-prefix accessor.
    pub fn h265() -> Self {
        Self::for_codec(VideoCodec::H265)
    }

    fn for_codec(codec: VideoCodec) -> Self {
        Self {
            codec,
            bitrate: 4_000_000,
            keyframe_interval: 60,
            framerate: Rate::Any,
            configured: false,
            state: None,
            last_caps: None,
            emitted: 0,
        }
    }

    /// Set the target average bitrate (bits/sec).
    pub fn with_bitrate(mut self, bits_per_sec: u32) -> Self {
        self.bitrate = bits_per_sec;
        self
    }

    /// Set the maximum keyframe interval (frames between IDRs).
    pub fn with_keyframe_interval(mut self, frames: u32) -> Self {
        self.keyframe_interval = frames;
        self
    }

    /// Count of encoded access units pushed downstream.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn output_caps(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        }
    }

    /// Encode one NV12 frame, packing any produced access units into the state
    /// collector for `process` to drain.
    fn feed(&mut self, nv12: &[u8], pts_ns: u64) -> Result<(), G2gError> {
        let Some(state) = self.state.as_mut() else {
            return Err(G2gError::NotConfigured);
        };
        // SAFETY: the session is live; the pixel buffer is built from `nv12`
        // (which outlives the synchronous encode), fed, then completed so the
        // callback fires before we return; the refcon points at the boxed
        // collector that lives as long as the session.
        unsafe { encode_into(state, nv12, pts_ns)? };
        if let Some(err) = state.collector.error.take() {
            return Err(err);
        }
        Ok(())
    }

    async fn emit(&mut self, e: &EncodedFrame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let (w, h) = match &self.state {
            Some(s) => (s.width, s.height),
            None => return Err(G2gError::NotConfigured),
        };
        // Never emit an `Any` rate (a downstream transform cannot fixate() it);
        // default to 30/1 when the input caps did not declare one.
        let framerate = match &self.framerate {
            Rate::Fixed(q) => Rate::Fixed(*q),
            _ => Rate::Fixed(30 << 16),
        };
        let new_caps = h264_caps(self.codec, w, h, framerate);
        if self.last_caps.as_ref() != Some(&new_caps) {
            out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                .await?;
            self.last_caps = Some(new_caps);
        }
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(e.annexb.clone())),
            timing: FrameTiming {
                pts_ns: e.pts_ns,
                dts_ns: e.pts_ns, // no frame reordering: DTS == PTS
                duration_ns: 0,
                capture_ns: e.pts_ns,
                keyframe: e.keyframe,
                ..FrameTiming::default()
            },
            sequence: self.emitted,
            meta: Default::default(),
        };
        self.emitted += 1;
        out.push(PipelinePacket::DataFrame(frame)).await?;
        Ok(())
    }

    /// Drain everything the callback has packed and emit it downstream.
    async fn drain(&mut self, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let encoded = match self.state.as_mut() {
            Some(st) => core::mem::take(&mut st.collector.frames),
            None => Vec::new(),
        };
        for e in encoded {
            self.emit(&e, out).await?;
        }
        Ok(())
    }
}

impl AsyncElement for VtEncode {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&self.output_caps())
    }

    /// Native `DerivedOutput`: NV12 in, H.264 at the same dims / framerate out.
    /// Mirrors `MfEncode`.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let codec = self.codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            derive_output_caps(codec, input)
        }))
    }

    fn properties(&self) -> &'static [PropertySpec] {
        const PROPS: &[PropertySpec] = &[
            PropertySpec::new("bitrate", PropKind::Uint, "target bitrate, bits/second")
                .with_default("4000000"),
            PropertySpec::new(
                "max-keyframe-interval",
                PropKind::Uint,
                "maximum frames between keyframes",
            )
            .with_default("60"),
        ];
        PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "bitrate" => {
                self.bitrate = value.as_uint().ok_or(PropError::Type)? as u32;
                Ok(())
            }
            "max-keyframe-interval" => {
                self.keyframe_interval = (value.as_uint().ok_or(PropError::Type)? as u32).max(1);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "bitrate" => Some(PropValue::Uint(self.bitrate as u64)),
            "max-keyframe-interval" => Some(PropValue::Uint(self.keyframe_interval as u64)),
            _ => None,
        }
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h, framerate) = match absolute_caps {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                framerate,
            } if *w % 2 == 0 && *h % 2 == 0 => (*w, *h, framerate.clone()),
            _ => return Err(G2gError::CapsMismatch),
        };
        self.framerate = framerate;
        // SAFETY: dims are validated even/non-zero; the create out-param is a
        // valid slot; the session + boxed collector are stored in the returned
        // state, keeping the refcon's target alive for the session's life.
        let state =
            unsafe { build_session(self.codec, w, h, self.bitrate, self.keyframe_interval)? };
        self.state = Some(state);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> g2g_core::ElementMetadata {
        g2g_core::ElementMetadata::new(
            "VideoToolbox H.264 / H.265 encoder",
            "Codec/Encoder/Video",
            "Hardware H.264 / H.265 encode on macOS via VideoToolbox (NV12 -> Annex-B)",
            "g2g",
        )
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
                    self.drain(out).await?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // A mid-stream geometry / format change would need a session
                    // rebuild; reject anything but the exact configured NV12 shape
                    // loud (the bare format check let a new resolution slip past).
                    // The runner's pre-fixed output caps (our compressed codec)
                    // are forwarded so the sink sees them before the first access
                    // unit, like FfmpegVideoDec.
                    let session_dims = self.state.as_ref().map(|s| (s.width, s.height));
                    match &c {
                        Caps::RawVideo {
                            format: RawVideoFormat::Nv12,
                            width: Dim::Fixed(w),
                            height: Dim::Fixed(h),
                            ..
                        } if session_dims == Some((*w, *h)) => {}
                        Caps::CompressedVideo { codec, .. } if *codec == self.codec => {
                            out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                            self.last_caps = Some(c);
                        }
                        _ => return Err(G2gError::CapsMismatch),
                    }
                }
                PipelinePacket::Flush => {
                    if let Some(st) = self.state.as_ref() {
                        // SAFETY: live session; complete + discard in-flight frames.
                        unsafe { complete_frames(&st.session) };
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
                        // SAFETY: live session; flush delayed frames at EOS.
                        unsafe { complete_frames(&st.session) };
                    }
                    self.drain(out).await?;
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

impl PadTemplates for VtEncode {
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

fn h264_caps(codec: VideoCodec, w: u32, h: u32, framerate: Rate) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate,
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

/// VideoToolbox compression output callback: VT hands us an encoded
/// `CMSampleBuffer`; we pull its AVCC bytes, frame them as Annex-B, prepend the
/// SPS/PPS (from the sample's format description) on a keyframe, and push the
/// access unit into the `Collector` the refcon points at.
///
/// SAFETY: invoked by VideoToolbox during `EncodeFrame` / `CompleteFrames` with a
/// valid (or null, on a dropped frame) sample; `refcon` is the `*mut Collector`
/// passed at session create, which outlives the session.
unsafe extern "C-unwind" fn output_callback(
    refcon: *mut c_void,
    _source_frame_refcon: *mut c_void,
    status: OSStatus,
    _info_flags: VTEncodeInfoFlags,
    sample_buffer: *mut CMSampleBuffer,
) {
    // SAFETY: refcon is the boxed Collector's stable address (see Collector).
    let collector = unsafe { &mut *(refcon as *mut Collector) };
    if status != 0 {
        collector.error = Some(G2gError::Hardware(HardwareError::Other));
        return;
    }
    if sample_buffer.is_null() {
        return; // dropped frame, not an error
    }
    let sample = unsafe { &*sample_buffer };
    let codec = collector.codec;
    match unsafe { sample_to_annexb(codec, sample) } {
        Ok((annexb, pts_ns, keyframe)) => collector.frames.push(EncodedFrame {
            annexb,
            pts_ns,
            keyframe,
        }),
        Err(e) => collector.error = Some(e),
    }
}

/// Pull the encoded bytes (AVCC) out of `sample`, frame them as Annex-B, and on a
/// keyframe prepend the SPS/PPS from the format description so the elementary
/// stream is self-contained.
///
/// SAFETY: `sample` is a valid encoded `CMSampleBuffer` for the call's duration.
unsafe fn sample_to_annexb(
    codec: VideoCodec,
    sample: &CMSampleBuffer,
) -> Result<(Box<[u8]>, u64, bool), G2gError> {
    let hw = || G2gError::Hardware(HardwareError::Other);

    // The encoded bytes live in the sample's block buffer in AVCC framing.
    let block: CFRetained<CMBlockBuffer> = unsafe { sample.data_buffer() }.ok_or_else(hw)?;
    let mut total_len: usize = 0;
    let mut len_at: usize = 0;
    // c_char (i8) pointer per `CMBlockBuffer::data_pointer`; cast to u8 for the slice.
    let mut data_ptr: *mut c_char = ptr::null_mut();
    // objc2 exposes this as `CMBlockBuffer::data_pointer` (associated fn taking
    // &CMBlockBuffer); the out-params are usize lengths.
    let st = unsafe {
        CMBlockBuffer::data_pointer(&block, 0, &mut len_at, &mut total_len, &mut data_ptr)
    };
    if st != 0 || data_ptr.is_null() || total_len == 0 {
        return Err(hw());
    }
    // SAFETY: VideoToolbox guarantees `total_len` bytes at `data_ptr` for the
    // contiguous block (the encoder output is a single contiguous block).
    let avcc = unsafe { core::slice::from_raw_parts(data_ptr as *const u8, total_len) };
    let mut annexb = avcc_to_annexb(avcc);

    let pts_ns = cmtime_to_ns(unsafe { sample_pts(sample) });
    let keyframe = au_is_keyframe(codec, &annexb);
    if keyframe {
        // Prepend the parameter sets (Annex-B framed) ahead of the IRAP picture.
        if let Some(fmt) = unsafe { sample.format_description() } {
            let mut prefix = unsafe { parameter_sets_annexb(codec, &fmt)? };
            prefix.append(&mut annexb);
            annexb = prefix;
        }
    }
    Ok((annexb.into_boxed_slice(), pts_ns, keyframe))
}

/// Extract the parameter sets (H.264 SPS/PPS, or H.265 VPS/SPS/PPS) from a
/// `CMVideoFormatDescription`, each Annex-B framed (4-byte start code), in order.
///
/// SAFETY: `fmt` is a valid `CMVideoFormatDescription` for `codec`.
unsafe fn parameter_sets_annexb(
    codec: VideoCodec,
    fmt: &CMFormatDescription,
) -> Result<Vec<u8>, G2gError> {
    let hw = || G2gError::Hardware(HardwareError::Other);
    // The H.264 and HEVC accessors share the exact same signature; pick by codec.
    // SAFETY: all pointers are valid stack slots / null (allowed by the API).
    let get = |fmt: &CMFormatDescription,
               i: usize,
               p: *mut *const u8,
               sz: *mut usize,
               count: *mut usize|
     -> OSStatus {
        unsafe {
            match codec {
                VideoCodec::H265 => CMVideoFormatDescriptionGetHEVCParameterSetAtIndex(
                    fmt,
                    i,
                    p,
                    sz,
                    count,
                    ptr::null_mut(),
                ),
                _ => CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                    fmt,
                    i,
                    p,
                    sz,
                    count,
                    ptr::null_mut(),
                ),
            }
        }
    };
    let mut out = Vec::new();
    // First query index 0 to learn the parameter-set count.
    let mut count: usize = 0;
    let mut ptr0: *const u8 = ptr::null();
    let mut size0: usize = 0;
    let st = get(fmt, 0, &mut ptr0, &mut size0, &mut count);
    if st != 0 {
        return Err(hw());
    }
    for i in 0..count {
        let mut p: *const u8 = ptr::null();
        let mut sz: usize = 0;
        let st = get(fmt, i, &mut p, &mut sz, ptr::null_mut());
        if st != 0 || p.is_null() || sz == 0 {
            return Err(hw());
        }
        // SAFETY: VideoToolbox guarantees `sz` bytes at `p` for the format
        // description's lifetime (longer than this copy).
        let nal = unsafe { core::slice::from_raw_parts(p, sz) };
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
    }
    Ok(out)
}

/// Build a VideoToolbox compression session for `w`x`h`, wired to a fresh boxed
/// collector, with realtime + no-reorder + bitrate + keyframe-interval set.
///
/// SAFETY: out-param slot is valid; the boxed collector outlives the session
/// (stored in the returned state); property values are valid CF objects.
unsafe fn build_session(
    codec: VideoCodec,
    w: u32,
    h: u32,
    bitrate: u32,
    keyframe_interval: u32,
) -> Result<EncoderState, G2gError> {
    let hw = || G2gError::Hardware(HardwareError::Other);
    let mut collector = Box::new(Collector {
        frames: Vec::new(),
        error: None,
        codec,
    });
    let refcon = collector.as_mut() as *mut Collector as *mut c_void;
    let codec_type = match codec {
        VideoCodec::H265 => CODEC_H265,
        _ => CODEC_H264,
    };

    let mut session: *mut VTCompressionSession = ptr::null_mut();
    // objc2 exposes this as `VTCompressionSession::create` (associated fn).
    // `output_callback` is the `VTCompressionOutputCallback`.
    let st = unsafe {
        VTCompressionSession::create(
            None,
            w as i32,
            h as i32,
            codec_type,
            None,
            None,
            None,
            Some(output_callback),
            refcon,
            NonNull::from(&mut session),
        )
    };
    if st != 0 {
        return Err(hw());
    }
    let session = unsafe { CFRetained::from_raw(NonNull::new(session).ok_or_else(hw)?) };

    // Low-latency tuning. VTSessionSetProperty takes the session as a VTSession,
    // a CFString key, and a CFType value.
    unsafe {
        set_bool(&session, kVTCompressionPropertyKey_RealTime, true)?;
        set_bool(
            &session,
            kVTCompressionPropertyKey_AllowFrameReordering,
            false,
        )?;
        set_u32(&session, kVTCompressionPropertyKey_AverageBitRate, bitrate)?;
        set_u32(
            &session,
            kVTCompressionPropertyKey_MaxKeyFrameInterval,
            keyframe_interval,
        )?;
    }

    Ok(EncoderState {
        session,
        collector,
        width: w,
        height: h,
    })
}

/// Encode one NV12 frame: wrap it in a `CVPixelBuffer`, submit, and complete so
/// the callback has fired before returning.
///
/// SAFETY: `state.session` is live; `nv12` outlives the synchronous encode.
unsafe fn encode_into(state: &EncoderState, nv12: &[u8], pts_ns: u64) -> Result<(), G2gError> {
    let hw = || G2gError::Hardware(HardwareError::Other);
    let pixel_buffer = unsafe { make_pixel_buffer(nv12, state.width, state.height)? };

    let mut info = VTEncodeInfoFlags::empty();
    // `encode_frame` takes the image buffer, PTS, duration, optional
    // frame-properties dict, source refcon, and an info-flags out-param. A
    // CVPixelBuffer is a CVImageBuffer (typedef).
    let image: &CVImageBuffer =
        unsafe { &*(CFRetained::as_ptr(&pixel_buffer).as_ptr() as *const CVImageBuffer) };
    let st = unsafe {
        state.session.encode_frame(
            image,
            cm_time(pts_ns),
            cm_time_invalid(),
            None,
            ptr::null_mut(),
            &mut info,
        )
    };
    if st != 0 {
        return Err(hw());
    }
    // Drain this frame (no reordering, so one input yields one output promptly).
    // A pipelined drain that doesn't complete every frame is a follow-up.
    unsafe { complete_frames(&state.session) };
    Ok(())
}

/// Complete (flush) all pending frames so their callbacks fire.
///
/// SAFETY: `session` is live.
unsafe fn complete_frames(session: &VTCompressionSession) {
    // Completing up to an invalid PTS flushes everything.
    let _ = unsafe { session.complete_frames(cm_time_invalid()) };
}

/// Build a bi-planar NV12 `CVPixelBuffer` and copy `nv12` into its planes
/// (inverse of `VtDecode::pack_nv12`).
///
/// SAFETY: `nv12` is `w*h * 3/2` bytes; the created buffer is released on drop.
unsafe fn make_pixel_buffer(
    nv12: &[u8],
    w: u32,
    h: u32,
) -> Result<CFRetained<CVPixelBuffer>, G2gError> {
    let hw = || G2gError::Hardware(HardwareError::Other);
    let (wu, hu) = (w as usize, h as usize);
    if nv12.len() < wu * hu * 3 / 2 {
        return Err(hw());
    }
    let mut pb: *mut CVPixelBuffer = ptr::null_mut();
    // CVPixelBufferCreate(allocator, w, h, fourcc, attrs, out). None attributes
    // lets CoreVideo pick the plane strides.
    let st = unsafe {
        CVPixelBufferCreate(
            None,
            wu,
            hu,
            K_CV_PIXEL_FORMAT_420V,
            None,
            NonNull::from(&mut pb),
        )
    };
    if st != 0 {
        return Err(hw());
    }
    let pb = unsafe { CFRetained::from_raw(NonNull::new(pb).ok_or_else(hw)?) };

    // SAFETY: lock for write, copy the planes in (stripping into the buffer's own
    // per-row stride), unlock.
    let lock = unsafe { CVPixelBufferLockBaseAddress(&pb, CVPixelBufferLockFlags::empty()) };
    if lock != 0 {
        return Err(hw());
    }
    let mut src = 0usize;
    for plane in 0..2usize {
        let base = CVPixelBufferGetBaseAddressOfPlane(&pb, plane) as *mut u8;
        if base.is_null() {
            unsafe { CVPixelBufferUnlockBaseAddress(&pb, CVPixelBufferLockFlags::empty()) };
            return Err(hw());
        }
        let stride = CVPixelBufferGetBytesPerRowOfPlane(&pb, plane);
        let pw = CVPixelBufferGetWidthOfPlane(&pb, plane);
        let ph = CVPixelBufferGetHeightOfPlane(&pb, plane);
        let row_bytes = if plane == 0 { pw } else { pw * 2 };
        for row in 0..ph {
            if src + row_bytes > nv12.len() {
                break;
            }
            // SAFETY: row < plane height; row_bytes <= stride; base valid for the plane.
            let dst = unsafe { core::slice::from_raw_parts_mut(base.add(row * stride), row_bytes) };
            dst.copy_from_slice(&nv12[src..src + row_bytes]);
            src += row_bytes;
        }
    }
    unsafe { CVPixelBufferUnlockBaseAddress(&pb, CVPixelBufferLockFlags::empty()) };
    Ok(pb)
}

/// Set a CFBoolean session property.
unsafe fn set_bool(
    session: &VTCompressionSession,
    key: &objc2_core_foundation::CFString,
    value: bool,
) -> Result<(), G2gError> {
    // `CFBoolean::new` returns the shared `&'static CFBoolean` singleton.
    let v = objc2_core_foundation::CFBoolean::new(value);
    let st = unsafe { VTSessionSetProperty(session, key, Some(v as &_)) };
    if st != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    Ok(())
}

/// Set a CFNumber (u32) session property. NOTE: CFNumber handling is compile-pending.
unsafe fn set_u32(
    session: &VTCompressionSession,
    key: &objc2_core_foundation::CFString,
    value: u32,
) -> Result<(), G2gError> {
    let n = value as i64;
    let num = unsafe {
        objc2_core_foundation::CFNumber::new(
            None,
            objc2_core_foundation::CFNumberType::SInt64Type,
            &n as *const i64 as *const c_void,
        )
    };
    let st = unsafe { VTSessionSetProperty(session, key, num.as_deref().map(|x| x as &_)) };
    if st != 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    Ok(())
}

/// A valid `CMTime` for `pts_ns` at nanosecond timescale.
fn cm_time(pts_ns: u64) -> CMTime {
    CMTime {
        value: pts_ns as i64,
        timescale: 1_000_000_000,
        flags: CMTimeFlags::Valid,
        epoch: 0,
    }
}

/// An invalid `CMTime` (unknown duration).
fn cm_time_invalid() -> CMTime {
    CMTime {
        value: 0,
        timescale: 0,
        flags: CMTimeFlags::empty(),
        epoch: 0,
    }
}

/// The sample's presentation timestamp.
///
/// SAFETY: `sample` is a valid `CMSampleBuffer`.
unsafe fn sample_pts(sample: &CMSampleBuffer) -> CMTime {
    unsafe { sample.presentation_time_stamp() }
}

/// Convert a valid `CMTime` to nanoseconds.
fn cmtime_to_ns(t: CMTime) -> u64 {
    if t.timescale <= 0 || !t.flags.contains(CMTimeFlags::Valid) {
        return 0;
    }
    ((t.value as i128 * 1_000_000_000) / t.timescale as i128) as u64
}
