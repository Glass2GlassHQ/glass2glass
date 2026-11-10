// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! Windows H.264 / H.265 decode element wrapping a Media Foundation decoder
//! MFT (`CLSID_MSH264DecoderMFT` or `CLSID_MSH265DecoderMFT`, an
//! `IMFTransform`). Codec is selected with [`MfDecode::with_codec`]; the
//! default is H.264.
//!
//! Consumes Annex-B H.264 `DataFrame`s (the bitstream `RtspSrc` /
//! `H264Parse` already emit, `MemoryDomain::System`) and produces decoded
//! NV12 frames, also `MemoryDomain::System` (CPU copy out of the MFT's output
//! buffer). A `CapsChanged(Nv12, w, h)` is emitted before the first decoded
//! frame and again whenever the decoder signals a resolution change.
//!
//! The same pipeline carries H.265/HEVC when constructed with
//! `with_codec(VideoCodec::H265)`. The MS HEVC decoder MFT ships as the
//! Store "HEVC Video Extensions" on many SKUs, so its `CoCreateInstance` can
//! fail with `REGDB_E_CLASSNOTREG` when absent; that surfaces as a loud
//! `Hardware` error at `configure_pipeline` rather than a silent fallback.
//!
//! Threading: COM is initialised multi-threaded (MTA) in `configure_pipeline`
//! and every `IMFTransform` call runs on that same thread. `MfDecode` is
//! therefore `!Send` and must run on a current-thread / single-thread executor.
//!
//! NV12 stride: the MFT's output buffer may carry per-row padding when the
//! reported `MF_MT_DEFAULT_STRIDE` exceeds the width (hardware MFTs align rows
//! up; the MS software decoder packs tightly). `pack_nv12` strips that
//! padding so downstream always sees tightly-packed NV12.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use windows::Win32::Media::MediaFoundation::{
    CLSID_MSH264DecoderMFT, CLSID_MSH265DecoderMFT, IMFTransform, MFCreateMediaType,
    MFMediaType_Video, MFShutdown, MFStartup, MFVideoFormat_H264, MFVideoFormat_HEVC,
    MFVideoFormat_NV12, MFSTARTUP_FULL, MFT_MESSAGE_COMMAND_FLUSH,
    MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, MFT_MESSAGE_NOTIFY_START_OF_STREAM, MF_MT_DEFAULT_STRIDE,
    MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_VERSION,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
};

use crate::inputstamp::{InputStamp, PendingInputStamps};
use crate::mfcommon::{
    feed_sample, make_input_sample, mf_err, mf_ticks_to_ns, ns_to_mf_ticks,
    output_provides_samples, pack_u32_pair, process_output, read_sample_bytes, sample_time_ns,
    send_drain_command, unpack_u32_pair, OutputPull,
};
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, Interlace, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    Rate, RawVideoFormat, Segment, VideoCodec,
};

/// Live MFT plus the negotiated output geometry. Recreated nowhere — the
/// transform persists for the element's lifetime; geometry/`out_size` are
/// updated on a stream-change renegotiation.
#[derive(Debug)]
struct DecoderState {
    transform: IMFTransform,
    width: u32,
    height: u32,
    out_size: u32,
    /// Row pitch in bytes of the MFT's NV12 output (`MF_MT_DEFAULT_STRIDE`),
    /// `>= width`. The MS software decoder packs tightly (`stride == width`),
    /// but a DXVA / hardware MFT aligns rows up, so the contiguous output
    /// buffer carries per-row padding that must be stripped before the packed
    /// NV12 the downstream sinks expect.
    stride: u32,
    /// True when the MFT allocates its own output samples
    /// (`MFT_OUTPUT_STREAM_PROVIDES_SAMPLES`): `ProcessOutput` is then called
    /// with a null sample and the MFT fills it. Otherwise a system-memory
    /// output sample is pre-allocated.
    provides_samples: bool,
}

/// One decoded picture, packed system NV12, plus its geometry and timestamp.
#[derive(Debug)]
struct DecodedPicture {
    nv12: Box<[u8]>,
    width: u32,
    height: u32,
    pts_ns: u64,
}

/// Result of a single `ProcessOutput` attempt.
#[derive(Debug)]
enum OutputStep {
    Frame(DecodedPicture),
    NeedInput,
    StreamChange,
}

/// # Example
///
/// ```ignore
/// use g2g_core::VideoCodec;
/// use g2g_plugins::mfdecode::MfDecode;
///
/// let element = MfDecode::new().with_codec(VideoCodec::H265);
/// ```
#[derive(Debug)]
pub struct MfDecode {
    /// Bitstream codec the MFT decodes (`H264` default, or `H265`). Selects the
    /// decoder CLSID and the input media subtype.
    codec: VideoCodec,
    state: Option<DecoderState>,
    com_started: bool,
    configured: bool,
    last_caps: Option<Caps>,
    /// Most recent input caps received via
    /// `PipelinePacket::CapsChanged`. Used to validate the format on
    /// mid-stream changes and to debug-assert agreement between the
    /// declared `DerivedOutput` closure and the decode-time output
    /// geometry. See `ffmpegdec.rs` for the same field with full notes.
    input_caps: Option<Caps>,
    input_stamps: PendingInputStamps,
    emitted: u64,
    segment: Segment,
}

// SAFETY: `IMFTransform` is a COM interface and thus `!Send` by default. The
// framework's `multi-thread` runner requires `Send` elements so it can hand a
// task between worker threads. We uphold that for `MfDecode` by construction
// and contract: COM is initialised multi-threaded (MTA, free-threaded), the MS
// H.264 decoder MFT is callable from any MTA thread without marshaling, and the
// runner drives a single element through `&mut self` (never concurrently). The
// element is moved between threads but never shared, so there is no data race.
// Callers must keep driving threads in the MTA (the default for tokio's
// multi-thread workers once any thread calls `MFStartup`/`CoInitializeEx`).
unsafe impl Send for MfDecode {}

impl Default for MfDecode {
    fn default() -> Self {
        Self::new()
    }
}

impl MfDecode {
    pub fn new() -> Self {
        Self {
            codec: VideoCodec::H264,
            state: None,
            com_started: false,
            configured: false,
            last_caps: None,
            input_caps: None,
            input_stamps: PendingInputStamps::default(),
            emitted: 0,
            segment: Segment::new(),
        }
    }

    /// Select the bitstream codec to decode. Only `H264` (the default) and
    /// `H265` map to a Media Foundation decoder MFT; any other codec is
    /// rejected loud at `configure_pipeline`. Call before `configure_pipeline`.
    pub fn with_codec(mut self, codec: VideoCodec) -> Self {
        self.codec = codec;
        self
    }

    /// The configured bitstream codec.
    pub fn codec(&self) -> VideoCodec {
        self.codec
    }

    /// Count of decoded `DataFrame`s pushed downstream. Useful in tests.
    pub fn decoded_count(&self) -> u64 {
        self.emitted
    }

    /// Feed one access unit, then drain whatever the decoder is ready to
    /// release into `decoded`. Decoders buffer for B-frame reordering, so
    /// early inputs commonly yield zero outputs.
    fn feed(
        &mut self,
        data: &[u8],
        pts_ns: u64,
        stamp: InputStamp,
        decoded: &mut Vec<DecodedPicture>,
    ) -> Result<(), G2gError> {
        let sample = make_input_sample(data, pts_ns)?;
        // the decoded sample carries the pts back rounded down to 100 ns ticks
        self.input_stamps
            .insert(mf_ticks_to_ns(ns_to_mf_ticks(pts_ns)), stamp);
        feed_sample(self, &sample, Self::transform, |element| {
            element.drain(decoded)
        })
    }

    fn transform(&self) -> Result<&IMFTransform, G2gError> {
        Ok(&self
            .state
            .as_ref()
            .ok_or(G2gError::NotConfigured)?
            .transform)
    }

    /// Pull outputs until the MFT needs more input, renegotiating the output
    /// type on a stream change.
    fn drain(&mut self, decoded: &mut Vec<DecodedPicture>) -> Result<(), G2gError> {
        loop {
            match self.process_output()? {
                OutputStep::Frame(f) => decoded.push(f),
                OutputStep::NeedInput => return Ok(()),
                OutputStep::StreamChange => self.renegotiate()?,
            }
        }
    }

    /// Drain on end-of-stream: send the MFT a DRAIN command so it flushes
    /// reordered pictures, then collect them.
    fn drain_eos(&mut self, decoded: &mut Vec<DecodedPicture>) -> Result<(), G2gError> {
        send_drain_command(self.transform()?)?;
        self.drain(decoded)
    }

    fn process_output(&self) -> Result<OutputStep, G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        let sample = match process_output(&st.transform, st.provides_samples, st.out_size)? {
            OutputPull::Sample(sample) => sample,
            OutputPull::NeedInput => return Ok(OutputStep::NeedInput),
            OutputPull::StreamChange => return Ok(OutputStep::StreamChange),
        };
        let pts_ns = sample_time_ns(&sample);
        let nv12 = read_sample_bytes(&sample, |src| {
            pack_nv12(
                src,
                st.width as usize,
                st.height as usize,
                st.stride as usize,
            )
        })??;
        Ok(OutputStep::Frame(DecodedPicture {
            nv12,
            width: st.width,
            height: st.height,
            pts_ns,
        }))
    }

    /// Re-pick the NV12 output type after a stream change and refresh the
    /// cached geometry / output buffer size from it.
    fn renegotiate(&mut self) -> Result<(), G2gError> {
        let st = self.state.as_mut().ok_or(G2gError::NotConfigured)?;
        let (w, h, stride) = set_nv12_output(&st.transform)?;
        st.out_size = output_buffer_size(&st.transform, w, h)?;
        st.provides_samples = output_provides_samples(&st.transform)?;
        st.width = w;
        st.height = h;
        st.stride = stride;
        Ok(())
    }
}

impl AsyncElement for MfDecode {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Consumes the configured codec at any geometry; intersecting narrows
        // the proposal and rejects a mismatched codec.
        let supported = Caps::CompressedVideo {
            codec: self.codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        upstream_caps.intersect(&supported)
    }

    /// Native `DerivedOutput`: accepts H.264 at any
    /// geometry and produces NV12 at the same dims/framerate. The closure
    /// validates the input format and returns an empty set on mismatch, so
    /// the solver rejects non-H.264 upstream at negotiation time instead of
    /// via the dynamic `intercept_caps` callback. Mixed chains get real
    /// per-link caps from the solver: H.264 to the decoder, NV12 to the
    /// sink. Mirrors `FfmpegH264Dec` (step 5k); the MFT only ever emits
    /// NV12, so there is no output-format choice.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let codec = self.codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            derive_output_caps(codec, input)
        }))
    }

    fn metadata(&self) -> g2g_core::ElementMetadata {
        g2g_core::ElementMetadata::new(
            "Media Foundation H.264 / H.265 decoder",
            "Codec/Decoder/Video",
            "Decodes H.264 or H.265 to system-memory NV12 with a Media Foundation decoder MFT",
            crate::AARON_BOXER,
        )
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = match absolute_caps {
            Caps::CompressedVideo {
                codec,
                width,
                height,
                ..
            } if *codec == self.codec => (width.fixed().unwrap_or(0), height.fixed().unwrap_or(0)),
            _ => return Err(G2gError::CapsMismatch),
        };

        // SAFETY: COM/MF startup on the calling thread. MfDecode is !Send, so
        // every later COM call lands on this same thread (MTA apartment
        // affinity). CoInitializeEx returning S_FALSE (already initialised) is
        // not an error for our use.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            MFStartup(MF_VERSION, MFSTARTUP_FULL).map_err(mf_err)?;
        }
        self.com_started = true;

        self.state = Some(init_decoder(self.codec, w, h)?);
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
            let mut decoded = Vec::new();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.feed(
                        slice,
                        frame.timing.pts_ns,
                        InputStamp::of(&frame),
                        &mut decoded,
                    )?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // M16 workaround #3 Phase A: validate + record.
                    // Reject an incompatible mid-stream format change
                    // (e.g. H.264 -> VP9, or a codec swap) loud; previously
                    // dropped silently. Output `CapsChanged` is still emitted
                    // from decoded geometry at the decode boundary so
                    // the ordering invariant from §3 is preserved. The runner's
                    // pre-fixed output caps (our NV12) are forwarded so the sink
                    // sees them before the first decoded frame (M733/M734, see
                    // `ffmpegdec.rs` "Two callers").
                    match &c {
                        Caps::CompressedVideo { codec, .. } if *codec == self.codec => {
                            self.input_caps = Some(c);
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
                        // SAFETY: flush message on the owning thread.
                        unsafe {
                            st.transform
                                .ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0)
                                .map_err(mf_err)?;
                        }
                    }
                    self.input_stamps = PendingInputStamps::default();
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    self.drain_eos(&mut decoded)?;
                }
                PipelinePacket::Segment(seg) => {
                    self.segment = seg;
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }

            // `Any` when the stream declared no rate, never a guess.
            let (out_framerate, out_pixel_aspect) = match &self.input_caps {
                Some(Caps::CompressedVideo {
                    framerate,
                    pixel_aspect,
                    ..
                }) => (framerate.clone(), *pixel_aspect),
                _ => (Rate::Any, g2g_core::PixelAspect::Any),
            };
            for d in decoded {
                let input = self.input_stamps.take(d.pts_ns);
                if !self.segment.presents(d.pts_ns) {
                    g2g_core::g2g_log!(
                        g2g_core::log::Target::category(g2g_core::AsyncElement::log_category(self)),
                        "dropping frame pts {} outside the segment",
                        g2g_core::log::ClockTime(d.pts_ns)
                    );
                    continue;
                }
                let new_caps =
                    nv12_caps(d.width, d.height, out_framerate.clone(), out_pixel_aspect);
                if self.last_caps.as_ref() != Some(&new_caps) {
                    // M16 workaround #3 Phase A debug assertion:
                    // decode-time output must overlap the closure's
                    // derivation of the recorded input. See
                    // `ffmpegdec.rs` for the full rationale.
                    #[cfg(debug_assertions)]
                    if let Some(input) = self.input_caps.as_ref() {
                        let expected = derive_output_caps(self.codec, input);
                        debug_assert!(
                            !expected
                                .intersect(&CapsSet::one(new_caps.clone()))
                                .is_empty(),
                            "mfdecode decode-time output {new_caps:?} inconsistent with derive_output_caps({input:?}) = {expected:?}"
                        );
                    }
                    out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                        .await?;
                    self.last_caps = Some(new_caps.clone());
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(d.nv12)),
                    timing: FrameTiming {
                        pts_ns: d.pts_ns,
                        dts_ns: d.pts_ns,
                        duration_ns: 0,
                        capture_ns: d.pts_ns,
                        arrival_ns: input.arrival_ns,
                        ..FrameTiming::default()
                    },
                    sequence: self.emitted,
                    meta: input.meta,
                };
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
        })
    }
}

impl PadTemplates for MfDecode {
    /// Consumes H.264 or H.265 and produces NV12, both at any geometry (the
    /// MFT derives the output dims from the stream). The static superset
    /// advertises both codecs on the sink; the configured instance narrows to
    /// one via `intercept_caps` / `caps_constraint_as_transform`.
    fn pad_templates() -> Vec<PadTemplate> {
        let compressed = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
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

impl Drop for MfDecode {
    fn drop(&mut self) {
        // Release the MFT before tearing COM/MF down.
        self.state = None;
        if self.com_started {
            // SAFETY: paired with the CoInitializeEx/MFStartup in
            // configure_pipeline, on the same (owning) thread.
            unsafe {
                let _ = MFShutdown();
                CoUninitialize();
            }
        }
    }
}

/// Decoder MFT CLSID and input media subtype for a codec. Only H.264 and H.265
/// map to a Media Foundation decoder; any other codec is rejected loud.
fn decoder_ids(codec: VideoCodec) -> Result<(windows::core::GUID, windows::core::GUID), G2gError> {
    match codec {
        VideoCodec::H264 => Ok((CLSID_MSH264DecoderMFT, MFVideoFormat_H264)),
        VideoCodec::H265 => Ok((CLSID_MSH265DecoderMFT, MFVideoFormat_HEVC)),
        _ => Err(G2gError::CapsMismatch),
    }
}

/// Create the decoder MFT, set the bitstream input type and an NV12 output
/// type, and put it into streaming mode.
fn init_decoder(codec: VideoCodec, width: u32, height: u32) -> Result<DecoderState, G2gError> {
    let (clsid, subtype) = decoder_ids(codec)?;
    // SAFETY: COM object creation on the owning thread.
    let transform: IMFTransform =
        unsafe { CoCreateInstance(&clsid, None, CLSCTX_INPROC_SERVER) }.map_err(mf_err)?;

    // SAFETY: media-type configuration on the owning thread; all arguments are
    // valid for the duration of each call.
    unsafe {
        let input = MFCreateMediaType().map_err(mf_err)?;
        input
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .map_err(mf_err)?;
        input.SetGUID(&MF_MT_SUBTYPE, &subtype).map_err(mf_err)?;
        if width != 0 && height != 0 {
            input
                .SetUINT64(&MF_MT_FRAME_SIZE, pack_u32_pair(width, height))
                .map_err(mf_err)?;
        }
        transform.SetInputType(0, &input, 0).map_err(mf_err)?;
    }

    let (w, h, stride) = set_nv12_output(&transform)?;
    let out_size = output_buffer_size(&transform, w, h)?;
    let provides_samples = output_provides_samples(&transform)?;

    // SAFETY: streaming-mode messages on the owning thread.
    unsafe {
        transform
            .ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)
            .map_err(mf_err)?;
        transform
            .ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)
            .map_err(mf_err)?;
    }

    Ok(DecoderState {
        transform,
        width: w,
        height: h,
        out_size,
        stride,
        provides_samples,
    })
}

/// Select the first NV12 output type the MFT offers and return its geometry
/// and row stride. The stride (`MF_MT_DEFAULT_STRIDE`) is the source row pitch
/// for the de-pad in [`pack_nv12`]; it is floored at `width` so a missing or
/// bottom-up attribute degrades to the tightly-packed assumption.
fn set_nv12_output(transform: &IMFTransform) -> Result<(u32, u32, u32), G2gError> {
    let mut i = 0u32;
    loop {
        // SAFETY: type enumeration on the owning thread.
        let candidate = unsafe { transform.GetOutputAvailableType(0, i) };
        match candidate {
            Ok(t) => {
                // SAFETY: reading attributes off a valid media type.
                let subtype = unsafe { t.GetGUID(&MF_MT_SUBTYPE) }.map_err(mf_err)?;
                if subtype == MFVideoFormat_NV12 {
                    // SAFETY: applying the chosen output type on the owning thread.
                    unsafe { transform.SetOutputType(0, &t, 0) }.map_err(mf_err)?;
                    // SAFETY: reading the frame-size attribute off the type.
                    let size = unsafe { t.GetUINT64(&MF_MT_FRAME_SIZE) }.unwrap_or(0);
                    let (w, h) = unpack_u32_pair(size);
                    // SAFETY: reading the (optional) default-stride attribute.
                    // Absent on the packed software path; present (>= width)
                    // when a hardware MFT aligns rows up.
                    let stride = unsafe { t.GetUINT32(&MF_MT_DEFAULT_STRIDE) }.unwrap_or(0);
                    return Ok((w, h, stride.max(w)));
                }
                i += 1;
            }
            // No more types (typically MF_E_NO_MORE_TYPES): no NV12 path.
            Err(e) => return Err(mf_err(e)),
        }
    }
}

/// Output buffer bytes to allocate per `ProcessOutput`: the MFT's reported
/// size, floored at a tightly-packed NV12 frame so a zero/early estimate
/// never under-allocates.
fn output_buffer_size(transform: &IMFTransform, w: u32, h: u32) -> Result<u32, G2gError> {
    // SAFETY: stream-info query on the owning thread.
    let info = unsafe { transform.GetOutputStreamInfo(0) }.map_err(mf_err)?;
    let nv12 = RawVideoFormat::Nv12
        .unpadded_frame_bytes(w, h)
        .map_or(u32::MAX, |bytes| u32::try_from(bytes).unwrap_or(u32::MAX));
    Ok(info.cbSize.max(nv12))
}

/// De-pad a strided NV12 source buffer into a tightly-packed
/// `width * height * 3 / 2` buffer: the Y plane is `height` rows and the
/// interleaved UV plane `height / 2` rows, each `width` bytes wide read from a
/// `stride`-byte source pitch. When `stride == width` this is a single
/// contiguous copy. Rows beyond what the source actually holds are skipped
/// (left zero) rather than panicking, so a short buffer fails safe.
fn pack_nv12(
    src: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Box<[u8]>, G2gError> {
    if width == 0 || height == 0 || stride < width {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let rows = height + height / 2; // Y plane + half-height interleaved UV
    let mut dst = alloc::vec![0u8; width * rows].into_boxed_slice();
    for row in 0..rows {
        let src_start = row * stride;
        let dst_start = row * width;
        let Some(src_row) = src.get(src_start..src_start + width) else {
            break; // source shorter than advertised; leave the rest zeroed
        };
        dst[dst_start..dst_start + width].copy_from_slice(src_row);
    }
    Ok(dst)
}

fn nv12_caps(w: u32, h: u32, framerate: Rate, pixel_aspect: g2g_core::PixelAspect) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate,
        interlace: Interlace::Any,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        pixel_aspect,
    }
}

/// Single source of truth for the decoder's output-side caps derivation.
/// Shared by the `DerivedOutput` constraint closure and the
/// workaround-#3 Phase A debug assertion. The MFT only emits NV12, so
/// there's no output-format choice (unlike `ffmpegdec`'s helper); an input
/// whose codec differs from the configured one yields an empty set.
fn derive_output_caps(codec: VideoCodec, input: &Caps) -> CapsSet {
    match input {
        Caps::CompressedVideo {
            codec: c,
            width,
            height,
            framerate,
            pixel_aspect,
            ..
        } if *c == codec => CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: width.clone(),
            height: height.clone(),
            framerate: framerate.clone(),
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: *pixel_aspect,
        }),
        _ => CapsSet::from_alternatives(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn nv12_caps_are_fixed() {
        assert_eq!(
            nv12_caps(640, 480, Rate::Fixed(30 << 16), g2g_core::PixelAspect::Any),
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
                interlace: Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                pixel_aspect: g2g_core::PixelAspect::Any,
            }
        );
    }

    #[test]
    fn intercept_rejects_non_h264() {
        let dec = MfDecode::new();
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(dec.intercept_caps(&vp9), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn intercept_narrows_h264_geometry() {
        let dec = MfDecode::new();
        let proposal = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(dec.intercept_caps(&proposal), Ok(proposal));
    }

    #[test]
    fn codec_defaults_h264_and_selects_h265() {
        assert_eq!(MfDecode::new().codec(), VideoCodec::H264);
        assert_eq!(
            MfDecode::new().with_codec(VideoCodec::H265).codec(),
            VideoCodec::H265
        );
    }

    #[test]
    fn decoder_ids_map_supported_codecs() {
        assert_eq!(
            decoder_ids(VideoCodec::H264).unwrap(),
            (CLSID_MSH264DecoderMFT, MFVideoFormat_H264)
        );
        assert_eq!(
            decoder_ids(VideoCodec::H265).unwrap(),
            (CLSID_MSH265DecoderMFT, MFVideoFormat_HEVC)
        );
        assert_eq!(decoder_ids(VideoCodec::Av1), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn hevc_instance_accepts_h265_rejects_h264() {
        let dec = MfDecode::new().with_codec(VideoCodec::H265);
        let h265 = Caps::CompressedVideo {
            codec: VideoCodec::H265,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(dec.intercept_caps(&h265), Ok(h265.clone()));
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(dec.intercept_caps(&h264), Err(G2gError::CapsMismatch));
        // The DerivedOutput closure derives NV12 from H.265 and rejects H.264.
        let CapsConstraint::DerivedOutput(f) = dec.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        assert!(!f(&h265).is_empty());
        assert!(f(&h264).is_empty());
    }

    #[test]
    fn pack_nv12_packed_source_is_identity() {
        // stride == width: a 4x2 NV12 frame (Y=8 bytes, UV=4 bytes) copies
        // through unchanged.
        let src: Vec<u8> = (0..12).collect();
        let out = pack_nv12(&src, 4, 2, 4).unwrap();
        assert_eq!(&out[..], &src[..]);
        assert_eq!(out.len(), 4 * 2 * 3 / 2);
    }

    #[test]
    fn pack_nv12_strips_row_stride_padding() {
        // 4x2 NV12 with stride 6: each of the 3 rows (2 Y + 1 UV) holds 4 data
        // bytes then 2 pad bytes. The packed output drops the padding.
        let width = 4;
        let height = 2;
        let stride = 6;
        let rows = height + height / 2; // 3
        let mut src = Vec::new();
        for row in 0..rows {
            for col in 0..width {
                src.push((row * 10 + col) as u8); // data
            }
            src.push(0xFF); // pad
            src.push(0xFF); // pad
        }
        let out = pack_nv12(&src, width, height, stride).unwrap();
        let expected: Vec<u8> = vec![
            0, 1, 2, 3, // Y row 0
            10, 11, 12, 13, // Y row 1
            20, 21, 22, 23, // UV row
        ];
        assert_eq!(&out[..], &expected[..]);
        assert!(!out.contains(&0xFF), "padding bytes must be stripped");
    }

    #[test]
    fn pack_nv12_rejects_bad_geometry() {
        // stride < width is impossible NV12; zero dims have no pixels.
        assert!(pack_nv12(&[0; 16], 8, 2, 4).is_err());
        assert!(pack_nv12(&[0; 16], 0, 2, 4).is_err());
    }

    #[test]
    fn pack_nv12_short_source_fails_safe() {
        // A source shorter than advertised leaves the missing tail zeroed
        // rather than panicking. 4x2 needs 3 rows; supply only the first.
        let src: Vec<u8> = vec![1, 2, 3, 4];
        let out = pack_nv12(&src, 4, 2, 4).unwrap();
        assert_eq!(&out[0..4], &[1, 2, 3, 4]);
        assert_eq!(&out[4..], &[0; 8]); // remaining rows zeroed
    }

    #[test]
    fn pad_templates_are_h264_in_nv12_out() {
        use g2g_core::{PadDirection, PadTemplates};
        let sink = MfDecode::pad_template(PadDirection::Sink).expect("has sink pad");
        let source = MfDecode::pad_template(PadDirection::Source).expect("has source pad");
        // H.264 in, NV12 out.
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
        };
        assert!(matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&h264)));
        assert!(matches!(source.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&nv12)));
        assert!(!matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&nv12)));
    }

    #[test]
    fn caps_constraint_is_derived_output_h264_to_nv12() {
        // M16 step 5l: DerivedOutput closure validates H.264 input and
        // emits NV12 at the same dims/rate; non-H.264 yields an empty set.
        let dec = MfDecode::new();
        let CapsConstraint::DerivedOutput(f) = dec.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(
            f(&h264).alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(30 << 16),
                interlace: Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                pixel_aspect: g2g_core::PixelAspect::Any,
            }]
        );

        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert!(f(&vp9).is_empty());
    }
}
