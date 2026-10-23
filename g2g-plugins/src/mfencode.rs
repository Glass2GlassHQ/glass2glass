//! Windows H.264 / H.265 encode element wrapping a Media Foundation encoder
//! MFT. The encode-side mirror of `mfdecode`. Codec is selected with
//! [`MfEncode::with_codec`]; the default is H.264.
//!
//! H.264 uses the MS H.264 Encoder MFT at a fixed CLSID
//! (`CLSID_MSH264EncoderMFT`). H.265/HEVC has no fixed CLSID, so M30
//! enumerates an encoder via `MFTEnumEx` for the HEVC output subtype. Both a
//! synchronous MFT (the `ProcessInput`/`ProcessOutput` loop below) and an
//! asynchronous, event-driven MFT (the common shape of a hardware HEVC encoder,
//! driven by its `IMFMediaEventGenerator`) are supported; the path is chosen
//! from the MFT's `MF_TRANSFORM_ASYNC` attribute at `configure_pipeline`.
//!
//! M19: consumes raw NV12 `DataFrame`s (`MemoryDomain::System`, tightly
//! packed) and produces Annex-B H.264 access units, also
//! `MemoryDomain::System` (one encoded sample per input picture; the MFT
//! emits NALUs with start codes, SPS/PPS attached to each IDR). A
//! `CapsChanged(H264, w, h)` is emitted before the first encoded frame.
//!
//! Latency: `MF_LOW_LATENCY` is set on the MFT's attribute store (the
//! attribute alias of `CODECAPI_AVLowLatencyMode`), so the encoder runs
//! without B-frames or lookahead and releases one output per input. Set
//! best-effort: an MFT that does not recognise it just encodes with its
//! defaults.
//!
//! Threading: COM is initialised multi-threaded (MTA) in `configure_pipeline`
//! and every `IMFTransform` call runs on that same thread; `Send` is asserted
//! under the same documented contract as `MfDecode`.
//!
//! Geometry: unlike the decoder (which derives dims from the bitstream), the
//! encoder's media types need concrete dims up front, so `configure_pipeline`
//! requires fixed, even NV12 dims. A mid-stream `CapsChanged` to different
//! dims drains the current MFT and rebuilds it at the new geometry.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use windows::core::Interface;
use windows::Win32::Media::MediaFoundation::{
    eAVEncH264VProfile_Main, CLSID_MSH264EncoderMFT, IMFActivate, IMFMediaEventGenerator,
    IMFSample, IMFTransform, METransformDrainComplete, METransformHaveOutput, METransformNeedInput,
    MFCreateMediaType, MFMediaType_Video, MFShutdown, MFStartup, MFTEnumEx, MFVideoFormat_H264,
    MFVideoFormat_HEVC, MFVideoFormat_NV12, MFVideoInterlace_Progressive,
    MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS, MFSTARTUP_FULL, MFT_CATEGORY_VIDEO_ENCODER,
    MFT_ENUM_FLAG_ASYNCMFT, MFT_ENUM_FLAG_HARDWARE, MFT_ENUM_FLAG_SORTANDFILTER,
    MFT_ENUM_FLAG_SYNCMFT, MFT_MESSAGE_COMMAND_DRAIN, MFT_MESSAGE_COMMAND_FLUSH,
    MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, MFT_MESSAGE_NOTIFY_END_OF_STREAM,
    MFT_MESSAGE_NOTIFY_START_OF_STREAM, MFT_REGISTER_TYPE_INFO, MF_EVENT_FLAG_NO_WAIT,
    MF_E_NO_EVENTS_AVAILABLE, MF_LOW_LATENCY, MF_MT_AVG_BITRATE, MF_MT_FRAME_RATE,
    MF_MT_FRAME_SIZE, MF_MT_INTERLACE_MODE, MF_MT_MAJOR_TYPE, MF_MT_MPEG2_PROFILE, MF_MT_SUBTYPE,
    MF_TRANSFORM_ASYNC, MF_TRANSFORM_ASYNC_UNLOCK, MF_VERSION,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoTaskMemFree, CoUninitialize, CLSCTX_INPROC_SERVER,
    COINIT_MULTITHREADED,
};

use alloc::collections::VecDeque;

use crate::mfcommon::{
    feed_sample, make_input_sample, mf_err, mf_ticks_to_ns, ns_to_mf_ticks,
    output_provides_samples, pack_u32_pair, process_output, read_sample_bytes, sample_time_ns,
    send_drain_command, OutputPull,
};
use g2g_core::caps::framerate_q16_to_fraction;
use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    FrameTiming, G2gError, HardwareError, Interlace, MemoryDomain, OutputSink, PadTemplate,
    PadTemplates, PipelinePacket, Rate, RawVideoFormat, VideoCodec,
};

/// Default target bitrate (bits/s) when the caller doesn't override it.
const DEFAULT_BITRATE: u32 = 4_000_000;
// the encoder types need an MF_MT_FRAME_RATE even when the caps declare none
const ENCODER_FRAMERATE_WHEN_UNDECLARED: (u32, u32) = (30, 1);

/// Live MFT plus the configured input geometry. Rebuilt (not renegotiated in
/// place) on a mid-stream geometry change: a fresh MFT at the new dims is
/// deterministic, while re-setting media types on a draining encoder is not.
#[derive(Debug)]
struct EncoderState {
    transform: IMFTransform,
    /// Negotiated output media subtype (`MFVideoFormat_H264`/`_HEVC`), reused
    /// to re-pick the output type on a stream-change renegotiation.
    out_subtype: windows::core::GUID,
    width: u32,
    height: u32,
    out_size: u32,
    /// Per-frame duration in 100-ns units, derived from the framerate the MFT
    /// was configured with. Stamped on an input sample that has no duration of
    /// its own.
    duration_hns: i64,
    /// True when the MFT allocates its own output samples; the MS software
    /// encoder does not, but a substituted hardware encoder MFT might.
    provides_samples: bool,
    /// True for an asynchronous (event-driven) MFT, the common shape of a
    /// hardware encoder. When set, input/output is driven by the event
    /// generator below instead of the sync `ProcessInput`/`ProcessOutput` poll.
    async_mode: bool,
    /// Event generator for the async path (`Some` iff `async_mode`).
    event_gen: Option<IMFMediaEventGenerator>,
    /// Input samples handed in but not yet accepted by the async MFT, fed on
    /// the next `METransformNeedInput`.
    pending_input: VecDeque<IMFSample>,
    /// Outstanding `METransformNeedInput` events with no input queued to
    /// satisfy them: the next handed-in sample is fed immediately.
    need_input_credits: u32,
}

/// One encoded access unit plus the caps fields it was produced under, so
/// frames drained across a mid-stream geometry change still emit under the
/// caps they were encoded with.
#[derive(Debug)]
struct EncodedChunk {
    data: Box<[u8]>,
    width: u32,
    height: u32,
    framerate: Rate,
    pts_ns: u64,
    duration_ns: u64,
}

/// Result of a single `ProcessOutput` attempt.
#[derive(Debug)]
enum OutputStep {
    Frame(EncodedChunk),
    NeedInput,
    StreamChange,
}

/// Outcome of handling one async MFT event.
#[derive(Debug, PartialEq, Eq)]
enum PumpResult {
    /// An event was handled (input fed, output pulled, or ignored).
    Pumped,
    /// Non-blocking pump found the event queue empty.
    NoEvents,
    /// The MFT signalled `METransformDrainComplete`.
    DrainComplete,
}

/// # Example
///
/// ```ignore
/// use g2g_core::VideoCodec;
/// use g2g_plugins::mfencode::MfEncode;
///
/// let encoder = MfEncode::new()
///     .with_codec(VideoCodec::H265)
///     .with_bitrate(4_000_000)
///     .with_hardware();
/// ```
#[derive(Debug)]
pub struct MfEncode {
    /// Output codec the MFT produces (`H264` default, or `H265`). Selects the
    /// encoder MFT and the output media subtype.
    codec: VideoCodec,
    state: Option<EncoderState>,
    com_started: bool,
    configured: bool,
    bitrate: u32,
    /// Prefer a hardware encoder MFT (enumerated via `MFTEnumEx`) even for
    /// H.264, which otherwise uses the fixed-CLSID MS software encoder. A
    /// hardware encoder is commonly an asynchronous MFT, driven by the
    /// event-based path. H.265 always enumerates regardless.
    prefer_hardware: bool,
    /// Negotiated input framerate, echoed on the H.264 output caps and used
    /// to derive the per-frame sample duration.
    framerate: Rate,
    last_caps: Option<Caps>,
    /// Most recent input caps received via `PipelinePacket::CapsChanged`;
    /// validated on mid-stream changes and debug-asserted against the
    /// declared `DerivedOutput` closure, as in `mfdecode`.
    input_caps: Option<Caps>,
    emitted: u64,
    /// M12: the downstream consumer's allocation proposal, recorded in
    /// `configure_allocation`. The encoder emits system-memory bitstream
    /// buffers, so the request is informational.
    requested_alloc: Option<AllocationParams>,
}

// SAFETY: `IMFTransform` is a COM interface and thus `!Send` by default. Same
// contract as `MfDecode`: COM is initialised multi-threaded (MTA), the MS
// H.264 encoder MFT is callable from any MTA thread without marshaling, and
// the runner drives the element through `&mut self` (moved between threads,
// never shared), so there is no data race.
unsafe impl Send for MfEncode {}

impl Default for MfEncode {
    fn default() -> Self {
        Self::new()
    }
}

impl MfEncode {
    pub fn new() -> Self {
        Self {
            codec: VideoCodec::H264,
            state: None,
            com_started: false,
            configured: false,
            bitrate: DEFAULT_BITRATE,
            prefer_hardware: false,
            framerate: Rate::Any,
            last_caps: None,
            input_caps: None,
            emitted: 0,
            requested_alloc: None,
        }
    }

    /// Select the output codec. Only `H264` (the default) and `H265` map to a
    /// Media Foundation encoder MFT; any other codec is rejected loud at
    /// `configure_pipeline`. Call before `configure_pipeline`.
    pub fn with_codec(mut self, codec: VideoCodec) -> Self {
        self.codec = codec;
        self
    }

    /// The configured output codec.
    pub fn codec(&self) -> VideoCodec {
        self.codec
    }

    /// Prefer a hardware encoder MFT (enumerated via `MFTEnumEx`) even for
    /// H.264. Hardware encoders are commonly asynchronous MFTs, driven by the
    /// event-based path; the default H.264 route uses the fixed-CLSID MS
    /// software encoder. Call before `configure_pipeline`.
    pub fn with_hardware(mut self) -> Self {
        self.prefer_hardware = true;
        self
    }

    /// Whether the live encoder MFT is asynchronous (event-driven). `None`
    /// before `configure_pipeline`. Useful in tests to confirm the async path
    /// is exercised.
    pub fn is_async(&self) -> Option<bool> {
        self.state.as_ref().map(|s| s.async_mode)
    }

    /// Target average bitrate in bits/s (`MF_MT_AVG_BITRATE`). Applies at the
    /// next encoder (re)build; call before `configure_pipeline`.
    pub fn with_bitrate(mut self, bits_per_sec: u32) -> Self {
        self.bitrate = bits_per_sec;
        self
    }

    /// The configured target bitrate in bits/s.
    pub fn bitrate(&self) -> u32 {
        self.bitrate
    }

    /// Count of encoded `DataFrame`s pushed downstream. Useful in tests.
    pub fn encoded_count(&self) -> u64 {
        self.emitted
    }

    /// The downstream consumer's recorded M12 allocation proposal, if any.
    pub fn requested_alloc(&self) -> Option<AllocationParams> {
        self.requested_alloc
    }

    /// Feed one raw picture, then drain whatever the encoder is ready to
    /// release into `encoded`.
    fn feed(
        &mut self,
        data: &[u8],
        timing: &FrameTiming,
        encoded: &mut Vec<EncodedChunk>,
    ) -> Result<(), G2gError> {
        let (rate_duration_hns, async_mode) = {
            let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
            (st.duration_hns, st.async_mode)
        };
        let sample = make_input_sample(data, timing.pts_ns)?;
        let duration_hns = sample_duration_hns(timing.duration_ns, rate_duration_hns);
        // SAFETY: setting an attribute on the sample just built, on the owning thread.
        unsafe { sample.SetSampleDuration(duration_hns) }.map_err(mf_err)?;
        if async_mode {
            return self.feed_async(sample, encoded);
        }
        feed_sample(self, &sample, Self::transform, |element| {
            element.drain(encoded)
        })
    }

    fn transform(&self) -> Result<&IMFTransform, G2gError> {
        Ok(&self
            .state
            .as_ref()
            .ok_or(G2gError::NotConfigured)?
            .transform)
    }

    /// Pull outputs until the MFT needs more input.
    fn drain(&mut self, encoded: &mut Vec<EncodedChunk>) -> Result<(), G2gError> {
        loop {
            match self.process_output()? {
                OutputStep::Frame(f) => encoded.push(f),
                OutputStep::NeedInput => return Ok(()),
                OutputStep::StreamChange => self.renegotiate()?,
            }
        }
    }

    /// Drain on end-of-stream (or before a geometry rebuild): send the MFT a
    /// DRAIN command so it flushes buffered pictures, then collect them.
    fn drain_eos(&mut self, encoded: &mut Vec<EncodedChunk>) -> Result<(), G2gError> {
        if self
            .state
            .as_ref()
            .ok_or(G2gError::NotConfigured)?
            .async_mode
        {
            return self.drain_eos_async(encoded);
        }
        send_drain_command(self.transform()?)?;
        self.drain(encoded)
    }

    /// Async path: queue an input sample, feeding it now if the MFT has already
    /// asked for input, then drain whatever events are immediately ready.
    fn feed_async(
        &mut self,
        sample: IMFSample,
        encoded: &mut Vec<EncodedChunk>,
    ) -> Result<(), G2gError> {
        {
            let st = self.state.as_mut().ok_or(G2gError::NotConfigured)?;
            if st.need_input_credits > 0 {
                st.need_input_credits -= 1;
                // SAFETY: ProcessInput on the owning thread; `sample` is valid.
                unsafe { st.transform.ProcessInput(0, &sample, 0) }.map_err(mf_err)?;
            } else {
                st.pending_input.push_back(sample);
            }
        }
        while matches!(self.pump_one(encoded, false)?, PumpResult::Pumped) {}
        Ok(())
    }

    /// Async path drain: feed any still-queued input (blocking for the MFT's
    /// input requests), then DRAIN and pump until `METransformDrainComplete`.
    fn drain_eos_async(&mut self, encoded: &mut Vec<EncodedChunk>) -> Result<(), G2gError> {
        // Push out queued input first; the MFT signals NeedInput as it accepts.
        while !self
            .state
            .as_ref()
            .ok_or(G2gError::NotConfigured)?
            .pending_input
            .is_empty()
        {
            if matches!(self.pump_one(encoded, true)?, PumpResult::DrainComplete) {
                return Ok(());
            }
        }
        {
            let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
            // SAFETY: end-of-stream + drain messages on the owning thread.
            unsafe {
                st.transform
                    .ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0)
                    .map_err(mf_err)?;
                st.transform
                    .ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0)
                    .map_err(mf_err)?;
            }
        }
        loop {
            if matches!(self.pump_one(encoded, true)?, PumpResult::DrainComplete) {
                return Ok(());
            }
        }
    }

    /// Handle one async event. `block` waits for the next event; otherwise a
    /// drained queue returns `NoEvents`. `NeedInput` feeds a queued sample (or
    /// banks a credit), `HaveOutput` pulls an encoded frame.
    fn pump_one(
        &mut self,
        encoded: &mut Vec<EncodedChunk>,
        block: bool,
    ) -> Result<PumpResult, G2gError> {
        let event = {
            let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
            let gen = st.event_gen.as_ref().ok_or(G2gError::NotConfigured)?;
            let flags = if block {
                MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS(0)
            } else {
                MF_EVENT_FLAG_NO_WAIT
            };
            // SAFETY: event query on the owning thread.
            match unsafe { gen.GetEvent(flags) } {
                Ok(ev) => ev,
                Err(e) if e.code() == MF_E_NO_EVENTS_AVAILABLE => return Ok(PumpResult::NoEvents),
                Err(e) => return Err(mf_err(e)),
            }
        };
        // SAFETY: reading the event type off a valid event.
        let ty = unsafe { event.GetType() }.map_err(mf_err)? as i32;
        if ty == METransformNeedInput.0 {
            let st = self.state.as_mut().ok_or(G2gError::NotConfigured)?;
            if let Some(s) = st.pending_input.pop_front() {
                // SAFETY: ProcessInput on the owning thread.
                unsafe { st.transform.ProcessInput(0, &s, 0) }.map_err(mf_err)?;
            } else {
                st.need_input_credits += 1;
            }
        } else if ty == METransformHaveOutput.0 {
            match self.process_output()? {
                OutputStep::Frame(c) => encoded.push(c),
                OutputStep::NeedInput => {}
                OutputStep::StreamChange => self.renegotiate()?,
            }
        } else if ty == METransformDrainComplete.0 {
            return Ok(PumpResult::DrainComplete);
        }
        Ok(PumpResult::Pumped)
    }

    fn process_output(&self) -> Result<OutputStep, G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        let sample = match process_output(&st.transform, st.provides_samples, st.out_size)? {
            OutputPull::Sample(sample) => sample,
            OutputPull::NeedInput => return Ok(OutputStep::NeedInput),
            OutputPull::StreamChange => return Ok(OutputStep::StreamChange),
        };
        // SAFETY: reading an attribute off a valid sample on the owning thread.
        let duration_ns = unsafe { sample.GetSampleDuration() }
            .map(mf_ticks_to_ns)
            .unwrap_or(0);
        Ok(OutputStep::Frame(EncodedChunk {
            data: read_sample_bytes(&sample, |bytes| bytes.into())?,
            width: st.width,
            height: st.height,
            framerate: self.framerate.clone(),
            pts_ns: sample_time_ns(&sample),
            duration_ns,
        }))
    }

    /// Re-pick the H.264 output type after a stream change (rare for an
    /// encoder, but the MFT contract allows it) and refresh the output buffer
    /// size from it.
    fn renegotiate(&mut self) -> Result<(), G2gError> {
        let st = self.state.as_mut().ok_or(G2gError::NotConfigured)?;
        set_output_from_available(&st.transform, st.out_subtype)?;
        st.out_size = output_buffer_size(&st.transform, st.width, st.height)?;
        st.provides_samples = output_provides_samples(&st.transform)?;
        Ok(())
    }
}

impl AsyncElement for MfEncode {
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
        // Consumes packed NV12 at any geometry; intersecting narrows the
        // proposal and rejects everything else.
        let supported = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        upstream_caps.intersect(&supported)
    }

    /// Native `DerivedOutput`: accepts NV12 at any geometry and produces
    /// H.264 at the same dims/framerate, the inverse of `MfDecode`'s
    /// constraint. The closure returns an empty set on a non-NV12 input, so
    /// the solver rejects incompatible upstream at negotiation time.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let codec = self.codec;
        CapsConstraint::DerivedOutput(Box::new(move |input: &Caps| {
            derive_output_caps(codec, input)
        }))
    }

    /// M12: record the downstream consumer's allocation proposal. The encoder
    /// emits system-memory bitstream buffers sized per encoded frame, so the
    /// request is recorded for diagnostics.
    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.requested_alloc = Some(*params);
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // The encoder's media types need concrete dims up front (no
        // bitstream to derive them from), and NV12 needs even dims.
        let (w, h, framerate) = match absolute_caps {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(w),
                height: Dim::Fixed(h),
                framerate,
                ..
            } => (*w, *h, framerate.clone()),
            _ => return Err(G2gError::CapsMismatch),
        };
        if w == 0 || h == 0 || w % 2 != 0 || h % 2 != 0 {
            return Err(G2gError::CapsMismatch);
        }

        // SAFETY: COM/MF startup on the calling thread; every later COM call
        // lands on this same thread (see the Send contract above).
        // CoInitializeEx returning S_FALSE (already initialised) is fine.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            MFStartup(MF_VERSION, MFSTARTUP_FULL).map_err(mf_err)?;
        }
        self.com_started = true;

        self.state = Some(init_encoder(
            self.codec,
            w,
            h,
            rate_to_ratio(&framerate),
            self.bitrate,
            self.prefer_hardware,
        )?);
        self.framerate = framerate;
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
            let mut encoded = Vec::new();
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    self.feed(slice, &frame.timing, &mut encoded)?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // The runner's pre-fixed output caps (our compressed codec)
                    // are forwarded so the sink sees them before the first
                    // access unit (M733/M734, see `ffmpegdec.rs` "Two callers").
                    if matches!(&c, Caps::CompressedVideo { codec, .. } if *codec == self.codec) {
                        out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                        self.last_caps = Some(c);
                        return Ok(());
                    }
                    let (w, h, framerate) = match &c {
                        Caps::RawVideo {
                            format: RawVideoFormat::Nv12,
                            width: Dim::Fixed(w),
                            height: Dim::Fixed(h),
                            framerate,
                            ..
                        } => (*w, *h, framerate.clone()),
                        _ => return Err(G2gError::CapsMismatch),
                    };
                    if w == 0 || h == 0 || w % 2 != 0 || h % 2 != 0 {
                        return Err(G2gError::CapsMismatch);
                    }
                    // A geometry change rebuilds the MFT: drain the old
                    // encoder first so buffered pictures emit under the caps
                    // they were encoded with (the chunks carry them), then
                    // start fresh at the new dims. A framerate-only change is
                    // informational.
                    let geometry_changed = self
                        .state
                        .as_ref()
                        .is_some_and(|st| (st.width, st.height) != (w, h));
                    if geometry_changed {
                        self.drain_eos(&mut encoded)?;
                        self.state = Some(init_encoder(
                            self.codec,
                            w,
                            h,
                            rate_to_ratio(&framerate),
                            self.bitrate,
                            self.prefer_hardware,
                        )?);
                    }
                    // Track the rate even on a framerate-only change (no rebuild),
                    // so the reported output caps are not stale.
                    self.framerate = framerate;
                    self.input_caps = Some(c);
                }
                PipelinePacket::Flush => {
                    if let Some(st) = self.state.as_mut() {
                        // SAFETY: flush message on the owning thread.
                        unsafe {
                            st.transform
                                .ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0)
                                .map_err(mf_err)?;
                        }
                        // Drop any input the async MFT never accepted.
                        st.pending_input.clear();
                        st.need_input_credits = 0;
                    }
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    self.drain_eos(&mut encoded)?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }

            for c in encoded {
                let new_caps = compressed_caps(self.codec, c.width, c.height, c.framerate);
                if self.last_caps.as_ref() != Some(&new_caps) {
                    // Encode-time output must overlap the closure's
                    // derivation of the recorded input (same Phase A
                    // assertion as the decoders).
                    #[cfg(debug_assertions)]
                    if let Some(input) = self.input_caps.as_ref() {
                        let expected = derive_output_caps(self.codec, input);
                        debug_assert!(
                            !expected
                                .intersect(&CapsSet::one(new_caps.clone()))
                                .is_empty(),
                            "mfencode encode-time output {new_caps:?} inconsistent with derive_output_caps({input:?}) = {expected:?}"
                        );
                    }
                    out.push(PipelinePacket::CapsChanged(new_caps.clone()))
                        .await?;
                    self.last_caps = Some(new_caps);
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(c.data)),
                    timing: FrameTiming {
                        pts_ns: c.pts_ns,
                        dts_ns: c.pts_ns,
                        duration_ns: c.duration_ns,
                        capture_ns: c.pts_ns,
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

impl PadTemplates for MfEncode {
    /// Consumes NV12 and produces H.264 or H.265, both at any geometry; the
    /// inverse of `MfDecode`'s templates. The static superset advertises both
    /// codecs on the source pad; the configured instance narrows to one.
    fn pad_templates() -> Vec<PadTemplate> {
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let compressed = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
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

impl Drop for MfEncode {
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

/// The output media subtype an encoder produces for a codec. Only H.264 and
/// H.265 map to a Media Foundation encoder; any other codec is rejected loud.
fn encoder_subtype(codec: VideoCodec) -> Result<windows::core::GUID, G2gError> {
    match codec {
        VideoCodec::H264 => Ok(MFVideoFormat_H264),
        VideoCodec::H265 => Ok(MFVideoFormat_HEVC),
        _ => Err(G2gError::CapsMismatch),
    }
}

/// Instantiate the encoder MFT for a codec, returning `(transform, is_async)`.
/// H.264 uses the fixed-CLSID MS software encoder (synchronous) unless
/// `prefer_hardware`; H.265 (no fixed CLSID) and the hardware H.264 path are
/// enumerated by output subtype. An enumerated asynchronous MFT is unlocked
/// for use and driven by the event-based path.
fn create_encoder_transform(
    codec: VideoCodec,
    prefer_hardware: bool,
) -> Result<(IMFTransform, bool), G2gError> {
    match codec {
        VideoCodec::H264 if !prefer_hardware => {
            // SAFETY: COM object creation on the owning thread.
            let t =
                unsafe { CoCreateInstance(&CLSID_MSH264EncoderMFT, None, CLSCTX_INPROC_SERVER) }
                    .map_err(mf_err)?;
            Ok((t, false))
        }
        VideoCodec::H264 => enumerate_encoder(MFVideoFormat_H264),
        VideoCodec::H265 => enumerate_encoder(MFVideoFormat_HEVC),
        _ => Err(G2gError::CapsMismatch),
    }
}

/// Find and activate an encoder MFT that outputs `output_subtype` via
/// `MFTEnumEx`, preferring the first match (sort-and-filter orders hardware
/// first). An asynchronous MFT is unlocked (`MF_TRANSFORM_ASYNC_UNLOCK`) so it
/// can be driven by the event loop. Returns `(transform, is_async)`; fails
/// `Hardware` when no encoder is registered.
fn enumerate_encoder(
    output_subtype: windows::core::GUID,
) -> Result<(IMFTransform, bool), G2gError> {
    let out_info = MFT_REGISTER_TYPE_INFO {
        guidMajorType: MFMediaType_Video,
        guidSubtype: output_subtype,
    };
    let flags = MFT_ENUM_FLAG_SYNCMFT
        | MFT_ENUM_FLAG_ASYNCMFT
        | MFT_ENUM_FLAG_HARDWARE
        | MFT_ENUM_FLAG_SORTANDFILTER;

    let mut activates: *mut Option<IMFActivate> = core::ptr::null_mut();
    let mut count = 0u32;
    // SAFETY: MFTEnumEx allocates a CoTaskMem array of `count` activation
    // objects, freed below. We pass our output type-info and no input
    // constraint.
    unsafe {
        MFTEnumEx(
            MFT_CATEGORY_VIDEO_ENCODER,
            flags,
            None,
            Some(&out_info),
            &mut activates,
            &mut count,
        )
        .map_err(mf_err)?;
    }

    let mut chosen: Option<Result<(IMFTransform, bool), G2gError>> = None;
    // SAFETY: `activates` points to `count` initialised `Option<IMFActivate>`
    // entries when count > 0. We take ownership of each entry (releasing the
    // COM ref on drop), activating the first one.
    unsafe {
        for i in 0..count as usize {
            let entry = core::ptr::read(activates.add(i));
            if let Some(activate) = entry {
                if chosen.is_none() {
                    let is_async = activate.GetUINT32(&MF_TRANSFORM_ASYNC).unwrap_or(0) != 0;
                    chosen = Some(activate_encoder(&activate, is_async));
                }
                // `activate` drops here, releasing the enumerated ref.
            }
        }
        if !activates.is_null() {
            CoTaskMemFree(Some(activates.cast()));
        }
    }

    chosen.unwrap_or(Err(G2gError::Hardware(HardwareError::Other)))
}

/// Activate one enumerated encoder, unlocking it first when asynchronous so the
/// event-driven path may drive it.
fn activate_encoder(
    activate: &IMFActivate,
    is_async: bool,
) -> Result<(IMFTransform, bool), G2gError> {
    // SAFETY: object activation + attribute set on the owning thread.
    let transform = unsafe { activate.ActivateObject::<IMFTransform>() }.map_err(mf_err)?;
    if is_async {
        // SAFETY: unlocking the async MFT before any media type is set.
        unsafe {
            let attrs = transform.GetAttributes().map_err(mf_err)?;
            attrs
                .SetUINT32(&MF_TRANSFORM_ASYNC_UNLOCK, 1)
                .map_err(mf_err)?;
        }
    }
    Ok((transform, is_async))
}

/// Create the encoder MFT for `codec`, set the output type then the NV12 input
/// type (the encoder contract requires output before input), and put it into
/// streaming mode.
fn init_encoder(
    codec: VideoCodec,
    width: u32,
    height: u32,
    fps: (u32, u32),
    bitrate: u32,
    prefer_hardware: bool,
) -> Result<EncoderState, G2gError> {
    let out_subtype = encoder_subtype(codec)?;
    let (transform, async_mode) = create_encoder_transform(codec, prefer_hardware)?;

    // Low-latency mode (the attribute alias of CODECAPI_AVLowLatencyMode):
    // no B-frames / lookahead, one output per input. Best-effort, set before
    // the media types so it shapes the encoder configuration.
    // SAFETY: attribute-store access on the owning thread.
    unsafe {
        if let Ok(attrs) = transform.GetAttributes() {
            let _ = attrs.SetUINT32(&MF_LOW_LATENCY, 1);
        }
    }

    let (num, den) = fps;

    // SAFETY: media-type configuration on the owning thread; all arguments
    // are valid for the duration of each call.
    unsafe {
        let output = MFCreateMediaType().map_err(mf_err)?;
        output
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .map_err(mf_err)?;
        output
            .SetGUID(&MF_MT_SUBTYPE, &out_subtype)
            .map_err(mf_err)?;
        output
            .SetUINT32(&MF_MT_AVG_BITRATE, bitrate)
            .map_err(mf_err)?;
        output
            .SetUINT64(&MF_MT_FRAME_SIZE, pack_u32_pair(width, height))
            .map_err(mf_err)?;
        output
            .SetUINT64(&MF_MT_FRAME_RATE, pack_u32_pair(num, den))
            .map_err(mf_err)?;
        output
            .SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)
            .map_err(mf_err)?;
        // H.264 pins Main profile via MF_MT_MPEG2_PROFILE; HEVC has no MS
        // software encoder and an enumerated HW encoder picks its own profile,
        // so leave it unset there.
        if codec == VideoCodec::H264 {
            output
                .SetUINT32(&MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_Main.0 as u32)
                .map_err(mf_err)?;
        }
        transform.SetOutputType(0, &output, 0).map_err(mf_err)?;

        let input = MFCreateMediaType().map_err(mf_err)?;
        input
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .map_err(mf_err)?;
        input
            .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)
            .map_err(mf_err)?;
        input
            .SetUINT64(&MF_MT_FRAME_SIZE, pack_u32_pair(width, height))
            .map_err(mf_err)?;
        input
            .SetUINT64(&MF_MT_FRAME_RATE, pack_u32_pair(num, den))
            .map_err(mf_err)?;
        input
            .SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)
            .map_err(mf_err)?;
        transform.SetInputType(0, &input, 0).map_err(mf_err)?;
    }

    let out_size = output_buffer_size(&transform, width, height)?;
    let provides_samples = output_provides_samples(&transform)?;

    // An async MFT is driven through its event generator.
    let event_gen = if async_mode {
        Some(transform.cast::<IMFMediaEventGenerator>().map_err(mf_err)?)
    } else {
        None
    };

    // SAFETY: streaming-mode messages on the owning thread.
    unsafe {
        transform
            .ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)
            .map_err(mf_err)?;
        transform
            .ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)
            .map_err(mf_err)?;
    }

    Ok(EncoderState {
        transform,
        out_subtype,
        width,
        height,
        out_size,
        duration_hns: frame_duration_hns(num, den),
        provides_samples,
        async_mode,
        event_gen,
        pending_input: VecDeque::new(),
        need_input_credits: 0,
    })
}

/// Re-select the configured output subtype from the MFT's available types
/// (stream change recovery).
fn set_output_from_available(
    transform: &IMFTransform,
    out_subtype: windows::core::GUID,
) -> Result<(), G2gError> {
    let mut i = 0u32;
    loop {
        // SAFETY: type enumeration on the owning thread.
        let candidate = unsafe { transform.GetOutputAvailableType(0, i) };
        match candidate {
            Ok(t) => {
                // SAFETY: reading attributes off a valid media type.
                let subtype = unsafe { t.GetGUID(&MF_MT_SUBTYPE) }.map_err(mf_err)?;
                if subtype == out_subtype {
                    // SAFETY: applying the chosen output type on the owning thread.
                    unsafe { transform.SetOutputType(0, &t, 0) }.map_err(mf_err)?;
                    return Ok(());
                }
                i += 1;
            }
            Err(e) => return Err(mf_err(e)),
        }
    }
}

/// Output buffer bytes to allocate per `ProcessOutput`: the MFT's reported
/// size, floored at the raw NV12 frame size so a zero/early estimate never
/// under-allocates (an encoded frame is in practice far smaller).
fn output_buffer_size(transform: &IMFTransform, w: u32, h: u32) -> Result<u32, G2gError> {
    // SAFETY: stream-info query on the owning thread.
    let info = unsafe { transform.GetOutputStreamInfo(0) }.map_err(mf_err)?;
    let nv12 = RawVideoFormat::Nv12
        .unpadded_frame_bytes(w, h)
        .map_or(u32::MAX, |bytes| u32::try_from(bytes).unwrap_or(u32::MAX));
    Ok(info.cbSize.max(nv12))
}

fn compressed_caps(codec: VideoCodec, w: u32, h: u32, framerate: Rate) -> Caps {
    Caps::CompressedVideo {
        codec,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate,
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
        alignment: g2g_core::BitstreamAlignment::Unspecified,
    }
}

/// Single source of truth for the encoder's output-side caps derivation,
/// shared by the `DerivedOutput` constraint closure and the debug assertion.
/// The inverse of `mfdecode::derive_output_caps`: NV12 in, the configured
/// codec out.
fn derive_output_caps(codec: VideoCodec, input: &Caps) -> CapsSet {
    match input {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width,
            height,
            framerate,
            ..
        } => CapsSet::one(Caps::CompressedVideo {
            codec,
            width: width.clone(),
            height: height.clone(),
            framerate: framerate.clone(),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        }),
        _ => CapsSet::from_alternatives(Vec::new()),
    }
}

/// Convert a Q16 fixed-point framerate to the (numerator, denominator) ratio
/// MF media types carry, defaulting to 30/1 when the rate is open.
fn rate_to_ratio(rate: &Rate) -> (u32, u32) {
    match rate {
        Rate::Fixed(q16) if *q16 > 0 => framerate_q16_to_fraction(*q16),
        _ => ENCODER_FRAMERATE_WHEN_UNDECLARED,
    }
}

// the MS encoder rejects a sample with no duration (MF_E_NO_SAMPLE_DURATION)
fn sample_duration_hns(frame_duration_ns: u64, rate_duration_hns: i64) -> i64 {
    if frame_duration_ns > 0 {
        return ns_to_mf_ticks(frame_duration_ns);
    }
    rate_duration_hns
}

/// Per-frame duration in 100-ns units for an `fps = num/den` stream.
fn frame_duration_hns(num: u32, den: u32) -> i64 {
    if num == 0 {
        return 0;
    }
    (10_000_000i64 * den as i64) / num as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intercept_rejects_compressed_input() {
        let enc = MfEncode::new();
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert_eq!(enc.intercept_caps(&h264), Err(G2gError::CapsMismatch));
    }

    #[test]
    fn intercept_narrows_nv12_geometry() {
        let enc = MfEncode::new();
        let proposal = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(enc.intercept_caps(&proposal), Ok(proposal));
    }

    #[test]
    fn caps_constraint_is_derived_output_nv12_to_h264() {
        let enc = MfEncode::new();
        let CapsConstraint::DerivedOutput(f) = enc.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(
            f(&nv12).alternatives(),
            &[Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(30 << 16),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                alignment: g2g_core::BitstreamAlignment::Unspecified,
            }]
        );

        let rgba = Caps::RawVideo {
            format: RawVideoFormat::Rgba8,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(f(&rgba).is_empty());
    }

    #[test]
    fn pad_templates_are_nv12_in_h264_out() {
        use g2g_core::{PadDirection, PadTemplates};
        let sink = MfEncode::pad_template(PadDirection::Sink).expect("has sink pad");
        let source = MfEncode::pad_template(PadDirection::Source).expect("has source pad");
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            alignment: g2g_core::BitstreamAlignment::Unspecified,
        };
        assert!(matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&nv12)));
        assert!(matches!(source.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&h264)));
        assert!(!matches!(sink.caps, g2g_core::PadCaps::Fixed(ref s) if s.accepts(&h264)));
    }

    #[test]
    fn bitrate_builder_defaults_and_overrides() {
        assert_eq!(MfEncode::new().bitrate(), DEFAULT_BITRATE);
        assert_eq!(MfEncode::new().with_bitrate(750_000).bitrate(), 750_000);
    }

    #[test]
    fn codec_defaults_h264_and_selects_h265() {
        assert_eq!(MfEncode::new().codec(), VideoCodec::H264);
        assert_eq!(
            MfEncode::new().with_codec(VideoCodec::H265).codec(),
            VideoCodec::H265
        );
    }

    #[test]
    fn async_state_is_unknown_before_configure() {
        // is_async() reports the live MFT's mode; None until configured.
        assert_eq!(MfEncode::new().is_async(), None);
        assert_eq!(MfEncode::new().with_hardware().is_async(), None);
    }

    #[test]
    fn encoder_subtype_maps_supported_codecs() {
        assert_eq!(
            encoder_subtype(VideoCodec::H264).unwrap(),
            MFVideoFormat_H264
        );
        assert_eq!(
            encoder_subtype(VideoCodec::H265).unwrap(),
            MFVideoFormat_HEVC
        );
        assert_eq!(
            encoder_subtype(VideoCodec::Vp9),
            Err(G2gError::CapsMismatch)
        );
    }

    #[test]
    fn hevc_instance_derives_hevc_output_from_nv12() {
        let enc = MfEncode::new().with_codec(VideoCodec::H265);
        let CapsConstraint::DerivedOutput(f) = enc.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Fixed(30 << 16),
            interlace: Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert_eq!(
            f(&nv12).alternatives(),
            &[Caps::CompressedVideo {
                codec: VideoCodec::H265,
                width: Dim::Fixed(1280),
                height: Dim::Fixed(720),
                framerate: Rate::Fixed(30 << 16),
                colorimetry: g2g_core::Colorimetry::UNKNOWN,
                alignment: g2g_core::BitstreamAlignment::Unspecified,
            }]
        );
    }

    #[test]
    fn rate_to_ratio_reduces_q16() {
        // integer fps reduces fully
        assert_eq!(rate_to_ratio(&Rate::Fixed(30 << 16)), (30, 1));
        // 30.5 fps = (30.5 * 65536) / 65536 = 61/2
        assert_eq!(rate_to_ratio(&Rate::Fixed((61 << 16) / 2)), (61, 2));
        // open / zero rates default to 30/1
        assert_eq!(rate_to_ratio(&Rate::Any), ENCODER_FRAMERATE_WHEN_UNDECLARED);
        assert_eq!(
            rate_to_ratio(&Rate::Fixed(0)),
            ENCODER_FRAMERATE_WHEN_UNDECLARED
        );
    }

    #[test]
    fn frame_duration_is_hns_per_frame() {
        assert_eq!(frame_duration_hns(30, 1), 333_333);
        assert_eq!(frame_duration_hns(61, 2), 327_868);
        assert_eq!(frame_duration_hns(0, 1), 0);
    }

    #[test]
    fn sample_duration_prefers_the_frame_over_the_configured_rate() {
        const FRAME_DURATION_NS: u64 = 40_000_000;
        let thirty = frame_duration_hns(30, 1);
        assert_eq!(
            sample_duration_hns(FRAME_DURATION_NS, thirty),
            ns_to_mf_ticks(FRAME_DURATION_NS)
        );
        assert_eq!(sample_duration_hns(0, thirty), thirty);
    }
}
