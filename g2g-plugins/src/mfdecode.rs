//! Windows H.264 / H.265 decode element wrapping a Media Foundation decoder
//! MFT (`CLSID_MSH264DecoderMFT` or `CLSID_MSH265DecoderMFT`, an
//! `IMFTransform`). Codec is selected with [`MfDecode::with_codec`]; the
//! default is H.264.
//!
//! M13: consumes Annex-B H.264 `DataFrame`s (the bitstream `RtspSrc` /
//! `H264Parse` already emit, `MemoryDomain::System`) and produces decoded
//! NV12 frames, also `MemoryDomain::System` (CPU copy out of the MFT's output
//! buffer). A `CapsChanged(Nv12, w, h)` is emitted before the first decoded
//! frame and again whenever the decoder signals a resolution change.
//!
//! M30: the same pipeline carries H.265/HEVC when constructed with
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
//! up; the MS software decoder packs tightly). `copy_sample` strips that
//! padding via `pack_nv12` so downstream always sees tightly-packed NV12.
//!
//! DXVA / D3D11 (`with_d3d11`): opts into GPU decode. A hardware D3D11 device +
//! DXGI manager is handed to the sync MFT (`MFT_MESSAGE_SET_D3D_MANAGER`), so
//! decode runs on the GPU and the MFT allocates D3D11-backed output samples.
//! Each frame is emitted as `MemoryDomain::D3D11Texture` (zero-copy: the
//! texture stays on the GPU, the owning `IMFSample` is its keep-alive). The
//! default software path is unchanged (system NV12).

use core::future::Future;
use core::mem::ManuallyDrop;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use windows::core::Interface;
use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
use windows::Win32::Graphics::Direct3D11::{
    D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Multithread, ID3D11Texture2D,
    D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_VIDEO_SUPPORT, D3D11_SDK_VERSION,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_NV12;
use windows::Win32::Media::MediaFoundation::{
    CLSID_MSH264DecoderMFT, CLSID_MSH265DecoderMFT, IMFDXGIBuffer, IMFDXGIDeviceManager, IMFSample,
    IMFTransform, MFCreateDXGIDeviceManager, MFCreateMediaType, MFCreateMemoryBuffer,
    MFCreateSample, MFMediaType_Video, MFShutdown, MFStartup, MFVideoFormat_H264,
    MFVideoFormat_HEVC, MFVideoFormat_NV12, MFSTARTUP_FULL, MFT_MESSAGE_COMMAND_DRAIN,
    MFT_MESSAGE_COMMAND_FLUSH, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING,
    MFT_MESSAGE_NOTIFY_START_OF_STREAM, MFT_MESSAGE_SET_D3D_MANAGER, MFT_OUTPUT_DATA_BUFFER,
    MFT_OUTPUT_STREAM_PROVIDES_SAMPLES, MF_E_NOTACCEPTING, MF_E_TRANSFORM_NEED_MORE_INPUT,
    MF_E_TRANSFORM_STREAM_CHANGE, MF_MT_DEFAULT_STRIDE, MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE,
    MF_MT_SUBTYPE, MF_VERSION,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    D3D11KeepAlive, Dim, FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink,
    OwnedD3D11Texture, PadTemplate, PadTemplates, PipelinePacket, Rate, RawVideoFormat, VideoCodec,
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
    /// (`MFT_OUTPUT_STREAM_PROVIDES_SAMPLES`), as the DXVA / D3D11 path does:
    /// `ProcessOutput` is then called with a null sample and the MFT fills it
    /// with a (D3D11-backed) sample we take ownership of. The software path
    /// clears this and we pre-allocate a system-memory output sample.
    provides_samples: bool,
    /// `ID3D11Device` (as `u64`) for the DXVA path, stamped onto every emitted
    /// [`OwnedD3D11Texture`] so a consumer uses the right device. `0` on the
    /// software path.
    device_ptr: u64,
    /// D3D11 device backing the DXVA decode, kept alive for the decoder's
    /// lifetime. `None` on the software path. Held so the device (and the
    /// DXGI manager that references it) outlive every output texture.
    _d3d_device: Option<ID3D11Device>,
    /// DXGI device manager handed to the MFT via `MFT_MESSAGE_SET_D3D_MANAGER`.
    /// `None` on the software path.
    _dxgi_manager: Option<IMFDXGIDeviceManager>,
}

/// One decoded picture: either copied out to packed system NV12 (software /
/// Phase-2 readback) or left in a D3D11 texture (DXVA zero-copy, Phase 3).
#[derive(Debug)]
enum DecodedPayload {
    System(Box<[u8]>),
    D3D11(OwnedD3D11Texture),
}

/// One decoded picture plus its geometry and timestamp.
#[derive(Debug)]
struct DecodedPicture {
    payload: DecodedPayload,
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

#[derive(Debug)]
pub struct MfDecode {
    /// Bitstream codec the MFT decodes (`H264` default, or `H265`). Selects the
    /// decoder CLSID and the input media subtype.
    codec: VideoCodec,
    state: Option<DecoderState>,
    com_started: bool,
    configured: bool,
    /// Opt into DXVA / D3D11 hardware decode (`with_d3d11`). When set,
    /// `configure_pipeline` creates a D3D11 device + DXGI manager and hands it
    /// to the MFT, so decode runs on the GPU. Default `false` (the MS software
    /// decoder, system-memory output).
    use_d3d11: bool,
    last_caps: Option<Caps>,
    /// M16 workaround #3 Phase A: most recent input caps received via
    /// `PipelinePacket::CapsChanged`. Used to validate the format on
    /// mid-stream changes and to debug-assert agreement between the
    /// declared `DerivedOutput` closure and the decode-time output
    /// geometry. See `ffmpegdec.rs` for the same field with full notes.
    input_caps: Option<Caps>,
    emitted: u64,
    /// M12 / W1: the downstream consumer's allocation proposal, recorded in
    /// `configure_allocation`. A `MemoryDomainKind::D3D11Texture` request from a
    /// GPU sink (`D3D11Sink`) is satisfied by construction on the `with_d3d11`
    /// path (it already emits texture-resident frames). Mirrors
    /// `FfmpegH264Dec::requested_alloc`.
    requested_alloc: Option<AllocationParams>,
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
            use_d3d11: false,
            last_caps: None,
            input_caps: None,
            emitted: 0,
            requested_alloc: None,
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

    /// The downstream consumer's recorded M12 allocation proposal, if any
    /// (see [`AsyncElement::configure_allocation`]).
    pub fn requested_alloc(&self) -> Option<AllocationParams> {
        self.requested_alloc
    }

    /// Enable DXVA / D3D11 hardware decode. `configure_pipeline` then creates a
    /// hardware D3D11 device and hands the MFT a DXGI device manager, so decode
    /// runs on the GPU and each frame is emitted as `MemoryDomain::D3D11Texture`
    /// (zero-copy: the decoded NV12 stays in a GPU texture). Fails loud
    /// (`Hardware`) if no D3D11 device is available. The default is the MS
    /// software decoder emitting system-memory NV12.
    pub fn with_d3d11(mut self) -> Self {
        self.use_d3d11 = true;
        self
    }

    /// Whether DXVA / D3D11 hardware decode is enabled (see [`with_d3d11`]).
    pub fn uses_d3d11(&self) -> bool {
        self.use_d3d11
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
        decoded: &mut Vec<DecodedPicture>,
    ) -> Result<(), G2gError> {
        let sample = make_input_sample(data, pts_ns)?;
        let mut guard = 0u32;
        // ProcessInput returns MF_E_NOTACCEPTING when the MFT must release
        // outputs before it can take more input; drain, then retry.
        while !self.process_input(&sample)? {
            self.drain(decoded)?;
            guard += 1;
            if guard > 64 {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
        }
        self.drain(decoded)
    }

    fn process_input(&self, sample: &IMFSample) -> Result<bool, G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        // SAFETY: COM call on the element's owning thread; `sample` is a valid
        // IMFSample we just allocated.
        let r = unsafe { st.transform.ProcessInput(0, sample, 0) };
        match r {
            Ok(()) => Ok(true),
            Err(e) if e.code() == MF_E_NOTACCEPTING => Ok(false),
            Err(e) => Err(mf_err(e)),
        }
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
        {
            let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
            // SAFETY: drain message on the owning thread.
            unsafe {
                st.transform
                    .ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0)
                    .map_err(mf_err)?;
            }
        }
        self.drain(decoded)
    }

    fn process_output(&self) -> Result<OutputStep, G2gError> {
        let st = self.state.as_ref().ok_or(G2gError::NotConfigured)?;
        // SAFETY: every call runs on the element's owning COM thread. We supply
        // an output sample only on the software path; the DXVA path sets
        // `provides_samples`, so the MFT allocates a (D3D11-backed) sample and
        // writes it into the struct, which we take ownership of below. The refs
        // in the FFI struct are reclaimed right after ProcessOutput regardless
        // of its result.
        unsafe {
            let preallocated = if st.provides_samples {
                None
            } else {
                let buffer = MFCreateMemoryBuffer(st.out_size).map_err(mf_err)?;
                let sample = MFCreateSample().map_err(mf_err)?;
                sample.AddBuffer(&buffer).map_err(mf_err)?;
                Some(sample)
            };

            let mut out = [MFT_OUTPUT_DATA_BUFFER {
                dwStreamID: 0,
                pSample: ManuallyDrop::new(preallocated),
                dwStatus: 0,
                pEvents: ManuallyDrop::new(None),
            }];
            let mut status = 0u32;
            let r = st.transform.ProcessOutput(0, &mut out, &mut status);

            // Take the output sample (ours on the software path, the MFT's on
            // the DXVA path) and release any events.
            let out_sample = ManuallyDrop::into_inner(core::mem::replace(
                &mut out[0].pSample,
                ManuallyDrop::new(None),
            ));
            drop(ManuallyDrop::into_inner(core::mem::replace(
                &mut out[0].pEvents,
                ManuallyDrop::new(None),
            )));

            match r {
                Ok(()) => {
                    let sample = out_sample.ok_or(G2gError::Hardware(HardwareError::Other))?;
                    let pts_ns = sample
                        .GetSampleTime()
                        .map(|hns| (hns.max(0) as u64).saturating_mul(100))
                        .unwrap_or(0);
                    let payload = if st.device_ptr != 0 {
                        // DXVA zero-copy (Phase 3): hand the decoded D3D11
                        // texture downstream, keeping the sample alive so the
                        // texture stays valid until the consumer drops it.
                        DecodedPayload::D3D11(extract_texture(
                            sample,
                            st.width,
                            st.height,
                            st.device_ptr,
                        )?)
                    } else {
                        // Software / readback path: copy to packed system NV12.
                        DecodedPayload::System(copy_sample(
                            &sample, st.width, st.height, st.stride,
                        )?)
                    };
                    Ok(OutputStep::Frame(DecodedPicture {
                        payload,
                        width: st.width,
                        height: st.height,
                        pts_ns,
                    }))
                }
                Err(e) if e.code() == MF_E_TRANSFORM_NEED_MORE_INPUT => Ok(OutputStep::NeedInput),
                Err(e) if e.code() == MF_E_TRANSFORM_STREAM_CHANGE => Ok(OutputStep::StreamChange),
                Err(e) => Err(mf_err(e)),
            }
        }
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

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Consumes the configured codec at any geometry; intersecting narrows
        // the proposal and rejects a mismatched codec.
        let supported = Caps::CompressedVideo {
            codec: self.codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    /// M16 step 5l: native `DerivedOutput` — accepts H.264 at any
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

    /// M12 / W1: record the downstream consumer's allocation proposal. A
    /// `MemoryDomainKind::D3D11Texture` request (from `D3D11Sink`) is honoured
    /// by construction on the `with_d3d11` path, which already emits
    /// texture-resident frames; the software path emits system memory, so a
    /// texture request there is unsatisfiable and stays recorded for
    /// diagnostics rather than silently changing the output domain.
    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.requested_alloc = Some(*params);
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = match absolute_caps {
            Caps::CompressedVideo {
                codec,
                width,
                height,
                ..
            } if *codec == self.codec => (fixed_or_zero(width), fixed_or_zero(height)),
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

        self.state = Some(init_decoder(self.codec, w, h, self.use_d3d11)?);
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
                    let Some(slice) = frame.domain.as_system_slice() else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed(slice, frame.timing.pts_ns, &mut decoded)?;
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
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    self.drain_eos(&mut decoded)?;
                }
                PipelinePacket::Segment(seg) => {
                    out.push(PipelinePacket::Segment(seg)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }

            // A compressed stream's rate is advisory (per-frame PTS carries the
            // real timing); never emit `Rate::Any` (a downstream transform
            // cannot fixate() it), default 30/1 like `ffmpegdec`.
            let out_framerate = match &self.input_caps {
                Some(Caps::CompressedVideo {
                    framerate: Rate::Fixed(q),
                    ..
                }) => Rate::Fixed(*q),
                _ => Rate::Fixed(30 << 16),
            };
            for d in decoded {
                let new_caps = nv12_caps(d.width, d.height, out_framerate.clone());
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
                let domain = match d.payload {
                    DecodedPayload::System(bytes) => {
                        MemoryDomain::System(SystemSlice::from_boxed(bytes))
                    }
                    DecodedPayload::D3D11(texture) => MemoryDomain::D3D11Texture(texture),
                };
                let frame = Frame {
                    domain,
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

impl PadTemplates for MfDecode {
    /// Consumes H.264 or H.265 and produces NV12, both at any geometry (the
    /// MFT derives the output dims from the stream). The static superset
    /// advertises both codecs on the sink; the configured instance narrows to
    /// one via `intercept_caps` / `caps_constraint_as_transform`. Memory domain
    /// (System vs D3D11Texture) is not encoded in caps, so the templates are
    /// backend-independent.
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
/// type, and put it into streaming mode. When `use_d3d11`, a hardware D3D11
/// device and DXGI manager are created and handed to the MFT first, so it
/// decodes via DXVA and allocates its own (D3D11-backed) output samples.
fn init_decoder(
    codec: VideoCodec,
    width: u32,
    height: u32,
    use_d3d11: bool,
) -> Result<DecoderState, G2gError> {
    let (clsid, subtype) = decoder_ids(codec)?;
    // SAFETY: COM object creation on the owning thread.
    let transform: IMFTransform =
        unsafe { CoCreateInstance(&clsid, None, CLSCTX_INPROC_SERVER) }.map_err(mf_err)?;

    // DXVA: build the device + manager and hand it to the MFT before any media
    // type is set (the MFT switches to D3D11 output allocation on receipt).
    let (d3d_device, dxgi_manager) = if use_d3d11 {
        let (device, manager, token) = create_d3d11_device_and_manager()?;
        // SAFETY: ResetDevice associates the device with the manager; the
        // SET_D3D_MANAGER message hands the manager (as a ULONG_PTR) to the MFT.
        unsafe {
            manager.ResetDevice(&device, token).map_err(mf_err)?;
            transform
                .ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER, manager.as_raw() as usize)
                .map_err(mf_err)?;
        }
        (Some(device), Some(manager))
    } else {
        (None, None)
    };

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
                .SetUINT64(&MF_MT_FRAME_SIZE, pack_size(width, height))
                .map_err(mf_err)?;
        }
        transform.SetInputType(0, &input, 0).map_err(mf_err)?;
    }

    let (w, h, stride) = set_nv12_output(&transform)?;
    let out_size = output_buffer_size(&transform, w, h)?;
    let provides_samples = output_provides_samples(&transform)?;
    let device_ptr = d3d_device.as_ref().map(|d| d.as_raw() as u64).unwrap_or(0);

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
        device_ptr,
        _d3d_device: d3d_device,
        _dxgi_manager: dxgi_manager,
    })
}

/// Create a hardware D3D11 device with video support and wrap it in a Media
/// Foundation DXGI device manager. Returns the device, the manager, and the
/// manager's reset token. The device is created with multithread protection
/// on, which Media Foundation requires for a shared decode device.
fn create_d3d11_device_and_manager() -> Result<(ID3D11Device, IMFDXGIDeviceManager, u32), G2gError>
{
    // SAFETY: D3D11/MF object creation on the owning thread; out-params are
    // initialised by the calls before we read them.
    unsafe {
        let mut device: Option<ID3D11Device> = None;
        let mut context: Option<ID3D11DeviceContext> = None;
        D3D11CreateDevice(
            None,
            D3D_DRIVER_TYPE_HARDWARE,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            None, // default feature levels
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
        .map_err(mf_err)?;
        let device = device.ok_or(G2gError::Hardware(HardwareError::Other))?;
        let context = context.ok_or(G2gError::Hardware(HardwareError::Other))?;

        // MF requires the decode device to be multithread-protected.
        let multithread: ID3D11Multithread = context.cast().map_err(mf_err)?;
        let _ = multithread.SetMultithreadProtected(true);

        let mut token = 0u32;
        let mut manager: Option<IMFDXGIDeviceManager> = None;
        MFCreateDXGIDeviceManager(&mut token, &mut manager).map_err(mf_err)?;
        let manager = manager.ok_or(G2gError::Hardware(HardwareError::Other))?;

        Ok((device, manager, token))
    }
}

/// Whether the MFT allocates its own output samples (the DXVA / D3D11 path).
fn output_provides_samples(transform: &IMFTransform) -> Result<bool, G2gError> {
    // SAFETY: stream-info query on the owning thread.
    let info = unsafe { transform.GetOutputStreamInfo(0) }.map_err(mf_err)?;
    Ok(info.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 as u32 != 0)
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
                    let (w, h) = unpack_size(size);
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
    let nv12 = w.saturating_mul(h).saturating_mul(3) / 2;
    Ok(info.cbSize.max(nv12))
}

/// Wrap an access unit in an `IMFSample` backed by a copied memory buffer,
/// stamped with the presentation time (MF uses 100-ns units).
fn make_input_sample(data: &[u8], pts_ns: u64) -> Result<IMFSample, G2gError> {
    let len = data.len() as u32;
    // SAFETY: buffer allocation, locked copy, and sample assembly on the
    // owning thread. `ptr` is valid for `len` bytes between Lock and Unlock.
    unsafe {
        let buffer = MFCreateMemoryBuffer(len).map_err(mf_err)?;
        let mut ptr: *mut u8 = core::ptr::null_mut();
        buffer.Lock(&mut ptr, None, None).map_err(mf_err)?;
        core::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        buffer.Unlock().map_err(mf_err)?;
        buffer.SetCurrentLength(len).map_err(mf_err)?;

        let sample = MFCreateSample().map_err(mf_err)?;
        sample.AddBuffer(&buffer).map_err(mf_err)?;
        sample
            .SetSampleTime((pts_ns / 100) as i64)
            .map_err(mf_err)?;
        Ok(sample)
    }
}

/// Copy the decoded pixels out of a sample into a tightly-packed NV12 buffer,
/// stripping any per-row stride padding the MFT applied.
fn copy_sample(
    sample: &IMFSample,
    width: u32,
    height: u32,
    stride: u32,
) -> Result<Box<[u8]>, G2gError> {
    // SAFETY: contiguous-buffer access on the owning thread; `ptr` is valid
    // for `len` bytes between Lock and Unlock, where we copy it out.
    unsafe {
        let buffer = sample.ConvertToContiguousBuffer().map_err(mf_err)?;
        let mut ptr: *mut u8 = core::ptr::null_mut();
        let mut len: u32 = 0;
        buffer
            .Lock(&mut ptr, None, Some(&mut len))
            .map_err(mf_err)?;
        let src = core::slice::from_raw_parts(ptr, len as usize);
        let packed = pack_nv12(src, width as usize, height as usize, stride as usize);
        buffer.Unlock().map_err(mf_err)?;
        packed
    }
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

/// Extract the decoded D3D11 texture from a DXVA output sample (whose first
/// buffer is an `IMFDXGIBuffer` over the decoder's output texture array) into
/// an [`OwnedD3D11Texture`]. The keep-alive owns the `IMFSample`, so the
/// texture stays valid until the downstream consumer drops the frame. NV12
/// (the negotiated output format) is the texture's `DXGI_FORMAT`.
fn extract_texture(
    sample: IMFSample,
    width: u32,
    height: u32,
    device_ptr: u64,
) -> Result<OwnedD3D11Texture, G2gError> {
    // SAFETY: COM calls on the owning thread. The DXVA path guarantees the
    // sample's first buffer is an `IMFDXGIBuffer`; `GetResource` hands back an
    // owned ref to the `ID3D11Texture2D` and `GetSubresourceIndex` the slot
    // within the decoder's texture array.
    let (texture_ptr, subresource) = unsafe {
        let buffer = sample.GetBufferByIndex(0).map_err(mf_err)?;
        let dxgi: IMFDXGIBuffer = buffer.cast().map_err(mf_err)?;
        let mut raw: *mut core::ffi::c_void = core::ptr::null_mut();
        dxgi.GetResource(&ID3D11Texture2D::IID, &mut raw)
            .map_err(mf_err)?;
        if raw.is_null() {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        // Take ownership of the AddRef'd ref just to read the pointer value,
        // then drop it: the IMFSample (held by the keep-alive below) keeps the
        // texture alive, so this extra ref is redundant.
        let texture = ID3D11Texture2D::from_raw(raw);
        let ptr = texture.as_raw() as u64;
        let subresource = dxgi.GetSubresourceIndex().map_err(mf_err)?;
        (ptr, subresource)
    };
    // Arc, not Box: shareable so a tee can fan the D3D11 texture out to several
    // consumers zero-copy (M213).
    let keep_alive = alloc::sync::Arc::new(SampleOwner(sample));
    Ok(OwnedD3D11Texture::new(
        texture_ptr,
        subresource,
        width,
        height,
        DXGI_FORMAT_NV12.0 as u32,
        device_ptr,
        keep_alive,
    ))
}

/// Keeps a DXVA output `IMFSample` alive so its D3D11 texture stays valid while
/// the frame travels downstream. Boxed as the [`D3D11KeepAlive`] of an
/// [`OwnedD3D11Texture`]; dropping it releases the sample back to the decoder's
/// output texture pool.
struct SampleOwner(#[allow(dead_code)] IMFSample);

// SAFETY: an `IMFSample` is a COM object and `!Send` by default. We uphold
// `Send` by the same contract as `MfDecode` itself: COM is initialised MTA
// (free-threaded) and the sample is owned and moved, never aliased across
// threads.
unsafe impl Send for SampleOwner {}

// SAFETY: `Sync` (M213) lets a tee share the keep-alive across branches that
// read the D3D11 texture concurrently. Sound because the owner is inert (it only
// pins the `IMFSample`, no interior mutability), COM is MTA / free-threaded, the
// decoded texture is read-only, and the final release is serialized by the `Arc`.
unsafe impl Sync for SampleOwner {}

impl core::fmt::Debug for SampleOwner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("SampleOwner(<IMFSample>)")
    }
}

impl D3D11KeepAlive for SampleOwner {}

fn nv12_caps(w: u32, h: u32, framerate: Rate) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate,
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
        } if *c == codec => CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: width.clone(),
            height: height.clone(),
            framerate: framerate.clone(),
        }),
        _ => CapsSet::from_alternatives(Vec::new()),
    }
}

fn fixed_or_zero(d: &Dim) -> u32 {
    match d {
        Dim::Fixed(v) => *v,
        _ => 0,
    }
}

/// MF packs a frame size as `(width << 32) | height` in a `UINT64` attribute.
fn pack_size(width: u32, height: u32) -> u64 {
    ((width as u64) << 32) | (height as u64)
}

fn unpack_size(v: u64) -> (u32, u32) {
    ((v >> 32) as u32, (v & 0xFFFF_FFFF) as u32)
}

fn mf_err(e: windows::core::Error) -> G2gError {
    G2gError::Hardware(HardwareError::MediaFoundation(e.code().0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn frame_size_packs_and_unpacks() {
        assert_eq!(unpack_size(pack_size(1920, 1080)), (1920, 1080));
        assert_eq!(pack_size(1280, 720), (1280u64 << 32) | 720);
    }

    #[test]
    fn nv12_caps_are_fixed() {
        assert_eq!(
            nv12_caps(640, 480, Rate::Fixed(30 << 16)),
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
            }
        );
    }

    #[test]
    fn fixed_or_zero_reads_dims() {
        assert_eq!(fixed_or_zero(&Dim::Fixed(720)), 720);
        assert_eq!(fixed_or_zero(&Dim::Any), 0);
    }

    #[test]
    fn intercept_rejects_non_h264() {
        let dec = MfDecode::new();
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
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
        };
        assert_eq!(dec.intercept_caps(&proposal), Ok(proposal));
    }

    #[test]
    fn d3d11_opt_in_defaults_off() {
        assert!(!MfDecode::new().uses_d3d11());
        assert!(MfDecode::new().with_d3d11().uses_d3d11());
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
        };
        assert_eq!(dec.intercept_caps(&h265), Ok(h265.clone()));
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Any,
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
    fn records_downstream_d3d11_allocation_proposal() {
        // W1: a D3D11Sink proposes texture-resident buffers; the runner conveys
        // that to the decoder's configure_allocation. The with_d3d11 path emits
        // D3D11 textures, so the request is honoured by construction; here we
        // assert the proposal is recorded (the GPU path's allocation handshake).
        use g2g_core::MemoryDomainKind;
        let mut dec = MfDecode::new().with_d3d11();
        assert_eq!(dec.requested_alloc(), None);
        let proposal = AllocationParams::d3d11(1920 * 1080 * 3 / 2, 3, 256);
        AsyncElement::configure_allocation(&mut dec, &proposal);
        let recorded = dec.requested_alloc().expect("proposal recorded");
        assert_eq!(recorded.domain, MemoryDomainKind::D3D11Texture);
        assert_eq!(recorded.min_buffers, 3);
        assert_eq!(recorded.align, 256);
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
        };
        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Any,
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
        };
        assert_eq!(
            f(&h264).alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(30 << 16),
            }]
        );

        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
        };
        assert!(f(&vp9).is_empty());
    }
}
