//! Windows H.264 decode element wrapping the Media Foundation H.264 Decoder
//! MFT (`CLSID_MSH264DecoderMFT`, an `IMFTransform`).
//!
//! M13: consumes Annex-B H.264 `DataFrame`s (the bitstream `RtspSrc` /
//! `H264Parse` already emit, `MemoryDomain::System`) and produces decoded
//! NV12 frames, also `MemoryDomain::System` (CPU copy out of the MFT's output
//! buffer). A `CapsChanged(Nv12, w, h)` is emitted before the first decoded
//! frame and again whenever the decoder signals a resolution change.
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
//! Deferred:
//! - D3D11 zero-copy output (would need a new `MemoryDomain` variant); this
//!   element always copies decoded pixels into a `System` slice.
//! - DXVA hardware-accelerated decode (would set `MF_SA_D3D11_AWARE`); the MS
//!   software decoder path is used.

use core::future::Future;
use core::mem::ManuallyDrop;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use windows::Win32::Media::MediaFoundation::{
    IMFSample, IMFTransform, MFCreateMediaType, MFCreateMemoryBuffer, MFCreateSample, MFShutdown,
    MFStartup, CLSID_MSH264DecoderMFT, MFMediaType_Video, MFSTARTUP_FULL, MFT_MESSAGE_COMMAND_DRAIN,
    MFT_MESSAGE_COMMAND_FLUSH, MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, MFT_MESSAGE_NOTIFY_START_OF_STREAM,
    MFT_OUTPUT_DATA_BUFFER, MFVideoFormat_H264, MFVideoFormat_NV12, MF_E_NOTACCEPTING,
    MF_E_TRANSFORM_NEED_MORE_INPUT, MF_E_TRANSFORM_STREAM_CHANGE, MF_MT_DEFAULT_STRIDE,
    MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE, MF_MT_SUBTYPE, MF_VERSION,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED,
};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError,
    HardwareError, MemoryDomain, OutputSink, PipelinePacket, Rate, VideoCodec, RawVideoFormat,
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
}

/// One decoded picture, pixels already copied out of the MFT buffer.
#[derive(Debug)]
struct DecodedNv12 {
    bytes: Box<[u8]>,
    width: u32,
    height: u32,
    pts_ns: u64,
}

/// Result of a single `ProcessOutput` attempt.
#[derive(Debug)]
enum OutputStep {
    Frame(DecodedNv12),
    NeedInput,
    StreamChange,
}

#[derive(Debug)]
pub struct MfDecode {
    state: Option<DecoderState>,
    com_started: bool,
    configured: bool,
    last_caps: Option<Caps>,
    /// M16 workaround #3 Phase A: most recent input caps received via
    /// `PipelinePacket::CapsChanged`. Used to validate the format on
    /// mid-stream changes and to debug-assert agreement between the
    /// declared `DerivedOutput` closure and the decode-time output
    /// geometry. See `ffmpegdec.rs` for the same field with full notes.
    input_caps: Option<Caps>,
    emitted: u64,
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
            state: None,
            com_started: false,
            configured: false,
            last_caps: None,
            input_caps: None,
            emitted: 0,
        }
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
        decoded: &mut Vec<DecodedNv12>,
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
    fn drain(&mut self, decoded: &mut Vec<DecodedNv12>) -> Result<(), G2gError> {
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
    fn drain_eos(&mut self, decoded: &mut Vec<DecodedNv12>) -> Result<(), G2gError> {
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
        // SAFETY: every call runs on the element's owning COM thread. The
        // output sample is caller-allocated; the refs handed to the
        // MFT_OUTPUT_DATA_BUFFER are reclaimed and released right after the
        // ProcessOutput call regardless of its result.
        unsafe {
            let buffer = MFCreateMemoryBuffer(st.out_size).map_err(mf_err)?;
            let sample = MFCreateSample().map_err(mf_err)?;
            sample.AddBuffer(&buffer).map_err(mf_err)?;

            let mut out = [MFT_OUTPUT_DATA_BUFFER {
                dwStreamID: 0,
                pSample: ManuallyDrop::new(Some(sample.clone())),
                dwStatus: 0,
                pEvents: ManuallyDrop::new(None),
            }];
            let mut status = 0u32;
            let r = st.transform.ProcessOutput(0, &mut out, &mut status);

            // Balance the refs we placed into the FFI struct.
            drop(ManuallyDrop::into_inner(core::mem::replace(
                &mut out[0].pSample,
                ManuallyDrop::new(None),
            )));
            drop(ManuallyDrop::into_inner(core::mem::replace(
                &mut out[0].pEvents,
                ManuallyDrop::new(None),
            )));

            match r {
                Ok(()) => {
                    let pts_ns = sample
                        .GetSampleTime()
                        .map(|hns| (hns.max(0) as u64).saturating_mul(100))
                        .unwrap_or(0);
                    let bytes = copy_sample(&sample, st.width, st.height, st.stride)?;
                    Ok(OutputStep::Frame(DecodedNv12 {
                        bytes,
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
        st.width = w;
        st.height = h;
        st.stride = stride;
        Ok(())
    }
}

impl AsyncElement for MfDecode {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        // Consumes H.264 at any geometry; intersecting narrows the proposal
        // and rejects non-H.264 inputs.
        let supported = Caps::CompressedVideo {
            codec: VideoCodec::H264,
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
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| derive_output_caps(input)))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let (w, h) = match absolute_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width,
                height,
                ..
            } => (fixed_or_zero(width), fixed_or_zero(height)),
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

        self.state = Some(init_decoder(w, h)?);
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
                    let MemoryDomain::System(slice) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    self.feed(slice.as_slice(), frame.timing.pts_ns, &mut decoded)?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // M16 workaround #3 Phase A: validate + record.
                    // Reject an incompatible mid-stream format change
                    // (e.g. H.264 -> VP9) loud; previously dropped
                    // silently. Output `CapsChanged` is still emitted
                    // from decoded geometry at the decode boundary so
                    // the ordering invariant from §3 is preserved.
                    match &c {
                        Caps::CompressedVideo {
                            codec: VideoCodec::H264,
                            ..
                        } => {}
                        _ => return Err(G2gError::CapsMismatch),
                    }
                    self.input_caps = Some(c);
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
            }

            for d in decoded {
                let new_caps = nv12_caps(d.width, d.height);
                if self.last_caps.as_ref() != Some(&new_caps) {
                    // M16 workaround #3 Phase A debug assertion:
                    // decode-time output must overlap the closure's
                    // derivation of the recorded input. See
                    // `ffmpegdec.rs` for the full rationale.
                    #[cfg(debug_assertions)]
                    if let Some(input) = self.input_caps.as_ref() {
                        let expected = derive_output_caps(input);
                        debug_assert!(
                            !expected
                                .intersect(&CapsSet::one(new_caps.clone()))
                                .is_empty(),
                            "mfdecode decode-time output {new_caps:?} inconsistent with derive_output_caps({input:?}) = {expected:?}"
                        );
                    }
                    out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                    self.last_caps = Some(new_caps.clone());
                }
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(d.bytes)),
                    timing: FrameTiming {
                        pts_ns: d.pts_ns,
                        dts_ns: d.pts_ns,
                        duration_ns: 0,
                        capture_ns: d.pts_ns,
                        ..FrameTiming::default()
                    },
                    sequence: self.emitted,
                };
                self.emitted += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }
            Ok(())
        })
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

/// Create the decoder MFT, set the H.264 input type and an NV12 output type,
/// and put it into streaming mode.
fn init_decoder(width: u32, height: u32) -> Result<DecoderState, G2gError> {
    // SAFETY: COM object creation + media-type configuration on the owning
    // thread; all arguments are valid for the duration of each call.
    let transform: IMFTransform = unsafe {
        let transform: IMFTransform =
            CoCreateInstance(&CLSID_MSH264DecoderMFT, None, CLSCTX_INPROC_SERVER).map_err(mf_err)?;

        let input = MFCreateMediaType().map_err(mf_err)?;
        input
            .SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)
            .map_err(mf_err)?;
        input
            .SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264)
            .map_err(mf_err)?;
        if width != 0 && height != 0 {
            input
                .SetUINT64(&MF_MT_FRAME_SIZE, pack_size(width, height))
                .map_err(mf_err)?;
        }
        transform.SetInputType(0, &input, 0).map_err(mf_err)?;
        transform
    };

    let (w, h, stride) = set_nv12_output(&transform)?;
    let out_size = output_buffer_size(&transform, w, h)?;

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
fn pack_nv12(src: &[u8], width: usize, height: usize, stride: usize) -> Result<Box<[u8]>, G2gError> {
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

fn nv12_caps(w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

/// Single source of truth for the decoder's output-side caps derivation.
/// Shared by the `DerivedOutput` constraint closure and the
/// workaround-#3 Phase A debug assertion. The MFT only emits NV12, so
/// there's no output-format choice (unlike `ffmpegdec`'s helper).
fn derive_output_caps(input: &Caps) -> CapsSet {
    match input {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width,
            height,
            framerate,
        } => CapsSet::one(Caps::RawVideo {
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
            nv12_caps(640, 480),
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Any,
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
