//! Native NVDEC H.264 decode element (`nvdec` feature): the decode half of the
//! gst-`nvcodec`-style pair, the counterpart to the native [`crate::nvenc::NvEnc`]
//! encoder. It promotes NVIDIA hardware decode from a `FfmpegVideoDec` backend
//! flag (`Backend::NvdecCuda`, which reaches NVDEC *through* libavcodec's cuvid
//! hwaccel) to a first-class element that drives the NVIDIA Video Codec SDK's
//! NVCUVID API directly, so the decode path no longer depends on libavcodec.
//!
//! `Caps::CompressedVideo{H264}` (Annex-B) in, `Caps::RawVideo{Nv12}` out in CUDA
//! device memory (`MemoryDomain::Cuda`), the zero-copy hwframe domain a downstream
//! `CudaToWgpu` / `CudaGlSink` / `NvEnc` consumes with no PCIe download. With
//! `NvDec -> ... -> NvEnc` both native, the whole H.264 transcode loop stays on
//! the GPU and out of libavcodec.
//!
//! NVCUVID is callback-driven: a *parser* (`cuvidCreateVideoParser`) is fed the
//! elementary stream and synchronously invokes three callbacks from inside
//! `cuvidParseVideoData`, a sequence callback (creates the decoder once the SPS is
//! parsed), a decode callback (`cuvidDecodePicture`), and a display callback
//! (a picture is ready in display order). Because the display callback cannot
//! `await`, it maps the surface (`cuvidMapVideoFrame64`) and pushes a ready frame
//! onto a queue; `process` drains the queue and emits downstream after the parse
//! call returns. The callbacks reach element state through a `*mut DecoderState`
//! passed as the parser's user-data; that pointer targets a heap `Box` so it
//! stays valid even as the runner moves the element between worker threads.
//!
//! Bindings are hand-rolled FFI linking `libnvcuvid` + `libcuda` directly (no
//! `cudarc`), matching [`crate::nvenc`] and the `cuda` module. NVCUVID exports
//! real symbols (unlike NVENC's `CreateInstance` dispatch table), so the calls
//! are plain `extern "C"`. The version-free structs are transcribed `#[repr(C)]`
//! with compile-time size assertions checked against the installed `cuviddec.h` /
//! `nvcuvid.h` (field offsets verified with `offsetof`); the per-picture
//! `CUVIDPICPARAMS` is opaque (the parser fills it and we pass the pointer
//! straight to `cuvidDecodePicture`).
//!
//! Each mapped output frame carries a [`CudaKeepAlive`] that `cuvidUnmapVideoFrame64`s
//! on drop, and an `Arc` to the decoder so the decoder / context outlive any frame
//! still in flight. The element owns its own CUDA context (created at configure).
//!
//! Deferred (v1): mid-stream resolution change (decoder reconfigure), HEVC / other
//! codecs via the codec enum, 10-bit output, and a `display_delay` knob (fixed at
//! a low-latency 1 today).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use g2g_core::memory::{CudaKeepAlive, DomainSet, MemoryDomainKind, OwnedCudaBuffer};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, RawVideoFormat, Rate, SystemSlice, VideoCodec,
};

/// Number of decode surfaces the parser cycles through. Also the cap on the
/// decoder's surface pool; the sequence callback clamps the stream's minimum into
/// this. Bigger = more reorder / in-flight headroom at a memory cost.
const NUM_DECODE_SURFACES: u32 = 20;
/// Max output surfaces mapped at once (frames held downstream before release).
const NUM_OUTPUT_SURFACES: u32 = 8;

/// Native NVDEC H.264 decoder. Annex-B in, CUDA NV12 out. See the module docs.
pub struct NvDec {
    width: u32,
    height: u32,
    framerate: Rate,
    /// Our CUDA context (created at configure), shared into every output frame's
    /// keep-alive so unmap can run in it.
    context: u64,
    /// `CUvideoparser`; created at configure, destroyed on drop.
    parser: *mut core::ffi::c_void,
    /// Callback shuttle on the heap so its address is stable as the runner moves
    /// the element; the parser holds a raw pointer to it as user-data.
    state: Box<DecoderState>,
    emitted: u64,
    caps_sent: bool,
    configured: bool,
    /// The memory domain the negotiation settled this decoder's output on (M352).
    /// `Cuda` keeps frames device-resident (zero-copy, the default); `System`
    /// downloads each decoded surface to host memory before emitting. Chosen in
    /// `configure_allocation` by reconciling the downstream proposal against
    /// [`Self::OUTPUT_DOMAINS`].
    out_domain: MemoryDomainKind,
}

/// State the parser callbacks read and write (decoder handle, geometry, the ready
/// queue, the first error). Lives in a `Box` owned by [`NvDec`]; the parser is
/// given a raw pointer to it.
struct DecoderState {
    context: u64,
    ctx_lock: *mut core::ffi::c_void,
    /// The `cudaVideoCodec` the parser / decoder were created for (H.264 or HEVC),
    /// from the negotiated input caps.
    codec_cuvid: i32,
    /// `CUvideodecoder`, created in the sequence callback. Raw copy for the decode
    /// / display callbacks; ownership / destruction is the `Arc`'s.
    decoder: *mut core::ffi::c_void,
    /// Shared decoder owner; cloned into each frame keep-alive so the decoder and
    /// context outlive frames still referenced downstream.
    decoder_owner: Option<Arc<CuvidDecoder>>,
    /// Display geometry (the cropped output dims). Chroma offset uses `target_height`.
    target_width: u32,
    target_height: u32,
    /// Frames mapped and ready to emit (drained by `process` after each parse).
    ready: Vec<ReadyFrame>,
    /// First error raised inside a callback, surfaced after the parse returns.
    error: Option<G2gError>,
}

/// A decoded, mapped NV12 surface ready to hand downstream.
struct ReadyFrame {
    buffer: OwnedCudaBuffer,
    pts_ns: u64,
}

// SAFETY: `NvDec` holds raw NVCUVID/CUDA handles and a `Box<DecoderState>` with
// raw pointers. The runner moves the element between worker tasks but drives it
// through `&mut self` only (never concurrently), so the handles are owned and
// moved, never aliased, the same contract as `FfmpegVideoDec` / `NvEnc`.
unsafe impl Send for NvDec {}

impl core::fmt::Debug for NvDec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NvDec")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("open", &!self.parser.is_null())
            .field("emitted", &self.emitted)
            .field("configured", &self.configured)
            .finish()
    }
}

impl Default for NvDec {
    fn default() -> Self {
        Self::new()
    }
}

impl NvDec {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            framerate: Rate::Any,
            context: 0,
            parser: core::ptr::null_mut(),
            state: Box::new(DecoderState {
                context: 0,
                ctx_lock: core::ptr::null_mut(),
                codec_cuvid: ffi::CUDA_VIDEO_CODEC_H264,
                decoder: core::ptr::null_mut(),
                decoder_owner: None,
                target_width: 0,
                target_height: 0,
                ready: Vec::new(),
                error: None,
            }),
            emitted: 0,
            caps_sent: false,
            configured: false,
            out_domain: MemoryDomainKind::Cuda,
        }
    }

    /// Domains this decoder can emit (M352): `Cuda` (device-resident, zero-copy)
    /// or `System` (downloaded). The producer-capability half of the M351
    /// two-sided allocation-domain negotiation.
    const OUTPUT_DOMAINS: DomainSet =
        DomainSet::only(MemoryDomainKind::Cuda).with(MemoryDomainKind::System);

    /// Frames decoded and emitted so far.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// Accepted input codecs: H.264 and HEVC (NVCUVID decodes both; AV1 needs an
    /// Ampere+ NVDEC and is a follow-up).
    fn input_codecs() -> [VideoCodec; 2] {
        [VideoCodec::H264, VideoCodec::H265]
    }

    /// Open-geometry input caps, one alternative per accepted codec.
    fn input_caps_set() -> CapsSet {
        CapsSet::from_alternatives(
            Self::input_codecs()
                .into_iter()
                .map(|codec| Caps::CompressedVideo {
                    codec,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                })
                .collect(),
        )
    }

    /// The `cudaVideoCodec` value for a supported input codec.
    fn cuvid_codec(codec: VideoCodec) -> Option<i32> {
        match codec {
            VideoCodec::H264 => Some(ffi::CUDA_VIDEO_CODEC_H264),
            VideoCodec::H265 => Some(ffi::CUDA_VIDEO_CODEC_HEVC),
            _ => None,
        }
    }

    fn output_caps(&self) -> Caps {
        // Actual decoded display geometry once known, else the negotiated dims.
        let (w, h) = if self.state.target_width != 0 {
            (self.state.target_width, self.state.target_height)
        } else {
            (self.width, self.height)
        };
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: self.framerate.clone(),
        }
    }

    /// Bring up the CUDA context, the context lock, and the NVCUVID parser. The
    /// decoder is created lazily in the sequence callback (it needs the parsed SPS
    /// geometry). Fails loud if NVDEC / the driver is unavailable.
    fn open(&mut self) -> Result<(), G2gError> {
        // SAFETY: standard CUDA driver bring-up; each result is checked and we
        // bail before using a handle on failure.
        let context = unsafe {
            cuchk(ffi::cu_init(0))?;
            let mut dev = 0i32;
            cuchk(ffi::cu_device_get(&mut dev, 0))?;
            let mut ctx: *mut core::ffi::c_void = core::ptr::null_mut();
            cuchk(ffi::cu_ctx_create(&mut ctx, 0, dev))?;
            if ctx.is_null() {
                return Err(hw());
            }
            ctx as u64
        };
        self.context = context;
        self.state.context = context;

        let _ctx = ContextGuard::push(context)?;
        // SAFETY: valid context; on success `ctx_lock` receives the lock handle.
        let ctx_lock = unsafe {
            let mut lock: *mut core::ffi::c_void = core::ptr::null_mut();
            cuchk(ffi::cuvid_ctx_lock_create(&mut lock, context as *mut core::ffi::c_void))?;
            lock
        };
        self.state.ctx_lock = ctx_lock;

        // Create the parser, pointing it at the heap `DecoderState` as user-data.
        let user = self.state.as_mut() as *mut DecoderState as *mut core::ffi::c_void;
        // SAFETY: the NVCUVID param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then fill.
        let mut params: ffi::ParserParams = unsafe { core::mem::zeroed() };
        params.codec_type = self.state.codec_cuvid;
        params.max_num_decode_surfaces = NUM_DECODE_SURFACES;
        // Low latency: a single-frame display delay (recommended 2..4 for higher
        // throughput / heavier reorder; 1 keeps glass-to-glass tight).
        params.max_display_delay = 1;
        params.user_data = user;
        params.pfn_sequence_callback = Some(handle_sequence);
        params.pfn_decode_picture = Some(handle_decode);
        params.pfn_display_picture = Some(handle_display);
        let mut parser: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: `params` is fully initialized; on success `parser` receives a
        // valid handle that retains the `user` pointer (stable: it is the boxed
        // state, which does not move when `self` moves).
        cuchk(unsafe { ffi::cuvid_create_video_parser(&mut parser, &mut params) })?;
        self.parser = parser;
        Ok(())
    }

    /// Feed one Annex-B access unit (or an EOS flush) to the parser, then drain
    /// whatever frames the display callback produced.
    fn parse(&mut self, payload: &[u8], pts_ns: u64, eos: bool) -> Result<Vec<ReadyFrame>, G2gError> {
        let _ctx = ContextGuard::push(self.context)?;
        // SAFETY: the NVCUVID param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then fill.
        let mut pkt: ffi::SourceDataPacket = unsafe { core::mem::zeroed() };
        if eos {
            pkt.flags = ffi::CUVID_PKT_ENDOFSTREAM;
        } else {
            pkt.flags = ffi::CUVID_PKT_TIMESTAMP;
            pkt.payload_size = payload.len() as u64;
            pkt.payload = payload.as_ptr();
            pkt.timestamp = pts_ns as i64;
        }
        // SAFETY: valid parser; `pkt` describes `payload` (or an empty EOS packet)
        // and is only read for the duration of the call. The callbacks run
        // synchronously here, with `self.context` current, and route through the
        // user-data pointer to `self.state`.
        let rc = unsafe { ffi::cuvid_parse_video_data(self.parser, &mut pkt) };
        // A callback error takes precedence over the parse return code.
        if let Some(e) = self.state.error.take() {
            return Err(e);
        }
        cuchk(rc)?;
        Ok(core::mem::take(&mut self.state.ready))
    }

    async fn emit(
        &mut self,
        frames: Vec<ReadyFrame>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        if frames.is_empty() {
            return Ok(());
        }
        if !self.caps_sent {
            out.push(PipelinePacket::CapsChanged(self.output_caps())).await?;
            self.caps_sent = true;
        }
        for f in frames {
            // M352: keep the surface on the GPU (zero-copy) unless negotiation
            // settled this decoder's output on System, in which case download it
            // device->host before emitting.
            let domain = if self.out_domain == MemoryDomainKind::System {
                // SAFETY: `f.buffer`'s plane pointers are valid CUDA device memory
                // in its context, pinned by the buffer's keep-alive owner for the
                // duration of this copy.
                let bytes = unsafe { crate::cuda::download_nv12(&f.buffer)? };
                MemoryDomain::System(SystemSlice::from_boxed(bytes))
            } else {
                MemoryDomain::Cuda(f.buffer)
            };
            let frame = g2g_core::frame::Frame::new(
                domain,
                g2g_core::FrameTiming { pts_ns: f.pts_ns, dts_ns: f.pts_ns, ..Default::default() },
                self.emitted,
            );
            self.emitted += 1;
            out.push(PipelinePacket::DataFrame(frame)).await?;
        }
        Ok(())
    }
}

impl Drop for NvDec {
    fn drop(&mut self) {
        // Destroy the parser first so no further callback can fire, then let the
        // boxed state drop (releasing the decoder `Arc`; the decoder / ctx-lock /
        // context are torn down when the last frame referencing them is gone).
        if !self.parser.is_null() {
            // SAFETY: `parser` was created in `open` and is destroyed once.
            unsafe {
                let _ = ffi::cuvid_destroy_video_parser(self.parser);
            }
            self.parser = core::ptr::null_mut();
        }
        // The ctx-lock and context are owned by `CuvidDecoder`, which frees them
        // (via the `Arc`) once the last frame referencing it drops. But the
        // decoder is created lazily on the first decoded sequence: if we were
        // configured but never decoded a picture, `decoder_owner` is `None` and
        // nothing else owns them. With the parser already destroyed no callback
        // can still create one, so free them here, mirroring `CuvidDecoder::drop`.
        if self.state.decoder_owner.is_none() {
            // SAFETY: created together in `open`; destroyed once, here, only when
            // no `CuvidDecoder` took ownership. Best-effort; failures unactionable.
            unsafe {
                if !self.state.ctx_lock.is_null() {
                    let _ = ffi::cuvid_ctx_lock_destroy(self.state.ctx_lock);
                    self.state.ctx_lock = core::ptr::null_mut();
                }
                if self.state.context != 0 {
                    let _ = ffi::cu_ctx_destroy(self.state.context as *mut core::ffi::c_void);
                    self.state.context = 0;
                }
            }
        }
    }
}

/// Owns the `CUvideodecoder`, its context lock, and the CUDA context, tearing
/// them down (in that order) when the last reference, the decoder itself or any
/// frame keep-alive still in flight, drops. Boxed as the [`CudaKeepAlive`] of
/// every emitted frame.
struct CuvidDecoder {
    decoder: *mut core::ffi::c_void,
    ctx_lock: *mut core::ffi::c_void,
    context: u64,
}

// SAFETY: the handles are owned and inert. `Send` + `Sync` let an output frame
// cross worker threads and fan out through a tee (M213): NVCUVID serializes
// decoder access through the context lock, and unmap-on-drop is the only
// operation a shared keep-alive performs, so concurrent read-only sharing of the
// decoded surface is sound, the same contract as `FfmpegVideoDec`'s frame owner.
unsafe impl Send for CuvidDecoder {}
// SAFETY: as for `Send` above, concurrent read-only sharing of the decoded
// surface is sound (access serialized by the context lock).
unsafe impl Sync for CuvidDecoder {}

impl core::fmt::Debug for CuvidDecoder {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("CuvidDecoder(<CUvideodecoder>)")
    }
}

impl Drop for CuvidDecoder {
    fn drop(&mut self) {
        // SAFETY: all three handles were created together and are destroyed once,
        // here, after every mapped frame has been unmapped (frames hold an `Arc`
        // to this, so this drop runs only once none remain). Context destroyed
        // last. Best-effort; failures are unactionable.
        unsafe {
            if !self.decoder.is_null() {
                let _ = ffi::cuvid_destroy_decoder(self.decoder);
            }
            if !self.ctx_lock.is_null() {
                let _ = ffi::cuvid_ctx_lock_destroy(self.ctx_lock);
            }
            if self.context != 0 {
                let _ = ffi::cu_ctx_destroy(self.context as *mut core::ffi::c_void);
            }
        }
    }
}

/// Unmaps one NVDEC output surface on drop, releasing it to the decoder's pool.
/// Holds an `Arc` to the decoder so it (and the context unmap runs in) outlive
/// the frame.
struct CuvidMappedFrame {
    owner: Arc<CuvidDecoder>,
    dev_ptr: u64,
}

// SAFETY: see `CuvidDecoder`; this only unmaps (serialized by the context lock)
// and pins the decoder via the `Arc`.
unsafe impl Send for CuvidMappedFrame {}
// SAFETY: as for `Send` above; only unmaps (serialized) and pins the decoder.
unsafe impl Sync for CuvidMappedFrame {}

impl core::fmt::Debug for CuvidMappedFrame {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("CuvidMappedFrame")
    }
}

impl CudaKeepAlive for CuvidMappedFrame {}

impl Drop for CuvidMappedFrame {
    fn drop(&mut self) {
        // SAFETY: `dev_ptr` was returned by `cuvidMapVideoFrame64` on
        // `owner.decoder` and is unmapped once. Push the context first so the
        // unmap runs in it; best-effort.
        unsafe {
            let _ = ffi::cu_ctx_push_current(self.owner.context as *mut core::ffi::c_void);
            let _ = ffi::cuvid_unmap_video_frame(self.owner.decoder, self.dev_ptr);
            let mut popped = core::ptr::null_mut();
            let _ = ffi::cu_ctx_pop_current(&mut popped);
        }
    }
}

// --- Parser callbacks (called synchronously from inside cuvidParseVideoData) ---

/// Record the first callback error on the state and return the failure code the
/// callback ABI expects (0 = fail).
fn fail(state: &mut DecoderState, err: G2gError) -> i32 {
    if state.error.is_none() {
        state.error = Some(err);
    }
    0
}

/// Sequence callback: the parser hands us the decoded stream format. Create the
/// decoder (once) and report the surface count back to the parser.
extern "C" fn handle_sequence(user: *mut core::ffi::c_void, fmt: *mut ffi::VideoFormat) -> i32 {
    // SAFETY: `user` is the boxed `DecoderState` pointer set in `open`; `fmt` is a
    // valid format struct for the duration of the callback.
    let state = unsafe { &mut *(user as *mut DecoderState) };
    // SAFETY: `fmt` is the valid format struct the parser passes for this callback.
    let f = unsafe { &*fmt };

    let num_surfaces = (f.min_num_decode_surfaces as u32).clamp(1, NUM_DECODE_SURFACES);
    // Display (cropped) geometry, rounded up to even for NV12.
    let disp_w = (f.display_area.right - f.display_area.left).max(0) as u32;
    let disp_h = (f.display_area.bottom - f.display_area.top).max(0) as u32;
    let target_w = if disp_w != 0 { disp_w } else { f.coded_width };
    let target_h = if disp_h != 0 { disp_h } else { f.coded_height };
    let target_w = (target_w + 1) & !1;
    let target_h = (target_h + 1) & !1;

    if !state.decoder.is_null() {
        // Mid-stream format change: reconfigure is deferred (v1). Keep the
        // existing decoder if the geometry matches, else fail loud.
        if state.target_width == target_w && state.target_height == target_h {
            return num_surfaces as i32;
        }
        return fail(state, hw());
    }

    // SAFETY: the NVCUVID param structs are plain old data (ints, pointers,
    // reserved arrays); all-zero is a valid initial state we then fill.
    let mut info: ffi::DecodeCreateInfo = unsafe { core::mem::zeroed() };
    info.width = f.coded_width as u64;
    info.height = f.coded_height as u64;
    info.num_decode_surfaces = num_surfaces as u64;
    info.codec_type = state.codec_cuvid;
    info.chroma_format = f.chroma_format;
    info.creation_flags = ffi::CUDA_VIDEO_CREATE_PREFER_CUVID;
    info.bit_depth_minus8 = f.bit_depth_luma_minus8 as u64;
    info.max_width = f.coded_width as u64;
    info.max_height = f.coded_height as u64;
    info.display_area_left = f.display_area.left as i16;
    info.display_area_top = f.display_area.top as i16;
    info.display_area_right = f.display_area.right as i16;
    info.display_area_bottom = f.display_area.bottom as i16;
    info.output_format = ffi::CUDA_VIDEO_SURFACE_FORMAT_NV12;
    info.deinterlace_mode = ffi::CUDA_VIDEO_DEINTERLACE_WEAVE;
    info.target_width = target_w as u64;
    info.target_height = target_h as u64;
    info.num_output_surfaces = NUM_OUTPUT_SURFACES as u64;
    info.vid_lock = state.ctx_lock;

    let mut decoder: *mut core::ffi::c_void = core::ptr::null_mut();
    // SAFETY: `info` is fully initialized; on success `decoder` is a valid handle.
    let rc = unsafe { ffi::cuvid_create_decoder(&mut decoder, &mut info) };
    if rc != 0 || decoder.is_null() {
        return fail(state, G2gError::Hardware(HardwareError::Cuda(rc)));
    }
    state.decoder = decoder;
    state.target_width = target_w;
    state.target_height = target_h;
    state.decoder_owner = Some(Arc::new(CuvidDecoder {
        decoder,
        ctx_lock: state.ctx_lock,
        context: state.context,
    }));
    num_surfaces as i32
}

/// Decode callback: submit the parser-filled picture params to the hardware.
extern "C" fn handle_decode(user: *mut core::ffi::c_void, pic: *mut core::ffi::c_void) -> i32 {
    // SAFETY: `user` is the boxed state; `pic` is the parser's `CUVIDPICPARAMS`,
    // opaque to us and passed straight through to the decoder.
    let state = unsafe { &mut *(user as *mut DecoderState) };
    if state.decoder.is_null() {
        return fail(state, hw());
    }
    // SAFETY: valid decoder + parser-owned picture params.
    let rc = unsafe { ffi::cuvid_decode_picture(state.decoder, pic) };
    if rc != 0 {
        return fail(state, G2gError::Hardware(HardwareError::Cuda(rc)));
    }
    1
}

/// Display callback: a picture is ready in display order. Map it to a device
/// pointer, wrap it as an `OwnedCudaBuffer`, and queue it for `process` to emit.
extern "C" fn handle_display(user: *mut core::ffi::c_void, disp: *mut ffi::ParserDispInfo) -> i32 {
    // SAFETY: `user` is the boxed state; `disp` is valid for the callback.
    let state = unsafe { &mut *(user as *mut DecoderState) };
    // SAFETY: `disp` is the valid display-info struct the parser passes here.
    let d = unsafe { &*disp };
    let Some(owner) = state.decoder_owner.clone() else {
        return fail(state, hw());
    };

    // SAFETY: the NVCUVID param structs are plain old data (ints, pointers,
    // reserved arrays); all-zero is a valid initial state we then fill.
    let mut proc: ffi::ProcParams = unsafe { core::mem::zeroed() };
    proc.progressive_frame = d.progressive_frame;
    proc.top_field_first = d.top_field_first;
    proc.second_field = 0;
    proc.unpaired_field = (d.repeat_first_field < 0) as i32;

    let mut dev_ptr: u64 = 0;
    let mut pitch: u32 = 0;
    // SAFETY: valid decoder + picture index from the parser; on success `dev_ptr`
    // / `pitch` describe a mapped NV12 surface valid until unmap.
    let rc = unsafe {
        ffi::cuvid_map_video_frame(
            owner.decoder,
            d.picture_index,
            &mut dev_ptr,
            &mut pitch,
            &mut proc,
        )
    };
    if rc != 0 || dev_ptr == 0 {
        return fail(state, G2gError::Hardware(HardwareError::Cuda(rc)));
    }

    // NV12: chroma plane follows luma at pitch * target_height.
    let chroma_ptr = dev_ptr + (pitch as u64) * (state.target_height as u64);
    let buffer = OwnedCudaBuffer::new(
        dev_ptr,
        chroma_ptr,
        pitch,
        pitch,
        state.target_width,
        state.target_height,
        state.context,
        Arc::new(CuvidMappedFrame { owner, dev_ptr }),
    );
    state.ready.push(ReadyFrame { buffer, pts_ns: d.timestamp as u64 });
    1
}

/// Map a CUDA / CUVID result to a `Result`, carrying the code on failure.
fn cuchk(code: i32) -> Result<(), G2gError> {
    if code == 0 {
        Ok(())
    } else {
        Err(G2gError::Hardware(HardwareError::Cuda(code)))
    }
}

/// Shorthand for the generic hardware error.
fn hw() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Pushes a CUDA context current for a scope and pops it on drop.
struct ContextGuard;

impl ContextGuard {
    fn push(context: u64) -> Result<Self, G2gError> {
        // SAFETY: `context` is the valid context created in `open`.
        let code = unsafe { ffi::cu_ctx_push_current(context as *mut core::ffi::c_void) };
        if code == 0 {
            Ok(ContextGuard)
        } else {
            Err(G2gError::Hardware(HardwareError::Cuda(code)))
        }
    }
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        let mut popped: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: balances the push in `ContextGuard::push`; best-effort.
        unsafe {
            let _ = ffi::cu_ctx_pop_current(&mut popped);
        }
    }
}

impl AsyncElement for NvDec {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for alt in Self::input_caps_set().alternatives() {
            if let Ok(narrowed) = upstream_caps.intersect(alt) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    /// Native `DerivedOutput`: H.264 or HEVC (any geometry) in, NV12 at the same
    /// dims and framerate out. Any other input yields an empty set, rejected at
    /// solve. The runtime `CapsChanged` carries the actual decoded (cropped) dims.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo {
                codec: VideoCodec::H264 | VideoCodec::H265,
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
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::CompressedVideo { codec, width, height, framerate } = absolute_caps else {
            return Err(G2gError::CapsMismatch);
        };
        // Pick the NVCUVID codec before opening the parser; reject unsupported ones.
        self.state.codec_cuvid = Self::cuvid_codec(*codec).ok_or(G2gError::CapsMismatch)?;
        if let Dim::Fixed(w) = width {
            self.width = *w;
        }
        if let Dim::Fixed(h) = height {
            self.height = *h;
        }
        self.framerate = framerate.clone();
        self.open()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    /// NVDEC emits NV12 in CUDA device memory (the zero-copy hwframe domain),
    /// so a downstream link from this element is a GPU link (M285). This is the
    /// *preferred* domain; [`output_domains`](Self::output_domains) widens it to
    /// the full set the decoder can satisfy.
    fn output_memory(&self) -> g2g_core::memory::MemoryDomainKind {
        g2g_core::memory::MemoryDomainKind::Cuda
    }

    /// M352: the decoder can keep frames on the GPU *or* download them to System,
    /// so it advertises both. The runner's allocation cascade narrows this against
    /// the downstream consumers' accepted domains (a tee join over the branches),
    /// and [`configure_allocation`](Self::configure_allocation) settles the choice.
    fn output_domains(&self) -> DomainSet {
        Self::OUTPUT_DOMAINS
    }

    /// Receive the (possibly tee-joined) downstream allocation proposal and settle
    /// this decoder's output domain (M352). Reconciles the consumer's accepted
    /// domains against what NVDEC can emit (`resolve_for_producer`, the
    /// producer-side of the M351 negotiation): a CUDA-capable consumer keeps the
    /// frame device-resident (zero-copy), a System-only consumer makes the decoder
    /// download. No reconcilable domain leaves the default (`Cuda`) in place; the
    /// consumer then rejects the domain at `process` as it would today.
    fn configure_allocation(&mut self, params: &AllocationParams) {
        if let Ok(resolved) = params.resolve_for_producer(Self::OUTPUT_DOMAINS) {
            self.out_domain = resolved.domain;
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "NVDEC H.264 / HEVC decoder",
            "Codec/Decoder/Video/Hardware",
            "Zero-copy H.264 / HEVC decode to CUDA NV12 surfaces via the NVIDIA Video Codec SDK (NVCUVID)",
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
                    let frames = self.parse(slice.as_slice(), frame.timing.pts_ns, false)?;
                    self.emit(frames, out).await?;
                }
                PipelinePacket::Eos => {
                    // Flush the parser's display queue; the runner forwards EOS.
                    let frames = self.parse(&[], 0, true)?;
                    self.emit(frames, out).await?;
                }
                PipelinePacket::CapsChanged(_) => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

impl PadTemplates for NvDec {
    fn pad_templates() -> Vec<PadTemplate> {
        let out = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(Self::input_caps_set()),
            PadTemplate::source(CapsSet::one(out)),
        ])
    }
}

/// Thin hand-rolled FFI for the NVCUVID decode API (`cuviddec.h` / `nvcuvid.h`)
/// plus the `libcuda` context calls. Only the surface this element uses is
/// transcribed; the per-picture `CUVIDPICPARAMS` is opaque (passed straight
/// through). Every `#[repr(C)]` struct carries a compile-time size assertion
/// checked against the installed headers; field offsets are correct by faithful
/// transcription (verified with `offsetof`). `unsigned long` is 8 bytes here.
// The FFI items are `pub` for 1:1 correspondence with the C headers even though
// this module is private (only `super` uses them), the same shape as the
// `crate::cuda` / `crate::nvenc` FFI blocks.
#[allow(non_upper_case_globals, unreachable_pub)]
mod ffi {
    use core::ffi::c_void;

    // Codec / format enum values (cuviddec.h).
    pub const CUDA_VIDEO_CODEC_H264: i32 = 4;
    pub const CUDA_VIDEO_CODEC_HEVC: i32 = 8;
    pub const CUDA_VIDEO_SURFACE_FORMAT_NV12: i32 = 0;
    pub const CUDA_VIDEO_DEINTERLACE_WEAVE: i32 = 0;
    pub const CUDA_VIDEO_CREATE_PREFER_CUVID: u64 = 0x04;
    // Packet flags (nvcuvid.h).
    pub const CUVID_PKT_ENDOFSTREAM: u64 = 0x01;
    pub const CUVID_PKT_TIMESTAMP: u64 = 0x02;

    /// `CUVIDSOURCEDATAPACKET` (32 bytes). `unsigned long` flags / payload_size.
    #[repr(C)]
    pub struct SourceDataPacket {
        pub flags: u64,
        pub payload_size: u64,
        pub payload: *const u8,
        pub timestamp: i64,
    }
    const _: () = assert!(core::mem::size_of::<SourceDataPacket>() == 32);

    /// `CUVIDPARSERDISPINFO` (24 bytes).
    #[repr(C)]
    pub struct ParserDispInfo {
        pub picture_index: i32,
        pub progressive_frame: i32,
        pub top_field_first: i32,
        pub repeat_first_field: i32,
        pub timestamp: i64,
    }
    const _: () = assert!(core::mem::size_of::<ParserDispInfo>() == 24);

    pub type SequenceCb = extern "C" fn(*mut c_void, *mut VideoFormat) -> i32;
    pub type DecodeCb = extern "C" fn(*mut c_void, *mut c_void) -> i32;
    pub type DisplayCb = extern "C" fn(*mut c_void, *mut ParserDispInfo) -> i32;

    /// `CUVIDPARSERPARAMS` (136 bytes). The `bAnnexb:1/uReserved:31` bitfield is
    /// one `u32` (`annexb_bits`); unused callbacks / reserved pointers are null.
    #[repr(C)]
    pub struct ParserParams {
        pub codec_type: i32,
        pub max_num_decode_surfaces: u32,
        pub clock_rate: u32,
        pub error_threshold: u32,
        pub max_display_delay: u32,
        pub annexb_bits: u32,
        pub reserved1: [u32; 4],
        pub user_data: *mut c_void,
        pub pfn_sequence_callback: Option<SequenceCb>,
        pub pfn_decode_picture: Option<DecodeCb>,
        pub pfn_display_picture: Option<DisplayCb>,
        pub pfn_get_operating_point: *mut c_void,
        pub pfn_get_sei_msg: *mut c_void,
        pub reserved2: [*mut c_void; 5],
        pub ext_video_info: *mut c_void,
    }
    const _: () = assert!(core::mem::size_of::<ParserParams>() == 136);

    /// `CUVIDEOFORMAT` (64 bytes). The parser fills it; we read geometry / chroma.
    #[repr(C)]
    pub struct VideoFormat {
        pub codec: i32,
        pub frame_rate_numerator: u32,
        pub frame_rate_denominator: u32,
        pub progressive_sequence: u8,
        pub bit_depth_luma_minus8: u8,
        pub bit_depth_chroma_minus8: u8,
        pub min_num_decode_surfaces: u8,
        pub coded_width: u32,
        pub coded_height: u32,
        pub display_area: Rect,
        pub chroma_format: i32,
        pub bitrate: u32,
        pub display_aspect_ratio_x: i32,
        pub display_aspect_ratio_y: i32,
        pub video_signal_description: [u8; 4],
        pub seqhdr_data_length: u32,
    }
    const _: () = assert!(core::mem::size_of::<VideoFormat>() == 64);

    /// `int` display rectangle inside `CUVIDEOFORMAT`.
    #[repr(C)]
    pub struct Rect {
        pub left: i32,
        pub top: i32,
        pub right: i32,
        pub bottom: i32,
    }

    /// `CUVIDDECODECREATEINFO` (176 bytes). `unsigned long` fields are 8 bytes;
    /// the two `short` rectangles are flattened into named `i16` fields.
    #[repr(C)]
    pub struct DecodeCreateInfo {
        pub width: u64,
        pub height: u64,
        pub num_decode_surfaces: u64,
        pub codec_type: i32,
        pub chroma_format: i32,
        pub creation_flags: u64,
        pub bit_depth_minus8: u64,
        pub intra_decode_only: u64,
        pub max_width: u64,
        pub max_height: u64,
        pub reserved1: u64,
        pub display_area_left: i16,
        pub display_area_top: i16,
        pub display_area_right: i16,
        pub display_area_bottom: i16,
        pub output_format: i32,
        pub deinterlace_mode: i32,
        pub target_width: u64,
        pub target_height: u64,
        pub num_output_surfaces: u64,
        pub vid_lock: *mut c_void,
        pub target_rect_left: i16,
        pub target_rect_top: i16,
        pub target_rect_right: i16,
        pub target_rect_bottom: i16,
        pub enable_histogram: u64,
        pub reserved2: [u64; 4],
    }
    const _: () = assert!(core::mem::size_of::<DecodeCreateInfo>() == 176);

    /// `CUVIDPROCPARAMS` (264 bytes). We set the field/progressive flags; the rest
    /// (raw-YUV I/O, stream, reserved) stays zero.
    #[repr(C)]
    pub struct ProcParams {
        pub progressive_frame: i32,
        pub second_field: i32,
        pub top_field_first: i32,
        pub unpaired_field: i32,
        pub reserved_flags: u32,
        pub reserved_zero: u32,
        pub raw_input_dptr: u64,
        pub raw_input_pitch: u32,
        pub raw_input_format: u32,
        pub raw_output_dptr: u64,
        pub raw_output_pitch: u32,
        pub reserved1: u32,
        pub output_stream: *mut c_void,
        pub reserved: [u32; 46],
        pub histogram_dptr: *mut u64,
        pub reserved2: [*mut c_void; 1],
    }
    const _: () = assert!(core::mem::size_of::<ProcParams>() == 264);

    // NVCUVID exports plain symbols; alias them to snake_case via link_name.
    #[link(name = "nvcuvid")]
    extern "C" {
        #[link_name = "cuvidCreateVideoParser"]
        pub fn cuvid_create_video_parser(
            parser: *mut *mut c_void,
            params: *mut ParserParams,
        ) -> i32;
        #[link_name = "cuvidParseVideoData"]
        pub fn cuvid_parse_video_data(parser: *mut c_void, pkt: *mut SourceDataPacket) -> i32;
        #[link_name = "cuvidDestroyVideoParser"]
        pub fn cuvid_destroy_video_parser(parser: *mut c_void) -> i32;
        #[link_name = "cuvidCreateDecoder"]
        pub fn cuvid_create_decoder(decoder: *mut *mut c_void, info: *mut DecodeCreateInfo) -> i32;
        #[link_name = "cuvidDestroyDecoder"]
        pub fn cuvid_destroy_decoder(decoder: *mut c_void) -> i32;
        #[link_name = "cuvidDecodePicture"]
        pub fn cuvid_decode_picture(decoder: *mut c_void, pic: *mut c_void) -> i32;
        #[link_name = "cuvidMapVideoFrame64"]
        pub fn cuvid_map_video_frame(
            decoder: *mut c_void,
            pic_idx: i32,
            dev_ptr: *mut u64,
            pitch: *mut u32,
            proc: *mut ProcParams,
        ) -> i32;
        #[link_name = "cuvidUnmapVideoFrame64"]
        pub fn cuvid_unmap_video_frame(decoder: *mut c_void, dev_ptr: u64) -> i32;
        #[link_name = "cuvidCtxLockCreate"]
        pub fn cuvid_ctx_lock_create(lock: *mut *mut c_void, ctx: *mut c_void) -> i32;
        #[link_name = "cuvidCtxLockDestroy"]
        pub fn cuvid_ctx_lock_destroy(lock: *mut c_void) -> i32;
    }

    #[link(name = "cuda")]
    extern "C" {
        #[link_name = "cuInit"]
        pub fn cu_init(flags: u32) -> i32;
        #[link_name = "cuDeviceGet"]
        pub fn cu_device_get(dev: *mut i32, ordinal: i32) -> i32;
        #[link_name = "cuCtxCreate_v2"]
        pub fn cu_ctx_create(pctx: *mut *mut c_void, flags: u32, dev: i32) -> i32;
        #[link_name = "cuCtxDestroy_v2"]
        pub fn cu_ctx_destroy(ctx: *mut c_void) -> i32;
        #[link_name = "cuCtxPushCurrent_v2"]
        pub fn cu_ctx_push_current(ctx: *mut c_void) -> i32;
        #[link_name = "cuCtxPopCurrent_v2"]
        pub fn cu_ctx_pop_current(pctx: *mut *mut c_void) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h264(w: u32, h: u32) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    // --- Pure-logic coverage (no GPU): caps + struct layout ---

    #[test]
    fn caps_constraint_is_h264_to_nv12() {
        let d = NvDec::new();
        let CapsConstraint::DerivedOutput(derive) = d.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = derive(&h264(1920, 1080));
        assert!(out.accepts(&Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
        }));
        // Non-H.264 (e.g. AV1) yields an empty set, rejected at solve.
        let av1 = Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert!(derive(&av1).alternatives().is_empty());
    }

    #[test]
    fn output_caps_use_decoded_dims_when_known() {
        let mut d = NvDec::new();
        d.width = 1920;
        d.height = 1080;
        d.framerate = Rate::Fixed(30 << 16);
        // Before decode: negotiated dims.
        assert!(matches!(
            d.output_caps(),
            Caps::RawVideo { width: Dim::Fixed(1920), height: Dim::Fixed(1080), .. }
        ));
        // After the sequence callback learns the real (cropped) display dims.
        d.state.target_width = 1280;
        d.state.target_height = 720;
        assert!(matches!(
            d.output_caps(),
            Caps::RawVideo { width: Dim::Fixed(1280), height: Dim::Fixed(720), .. }
        ));
    }

    // --- On-hardware round trip (RTX 3060): encode an NV12 CUDA surface with the
    // native NvEnc, then decode the Annex-B back through NvDec to NV12 in CUDA
    // device memory. Exercises the whole native loop; skips with no GPU. Needs the
    // `nvenc` feature for the encode leg. ---

    #[cfg(feature = "nvenc")]
    #[tokio::test]
    async fn nvenc_to_nvdec_round_trip_on_gpu() {
        gpu_round_trip(VideoCodec::H264).await;
    }

    #[cfg(feature = "nvenc")]
    #[tokio::test]
    async fn nvenc_to_nvdec_hevc_round_trip_on_gpu() {
        gpu_round_trip(VideoCodec::H265).await;
    }

    /// CUDA driver FFI to synthesize an NV12 surface for the encode leg and to
    /// read a decoded luma plane back for verification.
    #[cfg(feature = "nvenc")]
    #[allow(unreachable_pub)]
    mod cu {
        use core::ffi::c_void;
        #[link(name = "cuda")]
        extern "C" {
            #[link_name = "cuInit"]
            pub fn cu_init(flags: u32) -> i32;
            #[link_name = "cuDeviceGet"]
            pub fn cu_device_get(dev: *mut i32, ordinal: i32) -> i32;
            #[link_name = "cuCtxCreate_v2"]
            pub fn cu_ctx_create(pctx: *mut *mut c_void, flags: u32, dev: i32) -> i32;
            #[link_name = "cuCtxDestroy_v2"]
            pub fn cu_ctx_destroy(ctx: *mut c_void) -> i32;
            #[link_name = "cuCtxPushCurrent_v2"]
            pub fn cu_ctx_push_current(ctx: *mut c_void) -> i32;
            #[link_name = "cuCtxPopCurrent_v2"]
            pub fn cu_ctx_pop_current(pctx: *mut *mut c_void) -> i32;
            #[link_name = "cuMemAlloc_v2"]
            pub fn cu_mem_alloc(dptr: *mut u64, bytesize: usize) -> i32;
            #[link_name = "cuMemFree_v2"]
            pub fn cu_mem_free(dptr: u64) -> i32;
            #[link_name = "cuMemcpyHtoD_v2"]
            pub fn cu_memcpy_htod(dst: u64, src: *const c_void, bytesize: usize) -> i32;
            #[link_name = "cuMemcpyDtoH_v2"]
            pub fn cu_memcpy_dtoh(dst: *mut c_void, src: u64, bytesize: usize) -> i32;
        }
    }

    #[cfg(feature = "nvenc")]
    #[derive(Debug)]
    struct DevAlloc {
        dptr: u64,
        ctx: u64,
    }
    #[cfg(feature = "nvenc")]
    impl CudaKeepAlive for DevAlloc {}
    #[cfg(feature = "nvenc")]
    impl Drop for DevAlloc {
        fn drop(&mut self) {
            // SAFETY: free on the allocating context; best-effort.
            unsafe {
                let _ = cu::cu_ctx_push_current(self.ctx as *mut core::ffi::c_void);
                let _ = cu::cu_mem_free(self.dptr);
                let mut popped = core::ptr::null_mut();
                let _ = cu::cu_ctx_pop_current(&mut popped);
            }
        }
    }

    #[cfg(feature = "nvenc")]
    async fn gpu_round_trip(codec: VideoCodec) {
        use crate::nvenc::NvEnc;
        use core::future::Future;
        use core::pin::Pin;
        use g2g_core::frame::Frame;
        use g2g_core::memory::SystemSlice;
        use g2g_core::{FrameTiming, PushOutcome};

        const W: u32 = 320;
        const H: u32 = 240;
        let (w, h) = (W as usize, H as usize);
        let size = w * h * 3 / 2;

        // Bring up a context for the encode leg's source surfaces.
        // SAFETY: standard CUDA driver bring-up; each result is checked and the
        // path bails before using a handle on failure.
        let ctx = unsafe {
            if cu::cu_init(0) != 0 {
                std::eprintln!("skipping: cuInit failed (no NVIDIA GPU)");
                return;
            }
            let mut dev = 0i32;
            if cu::cu_device_get(&mut dev, 0) != 0 {
                std::eprintln!("skipping: no CUDA device");
                return;
            }
            let mut ctx: *mut core::ffi::c_void = core::ptr::null_mut();
            if cu::cu_ctx_create(&mut ctx, 0, dev) != 0 || ctx.is_null() {
                std::eprintln!("skipping: cuCtxCreate failed");
                return;
            }
            ctx as u64
        };
        struct CtxGuard(u64);
        impl Drop for CtxGuard {
            fn drop(&mut self) {
                // SAFETY: the context created just above, destroyed once.
                unsafe {
                    let _ = cu::cu_ctx_destroy(self.0 as *mut core::ffi::c_void);
                }
            }
        }
        let _ctx_guard = CtxGuard(ctx);

        // A moving NV12 pattern as a CUDA-resident frame for the encoder.
        let make_frame = |seq: u64| -> Option<Frame> {
            let mut host = alloc::vec![0u8; size];
            for y in 0..h {
                for x in 0..w {
                    host[y * w + x] = ((x + y + seq as usize * 9) & 0xff) as u8;
                }
            }
            for c in &mut host[w * h..] {
                *c = 128;
            }
            // SAFETY: alloc + upload one NV12 surface in `ctx`; `host` outlives it.
            unsafe {
                let _ = cu::cu_ctx_push_current(ctx as *mut core::ffi::c_void);
                let mut dptr = 0u64;
                let ok = cu::cu_mem_alloc(&mut dptr, size) == 0
                    && cu::cu_memcpy_htod(dptr, host.as_ptr() as *const core::ffi::c_void, size) == 0;
                let mut popped = core::ptr::null_mut();
                let _ = cu::cu_ctx_pop_current(&mut popped);
                if !ok {
                    return None;
                }
                Some(Frame::new(
                    MemoryDomain::Cuda(OwnedCudaBuffer::new(
                        dptr,
                        dptr + (w * h) as u64,
                        W,
                        W,
                        W,
                        H,
                        ctx,
                        Arc::new(DevAlloc { dptr, ctx }),
                    )),
                    FrameTiming { pts_ns: seq * 33_000_000, ..FrameTiming::default() },
                    seq,
                ))
            }
        };

        // Sink collecting H.264 Annex-B access units (System memory).
        #[derive(Default)]
        struct AuSink {
            aus: Vec<Vec<u8>>,
        }
        impl OutputSink for AuSink {
            fn push<'a>(
                &'a mut self,
                packet: PipelinePacket,
            ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
                Box::pin(async move {
                    if let PipelinePacket::DataFrame(f) = packet {
                        if let MemoryDomain::System(s) = &f.domain {
                            self.aus.push(s.as_slice().to_vec());
                        }
                    }
                    Ok(PushOutcome::Accepted)
                })
            }
        }

        let nv12_caps = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
        };
        let mut enc = NvEnc::new().with_codec(codec);
        enc.configure_pipeline(&nv12_caps).unwrap();
        let mut au_sink = AuSink::default();
        for i in 0..10u64 {
            let Some(frame) = make_frame(i) else {
                std::eprintln!("skipping: CUDA alloc/upload failed");
                return;
            };
            if enc.process(PipelinePacket::DataFrame(frame), &mut au_sink).await.is_err() {
                std::eprintln!("skipping: NVENC unavailable on this host");
                return;
            }
        }
        enc.process(PipelinePacket::Eos, &mut au_sink).await.unwrap();
        assert!(!au_sink.aus.is_empty(), "NVENC produced access units to decode");

        // Decode the Annex-B back through the native NvDec; capture NV12 Cuda
        // frames and verify geometry + that the luma plane holds real content.
        #[derive(Default)]
        struct CudaSink {
            caps: Vec<Caps>,
            dims: Vec<(u32, u32)>,
            luma_varied: bool,
            count: usize,
        }
        impl OutputSink for CudaSink {
            fn push<'a>(
                &'a mut self,
                packet: PipelinePacket,
            ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
                Box::pin(async move {
                    match packet {
                        PipelinePacket::CapsChanged(c) => self.caps.push(c),
                        PipelinePacket::DataFrame(f) => {
                            if let MemoryDomain::Cuda(buf) = &f.domain {
                                self.dims.push((buf.width, buf.height));
                                self.count += 1;
                                // Download the first 64 luma bytes; a real decoded
                                // frame is not uniform.
                                if !self.luma_varied {
                                    let mut row = alloc::vec![0u8; 64];
                                    // SAFETY: `buf.luma_ptr` is a valid device ptr
                                    // in `buf.context`; copy a small prefix out.
                                    unsafe {
                                        let _ = cu::cu_ctx_push_current(
                                            buf.context as *mut core::ffi::c_void,
                                        );
                                        let _ = cu::cu_memcpy_dtoh(
                                            row.as_mut_ptr() as *mut core::ffi::c_void,
                                            buf.luma_ptr,
                                            row.len(),
                                        );
                                        let mut popped = core::ptr::null_mut();
                                        let _ = cu::cu_ctx_pop_current(&mut popped);
                                    }
                                    if row.iter().any(|&b| b != row[0]) {
                                        self.luma_varied = true;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                    Ok(PushOutcome::Accepted)
                })
            }
        }

        let mut dec = NvDec::new();
        let in_caps = Caps::CompressedVideo {
            codec,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
        };
        if dec.configure_pipeline(&in_caps).is_err() {
            std::eprintln!("skipping: NVDEC unavailable on this host");
            return;
        }
        let mut cuda_sink = CudaSink::default();
        for au in &au_sink.aus {
            let f = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            dec.process(PipelinePacket::DataFrame(f), &mut cuda_sink).await.expect("decode AU");
        }
        dec.process(PipelinePacket::Eos, &mut cuda_sink).await.expect("flush decoder");

        assert!(cuda_sink.count > 0, "NvDec produced decoded NV12 CUDA frames");
        assert_eq!(
            cuda_sink.caps,
            std::vec![Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(W),
                height: Dim::Fixed(H),
                framerate: Rate::Fixed(30 << 16),
            }],
            "NV12 output caps announced once at the decoded geometry"
        );
        assert!(
            cuda_sink.dims.iter().all(|&d| d == (W, H)),
            "every decoded frame is {W}x{H}, got {:?}",
            cuda_sink.dims
        );
        assert!(cuda_sink.luma_varied, "decoded luma holds real (non-uniform) content");
    }
}
