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
//! on drop, and an `Arc` to the decoder so the decoder outlives any frame still in
//! flight. The CUDA context and the NVCUVID context lock are a separate `Arc`
//! ([`CuvidContext`]) shared by every decoder the element builds, so a mid-stream
//! rebuild cannot tear them out from under an older decoder's frames.
//!
//! A mid-stream format change re-enters the sequence callback: a new coded size
//! that still fits the live decoder's ceiling (and keeps its surface format) is
//! applied in place with `cuvidReconfigureDecoder`, anything else (a bigger
//! picture, a bit-depth change) builds a fresh decoder. Either way the next
//! emitted frame carries a new `CapsChanged`.
//!
//! Bit depth follows the stream: 8-bit decodes to NV12, 10-/12-bit to NVDEC's
//! `P016` surface, announced as [`RawVideoFormat::P010`] (16-bit samples, value in
//! the top bits).

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;

use core::sync::atomic::{AtomicU32, Ordering};

use g2g_core::g2g_error;
use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{CudaKeepAlive, DomainSet, MemoryDomainKind, OwnedCudaBuffer};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, G2gError, HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates,
    PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
    SystemSlice, VideoCodec,
};

/// Number of decode surfaces the parser cycles through. Also the cap on the
/// decoder's surface pool; the sequence callback clamps the stream's minimum into
/// this. Bigger = more reorder / in-flight headroom at a memory cost.
const NUM_DECODE_SURFACES: u32 = 20;
/// Default max output surfaces mapped at once, i.e. how many decoded frames the
/// rest of the pipeline may hold before a map fails the decode outright. A
/// display chain holds several at a time (the link queues plus the frame on
/// screen), so the default leaves headroom above that.
const DEFAULT_NUM_OUTPUT_SURFACES: u32 = 20;
/// Upper bound the `num-output-surfaces` property accepts, as gst-nvcodec's.
const NUM_OUTPUT_SURFACES_LIMIT: u32 = 64;
/// Default parser display delay, in frames: one, the low-latency setting.
const DEFAULT_MAX_DISPLAY_DELAY: u32 = 1;
/// Upper bound the `max-display-delay` property accepts, as gst-nvcodec's.
const MAX_DISPLAY_DELAY_LIMIT: u32 = 16;

/// Native NVDEC H.264 decoder. Annex-B in, CUDA NV12 out. See the module docs.
///
/// # Example
///
/// ```no_run
/// use g2g_plugins::nvdec::NvDec;
///
/// let dec = NvDec::new().with_max_display_delay(2);
/// ```
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
    /// Caps of the last frame emitted; a decoded frame whose format / geometry
    /// differs (a mid-stream resolution or bit-depth change) re-announces.
    last_caps: Option<Caps>,
    /// Frames the parser holds back before displaying (`CUVIDPARSERPARAMS::
    /// ulMaxDisplayDelay`): 1 keeps glass-to-glass tight, higher values pipeline
    /// decode against display at the cost of latency. Applied when the parser
    /// opens at configure.
    max_display_delay: u32,
    /// Decoded frames that may be mapped at once (`CUVIDDECODECREATEINFO::
    /// ulNumOutputSurfaces`). Every frame still held downstream occupies one, so
    /// a chain that queues more than this decodes until the pool is empty and
    /// then fails. Applied when the decoder is created.
    num_output_surfaces: u32,
    /// CUDA device this decoder opens its context on, carried onto every emitted
    /// frame so a consumer knows which GPU the surface lives on. Read at
    /// configure, when the context is created.
    cuda_device_id: i32,
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
    /// CUDA context + NVCUVID context lock, shared by every decoder this element
    /// builds (see [`CuvidContext`]). `None` until `open`.
    cuda: Option<Arc<CuvidContext>>,
    /// The `cudaVideoCodec` the parser / decoder were created for (H.264, HEVC or
    /// AV1), from the negotiated input caps.
    codec_cuvid: i32,
    /// `CUvideodecoder`, created in the sequence callback. Raw copy for the decode
    /// / display callbacks; ownership / destruction is the `Arc`'s.
    decoder: *mut core::ffi::c_void,
    /// Shared decoder owner; cloned into each frame keep-alive so the decoder
    /// outlives frames still referenced downstream.
    decoder_owner: Option<Arc<CuvidDecoder>>,
    /// Coded dimensions the live decoder was created for: its reconfigure
    /// ceiling, a bigger picture needs a fresh decoder.
    max_width: u32,
    max_height: u32,
    /// Coded dimensions the decoder is currently set to (moves with each
    /// reconfigure, unlike the ceiling above).
    coded_width: u32,
    coded_height: u32,
    /// Display geometry (the cropped output dims). Chroma offset uses `target_height`.
    target_width: u32,
    target_height: u32,
    /// Decode-surface count the live decoder was created with; a reconfigure must
    /// stay within it, and the sequence callback keeps reporting it to the parser.
    num_decode_surfaces: u32,
    /// Output-surface count to create the decoder with, copied from the element
    /// when the parser opens (the sequence callback has only this state).
    num_output_surfaces: u32,
    /// Decoders built so far: 1 for a stream whose format changes stayed within
    /// what `cuvidReconfigureDecoder` can apply in place.
    decoders_created: u32,
    /// The live decoder's `cudaVideoSurfaceFormat` and the caps format it maps to
    /// (NV12 for 8-bit, P010 for 10-/12-bit).
    surface_format: i32,
    out_format: RawVideoFormat,
    /// Frames mapped and ready to emit (drained by `process` after each parse).
    ready: Vec<ReadyFrame>,
    /// First error raised inside a callback, surfaced after the parse returns.
    error: Option<G2gError>,
}

/// A decoded, mapped surface ready to hand downstream, with the pixel format the
/// decoder produced it in (the frame's own geometry lives in `buffer`).
struct ReadyFrame {
    buffer: OwnedCudaBuffer,
    pts_ns: u64,
    format: RawVideoFormat,
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
                cuda: None,
                codec_cuvid: ffi::CUDA_VIDEO_CODEC_H264,
                decoder: core::ptr::null_mut(),
                decoder_owner: None,
                max_width: 0,
                max_height: 0,
                coded_width: 0,
                coded_height: 0,
                target_width: 0,
                target_height: 0,
                num_decode_surfaces: 0,
                num_output_surfaces: DEFAULT_NUM_OUTPUT_SURFACES,
                decoders_created: 0,
                surface_format: ffi::CUDA_VIDEO_SURFACE_FORMAT_NV12,
                out_format: RawVideoFormat::Nv12,
                ready: Vec::new(),
                error: None,
            }),
            emitted: 0,
            last_caps: None,
            max_display_delay: DEFAULT_MAX_DISPLAY_DELAY,
            num_output_surfaces: DEFAULT_NUM_OUTPUT_SURFACES,
            cuda_device_id: crate::cudadeviceid::DEFAULT_CUDA_DEVICE_ID,
            configured: false,
            out_domain: MemoryDomainKind::Cuda,
        }
    }

    /// Decode on CUDA device `ordinal` instead of device 0, on a host with more
    /// than one NVIDIA GPU. Also the `cuda-device-id` property; read when the
    /// context opens at configure, and the ordinal every emitted frame carries.
    /// An ordinal the driver does not have fails the configure.
    pub fn with_cuda_device_id(mut self, ordinal: i32) -> Self {
        self.cuda_device_id = ordinal;
        self
    }

    /// Frames the parser holds back before displaying (0..=16, default 1). Higher
    /// values pipeline decode against display (NVIDIA recommends 2..4 for
    /// throughput) at the cost of latency. Also the `max-display-delay` property;
    /// applied when the parser opens at configure.
    pub fn with_max_display_delay(mut self, frames: u32) -> Self {
        self.max_display_delay = frames.min(MAX_DISPLAY_DELAY_LIMIT);
        self
    }

    /// Decoded frames that may be held downstream at once (1..=64, default 20).
    /// Also the `num-output-surfaces` property. A chain that holds more frames
    /// than this fails the decode once the pool empties, so raise it for a deep
    /// one; each surface costs a full frame of device memory.
    pub fn with_num_output_surfaces(mut self, surfaces: u32) -> Self {
        self.num_output_surfaces = surfaces.clamp(1, NUM_OUTPUT_SURFACES_LIMIT);
        self
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

    /// NVDEC decoders built for this stream: one, unless a mid-stream format
    /// change went beyond what an in-place reconfigure can apply.
    pub fn decoders_created(&self) -> u32 {
        self.state.decoders_created
    }

    /// Accepted input codecs. AV1 needs an Ampere+ NVDEC; an older GPU fails
    /// decoder creation at the first sequence rather than at negotiation.
    fn input_codecs() -> [VideoCodec; 3] {
        [VideoCodec::H264, VideoCodec::H265, VideoCodec::Av1]
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
                    colorimetry: g2g_core::Colorimetry::UNKNOWN,
                })
                .collect(),
        )
    }

    /// The `cudaVideoCodec` value for a supported input codec.
    fn cuvid_codec(codec: VideoCodec) -> Option<i32> {
        match codec {
            VideoCodec::H264 => Some(ffi::CUDA_VIDEO_CODEC_H264),
            VideoCodec::H265 => Some(ffi::CUDA_VIDEO_CODEC_HEVC),
            VideoCodec::Av1 => Some(ffi::CUDA_VIDEO_CODEC_AV1),
            _ => None,
        }
    }

    /// The output caps for one decoded frame's format and geometry.
    fn frame_caps(&self, format: RawVideoFormat, w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: self.framerate.clone(),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
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
            let rc = ffi::cu_device_get(&mut dev, self.cuda_device_id);
            if rc != 0 {
                g2g_error!(
                    Target::category(short_type_name::<NvDec>()),
                    "no CUDA device {} (cuDeviceGet returned {}): set cuda-device-id to an ordinal this host has",
                    self.cuda_device_id,
                    rc
                );
                return Err(G2gError::Hardware(HardwareError::Cuda(rc)));
            }
            let mut ctx: *mut core::ffi::c_void = core::ptr::null_mut();
            cuchk(ffi::cu_ctx_create(&mut ctx, 0, dev))?;
            if ctx.is_null() {
                return Err(hw());
            }
            ctx as u64
        };
        self.context = context;

        let _ctx = ContextGuard::push(context)?;
        // SAFETY: valid context; on success `ctx_lock` receives the lock handle.
        let ctx_lock = unsafe {
            let mut lock: *mut core::ffi::c_void = core::ptr::null_mut();
            cuchk(ffi::cuvid_ctx_lock_create(
                &mut lock,
                context as *mut core::ffi::c_void,
            ))?;
            lock
        };
        self.state.cuda = Some(Arc::new(CuvidContext {
            ctx_lock,
            context,
            device_ordinal: self.cuda_device_id,
        }));

        self.state.num_output_surfaces = self.num_output_surfaces;

        // Create the parser, pointing it at the heap `DecoderState` as user-data.
        let user = self.state.as_mut() as *mut DecoderState as *mut core::ffi::c_void;
        // SAFETY: the NVCUVID param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then fill.
        let mut params: ffi::ParserParams = unsafe { core::mem::zeroed() };
        params.codec_type = self.state.codec_cuvid;
        params.max_num_decode_surfaces = NUM_DECODE_SURFACES;
        params.max_display_delay = self.max_display_delay;
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
    fn parse(
        &mut self,
        payload: &[u8],
        pts_ns: u64,
        eos: bool,
    ) -> Result<Vec<ReadyFrame>, G2gError> {
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
        for f in frames {
            // Announce the frame's own format / geometry, so a mid-stream
            // resolution or bit-depth change re-announces before its first frame.
            let caps = self.frame_caps(f.format, f.buffer.width, f.buffer.height);
            if self.last_caps.as_ref() != Some(&caps) {
                out.push(PipelinePacket::CapsChanged(caps.clone())).await?;
                self.last_caps = Some(caps);
            }
            // M352: keep the surface on the GPU (zero-copy) unless negotiation
            // settled this decoder's output on System, in which case download it
            // device->host before emitting.
            let domain = if self.out_domain == MemoryDomainKind::System {
                // SAFETY: `f.buffer`'s plane pointers are valid CUDA device memory
                // in its context, pinned by the buffer's keep-alive owner for the
                // duration of this copy.
                let bytes =
                    unsafe { crate::cuda::download_nv12(&f.buffer, f.format.bytes_per_sample())? };
                MemoryDomain::System(SystemSlice::from_boxed(bytes))
            } else {
                MemoryDomain::Cuda(f.buffer)
            };
            let frame = g2g_core::frame::Frame::new(
                domain,
                g2g_core::FrameTiming {
                    pts_ns: f.pts_ns,
                    dts_ns: f.pts_ns,
                    ..Default::default()
                },
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
        // boxed state drop (releasing the decoder / context `Arc`s; each is torn
        // down once the last frame referencing it is gone).
        if !self.parser.is_null() {
            // SAFETY: `parser` was created in `open` and is destroyed once.
            unsafe {
                let _ = ffi::cuvid_destroy_video_parser(self.parser);
            }
            self.parser = core::ptr::null_mut();
        }
    }
}

/// Owns the CUDA context and the NVCUVID context lock, destroying them (lock
/// first) when the last reference drops. Held by the element and by every
/// decoder it builds, so a mid-stream decoder rebuild leaves the older decoder's
/// context intact for as long as its frames live.
struct CuvidContext {
    ctx_lock: *mut core::ffi::c_void,
    context: u64,
    /// Ordinal of the device the context was created on, carried onto every
    /// frame's `OwnedCudaBuffer` so a consumer can name the device.
    device_ordinal: i32,
}

// SAFETY: the handles are owned and inert; see `CuvidDecoder` below for the
// shared-access contract they are used under.
unsafe impl Send for CuvidContext {}
// SAFETY: as for `Send` above.
unsafe impl Sync for CuvidContext {}

impl core::fmt::Debug for CuvidContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("CuvidContext(<CUcontext>)")
    }
}

impl Drop for CuvidContext {
    fn drop(&mut self) {
        // SAFETY: both handles were created together in `open` and are destroyed
        // once, here, after every decoder built on them has been destroyed (each
        // holds an `Arc` to this). Best-effort; failures are unactionable.
        unsafe {
            if !self.ctx_lock.is_null() {
                let _ = ffi::cuvid_ctx_lock_destroy(self.ctx_lock);
            }
            if self.context != 0 {
                let _ = ffi::cu_ctx_destroy(self.context as *mut core::ffi::c_void);
            }
        }
    }
}

/// Owns one `CUvideodecoder` and pins the context it lives in, tearing the
/// decoder down when the last reference, the element itself or any frame
/// keep-alive still in flight, drops. Boxed as the [`CudaKeepAlive`] of every
/// emitted frame.
struct CuvidDecoder {
    decoder: *mut core::ffi::c_void,
    ctx: Arc<CuvidContext>,
    /// Output surfaces mapped right now. Only read to tell an exhausted pool
    /// apart from any other map failure, so the error can name the cause.
    mapped: AtomicU32,
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
        // SAFETY: `decoder` is destroyed once, here, after every mapped frame has
        // been unmapped (frames hold an `Arc` to this, so this drop runs only once
        // none remain). The context outlives it via `ctx`. Best-effort.
        unsafe {
            if !self.decoder.is_null() {
                let _ = ffi::cuvid_destroy_decoder(self.decoder);
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
        self.owner.mapped.fetch_sub(1, Ordering::Relaxed);
        // SAFETY: `dev_ptr` was returned by `cuvidMapVideoFrame64` on
        // `owner.decoder` and is unmapped once. Push the context first so the
        // unmap runs in it; best-effort.
        unsafe {
            let _ = ffi::cu_ctx_push_current(self.owner.ctx.context as *mut core::ffi::c_void);
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
/// decoder, or bring the live one to the new format (in place where NVDEC allows
/// it), and report the surface count back to the parser.
extern "C" fn handle_sequence(user: *mut core::ffi::c_void, fmt: *mut ffi::VideoFormat) -> i32 {
    // SAFETY: `user` is the boxed `DecoderState` pointer set in `open`; `fmt` is a
    // valid format struct for the duration of the callback.
    let state = unsafe { &mut *(user as *mut DecoderState) };
    // SAFETY: `fmt` is the valid format struct the parser passes for this callback.
    let f = unsafe { &*fmt };

    let num_surfaces = (f.min_num_decode_surfaces as u32).clamp(1, NUM_DECODE_SURFACES);
    // Display (cropped) geometry, rounded up to even for 4:2:0 chroma.
    let disp_w = (f.display_area.right - f.display_area.left).max(0) as u32;
    let disp_h = (f.display_area.bottom - f.display_area.top).max(0) as u32;
    let target_w = if disp_w != 0 { disp_w } else { f.coded_width };
    let target_h = if disp_h != 0 { disp_h } else { f.coded_height };
    let target_w = (target_w + 1) & !1;
    let target_h = (target_h + 1) & !1;
    // 8-bit decodes to NV12; anything deeper to the 16-bit semi-planar surface,
    // which g2g calls P010 (samples in the top bits of each 16-bit word).
    let (surface_format, out_format) = if f.bit_depth_luma_minus8 > 0 {
        (ffi::CUDA_VIDEO_SURFACE_FORMAT_P016, RawVideoFormat::P010)
    } else {
        (ffi::CUDA_VIDEO_SURFACE_FORMAT_NV12, RawVideoFormat::Nv12)
    };

    if state.decoder_owner.is_some() {
        let fits = f.coded_width <= state.max_width
            && f.coded_height <= state.max_height
            && num_surfaces <= state.num_decode_surfaces
            && surface_format == state.surface_format;
        if fits {
            if state.coded_width == f.coded_width
                && state.coded_height == f.coded_height
                && state.target_width == target_w
                && state.target_height == target_h
            {
                // Same geometry and format: nothing to do.
                return state.num_decode_surfaces as i32;
            }
            // SAFETY: the NVCUVID param structs are plain old data (ints,
            // reserved arrays); all-zero is a valid initial state we then fill.
            let mut re: ffi::ReconfigureDecoderInfo = unsafe { core::mem::zeroed() };
            re.width = f.coded_width;
            re.height = f.coded_height;
            re.target_width = target_w;
            re.target_height = target_h;
            // Must stay at the count the decoder was created with.
            re.num_decode_surfaces = state.num_decode_surfaces;
            re.display_area_left = f.display_area.left as i16;
            re.display_area_top = f.display_area.top as i16;
            re.display_area_right = f.display_area.right as i16;
            re.display_area_bottom = f.display_area.bottom as i16;
            // SAFETY: valid decoder; `re` is fully initialized and only read for
            // the duration of the call.
            let rc = unsafe { ffi::cuvid_reconfigure_decoder(state.decoder, &mut re) };
            if rc != 0 {
                return fail(state, G2gError::Hardware(HardwareError::Cuda(rc)));
            }
            state.coded_width = f.coded_width;
            state.coded_height = f.coded_height;
            state.target_width = target_w;
            state.target_height = target_h;
            return state.num_decode_surfaces as i32;
        }
        // Beyond what the live decoder can be reconfigured to (a bigger picture,
        // a deeper bit depth, more surfaces): build a fresh one. The old decoder
        // lives on in any frame still downstream and dies with the last of them.
    }

    let Some(cuda) = state.cuda.clone() else {
        return fail(state, hw());
    };

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
    info.output_format = surface_format;
    info.deinterlace_mode = ffi::CUDA_VIDEO_DEINTERLACE_WEAVE;
    info.target_width = target_w as u64;
    info.target_height = target_h as u64;
    info.num_output_surfaces = state.num_output_surfaces as u64;
    info.vid_lock = cuda.ctx_lock;

    let mut decoder: *mut core::ffi::c_void = core::ptr::null_mut();
    // SAFETY: `info` is fully initialized; on success `decoder` is a valid handle.
    let rc = unsafe { ffi::cuvid_create_decoder(&mut decoder, &mut info) };
    if rc != 0 || decoder.is_null() {
        return fail(state, G2gError::Hardware(HardwareError::Cuda(rc)));
    }
    state.decoder = decoder;
    state.max_width = f.coded_width;
    state.max_height = f.coded_height;
    state.coded_width = f.coded_width;
    state.coded_height = f.coded_height;
    state.target_width = target_w;
    state.target_height = target_h;
    state.num_decode_surfaces = num_surfaces;
    state.decoders_created += 1;
    state.surface_format = surface_format;
    state.out_format = out_format;
    state.decoder_owner = Some(Arc::new(CuvidDecoder {
        decoder,
        ctx: cuda,
        mapped: AtomicU32::new(0),
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
        if owner.mapped.load(Ordering::Relaxed) >= state.num_output_surfaces {
            g2g_error!(
                Target::category(short_type_name::<NvDec>()),
                "the pipeline is holding all {} decoded frames NVDEC can map at once, so there is none left to decode into: raise num-output-surfaces (max {})",
                state.num_output_surfaces,
                NUM_OUTPUT_SURFACES_LIMIT
            );
        }
        return fail(state, G2gError::Hardware(HardwareError::Cuda(rc)));
    }
    owner.mapped.fetch_add(1, Ordering::Relaxed);

    // Semi-planar: the chroma plane follows luma at pitch * target_height bytes,
    // at 8 or 16 bits per sample.
    let chroma_ptr = dev_ptr + (pitch as u64) * (state.target_height as u64);
    let context = owner.ctx.context;
    let device_ordinal = owner.ctx.device_ordinal;
    let buffer = OwnedCudaBuffer::new(
        dev_ptr,
        chroma_ptr,
        pitch,
        pitch,
        state.target_width,
        state.target_height,
        context,
        device_ordinal,
        Arc::new(CuvidMappedFrame { owner, dev_ptr }),
    );
    state.ready.push(ReadyFrame {
        buffer,
        pts_ns: d.timestamp as u64,
        format: state.out_format,
    });
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

    /// Reads host memory, so it takes system frames only. The allocation
    /// cascade turns that into a download demand on a GPU producer.
    fn input_domains(&self) -> g2g_core::memory::DomainSet {
        g2g_core::memory::DomainSet::only(g2g_core::memory::MemoryDomainKind::System)
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for alt in Self::input_caps_set().alternatives() {
            if let Ok(narrowed) = upstream_caps.intersect(alt) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
    }

    /// Native `DerivedOutput`: a supported codec (any geometry) in, NV12 or (for a
    /// 10-/12-bit stream) P010 at the same dims and framerate out. Any other input
    /// yields an empty set, rejected at solve. The runtime `CapsChanged` carries
    /// the actual decoded (cropped) dims and the depth the stream turned out to be.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::CompressedVideo {
                codec: VideoCodec::H264 | VideoCodec::H265 | VideoCodec::Av1,
                width,
                height,
                framerate,
                ..
            } => CapsSet::from_alternatives(
                [RawVideoFormat::Nv12, RawVideoFormat::P010]
                    .into_iter()
                    .map(|format| Caps::RawVideo {
                        format,
                        width: width.clone(),
                        height: height.clone(),
                        framerate: framerate.clone(),
                        interlace: g2g_core::Interlace::Any,
                        colorimetry: g2g_core::Colorimetry::UNKNOWN,
                    })
                    .collect(),
            ),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::CompressedVideo {
            codec,
            width,
            height,
            framerate,
            ..
        } = absolute_caps
        else {
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

    /// The domain this decoder emits into: CUDA device memory (the zero-copy
    /// hwframe domain, and the default) until the allocation cascade settles it
    /// on System for a host-memory consumer. Reporting the settled domain rather
    /// than the default is what keeps a graph dump honest about which links are
    /// GPU links (M285). [`output_domains`](Self::output_domains) is the full set
    /// it can satisfy.
    fn output_memory(&self) -> g2g_core::memory::MemoryDomainKind {
        self.out_domain
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
            "NVDEC H.264 / HEVC / AV1 decoder",
            "Codec/Decoder/Video/Hardware",
            "Zero-copy H.264 / HEVC / AV1 decode to CUDA NV12 / P010 surfaces via the NVIDIA Video Codec SDK (NVCUVID)",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        NVDEC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "max-display-delay" => {
                let frames = value.as_uint().ok_or(PropError::Type)?;
                if frames > MAX_DISPLAY_DELAY_LIMIT as u64 {
                    return Err(PropError::Value);
                }
                self.max_display_delay = frames as u32;
                Ok(())
            }
            "num-output-surfaces" => {
                let surfaces = value.as_uint().ok_or(PropError::Type)?;
                if surfaces == 0 || surfaces > NUM_OUTPUT_SURFACES_LIMIT as u64 {
                    return Err(PropError::Value);
                }
                self.num_output_surfaces = surfaces as u32;
                Ok(())
            }
            "cuda-device-id" => crate::cudadeviceid::set_cuda_device_id(
                &mut self.cuda_device_id,
                self.state.cuda.is_some(),
                &value,
            ),
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "max-display-delay" => Some(PropValue::Uint(self.max_display_delay as u64)),
            "num-output-surfaces" => Some(PropValue::Uint(self.num_output_surfaces as u64)),
            "cuda-device-id" => Some(crate::cudadeviceid::get_cuda_device_id(self.cuda_device_id)),
            _ => None,
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
                    let slice = frame
                        .domain
                        .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
                    let frames = self.parse(slice, frame.timing.pts_ns, false)?;
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
        let out = CapsSet::from_alternatives(
            [RawVideoFormat::Nv12, RawVideoFormat::P010]
                .into_iter()
                .map(|format| Caps::RawVideo {
                    format,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                    interlace: g2g_core::Interlace::Any,
                    colorimetry: g2g_core::Colorimetry::UNKNOWN,
                })
                .collect(),
        );
        Vec::from([
            PadTemplate::sink(Self::input_caps_set()),
            PadTemplate::source(out),
        ])
    }
}

/// Settable properties: the parser's display delay, the output-surface pool, and
/// the CUDA device, so a `gst-launch` line can trade latency for decode/display
/// pipelining, give a deep chain room to hold frames, or pin the GPU, without the
/// builder. Named as gst-nvcodec's decoders name them.
static NVDEC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "max-display-delay",
        PropKind::Uint,
        "frames the parser holds back before display, 0..16 (default 1, low latency)",
    ),
    PropertySpec::new(
        "num-output-surfaces",
        PropKind::Uint,
        "decoded frames the pipeline may hold at once, 1..64 (default 20); a deeper chain needs more, at a frame of device memory each",
    ),
    crate::cudadeviceid::CUDA_DEVICE_ID_PROP,
];

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
    pub const CUDA_VIDEO_CODEC_AV1: i32 = 11;
    pub const CUDA_VIDEO_SURFACE_FORMAT_NV12: i32 = 0;
    /// 16-bit semi-planar YUV, the 10-/12-bit output surface (g2g `P010`).
    pub const CUDA_VIDEO_SURFACE_FORMAT_P016: i32 = 1;
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

    /// `CUVIDRECONFIGUREDECODERINFO` (128 bytes), the in-place decoder reset for a
    /// mid-stream resolution change. The two `short` rectangles are flattened into
    /// named `i16` fields, as in [`DecodeCreateInfo`].
    #[repr(C)]
    pub struct ReconfigureDecoderInfo {
        pub width: u32,
        pub height: u32,
        pub target_width: u32,
        pub target_height: u32,
        pub num_decode_surfaces: u32,
        pub reserved1: [u32; 12],
        pub display_area_left: i16,
        pub display_area_top: i16,
        pub display_area_right: i16,
        pub display_area_bottom: i16,
        pub target_rect_left: i16,
        pub target_rect_top: i16,
        pub target_rect_right: i16,
        pub target_rect_bottom: i16,
        pub reserved2: [u32; 11],
    }
    const _: () = assert!(core::mem::size_of::<ReconfigureDecoderInfo>() == 128);

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
        #[link_name = "cuvidReconfigureDecoder"]
        pub fn cuvid_reconfigure_decoder(
            decoder: *mut c_void,
            info: *mut ReconfigureDecoderInfo,
        ) -> i32;
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
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        }
    }

    // --- Pure-logic coverage (no GPU): caps + struct layout ---

    #[test]
    fn caps_constraint_covers_the_decodable_codecs_and_depths() {
        let d = NvDec::new();
        let CapsConstraint::DerivedOutput(derive) = d.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let out = derive(&h264(1920, 1080));
        // Both depths are offered; the runtime CapsChanged says which it is.
        for format in [RawVideoFormat::Nv12, RawVideoFormat::P010] {
            assert!(out.accepts(&Caps::RawVideo {
                format,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            }));
        }
        // AV1 decodes too (Ampere+).
        let av1 = Caps::CompressedVideo {
            codec: VideoCodec::Av1,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(!derive(&av1).alternatives().is_empty());
        // A codec NVDEC has no decoder for yields an empty set, rejected at solve.
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        assert!(derive(&vp9).alternatives().is_empty());
    }

    #[test]
    fn frame_caps_track_the_decoded_format_and_dims() {
        let mut d = NvDec::new();
        d.framerate = Rate::Fixed(30 << 16);
        assert_eq!(
            d.frame_caps(RawVideoFormat::Nv12, 1280, 720),
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(1280),
                height: Dim::Fixed(720),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            }
        );
        // A 10-bit stream announces the semi-planar 16-bit surface instead.
        assert!(matches!(
            d.frame_caps(RawVideoFormat::P010, 640, 480),
            Caps::RawVideo {
                format: RawVideoFormat::P010,
                width: Dim::Fixed(640),
                ..
            }
        ));
    }

    #[test]
    fn max_display_delay_property_round_trips() {
        let mut d = NvDec::new();
        assert_eq!(
            d.get_property("max-display-delay"),
            Some(PropValue::Uint(DEFAULT_MAX_DISPLAY_DELAY as u64))
        );
        d.set_property("max-display-delay", PropValue::Uint(4))
            .unwrap();
        assert_eq!(d.max_display_delay, 4);
        assert_eq!(
            d.get_property("max-display-delay"),
            Some(PropValue::Uint(4))
        );
        // Out of the 0..=16 range NVCUVID accepts.
        assert_eq!(
            d.set_property("max-display-delay", PropValue::Uint(64)),
            Err(PropError::Value)
        );
        assert!(d.properties().iter().any(|s| s.name == "max-display-delay"));
        assert_eq!(NvDec::new().with_max_display_delay(3).max_display_delay, 3);
    }

    #[test]
    fn num_output_surfaces_property_round_trips() {
        let mut d = NvDec::new();
        assert_eq!(
            d.get_property("num-output-surfaces"),
            Some(PropValue::Uint(DEFAULT_NUM_OUTPUT_SURFACES as u64))
        );
        d.set_property("num-output-surfaces", PropValue::Uint(32))
            .unwrap();
        assert_eq!(d.num_output_surfaces, 32);
        assert_eq!(
            d.get_property("num-output-surfaces"),
            Some(PropValue::Uint(32))
        );
        // A pool of zero could never map a frame, and 64 is NVCUVID's ceiling.
        assert_eq!(
            d.set_property("num-output-surfaces", PropValue::Uint(0)),
            Err(PropError::Value)
        );
        assert_eq!(
            d.set_property("num-output-surfaces", PropValue::Uint(65)),
            Err(PropError::Value)
        );
        assert!(d
            .properties()
            .iter()
            .any(|s| s.name == "num-output-surfaces"));
        assert_eq!(
            NvDec::new()
                .with_num_output_surfaces(99)
                .num_output_surfaces,
            NUM_OUTPUT_SURFACES_LIMIT
        );
    }

    // --- On-hardware fixture decodes (RTX 3060): mid-stream resolution change,
    // 10-bit (P010) output, AV1, and the display-delay knob. Each skips cleanly
    // when NVDEC is unavailable. ---

    /// 6 frames at 640x480 then 6 at 320x240, concatenated Annex-B.
    const H264_RECONFIG: &[u8] =
        include_bytes!("../tests/fixtures/h264_reconfig_640x480_to_320x240.h264");
    /// 640x480 Main 10 (10-bit) HEVC.
    const HEVC_MAIN10: &[u8] = include_bytes!("../tests/fixtures/h265_640x480_main10.hevc");
    /// 640x480 AV1, a raw low-overhead OBU stream.
    const AV1_CLIP: &[u8] = include_bytes!("../tests/fixtures/av1_640x480.obu");
    /// 640x480 H.264, single resolution.
    const H264_CLIP: &[u8] = include_bytes!("../tests/fixtures/h264_640x480.h264");

    /// Records what reached the sink: announced caps, per-frame geometry, and the
    /// first 64 bytes of the first frame's luma plane (a real decode is not flat).
    #[derive(Default)]
    struct RecordSink {
        caps: Vec<Caps>,
        dims: Vec<(u32, u32)>,
        luma_head: Option<Vec<u8>>,
    }

    impl OutputSink for RecordSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            let packet = packet_slot.take().expect("poll_push without a packet");
            core::task::Poll::Ready({
                match packet {
                    PipelinePacket::CapsChanged(c) => self.caps.push(c),
                    PipelinePacket::DataFrame(f) => {
                        if let MemoryDomain::Cuda(buf) = &f.domain {
                            self.dims.push((buf.width, buf.height));
                            if self.luma_head.is_none() {
                                let mut row = alloc::vec![0u8; 64];
                                // SAFETY: `buf.luma_ptr` is valid device memory in
                                // `buf.context`; copy a small prefix out of it.
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
                                self.luma_head = Some(row);
                            }
                        }
                    }
                    _ => {}
                }
                Ok(g2g_core::PushOutcome::Accepted)
            })
        }
    }

    /// Byte offsets of each Annex-B NAL payload (just past its start code).
    fn start_code_offsets(data: &[u8]) -> Vec<usize> {
        let mut offs = Vec::new();
        let mut i = 0;
        while i + 3 <= data.len() {
            if data[i] == 0 && data[i + 1] == 0 {
                if data[i + 2] == 1 {
                    offs.push(i + 3);
                    i += 3;
                    continue;
                }
                if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                    offs.push(i + 4);
                    i += 4;
                    continue;
                }
            }
            i += 1;
        }
        offs
    }

    /// Split an Annex-B stream into access units, one per picture: `is_vcl`
    /// decides which NAL types close a unit (the fixtures are single-slice).
    fn split_access_units(stream: &[u8], is_vcl: fn(&[u8]) -> bool) -> Vec<Vec<u8>> {
        let mut units = Vec::new();
        let mut cur: Vec<u8> = Vec::new();
        let starts = start_code_offsets(stream);
        for (k, &begin) in starts.iter().enumerate() {
            let end = starts.get(k + 1).copied().unwrap_or(stream.len());
            let nal = &stream[begin..end];
            cur.extend_from_slice(&[0, 0, 0, 1]);
            cur.extend_from_slice(nal);
            if is_vcl(nal) {
                units.push(core::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            if let Some(last) = units.last_mut() {
                last.extend_from_slice(&cur);
            }
        }
        units
    }

    fn h264_units(stream: &[u8]) -> Vec<Vec<u8>> {
        split_access_units(stream, |nal| {
            matches!(nal.first().map(|b| b & 0x1F), Some(1) | Some(5))
        })
    }

    fn h265_units(stream: &[u8]) -> Vec<Vec<u8>> {
        // HEVC NAL header is two bytes; VCL types are 0..=31.
        split_access_units(stream, |nal| {
            nal.first().map(|b| (b >> 1) & 0x3F).unwrap_or(63) <= 31
        })
    }

    /// Split a low-overhead AV1 OBU stream into temporal units (each starts at a
    /// temporal delimiter, OBU type 2). Sizes are leb128 after the header byte.
    fn av1_temporal_units(stream: &[u8]) -> Vec<Vec<u8>> {
        let mut units: Vec<Vec<u8>> = Vec::new();
        let mut cur: Vec<u8> = Vec::new();
        let mut i = 0usize;
        while i < stream.len() {
            let header = stream[i];
            let obu_type = (header >> 3) & 0xF;
            let has_extension = header & 0x04 != 0;
            let has_size = header & 0x02 != 0;
            let mut p = i + 1 + usize::from(has_extension);
            if !has_size || p >= stream.len() {
                break;
            }
            // leb128 payload size.
            let mut size = 0usize;
            let mut shift = 0;
            loop {
                if p >= stream.len() || shift > 56 {
                    return units;
                }
                let b = stream[p];
                p += 1;
                size |= ((b & 0x7f) as usize) << shift;
                shift += 7;
                if b & 0x80 == 0 {
                    break;
                }
            }
            let end = match p.checked_add(size) {
                Some(e) if e <= stream.len() => e,
                _ => return units,
            };
            if obu_type == 2 && !cur.is_empty() {
                units.push(core::mem::take(&mut cur));
            }
            cur.extend_from_slice(&stream[i..end]);
            i = end;
        }
        if !cur.is_empty() {
            units.push(cur);
        }
        units
    }

    /// A system-memory frame carrying one access unit.
    fn au_frame(au: &[u8], pts_ns: u64) -> g2g_core::frame::Frame {
        g2g_core::frame::Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(au.to_vec().into_boxed_slice())),
            g2g_core::FrameTiming {
                pts_ns,
                ..Default::default()
            },
            0,
        )
    }

    /// Configure a decoder for `codec` at `w`x`h`, or `None` if NVDEC is
    /// unavailable on this host (the test then skips).
    fn open_dec(codec: VideoCodec, w: u32, h: u32, display_delay: u32) -> Option<NvDec> {
        let mut dec = NvDec::new().with_max_display_delay(display_delay);
        let caps = Caps::CompressedVideo {
            codec,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        match dec.configure_pipeline(&caps) {
            Ok(_) => Some(dec),
            Err(_) => {
                std::eprintln!("skipping: NVDEC unavailable on this host");
                None
            }
        }
    }

    #[tokio::test]
    async fn mid_stream_resolution_change_reconfigures_the_decoder() {
        let Some(mut dec) = open_dec(VideoCodec::H264, 640, 480, 1) else {
            return;
        };
        let units = h264_units(H264_RECONFIG);
        assert!(units.len() > 6, "fixture split into per-picture AUs");
        let mut sink = RecordSink::default();
        for (i, au) in units.iter().enumerate() {
            dec.process(
                PipelinePacket::DataFrame(au_frame(au, i as u64 * 33_000_000)),
                &mut sink,
            )
            .await
            .expect("decode AU");
        }
        dec.process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("flush");

        assert!(
            sink.dims.contains(&(640, 480)) && sink.dims.contains(&(320, 240)),
            "both resolutions decoded, got {:?}",
            sink.dims
        );
        let announced: Vec<(RawVideoFormat, u32, u32)> = sink
            .caps
            .iter()
            .filter_map(|c| match c {
                Caps::RawVideo {
                    format,
                    width: Dim::Fixed(w),
                    height: Dim::Fixed(h),
                    ..
                } => Some((*format, *w, *h)),
                _ => None,
            })
            .collect();
        assert_eq!(
            announced,
            std::vec![
                (RawVideoFormat::Nv12, 640, 480),
                (RawVideoFormat::Nv12, 320, 240)
            ],
            "one CapsChanged per resolution, in stream order"
        );
        assert_eq!(
            dec.decoders_created(),
            1,
            "a shrink within the decoder's ceiling reconfigures in place"
        );
    }

    #[tokio::test]
    async fn hevc_main10_decodes_to_p010_surfaces() {
        let Some(mut dec) = open_dec(VideoCodec::H265, 640, 480, 1) else {
            return;
        };
        let units = h265_units(HEVC_MAIN10);
        assert!(!units.is_empty(), "fixture split into AUs");
        let mut sink = RecordSink::default();
        for (i, au) in units.iter().enumerate() {
            dec.process(
                PipelinePacket::DataFrame(au_frame(au, i as u64 * 33_000_000)),
                &mut sink,
            )
            .await
            .expect("decode AU");
        }
        dec.process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("flush");

        assert!(!sink.dims.is_empty(), "10-bit stream produced frames");
        assert_eq!(
            sink.caps,
            std::vec![Caps::RawVideo {
                format: RawVideoFormat::P010,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            }],
            "a 10-bit stream announces the P010 surface"
        );
        assert!(sink.dims.iter().all(|&d| d == (640, 480)));
        // 16-bit samples with the value in the top 10 bits: the low 6 bits of
        // each little-endian word are zero, and the picture is not flat.
        let head = sink.luma_head.expect("luma read back");
        assert!(
            head.as_chunks::<2>().0.iter().all(|w| w[0] & 0x3f == 0),
            "P010 samples sit in the top 10 bits, got {:?}",
            &head[..8]
        );
        let samples: Vec<u16> = head
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| u16::from_le_bytes([w[0], w[1]]) >> 6)
            .collect();
        assert!(
            samples.iter().any(|&s| s != samples[0]),
            "decoded luma is uniform; decode likely failed"
        );
    }

    #[tokio::test]
    async fn av1_decodes_on_gpu() {
        let Some(mut dec) = open_dec(VideoCodec::Av1, 640, 480, 1) else {
            return;
        };
        let units = av1_temporal_units(AV1_CLIP);
        assert!(units.len() > 1, "fixture split into temporal units");
        let mut sink = RecordSink::default();
        for (i, tu) in units.iter().enumerate() {
            if dec
                .process(
                    PipelinePacket::DataFrame(au_frame(tu, i as u64 * 33_000_000)),
                    &mut sink,
                )
                .await
                .is_err()
            {
                // Pre-Ampere NVDEC has no AV1 decoder: the first sequence fails.
                std::eprintln!("skipping: no AV1 decode on this GPU");
                return;
            }
        }
        dec.process(PipelinePacket::Eos, &mut sink)
            .await
            .expect("flush");

        assert!(sink.dims.len() > 1, "AV1 clip decoded to several frames");
        assert!(sink.dims.iter().all(|&d| d == (640, 480)));
        assert_eq!(
            sink.caps,
            std::vec![Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            }]
        );
        let head = sink.luma_head.expect("luma read back");
        assert!(
            head.iter().any(|&b| b != head[0]),
            "decoded AV1 luma is uniform; decode likely failed"
        );
    }

    #[tokio::test]
    async fn display_delay_lags_output_by_the_configured_frames() {
        // Same stream, two display delays: the deeper one holds frames back in
        // the parser, so fewer have been emitted by the time the input ends.
        async fn decode(delay: u32) -> Option<(usize, usize)> {
            let mut dec = open_dec(VideoCodec::H264, 640, 480, delay)?;
            let units = h264_units(H264_CLIP);
            let mut sink = RecordSink::default();
            for (i, au) in units.iter().enumerate() {
                dec.process(
                    PipelinePacket::DataFrame(au_frame(au, i as u64 * 33_000_000)),
                    &mut sink,
                )
                .await
                .expect("decode AU");
            }
            let before_flush = sink.dims.len();
            dec.process(PipelinePacket::Eos, &mut sink)
                .await
                .expect("flush");
            Some((before_flush, sink.dims.len()))
        }

        let Some((low_before, low_total)) = decode(1).await else {
            return;
        };
        let (deep_before, deep_total) = decode(8).await.expect("second decoder");
        assert!(
            low_total > 0 && low_total == deep_total,
            "same frame count either way, got {low_total} vs {deep_total}"
        );
        // The parser holds frames back before displaying; how many it can hold is
        // also bounded by the decode-surface pool, so assert the lag, not a count.
        assert!(
            low_before > deep_before,
            "a deeper display delay must lag: {low_before} of {low_total} emitted at delay 1 vs {deep_before} at delay 8"
        );
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
    /// read a decoded luma plane back for verification. The allocate / upload
    /// half is only used by the `nvenc` round trips.
    #[allow(unreachable_pub, dead_code)]
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
        // One NVENC session at a time across this binary's test threads.
        let _lock = crate::nvenc::tests::encode_session_lock().await;
        use g2g_core::frame::Frame;
        use g2g_core::memory::SystemSlice;
        use g2g_core::FrameTiming;

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
                    && cu::cu_memcpy_htod(dptr, host.as_ptr() as *const core::ffi::c_void, size)
                        == 0;
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
                        crate::cudadeviceid::DEFAULT_CUDA_DEVICE_ID,
                        Arc::new(DevAlloc { dptr, ctx }),
                    )),
                    FrameTiming {
                        pts_ns: seq * 33_000_000,
                        ..FrameTiming::default()
                    },
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
            fn poll_push(
                &mut self,
                _cx: &mut core::task::Context<'_>,
                packet_slot: &mut Option<PipelinePacket>,
            ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
                let packet = packet_slot.take().expect("poll_push without a packet");
                core::task::Poll::Ready({
                    if let PipelinePacket::DataFrame(f) = packet {
                        if let Some(s) = f.domain.as_system_slice() {
                            self.aus.push(s.to_vec());
                        }
                    }
                    Ok(g2g_core::PushOutcome::Accepted)
                })
            }
        }

        let nv12_caps = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
            interlace: g2g_core::Interlace::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
        };
        let mut enc = NvEnc::new().with_codec(codec);
        enc.configure_pipeline(&nv12_caps).unwrap();
        let mut au_sink = AuSink::default();
        for i in 0..10u64 {
            let Some(frame) = make_frame(i) else {
                std::eprintln!("skipping: CUDA alloc/upload failed");
                return;
            };
            if enc
                .process(PipelinePacket::DataFrame(frame), &mut au_sink)
                .await
                .is_err()
            {
                std::eprintln!("skipping: NVENC unavailable on this host");
                return;
            }
        }
        enc.process(PipelinePacket::Eos, &mut au_sink)
            .await
            .unwrap();
        assert!(
            !au_sink.aus.is_empty(),
            "NVENC produced access units to decode"
        );

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
            fn poll_push(
                &mut self,
                _cx: &mut core::task::Context<'_>,
                packet_slot: &mut Option<PipelinePacket>,
            ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
                let packet = packet_slot.take().expect("poll_push without a packet");
                core::task::Poll::Ready({
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
                    Ok(g2g_core::PushOutcome::Accepted)
                })
            }
        }

        let mut dec = NvDec::new();
        let in_caps = Caps::CompressedVideo {
            codec,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
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
            dec.process(PipelinePacket::DataFrame(f), &mut cuda_sink)
                .await
                .expect("decode AU");
        }
        dec.process(PipelinePacket::Eos, &mut cuda_sink)
            .await
            .expect("flush decoder");

        assert!(
            cuda_sink.count > 0,
            "NvDec produced decoded NV12 CUDA frames"
        );
        assert_eq!(
            cuda_sink.caps,
            std::vec![Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(W),
                height: Dim::Fixed(H),
                framerate: Rate::Fixed(30 << 16),
                interlace: g2g_core::Interlace::Any,
                colorimetry: g2g_core::Colorimetry::UNKNOWN
            }],
            "NV12 output caps announced once at the decoded geometry"
        );
        assert!(
            cuda_sink.dims.iter().all(|&d| d == (W, H)),
            "every decoded frame is {W}x{H}, got {:?}",
            cuda_sink.dims
        );
        assert!(
            cuda_sink.luma_varied,
            "decoded luma holds real (non-uniform) content"
        );
    }
}
