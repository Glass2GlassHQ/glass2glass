//! Linux H.264 decode element using ffmpeg / libavcodec.
//!
//! M13 (Linux production path): consumes Annex-B H.264 `DataFrame`s (the
//! bitstream `RtspSrc` / `H264Parse` already emit, `MemoryDomain::System`)
//! and produces decoded I420 frames, also `MemoryDomain::System` (CPU copy
//! out of libavcodec's frame buffer). A `CapsChanged(I420, w, h)` is emitted
//! before the first decoded frame and again whenever the decoder signals a
//! resolution change.
//!
//! Why this element exists alongside `VaapiH264Dec`: cros-codecs 0.0.6
//! cannot allocate decoder output surfaces on AMD desktop GPUs (see
//! `vaapidec.rs` module header for the full diagnosis). libavcodec's
//! software H.264 decoder works on every Linux system out of the box; this
//! is the production-ready baseline.
//!
//! NVDEC (NVIDIA) hardware decode is available via the same element by
//! constructing with [`Backend::NvdecCuvid`]: libavcodec ships an
//! `h264_cuvid` standalone codec (not a hwaccel hook), so we just swap the
//! codec lookup. The decoder emits `Pixel::NV12` straight to system memory,
//! so no `AVHWDeviceContext` setup or `av_hwframe_transfer_data` plumbing is
//! needed. The `AsyncElement` shape is identical to the software backend;
//! output caps are still NV12 / I420 from the caller's chosen
//! [`OutputFormat`]. Requires the libavcodec build to include `h264_cuvid`
//! (Fedora `ffmpeg-free` includes it; check `ffmpeg -decoders | grep cuvid`)
//! and a working NVIDIA driver + `libnvcuvid.so` at runtime.
//!
//! VAAPI hwaccel through ffmpeg (`h264_vaapi` + `AV_HWDEVICE_TYPE_VAAPI`)
//! follows the same pattern as a future fourth backend variant, but does
//! need an `AVHWDeviceContext` and a `get_format` hook (it is a true
//! hwaccel, not a standalone codec like cuvid).
//!
//! [`Backend::NvdecCuda`] (C3) is that true-hwaccel path for NVIDIA: instead
//! of the standalone `h264_cuvid` codec (which copies NV12 back to system
//! memory), it attaches an `AV_HWDEVICE_TYPE_CUDA` device to the generic
//! `h264` decoder and registers a `get_format` hook that selects
//! `AV_PIX_FMT_CUDA`. Decoded NV12 then stays resident in GPU memory: each
//! frame is emitted as `MemoryDomain::Cuda`, carrying the two NV12 plane
//! device pointers and the `CUcontext`, with the owning `AVFrame` boxed as
//! the buffer's `CudaKeepAlive` so the pointers stay valid until a downstream
//! GPU consumer drops the frame. This removes cuvid's device->host copy; the
//! payoff lands when a CUDA-consuming sink (Phase 3) takes the handoff.
//! Output is always NV12 (the decoder's native device layout); requesting
//! `OutputFormat::I420` with this backend is rejected loud at configure time
//! (a GPU colour convert would be needed and is out of scope here).
//!
//! Pipeline:
//!
//! ```text
//! RtspSrc ─► H264Parse ─► FfmpegH264Dec ─► [downstream sink / ML]
//!  (System/H264 Annex-B)        (System/I420)
//! ```
//!
//! Threading: `ffmpeg::decoder::Video` wraps a raw `AVCodecContext*`, which
//! is `!Send` and `!Sync` by default. The element is moved between worker
//! threads but never shared (the runner holds at most one `&mut self`
//! reference at a time), so an `unsafe impl Send` is sound on the same
//! grounds as `MfDecode` and `VaapiH264Dec`: ownership transfer, never
//! aliasing.
//!
//! Output format: I420 by default; NV12 selectable via
//! [`FfmpegH264Dec::with_output_format`]. NV12 is what KMS overlay planes
//! prefer, so the `KmsSink` path opts into it. The conversion is a direct
//! interleave of the U/V planes after the YUV420P decode (no swscale).
//!
//! Deferred:
//! - VAAPI hwaccel: open `h264_vaapi` codec with an attached
//!   `AVHWDeviceContext(VAAPI)`, register `get_format` to claim
//!   `AV_PIX_FMT_VAAPI`, and `av_hwframe_transfer_data` the decoded surface
//!   into System memory. This stays inside this module — public shape
//!   (`AsyncElement`, input/output caps) doesn't change.
//! - YUV444P / 10-bit pixel formats. Mainline H.264 cameras emit YUV420P;
//!   other formats are rejected with `CapsMismatch` so the failure is loud.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use ffmpeg_next as ffmpeg;
use ffmpeg::codec::{self, Flags, Id};
use ffmpeg::format::Pixel;
use ffmpeg::frame::Video as FfVideo;
use ffmpeg::packet::Packet;
use ffmpeg::Dictionary;
use ffmpeg::Error as FfError;

use g2g_core::frame::Frame;
use g2g_core::memory::{OwnedCudaBuffer, SystemSlice};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, CudaKeepAlive,
    Dim, FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink, PipelinePacket, Rate,
    VideoCodec, RawVideoFormat,
};

/// Pixel layout emitted on the decoder's output side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Planar Y / U / V (default). Byte length = `w*h + 2 * ceil(w/2) * ceil(h/2)`.
    I420,
    /// Y plane followed by interleaved U/V (NV12). Same total byte length
    /// as I420; what KMS overlay planes (and many GPU samplers) prefer.
    Nv12,
}

impl OutputFormat {
    fn raw_format(self) -> RawVideoFormat {
        match self {
            OutputFormat::I420 => RawVideoFormat::I420,
            OutputFormat::Nv12 => RawVideoFormat::Nv12,
        }
    }
}

/// libavcodec decoder backend. The element shape (input H.264 Annex-B,
/// output NV12 / I420 in system memory) is identical across variants —
/// only the codec used internally and the path through libavcodec change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// libavcodec's built-in software H.264 decoder. Works everywhere
    /// libavcodec is present. Default.
    Software,
    /// NVIDIA NVDEC via the `h264_cuvid` standalone codec. Requires the
    /// libavcodec build to include `h264_cuvid` and a working NVIDIA
    /// driver (`libnvcuvid.so`) at runtime. cuvid emits NV12 directly to
    /// system memory, so no AVHWDeviceContext / hwframe transfer is
    /// needed; the only difference from the software path is the source
    /// `AVFrame`'s pixel format.
    NvdecCuvid,
    /// NVIDIA NVDEC via the generic `h264` decoder with an attached
    /// `AV_HWDEVICE_TYPE_CUDA` device and a `get_format` hook selecting
    /// `AV_PIX_FMT_CUDA`. Decoded NV12 stays in GPU memory; frames are
    /// emitted as `MemoryDomain::Cuda` for a zero-copy handoff to a CUDA
    /// consumer. Output is always NV12; `OutputFormat::I420` is rejected
    /// at configure time. Requires libavcodec built with the CUDA hwaccel
    /// and a working NVIDIA driver at runtime.
    NvdecCuda,
}

/// One decoded picture. Either the pixels already copied out of the
/// libavcodec frame into system memory (Software / NvdecCuvid), or a CUDA
/// device buffer still resident on the GPU (NvdecCuda).
struct DecodedPicture {
    payload: DecodedPayload,
    width: u32,
    height: u32,
    pts_ns: u64,
    /// Source-side wall-clock stamp threaded through from the input
    /// frame so glass-to-glass latency survives decode. Looked up in
    /// `pts_to_arrival` after libavcodec echoes the input pts back.
    arrival_ns: u64,
}

/// Where a decoded picture's pixels live.
enum DecodedPayload {
    /// Packed `OutputFormat` bytes in system memory (CPU decode path or
    /// cuvid's system-memory output).
    System(Box<[u8]>),
    /// NV12 still in CUDA device memory (NvdecCuda zero-copy path).
    Cuda(OwnedCudaBuffer),
}

pub struct FfmpegH264Dec {
    decoder: Option<ffmpeg::decoder::Video>,
    last_caps: Option<Caps>,
    configured: bool,
    emitted: u64,
    output_format: OutputFormat,
    backend: Backend,
    /// Number of internal output surfaces requested from `h264_cuvid`.
    /// `None` keeps the cuvid default (25 — throughput-oriented, adds
    /// ~25 frames of in-decoder latency). The `NvdecCuvid` constructor
    /// path defaults this to `Some(4)` for low-latency live use; callers
    /// who need throughput at the cost of latency can override via
    /// [`FfmpegH264Dec::with_cuvid_surfaces`]. Ignored on the software
    /// backend.
    cuvid_surfaces: Option<u32>,
    /// Set `AV_CODEC_FLAG_LOW_DELAY` at codec-open time. Tells the
    /// decoder to emit each picture as soon as it's decoded rather than
    /// holding it for reorder. Defaulted on for NVDEC (cuvid otherwise
    /// happily pipelines several frames before releasing the first), off
    /// for software (correctness on B-frame streams).
    low_delay: bool,
    /// M16 workaround #3 Phase A: most recent input caps received via
    /// `PipelinePacket::CapsChanged`. Used to validate the format on
    /// mid-stream changes and to debug-assert that the decode-time
    /// output geometry agrees with the declared `DerivedOutput`
    /// closure. Phase B will use this as the input to a runner-side
    /// downstream subgraph re-solve.
    input_caps: Option<Caps>,
    /// Map input pts -> input arrival_ns. Survives the B-frame
    /// reordering libavcodec does internally because we key on pts,
    /// which the codec layer echoes verbatim.
    pts_to_arrival: alloc::collections::BTreeMap<u64, u64>,
    /// `CUcontext` (as `u64`) of the CUDA hwdevice created for the
    /// `NvdecCuda` backend, extracted at `configure_pipeline`. `0` for the
    /// system-memory backends. Stamped onto every emitted `OwnedCudaBuffer`
    /// so a consumer can push the right context before touching the memory.
    cuda_context: u64,
    /// M12 / C3 step 3: the downstream consumer's allocation proposal,
    /// recorded in `configure_allocation`. A `MemoryDomainKind::Cuda` request
    /// from a GPU sink (`CudaGlSink`) is satisfied by construction on the
    /// `NvdecCuda` backend (it already emits device-resident frames). The
    /// `min_buffers` hint sizes the CUDA hwframe pool's `extra_hw_frames` at
    /// open time: the runner's M12 allocation query now runs before
    /// `configure_pipeline`, so this is recorded by the time the decoder opens.
    requested_alloc: Option<AllocationParams>,
}

// SAFETY: `ffmpeg::decoder::Video` wraps a raw `*mut AVCodecContext` and is
// `!Send` by default. The multi-thread runner requires `Send` so it can move
// the element between worker tasks. We uphold that by construction: the
// runner drives the element through `&mut self` (never concurrently), so the
// context is owned and moved, never aliased.
unsafe impl Send for FfmpegH264Dec {}

impl core::fmt::Debug for FfmpegH264Dec {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FfmpegH264Dec")
            .field("configured", &self.configured)
            .field("emitted", &self.emitted)
            .finish()
    }
}

impl Default for FfmpegH264Dec {
    fn default() -> Self {
        Self::new()
    }
}

impl FfmpegH264Dec {
    pub fn new() -> Self {
        Self {
            decoder: None,
            last_caps: None,
            configured: false,
            emitted: 0,
            output_format: OutputFormat::I420,
            backend: Backend::Software,
            cuvid_surfaces: None,
            low_delay: false,
            input_caps: None,
            pts_to_arrival: alloc::collections::BTreeMap::new(),
            cuda_context: 0,
            requested_alloc: None,
        }
    }

    /// The downstream consumer's recorded M12 allocation proposal, if any
    /// (see [`AsyncElement::configure_allocation`]).
    pub fn requested_alloc(&self) -> Option<AllocationParams> {
        self.requested_alloc
    }

    /// Whether this backend emits frames in CUDA device memory
    /// (`MemoryDomain::Cuda`) rather than copying them to system memory.
    fn outputs_cuda(&self) -> bool {
        matches!(self.backend, Backend::NvdecCuda)
    }

    /// Switch the output layout. NV12 is required by `KmsSink` and most GPU
    /// samplers; I420 is the default for backwards compatibility with
    /// existing tests and the documented Linux ML path.
    pub fn with_output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    pub fn output_format(&self) -> OutputFormat {
        self.output_format
    }

    /// Select the libavcodec backend. Defaults to [`Backend::Software`].
    /// [`Backend::NvdecCuvid`] opens `h264_cuvid` at `configure_pipeline`
    /// time and fails loud (`HardwareError::Other`) if the libavcodec
    /// build doesn't include it or the NVIDIA driver isn't reachable.
    ///
    /// Switching *to* `NvdecCuvid` from the default also enables
    /// latency-oriented tuning: `surfaces=4` (down from cuvid's default
    /// 25, which costs ~25 frames of in-decoder buffering) and the
    /// `AV_CODEC_FLAG_LOW_DELAY` codec flag (release each picture as
    /// soon as it's decoded). Switching back to `Software` clears those
    /// defaults. Override either explicitly via
    /// [`Self::with_cuvid_surfaces`] or [`Self::with_low_delay`] *after*
    /// `with_backend`.
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        match backend {
            Backend::Software => {
                self.cuvid_surfaces = None;
                self.low_delay = false;
            }
            Backend::NvdecCuvid => {
                self.cuvid_surfaces = Some(4);
                self.low_delay = true;
            }
            Backend::NvdecCuda => {
                // The generic CUDA hwaccel ignores the cuvid-private
                // `surfaces` option; pool depth is controlled via
                // `extra_hw_frames` at open time instead. Low-delay still
                // applies (release each picture as soon as decoded). The
                // device frame is NV12, so force the output layout to match
                // what we emit; an I420 request is rejected at configure.
                self.cuvid_surfaces = None;
                self.low_delay = true;
                self.output_format = OutputFormat::Nv12;
            }
        }
        self
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Override the `h264_cuvid` `surfaces` AVOption. Higher = more
    /// throughput headroom but more in-decoder latency (each extra
    /// surface is one extra frame the decoder may hold before releasing
    /// output). `None` reverts to the cuvid default (25). Ignored on the
    /// software backend.
    pub fn with_cuvid_surfaces(mut self, surfaces: Option<u32>) -> Self {
        self.cuvid_surfaces = surfaces;
        self
    }

    pub fn cuvid_surfaces(&self) -> Option<u32> {
        self.cuvid_surfaces
    }

    /// Set `AV_CODEC_FLAG_LOW_DELAY`. Defaults: on for NVDEC (the only
    /// way to keep cuvid from holding several frames before output),
    /// off for software. Setting this on a sw path with reordered
    /// streams (B-frames) is normally not what you want.
    pub fn with_low_delay(mut self, on: bool) -> Self {
        self.low_delay = on;
        self
    }

    pub fn low_delay(&self) -> bool {
        self.low_delay
    }

    /// Count of decoded `DataFrame`s pushed downstream. Useful in tests.
    pub fn decoded_count(&self) -> u64 {
        self.emitted
    }

    /// Send one access unit to the decoder and drain whatever it is ready
    /// to release. libavcodec buffers for B-frame reordering, so early
    /// inputs commonly yield zero outputs.
    fn feed_access_unit(
        &mut self,
        bitstream: &[u8],
        pts_ns: u64,
        arrival_ns: u64,
        decoded: &mut Vec<DecodedPicture>,
    ) -> Result<(), G2gError> {
        let mut packet = Packet::copy(bitstream);
        // libavcodec uses the packet's PTS verbatim; the unit is opaque to
        // the codec layer and is echoed back on the decoded frame. We feed
        // nanoseconds straight through.
        packet.set_pts(Some(pts_ns as i64));
        packet.set_dts(Some(pts_ns as i64));
        if arrival_ns != 0 {
            self.pts_to_arrival.insert(pts_ns, arrival_ns);
        }

        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
        decoder
            .send_packet(&packet)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.drain_frames(decoded)
    }

    fn drain_frames(&mut self, decoded: &mut Vec<DecodedPicture>) -> Result<(), G2gError> {
        let format = self.output_format;
        let outputs_cuda = self.outputs_cuda();
        let cuda_context = self.cuda_context;
        let decoder = self.decoder.as_mut().ok_or(G2gError::NotConfigured)?;
        loop {
            // Fresh frame per iteration: the CUDA path moves the whole
            // `AVFrame` into the emitted buffer's keep-alive, so it cannot be
            // a reused scratch frame. The system path copies out and drops it.
            let mut frame = FfVideo::empty();
            match decoder.receive_frame(&mut frame) {
                Ok(()) => {
                    // libavcodec returns the PTS we fed in (or AV_NOPTS_VALUE
                    // = INT64_MIN if it could not propagate one); treat the
                    // sentinel as zero so we don't return a wild timestamp.
                    let pts_ns = match frame.pts() {
                        Some(p) if p >= 0 => p as u64,
                        _ => 0,
                    };
                    let width = frame.width();
                    let height = frame.height();
                    let payload = if outputs_cuda {
                        // SAFETY: the NvdecCuda backend decodes into
                        // `AV_PIX_FMT_CUDA` frames; `cuda_context` is the
                        // `CUcontext` its hwdevice was created with. The
                        // helper reads the device pointers and moves the
                        // frame into the buffer's keep-alive.
                        DecodedPayload::Cuda(unsafe {
                            cuda_buffer_from_frame(frame, cuda_context)?
                        })
                    } else {
                        DecodedPayload::System(copy_yuv420(&frame, format)?)
                    };
                    let arrival_ns = self.pts_to_arrival.remove(&pts_ns).unwrap_or(0);
                    decoded.push(DecodedPicture {
                        payload,
                        width,
                        height,
                        pts_ns,
                        arrival_ns,
                    });
                }
                Err(FfError::Other { errno }) if errno == ffmpeg::error::EAGAIN => {
                    // Need more input.
                    return Ok(());
                }
                Err(FfError::Eof) => return Ok(()),
                Err(_) => return Err(G2gError::Hardware(HardwareError::Other)),
            }
        }
    }

    fn drain_eos(&mut self, decoded: &mut Vec<DecodedPicture>) -> Result<(), G2gError> {
        if let Some(d) = self.decoder.as_mut() {
            d.send_eof()
                .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        }
        self.drain_frames(decoded)
    }
}

impl AsyncElement for FfmpegH264Dec {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        let supported = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        upstream_caps.intersect(&supported)
    }

    /// M16 step 5k: native `DerivedOutput` — accepts H.264 with any
    /// geometry and produces NV12 or I420 (chosen at construction) at
    /// the same dims and framerate. The closure validates the input
    /// format and returns an empty set on mismatch, so the solver
    /// rejects non-H.264 upstream at negotiation time instead of via
    /// the dynamic `intercept_caps` callback. Mixed chains containing
    /// this decoder now get real per-link caps from the solver: H.264
    /// to the decoder, NV12/I420 to the sink. Coupled with 5j (NV12
    /// sinks tolerate mid-stream dim changes), the production
    /// `rtsp → ffmpegdec → wayland/kms` chain switches from the
    /// legacy single-fixated cascade to the per-link path without
    /// regression.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        let out_fmt = self.output_format.raw_format();
        CapsConstraint::DerivedOutput(alloc::boxed::Box::new(move |input: &Caps| {
            derive_output_caps(input, out_fmt)
        }))
    }

    /// M12 / C3 step 3: record the downstream consumer's allocation proposal.
    /// A `MemoryDomainKind::Cuda` request (from `CudaGlSink`) is honoured by
    /// construction on the `NvdecCuda` backend, which already emits
    /// device-resident frames; the other backends emit system memory, so a
    /// Cuda request there is simply unsatisfiable and stays recorded for
    /// diagnostics rather than silently changing the output domain.
    fn configure_allocation(&mut self, params: &AllocationParams) {
        self.requested_alloc = Some(*params);
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::CompressedVideo {
                codec: VideoCodec::H264,
                ..
            } => {}
            _ => return Err(G2gError::CapsMismatch),
        }

        // ffmpeg::init() registers codecs once per process; calling it
        // repeatedly is safe and cheap.
        ffmpeg::init().map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // NvdecCuda needs NV12 out (the device frame's native layout); reject
        // an I420 request loud rather than silently emit mismatched caps.
        if self.backend == Backend::NvdecCuda && self.output_format != OutputFormat::Nv12 {
            return Err(G2gError::CapsMismatch);
        }

        let codec = match self.backend {
            // The generic `h264` decoder hosts the CUDA hwaccel; NvdecCuda
            // attaches a CUDA device + get_format hook to it below.
            Backend::Software | Backend::NvdecCuda => codec::decoder::find(Id::H264),
            // `h264_cuvid` is a standalone codec entry, not a hwaccel hook,
            // so it's found by name rather than by AVCodecID. If absent, the
            // libavcodec build wasn't compiled with `--enable-cuvid` or
            // libnvcuvid isn't loadable at runtime; either way the right
            // answer is to fail loud so the caller picks Software explicitly.
            Backend::NvdecCuvid => codec::decoder::find_by_name("h264_cuvid"),
        }
        .ok_or(G2gError::Hardware(HardwareError::Other))?;

        let mut decoder_ctx = codec::decoder::new();
        if self.low_delay {
            // `AV_CODEC_FLAG_LOW_DELAY` tells the codec to release each
            // picture as soon as it's decoded rather than holding it for
            // reorder. Essential on cuvid (otherwise the internal pipeline
            // depth dominates p50); set explicitly here so the policy is
            // visible alongside the surface count.
            decoder_ctx.set_flags(Flags::LOW_DELAY);
        }
        // cuvid's tunables live as AVOptions on the codec's private data,
        // applied at `avcodec_open2` via an `AVDictionary`. The `surfaces`
        // option is the dominant latency knob — default 25 ~= 25 frames
        // of in-decoder buffering. The software backend ignores cuvid
        // options, so it's harmless to leave the dictionary empty there.
        let mut opts = Dictionary::new();
        if self.backend == Backend::NvdecCuvid {
            if let Some(n) = self.cuvid_surfaces {
                let v = alloc::format!("{n}");
                opts.set("surfaces", &v);
            }
        }

        // NvdecCuda: create a CUDA hwdevice, hand the generic h264 decoder a
        // reference to it plus a get_format hook that selects AV_PIX_FMT_CUDA,
        // so decoded NV12 stays in device memory. Done on the raw
        // AVCodecContext before open, the canonical `hw_decode.c` setup.
        if self.backend == Backend::NvdecCuda {
            // SAFETY: standard ffmpeg hwaccel init on a freshly-allocated,
            // not-yet-opened context. `av_hwdevice_ctx_create` initialises
            // `hw_device_ctx`; we read the CUcontext out of the device's
            // hwctx, hand the codec its own ref, drop ours, and install the
            // get_format callback. A successful create guarantees the device
            // `data` and CUDA `hwctx` are non-null, so only `hw_device_ctx`
            // itself is checked explicitly below.
            unsafe {
                let mut hw_device_ctx: *mut ffmpeg::ffi::AVBufferRef = core::ptr::null_mut();
                let ret = ffmpeg::ffi::av_hwdevice_ctx_create(
                    &mut hw_device_ctx,
                    ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
                    core::ptr::null(),     // default CUDA device
                    core::ptr::null_mut(), // no options
                    0,
                );
                if ret < 0 || hw_device_ctx.is_null() {
                    return Err(G2gError::Hardware(HardwareError::Other));
                }

                // Walk the device buffer to its AVCUDADeviceContext and read
                // the CUcontext. The device stays alive for the decoder's
                // lifetime via the codec's own ref taken below, so this
                // pointer remains valid for every frame we emit.
                let dev_ctx = (*hw_device_ctx).data as *const ffmpeg::ffi::AVHWDeviceContext;
                let cuda_dev = (*dev_ctx).hwctx as *const AVCUDADeviceContextHead;
                self.cuda_context = (*cuda_dev).cuda_ctx as u64;

                let raw = decoder_ctx.as_mut_ptr();
                (*raw).hw_device_ctx = ffmpeg::ffi::av_buffer_ref(hw_device_ctx);
                (*raw).get_format = Some(get_cuda_format);
                // Pool headroom: the decoder must keep enough surfaces for the
                // frames we hold downstream (link capacity) plus its own
                // reorder/reference set. Without this the pool can starve once
                // a few frames are in flight to the sink. The downstream
                // consumer's M12 proposal (now recorded before this open, via
                // the allocation-query reorder) carries its hold count in
                // `min_buffers`; size the pool to that plus a reorder margin,
                // falling back to 8 when no consumer proposed.
                const REORDER_MARGIN: usize = 4;
                let headroom = self
                    .requested_alloc
                    .map(|p| (p.min_buffers + REORDER_MARGIN) as i32)
                    .unwrap_or(8);
                (*raw).extra_hw_frames = headroom;

                // The codec now owns a ref; release ours so only the decoder
                // keeps the device alive.
                ffmpeg::ffi::av_buffer_unref(&mut hw_device_ctx);
            }
        }

        let decoder = decoder_ctx
            .open_as_with(codec, opts)
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?
            .video()
            .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        self.decoder = Some(decoder);
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
                    self.feed_access_unit(
                        slice.as_slice(),
                        frame.timing.pts_ns,
                        frame.timing.arrival_ns,
                        &mut decoded,
                    )?;
                }
                PipelinePacket::CapsChanged(c) => {
                    // M16 workaround #3 Phase A: validate the format
                    // (loud-reject an incompatible mid-stream switch like
                    // H.264 -> VP9, which previously was silently dropped)
                    // and record `c` as the current input caps. The
                    // ordering invariant (§3) is preserved: we still emit
                    // our own output `CapsChanged` at the decode boundary
                    // from decoded-frame geometry, not eagerly here.
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
                    if let Some(d) = self.decoder.as_mut() {
                        d.flush();
                    }
                    self.last_caps = None;
                    out.push(PipelinePacket::Flush).await?;
                    return Ok(());
                }
                PipelinePacket::Eos => {
                    self.drain_eos(&mut decoded)?;
                }
            }

            let out_format = self.output_format;
            for d in decoded {
                let new_caps = yuv420_caps(out_format, d.width, d.height);
                if self.last_caps.as_ref() != Some(&new_caps) {
                    // M16 workaround #3 Phase A debug assertion: the
                    // decode-time output caps must be consistent with
                    // the declared `DerivedOutput` closure applied to
                    // the most recently recorded input caps. They need
                    // not be equal (decode-time leaves framerate `Any`,
                    // closure may carry a `Fixed`); they must overlap
                    // under `Caps::intersect`. Disagreement here means
                    // the closure is buggy or the upstream caps are
                    // stale.
                    #[cfg(debug_assertions)]
                    if let Some(input) = self.input_caps.as_ref() {
                        let expected = derive_output_caps(input, out_format.raw_format());
                        debug_assert!(
                            !expected
                                .intersect(&CapsSet::one(new_caps.clone()))
                                .is_empty(),
                            "ffmpegdec decode-time output {new_caps:?} inconsistent with derive_output_caps({input:?}) = {expected:?}"
                        );
                    }
                    out.push(PipelinePacket::CapsChanged(new_caps.clone())).await?;
                    self.last_caps = Some(new_caps.clone());
                }
                let domain = match d.payload {
                    DecodedPayload::System(bytes) => {
                        MemoryDomain::System(SystemSlice::from_boxed(bytes))
                    }
                    DecodedPayload::Cuda(buf) => MemoryDomain::Cuda(buf),
                };
                let frame = Frame {
                    domain,
                    timing: FrameTiming {
                        pts_ns: d.pts_ns,
                        dts_ns: d.pts_ns,
                        duration_ns: 0,
                        capture_ns: d.pts_ns,
                        arrival_ns: d.arrival_ns,
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

/// Single source of truth for the decoder's output-side caps derivation.
/// Called by the `DerivedOutput` constraint closure (startup negotiation)
/// and the workaround-#3 Phase A debug assertion (mid-stream consistency
/// check between recorded input caps and decode-time output geometry).
///
/// Non-H.264 input yields an empty `CapsSet`: the solver treats that as a
/// negotiation failure; the runtime mid-stream check refuses
/// `CapsMismatch` before it ever reaches here.
fn derive_output_caps(input: &Caps, out_fmt: RawVideoFormat) -> CapsSet {
    match input {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width,
            height,
            framerate,
        } => CapsSet::one(Caps::RawVideo {
            format: out_fmt,
            width: width.clone(),
            height: height.clone(),
            framerate: framerate.clone(),
        }),
        _ => CapsSet::from_alternatives(alloc::vec::Vec::new()),
    }
}

fn yuv420_caps(format: OutputFormat, w: u32, h: u32) -> Caps {
    Caps::RawVideo {
        format: format.raw_format(),
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Any,
    }
}

/// Copy 8-bit 4:2:0 YUV pixels out of a libavcodec frame into a packed
/// `width * height * 3 / 2` buffer in either I420 or NV12 layout.
///
/// - **I420**: Y (w*h), then U (cw*ch), then V (cw*ch).
/// - **NV12**: Y (w*h), then interleaved UV pairs (2*cw*ch).
///
/// where `cw = ceil(w/2)` and `ch = ceil(h/2)`. Source plane pitches may
/// exceed the visible width due to libavcodec's alignment, so rows are
/// copied individually.
///
/// Two source pixel layouts are accepted: planar `YUV420P` (the software
/// H.264 decoder's native output) and semi-planar `NV12` (h264_cuvid's
/// native output). Any other format is rejected loud — those streams need
/// a `ColorConvert` element upstream of any I420/NV12 consumer.
fn copy_yuv420(frame: &FfVideo, format: OutputFormat) -> Result<Box<[u8]>, G2gError> {
    let src = match frame.format() {
        // YUVJ420P is YUV420P with JPEG (full) range. Same plane layout, so
        // accept it; range fidelity is preserved in the pixel values and can
        // be advertised by a future colour-metadata field on `Caps::Video`.
        Pixel::YUV420P | Pixel::YUVJ420P => SourceLayout::Planar420,
        Pixel::NV12 => SourceLayout::SemiPlanar420,
        _ => return Err(G2gError::CapsMismatch),
    };
    let required_planes = match src {
        SourceLayout::Planar420 => 3,
        SourceLayout::SemiPlanar420 => 2,
    };
    if frame.planes() < required_planes {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let w = frame.width() as usize;
    let h = frame.height() as usize;
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let y_size = w * h;
    let c_size = cw * ch;
    let total = y_size + 2 * c_size;

    let mut out = alloc::vec![0u8; total];

    // Y plane (full resolution).
    let y_src = frame.data(0);
    let y_pitch = frame.stride(0);
    for row in 0..h {
        let src_off = row * y_pitch;
        let dst_off = row * w;
        out[dst_off..dst_off + w].copy_from_slice(&y_src[src_off..src_off + w]);
    }
    match (src, format) {
        // Planar source -> planar I420: copy U then V at half-res.
        (SourceLayout::Planar420, OutputFormat::I420) => {
            let u_src = frame.data(1);
            let u_pitch = frame.stride(1);
            let v_src = frame.data(2);
            let v_pitch = frame.stride(2);
            for row in 0..ch {
                let dst_off = y_size + row * cw;
                out[dst_off..dst_off + cw]
                    .copy_from_slice(&u_src[row * u_pitch..row * u_pitch + cw]);
            }
            for row in 0..ch {
                let dst_off = y_size + c_size + row * cw;
                out[dst_off..dst_off + cw]
                    .copy_from_slice(&v_src[row * v_pitch..row * v_pitch + cw]);
            }
        }
        // Planar source -> semi-planar NV12: interleave U and V.
        (SourceLayout::Planar420, OutputFormat::Nv12) => {
            let u_src = frame.data(1);
            let u_pitch = frame.stride(1);
            let v_src = frame.data(2);
            let v_pitch = frame.stride(2);
            for row in 0..ch {
                let u_row = &u_src[row * u_pitch..row * u_pitch + cw];
                let v_row = &v_src[row * v_pitch..row * v_pitch + cw];
                let dst_base = y_size + row * 2 * cw;
                for col in 0..cw {
                    out[dst_base + 2 * col] = u_row[col];
                    out[dst_base + 2 * col + 1] = v_row[col];
                }
            }
        }
        // Semi-planar source (NV12) -> NV12 output: row copy of the
        // interleaved UV plane, honouring source pitch.
        (SourceLayout::SemiPlanar420, OutputFormat::Nv12) => {
            let uv_src = frame.data(1);
            let uv_pitch = frame.stride(1);
            let uv_row_bytes = 2 * cw;
            for row in 0..ch {
                let dst_base = y_size + row * uv_row_bytes;
                out[dst_base..dst_base + uv_row_bytes]
                    .copy_from_slice(&uv_src[row * uv_pitch..row * uv_pitch + uv_row_bytes]);
            }
        }
        // Semi-planar source -> planar I420: de-interleave UV into U then V.
        (SourceLayout::SemiPlanar420, OutputFormat::I420) => {
            let uv_src = frame.data(1);
            let uv_pitch = frame.stride(1);
            for row in 0..ch {
                let uv_row = &uv_src[row * uv_pitch..row * uv_pitch + 2 * cw];
                let u_dst_base = y_size + row * cw;
                let v_dst_base = y_size + c_size + row * cw;
                for col in 0..cw {
                    out[u_dst_base + col] = uv_row[2 * col];
                    out[v_dst_base + col] = uv_row[2 * col + 1];
                }
            }
        }
    }

    Ok(out.into_boxed_slice())
}

#[derive(Clone, Copy)]
enum SourceLayout {
    Planar420,
    SemiPlanar420,
}

/// `get_format` hook for the `NvdecCuda` backend: pick `AV_PIX_FMT_CUDA` if the
/// decoder offers it (keeping frames in device memory), else `AV_PIX_FMT_NONE`
/// so libavcodec fails the format selection loud rather than silently falling
/// back to a software pixel format we don't expect.
///
/// # Safety
/// Called by libavcodec with a valid `*ctx` and a `formats` array terminated
/// by `AV_PIX_FMT_NONE`. We only read the array.
unsafe extern "C" fn get_cuda_format(
    _ctx: *mut ffmpeg::ffi::AVCodecContext,
    mut formats: *const ffmpeg::ffi::AVPixelFormat,
) -> ffmpeg::ffi::AVPixelFormat {
    if formats.is_null() {
        return ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE;
    }
    // SAFETY: the list is non-null and AV_PIX_FMT_NONE-terminated per the
    // get_format contract; we walk it without passing the terminator.
    unsafe {
        while *formats != ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *formats == ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_CUDA {
                return ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_CUDA;
            }
            formats = formats.add(1);
        }
    }
    ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE
}

/// Read the NV12 plane device pointers out of a decoded `AV_PIX_FMT_CUDA`
/// frame and wrap them in an [`OwnedCudaBuffer`], moving the `AVFrame` into the
/// buffer's keep-alive so the device memory stays referenced until a
/// downstream consumer drops the frame.
///
/// # Safety
/// `frame` must be a decoded CUDA hwframe (`format == AV_PIX_FMT_CUDA`) from a
/// decoder whose hwdevice was created with `cuda_context`. The plane pointers
/// are valid for the lifetime of the owned frame.
unsafe fn cuda_buffer_from_frame(
    frame: FfVideo,
    cuda_context: u64,
) -> Result<OwnedCudaBuffer, G2gError> {
    // SAFETY: `as_ptr` yields the owned AVFrame pointer; reading these public
    // fields is sound while we hold the frame. The reads copy into locals, so
    // moving `frame` into the keep-alive afterwards leaves no dangling borrow.
    let (luma_ptr, chroma_ptr, luma_pitch, chroma_pitch, width, height) = unsafe {
        let f = frame.as_ptr();
        (
            (*f).data[0] as u64,
            (*f).data[1] as u64,
            (*f).linesize[0] as u32,
            (*f).linesize[1] as u32,
            (*f).width as u32,
            (*f).height as u32,
        )
    };
    if luma_ptr == 0 || chroma_ptr == 0 {
        // Not a device frame (would-be system fallback); fail loud rather than
        // hand a downstream CUDA consumer a host or null pointer.
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    let keep_alive = Box::new(CudaFrameOwner(frame));
    Ok(OwnedCudaBuffer::new(
        luma_ptr,
        chroma_ptr,
        luma_pitch,
        chroma_pitch,
        width,
        height,
        cuda_context,
        keep_alive,
    ))
}

/// Owns a decoded CUDA `AVFrame` so its device pointers stay valid while the
/// frame travels downstream. Boxed as the [`CudaKeepAlive`] of an
/// [`OwnedCudaBuffer`]; dropping it `av_frame_free`s the frame, returning the
/// surface to the decoder's hwframe pool.
struct CudaFrameOwner(FfVideo);

// SAFETY: an `AVFrame` (like the decoder's `AVCodecContext`) is `!Send` by
// default. We uphold `Send` by construction: the runner moves the frame
// between worker tasks but never shares it, so the frame is owned and moved,
// never aliased, the same contract as the decoder itself.
unsafe impl Send for CudaFrameOwner {}

impl core::fmt::Debug for CudaFrameOwner {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("CudaFrameOwner(<AVFrame>)")
    }
}

impl CudaKeepAlive for CudaFrameOwner {}

/// Leading fields of libavutil's `AVCUDADeviceContext` (from
/// `hwcontext_cuda.h`). We mirror only the head rather than depend on
/// ffmpeg-sys-next having bound the optional CUDA header at build time. The
/// field order (`cuda_ctx`, `stream`) is stable public ABI; we read just
/// `cuda_ctx`.
#[repr(C)]
#[derive(Debug)]
struct AVCUDADeviceContextHead {
    cuda_ctx: *mut core::ffi::c_void,
    stream: *mut core::ffi::c_void,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_constraint_is_derived_output_h264_to_chosen_format() {
        // M16 step 5k: DerivedOutput closure validates H.264 input and
        // emits the configured output format at the same dims/rate.
        let dec = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
        let c = dec.caps_constraint_as_transform();
        let CapsConstraint::DerivedOutput(f) = c else {
            panic!("expected DerivedOutput");
        };
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
        };
        let out = f(&h264);
        assert_eq!(
            out.alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(30 << 16),
            }]
        );

        // Non-H.264 input → empty CapsSet (solver rejects with EmptyLink).
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
        };
        assert!(f(&vp9).is_empty());
    }

    #[test]
    fn i420_caps_are_fixed() {
        assert_eq!(
            yuv420_caps(OutputFormat::I420, 640, 480),
            Caps::RawVideo {
                format: RawVideoFormat::I420,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Any,
            }
        );
    }

    #[test]
    fn nv12_caps_advertise_nv12_format() {
        assert_eq!(
            yuv420_caps(OutputFormat::Nv12, 1280, 720),
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(1280),
                height: Dim::Fixed(720),
                framerate: Rate::Any,
            }
        );
    }

    #[test]
    fn default_output_format_is_i420() {
        assert_eq!(FfmpegH264Dec::new().output_format(), OutputFormat::I420);
    }

    #[test]
    fn with_output_format_overrides_default() {
        let dec = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
        assert_eq!(dec.output_format(), OutputFormat::Nv12);
    }

    #[test]
    fn intercept_rejects_non_h264() {
        let dec = FfmpegH264Dec::new();
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
        let dec = FfmpegH264Dec::new();
        let proposal = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
        };
        assert_eq!(dec.intercept_caps(&proposal), Ok(proposal));
    }

    #[test]
    fn default_backend_is_software() {
        assert_eq!(FfmpegH264Dec::new().backend(), Backend::Software);
    }

    #[test]
    fn with_backend_overrides_default() {
        let dec = FfmpegH264Dec::new().with_backend(Backend::NvdecCuvid);
        assert_eq!(dec.backend(), Backend::NvdecCuvid);
    }

    #[test]
    fn caps_constraint_independent_of_backend() {
        // The NVDEC backend changes the libavcodec codec used internally,
        // not the negotiation surface: input is still H.264, output is
        // still the configured OutputFormat at the same geometry. Solver
        // sees no difference, so chains compose identically.
        let sw = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
        let nv = FfmpegH264Dec::new()
            .with_output_format(OutputFormat::Nv12)
            .with_backend(Backend::NvdecCuvid);
        let h264 = Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
        };
        let CapsConstraint::DerivedOutput(f_sw) = sw.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        let CapsConstraint::DerivedOutput(f_nv) = nv.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        assert_eq!(f_sw(&h264).alternatives(), f_nv(&h264).alternatives());
    }

    #[test]
    fn software_backend_does_not_set_cuvid_defaults() {
        // The latency tuning only makes sense for NVDEC; the software
        // path keeps libavcodec's defaults so existing behavior and
        // existing CHANGELOG entries are unchanged.
        let dec = FfmpegH264Dec::new();
        assert!(!dec.low_delay());
        assert_eq!(dec.cuvid_surfaces(), None);
    }

    #[test]
    fn nvdec_backend_defaults_to_low_latency_tuning() {
        // surfaces=4 (down from cuvid's 25) and AV_CODEC_FLAG_LOW_DELAY
        // are the two settings that recover the ~80 ms of in-decoder
        // buffering otherwise visible in glass-to-glass numbers. Locking
        // the defaults here so a refactor can't silently revert.
        let dec = FfmpegH264Dec::new().with_backend(Backend::NvdecCuvid);
        assert!(dec.low_delay());
        assert_eq!(dec.cuvid_surfaces(), Some(4));
    }

    #[test]
    fn switching_back_to_software_clears_nvdec_tuning() {
        let dec = FfmpegH264Dec::new()
            .with_backend(Backend::NvdecCuvid)
            .with_backend(Backend::Software);
        assert!(!dec.low_delay());
        assert_eq!(dec.cuvid_surfaces(), None);
    }

    #[test]
    fn cuvid_surfaces_override_survives_after_with_backend() {
        // Override order: with_backend first, then with_cuvid_surfaces
        // (so the override wins over the NVDEC default).
        let dec = FfmpegH264Dec::new()
            .with_backend(Backend::NvdecCuvid)
            .with_cuvid_surfaces(Some(2));
        assert_eq!(dec.cuvid_surfaces(), Some(2));
    }

    #[test]
    fn unconfigured_decoder_reports_zero_decoded() {
        let dec = FfmpegH264Dec::new();
        assert_eq!(dec.decoded_count(), 0);
    }

    #[test]
    fn nvdec_cuda_backend_forces_nv12_and_low_delay() {
        // The CUDA device frame is NV12, so the backend pins the output
        // layout to match what it emits and enables low-delay release. It
        // does not use the cuvid-private `surfaces` knob.
        let dec = FfmpegH264Dec::new().with_backend(Backend::NvdecCuda);
        assert_eq!(dec.backend(), Backend::NvdecCuda);
        assert_eq!(dec.output_format(), OutputFormat::Nv12);
        assert!(dec.low_delay());
        assert_eq!(dec.cuvid_surfaces(), None);
    }

    #[test]
    fn nvdec_cuda_backend_overrides_prior_i420() {
        // with_backend after a with_output_format(I420) still lands on NV12
        // (the backend's requirement wins), so the common build order is safe.
        let dec = FfmpegH264Dec::new()
            .with_output_format(OutputFormat::I420)
            .with_backend(Backend::NvdecCuda);
        assert_eq!(dec.output_format(), OutputFormat::Nv12);
    }

    #[test]
    fn nvdec_cuda_caps_constraint_is_nv12() {
        // Negotiation surface is unchanged from the other backends: H.264 in,
        // NV12 out at the same geometry. Only the memory domain of the emitted
        // frame differs, which caps do not encode.
        let dec = FfmpegH264Dec::new().with_backend(Backend::NvdecCuda);
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
    }

    #[test]
    fn records_downstream_cuda_allocation_proposal() {
        // C3 step 3: a CudaGlSink proposes device-resident buffers; the runner
        // conveys that to the decoder's configure_allocation. The NvdecCuda
        // backend already emits Cuda frames, so the request is honoured by
        // construction; here we assert the proposal is recorded (the handshake
        // the GPU path's allocation query depends on).
        use g2g_core::MemoryDomainKind;
        let mut dec = FfmpegH264Dec::new().with_backend(Backend::NvdecCuda);
        assert_eq!(dec.requested_alloc(), None);
        let proposal = AllocationParams::cuda(1920 * 1080 * 3 / 2, 3, 256);
        AsyncElement::configure_allocation(&mut dec, &proposal);
        let recorded = dec.requested_alloc().expect("proposal recorded");
        assert_eq!(recorded.domain, MemoryDomainKind::Cuda);
        assert_eq!(recorded.min_buffers, 3);
        assert_eq!(recorded.align, 256);
    }
}
