//! Linux video decode element using ffmpeg / libavcodec.
//!
//! M13 shipped H.264; M111 generalized it to any libavcodec video decoder the
//! negotiated caps name (H.264 / H.265 / VP8 / VP9 / AV1), so the MKV / TS
//! demuxers' VP9 / AV1 elementary streams decode. Construction is unchanged: the
//! codec is read from the input caps at `configure_pipeline` and the matching
//! decoder opened. `FfmpegVideoDec` is the preferred name now (`FfmpegH264Dec`
//! remains a back-compat alias).
//!
//! M13 (Linux production path): consumes Annex-B `DataFrame`s (the
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
//! [`Backend::Vaapi`] is the Linux AMD / Intel hardware path. It attaches an
//! `AV_HWDEVICE_TYPE_VAAPI` device to the generic decoder and a `get_format`
//! hook selecting `AV_PIX_FMT_VAAPI` (a true hwaccel, not a standalone codec
//! like cuvid), decodes on the GPU, then `av_hwframe_transfer_data`s the
//! surface into system memory (NV12 on radeonsi / Intel) and packs it like the
//! software path. Unlike `VaapiH264Dec` (cros-codecs, blocked on Mesa
//! `radeonsi` GBM surface allocation), it uses libavcodec's own hwframe pool,
//! so it works on AMD desktop / iGPU. Pin the render node with
//! [`FfmpegH264Dec::with_vaapi_device`] on multi-GPU hosts.
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
//! - 10-bit pixel formats (`YUV420P10` / `P010`). Mainline H.264 cameras emit
//!   8-bit YUV420P; `YUV444P` is now accepted (chroma box-averaged to 4:2:0),
//!   but 10-bit and other formats are still rejected with `CapsMismatch`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use ffmpeg_next as ffmpeg;
use ffmpeg::codec::{self, Flags, Id};
use ffmpeg::format::Pixel;
use ffmpeg::frame::Video as FfVideo;
use ffmpeg::packet::Packet;
use ffmpeg::Dictionary;
use ffmpeg::Error as FfError;

use crate::yuv::downsample_chroma_420;
use g2g_core::frame::Frame;
use g2g_core::memory::{OwnedCudaBuffer, SystemSlice};
use g2g_core::{
    AllocationParams, AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, CudaKeepAlive,
    Dim, ElementMetadata, FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink,
    PadTemplate, PadTemplates, PipelinePacket, PropError, PropKind, PropValue, PropertySpec, Rate,
    VideoCodec, RawVideoFormat,
};

/// Cap on pending input-pts -> arrival entries, so frames the decoder drops
/// (never echoing their pts back) can't grow the map without bound.
const MAX_PENDING_ARRIVALS: usize = 1024;

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
    /// VAAPI hardware decode via the generic decoder with an attached
    /// `AV_HWDEVICE_TYPE_VAAPI` device and a `get_format` hook selecting
    /// `AV_PIX_FMT_VAAPI`. The decoded surface is downloaded to system memory
    /// with `av_hwframe_transfer_data` (radeonsi / Intel transfer to NV12),
    /// then packed into the element's `OutputFormat` like the software path, so
    /// frames are emitted as `MemoryDomain::System`. This is the Linux AMD /
    /// Intel hardware path: unlike `VaapiH264Dec` (cros-codecs), it allocates
    /// surfaces through libavcodec's hwframe pool rather than GBM, so it works
    /// on Mesa `radeonsi` where cros-codecs 0.0.6 cannot. Pin the render node
    /// with [`FfmpegH264Dec::with_vaapi_device`] on multi-GPU hosts. Requires
    /// libavcodec built with the VAAPI hwaccel and a libva-capable render node.
    Vaapi,
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
    /// which the codec layer echoes verbatim. Bounded by
    /// [`MAX_PENDING_ARRIVALS`] so decoder-dropped frames can't leak.
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
    /// DRM render node the `Vaapi` backend opens its hwdevice on (e.g.
    /// `/dev/dri/renderD128`). `None` lets libva pick its default device, which
    /// on a multi-GPU host may not be the intended GPU. Ignored by the other
    /// backends.
    vaapi_device: Option<String>,
}

/// Preferred name now that this element decodes more than H.264 (also H.265 /
/// VP8 / VP9 / AV1). The struct keeps its original name as a back-compat alias.
pub type FfmpegVideoDec = FfmpegH264Dec;

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
            vaapi_device: None,
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
            Backend::Vaapi => {
                // The downloaded surface is packed by `copy_yuv420` like the
                // software path, so either output layout works (no forced
                // format). cuvid's `surfaces` knob doesn't apply; leave
                // `low_delay` off so B-frame reorder stays correct (VAAPI has
                // no cuvid-style deep internal pipeline to flatten).
                self.cuvid_surfaces = None;
                self.low_delay = false;
            }
        }
        self
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Pin the DRM render node the `Backend::Vaapi` hwdevice opens, e.g.
    /// `/dev/dri/renderD128`. `None` (the default) lets libva choose, which on
    /// a multi-GPU host may not select the intended GPU. Only consulted by the
    /// `Vaapi` backend.
    pub fn with_vaapi_device(mut self, path: Option<&str>) -> Self {
        self.vaapi_device = path.map(String::from);
        self
    }

    pub fn vaapi_device(&self) -> Option<&str> {
        self.vaapi_device.as_deref()
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
            // A decoder-dropped frame never echoes its pts back, so its entry
            // would linger; cap the map and evict the oldest, losing only a
            // latency sample for a frame the decoder discarded.
            while self.pts_to_arrival.len() > MAX_PENDING_ARRIVALS {
                self.pts_to_arrival.pop_first();
            }
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
        let transfers_from_hw = matches!(self.backend, Backend::Vaapi);
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
                    } else if transfers_from_hw {
                        // VAAPI: the decoded frame is a GPU surface
                        // (AV_PIX_FMT_VAAPI). Download it into a system-memory
                        // frame (NV12 on radeonsi / Intel), then pack like the
                        // software path.
                        let sw = transfer_hw_to_sw(&frame)?;
                        DecodedPayload::System(copy_yuv420(&sw, format)?)
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

impl PadTemplates for FfmpegH264Dec {
    /// Static superset for auto-plug: any supported codec in (any geometry), raw
    /// NV12 or I420 out. A constructed instance narrows the source pad to its
    /// configured `OutputFormat` via `caps_constraint_as_transform`; the template
    /// lists both formats the type can ever emit so the registry search can route
    /// either way, and every codec it can decode so `decodebin` autoplugs it for
    /// H.265 / VP8 / VP9 / AV1 too, not just H.264.
    fn pad_templates() -> Vec<PadTemplate> {
        let any_geometry = |format| Caps::RawVideo {
            format,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        let any_codec = |codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(
                SUPPORTED_CODECS.into_iter().map(any_codec).collect(),
            )),
            PadTemplate::source(CapsSet::from_alternatives(Vec::from([
                any_geometry(RawVideoFormat::Nv12),
                any_geometry(RawVideoFormat::I420),
            ]))),
        ])
    }
}

impl AsyncElement for FfmpegH264Dec {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        for codec in SUPPORTED_CODECS {
            let candidate = Caps::CompressedVideo {
                codec,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            };
            if let Ok(narrowed) = upstream_caps.intersect(&candidate) {
                return Ok(narrowed);
            }
        }
        Err(G2gError::CapsMismatch)
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
        let codec_kind = match absolute_caps {
            Caps::CompressedVideo { codec, .. } if SUPPORTED_CODECS.contains(codec) => *codec,
            _ => return Err(G2gError::CapsMismatch),
        };

        // ffmpeg::init() registers codecs once per process; calling it
        // repeatedly is safe and cheap.
        ffmpeg::init().map_err(|_| G2gError::Hardware(HardwareError::Other))?;

        // NvdecCuda needs NV12 out (the device frame's native layout); reject
        // an I420 request loud rather than silently emit mismatched caps.
        if self.backend == Backend::NvdecCuda && self.output_format != OutputFormat::Nv12 {
            return Err(G2gError::CapsMismatch);
        }

        let codec = match self.backend {
            // The generic decoder hosts the CUDA / VAAPI hwaccel; NvdecCuda and
            // Vaapi attach a device + get_format hook to it below.
            Backend::Software | Backend::NvdecCuda | Backend::Vaapi => {
                codec::decoder::find(codec_id(codec_kind))
            }
            // The `*_cuvid` decoders are standalone codec entries, not hwaccel
            // hooks, so they're found by name rather than by AVCodecID. If absent,
            // the libavcodec build wasn't compiled with `--enable-cuvid` or
            // libnvcuvid isn't loadable at runtime; either way the right answer is
            // to fail loud so the caller picks Software explicitly.
            Backend::NvdecCuvid => codec::decoder::find_by_name(cuvid_name(codec_kind)),
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

        // Vaapi: create a VAAPI hwdevice (optionally pinned to a render node)
        // and hand the generic decoder a reference plus a get_format hook
        // selecting AV_PIX_FMT_VAAPI, so decode runs on the GPU. The decoded
        // surface is downloaded to system memory per frame in `drain_frames`;
        // we keep no device handle on the element (unlike NvdecCuda's
        // CUcontext), the decoder's own ref keeps the device alive.
        if self.backend == Backend::Vaapi {
            // A non-empty device string must be a valid C string; an interior
            // NUL is a caller error, surfaced loud rather than truncating.
            let device = match self.vaapi_device.as_deref() {
                Some(path) => Some(
                    alloc::ffi::CString::new(path)
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?,
                ),
                None => None,
            };
            let device_ptr = device.as_ref().map_or(core::ptr::null(), |c| c.as_ptr());
            // SAFETY: standard ffmpeg hwaccel init on a freshly-allocated,
            // not-yet-opened context, mirroring the NvdecCuda path above.
            // `av_hwdevice_ctx_create` initialises `hw_device_ctx`; we hand the
            // codec its own ref, install the get_format callback, then drop our
            // ref. `device_ptr` is either null or points into `device`, which
            // outlives the create call.
            unsafe {
                let mut hw_device_ctx: *mut ffmpeg::ffi::AVBufferRef = core::ptr::null_mut();
                let ret = ffmpeg::ffi::av_hwdevice_ctx_create(
                    &mut hw_device_ctx,
                    ffmpeg::ffi::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
                    device_ptr,
                    core::ptr::null_mut(), // no options
                    0,
                );
                if ret < 0 || hw_device_ctx.is_null() {
                    return Err(G2gError::Hardware(HardwareError::Other));
                }
                let raw = decoder_ctx.as_mut_ptr();
                (*raw).hw_device_ctx = ffmpeg::ffi::av_buffer_ref(hw_device_ctx);
                (*raw).get_format = Some(get_vaapi_format);
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

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "FFmpeg video decoder",
            "Codec/Decoder/Video",
            "Decodes H.264 / H.265 / VP8 / VP9 / AV1 via libavcodec",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FFMPEGDEC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "device" => {
                // VAAPI render node (only consulted by Backend::Vaapi). An empty
                // value clears the pin so libva picks its default device.
                let path = value.as_str().ok_or(PropError::Type)?;
                self.vaapi_device = if path.is_empty() { None } else { Some(path.into()) };
                Ok(())
            }
            "output-format" => {
                self.output_format = match value.as_str().ok_or(PropError::Type)? {
                    "i420" | "I420" => OutputFormat::I420,
                    "nv12" | "NV12" => OutputFormat::Nv12,
                    _ => return Err(PropError::Value),
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            // Empty string when unpinned (= libva default), like other
            // always-present string properties (see filesrc `location`).
            "device" => {
                Some(PropValue::Str(self.vaapi_device.clone().unwrap_or_default()))
            }
            "output-format" => Some(PropValue::Str(
                match self.output_format {
                    OutputFormat::I420 => "i420",
                    OutputFormat::Nv12 => "nv12",
                }
                .into(),
            )),
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
                    // Two callers:
                    //   1) An H.264 input CapsChanged from upstream: record
                    //      it (M16 workaround #3 Phase A validates the
                    //      format; ordering invariant §3 lets us emit our
                    //      own output `CapsChanged` later from decoded-frame
                    //      geometry, not eagerly here).
                    //   2) The runner's transform arm pre-fixed forward
                    //      output caps for a strict-NV12 downstream sink
                    //      (runner.rs:1281). Those carry our output
                    //      `RawVideoFormat` and must be forwarded so the
                    //      sink sees its expected caps before the first
                    //      decoded frame; suppress re-emission from the
                    //      decode loop by recording `last_caps`.
                    match &c {
                        Caps::CompressedVideo { codec, .. }
                            if SUPPORTED_CODECS.contains(codec) =>
                        {
                            self.input_caps = Some(c);
                        }
                        Caps::RawVideo { format, .. }
                            if *format == self.output_format.raw_format() =>
                        {
                            out.push(PipelinePacket::CapsChanged(c.clone())).await?;
                            self.last_caps = Some(c);
                        }
                        _ => return Err(G2gError::CapsMismatch),
                    }
                }
                PipelinePacket::Flush => {
                    if let Some(d) = self.decoder.as_mut() {
                        d.flush();
                    }
                    self.pts_to_arrival.clear();
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
                    return Ok(());
                }
            }

            let out_format = self.output_format;
            // Carry the negotiated framerate (from the input caps) into the
            // output caps. Emitting `Rate::Any` here breaks the mid-stream
            // forward-caps resolve downstream: a format/geometry-changing
            // transform (videoconvert / videoscale) cannot `fixate()` an `Any`
            // framerate, so it falls back to forwarding our caps verbatim, and a
            // constraining capsfilter then rejects them (the decode -> scale ->
            // fixed-format chain). A compressed stream's rate is advisory anyway
            // (per-frame PTS carries the real timing); default to 30/1 when the
            // container did not declare one.
            let out_framerate = match &self.input_caps {
                Some(Caps::CompressedVideo { framerate: Rate::Fixed(q), .. }) => Rate::Fixed(*q),
                _ => Rate::Fixed(30 << 16),
            };
            for d in decoded {
                let new_caps = yuv420_caps(out_format, d.width, d.height, out_framerate.clone());
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
                        keyframe: true, // raw decoded frames are each independently presentable
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

/// Single source of truth for the decoder's output-side caps derivation.
/// Called by the `DerivedOutput` constraint closure (startup negotiation)
/// and the workaround-#3 Phase A debug assertion (mid-stream consistency
/// check between recorded input caps and decode-time output geometry).
///
/// Non-H.264 input yields an empty `CapsSet`: the solver treats that as a
/// negotiation failure; the runtime mid-stream check refuses
/// `CapsMismatch` before it ever reaches here.
/// The compressed codecs this element can open via libavcodec.
const SUPPORTED_CODECS: [VideoCodec; 6] = [
    VideoCodec::H264,
    VideoCodec::H265,
    VideoCodec::Vp8,
    VideoCodec::Vp9,
    VideoCodec::Av1,
    VideoCodec::Mpeg4Part2,
];

/// `FfmpegVideoDec`'s settable properties: the VAAPI render node (for the
/// `Backend::Vaapi` / `ffmpegvaapidec` path) and the decoded output layout, so a
/// `gst-launch` line can pin the GPU and pick I420 / NV12 without the builder.
static FFMPEGDEC_PROPS: &[PropertySpec] = &[
    PropertySpec::new(
        "device",
        PropKind::Str,
        "VAAPI render node, e.g. /dev/dri/renderD128 (Backend::Vaapi only; empty = libva default)",
    ),
    PropertySpec::new("output-format", PropKind::Str, "decoded pixel layout: i420 | nv12"),
];

/// The libavcodec `AVCodecID` for a g2g codec (generic + CUDA-hwaccel path).
fn codec_id(codec: VideoCodec) -> Id {
    match codec {
        VideoCodec::H264 => Id::H264,
        VideoCodec::H265 => Id::HEVC,
        VideoCodec::Vp8 => Id::VP8,
        VideoCodec::Vp9 => Id::VP9,
        VideoCodec::Av1 => Id::AV1,
        VideoCodec::Mjpeg => Id::MJPEG,
        VideoCodec::Mpeg4Part2 => Id::MPEG4,
        _ => unreachable!("ffmpegdec negotiates only known VideoCodec variants"),
    }
}

/// The standalone `*_cuvid` decoder name for a codec (NvdecCuvid backend).
fn cuvid_name(codec: VideoCodec) -> &'static str {
    match codec {
        VideoCodec::H264 => "h264_cuvid",
        VideoCodec::H265 => "hevc_cuvid",
        VideoCodec::Vp8 => "vp8_cuvid",
        VideoCodec::Vp9 => "vp9_cuvid",
        VideoCodec::Av1 => "av1_cuvid",
        VideoCodec::Mjpeg => "mjpeg_cuvid",
        _ => unreachable!("ffmpegdec negotiates only known VideoCodec variants"),
    }
}

fn derive_output_caps(input: &Caps, out_fmt: RawVideoFormat) -> CapsSet {
    match input {
        Caps::CompressedVideo { codec, width, height, framerate }
            if SUPPORTED_CODECS.contains(codec) =>
        {
            CapsSet::one(Caps::RawVideo {
                format: out_fmt,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            })
        }
        _ => CapsSet::from_alternatives(alloc::vec::Vec::new()),
    }
}

fn yuv420_caps(format: OutputFormat, w: u32, h: u32, framerate: Rate) -> Caps {
    Caps::RawVideo {
        format: format.raw_format(),
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate,
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
/// Three source pixel layouts are accepted: planar `YUV420P` (the software
/// H.264 decoder's native output), semi-planar `NV12` (h264_cuvid's native
/// output), and planar `YUV444P` (High 4:4:4 profile), whose full-resolution
/// chroma is box-averaged down to 4:2:0 (lossy). Any other format (e.g.
/// 10-bit) is rejected loud — those streams need a `ColorConvert` element
/// upstream of any I420/NV12 consumer.
fn copy_yuv420(frame: &FfVideo, format: OutputFormat) -> Result<Box<[u8]>, G2gError> {
    let src = match frame.format() {
        // YUVJ420P is YUV420P with JPEG (full) range. Same plane layout, so
        // accept it; range fidelity is preserved in the pixel values and can
        // be advertised by a future colour-metadata field on `Caps::Video`.
        Pixel::YUV420P | Pixel::YUVJ420P => SourceLayout::Planar420,
        Pixel::NV12 => SourceLayout::SemiPlanar420,
        // 4:4:4 (High 4:4:4 profile). Full-resolution chroma is box-averaged
        // down to 4:2:0; lossy in chroma, but keeps the output contract.
        Pixel::YUV444P | Pixel::YUVJ444P => SourceLayout::Planar444,
        _ => return Err(G2gError::CapsMismatch),
    };
    let required_planes = match src {
        SourceLayout::Planar420 | SourceLayout::Planar444 => 3,
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
        // 4:4:4 source -> planar I420: box-average each full-res U/V plane to
        // half resolution (tightly packed, row stride `cw`), then store.
        (SourceLayout::Planar444, OutputFormat::I420) => {
            let u_ds = downsample_chroma_420(frame.data(1), frame.stride(1), w, h);
            let v_ds = downsample_chroma_420(frame.data(2), frame.stride(2), w, h);
            out[y_size..y_size + c_size].copy_from_slice(&u_ds);
            out[y_size + c_size..y_size + 2 * c_size].copy_from_slice(&v_ds);
        }
        // 4:4:4 source -> semi-planar NV12: downsample, then interleave U/V.
        (SourceLayout::Planar444, OutputFormat::Nv12) => {
            let u_ds = downsample_chroma_420(frame.data(1), frame.stride(1), w, h);
            let v_ds = downsample_chroma_420(frame.data(2), frame.stride(2), w, h);
            for row in 0..ch {
                let dst_base = y_size + row * 2 * cw;
                for col in 0..cw {
                    out[dst_base + 2 * col] = u_ds[row * cw + col];
                    out[dst_base + 2 * col + 1] = v_ds[row * cw + col];
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
    /// Planar 4:4:4 (full-resolution chroma); downsampled to 4:2:0 on output.
    Planar444,
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
    formats: *const ffmpeg::ffi::AVPixelFormat,
) -> ffmpeg::ffi::AVPixelFormat {
    // SAFETY: `formats` is null or an AV_PIX_FMT_NONE-terminated array per the
    // get_format contract; `pick_hw_format` only reads it.
    unsafe { pick_hw_format(formats, ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_CUDA) }
}

/// `get_format` hook for the `Vaapi` backend: pick `AV_PIX_FMT_VAAPI` so the
/// decoder produces GPU surfaces (downloaded to system memory after decode),
/// else `AV_PIX_FMT_NONE` to fail the selection loud rather than silently fall
/// back to software under a hwaccel backend.
///
/// # Safety
/// Called by libavcodec with a valid `*ctx` and an `AV_PIX_FMT_NONE`-terminated
/// `formats` array. We only read the array.
unsafe extern "C" fn get_vaapi_format(
    _ctx: *mut ffmpeg::ffi::AVCodecContext,
    formats: *const ffmpeg::ffi::AVPixelFormat,
) -> ffmpeg::ffi::AVPixelFormat {
    // SAFETY: see get_cuda_format.
    unsafe { pick_hw_format(formats, ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_VAAPI) }
}

/// Walk libavcodec's `AV_PIX_FMT_NONE`-terminated offered-format list and return
/// `wanted` if present, else `AV_PIX_FMT_NONE` so the codec fails the format
/// selection loud rather than silently picking a software fallback.
///
/// # Safety
/// `formats` must be null or a valid `AV_PIX_FMT_NONE`-terminated array.
unsafe fn pick_hw_format(
    mut formats: *const ffmpeg::ffi::AVPixelFormat,
    wanted: ffmpeg::ffi::AVPixelFormat,
) -> ffmpeg::ffi::AVPixelFormat {
    if formats.is_null() {
        return ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE;
    }
    // SAFETY: the list is non-null and AV_PIX_FMT_NONE-terminated per the
    // get_format contract; we walk it without passing the terminator.
    unsafe {
        while *formats != ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE {
            if *formats == wanted {
                return wanted;
            }
            formats = formats.add(1);
        }
    }
    ffmpeg::ffi::AVPixelFormat::AV_PIX_FMT_NONE
}

/// Download a decoded hardware surface (e.g. `AV_PIX_FMT_VAAPI`) into a freshly
/// allocated system-memory frame. The destination format is left unset so
/// libavcodec picks the surface's preferred transfer format (NV12 on radeonsi /
/// Intel VAAPI); [`copy_yuv420`] then packs NV12 or planar into the element's
/// `OutputFormat`. Geometry and plane data come from the surface; the caller
/// reads pts / width / height off the source frame before this call.
fn transfer_hw_to_sw(hw: &FfVideo) -> Result<FfVideo, G2gError> {
    let mut sw = FfVideo::empty();
    // SAFETY: `hw` is a decoded hardware-surface frame from a VAAPI-configured
    // decoder; `sw` is a freshly allocated empty frame. av_hwframe_transfer_data
    // allocates sw's buffers and downloads the surface. A non-negative return
    // means sw holds valid system-memory planes read by `copy_yuv420`.
    let ret = unsafe { ffmpeg::ffi::av_hwframe_transfer_data(sw.as_mut_ptr(), hw.as_ptr(), 0) };
    if ret < 0 {
        return Err(G2gError::Hardware(HardwareError::Other));
    }
    Ok(sw)
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
    // Arc, not Box: the keep-alive is shareable so a tee can fan the GPU frame
    // out to several consumers zero-copy (M213).
    let keep_alive = alloc::sync::Arc::new(CudaFrameOwner(frame));
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
// The field is never read: it is a drop-guard, held only so the owned `FfVideo`
// (and thus the `AVFrame`) lives until this owner is dropped.
#[allow(dead_code)]
struct CudaFrameOwner(FfVideo);

// SAFETY: an `AVFrame` (like the decoder's `AVCodecContext`) is `!Send` by
// default. We uphold `Send` by construction: the runner moves the frame
// between worker tasks but never shares it, so the frame is owned and moved,
// never aliased, the same contract as the decoder itself.
unsafe impl Send for CudaFrameOwner {}

// SAFETY: `Sync` (M213) lets a tee share the keep-alive across branches that
// read the decoded surface concurrently. Sound because the owner is inert: it
// pins the `AVFrame` and exposes no interior mutability; the device memory is an
// immutable decoded surface safe for concurrent read, and the final drop is
// serialized by the `Arc` refcount. Same contract as `Send` above.
unsafe impl Sync for CudaFrameOwner {}

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
    fn caps_constraint_derives_output_for_supported_codecs() {
        // M16 step 5k: DerivedOutput closure validates a supported codec input
        // and emits the configured output format at the same dims/rate.
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

        // VP9 (and the other supported codecs) now derive output too (M111).
        let vp9 = Caps::CompressedVideo {
            codec: VideoCodec::Vp9,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
        };
        assert_eq!(
            f(&vp9).alternatives(),
            &[Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(1920),
                height: Dim::Fixed(1080),
                framerate: Rate::Fixed(30 << 16),
            }]
        );
        // A non-compressed input has no codec to decode → empty CapsSet.
        let raw = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(64),
            height: Dim::Fixed(64),
            framerate: Rate::Any,
        };
        assert!(f(&raw).is_empty());
    }

    #[test]
    fn i420_caps_are_fixed() {
        assert_eq!(
            yuv420_caps(OutputFormat::I420, 640, 480, Rate::Fixed(30 << 16)),
            Caps::RawVideo {
                format: RawVideoFormat::I420,
                width: Dim::Fixed(640),
                height: Dim::Fixed(480),
                framerate: Rate::Fixed(30 << 16),
            }
        );
    }

    #[test]
    fn nv12_caps_advertise_nv12_format() {
        assert_eq!(
            yuv420_caps(OutputFormat::Nv12, 1280, 720, Rate::Fixed(30 << 16)),
            Caps::RawVideo {
                format: RawVideoFormat::Nv12,
                width: Dim::Fixed(1280),
                height: Dim::Fixed(720),
                framerate: Rate::Fixed(30 << 16),
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
    fn intercept_accepts_supported_codecs_rejects_raw() {
        let dec = FfmpegH264Dec::new();
        // VP9 / AV1 / H.265 are accepted now (M111), each narrowed to itself.
        for codec in [VideoCodec::Vp9, VideoCodec::Av1, VideoCodec::H265, VideoCodec::Vp8] {
            let caps = Caps::CompressedVideo {
                codec,
                width: Dim::Any,
                height: Dim::Any,
                framerate: Rate::Any,
            };
            assert_eq!(dec.intercept_caps(&caps), Ok(caps));
        }
        // A raw input has no codec to decode and is rejected.
        let raw = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
        };
        assert_eq!(dec.intercept_caps(&raw), Err(G2gError::CapsMismatch));
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
    fn codec_maps_to_libavcodec_id_and_cuvid_name() {
        assert_eq!(codec_id(VideoCodec::H264), Id::H264);
        assert_eq!(codec_id(VideoCodec::H265), Id::HEVC);
        assert_eq!(codec_id(VideoCodec::Vp8), Id::VP8);
        assert_eq!(codec_id(VideoCodec::Vp9), Id::VP9);
        assert_eq!(codec_id(VideoCodec::Av1), Id::AV1);
        assert_eq!(cuvid_name(VideoCodec::H264), "h264_cuvid");
        assert_eq!(cuvid_name(VideoCodec::Vp9), "vp9_cuvid");
        assert_eq!(cuvid_name(VideoCodec::Av1), "av1_cuvid");
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
    fn vaapi_backend_selectable_and_keeps_output_format() {
        // Unlike NvdecCuda, the VAAPI path downloads to system memory and packs
        // via copy_yuv420, so it honours either output layout (no forced NV12).
        let dec = FfmpegH264Dec::new()
            .with_output_format(OutputFormat::I420)
            .with_backend(Backend::Vaapi);
        assert_eq!(dec.backend(), Backend::Vaapi);
        assert_eq!(dec.output_format(), OutputFormat::I420);
        // VAAPI has no cuvid-style deep pipeline, so low-delay stays off for
        // B-frame reorder correctness, and the cuvid surface knob is unset.
        assert!(!dec.low_delay());
        assert_eq!(dec.cuvid_surfaces(), None);
    }

    #[test]
    fn with_vaapi_device_stores_render_node() {
        let dec = FfmpegH264Dec::new()
            .with_backend(Backend::Vaapi)
            .with_vaapi_device(Some("/dev/dri/renderD128"));
        assert_eq!(dec.vaapi_device(), Some("/dev/dri/renderD128"));
        // Default is None (libva picks its default device).
        assert_eq!(FfmpegH264Dec::new().vaapi_device(), None);
    }

    #[test]
    fn device_property_sets_and_reads_vaapi_render_node() {
        let mut dec = FfmpegH264Dec::new().with_backend(Backend::Vaapi);
        // Unset reads back as the empty string (= libva default).
        assert_eq!(dec.get_property("device"), Some(PropValue::Str(String::new())));
        dec.set_property("device", PropValue::Str("/dev/dri/renderD128".into()))
            .expect("device is a known property");
        assert_eq!(dec.vaapi_device(), Some("/dev/dri/renderD128"));
        assert_eq!(
            dec.get_property("device"),
            Some(PropValue::Str("/dev/dri/renderD128".into()))
        );
        // An empty value clears the pin back to the libva default.
        dec.set_property("device", PropValue::Str(String::new())).unwrap();
        assert_eq!(dec.vaapi_device(), None);
    }

    #[test]
    fn output_format_property_round_trips_and_rejects_bad_value() {
        let mut dec = FfmpegH264Dec::new();
        assert_eq!(dec.get_property("output-format"), Some(PropValue::Str("i420".into())));
        dec.set_property("output-format", PropValue::Str("nv12".into())).unwrap();
        assert_eq!(dec.output_format(), OutputFormat::Nv12);
        assert_eq!(dec.set_property("output-format", PropValue::Str("rgb".into())), Err(PropError::Value));
        // A type mismatch and an unknown name are distinct errors.
        assert_eq!(dec.set_property("device", PropValue::Int(7)), Err(PropError::Type));
        assert_eq!(dec.set_property("nope", PropValue::Str("x".into())), Err(PropError::Unknown));
    }

    #[test]
    fn declares_device_and_output_format_properties() {
        let names: Vec<&str> = FfmpegH264Dec::new().properties().iter().map(|p| p.name).collect();
        assert!(names.contains(&"device"), "device property declared: {names:?}");
        assert!(names.contains(&"output-format"), "output-format declared: {names:?}");
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

    // Minimal recording `OutputSink` for direct `process` probes. Push
    // results are collected into a `RefCell<Vec<_>>` we inspect after.
    struct RecSink<'a>(&'a core::cell::RefCell<Vec<PipelinePacket>>);
    impl<'a> OutputSink for RecSink<'a> {
        fn push<'b>(
            &'b mut self,
            packet: PipelinePacket,
        ) -> Pin<Box<dyn Future<Output = Result<g2g_core::PushOutcome, G2gError>> + 'b>> {
            let log = self.0;
            Box::pin(async move {
                log.borrow_mut().push(packet);
                Ok(g2g_core::PushOutcome::Accepted)
            })
        }
    }

    fn h264_caps(w: u32, h: u32) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    /// Regression: when a strict-NV12 downstream sink (e.g. `WaylandSink`)
    /// declares `Accepts(NV12, Any, ..)`, the solver pre-fixes the
    /// transform's forward output caps, and the runner's transform arm
    /// (runner.rs `run_source_transform_sink_inner`, the
    /// `Some(PipelinePacket::CapsChanged(new_caps))` branch) pushes
    /// `CapsChanged(forward_caps)` through `transform.process`. The
    /// decoder used to reject anything that wasn't H.264, killing the
    /// transform arm and surfacing as `G2gError::Shutdown` to the source
    /// (it caught the wayland_smoke harness). The decoder must instead
    /// forward such a packet downstream so the sink sees its expected
    /// caps before the first decoded frame.
    #[tokio::test]
    async fn process_caps_changed_with_output_format_forwards_to_downstream() {
        let mut dec = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
        dec.configure_pipeline(&h264_caps(1280, 720))
            .expect("configure H.264 input");

        let nv12 = Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Fixed(30 << 16),
        };
        let log = core::cell::RefCell::new(Vec::new());
        let mut rec = RecSink(&log);
        dec.process(PipelinePacket::CapsChanged(nv12.clone()), &mut rec)
            .await
            .expect("decoder must accept and forward its own output-format CapsChanged");

        let recorded = log.borrow();
        assert_eq!(recorded.len(), 1, "exactly one packet forwarded");
        match &recorded[0] {
            PipelinePacket::CapsChanged(c) => assert_eq!(c, &nv12),
            other => panic!("expected CapsChanged, got {other:?}"),
        }
    }

    /// H.264 input caps are still accepted but NOT forwarded (the decoder
    /// emits its own output `CapsChanged` from decoded-frame geometry later;
    /// pinning this stops a future "forward everything" refactor from
    /// breaking the ordering invariant §3).
    #[tokio::test]
    async fn process_caps_changed_with_h264_input_records_but_does_not_forward() {
        let mut dec = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
        dec.configure_pipeline(&h264_caps(640, 480))
            .expect("configure H.264 input");

        let log = core::cell::RefCell::new(Vec::new());
        let mut rec = RecSink(&log);
        dec.process(PipelinePacket::CapsChanged(h264_caps(640, 480)), &mut rec)
            .await
            .expect("H.264 input CapsChanged must be accepted");

        assert!(log.borrow().is_empty(), "H.264 input caps must not be forwarded");
        assert!(dec.input_caps.is_some(), "input caps must be recorded");
    }

    /// Inverse: a raw format that does not match the decoder's configured
    /// output must still fail loud. The forward-caps acceptance is narrow:
    /// only the exact `OutputFormat` the decoder emits. Stops a regression
    /// where the decoder silently accepts any RawVideo caps.
    #[tokio::test]
    async fn process_caps_changed_with_wrong_raw_format_rejects() {
        let mut dec = FfmpegH264Dec::new().with_output_format(OutputFormat::Nv12);
        dec.configure_pipeline(&h264_caps(640, 480))
            .expect("configure H.264 input");

        let i420 = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        let log = core::cell::RefCell::new(Vec::new());
        let mut rec = RecSink(&log);
        let err = dec
            .process(PipelinePacket::CapsChanged(i420), &mut rec)
            .await;
        assert_eq!(err, Err(G2gError::CapsMismatch));
        assert!(log.borrow().is_empty(), "rejected caps must not be forwarded");
    }
}
