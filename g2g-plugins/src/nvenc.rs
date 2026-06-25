//! Native NVENC H.264 encode element (`nvenc` feature): the zero-copy,
//! device-resident counterpart of [`crate::ffmpegenc::FfmpegH264Enc`]. Where the
//! ffmpeg encoder takes system-memory I420 and copies it into libavcodec, this
//! element ingests an NVDEC/CUDA NV12 surface ([`MemoryDomain::Cuda`]) *in place*
//! and drives the NVIDIA Video Codec SDK (`nvEncodeAPI`) directly, so the pixels
//! never leave the GPU. It is the encode-side mirror of the M220 `CudaToWgpu`
//! bridge (which imports NVDEC output for wgpu): here the NVDEC output is handed
//! straight to NVENC, closing the native `NvdecCuda -> NvEnc` loop with no PCIe
//! download.
//!
//! `Caps::RawVideo{Nv12}` in, `Caps::CompressedVideo{H264}` Annex-B out. Caps do
//! not encode the memory domain, so negotiation is identical to a system-memory
//! encoder; at runtime the frame must be `MemoryDomain::Cuda` (an `UnsupportedDomain`
//! error otherwise, the symmetric contract `FfmpegH264Enc` upholds for `System`).
//! The incoming `OwnedCudaBuffer` must be a contiguous NV12 surface (chroma at
//! `luma_ptr + luma_pitch * height`, a single base pointer + pitch), which is how
//! NVDEC lays out its hwframe pool; NVENC registers that one device pointer.
//!
//! Bindings are a thin hand-rolled FFI linking `libnvidia-encode` and `libcuda`
//! directly (no `cudarc`), matching the decision in DESIGN-C3-cuda.md §6 and the
//! [`crate::cuda`] module: the SDK's giant version-tagged structs are transcribed
//! `#[repr(C)]` with compile-time size assertions checked against the installed
//! `nvEncodeAPI.h` (SDK 13.0), and the one field-heavy codec-config union is left
//! opaque (the driver fills it via `nvEncGetEncodePresetConfigEx`, we only tweak
//! rate control / GOP). The session opens lazily on the first frame, on that
//! frame's `CUcontext`, so it runs in the same context as the NVDEC source.
//!
//! Low latency, like `FfmpegH264Enc`: preset P4 + the LOW_LATENCY tuning info,
//! CBR, no B-frames (`frameIntervalP = 1`), and an *infinite GOP* so IDRs are
//! emitted only on demand (the first frame, and on a downstream PLI via
//! [`Reconfigure::ForceKeyframe`]); each forced IDR carries in-band SPS/PPS
//! (`OUTPUT_SPSPPS`), the Annex-B parameter sets a network sink expects. The
//! NV12 nanosecond PTS round-trips through NVENC's `inputTimeStamp`.
//!
//! Threading: the encoder is a raw session handle plus a CUDA context, driven
//! through `&mut self` only and never shared; `unsafe impl Send` rests on the
//! same ownership-transfer contract as `FfmpegH264Enc` / `FfmpegVideoDec`.
//!
//! Deferred (v1): system-memory NV12 input (host upload, already covered by
//! `FfmpegH264Enc` for I420), HEVC via the HEVC GUID, finite-GOP periodic IDRs
//! with `repeatSPSPPS`, an output-bitstream-buffer pool (one is allocated and
//! freed per frame today), and runtime bitrate retarget via `nvEncReconfigureEncoder`.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::memory::OwnedCudaBuffer;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, G2gError,
    HardwareError, MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, RawVideoFormat, Rate, VideoCodec,
};

/// Default constant target bitrate (bits/second). 4 Mbps, a 1080p30 default that
/// matches [`crate::ffmpegenc`].
const DEFAULT_BITRATE_BPS: u32 = 4_000_000;

/// Native H.264 encoder over the NVIDIA Video Codec SDK. NV12 CUDA surfaces in,
/// H.264 Annex-B out. See the module docs.
pub struct NvEnc {
    width: u32,
    height: u32,
    framerate: Rate,
    /// Negotiated input pixel format: NV12 (the NVDEC hwframe domain) or a packed
    /// 8-bit RGBA (the GPU-render domain, e.g. via `WgpuToCuda`). NVENC color
    /// converts RGBA internally, so both produce H.264.
    input_format: RawVideoFormat,
    bitrate_bps: u32,
    /// Open NVENC session (lazy): the SDK function table, the encoder handle, and
    /// the CUDA context it was opened on. `None` until the first frame.
    session: Option<Session>,
    /// Monotonic input frame index stamped into `NV_ENC_PIC_PARAMS::frameIdx`.
    frame_no: u32,
    emitted: u64,
    caps_sent: bool,
    /// Latched IDR request. Starts `true` so the very first frame is a keyframe;
    /// a downstream PLI re-latches it.
    force_keyframe: bool,
    configured: bool,
}

/// A live NVENC session: the loaded API function table, the opaque encoder
/// handle, and the CUDA context the handle and all input surfaces live in.
struct Session {
    funcs: Box<ffi::NvEncodeApiFunctionList>,
    encoder: *mut core::ffi::c_void,
    context: u64,
}

// SAFETY: `Session` holds raw NVENC/CUDA handles. The runner moves `NvEnc`
// between worker tasks but drives it through `&mut self` only (never
// concurrently), so the handles are owned and moved, never aliased, the same
// contract upheld by `FfmpegH264Enc` / `FfmpegVideoDec`.
unsafe impl Send for NvEnc {}

impl core::fmt::Debug for NvEnc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NvEnc")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bitrate_bps", &self.bitrate_bps)
            .field("open", &self.session.is_some())
            .field("emitted", &self.emitted)
            .field("configured", &self.configured)
            .finish()
    }
}

impl Default for NvEnc {
    fn default() -> Self {
        Self::new()
    }
}

impl NvEnc {
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            framerate: Rate::Any,
            input_format: RawVideoFormat::Nv12,
            bitrate_bps: DEFAULT_BITRATE_BPS,
            session: None,
            frame_no: 0,
            emitted: 0,
            caps_sent: false,
            force_keyframe: true,
            configured: false,
        }
    }

    /// Set the constant target bitrate (bits/second). Default 4 Mbps.
    pub fn with_bitrate(mut self, bps: u32) -> Self {
        self.bitrate_bps = bps.max(1);
        self
    }

    /// Count of H.264 access units emitted.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    /// The accepted input formats: NV12 (the NVDEC hwframe domain) and packed
    /// 8-bit RGBA (the GPU-render domain). NVENC color converts RGBA internally.
    fn input_formats() -> [RawVideoFormat; 2] {
        [RawVideoFormat::Nv12, RawVideoFormat::Rgba8]
    }

    /// Open-geometry input caps, one alternative per accepted format.
    fn input_caps_set() -> CapsSet {
        CapsSet::from_alternatives(
            Self::input_formats()
                .into_iter()
                .map(|format| Caps::RawVideo {
                    format,
                    width: Dim::Any,
                    height: Dim::Any,
                    framerate: Rate::Any,
                })
                .collect(),
        )
    }

    /// The NVENC input buffer format for the negotiated input. RGBA maps to
    /// `ABGR` (NVENC's word-ordered `A8B8G8R8` is byte order R,G,B,A, exactly
    /// wgpu's `Rgba8`); `Bgra8` maps to `ARGB`.
    fn nv_buffer_format(&self) -> u32 {
        match self.input_format {
            RawVideoFormat::Rgba8 => ffi::NV_ENC_BUFFER_FORMAT_ABGR,
            RawVideoFormat::Bgra8 => ffi::NV_ENC_BUFFER_FORMAT_ARGB,
            _ => ffi::NV_ENC_BUFFER_FORMAT_NV12,
        }
    }

    fn output_caps(&self) -> Caps {
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: self.framerate.clone(),
        }
    }

    /// Frames per second from the negotiated framerate (Q16.16), defaulting to 30.
    fn fps(&self) -> u32 {
        match self.framerate {
            Rate::Fixed(q16) => (q16 >> 16).max(1),
            _ => 30,
        }
    }

    /// Open the NVENC session on `context` (lazy, first frame). Loads the API
    /// function table, opens a CUDA encode session, pulls the low-latency preset
    /// config, tweaks rate control + GOP, and initializes the encoder for the
    /// negotiated geometry. Fails loud (`HardwareError::Other`) if NVENC is
    /// unavailable so the caller can fall back to `FfmpegH264Enc`.
    fn open_session(&mut self, context: u64) -> Result<(), G2gError> {
        // Load the SDK dispatch table.
        let mut funcs: Box<ffi::NvEncodeApiFunctionList> =
            // SAFETY: the function-list struct is plain old data; all-zero is the valid
            // "unset" state we then version-tag before NvEncodeAPICreateInstance fills it.
            Box::new(unsafe { core::mem::zeroed() });
        funcs.version = ffi::NV_ENCODE_API_FUNCTION_LIST_VER;
        // SAFETY: `funcs` is a zeroed, version-tagged function-list struct; the
        // SDK fills it with entry points and only reads `version`.
        nvchk(unsafe { ffi::NvEncodeAPICreateInstance(funcs.as_mut()) })?;

        // Everything below runs in `context` (push now, pop at the end).
        let _ctx = ContextGuard::push(context)?;

        // Open a CUDA encode session.
        // SAFETY: the NVENC param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then version-tag.
        let mut open: ffi::OpenEncodeSessionExParams = unsafe { core::mem::zeroed() };
        open.version = ffi::NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
        open.device_type = ffi::NV_ENC_DEVICE_TYPE_CUDA;
        open.device = context as *mut core::ffi::c_void;
        open.api_version = ffi::NVENCAPI_VERSION;
        let open_fn = funcs.nv_enc_open_encode_session_ex.ok_or(hw())?;
        let mut encoder: *mut core::ffi::c_void = core::ptr::null_mut();
        // SAFETY: `open` is fully initialized and version-tagged; on success
        // `encoder` receives a valid session handle.
        nvchk(unsafe { open_fn(&mut open, &mut encoder) })?;
        if encoder.is_null() {
            return Err(hw());
        }

        // Build the session now so a failure below still destroys the handle on
        // drop.
        self.session = Some(Session { funcs, encoder, context });

        // Pull the low-latency preset config (the driver fills the codec-specific
        // union for us), then tweak rate control + GOP.
        let funcs = &self.session.as_ref().unwrap().funcs;
        // SAFETY: the NVENC param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then version-tag.
        let mut preset: ffi::PresetConfig = unsafe { core::mem::zeroed() };
        preset.version = ffi::NV_ENC_PRESET_CONFIG_VER;
        preset.preset_cfg.version = ffi::NV_ENC_CONFIG_VER;
        let preset_fn = funcs.nv_enc_get_encode_preset_config_ex.ok_or(hw())?;
        // SAFETY: valid encoder handle; GUIDs passed by value per the C ABI; the
        // driver writes the filled config into `preset.preset_cfg`.
        nvchk(unsafe {
            preset_fn(
                encoder,
                ffi::NV_ENC_CODEC_H264_GUID,
                ffi::NV_ENC_PRESET_P4_GUID,
                ffi::NV_ENC_TUNING_INFO_LOW_LATENCY,
                &mut preset,
            )
        })?;

        let mut config = preset.preset_cfg;
        config.version = ffi::NV_ENC_CONFIG_VER;
        config.rc_params.version = ffi::NV_ENC_RC_PARAMS_VER;
        // Infinite GOP: IDRs only on demand (first frame + downstream PLI), the
        // low-latency live-streaming model. No B-frames.
        config.gop_length = ffi::NVENC_INFINITE_GOPLENGTH;
        config.frame_interval_p = 1;
        // Constant bitrate at the configured target.
        config.rc_params.rate_control_mode = ffi::NV_ENC_PARAMS_RC_CBR;
        config.rc_params.average_bit_rate = self.bitrate_bps;
        config.rc_params.max_bit_rate = self.bitrate_bps;

        let fps = self.fps();
        // SAFETY: the NVENC param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then version-tag.
        let mut init: ffi::InitializeParams = unsafe { core::mem::zeroed() };
        init.version = ffi::NV_ENC_INITIALIZE_PARAMS_VER;
        init.encode_guid = ffi::NV_ENC_CODEC_H264_GUID;
        init.preset_guid = ffi::NV_ENC_PRESET_P4_GUID;
        init.encode_width = self.width;
        init.encode_height = self.height;
        init.dar_width = self.width;
        init.dar_height = self.height;
        init.frame_rate_num = fps;
        init.frame_rate_den = 1;
        init.enable_ptd = 1; // let NVENC decide picture types (we force IDRs per-pic)
        init.tuning_info = ffi::NV_ENC_TUNING_INFO_LOW_LATENCY;
        init.encode_config = &mut config;
        let init_fn = funcs.nv_enc_initialize_encoder.ok_or(hw())?;
        // SAFETY: valid encoder handle; `init` and the `config` it points at are
        // fully initialized and live across this call (the driver copies them).
        nvchk(unsafe { init_fn(encoder, &mut init) })?;
        Ok(())
    }

    /// Encode one CUDA NV12 surface, returning any ready `(annex_b, pts_ns)`
    /// access units. Opens the session lazily on `buf.context`.
    fn encode(&mut self, buf: &OwnedCudaBuffer, pts_ns: u64) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        if self.session.is_none() {
            self.open_session(buf.context)?;
        }
        let session = self.session.as_ref().ok_or(G2gError::NotConfigured)?;
        if session.context != buf.context {
            // The NVDEC pool keeps one context for the stream; a mid-stream
            // context change is unsupported (would need a session re-open).
            return Err(hw());
        }
        let _ctx = ContextGuard::push(buf.context)?;

        let force = core::mem::take(&mut self.force_keyframe);
        // SAFETY: `buf`'s NV12 device pointer is valid in the pushed context for
        // the life of this call (the frame's keep-alive pins it); the session
        // handle and function table are live.
        let pkt = unsafe { self.encode_locked(buf, pts_ns, force) }?;
        Ok(pkt)
    }

    /// Register -> map -> encode -> drain one surface. Must run with `buf.context`
    /// current.
    ///
    /// # Safety
    /// `buf`'s pointers must be valid device memory in the current CUDA context,
    /// and a session must be open.
    unsafe fn encode_locked(
        &mut self,
        buf: &OwnedCudaBuffer,
        pts_ns: u64,
        force_keyframe: bool,
    ) -> Result<Vec<(Vec<u8>, u64)>, G2gError> {
        let buffer_format = self.nv_buffer_format();
        let session = self.session.as_ref().ok_or(G2gError::NotConfigured)?;
        let enc = session.encoder;
        let f = &session.funcs;

        // Register the input surface (NV12 two-plane, or packed RGBA) as a CUDA
        // device pointer. For RGBA the pitch is the full row in bytes and the
        // single plane lives at `luma_ptr`; NVENC color converts internally.
        // SAFETY: the NVENC param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then version-tag.
        let mut reg: ffi::RegisterResource = unsafe { core::mem::zeroed() };
        reg.version = ffi::NV_ENC_REGISTER_RESOURCE_VER;
        reg.resource_type = ffi::NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
        reg.width = self.width;
        reg.height = self.height;
        reg.pitch = buf.luma_pitch;
        reg.resource_to_register = buf.luma_ptr as *mut core::ffi::c_void;
        reg.buffer_format = buffer_format;
        reg.buffer_usage = ffi::NV_ENC_INPUT_IMAGE;
        // SAFETY: valid encoder + fully-initialized register struct.
        nvchk(unsafe { (f.nv_enc_register_resource.ok_or(hw())?)(enc, &mut reg) })?;
        let registered = reg.registered_resource;

        // Map it to an input buffer, with explicit cleanup on any later failure.
        // SAFETY: the NVENC param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then version-tag.
        let mut map: ffi::MapInputResource = unsafe { core::mem::zeroed() };
        map.version = ffi::NV_ENC_MAP_INPUT_RESOURCE_VER;
        map.registered_resource = registered;
        let map_fn = f.nv_enc_map_input_resource.ok_or(hw())?;
        // SAFETY: valid encoder + registered handle.
        if let Err(e) = nvchk(unsafe { map_fn(enc, &mut map) }) {
            // SAFETY: `registered` is the live registration from just above.
            unsafe { self.unregister(registered) };
            return Err(e);
        }
        let mapped = map.mapped_resource;

        // Allocate an output bitstream buffer.
        // SAFETY: the NVENC param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then version-tag.
        let mut create_bs: ffi::CreateBitstreamBuffer = unsafe { core::mem::zeroed() };
        create_bs.version = ffi::NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
        let create_fn = f.nv_enc_create_bitstream_buffer.ok_or(hw())?;
        // SAFETY: valid encoder + version-tagged struct.
        if let Err(e) = nvchk(unsafe { create_fn(enc, &mut create_bs) }) {
            // SAFETY: `mapped`/`registered` are the live handles from above.
            unsafe {
                self.unmap(mapped);
                self.unregister(registered);
            }
            return Err(e);
        }
        let output = create_bs.bitstream_buffer;

        // Submit the picture.
        // SAFETY: the NVENC param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then version-tag.
        let mut pic: ffi::PicParams = unsafe { core::mem::zeroed() };
        pic.version = ffi::NV_ENC_PIC_PARAMS_VER;
        pic.input_width = self.width;
        pic.input_height = self.height;
        pic.input_pitch = buf.luma_pitch;
        pic.frame_idx = self.frame_no;
        pic.input_time_stamp = pts_ns;
        pic.input_buffer = mapped;
        pic.output_bitstream = output;
        pic.buffer_fmt = buffer_format;
        pic.picture_struct = ffi::NV_ENC_PIC_STRUCT_FRAME;
        if force_keyframe {
            // Force an IDR and write SPS/PPS in-band so each keyframe is a valid
            // Annex-B entry point.
            pic.encode_pic_flags = ffi::NV_ENC_PIC_FLAG_FORCEIDR | ffi::NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
        }
        self.frame_no = self.frame_no.wrapping_add(1);

        // SAFETY: valid encoder + fully-initialized picture referencing the
        // mapped input and output buffers.
        let status = unsafe { (f.nv_enc_encode_picture.ok_or(hw())?)(enc, &mut pic) };

        match status {
            ffi::NV_ENC_SUCCESS => {
                // Output ready (sync mode, no reorder): lock it, copy, release.
                // SAFETY: `output` was just produced; `mapped`/`registered` are live.
                let bytes = unsafe { self.lock_copy(output) };
                let ts = bytes.as_ref().map(|(_, t)| *t).unwrap_or(pts_ns);
                // SAFETY: `output`/`mapped`/`registered` are this frame's live handles.
                unsafe {
                    self.destroy_bitstream(output);
                    self.unmap(mapped);
                    self.unregister(registered);
                }
                let (data, _) = bytes?;
                Ok(Vec::from([(data, ts)]))
            }
            ffi::NV_ENC_ERR_NEED_MORE_INPUT => {
                // The encoder is buffering (should not happen with no B-frames +
                // zero reorder delay, but handle it): release this surface; its
                // output is owned by NVENC and will surface on a later frame. We
                // keep no per-frame state for this path in v1, so we cannot emit
                // the deferred output; treat as no packet this call. This is a
                // structural no-op for the low-latency config we initialize.
                // SAFETY: `output`/`mapped`/`registered` are this frame's live handles.
                unsafe {
                    self.destroy_bitstream(output);
                    self.unmap(mapped);
                    self.unregister(registered);
                }
                Ok(Vec::new())
            }
            _ => {
                // SAFETY: `output`/`mapped`/`registered` are this frame's live handles.
                unsafe {
                    self.destroy_bitstream(output);
                    self.unmap(mapped);
                    self.unregister(registered);
                }
                Err(hw())
            }
        }
    }

    /// Lock the output bitstream, copy its Annex-B bytes to a `Vec`, unlock.
    ///
    /// # Safety
    /// `output` must be a valid, encoded bitstream buffer of the open session.
    unsafe fn lock_copy(&self, output: *mut core::ffi::c_void) -> Result<(Vec<u8>, u64), G2gError> {
        let session = self.session.as_ref().ok_or(G2gError::NotConfigured)?;
        let enc = session.encoder;
        let f = &session.funcs;
        // SAFETY: the NVENC param structs are plain old data (ints, pointers,
        // reserved arrays); all-zero is a valid initial state we then version-tag.
        let mut lock: ffi::LockBitstream = unsafe { core::mem::zeroed() };
        lock.version = ffi::NV_ENC_LOCK_BITSTREAM_VER;
        lock.output_bitstream = output;
        // SAFETY: valid encoder + the output buffer just encoded.
        nvchk(unsafe { (f.nv_enc_lock_bitstream.ok_or(hw())?)(enc, &mut lock) })?;
        let size = lock.bitstream_size_in_bytes as usize;
        let ts = lock.output_time_stamp;
        let data = if lock.bitstream_buffer_ptr.is_null() || size == 0 {
            Vec::new()
        } else {
            // SAFETY: the driver guarantees `bitstream_buffer_ptr` points at
            // `size` valid bytes until the matching unlock below.
            unsafe {
                core::slice::from_raw_parts(lock.bitstream_buffer_ptr as *const u8, size).to_vec()
            }
        };
        // SAFETY: matching unlock for the lock above.
        nvchk(unsafe { (f.nv_enc_unlock_bitstream.ok_or(hw())?)(enc, output) })?;
        Ok((data, ts))
    }

    /// # Safety: `mapped` must be a live mapped input resource of the session.
    unsafe fn unmap(&self, mapped: *mut core::ffi::c_void) {
        if let Some(s) = self.session.as_ref() {
            if let Some(unmap) = s.funcs.nv_enc_unmap_input_resource {
                // SAFETY: valid encoder + mapped handle; best-effort cleanup.
                let _ = unsafe { unmap(s.encoder, mapped) };
            }
        }
    }

    /// # Safety: `registered` must be a live registered resource of the session.
    unsafe fn unregister(&self, registered: *mut core::ffi::c_void) {
        if let Some(s) = self.session.as_ref() {
            if let Some(unreg) = s.funcs.nv_enc_unregister_resource {
                // SAFETY: valid encoder + registered handle; best-effort cleanup.
                let _ = unsafe { unreg(s.encoder, registered) };
            }
        }
    }

    /// # Safety: `output` must be a live bitstream buffer of the session.
    unsafe fn destroy_bitstream(&self, output: *mut core::ffi::c_void) {
        if let Some(s) = self.session.as_ref() {
            if let Some(destroy) = s.funcs.nv_enc_destroy_bitstream_buffer {
                // SAFETY: valid encoder + output handle; best-effort cleanup.
                let _ = unsafe { destroy(s.encoder, output) };
            }
        }
    }

    async fn emit(
        &mut self,
        packets: Vec<(Vec<u8>, u64)>,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let caps = self.output_caps();
        let feedback = crate::encoder_base::emit_packets(
            &mut self.caps_sent,
            &mut self.emitted,
            packets,
            &caps,
            out,
        )
        .await?;
        if feedback.force_keyframe {
            self.force_keyframe = true;
        }
        Ok(())
    }
}

impl Drop for NvEnc {
    fn drop(&mut self) {
        if let Some(s) = self.session.take() {
            // Destroy on the session's context, best-effort.
            if let Ok(_ctx) = ContextGuard::push(s.context) {
                if let Some(destroy) = s.funcs.nv_enc_destroy_encoder {
                    // SAFETY: `s.encoder` was opened and not yet destroyed.
                    let _ = unsafe { destroy(s.encoder) };
                }
            }
        }
    }
}

/// Map an NVENC status to a `Result`.
fn nvchk(status: ffi::NvEncStatus) -> Result<(), G2gError> {
    if status == ffi::NV_ENC_SUCCESS {
        Ok(())
    } else {
        Err(G2gError::Hardware(HardwareError::Other))
    }
}

/// Shorthand for the generic NVENC hardware error.
fn hw() -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Pushes a CUDA context current for the duration of a scope and pops it on drop,
/// leaving the thread's context stack as it was found (the [`crate::cuda`]
/// download path's push/pop, RAII-wrapped).
struct ContextGuard;

impl ContextGuard {
    fn push(context: u64) -> Result<Self, G2gError> {
        // SAFETY: `context` is a valid CUcontext (it came from an
        // `OwnedCudaBuffer` / a session opened on it).
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

impl AsyncElement for NvEnc {
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

    /// Native `DerivedOutput`: NV12 or packed RGBA (any geometry) in, H.264 at the
    /// same dims and framerate out. Any other input yields an empty set, rejected
    /// at solve.
    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::DerivedOutput(Box::new(|input: &Caps| match input {
            Caps::RawVideo {
                format: RawVideoFormat::Nv12 | RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8,
                width,
                height,
                framerate,
            } => CapsSet::one(Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: width.clone(),
                height: height.clone(),
                framerate: framerate.clone(),
            }),
            _ => CapsSet::from_alternatives(Vec::new()),
        }))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        let Caps::RawVideo { format, width, height, framerate } = absolute_caps else {
            return Err(G2gError::CapsMismatch);
        };
        if !matches!(
            format,
            RawVideoFormat::Nv12 | RawVideoFormat::Rgba8 | RawVideoFormat::Bgra8
        ) {
            return Err(G2gError::CapsMismatch);
        }
        let (Dim::Fixed(w), Dim::Fixed(h)) = (width, height) else {
            return Err(G2gError::CapsMismatch);
        };
        self.width = *w;
        self.height = *h;
        self.input_format = *format;
        self.framerate = framerate.clone();
        // The NVENC session opens lazily on the first frame (it needs the frame's
        // CUDA context), so configure only records geometry.
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "NVENC H.264 encoder",
            "Codec/Encoder/Video/Hardware",
            "Zero-copy H.264 encode of CUDA NV12 surfaces via the NVIDIA Video Codec SDK",
            "g2g",
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        NVENC_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "bitrate" => {
                let bps = value.as_uint().ok_or(PropError::Type)?;
                self.bitrate_bps = (bps as u32).max(1);
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "bitrate" => Some(PropValue::Uint(self.bitrate_bps as u64)),
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
                    let MemoryDomain::Cuda(buf) = &frame.domain else {
                        return Err(G2gError::UnsupportedDomain);
                    };
                    let packets = self.encode(buf, frame.timing.pts_ns)?;
                    self.emit(packets, out).await?;
                }
                PipelinePacket::Eos => {
                    // The low-latency config emits each frame's output inline, so
                    // there is nothing buffered to flush; the runner forwards EOS.
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

impl PadTemplates for NvEnc {
    fn pad_templates() -> Vec<PadTemplate> {
        let out = Caps::CompressedVideo {
            codec: VideoCodec::H264,
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

/// Settable properties: the target bitrate, so a `gst-launch` line can set the
/// rate without the builder.
static NVENC_PROPS: &[PropertySpec] =
    &[PropertySpec::new("bitrate", PropKind::Uint, "constant target bitrate, bits/second")];

/// Thin hand-rolled FFI for the NVIDIA Video Codec SDK (`nvEncodeAPI.h`, SDK
/// 13.0) plus the two `libcuda` context calls NVENC needs. Only the surface this
/// element uses is transcribed; the field-heavy codec-config / per-picture unions
/// are left opaque (the driver fills the config via the preset call). Every
/// `#[repr(C)]` struct carries a compile-time size assertion checked against the
/// installed header, so a mismatched SDK fails the build rather than corrupting
/// the wire layout. Field offsets are correct by faithful transcription
/// (verified against `offsetof` on this header).
// The FFI items are `pub` for 1:1 correspondence with the C header even though
// this module is private (only `super` uses them), the same shape as the
// `crate::cuda` FFI block.
#[allow(non_upper_case_globals, unreachable_pub)]
mod ffi {
    use core::ffi::c_void;

    pub type NvEncStatus = i32;
    pub const NV_ENC_SUCCESS: NvEncStatus = 0;
    pub const NV_ENC_ERR_NEED_MORE_INPUT: NvEncStatus = 17;

    pub const NVENCAPI_MAJOR_VERSION: u32 = 13;
    pub const NVENCAPI_MINOR_VERSION: u32 = 0;
    pub const NVENCAPI_VERSION: u32 = NVENCAPI_MAJOR_VERSION | (NVENCAPI_MINOR_VERSION << 24);

    /// `NVENCAPI_STRUCT_VERSION(ver)` from the header.
    pub const fn struct_version(ver: u32) -> u32 {
        NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)
    }
    /// Some structs OR an extra high bit on top of `struct_version`.
    const fn struct_version_hi(ver: u32) -> u32 {
        struct_version(ver) | (1 << 31)
    }

    pub const NV_ENCODE_API_FUNCTION_LIST_VER: u32 = struct_version(2);
    pub const NV_ENC_INITIALIZE_PARAMS_VER: u32 = struct_version_hi(7);
    pub const NV_ENC_CONFIG_VER: u32 = struct_version_hi(9);
    pub const NV_ENC_PRESET_CONFIG_VER: u32 = struct_version_hi(5);
    pub const NV_ENC_RC_PARAMS_VER: u32 = struct_version(1);
    pub const NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER: u32 = struct_version(1);
    pub const NV_ENC_CREATE_BITSTREAM_BUFFER_VER: u32 = struct_version(1);
    pub const NV_ENC_REGISTER_RESOURCE_VER: u32 = struct_version(5);
    pub const NV_ENC_MAP_INPUT_RESOURCE_VER: u32 = struct_version(4);
    pub const NV_ENC_PIC_PARAMS_VER: u32 = struct_version_hi(7);
    pub const NV_ENC_LOCK_BITSTREAM_VER: u32 = struct_version_hi(2);

    // Enum values used (all int-sized C enums).
    pub const NV_ENC_DEVICE_TYPE_CUDA: u32 = 0x1;
    pub const NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR: u32 = 0x1;
    pub const NV_ENC_INPUT_IMAGE: u32 = 0x0;
    pub const NV_ENC_BUFFER_FORMAT_NV12: u32 = 0x1;
    /// Packed `A8R8G8B8` (word order), byte order B,G,R,A on little-endian.
    pub const NV_ENC_BUFFER_FORMAT_ARGB: u32 = 0x0100_0000;
    /// Packed `A8B8G8R8` (word order), byte order R,G,B,A, i.e. wgpu `Rgba8`.
    pub const NV_ENC_BUFFER_FORMAT_ABGR: u32 = 0x1000_0000;
    pub const NV_ENC_PIC_STRUCT_FRAME: u32 = 0x1;
    pub const NV_ENC_PARAMS_RC_CBR: u32 = 0x2;
    pub const NV_ENC_TUNING_INFO_LOW_LATENCY: u32 = 0x2;
    pub const NV_ENC_PIC_FLAG_FORCEIDR: u32 = 0x2;
    pub const NV_ENC_PIC_FLAG_OUTPUT_SPSPPS: u32 = 0x4;
    pub const NVENC_INFINITE_GOPLENGTH: u32 = 0xffff_ffff;

    /// `GUID` (`_GUID`): {u32, u16, u16, [u8;8]}, 16 bytes. Passed by value to
    /// the preset / initialize calls (SysV: one 16-byte INTEGER class struct).
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct Guid {
        pub data1: u32,
        pub data2: u16,
        pub data3: u16,
        pub data4: [u8; 8],
    }
    const _: () = assert!(core::mem::size_of::<Guid>() == 16);

    const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Guid {
        Guid { data1, data2, data3, data4 }
    }

    pub const NV_ENC_CODEC_H264_GUID: Guid = guid(
        0x6bc82762,
        0x4e63,
        0x4ca4,
        [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf],
    );
    pub const NV_ENC_PRESET_P4_GUID: Guid = guid(
        0x90a7b826,
        0xdf06,
        0x4862,
        [0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81],
    );

    /// `NV_ENC_QP` (3 x u32).
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct Qp {
        pub qp_inter_p: u32,
        pub qp_inter_b: u32,
        pub qp_intra: u32,
    }

    /// `NV_ENC_RC_PARAMS` (128 bytes). Only `rate_control_mode` /
    /// `average_bit_rate` / `max_bit_rate` are set; the rest comes from the preset.
    /// The C bitfield block is one `u32` (`bitfields`).
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct RcParams {
        pub version: u32,
        pub rate_control_mode: u32,
        pub const_qp: Qp,
        pub average_bit_rate: u32,
        pub max_bit_rate: u32,
        pub vbv_buffer_size: u32,
        pub vbv_initial_delay: u32,
        pub bitfields: u32,
        pub min_qp: Qp,
        pub max_qp: Qp,
        pub initial_rc_qp: Qp,
        pub temporal_layer_idx_mask: u32,
        pub temporal_layer_qp: [u8; 8],
        pub target_quality: u8,
        pub target_quality_lsb: u8,
        pub lookahead_depth: u16,
        pub low_delay_key_frame_scale: u8,
        pub y_dc_qp_index_offset: i8,
        pub u_dc_qp_index_offset: i8,
        pub v_dc_qp_index_offset: i8,
        pub qp_map_mode: u32,
        pub multi_pass: u32,
        pub alpha_layer_bitrate_ratio: u32,
        pub cb_qp_index_offset: i8,
        pub cr_qp_index_offset: i8,
        pub reserved2: u16,
        pub lookahead_level: u32,
        pub view_bitrate_ratios: [u8; 7],
        pub reserved3: u8,
        pub reserved1: u32,
    }
    const _: () = assert!(core::mem::size_of::<RcParams>() == 128);

    /// `NV_ENC_CONFIG` (3584 bytes). The codec-specific union is opaque
    /// (`encode_codec_config`, 1792 bytes = the largest member); the driver fills
    /// it via the preset call.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct Config {
        pub version: u32,
        pub profile_guid: Guid,
        pub gop_length: u32,
        pub frame_interval_p: i32,
        pub mono_chrome_encoding: u32,
        pub frame_field_mode: u32,
        pub mv_precision: u32,
        pub rc_params: RcParams,
        pub encode_codec_config: [u32; 448],
        pub reserved: [u32; 278],
        pub reserved2: [*mut c_void; 64],
    }
    const _: () = assert!(core::mem::size_of::<Config>() == 3584);

    /// `NV_ENC_PRESET_CONFIG` (5128 bytes).
    #[repr(C)]
    pub struct PresetConfig {
        pub version: u32,
        pub reserved: u32,
        pub preset_cfg: Config,
        pub reserved1: [u32; 256],
        pub reserved2: [*mut c_void; 64],
    }
    const _: () = assert!(core::mem::size_of::<PresetConfig>() == 5128);

    /// `NV_ENC_INITIALIZE_PARAMS` (1800 bytes). Bitfield block + the ME-hint
    /// counts array are opaque (`bitfields`, `me_hint_counts`).
    #[repr(C)]
    pub struct InitializeParams {
        pub version: u32,
        pub encode_guid: Guid,
        pub preset_guid: Guid,
        pub encode_width: u32,
        pub encode_height: u32,
        pub dar_width: u32,
        pub dar_height: u32,
        pub frame_rate_num: u32,
        pub frame_rate_den: u32,
        pub enable_encode_async: u32,
        pub enable_ptd: u32,
        pub bitfields: u32,
        pub priv_data_size: u32,
        pub reserved: u32,
        pub priv_data: *mut c_void,
        pub encode_config: *mut Config,
        pub max_encode_width: u32,
        pub max_encode_height: u32,
        pub me_hint_counts: [u32; 8],
        pub tuning_info: u32,
        pub buffer_format: u32,
        pub num_state_buffers: u32,
        pub output_stats_level: u32,
        pub reserved1: [u32; 284],
        pub reserved2: [*mut c_void; 64],
    }
    const _: () = assert!(core::mem::size_of::<InitializeParams>() == 1800);

    /// `NV_ENC_OPEN_ENCODE_SESSIONEX_PARAMS` (1552 bytes).
    #[repr(C)]
    pub struct OpenEncodeSessionExParams {
        pub version: u32,
        pub device_type: u32,
        pub device: *mut c_void,
        pub reserved: *mut c_void,
        pub api_version: u32,
        pub reserved1: [u32; 253],
        pub reserved2: [*mut c_void; 64],
    }
    const _: () = assert!(core::mem::size_of::<OpenEncodeSessionExParams>() == 1552);

    /// `NV_ENC_CREATE_BITSTREAM_BUFFER` (776 bytes).
    #[repr(C)]
    pub struct CreateBitstreamBuffer {
        pub version: u32,
        pub size: u32,
        pub memory_heap: u32,
        pub reserved: u32,
        pub bitstream_buffer: *mut c_void,
        pub bitstream_buffer_ptr: *mut c_void,
        pub reserved1: [u32; 58],
        pub reserved2: [*mut c_void; 64],
    }
    const _: () = assert!(core::mem::size_of::<CreateBitstreamBuffer>() == 776);

    /// `NV_ENC_REGISTER_RESOURCE` (1536 bytes).
    #[repr(C)]
    pub struct RegisterResource {
        pub version: u32,
        pub resource_type: u32,
        pub width: u32,
        pub height: u32,
        pub pitch: u32,
        pub sub_resource_index: u32,
        pub resource_to_register: *mut c_void,
        pub registered_resource: *mut c_void,
        pub buffer_format: u32,
        pub buffer_usage: u32,
        pub p_input_fence_point: *mut c_void,
        pub chroma_offset: [u32; 2],
        pub chroma_offset_in: [u32; 2],
        pub reserved1: [u32; 244],
        pub reserved2: [*mut c_void; 61],
    }
    const _: () = assert!(core::mem::size_of::<RegisterResource>() == 1536);

    /// `NV_ENC_MAP_INPUT_RESOURCE` (1544 bytes).
    #[repr(C)]
    pub struct MapInputResource {
        pub version: u32,
        pub sub_resource_index: u32,
        pub input_resource: *mut c_void,
        pub registered_resource: *mut c_void,
        pub mapped_resource: *mut c_void,
        pub mapped_buffer_fmt: u32,
        pub reserved1: [u32; 251],
        pub reserved2: [*mut c_void; 63],
    }
    const _: () = assert!(core::mem::size_of::<MapInputResource>() == 1544);

    /// `NV_ENC_PIC_PARAMS` (3360 bytes). Everything from the codec-specific
    /// per-picture union onward (offset 80) is opaque (`tail`); we only set the
    /// leading scalar fields and leave the union zeroed.
    #[repr(C)]
    pub struct PicParams {
        pub version: u32,
        pub input_width: u32,
        pub input_height: u32,
        pub input_pitch: u32,
        pub encode_pic_flags: u32,
        pub frame_idx: u32,
        pub input_time_stamp: u64,
        pub input_duration: u64,
        pub input_buffer: *mut c_void,
        pub output_bitstream: *mut c_void,
        pub completion_event: *mut c_void,
        pub buffer_fmt: u32,
        pub picture_struct: u32,
        pub picture_type: u32,
        pub _pad: u32,
        pub tail: [u8; 3280],
    }
    const _: () = assert!(core::mem::size_of::<PicParams>() == 3360);

    /// `NV_ENC_LOCK_BITSTREAM` (1544 bytes). The C bitfield block is one `u32`
    /// (`bitfields`); we read `bitstream_size_in_bytes`, `output_time_stamp`, and
    /// `bitstream_buffer_ptr`.
    #[repr(C)]
    pub struct LockBitstream {
        pub version: u32,
        pub bitfields: u32,
        pub output_bitstream: *mut c_void,
        pub slice_offsets: *mut u32,
        pub frame_idx: u32,
        pub hw_encode_status: u32,
        pub num_slices: u32,
        pub bitstream_size_in_bytes: u32,
        pub output_time_stamp: u64,
        pub output_duration: u64,
        pub bitstream_buffer_ptr: *mut c_void,
        pub picture_type: u32,
        pub picture_struct: u32,
        pub frame_avg_qp: u32,
        pub frame_satd: u32,
        pub ltr_frame_idx: u32,
        pub ltr_frame_bitmap: u32,
        pub temporal_id: u32,
        pub intra_mb_count: u32,
        pub inter_mb_count: u32,
        pub average_mvx: i32,
        pub average_mvy: i32,
        pub alpha_layer_size_in_bytes: u32,
        pub output_stats_ptr_size: u32,
        pub reserved: u32,
        pub output_stats_ptr: *mut c_void,
        pub frame_idx_display: u32,
        pub reserved1: [u32; 219],
        pub reserved2: [*mut c_void; 63],
        pub reserved_internal: [u32; 8],
    }
    const _: () = assert!(core::mem::size_of::<LockBitstream>() == 1544);

    // Function-pointer types. The common shape is `(encoder, *mut Struct) ->
    // status`; the open / preset / destroy calls differ and are typed explicitly.
    pub type FnOpenSessionEx = unsafe extern "C" fn(
        *mut OpenEncodeSessionExParams,
        *mut *mut c_void,
    ) -> NvEncStatus;
    pub type FnInitialize = unsafe extern "C" fn(*mut c_void, *mut InitializeParams) -> NvEncStatus;
    pub type FnPresetEx = unsafe extern "C" fn(
        *mut c_void,
        Guid,
        Guid,
        u32,
        *mut PresetConfig,
    ) -> NvEncStatus;
    pub type FnCreateBitstream =
        unsafe extern "C" fn(*mut c_void, *mut CreateBitstreamBuffer) -> NvEncStatus;
    pub type FnRegister = unsafe extern "C" fn(*mut c_void, *mut RegisterResource) -> NvEncStatus;
    pub type FnMap = unsafe extern "C" fn(*mut c_void, *mut MapInputResource) -> NvEncStatus;
    pub type FnEncode = unsafe extern "C" fn(*mut c_void, *mut PicParams) -> NvEncStatus;
    pub type FnLock = unsafe extern "C" fn(*mut c_void, *mut LockBitstream) -> NvEncStatus;
    /// Handle-only calls: unlock / unmap / unregister / destroy-buffer take
    /// `(encoder, handle)`; destroy-encoder takes `(encoder)`.
    pub type FnHandle = unsafe extern "C" fn(*mut c_void, *mut c_void) -> NvEncStatus;
    pub type FnEncoder = unsafe extern "C" fn(*mut c_void) -> NvEncStatus;

    /// `NV_ENCODE_API_FUNCTION_LIST` (2552 bytes). Only the entry points this
    /// element calls are typed; the rest are opaque pointers, in exact order so
    /// the typed slots land at the right offsets (guarded by the size assertion).
    #[repr(C)]
    pub struct NvEncodeApiFunctionList {
        pub version: u32,
        pub reserved: u32,
        pub nv_enc_open_encode_session: *mut c_void,
        pub nv_enc_get_encode_guid_count: *mut c_void,
        pub nv_enc_get_encode_profile_guid_count: *mut c_void,
        pub nv_enc_get_encode_profile_guids: *mut c_void,
        pub nv_enc_get_encode_guids: *mut c_void,
        pub nv_enc_get_input_format_count: *mut c_void,
        pub nv_enc_get_input_formats: *mut c_void,
        pub nv_enc_get_encode_caps: *mut c_void,
        pub nv_enc_get_encode_preset_count: *mut c_void,
        pub nv_enc_get_encode_preset_guids: *mut c_void,
        pub nv_enc_get_encode_preset_config: *mut c_void,
        pub nv_enc_initialize_encoder: Option<FnInitialize>,
        pub nv_enc_create_input_buffer: *mut c_void,
        pub nv_enc_destroy_input_buffer: *mut c_void,
        pub nv_enc_create_bitstream_buffer: Option<FnCreateBitstream>,
        pub nv_enc_destroy_bitstream_buffer: Option<FnHandle>,
        pub nv_enc_encode_picture: Option<FnEncode>,
        pub nv_enc_lock_bitstream: Option<FnLock>,
        pub nv_enc_unlock_bitstream: Option<FnHandle>,
        pub nv_enc_lock_input_buffer: *mut c_void,
        pub nv_enc_unlock_input_buffer: *mut c_void,
        pub nv_enc_get_encode_stats: *mut c_void,
        pub nv_enc_get_sequence_params: *mut c_void,
        pub nv_enc_register_async_event: *mut c_void,
        pub nv_enc_unregister_async_event: *mut c_void,
        pub nv_enc_map_input_resource: Option<FnMap>,
        pub nv_enc_unmap_input_resource: Option<FnHandle>,
        pub nv_enc_destroy_encoder: Option<FnEncoder>,
        pub nv_enc_invalidate_ref_frames: *mut c_void,
        pub nv_enc_open_encode_session_ex: Option<FnOpenSessionEx>,
        pub nv_enc_register_resource: Option<FnRegister>,
        pub nv_enc_unregister_resource: Option<FnHandle>,
        pub nv_enc_reconfigure_encoder: *mut c_void,
        pub reserved1: *mut c_void,
        pub nv_enc_create_mv_buffer: *mut c_void,
        pub nv_enc_destroy_mv_buffer: *mut c_void,
        pub nv_enc_run_motion_estimation_only: *mut c_void,
        pub nv_enc_get_last_error_string: *mut c_void,
        pub nv_enc_set_io_cuda_streams: *mut c_void,
        pub nv_enc_get_encode_preset_config_ex: Option<FnPresetEx>,
        pub nv_enc_get_sequence_param_ex: *mut c_void,
        pub nv_enc_restore_encoder_state: *mut c_void,
        pub nv_enc_lookahead_picture: *mut c_void,
        pub reserved2: [*mut c_void; 275],
    }
    const _: () = assert!(core::mem::size_of::<NvEncodeApiFunctionList>() == 2552);

    #[link(name = "nvidia-encode")]
    extern "C" {
        /// Populate the function table (must have `version` set on input).
        #[allow(non_snake_case)]
        pub fn NvEncodeAPICreateInstance(list: *mut NvEncodeApiFunctionList) -> NvEncStatus;
    }

    #[link(name = "cuda")]
    extern "C" {
        /// Push `ctx` onto the calling thread's current-context stack.
        #[link_name = "cuCtxPushCurrent_v2"]
        pub fn cu_ctx_push_current(ctx: *mut c_void) -> i32;
        /// Pop the current context, returning it through `pctx`.
        #[link_name = "cuCtxPopCurrent_v2"]
        pub fn cu_ctx_pop_current(pctx: *mut *mut c_void) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nv12(w: u32, h: u32) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            framerate: Rate::Fixed(30 << 16),
        }
    }

    // --- Pure-logic coverage (no GPU): caps, domain, and the version macro ---

    #[test]
    fn struct_versions_match_the_sdk_macro() {
        // The probed values from nvEncodeAPI.h 13.0 (NVENCAPI_STRUCT_VERSION,
        // with the high bit for the *_VER_HI structs). Catches a macro slip.
        assert_eq!(ffi::NV_ENCODE_API_FUNCTION_LIST_VER, 1879179277);
        assert_eq!(ffi::NV_ENC_INITIALIZE_PARAMS_VER, 4026990605);
        assert_eq!(ffi::NV_ENC_CONFIG_VER, 4027121677);
        assert_eq!(ffi::NV_ENC_PRESET_CONFIG_VER, 4026859533);
        assert_eq!(ffi::NV_ENC_RC_PARAMS_VER, 1879113741);
        assert_eq!(ffi::NV_ENC_REGISTER_RESOURCE_VER, 1879375885);
        assert_eq!(ffi::NV_ENC_MAP_INPUT_RESOURCE_VER, 1879310349);
        assert_eq!(ffi::NV_ENC_PIC_PARAMS_VER, 4026990605);
        assert_eq!(ffi::NV_ENC_LOCK_BITSTREAM_VER, 4026662925);
        assert_eq!(ffi::NVENCAPI_VERSION, 13);
    }

    #[test]
    fn h264_codec_guid_bytes() {
        // {6BC82762-4E63-4ca4-AA85-1E50F321F6BF}
        let g = ffi::NV_ENC_CODEC_H264_GUID;
        assert_eq!(g.data1, 0x6bc82762);
        assert_eq!(g.data2, 0x4e63);
        assert_eq!(g.data3, 0x4ca4);
        assert_eq!(g.data4, [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf]);
    }

    #[test]
    fn caps_constraint_is_nv12_to_h264() {
        let e = NvEnc::new();
        let CapsConstraint::DerivedOutput(derive) = e.caps_constraint_as_transform() else {
            panic!("expected DerivedOutput");
        };
        // NV12 in -> H.264 at the same geometry.
        let out = derive(&nv12(1920, 1080));
        assert!(out.accepts(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(1920),
            height: Dim::Fixed(1080),
            framerate: Rate::Fixed(30 << 16),
        }));
        // Non-NV12 (I420) yields an empty set, rejected at solve.
        let i420 = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        };
        assert!(derive(&i420).alternatives().is_empty());
    }

    #[test]
    fn configure_rejects_non_nv12() {
        let mut ok = NvEnc::new();
        assert!(ok.configure_pipeline(&nv12(1280, 720)).is_ok());
        assert_eq!(ok.width, 1280);
        assert_eq!(ok.height, 720);

        let mut bad = NvEnc::new();
        let i420 = Caps::RawVideo {
            format: RawVideoFormat::I420,
            width: Dim::Fixed(1280),
            height: Dim::Fixed(720),
            framerate: Rate::Any,
        };
        assert_eq!(bad.configure_pipeline(&i420).err(), Some(G2gError::CapsMismatch));
    }

    #[tokio::test]
    async fn system_memory_frame_is_rejected() {
        // NvEnc is the device-resident encoder: a System frame is the wrong
        // domain (FfmpegH264Enc handles that path). No GPU needed for this.
        use g2g_core::frame::Frame;
        use g2g_core::memory::SystemSlice;
        use g2g_core::{FrameTiming, PushOutcome};
        use core::future::Future;
        use core::pin::Pin;

        struct NullSink;
        impl OutputSink for NullSink {
            fn push<'a>(
                &'a mut self,
                _p: PipelinePacket,
            ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
                Box::pin(async { Ok(PushOutcome::Accepted) })
            }
        }

        let mut enc = NvEnc::new();
        enc.configure_pipeline(&nv12(320, 240)).unwrap();
        let frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(alloc::vec![0u8; 320 * 240 * 3 / 2].into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        let err = enc.process(PipelinePacket::DataFrame(frame), &mut NullSink).await;
        assert_eq!(err.err(), Some(G2gError::UnsupportedDomain));
    }

    // --- On-hardware round trip (RTX 3060 host): NV12 CUDA surface -> NvEnc ->
    // Annex-B, decoded back through FfmpegVideoDec. Skips cleanly with no NVIDIA
    // GPU / SDK. Needs the `ffmpeg` feature for the decode-back leg. ---

    #[cfg(feature = "ffmpeg")]
    #[tokio::test]
    async fn nvenc_round_trips_a_cuda_nv12_surface() {
        gpu_round_trip().await;
    }

    /// CUDA driver FFI used only to synthesize a device-resident NV12 frame for
    /// the round-trip test (allocate, upload, free).
    #[cfg(feature = "ffmpeg")]
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
            #[link_name = "cuMemAlloc_v2"]
            pub fn cu_mem_alloc(dptr: *mut u64, bytesize: usize) -> i32;
            #[link_name = "cuMemFree_v2"]
            pub fn cu_mem_free(dptr: u64) -> i32;
            #[link_name = "cuMemcpyHtoD_v2"]
            pub fn cu_memcpy_htod(dst: u64, src: *const c_void, bytesize: usize) -> i32;
        }
    }

    /// Frees one device allocation when the synthesized frame is dropped.
    #[cfg(feature = "ffmpeg")]
    #[derive(Debug)]
    struct DevAlloc {
        dptr: u64,
        ctx: u64,
    }
    #[cfg(feature = "ffmpeg")]
    impl g2g_core::memory::CudaKeepAlive for DevAlloc {}
    #[cfg(feature = "ffmpeg")]
    impl Drop for DevAlloc {
        fn drop(&mut self) {
            // SAFETY: free on the allocating context; best-effort.
            unsafe {
                let _ = ffi::cu_ctx_push_current(self.ctx as *mut core::ffi::c_void);
                let _ = cu::cu_mem_free(self.dptr);
                let mut popped = core::ptr::null_mut();
                let _ = ffi::cu_ctx_pop_current(&mut popped);
            }
        }
    }

    #[cfg(feature = "ffmpeg")]
    async fn gpu_round_trip() {
        use alloc::sync::Arc;
        use core::future::Future;
        use core::pin::Pin;
        use g2g_core::frame::Frame;
        use g2g_core::memory::SystemSlice;
        use g2g_core::{FrameTiming, PushOutcome};

        const W: u32 = 320;
        const H: u32 = 240;
        let (w, h) = (W as usize, H as usize);
        let size = w * h * 3 / 2; // tight NV12, pitch == width

        // Bring up a CUDA context; skip if no NVIDIA GPU on this host.
        // SAFETY: standard CUDA driver bring-up; every call's result is checked
        // and the path bails before using a handle on failure.
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
        // Tear the context down last (after all DevAlloc drops).
        struct CtxGuard(u64);
        impl Drop for CtxGuard {
            fn drop(&mut self) {
                // SAFETY: `self.0` is the context created above, destroyed once.
                unsafe {
                    let _ = cu::cu_ctx_destroy(self.0 as *mut core::ffi::c_void);
                }
            }
        }
        let _ctx_guard = CtxGuard(ctx);

        // Build a moving NV12 pattern as a CUDA-resident frame.
        let make_frame = |seq: u64| -> Option<Frame> {
            let mut host = alloc::vec![0u8; size];
            for y in 0..h {
                for x in 0..w {
                    host[y * w + x] = ((x + y + seq as usize * 7) & 0xff) as u8;
                }
            }
            for c in &mut host[w * h..] {
                *c = 128; // neutral chroma
            }
            // SAFETY: allocate + upload one NV12 surface in the test context,
            // pushed/popped around the calls; `host` outlives the copy.
            unsafe {
                let _ = ffi::cu_ctx_push_current(ctx as *mut core::ffi::c_void);
                let mut dptr = 0u64;
                let ok = cu::cu_mem_alloc(&mut dptr, size) == 0
                    && cu::cu_memcpy_htod(dptr, host.as_ptr() as *const core::ffi::c_void, size) == 0;
                let mut popped = core::ptr::null_mut();
                let _ = ffi::cu_ctx_pop_current(&mut popped);
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

        #[derive(Default)]
        struct CaptureSink {
            caps: Vec<Caps>,
            frames: Vec<Vec<u8>>,
        }
        impl OutputSink for CaptureSink {
            fn push<'a>(
                &'a mut self,
                packet: PipelinePacket,
            ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
                Box::pin(async move {
                    match packet {
                        PipelinePacket::CapsChanged(c) => self.caps.push(c),
                        PipelinePacket::DataFrame(f) => {
                            if let MemoryDomain::System(s) = &f.domain {
                                self.frames.push(s.as_slice().to_vec());
                            }
                        }
                        _ => {}
                    }
                    Ok(PushOutcome::Accepted)
                })
            }
        }

        let mut enc = NvEnc::new();
        enc.configure_pipeline(&nv12(W, H)).unwrap();
        let mut sink = CaptureSink::default();
        for i in 0..10u64 {
            let Some(frame) = make_frame(i) else {
                std::eprintln!("skipping: CUDA alloc/upload failed");
                return;
            };
            // First frame opens the session; if NVENC is unavailable, skip.
            if enc.process(PipelinePacket::DataFrame(frame), &mut sink).await.is_err() {
                std::eprintln!("skipping: NVENC unavailable on this host");
                return;
            }
        }
        enc.process(PipelinePacket::Eos, &mut sink).await.unwrap();

        assert!(!sink.frames.is_empty(), "NVENC produced H.264 access units");
        assert_eq!(
            sink.caps,
            alloc::vec![Caps::CompressedVideo {
                codec: VideoCodec::H264,
                width: Dim::Fixed(W),
                height: Dim::Fixed(H),
                framerate: Rate::Fixed(30 << 16),
            }],
            "output caps announced once"
        );
        let first = &sink.frames[0];
        let annex_b = first.starts_with(&[0, 0, 0, 1]) || first.starts_with(&[0, 0, 1]);
        assert!(annex_b, "NVENC output is Annex-B framed, got {:?}", &first[..4.min(first.len())]);

        // Decode the stream back to prove it is a real, decodable H.264 bitstream.
        let mut dec = crate::ffmpegdec::FfmpegVideoDec::new();
        dec.configure_pipeline(&Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(W),
            height: Dim::Fixed(H),
            framerate: Rate::Fixed(30 << 16),
        })
        .expect("open H.264 decoder");
        let mut dsink = CaptureSink::default();
        for au in &sink.frames {
            let f = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
                FrameTiming::default(),
                0,
            );
            dec.process(PipelinePacket::DataFrame(f), &mut dsink).await.expect("decode AU");
        }
        dec.process(PipelinePacket::Eos, &mut dsink).await.expect("drain decoder");

        let geometry = dsink.caps.iter().find_map(|c| match c {
            Caps::RawVideo { width: Dim::Fixed(w), height: Dim::Fixed(h), .. } => Some((*w, *h)),
            _ => None,
        });
        assert_eq!(geometry, Some((W, H)), "NVENC stream decodes back to {W}x{H}");
        assert!(!dsink.frames.is_empty(), "NVENC stream decoded to raw frames");
    }
}
