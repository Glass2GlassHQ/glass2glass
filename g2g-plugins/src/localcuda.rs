//! Local zero-copy CUDA transport elements (M556 phase 2, `local-ipc`): the
//! GPU-resident analog of the [`RemoteSink`](crate::remotesink) /
//! [`RemoteSrc`](crate::remotesrc) pair. `LocalCudaSink` (Unix-socket client) and
//! `LocalCudaSrc` (Unix-socket server) carry a `MemoryDomain::Cuda` NV12 frame
//! from one process to another *on the same machine + GPU* without a
//! device->host->device round trip, by sharing the frame's VRAM through a CUDA
//! IPC handle (see [`crate::localipc`]).
//!
//! # Lifetime and the one device-to-device copy
//!
//! The hard part of sharing device memory across processes is lifetime: the
//! producer's allocation must stay valid until the consumer is done reading it.
//! Rather than couple the two processes' whole pipelines, the receive side takes
//! one **on-GPU** copy (`cuMemcpyDtoD`, no host / PCIe round trip) of the mapped
//! allocation into its own buffer, then acknowledges; the producer holds the
//! source frame only until that ack (one frame in flight), then frees it. So the
//! per-frame protocol over the socket is a synchronous ping-pong:
//!   sink -> src: leading `CapsChanged` (NV12 dims), then one frame descriptor
//!                (IPC handle + geometry + timing) per frame, then `Eos`.
//!   src -> sink: one `u64` ack per frame, written after the device->device copy.
//! The win over the CPU wire codec is that nothing crosses PCIe: the frame stays
//! in VRAM the whole way. Eliminating even the receive-side d2d copy (the
//! consumer reads the mapped pointer directly, e.g. NVENC-from-mapped) is the
//! true-zero-copy-consume follow-up; it needs a consume-then-ack handshake this
//! simpler design trades away for runner-independence.
//!
//! Contract: NVIDIA-only (the `cuda` gate); both ends single-thread executors
//! (the CUDA context is thread-affine, as `MfDecode` documents). The sink handles
//! both a whole-allocation frame (`CudaUpload`, base == `luma_ptr`) and a frame
//! whose planes are a *sub-allocation* of a larger pool (an NVDEC decode pool):
//! it resolves the enclosing allocation with `cuMemGetAddressRange` and carries
//! each plane's offset within it, so nothing assumes `luma_ptr` is the base.

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use std::os::unix::net::UnixListener as StdUnixListener;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::UnixStream;

use g2g_core::memory::{CudaKeepAlive, OwnedCudaBuffer};
use g2g_core::pad_template::{PadTemplate, PadTemplates};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, ElementMetadata, Frame,
    FrameTiming, G2gError, HardwareError, MemoryDomain, OutputSink, PipelinePacket, PropError,
    PropKind, PropValue, PropertySpec, Rate, RawVideoFormat,
};

/// The `location` property shared by both ends: the Unix socket path.
const LOCATION_PROP: &[PropertySpec] = &[PropertySpec::new(
    "location",
    PropKind::Str,
    "Unix socket path",
)];

use crate::localipc;

// ---- wire framing over the Unix socket ----

const TAG_CAPS: u8 = 0;
const TAG_FRAME: u8 = 1;
const TAG_EOS: u8 = 2;

/// Fixed length of a serialized frame descriptor (see `encode_desc`).
const DESC_LEN: usize = 8 * 3      // alloc_size, luma_off, chroma_off
    + 4 * 4                        // luma_pitch, chroma_pitch, width, height
    + 8                            // sequence
    + 8 * 5 + 1                    // timing (pts,dts,dur,cap,arr) + keyframe
    + localipc::CUDA_IPC_HANDLE_SIZE;

/// The per-frame descriptor crossing the socket: how to map + slice the shared
/// allocation, plus the frame's timing / sequence. Pure data (the `handle` is the
/// only device-specific field, and it is plain bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
struct FrameDesc {
    alloc_size: u64,
    luma_off: u64,
    chroma_off: u64,
    luma_pitch: u32,
    chroma_pitch: u32,
    width: u32,
    height: u32,
    sequence: u64,
    timing: FrameTiming,
    handle: localipc::CudaIpcHandle,
}

fn encode_desc(d: &FrameDesc) -> Vec<u8> {
    let mut b = Vec::with_capacity(DESC_LEN);
    b.extend_from_slice(&d.alloc_size.to_le_bytes());
    b.extend_from_slice(&d.luma_off.to_le_bytes());
    b.extend_from_slice(&d.chroma_off.to_le_bytes());
    b.extend_from_slice(&d.luma_pitch.to_le_bytes());
    b.extend_from_slice(&d.chroma_pitch.to_le_bytes());
    b.extend_from_slice(&d.width.to_le_bytes());
    b.extend_from_slice(&d.height.to_le_bytes());
    b.extend_from_slice(&d.sequence.to_le_bytes());
    b.extend_from_slice(&d.timing.pts_ns.to_le_bytes());
    b.extend_from_slice(&d.timing.dts_ns.to_le_bytes());
    b.extend_from_slice(&d.timing.duration_ns.to_le_bytes());
    b.extend_from_slice(&d.timing.capture_ns.to_le_bytes());
    b.extend_from_slice(&d.timing.arrival_ns.to_le_bytes());
    b.push(d.timing.keyframe as u8);
    b.extend_from_slice(&d.handle);
    b
}

fn decode_desc(b: &[u8]) -> Option<FrameDesc> {
    if b.len() < DESC_LEN {
        return None;
    }
    let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    let mut handle = [0u8; localipc::CUDA_IPC_HANDLE_SIZE];
    let hoff = DESC_LEN - localipc::CUDA_IPC_HANDLE_SIZE;
    handle.copy_from_slice(&b[hoff..DESC_LEN]);
    Some(FrameDesc {
        alloc_size: u64_at(0),
        luma_off: u64_at(8),
        chroma_off: u64_at(16),
        luma_pitch: u32_at(24),
        chroma_pitch: u32_at(28),
        width: u32_at(32),
        height: u32_at(36),
        sequence: u64_at(40),
        timing: FrameTiming {
            pts_ns: u64_at(48),
            dts_ns: u64_at(56),
            duration_ns: u64_at(64),
            capture_ns: u64_at(72),
            arrival_ns: u64_at(80),
            keyframe: b[88] != 0,
        },
        handle,
    })
}

fn io_err(_: std::io::Error) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}

/// Write one tagged, length-framed message: `[tag][u32 len][payload]`.
async fn write_msg<W: AsyncWrite + Unpin>(
    sock: &mut W,
    tag: u8,
    payload: &[u8],
) -> Result<(), G2gError> {
    sock.write_all(&[tag]).await.map_err(io_err)?;
    sock.write_all(&(payload.len() as u32).to_le_bytes())
        .await
        .map_err(io_err)?;
    sock.write_all(payload).await.map_err(io_err)?;
    Ok(())
}

/// Read one tagged message; `Ok(None)` on a clean EOF at a message boundary.
async fn read_msg<R: AsyncRead + Unpin>(sock: &mut R) -> Result<Option<(u8, Vec<u8>)>, G2gError> {
    let mut tag = [0u8; 1];
    match sock.read_exact(&mut tag).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(io_err(e)),
    }
    let mut len = [0u8; 4];
    sock.read_exact(&mut len).await.map_err(io_err)?;
    let mut payload = vec![0u8; u32::from_le_bytes(len) as usize];
    sock.read_exact(&mut payload).await.map_err(io_err)?;
    Ok(Some((tag[0], payload)))
}

/// NV12 plane byte sizes (luma, chroma) for a frame descriptor.
fn plane_sizes(d: &FrameDesc) -> (u64, u64) {
    let luma = d.luma_pitch as u64 * d.height as u64;
    let chroma = d.chroma_pitch as u64 * (d.height as u64).div_ceil(2);
    (luma, chroma)
}

/// Both planes must lie within the exported allocation. Never trust the transport:
/// a bogus offset / pitch must fail the map rather than read out of bounds or copy
/// past the allocation.
fn planes_in_bounds(d: &FrameDesc) -> bool {
    let (luma, chroma) = plane_sizes(d);
    d.luma_off
        .checked_add(luma)
        .is_some_and(|e| e <= d.alloc_size)
        && d.chroma_off
            .checked_add(chroma)
            .is_some_and(|e| e <= d.alloc_size)
}

fn nv12(width: u32, height: u32) -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Fixed(width),
        height: Dim::Fixed(height),
        // A source must advertise a fixable rate: negotiation's fixate() rejects
        // Rate::Any. Per-frame timing crosses in the descriptor; this nominal
        // framerate only satisfies fixation (the transport is rate-agnostic).
        framerate: Rate::Fixed(30 << 16),
    }
}

fn dims_of(caps: &Caps) -> Option<(u32, u32)> {
    match caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(w),
            height: Dim::Fixed(h),
            ..
        } => Some((*w, *h)),
        _ => None,
    }
}

// ---- sink (client) ----

#[derive(Debug)]
pub struct LocalCudaSink {
    path: String,
    socket: Option<UnixStream>,
    configured_caps: Option<Caps>,
    caps_sent: bool,
    configured: bool,
    sent: u64,
}

impl LocalCudaSink {
    /// Send NV12 CUDA frames to `path` (the [`LocalCudaSrc`] Unix socket).
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            socket: None,
            configured_caps: None,
            caps_sent: false,
            configured: false,
            sent: 0,
        }
    }

    /// Frames handed to the peer (and acked). Useful in tests.
    pub fn sent(&self) -> u64 {
        self.sent
    }
}

impl PadTemplates for LocalCudaSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink(CapsSet::one(nv12_any()))])
    }
}

impl AsyncElement for LocalCudaSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        upstream_caps.intersect(&nv12_any())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::Accepts(CapsSet::one(nv12_any()))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.configured_caps = Some(absolute_caps.clone());
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            if self.socket.is_none() {
                self.socket = Some(UnixStream::connect(&self.path).await.map_err(io_err)?);
            }
            // Announce the NV12 dims once so the receiver discovers its caps.
            if !self.caps_sent {
                let (w, h) = self
                    .configured_caps
                    .as_ref()
                    .and_then(dims_of)
                    .ok_or(G2gError::CapsMismatch)?;
                let mut payload = Vec::with_capacity(8);
                payload.extend_from_slice(&w.to_le_bytes());
                payload.extend_from_slice(&h.to_le_bytes());
                let sock = self.socket.as_mut().unwrap();
                write_msg(sock, TAG_CAPS, &payload).await?;
                self.caps_sent = true;
            }
            match packet {
                PipelinePacket::DataFrame(frame) => {
                    let MemoryDomain::Cuda(buf) = &frame.domain else {
                        // Only device-resident CUDA frames share via IPC; a CPU
                        // frame belongs on the wire codec instead.
                        return Err(G2gError::UnsupportedDomain);
                    };
                    // Resolve the enclosing allocation: for CudaUpload it is the
                    // luma pointer itself (offset 0); for an NVDEC pool frame the
                    // planes sit at offsets within a larger pool allocation. CUDA
                    // IPC exports whole allocations, so we export the base and carry
                    // each plane's offset.
                    // SAFETY: `luma_ptr` is a live device pointer in the frame's
                    // (current) CUDA context; the frame is held for the whole
                    // send + ack below, so the allocation stays valid until the
                    // receiver is done with it.
                    let (base, alloc_size) = unsafe { localipc::address_range(buf.luma_ptr)? };
                    let luma_off = buf.luma_ptr.saturating_sub(base);
                    let chroma_off = buf.chroma_ptr.saturating_sub(base);
                    // SAFETY: `base` is the base of that live allocation.
                    let handle = unsafe { localipc::ipc_export(base)? };
                    let desc = FrameDesc {
                        alloc_size,
                        luma_off,
                        chroma_off,
                        luma_pitch: buf.luma_pitch,
                        chroma_pitch: buf.chroma_pitch,
                        width: buf.width,
                        height: buf.height,
                        sequence: frame.sequence,
                        timing: frame.timing,
                        handle,
                    };
                    let sock = self.socket.as_mut().unwrap();
                    write_msg(sock, TAG_FRAME, &encode_desc(&desc)).await?;
                    // Wait for the receiver's ack (it has taken its on-GPU copy),
                    // so `frame` (and its VRAM) stays alive exactly until then.
                    let mut ack = [0u8; 8];
                    sock.read_exact(&mut ack).await.map_err(io_err)?;
                    self.sent += 1;
                    // `frame` drops here: the producer may now free the source.
                }
                PipelinePacket::Eos => {
                    if let Some(sock) = self.socket.as_mut() {
                        let _ = write_msg(sock, TAG_EOS, &[]).await;
                        let _ = sock.shutdown().await;
                    }
                }
                PipelinePacket::CapsChanged(_)
                | PipelinePacket::Flush
                | PipelinePacket::Segment(_) => {}
                // future PipelinePacket variants: no-op (terminal sink).
                _ => {}
            }
            Ok(())
        })
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LOCATION_PROP
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.path = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.path.clone())),
            _ => None,
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Local CUDA sink",
            "Sink/Network",
            "Shares NV12 CUDA frames with a same-machine peer over a Unix socket via CUDA IPC (no PCIe round trip)",
            "g2g",
        )
    }
}

// ---- source (server) ----

/// Owns the receive-side CUDA context; destroyed once every frame that borrows
/// it (their dest allocations) is released.
#[derive(Debug)]
struct SrcCtx(u64);

// SAFETY: the context is created and used on the single executor thread; frames
// reference its device memory read-downstream only (documented thread-affinity).
unsafe impl Send for SrcCtx {}
// SAFETY: see above.
unsafe impl Sync for SrcCtx {}

impl Drop for SrcCtx {
    fn drop(&mut self) {
        localipc::destroy_context(self.0);
    }
}

/// Keep-alive for a received frame: frees the receive-side dest allocation on
/// drop and pins the context for the frame's lifetime.
#[derive(Debug)]
struct DestAlloc {
    dptr: u64,
    _ctx: Arc<SrcCtx>,
}

impl Drop for DestAlloc {
    fn drop(&mut self) {
        // SAFETY: `dptr` is our dest allocation, freed once; the context is still
        // pinned (`_ctx`) and current on this (single) thread.
        unsafe {
            let _ = localipc::free(self.dptr);
        }
    }
}

impl CudaKeepAlive for DestAlloc {}

/// Zero-copy keep-alive: the emitted frame points directly at the *producer's*
/// mapped VRAM (no receive-side copy). On drop (the frame is fully consumed
/// downstream) it closes the IPC mapping and signals the run loop, via the
/// oneshot, that the producer may now free the source. Pins the context.
struct IpcMapping {
    base: u64,
    _ctx: Arc<SrcCtx>,
    done: Option<tokio::sync::oneshot::Sender<()>>,
}

impl core::fmt::Debug for IpcMapping {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IpcMapping")
            .field("base", &self.base)
            .finish_non_exhaustive()
    }
}

impl Drop for IpcMapping {
    fn drop(&mut self) {
        // Close first (importer unmaps before the exporter is told to free), then
        // signal the run loop to ack the producer.
        // SAFETY: `base` came from ipc_open, closed once; the context is pinned
        // (`_ctx`) and current on this (single) thread.
        unsafe {
            let _ = localipc::ipc_close(self.base);
        }
        if let Some(tx) = self.done.take() {
            let _ = tx.send(());
        }
    }
}

impl CudaKeepAlive for IpcMapping {}

#[derive(Debug)]
pub struct LocalCudaSrc {
    path: String,
    listener: Option<StdUnixListener>,
    socket: Option<UnixStream>,
    discovered: Option<Caps>,
    ctx: Option<Arc<SrcCtx>>,
    configured: bool,
    frame_limit: u64,
    /// When set, emit the producer's mapped VRAM directly (true zero copy) and
    /// ack only once the frame is fully consumed downstream, instead of taking a
    /// receive-side device->device copy and acking immediately.
    direct: bool,
}

impl LocalCudaSrc {
    /// Receive NV12 CUDA frames on the Unix socket `path` (the [`LocalCudaSink`]
    /// connects here). Any stale socket file at `path` is removed on bind.
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            listener: None,
            socket: None,
            discovered: None,
            ctx: None,
            configured: false,
            frame_limit: 0,
            direct: false,
        }
    }

    /// Stop after `n` frames and emit EOS (the bounded / test path).
    pub fn with_frame_limit(mut self, n: u64) -> Self {
        self.frame_limit = n;
        self
    }

    /// Enable true zero-copy consume: emit the producer's mapped VRAM directly
    /// (no receive-side device->device copy), acking the producer only once each
    /// frame is fully consumed downstream. Requires a consumer that releases each
    /// frame promptly (one frame in flight); a consumer that holds or fans out
    /// frames would stall the producer, so the default (copy) mode suits those.
    pub fn zero_copy(mut self) -> Self {
        self.direct = true;
        self
    }

    /// Accept the sender and read its first message (the caps). Idempotent.
    async fn ensure_connected(&mut self) -> Result<Caps, G2gError> {
        if let Some(caps) = &self.discovered {
            return Ok(caps.clone());
        }
        let listener = match self.listener.take() {
            Some(l) => l,
            None => {
                // Fresh bind: clear any stale socket file first.
                let _ = std::fs::remove_file(&self.path);
                StdUnixListener::bind(&self.path).map_err(io_err)?
            }
        };
        listener.set_nonblocking(true).map_err(io_err)?;
        let listener = tokio::net::UnixListener::from_std(listener).map_err(io_err)?;
        let (mut stream, _) = listener.accept().await.map_err(io_err)?;
        // First message is the NV12 caps (dims).
        let (tag, payload) = read_msg(&mut stream)
            .await?
            .ok_or(G2gError::NotConfigured)?;
        if tag != TAG_CAPS || payload.len() < 8 {
            return Err(G2gError::Hardware(HardwareError::Other));
        }
        let w = u32::from_le_bytes(payload[0..4].try_into().unwrap());
        let h = u32::from_le_bytes(payload[4..8].try_into().unwrap());
        let caps = nv12(w, h);
        self.socket = Some(stream);
        self.discovered = Some(caps.clone());
        Ok(caps)
    }
}

impl SourceLoop for LocalCudaSrc {
    type RunFuture<'a>
        = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;
    type CapsFuture<'a>
        = Pin<Box<dyn Future<Output = Result<Caps, G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        Box::pin(async move { self.ensure_connected().await })
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        Box::pin(async move {
            let caps = self.ensure_connected().await?;
            Ok(CapsConstraint::Produces(CapsSet::one(caps)))
        })
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        // Our own CUDA context on device 0 (same device as the sender). Created
        // on this executor thread; the context is thread-affine, so the runner
        // must be single-thread (documented).
        let ctx = localipc::init_context(0)?;
        self.ctx = Some(Arc::new(SrcCtx(ctx)));
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        LOCATION_PROP
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "location" => {
                self.path = value.as_str().ok_or(PropError::Type)?.into();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "location" => Some(PropValue::Str(self.path.clone())),
            _ => None,
        }
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Local CUDA source",
            "Source/Network",
            "Receives NV12 CUDA frames from a same-machine LocalCudaSink over a Unix socket via CUDA IPC",
            "g2g",
        )
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let caps = self.discovered.clone().ok_or(G2gError::NotConfigured)?;
            let ctx = self.ctx.clone().ok_or(G2gError::NotConfigured)?;
            let direct = self.direct;
            // Split so a frame descriptor can be read while the ack for the
            // previous frame is written (the zero-copy path waits on consumption
            // between the read and the ack).
            let (mut rd, mut wr) = self
                .socket
                .take()
                .ok_or(G2gError::NotConfigured)?
                .into_split();
            out.push(PipelinePacket::CapsChanged(caps)).await?;

            let limit = self.frame_limit;
            let mut emitted = 0u64;
            loop {
                let (tag, payload) = match read_msg(&mut rd).await? {
                    Some(m) => m,
                    None => {
                        out.push(PipelinePacket::Eos).await?;
                        break;
                    }
                };
                match tag {
                    TAG_EOS => {
                        out.push(PipelinePacket::Eos).await?;
                        break;
                    }
                    TAG_FRAME => {
                        let desc = decode_desc(&payload)
                            .ok_or(G2gError::Hardware(HardwareError::Other))?;
                        if !planes_in_bounds(&desc) {
                            return Err(G2gError::Hardware(HardwareError::Other));
                        }
                        if direct {
                            // True zero copy: emit the producer's mapped VRAM
                            // directly; ack only once the frame is fully consumed
                            // downstream (its keep-alive drops), so the producer
                            // holds the source exactly that long.
                            // SAFETY: our context (device 0) is current; the sender
                            // holds the source allocation alive until we ack.
                            let base = unsafe { localipc::ipc_open(&desc.handle)? };
                            let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
                            let buf = OwnedCudaBuffer::new(
                                base + desc.luma_off,
                                base + desc.chroma_off,
                                desc.luma_pitch,
                                desc.chroma_pitch,
                                desc.width,
                                desc.height,
                                ctx.0,
                                Arc::new(IpcMapping {
                                    base,
                                    _ctx: Arc::clone(&ctx),
                                    done: Some(done_tx),
                                }),
                            );
                            out.push(PipelinePacket::DataFrame(Frame {
                                domain: MemoryDomain::Cuda(buf),
                                timing: desc.timing,
                                sequence: desc.sequence,
                                meta: Default::default(),
                            }))
                            .await?;
                            // Wait until downstream has released the frame (the
                            // IpcMapping dropped, closing our mapping), then tell
                            // the producer it may free the source.
                            let _ = done_rx.await;
                            wr.write_all(&desc.sequence.to_le_bytes())
                                .await
                                .map_err(io_err)?;
                        } else {
                            // Copy mode: take one on-GPU copy into our own packed
                            // buffer, close the mapping, ack immediately (the
                            // producer may free while our copy lives on
                            // independently). The two planes are copied separately
                            // (a pool frame's luma / chroma need not be contiguous)
                            // into a tight luma||chroma destination.
                            let (luma_sz, chroma_sz) = plane_sizes(&desc);
                            let dest_size = (luma_sz + chroma_sz) as usize;
                            // SAFETY: our context (device 0) is current; the sender
                            // holds the source allocation alive until our ack below;
                            // the offsets are bounds-checked (planes_in_bounds).
                            let dest = unsafe {
                                let src_base = localipc::ipc_open(&desc.handle)?;
                                let dest = match localipc::alloc(dest_size) {
                                    Ok(d) => d,
                                    Err(e) => {
                                        let _ = localipc::ipc_close(src_base);
                                        return Err(e);
                                    }
                                };
                                let copied = localipc::dtod(
                                    dest,
                                    src_base + desc.luma_off,
                                    luma_sz as usize,
                                )
                                .and_then(|_| {
                                    localipc::dtod(
                                        dest + luma_sz,
                                        src_base + desc.chroma_off,
                                        chroma_sz as usize,
                                    )
                                });
                                let _ = localipc::ipc_close(src_base);
                                if let Err(e) = copied {
                                    let _ = localipc::free(dest);
                                    return Err(e);
                                }
                                dest
                            };
                            wr.write_all(&desc.sequence.to_le_bytes())
                                .await
                                .map_err(io_err)?;
                            let buf = OwnedCudaBuffer::new(
                                dest,
                                dest + luma_sz,
                                desc.luma_pitch,
                                desc.chroma_pitch,
                                desc.width,
                                desc.height,
                                ctx.0,
                                Arc::new(DestAlloc {
                                    dptr: dest,
                                    _ctx: Arc::clone(&ctx),
                                }),
                            );
                            out.push(PipelinePacket::DataFrame(Frame {
                                domain: MemoryDomain::Cuda(buf),
                                timing: desc.timing,
                                sequence: desc.sequence,
                                meta: Default::default(),
                            }))
                            .await?;
                        }
                        emitted += 1;
                        if limit != 0 && emitted >= limit {
                            out.push(PipelinePacket::Eos).await?;
                            break;
                        }
                    }
                    _ => {} // caps re-announce or unknown: ignore
                }
            }
            Ok(emitted)
        })
    }
}

/// NV12 caps with open geometry (the sink's accepted set / caps do not encode
/// the memory domain).
fn nv12_any() -> Caps {
    Caps::RawVideo {
        format: RawVideoFormat::Nv12,
        width: Dim::Any,
        height: Dim::Any,
        framerate: Rate::Any,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_descriptor_round_trips() {
        let mut handle = [0u8; localipc::CUDA_IPC_HANDLE_SIZE];
        for (i, b) in handle.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(5).wrapping_add(2);
        }
        let d = FrameDesc {
            alloc_size: 3_133_440,
            luma_off: 0,
            chroma_off: 2_088_960,
            luma_pitch: 2048,
            chroma_pitch: 2048,
            width: 1920,
            height: 1080,
            sequence: 77,
            timing: FrameTiming {
                pts_ns: 1_000_000,
                dts_ns: 900_000,
                duration_ns: 33_000,
                capture_ns: 5,
                arrival_ns: 6,
                keyframe: true,
            },
            handle,
        };
        let bytes = encode_desc(&d);
        assert_eq!(bytes.len(), DESC_LEN);
        assert_eq!(decode_desc(&bytes), Some(d));
    }

    #[test]
    fn short_descriptor_rejected() {
        assert_eq!(decode_desc(&[0u8; DESC_LEN - 1]), None);
    }
}
